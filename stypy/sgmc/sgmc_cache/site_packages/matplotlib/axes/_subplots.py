
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: from six.moves import map
6: 
7: from matplotlib.gridspec import GridSpec, SubplotSpec
8: from matplotlib import docstring
9: import matplotlib.artist as martist
10: from matplotlib.axes._axes import Axes
11: 
12: import warnings
13: from matplotlib.cbook import mplDeprecation
14: 
15: 
16: class SubplotBase(object):
17:     '''
18:     Base class for subplots, which are :class:`Axes` instances with
19:     additional methods to facilitate generating and manipulating a set
20:     of :class:`Axes` within a figure.
21:     '''
22: 
23:     def __init__(self, fig, *args, **kwargs):
24:         '''
25:         *fig* is a :class:`matplotlib.figure.Figure` instance.
26: 
27:         *args* is the tuple (*numRows*, *numCols*, *plotNum*), where
28:         the array of subplots in the figure has dimensions *numRows*,
29:         *numCols*, and where *plotNum* is the number of the subplot
30:         being created.  *plotNum* starts at 1 in the upper left
31:         corner and increases to the right.
32: 
33: 
34:         If *numRows* <= *numCols* <= *plotNum* < 10, *args* can be the
35:         decimal integer *numRows* * 100 + *numCols* * 10 + *plotNum*.
36:         '''
37: 
38:         self.figure = fig
39: 
40:         if len(args) == 1:
41:             if isinstance(args[0], SubplotSpec):
42:                 self._subplotspec = args[0]
43:             else:
44:                 try:
45:                     s = str(int(args[0]))
46:                     rows, cols, num = map(int, s)
47:                 except ValueError:
48:                     raise ValueError(
49:                         'Single argument to subplot must be a 3-digit '
50:                         'integer')
51:                 self._subplotspec = GridSpec(rows, cols)[num - 1]
52:                 # num - 1 for converting from MATLAB to python indexing
53:         elif len(args) == 3:
54:             rows, cols, num = args
55:             rows = int(rows)
56:             cols = int(cols)
57:             if isinstance(num, tuple) and len(num) == 2:
58:                 num = [int(n) for n in num]
59:                 self._subplotspec = GridSpec(rows, cols)[num[0] - 1:num[1]]
60:             else:
61:                 if num < 1 or num > rows*cols:
62:                     raise ValueError(
63:                         "num must be 1 <= num <= {maxn}, not {num}".format(
64:                             maxn=rows*cols, num=num))
65:                 self._subplotspec = GridSpec(rows, cols)[int(num) - 1]
66:                 # num - 1 for converting from MATLAB to python indexing
67:         else:
68:             raise ValueError('Illegal argument(s) to subplot: %s' % (args,))
69: 
70:         self.update_params()
71: 
72:         # _axes_class is set in the subplot_class_factory
73:         self._axes_class.__init__(self, fig, self.figbox, **kwargs)
74: 
75:     def __reduce__(self):
76:         # get the first axes class which does not
77:         # inherit from a subplotbase
78: 
79:         def not_subplotbase(c):
80:             return issubclass(c, Axes) and not issubclass(c, SubplotBase)
81: 
82:         axes_class = [c for c in self.__class__.mro()
83:                       if not_subplotbase(c)][0]
84:         r = [_PicklableSubplotClassConstructor(),
85:              (axes_class,),
86:              self.__getstate__()]
87:         return tuple(r)
88: 
89:     def get_geometry(self):
90:         '''get the subplot geometry, e.g., 2,2,3'''
91:         rows, cols, num1, num2 = self.get_subplotspec().get_geometry()
92:         return rows, cols, num1 + 1  # for compatibility
93: 
94:     # COVERAGE NOTE: Never used internally or from examples
95:     def change_geometry(self, numrows, numcols, num):
96:         '''change subplot geometry, e.g., from 1,1,1 to 2,2,3'''
97:         self._subplotspec = GridSpec(numrows, numcols)[num - 1]
98:         self.update_params()
99:         self.set_position(self.figbox)
100: 
101:     def get_subplotspec(self):
102:         '''get the SubplotSpec instance associated with the subplot'''
103:         return self._subplotspec
104: 
105:     def set_subplotspec(self, subplotspec):
106:         '''set the SubplotSpec instance associated with the subplot'''
107:         self._subplotspec = subplotspec
108: 
109:     def update_params(self):
110:         '''update the subplot position from fig.subplotpars'''
111: 
112:         self.figbox, self.rowNum, self.colNum, self.numRows, self.numCols = \
113:             self.get_subplotspec().get_position(self.figure,
114:                                                 return_all=True)
115: 
116:     def is_first_col(self):
117:         return self.colNum == 0
118: 
119:     def is_first_row(self):
120:         return self.rowNum == 0
121: 
122:     def is_last_row(self):
123:         return self.rowNum == self.numRows - 1
124: 
125:     def is_last_col(self):
126:         return self.colNum == self.numCols - 1
127: 
128:     # COVERAGE NOTE: Never used internally.
129:     def label_outer(self):
130:         '''Only show "outer" labels and tick labels.
131: 
132:         x-labels are only kept for subplots on the last row; y-labels only for
133:         subplots on the first column.
134:         '''
135:         lastrow = self.is_last_row()
136:         firstcol = self.is_first_col()
137:         if not lastrow:
138:             for label in self.get_xticklabels(which="both"):
139:                 label.set_visible(False)
140:             self.get_xaxis().get_offset_text().set_visible(False)
141:             self.set_xlabel("")
142:         if not firstcol:
143:             for label in self.get_yticklabels(which="both"):
144:                 label.set_visible(False)
145:             self.get_yaxis().get_offset_text().set_visible(False)
146:             self.set_ylabel("")
147: 
148:     def _make_twin_axes(self, *kl, **kwargs):
149:         '''
150:         make a twinx axes of self. This is used for twinx and twiny.
151:         '''
152:         from matplotlib.projections import process_projection_requirements
153:         kl = (self.get_subplotspec(),) + kl
154:         projection_class, kwargs, key = process_projection_requirements(
155:             self.figure, *kl, **kwargs)
156: 
157:         ax2 = subplot_class_factory(projection_class)(self.figure,
158:                                                       *kl, **kwargs)
159:         self.figure.add_subplot(ax2)
160:         return ax2
161: 
162: _subplot_classes = {}
163: 
164: 
165: def subplot_class_factory(axes_class=None):
166:     # This makes a new class that inherits from SubplotBase and the
167:     # given axes_class (which is assumed to be a subclass of Axes).
168:     # This is perhaps a little bit roundabout to make a new class on
169:     # the fly like this, but it means that a new Subplot class does
170:     # not have to be created for every type of Axes.
171:     if axes_class is None:
172:         axes_class = Axes
173: 
174:     new_class = _subplot_classes.get(axes_class)
175:     if new_class is None:
176:         new_class = type(str("%sSubplot") % (axes_class.__name__),
177:                          (SubplotBase, axes_class),
178:                          {'_axes_class': axes_class})
179:         _subplot_classes[axes_class] = new_class
180: 
181:     return new_class
182: 
183: # This is provided for backward compatibility
184: Subplot = subplot_class_factory()
185: 
186: 
187: class _PicklableSubplotClassConstructor(object):
188:     '''
189:     This stub class exists to return the appropriate subplot
190:     class when __call__-ed with an axes class. This is purely to
191:     allow Pickling of Axes and Subplots.
192:     '''
193:     def __call__(self, axes_class):
194:         # create a dummy object instance
195:         subplot_instance = _PicklableSubplotClassConstructor()
196:         subplot_class = subplot_class_factory(axes_class)
197:         # update the class to the desired subplot class
198:         subplot_instance.__class__ = subplot_class
199:         return subplot_instance
200: 
201: 
202: docstring.interpd.update(Axes=martist.kwdoc(Axes))
203: docstring.interpd.update(Subplot=martist.kwdoc(Axes))
204: 
205: '''
206: # this is some discarded code I was using to find the minimum positive
207: # data point for some log scaling fixes.  I realized there was a
208: # cleaner way to do it, but am keeping this around as an example for
209: # how to get the data out of the axes.  Might want to make something
210: # like this a method one day, or better yet make get_verts an Artist
211: # method
212: 
213:             minx, maxx = self.get_xlim()
214:             if minx<=0 or maxx<=0:
215:                 # find the min pos value in the data
216:                 xs = []
217:                 for line in self.lines:
218:                     xs.extend(line.get_xdata(orig=False))
219:                 for patch in self.patches:
220:                     xs.extend([x for x,y in patch.get_verts()])
221:                 for collection in self.collections:
222:                     xs.extend([x for x,y in collection.get_verts()])
223:                 posx = [x for x in xs if x>0]
224:                 if len(posx):
225: 
226:                     minx = min(posx)
227:                     maxx = max(posx)
228:                     # warning, probably breaks inverted axis
229:                     self.set_xlim((0.1*minx, maxx))
230: 
231: '''
232: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/axes/')
import_217161 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_217161) is not StypyTypeError):

    if (import_217161 != 'pyd_module'):
        __import__(import_217161)
        sys_modules_217162 = sys.modules[import_217161]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_217162.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_217161)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/axes/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from six.moves import map' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/axes/')
import_217163 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'six.moves')

if (type(import_217163) is not StypyTypeError):

    if (import_217163 != 'pyd_module'):
        __import__(import_217163)
        sys_modules_217164 = sys.modules[import_217163]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'six.moves', sys_modules_217164.module_type_store, module_type_store, ['map'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_217164, sys_modules_217164.module_type_store, module_type_store)
    else:
        from six.moves import map

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'six.moves', None, module_type_store, ['map'], [map])

else:
    # Assigning a type to the variable 'six.moves' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'six.moves', import_217163)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/axes/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from matplotlib.gridspec import GridSpec, SubplotSpec' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/axes/')
import_217165 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.gridspec')

if (type(import_217165) is not StypyTypeError):

    if (import_217165 != 'pyd_module'):
        __import__(import_217165)
        sys_modules_217166 = sys.modules[import_217165]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.gridspec', sys_modules_217166.module_type_store, module_type_store, ['GridSpec', 'SubplotSpec'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_217166, sys_modules_217166.module_type_store, module_type_store)
    else:
        from matplotlib.gridspec import GridSpec, SubplotSpec

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.gridspec', None, module_type_store, ['GridSpec', 'SubplotSpec'], [GridSpec, SubplotSpec])

else:
    # Assigning a type to the variable 'matplotlib.gridspec' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.gridspec', import_217165)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/axes/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from matplotlib import docstring' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/axes/')
import_217167 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib')

if (type(import_217167) is not StypyTypeError):

    if (import_217167 != 'pyd_module'):
        __import__(import_217167)
        sys_modules_217168 = sys.modules[import_217167]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib', sys_modules_217168.module_type_store, module_type_store, ['docstring'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_217168, sys_modules_217168.module_type_store, module_type_store)
    else:
        from matplotlib import docstring

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib', None, module_type_store, ['docstring'], [docstring])

else:
    # Assigning a type to the variable 'matplotlib' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib', import_217167)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/axes/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import matplotlib.artist' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/axes/')
import_217169 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.artist')

if (type(import_217169) is not StypyTypeError):

    if (import_217169 != 'pyd_module'):
        __import__(import_217169)
        sys_modules_217170 = sys.modules[import_217169]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'martist', sys_modules_217170.module_type_store, module_type_store)
    else:
        import matplotlib.artist as martist

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'martist', matplotlib.artist, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.artist' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.artist', import_217169)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/axes/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from matplotlib.axes._axes import Axes' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/axes/')
import_217171 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.axes._axes')

if (type(import_217171) is not StypyTypeError):

    if (import_217171 != 'pyd_module'):
        __import__(import_217171)
        sys_modules_217172 = sys.modules[import_217171]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.axes._axes', sys_modules_217172.module_type_store, module_type_store, ['Axes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_217172, sys_modules_217172.module_type_store, module_type_store)
    else:
        from matplotlib.axes._axes import Axes

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.axes._axes', None, module_type_store, ['Axes'], [Axes])

else:
    # Assigning a type to the variable 'matplotlib.axes._axes' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib.axes._axes', import_217171)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/axes/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import warnings' statement (line 12)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from matplotlib.cbook import mplDeprecation' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/axes/')
import_217173 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.cbook')

if (type(import_217173) is not StypyTypeError):

    if (import_217173 != 'pyd_module'):
        __import__(import_217173)
        sys_modules_217174 = sys.modules[import_217173]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.cbook', sys_modules_217174.module_type_store, module_type_store, ['mplDeprecation'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_217174, sys_modules_217174.module_type_store, module_type_store)
    else:
        from matplotlib.cbook import mplDeprecation

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.cbook', None, module_type_store, ['mplDeprecation'], [mplDeprecation])

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.cbook', import_217173)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/axes/')

# Declaration of the 'SubplotBase' class

class SubplotBase(object, ):
    unicode_217175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, (-1)), 'unicode', u'\n    Base class for subplots, which are :class:`Axes` instances with\n    additional methods to facilitate generating and manipulating a set\n    of :class:`Axes` within a figure.\n    ')

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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotBase.__init__', ['fig'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['fig'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_217176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, (-1)), 'unicode', u'\n        *fig* is a :class:`matplotlib.figure.Figure` instance.\n\n        *args* is the tuple (*numRows*, *numCols*, *plotNum*), where\n        the array of subplots in the figure has dimensions *numRows*,\n        *numCols*, and where *plotNum* is the number of the subplot\n        being created.  *plotNum* starts at 1 in the upper left\n        corner and increases to the right.\n\n\n        If *numRows* <= *numCols* <= *plotNum* < 10, *args* can be the\n        decimal integer *numRows* * 100 + *numCols* * 10 + *plotNum*.\n        ')
        
        # Assigning a Name to a Attribute (line 38):
        
        # Assigning a Name to a Attribute (line 38):
        # Getting the type of 'fig' (line 38)
        fig_217177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 22), 'fig')
        # Getting the type of 'self' (line 38)
        self_217178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self')
        # Setting the type of the member 'figure' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_217178, 'figure', fig_217177)
        
        
        
        # Call to len(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'args' (line 40)
        args_217180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'args', False)
        # Processing the call keyword arguments (line 40)
        kwargs_217181 = {}
        # Getting the type of 'len' (line 40)
        len_217179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'len', False)
        # Calling len(args, kwargs) (line 40)
        len_call_result_217182 = invoke(stypy.reporting.localization.Localization(__file__, 40, 11), len_217179, *[args_217180], **kwargs_217181)
        
        int_217183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 24), 'int')
        # Applying the binary operator '==' (line 40)
        result_eq_217184 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 11), '==', len_call_result_217182, int_217183)
        
        # Testing the type of an if condition (line 40)
        if_condition_217185 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 8), result_eq_217184)
        # Assigning a type to the variable 'if_condition_217185' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'if_condition_217185', if_condition_217185)
        # SSA begins for if statement (line 40)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to isinstance(...): (line 41)
        # Processing the call arguments (line 41)
        
        # Obtaining the type of the subscript
        int_217187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 31), 'int')
        # Getting the type of 'args' (line 41)
        args_217188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 26), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 41)
        getitem___217189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 26), args_217188, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 41)
        subscript_call_result_217190 = invoke(stypy.reporting.localization.Localization(__file__, 41, 26), getitem___217189, int_217187)
        
        # Getting the type of 'SubplotSpec' (line 41)
        SubplotSpec_217191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 35), 'SubplotSpec', False)
        # Processing the call keyword arguments (line 41)
        kwargs_217192 = {}
        # Getting the type of 'isinstance' (line 41)
        isinstance_217186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 41)
        isinstance_call_result_217193 = invoke(stypy.reporting.localization.Localization(__file__, 41, 15), isinstance_217186, *[subscript_call_result_217190, SubplotSpec_217191], **kwargs_217192)
        
        # Testing the type of an if condition (line 41)
        if_condition_217194 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 12), isinstance_call_result_217193)
        # Assigning a type to the variable 'if_condition_217194' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'if_condition_217194', if_condition_217194)
        # SSA begins for if statement (line 41)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Attribute (line 42):
        
        # Assigning a Subscript to a Attribute (line 42):
        
        # Obtaining the type of the subscript
        int_217195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 41), 'int')
        # Getting the type of 'args' (line 42)
        args_217196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 36), 'args')
        # Obtaining the member '__getitem__' of a type (line 42)
        getitem___217197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 36), args_217196, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 42)
        subscript_call_result_217198 = invoke(stypy.reporting.localization.Localization(__file__, 42, 36), getitem___217197, int_217195)
        
        # Getting the type of 'self' (line 42)
        self_217199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'self')
        # Setting the type of the member '_subplotspec' of a type (line 42)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 16), self_217199, '_subplotspec', subscript_call_result_217198)
        # SSA branch for the else part of an if statement (line 41)
        module_type_store.open_ssa_branch('else')
        
        
        # SSA begins for try-except statement (line 44)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 45):
        
        # Assigning a Call to a Name (line 45):
        
        # Call to str(...): (line 45)
        # Processing the call arguments (line 45)
        
        # Call to int(...): (line 45)
        # Processing the call arguments (line 45)
        
        # Obtaining the type of the subscript
        int_217202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 37), 'int')
        # Getting the type of 'args' (line 45)
        args_217203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 32), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 45)
        getitem___217204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 32), args_217203, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 45)
        subscript_call_result_217205 = invoke(stypy.reporting.localization.Localization(__file__, 45, 32), getitem___217204, int_217202)
        
        # Processing the call keyword arguments (line 45)
        kwargs_217206 = {}
        # Getting the type of 'int' (line 45)
        int_217201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 28), 'int', False)
        # Calling int(args, kwargs) (line 45)
        int_call_result_217207 = invoke(stypy.reporting.localization.Localization(__file__, 45, 28), int_217201, *[subscript_call_result_217205], **kwargs_217206)
        
        # Processing the call keyword arguments (line 45)
        kwargs_217208 = {}
        # Getting the type of 'str' (line 45)
        str_217200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 24), 'str', False)
        # Calling str(args, kwargs) (line 45)
        str_call_result_217209 = invoke(stypy.reporting.localization.Localization(__file__, 45, 24), str_217200, *[int_call_result_217207], **kwargs_217208)
        
        # Assigning a type to the variable 's' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 20), 's', str_call_result_217209)
        
        # Assigning a Call to a Tuple (line 46):
        
        # Assigning a Call to a Name:
        
        # Call to map(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'int' (line 46)
        int_217211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 42), 'int', False)
        # Getting the type of 's' (line 46)
        s_217212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 47), 's', False)
        # Processing the call keyword arguments (line 46)
        kwargs_217213 = {}
        # Getting the type of 'map' (line 46)
        map_217210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 38), 'map', False)
        # Calling map(args, kwargs) (line 46)
        map_call_result_217214 = invoke(stypy.reporting.localization.Localization(__file__, 46, 38), map_217210, *[int_217211, s_217212], **kwargs_217213)
        
        # Assigning a type to the variable 'call_assignment_217139' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'call_assignment_217139', map_call_result_217214)
        
        # Assigning a Call to a Name (line 46):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_217217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 20), 'int')
        # Processing the call keyword arguments
        kwargs_217218 = {}
        # Getting the type of 'call_assignment_217139' (line 46)
        call_assignment_217139_217215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'call_assignment_217139', False)
        # Obtaining the member '__getitem__' of a type (line 46)
        getitem___217216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 20), call_assignment_217139_217215, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_217219 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___217216, *[int_217217], **kwargs_217218)
        
        # Assigning a type to the variable 'call_assignment_217140' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'call_assignment_217140', getitem___call_result_217219)
        
        # Assigning a Name to a Name (line 46):
        # Getting the type of 'call_assignment_217140' (line 46)
        call_assignment_217140_217220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'call_assignment_217140')
        # Assigning a type to the variable 'rows' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'rows', call_assignment_217140_217220)
        
        # Assigning a Call to a Name (line 46):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_217223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 20), 'int')
        # Processing the call keyword arguments
        kwargs_217224 = {}
        # Getting the type of 'call_assignment_217139' (line 46)
        call_assignment_217139_217221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'call_assignment_217139', False)
        # Obtaining the member '__getitem__' of a type (line 46)
        getitem___217222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 20), call_assignment_217139_217221, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_217225 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___217222, *[int_217223], **kwargs_217224)
        
        # Assigning a type to the variable 'call_assignment_217141' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'call_assignment_217141', getitem___call_result_217225)
        
        # Assigning a Name to a Name (line 46):
        # Getting the type of 'call_assignment_217141' (line 46)
        call_assignment_217141_217226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'call_assignment_217141')
        # Assigning a type to the variable 'cols' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'cols', call_assignment_217141_217226)
        
        # Assigning a Call to a Name (line 46):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_217229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 20), 'int')
        # Processing the call keyword arguments
        kwargs_217230 = {}
        # Getting the type of 'call_assignment_217139' (line 46)
        call_assignment_217139_217227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'call_assignment_217139', False)
        # Obtaining the member '__getitem__' of a type (line 46)
        getitem___217228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 20), call_assignment_217139_217227, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_217231 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___217228, *[int_217229], **kwargs_217230)
        
        # Assigning a type to the variable 'call_assignment_217142' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'call_assignment_217142', getitem___call_result_217231)
        
        # Assigning a Name to a Name (line 46):
        # Getting the type of 'call_assignment_217142' (line 46)
        call_assignment_217142_217232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'call_assignment_217142')
        # Assigning a type to the variable 'num' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 32), 'num', call_assignment_217142_217232)
        # SSA branch for the except part of a try statement (line 44)
        # SSA branch for the except 'ValueError' branch of a try statement (line 44)
        module_type_store.open_ssa_branch('except')
        
        # Call to ValueError(...): (line 48)
        # Processing the call arguments (line 48)
        unicode_217234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 24), 'unicode', u'Single argument to subplot must be a 3-digit integer')
        # Processing the call keyword arguments (line 48)
        kwargs_217235 = {}
        # Getting the type of 'ValueError' (line 48)
        ValueError_217233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 48)
        ValueError_call_result_217236 = invoke(stypy.reporting.localization.Localization(__file__, 48, 26), ValueError_217233, *[unicode_217234], **kwargs_217235)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 48, 20), ValueError_call_result_217236, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 44)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Attribute (line 51):
        
        # Assigning a Subscript to a Attribute (line 51):
        
        # Obtaining the type of the subscript
        # Getting the type of 'num' (line 51)
        num_217237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 57), 'num')
        int_217238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 63), 'int')
        # Applying the binary operator '-' (line 51)
        result_sub_217239 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 57), '-', num_217237, int_217238)
        
        
        # Call to GridSpec(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'rows' (line 51)
        rows_217241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 45), 'rows', False)
        # Getting the type of 'cols' (line 51)
        cols_217242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 51), 'cols', False)
        # Processing the call keyword arguments (line 51)
        kwargs_217243 = {}
        # Getting the type of 'GridSpec' (line 51)
        GridSpec_217240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 36), 'GridSpec', False)
        # Calling GridSpec(args, kwargs) (line 51)
        GridSpec_call_result_217244 = invoke(stypy.reporting.localization.Localization(__file__, 51, 36), GridSpec_217240, *[rows_217241, cols_217242], **kwargs_217243)
        
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___217245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 36), GridSpec_call_result_217244, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_217246 = invoke(stypy.reporting.localization.Localization(__file__, 51, 36), getitem___217245, result_sub_217239)
        
        # Getting the type of 'self' (line 51)
        self_217247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'self')
        # Setting the type of the member '_subplotspec' of a type (line 51)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 16), self_217247, '_subplotspec', subscript_call_result_217246)
        # SSA join for if statement (line 41)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 40)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to len(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'args' (line 53)
        args_217249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'args', False)
        # Processing the call keyword arguments (line 53)
        kwargs_217250 = {}
        # Getting the type of 'len' (line 53)
        len_217248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 13), 'len', False)
        # Calling len(args, kwargs) (line 53)
        len_call_result_217251 = invoke(stypy.reporting.localization.Localization(__file__, 53, 13), len_217248, *[args_217249], **kwargs_217250)
        
        int_217252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 26), 'int')
        # Applying the binary operator '==' (line 53)
        result_eq_217253 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 13), '==', len_call_result_217251, int_217252)
        
        # Testing the type of an if condition (line 53)
        if_condition_217254 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 13), result_eq_217253)
        # Assigning a type to the variable 'if_condition_217254' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 13), 'if_condition_217254', if_condition_217254)
        # SSA begins for if statement (line 53)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Tuple (line 54):
        
        # Assigning a Subscript to a Name (line 54):
        
        # Obtaining the type of the subscript
        int_217255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 12), 'int')
        # Getting the type of 'args' (line 54)
        args_217256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 30), 'args')
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___217257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), args_217256, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_217258 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), getitem___217257, int_217255)
        
        # Assigning a type to the variable 'tuple_var_assignment_217143' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'tuple_var_assignment_217143', subscript_call_result_217258)
        
        # Assigning a Subscript to a Name (line 54):
        
        # Obtaining the type of the subscript
        int_217259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 12), 'int')
        # Getting the type of 'args' (line 54)
        args_217260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 30), 'args')
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___217261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), args_217260, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_217262 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), getitem___217261, int_217259)
        
        # Assigning a type to the variable 'tuple_var_assignment_217144' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'tuple_var_assignment_217144', subscript_call_result_217262)
        
        # Assigning a Subscript to a Name (line 54):
        
        # Obtaining the type of the subscript
        int_217263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 12), 'int')
        # Getting the type of 'args' (line 54)
        args_217264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 30), 'args')
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___217265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), args_217264, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_217266 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), getitem___217265, int_217263)
        
        # Assigning a type to the variable 'tuple_var_assignment_217145' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'tuple_var_assignment_217145', subscript_call_result_217266)
        
        # Assigning a Name to a Name (line 54):
        # Getting the type of 'tuple_var_assignment_217143' (line 54)
        tuple_var_assignment_217143_217267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'tuple_var_assignment_217143')
        # Assigning a type to the variable 'rows' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'rows', tuple_var_assignment_217143_217267)
        
        # Assigning a Name to a Name (line 54):
        # Getting the type of 'tuple_var_assignment_217144' (line 54)
        tuple_var_assignment_217144_217268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'tuple_var_assignment_217144')
        # Assigning a type to the variable 'cols' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 18), 'cols', tuple_var_assignment_217144_217268)
        
        # Assigning a Name to a Name (line 54):
        # Getting the type of 'tuple_var_assignment_217145' (line 54)
        tuple_var_assignment_217145_217269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'tuple_var_assignment_217145')
        # Assigning a type to the variable 'num' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'num', tuple_var_assignment_217145_217269)
        
        # Assigning a Call to a Name (line 55):
        
        # Assigning a Call to a Name (line 55):
        
        # Call to int(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'rows' (line 55)
        rows_217271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 23), 'rows', False)
        # Processing the call keyword arguments (line 55)
        kwargs_217272 = {}
        # Getting the type of 'int' (line 55)
        int_217270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'int', False)
        # Calling int(args, kwargs) (line 55)
        int_call_result_217273 = invoke(stypy.reporting.localization.Localization(__file__, 55, 19), int_217270, *[rows_217271], **kwargs_217272)
        
        # Assigning a type to the variable 'rows' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'rows', int_call_result_217273)
        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Call to int(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'cols' (line 56)
        cols_217275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 23), 'cols', False)
        # Processing the call keyword arguments (line 56)
        kwargs_217276 = {}
        # Getting the type of 'int' (line 56)
        int_217274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 19), 'int', False)
        # Calling int(args, kwargs) (line 56)
        int_call_result_217277 = invoke(stypy.reporting.localization.Localization(__file__, 56, 19), int_217274, *[cols_217275], **kwargs_217276)
        
        # Assigning a type to the variable 'cols' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'cols', int_call_result_217277)
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'num' (line 57)
        num_217279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 26), 'num', False)
        # Getting the type of 'tuple' (line 57)
        tuple_217280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 31), 'tuple', False)
        # Processing the call keyword arguments (line 57)
        kwargs_217281 = {}
        # Getting the type of 'isinstance' (line 57)
        isinstance_217278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 57)
        isinstance_call_result_217282 = invoke(stypy.reporting.localization.Localization(__file__, 57, 15), isinstance_217278, *[num_217279, tuple_217280], **kwargs_217281)
        
        
        
        # Call to len(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'num' (line 57)
        num_217284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 46), 'num', False)
        # Processing the call keyword arguments (line 57)
        kwargs_217285 = {}
        # Getting the type of 'len' (line 57)
        len_217283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 42), 'len', False)
        # Calling len(args, kwargs) (line 57)
        len_call_result_217286 = invoke(stypy.reporting.localization.Localization(__file__, 57, 42), len_217283, *[num_217284], **kwargs_217285)
        
        int_217287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 54), 'int')
        # Applying the binary operator '==' (line 57)
        result_eq_217288 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 42), '==', len_call_result_217286, int_217287)
        
        # Applying the binary operator 'and' (line 57)
        result_and_keyword_217289 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 15), 'and', isinstance_call_result_217282, result_eq_217288)
        
        # Testing the type of an if condition (line 57)
        if_condition_217290 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 12), result_and_keyword_217289)
        # Assigning a type to the variable 'if_condition_217290' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'if_condition_217290', if_condition_217290)
        # SSA begins for if statement (line 57)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a ListComp to a Name (line 58):
        
        # Assigning a ListComp to a Name (line 58):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'num' (line 58)
        num_217295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 39), 'num')
        comprehension_217296 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 23), num_217295)
        # Assigning a type to the variable 'n' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 23), 'n', comprehension_217296)
        
        # Call to int(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'n' (line 58)
        n_217292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 27), 'n', False)
        # Processing the call keyword arguments (line 58)
        kwargs_217293 = {}
        # Getting the type of 'int' (line 58)
        int_217291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 23), 'int', False)
        # Calling int(args, kwargs) (line 58)
        int_call_result_217294 = invoke(stypy.reporting.localization.Localization(__file__, 58, 23), int_217291, *[n_217292], **kwargs_217293)
        
        list_217297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 23), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 23), list_217297, int_call_result_217294)
        # Assigning a type to the variable 'num' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'num', list_217297)
        
        # Assigning a Subscript to a Attribute (line 59):
        
        # Assigning a Subscript to a Attribute (line 59):
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_217298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 61), 'int')
        # Getting the type of 'num' (line 59)
        num_217299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 57), 'num')
        # Obtaining the member '__getitem__' of a type (line 59)
        getitem___217300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 57), num_217299, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 59)
        subscript_call_result_217301 = invoke(stypy.reporting.localization.Localization(__file__, 59, 57), getitem___217300, int_217298)
        
        int_217302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 66), 'int')
        # Applying the binary operator '-' (line 59)
        result_sub_217303 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 57), '-', subscript_call_result_217301, int_217302)
        
        
        # Obtaining the type of the subscript
        int_217304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 72), 'int')
        # Getting the type of 'num' (line 59)
        num_217305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 68), 'num')
        # Obtaining the member '__getitem__' of a type (line 59)
        getitem___217306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 68), num_217305, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 59)
        subscript_call_result_217307 = invoke(stypy.reporting.localization.Localization(__file__, 59, 68), getitem___217306, int_217304)
        
        slice_217308 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 59, 36), result_sub_217303, subscript_call_result_217307, None)
        
        # Call to GridSpec(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'rows' (line 59)
        rows_217310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 45), 'rows', False)
        # Getting the type of 'cols' (line 59)
        cols_217311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 51), 'cols', False)
        # Processing the call keyword arguments (line 59)
        kwargs_217312 = {}
        # Getting the type of 'GridSpec' (line 59)
        GridSpec_217309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 36), 'GridSpec', False)
        # Calling GridSpec(args, kwargs) (line 59)
        GridSpec_call_result_217313 = invoke(stypy.reporting.localization.Localization(__file__, 59, 36), GridSpec_217309, *[rows_217310, cols_217311], **kwargs_217312)
        
        # Obtaining the member '__getitem__' of a type (line 59)
        getitem___217314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 36), GridSpec_call_result_217313, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 59)
        subscript_call_result_217315 = invoke(stypy.reporting.localization.Localization(__file__, 59, 36), getitem___217314, slice_217308)
        
        # Getting the type of 'self' (line 59)
        self_217316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'self')
        # Setting the type of the member '_subplotspec' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 16), self_217316, '_subplotspec', subscript_call_result_217315)
        # SSA branch for the else part of an if statement (line 57)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'num' (line 61)
        num_217317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 19), 'num')
        int_217318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 25), 'int')
        # Applying the binary operator '<' (line 61)
        result_lt_217319 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 19), '<', num_217317, int_217318)
        
        
        # Getting the type of 'num' (line 61)
        num_217320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'num')
        # Getting the type of 'rows' (line 61)
        rows_217321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 36), 'rows')
        # Getting the type of 'cols' (line 61)
        cols_217322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 41), 'cols')
        # Applying the binary operator '*' (line 61)
        result_mul_217323 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 36), '*', rows_217321, cols_217322)
        
        # Applying the binary operator '>' (line 61)
        result_gt_217324 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 30), '>', num_217320, result_mul_217323)
        
        # Applying the binary operator 'or' (line 61)
        result_or_keyword_217325 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 19), 'or', result_lt_217319, result_gt_217324)
        
        # Testing the type of an if condition (line 61)
        if_condition_217326 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 16), result_or_keyword_217325)
        # Assigning a type to the variable 'if_condition_217326' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'if_condition_217326', if_condition_217326)
        # SSA begins for if statement (line 61)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 62)
        # Processing the call arguments (line 62)
        
        # Call to format(...): (line 63)
        # Processing the call keyword arguments (line 63)
        # Getting the type of 'rows' (line 64)
        rows_217330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 33), 'rows', False)
        # Getting the type of 'cols' (line 64)
        cols_217331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 38), 'cols', False)
        # Applying the binary operator '*' (line 64)
        result_mul_217332 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 33), '*', rows_217330, cols_217331)
        
        keyword_217333 = result_mul_217332
        # Getting the type of 'num' (line 64)
        num_217334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 48), 'num', False)
        keyword_217335 = num_217334
        kwargs_217336 = {'num': keyword_217335, 'maxn': keyword_217333}
        unicode_217328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 24), 'unicode', u'num must be 1 <= num <= {maxn}, not {num}')
        # Obtaining the member 'format' of a type (line 63)
        format_217329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 24), unicode_217328, 'format')
        # Calling format(args, kwargs) (line 63)
        format_call_result_217337 = invoke(stypy.reporting.localization.Localization(__file__, 63, 24), format_217329, *[], **kwargs_217336)
        
        # Processing the call keyword arguments (line 62)
        kwargs_217338 = {}
        # Getting the type of 'ValueError' (line 62)
        ValueError_217327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 62)
        ValueError_call_result_217339 = invoke(stypy.reporting.localization.Localization(__file__, 62, 26), ValueError_217327, *[format_call_result_217337], **kwargs_217338)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 62, 20), ValueError_call_result_217339, 'raise parameter', BaseException)
        # SSA join for if statement (line 61)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Attribute (line 65):
        
        # Assigning a Subscript to a Attribute (line 65):
        
        # Obtaining the type of the subscript
        
        # Call to int(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'num' (line 65)
        num_217341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 61), 'num', False)
        # Processing the call keyword arguments (line 65)
        kwargs_217342 = {}
        # Getting the type of 'int' (line 65)
        int_217340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 57), 'int', False)
        # Calling int(args, kwargs) (line 65)
        int_call_result_217343 = invoke(stypy.reporting.localization.Localization(__file__, 65, 57), int_217340, *[num_217341], **kwargs_217342)
        
        int_217344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 68), 'int')
        # Applying the binary operator '-' (line 65)
        result_sub_217345 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 57), '-', int_call_result_217343, int_217344)
        
        
        # Call to GridSpec(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'rows' (line 65)
        rows_217347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 45), 'rows', False)
        # Getting the type of 'cols' (line 65)
        cols_217348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 51), 'cols', False)
        # Processing the call keyword arguments (line 65)
        kwargs_217349 = {}
        # Getting the type of 'GridSpec' (line 65)
        GridSpec_217346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 36), 'GridSpec', False)
        # Calling GridSpec(args, kwargs) (line 65)
        GridSpec_call_result_217350 = invoke(stypy.reporting.localization.Localization(__file__, 65, 36), GridSpec_217346, *[rows_217347, cols_217348], **kwargs_217349)
        
        # Obtaining the member '__getitem__' of a type (line 65)
        getitem___217351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 36), GridSpec_call_result_217350, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 65)
        subscript_call_result_217352 = invoke(stypy.reporting.localization.Localization(__file__, 65, 36), getitem___217351, result_sub_217345)
        
        # Getting the type of 'self' (line 65)
        self_217353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'self')
        # Setting the type of the member '_subplotspec' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 16), self_217353, '_subplotspec', subscript_call_result_217352)
        # SSA join for if statement (line 57)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 53)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 68)
        # Processing the call arguments (line 68)
        unicode_217355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 29), 'unicode', u'Illegal argument(s) to subplot: %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 68)
        tuple_217356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 69), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 68)
        # Adding element type (line 68)
        # Getting the type of 'args' (line 68)
        args_217357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 69), 'args', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 69), tuple_217356, args_217357)
        
        # Applying the binary operator '%' (line 68)
        result_mod_217358 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 29), '%', unicode_217355, tuple_217356)
        
        # Processing the call keyword arguments (line 68)
        kwargs_217359 = {}
        # Getting the type of 'ValueError' (line 68)
        ValueError_217354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 68)
        ValueError_call_result_217360 = invoke(stypy.reporting.localization.Localization(__file__, 68, 18), ValueError_217354, *[result_mod_217358], **kwargs_217359)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 68, 12), ValueError_call_result_217360, 'raise parameter', BaseException)
        # SSA join for if statement (line 53)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 40)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to update_params(...): (line 70)
        # Processing the call keyword arguments (line 70)
        kwargs_217363 = {}
        # Getting the type of 'self' (line 70)
        self_217361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'self', False)
        # Obtaining the member 'update_params' of a type (line 70)
        update_params_217362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), self_217361, 'update_params')
        # Calling update_params(args, kwargs) (line 70)
        update_params_call_result_217364 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), update_params_217362, *[], **kwargs_217363)
        
        
        # Call to __init__(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'self' (line 73)
        self_217368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 34), 'self', False)
        # Getting the type of 'fig' (line 73)
        fig_217369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 40), 'fig', False)
        # Getting the type of 'self' (line 73)
        self_217370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 45), 'self', False)
        # Obtaining the member 'figbox' of a type (line 73)
        figbox_217371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 45), self_217370, 'figbox')
        # Processing the call keyword arguments (line 73)
        # Getting the type of 'kwargs' (line 73)
        kwargs_217372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 60), 'kwargs', False)
        kwargs_217373 = {'kwargs_217372': kwargs_217372}
        # Getting the type of 'self' (line 73)
        self_217365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self', False)
        # Obtaining the member '_axes_class' of a type (line 73)
        _axes_class_217366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_217365, '_axes_class')
        # Obtaining the member '__init__' of a type (line 73)
        init___217367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), _axes_class_217366, '__init__')
        # Calling __init__(args, kwargs) (line 73)
        init___call_result_217374 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), init___217367, *[self_217368, fig_217369, figbox_217371], **kwargs_217373)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __reduce__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__reduce__'
        module_type_store = module_type_store.open_function_context('__reduce__', 75, 4, False)
        # Assigning a type to the variable 'self' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubplotBase.__reduce__.__dict__.__setitem__('stypy_localization', localization)
        SubplotBase.__reduce__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubplotBase.__reduce__.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubplotBase.__reduce__.__dict__.__setitem__('stypy_function_name', 'SubplotBase.__reduce__')
        SubplotBase.__reduce__.__dict__.__setitem__('stypy_param_names_list', [])
        SubplotBase.__reduce__.__dict__.__setitem__('stypy_varargs_param_name', None)
        SubplotBase.__reduce__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SubplotBase.__reduce__.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubplotBase.__reduce__.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubplotBase.__reduce__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubplotBase.__reduce__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotBase.__reduce__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__reduce__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__reduce__(...)' code ##################


        @norecursion
        def not_subplotbase(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'not_subplotbase'
            module_type_store = module_type_store.open_function_context('not_subplotbase', 79, 8, False)
            
            # Passed parameters checking function
            not_subplotbase.stypy_localization = localization
            not_subplotbase.stypy_type_of_self = None
            not_subplotbase.stypy_type_store = module_type_store
            not_subplotbase.stypy_function_name = 'not_subplotbase'
            not_subplotbase.stypy_param_names_list = ['c']
            not_subplotbase.stypy_varargs_param_name = None
            not_subplotbase.stypy_kwargs_param_name = None
            not_subplotbase.stypy_call_defaults = defaults
            not_subplotbase.stypy_call_varargs = varargs
            not_subplotbase.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'not_subplotbase', ['c'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'not_subplotbase', localization, ['c'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'not_subplotbase(...)' code ##################

            
            # Evaluating a boolean operation
            
            # Call to issubclass(...): (line 80)
            # Processing the call arguments (line 80)
            # Getting the type of 'c' (line 80)
            c_217376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 30), 'c', False)
            # Getting the type of 'Axes' (line 80)
            Axes_217377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 33), 'Axes', False)
            # Processing the call keyword arguments (line 80)
            kwargs_217378 = {}
            # Getting the type of 'issubclass' (line 80)
            issubclass_217375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 19), 'issubclass', False)
            # Calling issubclass(args, kwargs) (line 80)
            issubclass_call_result_217379 = invoke(stypy.reporting.localization.Localization(__file__, 80, 19), issubclass_217375, *[c_217376, Axes_217377], **kwargs_217378)
            
            
            
            # Call to issubclass(...): (line 80)
            # Processing the call arguments (line 80)
            # Getting the type of 'c' (line 80)
            c_217381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 58), 'c', False)
            # Getting the type of 'SubplotBase' (line 80)
            SubplotBase_217382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 61), 'SubplotBase', False)
            # Processing the call keyword arguments (line 80)
            kwargs_217383 = {}
            # Getting the type of 'issubclass' (line 80)
            issubclass_217380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 47), 'issubclass', False)
            # Calling issubclass(args, kwargs) (line 80)
            issubclass_call_result_217384 = invoke(stypy.reporting.localization.Localization(__file__, 80, 47), issubclass_217380, *[c_217381, SubplotBase_217382], **kwargs_217383)
            
            # Applying the 'not' unary operator (line 80)
            result_not__217385 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 43), 'not', issubclass_call_result_217384)
            
            # Applying the binary operator 'and' (line 80)
            result_and_keyword_217386 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 19), 'and', issubclass_call_result_217379, result_not__217385)
            
            # Assigning a type to the variable 'stypy_return_type' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'stypy_return_type', result_and_keyword_217386)
            
            # ################# End of 'not_subplotbase(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'not_subplotbase' in the type store
            # Getting the type of 'stypy_return_type' (line 79)
            stypy_return_type_217387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_217387)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'not_subplotbase'
            return stypy_return_type_217387

        # Assigning a type to the variable 'not_subplotbase' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'not_subplotbase', not_subplotbase)
        
        # Assigning a Subscript to a Name (line 82):
        
        # Assigning a Subscript to a Name (line 82):
        
        # Obtaining the type of the subscript
        int_217388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 45), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to mro(...): (line 82)
        # Processing the call keyword arguments (line 82)
        kwargs_217397 = {}
        # Getting the type of 'self' (line 82)
        self_217394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 33), 'self', False)
        # Obtaining the member '__class__' of a type (line 82)
        class___217395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 33), self_217394, '__class__')
        # Obtaining the member 'mro' of a type (line 82)
        mro_217396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 33), class___217395, 'mro')
        # Calling mro(args, kwargs) (line 82)
        mro_call_result_217398 = invoke(stypy.reporting.localization.Localization(__file__, 82, 33), mro_217396, *[], **kwargs_217397)
        
        comprehension_217399 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 22), mro_call_result_217398)
        # Assigning a type to the variable 'c' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 22), 'c', comprehension_217399)
        
        # Call to not_subplotbase(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'c' (line 83)
        c_217391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 41), 'c', False)
        # Processing the call keyword arguments (line 83)
        kwargs_217392 = {}
        # Getting the type of 'not_subplotbase' (line 83)
        not_subplotbase_217390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 25), 'not_subplotbase', False)
        # Calling not_subplotbase(args, kwargs) (line 83)
        not_subplotbase_call_result_217393 = invoke(stypy.reporting.localization.Localization(__file__, 83, 25), not_subplotbase_217390, *[c_217391], **kwargs_217392)
        
        # Getting the type of 'c' (line 82)
        c_217389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 22), 'c')
        list_217400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 22), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 22), list_217400, c_217389)
        # Obtaining the member '__getitem__' of a type (line 82)
        getitem___217401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 22), list_217400, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 82)
        subscript_call_result_217402 = invoke(stypy.reporting.localization.Localization(__file__, 82, 22), getitem___217401, int_217388)
        
        # Assigning a type to the variable 'axes_class' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'axes_class', subscript_call_result_217402)
        
        # Assigning a List to a Name (line 84):
        
        # Assigning a List to a Name (line 84):
        
        # Obtaining an instance of the builtin type 'list' (line 84)
        list_217403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 84)
        # Adding element type (line 84)
        
        # Call to _PicklableSubplotClassConstructor(...): (line 84)
        # Processing the call keyword arguments (line 84)
        kwargs_217405 = {}
        # Getting the type of '_PicklableSubplotClassConstructor' (line 84)
        _PicklableSubplotClassConstructor_217404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 13), '_PicklableSubplotClassConstructor', False)
        # Calling _PicklableSubplotClassConstructor(args, kwargs) (line 84)
        _PicklableSubplotClassConstructor_call_result_217406 = invoke(stypy.reporting.localization.Localization(__file__, 84, 13), _PicklableSubplotClassConstructor_217404, *[], **kwargs_217405)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 12), list_217403, _PicklableSubplotClassConstructor_call_result_217406)
        # Adding element type (line 84)
        
        # Obtaining an instance of the builtin type 'tuple' (line 85)
        tuple_217407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 85)
        # Adding element type (line 85)
        # Getting the type of 'axes_class' (line 85)
        axes_class_217408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 14), 'axes_class')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 14), tuple_217407, axes_class_217408)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 12), list_217403, tuple_217407)
        # Adding element type (line 84)
        
        # Call to __getstate__(...): (line 86)
        # Processing the call keyword arguments (line 86)
        kwargs_217411 = {}
        # Getting the type of 'self' (line 86)
        self_217409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 13), 'self', False)
        # Obtaining the member '__getstate__' of a type (line 86)
        getstate___217410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 13), self_217409, '__getstate__')
        # Calling __getstate__(args, kwargs) (line 86)
        getstate___call_result_217412 = invoke(stypy.reporting.localization.Localization(__file__, 86, 13), getstate___217410, *[], **kwargs_217411)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 12), list_217403, getstate___call_result_217412)
        
        # Assigning a type to the variable 'r' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'r', list_217403)
        
        # Call to tuple(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'r' (line 87)
        r_217414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 21), 'r', False)
        # Processing the call keyword arguments (line 87)
        kwargs_217415 = {}
        # Getting the type of 'tuple' (line 87)
        tuple_217413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'tuple', False)
        # Calling tuple(args, kwargs) (line 87)
        tuple_call_result_217416 = invoke(stypy.reporting.localization.Localization(__file__, 87, 15), tuple_217413, *[r_217414], **kwargs_217415)
        
        # Assigning a type to the variable 'stypy_return_type' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'stypy_return_type', tuple_call_result_217416)
        
        # ################# End of '__reduce__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__reduce__' in the type store
        # Getting the type of 'stypy_return_type' (line 75)
        stypy_return_type_217417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217417)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__reduce__'
        return stypy_return_type_217417


    @norecursion
    def get_geometry(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_geometry'
        module_type_store = module_type_store.open_function_context('get_geometry', 89, 4, False)
        # Assigning a type to the variable 'self' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubplotBase.get_geometry.__dict__.__setitem__('stypy_localization', localization)
        SubplotBase.get_geometry.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubplotBase.get_geometry.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubplotBase.get_geometry.__dict__.__setitem__('stypy_function_name', 'SubplotBase.get_geometry')
        SubplotBase.get_geometry.__dict__.__setitem__('stypy_param_names_list', [])
        SubplotBase.get_geometry.__dict__.__setitem__('stypy_varargs_param_name', None)
        SubplotBase.get_geometry.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SubplotBase.get_geometry.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubplotBase.get_geometry.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubplotBase.get_geometry.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubplotBase.get_geometry.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotBase.get_geometry', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_geometry', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_geometry(...)' code ##################

        unicode_217418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 8), 'unicode', u'get the subplot geometry, e.g., 2,2,3')
        
        # Assigning a Call to a Tuple (line 91):
        
        # Assigning a Call to a Name:
        
        # Call to get_geometry(...): (line 91)
        # Processing the call keyword arguments (line 91)
        kwargs_217424 = {}
        
        # Call to get_subplotspec(...): (line 91)
        # Processing the call keyword arguments (line 91)
        kwargs_217421 = {}
        # Getting the type of 'self' (line 91)
        self_217419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 33), 'self', False)
        # Obtaining the member 'get_subplotspec' of a type (line 91)
        get_subplotspec_217420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 33), self_217419, 'get_subplotspec')
        # Calling get_subplotspec(args, kwargs) (line 91)
        get_subplotspec_call_result_217422 = invoke(stypy.reporting.localization.Localization(__file__, 91, 33), get_subplotspec_217420, *[], **kwargs_217421)
        
        # Obtaining the member 'get_geometry' of a type (line 91)
        get_geometry_217423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 33), get_subplotspec_call_result_217422, 'get_geometry')
        # Calling get_geometry(args, kwargs) (line 91)
        get_geometry_call_result_217425 = invoke(stypy.reporting.localization.Localization(__file__, 91, 33), get_geometry_217423, *[], **kwargs_217424)
        
        # Assigning a type to the variable 'call_assignment_217146' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'call_assignment_217146', get_geometry_call_result_217425)
        
        # Assigning a Call to a Name (line 91):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_217428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'int')
        # Processing the call keyword arguments
        kwargs_217429 = {}
        # Getting the type of 'call_assignment_217146' (line 91)
        call_assignment_217146_217426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'call_assignment_217146', False)
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___217427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), call_assignment_217146_217426, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_217430 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___217427, *[int_217428], **kwargs_217429)
        
        # Assigning a type to the variable 'call_assignment_217147' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'call_assignment_217147', getitem___call_result_217430)
        
        # Assigning a Name to a Name (line 91):
        # Getting the type of 'call_assignment_217147' (line 91)
        call_assignment_217147_217431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'call_assignment_217147')
        # Assigning a type to the variable 'rows' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'rows', call_assignment_217147_217431)
        
        # Assigning a Call to a Name (line 91):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_217434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'int')
        # Processing the call keyword arguments
        kwargs_217435 = {}
        # Getting the type of 'call_assignment_217146' (line 91)
        call_assignment_217146_217432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'call_assignment_217146', False)
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___217433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), call_assignment_217146_217432, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_217436 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___217433, *[int_217434], **kwargs_217435)
        
        # Assigning a type to the variable 'call_assignment_217148' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'call_assignment_217148', getitem___call_result_217436)
        
        # Assigning a Name to a Name (line 91):
        # Getting the type of 'call_assignment_217148' (line 91)
        call_assignment_217148_217437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'call_assignment_217148')
        # Assigning a type to the variable 'cols' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 14), 'cols', call_assignment_217148_217437)
        
        # Assigning a Call to a Name (line 91):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_217440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'int')
        # Processing the call keyword arguments
        kwargs_217441 = {}
        # Getting the type of 'call_assignment_217146' (line 91)
        call_assignment_217146_217438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'call_assignment_217146', False)
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___217439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), call_assignment_217146_217438, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_217442 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___217439, *[int_217440], **kwargs_217441)
        
        # Assigning a type to the variable 'call_assignment_217149' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'call_assignment_217149', getitem___call_result_217442)
        
        # Assigning a Name to a Name (line 91):
        # Getting the type of 'call_assignment_217149' (line 91)
        call_assignment_217149_217443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'call_assignment_217149')
        # Assigning a type to the variable 'num1' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'num1', call_assignment_217149_217443)
        
        # Assigning a Call to a Name (line 91):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_217446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'int')
        # Processing the call keyword arguments
        kwargs_217447 = {}
        # Getting the type of 'call_assignment_217146' (line 91)
        call_assignment_217146_217444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'call_assignment_217146', False)
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___217445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), call_assignment_217146_217444, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_217448 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___217445, *[int_217446], **kwargs_217447)
        
        # Assigning a type to the variable 'call_assignment_217150' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'call_assignment_217150', getitem___call_result_217448)
        
        # Assigning a Name to a Name (line 91):
        # Getting the type of 'call_assignment_217150' (line 91)
        call_assignment_217150_217449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'call_assignment_217150')
        # Assigning a type to the variable 'num2' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 26), 'num2', call_assignment_217150_217449)
        
        # Obtaining an instance of the builtin type 'tuple' (line 92)
        tuple_217450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 92)
        # Adding element type (line 92)
        # Getting the type of 'rows' (line 92)
        rows_217451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'rows')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 15), tuple_217450, rows_217451)
        # Adding element type (line 92)
        # Getting the type of 'cols' (line 92)
        cols_217452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 21), 'cols')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 15), tuple_217450, cols_217452)
        # Adding element type (line 92)
        # Getting the type of 'num1' (line 92)
        num1_217453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 27), 'num1')
        int_217454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 34), 'int')
        # Applying the binary operator '+' (line 92)
        result_add_217455 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 27), '+', num1_217453, int_217454)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 15), tuple_217450, result_add_217455)
        
        # Assigning a type to the variable 'stypy_return_type' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'stypy_return_type', tuple_217450)
        
        # ################# End of 'get_geometry(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_geometry' in the type store
        # Getting the type of 'stypy_return_type' (line 89)
        stypy_return_type_217456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217456)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_geometry'
        return stypy_return_type_217456


    @norecursion
    def change_geometry(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'change_geometry'
        module_type_store = module_type_store.open_function_context('change_geometry', 95, 4, False)
        # Assigning a type to the variable 'self' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubplotBase.change_geometry.__dict__.__setitem__('stypy_localization', localization)
        SubplotBase.change_geometry.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubplotBase.change_geometry.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubplotBase.change_geometry.__dict__.__setitem__('stypy_function_name', 'SubplotBase.change_geometry')
        SubplotBase.change_geometry.__dict__.__setitem__('stypy_param_names_list', ['numrows', 'numcols', 'num'])
        SubplotBase.change_geometry.__dict__.__setitem__('stypy_varargs_param_name', None)
        SubplotBase.change_geometry.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SubplotBase.change_geometry.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubplotBase.change_geometry.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubplotBase.change_geometry.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubplotBase.change_geometry.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotBase.change_geometry', ['numrows', 'numcols', 'num'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'change_geometry', localization, ['numrows', 'numcols', 'num'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'change_geometry(...)' code ##################

        unicode_217457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'unicode', u'change subplot geometry, e.g., from 1,1,1 to 2,2,3')
        
        # Assigning a Subscript to a Attribute (line 97):
        
        # Assigning a Subscript to a Attribute (line 97):
        
        # Obtaining the type of the subscript
        # Getting the type of 'num' (line 97)
        num_217458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 55), 'num')
        int_217459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 61), 'int')
        # Applying the binary operator '-' (line 97)
        result_sub_217460 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 55), '-', num_217458, int_217459)
        
        
        # Call to GridSpec(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'numrows' (line 97)
        numrows_217462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 37), 'numrows', False)
        # Getting the type of 'numcols' (line 97)
        numcols_217463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 46), 'numcols', False)
        # Processing the call keyword arguments (line 97)
        kwargs_217464 = {}
        # Getting the type of 'GridSpec' (line 97)
        GridSpec_217461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 28), 'GridSpec', False)
        # Calling GridSpec(args, kwargs) (line 97)
        GridSpec_call_result_217465 = invoke(stypy.reporting.localization.Localization(__file__, 97, 28), GridSpec_217461, *[numrows_217462, numcols_217463], **kwargs_217464)
        
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___217466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 28), GridSpec_call_result_217465, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_217467 = invoke(stypy.reporting.localization.Localization(__file__, 97, 28), getitem___217466, result_sub_217460)
        
        # Getting the type of 'self' (line 97)
        self_217468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'self')
        # Setting the type of the member '_subplotspec' of a type (line 97)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), self_217468, '_subplotspec', subscript_call_result_217467)
        
        # Call to update_params(...): (line 98)
        # Processing the call keyword arguments (line 98)
        kwargs_217471 = {}
        # Getting the type of 'self' (line 98)
        self_217469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'self', False)
        # Obtaining the member 'update_params' of a type (line 98)
        update_params_217470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), self_217469, 'update_params')
        # Calling update_params(args, kwargs) (line 98)
        update_params_call_result_217472 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), update_params_217470, *[], **kwargs_217471)
        
        
        # Call to set_position(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'self' (line 99)
        self_217475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 26), 'self', False)
        # Obtaining the member 'figbox' of a type (line 99)
        figbox_217476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 26), self_217475, 'figbox')
        # Processing the call keyword arguments (line 99)
        kwargs_217477 = {}
        # Getting the type of 'self' (line 99)
        self_217473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'self', False)
        # Obtaining the member 'set_position' of a type (line 99)
        set_position_217474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), self_217473, 'set_position')
        # Calling set_position(args, kwargs) (line 99)
        set_position_call_result_217478 = invoke(stypy.reporting.localization.Localization(__file__, 99, 8), set_position_217474, *[figbox_217476], **kwargs_217477)
        
        
        # ################# End of 'change_geometry(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'change_geometry' in the type store
        # Getting the type of 'stypy_return_type' (line 95)
        stypy_return_type_217479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217479)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'change_geometry'
        return stypy_return_type_217479


    @norecursion
    def get_subplotspec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_subplotspec'
        module_type_store = module_type_store.open_function_context('get_subplotspec', 101, 4, False)
        # Assigning a type to the variable 'self' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubplotBase.get_subplotspec.__dict__.__setitem__('stypy_localization', localization)
        SubplotBase.get_subplotspec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubplotBase.get_subplotspec.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubplotBase.get_subplotspec.__dict__.__setitem__('stypy_function_name', 'SubplotBase.get_subplotspec')
        SubplotBase.get_subplotspec.__dict__.__setitem__('stypy_param_names_list', [])
        SubplotBase.get_subplotspec.__dict__.__setitem__('stypy_varargs_param_name', None)
        SubplotBase.get_subplotspec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SubplotBase.get_subplotspec.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubplotBase.get_subplotspec.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubplotBase.get_subplotspec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubplotBase.get_subplotspec.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotBase.get_subplotspec', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_subplotspec', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_subplotspec(...)' code ##################

        unicode_217480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 8), 'unicode', u'get the SubplotSpec instance associated with the subplot')
        # Getting the type of 'self' (line 103)
        self_217481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'self')
        # Obtaining the member '_subplotspec' of a type (line 103)
        _subplotspec_217482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 15), self_217481, '_subplotspec')
        # Assigning a type to the variable 'stypy_return_type' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'stypy_return_type', _subplotspec_217482)
        
        # ################# End of 'get_subplotspec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_subplotspec' in the type store
        # Getting the type of 'stypy_return_type' (line 101)
        stypy_return_type_217483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217483)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_subplotspec'
        return stypy_return_type_217483


    @norecursion
    def set_subplotspec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_subplotspec'
        module_type_store = module_type_store.open_function_context('set_subplotspec', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubplotBase.set_subplotspec.__dict__.__setitem__('stypy_localization', localization)
        SubplotBase.set_subplotspec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubplotBase.set_subplotspec.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubplotBase.set_subplotspec.__dict__.__setitem__('stypy_function_name', 'SubplotBase.set_subplotspec')
        SubplotBase.set_subplotspec.__dict__.__setitem__('stypy_param_names_list', ['subplotspec'])
        SubplotBase.set_subplotspec.__dict__.__setitem__('stypy_varargs_param_name', None)
        SubplotBase.set_subplotspec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SubplotBase.set_subplotspec.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubplotBase.set_subplotspec.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubplotBase.set_subplotspec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubplotBase.set_subplotspec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotBase.set_subplotspec', ['subplotspec'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_subplotspec', localization, ['subplotspec'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_subplotspec(...)' code ##################

        unicode_217484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'unicode', u'set the SubplotSpec instance associated with the subplot')
        
        # Assigning a Name to a Attribute (line 107):
        
        # Assigning a Name to a Attribute (line 107):
        # Getting the type of 'subplotspec' (line 107)
        subplotspec_217485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 28), 'subplotspec')
        # Getting the type of 'self' (line 107)
        self_217486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'self')
        # Setting the type of the member '_subplotspec' of a type (line 107)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 8), self_217486, '_subplotspec', subplotspec_217485)
        
        # ################# End of 'set_subplotspec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_subplotspec' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_217487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217487)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_subplotspec'
        return stypy_return_type_217487


    @norecursion
    def update_params(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update_params'
        module_type_store = module_type_store.open_function_context('update_params', 109, 4, False)
        # Assigning a type to the variable 'self' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubplotBase.update_params.__dict__.__setitem__('stypy_localization', localization)
        SubplotBase.update_params.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubplotBase.update_params.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubplotBase.update_params.__dict__.__setitem__('stypy_function_name', 'SubplotBase.update_params')
        SubplotBase.update_params.__dict__.__setitem__('stypy_param_names_list', [])
        SubplotBase.update_params.__dict__.__setitem__('stypy_varargs_param_name', None)
        SubplotBase.update_params.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SubplotBase.update_params.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubplotBase.update_params.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubplotBase.update_params.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubplotBase.update_params.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotBase.update_params', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update_params', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update_params(...)' code ##################

        unicode_217488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 8), 'unicode', u'update the subplot position from fig.subplotpars')
        
        # Assigning a Call to a Tuple (line 112):
        
        # Assigning a Call to a Name:
        
        # Call to get_position(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'self' (line 113)
        self_217494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 48), 'self', False)
        # Obtaining the member 'figure' of a type (line 113)
        figure_217495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 48), self_217494, 'figure')
        # Processing the call keyword arguments (line 113)
        # Getting the type of 'True' (line 114)
        True_217496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 59), 'True', False)
        keyword_217497 = True_217496
        kwargs_217498 = {'return_all': keyword_217497}
        
        # Call to get_subplotspec(...): (line 113)
        # Processing the call keyword arguments (line 113)
        kwargs_217491 = {}
        # Getting the type of 'self' (line 113)
        self_217489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'self', False)
        # Obtaining the member 'get_subplotspec' of a type (line 113)
        get_subplotspec_217490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 12), self_217489, 'get_subplotspec')
        # Calling get_subplotspec(args, kwargs) (line 113)
        get_subplotspec_call_result_217492 = invoke(stypy.reporting.localization.Localization(__file__, 113, 12), get_subplotspec_217490, *[], **kwargs_217491)
        
        # Obtaining the member 'get_position' of a type (line 113)
        get_position_217493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 12), get_subplotspec_call_result_217492, 'get_position')
        # Calling get_position(args, kwargs) (line 113)
        get_position_call_result_217499 = invoke(stypy.reporting.localization.Localization(__file__, 113, 12), get_position_217493, *[figure_217495], **kwargs_217498)
        
        # Assigning a type to the variable 'call_assignment_217151' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'call_assignment_217151', get_position_call_result_217499)
        
        # Assigning a Call to a Name (line 112):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_217502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 8), 'int')
        # Processing the call keyword arguments
        kwargs_217503 = {}
        # Getting the type of 'call_assignment_217151' (line 112)
        call_assignment_217151_217500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'call_assignment_217151', False)
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___217501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), call_assignment_217151_217500, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_217504 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___217501, *[int_217502], **kwargs_217503)
        
        # Assigning a type to the variable 'call_assignment_217152' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'call_assignment_217152', getitem___call_result_217504)
        
        # Assigning a Name to a Attribute (line 112):
        # Getting the type of 'call_assignment_217152' (line 112)
        call_assignment_217152_217505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'call_assignment_217152')
        # Getting the type of 'self' (line 112)
        self_217506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'self')
        # Setting the type of the member 'figbox' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), self_217506, 'figbox', call_assignment_217152_217505)
        
        # Assigning a Call to a Name (line 112):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_217509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 8), 'int')
        # Processing the call keyword arguments
        kwargs_217510 = {}
        # Getting the type of 'call_assignment_217151' (line 112)
        call_assignment_217151_217507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'call_assignment_217151', False)
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___217508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), call_assignment_217151_217507, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_217511 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___217508, *[int_217509], **kwargs_217510)
        
        # Assigning a type to the variable 'call_assignment_217153' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'call_assignment_217153', getitem___call_result_217511)
        
        # Assigning a Name to a Attribute (line 112):
        # Getting the type of 'call_assignment_217153' (line 112)
        call_assignment_217153_217512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'call_assignment_217153')
        # Getting the type of 'self' (line 112)
        self_217513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 21), 'self')
        # Setting the type of the member 'rowNum' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 21), self_217513, 'rowNum', call_assignment_217153_217512)
        
        # Assigning a Call to a Name (line 112):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_217516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 8), 'int')
        # Processing the call keyword arguments
        kwargs_217517 = {}
        # Getting the type of 'call_assignment_217151' (line 112)
        call_assignment_217151_217514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'call_assignment_217151', False)
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___217515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), call_assignment_217151_217514, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_217518 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___217515, *[int_217516], **kwargs_217517)
        
        # Assigning a type to the variable 'call_assignment_217154' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'call_assignment_217154', getitem___call_result_217518)
        
        # Assigning a Name to a Attribute (line 112):
        # Getting the type of 'call_assignment_217154' (line 112)
        call_assignment_217154_217519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'call_assignment_217154')
        # Getting the type of 'self' (line 112)
        self_217520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 34), 'self')
        # Setting the type of the member 'colNum' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 34), self_217520, 'colNum', call_assignment_217154_217519)
        
        # Assigning a Call to a Name (line 112):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_217523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 8), 'int')
        # Processing the call keyword arguments
        kwargs_217524 = {}
        # Getting the type of 'call_assignment_217151' (line 112)
        call_assignment_217151_217521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'call_assignment_217151', False)
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___217522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), call_assignment_217151_217521, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_217525 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___217522, *[int_217523], **kwargs_217524)
        
        # Assigning a type to the variable 'call_assignment_217155' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'call_assignment_217155', getitem___call_result_217525)
        
        # Assigning a Name to a Attribute (line 112):
        # Getting the type of 'call_assignment_217155' (line 112)
        call_assignment_217155_217526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'call_assignment_217155')
        # Getting the type of 'self' (line 112)
        self_217527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 47), 'self')
        # Setting the type of the member 'numRows' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 47), self_217527, 'numRows', call_assignment_217155_217526)
        
        # Assigning a Call to a Name (line 112):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_217530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 8), 'int')
        # Processing the call keyword arguments
        kwargs_217531 = {}
        # Getting the type of 'call_assignment_217151' (line 112)
        call_assignment_217151_217528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'call_assignment_217151', False)
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___217529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), call_assignment_217151_217528, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_217532 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___217529, *[int_217530], **kwargs_217531)
        
        # Assigning a type to the variable 'call_assignment_217156' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'call_assignment_217156', getitem___call_result_217532)
        
        # Assigning a Name to a Attribute (line 112):
        # Getting the type of 'call_assignment_217156' (line 112)
        call_assignment_217156_217533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'call_assignment_217156')
        # Getting the type of 'self' (line 112)
        self_217534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 61), 'self')
        # Setting the type of the member 'numCols' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 61), self_217534, 'numCols', call_assignment_217156_217533)
        
        # ################# End of 'update_params(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_params' in the type store
        # Getting the type of 'stypy_return_type' (line 109)
        stypy_return_type_217535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217535)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_params'
        return stypy_return_type_217535


    @norecursion
    def is_first_col(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_first_col'
        module_type_store = module_type_store.open_function_context('is_first_col', 116, 4, False)
        # Assigning a type to the variable 'self' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubplotBase.is_first_col.__dict__.__setitem__('stypy_localization', localization)
        SubplotBase.is_first_col.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubplotBase.is_first_col.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubplotBase.is_first_col.__dict__.__setitem__('stypy_function_name', 'SubplotBase.is_first_col')
        SubplotBase.is_first_col.__dict__.__setitem__('stypy_param_names_list', [])
        SubplotBase.is_first_col.__dict__.__setitem__('stypy_varargs_param_name', None)
        SubplotBase.is_first_col.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SubplotBase.is_first_col.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubplotBase.is_first_col.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubplotBase.is_first_col.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubplotBase.is_first_col.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotBase.is_first_col', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_first_col', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_first_col(...)' code ##################

        
        # Getting the type of 'self' (line 117)
        self_217536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'self')
        # Obtaining the member 'colNum' of a type (line 117)
        colNum_217537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 15), self_217536, 'colNum')
        int_217538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 30), 'int')
        # Applying the binary operator '==' (line 117)
        result_eq_217539 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 15), '==', colNum_217537, int_217538)
        
        # Assigning a type to the variable 'stypy_return_type' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'stypy_return_type', result_eq_217539)
        
        # ################# End of 'is_first_col(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_first_col' in the type store
        # Getting the type of 'stypy_return_type' (line 116)
        stypy_return_type_217540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217540)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_first_col'
        return stypy_return_type_217540


    @norecursion
    def is_first_row(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_first_row'
        module_type_store = module_type_store.open_function_context('is_first_row', 119, 4, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubplotBase.is_first_row.__dict__.__setitem__('stypy_localization', localization)
        SubplotBase.is_first_row.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubplotBase.is_first_row.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubplotBase.is_first_row.__dict__.__setitem__('stypy_function_name', 'SubplotBase.is_first_row')
        SubplotBase.is_first_row.__dict__.__setitem__('stypy_param_names_list', [])
        SubplotBase.is_first_row.__dict__.__setitem__('stypy_varargs_param_name', None)
        SubplotBase.is_first_row.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SubplotBase.is_first_row.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubplotBase.is_first_row.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubplotBase.is_first_row.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubplotBase.is_first_row.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotBase.is_first_row', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_first_row', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_first_row(...)' code ##################

        
        # Getting the type of 'self' (line 120)
        self_217541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'self')
        # Obtaining the member 'rowNum' of a type (line 120)
        rowNum_217542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 15), self_217541, 'rowNum')
        int_217543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 30), 'int')
        # Applying the binary operator '==' (line 120)
        result_eq_217544 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 15), '==', rowNum_217542, int_217543)
        
        # Assigning a type to the variable 'stypy_return_type' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'stypy_return_type', result_eq_217544)
        
        # ################# End of 'is_first_row(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_first_row' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_217545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217545)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_first_row'
        return stypy_return_type_217545


    @norecursion
    def is_last_row(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_last_row'
        module_type_store = module_type_store.open_function_context('is_last_row', 122, 4, False)
        # Assigning a type to the variable 'self' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubplotBase.is_last_row.__dict__.__setitem__('stypy_localization', localization)
        SubplotBase.is_last_row.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubplotBase.is_last_row.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubplotBase.is_last_row.__dict__.__setitem__('stypy_function_name', 'SubplotBase.is_last_row')
        SubplotBase.is_last_row.__dict__.__setitem__('stypy_param_names_list', [])
        SubplotBase.is_last_row.__dict__.__setitem__('stypy_varargs_param_name', None)
        SubplotBase.is_last_row.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SubplotBase.is_last_row.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubplotBase.is_last_row.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubplotBase.is_last_row.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubplotBase.is_last_row.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotBase.is_last_row', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_last_row', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_last_row(...)' code ##################

        
        # Getting the type of 'self' (line 123)
        self_217546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 'self')
        # Obtaining the member 'rowNum' of a type (line 123)
        rowNum_217547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 15), self_217546, 'rowNum')
        # Getting the type of 'self' (line 123)
        self_217548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 30), 'self')
        # Obtaining the member 'numRows' of a type (line 123)
        numRows_217549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 30), self_217548, 'numRows')
        int_217550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 45), 'int')
        # Applying the binary operator '-' (line 123)
        result_sub_217551 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 30), '-', numRows_217549, int_217550)
        
        # Applying the binary operator '==' (line 123)
        result_eq_217552 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 15), '==', rowNum_217547, result_sub_217551)
        
        # Assigning a type to the variable 'stypy_return_type' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'stypy_return_type', result_eq_217552)
        
        # ################# End of 'is_last_row(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_last_row' in the type store
        # Getting the type of 'stypy_return_type' (line 122)
        stypy_return_type_217553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217553)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_last_row'
        return stypy_return_type_217553


    @norecursion
    def is_last_col(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_last_col'
        module_type_store = module_type_store.open_function_context('is_last_col', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubplotBase.is_last_col.__dict__.__setitem__('stypy_localization', localization)
        SubplotBase.is_last_col.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubplotBase.is_last_col.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubplotBase.is_last_col.__dict__.__setitem__('stypy_function_name', 'SubplotBase.is_last_col')
        SubplotBase.is_last_col.__dict__.__setitem__('stypy_param_names_list', [])
        SubplotBase.is_last_col.__dict__.__setitem__('stypy_varargs_param_name', None)
        SubplotBase.is_last_col.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SubplotBase.is_last_col.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubplotBase.is_last_col.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubplotBase.is_last_col.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubplotBase.is_last_col.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotBase.is_last_col', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_last_col', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_last_col(...)' code ##################

        
        # Getting the type of 'self' (line 126)
        self_217554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 15), 'self')
        # Obtaining the member 'colNum' of a type (line 126)
        colNum_217555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 15), self_217554, 'colNum')
        # Getting the type of 'self' (line 126)
        self_217556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 30), 'self')
        # Obtaining the member 'numCols' of a type (line 126)
        numCols_217557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 30), self_217556, 'numCols')
        int_217558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 45), 'int')
        # Applying the binary operator '-' (line 126)
        result_sub_217559 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 30), '-', numCols_217557, int_217558)
        
        # Applying the binary operator '==' (line 126)
        result_eq_217560 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 15), '==', colNum_217555, result_sub_217559)
        
        # Assigning a type to the variable 'stypy_return_type' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'stypy_return_type', result_eq_217560)
        
        # ################# End of 'is_last_col(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_last_col' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_217561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217561)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_last_col'
        return stypy_return_type_217561


    @norecursion
    def label_outer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'label_outer'
        module_type_store = module_type_store.open_function_context('label_outer', 129, 4, False)
        # Assigning a type to the variable 'self' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubplotBase.label_outer.__dict__.__setitem__('stypy_localization', localization)
        SubplotBase.label_outer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubplotBase.label_outer.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubplotBase.label_outer.__dict__.__setitem__('stypy_function_name', 'SubplotBase.label_outer')
        SubplotBase.label_outer.__dict__.__setitem__('stypy_param_names_list', [])
        SubplotBase.label_outer.__dict__.__setitem__('stypy_varargs_param_name', None)
        SubplotBase.label_outer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SubplotBase.label_outer.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubplotBase.label_outer.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubplotBase.label_outer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubplotBase.label_outer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotBase.label_outer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'label_outer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'label_outer(...)' code ##################

        unicode_217562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, (-1)), 'unicode', u'Only show "outer" labels and tick labels.\n\n        x-labels are only kept for subplots on the last row; y-labels only for\n        subplots on the first column.\n        ')
        
        # Assigning a Call to a Name (line 135):
        
        # Assigning a Call to a Name (line 135):
        
        # Call to is_last_row(...): (line 135)
        # Processing the call keyword arguments (line 135)
        kwargs_217565 = {}
        # Getting the type of 'self' (line 135)
        self_217563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 18), 'self', False)
        # Obtaining the member 'is_last_row' of a type (line 135)
        is_last_row_217564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 18), self_217563, 'is_last_row')
        # Calling is_last_row(args, kwargs) (line 135)
        is_last_row_call_result_217566 = invoke(stypy.reporting.localization.Localization(__file__, 135, 18), is_last_row_217564, *[], **kwargs_217565)
        
        # Assigning a type to the variable 'lastrow' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'lastrow', is_last_row_call_result_217566)
        
        # Assigning a Call to a Name (line 136):
        
        # Assigning a Call to a Name (line 136):
        
        # Call to is_first_col(...): (line 136)
        # Processing the call keyword arguments (line 136)
        kwargs_217569 = {}
        # Getting the type of 'self' (line 136)
        self_217567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), 'self', False)
        # Obtaining the member 'is_first_col' of a type (line 136)
        is_first_col_217568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 19), self_217567, 'is_first_col')
        # Calling is_first_col(args, kwargs) (line 136)
        is_first_col_call_result_217570 = invoke(stypy.reporting.localization.Localization(__file__, 136, 19), is_first_col_217568, *[], **kwargs_217569)
        
        # Assigning a type to the variable 'firstcol' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'firstcol', is_first_col_call_result_217570)
        
        
        # Getting the type of 'lastrow' (line 137)
        lastrow_217571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'lastrow')
        # Applying the 'not' unary operator (line 137)
        result_not__217572 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 11), 'not', lastrow_217571)
        
        # Testing the type of an if condition (line 137)
        if_condition_217573 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 8), result_not__217572)
        # Assigning a type to the variable 'if_condition_217573' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'if_condition_217573', if_condition_217573)
        # SSA begins for if statement (line 137)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to get_xticklabels(...): (line 138)
        # Processing the call keyword arguments (line 138)
        unicode_217576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 52), 'unicode', u'both')
        keyword_217577 = unicode_217576
        kwargs_217578 = {'which': keyword_217577}
        # Getting the type of 'self' (line 138)
        self_217574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 25), 'self', False)
        # Obtaining the member 'get_xticklabels' of a type (line 138)
        get_xticklabels_217575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 25), self_217574, 'get_xticklabels')
        # Calling get_xticklabels(args, kwargs) (line 138)
        get_xticklabels_call_result_217579 = invoke(stypy.reporting.localization.Localization(__file__, 138, 25), get_xticklabels_217575, *[], **kwargs_217578)
        
        # Testing the type of a for loop iterable (line 138)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 138, 12), get_xticklabels_call_result_217579)
        # Getting the type of the for loop variable (line 138)
        for_loop_var_217580 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 138, 12), get_xticklabels_call_result_217579)
        # Assigning a type to the variable 'label' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'label', for_loop_var_217580)
        # SSA begins for a for statement (line 138)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to set_visible(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'False' (line 139)
        False_217583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 34), 'False', False)
        # Processing the call keyword arguments (line 139)
        kwargs_217584 = {}
        # Getting the type of 'label' (line 139)
        label_217581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'label', False)
        # Obtaining the member 'set_visible' of a type (line 139)
        set_visible_217582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 16), label_217581, 'set_visible')
        # Calling set_visible(args, kwargs) (line 139)
        set_visible_call_result_217585 = invoke(stypy.reporting.localization.Localization(__file__, 139, 16), set_visible_217582, *[False_217583], **kwargs_217584)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_visible(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'False' (line 140)
        False_217594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 59), 'False', False)
        # Processing the call keyword arguments (line 140)
        kwargs_217595 = {}
        
        # Call to get_offset_text(...): (line 140)
        # Processing the call keyword arguments (line 140)
        kwargs_217591 = {}
        
        # Call to get_xaxis(...): (line 140)
        # Processing the call keyword arguments (line 140)
        kwargs_217588 = {}
        # Getting the type of 'self' (line 140)
        self_217586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'self', False)
        # Obtaining the member 'get_xaxis' of a type (line 140)
        get_xaxis_217587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 12), self_217586, 'get_xaxis')
        # Calling get_xaxis(args, kwargs) (line 140)
        get_xaxis_call_result_217589 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), get_xaxis_217587, *[], **kwargs_217588)
        
        # Obtaining the member 'get_offset_text' of a type (line 140)
        get_offset_text_217590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 12), get_xaxis_call_result_217589, 'get_offset_text')
        # Calling get_offset_text(args, kwargs) (line 140)
        get_offset_text_call_result_217592 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), get_offset_text_217590, *[], **kwargs_217591)
        
        # Obtaining the member 'set_visible' of a type (line 140)
        set_visible_217593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 12), get_offset_text_call_result_217592, 'set_visible')
        # Calling set_visible(args, kwargs) (line 140)
        set_visible_call_result_217596 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), set_visible_217593, *[False_217594], **kwargs_217595)
        
        
        # Call to set_xlabel(...): (line 141)
        # Processing the call arguments (line 141)
        unicode_217599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 28), 'unicode', u'')
        # Processing the call keyword arguments (line 141)
        kwargs_217600 = {}
        # Getting the type of 'self' (line 141)
        self_217597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'self', False)
        # Obtaining the member 'set_xlabel' of a type (line 141)
        set_xlabel_217598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 12), self_217597, 'set_xlabel')
        # Calling set_xlabel(args, kwargs) (line 141)
        set_xlabel_call_result_217601 = invoke(stypy.reporting.localization.Localization(__file__, 141, 12), set_xlabel_217598, *[unicode_217599], **kwargs_217600)
        
        # SSA join for if statement (line 137)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'firstcol' (line 142)
        firstcol_217602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 15), 'firstcol')
        # Applying the 'not' unary operator (line 142)
        result_not__217603 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 11), 'not', firstcol_217602)
        
        # Testing the type of an if condition (line 142)
        if_condition_217604 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 8), result_not__217603)
        # Assigning a type to the variable 'if_condition_217604' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'if_condition_217604', if_condition_217604)
        # SSA begins for if statement (line 142)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to get_yticklabels(...): (line 143)
        # Processing the call keyword arguments (line 143)
        unicode_217607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 52), 'unicode', u'both')
        keyword_217608 = unicode_217607
        kwargs_217609 = {'which': keyword_217608}
        # Getting the type of 'self' (line 143)
        self_217605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 25), 'self', False)
        # Obtaining the member 'get_yticklabels' of a type (line 143)
        get_yticklabels_217606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 25), self_217605, 'get_yticklabels')
        # Calling get_yticklabels(args, kwargs) (line 143)
        get_yticklabels_call_result_217610 = invoke(stypy.reporting.localization.Localization(__file__, 143, 25), get_yticklabels_217606, *[], **kwargs_217609)
        
        # Testing the type of a for loop iterable (line 143)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 143, 12), get_yticklabels_call_result_217610)
        # Getting the type of the for loop variable (line 143)
        for_loop_var_217611 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 143, 12), get_yticklabels_call_result_217610)
        # Assigning a type to the variable 'label' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'label', for_loop_var_217611)
        # SSA begins for a for statement (line 143)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to set_visible(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'False' (line 144)
        False_217614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 34), 'False', False)
        # Processing the call keyword arguments (line 144)
        kwargs_217615 = {}
        # Getting the type of 'label' (line 144)
        label_217612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'label', False)
        # Obtaining the member 'set_visible' of a type (line 144)
        set_visible_217613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 16), label_217612, 'set_visible')
        # Calling set_visible(args, kwargs) (line 144)
        set_visible_call_result_217616 = invoke(stypy.reporting.localization.Localization(__file__, 144, 16), set_visible_217613, *[False_217614], **kwargs_217615)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_visible(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'False' (line 145)
        False_217625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 59), 'False', False)
        # Processing the call keyword arguments (line 145)
        kwargs_217626 = {}
        
        # Call to get_offset_text(...): (line 145)
        # Processing the call keyword arguments (line 145)
        kwargs_217622 = {}
        
        # Call to get_yaxis(...): (line 145)
        # Processing the call keyword arguments (line 145)
        kwargs_217619 = {}
        # Getting the type of 'self' (line 145)
        self_217617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'self', False)
        # Obtaining the member 'get_yaxis' of a type (line 145)
        get_yaxis_217618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 12), self_217617, 'get_yaxis')
        # Calling get_yaxis(args, kwargs) (line 145)
        get_yaxis_call_result_217620 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), get_yaxis_217618, *[], **kwargs_217619)
        
        # Obtaining the member 'get_offset_text' of a type (line 145)
        get_offset_text_217621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 12), get_yaxis_call_result_217620, 'get_offset_text')
        # Calling get_offset_text(args, kwargs) (line 145)
        get_offset_text_call_result_217623 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), get_offset_text_217621, *[], **kwargs_217622)
        
        # Obtaining the member 'set_visible' of a type (line 145)
        set_visible_217624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 12), get_offset_text_call_result_217623, 'set_visible')
        # Calling set_visible(args, kwargs) (line 145)
        set_visible_call_result_217627 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), set_visible_217624, *[False_217625], **kwargs_217626)
        
        
        # Call to set_ylabel(...): (line 146)
        # Processing the call arguments (line 146)
        unicode_217630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 28), 'unicode', u'')
        # Processing the call keyword arguments (line 146)
        kwargs_217631 = {}
        # Getting the type of 'self' (line 146)
        self_217628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'self', False)
        # Obtaining the member 'set_ylabel' of a type (line 146)
        set_ylabel_217629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 12), self_217628, 'set_ylabel')
        # Calling set_ylabel(args, kwargs) (line 146)
        set_ylabel_call_result_217632 = invoke(stypy.reporting.localization.Localization(__file__, 146, 12), set_ylabel_217629, *[unicode_217630], **kwargs_217631)
        
        # SSA join for if statement (line 142)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'label_outer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'label_outer' in the type store
        # Getting the type of 'stypy_return_type' (line 129)
        stypy_return_type_217633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217633)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'label_outer'
        return stypy_return_type_217633


    @norecursion
    def _make_twin_axes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_make_twin_axes'
        module_type_store = module_type_store.open_function_context('_make_twin_axes', 148, 4, False)
        # Assigning a type to the variable 'self' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubplotBase._make_twin_axes.__dict__.__setitem__('stypy_localization', localization)
        SubplotBase._make_twin_axes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubplotBase._make_twin_axes.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubplotBase._make_twin_axes.__dict__.__setitem__('stypy_function_name', 'SubplotBase._make_twin_axes')
        SubplotBase._make_twin_axes.__dict__.__setitem__('stypy_param_names_list', [])
        SubplotBase._make_twin_axes.__dict__.__setitem__('stypy_varargs_param_name', 'kl')
        SubplotBase._make_twin_axes.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        SubplotBase._make_twin_axes.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubplotBase._make_twin_axes.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubplotBase._make_twin_axes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubplotBase._make_twin_axes.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotBase._make_twin_axes', [], 'kl', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_make_twin_axes', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_make_twin_axes(...)' code ##################

        unicode_217634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, (-1)), 'unicode', u'\n        make a twinx axes of self. This is used for twinx and twiny.\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 152, 8))
        
        # 'from matplotlib.projections import process_projection_requirements' statement (line 152)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/axes/')
        import_217635 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 152, 8), 'matplotlib.projections')

        if (type(import_217635) is not StypyTypeError):

            if (import_217635 != 'pyd_module'):
                __import__(import_217635)
                sys_modules_217636 = sys.modules[import_217635]
                import_from_module(stypy.reporting.localization.Localization(__file__, 152, 8), 'matplotlib.projections', sys_modules_217636.module_type_store, module_type_store, ['process_projection_requirements'])
                nest_module(stypy.reporting.localization.Localization(__file__, 152, 8), __file__, sys_modules_217636, sys_modules_217636.module_type_store, module_type_store)
            else:
                from matplotlib.projections import process_projection_requirements

                import_from_module(stypy.reporting.localization.Localization(__file__, 152, 8), 'matplotlib.projections', None, module_type_store, ['process_projection_requirements'], [process_projection_requirements])

        else:
            # Assigning a type to the variable 'matplotlib.projections' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'matplotlib.projections', import_217635)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/axes/')
        
        
        # Assigning a BinOp to a Name (line 153):
        
        # Assigning a BinOp to a Name (line 153):
        
        # Obtaining an instance of the builtin type 'tuple' (line 153)
        tuple_217637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 153)
        # Adding element type (line 153)
        
        # Call to get_subplotspec(...): (line 153)
        # Processing the call keyword arguments (line 153)
        kwargs_217640 = {}
        # Getting the type of 'self' (line 153)
        self_217638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 14), 'self', False)
        # Obtaining the member 'get_subplotspec' of a type (line 153)
        get_subplotspec_217639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 14), self_217638, 'get_subplotspec')
        # Calling get_subplotspec(args, kwargs) (line 153)
        get_subplotspec_call_result_217641 = invoke(stypy.reporting.localization.Localization(__file__, 153, 14), get_subplotspec_217639, *[], **kwargs_217640)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 14), tuple_217637, get_subplotspec_call_result_217641)
        
        # Getting the type of 'kl' (line 153)
        kl_217642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 41), 'kl')
        # Applying the binary operator '+' (line 153)
        result_add_217643 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 13), '+', tuple_217637, kl_217642)
        
        # Assigning a type to the variable 'kl' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'kl', result_add_217643)
        
        # Assigning a Call to a Tuple (line 154):
        
        # Assigning a Call to a Name:
        
        # Call to process_projection_requirements(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'self' (line 155)
        self_217645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'self', False)
        # Obtaining the member 'figure' of a type (line 155)
        figure_217646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 12), self_217645, 'figure')
        # Getting the type of 'kl' (line 155)
        kl_217647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 26), 'kl', False)
        # Processing the call keyword arguments (line 154)
        # Getting the type of 'kwargs' (line 155)
        kwargs_217648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 32), 'kwargs', False)
        kwargs_217649 = {'kwargs_217648': kwargs_217648}
        # Getting the type of 'process_projection_requirements' (line 154)
        process_projection_requirements_217644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 40), 'process_projection_requirements', False)
        # Calling process_projection_requirements(args, kwargs) (line 154)
        process_projection_requirements_call_result_217650 = invoke(stypy.reporting.localization.Localization(__file__, 154, 40), process_projection_requirements_217644, *[figure_217646, kl_217647], **kwargs_217649)
        
        # Assigning a type to the variable 'call_assignment_217157' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'call_assignment_217157', process_projection_requirements_call_result_217650)
        
        # Assigning a Call to a Name (line 154):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_217653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 8), 'int')
        # Processing the call keyword arguments
        kwargs_217654 = {}
        # Getting the type of 'call_assignment_217157' (line 154)
        call_assignment_217157_217651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'call_assignment_217157', False)
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___217652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), call_assignment_217157_217651, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_217655 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___217652, *[int_217653], **kwargs_217654)
        
        # Assigning a type to the variable 'call_assignment_217158' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'call_assignment_217158', getitem___call_result_217655)
        
        # Assigning a Name to a Name (line 154):
        # Getting the type of 'call_assignment_217158' (line 154)
        call_assignment_217158_217656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'call_assignment_217158')
        # Assigning a type to the variable 'projection_class' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'projection_class', call_assignment_217158_217656)
        
        # Assigning a Call to a Name (line 154):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_217659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 8), 'int')
        # Processing the call keyword arguments
        kwargs_217660 = {}
        # Getting the type of 'call_assignment_217157' (line 154)
        call_assignment_217157_217657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'call_assignment_217157', False)
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___217658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), call_assignment_217157_217657, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_217661 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___217658, *[int_217659], **kwargs_217660)
        
        # Assigning a type to the variable 'call_assignment_217159' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'call_assignment_217159', getitem___call_result_217661)
        
        # Assigning a Name to a Name (line 154):
        # Getting the type of 'call_assignment_217159' (line 154)
        call_assignment_217159_217662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'call_assignment_217159')
        # Assigning a type to the variable 'kwargs' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 26), 'kwargs', call_assignment_217159_217662)
        
        # Assigning a Call to a Name (line 154):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_217665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 8), 'int')
        # Processing the call keyword arguments
        kwargs_217666 = {}
        # Getting the type of 'call_assignment_217157' (line 154)
        call_assignment_217157_217663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'call_assignment_217157', False)
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___217664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), call_assignment_217157_217663, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_217667 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___217664, *[int_217665], **kwargs_217666)
        
        # Assigning a type to the variable 'call_assignment_217160' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'call_assignment_217160', getitem___call_result_217667)
        
        # Assigning a Name to a Name (line 154):
        # Getting the type of 'call_assignment_217160' (line 154)
        call_assignment_217160_217668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'call_assignment_217160')
        # Assigning a type to the variable 'key' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 34), 'key', call_assignment_217160_217668)
        
        # Assigning a Call to a Name (line 157):
        
        # Assigning a Call to a Name (line 157):
        
        # Call to (...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'self' (line 157)
        self_217673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 54), 'self', False)
        # Obtaining the member 'figure' of a type (line 157)
        figure_217674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 54), self_217673, 'figure')
        # Getting the type of 'kl' (line 158)
        kl_217675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 55), 'kl', False)
        # Processing the call keyword arguments (line 157)
        # Getting the type of 'kwargs' (line 158)
        kwargs_217676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 61), 'kwargs', False)
        kwargs_217677 = {'kwargs_217676': kwargs_217676}
        
        # Call to subplot_class_factory(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'projection_class' (line 157)
        projection_class_217670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 36), 'projection_class', False)
        # Processing the call keyword arguments (line 157)
        kwargs_217671 = {}
        # Getting the type of 'subplot_class_factory' (line 157)
        subplot_class_factory_217669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 14), 'subplot_class_factory', False)
        # Calling subplot_class_factory(args, kwargs) (line 157)
        subplot_class_factory_call_result_217672 = invoke(stypy.reporting.localization.Localization(__file__, 157, 14), subplot_class_factory_217669, *[projection_class_217670], **kwargs_217671)
        
        # Calling (args, kwargs) (line 157)
        _call_result_217678 = invoke(stypy.reporting.localization.Localization(__file__, 157, 14), subplot_class_factory_call_result_217672, *[figure_217674, kl_217675], **kwargs_217677)
        
        # Assigning a type to the variable 'ax2' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'ax2', _call_result_217678)
        
        # Call to add_subplot(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'ax2' (line 159)
        ax2_217682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 32), 'ax2', False)
        # Processing the call keyword arguments (line 159)
        kwargs_217683 = {}
        # Getting the type of 'self' (line 159)
        self_217679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 159)
        figure_217680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), self_217679, 'figure')
        # Obtaining the member 'add_subplot' of a type (line 159)
        add_subplot_217681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), figure_217680, 'add_subplot')
        # Calling add_subplot(args, kwargs) (line 159)
        add_subplot_call_result_217684 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), add_subplot_217681, *[ax2_217682], **kwargs_217683)
        
        # Getting the type of 'ax2' (line 160)
        ax2_217685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'ax2')
        # Assigning a type to the variable 'stypy_return_type' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'stypy_return_type', ax2_217685)
        
        # ################# End of '_make_twin_axes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_make_twin_axes' in the type store
        # Getting the type of 'stypy_return_type' (line 148)
        stypy_return_type_217686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217686)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_make_twin_axes'
        return stypy_return_type_217686


# Assigning a type to the variable 'SubplotBase' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'SubplotBase', SubplotBase)

# Assigning a Dict to a Name (line 162):

# Assigning a Dict to a Name (line 162):

# Obtaining an instance of the builtin type 'dict' (line 162)
dict_217687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 162)

# Assigning a type to the variable '_subplot_classes' (line 162)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 0), '_subplot_classes', dict_217687)

@norecursion
def subplot_class_factory(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 165)
    None_217688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 37), 'None')
    defaults = [None_217688]
    # Create a new context for function 'subplot_class_factory'
    module_type_store = module_type_store.open_function_context('subplot_class_factory', 165, 0, False)
    
    # Passed parameters checking function
    subplot_class_factory.stypy_localization = localization
    subplot_class_factory.stypy_type_of_self = None
    subplot_class_factory.stypy_type_store = module_type_store
    subplot_class_factory.stypy_function_name = 'subplot_class_factory'
    subplot_class_factory.stypy_param_names_list = ['axes_class']
    subplot_class_factory.stypy_varargs_param_name = None
    subplot_class_factory.stypy_kwargs_param_name = None
    subplot_class_factory.stypy_call_defaults = defaults
    subplot_class_factory.stypy_call_varargs = varargs
    subplot_class_factory.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'subplot_class_factory', ['axes_class'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'subplot_class_factory', localization, ['axes_class'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'subplot_class_factory(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 171)
    # Getting the type of 'axes_class' (line 171)
    axes_class_217689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 7), 'axes_class')
    # Getting the type of 'None' (line 171)
    None_217690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 21), 'None')
    
    (may_be_217691, more_types_in_union_217692) = may_be_none(axes_class_217689, None_217690)

    if may_be_217691:

        if more_types_in_union_217692:
            # Runtime conditional SSA (line 171)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 172):
        
        # Assigning a Name to a Name (line 172):
        # Getting the type of 'Axes' (line 172)
        Axes_217693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 21), 'Axes')
        # Assigning a type to the variable 'axes_class' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'axes_class', Axes_217693)

        if more_types_in_union_217692:
            # SSA join for if statement (line 171)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 174):
    
    # Assigning a Call to a Name (line 174):
    
    # Call to get(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'axes_class' (line 174)
    axes_class_217696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 37), 'axes_class', False)
    # Processing the call keyword arguments (line 174)
    kwargs_217697 = {}
    # Getting the type of '_subplot_classes' (line 174)
    _subplot_classes_217694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), '_subplot_classes', False)
    # Obtaining the member 'get' of a type (line 174)
    get_217695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 16), _subplot_classes_217694, 'get')
    # Calling get(args, kwargs) (line 174)
    get_call_result_217698 = invoke(stypy.reporting.localization.Localization(__file__, 174, 16), get_217695, *[axes_class_217696], **kwargs_217697)
    
    # Assigning a type to the variable 'new_class' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'new_class', get_call_result_217698)
    
    # Type idiom detected: calculating its left and rigth part (line 175)
    # Getting the type of 'new_class' (line 175)
    new_class_217699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 7), 'new_class')
    # Getting the type of 'None' (line 175)
    None_217700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 20), 'None')
    
    (may_be_217701, more_types_in_union_217702) = may_be_none(new_class_217699, None_217700)

    if may_be_217701:

        if more_types_in_union_217702:
            # Runtime conditional SSA (line 175)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 176):
        
        # Assigning a Call to a Name (line 176):
        
        # Call to type(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Call to str(...): (line 176)
        # Processing the call arguments (line 176)
        unicode_217705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 29), 'unicode', u'%sSubplot')
        # Processing the call keyword arguments (line 176)
        kwargs_217706 = {}
        # Getting the type of 'str' (line 176)
        str_217704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 25), 'str', False)
        # Calling str(args, kwargs) (line 176)
        str_call_result_217707 = invoke(stypy.reporting.localization.Localization(__file__, 176, 25), str_217704, *[unicode_217705], **kwargs_217706)
        
        # Getting the type of 'axes_class' (line 176)
        axes_class_217708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 45), 'axes_class', False)
        # Obtaining the member '__name__' of a type (line 176)
        name___217709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 45), axes_class_217708, '__name__')
        # Applying the binary operator '%' (line 176)
        result_mod_217710 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 25), '%', str_call_result_217707, name___217709)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 177)
        tuple_217711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 177)
        # Adding element type (line 177)
        # Getting the type of 'SubplotBase' (line 177)
        SubplotBase_217712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 26), 'SubplotBase', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 26), tuple_217711, SubplotBase_217712)
        # Adding element type (line 177)
        # Getting the type of 'axes_class' (line 177)
        axes_class_217713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 39), 'axes_class', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 26), tuple_217711, axes_class_217713)
        
        
        # Obtaining an instance of the builtin type 'dict' (line 178)
        dict_217714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 25), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 178)
        # Adding element type (key, value) (line 178)
        unicode_217715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 26), 'unicode', u'_axes_class')
        # Getting the type of 'axes_class' (line 178)
        axes_class_217716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 41), 'axes_class', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 25), dict_217714, (unicode_217715, axes_class_217716))
        
        # Processing the call keyword arguments (line 176)
        kwargs_217717 = {}
        # Getting the type of 'type' (line 176)
        type_217703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 20), 'type', False)
        # Calling type(args, kwargs) (line 176)
        type_call_result_217718 = invoke(stypy.reporting.localization.Localization(__file__, 176, 20), type_217703, *[result_mod_217710, tuple_217711, dict_217714], **kwargs_217717)
        
        # Assigning a type to the variable 'new_class' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'new_class', type_call_result_217718)
        
        # Assigning a Name to a Subscript (line 179):
        
        # Assigning a Name to a Subscript (line 179):
        # Getting the type of 'new_class' (line 179)
        new_class_217719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 39), 'new_class')
        # Getting the type of '_subplot_classes' (line 179)
        _subplot_classes_217720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), '_subplot_classes')
        # Getting the type of 'axes_class' (line 179)
        axes_class_217721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 25), 'axes_class')
        # Storing an element on a container (line 179)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 8), _subplot_classes_217720, (axes_class_217721, new_class_217719))

        if more_types_in_union_217702:
            # SSA join for if statement (line 175)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'new_class' (line 181)
    new_class_217722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 11), 'new_class')
    # Assigning a type to the variable 'stypy_return_type' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'stypy_return_type', new_class_217722)
    
    # ################# End of 'subplot_class_factory(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'subplot_class_factory' in the type store
    # Getting the type of 'stypy_return_type' (line 165)
    stypy_return_type_217723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_217723)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'subplot_class_factory'
    return stypy_return_type_217723

# Assigning a type to the variable 'subplot_class_factory' (line 165)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), 'subplot_class_factory', subplot_class_factory)

# Assigning a Call to a Name (line 184):

# Assigning a Call to a Name (line 184):

# Call to subplot_class_factory(...): (line 184)
# Processing the call keyword arguments (line 184)
kwargs_217725 = {}
# Getting the type of 'subplot_class_factory' (line 184)
subplot_class_factory_217724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 10), 'subplot_class_factory', False)
# Calling subplot_class_factory(args, kwargs) (line 184)
subplot_class_factory_call_result_217726 = invoke(stypy.reporting.localization.Localization(__file__, 184, 10), subplot_class_factory_217724, *[], **kwargs_217725)

# Assigning a type to the variable 'Subplot' (line 184)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 0), 'Subplot', subplot_class_factory_call_result_217726)
# Declaration of the '_PicklableSubplotClassConstructor' class

class _PicklableSubplotClassConstructor(object, ):
    unicode_217727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, (-1)), 'unicode', u'\n    This stub class exists to return the appropriate subplot\n    class when __call__-ed with an axes class. This is purely to\n    allow Pickling of Axes and Subplots.\n    ')

    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 193, 4, False)
        # Assigning a type to the variable 'self' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _PicklableSubplotClassConstructor.__call__.__dict__.__setitem__('stypy_localization', localization)
        _PicklableSubplotClassConstructor.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _PicklableSubplotClassConstructor.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _PicklableSubplotClassConstructor.__call__.__dict__.__setitem__('stypy_function_name', '_PicklableSubplotClassConstructor.__call__')
        _PicklableSubplotClassConstructor.__call__.__dict__.__setitem__('stypy_param_names_list', ['axes_class'])
        _PicklableSubplotClassConstructor.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        _PicklableSubplotClassConstructor.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _PicklableSubplotClassConstructor.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _PicklableSubplotClassConstructor.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _PicklableSubplotClassConstructor.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _PicklableSubplotClassConstructor.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_PicklableSubplotClassConstructor.__call__', ['axes_class'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['axes_class'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Assigning a Call to a Name (line 195):
        
        # Assigning a Call to a Name (line 195):
        
        # Call to _PicklableSubplotClassConstructor(...): (line 195)
        # Processing the call keyword arguments (line 195)
        kwargs_217729 = {}
        # Getting the type of '_PicklableSubplotClassConstructor' (line 195)
        _PicklableSubplotClassConstructor_217728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 27), '_PicklableSubplotClassConstructor', False)
        # Calling _PicklableSubplotClassConstructor(args, kwargs) (line 195)
        _PicklableSubplotClassConstructor_call_result_217730 = invoke(stypy.reporting.localization.Localization(__file__, 195, 27), _PicklableSubplotClassConstructor_217728, *[], **kwargs_217729)
        
        # Assigning a type to the variable 'subplot_instance' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'subplot_instance', _PicklableSubplotClassConstructor_call_result_217730)
        
        # Assigning a Call to a Name (line 196):
        
        # Assigning a Call to a Name (line 196):
        
        # Call to subplot_class_factory(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'axes_class' (line 196)
        axes_class_217732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 46), 'axes_class', False)
        # Processing the call keyword arguments (line 196)
        kwargs_217733 = {}
        # Getting the type of 'subplot_class_factory' (line 196)
        subplot_class_factory_217731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 24), 'subplot_class_factory', False)
        # Calling subplot_class_factory(args, kwargs) (line 196)
        subplot_class_factory_call_result_217734 = invoke(stypy.reporting.localization.Localization(__file__, 196, 24), subplot_class_factory_217731, *[axes_class_217732], **kwargs_217733)
        
        # Assigning a type to the variable 'subplot_class' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'subplot_class', subplot_class_factory_call_result_217734)
        
        # Assigning a Name to a Attribute (line 198):
        
        # Assigning a Name to a Attribute (line 198):
        # Getting the type of 'subplot_class' (line 198)
        subplot_class_217735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 37), 'subplot_class')
        # Getting the type of 'subplot_instance' (line 198)
        subplot_instance_217736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'subplot_instance')
        # Setting the type of the member '__class__' of a type (line 198)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), subplot_instance_217736, '__class__', subplot_class_217735)
        # Getting the type of 'subplot_instance' (line 199)
        subplot_instance_217737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 15), 'subplot_instance')
        # Assigning a type to the variable 'stypy_return_type' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'stypy_return_type', subplot_instance_217737)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 193)
        stypy_return_type_217738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_217738)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_217738


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 187, 0, False)
        # Assigning a type to the variable 'self' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_PicklableSubplotClassConstructor.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_PicklableSubplotClassConstructor' (line 187)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), '_PicklableSubplotClassConstructor', _PicklableSubplotClassConstructor)

# Call to update(...): (line 202)
# Processing the call keyword arguments (line 202)

# Call to kwdoc(...): (line 202)
# Processing the call arguments (line 202)
# Getting the type of 'Axes' (line 202)
Axes_217744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 44), 'Axes', False)
# Processing the call keyword arguments (line 202)
kwargs_217745 = {}
# Getting the type of 'martist' (line 202)
martist_217742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 30), 'martist', False)
# Obtaining the member 'kwdoc' of a type (line 202)
kwdoc_217743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 30), martist_217742, 'kwdoc')
# Calling kwdoc(args, kwargs) (line 202)
kwdoc_call_result_217746 = invoke(stypy.reporting.localization.Localization(__file__, 202, 30), kwdoc_217743, *[Axes_217744], **kwargs_217745)

keyword_217747 = kwdoc_call_result_217746
kwargs_217748 = {'Axes': keyword_217747}
# Getting the type of 'docstring' (line 202)
docstring_217739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 0), 'docstring', False)
# Obtaining the member 'interpd' of a type (line 202)
interpd_217740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 0), docstring_217739, 'interpd')
# Obtaining the member 'update' of a type (line 202)
update_217741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 0), interpd_217740, 'update')
# Calling update(args, kwargs) (line 202)
update_call_result_217749 = invoke(stypy.reporting.localization.Localization(__file__, 202, 0), update_217741, *[], **kwargs_217748)


# Call to update(...): (line 203)
# Processing the call keyword arguments (line 203)

# Call to kwdoc(...): (line 203)
# Processing the call arguments (line 203)
# Getting the type of 'Axes' (line 203)
Axes_217755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 47), 'Axes', False)
# Processing the call keyword arguments (line 203)
kwargs_217756 = {}
# Getting the type of 'martist' (line 203)
martist_217753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 33), 'martist', False)
# Obtaining the member 'kwdoc' of a type (line 203)
kwdoc_217754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 33), martist_217753, 'kwdoc')
# Calling kwdoc(args, kwargs) (line 203)
kwdoc_call_result_217757 = invoke(stypy.reporting.localization.Localization(__file__, 203, 33), kwdoc_217754, *[Axes_217755], **kwargs_217756)

keyword_217758 = kwdoc_call_result_217757
kwargs_217759 = {'Subplot': keyword_217758}
# Getting the type of 'docstring' (line 203)
docstring_217750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 0), 'docstring', False)
# Obtaining the member 'interpd' of a type (line 203)
interpd_217751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 0), docstring_217750, 'interpd')
# Obtaining the member 'update' of a type (line 203)
update_217752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 0), interpd_217751, 'update')
# Calling update(args, kwargs) (line 203)
update_call_result_217760 = invoke(stypy.reporting.localization.Localization(__file__, 203, 0), update_217752, *[], **kwargs_217759)

unicode_217761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, (-1)), 'unicode', u'\n# this is some discarded code I was using to find the minimum positive\n# data point for some log scaling fixes.  I realized there was a\n# cleaner way to do it, but am keeping this around as an example for\n# how to get the data out of the axes.  Might want to make something\n# like this a method one day, or better yet make get_verts an Artist\n# method\n\n            minx, maxx = self.get_xlim()\n            if minx<=0 or maxx<=0:\n                # find the min pos value in the data\n                xs = []\n                for line in self.lines:\n                    xs.extend(line.get_xdata(orig=False))\n                for patch in self.patches:\n                    xs.extend([x for x,y in patch.get_verts()])\n                for collection in self.collections:\n                    xs.extend([x for x,y in collection.get_verts()])\n                posx = [x for x in xs if x>0]\n                if len(posx):\n\n                    minx = min(posx)\n                    maxx = max(posx)\n                    # warning, probably breaks inverted axis\n                    self.set_xlim((0.1*minx, maxx))\n\n')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
