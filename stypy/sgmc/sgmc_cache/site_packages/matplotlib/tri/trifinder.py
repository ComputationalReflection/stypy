
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: 
6: from matplotlib.tri import Triangulation
7: import matplotlib._tri as _tri
8: import numpy as np
9: 
10: 
11: class TriFinder(object):
12:     '''
13:     Abstract base class for classes used to find the triangles of a
14:     Triangulation in which (x,y) points lie.
15: 
16:     Rather than instantiate an object of a class derived from TriFinder, it is
17:     usually better to use the function
18:     :func:`matplotlib.tri.Triangulation.get_trifinder`.
19: 
20:     Derived classes implement __call__(x,y) where x,y are array_like point
21:     coordinates of the same shape.
22:     '''
23:     def __init__(self, triangulation):
24:         if not isinstance(triangulation, Triangulation):
25:             raise ValueError('Expected a Triangulation object')
26:         self._triangulation = triangulation
27: 
28: 
29: class TrapezoidMapTriFinder(TriFinder):
30:     '''
31:     :class:`~matplotlib.tri.TriFinder` class implemented using the trapezoid
32:     map algorithm from the book "Computational Geometry, Algorithms and
33:     Applications", second edition, by M. de Berg, M. van Kreveld, M. Overmars
34:     and O. Schwarzkopf.
35: 
36:     The triangulation must be valid, i.e. it must not have duplicate points,
37:     triangles formed from colinear points, or overlapping triangles.  The
38:     algorithm has some tolerance to triangles formed from colinear points, but
39:     this should not be relied upon.
40:     '''
41:     def __init__(self, triangulation):
42:         TriFinder.__init__(self, triangulation)
43:         self._cpp_trifinder = _tri.TrapezoidMapTriFinder(
44:             triangulation.get_cpp_triangulation())
45:         self._initialize()
46: 
47:     def __call__(self, x, y):
48:         '''
49:         Return an array containing the indices of the triangles in which the
50:         specified x,y points lie, or -1 for points that do not lie within a
51:         triangle.
52: 
53:         *x*, *y* are array_like x and y coordinates of the same shape and any
54:         number of dimensions.
55: 
56:         Returns integer array with the same shape and *x* and *y*.
57:         '''
58:         x = np.asarray(x, dtype=np.float64)
59:         y = np.asarray(y, dtype=np.float64)
60:         if x.shape != y.shape:
61:             raise ValueError("x and y must be array-like with the same shape")
62: 
63:         # C++ does the heavy lifting, and expects 1D arrays.
64:         indices = (self._cpp_trifinder.find_many(x.ravel(), y.ravel())
65:                    .reshape(x.shape))
66:         return indices
67: 
68:     def _get_tree_stats(self):
69:         '''
70:         Return a python list containing the statistics about the node tree:
71:             0: number of nodes (tree size)
72:             1: number of unique nodes
73:             2: number of trapezoids (tree leaf nodes)
74:             3: number of unique trapezoids
75:             4: maximum parent count (max number of times a node is repeated in
76:                    tree)
77:             5: maximum depth of tree (one more than the maximum number of
78:                    comparisons needed to search through the tree)
79:             6: mean of all trapezoid depths (one more than the average number
80:                    of comparisons needed to search through the tree)
81:         '''
82:         return self._cpp_trifinder.get_tree_stats()
83: 
84:     def _initialize(self):
85:         '''
86:         Initialize the underlying C++ object.  Can be called multiple times if,
87:         for example, the triangulation is modified.
88:         '''
89:         self._cpp_trifinder.initialize()
90: 
91:     def _print_tree(self):
92:         '''
93:         Print a text representation of the node tree, which is useful for
94:         debugging purposes.
95:         '''
96:         self._cpp_trifinder.print_tree()
97: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_295135 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_295135) is not StypyTypeError):

    if (import_295135 != 'pyd_module'):
        __import__(import_295135)
        sys_modules_295136 = sys.modules[import_295135]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_295136.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_295135)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from matplotlib.tri import Triangulation' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_295137 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.tri')

if (type(import_295137) is not StypyTypeError):

    if (import_295137 != 'pyd_module'):
        __import__(import_295137)
        sys_modules_295138 = sys.modules[import_295137]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.tri', sys_modules_295138.module_type_store, module_type_store, ['Triangulation'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_295138, sys_modules_295138.module_type_store, module_type_store)
    else:
        from matplotlib.tri import Triangulation

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.tri', None, module_type_store, ['Triangulation'], [Triangulation])

else:
    # Assigning a type to the variable 'matplotlib.tri' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.tri', import_295137)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import matplotlib._tri' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_295139 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib._tri')

if (type(import_295139) is not StypyTypeError):

    if (import_295139 != 'pyd_module'):
        __import__(import_295139)
        sys_modules_295140 = sys.modules[import_295139]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), '_tri', sys_modules_295140.module_type_store, module_type_store)
    else:
        import matplotlib._tri as _tri

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), '_tri', matplotlib._tri, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib._tri' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib._tri', import_295139)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import numpy' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/tri/')
import_295141 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_295141) is not StypyTypeError):

    if (import_295141 != 'pyd_module'):
        __import__(import_295141)
        sys_modules_295142 = sys.modules[import_295141]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', sys_modules_295142.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_295141)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/tri/')

# Declaration of the 'TriFinder' class

class TriFinder(object, ):
    unicode_295143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, (-1)), 'unicode', u'\n    Abstract base class for classes used to find the triangles of a\n    Triangulation in which (x,y) points lie.\n\n    Rather than instantiate an object of a class derived from TriFinder, it is\n    usually better to use the function\n    :func:`matplotlib.tri.Triangulation.get_trifinder`.\n\n    Derived classes implement __call__(x,y) where x,y are array_like point\n    coordinates of the same shape.\n    ')

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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TriFinder.__init__', ['triangulation'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['triangulation'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        
        
        # Call to isinstance(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'triangulation' (line 24)
        triangulation_295145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 26), 'triangulation', False)
        # Getting the type of 'Triangulation' (line 24)
        Triangulation_295146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 41), 'Triangulation', False)
        # Processing the call keyword arguments (line 24)
        kwargs_295147 = {}
        # Getting the type of 'isinstance' (line 24)
        isinstance_295144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 24)
        isinstance_call_result_295148 = invoke(stypy.reporting.localization.Localization(__file__, 24, 15), isinstance_295144, *[triangulation_295145, Triangulation_295146], **kwargs_295147)
        
        # Applying the 'not' unary operator (line 24)
        result_not__295149 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 11), 'not', isinstance_call_result_295148)
        
        # Testing the type of an if condition (line 24)
        if_condition_295150 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 24, 8), result_not__295149)
        # Assigning a type to the variable 'if_condition_295150' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'if_condition_295150', if_condition_295150)
        # SSA begins for if statement (line 24)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 25)
        # Processing the call arguments (line 25)
        unicode_295152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 29), 'unicode', u'Expected a Triangulation object')
        # Processing the call keyword arguments (line 25)
        kwargs_295153 = {}
        # Getting the type of 'ValueError' (line 25)
        ValueError_295151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 25)
        ValueError_call_result_295154 = invoke(stypy.reporting.localization.Localization(__file__, 25, 18), ValueError_295151, *[unicode_295152], **kwargs_295153)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 25, 12), ValueError_call_result_295154, 'raise parameter', BaseException)
        # SSA join for if statement (line 24)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 26):
        # Getting the type of 'triangulation' (line 26)
        triangulation_295155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 30), 'triangulation')
        # Getting the type of 'self' (line 26)
        self_295156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self')
        # Setting the type of the member '_triangulation' of a type (line 26)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_295156, '_triangulation', triangulation_295155)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'TriFinder' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'TriFinder', TriFinder)
# Declaration of the 'TrapezoidMapTriFinder' class
# Getting the type of 'TriFinder' (line 29)
TriFinder_295157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 28), 'TriFinder')

class TrapezoidMapTriFinder(TriFinder_295157, ):
    unicode_295158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, (-1)), 'unicode', u'\n    :class:`~matplotlib.tri.TriFinder` class implemented using the trapezoid\n    map algorithm from the book "Computational Geometry, Algorithms and\n    Applications", second edition, by M. de Berg, M. van Kreveld, M. Overmars\n    and O. Schwarzkopf.\n\n    The triangulation must be valid, i.e. it must not have duplicate points,\n    triangles formed from colinear points, or overlapping triangles.  The\n    algorithm has some tolerance to triangles formed from colinear points, but\n    this should not be relied upon.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TrapezoidMapTriFinder.__init__', ['triangulation'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['triangulation'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'self' (line 42)
        self_295161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 27), 'self', False)
        # Getting the type of 'triangulation' (line 42)
        triangulation_295162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 33), 'triangulation', False)
        # Processing the call keyword arguments (line 42)
        kwargs_295163 = {}
        # Getting the type of 'TriFinder' (line 42)
        TriFinder_295159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'TriFinder', False)
        # Obtaining the member '__init__' of a type (line 42)
        init___295160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), TriFinder_295159, '__init__')
        # Calling __init__(args, kwargs) (line 42)
        init___call_result_295164 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), init___295160, *[self_295161, triangulation_295162], **kwargs_295163)
        
        
        # Assigning a Call to a Attribute (line 43):
        
        # Call to TrapezoidMapTriFinder(...): (line 43)
        # Processing the call arguments (line 43)
        
        # Call to get_cpp_triangulation(...): (line 44)
        # Processing the call keyword arguments (line 44)
        kwargs_295169 = {}
        # Getting the type of 'triangulation' (line 44)
        triangulation_295167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'triangulation', False)
        # Obtaining the member 'get_cpp_triangulation' of a type (line 44)
        get_cpp_triangulation_295168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 12), triangulation_295167, 'get_cpp_triangulation')
        # Calling get_cpp_triangulation(args, kwargs) (line 44)
        get_cpp_triangulation_call_result_295170 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), get_cpp_triangulation_295168, *[], **kwargs_295169)
        
        # Processing the call keyword arguments (line 43)
        kwargs_295171 = {}
        # Getting the type of '_tri' (line 43)
        _tri_295165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 30), '_tri', False)
        # Obtaining the member 'TrapezoidMapTriFinder' of a type (line 43)
        TrapezoidMapTriFinder_295166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 30), _tri_295165, 'TrapezoidMapTriFinder')
        # Calling TrapezoidMapTriFinder(args, kwargs) (line 43)
        TrapezoidMapTriFinder_call_result_295172 = invoke(stypy.reporting.localization.Localization(__file__, 43, 30), TrapezoidMapTriFinder_295166, *[get_cpp_triangulation_call_result_295170], **kwargs_295171)
        
        # Getting the type of 'self' (line 43)
        self_295173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self')
        # Setting the type of the member '_cpp_trifinder' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_295173, '_cpp_trifinder', TrapezoidMapTriFinder_call_result_295172)
        
        # Call to _initialize(...): (line 45)
        # Processing the call keyword arguments (line 45)
        kwargs_295176 = {}
        # Getting the type of 'self' (line 45)
        self_295174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self', False)
        # Obtaining the member '_initialize' of a type (line 45)
        _initialize_295175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_295174, '_initialize')
        # Calling _initialize(args, kwargs) (line 45)
        _initialize_call_result_295177 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), _initialize_295175, *[], **kwargs_295176)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 47, 4, False)
        # Assigning a type to the variable 'self' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TrapezoidMapTriFinder.__call__.__dict__.__setitem__('stypy_localization', localization)
        TrapezoidMapTriFinder.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TrapezoidMapTriFinder.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TrapezoidMapTriFinder.__call__.__dict__.__setitem__('stypy_function_name', 'TrapezoidMapTriFinder.__call__')
        TrapezoidMapTriFinder.__call__.__dict__.__setitem__('stypy_param_names_list', ['x', 'y'])
        TrapezoidMapTriFinder.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TrapezoidMapTriFinder.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TrapezoidMapTriFinder.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TrapezoidMapTriFinder.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TrapezoidMapTriFinder.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TrapezoidMapTriFinder.__call__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TrapezoidMapTriFinder.__call__', ['x', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['x', 'y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        unicode_295178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, (-1)), 'unicode', u'\n        Return an array containing the indices of the triangles in which the\n        specified x,y points lie, or -1 for points that do not lie within a\n        triangle.\n\n        *x*, *y* are array_like x and y coordinates of the same shape and any\n        number of dimensions.\n\n        Returns integer array with the same shape and *x* and *y*.\n        ')
        
        # Assigning a Call to a Name (line 58):
        
        # Call to asarray(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'x' (line 58)
        x_295181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 23), 'x', False)
        # Processing the call keyword arguments (line 58)
        # Getting the type of 'np' (line 58)
        np_295182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 32), 'np', False)
        # Obtaining the member 'float64' of a type (line 58)
        float64_295183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 32), np_295182, 'float64')
        keyword_295184 = float64_295183
        kwargs_295185 = {'dtype': keyword_295184}
        # Getting the type of 'np' (line 58)
        np_295179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 58)
        asarray_295180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 12), np_295179, 'asarray')
        # Calling asarray(args, kwargs) (line 58)
        asarray_call_result_295186 = invoke(stypy.reporting.localization.Localization(__file__, 58, 12), asarray_295180, *[x_295181], **kwargs_295185)
        
        # Assigning a type to the variable 'x' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'x', asarray_call_result_295186)
        
        # Assigning a Call to a Name (line 59):
        
        # Call to asarray(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'y' (line 59)
        y_295189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 23), 'y', False)
        # Processing the call keyword arguments (line 59)
        # Getting the type of 'np' (line 59)
        np_295190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 32), 'np', False)
        # Obtaining the member 'float64' of a type (line 59)
        float64_295191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 32), np_295190, 'float64')
        keyword_295192 = float64_295191
        kwargs_295193 = {'dtype': keyword_295192}
        # Getting the type of 'np' (line 59)
        np_295187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 59)
        asarray_295188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 12), np_295187, 'asarray')
        # Calling asarray(args, kwargs) (line 59)
        asarray_call_result_295194 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), asarray_295188, *[y_295189], **kwargs_295193)
        
        # Assigning a type to the variable 'y' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'y', asarray_call_result_295194)
        
        
        # Getting the type of 'x' (line 60)
        x_295195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'x')
        # Obtaining the member 'shape' of a type (line 60)
        shape_295196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 11), x_295195, 'shape')
        # Getting the type of 'y' (line 60)
        y_295197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 22), 'y')
        # Obtaining the member 'shape' of a type (line 60)
        shape_295198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 22), y_295197, 'shape')
        # Applying the binary operator '!=' (line 60)
        result_ne_295199 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 11), '!=', shape_295196, shape_295198)
        
        # Testing the type of an if condition (line 60)
        if_condition_295200 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 8), result_ne_295199)
        # Assigning a type to the variable 'if_condition_295200' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'if_condition_295200', if_condition_295200)
        # SSA begins for if statement (line 60)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 61)
        # Processing the call arguments (line 61)
        unicode_295202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 29), 'unicode', u'x and y must be array-like with the same shape')
        # Processing the call keyword arguments (line 61)
        kwargs_295203 = {}
        # Getting the type of 'ValueError' (line 61)
        ValueError_295201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 61)
        ValueError_call_result_295204 = invoke(stypy.reporting.localization.Localization(__file__, 61, 18), ValueError_295201, *[unicode_295202], **kwargs_295203)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 61, 12), ValueError_call_result_295204, 'raise parameter', BaseException)
        # SSA join for if statement (line 60)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 64):
        
        # Call to reshape(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'x' (line 65)
        x_295219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 28), 'x', False)
        # Obtaining the member 'shape' of a type (line 65)
        shape_295220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 28), x_295219, 'shape')
        # Processing the call keyword arguments (line 64)
        kwargs_295221 = {}
        
        # Call to find_many(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Call to ravel(...): (line 64)
        # Processing the call keyword arguments (line 64)
        kwargs_295210 = {}
        # Getting the type of 'x' (line 64)
        x_295208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 49), 'x', False)
        # Obtaining the member 'ravel' of a type (line 64)
        ravel_295209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 49), x_295208, 'ravel')
        # Calling ravel(args, kwargs) (line 64)
        ravel_call_result_295211 = invoke(stypy.reporting.localization.Localization(__file__, 64, 49), ravel_295209, *[], **kwargs_295210)
        
        
        # Call to ravel(...): (line 64)
        # Processing the call keyword arguments (line 64)
        kwargs_295214 = {}
        # Getting the type of 'y' (line 64)
        y_295212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 60), 'y', False)
        # Obtaining the member 'ravel' of a type (line 64)
        ravel_295213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 60), y_295212, 'ravel')
        # Calling ravel(args, kwargs) (line 64)
        ravel_call_result_295215 = invoke(stypy.reporting.localization.Localization(__file__, 64, 60), ravel_295213, *[], **kwargs_295214)
        
        # Processing the call keyword arguments (line 64)
        kwargs_295216 = {}
        # Getting the type of 'self' (line 64)
        self_295205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 19), 'self', False)
        # Obtaining the member '_cpp_trifinder' of a type (line 64)
        _cpp_trifinder_295206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 19), self_295205, '_cpp_trifinder')
        # Obtaining the member 'find_many' of a type (line 64)
        find_many_295207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 19), _cpp_trifinder_295206, 'find_many')
        # Calling find_many(args, kwargs) (line 64)
        find_many_call_result_295217 = invoke(stypy.reporting.localization.Localization(__file__, 64, 19), find_many_295207, *[ravel_call_result_295211, ravel_call_result_295215], **kwargs_295216)
        
        # Obtaining the member 'reshape' of a type (line 64)
        reshape_295218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 19), find_many_call_result_295217, 'reshape')
        # Calling reshape(args, kwargs) (line 64)
        reshape_call_result_295222 = invoke(stypy.reporting.localization.Localization(__file__, 64, 19), reshape_295218, *[shape_295220], **kwargs_295221)
        
        # Assigning a type to the variable 'indices' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'indices', reshape_call_result_295222)
        # Getting the type of 'indices' (line 66)
        indices_295223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 15), 'indices')
        # Assigning a type to the variable 'stypy_return_type' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'stypy_return_type', indices_295223)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 47)
        stypy_return_type_295224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_295224)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_295224


    @norecursion
    def _get_tree_stats(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_tree_stats'
        module_type_store = module_type_store.open_function_context('_get_tree_stats', 68, 4, False)
        # Assigning a type to the variable 'self' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TrapezoidMapTriFinder._get_tree_stats.__dict__.__setitem__('stypy_localization', localization)
        TrapezoidMapTriFinder._get_tree_stats.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TrapezoidMapTriFinder._get_tree_stats.__dict__.__setitem__('stypy_type_store', module_type_store)
        TrapezoidMapTriFinder._get_tree_stats.__dict__.__setitem__('stypy_function_name', 'TrapezoidMapTriFinder._get_tree_stats')
        TrapezoidMapTriFinder._get_tree_stats.__dict__.__setitem__('stypy_param_names_list', [])
        TrapezoidMapTriFinder._get_tree_stats.__dict__.__setitem__('stypy_varargs_param_name', None)
        TrapezoidMapTriFinder._get_tree_stats.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TrapezoidMapTriFinder._get_tree_stats.__dict__.__setitem__('stypy_call_defaults', defaults)
        TrapezoidMapTriFinder._get_tree_stats.__dict__.__setitem__('stypy_call_varargs', varargs)
        TrapezoidMapTriFinder._get_tree_stats.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TrapezoidMapTriFinder._get_tree_stats.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TrapezoidMapTriFinder._get_tree_stats', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_tree_stats', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_tree_stats(...)' code ##################

        unicode_295225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, (-1)), 'unicode', u'\n        Return a python list containing the statistics about the node tree:\n            0: number of nodes (tree size)\n            1: number of unique nodes\n            2: number of trapezoids (tree leaf nodes)\n            3: number of unique trapezoids\n            4: maximum parent count (max number of times a node is repeated in\n                   tree)\n            5: maximum depth of tree (one more than the maximum number of\n                   comparisons needed to search through the tree)\n            6: mean of all trapezoid depths (one more than the average number\n                   of comparisons needed to search through the tree)\n        ')
        
        # Call to get_tree_stats(...): (line 82)
        # Processing the call keyword arguments (line 82)
        kwargs_295229 = {}
        # Getting the type of 'self' (line 82)
        self_295226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'self', False)
        # Obtaining the member '_cpp_trifinder' of a type (line 82)
        _cpp_trifinder_295227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 15), self_295226, '_cpp_trifinder')
        # Obtaining the member 'get_tree_stats' of a type (line 82)
        get_tree_stats_295228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 15), _cpp_trifinder_295227, 'get_tree_stats')
        # Calling get_tree_stats(args, kwargs) (line 82)
        get_tree_stats_call_result_295230 = invoke(stypy.reporting.localization.Localization(__file__, 82, 15), get_tree_stats_295228, *[], **kwargs_295229)
        
        # Assigning a type to the variable 'stypy_return_type' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'stypy_return_type', get_tree_stats_call_result_295230)
        
        # ################# End of '_get_tree_stats(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_tree_stats' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_295231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_295231)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_tree_stats'
        return stypy_return_type_295231


    @norecursion
    def _initialize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_initialize'
        module_type_store = module_type_store.open_function_context('_initialize', 84, 4, False)
        # Assigning a type to the variable 'self' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TrapezoidMapTriFinder._initialize.__dict__.__setitem__('stypy_localization', localization)
        TrapezoidMapTriFinder._initialize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TrapezoidMapTriFinder._initialize.__dict__.__setitem__('stypy_type_store', module_type_store)
        TrapezoidMapTriFinder._initialize.__dict__.__setitem__('stypy_function_name', 'TrapezoidMapTriFinder._initialize')
        TrapezoidMapTriFinder._initialize.__dict__.__setitem__('stypy_param_names_list', [])
        TrapezoidMapTriFinder._initialize.__dict__.__setitem__('stypy_varargs_param_name', None)
        TrapezoidMapTriFinder._initialize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TrapezoidMapTriFinder._initialize.__dict__.__setitem__('stypy_call_defaults', defaults)
        TrapezoidMapTriFinder._initialize.__dict__.__setitem__('stypy_call_varargs', varargs)
        TrapezoidMapTriFinder._initialize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TrapezoidMapTriFinder._initialize.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TrapezoidMapTriFinder._initialize', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_initialize', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_initialize(...)' code ##################

        unicode_295232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, (-1)), 'unicode', u'\n        Initialize the underlying C++ object.  Can be called multiple times if,\n        for example, the triangulation is modified.\n        ')
        
        # Call to initialize(...): (line 89)
        # Processing the call keyword arguments (line 89)
        kwargs_295236 = {}
        # Getting the type of 'self' (line 89)
        self_295233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self', False)
        # Obtaining the member '_cpp_trifinder' of a type (line 89)
        _cpp_trifinder_295234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_295233, '_cpp_trifinder')
        # Obtaining the member 'initialize' of a type (line 89)
        initialize_295235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), _cpp_trifinder_295234, 'initialize')
        # Calling initialize(args, kwargs) (line 89)
        initialize_call_result_295237 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), initialize_295235, *[], **kwargs_295236)
        
        
        # ################# End of '_initialize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_initialize' in the type store
        # Getting the type of 'stypy_return_type' (line 84)
        stypy_return_type_295238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_295238)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_initialize'
        return stypy_return_type_295238


    @norecursion
    def _print_tree(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_print_tree'
        module_type_store = module_type_store.open_function_context('_print_tree', 91, 4, False)
        # Assigning a type to the variable 'self' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TrapezoidMapTriFinder._print_tree.__dict__.__setitem__('stypy_localization', localization)
        TrapezoidMapTriFinder._print_tree.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TrapezoidMapTriFinder._print_tree.__dict__.__setitem__('stypy_type_store', module_type_store)
        TrapezoidMapTriFinder._print_tree.__dict__.__setitem__('stypy_function_name', 'TrapezoidMapTriFinder._print_tree')
        TrapezoidMapTriFinder._print_tree.__dict__.__setitem__('stypy_param_names_list', [])
        TrapezoidMapTriFinder._print_tree.__dict__.__setitem__('stypy_varargs_param_name', None)
        TrapezoidMapTriFinder._print_tree.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TrapezoidMapTriFinder._print_tree.__dict__.__setitem__('stypy_call_defaults', defaults)
        TrapezoidMapTriFinder._print_tree.__dict__.__setitem__('stypy_call_varargs', varargs)
        TrapezoidMapTriFinder._print_tree.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TrapezoidMapTriFinder._print_tree.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TrapezoidMapTriFinder._print_tree', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_print_tree', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_print_tree(...)' code ##################

        unicode_295239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, (-1)), 'unicode', u'\n        Print a text representation of the node tree, which is useful for\n        debugging purposes.\n        ')
        
        # Call to print_tree(...): (line 96)
        # Processing the call keyword arguments (line 96)
        kwargs_295243 = {}
        # Getting the type of 'self' (line 96)
        self_295240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'self', False)
        # Obtaining the member '_cpp_trifinder' of a type (line 96)
        _cpp_trifinder_295241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), self_295240, '_cpp_trifinder')
        # Obtaining the member 'print_tree' of a type (line 96)
        print_tree_295242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), _cpp_trifinder_295241, 'print_tree')
        # Calling print_tree(args, kwargs) (line 96)
        print_tree_call_result_295244 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), print_tree_295242, *[], **kwargs_295243)
        
        
        # ################# End of '_print_tree(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_print_tree' in the type store
        # Getting the type of 'stypy_return_type' (line 91)
        stypy_return_type_295245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_295245)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_print_tree'
        return stypy_return_type_295245


# Assigning a type to the variable 'TrapezoidMapTriFinder' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'TrapezoidMapTriFinder', TrapezoidMapTriFinder)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
