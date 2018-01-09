
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: 
6: from .geo import AitoffAxes, HammerAxes, LambertAxes, MollweideAxes
7: from .polar import PolarAxes
8: from matplotlib import axes
9: 
10: class ProjectionRegistry(object):
11:     '''
12:     Manages the set of projections available to the system.
13:     '''
14:     def __init__(self):
15:         self._all_projection_types = {}
16: 
17:     def register(self, *projections):
18:         '''
19:         Register a new set of projection(s).
20:         '''
21:         for projection in projections:
22:             name = projection.name
23:             self._all_projection_types[name] = projection
24: 
25:     def get_projection_class(self, name):
26:         '''
27:         Get a projection class from its *name*.
28:         '''
29:         return self._all_projection_types[name]
30: 
31:     def get_projection_names(self):
32:         '''
33:         Get a list of the names of all projections currently
34:         registered.
35:         '''
36:         return sorted(self._all_projection_types)
37: projection_registry = ProjectionRegistry()
38: 
39: projection_registry.register(
40:     axes.Axes,
41:     PolarAxes,
42:     AitoffAxes,
43:     HammerAxes,
44:     LambertAxes,
45:     MollweideAxes)
46: 
47: 
48: def register_projection(cls):
49:     projection_registry.register(cls)
50: 
51: 
52: def get_projection_class(projection=None):
53:     '''
54:     Get a projection class from its name.
55: 
56:     If *projection* is None, a standard rectilinear projection is
57:     returned.
58:     '''
59:     if projection is None:
60:         projection = 'rectilinear'
61: 
62:     try:
63:         return projection_registry.get_projection_class(projection)
64:     except KeyError:
65:         raise ValueError("Unknown projection '%s'" % projection)
66: 
67: 
68: def process_projection_requirements(figure, *args, **kwargs):
69:     '''
70:     Handle the args/kwargs to for add_axes/add_subplot/gca,
71:     returning::
72: 
73:         (axes_proj_class, proj_class_kwargs, proj_stack_key)
74: 
75:     Which can be used for new axes initialization/identification.
76: 
77:     .. note:: **kwargs** is modified in place.
78: 
79:     '''
80:     ispolar = kwargs.pop('polar', False)
81:     projection = kwargs.pop('projection', None)
82:     if ispolar:
83:         if projection is not None and projection != 'polar':
84:             raise ValueError(
85:                 "polar=True, yet projection=%r. "
86:                 "Only one of these arguments should be supplied." %
87:                 projection)
88:         projection = 'polar'
89: 
90:     if isinstance(projection, six.string_types) or projection is None:
91:         projection_class = get_projection_class(projection)
92:     elif hasattr(projection, '_as_mpl_axes'):
93:         projection_class, extra_kwargs = projection._as_mpl_axes()
94:         kwargs.update(**extra_kwargs)
95:     else:
96:         raise TypeError('projection must be a string, None or implement a '
97:                             '_as_mpl_axes method. Got %r' % projection)
98: 
99:     # Make the key without projection kwargs, this is used as a unique
100:     # lookup for axes instances
101:     key = figure._make_key(*args, **kwargs)
102: 
103:     return projection_class, kwargs, key
104: 
105: 
106: def get_projection_names():
107:     '''
108:     Get a list of acceptable projection names.
109:     '''
110:     return projection_registry.get_projection_names()
111: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/projections/')
import_285298 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_285298) is not StypyTypeError):

    if (import_285298 != 'pyd_module'):
        __import__(import_285298)
        sys_modules_285299 = sys.modules[import_285298]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_285299.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_285298)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/projections/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from matplotlib.projections.geo import AitoffAxes, HammerAxes, LambertAxes, MollweideAxes' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/projections/')
import_285300 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.projections.geo')

if (type(import_285300) is not StypyTypeError):

    if (import_285300 != 'pyd_module'):
        __import__(import_285300)
        sys_modules_285301 = sys.modules[import_285300]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.projections.geo', sys_modules_285301.module_type_store, module_type_store, ['AitoffAxes', 'HammerAxes', 'LambertAxes', 'MollweideAxes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_285301, sys_modules_285301.module_type_store, module_type_store)
    else:
        from matplotlib.projections.geo import AitoffAxes, HammerAxes, LambertAxes, MollweideAxes

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.projections.geo', None, module_type_store, ['AitoffAxes', 'HammerAxes', 'LambertAxes', 'MollweideAxes'], [AitoffAxes, HammerAxes, LambertAxes, MollweideAxes])

else:
    # Assigning a type to the variable 'matplotlib.projections.geo' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.projections.geo', import_285300)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/projections/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from matplotlib.projections.polar import PolarAxes' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/projections/')
import_285302 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.projections.polar')

if (type(import_285302) is not StypyTypeError):

    if (import_285302 != 'pyd_module'):
        __import__(import_285302)
        sys_modules_285303 = sys.modules[import_285302]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.projections.polar', sys_modules_285303.module_type_store, module_type_store, ['PolarAxes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_285303, sys_modules_285303.module_type_store, module_type_store)
    else:
        from matplotlib.projections.polar import PolarAxes

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.projections.polar', None, module_type_store, ['PolarAxes'], [PolarAxes])

else:
    # Assigning a type to the variable 'matplotlib.projections.polar' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.projections.polar', import_285302)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/projections/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from matplotlib import axes' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/projections/')
import_285304 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib')

if (type(import_285304) is not StypyTypeError):

    if (import_285304 != 'pyd_module'):
        __import__(import_285304)
        sys_modules_285305 = sys.modules[import_285304]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib', sys_modules_285305.module_type_store, module_type_store, ['axes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_285305, sys_modules_285305.module_type_store, module_type_store)
    else:
        from matplotlib import axes

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib', None, module_type_store, ['axes'], [axes])

else:
    # Assigning a type to the variable 'matplotlib' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib', import_285304)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/projections/')

# Declaration of the 'ProjectionRegistry' class

class ProjectionRegistry(object, ):
    unicode_285306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, (-1)), 'unicode', u'\n    Manages the set of projections available to the system.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 14, 4, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ProjectionRegistry.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Dict to a Attribute (line 15):
        
        # Assigning a Dict to a Attribute (line 15):
        
        # Obtaining an instance of the builtin type 'dict' (line 15)
        dict_285307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 37), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 15)
        
        # Getting the type of 'self' (line 15)
        self_285308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'self')
        # Setting the type of the member '_all_projection_types' of a type (line 15)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), self_285308, '_all_projection_types', dict_285307)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def register(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'register'
        module_type_store = module_type_store.open_function_context('register', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ProjectionRegistry.register.__dict__.__setitem__('stypy_localization', localization)
        ProjectionRegistry.register.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ProjectionRegistry.register.__dict__.__setitem__('stypy_type_store', module_type_store)
        ProjectionRegistry.register.__dict__.__setitem__('stypy_function_name', 'ProjectionRegistry.register')
        ProjectionRegistry.register.__dict__.__setitem__('stypy_param_names_list', [])
        ProjectionRegistry.register.__dict__.__setitem__('stypy_varargs_param_name', 'projections')
        ProjectionRegistry.register.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ProjectionRegistry.register.__dict__.__setitem__('stypy_call_defaults', defaults)
        ProjectionRegistry.register.__dict__.__setitem__('stypy_call_varargs', varargs)
        ProjectionRegistry.register.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ProjectionRegistry.register.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ProjectionRegistry.register', [], 'projections', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'register', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'register(...)' code ##################

        unicode_285309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, (-1)), 'unicode', u'\n        Register a new set of projection(s).\n        ')
        
        # Getting the type of 'projections' (line 21)
        projections_285310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 26), 'projections')
        # Testing the type of a for loop iterable (line 21)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 21, 8), projections_285310)
        # Getting the type of the for loop variable (line 21)
        for_loop_var_285311 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 21, 8), projections_285310)
        # Assigning a type to the variable 'projection' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'projection', for_loop_var_285311)
        # SSA begins for a for statement (line 21)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Attribute to a Name (line 22):
        
        # Assigning a Attribute to a Name (line 22):
        # Getting the type of 'projection' (line 22)
        projection_285312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 19), 'projection')
        # Obtaining the member 'name' of a type (line 22)
        name_285313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 19), projection_285312, 'name')
        # Assigning a type to the variable 'name' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'name', name_285313)
        
        # Assigning a Name to a Subscript (line 23):
        
        # Assigning a Name to a Subscript (line 23):
        # Getting the type of 'projection' (line 23)
        projection_285314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 47), 'projection')
        # Getting the type of 'self' (line 23)
        self_285315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'self')
        # Obtaining the member '_all_projection_types' of a type (line 23)
        _all_projection_types_285316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 12), self_285315, '_all_projection_types')
        # Getting the type of 'name' (line 23)
        name_285317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 39), 'name')
        # Storing an element on a container (line 23)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 12), _all_projection_types_285316, (name_285317, projection_285314))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'register(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'register' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_285318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_285318)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'register'
        return stypy_return_type_285318


    @norecursion
    def get_projection_class(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_projection_class'
        module_type_store = module_type_store.open_function_context('get_projection_class', 25, 4, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ProjectionRegistry.get_projection_class.__dict__.__setitem__('stypy_localization', localization)
        ProjectionRegistry.get_projection_class.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ProjectionRegistry.get_projection_class.__dict__.__setitem__('stypy_type_store', module_type_store)
        ProjectionRegistry.get_projection_class.__dict__.__setitem__('stypy_function_name', 'ProjectionRegistry.get_projection_class')
        ProjectionRegistry.get_projection_class.__dict__.__setitem__('stypy_param_names_list', ['name'])
        ProjectionRegistry.get_projection_class.__dict__.__setitem__('stypy_varargs_param_name', None)
        ProjectionRegistry.get_projection_class.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ProjectionRegistry.get_projection_class.__dict__.__setitem__('stypy_call_defaults', defaults)
        ProjectionRegistry.get_projection_class.__dict__.__setitem__('stypy_call_varargs', varargs)
        ProjectionRegistry.get_projection_class.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ProjectionRegistry.get_projection_class.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ProjectionRegistry.get_projection_class', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_projection_class', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_projection_class(...)' code ##################

        unicode_285319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, (-1)), 'unicode', u'\n        Get a projection class from its *name*.\n        ')
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 29)
        name_285320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 42), 'name')
        # Getting the type of 'self' (line 29)
        self_285321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'self')
        # Obtaining the member '_all_projection_types' of a type (line 29)
        _all_projection_types_285322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 15), self_285321, '_all_projection_types')
        # Obtaining the member '__getitem__' of a type (line 29)
        getitem___285323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 15), _all_projection_types_285322, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 29)
        subscript_call_result_285324 = invoke(stypy.reporting.localization.Localization(__file__, 29, 15), getitem___285323, name_285320)
        
        # Assigning a type to the variable 'stypy_return_type' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'stypy_return_type', subscript_call_result_285324)
        
        # ################# End of 'get_projection_class(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_projection_class' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_285325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_285325)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_projection_class'
        return stypy_return_type_285325


    @norecursion
    def get_projection_names(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_projection_names'
        module_type_store = module_type_store.open_function_context('get_projection_names', 31, 4, False)
        # Assigning a type to the variable 'self' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ProjectionRegistry.get_projection_names.__dict__.__setitem__('stypy_localization', localization)
        ProjectionRegistry.get_projection_names.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ProjectionRegistry.get_projection_names.__dict__.__setitem__('stypy_type_store', module_type_store)
        ProjectionRegistry.get_projection_names.__dict__.__setitem__('stypy_function_name', 'ProjectionRegistry.get_projection_names')
        ProjectionRegistry.get_projection_names.__dict__.__setitem__('stypy_param_names_list', [])
        ProjectionRegistry.get_projection_names.__dict__.__setitem__('stypy_varargs_param_name', None)
        ProjectionRegistry.get_projection_names.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ProjectionRegistry.get_projection_names.__dict__.__setitem__('stypy_call_defaults', defaults)
        ProjectionRegistry.get_projection_names.__dict__.__setitem__('stypy_call_varargs', varargs)
        ProjectionRegistry.get_projection_names.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ProjectionRegistry.get_projection_names.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ProjectionRegistry.get_projection_names', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_projection_names', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_projection_names(...)' code ##################

        unicode_285326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, (-1)), 'unicode', u'\n        Get a list of the names of all projections currently\n        registered.\n        ')
        
        # Call to sorted(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'self' (line 36)
        self_285328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 22), 'self', False)
        # Obtaining the member '_all_projection_types' of a type (line 36)
        _all_projection_types_285329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 22), self_285328, '_all_projection_types')
        # Processing the call keyword arguments (line 36)
        kwargs_285330 = {}
        # Getting the type of 'sorted' (line 36)
        sorted_285327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'sorted', False)
        # Calling sorted(args, kwargs) (line 36)
        sorted_call_result_285331 = invoke(stypy.reporting.localization.Localization(__file__, 36, 15), sorted_285327, *[_all_projection_types_285329], **kwargs_285330)
        
        # Assigning a type to the variable 'stypy_return_type' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'stypy_return_type', sorted_call_result_285331)
        
        # ################# End of 'get_projection_names(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_projection_names' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_285332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_285332)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_projection_names'
        return stypy_return_type_285332


# Assigning a type to the variable 'ProjectionRegistry' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'ProjectionRegistry', ProjectionRegistry)

# Assigning a Call to a Name (line 37):

# Assigning a Call to a Name (line 37):

# Call to ProjectionRegistry(...): (line 37)
# Processing the call keyword arguments (line 37)
kwargs_285334 = {}
# Getting the type of 'ProjectionRegistry' (line 37)
ProjectionRegistry_285333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 22), 'ProjectionRegistry', False)
# Calling ProjectionRegistry(args, kwargs) (line 37)
ProjectionRegistry_call_result_285335 = invoke(stypy.reporting.localization.Localization(__file__, 37, 22), ProjectionRegistry_285333, *[], **kwargs_285334)

# Assigning a type to the variable 'projection_registry' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'projection_registry', ProjectionRegistry_call_result_285335)

# Call to register(...): (line 39)
# Processing the call arguments (line 39)
# Getting the type of 'axes' (line 40)
axes_285338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'axes', False)
# Obtaining the member 'Axes' of a type (line 40)
Axes_285339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 4), axes_285338, 'Axes')
# Getting the type of 'PolarAxes' (line 41)
PolarAxes_285340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'PolarAxes', False)
# Getting the type of 'AitoffAxes' (line 42)
AitoffAxes_285341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'AitoffAxes', False)
# Getting the type of 'HammerAxes' (line 43)
HammerAxes_285342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'HammerAxes', False)
# Getting the type of 'LambertAxes' (line 44)
LambertAxes_285343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'LambertAxes', False)
# Getting the type of 'MollweideAxes' (line 45)
MollweideAxes_285344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'MollweideAxes', False)
# Processing the call keyword arguments (line 39)
kwargs_285345 = {}
# Getting the type of 'projection_registry' (line 39)
projection_registry_285336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'projection_registry', False)
# Obtaining the member 'register' of a type (line 39)
register_285337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 0), projection_registry_285336, 'register')
# Calling register(args, kwargs) (line 39)
register_call_result_285346 = invoke(stypy.reporting.localization.Localization(__file__, 39, 0), register_285337, *[Axes_285339, PolarAxes_285340, AitoffAxes_285341, HammerAxes_285342, LambertAxes_285343, MollweideAxes_285344], **kwargs_285345)


@norecursion
def register_projection(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'register_projection'
    module_type_store = module_type_store.open_function_context('register_projection', 48, 0, False)
    
    # Passed parameters checking function
    register_projection.stypy_localization = localization
    register_projection.stypy_type_of_self = None
    register_projection.stypy_type_store = module_type_store
    register_projection.stypy_function_name = 'register_projection'
    register_projection.stypy_param_names_list = ['cls']
    register_projection.stypy_varargs_param_name = None
    register_projection.stypy_kwargs_param_name = None
    register_projection.stypy_call_defaults = defaults
    register_projection.stypy_call_varargs = varargs
    register_projection.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'register_projection', ['cls'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'register_projection', localization, ['cls'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'register_projection(...)' code ##################

    
    # Call to register(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'cls' (line 49)
    cls_285349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 33), 'cls', False)
    # Processing the call keyword arguments (line 49)
    kwargs_285350 = {}
    # Getting the type of 'projection_registry' (line 49)
    projection_registry_285347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'projection_registry', False)
    # Obtaining the member 'register' of a type (line 49)
    register_285348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 4), projection_registry_285347, 'register')
    # Calling register(args, kwargs) (line 49)
    register_call_result_285351 = invoke(stypy.reporting.localization.Localization(__file__, 49, 4), register_285348, *[cls_285349], **kwargs_285350)
    
    
    # ################# End of 'register_projection(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'register_projection' in the type store
    # Getting the type of 'stypy_return_type' (line 48)
    stypy_return_type_285352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285352)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'register_projection'
    return stypy_return_type_285352

# Assigning a type to the variable 'register_projection' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'register_projection', register_projection)

@norecursion
def get_projection_class(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 52)
    None_285353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 36), 'None')
    defaults = [None_285353]
    # Create a new context for function 'get_projection_class'
    module_type_store = module_type_store.open_function_context('get_projection_class', 52, 0, False)
    
    # Passed parameters checking function
    get_projection_class.stypy_localization = localization
    get_projection_class.stypy_type_of_self = None
    get_projection_class.stypy_type_store = module_type_store
    get_projection_class.stypy_function_name = 'get_projection_class'
    get_projection_class.stypy_param_names_list = ['projection']
    get_projection_class.stypy_varargs_param_name = None
    get_projection_class.stypy_kwargs_param_name = None
    get_projection_class.stypy_call_defaults = defaults
    get_projection_class.stypy_call_varargs = varargs
    get_projection_class.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_projection_class', ['projection'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_projection_class', localization, ['projection'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_projection_class(...)' code ##################

    unicode_285354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, (-1)), 'unicode', u'\n    Get a projection class from its name.\n\n    If *projection* is None, a standard rectilinear projection is\n    returned.\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 59)
    # Getting the type of 'projection' (line 59)
    projection_285355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 7), 'projection')
    # Getting the type of 'None' (line 59)
    None_285356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), 'None')
    
    (may_be_285357, more_types_in_union_285358) = may_be_none(projection_285355, None_285356)

    if may_be_285357:

        if more_types_in_union_285358:
            # Runtime conditional SSA (line 59)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Str to a Name (line 60):
        
        # Assigning a Str to a Name (line 60):
        unicode_285359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 21), 'unicode', u'rectilinear')
        # Assigning a type to the variable 'projection' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'projection', unicode_285359)

        if more_types_in_union_285358:
            # SSA join for if statement (line 59)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # SSA begins for try-except statement (line 62)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to get_projection_class(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'projection' (line 63)
    projection_285362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 56), 'projection', False)
    # Processing the call keyword arguments (line 63)
    kwargs_285363 = {}
    # Getting the type of 'projection_registry' (line 63)
    projection_registry_285360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 15), 'projection_registry', False)
    # Obtaining the member 'get_projection_class' of a type (line 63)
    get_projection_class_285361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 15), projection_registry_285360, 'get_projection_class')
    # Calling get_projection_class(args, kwargs) (line 63)
    get_projection_class_call_result_285364 = invoke(stypy.reporting.localization.Localization(__file__, 63, 15), get_projection_class_285361, *[projection_285362], **kwargs_285363)
    
    # Assigning a type to the variable 'stypy_return_type' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'stypy_return_type', get_projection_class_call_result_285364)
    # SSA branch for the except part of a try statement (line 62)
    # SSA branch for the except 'KeyError' branch of a try statement (line 62)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 65)
    # Processing the call arguments (line 65)
    unicode_285366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 25), 'unicode', u"Unknown projection '%s'")
    # Getting the type of 'projection' (line 65)
    projection_285367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 53), 'projection', False)
    # Applying the binary operator '%' (line 65)
    result_mod_285368 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 25), '%', unicode_285366, projection_285367)
    
    # Processing the call keyword arguments (line 65)
    kwargs_285369 = {}
    # Getting the type of 'ValueError' (line 65)
    ValueError_285365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 65)
    ValueError_call_result_285370 = invoke(stypy.reporting.localization.Localization(__file__, 65, 14), ValueError_285365, *[result_mod_285368], **kwargs_285369)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 65, 8), ValueError_call_result_285370, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 62)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'get_projection_class(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_projection_class' in the type store
    # Getting the type of 'stypy_return_type' (line 52)
    stypy_return_type_285371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285371)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_projection_class'
    return stypy_return_type_285371

# Assigning a type to the variable 'get_projection_class' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'get_projection_class', get_projection_class)

@norecursion
def process_projection_requirements(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'process_projection_requirements'
    module_type_store = module_type_store.open_function_context('process_projection_requirements', 68, 0, False)
    
    # Passed parameters checking function
    process_projection_requirements.stypy_localization = localization
    process_projection_requirements.stypy_type_of_self = None
    process_projection_requirements.stypy_type_store = module_type_store
    process_projection_requirements.stypy_function_name = 'process_projection_requirements'
    process_projection_requirements.stypy_param_names_list = ['figure']
    process_projection_requirements.stypy_varargs_param_name = 'args'
    process_projection_requirements.stypy_kwargs_param_name = 'kwargs'
    process_projection_requirements.stypy_call_defaults = defaults
    process_projection_requirements.stypy_call_varargs = varargs
    process_projection_requirements.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'process_projection_requirements', ['figure'], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'process_projection_requirements', localization, ['figure'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'process_projection_requirements(...)' code ##################

    unicode_285372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, (-1)), 'unicode', u'\n    Handle the args/kwargs to for add_axes/add_subplot/gca,\n    returning::\n\n        (axes_proj_class, proj_class_kwargs, proj_stack_key)\n\n    Which can be used for new axes initialization/identification.\n\n    .. note:: **kwargs** is modified in place.\n\n    ')
    
    # Assigning a Call to a Name (line 80):
    
    # Assigning a Call to a Name (line 80):
    
    # Call to pop(...): (line 80)
    # Processing the call arguments (line 80)
    unicode_285375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 25), 'unicode', u'polar')
    # Getting the type of 'False' (line 80)
    False_285376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 34), 'False', False)
    # Processing the call keyword arguments (line 80)
    kwargs_285377 = {}
    # Getting the type of 'kwargs' (line 80)
    kwargs_285373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 14), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 80)
    pop_285374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 14), kwargs_285373, 'pop')
    # Calling pop(args, kwargs) (line 80)
    pop_call_result_285378 = invoke(stypy.reporting.localization.Localization(__file__, 80, 14), pop_285374, *[unicode_285375, False_285376], **kwargs_285377)
    
    # Assigning a type to the variable 'ispolar' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'ispolar', pop_call_result_285378)
    
    # Assigning a Call to a Name (line 81):
    
    # Assigning a Call to a Name (line 81):
    
    # Call to pop(...): (line 81)
    # Processing the call arguments (line 81)
    unicode_285381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 28), 'unicode', u'projection')
    # Getting the type of 'None' (line 81)
    None_285382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 42), 'None', False)
    # Processing the call keyword arguments (line 81)
    kwargs_285383 = {}
    # Getting the type of 'kwargs' (line 81)
    kwargs_285379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 17), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 81)
    pop_285380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 17), kwargs_285379, 'pop')
    # Calling pop(args, kwargs) (line 81)
    pop_call_result_285384 = invoke(stypy.reporting.localization.Localization(__file__, 81, 17), pop_285380, *[unicode_285381, None_285382], **kwargs_285383)
    
    # Assigning a type to the variable 'projection' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'projection', pop_call_result_285384)
    
    # Getting the type of 'ispolar' (line 82)
    ispolar_285385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 7), 'ispolar')
    # Testing the type of an if condition (line 82)
    if_condition_285386 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 4), ispolar_285385)
    # Assigning a type to the variable 'if_condition_285386' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'if_condition_285386', if_condition_285386)
    # SSA begins for if statement (line 82)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'projection' (line 83)
    projection_285387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'projection')
    # Getting the type of 'None' (line 83)
    None_285388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 29), 'None')
    # Applying the binary operator 'isnot' (line 83)
    result_is_not_285389 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 11), 'isnot', projection_285387, None_285388)
    
    
    # Getting the type of 'projection' (line 83)
    projection_285390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 38), 'projection')
    unicode_285391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 52), 'unicode', u'polar')
    # Applying the binary operator '!=' (line 83)
    result_ne_285392 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 38), '!=', projection_285390, unicode_285391)
    
    # Applying the binary operator 'and' (line 83)
    result_and_keyword_285393 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 11), 'and', result_is_not_285389, result_ne_285392)
    
    # Testing the type of an if condition (line 83)
    if_condition_285394 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 8), result_and_keyword_285393)
    # Assigning a type to the variable 'if_condition_285394' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'if_condition_285394', if_condition_285394)
    # SSA begins for if statement (line 83)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 84)
    # Processing the call arguments (line 84)
    unicode_285396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 16), 'unicode', u'polar=True, yet projection=%r. Only one of these arguments should be supplied.')
    # Getting the type of 'projection' (line 87)
    projection_285397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'projection', False)
    # Applying the binary operator '%' (line 85)
    result_mod_285398 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 16), '%', unicode_285396, projection_285397)
    
    # Processing the call keyword arguments (line 84)
    kwargs_285399 = {}
    # Getting the type of 'ValueError' (line 84)
    ValueError_285395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 84)
    ValueError_call_result_285400 = invoke(stypy.reporting.localization.Localization(__file__, 84, 18), ValueError_285395, *[result_mod_285398], **kwargs_285399)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 84, 12), ValueError_call_result_285400, 'raise parameter', BaseException)
    # SSA join for if statement (line 83)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Str to a Name (line 88):
    
    # Assigning a Str to a Name (line 88):
    unicode_285401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 21), 'unicode', u'polar')
    # Assigning a type to the variable 'projection' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'projection', unicode_285401)
    # SSA join for if statement (line 82)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'projection' (line 90)
    projection_285403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 18), 'projection', False)
    # Getting the type of 'six' (line 90)
    six_285404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 30), 'six', False)
    # Obtaining the member 'string_types' of a type (line 90)
    string_types_285405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 30), six_285404, 'string_types')
    # Processing the call keyword arguments (line 90)
    kwargs_285406 = {}
    # Getting the type of 'isinstance' (line 90)
    isinstance_285402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 90)
    isinstance_call_result_285407 = invoke(stypy.reporting.localization.Localization(__file__, 90, 7), isinstance_285402, *[projection_285403, string_types_285405], **kwargs_285406)
    
    
    # Getting the type of 'projection' (line 90)
    projection_285408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 51), 'projection')
    # Getting the type of 'None' (line 90)
    None_285409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 65), 'None')
    # Applying the binary operator 'is' (line 90)
    result_is__285410 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 51), 'is', projection_285408, None_285409)
    
    # Applying the binary operator 'or' (line 90)
    result_or_keyword_285411 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 7), 'or', isinstance_call_result_285407, result_is__285410)
    
    # Testing the type of an if condition (line 90)
    if_condition_285412 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 4), result_or_keyword_285411)
    # Assigning a type to the variable 'if_condition_285412' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'if_condition_285412', if_condition_285412)
    # SSA begins for if statement (line 90)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 91):
    
    # Assigning a Call to a Name (line 91):
    
    # Call to get_projection_class(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'projection' (line 91)
    projection_285414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 48), 'projection', False)
    # Processing the call keyword arguments (line 91)
    kwargs_285415 = {}
    # Getting the type of 'get_projection_class' (line 91)
    get_projection_class_285413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 27), 'get_projection_class', False)
    # Calling get_projection_class(args, kwargs) (line 91)
    get_projection_class_call_result_285416 = invoke(stypy.reporting.localization.Localization(__file__, 91, 27), get_projection_class_285413, *[projection_285414], **kwargs_285415)
    
    # Assigning a type to the variable 'projection_class' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'projection_class', get_projection_class_call_result_285416)
    # SSA branch for the else part of an if statement (line 90)
    module_type_store.open_ssa_branch('else')
    
    # Type idiom detected: calculating its left and rigth part (line 92)
    unicode_285417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 29), 'unicode', u'_as_mpl_axes')
    # Getting the type of 'projection' (line 92)
    projection_285418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 17), 'projection')
    
    (may_be_285419, more_types_in_union_285420) = may_provide_member(unicode_285417, projection_285418)

    if may_be_285419:

        if more_types_in_union_285420:
            # Runtime conditional SSA (line 92)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'projection' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 9), 'projection', remove_not_member_provider_from_union(projection_285418, u'_as_mpl_axes'))
        
        # Assigning a Call to a Tuple (line 93):
        
        # Assigning a Call to a Name:
        
        # Call to _as_mpl_axes(...): (line 93)
        # Processing the call keyword arguments (line 93)
        kwargs_285423 = {}
        # Getting the type of 'projection' (line 93)
        projection_285421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 41), 'projection', False)
        # Obtaining the member '_as_mpl_axes' of a type (line 93)
        _as_mpl_axes_285422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 41), projection_285421, '_as_mpl_axes')
        # Calling _as_mpl_axes(args, kwargs) (line 93)
        _as_mpl_axes_call_result_285424 = invoke(stypy.reporting.localization.Localization(__file__, 93, 41), _as_mpl_axes_285422, *[], **kwargs_285423)
        
        # Assigning a type to the variable 'call_assignment_285295' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'call_assignment_285295', _as_mpl_axes_call_result_285424)
        
        # Assigning a Call to a Name (line 93):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_285427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 8), 'int')
        # Processing the call keyword arguments
        kwargs_285428 = {}
        # Getting the type of 'call_assignment_285295' (line 93)
        call_assignment_285295_285425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'call_assignment_285295', False)
        # Obtaining the member '__getitem__' of a type (line 93)
        getitem___285426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), call_assignment_285295_285425, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_285429 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___285426, *[int_285427], **kwargs_285428)
        
        # Assigning a type to the variable 'call_assignment_285296' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'call_assignment_285296', getitem___call_result_285429)
        
        # Assigning a Name to a Name (line 93):
        # Getting the type of 'call_assignment_285296' (line 93)
        call_assignment_285296_285430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'call_assignment_285296')
        # Assigning a type to the variable 'projection_class' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'projection_class', call_assignment_285296_285430)
        
        # Assigning a Call to a Name (line 93):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_285433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 8), 'int')
        # Processing the call keyword arguments
        kwargs_285434 = {}
        # Getting the type of 'call_assignment_285295' (line 93)
        call_assignment_285295_285431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'call_assignment_285295', False)
        # Obtaining the member '__getitem__' of a type (line 93)
        getitem___285432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), call_assignment_285295_285431, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_285435 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___285432, *[int_285433], **kwargs_285434)
        
        # Assigning a type to the variable 'call_assignment_285297' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'call_assignment_285297', getitem___call_result_285435)
        
        # Assigning a Name to a Name (line 93):
        # Getting the type of 'call_assignment_285297' (line 93)
        call_assignment_285297_285436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'call_assignment_285297')
        # Assigning a type to the variable 'extra_kwargs' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 26), 'extra_kwargs', call_assignment_285297_285436)
        
        # Call to update(...): (line 94)
        # Processing the call keyword arguments (line 94)
        # Getting the type of 'extra_kwargs' (line 94)
        extra_kwargs_285439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 24), 'extra_kwargs', False)
        kwargs_285440 = {'extra_kwargs_285439': extra_kwargs_285439}
        # Getting the type of 'kwargs' (line 94)
        kwargs_285437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'kwargs', False)
        # Obtaining the member 'update' of a type (line 94)
        update_285438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), kwargs_285437, 'update')
        # Calling update(args, kwargs) (line 94)
        update_call_result_285441 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), update_285438, *[], **kwargs_285440)
        

        if more_types_in_union_285420:
            # Runtime conditional SSA for else branch (line 92)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_285419) or more_types_in_union_285420):
        # Assigning a type to the variable 'projection' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 9), 'projection', remove_member_provider_from_union(projection_285418, u'_as_mpl_axes'))
        
        # Call to TypeError(...): (line 96)
        # Processing the call arguments (line 96)
        unicode_285443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 24), 'unicode', u'projection must be a string, None or implement a _as_mpl_axes method. Got %r')
        # Getting the type of 'projection' (line 97)
        projection_285444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 60), 'projection', False)
        # Applying the binary operator '%' (line 96)
        result_mod_285445 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 24), '%', unicode_285443, projection_285444)
        
        # Processing the call keyword arguments (line 96)
        kwargs_285446 = {}
        # Getting the type of 'TypeError' (line 96)
        TypeError_285442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 14), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 96)
        TypeError_call_result_285447 = invoke(stypy.reporting.localization.Localization(__file__, 96, 14), TypeError_285442, *[result_mod_285445], **kwargs_285446)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 96, 8), TypeError_call_result_285447, 'raise parameter', BaseException)

        if (may_be_285419 and more_types_in_union_285420):
            # SSA join for if statement (line 92)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 90)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 101):
    
    # Assigning a Call to a Name (line 101):
    
    # Call to _make_key(...): (line 101)
    # Getting the type of 'args' (line 101)
    args_285450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 28), 'args', False)
    # Processing the call keyword arguments (line 101)
    # Getting the type of 'kwargs' (line 101)
    kwargs_285451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 36), 'kwargs', False)
    kwargs_285452 = {'kwargs_285451': kwargs_285451}
    # Getting the type of 'figure' (line 101)
    figure_285448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 10), 'figure', False)
    # Obtaining the member '_make_key' of a type (line 101)
    _make_key_285449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 10), figure_285448, '_make_key')
    # Calling _make_key(args, kwargs) (line 101)
    _make_key_call_result_285453 = invoke(stypy.reporting.localization.Localization(__file__, 101, 10), _make_key_285449, *[args_285450], **kwargs_285452)
    
    # Assigning a type to the variable 'key' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'key', _make_key_call_result_285453)
    
    # Obtaining an instance of the builtin type 'tuple' (line 103)
    tuple_285454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 103)
    # Adding element type (line 103)
    # Getting the type of 'projection_class' (line 103)
    projection_class_285455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 11), 'projection_class')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 11), tuple_285454, projection_class_285455)
    # Adding element type (line 103)
    # Getting the type of 'kwargs' (line 103)
    kwargs_285456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 29), 'kwargs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 11), tuple_285454, kwargs_285456)
    # Adding element type (line 103)
    # Getting the type of 'key' (line 103)
    key_285457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 37), 'key')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 11), tuple_285454, key_285457)
    
    # Assigning a type to the variable 'stypy_return_type' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type', tuple_285454)
    
    # ################# End of 'process_projection_requirements(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'process_projection_requirements' in the type store
    # Getting the type of 'stypy_return_type' (line 68)
    stypy_return_type_285458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285458)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'process_projection_requirements'
    return stypy_return_type_285458

# Assigning a type to the variable 'process_projection_requirements' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'process_projection_requirements', process_projection_requirements)

@norecursion
def get_projection_names(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_projection_names'
    module_type_store = module_type_store.open_function_context('get_projection_names', 106, 0, False)
    
    # Passed parameters checking function
    get_projection_names.stypy_localization = localization
    get_projection_names.stypy_type_of_self = None
    get_projection_names.stypy_type_store = module_type_store
    get_projection_names.stypy_function_name = 'get_projection_names'
    get_projection_names.stypy_param_names_list = []
    get_projection_names.stypy_varargs_param_name = None
    get_projection_names.stypy_kwargs_param_name = None
    get_projection_names.stypy_call_defaults = defaults
    get_projection_names.stypy_call_varargs = varargs
    get_projection_names.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_projection_names', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_projection_names', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_projection_names(...)' code ##################

    unicode_285459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, (-1)), 'unicode', u'\n    Get a list of acceptable projection names.\n    ')
    
    # Call to get_projection_names(...): (line 110)
    # Processing the call keyword arguments (line 110)
    kwargs_285462 = {}
    # Getting the type of 'projection_registry' (line 110)
    projection_registry_285460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'projection_registry', False)
    # Obtaining the member 'get_projection_names' of a type (line 110)
    get_projection_names_285461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 11), projection_registry_285460, 'get_projection_names')
    # Calling get_projection_names(args, kwargs) (line 110)
    get_projection_names_call_result_285463 = invoke(stypy.reporting.localization.Localization(__file__, 110, 11), get_projection_names_285461, *[], **kwargs_285462)
    
    # Assigning a type to the variable 'stypy_return_type' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'stypy_return_type', get_projection_names_call_result_285463)
    
    # ################# End of 'get_projection_names(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_projection_names' in the type store
    # Getting the type of 'stypy_return_type' (line 106)
    stypy_return_type_285464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285464)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_projection_names'
    return stypy_return_type_285464

# Assigning a type to the variable 'get_projection_names' (line 106)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), 'get_projection_names', get_projection_names)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
