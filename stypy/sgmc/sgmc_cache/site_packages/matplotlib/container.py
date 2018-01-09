
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: 
6: import matplotlib.cbook as cbook
7: import matplotlib.artist as martist
8: 
9: 
10: class Container(tuple):
11:     '''
12:     Base class for containers.
13:     '''
14: 
15:     def __repr__(self):
16:         return "<Container object of %d artists>" % (len(self))
17: 
18:     def __new__(cls, *kl, **kwargs):
19:         return tuple.__new__(cls, kl[0])
20: 
21:     def __init__(self, kl, label=None):
22: 
23:         self.eventson = False  # fire events only if eventson
24:         self._oid = 0  # an observer id
25:         self._propobservers = {}  # a dict from oids to funcs
26: 
27:         self._remove_method = None
28: 
29:         self.set_label(label)
30: 
31:     def set_remove_method(self, f):
32:         self._remove_method = f
33: 
34:     def remove(self):
35:         for c in cbook.flatten(
36:                 self, scalarp=lambda x: isinstance(x, martist.Artist)):
37:             if c is not None:
38:                 c.remove()
39: 
40:         if self._remove_method:
41:             self._remove_method(self)
42: 
43:     def __getstate__(self):
44:         d = self.__dict__.copy()
45:         # remove the unpicklable remove method, this will get re-added on load
46:         # (by the axes) if the artist lives on an axes.
47:         d['_remove_method'] = None
48:         return d
49: 
50:     def get_label(self):
51:         '''
52:         Get the label used for this artist in the legend.
53:         '''
54:         return self._label
55: 
56:     def set_label(self, s):
57:         '''
58:         Set the label to *s* for auto legend.
59: 
60:         ACCEPTS: string or anything printable with '%s' conversion.
61:         '''
62:         if s is not None:
63:             self._label = '%s' % (s, )
64:         else:
65:             self._label = None
66:         self.pchanged()
67: 
68:     def add_callback(self, func):
69:         '''
70:         Adds a callback function that will be called whenever one of
71:         the :class:`Artist`'s properties changes.
72: 
73:         Returns an *id* that is useful for removing the callback with
74:         :meth:`remove_callback` later.
75:         '''
76:         oid = self._oid
77:         self._propobservers[oid] = func
78:         self._oid += 1
79:         return oid
80: 
81:     def remove_callback(self, oid):
82:         '''
83:         Remove a callback based on its *id*.
84: 
85:         .. seealso::
86: 
87:             :meth:`add_callback`
88:                For adding callbacks
89: 
90:         '''
91:         try:
92:             del self._propobservers[oid]
93:         except KeyError:
94:             pass
95: 
96:     def pchanged(self):
97:         '''
98:         Fire an event when property changed, calling all of the
99:         registered callbacks.
100:         '''
101:         for oid, func in list(six.iteritems(self._propobservers)):
102:             func(self)
103: 
104:     def get_children(self):
105:         return [child for child in cbook.flatten(self) if child is not None]
106: 
107: 
108: class BarContainer(Container):
109: 
110:     def __init__(self, patches, errorbar=None, **kwargs):
111:         self.patches = patches
112:         self.errorbar = errorbar
113:         Container.__init__(self, patches, **kwargs)
114: 
115: 
116: class ErrorbarContainer(Container):
117: 
118:     def __init__(self, lines, has_xerr=False, has_yerr=False, **kwargs):
119:         self.lines = lines
120:         self.has_xerr = has_xerr
121:         self.has_yerr = has_yerr
122:         Container.__init__(self, lines, **kwargs)
123: 
124: 
125: class StemContainer(Container):
126: 
127:     def __init__(self, markerline_stemlines_baseline, **kwargs):
128:         markerline, stemlines, baseline = markerline_stemlines_baseline
129:         self.markerline = markerline
130:         self.stemlines = stemlines
131:         self.baseline = baseline
132:         Container.__init__(self, markerline_stemlines_baseline, **kwargs)
133: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_39072 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_39072) is not StypyTypeError):

    if (import_39072 != 'pyd_module'):
        __import__(import_39072)
        sys_modules_39073 = sys.modules[import_39072]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_39073.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_39072)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import matplotlib.cbook' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_39074 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.cbook')

if (type(import_39074) is not StypyTypeError):

    if (import_39074 != 'pyd_module'):
        __import__(import_39074)
        sys_modules_39075 = sys.modules[import_39074]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'cbook', sys_modules_39075.module_type_store, module_type_store)
    else:
        import matplotlib.cbook as cbook

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'cbook', matplotlib.cbook, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib.cbook', import_39074)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import matplotlib.artist' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_39076 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.artist')

if (type(import_39076) is not StypyTypeError):

    if (import_39076 != 'pyd_module'):
        __import__(import_39076)
        sys_modules_39077 = sys.modules[import_39076]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'martist', sys_modules_39077.module_type_store, module_type_store)
    else:
        import matplotlib.artist as martist

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'martist', matplotlib.artist, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.artist' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.artist', import_39076)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

# Declaration of the 'Container' class
# Getting the type of 'tuple' (line 10)
tuple_39078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 16), 'tuple')

class Container(tuple_39078, ):
    unicode_39079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, (-1)), 'unicode', u'\n    Base class for containers.\n    ')

    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Container.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        Container.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Container.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Container.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Container.stypy__repr__')
        Container.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        Container.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Container.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Container.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Container.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Container.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Container.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Container.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        unicode_39080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 15), 'unicode', u'<Container object of %d artists>')
        
        # Call to len(...): (line 16)
        # Processing the call arguments (line 16)
        # Getting the type of 'self' (line 16)
        self_39082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 57), 'self', False)
        # Processing the call keyword arguments (line 16)
        kwargs_39083 = {}
        # Getting the type of 'len' (line 16)
        len_39081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 53), 'len', False)
        # Calling len(args, kwargs) (line 16)
        len_call_result_39084 = invoke(stypy.reporting.localization.Localization(__file__, 16, 53), len_39081, *[self_39082], **kwargs_39083)
        
        # Applying the binary operator '%' (line 16)
        result_mod_39085 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 15), '%', unicode_39080, len_call_result_39084)
        
        # Assigning a type to the variable 'stypy_return_type' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'stypy_return_type', result_mod_39085)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_39086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39086)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_39086


    @norecursion
    def __new__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__new__'
        module_type_store = module_type_store.open_function_context('__new__', 18, 4, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Container.__new__.__dict__.__setitem__('stypy_localization', localization)
        Container.__new__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Container.__new__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Container.__new__.__dict__.__setitem__('stypy_function_name', 'Container.__new__')
        Container.__new__.__dict__.__setitem__('stypy_param_names_list', [])
        Container.__new__.__dict__.__setitem__('stypy_varargs_param_name', 'kl')
        Container.__new__.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        Container.__new__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Container.__new__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Container.__new__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Container.__new__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Container.__new__', [], 'kl', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__new__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__new__(...)' code ##################

        
        # Call to __new__(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'cls' (line 19)
        cls_39089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 29), 'cls', False)
        
        # Obtaining the type of the subscript
        int_39090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 37), 'int')
        # Getting the type of 'kl' (line 19)
        kl_39091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 34), 'kl', False)
        # Obtaining the member '__getitem__' of a type (line 19)
        getitem___39092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 34), kl_39091, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 19)
        subscript_call_result_39093 = invoke(stypy.reporting.localization.Localization(__file__, 19, 34), getitem___39092, int_39090)
        
        # Processing the call keyword arguments (line 19)
        kwargs_39094 = {}
        # Getting the type of 'tuple' (line 19)
        tuple_39087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'tuple', False)
        # Obtaining the member '__new__' of a type (line 19)
        new___39088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 15), tuple_39087, '__new__')
        # Calling __new__(args, kwargs) (line 19)
        new___call_result_39095 = invoke(stypy.reporting.localization.Localization(__file__, 19, 15), new___39088, *[cls_39089, subscript_call_result_39093], **kwargs_39094)
        
        # Assigning a type to the variable 'stypy_return_type' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'stypy_return_type', new___call_result_39095)
        
        # ################# End of '__new__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__new__' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_39096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39096)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__new__'
        return stypy_return_type_39096


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 21)
        None_39097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 33), 'None')
        defaults = [None_39097]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 21, 4, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Container.__init__', ['kl', 'label'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['kl', 'label'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 23):
        
        # Assigning a Name to a Attribute (line 23):
        # Getting the type of 'False' (line 23)
        False_39098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), 'False')
        # Getting the type of 'self' (line 23)
        self_39099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self')
        # Setting the type of the member 'eventson' of a type (line 23)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), self_39099, 'eventson', False_39098)
        
        # Assigning a Num to a Attribute (line 24):
        
        # Assigning a Num to a Attribute (line 24):
        int_39100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 20), 'int')
        # Getting the type of 'self' (line 24)
        self_39101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'self')
        # Setting the type of the member '_oid' of a type (line 24)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), self_39101, '_oid', int_39100)
        
        # Assigning a Dict to a Attribute (line 25):
        
        # Assigning a Dict to a Attribute (line 25):
        
        # Obtaining an instance of the builtin type 'dict' (line 25)
        dict_39102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 30), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 25)
        
        # Getting the type of 'self' (line 25)
        self_39103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self')
        # Setting the type of the member '_propobservers' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_39103, '_propobservers', dict_39102)
        
        # Assigning a Name to a Attribute (line 27):
        
        # Assigning a Name to a Attribute (line 27):
        # Getting the type of 'None' (line 27)
        None_39104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 30), 'None')
        # Getting the type of 'self' (line 27)
        self_39105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self')
        # Setting the type of the member '_remove_method' of a type (line 27)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), self_39105, '_remove_method', None_39104)
        
        # Call to set_label(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'label' (line 29)
        label_39108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 23), 'label', False)
        # Processing the call keyword arguments (line 29)
        kwargs_39109 = {}
        # Getting the type of 'self' (line 29)
        self_39106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self', False)
        # Obtaining the member 'set_label' of a type (line 29)
        set_label_39107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_39106, 'set_label')
        # Calling set_label(args, kwargs) (line 29)
        set_label_call_result_39110 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), set_label_39107, *[label_39108], **kwargs_39109)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_remove_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_remove_method'
        module_type_store = module_type_store.open_function_context('set_remove_method', 31, 4, False)
        # Assigning a type to the variable 'self' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Container.set_remove_method.__dict__.__setitem__('stypy_localization', localization)
        Container.set_remove_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Container.set_remove_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        Container.set_remove_method.__dict__.__setitem__('stypy_function_name', 'Container.set_remove_method')
        Container.set_remove_method.__dict__.__setitem__('stypy_param_names_list', ['f'])
        Container.set_remove_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        Container.set_remove_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Container.set_remove_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        Container.set_remove_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        Container.set_remove_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Container.set_remove_method.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Container.set_remove_method', ['f'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_remove_method', localization, ['f'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_remove_method(...)' code ##################

        
        # Assigning a Name to a Attribute (line 32):
        
        # Assigning a Name to a Attribute (line 32):
        # Getting the type of 'f' (line 32)
        f_39111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 30), 'f')
        # Getting the type of 'self' (line 32)
        self_39112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self')
        # Setting the type of the member '_remove_method' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), self_39112, '_remove_method', f_39111)
        
        # ################# End of 'set_remove_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_remove_method' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_39113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39113)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_remove_method'
        return stypy_return_type_39113


    @norecursion
    def remove(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'remove'
        module_type_store = module_type_store.open_function_context('remove', 34, 4, False)
        # Assigning a type to the variable 'self' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Container.remove.__dict__.__setitem__('stypy_localization', localization)
        Container.remove.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Container.remove.__dict__.__setitem__('stypy_type_store', module_type_store)
        Container.remove.__dict__.__setitem__('stypy_function_name', 'Container.remove')
        Container.remove.__dict__.__setitem__('stypy_param_names_list', [])
        Container.remove.__dict__.__setitem__('stypy_varargs_param_name', None)
        Container.remove.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Container.remove.__dict__.__setitem__('stypy_call_defaults', defaults)
        Container.remove.__dict__.__setitem__('stypy_call_varargs', varargs)
        Container.remove.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Container.remove.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Container.remove', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'remove', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'remove(...)' code ##################

        
        
        # Call to flatten(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'self' (line 36)
        self_39116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'self', False)
        # Processing the call keyword arguments (line 35)

        @norecursion
        def _stypy_temp_lambda_9(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_9'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_9', 36, 30, True)
            # Passed parameters checking function
            _stypy_temp_lambda_9.stypy_localization = localization
            _stypy_temp_lambda_9.stypy_type_of_self = None
            _stypy_temp_lambda_9.stypy_type_store = module_type_store
            _stypy_temp_lambda_9.stypy_function_name = '_stypy_temp_lambda_9'
            _stypy_temp_lambda_9.stypy_param_names_list = ['x']
            _stypy_temp_lambda_9.stypy_varargs_param_name = None
            _stypy_temp_lambda_9.stypy_kwargs_param_name = None
            _stypy_temp_lambda_9.stypy_call_defaults = defaults
            _stypy_temp_lambda_9.stypy_call_varargs = varargs
            _stypy_temp_lambda_9.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_9', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_9', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to isinstance(...): (line 36)
            # Processing the call arguments (line 36)
            # Getting the type of 'x' (line 36)
            x_39118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 51), 'x', False)
            # Getting the type of 'martist' (line 36)
            martist_39119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 54), 'martist', False)
            # Obtaining the member 'Artist' of a type (line 36)
            Artist_39120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 54), martist_39119, 'Artist')
            # Processing the call keyword arguments (line 36)
            kwargs_39121 = {}
            # Getting the type of 'isinstance' (line 36)
            isinstance_39117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 40), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 36)
            isinstance_call_result_39122 = invoke(stypy.reporting.localization.Localization(__file__, 36, 40), isinstance_39117, *[x_39118, Artist_39120], **kwargs_39121)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 36)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 30), 'stypy_return_type', isinstance_call_result_39122)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_9' in the type store
            # Getting the type of 'stypy_return_type' (line 36)
            stypy_return_type_39123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 30), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_39123)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_9'
            return stypy_return_type_39123

        # Assigning a type to the variable '_stypy_temp_lambda_9' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 30), '_stypy_temp_lambda_9', _stypy_temp_lambda_9)
        # Getting the type of '_stypy_temp_lambda_9' (line 36)
        _stypy_temp_lambda_9_39124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 30), '_stypy_temp_lambda_9')
        keyword_39125 = _stypy_temp_lambda_9_39124
        kwargs_39126 = {'scalarp': keyword_39125}
        # Getting the type of 'cbook' (line 35)
        cbook_39114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 17), 'cbook', False)
        # Obtaining the member 'flatten' of a type (line 35)
        flatten_39115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 17), cbook_39114, 'flatten')
        # Calling flatten(args, kwargs) (line 35)
        flatten_call_result_39127 = invoke(stypy.reporting.localization.Localization(__file__, 35, 17), flatten_39115, *[self_39116], **kwargs_39126)
        
        # Testing the type of a for loop iterable (line 35)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 35, 8), flatten_call_result_39127)
        # Getting the type of the for loop variable (line 35)
        for_loop_var_39128 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 35, 8), flatten_call_result_39127)
        # Assigning a type to the variable 'c' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'c', for_loop_var_39128)
        # SSA begins for a for statement (line 35)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 37)
        # Getting the type of 'c' (line 37)
        c_39129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'c')
        # Getting the type of 'None' (line 37)
        None_39130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 24), 'None')
        
        (may_be_39131, more_types_in_union_39132) = may_not_be_none(c_39129, None_39130)

        if may_be_39131:

            if more_types_in_union_39132:
                # Runtime conditional SSA (line 37)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to remove(...): (line 38)
            # Processing the call keyword arguments (line 38)
            kwargs_39135 = {}
            # Getting the type of 'c' (line 38)
            c_39133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'c', False)
            # Obtaining the member 'remove' of a type (line 38)
            remove_39134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 16), c_39133, 'remove')
            # Calling remove(args, kwargs) (line 38)
            remove_call_result_39136 = invoke(stypy.reporting.localization.Localization(__file__, 38, 16), remove_39134, *[], **kwargs_39135)
            

            if more_types_in_union_39132:
                # SSA join for if statement (line 37)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 40)
        self_39137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'self')
        # Obtaining the member '_remove_method' of a type (line 40)
        _remove_method_39138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 11), self_39137, '_remove_method')
        # Testing the type of an if condition (line 40)
        if_condition_39139 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 8), _remove_method_39138)
        # Assigning a type to the variable 'if_condition_39139' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'if_condition_39139', if_condition_39139)
        # SSA begins for if statement (line 40)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _remove_method(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'self' (line 41)
        self_39142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 32), 'self', False)
        # Processing the call keyword arguments (line 41)
        kwargs_39143 = {}
        # Getting the type of 'self' (line 41)
        self_39140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'self', False)
        # Obtaining the member '_remove_method' of a type (line 41)
        _remove_method_39141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 12), self_39140, '_remove_method')
        # Calling _remove_method(args, kwargs) (line 41)
        _remove_method_call_result_39144 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), _remove_method_39141, *[self_39142], **kwargs_39143)
        
        # SSA join for if statement (line 40)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'remove(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'remove' in the type store
        # Getting the type of 'stypy_return_type' (line 34)
        stypy_return_type_39145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39145)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'remove'
        return stypy_return_type_39145


    @norecursion
    def __getstate__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getstate__'
        module_type_store = module_type_store.open_function_context('__getstate__', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Container.__getstate__.__dict__.__setitem__('stypy_localization', localization)
        Container.__getstate__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Container.__getstate__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Container.__getstate__.__dict__.__setitem__('stypy_function_name', 'Container.__getstate__')
        Container.__getstate__.__dict__.__setitem__('stypy_param_names_list', [])
        Container.__getstate__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Container.__getstate__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Container.__getstate__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Container.__getstate__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Container.__getstate__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Container.__getstate__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Container.__getstate__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getstate__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getstate__(...)' code ##################

        
        # Assigning a Call to a Name (line 44):
        
        # Assigning a Call to a Name (line 44):
        
        # Call to copy(...): (line 44)
        # Processing the call keyword arguments (line 44)
        kwargs_39149 = {}
        # Getting the type of 'self' (line 44)
        self_39146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'self', False)
        # Obtaining the member '__dict__' of a type (line 44)
        dict___39147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 12), self_39146, '__dict__')
        # Obtaining the member 'copy' of a type (line 44)
        copy_39148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 12), dict___39147, 'copy')
        # Calling copy(args, kwargs) (line 44)
        copy_call_result_39150 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), copy_39148, *[], **kwargs_39149)
        
        # Assigning a type to the variable 'd' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'd', copy_call_result_39150)
        
        # Assigning a Name to a Subscript (line 47):
        
        # Assigning a Name to a Subscript (line 47):
        # Getting the type of 'None' (line 47)
        None_39151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 30), 'None')
        # Getting the type of 'd' (line 47)
        d_39152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'd')
        unicode_39153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 10), 'unicode', u'_remove_method')
        # Storing an element on a container (line 47)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 8), d_39152, (unicode_39153, None_39151))
        # Getting the type of 'd' (line 48)
        d_39154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'd')
        # Assigning a type to the variable 'stypy_return_type' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'stypy_return_type', d_39154)
        
        # ################# End of '__getstate__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getstate__' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_39155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39155)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getstate__'
        return stypy_return_type_39155


    @norecursion
    def get_label(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_label'
        module_type_store = module_type_store.open_function_context('get_label', 50, 4, False)
        # Assigning a type to the variable 'self' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Container.get_label.__dict__.__setitem__('stypy_localization', localization)
        Container.get_label.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Container.get_label.__dict__.__setitem__('stypy_type_store', module_type_store)
        Container.get_label.__dict__.__setitem__('stypy_function_name', 'Container.get_label')
        Container.get_label.__dict__.__setitem__('stypy_param_names_list', [])
        Container.get_label.__dict__.__setitem__('stypy_varargs_param_name', None)
        Container.get_label.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Container.get_label.__dict__.__setitem__('stypy_call_defaults', defaults)
        Container.get_label.__dict__.__setitem__('stypy_call_varargs', varargs)
        Container.get_label.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Container.get_label.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Container.get_label', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_label', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_label(...)' code ##################

        unicode_39156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, (-1)), 'unicode', u'\n        Get the label used for this artist in the legend.\n        ')
        # Getting the type of 'self' (line 54)
        self_39157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), 'self')
        # Obtaining the member '_label' of a type (line 54)
        _label_39158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 15), self_39157, '_label')
        # Assigning a type to the variable 'stypy_return_type' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'stypy_return_type', _label_39158)
        
        # ################# End of 'get_label(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_label' in the type store
        # Getting the type of 'stypy_return_type' (line 50)
        stypy_return_type_39159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39159)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_label'
        return stypy_return_type_39159


    @norecursion
    def set_label(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_label'
        module_type_store = module_type_store.open_function_context('set_label', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Container.set_label.__dict__.__setitem__('stypy_localization', localization)
        Container.set_label.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Container.set_label.__dict__.__setitem__('stypy_type_store', module_type_store)
        Container.set_label.__dict__.__setitem__('stypy_function_name', 'Container.set_label')
        Container.set_label.__dict__.__setitem__('stypy_param_names_list', ['s'])
        Container.set_label.__dict__.__setitem__('stypy_varargs_param_name', None)
        Container.set_label.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Container.set_label.__dict__.__setitem__('stypy_call_defaults', defaults)
        Container.set_label.__dict__.__setitem__('stypy_call_varargs', varargs)
        Container.set_label.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Container.set_label.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Container.set_label', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_label', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_label(...)' code ##################

        unicode_39160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, (-1)), 'unicode', u"\n        Set the label to *s* for auto legend.\n\n        ACCEPTS: string or anything printable with '%s' conversion.\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 62)
        # Getting the type of 's' (line 62)
        s_39161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 's')
        # Getting the type of 'None' (line 62)
        None_39162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'None')
        
        (may_be_39163, more_types_in_union_39164) = may_not_be_none(s_39161, None_39162)

        if may_be_39163:

            if more_types_in_union_39164:
                # Runtime conditional SSA (line 62)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Attribute (line 63):
            
            # Assigning a BinOp to a Attribute (line 63):
            unicode_39165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 26), 'unicode', u'%s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 63)
            tuple_39166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 34), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 63)
            # Adding element type (line 63)
            # Getting the type of 's' (line 63)
            s_39167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 34), 's')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 34), tuple_39166, s_39167)
            
            # Applying the binary operator '%' (line 63)
            result_mod_39168 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 26), '%', unicode_39165, tuple_39166)
            
            # Getting the type of 'self' (line 63)
            self_39169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'self')
            # Setting the type of the member '_label' of a type (line 63)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), self_39169, '_label', result_mod_39168)

            if more_types_in_union_39164:
                # Runtime conditional SSA for else branch (line 62)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_39163) or more_types_in_union_39164):
            
            # Assigning a Name to a Attribute (line 65):
            
            # Assigning a Name to a Attribute (line 65):
            # Getting the type of 'None' (line 65)
            None_39170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 26), 'None')
            # Getting the type of 'self' (line 65)
            self_39171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'self')
            # Setting the type of the member '_label' of a type (line 65)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), self_39171, '_label', None_39170)

            if (may_be_39163 and more_types_in_union_39164):
                # SSA join for if statement (line 62)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to pchanged(...): (line 66)
        # Processing the call keyword arguments (line 66)
        kwargs_39174 = {}
        # Getting the type of 'self' (line 66)
        self_39172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self', False)
        # Obtaining the member 'pchanged' of a type (line 66)
        pchanged_39173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_39172, 'pchanged')
        # Calling pchanged(args, kwargs) (line 66)
        pchanged_call_result_39175 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), pchanged_39173, *[], **kwargs_39174)
        
        
        # ################# End of 'set_label(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_label' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_39176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39176)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_label'
        return stypy_return_type_39176


    @norecursion
    def add_callback(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_callback'
        module_type_store = module_type_store.open_function_context('add_callback', 68, 4, False)
        # Assigning a type to the variable 'self' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Container.add_callback.__dict__.__setitem__('stypy_localization', localization)
        Container.add_callback.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Container.add_callback.__dict__.__setitem__('stypy_type_store', module_type_store)
        Container.add_callback.__dict__.__setitem__('stypy_function_name', 'Container.add_callback')
        Container.add_callback.__dict__.__setitem__('stypy_param_names_list', ['func'])
        Container.add_callback.__dict__.__setitem__('stypy_varargs_param_name', None)
        Container.add_callback.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Container.add_callback.__dict__.__setitem__('stypy_call_defaults', defaults)
        Container.add_callback.__dict__.__setitem__('stypy_call_varargs', varargs)
        Container.add_callback.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Container.add_callback.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Container.add_callback', ['func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_callback', localization, ['func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_callback(...)' code ##################

        unicode_39177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, (-1)), 'unicode', u"\n        Adds a callback function that will be called whenever one of\n        the :class:`Artist`'s properties changes.\n\n        Returns an *id* that is useful for removing the callback with\n        :meth:`remove_callback` later.\n        ")
        
        # Assigning a Attribute to a Name (line 76):
        
        # Assigning a Attribute to a Name (line 76):
        # Getting the type of 'self' (line 76)
        self_39178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 14), 'self')
        # Obtaining the member '_oid' of a type (line 76)
        _oid_39179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 14), self_39178, '_oid')
        # Assigning a type to the variable 'oid' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'oid', _oid_39179)
        
        # Assigning a Name to a Subscript (line 77):
        
        # Assigning a Name to a Subscript (line 77):
        # Getting the type of 'func' (line 77)
        func_39180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 35), 'func')
        # Getting the type of 'self' (line 77)
        self_39181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'self')
        # Obtaining the member '_propobservers' of a type (line 77)
        _propobservers_39182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), self_39181, '_propobservers')
        # Getting the type of 'oid' (line 77)
        oid_39183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 28), 'oid')
        # Storing an element on a container (line 77)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 8), _propobservers_39182, (oid_39183, func_39180))
        
        # Getting the type of 'self' (line 78)
        self_39184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self')
        # Obtaining the member '_oid' of a type (line 78)
        _oid_39185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_39184, '_oid')
        int_39186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 21), 'int')
        # Applying the binary operator '+=' (line 78)
        result_iadd_39187 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 8), '+=', _oid_39185, int_39186)
        # Getting the type of 'self' (line 78)
        self_39188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self')
        # Setting the type of the member '_oid' of a type (line 78)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_39188, '_oid', result_iadd_39187)
        
        # Getting the type of 'oid' (line 79)
        oid_39189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'oid')
        # Assigning a type to the variable 'stypy_return_type' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'stypy_return_type', oid_39189)
        
        # ################# End of 'add_callback(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_callback' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_39190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39190)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_callback'
        return stypy_return_type_39190


    @norecursion
    def remove_callback(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'remove_callback'
        module_type_store = module_type_store.open_function_context('remove_callback', 81, 4, False)
        # Assigning a type to the variable 'self' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Container.remove_callback.__dict__.__setitem__('stypy_localization', localization)
        Container.remove_callback.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Container.remove_callback.__dict__.__setitem__('stypy_type_store', module_type_store)
        Container.remove_callback.__dict__.__setitem__('stypy_function_name', 'Container.remove_callback')
        Container.remove_callback.__dict__.__setitem__('stypy_param_names_list', ['oid'])
        Container.remove_callback.__dict__.__setitem__('stypy_varargs_param_name', None)
        Container.remove_callback.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Container.remove_callback.__dict__.__setitem__('stypy_call_defaults', defaults)
        Container.remove_callback.__dict__.__setitem__('stypy_call_varargs', varargs)
        Container.remove_callback.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Container.remove_callback.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Container.remove_callback', ['oid'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'remove_callback', localization, ['oid'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'remove_callback(...)' code ##################

        unicode_39191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, (-1)), 'unicode', u'\n        Remove a callback based on its *id*.\n\n        .. seealso::\n\n            :meth:`add_callback`\n               For adding callbacks\n\n        ')
        
        
        # SSA begins for try-except statement (line 91)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        # Deleting a member
        # Getting the type of 'self' (line 92)
        self_39192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'self')
        # Obtaining the member '_propobservers' of a type (line 92)
        _propobservers_39193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 16), self_39192, '_propobservers')
        
        # Obtaining the type of the subscript
        # Getting the type of 'oid' (line 92)
        oid_39194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 36), 'oid')
        # Getting the type of 'self' (line 92)
        self_39195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'self')
        # Obtaining the member '_propobservers' of a type (line 92)
        _propobservers_39196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 16), self_39195, '_propobservers')
        # Obtaining the member '__getitem__' of a type (line 92)
        getitem___39197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 16), _propobservers_39196, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 92)
        subscript_call_result_39198 = invoke(stypy.reporting.localization.Localization(__file__, 92, 16), getitem___39197, oid_39194)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 12), _propobservers_39193, subscript_call_result_39198)
        # SSA branch for the except part of a try statement (line 91)
        # SSA branch for the except 'KeyError' branch of a try statement (line 91)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 91)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'remove_callback(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'remove_callback' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_39199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39199)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'remove_callback'
        return stypy_return_type_39199


    @norecursion
    def pchanged(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'pchanged'
        module_type_store = module_type_store.open_function_context('pchanged', 96, 4, False)
        # Assigning a type to the variable 'self' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Container.pchanged.__dict__.__setitem__('stypy_localization', localization)
        Container.pchanged.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Container.pchanged.__dict__.__setitem__('stypy_type_store', module_type_store)
        Container.pchanged.__dict__.__setitem__('stypy_function_name', 'Container.pchanged')
        Container.pchanged.__dict__.__setitem__('stypy_param_names_list', [])
        Container.pchanged.__dict__.__setitem__('stypy_varargs_param_name', None)
        Container.pchanged.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Container.pchanged.__dict__.__setitem__('stypy_call_defaults', defaults)
        Container.pchanged.__dict__.__setitem__('stypy_call_varargs', varargs)
        Container.pchanged.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Container.pchanged.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Container.pchanged', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pchanged', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pchanged(...)' code ##################

        unicode_39200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, (-1)), 'unicode', u'\n        Fire an event when property changed, calling all of the\n        registered callbacks.\n        ')
        
        
        # Call to list(...): (line 101)
        # Processing the call arguments (line 101)
        
        # Call to iteritems(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'self' (line 101)
        self_39204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 44), 'self', False)
        # Obtaining the member '_propobservers' of a type (line 101)
        _propobservers_39205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 44), self_39204, '_propobservers')
        # Processing the call keyword arguments (line 101)
        kwargs_39206 = {}
        # Getting the type of 'six' (line 101)
        six_39202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'six', False)
        # Obtaining the member 'iteritems' of a type (line 101)
        iteritems_39203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 30), six_39202, 'iteritems')
        # Calling iteritems(args, kwargs) (line 101)
        iteritems_call_result_39207 = invoke(stypy.reporting.localization.Localization(__file__, 101, 30), iteritems_39203, *[_propobservers_39205], **kwargs_39206)
        
        # Processing the call keyword arguments (line 101)
        kwargs_39208 = {}
        # Getting the type of 'list' (line 101)
        list_39201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 25), 'list', False)
        # Calling list(args, kwargs) (line 101)
        list_call_result_39209 = invoke(stypy.reporting.localization.Localization(__file__, 101, 25), list_39201, *[iteritems_call_result_39207], **kwargs_39208)
        
        # Testing the type of a for loop iterable (line 101)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 101, 8), list_call_result_39209)
        # Getting the type of the for loop variable (line 101)
        for_loop_var_39210 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 101, 8), list_call_result_39209)
        # Assigning a type to the variable 'oid' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'oid', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 8), for_loop_var_39210))
        # Assigning a type to the variable 'func' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'func', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 8), for_loop_var_39210))
        # SSA begins for a for statement (line 101)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to func(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'self' (line 102)
        self_39212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 17), 'self', False)
        # Processing the call keyword arguments (line 102)
        kwargs_39213 = {}
        # Getting the type of 'func' (line 102)
        func_39211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'func', False)
        # Calling func(args, kwargs) (line 102)
        func_call_result_39214 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), func_39211, *[self_39212], **kwargs_39213)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'pchanged(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pchanged' in the type store
        # Getting the type of 'stypy_return_type' (line 96)
        stypy_return_type_39215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39215)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pchanged'
        return stypy_return_type_39215


    @norecursion
    def get_children(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_children'
        module_type_store = module_type_store.open_function_context('get_children', 104, 4, False)
        # Assigning a type to the variable 'self' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Container.get_children.__dict__.__setitem__('stypy_localization', localization)
        Container.get_children.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Container.get_children.__dict__.__setitem__('stypy_type_store', module_type_store)
        Container.get_children.__dict__.__setitem__('stypy_function_name', 'Container.get_children')
        Container.get_children.__dict__.__setitem__('stypy_param_names_list', [])
        Container.get_children.__dict__.__setitem__('stypy_varargs_param_name', None)
        Container.get_children.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Container.get_children.__dict__.__setitem__('stypy_call_defaults', defaults)
        Container.get_children.__dict__.__setitem__('stypy_call_varargs', varargs)
        Container.get_children.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Container.get_children.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Container.get_children', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_children', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_children(...)' code ##################

        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to flatten(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'self' (line 105)
        self_39222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 49), 'self', False)
        # Processing the call keyword arguments (line 105)
        kwargs_39223 = {}
        # Getting the type of 'cbook' (line 105)
        cbook_39220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 35), 'cbook', False)
        # Obtaining the member 'flatten' of a type (line 105)
        flatten_39221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 35), cbook_39220, 'flatten')
        # Calling flatten(args, kwargs) (line 105)
        flatten_call_result_39224 = invoke(stypy.reporting.localization.Localization(__file__, 105, 35), flatten_39221, *[self_39222], **kwargs_39223)
        
        comprehension_39225 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 16), flatten_call_result_39224)
        # Assigning a type to the variable 'child' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'child', comprehension_39225)
        
        # Getting the type of 'child' (line 105)
        child_39217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 58), 'child')
        # Getting the type of 'None' (line 105)
        None_39218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 71), 'None')
        # Applying the binary operator 'isnot' (line 105)
        result_is_not_39219 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 58), 'isnot', child_39217, None_39218)
        
        # Getting the type of 'child' (line 105)
        child_39216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'child')
        list_39226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 16), list_39226, child_39216)
        # Assigning a type to the variable 'stypy_return_type' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'stypy_return_type', list_39226)
        
        # ################# End of 'get_children(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_children' in the type store
        # Getting the type of 'stypy_return_type' (line 104)
        stypy_return_type_39227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39227)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_children'
        return stypy_return_type_39227


# Assigning a type to the variable 'Container' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'Container', Container)
# Declaration of the 'BarContainer' class
# Getting the type of 'Container' (line 108)
Container_39228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 19), 'Container')

class BarContainer(Container_39228, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 110)
        None_39229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 41), 'None')
        defaults = [None_39229]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 110, 4, False)
        # Assigning a type to the variable 'self' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BarContainer.__init__', ['patches', 'errorbar'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['patches', 'errorbar'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 111):
        
        # Assigning a Name to a Attribute (line 111):
        # Getting the type of 'patches' (line 111)
        patches_39230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 23), 'patches')
        # Getting the type of 'self' (line 111)
        self_39231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'self')
        # Setting the type of the member 'patches' of a type (line 111)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), self_39231, 'patches', patches_39230)
        
        # Assigning a Name to a Attribute (line 112):
        
        # Assigning a Name to a Attribute (line 112):
        # Getting the type of 'errorbar' (line 112)
        errorbar_39232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 24), 'errorbar')
        # Getting the type of 'self' (line 112)
        self_39233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'self')
        # Setting the type of the member 'errorbar' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), self_39233, 'errorbar', errorbar_39232)
        
        # Call to __init__(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'self' (line 113)
        self_39236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 27), 'self', False)
        # Getting the type of 'patches' (line 113)
        patches_39237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 33), 'patches', False)
        # Processing the call keyword arguments (line 113)
        # Getting the type of 'kwargs' (line 113)
        kwargs_39238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 44), 'kwargs', False)
        kwargs_39239 = {'kwargs_39238': kwargs_39238}
        # Getting the type of 'Container' (line 113)
        Container_39234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'Container', False)
        # Obtaining the member '__init__' of a type (line 113)
        init___39235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), Container_39234, '__init__')
        # Calling __init__(args, kwargs) (line 113)
        init___call_result_39240 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), init___39235, *[self_39236, patches_39237], **kwargs_39239)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'BarContainer' (line 108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'BarContainer', BarContainer)
# Declaration of the 'ErrorbarContainer' class
# Getting the type of 'Container' (line 116)
Container_39241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), 'Container')

class ErrorbarContainer(Container_39241, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 118)
        False_39242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 39), 'False')
        # Getting the type of 'False' (line 118)
        False_39243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 55), 'False')
        defaults = [False_39242, False_39243]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 118, 4, False)
        # Assigning a type to the variable 'self' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ErrorbarContainer.__init__', ['lines', 'has_xerr', 'has_yerr'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['lines', 'has_xerr', 'has_yerr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 119):
        
        # Assigning a Name to a Attribute (line 119):
        # Getting the type of 'lines' (line 119)
        lines_39244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 21), 'lines')
        # Getting the type of 'self' (line 119)
        self_39245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'self')
        # Setting the type of the member 'lines' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), self_39245, 'lines', lines_39244)
        
        # Assigning a Name to a Attribute (line 120):
        
        # Assigning a Name to a Attribute (line 120):
        # Getting the type of 'has_xerr' (line 120)
        has_xerr_39246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 24), 'has_xerr')
        # Getting the type of 'self' (line 120)
        self_39247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'self')
        # Setting the type of the member 'has_xerr' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), self_39247, 'has_xerr', has_xerr_39246)
        
        # Assigning a Name to a Attribute (line 121):
        
        # Assigning a Name to a Attribute (line 121):
        # Getting the type of 'has_yerr' (line 121)
        has_yerr_39248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 24), 'has_yerr')
        # Getting the type of 'self' (line 121)
        self_39249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'self')
        # Setting the type of the member 'has_yerr' of a type (line 121)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), self_39249, 'has_yerr', has_yerr_39248)
        
        # Call to __init__(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'self' (line 122)
        self_39252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 27), 'self', False)
        # Getting the type of 'lines' (line 122)
        lines_39253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 33), 'lines', False)
        # Processing the call keyword arguments (line 122)
        # Getting the type of 'kwargs' (line 122)
        kwargs_39254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 42), 'kwargs', False)
        kwargs_39255 = {'kwargs_39254': kwargs_39254}
        # Getting the type of 'Container' (line 122)
        Container_39250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'Container', False)
        # Obtaining the member '__init__' of a type (line 122)
        init___39251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), Container_39250, '__init__')
        # Calling __init__(args, kwargs) (line 122)
        init___call_result_39256 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), init___39251, *[self_39252, lines_39253], **kwargs_39255)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'ErrorbarContainer' (line 116)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'ErrorbarContainer', ErrorbarContainer)
# Declaration of the 'StemContainer' class
# Getting the type of 'Container' (line 125)
Container_39257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 20), 'Container')

class StemContainer(Container_39257, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 127, 4, False)
        # Assigning a type to the variable 'self' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StemContainer.__init__', ['markerline_stemlines_baseline'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['markerline_stemlines_baseline'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Tuple (line 128):
        
        # Assigning a Subscript to a Name (line 128):
        
        # Obtaining the type of the subscript
        int_39258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 8), 'int')
        # Getting the type of 'markerline_stemlines_baseline' (line 128)
        markerline_stemlines_baseline_39259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 42), 'markerline_stemlines_baseline')
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___39260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), markerline_stemlines_baseline_39259, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_39261 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), getitem___39260, int_39258)
        
        # Assigning a type to the variable 'tuple_var_assignment_39069' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_39069', subscript_call_result_39261)
        
        # Assigning a Subscript to a Name (line 128):
        
        # Obtaining the type of the subscript
        int_39262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 8), 'int')
        # Getting the type of 'markerline_stemlines_baseline' (line 128)
        markerline_stemlines_baseline_39263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 42), 'markerline_stemlines_baseline')
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___39264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), markerline_stemlines_baseline_39263, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_39265 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), getitem___39264, int_39262)
        
        # Assigning a type to the variable 'tuple_var_assignment_39070' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_39070', subscript_call_result_39265)
        
        # Assigning a Subscript to a Name (line 128):
        
        # Obtaining the type of the subscript
        int_39266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 8), 'int')
        # Getting the type of 'markerline_stemlines_baseline' (line 128)
        markerline_stemlines_baseline_39267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 42), 'markerline_stemlines_baseline')
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___39268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), markerline_stemlines_baseline_39267, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_39269 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), getitem___39268, int_39266)
        
        # Assigning a type to the variable 'tuple_var_assignment_39071' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_39071', subscript_call_result_39269)
        
        # Assigning a Name to a Name (line 128):
        # Getting the type of 'tuple_var_assignment_39069' (line 128)
        tuple_var_assignment_39069_39270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_39069')
        # Assigning a type to the variable 'markerline' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'markerline', tuple_var_assignment_39069_39270)
        
        # Assigning a Name to a Name (line 128):
        # Getting the type of 'tuple_var_assignment_39070' (line 128)
        tuple_var_assignment_39070_39271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_39070')
        # Assigning a type to the variable 'stemlines' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 20), 'stemlines', tuple_var_assignment_39070_39271)
        
        # Assigning a Name to a Name (line 128):
        # Getting the type of 'tuple_var_assignment_39071' (line 128)
        tuple_var_assignment_39071_39272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_39071')
        # Assigning a type to the variable 'baseline' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 31), 'baseline', tuple_var_assignment_39071_39272)
        
        # Assigning a Name to a Attribute (line 129):
        
        # Assigning a Name to a Attribute (line 129):
        # Getting the type of 'markerline' (line 129)
        markerline_39273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 26), 'markerline')
        # Getting the type of 'self' (line 129)
        self_39274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self')
        # Setting the type of the member 'markerline' of a type (line 129)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_39274, 'markerline', markerline_39273)
        
        # Assigning a Name to a Attribute (line 130):
        
        # Assigning a Name to a Attribute (line 130):
        # Getting the type of 'stemlines' (line 130)
        stemlines_39275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 25), 'stemlines')
        # Getting the type of 'self' (line 130)
        self_39276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'self')
        # Setting the type of the member 'stemlines' of a type (line 130)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), self_39276, 'stemlines', stemlines_39275)
        
        # Assigning a Name to a Attribute (line 131):
        
        # Assigning a Name to a Attribute (line 131):
        # Getting the type of 'baseline' (line 131)
        baseline_39277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 24), 'baseline')
        # Getting the type of 'self' (line 131)
        self_39278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'self')
        # Setting the type of the member 'baseline' of a type (line 131)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), self_39278, 'baseline', baseline_39277)
        
        # Call to __init__(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'self' (line 132)
        self_39281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 27), 'self', False)
        # Getting the type of 'markerline_stemlines_baseline' (line 132)
        markerline_stemlines_baseline_39282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 33), 'markerline_stemlines_baseline', False)
        # Processing the call keyword arguments (line 132)
        # Getting the type of 'kwargs' (line 132)
        kwargs_39283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 66), 'kwargs', False)
        kwargs_39284 = {'kwargs_39283': kwargs_39283}
        # Getting the type of 'Container' (line 132)
        Container_39279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'Container', False)
        # Obtaining the member '__init__' of a type (line 132)
        init___39280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), Container_39279, '__init__')
        # Calling __init__(args, kwargs) (line 132)
        init___call_result_39285 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), init___39280, *[self_39281, markerline_stemlines_baseline_39282], **kwargs_39284)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'StemContainer' (line 125)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 0), 'StemContainer', StemContainer)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
