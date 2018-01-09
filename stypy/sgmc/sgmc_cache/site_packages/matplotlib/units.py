
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: The classes here provide support for using custom classes with
3: matplotlib, e.g., those that do not expose the array interface but know
4: how to convert themselves to arrays.  It also supports classes with
5: units and units conversion.  Use cases include converters for custom
6: objects, e.g., a list of datetime objects, as well as for objects that
7: are unit aware.  We don't assume any particular units implementation;
8: rather a units implementation must provide the register with the Registry
9: converter dictionary and a ConversionInterface.  For example,
10: here is a complete implementation which supports plotting with native
11: datetime objects::
12: 
13:     import matplotlib.units as units
14:     import matplotlib.dates as dates
15:     import matplotlib.ticker as ticker
16:     import datetime
17: 
18:     class DateConverter(units.ConversionInterface):
19: 
20:         @staticmethod
21:         def convert(value, unit, axis):
22:             'convert value to a scalar or array'
23:             return dates.date2num(value)
24: 
25:         @staticmethod
26:         def axisinfo(unit, axis):
27:             'return major and minor tick locators and formatters'
28:             if unit!='date': return None
29:             majloc = dates.AutoDateLocator()
30:             majfmt = dates.AutoDateFormatter(majloc)
31:             return AxisInfo(majloc=majloc,
32:                             majfmt=majfmt,
33:                             label='date')
34: 
35:         @staticmethod
36:         def default_units(x, axis):
37:             'return the default unit for x or None'
38:             return 'date'
39: 
40:     # finally we register our object type with a converter
41:     units.registry[datetime.date] = DateConverter()
42: 
43: '''
44: from __future__ import (absolute_import, division, print_function,
45:                         unicode_literals)
46: 
47: 
48: import six
49: from matplotlib.cbook import iterable, is_numlike, safe_first_element
50: import numpy as np
51: 
52: 
53: class AxisInfo(object):
54:     '''information to support default axis labeling and tick labeling, and
55:        default limits'''
56:     def __init__(self, majloc=None, minloc=None,
57:                  majfmt=None, minfmt=None, label=None,
58:                  default_limits=None):
59:         '''
60:         majloc and minloc: TickLocators for the major and minor ticks
61:         majfmt and minfmt: TickFormatters for the major and minor ticks
62:         label: the default axis label
63:         default_limits: the default min, max of the axis if no data is present
64:         If any of the above are None, the axis will simply use the default
65:         '''
66:         self.majloc = majloc
67:         self.minloc = minloc
68:         self.majfmt = majfmt
69:         self.minfmt = minfmt
70:         self.label = label
71:         self.default_limits = default_limits
72: 
73: 
74: class ConversionInterface(object):
75:     '''
76:     The minimal interface for a converter to take custom instances (or
77:     sequences) and convert them to values mpl can use
78:     '''
79:     @staticmethod
80:     def axisinfo(unit, axis):
81:         'return an units.AxisInfo instance for axis with the specified units'
82:         return None
83: 
84:     @staticmethod
85:     def default_units(x, axis):
86:         'return the default unit for x or None for the given axis'
87:         return None
88: 
89:     @staticmethod
90:     def convert(obj, unit, axis):
91:         '''
92:         convert obj using unit for the specified axis.  If obj is a sequence,
93:         return the converted sequence.  The output must be a sequence of
94:         scalars that can be used by the numpy array layer
95:         '''
96:         return obj
97: 
98:     @staticmethod
99:     def is_numlike(x):
100:         '''
101:         The matplotlib datalim, autoscaling, locators etc work with
102:         scalars which are the units converted to floats given the
103:         current unit.  The converter may be passed these floats, or
104:         arrays of them, even when units are set.  Derived conversion
105:         interfaces may opt to pass plain-ol unitless numbers through
106:         the conversion interface and this is a helper function for
107:         them.
108:         '''
109:         if iterable(x):
110:             for thisx in x:
111:                 return is_numlike(thisx)
112:         else:
113:             return is_numlike(x)
114: 
115: 
116: class Registry(dict):
117:     '''
118:     register types with conversion interface
119:     '''
120:     def __init__(self):
121:         dict.__init__(self)
122:         self._cached = {}
123: 
124:     def get_converter(self, x):
125:         'get the converter interface instance for x, or None'
126: 
127:         if not len(self):
128:             return None  # nothing registered
129:         # DISABLED idx = id(x)
130:         # DISABLED cached = self._cached.get(idx)
131:         # DISABLED if cached is not None: return cached
132: 
133:         converter = None
134:         classx = getattr(x, '__class__', None)
135: 
136:         if classx is not None:
137:             converter = self.get(classx)
138: 
139:         if isinstance(x, np.ndarray) and x.size:
140:             xravel = x.ravel()
141:             try:
142:                 # pass the first value of x that is not masked back to
143:                 # get_converter
144:                 if not np.all(xravel.mask):
145:                     # some elements are not masked
146:                     converter = self.get_converter(
147:                         xravel[np.argmin(xravel.mask)])
148:                     return converter
149:             except AttributeError:
150:                 # not a masked_array
151:                 # Make sure we don't recurse forever -- it's possible for
152:                 # ndarray subclasses to continue to return subclasses and
153:                 # not ever return a non-subclass for a single element.
154:                 next_item = xravel[0]
155:                 if (not isinstance(next_item, np.ndarray) or
156:                     next_item.shape != x.shape):
157:                     converter = self.get_converter(next_item)
158:                 return converter
159: 
160:         if converter is None:
161:             try:
162:                 thisx = safe_first_element(x)
163:             except (TypeError, StopIteration):
164:                 pass
165:             else:
166:                 if classx and classx != getattr(thisx, '__class__', None):
167:                     converter = self.get_converter(thisx)
168:                     return converter
169: 
170:         # DISABLED self._cached[idx] = converter
171:         return converter
172: 
173: 
174: registry = Registry()
175: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_162092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'unicode', u"\nThe classes here provide support for using custom classes with\nmatplotlib, e.g., those that do not expose the array interface but know\nhow to convert themselves to arrays.  It also supports classes with\nunits and units conversion.  Use cases include converters for custom\nobjects, e.g., a list of datetime objects, as well as for objects that\nare unit aware.  We don't assume any particular units implementation;\nrather a units implementation must provide the register with the Registry\nconverter dictionary and a ConversionInterface.  For example,\nhere is a complete implementation which supports plotting with native\ndatetime objects::\n\n    import matplotlib.units as units\n    import matplotlib.dates as dates\n    import matplotlib.ticker as ticker\n    import datetime\n\n    class DateConverter(units.ConversionInterface):\n\n        @staticmethod\n        def convert(value, unit, axis):\n            'convert value to a scalar or array'\n            return dates.date2num(value)\n\n        @staticmethod\n        def axisinfo(unit, axis):\n            'return major and minor tick locators and formatters'\n            if unit!='date': return None\n            majloc = dates.AutoDateLocator()\n            majfmt = dates.AutoDateFormatter(majloc)\n            return AxisInfo(majloc=majloc,\n                            majfmt=majfmt,\n                            label='date')\n\n        @staticmethod\n        def default_units(x, axis):\n            'return the default unit for x or None'\n            return 'date'\n\n    # finally we register our object type with a converter\n    units.registry[datetime.date] = DateConverter()\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 48, 0))

# 'import six' statement (line 48)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_162093 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'six')

if (type(import_162093) is not StypyTypeError):

    if (import_162093 != 'pyd_module'):
        __import__(import_162093)
        sys_modules_162094 = sys.modules[import_162093]
        import_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'six', sys_modules_162094.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'six', import_162093)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 49, 0))

# 'from matplotlib.cbook import iterable, is_numlike, safe_first_element' statement (line 49)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_162095 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 49, 0), 'matplotlib.cbook')

if (type(import_162095) is not StypyTypeError):

    if (import_162095 != 'pyd_module'):
        __import__(import_162095)
        sys_modules_162096 = sys.modules[import_162095]
        import_from_module(stypy.reporting.localization.Localization(__file__, 49, 0), 'matplotlib.cbook', sys_modules_162096.module_type_store, module_type_store, ['iterable', 'is_numlike', 'safe_first_element'])
        nest_module(stypy.reporting.localization.Localization(__file__, 49, 0), __file__, sys_modules_162096, sys_modules_162096.module_type_store, module_type_store)
    else:
        from matplotlib.cbook import iterable, is_numlike, safe_first_element

        import_from_module(stypy.reporting.localization.Localization(__file__, 49, 0), 'matplotlib.cbook', None, module_type_store, ['iterable', 'is_numlike', 'safe_first_element'], [iterable, is_numlike, safe_first_element])

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'matplotlib.cbook', import_162095)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 50, 0))

# 'import numpy' statement (line 50)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_162097 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 50, 0), 'numpy')

if (type(import_162097) is not StypyTypeError):

    if (import_162097 != 'pyd_module'):
        __import__(import_162097)
        sys_modules_162098 = sys.modules[import_162097]
        import_module(stypy.reporting.localization.Localization(__file__, 50, 0), 'np', sys_modules_162098.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 50, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'numpy', import_162097)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

# Declaration of the 'AxisInfo' class

class AxisInfo(object, ):
    unicode_162099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, (-1)), 'unicode', u'information to support default axis labeling and tick labeling, and\n       default limits')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 56)
        None_162100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), 'None')
        # Getting the type of 'None' (line 56)
        None_162101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 43), 'None')
        # Getting the type of 'None' (line 57)
        None_162102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 24), 'None')
        # Getting the type of 'None' (line 57)
        None_162103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 37), 'None')
        # Getting the type of 'None' (line 57)
        None_162104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 49), 'None')
        # Getting the type of 'None' (line 58)
        None_162105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 32), 'None')
        defaults = [None_162100, None_162101, None_162102, None_162103, None_162104, None_162105]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AxisInfo.__init__', ['majloc', 'minloc', 'majfmt', 'minfmt', 'label', 'default_limits'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['majloc', 'minloc', 'majfmt', 'minfmt', 'label', 'default_limits'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_162106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, (-1)), 'unicode', u'\n        majloc and minloc: TickLocators for the major and minor ticks\n        majfmt and minfmt: TickFormatters for the major and minor ticks\n        label: the default axis label\n        default_limits: the default min, max of the axis if no data is present\n        If any of the above are None, the axis will simply use the default\n        ')
        
        # Assigning a Name to a Attribute (line 66):
        # Getting the type of 'majloc' (line 66)
        majloc_162107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 22), 'majloc')
        # Getting the type of 'self' (line 66)
        self_162108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self')
        # Setting the type of the member 'majloc' of a type (line 66)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_162108, 'majloc', majloc_162107)
        
        # Assigning a Name to a Attribute (line 67):
        # Getting the type of 'minloc' (line 67)
        minloc_162109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 22), 'minloc')
        # Getting the type of 'self' (line 67)
        self_162110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self')
        # Setting the type of the member 'minloc' of a type (line 67)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_162110, 'minloc', minloc_162109)
        
        # Assigning a Name to a Attribute (line 68):
        # Getting the type of 'majfmt' (line 68)
        majfmt_162111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 22), 'majfmt')
        # Getting the type of 'self' (line 68)
        self_162112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self')
        # Setting the type of the member 'majfmt' of a type (line 68)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_162112, 'majfmt', majfmt_162111)
        
        # Assigning a Name to a Attribute (line 69):
        # Getting the type of 'minfmt' (line 69)
        minfmt_162113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 22), 'minfmt')
        # Getting the type of 'self' (line 69)
        self_162114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'self')
        # Setting the type of the member 'minfmt' of a type (line 69)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), self_162114, 'minfmt', minfmt_162113)
        
        # Assigning a Name to a Attribute (line 70):
        # Getting the type of 'label' (line 70)
        label_162115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 21), 'label')
        # Getting the type of 'self' (line 70)
        self_162116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'self')
        # Setting the type of the member 'label' of a type (line 70)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), self_162116, 'label', label_162115)
        
        # Assigning a Name to a Attribute (line 71):
        # Getting the type of 'default_limits' (line 71)
        default_limits_162117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 30), 'default_limits')
        # Getting the type of 'self' (line 71)
        self_162118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'self')
        # Setting the type of the member 'default_limits' of a type (line 71)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), self_162118, 'default_limits', default_limits_162117)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'AxisInfo' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'AxisInfo', AxisInfo)
# Declaration of the 'ConversionInterface' class

class ConversionInterface(object, ):
    unicode_162119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, (-1)), 'unicode', u'\n    The minimal interface for a converter to take custom instances (or\n    sequences) and convert them to values mpl can use\n    ')

    @staticmethod
    @norecursion
    def axisinfo(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'axisinfo'
        module_type_store = module_type_store.open_function_context('axisinfo', 79, 4, False)
        
        # Passed parameters checking function
        ConversionInterface.axisinfo.__dict__.__setitem__('stypy_localization', localization)
        ConversionInterface.axisinfo.__dict__.__setitem__('stypy_type_of_self', None)
        ConversionInterface.axisinfo.__dict__.__setitem__('stypy_type_store', module_type_store)
        ConversionInterface.axisinfo.__dict__.__setitem__('stypy_function_name', 'axisinfo')
        ConversionInterface.axisinfo.__dict__.__setitem__('stypy_param_names_list', ['unit', 'axis'])
        ConversionInterface.axisinfo.__dict__.__setitem__('stypy_varargs_param_name', None)
        ConversionInterface.axisinfo.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ConversionInterface.axisinfo.__dict__.__setitem__('stypy_call_defaults', defaults)
        ConversionInterface.axisinfo.__dict__.__setitem__('stypy_call_varargs', varargs)
        ConversionInterface.axisinfo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ConversionInterface.axisinfo.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, 'axisinfo', ['unit', 'axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'axisinfo', localization, ['axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'axisinfo(...)' code ##################

        unicode_162120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 8), 'unicode', u'return an units.AxisInfo instance for axis with the specified units')
        # Getting the type of 'None' (line 82)
        None_162121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'stypy_return_type', None_162121)
        
        # ################# End of 'axisinfo(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'axisinfo' in the type store
        # Getting the type of 'stypy_return_type' (line 79)
        stypy_return_type_162122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_162122)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'axisinfo'
        return stypy_return_type_162122


    @staticmethod
    @norecursion
    def default_units(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'default_units'
        module_type_store = module_type_store.open_function_context('default_units', 84, 4, False)
        
        # Passed parameters checking function
        ConversionInterface.default_units.__dict__.__setitem__('stypy_localization', localization)
        ConversionInterface.default_units.__dict__.__setitem__('stypy_type_of_self', None)
        ConversionInterface.default_units.__dict__.__setitem__('stypy_type_store', module_type_store)
        ConversionInterface.default_units.__dict__.__setitem__('stypy_function_name', 'default_units')
        ConversionInterface.default_units.__dict__.__setitem__('stypy_param_names_list', ['x', 'axis'])
        ConversionInterface.default_units.__dict__.__setitem__('stypy_varargs_param_name', None)
        ConversionInterface.default_units.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ConversionInterface.default_units.__dict__.__setitem__('stypy_call_defaults', defaults)
        ConversionInterface.default_units.__dict__.__setitem__('stypy_call_varargs', varargs)
        ConversionInterface.default_units.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ConversionInterface.default_units.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, 'default_units', ['x', 'axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'default_units', localization, ['axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'default_units(...)' code ##################

        unicode_162123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'unicode', u'return the default unit for x or None for the given axis')
        # Getting the type of 'None' (line 87)
        None_162124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'stypy_return_type', None_162124)
        
        # ################# End of 'default_units(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'default_units' in the type store
        # Getting the type of 'stypy_return_type' (line 84)
        stypy_return_type_162125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_162125)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'default_units'
        return stypy_return_type_162125


    @staticmethod
    @norecursion
    def convert(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'convert'
        module_type_store = module_type_store.open_function_context('convert', 89, 4, False)
        
        # Passed parameters checking function
        ConversionInterface.convert.__dict__.__setitem__('stypy_localization', localization)
        ConversionInterface.convert.__dict__.__setitem__('stypy_type_of_self', None)
        ConversionInterface.convert.__dict__.__setitem__('stypy_type_store', module_type_store)
        ConversionInterface.convert.__dict__.__setitem__('stypy_function_name', 'convert')
        ConversionInterface.convert.__dict__.__setitem__('stypy_param_names_list', ['obj', 'unit', 'axis'])
        ConversionInterface.convert.__dict__.__setitem__('stypy_varargs_param_name', None)
        ConversionInterface.convert.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ConversionInterface.convert.__dict__.__setitem__('stypy_call_defaults', defaults)
        ConversionInterface.convert.__dict__.__setitem__('stypy_call_varargs', varargs)
        ConversionInterface.convert.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ConversionInterface.convert.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, None, module_type_store, 'convert', ['obj', 'unit', 'axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'convert', localization, ['unit', 'axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'convert(...)' code ##################

        unicode_162126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, (-1)), 'unicode', u'\n        convert obj using unit for the specified axis.  If obj is a sequence,\n        return the converted sequence.  The output must be a sequence of\n        scalars that can be used by the numpy array layer\n        ')
        # Getting the type of 'obj' (line 96)
        obj_162127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), 'obj')
        # Assigning a type to the variable 'stypy_return_type' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'stypy_return_type', obj_162127)
        
        # ################# End of 'convert(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'convert' in the type store
        # Getting the type of 'stypy_return_type' (line 89)
        stypy_return_type_162128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_162128)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'convert'
        return stypy_return_type_162128


    @staticmethod
    @norecursion
    def is_numlike(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_numlike'
        module_type_store = module_type_store.open_function_context('is_numlike', 98, 4, False)
        
        # Passed parameters checking function
        ConversionInterface.is_numlike.__dict__.__setitem__('stypy_localization', localization)
        ConversionInterface.is_numlike.__dict__.__setitem__('stypy_type_of_self', None)
        ConversionInterface.is_numlike.__dict__.__setitem__('stypy_type_store', module_type_store)
        ConversionInterface.is_numlike.__dict__.__setitem__('stypy_function_name', 'is_numlike')
        ConversionInterface.is_numlike.__dict__.__setitem__('stypy_param_names_list', ['x'])
        ConversionInterface.is_numlike.__dict__.__setitem__('stypy_varargs_param_name', None)
        ConversionInterface.is_numlike.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ConversionInterface.is_numlike.__dict__.__setitem__('stypy_call_defaults', defaults)
        ConversionInterface.is_numlike.__dict__.__setitem__('stypy_call_varargs', varargs)
        ConversionInterface.is_numlike.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ConversionInterface.is_numlike.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, 'is_numlike', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_numlike', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_numlike(...)' code ##################

        unicode_162129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, (-1)), 'unicode', u'\n        The matplotlib datalim, autoscaling, locators etc work with\n        scalars which are the units converted to floats given the\n        current unit.  The converter may be passed these floats, or\n        arrays of them, even when units are set.  Derived conversion\n        interfaces may opt to pass plain-ol unitless numbers through\n        the conversion interface and this is a helper function for\n        them.\n        ')
        
        
        # Call to iterable(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'x' (line 109)
        x_162131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 20), 'x', False)
        # Processing the call keyword arguments (line 109)
        kwargs_162132 = {}
        # Getting the type of 'iterable' (line 109)
        iterable_162130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'iterable', False)
        # Calling iterable(args, kwargs) (line 109)
        iterable_call_result_162133 = invoke(stypy.reporting.localization.Localization(__file__, 109, 11), iterable_162130, *[x_162131], **kwargs_162132)
        
        # Testing the type of an if condition (line 109)
        if_condition_162134 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 8), iterable_call_result_162133)
        # Assigning a type to the variable 'if_condition_162134' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'if_condition_162134', if_condition_162134)
        # SSA begins for if statement (line 109)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'x' (line 110)
        x_162135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 25), 'x')
        # Testing the type of a for loop iterable (line 110)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 110, 12), x_162135)
        # Getting the type of the for loop variable (line 110)
        for_loop_var_162136 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 110, 12), x_162135)
        # Assigning a type to the variable 'thisx' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'thisx', for_loop_var_162136)
        # SSA begins for a for statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to is_numlike(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'thisx' (line 111)
        thisx_162138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 34), 'thisx', False)
        # Processing the call keyword arguments (line 111)
        kwargs_162139 = {}
        # Getting the type of 'is_numlike' (line 111)
        is_numlike_162137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 23), 'is_numlike', False)
        # Calling is_numlike(args, kwargs) (line 111)
        is_numlike_call_result_162140 = invoke(stypy.reporting.localization.Localization(__file__, 111, 23), is_numlike_162137, *[thisx_162138], **kwargs_162139)
        
        # Assigning a type to the variable 'stypy_return_type' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'stypy_return_type', is_numlike_call_result_162140)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 109)
        module_type_store.open_ssa_branch('else')
        
        # Call to is_numlike(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'x' (line 113)
        x_162142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 30), 'x', False)
        # Processing the call keyword arguments (line 113)
        kwargs_162143 = {}
        # Getting the type of 'is_numlike' (line 113)
        is_numlike_162141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'is_numlike', False)
        # Calling is_numlike(args, kwargs) (line 113)
        is_numlike_call_result_162144 = invoke(stypy.reporting.localization.Localization(__file__, 113, 19), is_numlike_162141, *[x_162142], **kwargs_162143)
        
        # Assigning a type to the variable 'stypy_return_type' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'stypy_return_type', is_numlike_call_result_162144)
        # SSA join for if statement (line 109)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'is_numlike(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_numlike' in the type store
        # Getting the type of 'stypy_return_type' (line 98)
        stypy_return_type_162145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_162145)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_numlike'
        return stypy_return_type_162145


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 74, 0, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ConversionInterface.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ConversionInterface' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'ConversionInterface', ConversionInterface)
# Declaration of the 'Registry' class
# Getting the type of 'dict' (line 116)
dict_162146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'dict')

class Registry(dict_162146, ):
    unicode_162147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, (-1)), 'unicode', u'\n    register types with conversion interface\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 120, 4, False)
        # Assigning a type to the variable 'self' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Registry.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'self' (line 121)
        self_162150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 22), 'self', False)
        # Processing the call keyword arguments (line 121)
        kwargs_162151 = {}
        # Getting the type of 'dict' (line 121)
        dict_162148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'dict', False)
        # Obtaining the member '__init__' of a type (line 121)
        init___162149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), dict_162148, '__init__')
        # Calling __init__(args, kwargs) (line 121)
        init___call_result_162152 = invoke(stypy.reporting.localization.Localization(__file__, 121, 8), init___162149, *[self_162150], **kwargs_162151)
        
        
        # Assigning a Dict to a Attribute (line 122):
        
        # Obtaining an instance of the builtin type 'dict' (line 122)
        dict_162153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 23), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 122)
        
        # Getting the type of 'self' (line 122)
        self_162154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'self')
        # Setting the type of the member '_cached' of a type (line 122)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), self_162154, '_cached', dict_162153)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_converter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_converter'
        module_type_store = module_type_store.open_function_context('get_converter', 124, 4, False)
        # Assigning a type to the variable 'self' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Registry.get_converter.__dict__.__setitem__('stypy_localization', localization)
        Registry.get_converter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Registry.get_converter.__dict__.__setitem__('stypy_type_store', module_type_store)
        Registry.get_converter.__dict__.__setitem__('stypy_function_name', 'Registry.get_converter')
        Registry.get_converter.__dict__.__setitem__('stypy_param_names_list', ['x'])
        Registry.get_converter.__dict__.__setitem__('stypy_varargs_param_name', None)
        Registry.get_converter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Registry.get_converter.__dict__.__setitem__('stypy_call_defaults', defaults)
        Registry.get_converter.__dict__.__setitem__('stypy_call_varargs', varargs)
        Registry.get_converter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Registry.get_converter.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Registry.get_converter', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_converter', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_converter(...)' code ##################

        unicode_162155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 8), 'unicode', u'get the converter interface instance for x, or None')
        
        
        
        # Call to len(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'self' (line 127)
        self_162157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 19), 'self', False)
        # Processing the call keyword arguments (line 127)
        kwargs_162158 = {}
        # Getting the type of 'len' (line 127)
        len_162156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'len', False)
        # Calling len(args, kwargs) (line 127)
        len_call_result_162159 = invoke(stypy.reporting.localization.Localization(__file__, 127, 15), len_162156, *[self_162157], **kwargs_162158)
        
        # Applying the 'not' unary operator (line 127)
        result_not__162160 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 11), 'not', len_call_result_162159)
        
        # Testing the type of an if condition (line 127)
        if_condition_162161 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 8), result_not__162160)
        # Assigning a type to the variable 'if_condition_162161' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'if_condition_162161', if_condition_162161)
        # SSA begins for if statement (line 127)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 128)
        None_162162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'stypy_return_type', None_162162)
        # SSA join for if statement (line 127)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 133):
        # Getting the type of 'None' (line 133)
        None_162163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 20), 'None')
        # Assigning a type to the variable 'converter' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'converter', None_162163)
        
        # Assigning a Call to a Name (line 134):
        
        # Call to getattr(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'x' (line 134)
        x_162165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 25), 'x', False)
        unicode_162166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 28), 'unicode', u'__class__')
        # Getting the type of 'None' (line 134)
        None_162167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 41), 'None', False)
        # Processing the call keyword arguments (line 134)
        kwargs_162168 = {}
        # Getting the type of 'getattr' (line 134)
        getattr_162164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 17), 'getattr', False)
        # Calling getattr(args, kwargs) (line 134)
        getattr_call_result_162169 = invoke(stypy.reporting.localization.Localization(__file__, 134, 17), getattr_162164, *[x_162165, unicode_162166, None_162167], **kwargs_162168)
        
        # Assigning a type to the variable 'classx' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'classx', getattr_call_result_162169)
        
        # Type idiom detected: calculating its left and rigth part (line 136)
        # Getting the type of 'classx' (line 136)
        classx_162170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'classx')
        # Getting the type of 'None' (line 136)
        None_162171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 25), 'None')
        
        (may_be_162172, more_types_in_union_162173) = may_not_be_none(classx_162170, None_162171)

        if may_be_162172:

            if more_types_in_union_162173:
                # Runtime conditional SSA (line 136)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 137):
            
            # Call to get(...): (line 137)
            # Processing the call arguments (line 137)
            # Getting the type of 'classx' (line 137)
            classx_162176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 33), 'classx', False)
            # Processing the call keyword arguments (line 137)
            kwargs_162177 = {}
            # Getting the type of 'self' (line 137)
            self_162174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 24), 'self', False)
            # Obtaining the member 'get' of a type (line 137)
            get_162175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 24), self_162174, 'get')
            # Calling get(args, kwargs) (line 137)
            get_call_result_162178 = invoke(stypy.reporting.localization.Localization(__file__, 137, 24), get_162175, *[classx_162176], **kwargs_162177)
            
            # Assigning a type to the variable 'converter' (line 137)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'converter', get_call_result_162178)

            if more_types_in_union_162173:
                # SSA join for if statement (line 136)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'x' (line 139)
        x_162180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 22), 'x', False)
        # Getting the type of 'np' (line 139)
        np_162181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 25), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 139)
        ndarray_162182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 25), np_162181, 'ndarray')
        # Processing the call keyword arguments (line 139)
        kwargs_162183 = {}
        # Getting the type of 'isinstance' (line 139)
        isinstance_162179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 139)
        isinstance_call_result_162184 = invoke(stypy.reporting.localization.Localization(__file__, 139, 11), isinstance_162179, *[x_162180, ndarray_162182], **kwargs_162183)
        
        # Getting the type of 'x' (line 139)
        x_162185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 41), 'x')
        # Obtaining the member 'size' of a type (line 139)
        size_162186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 41), x_162185, 'size')
        # Applying the binary operator 'and' (line 139)
        result_and_keyword_162187 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 11), 'and', isinstance_call_result_162184, size_162186)
        
        # Testing the type of an if condition (line 139)
        if_condition_162188 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 8), result_and_keyword_162187)
        # Assigning a type to the variable 'if_condition_162188' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'if_condition_162188', if_condition_162188)
        # SSA begins for if statement (line 139)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 140):
        
        # Call to ravel(...): (line 140)
        # Processing the call keyword arguments (line 140)
        kwargs_162191 = {}
        # Getting the type of 'x' (line 140)
        x_162189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 21), 'x', False)
        # Obtaining the member 'ravel' of a type (line 140)
        ravel_162190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 21), x_162189, 'ravel')
        # Calling ravel(args, kwargs) (line 140)
        ravel_call_result_162192 = invoke(stypy.reporting.localization.Localization(__file__, 140, 21), ravel_162190, *[], **kwargs_162191)
        
        # Assigning a type to the variable 'xravel' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'xravel', ravel_call_result_162192)
        
        
        # SSA begins for try-except statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        
        
        # Call to all(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'xravel' (line 144)
        xravel_162195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 30), 'xravel', False)
        # Obtaining the member 'mask' of a type (line 144)
        mask_162196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 30), xravel_162195, 'mask')
        # Processing the call keyword arguments (line 144)
        kwargs_162197 = {}
        # Getting the type of 'np' (line 144)
        np_162193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 23), 'np', False)
        # Obtaining the member 'all' of a type (line 144)
        all_162194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 23), np_162193, 'all')
        # Calling all(args, kwargs) (line 144)
        all_call_result_162198 = invoke(stypy.reporting.localization.Localization(__file__, 144, 23), all_162194, *[mask_162196], **kwargs_162197)
        
        # Applying the 'not' unary operator (line 144)
        result_not__162199 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 19), 'not', all_call_result_162198)
        
        # Testing the type of an if condition (line 144)
        if_condition_162200 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 16), result_not__162199)
        # Assigning a type to the variable 'if_condition_162200' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'if_condition_162200', if_condition_162200)
        # SSA begins for if statement (line 144)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 146):
        
        # Call to get_converter(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Obtaining the type of the subscript
        
        # Call to argmin(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'xravel' (line 147)
        xravel_162205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 41), 'xravel', False)
        # Obtaining the member 'mask' of a type (line 147)
        mask_162206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 41), xravel_162205, 'mask')
        # Processing the call keyword arguments (line 147)
        kwargs_162207 = {}
        # Getting the type of 'np' (line 147)
        np_162203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 31), 'np', False)
        # Obtaining the member 'argmin' of a type (line 147)
        argmin_162204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 31), np_162203, 'argmin')
        # Calling argmin(args, kwargs) (line 147)
        argmin_call_result_162208 = invoke(stypy.reporting.localization.Localization(__file__, 147, 31), argmin_162204, *[mask_162206], **kwargs_162207)
        
        # Getting the type of 'xravel' (line 147)
        xravel_162209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 24), 'xravel', False)
        # Obtaining the member '__getitem__' of a type (line 147)
        getitem___162210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 24), xravel_162209, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 147)
        subscript_call_result_162211 = invoke(stypy.reporting.localization.Localization(__file__, 147, 24), getitem___162210, argmin_call_result_162208)
        
        # Processing the call keyword arguments (line 146)
        kwargs_162212 = {}
        # Getting the type of 'self' (line 146)
        self_162201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 32), 'self', False)
        # Obtaining the member 'get_converter' of a type (line 146)
        get_converter_162202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 32), self_162201, 'get_converter')
        # Calling get_converter(args, kwargs) (line 146)
        get_converter_call_result_162213 = invoke(stypy.reporting.localization.Localization(__file__, 146, 32), get_converter_162202, *[subscript_call_result_162211], **kwargs_162212)
        
        # Assigning a type to the variable 'converter' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 20), 'converter', get_converter_call_result_162213)
        # Getting the type of 'converter' (line 148)
        converter_162214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 27), 'converter')
        # Assigning a type to the variable 'stypy_return_type' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'stypy_return_type', converter_162214)
        # SSA join for if statement (line 144)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the except part of a try statement (line 141)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 141)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Subscript to a Name (line 154):
        
        # Obtaining the type of the subscript
        int_162215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 35), 'int')
        # Getting the type of 'xravel' (line 154)
        xravel_162216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 28), 'xravel')
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___162217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 28), xravel_162216, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_162218 = invoke(stypy.reporting.localization.Localization(__file__, 154, 28), getitem___162217, int_162215)
        
        # Assigning a type to the variable 'next_item' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'next_item', subscript_call_result_162218)
        
        
        # Evaluating a boolean operation
        
        
        # Call to isinstance(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'next_item' (line 155)
        next_item_162220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 35), 'next_item', False)
        # Getting the type of 'np' (line 155)
        np_162221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 46), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 155)
        ndarray_162222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 46), np_162221, 'ndarray')
        # Processing the call keyword arguments (line 155)
        kwargs_162223 = {}
        # Getting the type of 'isinstance' (line 155)
        isinstance_162219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 24), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 155)
        isinstance_call_result_162224 = invoke(stypy.reporting.localization.Localization(__file__, 155, 24), isinstance_162219, *[next_item_162220, ndarray_162222], **kwargs_162223)
        
        # Applying the 'not' unary operator (line 155)
        result_not__162225 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 20), 'not', isinstance_call_result_162224)
        
        
        # Getting the type of 'next_item' (line 156)
        next_item_162226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 20), 'next_item')
        # Obtaining the member 'shape' of a type (line 156)
        shape_162227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 20), next_item_162226, 'shape')
        # Getting the type of 'x' (line 156)
        x_162228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 39), 'x')
        # Obtaining the member 'shape' of a type (line 156)
        shape_162229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 39), x_162228, 'shape')
        # Applying the binary operator '!=' (line 156)
        result_ne_162230 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 20), '!=', shape_162227, shape_162229)
        
        # Applying the binary operator 'or' (line 155)
        result_or_keyword_162231 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 20), 'or', result_not__162225, result_ne_162230)
        
        # Testing the type of an if condition (line 155)
        if_condition_162232 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 16), result_or_keyword_162231)
        # Assigning a type to the variable 'if_condition_162232' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'if_condition_162232', if_condition_162232)
        # SSA begins for if statement (line 155)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 157):
        
        # Call to get_converter(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'next_item' (line 157)
        next_item_162235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 51), 'next_item', False)
        # Processing the call keyword arguments (line 157)
        kwargs_162236 = {}
        # Getting the type of 'self' (line 157)
        self_162233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 32), 'self', False)
        # Obtaining the member 'get_converter' of a type (line 157)
        get_converter_162234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 32), self_162233, 'get_converter')
        # Calling get_converter(args, kwargs) (line 157)
        get_converter_call_result_162237 = invoke(stypy.reporting.localization.Localization(__file__, 157, 32), get_converter_162234, *[next_item_162235], **kwargs_162236)
        
        # Assigning a type to the variable 'converter' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'converter', get_converter_call_result_162237)
        # SSA join for if statement (line 155)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'converter' (line 158)
        converter_162238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 23), 'converter')
        # Assigning a type to the variable 'stypy_return_type' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'stypy_return_type', converter_162238)
        # SSA join for try-except statement (line 141)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 139)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 160)
        # Getting the type of 'converter' (line 160)
        converter_162239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 11), 'converter')
        # Getting the type of 'None' (line 160)
        None_162240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'None')
        
        (may_be_162241, more_types_in_union_162242) = may_be_none(converter_162239, None_162240)

        if may_be_162241:

            if more_types_in_union_162242:
                # Runtime conditional SSA (line 160)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # SSA begins for try-except statement (line 161)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 162):
            
            # Call to safe_first_element(...): (line 162)
            # Processing the call arguments (line 162)
            # Getting the type of 'x' (line 162)
            x_162244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 43), 'x', False)
            # Processing the call keyword arguments (line 162)
            kwargs_162245 = {}
            # Getting the type of 'safe_first_element' (line 162)
            safe_first_element_162243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 24), 'safe_first_element', False)
            # Calling safe_first_element(args, kwargs) (line 162)
            safe_first_element_call_result_162246 = invoke(stypy.reporting.localization.Localization(__file__, 162, 24), safe_first_element_162243, *[x_162244], **kwargs_162245)
            
            # Assigning a type to the variable 'thisx' (line 162)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'thisx', safe_first_element_call_result_162246)
            # SSA branch for the except part of a try statement (line 161)
            # SSA branch for the except 'Tuple' branch of a try statement (line 161)
            module_type_store.open_ssa_branch('except')
            pass
            # SSA branch for the else branch of a try statement (line 161)
            module_type_store.open_ssa_branch('except else')
            
            
            # Evaluating a boolean operation
            # Getting the type of 'classx' (line 166)
            classx_162247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 19), 'classx')
            
            # Getting the type of 'classx' (line 166)
            classx_162248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 30), 'classx')
            
            # Call to getattr(...): (line 166)
            # Processing the call arguments (line 166)
            # Getting the type of 'thisx' (line 166)
            thisx_162250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 48), 'thisx', False)
            unicode_162251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 55), 'unicode', u'__class__')
            # Getting the type of 'None' (line 166)
            None_162252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 68), 'None', False)
            # Processing the call keyword arguments (line 166)
            kwargs_162253 = {}
            # Getting the type of 'getattr' (line 166)
            getattr_162249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 40), 'getattr', False)
            # Calling getattr(args, kwargs) (line 166)
            getattr_call_result_162254 = invoke(stypy.reporting.localization.Localization(__file__, 166, 40), getattr_162249, *[thisx_162250, unicode_162251, None_162252], **kwargs_162253)
            
            # Applying the binary operator '!=' (line 166)
            result_ne_162255 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 30), '!=', classx_162248, getattr_call_result_162254)
            
            # Applying the binary operator 'and' (line 166)
            result_and_keyword_162256 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 19), 'and', classx_162247, result_ne_162255)
            
            # Testing the type of an if condition (line 166)
            if_condition_162257 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 166, 16), result_and_keyword_162256)
            # Assigning a type to the variable 'if_condition_162257' (line 166)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'if_condition_162257', if_condition_162257)
            # SSA begins for if statement (line 166)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 167):
            
            # Call to get_converter(...): (line 167)
            # Processing the call arguments (line 167)
            # Getting the type of 'thisx' (line 167)
            thisx_162260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 51), 'thisx', False)
            # Processing the call keyword arguments (line 167)
            kwargs_162261 = {}
            # Getting the type of 'self' (line 167)
            self_162258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 32), 'self', False)
            # Obtaining the member 'get_converter' of a type (line 167)
            get_converter_162259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 32), self_162258, 'get_converter')
            # Calling get_converter(args, kwargs) (line 167)
            get_converter_call_result_162262 = invoke(stypy.reporting.localization.Localization(__file__, 167, 32), get_converter_162259, *[thisx_162260], **kwargs_162261)
            
            # Assigning a type to the variable 'converter' (line 167)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 20), 'converter', get_converter_call_result_162262)
            # Getting the type of 'converter' (line 168)
            converter_162263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 27), 'converter')
            # Assigning a type to the variable 'stypy_return_type' (line 168)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 20), 'stypy_return_type', converter_162263)
            # SSA join for if statement (line 166)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for try-except statement (line 161)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_162242:
                # SSA join for if statement (line 160)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'converter' (line 171)
        converter_162264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 15), 'converter')
        # Assigning a type to the variable 'stypy_return_type' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'stypy_return_type', converter_162264)
        
        # ################# End of 'get_converter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_converter' in the type store
        # Getting the type of 'stypy_return_type' (line 124)
        stypy_return_type_162265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_162265)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_converter'
        return stypy_return_type_162265


# Assigning a type to the variable 'Registry' (line 116)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'Registry', Registry)

# Assigning a Call to a Name (line 174):

# Call to Registry(...): (line 174)
# Processing the call keyword arguments (line 174)
kwargs_162267 = {}
# Getting the type of 'Registry' (line 174)
Registry_162266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 'Registry', False)
# Calling Registry(args, kwargs) (line 174)
Registry_call_result_162268 = invoke(stypy.reporting.localization.Localization(__file__, 174, 11), Registry_162266, *[], **kwargs_162267)

# Assigning a type to the variable 'registry' (line 174)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 0), 'registry', Registry_call_result_162268)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
