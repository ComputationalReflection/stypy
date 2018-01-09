
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #===========================================================================
2: #
3: # StrConverter
4: #
5: #===========================================================================
6: 
7: 
8: '''StrConverter module containing class StrConverter.'''
9: 
10: #===========================================================================
11: # Place all imports after here.
12: #
13: from __future__ import (absolute_import, division, print_function,
14:                         unicode_literals)
15: 
16: import six
17: from six.moves import xrange
18: 
19: import matplotlib.units as units
20: from matplotlib.cbook import iterable
21: 
22: # Place all imports before here.
23: #===========================================================================
24: 
25: __all__ = [ 'StrConverter' ]
26: 
27: #===========================================================================
28: class StrConverter( units.ConversionInterface ):
29:    ''': A matplotlib converter class.  Provides matplotlib conversion
30:         functionality for string data values.
31: 
32:    Valid units for string are:
33:    - 'indexed' : Values are indexed as they are specified for plotting.
34:    - 'sorted'  : Values are sorted alphanumerically.
35:    - 'inverted' : Values are inverted so that the first value is on top.
36:    - 'sorted-inverted' :  A combination of 'sorted' and 'inverted'
37:    '''
38: 
39:    #------------------------------------------------------------------------
40:    @staticmethod
41:    def axisinfo( unit, axis ):
42:       ''': Returns information on how to handle an axis that has string data.
43: 
44:       = INPUT VARIABLES
45:       - axis    The axis using this converter.
46:       - unit    The units to use for a axis with string data.
47: 
48:       = RETURN VALUE
49:       - Returns a matplotlib AxisInfo data structure that contains
50:         minor/major formatters, major/minor locators, and default
51:         label information.
52:       '''
53: 
54:       return None
55: 
56:    #------------------------------------------------------------------------
57:    @staticmethod
58:    def convert( value, unit, axis ):
59:       ''': Convert value using unit to a float.  If value is a sequence, return
60:       the converted sequence.
61: 
62:       = INPUT VARIABLES
63:       - axis    The axis using this converter.
64:       - value   The value or list of values that need to be converted.
65:       - unit    The units to use for a axis with Epoch data.
66: 
67:       = RETURN VALUE
68:       - Returns the value parameter converted to floats.
69:       '''
70: 
71:       if ( units.ConversionInterface.is_numlike( value ) ):
72:          return value
73: 
74:       if ( value == [] ):
75:          return []
76: 
77:       # we delay loading to make matplotlib happy
78:       ax = axis.axes
79:       if axis is ax.get_xaxis():
80:          isXAxis = True
81:       else:
82:          isXAxis = False
83: 
84:       axis.get_major_ticks()
85:       ticks = axis.get_ticklocs()
86:       labels = axis.get_ticklabels()
87: 
88:       labels = [ l.get_text() for l in labels if l.get_text() ]
89: 
90:       if ( not labels ):
91:          ticks = []
92:          labels = []
93: 
94: 
95:       if ( not iterable( value ) ):
96:          value = [ value ]
97: 
98:       newValues = []
99:       for v in value:
100:          if ( (v not in labels) and (v not in newValues) ):
101:             newValues.append( v )
102: 
103:       for v in newValues:
104:          if ( labels ):
105:             labels.append( v )
106:          else:
107:             labels = [ v ]
108: 
109:       #DISABLED: This is disabled because matplotlib bar plots do not
110:       #DISABLED: recalculate the unit conversion of the data values
111:       #DISABLED: this is due to design and is not really a bug.
112:       #DISABLED: If this gets changed, then we can activate the following
113:       #DISABLED: block of code.  Note that this works for line plots.
114:       #DISABLED if ( unit ):
115:       #DISABLED    if ( unit.find( "sorted" ) > -1 ):
116:       #DISABLED       labels.sort()
117:       #DISABLED    if ( unit.find( "inverted" ) > -1 ):
118:       #DISABLED       labels = labels[ ::-1 ]
119: 
120:       # add padding (so they do not appear on the axes themselves)
121:       labels = [ '' ] + labels + [ '' ]
122:       ticks = list(xrange( len(labels) ))
123:       ticks[0] = 0.5
124:       ticks[-1] = ticks[-1] - 0.5
125: 
126:       axis.set_ticks( ticks )
127:       axis.set_ticklabels( labels )
128:       # we have to do the following lines to make ax.autoscale_view work
129:       loc = axis.get_major_locator()
130:       loc.set_bounds( ticks[0], ticks[-1] )
131: 
132:       if ( isXAxis ):
133:          ax.set_xlim( ticks[0], ticks[-1] )
134:       else:
135:          ax.set_ylim( ticks[0], ticks[-1] )
136: 
137:       result = []
138:       for v in value:
139:          # If v is not in labels then something went wrong with adding new
140:          # labels to the list of old labels.
141:          errmsg  = "This is due to a logic error in the StrConverter class.  "
142:          errmsg += "Please report this error and its message in bugzilla."
143:          assert ( v in labels ), errmsg
144:          result.append( ticks[ labels.index(v) ] )
145: 
146:       ax.viewLim.ignore(-1)
147:       return result
148: 
149:    #------------------------------------------------------------------------
150:    @staticmethod
151:    def default_units( value, axis ):
152:       ''': Return the default unit for value, or None.
153: 
154:       = INPUT VARIABLES
155:       - axis    The axis using this converter.
156:       - value   The value or list of values that need units.
157: 
158:       = RETURN VALUE
159:       - Returns the default units to use for value.
160:       Return the default unit for value, or None.
161:       '''
162: 
163:       # The default behavior for string indexing.
164:       return "indexed"
165: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_293105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 0), 'unicode', u'StrConverter module containing class StrConverter.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import six' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_293106 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six')

if (type(import_293106) is not StypyTypeError):

    if (import_293106 != 'pyd_module'):
        __import__(import_293106)
        sys_modules_293107 = sys.modules[import_293106]
        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', sys_modules_293107.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', import_293106)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from six.moves import xrange' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_293108 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'six.moves')

if (type(import_293108) is not StypyTypeError):

    if (import_293108 != 'pyd_module'):
        __import__(import_293108)
        sys_modules_293109 = sys.modules[import_293108]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'six.moves', sys_modules_293109.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_293109, sys_modules_293109.module_type_store, module_type_store)
    else:
        from six.moves import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'six.moves', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'six.moves' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'six.moves', import_293108)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import matplotlib.units' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_293110 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.units')

if (type(import_293110) is not StypyTypeError):

    if (import_293110 != 'pyd_module'):
        __import__(import_293110)
        sys_modules_293111 = sys.modules[import_293110]
        import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'units', sys_modules_293111.module_type_store, module_type_store)
    else:
        import matplotlib.units as units

        import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'units', matplotlib.units, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.units' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.units', import_293110)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from matplotlib.cbook import iterable' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_293112 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.cbook')

if (type(import_293112) is not StypyTypeError):

    if (import_293112 != 'pyd_module'):
        __import__(import_293112)
        sys_modules_293113 = sys.modules[import_293112]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.cbook', sys_modules_293113.module_type_store, module_type_store, ['iterable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_293113, sys_modules_293113.module_type_store, module_type_store)
    else:
        from matplotlib.cbook import iterable

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.cbook', None, module_type_store, ['iterable'], [iterable])

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.cbook', import_293112)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')


# Assigning a List to a Name (line 25):
__all__ = [u'StrConverter']
module_type_store.set_exportable_members([u'StrConverter'])

# Obtaining an instance of the builtin type 'list' (line 25)
list_293114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)
unicode_293115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 12), 'unicode', u'StrConverter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 10), list_293114, unicode_293115)

# Assigning a type to the variable '__all__' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), '__all__', list_293114)
# Declaration of the 'StrConverter' class
# Getting the type of 'units' (line 28)
units_293116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 20), 'units')
# Obtaining the member 'ConversionInterface' of a type (line 28)
ConversionInterface_293117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 20), units_293116, 'ConversionInterface')

class StrConverter(ConversionInterface_293117, ):
    unicode_293118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, (-1)), 'unicode', u": A matplotlib converter class.  Provides matplotlib conversion\n        functionality for string data values.\n\n   Valid units for string are:\n   - 'indexed' : Values are indexed as they are specified for plotting.\n   - 'sorted'  : Values are sorted alphanumerically.\n   - 'inverted' : Values are inverted so that the first value is on top.\n   - 'sorted-inverted' :  A combination of 'sorted' and 'inverted'\n   ")

    @staticmethod
    @norecursion
    def axisinfo(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'axisinfo'
        module_type_store = module_type_store.open_function_context('axisinfo', 40, 3, False)
        
        # Passed parameters checking function
        StrConverter.axisinfo.__dict__.__setitem__('stypy_localization', localization)
        StrConverter.axisinfo.__dict__.__setitem__('stypy_type_of_self', None)
        StrConverter.axisinfo.__dict__.__setitem__('stypy_type_store', module_type_store)
        StrConverter.axisinfo.__dict__.__setitem__('stypy_function_name', 'axisinfo')
        StrConverter.axisinfo.__dict__.__setitem__('stypy_param_names_list', ['unit', 'axis'])
        StrConverter.axisinfo.__dict__.__setitem__('stypy_varargs_param_name', None)
        StrConverter.axisinfo.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StrConverter.axisinfo.__dict__.__setitem__('stypy_call_defaults', defaults)
        StrConverter.axisinfo.__dict__.__setitem__('stypy_call_varargs', varargs)
        StrConverter.axisinfo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StrConverter.axisinfo.__dict__.__setitem__('stypy_declared_arg_number', 2)
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

        unicode_293119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, (-1)), 'unicode', u': Returns information on how to handle an axis that has string data.\n\n      = INPUT VARIABLES\n      - axis    The axis using this converter.\n      - unit    The units to use for a axis with string data.\n\n      = RETURN VALUE\n      - Returns a matplotlib AxisInfo data structure that contains\n        minor/major formatters, major/minor locators, and default\n        label information.\n      ')
        # Getting the type of 'None' (line 54)
        None_293120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 13), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 6), 'stypy_return_type', None_293120)
        
        # ################# End of 'axisinfo(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'axisinfo' in the type store
        # Getting the type of 'stypy_return_type' (line 40)
        stypy_return_type_293121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293121)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'axisinfo'
        return stypy_return_type_293121


    @staticmethod
    @norecursion
    def convert(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'convert'
        module_type_store = module_type_store.open_function_context('convert', 57, 3, False)
        
        # Passed parameters checking function
        StrConverter.convert.__dict__.__setitem__('stypy_localization', localization)
        StrConverter.convert.__dict__.__setitem__('stypy_type_of_self', None)
        StrConverter.convert.__dict__.__setitem__('stypy_type_store', module_type_store)
        StrConverter.convert.__dict__.__setitem__('stypy_function_name', 'convert')
        StrConverter.convert.__dict__.__setitem__('stypy_param_names_list', ['value', 'unit', 'axis'])
        StrConverter.convert.__dict__.__setitem__('stypy_varargs_param_name', None)
        StrConverter.convert.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StrConverter.convert.__dict__.__setitem__('stypy_call_defaults', defaults)
        StrConverter.convert.__dict__.__setitem__('stypy_call_varargs', varargs)
        StrConverter.convert.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StrConverter.convert.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, None, module_type_store, 'convert', ['value', 'unit', 'axis'], None, None, defaults, varargs, kwargs)

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

        unicode_293122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, (-1)), 'unicode', u': Convert value using unit to a float.  If value is a sequence, return\n      the converted sequence.\n\n      = INPUT VARIABLES\n      - axis    The axis using this converter.\n      - value   The value or list of values that need to be converted.\n      - unit    The units to use for a axis with Epoch data.\n\n      = RETURN VALUE\n      - Returns the value parameter converted to floats.\n      ')
        
        
        # Call to is_numlike(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'value' (line 71)
        value_293126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 49), 'value', False)
        # Processing the call keyword arguments (line 71)
        kwargs_293127 = {}
        # Getting the type of 'units' (line 71)
        units_293123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 11), 'units', False)
        # Obtaining the member 'ConversionInterface' of a type (line 71)
        ConversionInterface_293124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 11), units_293123, 'ConversionInterface')
        # Obtaining the member 'is_numlike' of a type (line 71)
        is_numlike_293125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 11), ConversionInterface_293124, 'is_numlike')
        # Calling is_numlike(args, kwargs) (line 71)
        is_numlike_call_result_293128 = invoke(stypy.reporting.localization.Localization(__file__, 71, 11), is_numlike_293125, *[value_293126], **kwargs_293127)
        
        # Testing the type of an if condition (line 71)
        if_condition_293129 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 6), is_numlike_call_result_293128)
        # Assigning a type to the variable 'if_condition_293129' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 6), 'if_condition_293129', if_condition_293129)
        # SSA begins for if statement (line 71)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'value' (line 72)
        value_293130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'value')
        # Assigning a type to the variable 'stypy_return_type' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 9), 'stypy_return_type', value_293130)
        # SSA join for if statement (line 71)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'value' (line 74)
        value_293131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 11), 'value')
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_293132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        
        # Applying the binary operator '==' (line 74)
        result_eq_293133 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 11), '==', value_293131, list_293132)
        
        # Testing the type of an if condition (line 74)
        if_condition_293134 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 6), result_eq_293133)
        # Assigning a type to the variable 'if_condition_293134' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 6), 'if_condition_293134', if_condition_293134)
        # SSA begins for if statement (line 74)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_293135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        
        # Assigning a type to the variable 'stypy_return_type' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 9), 'stypy_return_type', list_293135)
        # SSA join for if statement (line 74)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 78):
        # Getting the type of 'axis' (line 78)
        axis_293136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), 'axis')
        # Obtaining the member 'axes' of a type (line 78)
        axes_293137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 11), axis_293136, 'axes')
        # Assigning a type to the variable 'ax' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 6), 'ax', axes_293137)
        
        
        # Getting the type of 'axis' (line 79)
        axis_293138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 9), 'axis')
        
        # Call to get_xaxis(...): (line 79)
        # Processing the call keyword arguments (line 79)
        kwargs_293141 = {}
        # Getting the type of 'ax' (line 79)
        ax_293139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 17), 'ax', False)
        # Obtaining the member 'get_xaxis' of a type (line 79)
        get_xaxis_293140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 17), ax_293139, 'get_xaxis')
        # Calling get_xaxis(args, kwargs) (line 79)
        get_xaxis_call_result_293142 = invoke(stypy.reporting.localization.Localization(__file__, 79, 17), get_xaxis_293140, *[], **kwargs_293141)
        
        # Applying the binary operator 'is' (line 79)
        result_is__293143 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 9), 'is', axis_293138, get_xaxis_call_result_293142)
        
        # Testing the type of an if condition (line 79)
        if_condition_293144 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 6), result_is__293143)
        # Assigning a type to the variable 'if_condition_293144' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 6), 'if_condition_293144', if_condition_293144)
        # SSA begins for if statement (line 79)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 80):
        # Getting the type of 'True' (line 80)
        True_293145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 19), 'True')
        # Assigning a type to the variable 'isXAxis' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 9), 'isXAxis', True_293145)
        # SSA branch for the else part of an if statement (line 79)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 82):
        # Getting the type of 'False' (line 82)
        False_293146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'False')
        # Assigning a type to the variable 'isXAxis' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 9), 'isXAxis', False_293146)
        # SSA join for if statement (line 79)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to get_major_ticks(...): (line 84)
        # Processing the call keyword arguments (line 84)
        kwargs_293149 = {}
        # Getting the type of 'axis' (line 84)
        axis_293147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 6), 'axis', False)
        # Obtaining the member 'get_major_ticks' of a type (line 84)
        get_major_ticks_293148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 6), axis_293147, 'get_major_ticks')
        # Calling get_major_ticks(args, kwargs) (line 84)
        get_major_ticks_call_result_293150 = invoke(stypy.reporting.localization.Localization(__file__, 84, 6), get_major_ticks_293148, *[], **kwargs_293149)
        
        
        # Assigning a Call to a Name (line 85):
        
        # Call to get_ticklocs(...): (line 85)
        # Processing the call keyword arguments (line 85)
        kwargs_293153 = {}
        # Getting the type of 'axis' (line 85)
        axis_293151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 14), 'axis', False)
        # Obtaining the member 'get_ticklocs' of a type (line 85)
        get_ticklocs_293152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 14), axis_293151, 'get_ticklocs')
        # Calling get_ticklocs(args, kwargs) (line 85)
        get_ticklocs_call_result_293154 = invoke(stypy.reporting.localization.Localization(__file__, 85, 14), get_ticklocs_293152, *[], **kwargs_293153)
        
        # Assigning a type to the variable 'ticks' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 6), 'ticks', get_ticklocs_call_result_293154)
        
        # Assigning a Call to a Name (line 86):
        
        # Call to get_ticklabels(...): (line 86)
        # Processing the call keyword arguments (line 86)
        kwargs_293157 = {}
        # Getting the type of 'axis' (line 86)
        axis_293155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'axis', False)
        # Obtaining the member 'get_ticklabels' of a type (line 86)
        get_ticklabels_293156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 15), axis_293155, 'get_ticklabels')
        # Calling get_ticklabels(args, kwargs) (line 86)
        get_ticklabels_call_result_293158 = invoke(stypy.reporting.localization.Localization(__file__, 86, 15), get_ticklabels_293156, *[], **kwargs_293157)
        
        # Assigning a type to the variable 'labels' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 6), 'labels', get_ticklabels_call_result_293158)
        
        # Assigning a ListComp to a Name (line 88):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'labels' (line 88)
        labels_293167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 39), 'labels')
        comprehension_293168 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 17), labels_293167)
        # Assigning a type to the variable 'l' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 17), 'l', comprehension_293168)
        
        # Call to get_text(...): (line 88)
        # Processing the call keyword arguments (line 88)
        kwargs_293165 = {}
        # Getting the type of 'l' (line 88)
        l_293163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 49), 'l', False)
        # Obtaining the member 'get_text' of a type (line 88)
        get_text_293164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 49), l_293163, 'get_text')
        # Calling get_text(args, kwargs) (line 88)
        get_text_call_result_293166 = invoke(stypy.reporting.localization.Localization(__file__, 88, 49), get_text_293164, *[], **kwargs_293165)
        
        
        # Call to get_text(...): (line 88)
        # Processing the call keyword arguments (line 88)
        kwargs_293161 = {}
        # Getting the type of 'l' (line 88)
        l_293159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 17), 'l', False)
        # Obtaining the member 'get_text' of a type (line 88)
        get_text_293160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 17), l_293159, 'get_text')
        # Calling get_text(args, kwargs) (line 88)
        get_text_call_result_293162 = invoke(stypy.reporting.localization.Localization(__file__, 88, 17), get_text_293160, *[], **kwargs_293161)
        
        list_293169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 17), list_293169, get_text_call_result_293162)
        # Assigning a type to the variable 'labels' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 6), 'labels', list_293169)
        
        
        # Getting the type of 'labels' (line 90)
        labels_293170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 15), 'labels')
        # Applying the 'not' unary operator (line 90)
        result_not__293171 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 11), 'not', labels_293170)
        
        # Testing the type of an if condition (line 90)
        if_condition_293172 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 6), result_not__293171)
        # Assigning a type to the variable 'if_condition_293172' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 6), 'if_condition_293172', if_condition_293172)
        # SSA begins for if statement (line 90)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 91):
        
        # Obtaining an instance of the builtin type 'list' (line 91)
        list_293173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 91)
        
        # Assigning a type to the variable 'ticks' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 9), 'ticks', list_293173)
        
        # Assigning a List to a Name (line 92):
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_293174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        
        # Assigning a type to the variable 'labels' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 9), 'labels', list_293174)
        # SSA join for if statement (line 90)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to iterable(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'value' (line 95)
        value_293176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 25), 'value', False)
        # Processing the call keyword arguments (line 95)
        kwargs_293177 = {}
        # Getting the type of 'iterable' (line 95)
        iterable_293175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'iterable', False)
        # Calling iterable(args, kwargs) (line 95)
        iterable_call_result_293178 = invoke(stypy.reporting.localization.Localization(__file__, 95, 15), iterable_293175, *[value_293176], **kwargs_293177)
        
        # Applying the 'not' unary operator (line 95)
        result_not__293179 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 11), 'not', iterable_call_result_293178)
        
        # Testing the type of an if condition (line 95)
        if_condition_293180 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 6), result_not__293179)
        # Assigning a type to the variable 'if_condition_293180' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 6), 'if_condition_293180', if_condition_293180)
        # SSA begins for if statement (line 95)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 96):
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_293181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        # Getting the type of 'value' (line 96)
        value_293182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 19), 'value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 17), list_293181, value_293182)
        
        # Assigning a type to the variable 'value' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 9), 'value', list_293181)
        # SSA join for if statement (line 95)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 98):
        
        # Obtaining an instance of the builtin type 'list' (line 98)
        list_293183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 98)
        
        # Assigning a type to the variable 'newValues' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 6), 'newValues', list_293183)
        
        # Getting the type of 'value' (line 99)
        value_293184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 15), 'value')
        # Testing the type of a for loop iterable (line 99)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 99, 6), value_293184)
        # Getting the type of the for loop variable (line 99)
        for_loop_var_293185 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 99, 6), value_293184)
        # Assigning a type to the variable 'v' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 6), 'v', for_loop_var_293185)
        # SSA begins for a for statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'v' (line 100)
        v_293186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'v')
        # Getting the type of 'labels' (line 100)
        labels_293187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 24), 'labels')
        # Applying the binary operator 'notin' (line 100)
        result_contains_293188 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 15), 'notin', v_293186, labels_293187)
        
        
        # Getting the type of 'v' (line 100)
        v_293189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 37), 'v')
        # Getting the type of 'newValues' (line 100)
        newValues_293190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 46), 'newValues')
        # Applying the binary operator 'notin' (line 100)
        result_contains_293191 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 37), 'notin', v_293189, newValues_293190)
        
        # Applying the binary operator 'and' (line 100)
        result_and_keyword_293192 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 14), 'and', result_contains_293188, result_contains_293191)
        
        # Testing the type of an if condition (line 100)
        if_condition_293193 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 9), result_and_keyword_293192)
        # Assigning a type to the variable 'if_condition_293193' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 9), 'if_condition_293193', if_condition_293193)
        # SSA begins for if statement (line 100)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'v' (line 101)
        v_293196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'v', False)
        # Processing the call keyword arguments (line 101)
        kwargs_293197 = {}
        # Getting the type of 'newValues' (line 101)
        newValues_293194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'newValues', False)
        # Obtaining the member 'append' of a type (line 101)
        append_293195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), newValues_293194, 'append')
        # Calling append(args, kwargs) (line 101)
        append_call_result_293198 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), append_293195, *[v_293196], **kwargs_293197)
        
        # SSA join for if statement (line 100)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'newValues' (line 103)
        newValues_293199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'newValues')
        # Testing the type of a for loop iterable (line 103)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 103, 6), newValues_293199)
        # Getting the type of the for loop variable (line 103)
        for_loop_var_293200 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 103, 6), newValues_293199)
        # Assigning a type to the variable 'v' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 6), 'v', for_loop_var_293200)
        # SSA begins for a for statement (line 103)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'labels' (line 104)
        labels_293201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 14), 'labels')
        # Testing the type of an if condition (line 104)
        if_condition_293202 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 9), labels_293201)
        # Assigning a type to the variable 'if_condition_293202' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 9), 'if_condition_293202', if_condition_293202)
        # SSA begins for if statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'v' (line 105)
        v_293205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 27), 'v', False)
        # Processing the call keyword arguments (line 105)
        kwargs_293206 = {}
        # Getting the type of 'labels' (line 105)
        labels_293203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'labels', False)
        # Obtaining the member 'append' of a type (line 105)
        append_293204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), labels_293203, 'append')
        # Calling append(args, kwargs) (line 105)
        append_call_result_293207 = invoke(stypy.reporting.localization.Localization(__file__, 105, 12), append_293204, *[v_293205], **kwargs_293206)
        
        # SSA branch for the else part of an if statement (line 104)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 107):
        
        # Obtaining an instance of the builtin type 'list' (line 107)
        list_293208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 107)
        # Adding element type (line 107)
        # Getting the type of 'v' (line 107)
        v_293209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 23), 'v')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 21), list_293208, v_293209)
        
        # Assigning a type to the variable 'labels' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'labels', list_293208)
        # SSA join for if statement (line 104)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 121):
        
        # Obtaining an instance of the builtin type 'list' (line 121)
        list_293210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 121)
        # Adding element type (line 121)
        unicode_293211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 17), 'unicode', u'')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 15), list_293210, unicode_293211)
        
        # Getting the type of 'labels' (line 121)
        labels_293212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 24), 'labels')
        # Applying the binary operator '+' (line 121)
        result_add_293213 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 15), '+', list_293210, labels_293212)
        
        
        # Obtaining an instance of the builtin type 'list' (line 121)
        list_293214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 121)
        # Adding element type (line 121)
        unicode_293215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 35), 'unicode', u'')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 33), list_293214, unicode_293215)
        
        # Applying the binary operator '+' (line 121)
        result_add_293216 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 31), '+', result_add_293213, list_293214)
        
        # Assigning a type to the variable 'labels' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 6), 'labels', result_add_293216)
        
        # Assigning a Call to a Name (line 122):
        
        # Call to list(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Call to xrange(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Call to len(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'labels' (line 122)
        labels_293220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 31), 'labels', False)
        # Processing the call keyword arguments (line 122)
        kwargs_293221 = {}
        # Getting the type of 'len' (line 122)
        len_293219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 27), 'len', False)
        # Calling len(args, kwargs) (line 122)
        len_call_result_293222 = invoke(stypy.reporting.localization.Localization(__file__, 122, 27), len_293219, *[labels_293220], **kwargs_293221)
        
        # Processing the call keyword arguments (line 122)
        kwargs_293223 = {}
        # Getting the type of 'xrange' (line 122)
        xrange_293218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 19), 'xrange', False)
        # Calling xrange(args, kwargs) (line 122)
        xrange_call_result_293224 = invoke(stypy.reporting.localization.Localization(__file__, 122, 19), xrange_293218, *[len_call_result_293222], **kwargs_293223)
        
        # Processing the call keyword arguments (line 122)
        kwargs_293225 = {}
        # Getting the type of 'list' (line 122)
        list_293217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 14), 'list', False)
        # Calling list(args, kwargs) (line 122)
        list_call_result_293226 = invoke(stypy.reporting.localization.Localization(__file__, 122, 14), list_293217, *[xrange_call_result_293224], **kwargs_293225)
        
        # Assigning a type to the variable 'ticks' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 6), 'ticks', list_call_result_293226)
        
        # Assigning a Num to a Subscript (line 123):
        float_293227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 17), 'float')
        # Getting the type of 'ticks' (line 123)
        ticks_293228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 6), 'ticks')
        int_293229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 12), 'int')
        # Storing an element on a container (line 123)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 6), ticks_293228, (int_293229, float_293227))
        
        # Assigning a BinOp to a Subscript (line 124):
        
        # Obtaining the type of the subscript
        int_293230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 24), 'int')
        # Getting the type of 'ticks' (line 124)
        ticks_293231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 18), 'ticks')
        # Obtaining the member '__getitem__' of a type (line 124)
        getitem___293232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 18), ticks_293231, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 124)
        subscript_call_result_293233 = invoke(stypy.reporting.localization.Localization(__file__, 124, 18), getitem___293232, int_293230)
        
        float_293234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 30), 'float')
        # Applying the binary operator '-' (line 124)
        result_sub_293235 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 18), '-', subscript_call_result_293233, float_293234)
        
        # Getting the type of 'ticks' (line 124)
        ticks_293236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 6), 'ticks')
        int_293237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 12), 'int')
        # Storing an element on a container (line 124)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 6), ticks_293236, (int_293237, result_sub_293235))
        
        # Call to set_ticks(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'ticks' (line 126)
        ticks_293240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 22), 'ticks', False)
        # Processing the call keyword arguments (line 126)
        kwargs_293241 = {}
        # Getting the type of 'axis' (line 126)
        axis_293238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 6), 'axis', False)
        # Obtaining the member 'set_ticks' of a type (line 126)
        set_ticks_293239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 6), axis_293238, 'set_ticks')
        # Calling set_ticks(args, kwargs) (line 126)
        set_ticks_call_result_293242 = invoke(stypy.reporting.localization.Localization(__file__, 126, 6), set_ticks_293239, *[ticks_293240], **kwargs_293241)
        
        
        # Call to set_ticklabels(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'labels' (line 127)
        labels_293245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 27), 'labels', False)
        # Processing the call keyword arguments (line 127)
        kwargs_293246 = {}
        # Getting the type of 'axis' (line 127)
        axis_293243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 6), 'axis', False)
        # Obtaining the member 'set_ticklabels' of a type (line 127)
        set_ticklabels_293244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 6), axis_293243, 'set_ticklabels')
        # Calling set_ticklabels(args, kwargs) (line 127)
        set_ticklabels_call_result_293247 = invoke(stypy.reporting.localization.Localization(__file__, 127, 6), set_ticklabels_293244, *[labels_293245], **kwargs_293246)
        
        
        # Assigning a Call to a Name (line 129):
        
        # Call to get_major_locator(...): (line 129)
        # Processing the call keyword arguments (line 129)
        kwargs_293250 = {}
        # Getting the type of 'axis' (line 129)
        axis_293248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'axis', False)
        # Obtaining the member 'get_major_locator' of a type (line 129)
        get_major_locator_293249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 12), axis_293248, 'get_major_locator')
        # Calling get_major_locator(args, kwargs) (line 129)
        get_major_locator_call_result_293251 = invoke(stypy.reporting.localization.Localization(__file__, 129, 12), get_major_locator_293249, *[], **kwargs_293250)
        
        # Assigning a type to the variable 'loc' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 6), 'loc', get_major_locator_call_result_293251)
        
        # Call to set_bounds(...): (line 130)
        # Processing the call arguments (line 130)
        
        # Obtaining the type of the subscript
        int_293254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 28), 'int')
        # Getting the type of 'ticks' (line 130)
        ticks_293255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 22), 'ticks', False)
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___293256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 22), ticks_293255, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_293257 = invoke(stypy.reporting.localization.Localization(__file__, 130, 22), getitem___293256, int_293254)
        
        
        # Obtaining the type of the subscript
        int_293258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 38), 'int')
        # Getting the type of 'ticks' (line 130)
        ticks_293259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 32), 'ticks', False)
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___293260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 32), ticks_293259, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_293261 = invoke(stypy.reporting.localization.Localization(__file__, 130, 32), getitem___293260, int_293258)
        
        # Processing the call keyword arguments (line 130)
        kwargs_293262 = {}
        # Getting the type of 'loc' (line 130)
        loc_293252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 6), 'loc', False)
        # Obtaining the member 'set_bounds' of a type (line 130)
        set_bounds_293253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 6), loc_293252, 'set_bounds')
        # Calling set_bounds(args, kwargs) (line 130)
        set_bounds_call_result_293263 = invoke(stypy.reporting.localization.Localization(__file__, 130, 6), set_bounds_293253, *[subscript_call_result_293257, subscript_call_result_293261], **kwargs_293262)
        
        
        # Getting the type of 'isXAxis' (line 132)
        isXAxis_293264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 11), 'isXAxis')
        # Testing the type of an if condition (line 132)
        if_condition_293265 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 132, 6), isXAxis_293264)
        # Assigning a type to the variable 'if_condition_293265' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 6), 'if_condition_293265', if_condition_293265)
        # SSA begins for if statement (line 132)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_xlim(...): (line 133)
        # Processing the call arguments (line 133)
        
        # Obtaining the type of the subscript
        int_293268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 28), 'int')
        # Getting the type of 'ticks' (line 133)
        ticks_293269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), 'ticks', False)
        # Obtaining the member '__getitem__' of a type (line 133)
        getitem___293270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 22), ticks_293269, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 133)
        subscript_call_result_293271 = invoke(stypy.reporting.localization.Localization(__file__, 133, 22), getitem___293270, int_293268)
        
        
        # Obtaining the type of the subscript
        int_293272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 38), 'int')
        # Getting the type of 'ticks' (line 133)
        ticks_293273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 32), 'ticks', False)
        # Obtaining the member '__getitem__' of a type (line 133)
        getitem___293274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 32), ticks_293273, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 133)
        subscript_call_result_293275 = invoke(stypy.reporting.localization.Localization(__file__, 133, 32), getitem___293274, int_293272)
        
        # Processing the call keyword arguments (line 133)
        kwargs_293276 = {}
        # Getting the type of 'ax' (line 133)
        ax_293266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 9), 'ax', False)
        # Obtaining the member 'set_xlim' of a type (line 133)
        set_xlim_293267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 9), ax_293266, 'set_xlim')
        # Calling set_xlim(args, kwargs) (line 133)
        set_xlim_call_result_293277 = invoke(stypy.reporting.localization.Localization(__file__, 133, 9), set_xlim_293267, *[subscript_call_result_293271, subscript_call_result_293275], **kwargs_293276)
        
        # SSA branch for the else part of an if statement (line 132)
        module_type_store.open_ssa_branch('else')
        
        # Call to set_ylim(...): (line 135)
        # Processing the call arguments (line 135)
        
        # Obtaining the type of the subscript
        int_293280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 28), 'int')
        # Getting the type of 'ticks' (line 135)
        ticks_293281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 22), 'ticks', False)
        # Obtaining the member '__getitem__' of a type (line 135)
        getitem___293282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 22), ticks_293281, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
        subscript_call_result_293283 = invoke(stypy.reporting.localization.Localization(__file__, 135, 22), getitem___293282, int_293280)
        
        
        # Obtaining the type of the subscript
        int_293284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 38), 'int')
        # Getting the type of 'ticks' (line 135)
        ticks_293285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 32), 'ticks', False)
        # Obtaining the member '__getitem__' of a type (line 135)
        getitem___293286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 32), ticks_293285, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
        subscript_call_result_293287 = invoke(stypy.reporting.localization.Localization(__file__, 135, 32), getitem___293286, int_293284)
        
        # Processing the call keyword arguments (line 135)
        kwargs_293288 = {}
        # Getting the type of 'ax' (line 135)
        ax_293278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 9), 'ax', False)
        # Obtaining the member 'set_ylim' of a type (line 135)
        set_ylim_293279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 9), ax_293278, 'set_ylim')
        # Calling set_ylim(args, kwargs) (line 135)
        set_ylim_call_result_293289 = invoke(stypy.reporting.localization.Localization(__file__, 135, 9), set_ylim_293279, *[subscript_call_result_293283, subscript_call_result_293287], **kwargs_293288)
        
        # SSA join for if statement (line 132)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 137):
        
        # Obtaining an instance of the builtin type 'list' (line 137)
        list_293290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 137)
        
        # Assigning a type to the variable 'result' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 6), 'result', list_293290)
        
        # Getting the type of 'value' (line 138)
        value_293291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'value')
        # Testing the type of a for loop iterable (line 138)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 138, 6), value_293291)
        # Getting the type of the for loop variable (line 138)
        for_loop_var_293292 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 138, 6), value_293291)
        # Assigning a type to the variable 'v' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 6), 'v', for_loop_var_293292)
        # SSA begins for a for statement (line 138)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Str to a Name (line 141):
        unicode_293293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 19), 'unicode', u'This is due to a logic error in the StrConverter class.  ')
        # Assigning a type to the variable 'errmsg' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 9), 'errmsg', unicode_293293)
        
        # Getting the type of 'errmsg' (line 142)
        errmsg_293294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 9), 'errmsg')
        unicode_293295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 19), 'unicode', u'Please report this error and its message in bugzilla.')
        # Applying the binary operator '+=' (line 142)
        result_iadd_293296 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 9), '+=', errmsg_293294, unicode_293295)
        # Assigning a type to the variable 'errmsg' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 9), 'errmsg', result_iadd_293296)
        
        # Evaluating assert statement condition
        
        # Getting the type of 'v' (line 143)
        v_293297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 18), 'v')
        # Getting the type of 'labels' (line 143)
        labels_293298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 23), 'labels')
        # Applying the binary operator 'in' (line 143)
        result_contains_293299 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 18), 'in', v_293297, labels_293298)
        
        
        # Call to append(...): (line 144)
        # Processing the call arguments (line 144)
        
        # Obtaining the type of the subscript
        
        # Call to index(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'v' (line 144)
        v_293304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 44), 'v', False)
        # Processing the call keyword arguments (line 144)
        kwargs_293305 = {}
        # Getting the type of 'labels' (line 144)
        labels_293302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 31), 'labels', False)
        # Obtaining the member 'index' of a type (line 144)
        index_293303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 31), labels_293302, 'index')
        # Calling index(args, kwargs) (line 144)
        index_call_result_293306 = invoke(stypy.reporting.localization.Localization(__file__, 144, 31), index_293303, *[v_293304], **kwargs_293305)
        
        # Getting the type of 'ticks' (line 144)
        ticks_293307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 24), 'ticks', False)
        # Obtaining the member '__getitem__' of a type (line 144)
        getitem___293308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 24), ticks_293307, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 144)
        subscript_call_result_293309 = invoke(stypy.reporting.localization.Localization(__file__, 144, 24), getitem___293308, index_call_result_293306)
        
        # Processing the call keyword arguments (line 144)
        kwargs_293310 = {}
        # Getting the type of 'result' (line 144)
        result_293300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 9), 'result', False)
        # Obtaining the member 'append' of a type (line 144)
        append_293301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 9), result_293300, 'append')
        # Calling append(args, kwargs) (line 144)
        append_call_result_293311 = invoke(stypy.reporting.localization.Localization(__file__, 144, 9), append_293301, *[subscript_call_result_293309], **kwargs_293310)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to ignore(...): (line 146)
        # Processing the call arguments (line 146)
        int_293315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 24), 'int')
        # Processing the call keyword arguments (line 146)
        kwargs_293316 = {}
        # Getting the type of 'ax' (line 146)
        ax_293312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 6), 'ax', False)
        # Obtaining the member 'viewLim' of a type (line 146)
        viewLim_293313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 6), ax_293312, 'viewLim')
        # Obtaining the member 'ignore' of a type (line 146)
        ignore_293314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 6), viewLim_293313, 'ignore')
        # Calling ignore(args, kwargs) (line 146)
        ignore_call_result_293317 = invoke(stypy.reporting.localization.Localization(__file__, 146, 6), ignore_293314, *[int_293315], **kwargs_293316)
        
        # Getting the type of 'result' (line 147)
        result_293318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 13), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 6), 'stypy_return_type', result_293318)
        
        # ################# End of 'convert(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'convert' in the type store
        # Getting the type of 'stypy_return_type' (line 57)
        stypy_return_type_293319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293319)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'convert'
        return stypy_return_type_293319


    @staticmethod
    @norecursion
    def default_units(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'default_units'
        module_type_store = module_type_store.open_function_context('default_units', 150, 3, False)
        
        # Passed parameters checking function
        StrConverter.default_units.__dict__.__setitem__('stypy_localization', localization)
        StrConverter.default_units.__dict__.__setitem__('stypy_type_of_self', None)
        StrConverter.default_units.__dict__.__setitem__('stypy_type_store', module_type_store)
        StrConverter.default_units.__dict__.__setitem__('stypy_function_name', 'default_units')
        StrConverter.default_units.__dict__.__setitem__('stypy_param_names_list', ['value', 'axis'])
        StrConverter.default_units.__dict__.__setitem__('stypy_varargs_param_name', None)
        StrConverter.default_units.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StrConverter.default_units.__dict__.__setitem__('stypy_call_defaults', defaults)
        StrConverter.default_units.__dict__.__setitem__('stypy_call_varargs', varargs)
        StrConverter.default_units.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StrConverter.default_units.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, 'default_units', ['value', 'axis'], None, None, defaults, varargs, kwargs)

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

        unicode_293320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, (-1)), 'unicode', u': Return the default unit for value, or None.\n\n      = INPUT VARIABLES\n      - axis    The axis using this converter.\n      - value   The value or list of values that need units.\n\n      = RETURN VALUE\n      - Returns the default units to use for value.\n      Return the default unit for value, or None.\n      ')
        unicode_293321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 13), 'unicode', u'indexed')
        # Assigning a type to the variable 'stypy_return_type' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 6), 'stypy_return_type', unicode_293321)
        
        # ################# End of 'default_units(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'default_units' in the type store
        # Getting the type of 'stypy_return_type' (line 150)
        stypy_return_type_293322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293322)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'default_units'
        return stypy_return_type_293322


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StrConverter.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'StrConverter' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'StrConverter', StrConverter)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
