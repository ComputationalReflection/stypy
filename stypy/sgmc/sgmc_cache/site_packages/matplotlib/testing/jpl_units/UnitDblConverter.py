
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #===========================================================================
2: #
3: # UnitDblConverter
4: #
5: #===========================================================================
6: 
7: 
8: '''UnitDblConverter module containing class UnitDblConverter.'''
9: 
10: #===========================================================================
11: # Place all imports after here.
12: #
13: from __future__ import (absolute_import, division, print_function,
14:                         unicode_literals)
15: 
16: import six
17: 
18: import numpy as np
19: import matplotlib.units as units
20: import matplotlib.ticker as ticker
21: import matplotlib.projections.polar as polar
22: from matplotlib.cbook import iterable
23: #
24: # Place all imports before here.
25: #===========================================================================
26: 
27: __all__ = [ 'UnitDblConverter' ]
28: 
29: #===========================================================================
30: 
31: # A special function for use with the matplotlib FuncFormatter class
32: # for formatting axes with radian units.
33: # This was copied from matplotlib example code.
34: def rad_fn(x, pos = None ):
35:    '''Radian function formatter.'''
36:    n = int((x / np.pi) * 2.0 + 0.25)
37:    if n == 0:
38:       return str(x)
39:    elif n == 1:
40:       return r'$\pi/2$'
41:    elif n == 2:
42:       return r'$\pi$'
43:    elif n % 2 == 0:
44:       return r'$%s\pi$' % (n/2,)
45:    else:
46:       return r'$%s\pi/2$' % (n,)
47: 
48: #===========================================================================
49: class UnitDblConverter( units.ConversionInterface ):
50:    ''': A matplotlib converter class.  Provides matplotlib conversion
51:         functionality for the Monte UnitDbl class.
52:    '''
53: 
54:    # default for plotting
55:    defaults = {
56:                  "distance" : 'km',
57:                  "angle" : 'deg',
58:                  "time" : 'sec',
59:               }
60: 
61:    #------------------------------------------------------------------------
62:    @staticmethod
63:    def axisinfo( unit, axis ):
64:       ''': Returns information on how to handle an axis that has Epoch data.
65: 
66:       = INPUT VARIABLES
67:       - unit    The units to use for a axis with Epoch data.
68: 
69:       = RETURN VALUE
70:       - Returns a matplotlib AxisInfo data structure that contains
71:         minor/major formatters, major/minor locators, and default
72:         label information.
73:       '''
74:       # Delay-load due to circular dependencies.
75:       import matplotlib.testing.jpl_units as U
76: 
77:       # Check to see if the value used for units is a string unit value
78:       # or an actual instance of a UnitDbl so that we can use the unit
79:       # value for the default axis label value.
80:       if ( unit ):
81:          if ( isinstance( unit, six.string_types ) ):
82:             label = unit
83:          else:
84:             label = unit.label()
85:       else:
86:          label = None
87: 
88:       if ( label == "deg" ) and isinstance( axis.axes, polar.PolarAxes ):
89:          # If we want degrees for a polar plot, use the PolarPlotFormatter
90:          majfmt = polar.PolarAxes.ThetaFormatter()
91:       else:
92:          majfmt = U.UnitDblFormatter( useOffset = False )
93: 
94:       return units.AxisInfo( majfmt = majfmt, label = label )
95: 
96:    #------------------------------------------------------------------------
97:    @staticmethod
98:    def convert( value, unit, axis ):
99:       ''': Convert value using unit to a float.  If value is a sequence, return
100:       the converted sequence.
101: 
102:       = INPUT VARIABLES
103:       - value   The value or list of values that need to be converted.
104:       - unit    The units to use for a axis with Epoch data.
105: 
106:       = RETURN VALUE
107:       - Returns the value parameter converted to floats.
108:       '''
109:       # Delay-load due to circular dependencies.
110:       import matplotlib.testing.jpl_units as U
111: 
112:       isNotUnitDbl = True
113: 
114:       if ( iterable(value) and not isinstance(value, six.string_types) ):
115:          if ( len(value) == 0 ):
116:             return []
117:          else:
118:             return [ UnitDblConverter.convert( x, unit, axis ) for x in value ]
119: 
120:       # We need to check to see if the incoming value is actually a UnitDbl and
121:       # set a flag.  If we get an empty list, then just return an empty list.
122:       if ( isinstance(value, U.UnitDbl) ):
123:          isNotUnitDbl = False
124: 
125:       # If the incoming value behaves like a number, but is not a UnitDbl,
126:       # then just return it because we don't know how to convert it
127:       # (or it is already converted)
128:       if ( isNotUnitDbl and units.ConversionInterface.is_numlike( value ) ):
129:          return value
130: 
131:       # If no units were specified, then get the default units to use.
132:       if ( unit == None ):
133:          unit = UnitDblConverter.default_units( value, axis )
134: 
135:       # Convert the incoming UnitDbl value/values to float/floats
136:       if isinstance( axis.axes, polar.PolarAxes ) and (value.type() == "angle"):
137:          # Guarantee that units are radians for polar plots.
138:          return value.convert( "rad" )
139: 
140:       return value.convert( unit )
141: 
142:    #------------------------------------------------------------------------
143:    @staticmethod
144:    def default_units( value, axis ):
145:       ''': Return the default unit for value, or None.
146: 
147:       = INPUT VARIABLES
148:       - value   The value or list of values that need units.
149: 
150:       = RETURN VALUE
151:       - Returns the default units to use for value.
152:       Return the default unit for value, or None.
153:       '''
154: 
155:       # Determine the default units based on the user preferences set for
156:       # default units when printing a UnitDbl.
157:       if ( iterable(value) and not isinstance(value, six.string_types) ):
158:          return UnitDblConverter.default_units( value[0], axis )
159:       else:
160:          return UnitDblConverter.defaults[ value.type() ]
161: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_293680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 0), 'unicode', u'UnitDblConverter module containing class UnitDblConverter.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import six' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_293681 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six')

if (type(import_293681) is not StypyTypeError):

    if (import_293681 != 'pyd_module'):
        __import__(import_293681)
        sys_modules_293682 = sys.modules[import_293681]
        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', sys_modules_293682.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', import_293681)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import numpy' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_293683 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy')

if (type(import_293683) is not StypyTypeError):

    if (import_293683 != 'pyd_module'):
        __import__(import_293683)
        sys_modules_293684 = sys.modules[import_293683]
        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'np', sys_modules_293684.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy', import_293683)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import matplotlib.units' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_293685 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.units')

if (type(import_293685) is not StypyTypeError):

    if (import_293685 != 'pyd_module'):
        __import__(import_293685)
        sys_modules_293686 = sys.modules[import_293685]
        import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'units', sys_modules_293686.module_type_store, module_type_store)
    else:
        import matplotlib.units as units

        import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'units', matplotlib.units, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.units' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.units', import_293685)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import matplotlib.ticker' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_293687 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.ticker')

if (type(import_293687) is not StypyTypeError):

    if (import_293687 != 'pyd_module'):
        __import__(import_293687)
        sys_modules_293688 = sys.modules[import_293687]
        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'ticker', sys_modules_293688.module_type_store, module_type_store)
    else:
        import matplotlib.ticker as ticker

        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'ticker', matplotlib.ticker, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.ticker' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.ticker', import_293687)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'import matplotlib.projections.polar' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_293689 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'matplotlib.projections.polar')

if (type(import_293689) is not StypyTypeError):

    if (import_293689 != 'pyd_module'):
        __import__(import_293689)
        sys_modules_293690 = sys.modules[import_293689]
        import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'polar', sys_modules_293690.module_type_store, module_type_store)
    else:
        import matplotlib.projections.polar as polar

        import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'polar', matplotlib.projections.polar, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.projections.polar' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'matplotlib.projections.polar', import_293689)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from matplotlib.cbook import iterable' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_293691 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'matplotlib.cbook')

if (type(import_293691) is not StypyTypeError):

    if (import_293691 != 'pyd_module'):
        __import__(import_293691)
        sys_modules_293692 = sys.modules[import_293691]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'matplotlib.cbook', sys_modules_293692.module_type_store, module_type_store, ['iterable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_293692, sys_modules_293692.module_type_store, module_type_store)
    else:
        from matplotlib.cbook import iterable

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'matplotlib.cbook', None, module_type_store, ['iterable'], [iterable])

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'matplotlib.cbook', import_293691)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')


# Assigning a List to a Name (line 27):
__all__ = [u'UnitDblConverter']
module_type_store.set_exportable_members([u'UnitDblConverter'])

# Obtaining an instance of the builtin type 'list' (line 27)
list_293693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 27)
# Adding element type (line 27)
unicode_293694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 12), 'unicode', u'UnitDblConverter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_293693, unicode_293694)

# Assigning a type to the variable '__all__' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), '__all__', list_293693)

@norecursion
def rad_fn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 34)
    None_293695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 20), 'None')
    defaults = [None_293695]
    # Create a new context for function 'rad_fn'
    module_type_store = module_type_store.open_function_context('rad_fn', 34, 0, False)
    
    # Passed parameters checking function
    rad_fn.stypy_localization = localization
    rad_fn.stypy_type_of_self = None
    rad_fn.stypy_type_store = module_type_store
    rad_fn.stypy_function_name = 'rad_fn'
    rad_fn.stypy_param_names_list = ['x', 'pos']
    rad_fn.stypy_varargs_param_name = None
    rad_fn.stypy_kwargs_param_name = None
    rad_fn.stypy_call_defaults = defaults
    rad_fn.stypy_call_varargs = varargs
    rad_fn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rad_fn', ['x', 'pos'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rad_fn', localization, ['x', 'pos'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rad_fn(...)' code ##################

    unicode_293696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 3), 'unicode', u'Radian function formatter.')
    
    # Assigning a Call to a Name (line 36):
    
    # Call to int(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'x' (line 36)
    x_293698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'x', False)
    # Getting the type of 'np' (line 36)
    np_293699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'np', False)
    # Obtaining the member 'pi' of a type (line 36)
    pi_293700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 16), np_293699, 'pi')
    # Applying the binary operator 'div' (line 36)
    result_div_293701 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 12), 'div', x_293698, pi_293700)
    
    float_293702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 25), 'float')
    # Applying the binary operator '*' (line 36)
    result_mul_293703 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 11), '*', result_div_293701, float_293702)
    
    float_293704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 31), 'float')
    # Applying the binary operator '+' (line 36)
    result_add_293705 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 11), '+', result_mul_293703, float_293704)
    
    # Processing the call keyword arguments (line 36)
    kwargs_293706 = {}
    # Getting the type of 'int' (line 36)
    int_293697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 7), 'int', False)
    # Calling int(args, kwargs) (line 36)
    int_call_result_293707 = invoke(stypy.reporting.localization.Localization(__file__, 36, 7), int_293697, *[result_add_293705], **kwargs_293706)
    
    # Assigning a type to the variable 'n' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 3), 'n', int_call_result_293707)
    
    
    # Getting the type of 'n' (line 37)
    n_293708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 6), 'n')
    int_293709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 11), 'int')
    # Applying the binary operator '==' (line 37)
    result_eq_293710 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 6), '==', n_293708, int_293709)
    
    # Testing the type of an if condition (line 37)
    if_condition_293711 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 3), result_eq_293710)
    # Assigning a type to the variable 'if_condition_293711' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 3), 'if_condition_293711', if_condition_293711)
    # SSA begins for if statement (line 37)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to str(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'x' (line 38)
    x_293713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 17), 'x', False)
    # Processing the call keyword arguments (line 38)
    kwargs_293714 = {}
    # Getting the type of 'str' (line 38)
    str_293712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 13), 'str', False)
    # Calling str(args, kwargs) (line 38)
    str_call_result_293715 = invoke(stypy.reporting.localization.Localization(__file__, 38, 13), str_293712, *[x_293713], **kwargs_293714)
    
    # Assigning a type to the variable 'stypy_return_type' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 6), 'stypy_return_type', str_call_result_293715)
    # SSA branch for the else part of an if statement (line 37)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'n' (line 39)
    n_293716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'n')
    int_293717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 13), 'int')
    # Applying the binary operator '==' (line 39)
    result_eq_293718 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 8), '==', n_293716, int_293717)
    
    # Testing the type of an if condition (line 39)
    if_condition_293719 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 8), result_eq_293718)
    # Assigning a type to the variable 'if_condition_293719' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'if_condition_293719', if_condition_293719)
    # SSA begins for if statement (line 39)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    unicode_293720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 13), 'unicode', u'$\\pi/2$')
    # Assigning a type to the variable 'stypy_return_type' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 6), 'stypy_return_type', unicode_293720)
    # SSA branch for the else part of an if statement (line 39)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'n' (line 41)
    n_293721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'n')
    int_293722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 13), 'int')
    # Applying the binary operator '==' (line 41)
    result_eq_293723 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 8), '==', n_293721, int_293722)
    
    # Testing the type of an if condition (line 41)
    if_condition_293724 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 8), result_eq_293723)
    # Assigning a type to the variable 'if_condition_293724' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'if_condition_293724', if_condition_293724)
    # SSA begins for if statement (line 41)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    unicode_293725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 13), 'unicode', u'$\\pi$')
    # Assigning a type to the variable 'stypy_return_type' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 6), 'stypy_return_type', unicode_293725)
    # SSA branch for the else part of an if statement (line 41)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'n' (line 43)
    n_293726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'n')
    int_293727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 12), 'int')
    # Applying the binary operator '%' (line 43)
    result_mod_293728 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 8), '%', n_293726, int_293727)
    
    int_293729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 17), 'int')
    # Applying the binary operator '==' (line 43)
    result_eq_293730 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 8), '==', result_mod_293728, int_293729)
    
    # Testing the type of an if condition (line 43)
    if_condition_293731 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 8), result_eq_293730)
    # Assigning a type to the variable 'if_condition_293731' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'if_condition_293731', if_condition_293731)
    # SSA begins for if statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    unicode_293732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 13), 'unicode', u'$%s\\pi$')
    
    # Obtaining an instance of the builtin type 'tuple' (line 44)
    tuple_293733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 44)
    # Adding element type (line 44)
    # Getting the type of 'n' (line 44)
    n_293734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 27), 'n')
    int_293735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 29), 'int')
    # Applying the binary operator 'div' (line 44)
    result_div_293736 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 27), 'div', n_293734, int_293735)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 27), tuple_293733, result_div_293736)
    
    # Applying the binary operator '%' (line 44)
    result_mod_293737 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 13), '%', unicode_293732, tuple_293733)
    
    # Assigning a type to the variable 'stypy_return_type' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 6), 'stypy_return_type', result_mod_293737)
    # SSA branch for the else part of an if statement (line 43)
    module_type_store.open_ssa_branch('else')
    unicode_293738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 13), 'unicode', u'$%s\\pi/2$')
    
    # Obtaining an instance of the builtin type 'tuple' (line 46)
    tuple_293739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 46)
    # Adding element type (line 46)
    # Getting the type of 'n' (line 46)
    n_293740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 29), 'n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 29), tuple_293739, n_293740)
    
    # Applying the binary operator '%' (line 46)
    result_mod_293741 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 13), '%', unicode_293738, tuple_293739)
    
    # Assigning a type to the variable 'stypy_return_type' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 6), 'stypy_return_type', result_mod_293741)
    # SSA join for if statement (line 43)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 41)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 39)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 37)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'rad_fn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rad_fn' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_293742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_293742)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rad_fn'
    return stypy_return_type_293742

# Assigning a type to the variable 'rad_fn' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'rad_fn', rad_fn)
# Declaration of the 'UnitDblConverter' class
# Getting the type of 'units' (line 49)
units_293743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 24), 'units')
# Obtaining the member 'ConversionInterface' of a type (line 49)
ConversionInterface_293744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 24), units_293743, 'ConversionInterface')

class UnitDblConverter(ConversionInterface_293744, ):
    unicode_293745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, (-1)), 'unicode', u': A matplotlib converter class.  Provides matplotlib conversion\n        functionality for the Monte UnitDbl class.\n   ')

    @staticmethod
    @norecursion
    def axisinfo(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'axisinfo'
        module_type_store = module_type_store.open_function_context('axisinfo', 62, 3, False)
        
        # Passed parameters checking function
        UnitDblConverter.axisinfo.__dict__.__setitem__('stypy_localization', localization)
        UnitDblConverter.axisinfo.__dict__.__setitem__('stypy_type_of_self', None)
        UnitDblConverter.axisinfo.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitDblConverter.axisinfo.__dict__.__setitem__('stypy_function_name', 'axisinfo')
        UnitDblConverter.axisinfo.__dict__.__setitem__('stypy_param_names_list', ['unit', 'axis'])
        UnitDblConverter.axisinfo.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitDblConverter.axisinfo.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitDblConverter.axisinfo.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitDblConverter.axisinfo.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitDblConverter.axisinfo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitDblConverter.axisinfo.__dict__.__setitem__('stypy_declared_arg_number', 2)
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

        unicode_293746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, (-1)), 'unicode', u': Returns information on how to handle an axis that has Epoch data.\n\n      = INPUT VARIABLES\n      - unit    The units to use for a axis with Epoch data.\n\n      = RETURN VALUE\n      - Returns a matplotlib AxisInfo data structure that contains\n        minor/major formatters, major/minor locators, and default\n        label information.\n      ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 75, 6))
        
        # 'import matplotlib.testing.jpl_units' statement (line 75)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
        import_293747 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 75, 6), 'matplotlib.testing.jpl_units')

        if (type(import_293747) is not StypyTypeError):

            if (import_293747 != 'pyd_module'):
                __import__(import_293747)
                sys_modules_293748 = sys.modules[import_293747]
                import_module(stypy.reporting.localization.Localization(__file__, 75, 6), 'U', sys_modules_293748.module_type_store, module_type_store)
            else:
                import matplotlib.testing.jpl_units as U

                import_module(stypy.reporting.localization.Localization(__file__, 75, 6), 'U', matplotlib.testing.jpl_units, module_type_store)

        else:
            # Assigning a type to the variable 'matplotlib.testing.jpl_units' (line 75)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 6), 'matplotlib.testing.jpl_units', import_293747)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
        
        
        # Getting the type of 'unit' (line 80)
        unit_293749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'unit')
        # Testing the type of an if condition (line 80)
        if_condition_293750 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 6), unit_293749)
        # Assigning a type to the variable 'if_condition_293750' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 6), 'if_condition_293750', if_condition_293750)
        # SSA begins for if statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to isinstance(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'unit' (line 81)
        unit_293752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 26), 'unit', False)
        # Getting the type of 'six' (line 81)
        six_293753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 32), 'six', False)
        # Obtaining the member 'string_types' of a type (line 81)
        string_types_293754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 32), six_293753, 'string_types')
        # Processing the call keyword arguments (line 81)
        kwargs_293755 = {}
        # Getting the type of 'isinstance' (line 81)
        isinstance_293751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 14), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 81)
        isinstance_call_result_293756 = invoke(stypy.reporting.localization.Localization(__file__, 81, 14), isinstance_293751, *[unit_293752, string_types_293754], **kwargs_293755)
        
        # Testing the type of an if condition (line 81)
        if_condition_293757 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 9), isinstance_call_result_293756)
        # Assigning a type to the variable 'if_condition_293757' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 9), 'if_condition_293757', if_condition_293757)
        # SSA begins for if statement (line 81)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 82):
        # Getting the type of 'unit' (line 82)
        unit_293758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), 'unit')
        # Assigning a type to the variable 'label' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'label', unit_293758)
        # SSA branch for the else part of an if statement (line 81)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 84):
        
        # Call to label(...): (line 84)
        # Processing the call keyword arguments (line 84)
        kwargs_293761 = {}
        # Getting the type of 'unit' (line 84)
        unit_293759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), 'unit', False)
        # Obtaining the member 'label' of a type (line 84)
        label_293760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 20), unit_293759, 'label')
        # Calling label(args, kwargs) (line 84)
        label_call_result_293762 = invoke(stypy.reporting.localization.Localization(__file__, 84, 20), label_293760, *[], **kwargs_293761)
        
        # Assigning a type to the variable 'label' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'label', label_call_result_293762)
        # SSA join for if statement (line 81)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 80)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 86):
        # Getting the type of 'None' (line 86)
        None_293763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 17), 'None')
        # Assigning a type to the variable 'label' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 9), 'label', None_293763)
        # SSA join for if statement (line 80)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'label' (line 88)
        label_293764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'label')
        unicode_293765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 20), 'unicode', u'deg')
        # Applying the binary operator '==' (line 88)
        result_eq_293766 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 11), '==', label_293764, unicode_293765)
        
        
        # Call to isinstance(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'axis' (line 88)
        axis_293768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 44), 'axis', False)
        # Obtaining the member 'axes' of a type (line 88)
        axes_293769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 44), axis_293768, 'axes')
        # Getting the type of 'polar' (line 88)
        polar_293770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 55), 'polar', False)
        # Obtaining the member 'PolarAxes' of a type (line 88)
        PolarAxes_293771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 55), polar_293770, 'PolarAxes')
        # Processing the call keyword arguments (line 88)
        kwargs_293772 = {}
        # Getting the type of 'isinstance' (line 88)
        isinstance_293767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 32), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 88)
        isinstance_call_result_293773 = invoke(stypy.reporting.localization.Localization(__file__, 88, 32), isinstance_293767, *[axes_293769, PolarAxes_293771], **kwargs_293772)
        
        # Applying the binary operator 'and' (line 88)
        result_and_keyword_293774 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 9), 'and', result_eq_293766, isinstance_call_result_293773)
        
        # Testing the type of an if condition (line 88)
        if_condition_293775 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 6), result_and_keyword_293774)
        # Assigning a type to the variable 'if_condition_293775' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 6), 'if_condition_293775', if_condition_293775)
        # SSA begins for if statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 90):
        
        # Call to ThetaFormatter(...): (line 90)
        # Processing the call keyword arguments (line 90)
        kwargs_293779 = {}
        # Getting the type of 'polar' (line 90)
        polar_293776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 18), 'polar', False)
        # Obtaining the member 'PolarAxes' of a type (line 90)
        PolarAxes_293777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 18), polar_293776, 'PolarAxes')
        # Obtaining the member 'ThetaFormatter' of a type (line 90)
        ThetaFormatter_293778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 18), PolarAxes_293777, 'ThetaFormatter')
        # Calling ThetaFormatter(args, kwargs) (line 90)
        ThetaFormatter_call_result_293780 = invoke(stypy.reporting.localization.Localization(__file__, 90, 18), ThetaFormatter_293778, *[], **kwargs_293779)
        
        # Assigning a type to the variable 'majfmt' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 9), 'majfmt', ThetaFormatter_call_result_293780)
        # SSA branch for the else part of an if statement (line 88)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 92):
        
        # Call to UnitDblFormatter(...): (line 92)
        # Processing the call keyword arguments (line 92)
        # Getting the type of 'False' (line 92)
        False_293783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 50), 'False', False)
        keyword_293784 = False_293783
        kwargs_293785 = {'useOffset': keyword_293784}
        # Getting the type of 'U' (line 92)
        U_293781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 18), 'U', False)
        # Obtaining the member 'UnitDblFormatter' of a type (line 92)
        UnitDblFormatter_293782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 18), U_293781, 'UnitDblFormatter')
        # Calling UnitDblFormatter(args, kwargs) (line 92)
        UnitDblFormatter_call_result_293786 = invoke(stypy.reporting.localization.Localization(__file__, 92, 18), UnitDblFormatter_293782, *[], **kwargs_293785)
        
        # Assigning a type to the variable 'majfmt' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 9), 'majfmt', UnitDblFormatter_call_result_293786)
        # SSA join for if statement (line 88)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to AxisInfo(...): (line 94)
        # Processing the call keyword arguments (line 94)
        # Getting the type of 'majfmt' (line 94)
        majfmt_293789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 38), 'majfmt', False)
        keyword_293790 = majfmt_293789
        # Getting the type of 'label' (line 94)
        label_293791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 54), 'label', False)
        keyword_293792 = label_293791
        kwargs_293793 = {'label': keyword_293792, 'majfmt': keyword_293790}
        # Getting the type of 'units' (line 94)
        units_293787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 13), 'units', False)
        # Obtaining the member 'AxisInfo' of a type (line 94)
        AxisInfo_293788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 13), units_293787, 'AxisInfo')
        # Calling AxisInfo(args, kwargs) (line 94)
        AxisInfo_call_result_293794 = invoke(stypy.reporting.localization.Localization(__file__, 94, 13), AxisInfo_293788, *[], **kwargs_293793)
        
        # Assigning a type to the variable 'stypy_return_type' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 6), 'stypy_return_type', AxisInfo_call_result_293794)
        
        # ################# End of 'axisinfo(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'axisinfo' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_293795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293795)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'axisinfo'
        return stypy_return_type_293795


    @staticmethod
    @norecursion
    def convert(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'convert'
        module_type_store = module_type_store.open_function_context('convert', 97, 3, False)
        
        # Passed parameters checking function
        UnitDblConverter.convert.__dict__.__setitem__('stypy_localization', localization)
        UnitDblConverter.convert.__dict__.__setitem__('stypy_type_of_self', None)
        UnitDblConverter.convert.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitDblConverter.convert.__dict__.__setitem__('stypy_function_name', 'convert')
        UnitDblConverter.convert.__dict__.__setitem__('stypy_param_names_list', ['value', 'unit', 'axis'])
        UnitDblConverter.convert.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitDblConverter.convert.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitDblConverter.convert.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitDblConverter.convert.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitDblConverter.convert.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitDblConverter.convert.__dict__.__setitem__('stypy_declared_arg_number', 3)
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

        unicode_293796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, (-1)), 'unicode', u': Convert value using unit to a float.  If value is a sequence, return\n      the converted sequence.\n\n      = INPUT VARIABLES\n      - value   The value or list of values that need to be converted.\n      - unit    The units to use for a axis with Epoch data.\n\n      = RETURN VALUE\n      - Returns the value parameter converted to floats.\n      ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 110, 6))
        
        # 'import matplotlib.testing.jpl_units' statement (line 110)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
        import_293797 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 110, 6), 'matplotlib.testing.jpl_units')

        if (type(import_293797) is not StypyTypeError):

            if (import_293797 != 'pyd_module'):
                __import__(import_293797)
                sys_modules_293798 = sys.modules[import_293797]
                import_module(stypy.reporting.localization.Localization(__file__, 110, 6), 'U', sys_modules_293798.module_type_store, module_type_store)
            else:
                import matplotlib.testing.jpl_units as U

                import_module(stypy.reporting.localization.Localization(__file__, 110, 6), 'U', matplotlib.testing.jpl_units, module_type_store)

        else:
            # Assigning a type to the variable 'matplotlib.testing.jpl_units' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 6), 'matplotlib.testing.jpl_units', import_293797)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
        
        
        # Assigning a Name to a Name (line 112):
        # Getting the type of 'True' (line 112)
        True_293799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 21), 'True')
        # Assigning a type to the variable 'isNotUnitDbl' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 6), 'isNotUnitDbl', True_293799)
        
        
        # Evaluating a boolean operation
        
        # Call to iterable(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'value' (line 114)
        value_293801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 20), 'value', False)
        # Processing the call keyword arguments (line 114)
        kwargs_293802 = {}
        # Getting the type of 'iterable' (line 114)
        iterable_293800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 11), 'iterable', False)
        # Calling iterable(args, kwargs) (line 114)
        iterable_call_result_293803 = invoke(stypy.reporting.localization.Localization(__file__, 114, 11), iterable_293800, *[value_293801], **kwargs_293802)
        
        
        
        # Call to isinstance(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'value' (line 114)
        value_293805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 46), 'value', False)
        # Getting the type of 'six' (line 114)
        six_293806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 53), 'six', False)
        # Obtaining the member 'string_types' of a type (line 114)
        string_types_293807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 53), six_293806, 'string_types')
        # Processing the call keyword arguments (line 114)
        kwargs_293808 = {}
        # Getting the type of 'isinstance' (line 114)
        isinstance_293804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 35), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 114)
        isinstance_call_result_293809 = invoke(stypy.reporting.localization.Localization(__file__, 114, 35), isinstance_293804, *[value_293805, string_types_293807], **kwargs_293808)
        
        # Applying the 'not' unary operator (line 114)
        result_not__293810 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 31), 'not', isinstance_call_result_293809)
        
        # Applying the binary operator 'and' (line 114)
        result_and_keyword_293811 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 11), 'and', iterable_call_result_293803, result_not__293810)
        
        # Testing the type of an if condition (line 114)
        if_condition_293812 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 114, 6), result_and_keyword_293811)
        # Assigning a type to the variable 'if_condition_293812' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 6), 'if_condition_293812', if_condition_293812)
        # SSA begins for if statement (line 114)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to len(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'value' (line 115)
        value_293814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 18), 'value', False)
        # Processing the call keyword arguments (line 115)
        kwargs_293815 = {}
        # Getting the type of 'len' (line 115)
        len_293813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 14), 'len', False)
        # Calling len(args, kwargs) (line 115)
        len_call_result_293816 = invoke(stypy.reporting.localization.Localization(__file__, 115, 14), len_293813, *[value_293814], **kwargs_293815)
        
        int_293817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 28), 'int')
        # Applying the binary operator '==' (line 115)
        result_eq_293818 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 14), '==', len_call_result_293816, int_293817)
        
        # Testing the type of an if condition (line 115)
        if_condition_293819 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 9), result_eq_293818)
        # Assigning a type to the variable 'if_condition_293819' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 9), 'if_condition_293819', if_condition_293819)
        # SSA begins for if statement (line 115)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 116)
        list_293820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 116)
        
        # Assigning a type to the variable 'stypy_return_type' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'stypy_return_type', list_293820)
        # SSA branch for the else part of an if statement (line 115)
        module_type_store.open_ssa_branch('else')
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'value' (line 118)
        value_293828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 72), 'value')
        comprehension_293829 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 21), value_293828)
        # Assigning a type to the variable 'x' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 'x', comprehension_293829)
        
        # Call to convert(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'x' (line 118)
        x_293823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 47), 'x', False)
        # Getting the type of 'unit' (line 118)
        unit_293824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 50), 'unit', False)
        # Getting the type of 'axis' (line 118)
        axis_293825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 56), 'axis', False)
        # Processing the call keyword arguments (line 118)
        kwargs_293826 = {}
        # Getting the type of 'UnitDblConverter' (line 118)
        UnitDblConverter_293821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 'UnitDblConverter', False)
        # Obtaining the member 'convert' of a type (line 118)
        convert_293822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 21), UnitDblConverter_293821, 'convert')
        # Calling convert(args, kwargs) (line 118)
        convert_call_result_293827 = invoke(stypy.reporting.localization.Localization(__file__, 118, 21), convert_293822, *[x_293823, unit_293824, axis_293825], **kwargs_293826)
        
        list_293830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 21), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 21), list_293830, convert_call_result_293827)
        # Assigning a type to the variable 'stypy_return_type' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'stypy_return_type', list_293830)
        # SSA join for if statement (line 115)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 114)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to isinstance(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'value' (line 122)
        value_293832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 22), 'value', False)
        # Getting the type of 'U' (line 122)
        U_293833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 29), 'U', False)
        # Obtaining the member 'UnitDbl' of a type (line 122)
        UnitDbl_293834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 29), U_293833, 'UnitDbl')
        # Processing the call keyword arguments (line 122)
        kwargs_293835 = {}
        # Getting the type of 'isinstance' (line 122)
        isinstance_293831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 122)
        isinstance_call_result_293836 = invoke(stypy.reporting.localization.Localization(__file__, 122, 11), isinstance_293831, *[value_293832, UnitDbl_293834], **kwargs_293835)
        
        # Testing the type of an if condition (line 122)
        if_condition_293837 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 6), isinstance_call_result_293836)
        # Assigning a type to the variable 'if_condition_293837' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 6), 'if_condition_293837', if_condition_293837)
        # SSA begins for if statement (line 122)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 123):
        # Getting the type of 'False' (line 123)
        False_293838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 24), 'False')
        # Assigning a type to the variable 'isNotUnitDbl' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 9), 'isNotUnitDbl', False_293838)
        # SSA join for if statement (line 122)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'isNotUnitDbl' (line 128)
        isNotUnitDbl_293839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), 'isNotUnitDbl')
        
        # Call to is_numlike(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'value' (line 128)
        value_293843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 66), 'value', False)
        # Processing the call keyword arguments (line 128)
        kwargs_293844 = {}
        # Getting the type of 'units' (line 128)
        units_293840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 28), 'units', False)
        # Obtaining the member 'ConversionInterface' of a type (line 128)
        ConversionInterface_293841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 28), units_293840, 'ConversionInterface')
        # Obtaining the member 'is_numlike' of a type (line 128)
        is_numlike_293842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 28), ConversionInterface_293841, 'is_numlike')
        # Calling is_numlike(args, kwargs) (line 128)
        is_numlike_call_result_293845 = invoke(stypy.reporting.localization.Localization(__file__, 128, 28), is_numlike_293842, *[value_293843], **kwargs_293844)
        
        # Applying the binary operator 'and' (line 128)
        result_and_keyword_293846 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 11), 'and', isNotUnitDbl_293839, is_numlike_call_result_293845)
        
        # Testing the type of an if condition (line 128)
        if_condition_293847 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 128, 6), result_and_keyword_293846)
        # Assigning a type to the variable 'if_condition_293847' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 6), 'if_condition_293847', if_condition_293847)
        # SSA begins for if statement (line 128)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'value' (line 129)
        value_293848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'value')
        # Assigning a type to the variable 'stypy_return_type' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 9), 'stypy_return_type', value_293848)
        # SSA join for if statement (line 128)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 132)
        # Getting the type of 'unit' (line 132)
        unit_293849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 6), 'unit')
        # Getting the type of 'None' (line 132)
        None_293850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 19), 'None')
        
        (may_be_293851, more_types_in_union_293852) = may_be_none(unit_293849, None_293850)

        if may_be_293851:

            if more_types_in_union_293852:
                # Runtime conditional SSA (line 132)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 133):
            
            # Call to default_units(...): (line 133)
            # Processing the call arguments (line 133)
            # Getting the type of 'value' (line 133)
            value_293855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 48), 'value', False)
            # Getting the type of 'axis' (line 133)
            axis_293856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 55), 'axis', False)
            # Processing the call keyword arguments (line 133)
            kwargs_293857 = {}
            # Getting the type of 'UnitDblConverter' (line 133)
            UnitDblConverter_293853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'UnitDblConverter', False)
            # Obtaining the member 'default_units' of a type (line 133)
            default_units_293854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 16), UnitDblConverter_293853, 'default_units')
            # Calling default_units(args, kwargs) (line 133)
            default_units_call_result_293858 = invoke(stypy.reporting.localization.Localization(__file__, 133, 16), default_units_293854, *[value_293855, axis_293856], **kwargs_293857)
            
            # Assigning a type to the variable 'unit' (line 133)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 9), 'unit', default_units_call_result_293858)

            if more_types_in_union_293852:
                # SSA join for if statement (line 132)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'axis' (line 136)
        axis_293860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 21), 'axis', False)
        # Obtaining the member 'axes' of a type (line 136)
        axes_293861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 21), axis_293860, 'axes')
        # Getting the type of 'polar' (line 136)
        polar_293862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 32), 'polar', False)
        # Obtaining the member 'PolarAxes' of a type (line 136)
        PolarAxes_293863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 32), polar_293862, 'PolarAxes')
        # Processing the call keyword arguments (line 136)
        kwargs_293864 = {}
        # Getting the type of 'isinstance' (line 136)
        isinstance_293859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 9), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 136)
        isinstance_call_result_293865 = invoke(stypy.reporting.localization.Localization(__file__, 136, 9), isinstance_293859, *[axes_293861, PolarAxes_293863], **kwargs_293864)
        
        
        
        # Call to type(...): (line 136)
        # Processing the call keyword arguments (line 136)
        kwargs_293868 = {}
        # Getting the type of 'value' (line 136)
        value_293866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 55), 'value', False)
        # Obtaining the member 'type' of a type (line 136)
        type_293867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 55), value_293866, 'type')
        # Calling type(args, kwargs) (line 136)
        type_call_result_293869 = invoke(stypy.reporting.localization.Localization(__file__, 136, 55), type_293867, *[], **kwargs_293868)
        
        unicode_293870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 71), 'unicode', u'angle')
        # Applying the binary operator '==' (line 136)
        result_eq_293871 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 55), '==', type_call_result_293869, unicode_293870)
        
        # Applying the binary operator 'and' (line 136)
        result_and_keyword_293872 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 9), 'and', isinstance_call_result_293865, result_eq_293871)
        
        # Testing the type of an if condition (line 136)
        if_condition_293873 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 136, 6), result_and_keyword_293872)
        # Assigning a type to the variable 'if_condition_293873' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 6), 'if_condition_293873', if_condition_293873)
        # SSA begins for if statement (line 136)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to convert(...): (line 138)
        # Processing the call arguments (line 138)
        unicode_293876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 31), 'unicode', u'rad')
        # Processing the call keyword arguments (line 138)
        kwargs_293877 = {}
        # Getting the type of 'value' (line 138)
        value_293874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'value', False)
        # Obtaining the member 'convert' of a type (line 138)
        convert_293875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 16), value_293874, 'convert')
        # Calling convert(args, kwargs) (line 138)
        convert_call_result_293878 = invoke(stypy.reporting.localization.Localization(__file__, 138, 16), convert_293875, *[unicode_293876], **kwargs_293877)
        
        # Assigning a type to the variable 'stypy_return_type' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 9), 'stypy_return_type', convert_call_result_293878)
        # SSA join for if statement (line 136)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to convert(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'unit' (line 140)
        unit_293881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 28), 'unit', False)
        # Processing the call keyword arguments (line 140)
        kwargs_293882 = {}
        # Getting the type of 'value' (line 140)
        value_293879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 13), 'value', False)
        # Obtaining the member 'convert' of a type (line 140)
        convert_293880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 13), value_293879, 'convert')
        # Calling convert(args, kwargs) (line 140)
        convert_call_result_293883 = invoke(stypy.reporting.localization.Localization(__file__, 140, 13), convert_293880, *[unit_293881], **kwargs_293882)
        
        # Assigning a type to the variable 'stypy_return_type' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 6), 'stypy_return_type', convert_call_result_293883)
        
        # ################# End of 'convert(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'convert' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_293884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293884)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'convert'
        return stypy_return_type_293884


    @staticmethod
    @norecursion
    def default_units(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'default_units'
        module_type_store = module_type_store.open_function_context('default_units', 143, 3, False)
        
        # Passed parameters checking function
        UnitDblConverter.default_units.__dict__.__setitem__('stypy_localization', localization)
        UnitDblConverter.default_units.__dict__.__setitem__('stypy_type_of_self', None)
        UnitDblConverter.default_units.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitDblConverter.default_units.__dict__.__setitem__('stypy_function_name', 'default_units')
        UnitDblConverter.default_units.__dict__.__setitem__('stypy_param_names_list', ['value', 'axis'])
        UnitDblConverter.default_units.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitDblConverter.default_units.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitDblConverter.default_units.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitDblConverter.default_units.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitDblConverter.default_units.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitDblConverter.default_units.__dict__.__setitem__('stypy_declared_arg_number', 2)
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

        unicode_293885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, (-1)), 'unicode', u': Return the default unit for value, or None.\n\n      = INPUT VARIABLES\n      - value   The value or list of values that need units.\n\n      = RETURN VALUE\n      - Returns the default units to use for value.\n      Return the default unit for value, or None.\n      ')
        
        
        # Evaluating a boolean operation
        
        # Call to iterable(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'value' (line 157)
        value_293887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'value', False)
        # Processing the call keyword arguments (line 157)
        kwargs_293888 = {}
        # Getting the type of 'iterable' (line 157)
        iterable_293886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 11), 'iterable', False)
        # Calling iterable(args, kwargs) (line 157)
        iterable_call_result_293889 = invoke(stypy.reporting.localization.Localization(__file__, 157, 11), iterable_293886, *[value_293887], **kwargs_293888)
        
        
        
        # Call to isinstance(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'value' (line 157)
        value_293891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 46), 'value', False)
        # Getting the type of 'six' (line 157)
        six_293892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 53), 'six', False)
        # Obtaining the member 'string_types' of a type (line 157)
        string_types_293893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 53), six_293892, 'string_types')
        # Processing the call keyword arguments (line 157)
        kwargs_293894 = {}
        # Getting the type of 'isinstance' (line 157)
        isinstance_293890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 35), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 157)
        isinstance_call_result_293895 = invoke(stypy.reporting.localization.Localization(__file__, 157, 35), isinstance_293890, *[value_293891, string_types_293893], **kwargs_293894)
        
        # Applying the 'not' unary operator (line 157)
        result_not__293896 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 31), 'not', isinstance_call_result_293895)
        
        # Applying the binary operator 'and' (line 157)
        result_and_keyword_293897 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 11), 'and', iterable_call_result_293889, result_not__293896)
        
        # Testing the type of an if condition (line 157)
        if_condition_293898 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 6), result_and_keyword_293897)
        # Assigning a type to the variable 'if_condition_293898' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 6), 'if_condition_293898', if_condition_293898)
        # SSA begins for if statement (line 157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to default_units(...): (line 158)
        # Processing the call arguments (line 158)
        
        # Obtaining the type of the subscript
        int_293901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 54), 'int')
        # Getting the type of 'value' (line 158)
        value_293902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 48), 'value', False)
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___293903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 48), value_293902, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_293904 = invoke(stypy.reporting.localization.Localization(__file__, 158, 48), getitem___293903, int_293901)
        
        # Getting the type of 'axis' (line 158)
        axis_293905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 58), 'axis', False)
        # Processing the call keyword arguments (line 158)
        kwargs_293906 = {}
        # Getting the type of 'UnitDblConverter' (line 158)
        UnitDblConverter_293899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'UnitDblConverter', False)
        # Obtaining the member 'default_units' of a type (line 158)
        default_units_293900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 16), UnitDblConverter_293899, 'default_units')
        # Calling default_units(args, kwargs) (line 158)
        default_units_call_result_293907 = invoke(stypy.reporting.localization.Localization(__file__, 158, 16), default_units_293900, *[subscript_call_result_293904, axis_293905], **kwargs_293906)
        
        # Assigning a type to the variable 'stypy_return_type' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 9), 'stypy_return_type', default_units_call_result_293907)
        # SSA branch for the else part of an if statement (line 157)
        module_type_store.open_ssa_branch('else')
        
        # Obtaining the type of the subscript
        
        # Call to type(...): (line 160)
        # Processing the call keyword arguments (line 160)
        kwargs_293910 = {}
        # Getting the type of 'value' (line 160)
        value_293908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 43), 'value', False)
        # Obtaining the member 'type' of a type (line 160)
        type_293909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 43), value_293908, 'type')
        # Calling type(args, kwargs) (line 160)
        type_call_result_293911 = invoke(stypy.reporting.localization.Localization(__file__, 160, 43), type_293909, *[], **kwargs_293910)
        
        # Getting the type of 'UnitDblConverter' (line 160)
        UnitDblConverter_293912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 16), 'UnitDblConverter')
        # Obtaining the member 'defaults' of a type (line 160)
        defaults_293913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 16), UnitDblConverter_293912, 'defaults')
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___293914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 16), defaults_293913, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_293915 = invoke(stypy.reporting.localization.Localization(__file__, 160, 16), getitem___293914, type_call_result_293911)
        
        # Assigning a type to the variable 'stypy_return_type' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 9), 'stypy_return_type', subscript_call_result_293915)
        # SSA join for if statement (line 157)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'default_units(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'default_units' in the type store
        # Getting the type of 'stypy_return_type' (line 143)
        stypy_return_type_293916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293916)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'default_units'
        return stypy_return_type_293916


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 49, 0, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitDblConverter.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'UnitDblConverter' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'UnitDblConverter', UnitDblConverter)

# Assigning a Dict to a Name (line 55):

# Obtaining an instance of the builtin type 'dict' (line 55)
dict_293917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 55)
# Adding element type (key, value) (line 55)
unicode_293918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 17), 'unicode', u'distance')
unicode_293919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 30), 'unicode', u'km')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 14), dict_293917, (unicode_293918, unicode_293919))
# Adding element type (key, value) (line 55)
unicode_293920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 17), 'unicode', u'angle')
unicode_293921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 27), 'unicode', u'deg')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 14), dict_293917, (unicode_293920, unicode_293921))
# Adding element type (key, value) (line 55)
unicode_293922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 17), 'unicode', u'time')
unicode_293923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 26), 'unicode', u'sec')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 14), dict_293917, (unicode_293922, unicode_293923))

# Getting the type of 'UnitDblConverter'
UnitDblConverter_293924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnitDblConverter')
# Setting the type of the member 'defaults' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnitDblConverter_293924, 'defaults', dict_293917)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
