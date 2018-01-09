
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #===========================================================================
2: #
3: # EpochConverter
4: #
5: #===========================================================================
6: 
7: 
8: '''EpochConverter module containing class EpochConverter.'''
9: 
10: #===========================================================================
11: # Place all imports after here.
12: #
13: from __future__ import (absolute_import, division, print_function,
14:                         unicode_literals)
15: 
16: import six
17: 
18: import matplotlib.units as units
19: import matplotlib.dates as date_ticker
20: from matplotlib.cbook import iterable
21: #
22: # Place all imports before here.
23: #===========================================================================
24: 
25: __all__ = [ 'EpochConverter' ]
26: 
27: #===========================================================================
28: class EpochConverter( units.ConversionInterface ):
29:    ''': A matplotlib converter class.  Provides matplotlib conversion
30:         functionality for Monte Epoch and Duration classes.
31:    '''
32: 
33:    # julian date reference for "Jan 1, 0001" minus 1 day because
34:    # matplotlib really wants "Jan 0, 0001"
35:    jdRef = 1721425.5 - 1
36: 
37:    #------------------------------------------------------------------------
38:    @staticmethod
39:    def axisinfo( unit, axis ):
40:       ''': Returns information on how to handle an axis that has Epoch data.
41: 
42:       = INPUT VARIABLES
43:       - unit    The units to use for a axis with Epoch data.
44: 
45:       = RETURN VALUE
46:       - Returns a matplotlib AxisInfo data structure that contains
47:         minor/major formatters, major/minor locators, and default
48:         label information.
49:       '''
50: 
51:       majloc = date_ticker.AutoDateLocator()
52:       majfmt = date_ticker.AutoDateFormatter( majloc )
53: 
54:       return units.AxisInfo( majloc = majloc,
55:                              majfmt = majfmt,
56:                              label = unit )
57: 
58:    #------------------------------------------------------------------------
59:    @staticmethod
60:    def float2epoch( value, unit ):
61:       ''': Convert a matplotlib floating-point date into an Epoch of the
62:            specified units.
63: 
64:       = INPUT VARIABLES
65:       - value    The matplotlib floating-point date.
66:       - unit     The unit system to use for the Epoch.
67: 
68:       = RETURN VALUE
69:       - Returns the value converted to an Epoch in the sepcified time system.
70:       '''
71:       # Delay-load due to circular dependencies.
72:       import matplotlib.testing.jpl_units as U
73: 
74:       secPastRef = value * 86400.0 * U.UnitDbl( 1.0, 'sec' )
75:       return U.Epoch( unit, secPastRef, EpochConverter.jdRef )
76: 
77:    #------------------------------------------------------------------------
78:    @staticmethod
79:    def epoch2float( value, unit ):
80:       ''': Convert an Epoch value to a float suitible for plotting as a
81:            python datetime object.
82: 
83:       = INPUT VARIABLES
84:       - value   An Epoch or list of Epochs that need to be converted.
85:       - unit    The units to use for an axis with Epoch data.
86: 
87:       = RETURN VALUE
88:       - Returns the value parameter converted to floats.
89:       '''
90:       return value.julianDate( unit ) - EpochConverter.jdRef
91: 
92:    #------------------------------------------------------------------------
93:    @staticmethod
94:    def duration2float( value ):
95:       ''': Convert a Duration value to a float suitible for plotting as a
96:            python datetime object.
97: 
98:       = INPUT VARIABLES
99:       - value   A Duration or list of Durations that need to be converted.
100: 
101:       = RETURN VALUE
102:       - Returns the value parameter converted to floats.
103:       '''
104:       return value.days()
105: 
106:    #------------------------------------------------------------------------
107:    @staticmethod
108:    def convert( value, unit, axis ):
109:       ''': Convert value using unit to a float.  If value is a sequence, return
110:       the converted sequence.
111: 
112:       = INPUT VARIABLES
113:       - value   The value or list of values that need to be converted.
114:       - unit    The units to use for an axis with Epoch data.
115: 
116:       = RETURN VALUE
117:       - Returns the value parameter converted to floats.
118:       '''
119:       # Delay-load due to circular dependencies.
120:       import matplotlib.testing.jpl_units as U
121: 
122:       isNotEpoch = True
123:       isDuration = False
124: 
125:       if ( iterable(value) and not isinstance(value, six.string_types) ):
126:          if ( len(value) == 0 ):
127:             return []
128:          else:
129:             return [ EpochConverter.convert( x, unit, axis ) for x in value ]
130: 
131:       if ( isinstance(value, U.Epoch) ):
132:          isNotEpoch = False
133:       elif ( isinstance(value, U.Duration) ):
134:          isDuration = True
135: 
136:       if ( isNotEpoch and not isDuration and
137:            units.ConversionInterface.is_numlike( value ) ):
138:          return value
139: 
140:       if ( unit == None ):
141:          unit = EpochConverter.default_units( value, axis )
142: 
143:       if ( isDuration ):
144:          return EpochConverter.duration2float( value )
145:       else:
146:          return EpochConverter.epoch2float( value, unit )
147: 
148:    #------------------------------------------------------------------------
149:    @staticmethod
150:    def default_units( value, axis ):
151:       ''': Return the default unit for value, or None.
152: 
153:       = INPUT VARIABLES
154:       - value   The value or list of values that need units.
155: 
156:       = RETURN VALUE
157:       - Returns the default units to use for value.
158:       '''
159:       frame = None
160:       if ( iterable(value) and not isinstance(value, six.string_types) ):
161:          return EpochConverter.default_units( value[0], axis )
162:       else:
163:          frame = value.frame()
164: 
165:       return frame
166: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_292909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 0), 'unicode', u'EpochConverter module containing class EpochConverter.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import six' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_292910 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six')

if (type(import_292910) is not StypyTypeError):

    if (import_292910 != 'pyd_module'):
        __import__(import_292910)
        sys_modules_292911 = sys.modules[import_292910]
        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', sys_modules_292911.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', import_292910)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import matplotlib.units' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_292912 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib.units')

if (type(import_292912) is not StypyTypeError):

    if (import_292912 != 'pyd_module'):
        __import__(import_292912)
        sys_modules_292913 = sys.modules[import_292912]
        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'units', sys_modules_292913.module_type_store, module_type_store)
    else:
        import matplotlib.units as units

        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'units', matplotlib.units, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.units' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib.units', import_292912)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import matplotlib.dates' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_292914 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.dates')

if (type(import_292914) is not StypyTypeError):

    if (import_292914 != 'pyd_module'):
        __import__(import_292914)
        sys_modules_292915 = sys.modules[import_292914]
        import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'date_ticker', sys_modules_292915.module_type_store, module_type_store)
    else:
        import matplotlib.dates as date_ticker

        import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'date_ticker', matplotlib.dates, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.dates' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.dates', import_292914)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from matplotlib.cbook import iterable' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_292916 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.cbook')

if (type(import_292916) is not StypyTypeError):

    if (import_292916 != 'pyd_module'):
        __import__(import_292916)
        sys_modules_292917 = sys.modules[import_292916]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.cbook', sys_modules_292917.module_type_store, module_type_store, ['iterable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_292917, sys_modules_292917.module_type_store, module_type_store)
    else:
        from matplotlib.cbook import iterable

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.cbook', None, module_type_store, ['iterable'], [iterable])

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.cbook', import_292916)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')


# Assigning a List to a Name (line 25):
__all__ = [u'EpochConverter']
module_type_store.set_exportable_members([u'EpochConverter'])

# Obtaining an instance of the builtin type 'list' (line 25)
list_292918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)
unicode_292919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 12), 'unicode', u'EpochConverter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 10), list_292918, unicode_292919)

# Assigning a type to the variable '__all__' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), '__all__', list_292918)
# Declaration of the 'EpochConverter' class
# Getting the type of 'units' (line 28)
units_292920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 22), 'units')
# Obtaining the member 'ConversionInterface' of a type (line 28)
ConversionInterface_292921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 22), units_292920, 'ConversionInterface')

class EpochConverter(ConversionInterface_292921, ):
    unicode_292922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, (-1)), 'unicode', u': A matplotlib converter class.  Provides matplotlib conversion\n        functionality for Monte Epoch and Duration classes.\n   ')

    @staticmethod
    @norecursion
    def axisinfo(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'axisinfo'
        module_type_store = module_type_store.open_function_context('axisinfo', 38, 3, False)
        
        # Passed parameters checking function
        EpochConverter.axisinfo.__dict__.__setitem__('stypy_localization', localization)
        EpochConverter.axisinfo.__dict__.__setitem__('stypy_type_of_self', None)
        EpochConverter.axisinfo.__dict__.__setitem__('stypy_type_store', module_type_store)
        EpochConverter.axisinfo.__dict__.__setitem__('stypy_function_name', 'axisinfo')
        EpochConverter.axisinfo.__dict__.__setitem__('stypy_param_names_list', ['unit', 'axis'])
        EpochConverter.axisinfo.__dict__.__setitem__('stypy_varargs_param_name', None)
        EpochConverter.axisinfo.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EpochConverter.axisinfo.__dict__.__setitem__('stypy_call_defaults', defaults)
        EpochConverter.axisinfo.__dict__.__setitem__('stypy_call_varargs', varargs)
        EpochConverter.axisinfo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EpochConverter.axisinfo.__dict__.__setitem__('stypy_declared_arg_number', 2)
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

        unicode_292923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, (-1)), 'unicode', u': Returns information on how to handle an axis that has Epoch data.\n\n      = INPUT VARIABLES\n      - unit    The units to use for a axis with Epoch data.\n\n      = RETURN VALUE\n      - Returns a matplotlib AxisInfo data structure that contains\n        minor/major formatters, major/minor locators, and default\n        label information.\n      ')
        
        # Assigning a Call to a Name (line 51):
        
        # Call to AutoDateLocator(...): (line 51)
        # Processing the call keyword arguments (line 51)
        kwargs_292926 = {}
        # Getting the type of 'date_ticker' (line 51)
        date_ticker_292924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'date_ticker', False)
        # Obtaining the member 'AutoDateLocator' of a type (line 51)
        AutoDateLocator_292925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 15), date_ticker_292924, 'AutoDateLocator')
        # Calling AutoDateLocator(args, kwargs) (line 51)
        AutoDateLocator_call_result_292927 = invoke(stypy.reporting.localization.Localization(__file__, 51, 15), AutoDateLocator_292925, *[], **kwargs_292926)
        
        # Assigning a type to the variable 'majloc' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 6), 'majloc', AutoDateLocator_call_result_292927)
        
        # Assigning a Call to a Name (line 52):
        
        # Call to AutoDateFormatter(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'majloc' (line 52)
        majloc_292930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 46), 'majloc', False)
        # Processing the call keyword arguments (line 52)
        kwargs_292931 = {}
        # Getting the type of 'date_ticker' (line 52)
        date_ticker_292928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'date_ticker', False)
        # Obtaining the member 'AutoDateFormatter' of a type (line 52)
        AutoDateFormatter_292929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 15), date_ticker_292928, 'AutoDateFormatter')
        # Calling AutoDateFormatter(args, kwargs) (line 52)
        AutoDateFormatter_call_result_292932 = invoke(stypy.reporting.localization.Localization(__file__, 52, 15), AutoDateFormatter_292929, *[majloc_292930], **kwargs_292931)
        
        # Assigning a type to the variable 'majfmt' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 6), 'majfmt', AutoDateFormatter_call_result_292932)
        
        # Call to AxisInfo(...): (line 54)
        # Processing the call keyword arguments (line 54)
        # Getting the type of 'majloc' (line 54)
        majloc_292935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 38), 'majloc', False)
        keyword_292936 = majloc_292935
        # Getting the type of 'majfmt' (line 55)
        majfmt_292937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 38), 'majfmt', False)
        keyword_292938 = majfmt_292937
        # Getting the type of 'unit' (line 56)
        unit_292939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 37), 'unit', False)
        keyword_292940 = unit_292939
        kwargs_292941 = {'label': keyword_292940, 'majloc': keyword_292936, 'majfmt': keyword_292938}
        # Getting the type of 'units' (line 54)
        units_292933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 13), 'units', False)
        # Obtaining the member 'AxisInfo' of a type (line 54)
        AxisInfo_292934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 13), units_292933, 'AxisInfo')
        # Calling AxisInfo(args, kwargs) (line 54)
        AxisInfo_call_result_292942 = invoke(stypy.reporting.localization.Localization(__file__, 54, 13), AxisInfo_292934, *[], **kwargs_292941)
        
        # Assigning a type to the variable 'stypy_return_type' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 6), 'stypy_return_type', AxisInfo_call_result_292942)
        
        # ################# End of 'axisinfo(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'axisinfo' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_292943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292943)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'axisinfo'
        return stypy_return_type_292943


    @staticmethod
    @norecursion
    def float2epoch(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'float2epoch'
        module_type_store = module_type_store.open_function_context('float2epoch', 59, 3, False)
        
        # Passed parameters checking function
        EpochConverter.float2epoch.__dict__.__setitem__('stypy_localization', localization)
        EpochConverter.float2epoch.__dict__.__setitem__('stypy_type_of_self', None)
        EpochConverter.float2epoch.__dict__.__setitem__('stypy_type_store', module_type_store)
        EpochConverter.float2epoch.__dict__.__setitem__('stypy_function_name', 'float2epoch')
        EpochConverter.float2epoch.__dict__.__setitem__('stypy_param_names_list', ['value', 'unit'])
        EpochConverter.float2epoch.__dict__.__setitem__('stypy_varargs_param_name', None)
        EpochConverter.float2epoch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EpochConverter.float2epoch.__dict__.__setitem__('stypy_call_defaults', defaults)
        EpochConverter.float2epoch.__dict__.__setitem__('stypy_call_varargs', varargs)
        EpochConverter.float2epoch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EpochConverter.float2epoch.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, 'float2epoch', ['value', 'unit'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'float2epoch', localization, ['unit'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'float2epoch(...)' code ##################

        unicode_292944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, (-1)), 'unicode', u': Convert a matplotlib floating-point date into an Epoch of the\n           specified units.\n\n      = INPUT VARIABLES\n      - value    The matplotlib floating-point date.\n      - unit     The unit system to use for the Epoch.\n\n      = RETURN VALUE\n      - Returns the value converted to an Epoch in the sepcified time system.\n      ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 72, 6))
        
        # 'import matplotlib.testing.jpl_units' statement (line 72)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
        import_292945 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 72, 6), 'matplotlib.testing.jpl_units')

        if (type(import_292945) is not StypyTypeError):

            if (import_292945 != 'pyd_module'):
                __import__(import_292945)
                sys_modules_292946 = sys.modules[import_292945]
                import_module(stypy.reporting.localization.Localization(__file__, 72, 6), 'U', sys_modules_292946.module_type_store, module_type_store)
            else:
                import matplotlib.testing.jpl_units as U

                import_module(stypy.reporting.localization.Localization(__file__, 72, 6), 'U', matplotlib.testing.jpl_units, module_type_store)

        else:
            # Assigning a type to the variable 'matplotlib.testing.jpl_units' (line 72)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 6), 'matplotlib.testing.jpl_units', import_292945)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
        
        
        # Assigning a BinOp to a Name (line 74):
        # Getting the type of 'value' (line 74)
        value_292947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 19), 'value')
        float_292948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 27), 'float')
        # Applying the binary operator '*' (line 74)
        result_mul_292949 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 19), '*', value_292947, float_292948)
        
        
        # Call to UnitDbl(...): (line 74)
        # Processing the call arguments (line 74)
        float_292952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 48), 'float')
        unicode_292953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 53), 'unicode', u'sec')
        # Processing the call keyword arguments (line 74)
        kwargs_292954 = {}
        # Getting the type of 'U' (line 74)
        U_292950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 37), 'U', False)
        # Obtaining the member 'UnitDbl' of a type (line 74)
        UnitDbl_292951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 37), U_292950, 'UnitDbl')
        # Calling UnitDbl(args, kwargs) (line 74)
        UnitDbl_call_result_292955 = invoke(stypy.reporting.localization.Localization(__file__, 74, 37), UnitDbl_292951, *[float_292952, unicode_292953], **kwargs_292954)
        
        # Applying the binary operator '*' (line 74)
        result_mul_292956 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 35), '*', result_mul_292949, UnitDbl_call_result_292955)
        
        # Assigning a type to the variable 'secPastRef' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 6), 'secPastRef', result_mul_292956)
        
        # Call to Epoch(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'unit' (line 75)
        unit_292959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'unit', False)
        # Getting the type of 'secPastRef' (line 75)
        secPastRef_292960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 28), 'secPastRef', False)
        # Getting the type of 'EpochConverter' (line 75)
        EpochConverter_292961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 40), 'EpochConverter', False)
        # Obtaining the member 'jdRef' of a type (line 75)
        jdRef_292962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 40), EpochConverter_292961, 'jdRef')
        # Processing the call keyword arguments (line 75)
        kwargs_292963 = {}
        # Getting the type of 'U' (line 75)
        U_292957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 13), 'U', False)
        # Obtaining the member 'Epoch' of a type (line 75)
        Epoch_292958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 13), U_292957, 'Epoch')
        # Calling Epoch(args, kwargs) (line 75)
        Epoch_call_result_292964 = invoke(stypy.reporting.localization.Localization(__file__, 75, 13), Epoch_292958, *[unit_292959, secPastRef_292960, jdRef_292962], **kwargs_292963)
        
        # Assigning a type to the variable 'stypy_return_type' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 6), 'stypy_return_type', Epoch_call_result_292964)
        
        # ################# End of 'float2epoch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'float2epoch' in the type store
        # Getting the type of 'stypy_return_type' (line 59)
        stypy_return_type_292965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292965)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'float2epoch'
        return stypy_return_type_292965


    @staticmethod
    @norecursion
    def epoch2float(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'epoch2float'
        module_type_store = module_type_store.open_function_context('epoch2float', 78, 3, False)
        
        # Passed parameters checking function
        EpochConverter.epoch2float.__dict__.__setitem__('stypy_localization', localization)
        EpochConverter.epoch2float.__dict__.__setitem__('stypy_type_of_self', None)
        EpochConverter.epoch2float.__dict__.__setitem__('stypy_type_store', module_type_store)
        EpochConverter.epoch2float.__dict__.__setitem__('stypy_function_name', 'epoch2float')
        EpochConverter.epoch2float.__dict__.__setitem__('stypy_param_names_list', ['value', 'unit'])
        EpochConverter.epoch2float.__dict__.__setitem__('stypy_varargs_param_name', None)
        EpochConverter.epoch2float.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EpochConverter.epoch2float.__dict__.__setitem__('stypy_call_defaults', defaults)
        EpochConverter.epoch2float.__dict__.__setitem__('stypy_call_varargs', varargs)
        EpochConverter.epoch2float.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EpochConverter.epoch2float.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, 'epoch2float', ['value', 'unit'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'epoch2float', localization, ['unit'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'epoch2float(...)' code ##################

        unicode_292966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, (-1)), 'unicode', u': Convert an Epoch value to a float suitible for plotting as a\n           python datetime object.\n\n      = INPUT VARIABLES\n      - value   An Epoch or list of Epochs that need to be converted.\n      - unit    The units to use for an axis with Epoch data.\n\n      = RETURN VALUE\n      - Returns the value parameter converted to floats.\n      ')
        
        # Call to julianDate(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'unit' (line 90)
        unit_292969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 31), 'unit', False)
        # Processing the call keyword arguments (line 90)
        kwargs_292970 = {}
        # Getting the type of 'value' (line 90)
        value_292967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 13), 'value', False)
        # Obtaining the member 'julianDate' of a type (line 90)
        julianDate_292968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 13), value_292967, 'julianDate')
        # Calling julianDate(args, kwargs) (line 90)
        julianDate_call_result_292971 = invoke(stypy.reporting.localization.Localization(__file__, 90, 13), julianDate_292968, *[unit_292969], **kwargs_292970)
        
        # Getting the type of 'EpochConverter' (line 90)
        EpochConverter_292972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 40), 'EpochConverter')
        # Obtaining the member 'jdRef' of a type (line 90)
        jdRef_292973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 40), EpochConverter_292972, 'jdRef')
        # Applying the binary operator '-' (line 90)
        result_sub_292974 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 13), '-', julianDate_call_result_292971, jdRef_292973)
        
        # Assigning a type to the variable 'stypy_return_type' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 6), 'stypy_return_type', result_sub_292974)
        
        # ################# End of 'epoch2float(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'epoch2float' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_292975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292975)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'epoch2float'
        return stypy_return_type_292975


    @staticmethod
    @norecursion
    def duration2float(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'duration2float'
        module_type_store = module_type_store.open_function_context('duration2float', 93, 3, False)
        
        # Passed parameters checking function
        EpochConverter.duration2float.__dict__.__setitem__('stypy_localization', localization)
        EpochConverter.duration2float.__dict__.__setitem__('stypy_type_of_self', None)
        EpochConverter.duration2float.__dict__.__setitem__('stypy_type_store', module_type_store)
        EpochConverter.duration2float.__dict__.__setitem__('stypy_function_name', 'duration2float')
        EpochConverter.duration2float.__dict__.__setitem__('stypy_param_names_list', ['value'])
        EpochConverter.duration2float.__dict__.__setitem__('stypy_varargs_param_name', None)
        EpochConverter.duration2float.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EpochConverter.duration2float.__dict__.__setitem__('stypy_call_defaults', defaults)
        EpochConverter.duration2float.__dict__.__setitem__('stypy_call_varargs', varargs)
        EpochConverter.duration2float.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EpochConverter.duration2float.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, 'duration2float', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'duration2float', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'duration2float(...)' code ##################

        unicode_292976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, (-1)), 'unicode', u': Convert a Duration value to a float suitible for plotting as a\n           python datetime object.\n\n      = INPUT VARIABLES\n      - value   A Duration or list of Durations that need to be converted.\n\n      = RETURN VALUE\n      - Returns the value parameter converted to floats.\n      ')
        
        # Call to days(...): (line 104)
        # Processing the call keyword arguments (line 104)
        kwargs_292979 = {}
        # Getting the type of 'value' (line 104)
        value_292977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'value', False)
        # Obtaining the member 'days' of a type (line 104)
        days_292978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 13), value_292977, 'days')
        # Calling days(args, kwargs) (line 104)
        days_call_result_292980 = invoke(stypy.reporting.localization.Localization(__file__, 104, 13), days_292978, *[], **kwargs_292979)
        
        # Assigning a type to the variable 'stypy_return_type' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 6), 'stypy_return_type', days_call_result_292980)
        
        # ################# End of 'duration2float(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'duration2float' in the type store
        # Getting the type of 'stypy_return_type' (line 93)
        stypy_return_type_292981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292981)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'duration2float'
        return stypy_return_type_292981


    @staticmethod
    @norecursion
    def convert(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'convert'
        module_type_store = module_type_store.open_function_context('convert', 107, 3, False)
        
        # Passed parameters checking function
        EpochConverter.convert.__dict__.__setitem__('stypy_localization', localization)
        EpochConverter.convert.__dict__.__setitem__('stypy_type_of_self', None)
        EpochConverter.convert.__dict__.__setitem__('stypy_type_store', module_type_store)
        EpochConverter.convert.__dict__.__setitem__('stypy_function_name', 'convert')
        EpochConverter.convert.__dict__.__setitem__('stypy_param_names_list', ['value', 'unit', 'axis'])
        EpochConverter.convert.__dict__.__setitem__('stypy_varargs_param_name', None)
        EpochConverter.convert.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EpochConverter.convert.__dict__.__setitem__('stypy_call_defaults', defaults)
        EpochConverter.convert.__dict__.__setitem__('stypy_call_varargs', varargs)
        EpochConverter.convert.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EpochConverter.convert.__dict__.__setitem__('stypy_declared_arg_number', 3)
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

        unicode_292982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, (-1)), 'unicode', u': Convert value using unit to a float.  If value is a sequence, return\n      the converted sequence.\n\n      = INPUT VARIABLES\n      - value   The value or list of values that need to be converted.\n      - unit    The units to use for an axis with Epoch data.\n\n      = RETURN VALUE\n      - Returns the value parameter converted to floats.\n      ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 120, 6))
        
        # 'import matplotlib.testing.jpl_units' statement (line 120)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
        import_292983 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 120, 6), 'matplotlib.testing.jpl_units')

        if (type(import_292983) is not StypyTypeError):

            if (import_292983 != 'pyd_module'):
                __import__(import_292983)
                sys_modules_292984 = sys.modules[import_292983]
                import_module(stypy.reporting.localization.Localization(__file__, 120, 6), 'U', sys_modules_292984.module_type_store, module_type_store)
            else:
                import matplotlib.testing.jpl_units as U

                import_module(stypy.reporting.localization.Localization(__file__, 120, 6), 'U', matplotlib.testing.jpl_units, module_type_store)

        else:
            # Assigning a type to the variable 'matplotlib.testing.jpl_units' (line 120)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 6), 'matplotlib.testing.jpl_units', import_292983)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
        
        
        # Assigning a Name to a Name (line 122):
        # Getting the type of 'True' (line 122)
        True_292985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 19), 'True')
        # Assigning a type to the variable 'isNotEpoch' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 6), 'isNotEpoch', True_292985)
        
        # Assigning a Name to a Name (line 123):
        # Getting the type of 'False' (line 123)
        False_292986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 19), 'False')
        # Assigning a type to the variable 'isDuration' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 6), 'isDuration', False_292986)
        
        
        # Evaluating a boolean operation
        
        # Call to iterable(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'value' (line 125)
        value_292988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 20), 'value', False)
        # Processing the call keyword arguments (line 125)
        kwargs_292989 = {}
        # Getting the type of 'iterable' (line 125)
        iterable_292987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 11), 'iterable', False)
        # Calling iterable(args, kwargs) (line 125)
        iterable_call_result_292990 = invoke(stypy.reporting.localization.Localization(__file__, 125, 11), iterable_292987, *[value_292988], **kwargs_292989)
        
        
        
        # Call to isinstance(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'value' (line 125)
        value_292992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 46), 'value', False)
        # Getting the type of 'six' (line 125)
        six_292993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 53), 'six', False)
        # Obtaining the member 'string_types' of a type (line 125)
        string_types_292994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 53), six_292993, 'string_types')
        # Processing the call keyword arguments (line 125)
        kwargs_292995 = {}
        # Getting the type of 'isinstance' (line 125)
        isinstance_292991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 35), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 125)
        isinstance_call_result_292996 = invoke(stypy.reporting.localization.Localization(__file__, 125, 35), isinstance_292991, *[value_292992, string_types_292994], **kwargs_292995)
        
        # Applying the 'not' unary operator (line 125)
        result_not__292997 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 31), 'not', isinstance_call_result_292996)
        
        # Applying the binary operator 'and' (line 125)
        result_and_keyword_292998 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 11), 'and', iterable_call_result_292990, result_not__292997)
        
        # Testing the type of an if condition (line 125)
        if_condition_292999 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 6), result_and_keyword_292998)
        # Assigning a type to the variable 'if_condition_292999' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 6), 'if_condition_292999', if_condition_292999)
        # SSA begins for if statement (line 125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to len(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'value' (line 126)
        value_293001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 18), 'value', False)
        # Processing the call keyword arguments (line 126)
        kwargs_293002 = {}
        # Getting the type of 'len' (line 126)
        len_293000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 14), 'len', False)
        # Calling len(args, kwargs) (line 126)
        len_call_result_293003 = invoke(stypy.reporting.localization.Localization(__file__, 126, 14), len_293000, *[value_293001], **kwargs_293002)
        
        int_293004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 28), 'int')
        # Applying the binary operator '==' (line 126)
        result_eq_293005 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 14), '==', len_call_result_293003, int_293004)
        
        # Testing the type of an if condition (line 126)
        if_condition_293006 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 9), result_eq_293005)
        # Assigning a type to the variable 'if_condition_293006' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 9), 'if_condition_293006', if_condition_293006)
        # SSA begins for if statement (line 126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 127)
        list_293007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 127)
        
        # Assigning a type to the variable 'stypy_return_type' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'stypy_return_type', list_293007)
        # SSA branch for the else part of an if statement (line 126)
        module_type_store.open_ssa_branch('else')
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'value' (line 129)
        value_293015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 70), 'value')
        comprehension_293016 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 21), value_293015)
        # Assigning a type to the variable 'x' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 21), 'x', comprehension_293016)
        
        # Call to convert(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'x' (line 129)
        x_293010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 45), 'x', False)
        # Getting the type of 'unit' (line 129)
        unit_293011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 48), 'unit', False)
        # Getting the type of 'axis' (line 129)
        axis_293012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 54), 'axis', False)
        # Processing the call keyword arguments (line 129)
        kwargs_293013 = {}
        # Getting the type of 'EpochConverter' (line 129)
        EpochConverter_293008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 21), 'EpochConverter', False)
        # Obtaining the member 'convert' of a type (line 129)
        convert_293009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 21), EpochConverter_293008, 'convert')
        # Calling convert(args, kwargs) (line 129)
        convert_call_result_293014 = invoke(stypy.reporting.localization.Localization(__file__, 129, 21), convert_293009, *[x_293010, unit_293011, axis_293012], **kwargs_293013)
        
        list_293017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 21), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 21), list_293017, convert_call_result_293014)
        # Assigning a type to the variable 'stypy_return_type' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'stypy_return_type', list_293017)
        # SSA join for if statement (line 126)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 125)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to isinstance(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'value' (line 131)
        value_293019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 22), 'value', False)
        # Getting the type of 'U' (line 131)
        U_293020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 29), 'U', False)
        # Obtaining the member 'Epoch' of a type (line 131)
        Epoch_293021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 29), U_293020, 'Epoch')
        # Processing the call keyword arguments (line 131)
        kwargs_293022 = {}
        # Getting the type of 'isinstance' (line 131)
        isinstance_293018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 131)
        isinstance_call_result_293023 = invoke(stypy.reporting.localization.Localization(__file__, 131, 11), isinstance_293018, *[value_293019, Epoch_293021], **kwargs_293022)
        
        # Testing the type of an if condition (line 131)
        if_condition_293024 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 131, 6), isinstance_call_result_293023)
        # Assigning a type to the variable 'if_condition_293024' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 6), 'if_condition_293024', if_condition_293024)
        # SSA begins for if statement (line 131)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 132):
        # Getting the type of 'False' (line 132)
        False_293025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 22), 'False')
        # Assigning a type to the variable 'isNotEpoch' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 9), 'isNotEpoch', False_293025)
        # SSA branch for the else part of an if statement (line 131)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isinstance(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'value' (line 133)
        value_293027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 24), 'value', False)
        # Getting the type of 'U' (line 133)
        U_293028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 31), 'U', False)
        # Obtaining the member 'Duration' of a type (line 133)
        Duration_293029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 31), U_293028, 'Duration')
        # Processing the call keyword arguments (line 133)
        kwargs_293030 = {}
        # Getting the type of 'isinstance' (line 133)
        isinstance_293026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 13), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 133)
        isinstance_call_result_293031 = invoke(stypy.reporting.localization.Localization(__file__, 133, 13), isinstance_293026, *[value_293027, Duration_293029], **kwargs_293030)
        
        # Testing the type of an if condition (line 133)
        if_condition_293032 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 11), isinstance_call_result_293031)
        # Assigning a type to the variable 'if_condition_293032' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'if_condition_293032', if_condition_293032)
        # SSA begins for if statement (line 133)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 134):
        # Getting the type of 'True' (line 134)
        True_293033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 22), 'True')
        # Assigning a type to the variable 'isDuration' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 9), 'isDuration', True_293033)
        # SSA join for if statement (line 133)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 131)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'isNotEpoch' (line 136)
        isNotEpoch_293034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 11), 'isNotEpoch')
        
        # Getting the type of 'isDuration' (line 136)
        isDuration_293035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 30), 'isDuration')
        # Applying the 'not' unary operator (line 136)
        result_not__293036 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 26), 'not', isDuration_293035)
        
        # Applying the binary operator 'and' (line 136)
        result_and_keyword_293037 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 11), 'and', isNotEpoch_293034, result_not__293036)
        
        # Call to is_numlike(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'value' (line 137)
        value_293041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 49), 'value', False)
        # Processing the call keyword arguments (line 137)
        kwargs_293042 = {}
        # Getting the type of 'units' (line 137)
        units_293038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'units', False)
        # Obtaining the member 'ConversionInterface' of a type (line 137)
        ConversionInterface_293039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 11), units_293038, 'ConversionInterface')
        # Obtaining the member 'is_numlike' of a type (line 137)
        is_numlike_293040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 11), ConversionInterface_293039, 'is_numlike')
        # Calling is_numlike(args, kwargs) (line 137)
        is_numlike_call_result_293043 = invoke(stypy.reporting.localization.Localization(__file__, 137, 11), is_numlike_293040, *[value_293041], **kwargs_293042)
        
        # Applying the binary operator 'and' (line 136)
        result_and_keyword_293044 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 11), 'and', result_and_keyword_293037, is_numlike_call_result_293043)
        
        # Testing the type of an if condition (line 136)
        if_condition_293045 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 136, 6), result_and_keyword_293044)
        # Assigning a type to the variable 'if_condition_293045' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 6), 'if_condition_293045', if_condition_293045)
        # SSA begins for if statement (line 136)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'value' (line 138)
        value_293046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'value')
        # Assigning a type to the variable 'stypy_return_type' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 9), 'stypy_return_type', value_293046)
        # SSA join for if statement (line 136)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 140)
        # Getting the type of 'unit' (line 140)
        unit_293047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 6), 'unit')
        # Getting the type of 'None' (line 140)
        None_293048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 19), 'None')
        
        (may_be_293049, more_types_in_union_293050) = may_be_none(unit_293047, None_293048)

        if may_be_293049:

            if more_types_in_union_293050:
                # Runtime conditional SSA (line 140)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 141):
            
            # Call to default_units(...): (line 141)
            # Processing the call arguments (line 141)
            # Getting the type of 'value' (line 141)
            value_293053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 46), 'value', False)
            # Getting the type of 'axis' (line 141)
            axis_293054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 53), 'axis', False)
            # Processing the call keyword arguments (line 141)
            kwargs_293055 = {}
            # Getting the type of 'EpochConverter' (line 141)
            EpochConverter_293051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'EpochConverter', False)
            # Obtaining the member 'default_units' of a type (line 141)
            default_units_293052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 16), EpochConverter_293051, 'default_units')
            # Calling default_units(args, kwargs) (line 141)
            default_units_call_result_293056 = invoke(stypy.reporting.localization.Localization(__file__, 141, 16), default_units_293052, *[value_293053, axis_293054], **kwargs_293055)
            
            # Assigning a type to the variable 'unit' (line 141)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 9), 'unit', default_units_call_result_293056)

            if more_types_in_union_293050:
                # SSA join for if statement (line 140)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'isDuration' (line 143)
        isDuration_293057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 11), 'isDuration')
        # Testing the type of an if condition (line 143)
        if_condition_293058 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 6), isDuration_293057)
        # Assigning a type to the variable 'if_condition_293058' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 6), 'if_condition_293058', if_condition_293058)
        # SSA begins for if statement (line 143)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to duration2float(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'value' (line 144)
        value_293061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 47), 'value', False)
        # Processing the call keyword arguments (line 144)
        kwargs_293062 = {}
        # Getting the type of 'EpochConverter' (line 144)
        EpochConverter_293059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'EpochConverter', False)
        # Obtaining the member 'duration2float' of a type (line 144)
        duration2float_293060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 16), EpochConverter_293059, 'duration2float')
        # Calling duration2float(args, kwargs) (line 144)
        duration2float_call_result_293063 = invoke(stypy.reporting.localization.Localization(__file__, 144, 16), duration2float_293060, *[value_293061], **kwargs_293062)
        
        # Assigning a type to the variable 'stypy_return_type' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 9), 'stypy_return_type', duration2float_call_result_293063)
        # SSA branch for the else part of an if statement (line 143)
        module_type_store.open_ssa_branch('else')
        
        # Call to epoch2float(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'value' (line 146)
        value_293066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 44), 'value', False)
        # Getting the type of 'unit' (line 146)
        unit_293067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 51), 'unit', False)
        # Processing the call keyword arguments (line 146)
        kwargs_293068 = {}
        # Getting the type of 'EpochConverter' (line 146)
        EpochConverter_293064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'EpochConverter', False)
        # Obtaining the member 'epoch2float' of a type (line 146)
        epoch2float_293065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 16), EpochConverter_293064, 'epoch2float')
        # Calling epoch2float(args, kwargs) (line 146)
        epoch2float_call_result_293069 = invoke(stypy.reporting.localization.Localization(__file__, 146, 16), epoch2float_293065, *[value_293066, unit_293067], **kwargs_293068)
        
        # Assigning a type to the variable 'stypy_return_type' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 9), 'stypy_return_type', epoch2float_call_result_293069)
        # SSA join for if statement (line 143)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'convert(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'convert' in the type store
        # Getting the type of 'stypy_return_type' (line 107)
        stypy_return_type_293070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293070)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'convert'
        return stypy_return_type_293070


    @staticmethod
    @norecursion
    def default_units(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'default_units'
        module_type_store = module_type_store.open_function_context('default_units', 149, 3, False)
        
        # Passed parameters checking function
        EpochConverter.default_units.__dict__.__setitem__('stypy_localization', localization)
        EpochConverter.default_units.__dict__.__setitem__('stypy_type_of_self', None)
        EpochConverter.default_units.__dict__.__setitem__('stypy_type_store', module_type_store)
        EpochConverter.default_units.__dict__.__setitem__('stypy_function_name', 'default_units')
        EpochConverter.default_units.__dict__.__setitem__('stypy_param_names_list', ['value', 'axis'])
        EpochConverter.default_units.__dict__.__setitem__('stypy_varargs_param_name', None)
        EpochConverter.default_units.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EpochConverter.default_units.__dict__.__setitem__('stypy_call_defaults', defaults)
        EpochConverter.default_units.__dict__.__setitem__('stypy_call_varargs', varargs)
        EpochConverter.default_units.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EpochConverter.default_units.__dict__.__setitem__('stypy_declared_arg_number', 2)
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

        unicode_293071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, (-1)), 'unicode', u': Return the default unit for value, or None.\n\n      = INPUT VARIABLES\n      - value   The value or list of values that need units.\n\n      = RETURN VALUE\n      - Returns the default units to use for value.\n      ')
        
        # Assigning a Name to a Name (line 159):
        # Getting the type of 'None' (line 159)
        None_293072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 14), 'None')
        # Assigning a type to the variable 'frame' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 6), 'frame', None_293072)
        
        
        # Evaluating a boolean operation
        
        # Call to iterable(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'value' (line 160)
        value_293074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'value', False)
        # Processing the call keyword arguments (line 160)
        kwargs_293075 = {}
        # Getting the type of 'iterable' (line 160)
        iterable_293073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 11), 'iterable', False)
        # Calling iterable(args, kwargs) (line 160)
        iterable_call_result_293076 = invoke(stypy.reporting.localization.Localization(__file__, 160, 11), iterable_293073, *[value_293074], **kwargs_293075)
        
        
        
        # Call to isinstance(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'value' (line 160)
        value_293078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 46), 'value', False)
        # Getting the type of 'six' (line 160)
        six_293079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 53), 'six', False)
        # Obtaining the member 'string_types' of a type (line 160)
        string_types_293080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 53), six_293079, 'string_types')
        # Processing the call keyword arguments (line 160)
        kwargs_293081 = {}
        # Getting the type of 'isinstance' (line 160)
        isinstance_293077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 35), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 160)
        isinstance_call_result_293082 = invoke(stypy.reporting.localization.Localization(__file__, 160, 35), isinstance_293077, *[value_293078, string_types_293080], **kwargs_293081)
        
        # Applying the 'not' unary operator (line 160)
        result_not__293083 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 31), 'not', isinstance_call_result_293082)
        
        # Applying the binary operator 'and' (line 160)
        result_and_keyword_293084 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 11), 'and', iterable_call_result_293076, result_not__293083)
        
        # Testing the type of an if condition (line 160)
        if_condition_293085 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 6), result_and_keyword_293084)
        # Assigning a type to the variable 'if_condition_293085' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 6), 'if_condition_293085', if_condition_293085)
        # SSA begins for if statement (line 160)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to default_units(...): (line 161)
        # Processing the call arguments (line 161)
        
        # Obtaining the type of the subscript
        int_293088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 52), 'int')
        # Getting the type of 'value' (line 161)
        value_293089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 46), 'value', False)
        # Obtaining the member '__getitem__' of a type (line 161)
        getitem___293090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 46), value_293089, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 161)
        subscript_call_result_293091 = invoke(stypy.reporting.localization.Localization(__file__, 161, 46), getitem___293090, int_293088)
        
        # Getting the type of 'axis' (line 161)
        axis_293092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 56), 'axis', False)
        # Processing the call keyword arguments (line 161)
        kwargs_293093 = {}
        # Getting the type of 'EpochConverter' (line 161)
        EpochConverter_293086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'EpochConverter', False)
        # Obtaining the member 'default_units' of a type (line 161)
        default_units_293087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 16), EpochConverter_293086, 'default_units')
        # Calling default_units(args, kwargs) (line 161)
        default_units_call_result_293094 = invoke(stypy.reporting.localization.Localization(__file__, 161, 16), default_units_293087, *[subscript_call_result_293091, axis_293092], **kwargs_293093)
        
        # Assigning a type to the variable 'stypy_return_type' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 9), 'stypy_return_type', default_units_call_result_293094)
        # SSA branch for the else part of an if statement (line 160)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 163):
        
        # Call to frame(...): (line 163)
        # Processing the call keyword arguments (line 163)
        kwargs_293097 = {}
        # Getting the type of 'value' (line 163)
        value_293095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 17), 'value', False)
        # Obtaining the member 'frame' of a type (line 163)
        frame_293096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 17), value_293095, 'frame')
        # Calling frame(args, kwargs) (line 163)
        frame_call_result_293098 = invoke(stypy.reporting.localization.Localization(__file__, 163, 17), frame_293096, *[], **kwargs_293097)
        
        # Assigning a type to the variable 'frame' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 9), 'frame', frame_call_result_293098)
        # SSA join for if statement (line 160)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'frame' (line 165)
        frame_293099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 13), 'frame')
        # Assigning a type to the variable 'stypy_return_type' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 6), 'stypy_return_type', frame_293099)
        
        # ################# End of 'default_units(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'default_units' in the type store
        # Getting the type of 'stypy_return_type' (line 149)
        stypy_return_type_293100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293100)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'default_units'
        return stypy_return_type_293100


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EpochConverter.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'EpochConverter' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'EpochConverter', EpochConverter)

# Assigning a BinOp to a Name (line 35):
float_293101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 11), 'float')
int_293102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 23), 'int')
# Applying the binary operator '-' (line 35)
result_sub_293103 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 11), '-', float_293101, int_293102)

# Getting the type of 'EpochConverter'
EpochConverter_293104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'EpochConverter')
# Setting the type of the member 'jdRef' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), EpochConverter_293104, 'jdRef', result_sub_293103)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
