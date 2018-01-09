
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #===========================================================================
2: #
3: # UnitDbl
4: #
5: #===========================================================================
6: 
7: 
8: '''UnitDbl module.'''
9: 
10: #===========================================================================
11: # Place all imports after here.
12: #
13: from __future__ import (absolute_import, division, print_function,
14:                         unicode_literals)
15: 
16: import six
17: #
18: # Place all imports before here.
19: #===========================================================================
20: 
21: 
22: #===========================================================================
23: class UnitDbl(object):
24:    '''Class UnitDbl in development.
25:    '''
26:    #-----------------------------------------------------------------------
27:    # Unit conversion table.  Small subset of the full one but enough
28:    # to test the required functions.  First field is a scale factor to
29:    # convert the input units to the units of the second field.  Only
30:    # units in this table are allowed.
31:    allowed = {
32:                "m" : ( 0.001, "km" ),
33:                "km" : ( 1, "km" ),
34:                "mile" : ( 1.609344, "km" ),
35: 
36:                "rad" : ( 1, "rad" ),
37:                "deg" : ( 1.745329251994330e-02, "rad" ),
38: 
39:                "sec" : ( 1, "sec" ),
40:                "min" : ( 60.0, "sec" ),
41:                "hour" : ( 3600, "sec" ),
42:              }
43: 
44:    _types = {
45:               "km" : "distance",
46:               "rad" : "angle",
47:               "sec" : "time",
48:             }
49: 
50:    #-----------------------------------------------------------------------
51:    def __init__( self, value, units ):
52:       '''Create a new UnitDbl object.
53: 
54:       Units are internally converted to km, rad, and sec.  The only
55:       valid inputs for units are [ m, km, mile, rad, deg, sec, min, hour ].
56: 
57:       The field UnitDbl.value will contain the converted value.  Use
58:       the convert() method to get a specific type of units back.
59: 
60:       = ERROR CONDITIONS
61:       - If the input units are not in the allowed list, an error is thrown.
62: 
63:       = INPUT VARIABLES
64:       - value    The numeric value of the UnitDbl.
65:       - units    The string name of the units the value is in.
66:       '''
67:       self.checkUnits( units )
68: 
69:       data = self.allowed[ units ]
70:       self._value = float( value * data[0] )
71:       self._units = data[1]
72: 
73:    #-----------------------------------------------------------------------
74:    def convert( self, units ):
75:       '''Convert the UnitDbl to a specific set of units.
76: 
77:       = ERROR CONDITIONS
78:       - If the input units are not in the allowed list, an error is thrown.
79: 
80:       = INPUT VARIABLES
81:       - units    The string name of the units to convert to.
82: 
83:       = RETURN VALUE
84:       - Returns the value of the UnitDbl in the requested units as a floating
85:         point number.
86:       '''
87:       if self._units == units:
88:          return self._value
89: 
90:       self.checkUnits( units )
91: 
92:       data = self.allowed[ units ]
93:       if self._units != data[1]:
94:          msg = "Error trying to convert to different units.\n" \
95:                "   Invalid conversion requested.\n" \
96:                "   UnitDbl: %s\n" \
97:                "   Units:   %s\n" % ( str( self ), units )
98:          raise ValueError( msg )
99: 
100:       return self._value / data[0]
101: 
102:    #-----------------------------------------------------------------------
103:    def __abs__( self ):
104:       '''Return the absolute value of this UnitDbl.'''
105:       return UnitDbl( abs( self._value ), self._units )
106: 
107:    #-----------------------------------------------------------------------
108:    def __neg__( self ):
109:       '''Return the negative value of this UnitDbl.'''
110:       return UnitDbl( -self._value, self._units )
111: 
112:    #-----------------------------------------------------------------------
113:    def __nonzero__( self ):
114:       '''Test a UnitDbl for a non-zero value.
115: 
116:       = RETURN VALUE
117:       - Returns true if the value is non-zero.
118:       '''
119:       if six.PY3:
120:           return self._value.__bool__()
121:       else:
122:           return self._value.__nonzero__()
123: 
124:    if six.PY3:
125:       __bool__ = __nonzero__
126: 
127:    #-----------------------------------------------------------------------
128:    def __cmp__( self, rhs ):
129:       '''Compare two UnitDbl's.
130: 
131:       = ERROR CONDITIONS
132:       - If the input rhs units are not the same as our units,
133:         an error is thrown.
134: 
135:       = INPUT VARIABLES
136:       - rhs    The UnitDbl to compare against.
137: 
138:       = RETURN VALUE
139:       - Returns -1 if self < rhs, 0 if self == rhs, +1 if self > rhs.
140:       '''
141:       self.checkSameUnits( rhs, "compare" )
142:       return cmp( self._value, rhs._value )
143: 
144:    #-----------------------------------------------------------------------
145:    def __add__( self, rhs ):
146:       '''Add two UnitDbl's.
147: 
148:       = ERROR CONDITIONS
149:       - If the input rhs units are not the same as our units,
150:         an error is thrown.
151: 
152:       = INPUT VARIABLES
153:       - rhs    The UnitDbl to add.
154: 
155:       = RETURN VALUE
156:       - Returns the sum of ourselves and the input UnitDbl.
157:       '''
158:       self.checkSameUnits( rhs, "add" )
159:       return UnitDbl( self._value + rhs._value, self._units )
160: 
161:    #-----------------------------------------------------------------------
162:    def __sub__( self, rhs ):
163:       '''Subtract two UnitDbl's.
164: 
165:       = ERROR CONDITIONS
166:       - If the input rhs units are not the same as our units,
167:         an error is thrown.
168: 
169:       = INPUT VARIABLES
170:       - rhs    The UnitDbl to subtract.
171: 
172:       = RETURN VALUE
173:       - Returns the difference of ourselves and the input UnitDbl.
174:       '''
175:       self.checkSameUnits( rhs, "subtract" )
176:       return UnitDbl( self._value - rhs._value, self._units )
177: 
178:    #-----------------------------------------------------------------------
179:    def __mul__( self, rhs ):
180:       '''Scale a UnitDbl by a value.
181: 
182:       = INPUT VARIABLES
183:       - rhs    The scalar to multiply by.
184: 
185:       = RETURN VALUE
186:       - Returns the scaled UnitDbl.
187:       '''
188:       return UnitDbl( self._value * rhs, self._units )
189: 
190:    #-----------------------------------------------------------------------
191:    def __rmul__( self, lhs ):
192:       '''Scale a UnitDbl by a value.
193: 
194:       = INPUT VARIABLES
195:       - lhs    The scalar to multiply by.
196: 
197:       = RETURN VALUE
198:       - Returns the scaled UnitDbl.
199:       '''
200:       return UnitDbl( self._value * lhs, self._units )
201: 
202:    #-----------------------------------------------------------------------
203:    def __div__( self, rhs ):
204:       '''Divide a UnitDbl by a value.
205: 
206:       = INPUT VARIABLES
207:       - rhs    The scalar to divide by.
208: 
209:       = RETURN VALUE
210:       - Returns the scaled UnitDbl.
211:       '''
212:       return UnitDbl( self._value / rhs, self._units )
213: 
214:    #-----------------------------------------------------------------------
215:    def __str__( self ):
216:       '''Print the UnitDbl.'''
217:       return "%g *%s" % ( self._value, self._units )
218: 
219:    #-----------------------------------------------------------------------
220:    def __repr__( self ):
221:       '''Print the UnitDbl.'''
222:       return "UnitDbl( %g, '%s' )" % ( self._value, self._units )
223: 
224:    #-----------------------------------------------------------------------
225:    def type( self ):
226:       '''Return the type of UnitDbl data.'''
227:       return self._types[ self._units ]
228: 
229:    #-----------------------------------------------------------------------
230:    def range( start, stop, step=None ):
231:       '''Generate a range of UnitDbl objects.
232: 
233:       Similar to the Python range() method.  Returns the range [
234:       start, stop ) at the requested step.  Each element will be a
235:       UnitDbl object.
236: 
237:       = INPUT VARIABLES
238:       - start    The starting value of the range.
239:       - stop     The stop value of the range.
240:       - step     Optional step to use.  If set to None, then a UnitDbl of
241:                  value 1 w/ the units of the start is used.
242: 
243:       = RETURN VALUE
244:       - Returns a list contianing the requested UnitDbl values.
245:       '''
246:       if step is None:
247:          step = UnitDbl( 1, start._units )
248: 
249:       elems = []
250: 
251:       i = 0
252:       while True:
253:          d = start + i * step
254:          if d >= stop:
255:             break
256: 
257:          elems.append( d )
258:          i += 1
259: 
260:       return elems
261: 
262:    range = staticmethod( range )
263: 
264:    #-----------------------------------------------------------------------
265:    def checkUnits( self, units ):
266:       '''Check to see if some units are valid.
267: 
268:       = ERROR CONDITIONS
269:       - If the input units are not in the allowed list, an error is thrown.
270: 
271:       = INPUT VARIABLES
272:       - units    The string name of the units to check.
273:       '''
274:       if units not in self.allowed:
275:          msg = "Input units '%s' are not one of the supported types of %s" \
276:                % ( units, str( list(six.iterkeys(self.allowed)) ) )
277:          raise ValueError( msg )
278: 
279:    #-----------------------------------------------------------------------
280:    def checkSameUnits( self, rhs, func ):
281:       '''Check to see if units are the same.
282: 
283:       = ERROR CONDITIONS
284:       - If the units of the rhs UnitDbl are not the same as our units,
285:         an error is thrown.
286: 
287:       = INPUT VARIABLES
288:       - rhs    The UnitDbl to check for the same units
289:       - func   The name of the function doing the check.
290:       '''
291:       if self._units != rhs._units:
292:          msg = "Cannot %s units of different types.\n" \
293:                "LHS: %s\n" \
294:                "RHS: %s" % ( func, self._units, rhs._units )
295:          raise ValueError( msg )
296: 
297: #===========================================================================
298: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_293323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 0), 'unicode', u'UnitDbl module.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import six' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_293324 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six')

if (type(import_293324) is not StypyTypeError):

    if (import_293324 != 'pyd_module'):
        __import__(import_293324)
        sys_modules_293325 = sys.modules[import_293324]
        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', sys_modules_293325.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', import_293324)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

# Declaration of the 'UnitDbl' class

class UnitDbl(object, ):
    unicode_293326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, (-1)), 'unicode', u'Class UnitDbl in development.\n   ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 51, 3, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitDbl.__init__', ['value', 'units'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['value', 'units'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_293327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, (-1)), 'unicode', u'Create a new UnitDbl object.\n\n      Units are internally converted to km, rad, and sec.  The only\n      valid inputs for units are [ m, km, mile, rad, deg, sec, min, hour ].\n\n      The field UnitDbl.value will contain the converted value.  Use\n      the convert() method to get a specific type of units back.\n\n      = ERROR CONDITIONS\n      - If the input units are not in the allowed list, an error is thrown.\n\n      = INPUT VARIABLES\n      - value    The numeric value of the UnitDbl.\n      - units    The string name of the units the value is in.\n      ')
        
        # Call to checkUnits(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'units' (line 67)
        units_293330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 23), 'units', False)
        # Processing the call keyword arguments (line 67)
        kwargs_293331 = {}
        # Getting the type of 'self' (line 67)
        self_293328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 6), 'self', False)
        # Obtaining the member 'checkUnits' of a type (line 67)
        checkUnits_293329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 6), self_293328, 'checkUnits')
        # Calling checkUnits(args, kwargs) (line 67)
        checkUnits_call_result_293332 = invoke(stypy.reporting.localization.Localization(__file__, 67, 6), checkUnits_293329, *[units_293330], **kwargs_293331)
        
        
        # Assigning a Subscript to a Name (line 69):
        
        # Obtaining the type of the subscript
        # Getting the type of 'units' (line 69)
        units_293333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 27), 'units')
        # Getting the type of 'self' (line 69)
        self_293334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'self')
        # Obtaining the member 'allowed' of a type (line 69)
        allowed_293335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 13), self_293334, 'allowed')
        # Obtaining the member '__getitem__' of a type (line 69)
        getitem___293336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 13), allowed_293335, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 69)
        subscript_call_result_293337 = invoke(stypy.reporting.localization.Localization(__file__, 69, 13), getitem___293336, units_293333)
        
        # Assigning a type to the variable 'data' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 6), 'data', subscript_call_result_293337)
        
        # Assigning a Call to a Attribute (line 70):
        
        # Call to float(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'value' (line 70)
        value_293339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 27), 'value', False)
        
        # Obtaining the type of the subscript
        int_293340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 40), 'int')
        # Getting the type of 'data' (line 70)
        data_293341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 35), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 70)
        getitem___293342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 35), data_293341, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 70)
        subscript_call_result_293343 = invoke(stypy.reporting.localization.Localization(__file__, 70, 35), getitem___293342, int_293340)
        
        # Applying the binary operator '*' (line 70)
        result_mul_293344 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 27), '*', value_293339, subscript_call_result_293343)
        
        # Processing the call keyword arguments (line 70)
        kwargs_293345 = {}
        # Getting the type of 'float' (line 70)
        float_293338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 20), 'float', False)
        # Calling float(args, kwargs) (line 70)
        float_call_result_293346 = invoke(stypy.reporting.localization.Localization(__file__, 70, 20), float_293338, *[result_mul_293344], **kwargs_293345)
        
        # Getting the type of 'self' (line 70)
        self_293347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 6), 'self')
        # Setting the type of the member '_value' of a type (line 70)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 6), self_293347, '_value', float_call_result_293346)
        
        # Assigning a Subscript to a Attribute (line 71):
        
        # Obtaining the type of the subscript
        int_293348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 25), 'int')
        # Getting the type of 'data' (line 71)
        data_293349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 20), 'data')
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___293350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 20), data_293349, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_293351 = invoke(stypy.reporting.localization.Localization(__file__, 71, 20), getitem___293350, int_293348)
        
        # Getting the type of 'self' (line 71)
        self_293352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 6), 'self')
        # Setting the type of the member '_units' of a type (line 71)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 6), self_293352, '_units', subscript_call_result_293351)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def convert(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'convert'
        module_type_store = module_type_store.open_function_context('convert', 74, 3, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        UnitDbl.convert.__dict__.__setitem__('stypy_localization', localization)
        UnitDbl.convert.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnitDbl.convert.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitDbl.convert.__dict__.__setitem__('stypy_function_name', 'UnitDbl.convert')
        UnitDbl.convert.__dict__.__setitem__('stypy_param_names_list', ['units'])
        UnitDbl.convert.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitDbl.convert.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitDbl.convert.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitDbl.convert.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitDbl.convert.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitDbl.convert.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitDbl.convert', ['units'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'convert', localization, ['units'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'convert(...)' code ##################

        unicode_293353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, (-1)), 'unicode', u'Convert the UnitDbl to a specific set of units.\n\n      = ERROR CONDITIONS\n      - If the input units are not in the allowed list, an error is thrown.\n\n      = INPUT VARIABLES\n      - units    The string name of the units to convert to.\n\n      = RETURN VALUE\n      - Returns the value of the UnitDbl in the requested units as a floating\n        point number.\n      ')
        
        
        # Getting the type of 'self' (line 87)
        self_293354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 9), 'self')
        # Obtaining the member '_units' of a type (line 87)
        _units_293355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 9), self_293354, '_units')
        # Getting the type of 'units' (line 87)
        units_293356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 24), 'units')
        # Applying the binary operator '==' (line 87)
        result_eq_293357 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 9), '==', _units_293355, units_293356)
        
        # Testing the type of an if condition (line 87)
        if_condition_293358 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 6), result_eq_293357)
        # Assigning a type to the variable 'if_condition_293358' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 6), 'if_condition_293358', if_condition_293358)
        # SSA begins for if statement (line 87)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 88)
        self_293359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'self')
        # Obtaining the member '_value' of a type (line 88)
        _value_293360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 16), self_293359, '_value')
        # Assigning a type to the variable 'stypy_return_type' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 9), 'stypy_return_type', _value_293360)
        # SSA join for if statement (line 87)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to checkUnits(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'units' (line 90)
        units_293363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 23), 'units', False)
        # Processing the call keyword arguments (line 90)
        kwargs_293364 = {}
        # Getting the type of 'self' (line 90)
        self_293361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 6), 'self', False)
        # Obtaining the member 'checkUnits' of a type (line 90)
        checkUnits_293362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 6), self_293361, 'checkUnits')
        # Calling checkUnits(args, kwargs) (line 90)
        checkUnits_call_result_293365 = invoke(stypy.reporting.localization.Localization(__file__, 90, 6), checkUnits_293362, *[units_293363], **kwargs_293364)
        
        
        # Assigning a Subscript to a Name (line 92):
        
        # Obtaining the type of the subscript
        # Getting the type of 'units' (line 92)
        units_293366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 27), 'units')
        # Getting the type of 'self' (line 92)
        self_293367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 13), 'self')
        # Obtaining the member 'allowed' of a type (line 92)
        allowed_293368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 13), self_293367, 'allowed')
        # Obtaining the member '__getitem__' of a type (line 92)
        getitem___293369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 13), allowed_293368, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 92)
        subscript_call_result_293370 = invoke(stypy.reporting.localization.Localization(__file__, 92, 13), getitem___293369, units_293366)
        
        # Assigning a type to the variable 'data' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 6), 'data', subscript_call_result_293370)
        
        
        # Getting the type of 'self' (line 93)
        self_293371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 9), 'self')
        # Obtaining the member '_units' of a type (line 93)
        _units_293372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 9), self_293371, '_units')
        
        # Obtaining the type of the subscript
        int_293373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 29), 'int')
        # Getting the type of 'data' (line 93)
        data_293374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 24), 'data')
        # Obtaining the member '__getitem__' of a type (line 93)
        getitem___293375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 24), data_293374, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 93)
        subscript_call_result_293376 = invoke(stypy.reporting.localization.Localization(__file__, 93, 24), getitem___293375, int_293373)
        
        # Applying the binary operator '!=' (line 93)
        result_ne_293377 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 9), '!=', _units_293372, subscript_call_result_293376)
        
        # Testing the type of an if condition (line 93)
        if_condition_293378 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 6), result_ne_293377)
        # Assigning a type to the variable 'if_condition_293378' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 6), 'if_condition_293378', if_condition_293378)
        # SSA begins for if statement (line 93)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 94):
        unicode_293379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 15), 'unicode', u'Error trying to convert to different units.\n   Invalid conversion requested.\n   UnitDbl: %s\n   Units:   %s\n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 97)
        tuple_293380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 97)
        # Adding element type (line 97)
        
        # Call to str(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'self' (line 97)
        self_293382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 43), 'self', False)
        # Processing the call keyword arguments (line 97)
        kwargs_293383 = {}
        # Getting the type of 'str' (line 97)
        str_293381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 38), 'str', False)
        # Calling str(args, kwargs) (line 97)
        str_call_result_293384 = invoke(stypy.reporting.localization.Localization(__file__, 97, 38), str_293381, *[self_293382], **kwargs_293383)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 38), tuple_293380, str_call_result_293384)
        # Adding element type (line 97)
        # Getting the type of 'units' (line 97)
        units_293385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 51), 'units')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 38), tuple_293380, units_293385)
        
        # Applying the binary operator '%' (line 94)
        result_mod_293386 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 15), '%', unicode_293379, tuple_293380)
        
        # Assigning a type to the variable 'msg' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 9), 'msg', result_mod_293386)
        
        # Call to ValueError(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'msg' (line 98)
        msg_293388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 27), 'msg', False)
        # Processing the call keyword arguments (line 98)
        kwargs_293389 = {}
        # Getting the type of 'ValueError' (line 98)
        ValueError_293387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 15), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 98)
        ValueError_call_result_293390 = invoke(stypy.reporting.localization.Localization(__file__, 98, 15), ValueError_293387, *[msg_293388], **kwargs_293389)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 98, 9), ValueError_call_result_293390, 'raise parameter', BaseException)
        # SSA join for if statement (line 93)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'self' (line 100)
        self_293391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 13), 'self')
        # Obtaining the member '_value' of a type (line 100)
        _value_293392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 13), self_293391, '_value')
        
        # Obtaining the type of the subscript
        int_293393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 32), 'int')
        # Getting the type of 'data' (line 100)
        data_293394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'data')
        # Obtaining the member '__getitem__' of a type (line 100)
        getitem___293395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 27), data_293394, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 100)
        subscript_call_result_293396 = invoke(stypy.reporting.localization.Localization(__file__, 100, 27), getitem___293395, int_293393)
        
        # Applying the binary operator 'div' (line 100)
        result_div_293397 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 13), 'div', _value_293392, subscript_call_result_293396)
        
        # Assigning a type to the variable 'stypy_return_type' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 6), 'stypy_return_type', result_div_293397)
        
        # ################# End of 'convert(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'convert' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_293398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293398)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'convert'
        return stypy_return_type_293398


    @norecursion
    def __abs__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__abs__'
        module_type_store = module_type_store.open_function_context('__abs__', 103, 3, False)
        # Assigning a type to the variable 'self' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        UnitDbl.__abs__.__dict__.__setitem__('stypy_localization', localization)
        UnitDbl.__abs__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnitDbl.__abs__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitDbl.__abs__.__dict__.__setitem__('stypy_function_name', 'UnitDbl.__abs__')
        UnitDbl.__abs__.__dict__.__setitem__('stypy_param_names_list', [])
        UnitDbl.__abs__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitDbl.__abs__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitDbl.__abs__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitDbl.__abs__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitDbl.__abs__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitDbl.__abs__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitDbl.__abs__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__abs__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__abs__(...)' code ##################

        unicode_293399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 6), 'unicode', u'Return the absolute value of this UnitDbl.')
        
        # Call to UnitDbl(...): (line 105)
        # Processing the call arguments (line 105)
        
        # Call to abs(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'self' (line 105)
        self_293402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 27), 'self', False)
        # Obtaining the member '_value' of a type (line 105)
        _value_293403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 27), self_293402, '_value')
        # Processing the call keyword arguments (line 105)
        kwargs_293404 = {}
        # Getting the type of 'abs' (line 105)
        abs_293401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 22), 'abs', False)
        # Calling abs(args, kwargs) (line 105)
        abs_call_result_293405 = invoke(stypy.reporting.localization.Localization(__file__, 105, 22), abs_293401, *[_value_293403], **kwargs_293404)
        
        # Getting the type of 'self' (line 105)
        self_293406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 42), 'self', False)
        # Obtaining the member '_units' of a type (line 105)
        _units_293407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 42), self_293406, '_units')
        # Processing the call keyword arguments (line 105)
        kwargs_293408 = {}
        # Getting the type of 'UnitDbl' (line 105)
        UnitDbl_293400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 13), 'UnitDbl', False)
        # Calling UnitDbl(args, kwargs) (line 105)
        UnitDbl_call_result_293409 = invoke(stypy.reporting.localization.Localization(__file__, 105, 13), UnitDbl_293400, *[abs_call_result_293405, _units_293407], **kwargs_293408)
        
        # Assigning a type to the variable 'stypy_return_type' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 6), 'stypy_return_type', UnitDbl_call_result_293409)
        
        # ################# End of '__abs__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__abs__' in the type store
        # Getting the type of 'stypy_return_type' (line 103)
        stypy_return_type_293410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293410)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__abs__'
        return stypy_return_type_293410


    @norecursion
    def __neg__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__neg__'
        module_type_store = module_type_store.open_function_context('__neg__', 108, 3, False)
        # Assigning a type to the variable 'self' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        UnitDbl.__neg__.__dict__.__setitem__('stypy_localization', localization)
        UnitDbl.__neg__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnitDbl.__neg__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitDbl.__neg__.__dict__.__setitem__('stypy_function_name', 'UnitDbl.__neg__')
        UnitDbl.__neg__.__dict__.__setitem__('stypy_param_names_list', [])
        UnitDbl.__neg__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitDbl.__neg__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitDbl.__neg__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitDbl.__neg__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitDbl.__neg__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitDbl.__neg__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitDbl.__neg__', [], None, None, defaults, varargs, kwargs)

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

        unicode_293411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 6), 'unicode', u'Return the negative value of this UnitDbl.')
        
        # Call to UnitDbl(...): (line 110)
        # Processing the call arguments (line 110)
        
        # Getting the type of 'self' (line 110)
        self_293413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 23), 'self', False)
        # Obtaining the member '_value' of a type (line 110)
        _value_293414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 23), self_293413, '_value')
        # Applying the 'usub' unary operator (line 110)
        result___neg___293415 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 22), 'usub', _value_293414)
        
        # Getting the type of 'self' (line 110)
        self_293416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 36), 'self', False)
        # Obtaining the member '_units' of a type (line 110)
        _units_293417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 36), self_293416, '_units')
        # Processing the call keyword arguments (line 110)
        kwargs_293418 = {}
        # Getting the type of 'UnitDbl' (line 110)
        UnitDbl_293412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 13), 'UnitDbl', False)
        # Calling UnitDbl(args, kwargs) (line 110)
        UnitDbl_call_result_293419 = invoke(stypy.reporting.localization.Localization(__file__, 110, 13), UnitDbl_293412, *[result___neg___293415, _units_293417], **kwargs_293418)
        
        # Assigning a type to the variable 'stypy_return_type' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 6), 'stypy_return_type', UnitDbl_call_result_293419)
        
        # ################# End of '__neg__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__neg__' in the type store
        # Getting the type of 'stypy_return_type' (line 108)
        stypy_return_type_293420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293420)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__neg__'
        return stypy_return_type_293420


    @norecursion
    def __nonzero__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__nonzero__'
        module_type_store = module_type_store.open_function_context('__nonzero__', 113, 3, False)
        # Assigning a type to the variable 'self' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        UnitDbl.__nonzero__.__dict__.__setitem__('stypy_localization', localization)
        UnitDbl.__nonzero__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnitDbl.__nonzero__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitDbl.__nonzero__.__dict__.__setitem__('stypy_function_name', 'UnitDbl.__nonzero__')
        UnitDbl.__nonzero__.__dict__.__setitem__('stypy_param_names_list', [])
        UnitDbl.__nonzero__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitDbl.__nonzero__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitDbl.__nonzero__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitDbl.__nonzero__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitDbl.__nonzero__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitDbl.__nonzero__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitDbl.__nonzero__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__nonzero__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__nonzero__(...)' code ##################

        unicode_293421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, (-1)), 'unicode', u'Test a UnitDbl for a non-zero value.\n\n      = RETURN VALUE\n      - Returns true if the value is non-zero.\n      ')
        
        # Getting the type of 'six' (line 119)
        six_293422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 9), 'six')
        # Obtaining the member 'PY3' of a type (line 119)
        PY3_293423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 9), six_293422, 'PY3')
        # Testing the type of an if condition (line 119)
        if_condition_293424 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 6), PY3_293423)
        # Assigning a type to the variable 'if_condition_293424' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 6), 'if_condition_293424', if_condition_293424)
        # SSA begins for if statement (line 119)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __bool__(...): (line 120)
        # Processing the call keyword arguments (line 120)
        kwargs_293428 = {}
        # Getting the type of 'self' (line 120)
        self_293425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 17), 'self', False)
        # Obtaining the member '_value' of a type (line 120)
        _value_293426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 17), self_293425, '_value')
        # Obtaining the member '__bool__' of a type (line 120)
        bool___293427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 17), _value_293426, '__bool__')
        # Calling __bool__(args, kwargs) (line 120)
        bool___call_result_293429 = invoke(stypy.reporting.localization.Localization(__file__, 120, 17), bool___293427, *[], **kwargs_293428)
        
        # Assigning a type to the variable 'stypy_return_type' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 10), 'stypy_return_type', bool___call_result_293429)
        # SSA branch for the else part of an if statement (line 119)
        module_type_store.open_ssa_branch('else')
        
        # Call to __nonzero__(...): (line 122)
        # Processing the call keyword arguments (line 122)
        kwargs_293433 = {}
        # Getting the type of 'self' (line 122)
        self_293430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 17), 'self', False)
        # Obtaining the member '_value' of a type (line 122)
        _value_293431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 17), self_293430, '_value')
        # Obtaining the member '__nonzero__' of a type (line 122)
        nonzero___293432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 17), _value_293431, '__nonzero__')
        # Calling __nonzero__(args, kwargs) (line 122)
        nonzero___call_result_293434 = invoke(stypy.reporting.localization.Localization(__file__, 122, 17), nonzero___293432, *[], **kwargs_293433)
        
        # Assigning a type to the variable 'stypy_return_type' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 10), 'stypy_return_type', nonzero___call_result_293434)
        # SSA join for if statement (line 119)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__nonzero__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__nonzero__' in the type store
        # Getting the type of 'stypy_return_type' (line 113)
        stypy_return_type_293435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293435)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__nonzero__'
        return stypy_return_type_293435


    @norecursion
    def stypy__cmp__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__cmp__'
        module_type_store = module_type_store.open_function_context('__cmp__', 128, 3, False)
        # Assigning a type to the variable 'self' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        UnitDbl.stypy__cmp__.__dict__.__setitem__('stypy_localization', localization)
        UnitDbl.stypy__cmp__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnitDbl.stypy__cmp__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitDbl.stypy__cmp__.__dict__.__setitem__('stypy_function_name', 'UnitDbl.stypy__cmp__')
        UnitDbl.stypy__cmp__.__dict__.__setitem__('stypy_param_names_list', ['rhs'])
        UnitDbl.stypy__cmp__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitDbl.stypy__cmp__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitDbl.stypy__cmp__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitDbl.stypy__cmp__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitDbl.stypy__cmp__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitDbl.stypy__cmp__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitDbl.stypy__cmp__', ['rhs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__cmp__', localization, ['rhs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__cmp__(...)' code ##################

        unicode_293436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, (-1)), 'unicode', u"Compare two UnitDbl's.\n\n      = ERROR CONDITIONS\n      - If the input rhs units are not the same as our units,\n        an error is thrown.\n\n      = INPUT VARIABLES\n      - rhs    The UnitDbl to compare against.\n\n      = RETURN VALUE\n      - Returns -1 if self < rhs, 0 if self == rhs, +1 if self > rhs.\n      ")
        
        # Call to checkSameUnits(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'rhs' (line 141)
        rhs_293439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 27), 'rhs', False)
        unicode_293440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 32), 'unicode', u'compare')
        # Processing the call keyword arguments (line 141)
        kwargs_293441 = {}
        # Getting the type of 'self' (line 141)
        self_293437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 6), 'self', False)
        # Obtaining the member 'checkSameUnits' of a type (line 141)
        checkSameUnits_293438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 6), self_293437, 'checkSameUnits')
        # Calling checkSameUnits(args, kwargs) (line 141)
        checkSameUnits_call_result_293442 = invoke(stypy.reporting.localization.Localization(__file__, 141, 6), checkSameUnits_293438, *[rhs_293439, unicode_293440], **kwargs_293441)
        
        
        # Call to cmp(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'self' (line 142)
        self_293444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 18), 'self', False)
        # Obtaining the member '_value' of a type (line 142)
        _value_293445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 18), self_293444, '_value')
        # Getting the type of 'rhs' (line 142)
        rhs_293446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 31), 'rhs', False)
        # Obtaining the member '_value' of a type (line 142)
        _value_293447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 31), rhs_293446, '_value')
        # Processing the call keyword arguments (line 142)
        kwargs_293448 = {}
        # Getting the type of 'cmp' (line 142)
        cmp_293443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 13), 'cmp', False)
        # Calling cmp(args, kwargs) (line 142)
        cmp_call_result_293449 = invoke(stypy.reporting.localization.Localization(__file__, 142, 13), cmp_293443, *[_value_293445, _value_293447], **kwargs_293448)
        
        # Assigning a type to the variable 'stypy_return_type' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 6), 'stypy_return_type', cmp_call_result_293449)
        
        # ################# End of '__cmp__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__cmp__' in the type store
        # Getting the type of 'stypy_return_type' (line 128)
        stypy_return_type_293450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293450)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__cmp__'
        return stypy_return_type_293450


    @norecursion
    def __add__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__add__'
        module_type_store = module_type_store.open_function_context('__add__', 145, 3, False)
        # Assigning a type to the variable 'self' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        UnitDbl.__add__.__dict__.__setitem__('stypy_localization', localization)
        UnitDbl.__add__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnitDbl.__add__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitDbl.__add__.__dict__.__setitem__('stypy_function_name', 'UnitDbl.__add__')
        UnitDbl.__add__.__dict__.__setitem__('stypy_param_names_list', ['rhs'])
        UnitDbl.__add__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitDbl.__add__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitDbl.__add__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitDbl.__add__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitDbl.__add__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitDbl.__add__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitDbl.__add__', ['rhs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__add__', localization, ['rhs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__add__(...)' code ##################

        unicode_293451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, (-1)), 'unicode', u"Add two UnitDbl's.\n\n      = ERROR CONDITIONS\n      - If the input rhs units are not the same as our units,\n        an error is thrown.\n\n      = INPUT VARIABLES\n      - rhs    The UnitDbl to add.\n\n      = RETURN VALUE\n      - Returns the sum of ourselves and the input UnitDbl.\n      ")
        
        # Call to checkSameUnits(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'rhs' (line 158)
        rhs_293454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 27), 'rhs', False)
        unicode_293455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 32), 'unicode', u'add')
        # Processing the call keyword arguments (line 158)
        kwargs_293456 = {}
        # Getting the type of 'self' (line 158)
        self_293452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 6), 'self', False)
        # Obtaining the member 'checkSameUnits' of a type (line 158)
        checkSameUnits_293453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 6), self_293452, 'checkSameUnits')
        # Calling checkSameUnits(args, kwargs) (line 158)
        checkSameUnits_call_result_293457 = invoke(stypy.reporting.localization.Localization(__file__, 158, 6), checkSameUnits_293453, *[rhs_293454, unicode_293455], **kwargs_293456)
        
        
        # Call to UnitDbl(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'self' (line 159)
        self_293459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 22), 'self', False)
        # Obtaining the member '_value' of a type (line 159)
        _value_293460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 22), self_293459, '_value')
        # Getting the type of 'rhs' (line 159)
        rhs_293461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 36), 'rhs', False)
        # Obtaining the member '_value' of a type (line 159)
        _value_293462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 36), rhs_293461, '_value')
        # Applying the binary operator '+' (line 159)
        result_add_293463 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 22), '+', _value_293460, _value_293462)
        
        # Getting the type of 'self' (line 159)
        self_293464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 48), 'self', False)
        # Obtaining the member '_units' of a type (line 159)
        _units_293465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 48), self_293464, '_units')
        # Processing the call keyword arguments (line 159)
        kwargs_293466 = {}
        # Getting the type of 'UnitDbl' (line 159)
        UnitDbl_293458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 13), 'UnitDbl', False)
        # Calling UnitDbl(args, kwargs) (line 159)
        UnitDbl_call_result_293467 = invoke(stypy.reporting.localization.Localization(__file__, 159, 13), UnitDbl_293458, *[result_add_293463, _units_293465], **kwargs_293466)
        
        # Assigning a type to the variable 'stypy_return_type' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 6), 'stypy_return_type', UnitDbl_call_result_293467)
        
        # ################# End of '__add__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__add__' in the type store
        # Getting the type of 'stypy_return_type' (line 145)
        stypy_return_type_293468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293468)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__add__'
        return stypy_return_type_293468


    @norecursion
    def __sub__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__sub__'
        module_type_store = module_type_store.open_function_context('__sub__', 162, 3, False)
        # Assigning a type to the variable 'self' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        UnitDbl.__sub__.__dict__.__setitem__('stypy_localization', localization)
        UnitDbl.__sub__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnitDbl.__sub__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitDbl.__sub__.__dict__.__setitem__('stypy_function_name', 'UnitDbl.__sub__')
        UnitDbl.__sub__.__dict__.__setitem__('stypy_param_names_list', ['rhs'])
        UnitDbl.__sub__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitDbl.__sub__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitDbl.__sub__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitDbl.__sub__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitDbl.__sub__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitDbl.__sub__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitDbl.__sub__', ['rhs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__sub__', localization, ['rhs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__sub__(...)' code ##################

        unicode_293469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, (-1)), 'unicode', u"Subtract two UnitDbl's.\n\n      = ERROR CONDITIONS\n      - If the input rhs units are not the same as our units,\n        an error is thrown.\n\n      = INPUT VARIABLES\n      - rhs    The UnitDbl to subtract.\n\n      = RETURN VALUE\n      - Returns the difference of ourselves and the input UnitDbl.\n      ")
        
        # Call to checkSameUnits(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'rhs' (line 175)
        rhs_293472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 27), 'rhs', False)
        unicode_293473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 32), 'unicode', u'subtract')
        # Processing the call keyword arguments (line 175)
        kwargs_293474 = {}
        # Getting the type of 'self' (line 175)
        self_293470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 6), 'self', False)
        # Obtaining the member 'checkSameUnits' of a type (line 175)
        checkSameUnits_293471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 6), self_293470, 'checkSameUnits')
        # Calling checkSameUnits(args, kwargs) (line 175)
        checkSameUnits_call_result_293475 = invoke(stypy.reporting.localization.Localization(__file__, 175, 6), checkSameUnits_293471, *[rhs_293472, unicode_293473], **kwargs_293474)
        
        
        # Call to UnitDbl(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'self' (line 176)
        self_293477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 22), 'self', False)
        # Obtaining the member '_value' of a type (line 176)
        _value_293478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 22), self_293477, '_value')
        # Getting the type of 'rhs' (line 176)
        rhs_293479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 36), 'rhs', False)
        # Obtaining the member '_value' of a type (line 176)
        _value_293480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 36), rhs_293479, '_value')
        # Applying the binary operator '-' (line 176)
        result_sub_293481 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 22), '-', _value_293478, _value_293480)
        
        # Getting the type of 'self' (line 176)
        self_293482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 48), 'self', False)
        # Obtaining the member '_units' of a type (line 176)
        _units_293483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 48), self_293482, '_units')
        # Processing the call keyword arguments (line 176)
        kwargs_293484 = {}
        # Getting the type of 'UnitDbl' (line 176)
        UnitDbl_293476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 13), 'UnitDbl', False)
        # Calling UnitDbl(args, kwargs) (line 176)
        UnitDbl_call_result_293485 = invoke(stypy.reporting.localization.Localization(__file__, 176, 13), UnitDbl_293476, *[result_sub_293481, _units_293483], **kwargs_293484)
        
        # Assigning a type to the variable 'stypy_return_type' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 6), 'stypy_return_type', UnitDbl_call_result_293485)
        
        # ################# End of '__sub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__sub__' in the type store
        # Getting the type of 'stypy_return_type' (line 162)
        stypy_return_type_293486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293486)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__sub__'
        return stypy_return_type_293486


    @norecursion
    def __mul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__mul__'
        module_type_store = module_type_store.open_function_context('__mul__', 179, 3, False)
        # Assigning a type to the variable 'self' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        UnitDbl.__mul__.__dict__.__setitem__('stypy_localization', localization)
        UnitDbl.__mul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnitDbl.__mul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitDbl.__mul__.__dict__.__setitem__('stypy_function_name', 'UnitDbl.__mul__')
        UnitDbl.__mul__.__dict__.__setitem__('stypy_param_names_list', ['rhs'])
        UnitDbl.__mul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitDbl.__mul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitDbl.__mul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitDbl.__mul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitDbl.__mul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitDbl.__mul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitDbl.__mul__', ['rhs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__mul__', localization, ['rhs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__mul__(...)' code ##################

        unicode_293487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, (-1)), 'unicode', u'Scale a UnitDbl by a value.\n\n      = INPUT VARIABLES\n      - rhs    The scalar to multiply by.\n\n      = RETURN VALUE\n      - Returns the scaled UnitDbl.\n      ')
        
        # Call to UnitDbl(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'self' (line 188)
        self_293489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 22), 'self', False)
        # Obtaining the member '_value' of a type (line 188)
        _value_293490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 22), self_293489, '_value')
        # Getting the type of 'rhs' (line 188)
        rhs_293491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 36), 'rhs', False)
        # Applying the binary operator '*' (line 188)
        result_mul_293492 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 22), '*', _value_293490, rhs_293491)
        
        # Getting the type of 'self' (line 188)
        self_293493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 41), 'self', False)
        # Obtaining the member '_units' of a type (line 188)
        _units_293494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 41), self_293493, '_units')
        # Processing the call keyword arguments (line 188)
        kwargs_293495 = {}
        # Getting the type of 'UnitDbl' (line 188)
        UnitDbl_293488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 13), 'UnitDbl', False)
        # Calling UnitDbl(args, kwargs) (line 188)
        UnitDbl_call_result_293496 = invoke(stypy.reporting.localization.Localization(__file__, 188, 13), UnitDbl_293488, *[result_mul_293492, _units_293494], **kwargs_293495)
        
        # Assigning a type to the variable 'stypy_return_type' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 6), 'stypy_return_type', UnitDbl_call_result_293496)
        
        # ################# End of '__mul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__mul__' in the type store
        # Getting the type of 'stypy_return_type' (line 179)
        stypy_return_type_293497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293497)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__mul__'
        return stypy_return_type_293497


    @norecursion
    def __rmul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rmul__'
        module_type_store = module_type_store.open_function_context('__rmul__', 191, 3, False)
        # Assigning a type to the variable 'self' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        UnitDbl.__rmul__.__dict__.__setitem__('stypy_localization', localization)
        UnitDbl.__rmul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnitDbl.__rmul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitDbl.__rmul__.__dict__.__setitem__('stypy_function_name', 'UnitDbl.__rmul__')
        UnitDbl.__rmul__.__dict__.__setitem__('stypy_param_names_list', ['lhs'])
        UnitDbl.__rmul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitDbl.__rmul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitDbl.__rmul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitDbl.__rmul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitDbl.__rmul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitDbl.__rmul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitDbl.__rmul__', ['lhs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rmul__', localization, ['lhs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rmul__(...)' code ##################

        unicode_293498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, (-1)), 'unicode', u'Scale a UnitDbl by a value.\n\n      = INPUT VARIABLES\n      - lhs    The scalar to multiply by.\n\n      = RETURN VALUE\n      - Returns the scaled UnitDbl.\n      ')
        
        # Call to UnitDbl(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'self' (line 200)
        self_293500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 22), 'self', False)
        # Obtaining the member '_value' of a type (line 200)
        _value_293501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 22), self_293500, '_value')
        # Getting the type of 'lhs' (line 200)
        lhs_293502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 36), 'lhs', False)
        # Applying the binary operator '*' (line 200)
        result_mul_293503 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 22), '*', _value_293501, lhs_293502)
        
        # Getting the type of 'self' (line 200)
        self_293504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 41), 'self', False)
        # Obtaining the member '_units' of a type (line 200)
        _units_293505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 41), self_293504, '_units')
        # Processing the call keyword arguments (line 200)
        kwargs_293506 = {}
        # Getting the type of 'UnitDbl' (line 200)
        UnitDbl_293499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 13), 'UnitDbl', False)
        # Calling UnitDbl(args, kwargs) (line 200)
        UnitDbl_call_result_293507 = invoke(stypy.reporting.localization.Localization(__file__, 200, 13), UnitDbl_293499, *[result_mul_293503, _units_293505], **kwargs_293506)
        
        # Assigning a type to the variable 'stypy_return_type' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 6), 'stypy_return_type', UnitDbl_call_result_293507)
        
        # ################# End of '__rmul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rmul__' in the type store
        # Getting the type of 'stypy_return_type' (line 191)
        stypy_return_type_293508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293508)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rmul__'
        return stypy_return_type_293508


    @norecursion
    def __div__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__div__'
        module_type_store = module_type_store.open_function_context('__div__', 203, 3, False)
        # Assigning a type to the variable 'self' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        UnitDbl.__div__.__dict__.__setitem__('stypy_localization', localization)
        UnitDbl.__div__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnitDbl.__div__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitDbl.__div__.__dict__.__setitem__('stypy_function_name', 'UnitDbl.__div__')
        UnitDbl.__div__.__dict__.__setitem__('stypy_param_names_list', ['rhs'])
        UnitDbl.__div__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitDbl.__div__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitDbl.__div__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitDbl.__div__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitDbl.__div__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitDbl.__div__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitDbl.__div__', ['rhs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__div__', localization, ['rhs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__div__(...)' code ##################

        unicode_293509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, (-1)), 'unicode', u'Divide a UnitDbl by a value.\n\n      = INPUT VARIABLES\n      - rhs    The scalar to divide by.\n\n      = RETURN VALUE\n      - Returns the scaled UnitDbl.\n      ')
        
        # Call to UnitDbl(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'self' (line 212)
        self_293511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 22), 'self', False)
        # Obtaining the member '_value' of a type (line 212)
        _value_293512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 22), self_293511, '_value')
        # Getting the type of 'rhs' (line 212)
        rhs_293513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 36), 'rhs', False)
        # Applying the binary operator 'div' (line 212)
        result_div_293514 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 22), 'div', _value_293512, rhs_293513)
        
        # Getting the type of 'self' (line 212)
        self_293515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 41), 'self', False)
        # Obtaining the member '_units' of a type (line 212)
        _units_293516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 41), self_293515, '_units')
        # Processing the call keyword arguments (line 212)
        kwargs_293517 = {}
        # Getting the type of 'UnitDbl' (line 212)
        UnitDbl_293510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 13), 'UnitDbl', False)
        # Calling UnitDbl(args, kwargs) (line 212)
        UnitDbl_call_result_293518 = invoke(stypy.reporting.localization.Localization(__file__, 212, 13), UnitDbl_293510, *[result_div_293514, _units_293516], **kwargs_293517)
        
        # Assigning a type to the variable 'stypy_return_type' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 6), 'stypy_return_type', UnitDbl_call_result_293518)
        
        # ################# End of '__div__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__div__' in the type store
        # Getting the type of 'stypy_return_type' (line 203)
        stypy_return_type_293519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293519)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__div__'
        return stypy_return_type_293519


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 215, 3, False)
        # Assigning a type to the variable 'self' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        UnitDbl.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        UnitDbl.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnitDbl.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitDbl.stypy__str__.__dict__.__setitem__('stypy_function_name', 'UnitDbl.stypy__str__')
        UnitDbl.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        UnitDbl.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitDbl.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitDbl.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitDbl.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitDbl.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitDbl.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitDbl.stypy__str__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__str__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__str__(...)' code ##################

        unicode_293520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 6), 'unicode', u'Print the UnitDbl.')
        unicode_293521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 13), 'unicode', u'%g *%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 217)
        tuple_293522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 217)
        # Adding element type (line 217)
        # Getting the type of 'self' (line 217)
        self_293523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 26), 'self')
        # Obtaining the member '_value' of a type (line 217)
        _value_293524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 26), self_293523, '_value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 26), tuple_293522, _value_293524)
        # Adding element type (line 217)
        # Getting the type of 'self' (line 217)
        self_293525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 39), 'self')
        # Obtaining the member '_units' of a type (line 217)
        _units_293526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 39), self_293525, '_units')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 26), tuple_293522, _units_293526)
        
        # Applying the binary operator '%' (line 217)
        result_mod_293527 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 13), '%', unicode_293521, tuple_293522)
        
        # Assigning a type to the variable 'stypy_return_type' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 6), 'stypy_return_type', result_mod_293527)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 215)
        stypy_return_type_293528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293528)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_293528


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 220, 3, False)
        # Assigning a type to the variable 'self' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        UnitDbl.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        UnitDbl.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnitDbl.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitDbl.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'UnitDbl.stypy__repr__')
        UnitDbl.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        UnitDbl.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitDbl.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitDbl.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitDbl.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitDbl.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitDbl.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitDbl.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        unicode_293529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 6), 'unicode', u'Print the UnitDbl.')
        unicode_293530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 13), 'unicode', u"UnitDbl( %g, '%s' )")
        
        # Obtaining an instance of the builtin type 'tuple' (line 222)
        tuple_293531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 222)
        # Adding element type (line 222)
        # Getting the type of 'self' (line 222)
        self_293532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 39), 'self')
        # Obtaining the member '_value' of a type (line 222)
        _value_293533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 39), self_293532, '_value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 39), tuple_293531, _value_293533)
        # Adding element type (line 222)
        # Getting the type of 'self' (line 222)
        self_293534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 52), 'self')
        # Obtaining the member '_units' of a type (line 222)
        _units_293535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 52), self_293534, '_units')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 39), tuple_293531, _units_293535)
        
        # Applying the binary operator '%' (line 222)
        result_mod_293536 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 13), '%', unicode_293530, tuple_293531)
        
        # Assigning a type to the variable 'stypy_return_type' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 6), 'stypy_return_type', result_mod_293536)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 220)
        stypy_return_type_293537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293537)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_293537


    @norecursion
    def type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'type'
        module_type_store = module_type_store.open_function_context('type', 225, 3, False)
        # Assigning a type to the variable 'self' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        UnitDbl.type.__dict__.__setitem__('stypy_localization', localization)
        UnitDbl.type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnitDbl.type.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitDbl.type.__dict__.__setitem__('stypy_function_name', 'UnitDbl.type')
        UnitDbl.type.__dict__.__setitem__('stypy_param_names_list', [])
        UnitDbl.type.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitDbl.type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitDbl.type.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitDbl.type.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitDbl.type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitDbl.type.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitDbl.type', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'type', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'type(...)' code ##################

        unicode_293538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 6), 'unicode', u'Return the type of UnitDbl data.')
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 227)
        self_293539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 26), 'self')
        # Obtaining the member '_units' of a type (line 227)
        _units_293540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 26), self_293539, '_units')
        # Getting the type of 'self' (line 227)
        self_293541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 13), 'self')
        # Obtaining the member '_types' of a type (line 227)
        _types_293542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 13), self_293541, '_types')
        # Obtaining the member '__getitem__' of a type (line 227)
        getitem___293543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 13), _types_293542, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 227)
        subscript_call_result_293544 = invoke(stypy.reporting.localization.Localization(__file__, 227, 13), getitem___293543, _units_293540)
        
        # Assigning a type to the variable 'stypy_return_type' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 6), 'stypy_return_type', subscript_call_result_293544)
        
        # ################# End of 'type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'type' in the type store
        # Getting the type of 'stypy_return_type' (line 225)
        stypy_return_type_293545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293545)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'type'
        return stypy_return_type_293545


    @norecursion
    def range(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 230)
        None_293546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 32), 'None')
        defaults = [None_293546]
        # Create a new context for function 'range'
        module_type_store = module_type_store.open_function_context('range', 230, 3, False)
        # Assigning a type to the variable 'self' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        UnitDbl.range.__dict__.__setitem__('stypy_localization', localization)
        UnitDbl.range.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnitDbl.range.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitDbl.range.__dict__.__setitem__('stypy_function_name', 'UnitDbl.range')
        UnitDbl.range.__dict__.__setitem__('stypy_param_names_list', ['stop', 'step'])
        UnitDbl.range.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitDbl.range.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitDbl.range.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitDbl.range.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitDbl.range.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitDbl.range.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitDbl.range', ['stop', 'step'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'range', localization, ['stop', 'step'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'range(...)' code ##################

        unicode_293547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, (-1)), 'unicode', u'Generate a range of UnitDbl objects.\n\n      Similar to the Python range() method.  Returns the range [\n      start, stop ) at the requested step.  Each element will be a\n      UnitDbl object.\n\n      = INPUT VARIABLES\n      - start    The starting value of the range.\n      - stop     The stop value of the range.\n      - step     Optional step to use.  If set to None, then a UnitDbl of\n                 value 1 w/ the units of the start is used.\n\n      = RETURN VALUE\n      - Returns a list contianing the requested UnitDbl values.\n      ')
        
        # Type idiom detected: calculating its left and rigth part (line 246)
        # Getting the type of 'step' (line 246)
        step_293548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 9), 'step')
        # Getting the type of 'None' (line 246)
        None_293549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 17), 'None')
        
        (may_be_293550, more_types_in_union_293551) = may_be_none(step_293548, None_293549)

        if may_be_293550:

            if more_types_in_union_293551:
                # Runtime conditional SSA (line 246)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 247):
            
            # Call to UnitDbl(...): (line 247)
            # Processing the call arguments (line 247)
            int_293553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 25), 'int')
            # Getting the type of 'start' (line 247)
            start_293554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 28), 'start', False)
            # Obtaining the member '_units' of a type (line 247)
            _units_293555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 28), start_293554, '_units')
            # Processing the call keyword arguments (line 247)
            kwargs_293556 = {}
            # Getting the type of 'UnitDbl' (line 247)
            UnitDbl_293552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'UnitDbl', False)
            # Calling UnitDbl(args, kwargs) (line 247)
            UnitDbl_call_result_293557 = invoke(stypy.reporting.localization.Localization(__file__, 247, 16), UnitDbl_293552, *[int_293553, _units_293555], **kwargs_293556)
            
            # Assigning a type to the variable 'step' (line 247)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 9), 'step', UnitDbl_call_result_293557)

            if more_types_in_union_293551:
                # SSA join for if statement (line 246)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a List to a Name (line 249):
        
        # Obtaining an instance of the builtin type 'list' (line 249)
        list_293558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 249)
        
        # Assigning a type to the variable 'elems' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 6), 'elems', list_293558)
        
        # Assigning a Num to a Name (line 251):
        int_293559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 10), 'int')
        # Assigning a type to the variable 'i' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 6), 'i', int_293559)
        
        # Getting the type of 'True' (line 252)
        True_293560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'True')
        # Testing the type of an if condition (line 252)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 252, 6), True_293560)
        # SSA begins for while statement (line 252)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a BinOp to a Name (line 253):
        # Getting the type of 'start' (line 253)
        start_293561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 13), 'start')
        # Getting the type of 'i' (line 253)
        i_293562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 21), 'i')
        # Getting the type of 'step' (line 253)
        step_293563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 25), 'step')
        # Applying the binary operator '*' (line 253)
        result_mul_293564 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 21), '*', i_293562, step_293563)
        
        # Applying the binary operator '+' (line 253)
        result_add_293565 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 13), '+', start_293561, result_mul_293564)
        
        # Assigning a type to the variable 'd' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 9), 'd', result_add_293565)
        
        
        # Getting the type of 'd' (line 254)
        d_293566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'd')
        # Getting the type of 'stop' (line 254)
        stop_293567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 17), 'stop')
        # Applying the binary operator '>=' (line 254)
        result_ge_293568 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 12), '>=', d_293566, stop_293567)
        
        # Testing the type of an if condition (line 254)
        if_condition_293569 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 254, 9), result_ge_293568)
        # Assigning a type to the variable 'if_condition_293569' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 9), 'if_condition_293569', if_condition_293569)
        # SSA begins for if statement (line 254)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 254)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'd' (line 257)
        d_293572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 23), 'd', False)
        # Processing the call keyword arguments (line 257)
        kwargs_293573 = {}
        # Getting the type of 'elems' (line 257)
        elems_293570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 9), 'elems', False)
        # Obtaining the member 'append' of a type (line 257)
        append_293571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 9), elems_293570, 'append')
        # Calling append(args, kwargs) (line 257)
        append_call_result_293574 = invoke(stypy.reporting.localization.Localization(__file__, 257, 9), append_293571, *[d_293572], **kwargs_293573)
        
        
        # Getting the type of 'i' (line 258)
        i_293575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 9), 'i')
        int_293576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 14), 'int')
        # Applying the binary operator '+=' (line 258)
        result_iadd_293577 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 9), '+=', i_293575, int_293576)
        # Assigning a type to the variable 'i' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 9), 'i', result_iadd_293577)
        
        # SSA join for while statement (line 252)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'elems' (line 260)
        elems_293578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 13), 'elems')
        # Assigning a type to the variable 'stypy_return_type' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 6), 'stypy_return_type', elems_293578)
        
        # ################# End of 'range(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'range' in the type store
        # Getting the type of 'stypy_return_type' (line 230)
        stypy_return_type_293579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293579)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'range'
        return stypy_return_type_293579


    @norecursion
    def checkUnits(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'checkUnits'
        module_type_store = module_type_store.open_function_context('checkUnits', 265, 3, False)
        # Assigning a type to the variable 'self' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        UnitDbl.checkUnits.__dict__.__setitem__('stypy_localization', localization)
        UnitDbl.checkUnits.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnitDbl.checkUnits.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitDbl.checkUnits.__dict__.__setitem__('stypy_function_name', 'UnitDbl.checkUnits')
        UnitDbl.checkUnits.__dict__.__setitem__('stypy_param_names_list', ['units'])
        UnitDbl.checkUnits.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitDbl.checkUnits.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitDbl.checkUnits.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitDbl.checkUnits.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitDbl.checkUnits.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitDbl.checkUnits.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitDbl.checkUnits', ['units'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'checkUnits', localization, ['units'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'checkUnits(...)' code ##################

        unicode_293580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, (-1)), 'unicode', u'Check to see if some units are valid.\n\n      = ERROR CONDITIONS\n      - If the input units are not in the allowed list, an error is thrown.\n\n      = INPUT VARIABLES\n      - units    The string name of the units to check.\n      ')
        
        
        # Getting the type of 'units' (line 274)
        units_293581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 9), 'units')
        # Getting the type of 'self' (line 274)
        self_293582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 22), 'self')
        # Obtaining the member 'allowed' of a type (line 274)
        allowed_293583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 22), self_293582, 'allowed')
        # Applying the binary operator 'notin' (line 274)
        result_contains_293584 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 9), 'notin', units_293581, allowed_293583)
        
        # Testing the type of an if condition (line 274)
        if_condition_293585 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 274, 6), result_contains_293584)
        # Assigning a type to the variable 'if_condition_293585' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 6), 'if_condition_293585', if_condition_293585)
        # SSA begins for if statement (line 274)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 275):
        unicode_293586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 15), 'unicode', u"Input units '%s' are not one of the supported types of %s")
        
        # Obtaining an instance of the builtin type 'tuple' (line 276)
        tuple_293587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 276)
        # Adding element type (line 276)
        # Getting the type of 'units' (line 276)
        units_293588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 19), 'units')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 19), tuple_293587, units_293588)
        # Adding element type (line 276)
        
        # Call to str(...): (line 276)
        # Processing the call arguments (line 276)
        
        # Call to list(...): (line 276)
        # Processing the call arguments (line 276)
        
        # Call to iterkeys(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'self' (line 276)
        self_293593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 49), 'self', False)
        # Obtaining the member 'allowed' of a type (line 276)
        allowed_293594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 49), self_293593, 'allowed')
        # Processing the call keyword arguments (line 276)
        kwargs_293595 = {}
        # Getting the type of 'six' (line 276)
        six_293591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 36), 'six', False)
        # Obtaining the member 'iterkeys' of a type (line 276)
        iterkeys_293592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 36), six_293591, 'iterkeys')
        # Calling iterkeys(args, kwargs) (line 276)
        iterkeys_call_result_293596 = invoke(stypy.reporting.localization.Localization(__file__, 276, 36), iterkeys_293592, *[allowed_293594], **kwargs_293595)
        
        # Processing the call keyword arguments (line 276)
        kwargs_293597 = {}
        # Getting the type of 'list' (line 276)
        list_293590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 31), 'list', False)
        # Calling list(args, kwargs) (line 276)
        list_call_result_293598 = invoke(stypy.reporting.localization.Localization(__file__, 276, 31), list_293590, *[iterkeys_call_result_293596], **kwargs_293597)
        
        # Processing the call keyword arguments (line 276)
        kwargs_293599 = {}
        # Getting the type of 'str' (line 276)
        str_293589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 26), 'str', False)
        # Calling str(args, kwargs) (line 276)
        str_call_result_293600 = invoke(stypy.reporting.localization.Localization(__file__, 276, 26), str_293589, *[list_call_result_293598], **kwargs_293599)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 19), tuple_293587, str_call_result_293600)
        
        # Applying the binary operator '%' (line 275)
        result_mod_293601 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 15), '%', unicode_293586, tuple_293587)
        
        # Assigning a type to the variable 'msg' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 9), 'msg', result_mod_293601)
        
        # Call to ValueError(...): (line 277)
        # Processing the call arguments (line 277)
        # Getting the type of 'msg' (line 277)
        msg_293603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 27), 'msg', False)
        # Processing the call keyword arguments (line 277)
        kwargs_293604 = {}
        # Getting the type of 'ValueError' (line 277)
        ValueError_293602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 15), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 277)
        ValueError_call_result_293605 = invoke(stypy.reporting.localization.Localization(__file__, 277, 15), ValueError_293602, *[msg_293603], **kwargs_293604)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 277, 9), ValueError_call_result_293605, 'raise parameter', BaseException)
        # SSA join for if statement (line 274)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'checkUnits(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'checkUnits' in the type store
        # Getting the type of 'stypy_return_type' (line 265)
        stypy_return_type_293606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293606)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'checkUnits'
        return stypy_return_type_293606


    @norecursion
    def checkSameUnits(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'checkSameUnits'
        module_type_store = module_type_store.open_function_context('checkSameUnits', 280, 3, False)
        # Assigning a type to the variable 'self' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        UnitDbl.checkSameUnits.__dict__.__setitem__('stypy_localization', localization)
        UnitDbl.checkSameUnits.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnitDbl.checkSameUnits.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitDbl.checkSameUnits.__dict__.__setitem__('stypy_function_name', 'UnitDbl.checkSameUnits')
        UnitDbl.checkSameUnits.__dict__.__setitem__('stypy_param_names_list', ['rhs', 'func'])
        UnitDbl.checkSameUnits.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitDbl.checkSameUnits.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitDbl.checkSameUnits.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitDbl.checkSameUnits.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitDbl.checkSameUnits.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitDbl.checkSameUnits.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitDbl.checkSameUnits', ['rhs', 'func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'checkSameUnits', localization, ['rhs', 'func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'checkSameUnits(...)' code ##################

        unicode_293607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, (-1)), 'unicode', u'Check to see if units are the same.\n\n      = ERROR CONDITIONS\n      - If the units of the rhs UnitDbl are not the same as our units,\n        an error is thrown.\n\n      = INPUT VARIABLES\n      - rhs    The UnitDbl to check for the same units\n      - func   The name of the function doing the check.\n      ')
        
        
        # Getting the type of 'self' (line 291)
        self_293608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 9), 'self')
        # Obtaining the member '_units' of a type (line 291)
        _units_293609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 9), self_293608, '_units')
        # Getting the type of 'rhs' (line 291)
        rhs_293610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 24), 'rhs')
        # Obtaining the member '_units' of a type (line 291)
        _units_293611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 24), rhs_293610, '_units')
        # Applying the binary operator '!=' (line 291)
        result_ne_293612 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 9), '!=', _units_293609, _units_293611)
        
        # Testing the type of an if condition (line 291)
        if_condition_293613 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 291, 6), result_ne_293612)
        # Assigning a type to the variable 'if_condition_293613' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 6), 'if_condition_293613', if_condition_293613)
        # SSA begins for if statement (line 291)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 292):
        unicode_293614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 15), 'unicode', u'Cannot %s units of different types.\nLHS: %s\nRHS: %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 294)
        tuple_293615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 294)
        # Adding element type (line 294)
        # Getting the type of 'func' (line 294)
        func_293616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 29), 'func')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 29), tuple_293615, func_293616)
        # Adding element type (line 294)
        # Getting the type of 'self' (line 294)
        self_293617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 35), 'self')
        # Obtaining the member '_units' of a type (line 294)
        _units_293618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 35), self_293617, '_units')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 29), tuple_293615, _units_293618)
        # Adding element type (line 294)
        # Getting the type of 'rhs' (line 294)
        rhs_293619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 48), 'rhs')
        # Obtaining the member '_units' of a type (line 294)
        _units_293620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 48), rhs_293619, '_units')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 29), tuple_293615, _units_293620)
        
        # Applying the binary operator '%' (line 292)
        result_mod_293621 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 15), '%', unicode_293614, tuple_293615)
        
        # Assigning a type to the variable 'msg' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 9), 'msg', result_mod_293621)
        
        # Call to ValueError(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'msg' (line 295)
        msg_293623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 27), 'msg', False)
        # Processing the call keyword arguments (line 295)
        kwargs_293624 = {}
        # Getting the type of 'ValueError' (line 295)
        ValueError_293622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 15), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 295)
        ValueError_call_result_293625 = invoke(stypy.reporting.localization.Localization(__file__, 295, 15), ValueError_293622, *[msg_293623], **kwargs_293624)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 295, 9), ValueError_call_result_293625, 'raise parameter', BaseException)
        # SSA join for if statement (line 291)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'checkSameUnits(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'checkSameUnits' in the type store
        # Getting the type of 'stypy_return_type' (line 280)
        stypy_return_type_293626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293626)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'checkSameUnits'
        return stypy_return_type_293626


# Assigning a type to the variable 'UnitDbl' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'UnitDbl', UnitDbl)

# Assigning a Dict to a Name (line 31):

# Obtaining an instance of the builtin type 'dict' (line 31)
dict_293627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 31)
# Adding element type (key, value) (line 31)
unicode_293628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 15), 'unicode', u'm')

# Obtaining an instance of the builtin type 'tuple' (line 32)
tuple_293629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 32)
# Adding element type (line 32)
float_293630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 23), tuple_293629, float_293630)
# Adding element type (line 32)
unicode_293631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 30), 'unicode', u'km')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 23), tuple_293629, unicode_293631)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 13), dict_293627, (unicode_293628, tuple_293629))
# Adding element type (key, value) (line 31)
unicode_293632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 15), 'unicode', u'km')

# Obtaining an instance of the builtin type 'tuple' (line 33)
tuple_293633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 33)
# Adding element type (line 33)
int_293634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 24), tuple_293633, int_293634)
# Adding element type (line 33)
unicode_293635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 27), 'unicode', u'km')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 24), tuple_293633, unicode_293635)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 13), dict_293627, (unicode_293632, tuple_293633))
# Adding element type (key, value) (line 31)
unicode_293636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 15), 'unicode', u'mile')

# Obtaining an instance of the builtin type 'tuple' (line 34)
tuple_293637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 26), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 34)
# Adding element type (line 34)
float_293638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 26), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 26), tuple_293637, float_293638)
# Adding element type (line 34)
unicode_293639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 36), 'unicode', u'km')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 26), tuple_293637, unicode_293639)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 13), dict_293627, (unicode_293636, tuple_293637))
# Adding element type (key, value) (line 31)
unicode_293640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 15), 'unicode', u'rad')

# Obtaining an instance of the builtin type 'tuple' (line 36)
tuple_293641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 25), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 36)
# Adding element type (line 36)
int_293642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 25), tuple_293641, int_293642)
# Adding element type (line 36)
unicode_293643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 28), 'unicode', u'rad')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 25), tuple_293641, unicode_293643)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 13), dict_293627, (unicode_293640, tuple_293641))
# Adding element type (key, value) (line 31)
unicode_293644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 15), 'unicode', u'deg')

# Obtaining an instance of the builtin type 'tuple' (line 37)
tuple_293645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 25), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 37)
# Adding element type (line 37)
float_293646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 25), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 25), tuple_293645, float_293646)
# Adding element type (line 37)
unicode_293647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 48), 'unicode', u'rad')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 25), tuple_293645, unicode_293647)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 13), dict_293627, (unicode_293644, tuple_293645))
# Adding element type (key, value) (line 31)
unicode_293648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 15), 'unicode', u'sec')

# Obtaining an instance of the builtin type 'tuple' (line 39)
tuple_293649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 25), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 39)
# Adding element type (line 39)
int_293650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 25), tuple_293649, int_293650)
# Adding element type (line 39)
unicode_293651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 28), 'unicode', u'sec')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 25), tuple_293649, unicode_293651)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 13), dict_293627, (unicode_293648, tuple_293649))
# Adding element type (key, value) (line 31)
unicode_293652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 15), 'unicode', u'min')

# Obtaining an instance of the builtin type 'tuple' (line 40)
tuple_293653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 25), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 40)
# Adding element type (line 40)
float_293654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 25), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 25), tuple_293653, float_293654)
# Adding element type (line 40)
unicode_293655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 31), 'unicode', u'sec')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 25), tuple_293653, unicode_293655)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 13), dict_293627, (unicode_293652, tuple_293653))
# Adding element type (key, value) (line 31)
unicode_293656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 15), 'unicode', u'hour')

# Obtaining an instance of the builtin type 'tuple' (line 41)
tuple_293657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 26), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 41)
# Adding element type (line 41)
int_293658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 26), tuple_293657, int_293658)
# Adding element type (line 41)
unicode_293659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 32), 'unicode', u'sec')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 26), tuple_293657, unicode_293659)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 13), dict_293627, (unicode_293656, tuple_293657))

# Getting the type of 'UnitDbl'
UnitDbl_293660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnitDbl')
# Setting the type of the member 'allowed' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnitDbl_293660, 'allowed', dict_293627)

# Assigning a Dict to a Name (line 44):

# Obtaining an instance of the builtin type 'dict' (line 44)
dict_293661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 44)
# Adding element type (key, value) (line 44)
unicode_293662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 14), 'unicode', u'km')
unicode_293663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 21), 'unicode', u'distance')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 12), dict_293661, (unicode_293662, unicode_293663))
# Adding element type (key, value) (line 44)
unicode_293664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 14), 'unicode', u'rad')
unicode_293665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'unicode', u'angle')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 12), dict_293661, (unicode_293664, unicode_293665))
# Adding element type (key, value) (line 44)
unicode_293666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 14), 'unicode', u'sec')
unicode_293667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 22), 'unicode', u'time')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 12), dict_293661, (unicode_293666, unicode_293667))

# Getting the type of 'UnitDbl'
UnitDbl_293668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnitDbl')
# Setting the type of the member '_types' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnitDbl_293668, '_types', dict_293661)

# Getting the type of 'six' (line 124)
six_293669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 6), 'six')
# Obtaining the member 'PY3' of a type (line 124)
PY3_293670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 6), six_293669, 'PY3')
# Testing the type of an if condition (line 124)
if_condition_293671 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 3), PY3_293670)
# Assigning a type to the variable 'if_condition_293671' (line 124)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 3), 'if_condition_293671', if_condition_293671)
# SSA begins for if statement (line 124)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 125):
# Getting the type of 'UnitDbl'
UnitDbl_293672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnitDbl')
# Obtaining the member '__nonzero__' of a type
nonzero___293673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnitDbl_293672, '__nonzero__')
# Assigning a type to the variable '__bool__' (line 125)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 6), '__bool__', nonzero___293673)
# SSA join for if statement (line 124)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Call to a Name (line 262):

# Call to staticmethod(...): (line 262)
# Processing the call arguments (line 262)
# Getting the type of 'UnitDbl'
UnitDbl_293675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnitDbl', False)
# Obtaining the member 'range' of a type
range_293676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnitDbl_293675, 'range')
# Processing the call keyword arguments (line 262)
kwargs_293677 = {}
# Getting the type of 'staticmethod' (line 262)
staticmethod_293674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 262)
staticmethod_call_result_293678 = invoke(stypy.reporting.localization.Localization(__file__, 262, 11), staticmethod_293674, *[range_293676], **kwargs_293677)

# Getting the type of 'UnitDbl'
UnitDbl_293679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnitDbl')
# Setting the type of the member 'range' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnitDbl_293679, 'range', staticmethod_call_result_293678)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
