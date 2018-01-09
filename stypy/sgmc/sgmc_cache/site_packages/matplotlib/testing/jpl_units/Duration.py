
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #===========================================================================
2: #
3: # Duration
4: #
5: #===========================================================================
6: 
7: 
8: '''Duration module.'''
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
21: #===========================================================================
22: class Duration(object):
23:    '''Class Duration in development.
24:    '''
25:    allowed = [ "ET", "UTC" ]
26: 
27:    #-----------------------------------------------------------------------
28:    def __init__( self, frame, seconds ):
29:       '''Create a new Duration object.
30: 
31:       = ERROR CONDITIONS
32:       - If the input frame is not in the allowed list, an error is thrown.
33: 
34:       = INPUT VARIABLES
35:       - frame    The frame of the duration.  Must be 'ET' or 'UTC'
36:       - seconds  The number of seconds in the Duration.
37:       '''
38:       if frame not in self.allowed:
39:          msg = "Input frame '%s' is not one of the supported frames of %s" \
40:                % ( frame, str( self.allowed ) )
41:          raise ValueError( msg )
42: 
43:       self._frame = frame
44:       self._seconds = seconds
45: 
46:    #-----------------------------------------------------------------------
47:    def frame( self ):
48:       '''Return the frame the duration is in.'''
49:       return self._frame
50: 
51:    #-----------------------------------------------------------------------
52:    def __abs__( self ):
53:       '''Return the absolute value of the duration.'''
54:       return Duration( self._frame, abs( self._seconds ) )
55: 
56:    #-----------------------------------------------------------------------
57:    def __neg__( self ):
58:       '''Return the negative value of this Duration.'''
59:       return Duration( self._frame, -self._seconds )
60: 
61:    #-----------------------------------------------------------------------
62:    def seconds( self ):
63:       '''Return the number of seconds in the Duration.'''
64:       return self._seconds
65: 
66:    #-----------------------------------------------------------------------
67:    def __nonzero__( self ):
68:       '''Compare two Durations.
69: 
70:       = INPUT VARIABLES
71:       - rhs    The Duration to compare against.
72: 
73:       = RETURN VALUE
74:       - Returns -1 if self < rhs, 0 if self == rhs, +1 if self > rhs.
75:       '''
76:       return self._seconds != 0
77: 
78:    if six.PY3:
79:       __bool__ = __nonzero__
80: 
81:    #-----------------------------------------------------------------------
82:    def __cmp__( self, rhs ):
83:       '''Compare two Durations.
84: 
85:       = ERROR CONDITIONS
86:       - If the input rhs is not in the same frame, an error is thrown.
87: 
88:       = INPUT VARIABLES
89:       - rhs    The Duration to compare against.
90: 
91:       = RETURN VALUE
92:       - Returns -1 if self < rhs, 0 if self == rhs, +1 if self > rhs.
93:       '''
94:       self.checkSameFrame( rhs, "compare" )
95:       return cmp( self._seconds, rhs._seconds )
96: 
97:    #-----------------------------------------------------------------------
98:    def __add__( self, rhs ):
99:       '''Add two Durations.
100: 
101:       = ERROR CONDITIONS
102:       - If the input rhs is not in the same frame, an error is thrown.
103: 
104:       = INPUT VARIABLES
105:       - rhs    The Duration to add.
106: 
107:       = RETURN VALUE
108:       - Returns the sum of ourselves and the input Duration.
109:       '''
110:       # Delay-load due to circular dependencies.
111:       import matplotlib.testing.jpl_units as U
112: 
113:       if isinstance( rhs, U.Epoch ):
114:          return rhs + self
115: 
116:       self.checkSameFrame( rhs, "add" )
117:       return Duration( self._frame, self._seconds + rhs._seconds )
118: 
119:    #-----------------------------------------------------------------------
120:    def __sub__( self, rhs ):
121:       '''Subtract two Durations.
122: 
123:       = ERROR CONDITIONS
124:       - If the input rhs is not in the same frame, an error is thrown.
125: 
126:       = INPUT VARIABLES
127:       - rhs    The Duration to subtract.
128: 
129:       = RETURN VALUE
130:       - Returns the difference of ourselves and the input Duration.
131:       '''
132:       self.checkSameFrame( rhs, "sub" )
133:       return Duration( self._frame, self._seconds - rhs._seconds )
134: 
135:    #-----------------------------------------------------------------------
136:    def __mul__( self, rhs ):
137:       '''Scale a UnitDbl by a value.
138: 
139:       = INPUT VARIABLES
140:       - rhs    The scalar to multiply by.
141: 
142:       = RETURN VALUE
143:       - Returns the scaled Duration.
144:       '''
145:       return Duration( self._frame, self._seconds * float( rhs ) )
146: 
147:    #-----------------------------------------------------------------------
148:    def __rmul__( self, lhs ):
149:       '''Scale a Duration by a value.
150: 
151:       = INPUT VARIABLES
152:       - lhs    The scalar to multiply by.
153: 
154:       = RETURN VALUE
155:       - Returns the scaled Duration.
156:       '''
157:       return Duration( self._frame, self._seconds * float( lhs ) )
158: 
159:    #-----------------------------------------------------------------------
160:    def __div__( self, rhs ):
161:       '''Divide a Duration by a value.
162: 
163:       = INPUT VARIABLES
164:       - rhs    The scalar to divide by.
165: 
166:       = RETURN VALUE
167:       - Returns the scaled Duration.
168:       '''
169:       return Duration( self._frame, self._seconds / float( rhs ) )
170: 
171:    #-----------------------------------------------------------------------
172:    def __rdiv__( self, rhs ):
173:       '''Divide a Duration by a value.
174: 
175:       = INPUT VARIABLES
176:       - rhs    The scalar to divide by.
177: 
178:       = RETURN VALUE
179:       - Returns the scaled Duration.
180:       '''
181:       return Duration( self._frame, float( rhs ) / self._seconds )
182: 
183:    #-----------------------------------------------------------------------
184:    def __str__( self ):
185:       '''Print the Duration.'''
186:       return "%g %s" % ( self._seconds, self._frame )
187: 
188:    #-----------------------------------------------------------------------
189:    def __repr__( self ):
190:       '''Print the Duration.'''
191:       return "Duration( '%s', %g )" % ( self._frame, self._seconds )
192: 
193:    #-----------------------------------------------------------------------
194:    def checkSameFrame( self, rhs, func ):
195:       '''Check to see if frames are the same.
196: 
197:       = ERROR CONDITIONS
198:       - If the frame of the rhs Duration is not the same as our frame,
199:         an error is thrown.
200: 
201:       = INPUT VARIABLES
202:       - rhs    The Duration to check for the same frame
203:       - func   The name of the function doing the check.
204:       '''
205:       if self._frame != rhs._frame:
206:          msg = "Cannot %s Duration's with different frames.\n" \
207:                "LHS: %s\n" \
208:                "RHS: %s" % ( func, self._frame, rhs._frame )
209:          raise ValueError( msg )
210: 
211: #===========================================================================
212: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_292235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 0), 'unicode', u'Duration module.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import six' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_292236 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six')

if (type(import_292236) is not StypyTypeError):

    if (import_292236 != 'pyd_module'):
        __import__(import_292236)
        sys_modules_292237 = sys.modules[import_292236]
        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', sys_modules_292237.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', import_292236)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

# Declaration of the 'Duration' class

class Duration(object, ):
    unicode_292238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, (-1)), 'unicode', u'Class Duration in development.\n   ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 28, 3, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Duration.__init__', ['frame', 'seconds'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['frame', 'seconds'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_292239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, (-1)), 'unicode', u"Create a new Duration object.\n\n      = ERROR CONDITIONS\n      - If the input frame is not in the allowed list, an error is thrown.\n\n      = INPUT VARIABLES\n      - frame    The frame of the duration.  Must be 'ET' or 'UTC'\n      - seconds  The number of seconds in the Duration.\n      ")
        
        
        # Getting the type of 'frame' (line 38)
        frame_292240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 9), 'frame')
        # Getting the type of 'self' (line 38)
        self_292241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 22), 'self')
        # Obtaining the member 'allowed' of a type (line 38)
        allowed_292242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 22), self_292241, 'allowed')
        # Applying the binary operator 'notin' (line 38)
        result_contains_292243 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 9), 'notin', frame_292240, allowed_292242)
        
        # Testing the type of an if condition (line 38)
        if_condition_292244 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 6), result_contains_292243)
        # Assigning a type to the variable 'if_condition_292244' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 6), 'if_condition_292244', if_condition_292244)
        # SSA begins for if statement (line 38)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 39):
        unicode_292245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 15), 'unicode', u"Input frame '%s' is not one of the supported frames of %s")
        
        # Obtaining an instance of the builtin type 'tuple' (line 40)
        tuple_292246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 40)
        # Adding element type (line 40)
        # Getting the type of 'frame' (line 40)
        frame_292247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'frame')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 19), tuple_292246, frame_292247)
        # Adding element type (line 40)
        
        # Call to str(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'self' (line 40)
        self_292249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 31), 'self', False)
        # Obtaining the member 'allowed' of a type (line 40)
        allowed_292250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 31), self_292249, 'allowed')
        # Processing the call keyword arguments (line 40)
        kwargs_292251 = {}
        # Getting the type of 'str' (line 40)
        str_292248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 26), 'str', False)
        # Calling str(args, kwargs) (line 40)
        str_call_result_292252 = invoke(stypy.reporting.localization.Localization(__file__, 40, 26), str_292248, *[allowed_292250], **kwargs_292251)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 19), tuple_292246, str_call_result_292252)
        
        # Applying the binary operator '%' (line 39)
        result_mod_292253 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 15), '%', unicode_292245, tuple_292246)
        
        # Assigning a type to the variable 'msg' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 9), 'msg', result_mod_292253)
        
        # Call to ValueError(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'msg' (line 41)
        msg_292255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 27), 'msg', False)
        # Processing the call keyword arguments (line 41)
        kwargs_292256 = {}
        # Getting the type of 'ValueError' (line 41)
        ValueError_292254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 41)
        ValueError_call_result_292257 = invoke(stypy.reporting.localization.Localization(__file__, 41, 15), ValueError_292254, *[msg_292255], **kwargs_292256)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 41, 9), ValueError_call_result_292257, 'raise parameter', BaseException)
        # SSA join for if statement (line 38)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 43):
        # Getting the type of 'frame' (line 43)
        frame_292258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 20), 'frame')
        # Getting the type of 'self' (line 43)
        self_292259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 6), 'self')
        # Setting the type of the member '_frame' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 6), self_292259, '_frame', frame_292258)
        
        # Assigning a Name to a Attribute (line 44):
        # Getting the type of 'seconds' (line 44)
        seconds_292260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'seconds')
        # Getting the type of 'self' (line 44)
        self_292261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 6), 'self')
        # Setting the type of the member '_seconds' of a type (line 44)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 6), self_292261, '_seconds', seconds_292260)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def frame(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'frame'
        module_type_store = module_type_store.open_function_context('frame', 47, 3, False)
        # Assigning a type to the variable 'self' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Duration.frame.__dict__.__setitem__('stypy_localization', localization)
        Duration.frame.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Duration.frame.__dict__.__setitem__('stypy_type_store', module_type_store)
        Duration.frame.__dict__.__setitem__('stypy_function_name', 'Duration.frame')
        Duration.frame.__dict__.__setitem__('stypy_param_names_list', [])
        Duration.frame.__dict__.__setitem__('stypy_varargs_param_name', None)
        Duration.frame.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Duration.frame.__dict__.__setitem__('stypy_call_defaults', defaults)
        Duration.frame.__dict__.__setitem__('stypy_call_varargs', varargs)
        Duration.frame.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Duration.frame.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Duration.frame', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'frame', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'frame(...)' code ##################

        unicode_292262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 6), 'unicode', u'Return the frame the duration is in.')
        # Getting the type of 'self' (line 49)
        self_292263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 13), 'self')
        # Obtaining the member '_frame' of a type (line 49)
        _frame_292264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 13), self_292263, '_frame')
        # Assigning a type to the variable 'stypy_return_type' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 6), 'stypy_return_type', _frame_292264)
        
        # ################# End of 'frame(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'frame' in the type store
        # Getting the type of 'stypy_return_type' (line 47)
        stypy_return_type_292265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292265)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'frame'
        return stypy_return_type_292265


    @norecursion
    def __abs__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__abs__'
        module_type_store = module_type_store.open_function_context('__abs__', 52, 3, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Duration.__abs__.__dict__.__setitem__('stypy_localization', localization)
        Duration.__abs__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Duration.__abs__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Duration.__abs__.__dict__.__setitem__('stypy_function_name', 'Duration.__abs__')
        Duration.__abs__.__dict__.__setitem__('stypy_param_names_list', [])
        Duration.__abs__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Duration.__abs__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Duration.__abs__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Duration.__abs__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Duration.__abs__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Duration.__abs__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Duration.__abs__', [], None, None, defaults, varargs, kwargs)

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

        unicode_292266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 6), 'unicode', u'Return the absolute value of the duration.')
        
        # Call to Duration(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'self' (line 54)
        self_292268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 23), 'self', False)
        # Obtaining the member '_frame' of a type (line 54)
        _frame_292269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 23), self_292268, '_frame')
        
        # Call to abs(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'self' (line 54)
        self_292271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 41), 'self', False)
        # Obtaining the member '_seconds' of a type (line 54)
        _seconds_292272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 41), self_292271, '_seconds')
        # Processing the call keyword arguments (line 54)
        kwargs_292273 = {}
        # Getting the type of 'abs' (line 54)
        abs_292270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 36), 'abs', False)
        # Calling abs(args, kwargs) (line 54)
        abs_call_result_292274 = invoke(stypy.reporting.localization.Localization(__file__, 54, 36), abs_292270, *[_seconds_292272], **kwargs_292273)
        
        # Processing the call keyword arguments (line 54)
        kwargs_292275 = {}
        # Getting the type of 'Duration' (line 54)
        Duration_292267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 13), 'Duration', False)
        # Calling Duration(args, kwargs) (line 54)
        Duration_call_result_292276 = invoke(stypy.reporting.localization.Localization(__file__, 54, 13), Duration_292267, *[_frame_292269, abs_call_result_292274], **kwargs_292275)
        
        # Assigning a type to the variable 'stypy_return_type' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 6), 'stypy_return_type', Duration_call_result_292276)
        
        # ################# End of '__abs__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__abs__' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_292277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292277)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__abs__'
        return stypy_return_type_292277


    @norecursion
    def __neg__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__neg__'
        module_type_store = module_type_store.open_function_context('__neg__', 57, 3, False)
        # Assigning a type to the variable 'self' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Duration.__neg__.__dict__.__setitem__('stypy_localization', localization)
        Duration.__neg__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Duration.__neg__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Duration.__neg__.__dict__.__setitem__('stypy_function_name', 'Duration.__neg__')
        Duration.__neg__.__dict__.__setitem__('stypy_param_names_list', [])
        Duration.__neg__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Duration.__neg__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Duration.__neg__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Duration.__neg__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Duration.__neg__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Duration.__neg__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Duration.__neg__', [], None, None, defaults, varargs, kwargs)

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

        unicode_292278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 6), 'unicode', u'Return the negative value of this Duration.')
        
        # Call to Duration(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'self' (line 59)
        self_292280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 23), 'self', False)
        # Obtaining the member '_frame' of a type (line 59)
        _frame_292281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 23), self_292280, '_frame')
        
        # Getting the type of 'self' (line 59)
        self_292282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 37), 'self', False)
        # Obtaining the member '_seconds' of a type (line 59)
        _seconds_292283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 37), self_292282, '_seconds')
        # Applying the 'usub' unary operator (line 59)
        result___neg___292284 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 36), 'usub', _seconds_292283)
        
        # Processing the call keyword arguments (line 59)
        kwargs_292285 = {}
        # Getting the type of 'Duration' (line 59)
        Duration_292279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 13), 'Duration', False)
        # Calling Duration(args, kwargs) (line 59)
        Duration_call_result_292286 = invoke(stypy.reporting.localization.Localization(__file__, 59, 13), Duration_292279, *[_frame_292281, result___neg___292284], **kwargs_292285)
        
        # Assigning a type to the variable 'stypy_return_type' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 6), 'stypy_return_type', Duration_call_result_292286)
        
        # ################# End of '__neg__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__neg__' in the type store
        # Getting the type of 'stypy_return_type' (line 57)
        stypy_return_type_292287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292287)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__neg__'
        return stypy_return_type_292287


    @norecursion
    def seconds(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'seconds'
        module_type_store = module_type_store.open_function_context('seconds', 62, 3, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Duration.seconds.__dict__.__setitem__('stypy_localization', localization)
        Duration.seconds.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Duration.seconds.__dict__.__setitem__('stypy_type_store', module_type_store)
        Duration.seconds.__dict__.__setitem__('stypy_function_name', 'Duration.seconds')
        Duration.seconds.__dict__.__setitem__('stypy_param_names_list', [])
        Duration.seconds.__dict__.__setitem__('stypy_varargs_param_name', None)
        Duration.seconds.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Duration.seconds.__dict__.__setitem__('stypy_call_defaults', defaults)
        Duration.seconds.__dict__.__setitem__('stypy_call_varargs', varargs)
        Duration.seconds.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Duration.seconds.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Duration.seconds', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'seconds', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'seconds(...)' code ##################

        unicode_292288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 6), 'unicode', u'Return the number of seconds in the Duration.')
        # Getting the type of 'self' (line 64)
        self_292289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 13), 'self')
        # Obtaining the member '_seconds' of a type (line 64)
        _seconds_292290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 13), self_292289, '_seconds')
        # Assigning a type to the variable 'stypy_return_type' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 6), 'stypy_return_type', _seconds_292290)
        
        # ################# End of 'seconds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'seconds' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_292291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292291)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'seconds'
        return stypy_return_type_292291


    @norecursion
    def __nonzero__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__nonzero__'
        module_type_store = module_type_store.open_function_context('__nonzero__', 67, 3, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Duration.__nonzero__.__dict__.__setitem__('stypy_localization', localization)
        Duration.__nonzero__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Duration.__nonzero__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Duration.__nonzero__.__dict__.__setitem__('stypy_function_name', 'Duration.__nonzero__')
        Duration.__nonzero__.__dict__.__setitem__('stypy_param_names_list', [])
        Duration.__nonzero__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Duration.__nonzero__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Duration.__nonzero__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Duration.__nonzero__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Duration.__nonzero__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Duration.__nonzero__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Duration.__nonzero__', [], None, None, defaults, varargs, kwargs)

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

        unicode_292292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, (-1)), 'unicode', u'Compare two Durations.\n\n      = INPUT VARIABLES\n      - rhs    The Duration to compare against.\n\n      = RETURN VALUE\n      - Returns -1 if self < rhs, 0 if self == rhs, +1 if self > rhs.\n      ')
        
        # Getting the type of 'self' (line 76)
        self_292293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 13), 'self')
        # Obtaining the member '_seconds' of a type (line 76)
        _seconds_292294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 13), self_292293, '_seconds')
        int_292295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 30), 'int')
        # Applying the binary operator '!=' (line 76)
        result_ne_292296 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 13), '!=', _seconds_292294, int_292295)
        
        # Assigning a type to the variable 'stypy_return_type' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 6), 'stypy_return_type', result_ne_292296)
        
        # ################# End of '__nonzero__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__nonzero__' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_292297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292297)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__nonzero__'
        return stypy_return_type_292297


    @norecursion
    def stypy__cmp__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__cmp__'
        module_type_store = module_type_store.open_function_context('__cmp__', 82, 3, False)
        # Assigning a type to the variable 'self' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Duration.stypy__cmp__.__dict__.__setitem__('stypy_localization', localization)
        Duration.stypy__cmp__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Duration.stypy__cmp__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Duration.stypy__cmp__.__dict__.__setitem__('stypy_function_name', 'Duration.stypy__cmp__')
        Duration.stypy__cmp__.__dict__.__setitem__('stypy_param_names_list', ['rhs'])
        Duration.stypy__cmp__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Duration.stypy__cmp__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Duration.stypy__cmp__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Duration.stypy__cmp__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Duration.stypy__cmp__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Duration.stypy__cmp__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Duration.stypy__cmp__', ['rhs'], None, None, defaults, varargs, kwargs)

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

        unicode_292298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, (-1)), 'unicode', u'Compare two Durations.\n\n      = ERROR CONDITIONS\n      - If the input rhs is not in the same frame, an error is thrown.\n\n      = INPUT VARIABLES\n      - rhs    The Duration to compare against.\n\n      = RETURN VALUE\n      - Returns -1 if self < rhs, 0 if self == rhs, +1 if self > rhs.\n      ')
        
        # Call to checkSameFrame(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'rhs' (line 94)
        rhs_292301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 27), 'rhs', False)
        unicode_292302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 32), 'unicode', u'compare')
        # Processing the call keyword arguments (line 94)
        kwargs_292303 = {}
        # Getting the type of 'self' (line 94)
        self_292299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 6), 'self', False)
        # Obtaining the member 'checkSameFrame' of a type (line 94)
        checkSameFrame_292300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 6), self_292299, 'checkSameFrame')
        # Calling checkSameFrame(args, kwargs) (line 94)
        checkSameFrame_call_result_292304 = invoke(stypy.reporting.localization.Localization(__file__, 94, 6), checkSameFrame_292300, *[rhs_292301, unicode_292302], **kwargs_292303)
        
        
        # Call to cmp(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'self' (line 95)
        self_292306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 18), 'self', False)
        # Obtaining the member '_seconds' of a type (line 95)
        _seconds_292307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 18), self_292306, '_seconds')
        # Getting the type of 'rhs' (line 95)
        rhs_292308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 33), 'rhs', False)
        # Obtaining the member '_seconds' of a type (line 95)
        _seconds_292309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 33), rhs_292308, '_seconds')
        # Processing the call keyword arguments (line 95)
        kwargs_292310 = {}
        # Getting the type of 'cmp' (line 95)
        cmp_292305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 13), 'cmp', False)
        # Calling cmp(args, kwargs) (line 95)
        cmp_call_result_292311 = invoke(stypy.reporting.localization.Localization(__file__, 95, 13), cmp_292305, *[_seconds_292307, _seconds_292309], **kwargs_292310)
        
        # Assigning a type to the variable 'stypy_return_type' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 6), 'stypy_return_type', cmp_call_result_292311)
        
        # ################# End of '__cmp__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__cmp__' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_292312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292312)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__cmp__'
        return stypy_return_type_292312


    @norecursion
    def __add__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__add__'
        module_type_store = module_type_store.open_function_context('__add__', 98, 3, False)
        # Assigning a type to the variable 'self' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Duration.__add__.__dict__.__setitem__('stypy_localization', localization)
        Duration.__add__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Duration.__add__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Duration.__add__.__dict__.__setitem__('stypy_function_name', 'Duration.__add__')
        Duration.__add__.__dict__.__setitem__('stypy_param_names_list', ['rhs'])
        Duration.__add__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Duration.__add__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Duration.__add__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Duration.__add__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Duration.__add__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Duration.__add__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Duration.__add__', ['rhs'], None, None, defaults, varargs, kwargs)

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

        unicode_292313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, (-1)), 'unicode', u'Add two Durations.\n\n      = ERROR CONDITIONS\n      - If the input rhs is not in the same frame, an error is thrown.\n\n      = INPUT VARIABLES\n      - rhs    The Duration to add.\n\n      = RETURN VALUE\n      - Returns the sum of ourselves and the input Duration.\n      ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 111, 6))
        
        # 'import matplotlib.testing.jpl_units' statement (line 111)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
        import_292314 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 111, 6), 'matplotlib.testing.jpl_units')

        if (type(import_292314) is not StypyTypeError):

            if (import_292314 != 'pyd_module'):
                __import__(import_292314)
                sys_modules_292315 = sys.modules[import_292314]
                import_module(stypy.reporting.localization.Localization(__file__, 111, 6), 'U', sys_modules_292315.module_type_store, module_type_store)
            else:
                import matplotlib.testing.jpl_units as U

                import_module(stypy.reporting.localization.Localization(__file__, 111, 6), 'U', matplotlib.testing.jpl_units, module_type_store)

        else:
            # Assigning a type to the variable 'matplotlib.testing.jpl_units' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 6), 'matplotlib.testing.jpl_units', import_292314)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
        
        
        
        # Call to isinstance(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'rhs' (line 113)
        rhs_292317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 21), 'rhs', False)
        # Getting the type of 'U' (line 113)
        U_292318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 26), 'U', False)
        # Obtaining the member 'Epoch' of a type (line 113)
        Epoch_292319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 26), U_292318, 'Epoch')
        # Processing the call keyword arguments (line 113)
        kwargs_292320 = {}
        # Getting the type of 'isinstance' (line 113)
        isinstance_292316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 9), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 113)
        isinstance_call_result_292321 = invoke(stypy.reporting.localization.Localization(__file__, 113, 9), isinstance_292316, *[rhs_292317, Epoch_292319], **kwargs_292320)
        
        # Testing the type of an if condition (line 113)
        if_condition_292322 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 6), isinstance_call_result_292321)
        # Assigning a type to the variable 'if_condition_292322' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 6), 'if_condition_292322', if_condition_292322)
        # SSA begins for if statement (line 113)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'rhs' (line 114)
        rhs_292323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'rhs')
        # Getting the type of 'self' (line 114)
        self_292324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 22), 'self')
        # Applying the binary operator '+' (line 114)
        result_add_292325 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 16), '+', rhs_292323, self_292324)
        
        # Assigning a type to the variable 'stypy_return_type' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 9), 'stypy_return_type', result_add_292325)
        # SSA join for if statement (line 113)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to checkSameFrame(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'rhs' (line 116)
        rhs_292328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 27), 'rhs', False)
        unicode_292329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 32), 'unicode', u'add')
        # Processing the call keyword arguments (line 116)
        kwargs_292330 = {}
        # Getting the type of 'self' (line 116)
        self_292326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 6), 'self', False)
        # Obtaining the member 'checkSameFrame' of a type (line 116)
        checkSameFrame_292327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 6), self_292326, 'checkSameFrame')
        # Calling checkSameFrame(args, kwargs) (line 116)
        checkSameFrame_call_result_292331 = invoke(stypy.reporting.localization.Localization(__file__, 116, 6), checkSameFrame_292327, *[rhs_292328, unicode_292329], **kwargs_292330)
        
        
        # Call to Duration(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'self' (line 117)
        self_292333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 23), 'self', False)
        # Obtaining the member '_frame' of a type (line 117)
        _frame_292334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 23), self_292333, '_frame')
        # Getting the type of 'self' (line 117)
        self_292335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 36), 'self', False)
        # Obtaining the member '_seconds' of a type (line 117)
        _seconds_292336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 36), self_292335, '_seconds')
        # Getting the type of 'rhs' (line 117)
        rhs_292337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 52), 'rhs', False)
        # Obtaining the member '_seconds' of a type (line 117)
        _seconds_292338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 52), rhs_292337, '_seconds')
        # Applying the binary operator '+' (line 117)
        result_add_292339 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 36), '+', _seconds_292336, _seconds_292338)
        
        # Processing the call keyword arguments (line 117)
        kwargs_292340 = {}
        # Getting the type of 'Duration' (line 117)
        Duration_292332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 13), 'Duration', False)
        # Calling Duration(args, kwargs) (line 117)
        Duration_call_result_292341 = invoke(stypy.reporting.localization.Localization(__file__, 117, 13), Duration_292332, *[_frame_292334, result_add_292339], **kwargs_292340)
        
        # Assigning a type to the variable 'stypy_return_type' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 6), 'stypy_return_type', Duration_call_result_292341)
        
        # ################# End of '__add__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__add__' in the type store
        # Getting the type of 'stypy_return_type' (line 98)
        stypy_return_type_292342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292342)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__add__'
        return stypy_return_type_292342


    @norecursion
    def __sub__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__sub__'
        module_type_store = module_type_store.open_function_context('__sub__', 120, 3, False)
        # Assigning a type to the variable 'self' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Duration.__sub__.__dict__.__setitem__('stypy_localization', localization)
        Duration.__sub__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Duration.__sub__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Duration.__sub__.__dict__.__setitem__('stypy_function_name', 'Duration.__sub__')
        Duration.__sub__.__dict__.__setitem__('stypy_param_names_list', ['rhs'])
        Duration.__sub__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Duration.__sub__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Duration.__sub__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Duration.__sub__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Duration.__sub__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Duration.__sub__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Duration.__sub__', ['rhs'], None, None, defaults, varargs, kwargs)

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

        unicode_292343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, (-1)), 'unicode', u'Subtract two Durations.\n\n      = ERROR CONDITIONS\n      - If the input rhs is not in the same frame, an error is thrown.\n\n      = INPUT VARIABLES\n      - rhs    The Duration to subtract.\n\n      = RETURN VALUE\n      - Returns the difference of ourselves and the input Duration.\n      ')
        
        # Call to checkSameFrame(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'rhs' (line 132)
        rhs_292346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 27), 'rhs', False)
        unicode_292347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 32), 'unicode', u'sub')
        # Processing the call keyword arguments (line 132)
        kwargs_292348 = {}
        # Getting the type of 'self' (line 132)
        self_292344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 6), 'self', False)
        # Obtaining the member 'checkSameFrame' of a type (line 132)
        checkSameFrame_292345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 6), self_292344, 'checkSameFrame')
        # Calling checkSameFrame(args, kwargs) (line 132)
        checkSameFrame_call_result_292349 = invoke(stypy.reporting.localization.Localization(__file__, 132, 6), checkSameFrame_292345, *[rhs_292346, unicode_292347], **kwargs_292348)
        
        
        # Call to Duration(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'self' (line 133)
        self_292351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'self', False)
        # Obtaining the member '_frame' of a type (line 133)
        _frame_292352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 23), self_292351, '_frame')
        # Getting the type of 'self' (line 133)
        self_292353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 36), 'self', False)
        # Obtaining the member '_seconds' of a type (line 133)
        _seconds_292354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 36), self_292353, '_seconds')
        # Getting the type of 'rhs' (line 133)
        rhs_292355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 52), 'rhs', False)
        # Obtaining the member '_seconds' of a type (line 133)
        _seconds_292356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 52), rhs_292355, '_seconds')
        # Applying the binary operator '-' (line 133)
        result_sub_292357 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 36), '-', _seconds_292354, _seconds_292356)
        
        # Processing the call keyword arguments (line 133)
        kwargs_292358 = {}
        # Getting the type of 'Duration' (line 133)
        Duration_292350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 13), 'Duration', False)
        # Calling Duration(args, kwargs) (line 133)
        Duration_call_result_292359 = invoke(stypy.reporting.localization.Localization(__file__, 133, 13), Duration_292350, *[_frame_292352, result_sub_292357], **kwargs_292358)
        
        # Assigning a type to the variable 'stypy_return_type' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 6), 'stypy_return_type', Duration_call_result_292359)
        
        # ################# End of '__sub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__sub__' in the type store
        # Getting the type of 'stypy_return_type' (line 120)
        stypy_return_type_292360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292360)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__sub__'
        return stypy_return_type_292360


    @norecursion
    def __mul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__mul__'
        module_type_store = module_type_store.open_function_context('__mul__', 136, 3, False)
        # Assigning a type to the variable 'self' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Duration.__mul__.__dict__.__setitem__('stypy_localization', localization)
        Duration.__mul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Duration.__mul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Duration.__mul__.__dict__.__setitem__('stypy_function_name', 'Duration.__mul__')
        Duration.__mul__.__dict__.__setitem__('stypy_param_names_list', ['rhs'])
        Duration.__mul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Duration.__mul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Duration.__mul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Duration.__mul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Duration.__mul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Duration.__mul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Duration.__mul__', ['rhs'], None, None, defaults, varargs, kwargs)

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

        unicode_292361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, (-1)), 'unicode', u'Scale a UnitDbl by a value.\n\n      = INPUT VARIABLES\n      - rhs    The scalar to multiply by.\n\n      = RETURN VALUE\n      - Returns the scaled Duration.\n      ')
        
        # Call to Duration(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'self' (line 145)
        self_292363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 23), 'self', False)
        # Obtaining the member '_frame' of a type (line 145)
        _frame_292364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 23), self_292363, '_frame')
        # Getting the type of 'self' (line 145)
        self_292365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 36), 'self', False)
        # Obtaining the member '_seconds' of a type (line 145)
        _seconds_292366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 36), self_292365, '_seconds')
        
        # Call to float(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'rhs' (line 145)
        rhs_292368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 59), 'rhs', False)
        # Processing the call keyword arguments (line 145)
        kwargs_292369 = {}
        # Getting the type of 'float' (line 145)
        float_292367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 52), 'float', False)
        # Calling float(args, kwargs) (line 145)
        float_call_result_292370 = invoke(stypy.reporting.localization.Localization(__file__, 145, 52), float_292367, *[rhs_292368], **kwargs_292369)
        
        # Applying the binary operator '*' (line 145)
        result_mul_292371 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 36), '*', _seconds_292366, float_call_result_292370)
        
        # Processing the call keyword arguments (line 145)
        kwargs_292372 = {}
        # Getting the type of 'Duration' (line 145)
        Duration_292362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 13), 'Duration', False)
        # Calling Duration(args, kwargs) (line 145)
        Duration_call_result_292373 = invoke(stypy.reporting.localization.Localization(__file__, 145, 13), Duration_292362, *[_frame_292364, result_mul_292371], **kwargs_292372)
        
        # Assigning a type to the variable 'stypy_return_type' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 6), 'stypy_return_type', Duration_call_result_292373)
        
        # ################# End of '__mul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__mul__' in the type store
        # Getting the type of 'stypy_return_type' (line 136)
        stypy_return_type_292374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292374)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__mul__'
        return stypy_return_type_292374


    @norecursion
    def __rmul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rmul__'
        module_type_store = module_type_store.open_function_context('__rmul__', 148, 3, False)
        # Assigning a type to the variable 'self' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Duration.__rmul__.__dict__.__setitem__('stypy_localization', localization)
        Duration.__rmul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Duration.__rmul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Duration.__rmul__.__dict__.__setitem__('stypy_function_name', 'Duration.__rmul__')
        Duration.__rmul__.__dict__.__setitem__('stypy_param_names_list', ['lhs'])
        Duration.__rmul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Duration.__rmul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Duration.__rmul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Duration.__rmul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Duration.__rmul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Duration.__rmul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Duration.__rmul__', ['lhs'], None, None, defaults, varargs, kwargs)

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

        unicode_292375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, (-1)), 'unicode', u'Scale a Duration by a value.\n\n      = INPUT VARIABLES\n      - lhs    The scalar to multiply by.\n\n      = RETURN VALUE\n      - Returns the scaled Duration.\n      ')
        
        # Call to Duration(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'self' (line 157)
        self_292377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 23), 'self', False)
        # Obtaining the member '_frame' of a type (line 157)
        _frame_292378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 23), self_292377, '_frame')
        # Getting the type of 'self' (line 157)
        self_292379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 36), 'self', False)
        # Obtaining the member '_seconds' of a type (line 157)
        _seconds_292380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 36), self_292379, '_seconds')
        
        # Call to float(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'lhs' (line 157)
        lhs_292382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 59), 'lhs', False)
        # Processing the call keyword arguments (line 157)
        kwargs_292383 = {}
        # Getting the type of 'float' (line 157)
        float_292381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 52), 'float', False)
        # Calling float(args, kwargs) (line 157)
        float_call_result_292384 = invoke(stypy.reporting.localization.Localization(__file__, 157, 52), float_292381, *[lhs_292382], **kwargs_292383)
        
        # Applying the binary operator '*' (line 157)
        result_mul_292385 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 36), '*', _seconds_292380, float_call_result_292384)
        
        # Processing the call keyword arguments (line 157)
        kwargs_292386 = {}
        # Getting the type of 'Duration' (line 157)
        Duration_292376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 13), 'Duration', False)
        # Calling Duration(args, kwargs) (line 157)
        Duration_call_result_292387 = invoke(stypy.reporting.localization.Localization(__file__, 157, 13), Duration_292376, *[_frame_292378, result_mul_292385], **kwargs_292386)
        
        # Assigning a type to the variable 'stypy_return_type' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 6), 'stypy_return_type', Duration_call_result_292387)
        
        # ################# End of '__rmul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rmul__' in the type store
        # Getting the type of 'stypy_return_type' (line 148)
        stypy_return_type_292388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292388)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rmul__'
        return stypy_return_type_292388


    @norecursion
    def __div__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__div__'
        module_type_store = module_type_store.open_function_context('__div__', 160, 3, False)
        # Assigning a type to the variable 'self' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Duration.__div__.__dict__.__setitem__('stypy_localization', localization)
        Duration.__div__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Duration.__div__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Duration.__div__.__dict__.__setitem__('stypy_function_name', 'Duration.__div__')
        Duration.__div__.__dict__.__setitem__('stypy_param_names_list', ['rhs'])
        Duration.__div__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Duration.__div__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Duration.__div__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Duration.__div__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Duration.__div__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Duration.__div__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Duration.__div__', ['rhs'], None, None, defaults, varargs, kwargs)

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

        unicode_292389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, (-1)), 'unicode', u'Divide a Duration by a value.\n\n      = INPUT VARIABLES\n      - rhs    The scalar to divide by.\n\n      = RETURN VALUE\n      - Returns the scaled Duration.\n      ')
        
        # Call to Duration(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'self' (line 169)
        self_292391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 23), 'self', False)
        # Obtaining the member '_frame' of a type (line 169)
        _frame_292392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 23), self_292391, '_frame')
        # Getting the type of 'self' (line 169)
        self_292393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 36), 'self', False)
        # Obtaining the member '_seconds' of a type (line 169)
        _seconds_292394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 36), self_292393, '_seconds')
        
        # Call to float(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'rhs' (line 169)
        rhs_292396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 59), 'rhs', False)
        # Processing the call keyword arguments (line 169)
        kwargs_292397 = {}
        # Getting the type of 'float' (line 169)
        float_292395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 52), 'float', False)
        # Calling float(args, kwargs) (line 169)
        float_call_result_292398 = invoke(stypy.reporting.localization.Localization(__file__, 169, 52), float_292395, *[rhs_292396], **kwargs_292397)
        
        # Applying the binary operator 'div' (line 169)
        result_div_292399 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 36), 'div', _seconds_292394, float_call_result_292398)
        
        # Processing the call keyword arguments (line 169)
        kwargs_292400 = {}
        # Getting the type of 'Duration' (line 169)
        Duration_292390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 13), 'Duration', False)
        # Calling Duration(args, kwargs) (line 169)
        Duration_call_result_292401 = invoke(stypy.reporting.localization.Localization(__file__, 169, 13), Duration_292390, *[_frame_292392, result_div_292399], **kwargs_292400)
        
        # Assigning a type to the variable 'stypy_return_type' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 6), 'stypy_return_type', Duration_call_result_292401)
        
        # ################# End of '__div__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__div__' in the type store
        # Getting the type of 'stypy_return_type' (line 160)
        stypy_return_type_292402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292402)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__div__'
        return stypy_return_type_292402


    @norecursion
    def __rdiv__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rdiv__'
        module_type_store = module_type_store.open_function_context('__rdiv__', 172, 3, False)
        # Assigning a type to the variable 'self' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Duration.__rdiv__.__dict__.__setitem__('stypy_localization', localization)
        Duration.__rdiv__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Duration.__rdiv__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Duration.__rdiv__.__dict__.__setitem__('stypy_function_name', 'Duration.__rdiv__')
        Duration.__rdiv__.__dict__.__setitem__('stypy_param_names_list', ['rhs'])
        Duration.__rdiv__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Duration.__rdiv__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Duration.__rdiv__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Duration.__rdiv__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Duration.__rdiv__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Duration.__rdiv__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Duration.__rdiv__', ['rhs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rdiv__', localization, ['rhs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rdiv__(...)' code ##################

        unicode_292403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, (-1)), 'unicode', u'Divide a Duration by a value.\n\n      = INPUT VARIABLES\n      - rhs    The scalar to divide by.\n\n      = RETURN VALUE\n      - Returns the scaled Duration.\n      ')
        
        # Call to Duration(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'self' (line 181)
        self_292405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 23), 'self', False)
        # Obtaining the member '_frame' of a type (line 181)
        _frame_292406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 23), self_292405, '_frame')
        
        # Call to float(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'rhs' (line 181)
        rhs_292408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 43), 'rhs', False)
        # Processing the call keyword arguments (line 181)
        kwargs_292409 = {}
        # Getting the type of 'float' (line 181)
        float_292407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 36), 'float', False)
        # Calling float(args, kwargs) (line 181)
        float_call_result_292410 = invoke(stypy.reporting.localization.Localization(__file__, 181, 36), float_292407, *[rhs_292408], **kwargs_292409)
        
        # Getting the type of 'self' (line 181)
        self_292411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 51), 'self', False)
        # Obtaining the member '_seconds' of a type (line 181)
        _seconds_292412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 51), self_292411, '_seconds')
        # Applying the binary operator 'div' (line 181)
        result_div_292413 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 36), 'div', float_call_result_292410, _seconds_292412)
        
        # Processing the call keyword arguments (line 181)
        kwargs_292414 = {}
        # Getting the type of 'Duration' (line 181)
        Duration_292404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 13), 'Duration', False)
        # Calling Duration(args, kwargs) (line 181)
        Duration_call_result_292415 = invoke(stypy.reporting.localization.Localization(__file__, 181, 13), Duration_292404, *[_frame_292406, result_div_292413], **kwargs_292414)
        
        # Assigning a type to the variable 'stypy_return_type' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 6), 'stypy_return_type', Duration_call_result_292415)
        
        # ################# End of '__rdiv__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rdiv__' in the type store
        # Getting the type of 'stypy_return_type' (line 172)
        stypy_return_type_292416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292416)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rdiv__'
        return stypy_return_type_292416


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 184, 3, False)
        # Assigning a type to the variable 'self' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Duration.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        Duration.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Duration.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Duration.stypy__str__.__dict__.__setitem__('stypy_function_name', 'Duration.stypy__str__')
        Duration.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        Duration.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Duration.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Duration.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Duration.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Duration.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Duration.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Duration.stypy__str__', [], None, None, defaults, varargs, kwargs)

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

        unicode_292417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 6), 'unicode', u'Print the Duration.')
        unicode_292418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 13), 'unicode', u'%g %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 186)
        tuple_292419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 186)
        # Adding element type (line 186)
        # Getting the type of 'self' (line 186)
        self_292420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), 'self')
        # Obtaining the member '_seconds' of a type (line 186)
        _seconds_292421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 25), self_292420, '_seconds')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 25), tuple_292419, _seconds_292421)
        # Adding element type (line 186)
        # Getting the type of 'self' (line 186)
        self_292422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 40), 'self')
        # Obtaining the member '_frame' of a type (line 186)
        _frame_292423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 40), self_292422, '_frame')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 25), tuple_292419, _frame_292423)
        
        # Applying the binary operator '%' (line 186)
        result_mod_292424 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 13), '%', unicode_292418, tuple_292419)
        
        # Assigning a type to the variable 'stypy_return_type' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 6), 'stypy_return_type', result_mod_292424)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 184)
        stypy_return_type_292425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292425)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_292425


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 189, 3, False)
        # Assigning a type to the variable 'self' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Duration.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        Duration.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Duration.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Duration.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Duration.stypy__repr__')
        Duration.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        Duration.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Duration.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Duration.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Duration.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Duration.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Duration.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Duration.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        unicode_292426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 6), 'unicode', u'Print the Duration.')
        unicode_292427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 13), 'unicode', u"Duration( '%s', %g )")
        
        # Obtaining an instance of the builtin type 'tuple' (line 191)
        tuple_292428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 191)
        # Adding element type (line 191)
        # Getting the type of 'self' (line 191)
        self_292429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 40), 'self')
        # Obtaining the member '_frame' of a type (line 191)
        _frame_292430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 40), self_292429, '_frame')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 40), tuple_292428, _frame_292430)
        # Adding element type (line 191)
        # Getting the type of 'self' (line 191)
        self_292431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 53), 'self')
        # Obtaining the member '_seconds' of a type (line 191)
        _seconds_292432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 53), self_292431, '_seconds')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 40), tuple_292428, _seconds_292432)
        
        # Applying the binary operator '%' (line 191)
        result_mod_292433 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 13), '%', unicode_292427, tuple_292428)
        
        # Assigning a type to the variable 'stypy_return_type' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 6), 'stypy_return_type', result_mod_292433)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 189)
        stypy_return_type_292434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292434)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_292434


    @norecursion
    def checkSameFrame(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'checkSameFrame'
        module_type_store = module_type_store.open_function_context('checkSameFrame', 194, 3, False)
        # Assigning a type to the variable 'self' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Duration.checkSameFrame.__dict__.__setitem__('stypy_localization', localization)
        Duration.checkSameFrame.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Duration.checkSameFrame.__dict__.__setitem__('stypy_type_store', module_type_store)
        Duration.checkSameFrame.__dict__.__setitem__('stypy_function_name', 'Duration.checkSameFrame')
        Duration.checkSameFrame.__dict__.__setitem__('stypy_param_names_list', ['rhs', 'func'])
        Duration.checkSameFrame.__dict__.__setitem__('stypy_varargs_param_name', None)
        Duration.checkSameFrame.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Duration.checkSameFrame.__dict__.__setitem__('stypy_call_defaults', defaults)
        Duration.checkSameFrame.__dict__.__setitem__('stypy_call_varargs', varargs)
        Duration.checkSameFrame.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Duration.checkSameFrame.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Duration.checkSameFrame', ['rhs', 'func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'checkSameFrame', localization, ['rhs', 'func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'checkSameFrame(...)' code ##################

        unicode_292435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, (-1)), 'unicode', u'Check to see if frames are the same.\n\n      = ERROR CONDITIONS\n      - If the frame of the rhs Duration is not the same as our frame,\n        an error is thrown.\n\n      = INPUT VARIABLES\n      - rhs    The Duration to check for the same frame\n      - func   The name of the function doing the check.\n      ')
        
        
        # Getting the type of 'self' (line 205)
        self_292436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 9), 'self')
        # Obtaining the member '_frame' of a type (line 205)
        _frame_292437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 9), self_292436, '_frame')
        # Getting the type of 'rhs' (line 205)
        rhs_292438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 24), 'rhs')
        # Obtaining the member '_frame' of a type (line 205)
        _frame_292439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 24), rhs_292438, '_frame')
        # Applying the binary operator '!=' (line 205)
        result_ne_292440 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 9), '!=', _frame_292437, _frame_292439)
        
        # Testing the type of an if condition (line 205)
        if_condition_292441 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 6), result_ne_292440)
        # Assigning a type to the variable 'if_condition_292441' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 6), 'if_condition_292441', if_condition_292441)
        # SSA begins for if statement (line 205)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 206):
        unicode_292442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 15), 'unicode', u"Cannot %s Duration's with different frames.\nLHS: %s\nRHS: %s")
        
        # Obtaining an instance of the builtin type 'tuple' (line 208)
        tuple_292443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 208)
        # Adding element type (line 208)
        # Getting the type of 'func' (line 208)
        func_292444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 29), 'func')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 29), tuple_292443, func_292444)
        # Adding element type (line 208)
        # Getting the type of 'self' (line 208)
        self_292445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 35), 'self')
        # Obtaining the member '_frame' of a type (line 208)
        _frame_292446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 35), self_292445, '_frame')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 29), tuple_292443, _frame_292446)
        # Adding element type (line 208)
        # Getting the type of 'rhs' (line 208)
        rhs_292447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 48), 'rhs')
        # Obtaining the member '_frame' of a type (line 208)
        _frame_292448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 48), rhs_292447, '_frame')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 29), tuple_292443, _frame_292448)
        
        # Applying the binary operator '%' (line 206)
        result_mod_292449 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 15), '%', unicode_292442, tuple_292443)
        
        # Assigning a type to the variable 'msg' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 9), 'msg', result_mod_292449)
        
        # Call to ValueError(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'msg' (line 209)
        msg_292451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 27), 'msg', False)
        # Processing the call keyword arguments (line 209)
        kwargs_292452 = {}
        # Getting the type of 'ValueError' (line 209)
        ValueError_292450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 15), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 209)
        ValueError_call_result_292453 = invoke(stypy.reporting.localization.Localization(__file__, 209, 15), ValueError_292450, *[msg_292451], **kwargs_292452)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 209, 9), ValueError_call_result_292453, 'raise parameter', BaseException)
        # SSA join for if statement (line 205)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'checkSameFrame(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'checkSameFrame' in the type store
        # Getting the type of 'stypy_return_type' (line 194)
        stypy_return_type_292454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292454)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'checkSameFrame'
        return stypy_return_type_292454


# Assigning a type to the variable 'Duration' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'Duration', Duration)

# Assigning a List to a Name (line 25):

# Obtaining an instance of the builtin type 'list' (line 25)
list_292455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)
unicode_292456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 15), 'unicode', u'ET')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 13), list_292455, unicode_292456)
# Adding element type (line 25)
unicode_292457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 21), 'unicode', u'UTC')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 13), list_292455, unicode_292457)

# Getting the type of 'Duration'
Duration_292458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Duration')
# Setting the type of the member 'allowed' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Duration_292458, 'allowed', list_292455)

# Getting the type of 'six' (line 78)
six_292459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 6), 'six')
# Obtaining the member 'PY3' of a type (line 78)
PY3_292460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 6), six_292459, 'PY3')
# Testing the type of an if condition (line 78)
if_condition_292461 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 3), PY3_292460)
# Assigning a type to the variable 'if_condition_292461' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 3), 'if_condition_292461', if_condition_292461)
# SSA begins for if statement (line 78)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 79):
# Getting the type of 'Duration'
Duration_292462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Duration')
# Obtaining the member '__nonzero__' of a type
nonzero___292463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Duration_292462, '__nonzero__')
# Assigning a type to the variable '__bool__' (line 79)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 6), '__bool__', nonzero___292463)
# SSA join for if statement (line 78)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
