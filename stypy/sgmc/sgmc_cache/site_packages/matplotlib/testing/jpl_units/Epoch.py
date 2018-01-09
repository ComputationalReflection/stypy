
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #===========================================================================
2: #
3: # Epoch
4: #
5: #===========================================================================
6: 
7: 
8: '''Epoch module.'''
9: 
10: #===========================================================================
11: # Place all imports after here.
12: #
13: from __future__ import (absolute_import, division, print_function,
14:                         unicode_literals)
15: 
16: import six
17: 
18: import math
19: import datetime as DT
20: from matplotlib.dates import date2num
21: #
22: # Place all imports before here.
23: #===========================================================================
24: 
25: #===========================================================================
26: class Epoch(object):
27:    # Frame conversion offsets in seconds
28:    # t(TO) = t(FROM) + allowed[ FROM ][ TO ]
29:    allowed = {
30:       "ET" : {
31:          "UTC" : +64.1839,
32:          },
33:       "UTC" : {
34:          "ET" : -64.1839,
35:          },
36:       }
37: 
38:    #-----------------------------------------------------------------------
39:    def __init__( self, frame, sec=None, jd=None, daynum=None, dt=None ):
40:       '''Create a new Epoch object.
41: 
42:       Build an epoch 1 of 2 ways:
43: 
44:       Using seconds past a Julian date:
45:       #   Epoch( 'ET', sec=1e8, jd=2451545 )
46: 
47:       or using a matplotlib day number
48:       #   Epoch( 'ET', daynum=730119.5 )
49: 
50: 
51:       = ERROR CONDITIONS
52:       - If the input units are not in the allowed list, an error is thrown.
53: 
54:       = INPUT VARIABLES
55:       - frame    The frame of the epoch.  Must be 'ET' or 'UTC'
56:       - sec      The number of seconds past the input JD.
57:       - jd       The Julian date of the epoch.
58:       - daynum   The matplotlib day number of the epoch.
59:       - dt       A python datetime instance.
60:       '''
61:       if ( ( sec is None and jd is not None ) or
62:            ( sec is not None and jd is None ) or
63:            ( daynum is not None and ( sec is not None or jd is not None ) ) or
64:            ( daynum is None and dt is None and ( sec is None or jd is None ) ) or
65:            ( daynum is not None and dt is not None ) or
66:            ( dt is not None and ( sec is not None or jd is not None ) ) or
67:            ( (dt is not None) and not isinstance(dt, DT.datetime) ) ):
68:          msg = "Invalid inputs.  Must enter sec and jd together, " \
69:                "daynum by itself, or dt (must be a python datetime).\n" \
70:                "Sec = %s\nJD  = %s\ndnum= %s\ndt  = %s" \
71:                % ( str( sec ), str( jd ), str( daynum ), str( dt ) )
72:          raise ValueError( msg )
73: 
74:       if frame not in self.allowed:
75:          msg = "Input frame '%s' is not one of the supported frames of %s" \
76:                % ( frame, str( list(six.iterkeys(self.allowed) ) ) )
77:          raise ValueError(msg)
78: 
79:       self._frame = frame
80: 
81:       if dt is not None:
82:          daynum = date2num( dt )
83: 
84:       if daynum is not None:
85:          # 1-JAN-0001 in JD = 1721425.5
86:          jd = float( daynum ) + 1721425.5
87:          self._jd = math.floor( jd )
88:          self._seconds = ( jd - self._jd ) * 86400.0
89: 
90:       else:
91:          self._seconds = float( sec )
92:          self._jd = float( jd )
93: 
94:          # Resolve seconds down to [ 0, 86400 )
95:          deltaDays = int( math.floor( self._seconds / 86400.0 ) )
96:          self._jd += deltaDays
97:          self._seconds -= deltaDays * 86400.0
98: 
99:    #-----------------------------------------------------------------------
100:    def convert( self, frame ):
101:       if self._frame == frame:
102:          return self
103: 
104:       offset = self.allowed[ self._frame ][ frame ]
105: 
106:       return Epoch( frame, self._seconds + offset, self._jd )
107: 
108:    #-----------------------------------------------------------------------
109:    def frame( self ):
110:       return self._frame
111: 
112:    #-----------------------------------------------------------------------
113:    def julianDate( self, frame ):
114:       t = self
115:       if frame != self._frame:
116:          t = self.convert( frame )
117: 
118:       return t._jd + t._seconds / 86400.0
119: 
120:    #-----------------------------------------------------------------------
121:    def secondsPast( self, frame, jd ):
122:       t = self
123:       if frame != self._frame:
124:          t = self.convert( frame )
125: 
126:       delta = t._jd - jd
127:       return t._seconds + delta * 86400
128: 
129:    #-----------------------------------------------------------------------
130:    def __cmp__( self, rhs ):
131:       '''Compare two Epoch's.
132: 
133:       = INPUT VARIABLES
134:       - rhs    The Epoch to compare against.
135: 
136:       = RETURN VALUE
137:       - Returns -1 if self < rhs, 0 if self == rhs, +1 if self > rhs.
138:       '''
139:       t = self
140:       if self._frame != rhs._frame:
141:          t = self.convert( rhs._frame )
142: 
143:       if t._jd != rhs._jd:
144:          return cmp( t._jd, rhs._jd )
145: 
146:       return cmp( t._seconds, rhs._seconds )
147: 
148:    #-----------------------------------------------------------------------
149:    def __add__( self, rhs ):
150:       '''Add a duration to an Epoch.
151: 
152:       = INPUT VARIABLES
153:       - rhs    The Epoch to subtract.
154: 
155:       = RETURN VALUE
156:       - Returns the difference of ourselves and the input Epoch.
157:       '''
158:       t = self
159:       if self._frame != rhs.frame():
160:          t = self.convert( rhs._frame )
161: 
162:       sec = t._seconds + rhs.seconds()
163: 
164:       return Epoch( t._frame, sec, t._jd )
165: 
166:    #-----------------------------------------------------------------------
167:    def __sub__( self, rhs ):
168:       '''Subtract two Epoch's or a Duration from an Epoch.
169: 
170:       Valid:
171:       Duration = Epoch - Epoch
172:       Epoch = Epoch - Duration
173: 
174:       = INPUT VARIABLES
175:       - rhs    The Epoch to subtract.
176: 
177:       = RETURN VALUE
178:       - Returns either the duration between to Epoch's or the a new
179:         Epoch that is the result of subtracting a duration from an epoch.
180:       '''
181:       # Delay-load due to circular dependencies.
182:       import matplotlib.testing.jpl_units as U
183: 
184:       # Handle Epoch - Duration
185:       if isinstance( rhs, U.Duration ):
186:          return self + -rhs
187: 
188:       t = self
189:       if self._frame != rhs._frame:
190:          t = self.convert( rhs._frame )
191: 
192:       days = t._jd - rhs._jd
193:       sec = t._seconds - rhs._seconds
194: 
195:       return U.Duration( rhs._frame, days*86400 + sec )
196: 
197:    #-----------------------------------------------------------------------
198:    def __str__( self ):
199:       '''Print the Epoch.'''
200:       return "%22.15e %s" % ( self.julianDate( self._frame ), self._frame )
201: 
202:    #-----------------------------------------------------------------------
203:    def __repr__( self ):
204:       '''Print the Epoch.'''
205:       return str( self )
206: 
207:    #-----------------------------------------------------------------------
208:    def range( start, stop, step ):
209:       '''Generate a range of Epoch objects.
210: 
211:       Similar to the Python range() method.  Returns the range [
212:       start, stop ) at the requested step.  Each element will be a
213:       Epoch object.
214: 
215:       = INPUT VARIABLES
216:       - start    The starting value of the range.
217:       - stop     The stop value of the range.
218:       - step     Step to use.
219: 
220:       = RETURN VALUE
221:       - Returns a list contianing the requested Epoch values.
222:       '''
223:       elems = []
224: 
225:       i = 0
226:       while True:
227:          d = start + i * step
228:          if d >= stop:
229:             break
230: 
231:          elems.append( d )
232:          i += 1
233: 
234:       return elems
235: 
236:    range = staticmethod( range )
237: 
238: #===========================================================================
239: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_292464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 0), 'unicode', u'Epoch module.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import six' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_292465 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six')

if (type(import_292465) is not StypyTypeError):

    if (import_292465 != 'pyd_module'):
        __import__(import_292465)
        sys_modules_292466 = sys.modules[import_292465]
        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', sys_modules_292466.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', import_292465)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import math' statement (line 18)
import math

import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'math', math, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import datetime' statement (line 19)
import datetime as DT

import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'DT', DT, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from matplotlib.dates import date2num' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_292467 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.dates')

if (type(import_292467) is not StypyTypeError):

    if (import_292467 != 'pyd_module'):
        __import__(import_292467)
        sys_modules_292468 = sys.modules[import_292467]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.dates', sys_modules_292468.module_type_store, module_type_store, ['date2num'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_292468, sys_modules_292468.module_type_store, module_type_store)
    else:
        from matplotlib.dates import date2num

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.dates', None, module_type_store, ['date2num'], [date2num])

else:
    # Assigning a type to the variable 'matplotlib.dates' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.dates', import_292467)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

# Declaration of the 'Epoch' class

class Epoch(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 39)
        None_292469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 34), 'None')
        # Getting the type of 'None' (line 39)
        None_292470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 43), 'None')
        # Getting the type of 'None' (line 39)
        None_292471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 56), 'None')
        # Getting the type of 'None' (line 39)
        None_292472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 65), 'None')
        defaults = [None_292469, None_292470, None_292471, None_292472]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 39, 3, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Epoch.__init__', ['frame', 'sec', 'jd', 'daynum', 'dt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['frame', 'sec', 'jd', 'daynum', 'dt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_292473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, (-1)), 'unicode', u"Create a new Epoch object.\n\n      Build an epoch 1 of 2 ways:\n\n      Using seconds past a Julian date:\n      #   Epoch( 'ET', sec=1e8, jd=2451545 )\n\n      or using a matplotlib day number\n      #   Epoch( 'ET', daynum=730119.5 )\n\n\n      = ERROR CONDITIONS\n      - If the input units are not in the allowed list, an error is thrown.\n\n      = INPUT VARIABLES\n      - frame    The frame of the epoch.  Must be 'ET' or 'UTC'\n      - sec      The number of seconds past the input JD.\n      - jd       The Julian date of the epoch.\n      - daynum   The matplotlib day number of the epoch.\n      - dt       A python datetime instance.\n      ")
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        
        # Getting the type of 'sec' (line 61)
        sec_292474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 13), 'sec')
        # Getting the type of 'None' (line 61)
        None_292475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'None')
        # Applying the binary operator 'is' (line 61)
        result_is__292476 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 13), 'is', sec_292474, None_292475)
        
        
        # Getting the type of 'jd' (line 61)
        jd_292477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 29), 'jd')
        # Getting the type of 'None' (line 61)
        None_292478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 39), 'None')
        # Applying the binary operator 'isnot' (line 61)
        result_is_not_292479 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 29), 'isnot', jd_292477, None_292478)
        
        # Applying the binary operator 'and' (line 61)
        result_and_keyword_292480 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 13), 'and', result_is__292476, result_is_not_292479)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'sec' (line 62)
        sec_292481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 13), 'sec')
        # Getting the type of 'None' (line 62)
        None_292482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 24), 'None')
        # Applying the binary operator 'isnot' (line 62)
        result_is_not_292483 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 13), 'isnot', sec_292481, None_292482)
        
        
        # Getting the type of 'jd' (line 62)
        jd_292484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 33), 'jd')
        # Getting the type of 'None' (line 62)
        None_292485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 39), 'None')
        # Applying the binary operator 'is' (line 62)
        result_is__292486 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 33), 'is', jd_292484, None_292485)
        
        # Applying the binary operator 'and' (line 62)
        result_and_keyword_292487 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 13), 'and', result_is_not_292483, result_is__292486)
        
        # Applying the binary operator 'or' (line 61)
        result_or_keyword_292488 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 11), 'or', result_and_keyword_292480, result_and_keyword_292487)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'daynum' (line 63)
        daynum_292489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 13), 'daynum')
        # Getting the type of 'None' (line 63)
        None_292490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 27), 'None')
        # Applying the binary operator 'isnot' (line 63)
        result_is_not_292491 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 13), 'isnot', daynum_292489, None_292490)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'sec' (line 63)
        sec_292492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 38), 'sec')
        # Getting the type of 'None' (line 63)
        None_292493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 49), 'None')
        # Applying the binary operator 'isnot' (line 63)
        result_is_not_292494 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 38), 'isnot', sec_292492, None_292493)
        
        
        # Getting the type of 'jd' (line 63)
        jd_292495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 57), 'jd')
        # Getting the type of 'None' (line 63)
        None_292496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 67), 'None')
        # Applying the binary operator 'isnot' (line 63)
        result_is_not_292497 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 57), 'isnot', jd_292495, None_292496)
        
        # Applying the binary operator 'or' (line 63)
        result_or_keyword_292498 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 38), 'or', result_is_not_292494, result_is_not_292497)
        
        # Applying the binary operator 'and' (line 63)
        result_and_keyword_292499 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 13), 'and', result_is_not_292491, result_or_keyword_292498)
        
        # Applying the binary operator 'or' (line 61)
        result_or_keyword_292500 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 11), 'or', result_or_keyword_292488, result_and_keyword_292499)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'daynum' (line 64)
        daynum_292501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 13), 'daynum')
        # Getting the type of 'None' (line 64)
        None_292502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 23), 'None')
        # Applying the binary operator 'is' (line 64)
        result_is__292503 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 13), 'is', daynum_292501, None_292502)
        
        
        # Getting the type of 'dt' (line 64)
        dt_292504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 32), 'dt')
        # Getting the type of 'None' (line 64)
        None_292505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 38), 'None')
        # Applying the binary operator 'is' (line 64)
        result_is__292506 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 32), 'is', dt_292504, None_292505)
        
        # Applying the binary operator 'and' (line 64)
        result_and_keyword_292507 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 13), 'and', result_is__292503, result_is__292506)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'sec' (line 64)
        sec_292508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 49), 'sec')
        # Getting the type of 'None' (line 64)
        None_292509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 56), 'None')
        # Applying the binary operator 'is' (line 64)
        result_is__292510 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 49), 'is', sec_292508, None_292509)
        
        
        # Getting the type of 'jd' (line 64)
        jd_292511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 64), 'jd')
        # Getting the type of 'None' (line 64)
        None_292512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 70), 'None')
        # Applying the binary operator 'is' (line 64)
        result_is__292513 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 64), 'is', jd_292511, None_292512)
        
        # Applying the binary operator 'or' (line 64)
        result_or_keyword_292514 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 49), 'or', result_is__292510, result_is__292513)
        
        # Applying the binary operator 'and' (line 64)
        result_and_keyword_292515 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 13), 'and', result_and_keyword_292507, result_or_keyword_292514)
        
        # Applying the binary operator 'or' (line 61)
        result_or_keyword_292516 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 11), 'or', result_or_keyword_292500, result_and_keyword_292515)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'daynum' (line 65)
        daynum_292517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 13), 'daynum')
        # Getting the type of 'None' (line 65)
        None_292518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 27), 'None')
        # Applying the binary operator 'isnot' (line 65)
        result_is_not_292519 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 13), 'isnot', daynum_292517, None_292518)
        
        
        # Getting the type of 'dt' (line 65)
        dt_292520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 36), 'dt')
        # Getting the type of 'None' (line 65)
        None_292521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 46), 'None')
        # Applying the binary operator 'isnot' (line 65)
        result_is_not_292522 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 36), 'isnot', dt_292520, None_292521)
        
        # Applying the binary operator 'and' (line 65)
        result_and_keyword_292523 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 13), 'and', result_is_not_292519, result_is_not_292522)
        
        # Applying the binary operator 'or' (line 61)
        result_or_keyword_292524 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 11), 'or', result_or_keyword_292516, result_and_keyword_292523)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'dt' (line 66)
        dt_292525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 13), 'dt')
        # Getting the type of 'None' (line 66)
        None_292526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 23), 'None')
        # Applying the binary operator 'isnot' (line 66)
        result_is_not_292527 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 13), 'isnot', dt_292525, None_292526)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'sec' (line 66)
        sec_292528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 34), 'sec')
        # Getting the type of 'None' (line 66)
        None_292529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 45), 'None')
        # Applying the binary operator 'isnot' (line 66)
        result_is_not_292530 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 34), 'isnot', sec_292528, None_292529)
        
        
        # Getting the type of 'jd' (line 66)
        jd_292531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 53), 'jd')
        # Getting the type of 'None' (line 66)
        None_292532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 63), 'None')
        # Applying the binary operator 'isnot' (line 66)
        result_is_not_292533 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 53), 'isnot', jd_292531, None_292532)
        
        # Applying the binary operator 'or' (line 66)
        result_or_keyword_292534 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 34), 'or', result_is_not_292530, result_is_not_292533)
        
        # Applying the binary operator 'and' (line 66)
        result_and_keyword_292535 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 13), 'and', result_is_not_292527, result_or_keyword_292534)
        
        # Applying the binary operator 'or' (line 61)
        result_or_keyword_292536 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 11), 'or', result_or_keyword_292524, result_and_keyword_292535)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'dt' (line 67)
        dt_292537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 14), 'dt')
        # Getting the type of 'None' (line 67)
        None_292538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), 'None')
        # Applying the binary operator 'isnot' (line 67)
        result_is_not_292539 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 14), 'isnot', dt_292537, None_292538)
        
        
        
        # Call to isinstance(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'dt' (line 67)
        dt_292541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 49), 'dt', False)
        # Getting the type of 'DT' (line 67)
        DT_292542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 53), 'DT', False)
        # Obtaining the member 'datetime' of a type (line 67)
        datetime_292543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 53), DT_292542, 'datetime')
        # Processing the call keyword arguments (line 67)
        kwargs_292544 = {}
        # Getting the type of 'isinstance' (line 67)
        isinstance_292540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 38), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 67)
        isinstance_call_result_292545 = invoke(stypy.reporting.localization.Localization(__file__, 67, 38), isinstance_292540, *[dt_292541, datetime_292543], **kwargs_292544)
        
        # Applying the 'not' unary operator (line 67)
        result_not__292546 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 34), 'not', isinstance_call_result_292545)
        
        # Applying the binary operator 'and' (line 67)
        result_and_keyword_292547 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 13), 'and', result_is_not_292539, result_not__292546)
        
        # Applying the binary operator 'or' (line 61)
        result_or_keyword_292548 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 11), 'or', result_or_keyword_292536, result_and_keyword_292547)
        
        # Testing the type of an if condition (line 61)
        if_condition_292549 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 6), result_or_keyword_292548)
        # Assigning a type to the variable 'if_condition_292549' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 6), 'if_condition_292549', if_condition_292549)
        # SSA begins for if statement (line 61)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 68):
        unicode_292550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 15), 'unicode', u'Invalid inputs.  Must enter sec and jd together, daynum by itself, or dt (must be a python datetime).\nSec = %s\nJD  = %s\ndnum= %s\ndt  = %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 71)
        tuple_292551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 71)
        # Adding element type (line 71)
        
        # Call to str(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'sec' (line 71)
        sec_292553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 24), 'sec', False)
        # Processing the call keyword arguments (line 71)
        kwargs_292554 = {}
        # Getting the type of 'str' (line 71)
        str_292552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'str', False)
        # Calling str(args, kwargs) (line 71)
        str_call_result_292555 = invoke(stypy.reporting.localization.Localization(__file__, 71, 19), str_292552, *[sec_292553], **kwargs_292554)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 19), tuple_292551, str_call_result_292555)
        # Adding element type (line 71)
        
        # Call to str(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'jd' (line 71)
        jd_292557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 36), 'jd', False)
        # Processing the call keyword arguments (line 71)
        kwargs_292558 = {}
        # Getting the type of 'str' (line 71)
        str_292556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 31), 'str', False)
        # Calling str(args, kwargs) (line 71)
        str_call_result_292559 = invoke(stypy.reporting.localization.Localization(__file__, 71, 31), str_292556, *[jd_292557], **kwargs_292558)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 19), tuple_292551, str_call_result_292559)
        # Adding element type (line 71)
        
        # Call to str(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'daynum' (line 71)
        daynum_292561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 47), 'daynum', False)
        # Processing the call keyword arguments (line 71)
        kwargs_292562 = {}
        # Getting the type of 'str' (line 71)
        str_292560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 42), 'str', False)
        # Calling str(args, kwargs) (line 71)
        str_call_result_292563 = invoke(stypy.reporting.localization.Localization(__file__, 71, 42), str_292560, *[daynum_292561], **kwargs_292562)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 19), tuple_292551, str_call_result_292563)
        # Adding element type (line 71)
        
        # Call to str(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'dt' (line 71)
        dt_292565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 62), 'dt', False)
        # Processing the call keyword arguments (line 71)
        kwargs_292566 = {}
        # Getting the type of 'str' (line 71)
        str_292564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 57), 'str', False)
        # Calling str(args, kwargs) (line 71)
        str_call_result_292567 = invoke(stypy.reporting.localization.Localization(__file__, 71, 57), str_292564, *[dt_292565], **kwargs_292566)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 19), tuple_292551, str_call_result_292567)
        
        # Applying the binary operator '%' (line 68)
        result_mod_292568 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 15), '%', unicode_292550, tuple_292551)
        
        # Assigning a type to the variable 'msg' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 9), 'msg', result_mod_292568)
        
        # Call to ValueError(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'msg' (line 72)
        msg_292570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 27), 'msg', False)
        # Processing the call keyword arguments (line 72)
        kwargs_292571 = {}
        # Getting the type of 'ValueError' (line 72)
        ValueError_292569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 72)
        ValueError_call_result_292572 = invoke(stypy.reporting.localization.Localization(__file__, 72, 15), ValueError_292569, *[msg_292570], **kwargs_292571)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 72, 9), ValueError_call_result_292572, 'raise parameter', BaseException)
        # SSA join for if statement (line 61)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'frame' (line 74)
        frame_292573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 9), 'frame')
        # Getting the type of 'self' (line 74)
        self_292574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 22), 'self')
        # Obtaining the member 'allowed' of a type (line 74)
        allowed_292575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 22), self_292574, 'allowed')
        # Applying the binary operator 'notin' (line 74)
        result_contains_292576 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 9), 'notin', frame_292573, allowed_292575)
        
        # Testing the type of an if condition (line 74)
        if_condition_292577 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 6), result_contains_292576)
        # Assigning a type to the variable 'if_condition_292577' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 6), 'if_condition_292577', if_condition_292577)
        # SSA begins for if statement (line 74)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 75):
        unicode_292578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 15), 'unicode', u"Input frame '%s' is not one of the supported frames of %s")
        
        # Obtaining an instance of the builtin type 'tuple' (line 76)
        tuple_292579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 76)
        # Adding element type (line 76)
        # Getting the type of 'frame' (line 76)
        frame_292580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 19), 'frame')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 19), tuple_292579, frame_292580)
        # Adding element type (line 76)
        
        # Call to str(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Call to list(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Call to iterkeys(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'self' (line 76)
        self_292585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 49), 'self', False)
        # Obtaining the member 'allowed' of a type (line 76)
        allowed_292586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 49), self_292585, 'allowed')
        # Processing the call keyword arguments (line 76)
        kwargs_292587 = {}
        # Getting the type of 'six' (line 76)
        six_292583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 36), 'six', False)
        # Obtaining the member 'iterkeys' of a type (line 76)
        iterkeys_292584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 36), six_292583, 'iterkeys')
        # Calling iterkeys(args, kwargs) (line 76)
        iterkeys_call_result_292588 = invoke(stypy.reporting.localization.Localization(__file__, 76, 36), iterkeys_292584, *[allowed_292586], **kwargs_292587)
        
        # Processing the call keyword arguments (line 76)
        kwargs_292589 = {}
        # Getting the type of 'list' (line 76)
        list_292582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 31), 'list', False)
        # Calling list(args, kwargs) (line 76)
        list_call_result_292590 = invoke(stypy.reporting.localization.Localization(__file__, 76, 31), list_292582, *[iterkeys_call_result_292588], **kwargs_292589)
        
        # Processing the call keyword arguments (line 76)
        kwargs_292591 = {}
        # Getting the type of 'str' (line 76)
        str_292581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 26), 'str', False)
        # Calling str(args, kwargs) (line 76)
        str_call_result_292592 = invoke(stypy.reporting.localization.Localization(__file__, 76, 26), str_292581, *[list_call_result_292590], **kwargs_292591)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 19), tuple_292579, str_call_result_292592)
        
        # Applying the binary operator '%' (line 75)
        result_mod_292593 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 15), '%', unicode_292578, tuple_292579)
        
        # Assigning a type to the variable 'msg' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 9), 'msg', result_mod_292593)
        
        # Call to ValueError(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'msg' (line 77)
        msg_292595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 26), 'msg', False)
        # Processing the call keyword arguments (line 77)
        kwargs_292596 = {}
        # Getting the type of 'ValueError' (line 77)
        ValueError_292594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 15), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 77)
        ValueError_call_result_292597 = invoke(stypy.reporting.localization.Localization(__file__, 77, 15), ValueError_292594, *[msg_292595], **kwargs_292596)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 77, 9), ValueError_call_result_292597, 'raise parameter', BaseException)
        # SSA join for if statement (line 74)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 79):
        # Getting the type of 'frame' (line 79)
        frame_292598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'frame')
        # Getting the type of 'self' (line 79)
        self_292599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 6), 'self')
        # Setting the type of the member '_frame' of a type (line 79)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 6), self_292599, '_frame', frame_292598)
        
        # Type idiom detected: calculating its left and rigth part (line 81)
        # Getting the type of 'dt' (line 81)
        dt_292600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 6), 'dt')
        # Getting the type of 'None' (line 81)
        None_292601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 19), 'None')
        
        (may_be_292602, more_types_in_union_292603) = may_not_be_none(dt_292600, None_292601)

        if may_be_292602:

            if more_types_in_union_292603:
                # Runtime conditional SSA (line 81)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 82):
            
            # Call to date2num(...): (line 82)
            # Processing the call arguments (line 82)
            # Getting the type of 'dt' (line 82)
            dt_292605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 28), 'dt', False)
            # Processing the call keyword arguments (line 82)
            kwargs_292606 = {}
            # Getting the type of 'date2num' (line 82)
            date2num_292604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 18), 'date2num', False)
            # Calling date2num(args, kwargs) (line 82)
            date2num_call_result_292607 = invoke(stypy.reporting.localization.Localization(__file__, 82, 18), date2num_292604, *[dt_292605], **kwargs_292606)
            
            # Assigning a type to the variable 'daynum' (line 82)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 9), 'daynum', date2num_call_result_292607)

            if more_types_in_union_292603:
                # SSA join for if statement (line 81)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 84)
        # Getting the type of 'daynum' (line 84)
        daynum_292608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 6), 'daynum')
        # Getting the type of 'None' (line 84)
        None_292609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 23), 'None')
        
        (may_be_292610, more_types_in_union_292611) = may_not_be_none(daynum_292608, None_292609)

        if may_be_292610:

            if more_types_in_union_292611:
                # Runtime conditional SSA (line 84)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 86):
            
            # Call to float(...): (line 86)
            # Processing the call arguments (line 86)
            # Getting the type of 'daynum' (line 86)
            daynum_292613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 21), 'daynum', False)
            # Processing the call keyword arguments (line 86)
            kwargs_292614 = {}
            # Getting the type of 'float' (line 86)
            float_292612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 14), 'float', False)
            # Calling float(args, kwargs) (line 86)
            float_call_result_292615 = invoke(stypy.reporting.localization.Localization(__file__, 86, 14), float_292612, *[daynum_292613], **kwargs_292614)
            
            float_292616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 32), 'float')
            # Applying the binary operator '+' (line 86)
            result_add_292617 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 14), '+', float_call_result_292615, float_292616)
            
            # Assigning a type to the variable 'jd' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 9), 'jd', result_add_292617)
            
            # Assigning a Call to a Attribute (line 87):
            
            # Call to floor(...): (line 87)
            # Processing the call arguments (line 87)
            # Getting the type of 'jd' (line 87)
            jd_292620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 32), 'jd', False)
            # Processing the call keyword arguments (line 87)
            kwargs_292621 = {}
            # Getting the type of 'math' (line 87)
            math_292618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'math', False)
            # Obtaining the member 'floor' of a type (line 87)
            floor_292619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 20), math_292618, 'floor')
            # Calling floor(args, kwargs) (line 87)
            floor_call_result_292622 = invoke(stypy.reporting.localization.Localization(__file__, 87, 20), floor_292619, *[jd_292620], **kwargs_292621)
            
            # Getting the type of 'self' (line 87)
            self_292623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 9), 'self')
            # Setting the type of the member '_jd' of a type (line 87)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 9), self_292623, '_jd', floor_call_result_292622)
            
            # Assigning a BinOp to a Attribute (line 88):
            # Getting the type of 'jd' (line 88)
            jd_292624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 27), 'jd')
            # Getting the type of 'self' (line 88)
            self_292625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 32), 'self')
            # Obtaining the member '_jd' of a type (line 88)
            _jd_292626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 32), self_292625, '_jd')
            # Applying the binary operator '-' (line 88)
            result_sub_292627 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 27), '-', jd_292624, _jd_292626)
            
            float_292628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 45), 'float')
            # Applying the binary operator '*' (line 88)
            result_mul_292629 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 25), '*', result_sub_292627, float_292628)
            
            # Getting the type of 'self' (line 88)
            self_292630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 9), 'self')
            # Setting the type of the member '_seconds' of a type (line 88)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 9), self_292630, '_seconds', result_mul_292629)

            if more_types_in_union_292611:
                # Runtime conditional SSA for else branch (line 84)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_292610) or more_types_in_union_292611):
            
            # Assigning a Call to a Attribute (line 91):
            
            # Call to float(...): (line 91)
            # Processing the call arguments (line 91)
            # Getting the type of 'sec' (line 91)
            sec_292632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 32), 'sec', False)
            # Processing the call keyword arguments (line 91)
            kwargs_292633 = {}
            # Getting the type of 'float' (line 91)
            float_292631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 25), 'float', False)
            # Calling float(args, kwargs) (line 91)
            float_call_result_292634 = invoke(stypy.reporting.localization.Localization(__file__, 91, 25), float_292631, *[sec_292632], **kwargs_292633)
            
            # Getting the type of 'self' (line 91)
            self_292635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 9), 'self')
            # Setting the type of the member '_seconds' of a type (line 91)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 9), self_292635, '_seconds', float_call_result_292634)
            
            # Assigning a Call to a Attribute (line 92):
            
            # Call to float(...): (line 92)
            # Processing the call arguments (line 92)
            # Getting the type of 'jd' (line 92)
            jd_292637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 27), 'jd', False)
            # Processing the call keyword arguments (line 92)
            kwargs_292638 = {}
            # Getting the type of 'float' (line 92)
            float_292636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 20), 'float', False)
            # Calling float(args, kwargs) (line 92)
            float_call_result_292639 = invoke(stypy.reporting.localization.Localization(__file__, 92, 20), float_292636, *[jd_292637], **kwargs_292638)
            
            # Getting the type of 'self' (line 92)
            self_292640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 9), 'self')
            # Setting the type of the member '_jd' of a type (line 92)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 9), self_292640, '_jd', float_call_result_292639)
            
            # Assigning a Call to a Name (line 95):
            
            # Call to int(...): (line 95)
            # Processing the call arguments (line 95)
            
            # Call to floor(...): (line 95)
            # Processing the call arguments (line 95)
            # Getting the type of 'self' (line 95)
            self_292644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 38), 'self', False)
            # Obtaining the member '_seconds' of a type (line 95)
            _seconds_292645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 38), self_292644, '_seconds')
            float_292646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 54), 'float')
            # Applying the binary operator 'div' (line 95)
            result_div_292647 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 38), 'div', _seconds_292645, float_292646)
            
            # Processing the call keyword arguments (line 95)
            kwargs_292648 = {}
            # Getting the type of 'math' (line 95)
            math_292642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 26), 'math', False)
            # Obtaining the member 'floor' of a type (line 95)
            floor_292643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 26), math_292642, 'floor')
            # Calling floor(args, kwargs) (line 95)
            floor_call_result_292649 = invoke(stypy.reporting.localization.Localization(__file__, 95, 26), floor_292643, *[result_div_292647], **kwargs_292648)
            
            # Processing the call keyword arguments (line 95)
            kwargs_292650 = {}
            # Getting the type of 'int' (line 95)
            int_292641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 21), 'int', False)
            # Calling int(args, kwargs) (line 95)
            int_call_result_292651 = invoke(stypy.reporting.localization.Localization(__file__, 95, 21), int_292641, *[floor_call_result_292649], **kwargs_292650)
            
            # Assigning a type to the variable 'deltaDays' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 9), 'deltaDays', int_call_result_292651)
            
            # Getting the type of 'self' (line 96)
            self_292652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 9), 'self')
            # Obtaining the member '_jd' of a type (line 96)
            _jd_292653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 9), self_292652, '_jd')
            # Getting the type of 'deltaDays' (line 96)
            deltaDays_292654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 21), 'deltaDays')
            # Applying the binary operator '+=' (line 96)
            result_iadd_292655 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 9), '+=', _jd_292653, deltaDays_292654)
            # Getting the type of 'self' (line 96)
            self_292656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 9), 'self')
            # Setting the type of the member '_jd' of a type (line 96)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 9), self_292656, '_jd', result_iadd_292655)
            
            
            # Getting the type of 'self' (line 97)
            self_292657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 9), 'self')
            # Obtaining the member '_seconds' of a type (line 97)
            _seconds_292658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 9), self_292657, '_seconds')
            # Getting the type of 'deltaDays' (line 97)
            deltaDays_292659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 26), 'deltaDays')
            float_292660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 38), 'float')
            # Applying the binary operator '*' (line 97)
            result_mul_292661 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 26), '*', deltaDays_292659, float_292660)
            
            # Applying the binary operator '-=' (line 97)
            result_isub_292662 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 9), '-=', _seconds_292658, result_mul_292661)
            # Getting the type of 'self' (line 97)
            self_292663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 9), 'self')
            # Setting the type of the member '_seconds' of a type (line 97)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 9), self_292663, '_seconds', result_isub_292662)
            

            if (may_be_292610 and more_types_in_union_292611):
                # SSA join for if statement (line 84)
                module_type_store = module_type_store.join_ssa_context()


        
        
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
        module_type_store = module_type_store.open_function_context('convert', 100, 3, False)
        # Assigning a type to the variable 'self' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Epoch.convert.__dict__.__setitem__('stypy_localization', localization)
        Epoch.convert.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Epoch.convert.__dict__.__setitem__('stypy_type_store', module_type_store)
        Epoch.convert.__dict__.__setitem__('stypy_function_name', 'Epoch.convert')
        Epoch.convert.__dict__.__setitem__('stypy_param_names_list', ['frame'])
        Epoch.convert.__dict__.__setitem__('stypy_varargs_param_name', None)
        Epoch.convert.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Epoch.convert.__dict__.__setitem__('stypy_call_defaults', defaults)
        Epoch.convert.__dict__.__setitem__('stypy_call_varargs', varargs)
        Epoch.convert.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Epoch.convert.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Epoch.convert', ['frame'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'convert', localization, ['frame'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'convert(...)' code ##################

        
        
        # Getting the type of 'self' (line 101)
        self_292664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 9), 'self')
        # Obtaining the member '_frame' of a type (line 101)
        _frame_292665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 9), self_292664, '_frame')
        # Getting the type of 'frame' (line 101)
        frame_292666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'frame')
        # Applying the binary operator '==' (line 101)
        result_eq_292667 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 9), '==', _frame_292665, frame_292666)
        
        # Testing the type of an if condition (line 101)
        if_condition_292668 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 6), result_eq_292667)
        # Assigning a type to the variable 'if_condition_292668' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 6), 'if_condition_292668', if_condition_292668)
        # SSA begins for if statement (line 101)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 102)
        self_292669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 9), 'stypy_return_type', self_292669)
        # SSA join for if statement (line 101)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 104):
        
        # Obtaining the type of the subscript
        # Getting the type of 'frame' (line 104)
        frame_292670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 44), 'frame')
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 104)
        self_292671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 'self')
        # Obtaining the member '_frame' of a type (line 104)
        _frame_292672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 29), self_292671, '_frame')
        # Getting the type of 'self' (line 104)
        self_292673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'self')
        # Obtaining the member 'allowed' of a type (line 104)
        allowed_292674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 15), self_292673, 'allowed')
        # Obtaining the member '__getitem__' of a type (line 104)
        getitem___292675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 15), allowed_292674, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 104)
        subscript_call_result_292676 = invoke(stypy.reporting.localization.Localization(__file__, 104, 15), getitem___292675, _frame_292672)
        
        # Obtaining the member '__getitem__' of a type (line 104)
        getitem___292677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 15), subscript_call_result_292676, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 104)
        subscript_call_result_292678 = invoke(stypy.reporting.localization.Localization(__file__, 104, 15), getitem___292677, frame_292670)
        
        # Assigning a type to the variable 'offset' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 6), 'offset', subscript_call_result_292678)
        
        # Call to Epoch(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'frame' (line 106)
        frame_292680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'frame', False)
        # Getting the type of 'self' (line 106)
        self_292681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'self', False)
        # Obtaining the member '_seconds' of a type (line 106)
        _seconds_292682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 27), self_292681, '_seconds')
        # Getting the type of 'offset' (line 106)
        offset_292683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 43), 'offset', False)
        # Applying the binary operator '+' (line 106)
        result_add_292684 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 27), '+', _seconds_292682, offset_292683)
        
        # Getting the type of 'self' (line 106)
        self_292685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 51), 'self', False)
        # Obtaining the member '_jd' of a type (line 106)
        _jd_292686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 51), self_292685, '_jd')
        # Processing the call keyword arguments (line 106)
        kwargs_292687 = {}
        # Getting the type of 'Epoch' (line 106)
        Epoch_292679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'Epoch', False)
        # Calling Epoch(args, kwargs) (line 106)
        Epoch_call_result_292688 = invoke(stypy.reporting.localization.Localization(__file__, 106, 13), Epoch_292679, *[frame_292680, result_add_292684, _jd_292686], **kwargs_292687)
        
        # Assigning a type to the variable 'stypy_return_type' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 6), 'stypy_return_type', Epoch_call_result_292688)
        
        # ################# End of 'convert(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'convert' in the type store
        # Getting the type of 'stypy_return_type' (line 100)
        stypy_return_type_292689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292689)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'convert'
        return stypy_return_type_292689


    @norecursion
    def frame(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'frame'
        module_type_store = module_type_store.open_function_context('frame', 109, 3, False)
        # Assigning a type to the variable 'self' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Epoch.frame.__dict__.__setitem__('stypy_localization', localization)
        Epoch.frame.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Epoch.frame.__dict__.__setitem__('stypy_type_store', module_type_store)
        Epoch.frame.__dict__.__setitem__('stypy_function_name', 'Epoch.frame')
        Epoch.frame.__dict__.__setitem__('stypy_param_names_list', [])
        Epoch.frame.__dict__.__setitem__('stypy_varargs_param_name', None)
        Epoch.frame.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Epoch.frame.__dict__.__setitem__('stypy_call_defaults', defaults)
        Epoch.frame.__dict__.__setitem__('stypy_call_varargs', varargs)
        Epoch.frame.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Epoch.frame.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Epoch.frame', [], None, None, defaults, varargs, kwargs)

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

        # Getting the type of 'self' (line 110)
        self_292690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 13), 'self')
        # Obtaining the member '_frame' of a type (line 110)
        _frame_292691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 13), self_292690, '_frame')
        # Assigning a type to the variable 'stypy_return_type' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 6), 'stypy_return_type', _frame_292691)
        
        # ################# End of 'frame(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'frame' in the type store
        # Getting the type of 'stypy_return_type' (line 109)
        stypy_return_type_292692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292692)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'frame'
        return stypy_return_type_292692


    @norecursion
    def julianDate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'julianDate'
        module_type_store = module_type_store.open_function_context('julianDate', 113, 3, False)
        # Assigning a type to the variable 'self' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Epoch.julianDate.__dict__.__setitem__('stypy_localization', localization)
        Epoch.julianDate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Epoch.julianDate.__dict__.__setitem__('stypy_type_store', module_type_store)
        Epoch.julianDate.__dict__.__setitem__('stypy_function_name', 'Epoch.julianDate')
        Epoch.julianDate.__dict__.__setitem__('stypy_param_names_list', ['frame'])
        Epoch.julianDate.__dict__.__setitem__('stypy_varargs_param_name', None)
        Epoch.julianDate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Epoch.julianDate.__dict__.__setitem__('stypy_call_defaults', defaults)
        Epoch.julianDate.__dict__.__setitem__('stypy_call_varargs', varargs)
        Epoch.julianDate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Epoch.julianDate.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Epoch.julianDate', ['frame'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'julianDate', localization, ['frame'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'julianDate(...)' code ##################

        
        # Assigning a Name to a Name (line 114):
        # Getting the type of 'self' (line 114)
        self_292693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 10), 'self')
        # Assigning a type to the variable 't' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 6), 't', self_292693)
        
        
        # Getting the type of 'frame' (line 115)
        frame_292694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 9), 'frame')
        # Getting the type of 'self' (line 115)
        self_292695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 18), 'self')
        # Obtaining the member '_frame' of a type (line 115)
        _frame_292696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 18), self_292695, '_frame')
        # Applying the binary operator '!=' (line 115)
        result_ne_292697 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 9), '!=', frame_292694, _frame_292696)
        
        # Testing the type of an if condition (line 115)
        if_condition_292698 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 6), result_ne_292697)
        # Assigning a type to the variable 'if_condition_292698' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 6), 'if_condition_292698', if_condition_292698)
        # SSA begins for if statement (line 115)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 116):
        
        # Call to convert(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'frame' (line 116)
        frame_292701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 27), 'frame', False)
        # Processing the call keyword arguments (line 116)
        kwargs_292702 = {}
        # Getting the type of 'self' (line 116)
        self_292699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 13), 'self', False)
        # Obtaining the member 'convert' of a type (line 116)
        convert_292700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 13), self_292699, 'convert')
        # Calling convert(args, kwargs) (line 116)
        convert_call_result_292703 = invoke(stypy.reporting.localization.Localization(__file__, 116, 13), convert_292700, *[frame_292701], **kwargs_292702)
        
        # Assigning a type to the variable 't' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 9), 't', convert_call_result_292703)
        # SSA join for if statement (line 115)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 't' (line 118)
        t_292704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 13), 't')
        # Obtaining the member '_jd' of a type (line 118)
        _jd_292705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 13), t_292704, '_jd')
        # Getting the type of 't' (line 118)
        t_292706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 't')
        # Obtaining the member '_seconds' of a type (line 118)
        _seconds_292707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 21), t_292706, '_seconds')
        float_292708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 34), 'float')
        # Applying the binary operator 'div' (line 118)
        result_div_292709 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 21), 'div', _seconds_292707, float_292708)
        
        # Applying the binary operator '+' (line 118)
        result_add_292710 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 13), '+', _jd_292705, result_div_292709)
        
        # Assigning a type to the variable 'stypy_return_type' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 6), 'stypy_return_type', result_add_292710)
        
        # ################# End of 'julianDate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'julianDate' in the type store
        # Getting the type of 'stypy_return_type' (line 113)
        stypy_return_type_292711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292711)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'julianDate'
        return stypy_return_type_292711


    @norecursion
    def secondsPast(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'secondsPast'
        module_type_store = module_type_store.open_function_context('secondsPast', 121, 3, False)
        # Assigning a type to the variable 'self' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Epoch.secondsPast.__dict__.__setitem__('stypy_localization', localization)
        Epoch.secondsPast.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Epoch.secondsPast.__dict__.__setitem__('stypy_type_store', module_type_store)
        Epoch.secondsPast.__dict__.__setitem__('stypy_function_name', 'Epoch.secondsPast')
        Epoch.secondsPast.__dict__.__setitem__('stypy_param_names_list', ['frame', 'jd'])
        Epoch.secondsPast.__dict__.__setitem__('stypy_varargs_param_name', None)
        Epoch.secondsPast.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Epoch.secondsPast.__dict__.__setitem__('stypy_call_defaults', defaults)
        Epoch.secondsPast.__dict__.__setitem__('stypy_call_varargs', varargs)
        Epoch.secondsPast.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Epoch.secondsPast.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Epoch.secondsPast', ['frame', 'jd'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'secondsPast', localization, ['frame', 'jd'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'secondsPast(...)' code ##################

        
        # Assigning a Name to a Name (line 122):
        # Getting the type of 'self' (line 122)
        self_292712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 10), 'self')
        # Assigning a type to the variable 't' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 6), 't', self_292712)
        
        
        # Getting the type of 'frame' (line 123)
        frame_292713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 9), 'frame')
        # Getting the type of 'self' (line 123)
        self_292714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 18), 'self')
        # Obtaining the member '_frame' of a type (line 123)
        _frame_292715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 18), self_292714, '_frame')
        # Applying the binary operator '!=' (line 123)
        result_ne_292716 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 9), '!=', frame_292713, _frame_292715)
        
        # Testing the type of an if condition (line 123)
        if_condition_292717 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 6), result_ne_292716)
        # Assigning a type to the variable 'if_condition_292717' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 6), 'if_condition_292717', if_condition_292717)
        # SSA begins for if statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 124):
        
        # Call to convert(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'frame' (line 124)
        frame_292720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'frame', False)
        # Processing the call keyword arguments (line 124)
        kwargs_292721 = {}
        # Getting the type of 'self' (line 124)
        self_292718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 13), 'self', False)
        # Obtaining the member 'convert' of a type (line 124)
        convert_292719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 13), self_292718, 'convert')
        # Calling convert(args, kwargs) (line 124)
        convert_call_result_292722 = invoke(stypy.reporting.localization.Localization(__file__, 124, 13), convert_292719, *[frame_292720], **kwargs_292721)
        
        # Assigning a type to the variable 't' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 9), 't', convert_call_result_292722)
        # SSA join for if statement (line 123)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 126):
        # Getting the type of 't' (line 126)
        t_292723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 14), 't')
        # Obtaining the member '_jd' of a type (line 126)
        _jd_292724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 14), t_292723, '_jd')
        # Getting the type of 'jd' (line 126)
        jd_292725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 22), 'jd')
        # Applying the binary operator '-' (line 126)
        result_sub_292726 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 14), '-', _jd_292724, jd_292725)
        
        # Assigning a type to the variable 'delta' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 6), 'delta', result_sub_292726)
        # Getting the type of 't' (line 127)
        t_292727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 13), 't')
        # Obtaining the member '_seconds' of a type (line 127)
        _seconds_292728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 13), t_292727, '_seconds')
        # Getting the type of 'delta' (line 127)
        delta_292729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 26), 'delta')
        int_292730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 34), 'int')
        # Applying the binary operator '*' (line 127)
        result_mul_292731 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 26), '*', delta_292729, int_292730)
        
        # Applying the binary operator '+' (line 127)
        result_add_292732 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 13), '+', _seconds_292728, result_mul_292731)
        
        # Assigning a type to the variable 'stypy_return_type' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 6), 'stypy_return_type', result_add_292732)
        
        # ################# End of 'secondsPast(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'secondsPast' in the type store
        # Getting the type of 'stypy_return_type' (line 121)
        stypy_return_type_292733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292733)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'secondsPast'
        return stypy_return_type_292733


    @norecursion
    def stypy__cmp__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__cmp__'
        module_type_store = module_type_store.open_function_context('__cmp__', 130, 3, False)
        # Assigning a type to the variable 'self' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Epoch.stypy__cmp__.__dict__.__setitem__('stypy_localization', localization)
        Epoch.stypy__cmp__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Epoch.stypy__cmp__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Epoch.stypy__cmp__.__dict__.__setitem__('stypy_function_name', 'Epoch.stypy__cmp__')
        Epoch.stypy__cmp__.__dict__.__setitem__('stypy_param_names_list', ['rhs'])
        Epoch.stypy__cmp__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Epoch.stypy__cmp__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Epoch.stypy__cmp__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Epoch.stypy__cmp__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Epoch.stypy__cmp__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Epoch.stypy__cmp__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Epoch.stypy__cmp__', ['rhs'], None, None, defaults, varargs, kwargs)

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

        unicode_292734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, (-1)), 'unicode', u"Compare two Epoch's.\n\n      = INPUT VARIABLES\n      - rhs    The Epoch to compare against.\n\n      = RETURN VALUE\n      - Returns -1 if self < rhs, 0 if self == rhs, +1 if self > rhs.\n      ")
        
        # Assigning a Name to a Name (line 139):
        # Getting the type of 'self' (line 139)
        self_292735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 10), 'self')
        # Assigning a type to the variable 't' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 6), 't', self_292735)
        
        
        # Getting the type of 'self' (line 140)
        self_292736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 9), 'self')
        # Obtaining the member '_frame' of a type (line 140)
        _frame_292737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 9), self_292736, '_frame')
        # Getting the type of 'rhs' (line 140)
        rhs_292738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 24), 'rhs')
        # Obtaining the member '_frame' of a type (line 140)
        _frame_292739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 24), rhs_292738, '_frame')
        # Applying the binary operator '!=' (line 140)
        result_ne_292740 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 9), '!=', _frame_292737, _frame_292739)
        
        # Testing the type of an if condition (line 140)
        if_condition_292741 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 6), result_ne_292740)
        # Assigning a type to the variable 'if_condition_292741' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 6), 'if_condition_292741', if_condition_292741)
        # SSA begins for if statement (line 140)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 141):
        
        # Call to convert(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'rhs' (line 141)
        rhs_292744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 27), 'rhs', False)
        # Obtaining the member '_frame' of a type (line 141)
        _frame_292745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 27), rhs_292744, '_frame')
        # Processing the call keyword arguments (line 141)
        kwargs_292746 = {}
        # Getting the type of 'self' (line 141)
        self_292742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 13), 'self', False)
        # Obtaining the member 'convert' of a type (line 141)
        convert_292743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 13), self_292742, 'convert')
        # Calling convert(args, kwargs) (line 141)
        convert_call_result_292747 = invoke(stypy.reporting.localization.Localization(__file__, 141, 13), convert_292743, *[_frame_292745], **kwargs_292746)
        
        # Assigning a type to the variable 't' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 9), 't', convert_call_result_292747)
        # SSA join for if statement (line 140)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 't' (line 143)
        t_292748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 9), 't')
        # Obtaining the member '_jd' of a type (line 143)
        _jd_292749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 9), t_292748, '_jd')
        # Getting the type of 'rhs' (line 143)
        rhs_292750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 18), 'rhs')
        # Obtaining the member '_jd' of a type (line 143)
        _jd_292751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 18), rhs_292750, '_jd')
        # Applying the binary operator '!=' (line 143)
        result_ne_292752 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 9), '!=', _jd_292749, _jd_292751)
        
        # Testing the type of an if condition (line 143)
        if_condition_292753 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 6), result_ne_292752)
        # Assigning a type to the variable 'if_condition_292753' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 6), 'if_condition_292753', if_condition_292753)
        # SSA begins for if statement (line 143)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to cmp(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 't' (line 144)
        t_292755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 21), 't', False)
        # Obtaining the member '_jd' of a type (line 144)
        _jd_292756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 21), t_292755, '_jd')
        # Getting the type of 'rhs' (line 144)
        rhs_292757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 28), 'rhs', False)
        # Obtaining the member '_jd' of a type (line 144)
        _jd_292758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 28), rhs_292757, '_jd')
        # Processing the call keyword arguments (line 144)
        kwargs_292759 = {}
        # Getting the type of 'cmp' (line 144)
        cmp_292754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'cmp', False)
        # Calling cmp(args, kwargs) (line 144)
        cmp_call_result_292760 = invoke(stypy.reporting.localization.Localization(__file__, 144, 16), cmp_292754, *[_jd_292756, _jd_292758], **kwargs_292759)
        
        # Assigning a type to the variable 'stypy_return_type' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 9), 'stypy_return_type', cmp_call_result_292760)
        # SSA join for if statement (line 143)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to cmp(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 't' (line 146)
        t_292762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 18), 't', False)
        # Obtaining the member '_seconds' of a type (line 146)
        _seconds_292763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 18), t_292762, '_seconds')
        # Getting the type of 'rhs' (line 146)
        rhs_292764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 30), 'rhs', False)
        # Obtaining the member '_seconds' of a type (line 146)
        _seconds_292765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 30), rhs_292764, '_seconds')
        # Processing the call keyword arguments (line 146)
        kwargs_292766 = {}
        # Getting the type of 'cmp' (line 146)
        cmp_292761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 13), 'cmp', False)
        # Calling cmp(args, kwargs) (line 146)
        cmp_call_result_292767 = invoke(stypy.reporting.localization.Localization(__file__, 146, 13), cmp_292761, *[_seconds_292763, _seconds_292765], **kwargs_292766)
        
        # Assigning a type to the variable 'stypy_return_type' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 6), 'stypy_return_type', cmp_call_result_292767)
        
        # ################# End of '__cmp__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__cmp__' in the type store
        # Getting the type of 'stypy_return_type' (line 130)
        stypy_return_type_292768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292768)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__cmp__'
        return stypy_return_type_292768


    @norecursion
    def __add__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__add__'
        module_type_store = module_type_store.open_function_context('__add__', 149, 3, False)
        # Assigning a type to the variable 'self' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Epoch.__add__.__dict__.__setitem__('stypy_localization', localization)
        Epoch.__add__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Epoch.__add__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Epoch.__add__.__dict__.__setitem__('stypy_function_name', 'Epoch.__add__')
        Epoch.__add__.__dict__.__setitem__('stypy_param_names_list', ['rhs'])
        Epoch.__add__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Epoch.__add__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Epoch.__add__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Epoch.__add__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Epoch.__add__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Epoch.__add__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Epoch.__add__', ['rhs'], None, None, defaults, varargs, kwargs)

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

        unicode_292769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, (-1)), 'unicode', u'Add a duration to an Epoch.\n\n      = INPUT VARIABLES\n      - rhs    The Epoch to subtract.\n\n      = RETURN VALUE\n      - Returns the difference of ourselves and the input Epoch.\n      ')
        
        # Assigning a Name to a Name (line 158):
        # Getting the type of 'self' (line 158)
        self_292770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 10), 'self')
        # Assigning a type to the variable 't' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 6), 't', self_292770)
        
        
        # Getting the type of 'self' (line 159)
        self_292771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 9), 'self')
        # Obtaining the member '_frame' of a type (line 159)
        _frame_292772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 9), self_292771, '_frame')
        
        # Call to frame(...): (line 159)
        # Processing the call keyword arguments (line 159)
        kwargs_292775 = {}
        # Getting the type of 'rhs' (line 159)
        rhs_292773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 24), 'rhs', False)
        # Obtaining the member 'frame' of a type (line 159)
        frame_292774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 24), rhs_292773, 'frame')
        # Calling frame(args, kwargs) (line 159)
        frame_call_result_292776 = invoke(stypy.reporting.localization.Localization(__file__, 159, 24), frame_292774, *[], **kwargs_292775)
        
        # Applying the binary operator '!=' (line 159)
        result_ne_292777 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 9), '!=', _frame_292772, frame_call_result_292776)
        
        # Testing the type of an if condition (line 159)
        if_condition_292778 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 6), result_ne_292777)
        # Assigning a type to the variable 'if_condition_292778' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 6), 'if_condition_292778', if_condition_292778)
        # SSA begins for if statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 160):
        
        # Call to convert(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'rhs' (line 160)
        rhs_292781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 27), 'rhs', False)
        # Obtaining the member '_frame' of a type (line 160)
        _frame_292782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 27), rhs_292781, '_frame')
        # Processing the call keyword arguments (line 160)
        kwargs_292783 = {}
        # Getting the type of 'self' (line 160)
        self_292779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 13), 'self', False)
        # Obtaining the member 'convert' of a type (line 160)
        convert_292780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 13), self_292779, 'convert')
        # Calling convert(args, kwargs) (line 160)
        convert_call_result_292784 = invoke(stypy.reporting.localization.Localization(__file__, 160, 13), convert_292780, *[_frame_292782], **kwargs_292783)
        
        # Assigning a type to the variable 't' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 9), 't', convert_call_result_292784)
        # SSA join for if statement (line 159)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 162):
        # Getting the type of 't' (line 162)
        t_292785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 't')
        # Obtaining the member '_seconds' of a type (line 162)
        _seconds_292786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 12), t_292785, '_seconds')
        
        # Call to seconds(...): (line 162)
        # Processing the call keyword arguments (line 162)
        kwargs_292789 = {}
        # Getting the type of 'rhs' (line 162)
        rhs_292787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 25), 'rhs', False)
        # Obtaining the member 'seconds' of a type (line 162)
        seconds_292788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 25), rhs_292787, 'seconds')
        # Calling seconds(args, kwargs) (line 162)
        seconds_call_result_292790 = invoke(stypy.reporting.localization.Localization(__file__, 162, 25), seconds_292788, *[], **kwargs_292789)
        
        # Applying the binary operator '+' (line 162)
        result_add_292791 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 12), '+', _seconds_292786, seconds_call_result_292790)
        
        # Assigning a type to the variable 'sec' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 6), 'sec', result_add_292791)
        
        # Call to Epoch(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 't' (line 164)
        t_292793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 't', False)
        # Obtaining the member '_frame' of a type (line 164)
        _frame_292794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 20), t_292793, '_frame')
        # Getting the type of 'sec' (line 164)
        sec_292795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 30), 'sec', False)
        # Getting the type of 't' (line 164)
        t_292796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 35), 't', False)
        # Obtaining the member '_jd' of a type (line 164)
        _jd_292797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 35), t_292796, '_jd')
        # Processing the call keyword arguments (line 164)
        kwargs_292798 = {}
        # Getting the type of 'Epoch' (line 164)
        Epoch_292792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 13), 'Epoch', False)
        # Calling Epoch(args, kwargs) (line 164)
        Epoch_call_result_292799 = invoke(stypy.reporting.localization.Localization(__file__, 164, 13), Epoch_292792, *[_frame_292794, sec_292795, _jd_292797], **kwargs_292798)
        
        # Assigning a type to the variable 'stypy_return_type' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 6), 'stypy_return_type', Epoch_call_result_292799)
        
        # ################# End of '__add__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__add__' in the type store
        # Getting the type of 'stypy_return_type' (line 149)
        stypy_return_type_292800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292800)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__add__'
        return stypy_return_type_292800


    @norecursion
    def __sub__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__sub__'
        module_type_store = module_type_store.open_function_context('__sub__', 167, 3, False)
        # Assigning a type to the variable 'self' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Epoch.__sub__.__dict__.__setitem__('stypy_localization', localization)
        Epoch.__sub__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Epoch.__sub__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Epoch.__sub__.__dict__.__setitem__('stypy_function_name', 'Epoch.__sub__')
        Epoch.__sub__.__dict__.__setitem__('stypy_param_names_list', ['rhs'])
        Epoch.__sub__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Epoch.__sub__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Epoch.__sub__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Epoch.__sub__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Epoch.__sub__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Epoch.__sub__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Epoch.__sub__', ['rhs'], None, None, defaults, varargs, kwargs)

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

        unicode_292801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, (-1)), 'unicode', u"Subtract two Epoch's or a Duration from an Epoch.\n\n      Valid:\n      Duration = Epoch - Epoch\n      Epoch = Epoch - Duration\n\n      = INPUT VARIABLES\n      - rhs    The Epoch to subtract.\n\n      = RETURN VALUE\n      - Returns either the duration between to Epoch's or the a new\n        Epoch that is the result of subtracting a duration from an epoch.\n      ")
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 182, 6))
        
        # 'import matplotlib.testing.jpl_units' statement (line 182)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
        import_292802 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 182, 6), 'matplotlib.testing.jpl_units')

        if (type(import_292802) is not StypyTypeError):

            if (import_292802 != 'pyd_module'):
                __import__(import_292802)
                sys_modules_292803 = sys.modules[import_292802]
                import_module(stypy.reporting.localization.Localization(__file__, 182, 6), 'U', sys_modules_292803.module_type_store, module_type_store)
            else:
                import matplotlib.testing.jpl_units as U

                import_module(stypy.reporting.localization.Localization(__file__, 182, 6), 'U', matplotlib.testing.jpl_units, module_type_store)

        else:
            # Assigning a type to the variable 'matplotlib.testing.jpl_units' (line 182)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 6), 'matplotlib.testing.jpl_units', import_292802)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
        
        
        
        # Call to isinstance(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 'rhs' (line 185)
        rhs_292805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 21), 'rhs', False)
        # Getting the type of 'U' (line 185)
        U_292806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 26), 'U', False)
        # Obtaining the member 'Duration' of a type (line 185)
        Duration_292807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 26), U_292806, 'Duration')
        # Processing the call keyword arguments (line 185)
        kwargs_292808 = {}
        # Getting the type of 'isinstance' (line 185)
        isinstance_292804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 9), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 185)
        isinstance_call_result_292809 = invoke(stypy.reporting.localization.Localization(__file__, 185, 9), isinstance_292804, *[rhs_292805, Duration_292807], **kwargs_292808)
        
        # Testing the type of an if condition (line 185)
        if_condition_292810 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 6), isinstance_call_result_292809)
        # Assigning a type to the variable 'if_condition_292810' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 6), 'if_condition_292810', if_condition_292810)
        # SSA begins for if statement (line 185)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 186)
        self_292811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 16), 'self')
        
        # Getting the type of 'rhs' (line 186)
        rhs_292812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 24), 'rhs')
        # Applying the 'usub' unary operator (line 186)
        result___neg___292813 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 23), 'usub', rhs_292812)
        
        # Applying the binary operator '+' (line 186)
        result_add_292814 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 16), '+', self_292811, result___neg___292813)
        
        # Assigning a type to the variable 'stypy_return_type' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 9), 'stypy_return_type', result_add_292814)
        # SSA join for if statement (line 185)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 188):
        # Getting the type of 'self' (line 188)
        self_292815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 10), 'self')
        # Assigning a type to the variable 't' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 6), 't', self_292815)
        
        
        # Getting the type of 'self' (line 189)
        self_292816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 9), 'self')
        # Obtaining the member '_frame' of a type (line 189)
        _frame_292817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 9), self_292816, '_frame')
        # Getting the type of 'rhs' (line 189)
        rhs_292818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 24), 'rhs')
        # Obtaining the member '_frame' of a type (line 189)
        _frame_292819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 24), rhs_292818, '_frame')
        # Applying the binary operator '!=' (line 189)
        result_ne_292820 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 9), '!=', _frame_292817, _frame_292819)
        
        # Testing the type of an if condition (line 189)
        if_condition_292821 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 6), result_ne_292820)
        # Assigning a type to the variable 'if_condition_292821' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 6), 'if_condition_292821', if_condition_292821)
        # SSA begins for if statement (line 189)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 190):
        
        # Call to convert(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'rhs' (line 190)
        rhs_292824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 27), 'rhs', False)
        # Obtaining the member '_frame' of a type (line 190)
        _frame_292825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 27), rhs_292824, '_frame')
        # Processing the call keyword arguments (line 190)
        kwargs_292826 = {}
        # Getting the type of 'self' (line 190)
        self_292822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 13), 'self', False)
        # Obtaining the member 'convert' of a type (line 190)
        convert_292823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 13), self_292822, 'convert')
        # Calling convert(args, kwargs) (line 190)
        convert_call_result_292827 = invoke(stypy.reporting.localization.Localization(__file__, 190, 13), convert_292823, *[_frame_292825], **kwargs_292826)
        
        # Assigning a type to the variable 't' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 9), 't', convert_call_result_292827)
        # SSA join for if statement (line 189)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 192):
        # Getting the type of 't' (line 192)
        t_292828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 13), 't')
        # Obtaining the member '_jd' of a type (line 192)
        _jd_292829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 13), t_292828, '_jd')
        # Getting the type of 'rhs' (line 192)
        rhs_292830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 21), 'rhs')
        # Obtaining the member '_jd' of a type (line 192)
        _jd_292831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 21), rhs_292830, '_jd')
        # Applying the binary operator '-' (line 192)
        result_sub_292832 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 13), '-', _jd_292829, _jd_292831)
        
        # Assigning a type to the variable 'days' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 6), 'days', result_sub_292832)
        
        # Assigning a BinOp to a Name (line 193):
        # Getting the type of 't' (line 193)
        t_292833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 't')
        # Obtaining the member '_seconds' of a type (line 193)
        _seconds_292834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), t_292833, '_seconds')
        # Getting the type of 'rhs' (line 193)
        rhs_292835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 25), 'rhs')
        # Obtaining the member '_seconds' of a type (line 193)
        _seconds_292836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 25), rhs_292835, '_seconds')
        # Applying the binary operator '-' (line 193)
        result_sub_292837 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 12), '-', _seconds_292834, _seconds_292836)
        
        # Assigning a type to the variable 'sec' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 6), 'sec', result_sub_292837)
        
        # Call to Duration(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'rhs' (line 195)
        rhs_292840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 25), 'rhs', False)
        # Obtaining the member '_frame' of a type (line 195)
        _frame_292841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 25), rhs_292840, '_frame')
        # Getting the type of 'days' (line 195)
        days_292842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 37), 'days', False)
        int_292843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 42), 'int')
        # Applying the binary operator '*' (line 195)
        result_mul_292844 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 37), '*', days_292842, int_292843)
        
        # Getting the type of 'sec' (line 195)
        sec_292845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 50), 'sec', False)
        # Applying the binary operator '+' (line 195)
        result_add_292846 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 37), '+', result_mul_292844, sec_292845)
        
        # Processing the call keyword arguments (line 195)
        kwargs_292847 = {}
        # Getting the type of 'U' (line 195)
        U_292838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 13), 'U', False)
        # Obtaining the member 'Duration' of a type (line 195)
        Duration_292839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 13), U_292838, 'Duration')
        # Calling Duration(args, kwargs) (line 195)
        Duration_call_result_292848 = invoke(stypy.reporting.localization.Localization(__file__, 195, 13), Duration_292839, *[_frame_292841, result_add_292846], **kwargs_292847)
        
        # Assigning a type to the variable 'stypy_return_type' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 6), 'stypy_return_type', Duration_call_result_292848)
        
        # ################# End of '__sub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__sub__' in the type store
        # Getting the type of 'stypy_return_type' (line 167)
        stypy_return_type_292849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292849)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__sub__'
        return stypy_return_type_292849


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 198, 3, False)
        # Assigning a type to the variable 'self' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Epoch.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        Epoch.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Epoch.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Epoch.stypy__str__.__dict__.__setitem__('stypy_function_name', 'Epoch.stypy__str__')
        Epoch.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        Epoch.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Epoch.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Epoch.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Epoch.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Epoch.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Epoch.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Epoch.stypy__str__', [], None, None, defaults, varargs, kwargs)

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

        unicode_292850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 6), 'unicode', u'Print the Epoch.')
        unicode_292851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 13), 'unicode', u'%22.15e %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 200)
        tuple_292852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 200)
        # Adding element type (line 200)
        
        # Call to julianDate(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'self' (line 200)
        self_292855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 47), 'self', False)
        # Obtaining the member '_frame' of a type (line 200)
        _frame_292856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 47), self_292855, '_frame')
        # Processing the call keyword arguments (line 200)
        kwargs_292857 = {}
        # Getting the type of 'self' (line 200)
        self_292853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 30), 'self', False)
        # Obtaining the member 'julianDate' of a type (line 200)
        julianDate_292854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 30), self_292853, 'julianDate')
        # Calling julianDate(args, kwargs) (line 200)
        julianDate_call_result_292858 = invoke(stypy.reporting.localization.Localization(__file__, 200, 30), julianDate_292854, *[_frame_292856], **kwargs_292857)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 30), tuple_292852, julianDate_call_result_292858)
        # Adding element type (line 200)
        # Getting the type of 'self' (line 200)
        self_292859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 62), 'self')
        # Obtaining the member '_frame' of a type (line 200)
        _frame_292860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 62), self_292859, '_frame')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 30), tuple_292852, _frame_292860)
        
        # Applying the binary operator '%' (line 200)
        result_mod_292861 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 13), '%', unicode_292851, tuple_292852)
        
        # Assigning a type to the variable 'stypy_return_type' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 6), 'stypy_return_type', result_mod_292861)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 198)
        stypy_return_type_292862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292862)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_292862


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 203, 3, False)
        # Assigning a type to the variable 'self' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Epoch.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        Epoch.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Epoch.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Epoch.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Epoch.stypy__repr__')
        Epoch.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        Epoch.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Epoch.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Epoch.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Epoch.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Epoch.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Epoch.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Epoch.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        unicode_292863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 6), 'unicode', u'Print the Epoch.')
        
        # Call to str(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'self' (line 205)
        self_292865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 18), 'self', False)
        # Processing the call keyword arguments (line 205)
        kwargs_292866 = {}
        # Getting the type of 'str' (line 205)
        str_292864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 13), 'str', False)
        # Calling str(args, kwargs) (line 205)
        str_call_result_292867 = invoke(stypy.reporting.localization.Localization(__file__, 205, 13), str_292864, *[self_292865], **kwargs_292866)
        
        # Assigning a type to the variable 'stypy_return_type' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 6), 'stypy_return_type', str_call_result_292867)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 203)
        stypy_return_type_292868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292868)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_292868


    @norecursion
    def range(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'range'
        module_type_store = module_type_store.open_function_context('range', 208, 3, False)
        # Assigning a type to the variable 'self' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        Epoch.range.__dict__.__setitem__('stypy_localization', localization)
        Epoch.range.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Epoch.range.__dict__.__setitem__('stypy_type_store', module_type_store)
        Epoch.range.__dict__.__setitem__('stypy_function_name', 'Epoch.range')
        Epoch.range.__dict__.__setitem__('stypy_param_names_list', ['stop', 'step'])
        Epoch.range.__dict__.__setitem__('stypy_varargs_param_name', None)
        Epoch.range.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Epoch.range.__dict__.__setitem__('stypy_call_defaults', defaults)
        Epoch.range.__dict__.__setitem__('stypy_call_varargs', varargs)
        Epoch.range.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Epoch.range.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Epoch.range', ['stop', 'step'], None, None, defaults, varargs, kwargs)

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

        unicode_292869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, (-1)), 'unicode', u'Generate a range of Epoch objects.\n\n      Similar to the Python range() method.  Returns the range [\n      start, stop ) at the requested step.  Each element will be a\n      Epoch object.\n\n      = INPUT VARIABLES\n      - start    The starting value of the range.\n      - stop     The stop value of the range.\n      - step     Step to use.\n\n      = RETURN VALUE\n      - Returns a list contianing the requested Epoch values.\n      ')
        
        # Assigning a List to a Name (line 223):
        
        # Obtaining an instance of the builtin type 'list' (line 223)
        list_292870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 223)
        
        # Assigning a type to the variable 'elems' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 6), 'elems', list_292870)
        
        # Assigning a Num to a Name (line 225):
        int_292871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 10), 'int')
        # Assigning a type to the variable 'i' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 6), 'i', int_292871)
        
        # Getting the type of 'True' (line 226)
        True_292872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'True')
        # Testing the type of an if condition (line 226)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 6), True_292872)
        # SSA begins for while statement (line 226)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a BinOp to a Name (line 227):
        # Getting the type of 'start' (line 227)
        start_292873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 13), 'start')
        # Getting the type of 'i' (line 227)
        i_292874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 21), 'i')
        # Getting the type of 'step' (line 227)
        step_292875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 25), 'step')
        # Applying the binary operator '*' (line 227)
        result_mul_292876 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 21), '*', i_292874, step_292875)
        
        # Applying the binary operator '+' (line 227)
        result_add_292877 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 13), '+', start_292873, result_mul_292876)
        
        # Assigning a type to the variable 'd' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 9), 'd', result_add_292877)
        
        
        # Getting the type of 'd' (line 228)
        d_292878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'd')
        # Getting the type of 'stop' (line 228)
        stop_292879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 17), 'stop')
        # Applying the binary operator '>=' (line 228)
        result_ge_292880 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 12), '>=', d_292878, stop_292879)
        
        # Testing the type of an if condition (line 228)
        if_condition_292881 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 9), result_ge_292880)
        # Assigning a type to the variable 'if_condition_292881' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 9), 'if_condition_292881', if_condition_292881)
        # SSA begins for if statement (line 228)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 228)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'd' (line 231)
        d_292884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 23), 'd', False)
        # Processing the call keyword arguments (line 231)
        kwargs_292885 = {}
        # Getting the type of 'elems' (line 231)
        elems_292882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 9), 'elems', False)
        # Obtaining the member 'append' of a type (line 231)
        append_292883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 9), elems_292882, 'append')
        # Calling append(args, kwargs) (line 231)
        append_call_result_292886 = invoke(stypy.reporting.localization.Localization(__file__, 231, 9), append_292883, *[d_292884], **kwargs_292885)
        
        
        # Getting the type of 'i' (line 232)
        i_292887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 9), 'i')
        int_292888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 14), 'int')
        # Applying the binary operator '+=' (line 232)
        result_iadd_292889 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 9), '+=', i_292887, int_292888)
        # Assigning a type to the variable 'i' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 9), 'i', result_iadd_292889)
        
        # SSA join for while statement (line 226)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'elems' (line 234)
        elems_292890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 13), 'elems')
        # Assigning a type to the variable 'stypy_return_type' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 6), 'stypy_return_type', elems_292890)
        
        # ################# End of 'range(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'range' in the type store
        # Getting the type of 'stypy_return_type' (line 208)
        stypy_return_type_292891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292891)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'range'
        return stypy_return_type_292891


# Assigning a type to the variable 'Epoch' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'Epoch', Epoch)

# Assigning a Dict to a Name (line 29):

# Obtaining an instance of the builtin type 'dict' (line 29)
dict_292892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 29)
# Adding element type (key, value) (line 29)
unicode_292893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 6), 'unicode', u'ET')

# Obtaining an instance of the builtin type 'dict' (line 30)
dict_292894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 30)
# Adding element type (key, value) (line 30)
unicode_292895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 9), 'unicode', u'UTC')

float_292896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 18), 'float')
# Applying the 'uadd' unary operator (line 31)
result___pos___292897 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 17), 'uadd', float_292896)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 13), dict_292894, (unicode_292895, result___pos___292897))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 13), dict_292892, (unicode_292893, dict_292894))
# Adding element type (key, value) (line 29)
unicode_292898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 6), 'unicode', u'UTC')

# Obtaining an instance of the builtin type 'dict' (line 33)
dict_292899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 33)
# Adding element type (key, value) (line 33)
unicode_292900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 9), 'unicode', u'ET')
float_292901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 16), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 14), dict_292899, (unicode_292900, float_292901))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 13), dict_292892, (unicode_292898, dict_292899))

# Getting the type of 'Epoch'
Epoch_292902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Epoch')
# Setting the type of the member 'allowed' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Epoch_292902, 'allowed', dict_292892)

# Assigning a Call to a Name (line 236):

# Call to staticmethod(...): (line 236)
# Processing the call arguments (line 236)
# Getting the type of 'Epoch'
Epoch_292904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Epoch', False)
# Obtaining the member 'range' of a type
range_292905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Epoch_292904, 'range')
# Processing the call keyword arguments (line 236)
kwargs_292906 = {}
# Getting the type of 'staticmethod' (line 236)
staticmethod_292903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 236)
staticmethod_call_result_292907 = invoke(stypy.reporting.localization.Localization(__file__, 236, 11), staticmethod_292903, *[range_292905], **kwargs_292906)

# Getting the type of 'Epoch'
Epoch_292908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Epoch')
# Setting the type of the member 'range' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Epoch_292908, 'range', staticmethod_call_result_292907)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
