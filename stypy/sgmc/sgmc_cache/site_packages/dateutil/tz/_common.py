
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from six import PY3
2: 
3: from functools import wraps
4: 
5: from datetime import datetime, timedelta, tzinfo
6: 
7: 
8: ZERO = timedelta(0)
9: 
10: __all__ = ['tzname_in_python2', 'enfold']
11: 
12: 
13: def tzname_in_python2(namefunc):
14:     '''Change unicode output into bytestrings in Python 2
15: 
16:     tzname() API changed in Python 3. It used to return bytes, but was changed
17:     to unicode strings
18:     '''
19:     def adjust_encoding(*args, **kwargs):
20:         name = namefunc(*args, **kwargs)
21:         if name is not None and not PY3:
22:             name = name.encode()
23: 
24:         return name
25: 
26:     return adjust_encoding
27: 
28: 
29: # The following is adapted from Alexander Belopolsky's tz library
30: # https://github.com/abalkin/tz
31: if hasattr(datetime, 'fold'):
32:     # This is the pre-python 3.6 fold situation
33:     def enfold(dt, fold=1):
34:         '''
35:         Provides a unified interface for assigning the ``fold`` attribute to
36:         datetimes both before and after the implementation of PEP-495.
37: 
38:         :param fold:
39:             The value for the ``fold`` attribute in the returned datetime. This
40:             should be either 0 or 1.
41: 
42:         :return:
43:             Returns an object for which ``getattr(dt, 'fold', 0)`` returns
44:             ``fold`` for all versions of Python. In versions prior to
45:             Python 3.6, this is a ``_DatetimeWithFold`` object, which is a
46:             subclass of :py:class:`datetime.datetime` with the ``fold``
47:             attribute added, if ``fold`` is 1.
48: 
49:         .. versionadded:: 2.6.0
50:         '''
51:         return dt.replace(fold=fold)
52: 
53: else:
54:     class _DatetimeWithFold(datetime):
55:         '''
56:         This is a class designed to provide a PEP 495-compliant interface for
57:         Python versions before 3.6. It is used only for dates in a fold, so
58:         the ``fold`` attribute is fixed at ``1``.
59: 
60:         .. versionadded:: 2.6.0
61:         '''
62:         __slots__ = ()
63: 
64:         @property
65:         def fold(self):
66:             return 1
67: 
68:     def enfold(dt, fold=1):
69:         '''
70:         Provides a unified interface for assigning the ``fold`` attribute to
71:         datetimes both before and after the implementation of PEP-495.
72: 
73:         :param fold:
74:             The value for the ``fold`` attribute in the returned datetime. This
75:             should be either 0 or 1.
76: 
77:         :return:
78:             Returns an object for which ``getattr(dt, 'fold', 0)`` returns
79:             ``fold`` for all versions of Python. In versions prior to
80:             Python 3.6, this is a ``_DatetimeWithFold`` object, which is a
81:             subclass of :py:class:`datetime.datetime` with the ``fold``
82:             attribute added, if ``fold`` is 1.
83: 
84:         .. versionadded:: 2.6.0
85:         '''
86:         if getattr(dt, 'fold', 0) == fold:
87:             return dt
88: 
89:         args = dt.timetuple()[:6]
90:         args += (dt.microsecond, dt.tzinfo)
91: 
92:         if fold:
93:             return _DatetimeWithFold(*args)
94:         else:
95:             return datetime(*args)
96: 
97: 
98: def _validate_fromutc_inputs(f):
99:     '''
100:     The CPython version of ``fromutc`` checks that the input is a ``datetime``
101:     object and that ``self`` is attached as its ``tzinfo``.
102:     '''
103:     @wraps(f)
104:     def fromutc(self, dt):
105:         if not isinstance(dt, datetime):
106:             raise TypeError("fromutc() requires a datetime argument")
107:         if dt.tzinfo is not self:
108:             raise ValueError("dt.tzinfo is not self")
109: 
110:         return f(self, dt)
111: 
112:     return fromutc
113: 
114: 
115: class _tzinfo(tzinfo):
116:     '''
117:     Base class for all ``dateutil`` ``tzinfo`` objects.
118:     '''
119: 
120:     def is_ambiguous(self, dt):
121:         '''
122:         Whether or not the "wall time" of a given datetime is ambiguous in this
123:         zone.
124: 
125:         :param dt:
126:             A :py:class:`datetime.datetime`, naive or time zone aware.
127: 
128: 
129:         :return:
130:             Returns ``True`` if ambiguous, ``False`` otherwise.
131: 
132:         .. versionadded:: 2.6.0
133:         '''
134: 
135:         dt = dt.replace(tzinfo=self)
136: 
137:         wall_0 = enfold(dt, fold=0)
138:         wall_1 = enfold(dt, fold=1)
139: 
140:         same_offset = wall_0.utcoffset() == wall_1.utcoffset()
141:         same_dt = wall_0.replace(tzinfo=None) == wall_1.replace(tzinfo=None)
142: 
143:         return same_dt and not same_offset
144: 
145:     def _fold_status(self, dt_utc, dt_wall):
146:         '''
147:         Determine the fold status of a "wall" datetime, given a representation
148:         of the same datetime as a (naive) UTC datetime. This is calculated based
149:         on the assumption that ``dt.utcoffset() - dt.dst()`` is constant for all
150:         datetimes, and that this offset is the actual number of hours separating
151:         ``dt_utc`` and ``dt_wall``.
152: 
153:         :param dt_utc:
154:             Representation of the datetime as UTC
155: 
156:         :param dt_wall:
157:             Representation of the datetime as "wall time". This parameter must
158:             either have a `fold` attribute or have a fold-naive
159:             :class:`datetime.tzinfo` attached, otherwise the calculation may
160:             fail.
161:         '''
162:         if self.is_ambiguous(dt_wall):
163:             delta_wall = dt_wall - dt_utc
164:             _fold = int(delta_wall == (dt_utc.utcoffset() - dt_utc.dst()))
165:         else:
166:             _fold = 0
167: 
168:         return _fold
169: 
170:     def _fold(self, dt):
171:         return getattr(dt, 'fold', 0)
172: 
173:     def _fromutc(self, dt):
174:         '''
175:         Given a timezone-aware datetime in a given timezone, calculates a
176:         timezone-aware datetime in a new timezone.
177: 
178:         Since this is the one time that we *know* we have an unambiguous
179:         datetime object, we take this opportunity to determine whether the
180:         datetime is ambiguous and in a "fold" state (e.g. if it's the first
181:         occurence, chronologically, of the ambiguous datetime).
182: 
183:         :param dt:
184:             A timezone-aware :class:`datetime.datetime` object.
185:         '''
186: 
187:         # Re-implement the algorithm from Python's datetime.py
188:         dtoff = dt.utcoffset()
189:         if dtoff is None:
190:             raise ValueError("fromutc() requires a non-None utcoffset() "
191:                              "result")
192: 
193:         # The original datetime.py code assumes that `dst()` defaults to
194:         # zero during ambiguous times. PEP 495 inverts this presumption, so
195:         # for pre-PEP 495 versions of python, we need to tweak the algorithm.
196:         dtdst = dt.dst()
197:         if dtdst is None:
198:             raise ValueError("fromutc() requires a non-None dst() result")
199:         delta = dtoff - dtdst
200: 
201:         dt += delta
202:         # Set fold=1 so we can default to being in the fold for
203:         # ambiguous dates.
204:         dtdst = enfold(dt, fold=1).dst()
205:         if dtdst is None:
206:             raise ValueError("fromutc(): dt.dst gave inconsistent "
207:                              "results; cannot convert")
208:         return dt + dtdst
209: 
210:     @_validate_fromutc_inputs
211:     def fromutc(self, dt):
212:         '''
213:         Given a timezone-aware datetime in a given timezone, calculates a
214:         timezone-aware datetime in a new timezone.
215: 
216:         Since this is the one time that we *know* we have an unambiguous
217:         datetime object, we take this opportunity to determine whether the
218:         datetime is ambiguous and in a "fold" state (e.g. if it's the first
219:         occurance, chronologically, of the ambiguous datetime).
220: 
221:         :param dt:
222:             A timezone-aware :class:`datetime.datetime` object.
223:         '''
224:         dt_wall = self._fromutc(dt)
225: 
226:         # Calculate the fold status given the two datetimes.
227:         _fold = self._fold_status(dt, dt_wall)
228: 
229:         # Set the default fold value for ambiguous dates
230:         return enfold(dt_wall, fold=_fold)
231: 
232: 
233: class tzrangebase(_tzinfo):
234:     '''
235:     This is an abstract base class for time zones represented by an annual
236:     transition into and out of DST. Child classes should implement the following
237:     methods:
238: 
239:         * ``__init__(self, *args, **kwargs)``
240:         * ``transitions(self, year)`` - this is expected to return a tuple of
241:           datetimes representing the DST on and off transitions in standard
242:           time.
243: 
244:     A fully initialized ``tzrangebase`` subclass should also provide the
245:     following attributes:
246:         * ``hasdst``: Boolean whether or not the zone uses DST.
247:         * ``_dst_offset`` / ``_std_offset``: :class:`datetime.timedelta` objects
248:           representing the respective UTC offsets.
249:         * ``_dst_abbr`` / ``_std_abbr``: Strings representing the timezone short
250:           abbreviations in DST and STD, respectively.
251:         * ``_hasdst``: Whether or not the zone has DST.
252: 
253:     .. versionadded:: 2.6.0
254:     '''
255:     def __init__(self):
256:         raise NotImplementedError('tzrangebase is an abstract base class')
257: 
258:     def utcoffset(self, dt):
259:         isdst = self._isdst(dt)
260: 
261:         if isdst is None:
262:             return None
263:         elif isdst:
264:             return self._dst_offset
265:         else:
266:             return self._std_offset
267: 
268:     def dst(self, dt):
269:         isdst = self._isdst(dt)
270: 
271:         if isdst is None:
272:             return None
273:         elif isdst:
274:             return self._dst_base_offset
275:         else:
276:             return ZERO
277: 
278:     @tzname_in_python2
279:     def tzname(self, dt):
280:         if self._isdst(dt):
281:             return self._dst_abbr
282:         else:
283:             return self._std_abbr
284: 
285:     def fromutc(self, dt):
286:         ''' Given a datetime in UTC, return local time '''
287:         if not isinstance(dt, datetime):
288:             raise TypeError("fromutc() requires a datetime argument")
289: 
290:         if dt.tzinfo is not self:
291:             raise ValueError("dt.tzinfo is not self")
292: 
293:         # Get transitions - if there are none, fixed offset
294:         transitions = self.transitions(dt.year)
295:         if transitions is None:
296:             return dt + self.utcoffset(dt)
297: 
298:         # Get the transition times in UTC
299:         dston, dstoff = transitions
300: 
301:         dston -= self._std_offset
302:         dstoff -= self._std_offset
303: 
304:         utc_transitions = (dston, dstoff)
305:         dt_utc = dt.replace(tzinfo=None)
306: 
307:         isdst = self._naive_isdst(dt_utc, utc_transitions)
308: 
309:         if isdst:
310:             dt_wall = dt + self._dst_offset
311:         else:
312:             dt_wall = dt + self._std_offset
313: 
314:         _fold = int(not isdst and self.is_ambiguous(dt_wall))
315: 
316:         return enfold(dt_wall, fold=_fold)
317: 
318:     def is_ambiguous(self, dt):
319:         '''
320:         Whether or not the "wall time" of a given datetime is ambiguous in this
321:         zone.
322: 
323:         :param dt:
324:             A :py:class:`datetime.datetime`, naive or time zone aware.
325: 
326: 
327:         :return:
328:             Returns ``True`` if ambiguous, ``False`` otherwise.
329: 
330:         .. versionadded:: 2.6.0
331:         '''
332:         if not self.hasdst:
333:             return False
334: 
335:         start, end = self.transitions(dt.year)
336: 
337:         dt = dt.replace(tzinfo=None)
338:         return (end <= dt < end + self._dst_base_offset)
339: 
340:     def _isdst(self, dt):
341:         if not self.hasdst:
342:             return False
343:         elif dt is None:
344:             return None
345: 
346:         transitions = self.transitions(dt.year)
347: 
348:         if transitions is None:
349:             return False
350: 
351:         dt = dt.replace(tzinfo=None)
352: 
353:         isdst = self._naive_isdst(dt, transitions)
354: 
355:         # Handle ambiguous dates
356:         if not isdst and self.is_ambiguous(dt):
357:             return not self._fold(dt)
358:         else:
359:             return isdst
360: 
361:     def _naive_isdst(self, dt, transitions):
362:         dston, dstoff = transitions
363: 
364:         dt = dt.replace(tzinfo=None)
365: 
366:         if dston < dstoff:
367:             isdst = dston <= dt < dstoff
368:         else:
369:             isdst = not dstoff <= dt < dston
370: 
371:         return isdst
372: 
373:     @property
374:     def _dst_base_offset(self):
375:         return self._dst_offset - self._std_offset
376: 
377:     __hash__ = None
378: 
379:     def __ne__(self, other):
380:         return not (self == other)
381: 
382:     def __repr__(self):
383:         return "%s(...)" % self.__class__.__name__
384: 
385:     __reduce__ = object.__reduce__
386: 
387: 
388: def _total_seconds(td):
389:     # Python 2.6 doesn't have a total_seconds() method on timedelta objects
390:     return ((td.seconds + td.days * 86400) * 1000000 +
391:             td.microseconds) // 1000000
392: 
393: 
394: _total_seconds = getattr(timedelta, 'total_seconds', _total_seconds)
395: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from six import PY3' statement (line 1)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/dateutil/tz/')
import_324224 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'six')

if (type(import_324224) is not StypyTypeError):

    if (import_324224 != 'pyd_module'):
        __import__(import_324224)
        sys_modules_324225 = sys.modules[import_324224]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'six', sys_modules_324225.module_type_store, module_type_store, ['PY3'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_324225, sys_modules_324225.module_type_store, module_type_store)
    else:
        from six import PY3

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'six', None, module_type_store, ['PY3'], [PY3])

else:
    # Assigning a type to the variable 'six' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'six', import_324224)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/dateutil/tz/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from functools import wraps' statement (line 3)
try:
    from functools import wraps

except:
    wraps = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'functools', None, module_type_store, ['wraps'], [wraps])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from datetime import datetime, timedelta, tzinfo' statement (line 5)
try:
    from datetime import datetime, timedelta, tzinfo

except:
    datetime = UndefinedType
    timedelta = UndefinedType
    tzinfo = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'datetime', None, module_type_store, ['datetime', 'timedelta', 'tzinfo'], [datetime, timedelta, tzinfo])


# Assigning a Call to a Name (line 8):

# Assigning a Call to a Name (line 8):

# Call to timedelta(...): (line 8)
# Processing the call arguments (line 8)
int_324227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 17), 'int')
# Processing the call keyword arguments (line 8)
kwargs_324228 = {}
# Getting the type of 'timedelta' (line 8)
timedelta_324226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 7), 'timedelta', False)
# Calling timedelta(args, kwargs) (line 8)
timedelta_call_result_324229 = invoke(stypy.reporting.localization.Localization(__file__, 8, 7), timedelta_324226, *[int_324227], **kwargs_324228)

# Assigning a type to the variable 'ZERO' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'ZERO', timedelta_call_result_324229)

# Assigning a List to a Name (line 10):

# Assigning a List to a Name (line 10):
__all__ = ['tzname_in_python2', 'enfold']
module_type_store.set_exportable_members(['tzname_in_python2', 'enfold'])

# Obtaining an instance of the builtin type 'list' (line 10)
list_324230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
str_324231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 11), 'str', 'tzname_in_python2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_324230, str_324231)
# Adding element type (line 10)
str_324232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 32), 'str', 'enfold')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_324230, str_324232)

# Assigning a type to the variable '__all__' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), '__all__', list_324230)

@norecursion
def tzname_in_python2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'tzname_in_python2'
    module_type_store = module_type_store.open_function_context('tzname_in_python2', 13, 0, False)
    
    # Passed parameters checking function
    tzname_in_python2.stypy_localization = localization
    tzname_in_python2.stypy_type_of_self = None
    tzname_in_python2.stypy_type_store = module_type_store
    tzname_in_python2.stypy_function_name = 'tzname_in_python2'
    tzname_in_python2.stypy_param_names_list = ['namefunc']
    tzname_in_python2.stypy_varargs_param_name = None
    tzname_in_python2.stypy_kwargs_param_name = None
    tzname_in_python2.stypy_call_defaults = defaults
    tzname_in_python2.stypy_call_varargs = varargs
    tzname_in_python2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tzname_in_python2', ['namefunc'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tzname_in_python2', localization, ['namefunc'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tzname_in_python2(...)' code ##################

    str_324233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, (-1)), 'str', 'Change unicode output into bytestrings in Python 2\n\n    tzname() API changed in Python 3. It used to return bytes, but was changed\n    to unicode strings\n    ')

    @norecursion
    def adjust_encoding(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'adjust_encoding'
        module_type_store = module_type_store.open_function_context('adjust_encoding', 19, 4, False)
        
        # Passed parameters checking function
        adjust_encoding.stypy_localization = localization
        adjust_encoding.stypy_type_of_self = None
        adjust_encoding.stypy_type_store = module_type_store
        adjust_encoding.stypy_function_name = 'adjust_encoding'
        adjust_encoding.stypy_param_names_list = []
        adjust_encoding.stypy_varargs_param_name = 'args'
        adjust_encoding.stypy_kwargs_param_name = 'kwargs'
        adjust_encoding.stypy_call_defaults = defaults
        adjust_encoding.stypy_call_varargs = varargs
        adjust_encoding.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'adjust_encoding', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'adjust_encoding', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'adjust_encoding(...)' code ##################

        
        # Assigning a Call to a Name (line 20):
        
        # Assigning a Call to a Name (line 20):
        
        # Call to namefunc(...): (line 20)
        # Getting the type of 'args' (line 20)
        args_324235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 25), 'args', False)
        # Processing the call keyword arguments (line 20)
        # Getting the type of 'kwargs' (line 20)
        kwargs_324236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 33), 'kwargs', False)
        kwargs_324237 = {'kwargs_324236': kwargs_324236}
        # Getting the type of 'namefunc' (line 20)
        namefunc_324234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 15), 'namefunc', False)
        # Calling namefunc(args, kwargs) (line 20)
        namefunc_call_result_324238 = invoke(stypy.reporting.localization.Localization(__file__, 20, 15), namefunc_324234, *[args_324235], **kwargs_324237)
        
        # Assigning a type to the variable 'name' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'name', namefunc_call_result_324238)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'name' (line 21)
        name_324239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), 'name')
        # Getting the type of 'None' (line 21)
        None_324240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 23), 'None')
        # Applying the binary operator 'isnot' (line 21)
        result_is_not_324241 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 11), 'isnot', name_324239, None_324240)
        
        
        # Getting the type of 'PY3' (line 21)
        PY3_324242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 36), 'PY3')
        # Applying the 'not' unary operator (line 21)
        result_not__324243 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 32), 'not', PY3_324242)
        
        # Applying the binary operator 'and' (line 21)
        result_and_keyword_324244 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 11), 'and', result_is_not_324241, result_not__324243)
        
        # Testing the type of an if condition (line 21)
        if_condition_324245 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 21, 8), result_and_keyword_324244)
        # Assigning a type to the variable 'if_condition_324245' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'if_condition_324245', if_condition_324245)
        # SSA begins for if statement (line 21)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 22):
        
        # Assigning a Call to a Name (line 22):
        
        # Call to encode(...): (line 22)
        # Processing the call keyword arguments (line 22)
        kwargs_324248 = {}
        # Getting the type of 'name' (line 22)
        name_324246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 19), 'name', False)
        # Obtaining the member 'encode' of a type (line 22)
        encode_324247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 19), name_324246, 'encode')
        # Calling encode(args, kwargs) (line 22)
        encode_call_result_324249 = invoke(stypy.reporting.localization.Localization(__file__, 22, 19), encode_324247, *[], **kwargs_324248)
        
        # Assigning a type to the variable 'name' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'name', encode_call_result_324249)
        # SSA join for if statement (line 21)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'name' (line 24)
        name_324250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 15), 'name')
        # Assigning a type to the variable 'stypy_return_type' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'stypy_return_type', name_324250)
        
        # ################# End of 'adjust_encoding(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'adjust_encoding' in the type store
        # Getting the type of 'stypy_return_type' (line 19)
        stypy_return_type_324251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324251)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'adjust_encoding'
        return stypy_return_type_324251

    # Assigning a type to the variable 'adjust_encoding' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'adjust_encoding', adjust_encoding)
    # Getting the type of 'adjust_encoding' (line 26)
    adjust_encoding_324252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'adjust_encoding')
    # Assigning a type to the variable 'stypy_return_type' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type', adjust_encoding_324252)
    
    # ################# End of 'tzname_in_python2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tzname_in_python2' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_324253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_324253)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tzname_in_python2'
    return stypy_return_type_324253

# Assigning a type to the variable 'tzname_in_python2' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'tzname_in_python2', tzname_in_python2)

# Type idiom detected: calculating its left and rigth part (line 31)
str_324254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 21), 'str', 'fold')
# Getting the type of 'datetime' (line 31)
datetime_324255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'datetime')

(may_be_324256, more_types_in_union_324257) = may_provide_member(str_324254, datetime_324255)

if may_be_324256:

    if more_types_in_union_324257:
        # Runtime conditional SSA (line 31)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    # Assigning a type to the variable 'datetime' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'datetime', remove_not_member_provider_from_union(datetime_324255, 'fold'))

    @norecursion
    def enfold(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_324258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 24), 'int')
        defaults = [int_324258]
        # Create a new context for function 'enfold'
        module_type_store = module_type_store.open_function_context('enfold', 33, 4, False)
        
        # Passed parameters checking function
        enfold.stypy_localization = localization
        enfold.stypy_type_of_self = None
        enfold.stypy_type_store = module_type_store
        enfold.stypy_function_name = 'enfold'
        enfold.stypy_param_names_list = ['dt', 'fold']
        enfold.stypy_varargs_param_name = None
        enfold.stypy_kwargs_param_name = None
        enfold.stypy_call_defaults = defaults
        enfold.stypy_call_varargs = varargs
        enfold.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'enfold', ['dt', 'fold'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'enfold', localization, ['dt', 'fold'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'enfold(...)' code ##################

        str_324259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, (-1)), 'str', "\n        Provides a unified interface for assigning the ``fold`` attribute to\n        datetimes both before and after the implementation of PEP-495.\n\n        :param fold:\n            The value for the ``fold`` attribute in the returned datetime. This\n            should be either 0 or 1.\n\n        :return:\n            Returns an object for which ``getattr(dt, 'fold', 0)`` returns\n            ``fold`` for all versions of Python. In versions prior to\n            Python 3.6, this is a ``_DatetimeWithFold`` object, which is a\n            subclass of :py:class:`datetime.datetime` with the ``fold``\n            attribute added, if ``fold`` is 1.\n\n        .. versionadded:: 2.6.0\n        ")
        
        # Call to replace(...): (line 51)
        # Processing the call keyword arguments (line 51)
        # Getting the type of 'fold' (line 51)
        fold_324262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 31), 'fold', False)
        keyword_324263 = fold_324262
        kwargs_324264 = {'fold': keyword_324263}
        # Getting the type of 'dt' (line 51)
        dt_324260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'dt', False)
        # Obtaining the member 'replace' of a type (line 51)
        replace_324261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 15), dt_324260, 'replace')
        # Calling replace(args, kwargs) (line 51)
        replace_call_result_324265 = invoke(stypy.reporting.localization.Localization(__file__, 51, 15), replace_324261, *[], **kwargs_324264)
        
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', replace_call_result_324265)
        
        # ################# End of 'enfold(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'enfold' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_324266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324266)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'enfold'
        return stypy_return_type_324266

    # Assigning a type to the variable 'enfold' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'enfold', enfold)

    if more_types_in_union_324257:
        # Runtime conditional SSA for else branch (line 31)
        module_type_store.open_ssa_branch('idiom else')



if ((not may_be_324256) or more_types_in_union_324257):
    # Assigning a type to the variable 'datetime' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'datetime', remove_member_provider_from_union(datetime_324255, 'fold'))
    # Declaration of the '_DatetimeWithFold' class
    # Getting the type of 'datetime' (line 54)
    datetime_324267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 28), 'datetime')

    class _DatetimeWithFold(datetime_324267, ):
        str_324268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, (-1)), 'str', '\n        This is a class designed to provide a PEP 495-compliant interface for\n        Python versions before 3.6. It is used only for dates in a fold, so\n        the ``fold`` attribute is fixed at ``1``.\n\n        .. versionadded:: 2.6.0\n        ')
        
        # Assigning a Tuple to a Name (line 62):

        @norecursion
        def fold(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'fold'
            module_type_store = module_type_store.open_function_context('fold', 64, 8, False)
            # Assigning a type to the variable 'self' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            _DatetimeWithFold.fold.__dict__.__setitem__('stypy_localization', localization)
            _DatetimeWithFold.fold.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            _DatetimeWithFold.fold.__dict__.__setitem__('stypy_type_store', module_type_store)
            _DatetimeWithFold.fold.__dict__.__setitem__('stypy_function_name', '_DatetimeWithFold.fold')
            _DatetimeWithFold.fold.__dict__.__setitem__('stypy_param_names_list', [])
            _DatetimeWithFold.fold.__dict__.__setitem__('stypy_varargs_param_name', None)
            _DatetimeWithFold.fold.__dict__.__setitem__('stypy_kwargs_param_name', None)
            _DatetimeWithFold.fold.__dict__.__setitem__('stypy_call_defaults', defaults)
            _DatetimeWithFold.fold.__dict__.__setitem__('stypy_call_varargs', varargs)
            _DatetimeWithFold.fold.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            _DatetimeWithFold.fold.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, '_DatetimeWithFold.fold', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'fold', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'fold(...)' code ##################

            int_324269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 66)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'stypy_return_type', int_324269)
            
            # ################# End of 'fold(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'fold' in the type store
            # Getting the type of 'stypy_return_type' (line 64)
            stypy_return_type_324270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_324270)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'fold'
            return stypy_return_type_324270


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 54, 4, False)
            # Assigning a type to the variable 'self' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, '_DatetimeWithFold.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable '_DatetimeWithFold' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), '_DatetimeWithFold', _DatetimeWithFold)
    
    # Assigning a Tuple to a Name (line 62):
    
    # Obtaining an instance of the builtin type 'tuple' (line 62)
    tuple_324271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 62)
    
    # Getting the type of '_DatetimeWithFold'
    _DatetimeWithFold_324272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_DatetimeWithFold')
    # Setting the type of the member '__slots__' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _DatetimeWithFold_324272, '__slots__', tuple_324271)

    @norecursion
    def enfold(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_324273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 24), 'int')
        defaults = [int_324273]
        # Create a new context for function 'enfold'
        module_type_store = module_type_store.open_function_context('enfold', 68, 4, False)
        
        # Passed parameters checking function
        enfold.stypy_localization = localization
        enfold.stypy_type_of_self = None
        enfold.stypy_type_store = module_type_store
        enfold.stypy_function_name = 'enfold'
        enfold.stypy_param_names_list = ['dt', 'fold']
        enfold.stypy_varargs_param_name = None
        enfold.stypy_kwargs_param_name = None
        enfold.stypy_call_defaults = defaults
        enfold.stypy_call_varargs = varargs
        enfold.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'enfold', ['dt', 'fold'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'enfold', localization, ['dt', 'fold'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'enfold(...)' code ##################

        str_324274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, (-1)), 'str', "\n        Provides a unified interface for assigning the ``fold`` attribute to\n        datetimes both before and after the implementation of PEP-495.\n\n        :param fold:\n            The value for the ``fold`` attribute in the returned datetime. This\n            should be either 0 or 1.\n\n        :return:\n            Returns an object for which ``getattr(dt, 'fold', 0)`` returns\n            ``fold`` for all versions of Python. In versions prior to\n            Python 3.6, this is a ``_DatetimeWithFold`` object, which is a\n            subclass of :py:class:`datetime.datetime` with the ``fold``\n            attribute added, if ``fold`` is 1.\n\n        .. versionadded:: 2.6.0\n        ")
        
        
        
        # Call to getattr(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'dt' (line 86)
        dt_324276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'dt', False)
        str_324277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 23), 'str', 'fold')
        int_324278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 31), 'int')
        # Processing the call keyword arguments (line 86)
        kwargs_324279 = {}
        # Getting the type of 'getattr' (line 86)
        getattr_324275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 11), 'getattr', False)
        # Calling getattr(args, kwargs) (line 86)
        getattr_call_result_324280 = invoke(stypy.reporting.localization.Localization(__file__, 86, 11), getattr_324275, *[dt_324276, str_324277, int_324278], **kwargs_324279)
        
        # Getting the type of 'fold' (line 86)
        fold_324281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 37), 'fold')
        # Applying the binary operator '==' (line 86)
        result_eq_324282 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 11), '==', getattr_call_result_324280, fold_324281)
        
        # Testing the type of an if condition (line 86)
        if_condition_324283 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 8), result_eq_324282)
        # Assigning a type to the variable 'if_condition_324283' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'if_condition_324283', if_condition_324283)
        # SSA begins for if statement (line 86)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'dt' (line 87)
        dt_324284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 19), 'dt')
        # Assigning a type to the variable 'stypy_return_type' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'stypy_return_type', dt_324284)
        # SSA join for if statement (line 86)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 89):
        
        # Assigning a Subscript to a Name (line 89):
        
        # Obtaining the type of the subscript
        int_324285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 31), 'int')
        slice_324286 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 89, 15), None, int_324285, None)
        
        # Call to timetuple(...): (line 89)
        # Processing the call keyword arguments (line 89)
        kwargs_324289 = {}
        # Getting the type of 'dt' (line 89)
        dt_324287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'dt', False)
        # Obtaining the member 'timetuple' of a type (line 89)
        timetuple_324288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 15), dt_324287, 'timetuple')
        # Calling timetuple(args, kwargs) (line 89)
        timetuple_call_result_324290 = invoke(stypy.reporting.localization.Localization(__file__, 89, 15), timetuple_324288, *[], **kwargs_324289)
        
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___324291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 15), timetuple_call_result_324290, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 89)
        subscript_call_result_324292 = invoke(stypy.reporting.localization.Localization(__file__, 89, 15), getitem___324291, slice_324286)
        
        # Assigning a type to the variable 'args' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'args', subscript_call_result_324292)
        
        # Getting the type of 'args' (line 90)
        args_324293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'args')
        
        # Obtaining an instance of the builtin type 'tuple' (line 90)
        tuple_324294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 90)
        # Adding element type (line 90)
        # Getting the type of 'dt' (line 90)
        dt_324295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 17), 'dt')
        # Obtaining the member 'microsecond' of a type (line 90)
        microsecond_324296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 17), dt_324295, 'microsecond')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 17), tuple_324294, microsecond_324296)
        # Adding element type (line 90)
        # Getting the type of 'dt' (line 90)
        dt_324297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 33), 'dt')
        # Obtaining the member 'tzinfo' of a type (line 90)
        tzinfo_324298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 33), dt_324297, 'tzinfo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 17), tuple_324294, tzinfo_324298)
        
        # Applying the binary operator '+=' (line 90)
        result_iadd_324299 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 8), '+=', args_324293, tuple_324294)
        # Assigning a type to the variable 'args' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'args', result_iadd_324299)
        
        
        # Getting the type of 'fold' (line 92)
        fold_324300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'fold')
        # Testing the type of an if condition (line 92)
        if_condition_324301 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 8), fold_324300)
        # Assigning a type to the variable 'if_condition_324301' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'if_condition_324301', if_condition_324301)
        # SSA begins for if statement (line 92)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _DatetimeWithFold(...): (line 93)
        # Getting the type of 'args' (line 93)
        args_324303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 38), 'args', False)
        # Processing the call keyword arguments (line 93)
        kwargs_324304 = {}
        # Getting the type of '_DatetimeWithFold' (line 93)
        _DatetimeWithFold_324302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 19), '_DatetimeWithFold', False)
        # Calling _DatetimeWithFold(args, kwargs) (line 93)
        _DatetimeWithFold_call_result_324305 = invoke(stypy.reporting.localization.Localization(__file__, 93, 19), _DatetimeWithFold_324302, *[args_324303], **kwargs_324304)
        
        # Assigning a type to the variable 'stypy_return_type' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'stypy_return_type', _DatetimeWithFold_call_result_324305)
        # SSA branch for the else part of an if statement (line 92)
        module_type_store.open_ssa_branch('else')
        
        # Call to datetime(...): (line 95)
        # Getting the type of 'args' (line 95)
        args_324307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 29), 'args', False)
        # Processing the call keyword arguments (line 95)
        kwargs_324308 = {}
        # Getting the type of 'datetime' (line 95)
        datetime_324306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 19), 'datetime', False)
        # Calling datetime(args, kwargs) (line 95)
        datetime_call_result_324309 = invoke(stypy.reporting.localization.Localization(__file__, 95, 19), datetime_324306, *[args_324307], **kwargs_324308)
        
        # Assigning a type to the variable 'stypy_return_type' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'stypy_return_type', datetime_call_result_324309)
        # SSA join for if statement (line 92)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'enfold(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'enfold' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_324310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324310)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'enfold'
        return stypy_return_type_324310

    # Assigning a type to the variable 'enfold' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'enfold', enfold)

    if (may_be_324256 and more_types_in_union_324257):
        # SSA join for if statement (line 31)
        module_type_store = module_type_store.join_ssa_context()




@norecursion
def _validate_fromutc_inputs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_validate_fromutc_inputs'
    module_type_store = module_type_store.open_function_context('_validate_fromutc_inputs', 98, 0, False)
    
    # Passed parameters checking function
    _validate_fromutc_inputs.stypy_localization = localization
    _validate_fromutc_inputs.stypy_type_of_self = None
    _validate_fromutc_inputs.stypy_type_store = module_type_store
    _validate_fromutc_inputs.stypy_function_name = '_validate_fromutc_inputs'
    _validate_fromutc_inputs.stypy_param_names_list = ['f']
    _validate_fromutc_inputs.stypy_varargs_param_name = None
    _validate_fromutc_inputs.stypy_kwargs_param_name = None
    _validate_fromutc_inputs.stypy_call_defaults = defaults
    _validate_fromutc_inputs.stypy_call_varargs = varargs
    _validate_fromutc_inputs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_validate_fromutc_inputs', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_validate_fromutc_inputs', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_validate_fromutc_inputs(...)' code ##################

    str_324311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, (-1)), 'str', '\n    The CPython version of ``fromutc`` checks that the input is a ``datetime``\n    object and that ``self`` is attached as its ``tzinfo``.\n    ')

    @norecursion
    def fromutc(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fromutc'
        module_type_store = module_type_store.open_function_context('fromutc', 103, 4, False)
        
        # Passed parameters checking function
        fromutc.stypy_localization = localization
        fromutc.stypy_type_of_self = None
        fromutc.stypy_type_store = module_type_store
        fromutc.stypy_function_name = 'fromutc'
        fromutc.stypy_param_names_list = ['self', 'dt']
        fromutc.stypy_varargs_param_name = None
        fromutc.stypy_kwargs_param_name = None
        fromutc.stypy_call_defaults = defaults
        fromutc.stypy_call_varargs = varargs
        fromutc.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'fromutc', ['self', 'dt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fromutc', localization, ['self', 'dt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fromutc(...)' code ##################

        
        
        
        # Call to isinstance(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'dt' (line 105)
        dt_324313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 26), 'dt', False)
        # Getting the type of 'datetime' (line 105)
        datetime_324314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 30), 'datetime', False)
        # Processing the call keyword arguments (line 105)
        kwargs_324315 = {}
        # Getting the type of 'isinstance' (line 105)
        isinstance_324312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 105)
        isinstance_call_result_324316 = invoke(stypy.reporting.localization.Localization(__file__, 105, 15), isinstance_324312, *[dt_324313, datetime_324314], **kwargs_324315)
        
        # Applying the 'not' unary operator (line 105)
        result_not__324317 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 11), 'not', isinstance_call_result_324316)
        
        # Testing the type of an if condition (line 105)
        if_condition_324318 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 8), result_not__324317)
        # Assigning a type to the variable 'if_condition_324318' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'if_condition_324318', if_condition_324318)
        # SSA begins for if statement (line 105)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 106)
        # Processing the call arguments (line 106)
        str_324320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 28), 'str', 'fromutc() requires a datetime argument')
        # Processing the call keyword arguments (line 106)
        kwargs_324321 = {}
        # Getting the type of 'TypeError' (line 106)
        TypeError_324319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 106)
        TypeError_call_result_324322 = invoke(stypy.reporting.localization.Localization(__file__, 106, 18), TypeError_324319, *[str_324320], **kwargs_324321)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 106, 12), TypeError_call_result_324322, 'raise parameter', BaseException)
        # SSA join for if statement (line 105)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'dt' (line 107)
        dt_324323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), 'dt')
        # Obtaining the member 'tzinfo' of a type (line 107)
        tzinfo_324324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 11), dt_324323, 'tzinfo')
        # Getting the type of 'self' (line 107)
        self_324325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 28), 'self')
        # Applying the binary operator 'isnot' (line 107)
        result_is_not_324326 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 11), 'isnot', tzinfo_324324, self_324325)
        
        # Testing the type of an if condition (line 107)
        if_condition_324327 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 8), result_is_not_324326)
        # Assigning a type to the variable 'if_condition_324327' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'if_condition_324327', if_condition_324327)
        # SSA begins for if statement (line 107)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 108)
        # Processing the call arguments (line 108)
        str_324329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 29), 'str', 'dt.tzinfo is not self')
        # Processing the call keyword arguments (line 108)
        kwargs_324330 = {}
        # Getting the type of 'ValueError' (line 108)
        ValueError_324328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 108)
        ValueError_call_result_324331 = invoke(stypy.reporting.localization.Localization(__file__, 108, 18), ValueError_324328, *[str_324329], **kwargs_324330)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 108, 12), ValueError_call_result_324331, 'raise parameter', BaseException)
        # SSA join for if statement (line 107)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to f(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'self' (line 110)
        self_324333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 17), 'self', False)
        # Getting the type of 'dt' (line 110)
        dt_324334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 23), 'dt', False)
        # Processing the call keyword arguments (line 110)
        kwargs_324335 = {}
        # Getting the type of 'f' (line 110)
        f_324332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 15), 'f', False)
        # Calling f(args, kwargs) (line 110)
        f_call_result_324336 = invoke(stypy.reporting.localization.Localization(__file__, 110, 15), f_324332, *[self_324333, dt_324334], **kwargs_324335)
        
        # Assigning a type to the variable 'stypy_return_type' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'stypy_return_type', f_call_result_324336)
        
        # ################# End of 'fromutc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fromutc' in the type store
        # Getting the type of 'stypy_return_type' (line 103)
        stypy_return_type_324337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324337)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fromutc'
        return stypy_return_type_324337

    # Assigning a type to the variable 'fromutc' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'fromutc', fromutc)
    # Getting the type of 'fromutc' (line 112)
    fromutc_324338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'fromutc')
    # Assigning a type to the variable 'stypy_return_type' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type', fromutc_324338)
    
    # ################# End of '_validate_fromutc_inputs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_validate_fromutc_inputs' in the type store
    # Getting the type of 'stypy_return_type' (line 98)
    stypy_return_type_324339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_324339)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_validate_fromutc_inputs'
    return stypy_return_type_324339

# Assigning a type to the variable '_validate_fromutc_inputs' (line 98)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), '_validate_fromutc_inputs', _validate_fromutc_inputs)
# Declaration of the '_tzinfo' class
# Getting the type of 'tzinfo' (line 115)
tzinfo_324340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 14), 'tzinfo')

class _tzinfo(tzinfo_324340, ):
    str_324341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, (-1)), 'str', '\n    Base class for all ``dateutil`` ``tzinfo`` objects.\n    ')

    @norecursion
    def is_ambiguous(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_ambiguous'
        module_type_store = module_type_store.open_function_context('is_ambiguous', 120, 4, False)
        # Assigning a type to the variable 'self' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _tzinfo.is_ambiguous.__dict__.__setitem__('stypy_localization', localization)
        _tzinfo.is_ambiguous.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _tzinfo.is_ambiguous.__dict__.__setitem__('stypy_type_store', module_type_store)
        _tzinfo.is_ambiguous.__dict__.__setitem__('stypy_function_name', '_tzinfo.is_ambiguous')
        _tzinfo.is_ambiguous.__dict__.__setitem__('stypy_param_names_list', ['dt'])
        _tzinfo.is_ambiguous.__dict__.__setitem__('stypy_varargs_param_name', None)
        _tzinfo.is_ambiguous.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _tzinfo.is_ambiguous.__dict__.__setitem__('stypy_call_defaults', defaults)
        _tzinfo.is_ambiguous.__dict__.__setitem__('stypy_call_varargs', varargs)
        _tzinfo.is_ambiguous.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _tzinfo.is_ambiguous.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_tzinfo.is_ambiguous', ['dt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_ambiguous', localization, ['dt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_ambiguous(...)' code ##################

        str_324342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, (-1)), 'str', '\n        Whether or not the "wall time" of a given datetime is ambiguous in this\n        zone.\n\n        :param dt:\n            A :py:class:`datetime.datetime`, naive or time zone aware.\n\n\n        :return:\n            Returns ``True`` if ambiguous, ``False`` otherwise.\n\n        .. versionadded:: 2.6.0\n        ')
        
        # Assigning a Call to a Name (line 135):
        
        # Assigning a Call to a Name (line 135):
        
        # Call to replace(...): (line 135)
        # Processing the call keyword arguments (line 135)
        # Getting the type of 'self' (line 135)
        self_324345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 31), 'self', False)
        keyword_324346 = self_324345
        kwargs_324347 = {'tzinfo': keyword_324346}
        # Getting the type of 'dt' (line 135)
        dt_324343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 13), 'dt', False)
        # Obtaining the member 'replace' of a type (line 135)
        replace_324344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 13), dt_324343, 'replace')
        # Calling replace(args, kwargs) (line 135)
        replace_call_result_324348 = invoke(stypy.reporting.localization.Localization(__file__, 135, 13), replace_324344, *[], **kwargs_324347)
        
        # Assigning a type to the variable 'dt' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'dt', replace_call_result_324348)
        
        # Assigning a Call to a Name (line 137):
        
        # Assigning a Call to a Name (line 137):
        
        # Call to enfold(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'dt' (line 137)
        dt_324350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 24), 'dt', False)
        # Processing the call keyword arguments (line 137)
        int_324351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 33), 'int')
        keyword_324352 = int_324351
        kwargs_324353 = {'fold': keyword_324352}
        # Getting the type of 'enfold' (line 137)
        enfold_324349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 17), 'enfold', False)
        # Calling enfold(args, kwargs) (line 137)
        enfold_call_result_324354 = invoke(stypy.reporting.localization.Localization(__file__, 137, 17), enfold_324349, *[dt_324350], **kwargs_324353)
        
        # Assigning a type to the variable 'wall_0' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'wall_0', enfold_call_result_324354)
        
        # Assigning a Call to a Name (line 138):
        
        # Assigning a Call to a Name (line 138):
        
        # Call to enfold(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'dt' (line 138)
        dt_324356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 24), 'dt', False)
        # Processing the call keyword arguments (line 138)
        int_324357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 33), 'int')
        keyword_324358 = int_324357
        kwargs_324359 = {'fold': keyword_324358}
        # Getting the type of 'enfold' (line 138)
        enfold_324355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 17), 'enfold', False)
        # Calling enfold(args, kwargs) (line 138)
        enfold_call_result_324360 = invoke(stypy.reporting.localization.Localization(__file__, 138, 17), enfold_324355, *[dt_324356], **kwargs_324359)
        
        # Assigning a type to the variable 'wall_1' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'wall_1', enfold_call_result_324360)
        
        # Assigning a Compare to a Name (line 140):
        
        # Assigning a Compare to a Name (line 140):
        
        
        # Call to utcoffset(...): (line 140)
        # Processing the call keyword arguments (line 140)
        kwargs_324363 = {}
        # Getting the type of 'wall_0' (line 140)
        wall_0_324361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 22), 'wall_0', False)
        # Obtaining the member 'utcoffset' of a type (line 140)
        utcoffset_324362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 22), wall_0_324361, 'utcoffset')
        # Calling utcoffset(args, kwargs) (line 140)
        utcoffset_call_result_324364 = invoke(stypy.reporting.localization.Localization(__file__, 140, 22), utcoffset_324362, *[], **kwargs_324363)
        
        
        # Call to utcoffset(...): (line 140)
        # Processing the call keyword arguments (line 140)
        kwargs_324367 = {}
        # Getting the type of 'wall_1' (line 140)
        wall_1_324365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 44), 'wall_1', False)
        # Obtaining the member 'utcoffset' of a type (line 140)
        utcoffset_324366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 44), wall_1_324365, 'utcoffset')
        # Calling utcoffset(args, kwargs) (line 140)
        utcoffset_call_result_324368 = invoke(stypy.reporting.localization.Localization(__file__, 140, 44), utcoffset_324366, *[], **kwargs_324367)
        
        # Applying the binary operator '==' (line 140)
        result_eq_324369 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 22), '==', utcoffset_call_result_324364, utcoffset_call_result_324368)
        
        # Assigning a type to the variable 'same_offset' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'same_offset', result_eq_324369)
        
        # Assigning a Compare to a Name (line 141):
        
        # Assigning a Compare to a Name (line 141):
        
        
        # Call to replace(...): (line 141)
        # Processing the call keyword arguments (line 141)
        # Getting the type of 'None' (line 141)
        None_324372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 40), 'None', False)
        keyword_324373 = None_324372
        kwargs_324374 = {'tzinfo': keyword_324373}
        # Getting the type of 'wall_0' (line 141)
        wall_0_324370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 18), 'wall_0', False)
        # Obtaining the member 'replace' of a type (line 141)
        replace_324371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 18), wall_0_324370, 'replace')
        # Calling replace(args, kwargs) (line 141)
        replace_call_result_324375 = invoke(stypy.reporting.localization.Localization(__file__, 141, 18), replace_324371, *[], **kwargs_324374)
        
        
        # Call to replace(...): (line 141)
        # Processing the call keyword arguments (line 141)
        # Getting the type of 'None' (line 141)
        None_324378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 71), 'None', False)
        keyword_324379 = None_324378
        kwargs_324380 = {'tzinfo': keyword_324379}
        # Getting the type of 'wall_1' (line 141)
        wall_1_324376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 49), 'wall_1', False)
        # Obtaining the member 'replace' of a type (line 141)
        replace_324377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 49), wall_1_324376, 'replace')
        # Calling replace(args, kwargs) (line 141)
        replace_call_result_324381 = invoke(stypy.reporting.localization.Localization(__file__, 141, 49), replace_324377, *[], **kwargs_324380)
        
        # Applying the binary operator '==' (line 141)
        result_eq_324382 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 18), '==', replace_call_result_324375, replace_call_result_324381)
        
        # Assigning a type to the variable 'same_dt' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'same_dt', result_eq_324382)
        
        # Evaluating a boolean operation
        # Getting the type of 'same_dt' (line 143)
        same_dt_324383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 15), 'same_dt')
        
        # Getting the type of 'same_offset' (line 143)
        same_offset_324384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 31), 'same_offset')
        # Applying the 'not' unary operator (line 143)
        result_not__324385 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 27), 'not', same_offset_324384)
        
        # Applying the binary operator 'and' (line 143)
        result_and_keyword_324386 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 15), 'and', same_dt_324383, result_not__324385)
        
        # Assigning a type to the variable 'stypy_return_type' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'stypy_return_type', result_and_keyword_324386)
        
        # ################# End of 'is_ambiguous(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_ambiguous' in the type store
        # Getting the type of 'stypy_return_type' (line 120)
        stypy_return_type_324387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324387)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_ambiguous'
        return stypy_return_type_324387


    @norecursion
    def _fold_status(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_fold_status'
        module_type_store = module_type_store.open_function_context('_fold_status', 145, 4, False)
        # Assigning a type to the variable 'self' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _tzinfo._fold_status.__dict__.__setitem__('stypy_localization', localization)
        _tzinfo._fold_status.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _tzinfo._fold_status.__dict__.__setitem__('stypy_type_store', module_type_store)
        _tzinfo._fold_status.__dict__.__setitem__('stypy_function_name', '_tzinfo._fold_status')
        _tzinfo._fold_status.__dict__.__setitem__('stypy_param_names_list', ['dt_utc', 'dt_wall'])
        _tzinfo._fold_status.__dict__.__setitem__('stypy_varargs_param_name', None)
        _tzinfo._fold_status.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _tzinfo._fold_status.__dict__.__setitem__('stypy_call_defaults', defaults)
        _tzinfo._fold_status.__dict__.__setitem__('stypy_call_varargs', varargs)
        _tzinfo._fold_status.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _tzinfo._fold_status.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_tzinfo._fold_status', ['dt_utc', 'dt_wall'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_fold_status', localization, ['dt_utc', 'dt_wall'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_fold_status(...)' code ##################

        str_324388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, (-1)), 'str', '\n        Determine the fold status of a "wall" datetime, given a representation\n        of the same datetime as a (naive) UTC datetime. This is calculated based\n        on the assumption that ``dt.utcoffset() - dt.dst()`` is constant for all\n        datetimes, and that this offset is the actual number of hours separating\n        ``dt_utc`` and ``dt_wall``.\n\n        :param dt_utc:\n            Representation of the datetime as UTC\n\n        :param dt_wall:\n            Representation of the datetime as "wall time". This parameter must\n            either have a `fold` attribute or have a fold-naive\n            :class:`datetime.tzinfo` attached, otherwise the calculation may\n            fail.\n        ')
        
        
        # Call to is_ambiguous(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'dt_wall' (line 162)
        dt_wall_324391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 29), 'dt_wall', False)
        # Processing the call keyword arguments (line 162)
        kwargs_324392 = {}
        # Getting the type of 'self' (line 162)
        self_324389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 11), 'self', False)
        # Obtaining the member 'is_ambiguous' of a type (line 162)
        is_ambiguous_324390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 11), self_324389, 'is_ambiguous')
        # Calling is_ambiguous(args, kwargs) (line 162)
        is_ambiguous_call_result_324393 = invoke(stypy.reporting.localization.Localization(__file__, 162, 11), is_ambiguous_324390, *[dt_wall_324391], **kwargs_324392)
        
        # Testing the type of an if condition (line 162)
        if_condition_324394 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 8), is_ambiguous_call_result_324393)
        # Assigning a type to the variable 'if_condition_324394' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'if_condition_324394', if_condition_324394)
        # SSA begins for if statement (line 162)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 163):
        
        # Assigning a BinOp to a Name (line 163):
        # Getting the type of 'dt_wall' (line 163)
        dt_wall_324395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 25), 'dt_wall')
        # Getting the type of 'dt_utc' (line 163)
        dt_utc_324396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 35), 'dt_utc')
        # Applying the binary operator '-' (line 163)
        result_sub_324397 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 25), '-', dt_wall_324395, dt_utc_324396)
        
        # Assigning a type to the variable 'delta_wall' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'delta_wall', result_sub_324397)
        
        # Assigning a Call to a Name (line 164):
        
        # Assigning a Call to a Name (line 164):
        
        # Call to int(...): (line 164)
        # Processing the call arguments (line 164)
        
        # Getting the type of 'delta_wall' (line 164)
        delta_wall_324399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 24), 'delta_wall', False)
        
        # Call to utcoffset(...): (line 164)
        # Processing the call keyword arguments (line 164)
        kwargs_324402 = {}
        # Getting the type of 'dt_utc' (line 164)
        dt_utc_324400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 39), 'dt_utc', False)
        # Obtaining the member 'utcoffset' of a type (line 164)
        utcoffset_324401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 39), dt_utc_324400, 'utcoffset')
        # Calling utcoffset(args, kwargs) (line 164)
        utcoffset_call_result_324403 = invoke(stypy.reporting.localization.Localization(__file__, 164, 39), utcoffset_324401, *[], **kwargs_324402)
        
        
        # Call to dst(...): (line 164)
        # Processing the call keyword arguments (line 164)
        kwargs_324406 = {}
        # Getting the type of 'dt_utc' (line 164)
        dt_utc_324404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 60), 'dt_utc', False)
        # Obtaining the member 'dst' of a type (line 164)
        dst_324405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 60), dt_utc_324404, 'dst')
        # Calling dst(args, kwargs) (line 164)
        dst_call_result_324407 = invoke(stypy.reporting.localization.Localization(__file__, 164, 60), dst_324405, *[], **kwargs_324406)
        
        # Applying the binary operator '-' (line 164)
        result_sub_324408 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 39), '-', utcoffset_call_result_324403, dst_call_result_324407)
        
        # Applying the binary operator '==' (line 164)
        result_eq_324409 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 24), '==', delta_wall_324399, result_sub_324408)
        
        # Processing the call keyword arguments (line 164)
        kwargs_324410 = {}
        # Getting the type of 'int' (line 164)
        int_324398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'int', False)
        # Calling int(args, kwargs) (line 164)
        int_call_result_324411 = invoke(stypy.reporting.localization.Localization(__file__, 164, 20), int_324398, *[result_eq_324409], **kwargs_324410)
        
        # Assigning a type to the variable '_fold' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), '_fold', int_call_result_324411)
        # SSA branch for the else part of an if statement (line 162)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 166):
        
        # Assigning a Num to a Name (line 166):
        int_324412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 20), 'int')
        # Assigning a type to the variable '_fold' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), '_fold', int_324412)
        # SSA join for if statement (line 162)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of '_fold' (line 168)
        _fold_324413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 15), '_fold')
        # Assigning a type to the variable 'stypy_return_type' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'stypy_return_type', _fold_324413)
        
        # ################# End of '_fold_status(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_fold_status' in the type store
        # Getting the type of 'stypy_return_type' (line 145)
        stypy_return_type_324414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324414)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_fold_status'
        return stypy_return_type_324414


    @norecursion
    def _fold(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_fold'
        module_type_store = module_type_store.open_function_context('_fold', 170, 4, False)
        # Assigning a type to the variable 'self' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _tzinfo._fold.__dict__.__setitem__('stypy_localization', localization)
        _tzinfo._fold.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _tzinfo._fold.__dict__.__setitem__('stypy_type_store', module_type_store)
        _tzinfo._fold.__dict__.__setitem__('stypy_function_name', '_tzinfo._fold')
        _tzinfo._fold.__dict__.__setitem__('stypy_param_names_list', ['dt'])
        _tzinfo._fold.__dict__.__setitem__('stypy_varargs_param_name', None)
        _tzinfo._fold.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _tzinfo._fold.__dict__.__setitem__('stypy_call_defaults', defaults)
        _tzinfo._fold.__dict__.__setitem__('stypy_call_varargs', varargs)
        _tzinfo._fold.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _tzinfo._fold.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_tzinfo._fold', ['dt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_fold', localization, ['dt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_fold(...)' code ##################

        
        # Call to getattr(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'dt' (line 171)
        dt_324416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 23), 'dt', False)
        str_324417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 27), 'str', 'fold')
        int_324418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 35), 'int')
        # Processing the call keyword arguments (line 171)
        kwargs_324419 = {}
        # Getting the type of 'getattr' (line 171)
        getattr_324415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 15), 'getattr', False)
        # Calling getattr(args, kwargs) (line 171)
        getattr_call_result_324420 = invoke(stypy.reporting.localization.Localization(__file__, 171, 15), getattr_324415, *[dt_324416, str_324417, int_324418], **kwargs_324419)
        
        # Assigning a type to the variable 'stypy_return_type' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'stypy_return_type', getattr_call_result_324420)
        
        # ################# End of '_fold(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_fold' in the type store
        # Getting the type of 'stypy_return_type' (line 170)
        stypy_return_type_324421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324421)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_fold'
        return stypy_return_type_324421


    @norecursion
    def _fromutc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_fromutc'
        module_type_store = module_type_store.open_function_context('_fromutc', 173, 4, False)
        # Assigning a type to the variable 'self' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _tzinfo._fromutc.__dict__.__setitem__('stypy_localization', localization)
        _tzinfo._fromutc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _tzinfo._fromutc.__dict__.__setitem__('stypy_type_store', module_type_store)
        _tzinfo._fromutc.__dict__.__setitem__('stypy_function_name', '_tzinfo._fromutc')
        _tzinfo._fromutc.__dict__.__setitem__('stypy_param_names_list', ['dt'])
        _tzinfo._fromutc.__dict__.__setitem__('stypy_varargs_param_name', None)
        _tzinfo._fromutc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _tzinfo._fromutc.__dict__.__setitem__('stypy_call_defaults', defaults)
        _tzinfo._fromutc.__dict__.__setitem__('stypy_call_varargs', varargs)
        _tzinfo._fromutc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _tzinfo._fromutc.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_tzinfo._fromutc', ['dt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_fromutc', localization, ['dt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_fromutc(...)' code ##################

        str_324422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, (-1)), 'str', '\n        Given a timezone-aware datetime in a given timezone, calculates a\n        timezone-aware datetime in a new timezone.\n\n        Since this is the one time that we *know* we have an unambiguous\n        datetime object, we take this opportunity to determine whether the\n        datetime is ambiguous and in a "fold" state (e.g. if it\'s the first\n        occurence, chronologically, of the ambiguous datetime).\n\n        :param dt:\n            A timezone-aware :class:`datetime.datetime` object.\n        ')
        
        # Assigning a Call to a Name (line 188):
        
        # Assigning a Call to a Name (line 188):
        
        # Call to utcoffset(...): (line 188)
        # Processing the call keyword arguments (line 188)
        kwargs_324425 = {}
        # Getting the type of 'dt' (line 188)
        dt_324423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 16), 'dt', False)
        # Obtaining the member 'utcoffset' of a type (line 188)
        utcoffset_324424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 16), dt_324423, 'utcoffset')
        # Calling utcoffset(args, kwargs) (line 188)
        utcoffset_call_result_324426 = invoke(stypy.reporting.localization.Localization(__file__, 188, 16), utcoffset_324424, *[], **kwargs_324425)
        
        # Assigning a type to the variable 'dtoff' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'dtoff', utcoffset_call_result_324426)
        
        # Type idiom detected: calculating its left and rigth part (line 189)
        # Getting the type of 'dtoff' (line 189)
        dtoff_324427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 11), 'dtoff')
        # Getting the type of 'None' (line 189)
        None_324428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 20), 'None')
        
        (may_be_324429, more_types_in_union_324430) = may_be_none(dtoff_324427, None_324428)

        if may_be_324429:

            if more_types_in_union_324430:
                # Runtime conditional SSA (line 189)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to ValueError(...): (line 190)
            # Processing the call arguments (line 190)
            str_324432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 29), 'str', 'fromutc() requires a non-None utcoffset() result')
            # Processing the call keyword arguments (line 190)
            kwargs_324433 = {}
            # Getting the type of 'ValueError' (line 190)
            ValueError_324431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 190)
            ValueError_call_result_324434 = invoke(stypy.reporting.localization.Localization(__file__, 190, 18), ValueError_324431, *[str_324432], **kwargs_324433)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 190, 12), ValueError_call_result_324434, 'raise parameter', BaseException)

            if more_types_in_union_324430:
                # SSA join for if statement (line 189)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 196):
        
        # Assigning a Call to a Name (line 196):
        
        # Call to dst(...): (line 196)
        # Processing the call keyword arguments (line 196)
        kwargs_324437 = {}
        # Getting the type of 'dt' (line 196)
        dt_324435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'dt', False)
        # Obtaining the member 'dst' of a type (line 196)
        dst_324436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 16), dt_324435, 'dst')
        # Calling dst(args, kwargs) (line 196)
        dst_call_result_324438 = invoke(stypy.reporting.localization.Localization(__file__, 196, 16), dst_324436, *[], **kwargs_324437)
        
        # Assigning a type to the variable 'dtdst' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'dtdst', dst_call_result_324438)
        
        # Type idiom detected: calculating its left and rigth part (line 197)
        # Getting the type of 'dtdst' (line 197)
        dtdst_324439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 11), 'dtdst')
        # Getting the type of 'None' (line 197)
        None_324440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 20), 'None')
        
        (may_be_324441, more_types_in_union_324442) = may_be_none(dtdst_324439, None_324440)

        if may_be_324441:

            if more_types_in_union_324442:
                # Runtime conditional SSA (line 197)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to ValueError(...): (line 198)
            # Processing the call arguments (line 198)
            str_324444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 29), 'str', 'fromutc() requires a non-None dst() result')
            # Processing the call keyword arguments (line 198)
            kwargs_324445 = {}
            # Getting the type of 'ValueError' (line 198)
            ValueError_324443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 198)
            ValueError_call_result_324446 = invoke(stypy.reporting.localization.Localization(__file__, 198, 18), ValueError_324443, *[str_324444], **kwargs_324445)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 198, 12), ValueError_call_result_324446, 'raise parameter', BaseException)

            if more_types_in_union_324442:
                # SSA join for if statement (line 197)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Name (line 199):
        
        # Assigning a BinOp to a Name (line 199):
        # Getting the type of 'dtoff' (line 199)
        dtoff_324447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'dtoff')
        # Getting the type of 'dtdst' (line 199)
        dtdst_324448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 24), 'dtdst')
        # Applying the binary operator '-' (line 199)
        result_sub_324449 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 16), '-', dtoff_324447, dtdst_324448)
        
        # Assigning a type to the variable 'delta' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'delta', result_sub_324449)
        
        # Getting the type of 'dt' (line 201)
        dt_324450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'dt')
        # Getting the type of 'delta' (line 201)
        delta_324451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 14), 'delta')
        # Applying the binary operator '+=' (line 201)
        result_iadd_324452 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 8), '+=', dt_324450, delta_324451)
        # Assigning a type to the variable 'dt' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'dt', result_iadd_324452)
        
        
        # Assigning a Call to a Name (line 204):
        
        # Assigning a Call to a Name (line 204):
        
        # Call to dst(...): (line 204)
        # Processing the call keyword arguments (line 204)
        kwargs_324460 = {}
        
        # Call to enfold(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'dt' (line 204)
        dt_324454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 23), 'dt', False)
        # Processing the call keyword arguments (line 204)
        int_324455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 32), 'int')
        keyword_324456 = int_324455
        kwargs_324457 = {'fold': keyword_324456}
        # Getting the type of 'enfold' (line 204)
        enfold_324453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), 'enfold', False)
        # Calling enfold(args, kwargs) (line 204)
        enfold_call_result_324458 = invoke(stypy.reporting.localization.Localization(__file__, 204, 16), enfold_324453, *[dt_324454], **kwargs_324457)
        
        # Obtaining the member 'dst' of a type (line 204)
        dst_324459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 16), enfold_call_result_324458, 'dst')
        # Calling dst(args, kwargs) (line 204)
        dst_call_result_324461 = invoke(stypy.reporting.localization.Localization(__file__, 204, 16), dst_324459, *[], **kwargs_324460)
        
        # Assigning a type to the variable 'dtdst' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'dtdst', dst_call_result_324461)
        
        # Type idiom detected: calculating its left and rigth part (line 205)
        # Getting the type of 'dtdst' (line 205)
        dtdst_324462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 11), 'dtdst')
        # Getting the type of 'None' (line 205)
        None_324463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'None')
        
        (may_be_324464, more_types_in_union_324465) = may_be_none(dtdst_324462, None_324463)

        if may_be_324464:

            if more_types_in_union_324465:
                # Runtime conditional SSA (line 205)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to ValueError(...): (line 206)
            # Processing the call arguments (line 206)
            str_324467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 29), 'str', 'fromutc(): dt.dst gave inconsistent results; cannot convert')
            # Processing the call keyword arguments (line 206)
            kwargs_324468 = {}
            # Getting the type of 'ValueError' (line 206)
            ValueError_324466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 206)
            ValueError_call_result_324469 = invoke(stypy.reporting.localization.Localization(__file__, 206, 18), ValueError_324466, *[str_324467], **kwargs_324468)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 206, 12), ValueError_call_result_324469, 'raise parameter', BaseException)

            if more_types_in_union_324465:
                # SSA join for if statement (line 205)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'dt' (line 208)
        dt_324470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'dt')
        # Getting the type of 'dtdst' (line 208)
        dtdst_324471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 20), 'dtdst')
        # Applying the binary operator '+' (line 208)
        result_add_324472 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 15), '+', dt_324470, dtdst_324471)
        
        # Assigning a type to the variable 'stypy_return_type' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'stypy_return_type', result_add_324472)
        
        # ################# End of '_fromutc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_fromutc' in the type store
        # Getting the type of 'stypy_return_type' (line 173)
        stypy_return_type_324473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324473)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_fromutc'
        return stypy_return_type_324473


    @norecursion
    def fromutc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fromutc'
        module_type_store = module_type_store.open_function_context('fromutc', 210, 4, False)
        # Assigning a type to the variable 'self' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _tzinfo.fromutc.__dict__.__setitem__('stypy_localization', localization)
        _tzinfo.fromutc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _tzinfo.fromutc.__dict__.__setitem__('stypy_type_store', module_type_store)
        _tzinfo.fromutc.__dict__.__setitem__('stypy_function_name', '_tzinfo.fromutc')
        _tzinfo.fromutc.__dict__.__setitem__('stypy_param_names_list', ['dt'])
        _tzinfo.fromutc.__dict__.__setitem__('stypy_varargs_param_name', None)
        _tzinfo.fromutc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _tzinfo.fromutc.__dict__.__setitem__('stypy_call_defaults', defaults)
        _tzinfo.fromutc.__dict__.__setitem__('stypy_call_varargs', varargs)
        _tzinfo.fromutc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _tzinfo.fromutc.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_tzinfo.fromutc', ['dt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fromutc', localization, ['dt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fromutc(...)' code ##################

        str_324474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, (-1)), 'str', '\n        Given a timezone-aware datetime in a given timezone, calculates a\n        timezone-aware datetime in a new timezone.\n\n        Since this is the one time that we *know* we have an unambiguous\n        datetime object, we take this opportunity to determine whether the\n        datetime is ambiguous and in a "fold" state (e.g. if it\'s the first\n        occurance, chronologically, of the ambiguous datetime).\n\n        :param dt:\n            A timezone-aware :class:`datetime.datetime` object.\n        ')
        
        # Assigning a Call to a Name (line 224):
        
        # Assigning a Call to a Name (line 224):
        
        # Call to _fromutc(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'dt' (line 224)
        dt_324477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 32), 'dt', False)
        # Processing the call keyword arguments (line 224)
        kwargs_324478 = {}
        # Getting the type of 'self' (line 224)
        self_324475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 18), 'self', False)
        # Obtaining the member '_fromutc' of a type (line 224)
        _fromutc_324476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 18), self_324475, '_fromutc')
        # Calling _fromutc(args, kwargs) (line 224)
        _fromutc_call_result_324479 = invoke(stypy.reporting.localization.Localization(__file__, 224, 18), _fromutc_324476, *[dt_324477], **kwargs_324478)
        
        # Assigning a type to the variable 'dt_wall' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'dt_wall', _fromutc_call_result_324479)
        
        # Assigning a Call to a Name (line 227):
        
        # Assigning a Call to a Name (line 227):
        
        # Call to _fold_status(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'dt' (line 227)
        dt_324482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 34), 'dt', False)
        # Getting the type of 'dt_wall' (line 227)
        dt_wall_324483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 38), 'dt_wall', False)
        # Processing the call keyword arguments (line 227)
        kwargs_324484 = {}
        # Getting the type of 'self' (line 227)
        self_324480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'self', False)
        # Obtaining the member '_fold_status' of a type (line 227)
        _fold_status_324481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), self_324480, '_fold_status')
        # Calling _fold_status(args, kwargs) (line 227)
        _fold_status_call_result_324485 = invoke(stypy.reporting.localization.Localization(__file__, 227, 16), _fold_status_324481, *[dt_324482, dt_wall_324483], **kwargs_324484)
        
        # Assigning a type to the variable '_fold' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), '_fold', _fold_status_call_result_324485)
        
        # Call to enfold(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'dt_wall' (line 230)
        dt_wall_324487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 22), 'dt_wall', False)
        # Processing the call keyword arguments (line 230)
        # Getting the type of '_fold' (line 230)
        _fold_324488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 36), '_fold', False)
        keyword_324489 = _fold_324488
        kwargs_324490 = {'fold': keyword_324489}
        # Getting the type of 'enfold' (line 230)
        enfold_324486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 15), 'enfold', False)
        # Calling enfold(args, kwargs) (line 230)
        enfold_call_result_324491 = invoke(stypy.reporting.localization.Localization(__file__, 230, 15), enfold_324486, *[dt_wall_324487], **kwargs_324490)
        
        # Assigning a type to the variable 'stypy_return_type' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'stypy_return_type', enfold_call_result_324491)
        
        # ################# End of 'fromutc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fromutc' in the type store
        # Getting the type of 'stypy_return_type' (line 210)
        stypy_return_type_324492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324492)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fromutc'
        return stypy_return_type_324492


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 115, 0, False)
        # Assigning a type to the variable 'self' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_tzinfo.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_tzinfo' (line 115)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 0), '_tzinfo', _tzinfo)
# Declaration of the 'tzrangebase' class
# Getting the type of '_tzinfo' (line 233)
_tzinfo_324493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 18), '_tzinfo')

class tzrangebase(_tzinfo_324493, ):
    str_324494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, (-1)), 'str', '\n    This is an abstract base class for time zones represented by an annual\n    transition into and out of DST. Child classes should implement the following\n    methods:\n\n        * ``__init__(self, *args, **kwargs)``\n        * ``transitions(self, year)`` - this is expected to return a tuple of\n          datetimes representing the DST on and off transitions in standard\n          time.\n\n    A fully initialized ``tzrangebase`` subclass should also provide the\n    following attributes:\n        * ``hasdst``: Boolean whether or not the zone uses DST.\n        * ``_dst_offset`` / ``_std_offset``: :class:`datetime.timedelta` objects\n          representing the respective UTC offsets.\n        * ``_dst_abbr`` / ``_std_abbr``: Strings representing the timezone short\n          abbreviations in DST and STD, respectively.\n        * ``_hasdst``: Whether or not the zone has DST.\n\n    .. versionadded:: 2.6.0\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 255, 4, False)
        # Assigning a type to the variable 'self' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzrangebase.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to NotImplementedError(...): (line 256)
        # Processing the call arguments (line 256)
        str_324496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 34), 'str', 'tzrangebase is an abstract base class')
        # Processing the call keyword arguments (line 256)
        kwargs_324497 = {}
        # Getting the type of 'NotImplementedError' (line 256)
        NotImplementedError_324495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 14), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 256)
        NotImplementedError_call_result_324498 = invoke(stypy.reporting.localization.Localization(__file__, 256, 14), NotImplementedError_324495, *[str_324496], **kwargs_324497)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 256, 8), NotImplementedError_call_result_324498, 'raise parameter', BaseException)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def utcoffset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'utcoffset'
        module_type_store = module_type_store.open_function_context('utcoffset', 258, 4, False)
        # Assigning a type to the variable 'self' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzrangebase.utcoffset.__dict__.__setitem__('stypy_localization', localization)
        tzrangebase.utcoffset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzrangebase.utcoffset.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzrangebase.utcoffset.__dict__.__setitem__('stypy_function_name', 'tzrangebase.utcoffset')
        tzrangebase.utcoffset.__dict__.__setitem__('stypy_param_names_list', ['dt'])
        tzrangebase.utcoffset.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzrangebase.utcoffset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzrangebase.utcoffset.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzrangebase.utcoffset.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzrangebase.utcoffset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzrangebase.utcoffset.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzrangebase.utcoffset', ['dt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'utcoffset', localization, ['dt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'utcoffset(...)' code ##################

        
        # Assigning a Call to a Name (line 259):
        
        # Assigning a Call to a Name (line 259):
        
        # Call to _isdst(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'dt' (line 259)
        dt_324501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'dt', False)
        # Processing the call keyword arguments (line 259)
        kwargs_324502 = {}
        # Getting the type of 'self' (line 259)
        self_324499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 16), 'self', False)
        # Obtaining the member '_isdst' of a type (line 259)
        _isdst_324500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 16), self_324499, '_isdst')
        # Calling _isdst(args, kwargs) (line 259)
        _isdst_call_result_324503 = invoke(stypy.reporting.localization.Localization(__file__, 259, 16), _isdst_324500, *[dt_324501], **kwargs_324502)
        
        # Assigning a type to the variable 'isdst' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'isdst', _isdst_call_result_324503)
        
        # Type idiom detected: calculating its left and rigth part (line 261)
        # Getting the type of 'isdst' (line 261)
        isdst_324504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 11), 'isdst')
        # Getting the type of 'None' (line 261)
        None_324505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 20), 'None')
        
        (may_be_324506, more_types_in_union_324507) = may_be_none(isdst_324504, None_324505)

        if may_be_324506:

            if more_types_in_union_324507:
                # Runtime conditional SSA (line 261)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'None' (line 262)
            None_324508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 262)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'stypy_return_type', None_324508)

            if more_types_in_union_324507:
                # Runtime conditional SSA for else branch (line 261)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_324506) or more_types_in_union_324507):
            
            # Getting the type of 'isdst' (line 263)
            isdst_324509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 13), 'isdst')
            # Testing the type of an if condition (line 263)
            if_condition_324510 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 263, 13), isdst_324509)
            # Assigning a type to the variable 'if_condition_324510' (line 263)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 13), 'if_condition_324510', if_condition_324510)
            # SSA begins for if statement (line 263)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'self' (line 264)
            self_324511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 19), 'self')
            # Obtaining the member '_dst_offset' of a type (line 264)
            _dst_offset_324512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 19), self_324511, '_dst_offset')
            # Assigning a type to the variable 'stypy_return_type' (line 264)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'stypy_return_type', _dst_offset_324512)
            # SSA branch for the else part of an if statement (line 263)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'self' (line 266)
            self_324513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 19), 'self')
            # Obtaining the member '_std_offset' of a type (line 266)
            _std_offset_324514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 19), self_324513, '_std_offset')
            # Assigning a type to the variable 'stypy_return_type' (line 266)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'stypy_return_type', _std_offset_324514)
            # SSA join for if statement (line 263)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_324506 and more_types_in_union_324507):
                # SSA join for if statement (line 261)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'utcoffset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'utcoffset' in the type store
        # Getting the type of 'stypy_return_type' (line 258)
        stypy_return_type_324515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324515)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'utcoffset'
        return stypy_return_type_324515


    @norecursion
    def dst(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dst'
        module_type_store = module_type_store.open_function_context('dst', 268, 4, False)
        # Assigning a type to the variable 'self' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzrangebase.dst.__dict__.__setitem__('stypy_localization', localization)
        tzrangebase.dst.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzrangebase.dst.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzrangebase.dst.__dict__.__setitem__('stypy_function_name', 'tzrangebase.dst')
        tzrangebase.dst.__dict__.__setitem__('stypy_param_names_list', ['dt'])
        tzrangebase.dst.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzrangebase.dst.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzrangebase.dst.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzrangebase.dst.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzrangebase.dst.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzrangebase.dst.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzrangebase.dst', ['dt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dst', localization, ['dt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dst(...)' code ##################

        
        # Assigning a Call to a Name (line 269):
        
        # Assigning a Call to a Name (line 269):
        
        # Call to _isdst(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'dt' (line 269)
        dt_324518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 28), 'dt', False)
        # Processing the call keyword arguments (line 269)
        kwargs_324519 = {}
        # Getting the type of 'self' (line 269)
        self_324516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 16), 'self', False)
        # Obtaining the member '_isdst' of a type (line 269)
        _isdst_324517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 16), self_324516, '_isdst')
        # Calling _isdst(args, kwargs) (line 269)
        _isdst_call_result_324520 = invoke(stypy.reporting.localization.Localization(__file__, 269, 16), _isdst_324517, *[dt_324518], **kwargs_324519)
        
        # Assigning a type to the variable 'isdst' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'isdst', _isdst_call_result_324520)
        
        # Type idiom detected: calculating its left and rigth part (line 271)
        # Getting the type of 'isdst' (line 271)
        isdst_324521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 11), 'isdst')
        # Getting the type of 'None' (line 271)
        None_324522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 20), 'None')
        
        (may_be_324523, more_types_in_union_324524) = may_be_none(isdst_324521, None_324522)

        if may_be_324523:

            if more_types_in_union_324524:
                # Runtime conditional SSA (line 271)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'None' (line 272)
            None_324525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 272)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'stypy_return_type', None_324525)

            if more_types_in_union_324524:
                # Runtime conditional SSA for else branch (line 271)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_324523) or more_types_in_union_324524):
            
            # Getting the type of 'isdst' (line 273)
            isdst_324526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 13), 'isdst')
            # Testing the type of an if condition (line 273)
            if_condition_324527 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 13), isdst_324526)
            # Assigning a type to the variable 'if_condition_324527' (line 273)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 13), 'if_condition_324527', if_condition_324527)
            # SSA begins for if statement (line 273)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'self' (line 274)
            self_324528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 19), 'self')
            # Obtaining the member '_dst_base_offset' of a type (line 274)
            _dst_base_offset_324529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 19), self_324528, '_dst_base_offset')
            # Assigning a type to the variable 'stypy_return_type' (line 274)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'stypy_return_type', _dst_base_offset_324529)
            # SSA branch for the else part of an if statement (line 273)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'ZERO' (line 276)
            ZERO_324530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 19), 'ZERO')
            # Assigning a type to the variable 'stypy_return_type' (line 276)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'stypy_return_type', ZERO_324530)
            # SSA join for if statement (line 273)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_324523 and more_types_in_union_324524):
                # SSA join for if statement (line 271)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'dst(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dst' in the type store
        # Getting the type of 'stypy_return_type' (line 268)
        stypy_return_type_324531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324531)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dst'
        return stypy_return_type_324531


    @norecursion
    def tzname(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tzname'
        module_type_store = module_type_store.open_function_context('tzname', 278, 4, False)
        # Assigning a type to the variable 'self' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzrangebase.tzname.__dict__.__setitem__('stypy_localization', localization)
        tzrangebase.tzname.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzrangebase.tzname.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzrangebase.tzname.__dict__.__setitem__('stypy_function_name', 'tzrangebase.tzname')
        tzrangebase.tzname.__dict__.__setitem__('stypy_param_names_list', ['dt'])
        tzrangebase.tzname.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzrangebase.tzname.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzrangebase.tzname.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzrangebase.tzname.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzrangebase.tzname.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzrangebase.tzname.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzrangebase.tzname', ['dt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tzname', localization, ['dt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tzname(...)' code ##################

        
        
        # Call to _isdst(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'dt' (line 280)
        dt_324534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 23), 'dt', False)
        # Processing the call keyword arguments (line 280)
        kwargs_324535 = {}
        # Getting the type of 'self' (line 280)
        self_324532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 11), 'self', False)
        # Obtaining the member '_isdst' of a type (line 280)
        _isdst_324533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 11), self_324532, '_isdst')
        # Calling _isdst(args, kwargs) (line 280)
        _isdst_call_result_324536 = invoke(stypy.reporting.localization.Localization(__file__, 280, 11), _isdst_324533, *[dt_324534], **kwargs_324535)
        
        # Testing the type of an if condition (line 280)
        if_condition_324537 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 8), _isdst_call_result_324536)
        # Assigning a type to the variable 'if_condition_324537' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'if_condition_324537', if_condition_324537)
        # SSA begins for if statement (line 280)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 281)
        self_324538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 19), 'self')
        # Obtaining the member '_dst_abbr' of a type (line 281)
        _dst_abbr_324539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 19), self_324538, '_dst_abbr')
        # Assigning a type to the variable 'stypy_return_type' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'stypy_return_type', _dst_abbr_324539)
        # SSA branch for the else part of an if statement (line 280)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'self' (line 283)
        self_324540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 19), 'self')
        # Obtaining the member '_std_abbr' of a type (line 283)
        _std_abbr_324541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 19), self_324540, '_std_abbr')
        # Assigning a type to the variable 'stypy_return_type' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'stypy_return_type', _std_abbr_324541)
        # SSA join for if statement (line 280)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'tzname(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tzname' in the type store
        # Getting the type of 'stypy_return_type' (line 278)
        stypy_return_type_324542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324542)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tzname'
        return stypy_return_type_324542


    @norecursion
    def fromutc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fromutc'
        module_type_store = module_type_store.open_function_context('fromutc', 285, 4, False)
        # Assigning a type to the variable 'self' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzrangebase.fromutc.__dict__.__setitem__('stypy_localization', localization)
        tzrangebase.fromutc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzrangebase.fromutc.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzrangebase.fromutc.__dict__.__setitem__('stypy_function_name', 'tzrangebase.fromutc')
        tzrangebase.fromutc.__dict__.__setitem__('stypy_param_names_list', ['dt'])
        tzrangebase.fromutc.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzrangebase.fromutc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzrangebase.fromutc.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzrangebase.fromutc.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzrangebase.fromutc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzrangebase.fromutc.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzrangebase.fromutc', ['dt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fromutc', localization, ['dt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fromutc(...)' code ##################

        str_324543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 8), 'str', ' Given a datetime in UTC, return local time ')
        
        
        
        # Call to isinstance(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'dt' (line 287)
        dt_324545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 26), 'dt', False)
        # Getting the type of 'datetime' (line 287)
        datetime_324546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 30), 'datetime', False)
        # Processing the call keyword arguments (line 287)
        kwargs_324547 = {}
        # Getting the type of 'isinstance' (line 287)
        isinstance_324544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 287)
        isinstance_call_result_324548 = invoke(stypy.reporting.localization.Localization(__file__, 287, 15), isinstance_324544, *[dt_324545, datetime_324546], **kwargs_324547)
        
        # Applying the 'not' unary operator (line 287)
        result_not__324549 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 11), 'not', isinstance_call_result_324548)
        
        # Testing the type of an if condition (line 287)
        if_condition_324550 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 287, 8), result_not__324549)
        # Assigning a type to the variable 'if_condition_324550' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'if_condition_324550', if_condition_324550)
        # SSA begins for if statement (line 287)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 288)
        # Processing the call arguments (line 288)
        str_324552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 28), 'str', 'fromutc() requires a datetime argument')
        # Processing the call keyword arguments (line 288)
        kwargs_324553 = {}
        # Getting the type of 'TypeError' (line 288)
        TypeError_324551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 288)
        TypeError_call_result_324554 = invoke(stypy.reporting.localization.Localization(__file__, 288, 18), TypeError_324551, *[str_324552], **kwargs_324553)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 288, 12), TypeError_call_result_324554, 'raise parameter', BaseException)
        # SSA join for if statement (line 287)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'dt' (line 290)
        dt_324555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 11), 'dt')
        # Obtaining the member 'tzinfo' of a type (line 290)
        tzinfo_324556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 11), dt_324555, 'tzinfo')
        # Getting the type of 'self' (line 290)
        self_324557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 28), 'self')
        # Applying the binary operator 'isnot' (line 290)
        result_is_not_324558 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 11), 'isnot', tzinfo_324556, self_324557)
        
        # Testing the type of an if condition (line 290)
        if_condition_324559 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 290, 8), result_is_not_324558)
        # Assigning a type to the variable 'if_condition_324559' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'if_condition_324559', if_condition_324559)
        # SSA begins for if statement (line 290)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 291)
        # Processing the call arguments (line 291)
        str_324561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 29), 'str', 'dt.tzinfo is not self')
        # Processing the call keyword arguments (line 291)
        kwargs_324562 = {}
        # Getting the type of 'ValueError' (line 291)
        ValueError_324560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 291)
        ValueError_call_result_324563 = invoke(stypy.reporting.localization.Localization(__file__, 291, 18), ValueError_324560, *[str_324561], **kwargs_324562)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 291, 12), ValueError_call_result_324563, 'raise parameter', BaseException)
        # SSA join for if statement (line 290)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 294):
        
        # Assigning a Call to a Name (line 294):
        
        # Call to transitions(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'dt' (line 294)
        dt_324566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 39), 'dt', False)
        # Obtaining the member 'year' of a type (line 294)
        year_324567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 39), dt_324566, 'year')
        # Processing the call keyword arguments (line 294)
        kwargs_324568 = {}
        # Getting the type of 'self' (line 294)
        self_324564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 22), 'self', False)
        # Obtaining the member 'transitions' of a type (line 294)
        transitions_324565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 22), self_324564, 'transitions')
        # Calling transitions(args, kwargs) (line 294)
        transitions_call_result_324569 = invoke(stypy.reporting.localization.Localization(__file__, 294, 22), transitions_324565, *[year_324567], **kwargs_324568)
        
        # Assigning a type to the variable 'transitions' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'transitions', transitions_call_result_324569)
        
        # Type idiom detected: calculating its left and rigth part (line 295)
        # Getting the type of 'transitions' (line 295)
        transitions_324570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 11), 'transitions')
        # Getting the type of 'None' (line 295)
        None_324571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 26), 'None')
        
        (may_be_324572, more_types_in_union_324573) = may_be_none(transitions_324570, None_324571)

        if may_be_324572:

            if more_types_in_union_324573:
                # Runtime conditional SSA (line 295)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'dt' (line 296)
            dt_324574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 19), 'dt')
            
            # Call to utcoffset(...): (line 296)
            # Processing the call arguments (line 296)
            # Getting the type of 'dt' (line 296)
            dt_324577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 39), 'dt', False)
            # Processing the call keyword arguments (line 296)
            kwargs_324578 = {}
            # Getting the type of 'self' (line 296)
            self_324575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 24), 'self', False)
            # Obtaining the member 'utcoffset' of a type (line 296)
            utcoffset_324576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 24), self_324575, 'utcoffset')
            # Calling utcoffset(args, kwargs) (line 296)
            utcoffset_call_result_324579 = invoke(stypy.reporting.localization.Localization(__file__, 296, 24), utcoffset_324576, *[dt_324577], **kwargs_324578)
            
            # Applying the binary operator '+' (line 296)
            result_add_324580 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 19), '+', dt_324574, utcoffset_call_result_324579)
            
            # Assigning a type to the variable 'stypy_return_type' (line 296)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'stypy_return_type', result_add_324580)

            if more_types_in_union_324573:
                # SSA join for if statement (line 295)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Tuple (line 299):
        
        # Assigning a Subscript to a Name (line 299):
        
        # Obtaining the type of the subscript
        int_324581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 8), 'int')
        # Getting the type of 'transitions' (line 299)
        transitions_324582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 24), 'transitions')
        # Obtaining the member '__getitem__' of a type (line 299)
        getitem___324583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 8), transitions_324582, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 299)
        subscript_call_result_324584 = invoke(stypy.reporting.localization.Localization(__file__, 299, 8), getitem___324583, int_324581)
        
        # Assigning a type to the variable 'tuple_var_assignment_324217' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'tuple_var_assignment_324217', subscript_call_result_324584)
        
        # Assigning a Subscript to a Name (line 299):
        
        # Obtaining the type of the subscript
        int_324585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 8), 'int')
        # Getting the type of 'transitions' (line 299)
        transitions_324586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 24), 'transitions')
        # Obtaining the member '__getitem__' of a type (line 299)
        getitem___324587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 8), transitions_324586, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 299)
        subscript_call_result_324588 = invoke(stypy.reporting.localization.Localization(__file__, 299, 8), getitem___324587, int_324585)
        
        # Assigning a type to the variable 'tuple_var_assignment_324218' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'tuple_var_assignment_324218', subscript_call_result_324588)
        
        # Assigning a Name to a Name (line 299):
        # Getting the type of 'tuple_var_assignment_324217' (line 299)
        tuple_var_assignment_324217_324589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'tuple_var_assignment_324217')
        # Assigning a type to the variable 'dston' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'dston', tuple_var_assignment_324217_324589)
        
        # Assigning a Name to a Name (line 299):
        # Getting the type of 'tuple_var_assignment_324218' (line 299)
        tuple_var_assignment_324218_324590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'tuple_var_assignment_324218')
        # Assigning a type to the variable 'dstoff' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 15), 'dstoff', tuple_var_assignment_324218_324590)
        
        # Getting the type of 'dston' (line 301)
        dston_324591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'dston')
        # Getting the type of 'self' (line 301)
        self_324592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 17), 'self')
        # Obtaining the member '_std_offset' of a type (line 301)
        _std_offset_324593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 17), self_324592, '_std_offset')
        # Applying the binary operator '-=' (line 301)
        result_isub_324594 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 8), '-=', dston_324591, _std_offset_324593)
        # Assigning a type to the variable 'dston' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'dston', result_isub_324594)
        
        
        # Getting the type of 'dstoff' (line 302)
        dstoff_324595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'dstoff')
        # Getting the type of 'self' (line 302)
        self_324596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 18), 'self')
        # Obtaining the member '_std_offset' of a type (line 302)
        _std_offset_324597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 18), self_324596, '_std_offset')
        # Applying the binary operator '-=' (line 302)
        result_isub_324598 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 8), '-=', dstoff_324595, _std_offset_324597)
        # Assigning a type to the variable 'dstoff' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'dstoff', result_isub_324598)
        
        
        # Assigning a Tuple to a Name (line 304):
        
        # Assigning a Tuple to a Name (line 304):
        
        # Obtaining an instance of the builtin type 'tuple' (line 304)
        tuple_324599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 304)
        # Adding element type (line 304)
        # Getting the type of 'dston' (line 304)
        dston_324600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 27), 'dston')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 27), tuple_324599, dston_324600)
        # Adding element type (line 304)
        # Getting the type of 'dstoff' (line 304)
        dstoff_324601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 34), 'dstoff')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 27), tuple_324599, dstoff_324601)
        
        # Assigning a type to the variable 'utc_transitions' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'utc_transitions', tuple_324599)
        
        # Assigning a Call to a Name (line 305):
        
        # Assigning a Call to a Name (line 305):
        
        # Call to replace(...): (line 305)
        # Processing the call keyword arguments (line 305)
        # Getting the type of 'None' (line 305)
        None_324604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 35), 'None', False)
        keyword_324605 = None_324604
        kwargs_324606 = {'tzinfo': keyword_324605}
        # Getting the type of 'dt' (line 305)
        dt_324602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 17), 'dt', False)
        # Obtaining the member 'replace' of a type (line 305)
        replace_324603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 17), dt_324602, 'replace')
        # Calling replace(args, kwargs) (line 305)
        replace_call_result_324607 = invoke(stypy.reporting.localization.Localization(__file__, 305, 17), replace_324603, *[], **kwargs_324606)
        
        # Assigning a type to the variable 'dt_utc' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'dt_utc', replace_call_result_324607)
        
        # Assigning a Call to a Name (line 307):
        
        # Assigning a Call to a Name (line 307):
        
        # Call to _naive_isdst(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'dt_utc' (line 307)
        dt_utc_324610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 34), 'dt_utc', False)
        # Getting the type of 'utc_transitions' (line 307)
        utc_transitions_324611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 42), 'utc_transitions', False)
        # Processing the call keyword arguments (line 307)
        kwargs_324612 = {}
        # Getting the type of 'self' (line 307)
        self_324608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 16), 'self', False)
        # Obtaining the member '_naive_isdst' of a type (line 307)
        _naive_isdst_324609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 16), self_324608, '_naive_isdst')
        # Calling _naive_isdst(args, kwargs) (line 307)
        _naive_isdst_call_result_324613 = invoke(stypy.reporting.localization.Localization(__file__, 307, 16), _naive_isdst_324609, *[dt_utc_324610, utc_transitions_324611], **kwargs_324612)
        
        # Assigning a type to the variable 'isdst' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'isdst', _naive_isdst_call_result_324613)
        
        # Getting the type of 'isdst' (line 309)
        isdst_324614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 11), 'isdst')
        # Testing the type of an if condition (line 309)
        if_condition_324615 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 309, 8), isdst_324614)
        # Assigning a type to the variable 'if_condition_324615' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'if_condition_324615', if_condition_324615)
        # SSA begins for if statement (line 309)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 310):
        
        # Assigning a BinOp to a Name (line 310):
        # Getting the type of 'dt' (line 310)
        dt_324616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 22), 'dt')
        # Getting the type of 'self' (line 310)
        self_324617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 27), 'self')
        # Obtaining the member '_dst_offset' of a type (line 310)
        _dst_offset_324618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 27), self_324617, '_dst_offset')
        # Applying the binary operator '+' (line 310)
        result_add_324619 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 22), '+', dt_324616, _dst_offset_324618)
        
        # Assigning a type to the variable 'dt_wall' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'dt_wall', result_add_324619)
        # SSA branch for the else part of an if statement (line 309)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 312):
        
        # Assigning a BinOp to a Name (line 312):
        # Getting the type of 'dt' (line 312)
        dt_324620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 22), 'dt')
        # Getting the type of 'self' (line 312)
        self_324621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 27), 'self')
        # Obtaining the member '_std_offset' of a type (line 312)
        _std_offset_324622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 27), self_324621, '_std_offset')
        # Applying the binary operator '+' (line 312)
        result_add_324623 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 22), '+', dt_324620, _std_offset_324622)
        
        # Assigning a type to the variable 'dt_wall' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'dt_wall', result_add_324623)
        # SSA join for if statement (line 309)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 314):
        
        # Assigning a Call to a Name (line 314):
        
        # Call to int(...): (line 314)
        # Processing the call arguments (line 314)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'isdst' (line 314)
        isdst_324625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 24), 'isdst', False)
        # Applying the 'not' unary operator (line 314)
        result_not__324626 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 20), 'not', isdst_324625)
        
        
        # Call to is_ambiguous(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of 'dt_wall' (line 314)
        dt_wall_324629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 52), 'dt_wall', False)
        # Processing the call keyword arguments (line 314)
        kwargs_324630 = {}
        # Getting the type of 'self' (line 314)
        self_324627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 34), 'self', False)
        # Obtaining the member 'is_ambiguous' of a type (line 314)
        is_ambiguous_324628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 34), self_324627, 'is_ambiguous')
        # Calling is_ambiguous(args, kwargs) (line 314)
        is_ambiguous_call_result_324631 = invoke(stypy.reporting.localization.Localization(__file__, 314, 34), is_ambiguous_324628, *[dt_wall_324629], **kwargs_324630)
        
        # Applying the binary operator 'and' (line 314)
        result_and_keyword_324632 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 20), 'and', result_not__324626, is_ambiguous_call_result_324631)
        
        # Processing the call keyword arguments (line 314)
        kwargs_324633 = {}
        # Getting the type of 'int' (line 314)
        int_324624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 16), 'int', False)
        # Calling int(args, kwargs) (line 314)
        int_call_result_324634 = invoke(stypy.reporting.localization.Localization(__file__, 314, 16), int_324624, *[result_and_keyword_324632], **kwargs_324633)
        
        # Assigning a type to the variable '_fold' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), '_fold', int_call_result_324634)
        
        # Call to enfold(...): (line 316)
        # Processing the call arguments (line 316)
        # Getting the type of 'dt_wall' (line 316)
        dt_wall_324636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 22), 'dt_wall', False)
        # Processing the call keyword arguments (line 316)
        # Getting the type of '_fold' (line 316)
        _fold_324637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 36), '_fold', False)
        keyword_324638 = _fold_324637
        kwargs_324639 = {'fold': keyword_324638}
        # Getting the type of 'enfold' (line 316)
        enfold_324635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 15), 'enfold', False)
        # Calling enfold(args, kwargs) (line 316)
        enfold_call_result_324640 = invoke(stypy.reporting.localization.Localization(__file__, 316, 15), enfold_324635, *[dt_wall_324636], **kwargs_324639)
        
        # Assigning a type to the variable 'stypy_return_type' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'stypy_return_type', enfold_call_result_324640)
        
        # ################# End of 'fromutc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fromutc' in the type store
        # Getting the type of 'stypy_return_type' (line 285)
        stypy_return_type_324641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324641)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fromutc'
        return stypy_return_type_324641


    @norecursion
    def is_ambiguous(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_ambiguous'
        module_type_store = module_type_store.open_function_context('is_ambiguous', 318, 4, False)
        # Assigning a type to the variable 'self' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzrangebase.is_ambiguous.__dict__.__setitem__('stypy_localization', localization)
        tzrangebase.is_ambiguous.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzrangebase.is_ambiguous.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzrangebase.is_ambiguous.__dict__.__setitem__('stypy_function_name', 'tzrangebase.is_ambiguous')
        tzrangebase.is_ambiguous.__dict__.__setitem__('stypy_param_names_list', ['dt'])
        tzrangebase.is_ambiguous.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzrangebase.is_ambiguous.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzrangebase.is_ambiguous.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzrangebase.is_ambiguous.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzrangebase.is_ambiguous.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzrangebase.is_ambiguous.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzrangebase.is_ambiguous', ['dt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_ambiguous', localization, ['dt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_ambiguous(...)' code ##################

        str_324642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, (-1)), 'str', '\n        Whether or not the "wall time" of a given datetime is ambiguous in this\n        zone.\n\n        :param dt:\n            A :py:class:`datetime.datetime`, naive or time zone aware.\n\n\n        :return:\n            Returns ``True`` if ambiguous, ``False`` otherwise.\n\n        .. versionadded:: 2.6.0\n        ')
        
        
        # Getting the type of 'self' (line 332)
        self_324643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 15), 'self')
        # Obtaining the member 'hasdst' of a type (line 332)
        hasdst_324644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 15), self_324643, 'hasdst')
        # Applying the 'not' unary operator (line 332)
        result_not__324645 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 11), 'not', hasdst_324644)
        
        # Testing the type of an if condition (line 332)
        if_condition_324646 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 332, 8), result_not__324645)
        # Assigning a type to the variable 'if_condition_324646' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'if_condition_324646', if_condition_324646)
        # SSA begins for if statement (line 332)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 333)
        False_324647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'stypy_return_type', False_324647)
        # SSA join for if statement (line 332)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 335):
        
        # Assigning a Call to a Name:
        
        # Call to transitions(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'dt' (line 335)
        dt_324650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 38), 'dt', False)
        # Obtaining the member 'year' of a type (line 335)
        year_324651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 38), dt_324650, 'year')
        # Processing the call keyword arguments (line 335)
        kwargs_324652 = {}
        # Getting the type of 'self' (line 335)
        self_324648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 21), 'self', False)
        # Obtaining the member 'transitions' of a type (line 335)
        transitions_324649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 21), self_324648, 'transitions')
        # Calling transitions(args, kwargs) (line 335)
        transitions_call_result_324653 = invoke(stypy.reporting.localization.Localization(__file__, 335, 21), transitions_324649, *[year_324651], **kwargs_324652)
        
        # Assigning a type to the variable 'call_assignment_324219' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'call_assignment_324219', transitions_call_result_324653)
        
        # Assigning a Call to a Name (line 335):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_324656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 8), 'int')
        # Processing the call keyword arguments
        kwargs_324657 = {}
        # Getting the type of 'call_assignment_324219' (line 335)
        call_assignment_324219_324654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'call_assignment_324219', False)
        # Obtaining the member '__getitem__' of a type (line 335)
        getitem___324655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 8), call_assignment_324219_324654, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_324658 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___324655, *[int_324656], **kwargs_324657)
        
        # Assigning a type to the variable 'call_assignment_324220' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'call_assignment_324220', getitem___call_result_324658)
        
        # Assigning a Name to a Name (line 335):
        # Getting the type of 'call_assignment_324220' (line 335)
        call_assignment_324220_324659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'call_assignment_324220')
        # Assigning a type to the variable 'start' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'start', call_assignment_324220_324659)
        
        # Assigning a Call to a Name (line 335):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_324662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 8), 'int')
        # Processing the call keyword arguments
        kwargs_324663 = {}
        # Getting the type of 'call_assignment_324219' (line 335)
        call_assignment_324219_324660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'call_assignment_324219', False)
        # Obtaining the member '__getitem__' of a type (line 335)
        getitem___324661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 8), call_assignment_324219_324660, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_324664 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___324661, *[int_324662], **kwargs_324663)
        
        # Assigning a type to the variable 'call_assignment_324221' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'call_assignment_324221', getitem___call_result_324664)
        
        # Assigning a Name to a Name (line 335):
        # Getting the type of 'call_assignment_324221' (line 335)
        call_assignment_324221_324665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'call_assignment_324221')
        # Assigning a type to the variable 'end' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 15), 'end', call_assignment_324221_324665)
        
        # Assigning a Call to a Name (line 337):
        
        # Assigning a Call to a Name (line 337):
        
        # Call to replace(...): (line 337)
        # Processing the call keyword arguments (line 337)
        # Getting the type of 'None' (line 337)
        None_324668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 31), 'None', False)
        keyword_324669 = None_324668
        kwargs_324670 = {'tzinfo': keyword_324669}
        # Getting the type of 'dt' (line 337)
        dt_324666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 13), 'dt', False)
        # Obtaining the member 'replace' of a type (line 337)
        replace_324667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 13), dt_324666, 'replace')
        # Calling replace(args, kwargs) (line 337)
        replace_call_result_324671 = invoke(stypy.reporting.localization.Localization(__file__, 337, 13), replace_324667, *[], **kwargs_324670)
        
        # Assigning a type to the variable 'dt' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'dt', replace_call_result_324671)
        
        # Getting the type of 'end' (line 338)
        end_324672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 16), 'end')
        # Getting the type of 'dt' (line 338)
        dt_324673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 23), 'dt')
        # Applying the binary operator '<=' (line 338)
        result_le_324674 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 16), '<=', end_324672, dt_324673)
        # Getting the type of 'end' (line 338)
        end_324675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 28), 'end')
        # Getting the type of 'self' (line 338)
        self_324676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 34), 'self')
        # Obtaining the member '_dst_base_offset' of a type (line 338)
        _dst_base_offset_324677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 34), self_324676, '_dst_base_offset')
        # Applying the binary operator '+' (line 338)
        result_add_324678 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 28), '+', end_324675, _dst_base_offset_324677)
        
        # Applying the binary operator '<' (line 338)
        result_lt_324679 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 16), '<', dt_324673, result_add_324678)
        # Applying the binary operator '&' (line 338)
        result_and__324680 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 16), '&', result_le_324674, result_lt_324679)
        
        # Assigning a type to the variable 'stypy_return_type' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'stypy_return_type', result_and__324680)
        
        # ################# End of 'is_ambiguous(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_ambiguous' in the type store
        # Getting the type of 'stypy_return_type' (line 318)
        stypy_return_type_324681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324681)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_ambiguous'
        return stypy_return_type_324681


    @norecursion
    def _isdst(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_isdst'
        module_type_store = module_type_store.open_function_context('_isdst', 340, 4, False)
        # Assigning a type to the variable 'self' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzrangebase._isdst.__dict__.__setitem__('stypy_localization', localization)
        tzrangebase._isdst.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzrangebase._isdst.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzrangebase._isdst.__dict__.__setitem__('stypy_function_name', 'tzrangebase._isdst')
        tzrangebase._isdst.__dict__.__setitem__('stypy_param_names_list', ['dt'])
        tzrangebase._isdst.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzrangebase._isdst.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzrangebase._isdst.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzrangebase._isdst.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzrangebase._isdst.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzrangebase._isdst.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzrangebase._isdst', ['dt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_isdst', localization, ['dt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_isdst(...)' code ##################

        
        
        # Getting the type of 'self' (line 341)
        self_324682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 15), 'self')
        # Obtaining the member 'hasdst' of a type (line 341)
        hasdst_324683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 15), self_324682, 'hasdst')
        # Applying the 'not' unary operator (line 341)
        result_not__324684 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 11), 'not', hasdst_324683)
        
        # Testing the type of an if condition (line 341)
        if_condition_324685 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 341, 8), result_not__324684)
        # Assigning a type to the variable 'if_condition_324685' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'if_condition_324685', if_condition_324685)
        # SSA begins for if statement (line 341)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 342)
        False_324686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'stypy_return_type', False_324686)
        # SSA branch for the else part of an if statement (line 341)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 343)
        # Getting the type of 'dt' (line 343)
        dt_324687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 13), 'dt')
        # Getting the type of 'None' (line 343)
        None_324688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 19), 'None')
        
        (may_be_324689, more_types_in_union_324690) = may_be_none(dt_324687, None_324688)

        if may_be_324689:

            if more_types_in_union_324690:
                # Runtime conditional SSA (line 343)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'None' (line 344)
            None_324691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 344)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'stypy_return_type', None_324691)

            if more_types_in_union_324690:
                # SSA join for if statement (line 343)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 341)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 346):
        
        # Assigning a Call to a Name (line 346):
        
        # Call to transitions(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'dt' (line 346)
        dt_324694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 39), 'dt', False)
        # Obtaining the member 'year' of a type (line 346)
        year_324695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 39), dt_324694, 'year')
        # Processing the call keyword arguments (line 346)
        kwargs_324696 = {}
        # Getting the type of 'self' (line 346)
        self_324692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 22), 'self', False)
        # Obtaining the member 'transitions' of a type (line 346)
        transitions_324693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 22), self_324692, 'transitions')
        # Calling transitions(args, kwargs) (line 346)
        transitions_call_result_324697 = invoke(stypy.reporting.localization.Localization(__file__, 346, 22), transitions_324693, *[year_324695], **kwargs_324696)
        
        # Assigning a type to the variable 'transitions' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'transitions', transitions_call_result_324697)
        
        # Type idiom detected: calculating its left and rigth part (line 348)
        # Getting the type of 'transitions' (line 348)
        transitions_324698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 11), 'transitions')
        # Getting the type of 'None' (line 348)
        None_324699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 26), 'None')
        
        (may_be_324700, more_types_in_union_324701) = may_be_none(transitions_324698, None_324699)

        if may_be_324700:

            if more_types_in_union_324701:
                # Runtime conditional SSA (line 348)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'False' (line 349)
            False_324702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 349)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'stypy_return_type', False_324702)

            if more_types_in_union_324701:
                # SSA join for if statement (line 348)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 351):
        
        # Assigning a Call to a Name (line 351):
        
        # Call to replace(...): (line 351)
        # Processing the call keyword arguments (line 351)
        # Getting the type of 'None' (line 351)
        None_324705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 31), 'None', False)
        keyword_324706 = None_324705
        kwargs_324707 = {'tzinfo': keyword_324706}
        # Getting the type of 'dt' (line 351)
        dt_324703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 13), 'dt', False)
        # Obtaining the member 'replace' of a type (line 351)
        replace_324704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 13), dt_324703, 'replace')
        # Calling replace(args, kwargs) (line 351)
        replace_call_result_324708 = invoke(stypy.reporting.localization.Localization(__file__, 351, 13), replace_324704, *[], **kwargs_324707)
        
        # Assigning a type to the variable 'dt' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'dt', replace_call_result_324708)
        
        # Assigning a Call to a Name (line 353):
        
        # Assigning a Call to a Name (line 353):
        
        # Call to _naive_isdst(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'dt' (line 353)
        dt_324711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 34), 'dt', False)
        # Getting the type of 'transitions' (line 353)
        transitions_324712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 38), 'transitions', False)
        # Processing the call keyword arguments (line 353)
        kwargs_324713 = {}
        # Getting the type of 'self' (line 353)
        self_324709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 16), 'self', False)
        # Obtaining the member '_naive_isdst' of a type (line 353)
        _naive_isdst_324710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 16), self_324709, '_naive_isdst')
        # Calling _naive_isdst(args, kwargs) (line 353)
        _naive_isdst_call_result_324714 = invoke(stypy.reporting.localization.Localization(__file__, 353, 16), _naive_isdst_324710, *[dt_324711, transitions_324712], **kwargs_324713)
        
        # Assigning a type to the variable 'isdst' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'isdst', _naive_isdst_call_result_324714)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'isdst' (line 356)
        isdst_324715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 15), 'isdst')
        # Applying the 'not' unary operator (line 356)
        result_not__324716 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 11), 'not', isdst_324715)
        
        
        # Call to is_ambiguous(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'dt' (line 356)
        dt_324719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 43), 'dt', False)
        # Processing the call keyword arguments (line 356)
        kwargs_324720 = {}
        # Getting the type of 'self' (line 356)
        self_324717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 25), 'self', False)
        # Obtaining the member 'is_ambiguous' of a type (line 356)
        is_ambiguous_324718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 25), self_324717, 'is_ambiguous')
        # Calling is_ambiguous(args, kwargs) (line 356)
        is_ambiguous_call_result_324721 = invoke(stypy.reporting.localization.Localization(__file__, 356, 25), is_ambiguous_324718, *[dt_324719], **kwargs_324720)
        
        # Applying the binary operator 'and' (line 356)
        result_and_keyword_324722 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 11), 'and', result_not__324716, is_ambiguous_call_result_324721)
        
        # Testing the type of an if condition (line 356)
        if_condition_324723 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 356, 8), result_and_keyword_324722)
        # Assigning a type to the variable 'if_condition_324723' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'if_condition_324723', if_condition_324723)
        # SSA begins for if statement (line 356)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to _fold(...): (line 357)
        # Processing the call arguments (line 357)
        # Getting the type of 'dt' (line 357)
        dt_324726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 34), 'dt', False)
        # Processing the call keyword arguments (line 357)
        kwargs_324727 = {}
        # Getting the type of 'self' (line 357)
        self_324724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 23), 'self', False)
        # Obtaining the member '_fold' of a type (line 357)
        _fold_324725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 23), self_324724, '_fold')
        # Calling _fold(args, kwargs) (line 357)
        _fold_call_result_324728 = invoke(stypy.reporting.localization.Localization(__file__, 357, 23), _fold_324725, *[dt_324726], **kwargs_324727)
        
        # Applying the 'not' unary operator (line 357)
        result_not__324729 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 19), 'not', _fold_call_result_324728)
        
        # Assigning a type to the variable 'stypy_return_type' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'stypy_return_type', result_not__324729)
        # SSA branch for the else part of an if statement (line 356)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'isdst' (line 359)
        isdst_324730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 19), 'isdst')
        # Assigning a type to the variable 'stypy_return_type' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'stypy_return_type', isdst_324730)
        # SSA join for if statement (line 356)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_isdst(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_isdst' in the type store
        # Getting the type of 'stypy_return_type' (line 340)
        stypy_return_type_324731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324731)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_isdst'
        return stypy_return_type_324731


    @norecursion
    def _naive_isdst(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_naive_isdst'
        module_type_store = module_type_store.open_function_context('_naive_isdst', 361, 4, False)
        # Assigning a type to the variable 'self' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzrangebase._naive_isdst.__dict__.__setitem__('stypy_localization', localization)
        tzrangebase._naive_isdst.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzrangebase._naive_isdst.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzrangebase._naive_isdst.__dict__.__setitem__('stypy_function_name', 'tzrangebase._naive_isdst')
        tzrangebase._naive_isdst.__dict__.__setitem__('stypy_param_names_list', ['dt', 'transitions'])
        tzrangebase._naive_isdst.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzrangebase._naive_isdst.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzrangebase._naive_isdst.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzrangebase._naive_isdst.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzrangebase._naive_isdst.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzrangebase._naive_isdst.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzrangebase._naive_isdst', ['dt', 'transitions'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_naive_isdst', localization, ['dt', 'transitions'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_naive_isdst(...)' code ##################

        
        # Assigning a Name to a Tuple (line 362):
        
        # Assigning a Subscript to a Name (line 362):
        
        # Obtaining the type of the subscript
        int_324732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 8), 'int')
        # Getting the type of 'transitions' (line 362)
        transitions_324733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 24), 'transitions')
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___324734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 8), transitions_324733, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_324735 = invoke(stypy.reporting.localization.Localization(__file__, 362, 8), getitem___324734, int_324732)
        
        # Assigning a type to the variable 'tuple_var_assignment_324222' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'tuple_var_assignment_324222', subscript_call_result_324735)
        
        # Assigning a Subscript to a Name (line 362):
        
        # Obtaining the type of the subscript
        int_324736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 8), 'int')
        # Getting the type of 'transitions' (line 362)
        transitions_324737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 24), 'transitions')
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___324738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 8), transitions_324737, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_324739 = invoke(stypy.reporting.localization.Localization(__file__, 362, 8), getitem___324738, int_324736)
        
        # Assigning a type to the variable 'tuple_var_assignment_324223' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'tuple_var_assignment_324223', subscript_call_result_324739)
        
        # Assigning a Name to a Name (line 362):
        # Getting the type of 'tuple_var_assignment_324222' (line 362)
        tuple_var_assignment_324222_324740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'tuple_var_assignment_324222')
        # Assigning a type to the variable 'dston' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'dston', tuple_var_assignment_324222_324740)
        
        # Assigning a Name to a Name (line 362):
        # Getting the type of 'tuple_var_assignment_324223' (line 362)
        tuple_var_assignment_324223_324741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'tuple_var_assignment_324223')
        # Assigning a type to the variable 'dstoff' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 15), 'dstoff', tuple_var_assignment_324223_324741)
        
        # Assigning a Call to a Name (line 364):
        
        # Assigning a Call to a Name (line 364):
        
        # Call to replace(...): (line 364)
        # Processing the call keyword arguments (line 364)
        # Getting the type of 'None' (line 364)
        None_324744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 31), 'None', False)
        keyword_324745 = None_324744
        kwargs_324746 = {'tzinfo': keyword_324745}
        # Getting the type of 'dt' (line 364)
        dt_324742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 13), 'dt', False)
        # Obtaining the member 'replace' of a type (line 364)
        replace_324743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 13), dt_324742, 'replace')
        # Calling replace(args, kwargs) (line 364)
        replace_call_result_324747 = invoke(stypy.reporting.localization.Localization(__file__, 364, 13), replace_324743, *[], **kwargs_324746)
        
        # Assigning a type to the variable 'dt' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'dt', replace_call_result_324747)
        
        
        # Getting the type of 'dston' (line 366)
        dston_324748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 11), 'dston')
        # Getting the type of 'dstoff' (line 366)
        dstoff_324749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 19), 'dstoff')
        # Applying the binary operator '<' (line 366)
        result_lt_324750 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 11), '<', dston_324748, dstoff_324749)
        
        # Testing the type of an if condition (line 366)
        if_condition_324751 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 366, 8), result_lt_324750)
        # Assigning a type to the variable 'if_condition_324751' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'if_condition_324751', if_condition_324751)
        # SSA begins for if statement (line 366)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Compare to a Name (line 367):
        
        # Assigning a Compare to a Name (line 367):
        
        # Getting the type of 'dston' (line 367)
        dston_324752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 20), 'dston')
        # Getting the type of 'dt' (line 367)
        dt_324753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 29), 'dt')
        # Applying the binary operator '<=' (line 367)
        result_le_324754 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 20), '<=', dston_324752, dt_324753)
        # Getting the type of 'dstoff' (line 367)
        dstoff_324755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 34), 'dstoff')
        # Applying the binary operator '<' (line 367)
        result_lt_324756 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 20), '<', dt_324753, dstoff_324755)
        # Applying the binary operator '&' (line 367)
        result_and__324757 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 20), '&', result_le_324754, result_lt_324756)
        
        # Assigning a type to the variable 'isdst' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'isdst', result_and__324757)
        # SSA branch for the else part of an if statement (line 366)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a UnaryOp to a Name (line 369):
        
        # Assigning a UnaryOp to a Name (line 369):
        
        
        # Getting the type of 'dstoff' (line 369)
        dstoff_324758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 24), 'dstoff')
        # Getting the type of 'dt' (line 369)
        dt_324759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 34), 'dt')
        # Applying the binary operator '<=' (line 369)
        result_le_324760 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 24), '<=', dstoff_324758, dt_324759)
        # Getting the type of 'dston' (line 369)
        dston_324761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 39), 'dston')
        # Applying the binary operator '<' (line 369)
        result_lt_324762 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 24), '<', dt_324759, dston_324761)
        # Applying the binary operator '&' (line 369)
        result_and__324763 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 24), '&', result_le_324760, result_lt_324762)
        
        # Applying the 'not' unary operator (line 369)
        result_not__324764 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 20), 'not', result_and__324763)
        
        # Assigning a type to the variable 'isdst' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'isdst', result_not__324764)
        # SSA join for if statement (line 366)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'isdst' (line 371)
        isdst_324765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 15), 'isdst')
        # Assigning a type to the variable 'stypy_return_type' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'stypy_return_type', isdst_324765)
        
        # ################# End of '_naive_isdst(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_naive_isdst' in the type store
        # Getting the type of 'stypy_return_type' (line 361)
        stypy_return_type_324766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324766)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_naive_isdst'
        return stypy_return_type_324766


    @norecursion
    def _dst_base_offset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_dst_base_offset'
        module_type_store = module_type_store.open_function_context('_dst_base_offset', 373, 4, False)
        # Assigning a type to the variable 'self' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzrangebase._dst_base_offset.__dict__.__setitem__('stypy_localization', localization)
        tzrangebase._dst_base_offset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzrangebase._dst_base_offset.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzrangebase._dst_base_offset.__dict__.__setitem__('stypy_function_name', 'tzrangebase._dst_base_offset')
        tzrangebase._dst_base_offset.__dict__.__setitem__('stypy_param_names_list', [])
        tzrangebase._dst_base_offset.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzrangebase._dst_base_offset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzrangebase._dst_base_offset.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzrangebase._dst_base_offset.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzrangebase._dst_base_offset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzrangebase._dst_base_offset.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzrangebase._dst_base_offset', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_dst_base_offset', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_dst_base_offset(...)' code ##################

        # Getting the type of 'self' (line 375)
        self_324767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 15), 'self')
        # Obtaining the member '_dst_offset' of a type (line 375)
        _dst_offset_324768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 15), self_324767, '_dst_offset')
        # Getting the type of 'self' (line 375)
        self_324769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 34), 'self')
        # Obtaining the member '_std_offset' of a type (line 375)
        _std_offset_324770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 34), self_324769, '_std_offset')
        # Applying the binary operator '-' (line 375)
        result_sub_324771 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 15), '-', _dst_offset_324768, _std_offset_324770)
        
        # Assigning a type to the variable 'stypy_return_type' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'stypy_return_type', result_sub_324771)
        
        # ################# End of '_dst_base_offset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_dst_base_offset' in the type store
        # Getting the type of 'stypy_return_type' (line 373)
        stypy_return_type_324772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324772)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_dst_base_offset'
        return stypy_return_type_324772

    
    # Assigning a Name to a Name (line 377):

    @norecursion
    def __ne__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ne__'
        module_type_store = module_type_store.open_function_context('__ne__', 379, 4, False)
        # Assigning a type to the variable 'self' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzrangebase.__ne__.__dict__.__setitem__('stypy_localization', localization)
        tzrangebase.__ne__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzrangebase.__ne__.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzrangebase.__ne__.__dict__.__setitem__('stypy_function_name', 'tzrangebase.__ne__')
        tzrangebase.__ne__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        tzrangebase.__ne__.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzrangebase.__ne__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzrangebase.__ne__.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzrangebase.__ne__.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzrangebase.__ne__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzrangebase.__ne__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzrangebase.__ne__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__ne__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__ne__(...)' code ##################

        
        
        # Getting the type of 'self' (line 380)
        self_324773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 20), 'self')
        # Getting the type of 'other' (line 380)
        other_324774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 28), 'other')
        # Applying the binary operator '==' (line 380)
        result_eq_324775 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 20), '==', self_324773, other_324774)
        
        # Applying the 'not' unary operator (line 380)
        result_not__324776 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 15), 'not', result_eq_324775)
        
        # Assigning a type to the variable 'stypy_return_type' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'stypy_return_type', result_not__324776)
        
        # ################# End of '__ne__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ne__' in the type store
        # Getting the type of 'stypy_return_type' (line 379)
        stypy_return_type_324777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324777)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ne__'
        return stypy_return_type_324777


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 382, 4, False)
        # Assigning a type to the variable 'self' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzrangebase.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        tzrangebase.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzrangebase.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzrangebase.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'tzrangebase.stypy__repr__')
        tzrangebase.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        tzrangebase.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzrangebase.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzrangebase.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzrangebase.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzrangebase.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzrangebase.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzrangebase.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        str_324778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 15), 'str', '%s(...)')
        # Getting the type of 'self' (line 383)
        self_324779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 27), 'self')
        # Obtaining the member '__class__' of a type (line 383)
        class___324780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 27), self_324779, '__class__')
        # Obtaining the member '__name__' of a type (line 383)
        name___324781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 27), class___324780, '__name__')
        # Applying the binary operator '%' (line 383)
        result_mod_324782 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 15), '%', str_324778, name___324781)
        
        # Assigning a type to the variable 'stypy_return_type' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'stypy_return_type', result_mod_324782)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 382)
        stypy_return_type_324783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324783)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_324783

    
    # Assigning a Attribute to a Name (line 385):

# Assigning a type to the variable 'tzrangebase' (line 233)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 0), 'tzrangebase', tzrangebase)

# Assigning a Name to a Name (line 377):
# Getting the type of 'None' (line 377)
None_324784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 15), 'None')
# Getting the type of 'tzrangebase'
tzrangebase_324785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'tzrangebase')
# Setting the type of the member '__hash__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), tzrangebase_324785, '__hash__', None_324784)

# Assigning a Attribute to a Name (line 385):
# Getting the type of 'object' (line 385)
object_324786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 17), 'object')
# Obtaining the member '__reduce__' of a type (line 385)
reduce___324787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 17), object_324786, '__reduce__')
# Getting the type of 'tzrangebase'
tzrangebase_324788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'tzrangebase')
# Setting the type of the member '__reduce__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), tzrangebase_324788, '__reduce__', reduce___324787)

@norecursion
def _total_seconds(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_total_seconds'
    module_type_store = module_type_store.open_function_context('_total_seconds', 388, 0, False)
    
    # Passed parameters checking function
    _total_seconds.stypy_localization = localization
    _total_seconds.stypy_type_of_self = None
    _total_seconds.stypy_type_store = module_type_store
    _total_seconds.stypy_function_name = '_total_seconds'
    _total_seconds.stypy_param_names_list = ['td']
    _total_seconds.stypy_varargs_param_name = None
    _total_seconds.stypy_kwargs_param_name = None
    _total_seconds.stypy_call_defaults = defaults
    _total_seconds.stypy_call_varargs = varargs
    _total_seconds.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_total_seconds', ['td'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_total_seconds', localization, ['td'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_total_seconds(...)' code ##################

    # Getting the type of 'td' (line 390)
    td_324789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 13), 'td')
    # Obtaining the member 'seconds' of a type (line 390)
    seconds_324790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 13), td_324789, 'seconds')
    # Getting the type of 'td' (line 390)
    td_324791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 26), 'td')
    # Obtaining the member 'days' of a type (line 390)
    days_324792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 26), td_324791, 'days')
    int_324793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 36), 'int')
    # Applying the binary operator '*' (line 390)
    result_mul_324794 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 26), '*', days_324792, int_324793)
    
    # Applying the binary operator '+' (line 390)
    result_add_324795 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 13), '+', seconds_324790, result_mul_324794)
    
    int_324796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 45), 'int')
    # Applying the binary operator '*' (line 390)
    result_mul_324797 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 12), '*', result_add_324795, int_324796)
    
    # Getting the type of 'td' (line 391)
    td_324798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'td')
    # Obtaining the member 'microseconds' of a type (line 391)
    microseconds_324799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 12), td_324798, 'microseconds')
    # Applying the binary operator '+' (line 390)
    result_add_324800 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 12), '+', result_mul_324797, microseconds_324799)
    
    int_324801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 32), 'int')
    # Applying the binary operator '//' (line 390)
    result_floordiv_324802 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 11), '//', result_add_324800, int_324801)
    
    # Assigning a type to the variable 'stypy_return_type' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'stypy_return_type', result_floordiv_324802)
    
    # ################# End of '_total_seconds(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_total_seconds' in the type store
    # Getting the type of 'stypy_return_type' (line 388)
    stypy_return_type_324803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_324803)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_total_seconds'
    return stypy_return_type_324803

# Assigning a type to the variable '_total_seconds' (line 388)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 0), '_total_seconds', _total_seconds)

# Assigning a Call to a Name (line 394):

# Assigning a Call to a Name (line 394):

# Call to getattr(...): (line 394)
# Processing the call arguments (line 394)
# Getting the type of 'timedelta' (line 394)
timedelta_324805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 25), 'timedelta', False)
str_324806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 36), 'str', 'total_seconds')
# Getting the type of '_total_seconds' (line 394)
_total_seconds_324807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 53), '_total_seconds', False)
# Processing the call keyword arguments (line 394)
kwargs_324808 = {}
# Getting the type of 'getattr' (line 394)
getattr_324804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 17), 'getattr', False)
# Calling getattr(args, kwargs) (line 394)
getattr_call_result_324809 = invoke(stypy.reporting.localization.Localization(__file__, 394, 17), getattr_324804, *[timedelta_324805, str_324806, _total_seconds_324807], **kwargs_324808)

# Assigning a type to the variable '_total_seconds' (line 394)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 0), '_total_seconds', getattr_call_result_324809)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
