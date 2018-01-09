
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # This code was originally contributed by Jeffrey Harris.
2: import datetime
3: import struct
4: 
5: from six.moves import winreg
6: from six import text_type
7: 
8: try:
9:     import ctypes
10:     from ctypes import wintypes
11: except ValueError:
12:     # ValueError is raised on non-Windows systems for some horrible reason.
13:     raise ImportError("Running tzwin on non-Windows system")
14: 
15: from ._common import tzrangebase
16: 
17: __all__ = ["tzwin", "tzwinlocal", "tzres"]
18: 
19: ONEWEEK = datetime.timedelta(7)
20: 
21: TZKEYNAMENT = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Time Zones"
22: TZKEYNAME9X = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Time Zones"
23: TZLOCALKEYNAME = r"SYSTEM\CurrentControlSet\Control\TimeZoneInformation"
24: 
25: 
26: def _settzkeyname():
27:     handle = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
28:     try:
29:         winreg.OpenKey(handle, TZKEYNAMENT).Close()
30:         TZKEYNAME = TZKEYNAMENT
31:     except WindowsError:
32:         TZKEYNAME = TZKEYNAME9X
33:     handle.Close()
34:     return TZKEYNAME
35: 
36: 
37: TZKEYNAME = _settzkeyname()
38: 
39: 
40: class tzres(object):
41:     '''
42:     Class for accessing `tzres.dll`, which contains timezone name related
43:     resources.
44: 
45:     .. versionadded:: 2.5.0
46:     '''
47:     p_wchar = ctypes.POINTER(wintypes.WCHAR)        # Pointer to a wide char
48: 
49:     def __init__(self, tzres_loc='tzres.dll'):
50:         # Load the user32 DLL so we can load strings from tzres
51:         user32 = ctypes.WinDLL('user32')
52: 
53:         # Specify the LoadStringW function
54:         user32.LoadStringW.argtypes = (wintypes.HINSTANCE,
55:                                        wintypes.UINT,
56:                                        wintypes.LPWSTR,
57:                                        ctypes.c_int)
58: 
59:         self.LoadStringW = user32.LoadStringW
60:         self._tzres = ctypes.WinDLL(tzres_loc)
61:         self.tzres_loc = tzres_loc
62: 
63:     def load_name(self, offset):
64:         '''
65:         Load a timezone name from a DLL offset (integer).
66: 
67:         >>> from dateutil.tzwin import tzres
68:         >>> tzr = tzres()
69:         >>> print(tzr.load_name(112))
70:         'Eastern Standard Time'
71: 
72:         :param offset:
73:             A positive integer value referring to a string from the tzres dll.
74: 
75:         ..note:
76:             Offsets found in the registry are generally of the form
77:             `@tzres.dll,-114`. The offset in this case if 114, not -114.
78: 
79:         '''
80:         resource = self.p_wchar()
81:         lpBuffer = ctypes.cast(ctypes.byref(resource), wintypes.LPWSTR)
82:         nchar = self.LoadStringW(self._tzres._handle, offset, lpBuffer, 0)
83:         return resource[:nchar]
84: 
85:     def name_from_string(self, tzname_str):
86:         '''
87:         Parse strings as returned from the Windows registry into the time zone
88:         name as defined in the registry.
89: 
90:         >>> from dateutil.tzwin import tzres
91:         >>> tzr = tzres()
92:         >>> print(tzr.name_from_string('@tzres.dll,-251'))
93:         'Dateline Daylight Time'
94:         >>> print(tzr.name_from_string('Eastern Standard Time'))
95:         'Eastern Standard Time'
96: 
97:         :param tzname_str:
98:             A timezone name string as returned from a Windows registry key.
99: 
100:         :return:
101:             Returns the localized timezone string from tzres.dll if the string
102:             is of the form `@tzres.dll,-offset`, else returns the input string.
103:         '''
104:         if not tzname_str.startswith('@'):
105:             return tzname_str
106: 
107:         name_splt = tzname_str.split(',-')
108:         try:
109:             offset = int(name_splt[1])
110:         except:
111:             raise ValueError("Malformed timezone string.")
112: 
113:         return self.load_name(offset)
114: 
115: 
116: class tzwinbase(tzrangebase):
117:     '''tzinfo class based on win32's timezones available in the registry.'''
118:     def __init__(self):
119:         raise NotImplementedError('tzwinbase is an abstract base class')
120: 
121:     def __eq__(self, other):
122:         # Compare on all relevant dimensions, including name.
123:         if not isinstance(other, tzwinbase):
124:             return NotImplemented
125: 
126:         return  (self._std_offset == other._std_offset and
127:                  self._dst_offset == other._dst_offset and
128:                  self._stddayofweek == other._stddayofweek and
129:                  self._dstdayofweek == other._dstdayofweek and
130:                  self._stdweeknumber == other._stdweeknumber and
131:                  self._dstweeknumber == other._dstweeknumber and
132:                  self._stdhour == other._stdhour and
133:                  self._dsthour == other._dsthour and
134:                  self._stdminute == other._stdminute and
135:                  self._dstminute == other._dstminute and
136:                  self._std_abbr == other._std_abbr and
137:                  self._dst_abbr == other._dst_abbr)
138: 
139:     @staticmethod
140:     def list():
141:         '''Return a list of all time zones known to the system.'''
142:         with winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE) as handle:
143:             with winreg.OpenKey(handle, TZKEYNAME) as tzkey:
144:                 result = [winreg.EnumKey(tzkey, i)
145:                           for i in range(winreg.QueryInfoKey(tzkey)[0])]
146:         return result
147: 
148:     def display(self):
149:         return self._display
150: 
151:     def transitions(self, year):
152:         '''
153:         For a given year, get the DST on and off transition times, expressed
154:         always on the standard time side. For zones with no transitions, this
155:         function returns ``None``.
156: 
157:         :param year:
158:             The year whose transitions you would like to query.
159: 
160:         :return:
161:             Returns a :class:`tuple` of :class:`datetime.datetime` objects,
162:             ``(dston, dstoff)`` for zones with an annual DST transition, or
163:             ``None`` for fixed offset zones.
164:         '''
165: 
166:         if not self.hasdst:
167:             return None
168: 
169:         dston = picknthweekday(year, self._dstmonth, self._dstdayofweek,
170:                                self._dsthour, self._dstminute,
171:                                self._dstweeknumber)
172: 
173:         dstoff = picknthweekday(year, self._stdmonth, self._stddayofweek,
174:                                 self._stdhour, self._stdminute,
175:                                 self._stdweeknumber)
176: 
177:         # Ambiguous dates default to the STD side
178:         dstoff -= self._dst_base_offset
179: 
180:         return dston, dstoff
181: 
182:     def _get_hasdst(self):
183:         return self._dstmonth != 0
184: 
185:     @property
186:     def _dst_base_offset(self):
187:         return self._dst_base_offset_
188: 
189: 
190: class tzwin(tzwinbase):
191: 
192:     def __init__(self, name):
193:         self._name = name
194: 
195:         # multiple contexts only possible in 2.7 and 3.1, we still support 2.6
196:         with winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE) as handle:
197:             tzkeyname = text_type("{kn}\\{name}").format(kn=TZKEYNAME, name=name)
198:             with winreg.OpenKey(handle, tzkeyname) as tzkey:
199:                 keydict = valuestodict(tzkey)
200: 
201:         self._std_abbr = keydict["Std"]
202:         self._dst_abbr = keydict["Dlt"]
203: 
204:         self._display = keydict["Display"]
205: 
206:         # See http://ww_winreg.jsiinc.com/SUBA/tip0300/rh0398.htm
207:         tup = struct.unpack("=3l16h", keydict["TZI"])
208:         stdoffset = -tup[0]-tup[1]          # Bias + StandardBias * -1
209:         dstoffset = stdoffset-tup[2]        # + DaylightBias * -1
210:         self._std_offset = datetime.timedelta(minutes=stdoffset)
211:         self._dst_offset = datetime.timedelta(minutes=dstoffset)
212: 
213:         # for the meaning see the win32 TIME_ZONE_INFORMATION structure docs
214:         # http://msdn.microsoft.com/en-us/library/windows/desktop/ms725481(v=vs.85).aspx
215:         (self._stdmonth,
216:          self._stddayofweek,   # Sunday = 0
217:          self._stdweeknumber,  # Last = 5
218:          self._stdhour,
219:          self._stdminute) = tup[4:9]
220: 
221:         (self._dstmonth,
222:          self._dstdayofweek,   # Sunday = 0
223:          self._dstweeknumber,  # Last = 5
224:          self._dsthour,
225:          self._dstminute) = tup[12:17]
226: 
227:         self._dst_base_offset_ = self._dst_offset - self._std_offset
228:         self.hasdst = self._get_hasdst()
229: 
230:     def __repr__(self):
231:         return "tzwin(%s)" % repr(self._name)
232: 
233:     def __reduce__(self):
234:         return (self.__class__, (self._name,))
235: 
236: 
237: class tzwinlocal(tzwinbase):
238:     def __init__(self):
239:         with winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE) as handle:
240:             with winreg.OpenKey(handle, TZLOCALKEYNAME) as tzlocalkey:
241:                 keydict = valuestodict(tzlocalkey)
242: 
243:             self._std_abbr = keydict["StandardName"]
244:             self._dst_abbr = keydict["DaylightName"]
245: 
246:             try:
247:                 tzkeyname = text_type('{kn}\\{sn}').format(kn=TZKEYNAME,
248:                                                           sn=self._std_abbr)
249:                 with winreg.OpenKey(handle, tzkeyname) as tzkey:
250:                     _keydict = valuestodict(tzkey)
251:                     self._display = _keydict["Display"]
252:             except OSError:
253:                 self._display = None
254: 
255:         stdoffset = -keydict["Bias"]-keydict["StandardBias"]
256:         dstoffset = stdoffset-keydict["DaylightBias"]
257: 
258:         self._std_offset = datetime.timedelta(minutes=stdoffset)
259:         self._dst_offset = datetime.timedelta(minutes=dstoffset)
260: 
261:         # For reasons unclear, in this particular key, the day of week has been
262:         # moved to the END of the SYSTEMTIME structure.
263:         tup = struct.unpack("=8h", keydict["StandardStart"])
264: 
265:         (self._stdmonth,
266:          self._stdweeknumber,  # Last = 5
267:          self._stdhour,
268:          self._stdminute) = tup[1:5]
269: 
270:         self._stddayofweek = tup[7]
271: 
272:         tup = struct.unpack("=8h", keydict["DaylightStart"])
273: 
274:         (self._dstmonth,
275:          self._dstweeknumber,  # Last = 5
276:          self._dsthour,
277:          self._dstminute) = tup[1:5]
278: 
279:         self._dstdayofweek = tup[7]
280: 
281:         self._dst_base_offset_ = self._dst_offset - self._std_offset
282:         self.hasdst = self._get_hasdst()
283: 
284:     def __repr__(self):
285:         return "tzwinlocal()"
286: 
287:     def __str__(self):
288:         # str will return the standard name, not the daylight name.
289:         return "tzwinlocal(%s)" % repr(self._std_abbr)
290: 
291:     def __reduce__(self):
292:         return (self.__class__, ())
293: 
294: 
295: def picknthweekday(year, month, dayofweek, hour, minute, whichweek):
296:     ''' dayofweek == 0 means Sunday, whichweek 5 means last instance '''
297:     first = datetime.datetime(year, month, 1, hour, minute)
298: 
299:     # This will work if dayofweek is ISO weekday (1-7) or Microsoft-style (0-6),
300:     # Because 7 % 7 = 0
301:     weekdayone = first.replace(day=((dayofweek - first.isoweekday()) % 7) + 1)
302:     wd = weekdayone + ((whichweek - 1) * ONEWEEK)
303:     if (wd.month != month):
304:         wd -= ONEWEEK
305: 
306:     return wd
307: 
308: 
309: def valuestodict(key):
310:     '''Convert a registry key's values to a dictionary.'''
311:     dout = {}
312:     size = winreg.QueryInfoKey(key)[1]
313:     tz_res = None
314: 
315:     for i in range(size):
316:         key_name, value, dtype = winreg.EnumValue(key, i)
317:         if dtype == winreg.REG_DWORD or dtype == winreg.REG_DWORD_LITTLE_ENDIAN:
318:             # If it's a DWORD (32-bit integer), it's stored as unsigned - convert
319:             # that to a proper signed integer
320:             if value & (1 << 31):
321:                 value = value - (1 << 32)
322:         elif dtype == winreg.REG_SZ:
323:             # If it's a reference to the tzres DLL, load the actual string
324:             if value.startswith('@tzres'):
325:                 tz_res = tz_res or tzres()
326:                 value = tz_res.name_from_string(value)
327: 
328:             value = value.rstrip('\x00')    # Remove trailing nulls
329: 
330:         dout[key_name] = value
331: 
332:     return dout
333: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import datetime' statement (line 2)
import datetime

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'datetime', datetime, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import struct' statement (line 3)
import struct

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'struct', struct, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from six.moves import winreg' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/dateutil/tz/')
import_323276 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'six.moves')

if (type(import_323276) is not StypyTypeError):

    if (import_323276 != 'pyd_module'):
        __import__(import_323276)
        sys_modules_323277 = sys.modules[import_323276]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'six.moves', sys_modules_323277.module_type_store, module_type_store, ['winreg'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_323277, sys_modules_323277.module_type_store, module_type_store)
    else:
        from six.moves import winreg

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'six.moves', None, module_type_store, ['winreg'], [winreg])

else:
    # Assigning a type to the variable 'six.moves' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'six.moves', import_323276)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/dateutil/tz/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from six import text_type' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/dateutil/tz/')
import_323278 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'six')

if (type(import_323278) is not StypyTypeError):

    if (import_323278 != 'pyd_module'):
        __import__(import_323278)
        sys_modules_323279 = sys.modules[import_323278]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'six', sys_modules_323279.module_type_store, module_type_store, ['text_type'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_323279, sys_modules_323279.module_type_store, module_type_store)
    else:
        from six import text_type

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'six', None, module_type_store, ['text_type'], [text_type])

else:
    # Assigning a type to the variable 'six' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'six', import_323278)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/dateutil/tz/')



# SSA begins for try-except statement (line 8)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 4))

# 'import ctypes' statement (line 9)
import ctypes

import_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'ctypes', ctypes, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 4))

# 'from ctypes import wintypes' statement (line 10)
try:
    from ctypes import wintypes

except:
    wintypes = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'ctypes', None, module_type_store, ['wintypes'], [wintypes])

# SSA branch for the except part of a try statement (line 8)
# SSA branch for the except 'ValueError' branch of a try statement (line 8)
module_type_store.open_ssa_branch('except')

# Call to ImportError(...): (line 13)
# Processing the call arguments (line 13)
str_323281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 22), 'str', 'Running tzwin on non-Windows system')
# Processing the call keyword arguments (line 13)
kwargs_323282 = {}
# Getting the type of 'ImportError' (line 13)
ImportError_323280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'ImportError', False)
# Calling ImportError(args, kwargs) (line 13)
ImportError_call_result_323283 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), ImportError_323280, *[str_323281], **kwargs_323282)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 13, 4), ImportError_call_result_323283, 'raise parameter', BaseException)
# SSA join for try-except statement (line 8)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from dateutil.tz._common import tzrangebase' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/dateutil/tz/')
import_323284 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'dateutil.tz._common')

if (type(import_323284) is not StypyTypeError):

    if (import_323284 != 'pyd_module'):
        __import__(import_323284)
        sys_modules_323285 = sys.modules[import_323284]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'dateutil.tz._common', sys_modules_323285.module_type_store, module_type_store, ['tzrangebase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_323285, sys_modules_323285.module_type_store, module_type_store)
    else:
        from dateutil.tz._common import tzrangebase

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'dateutil.tz._common', None, module_type_store, ['tzrangebase'], [tzrangebase])

else:
    # Assigning a type to the variable 'dateutil.tz._common' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'dateutil.tz._common', import_323284)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/dateutil/tz/')


# Assigning a List to a Name (line 17):

# Assigning a List to a Name (line 17):
__all__ = ['tzwin', 'tzwinlocal', 'tzres']
module_type_store.set_exportable_members(['tzwin', 'tzwinlocal', 'tzres'])

# Obtaining an instance of the builtin type 'list' (line 17)
list_323286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
str_323287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 11), 'str', 'tzwin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_323286, str_323287)
# Adding element type (line 17)
str_323288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 20), 'str', 'tzwinlocal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_323286, str_323288)
# Adding element type (line 17)
str_323289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 34), 'str', 'tzres')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_323286, str_323289)

# Assigning a type to the variable '__all__' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), '__all__', list_323286)

# Assigning a Call to a Name (line 19):

# Assigning a Call to a Name (line 19):

# Call to timedelta(...): (line 19)
# Processing the call arguments (line 19)
int_323292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 29), 'int')
# Processing the call keyword arguments (line 19)
kwargs_323293 = {}
# Getting the type of 'datetime' (line 19)
datetime_323290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'datetime', False)
# Obtaining the member 'timedelta' of a type (line 19)
timedelta_323291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 10), datetime_323290, 'timedelta')
# Calling timedelta(args, kwargs) (line 19)
timedelta_call_result_323294 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), timedelta_323291, *[int_323292], **kwargs_323293)

# Assigning a type to the variable 'ONEWEEK' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'ONEWEEK', timedelta_call_result_323294)

# Assigning a Str to a Name (line 21):

# Assigning a Str to a Name (line 21):
str_323295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 14), 'str', 'SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Time Zones')
# Assigning a type to the variable 'TZKEYNAMENT' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'TZKEYNAMENT', str_323295)

# Assigning a Str to a Name (line 22):

# Assigning a Str to a Name (line 22):
str_323296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 14), 'str', 'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Time Zones')
# Assigning a type to the variable 'TZKEYNAME9X' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'TZKEYNAME9X', str_323296)

# Assigning a Str to a Name (line 23):

# Assigning a Str to a Name (line 23):
str_323297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 17), 'str', 'SYSTEM\\CurrentControlSet\\Control\\TimeZoneInformation')
# Assigning a type to the variable 'TZLOCALKEYNAME' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'TZLOCALKEYNAME', str_323297)

@norecursion
def _settzkeyname(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_settzkeyname'
    module_type_store = module_type_store.open_function_context('_settzkeyname', 26, 0, False)
    
    # Passed parameters checking function
    _settzkeyname.stypy_localization = localization
    _settzkeyname.stypy_type_of_self = None
    _settzkeyname.stypy_type_store = module_type_store
    _settzkeyname.stypy_function_name = '_settzkeyname'
    _settzkeyname.stypy_param_names_list = []
    _settzkeyname.stypy_varargs_param_name = None
    _settzkeyname.stypy_kwargs_param_name = None
    _settzkeyname.stypy_call_defaults = defaults
    _settzkeyname.stypy_call_varargs = varargs
    _settzkeyname.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_settzkeyname', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_settzkeyname', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_settzkeyname(...)' code ##################

    
    # Assigning a Call to a Name (line 27):
    
    # Assigning a Call to a Name (line 27):
    
    # Call to ConnectRegistry(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'None' (line 27)
    None_323300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 36), 'None', False)
    # Getting the type of 'winreg' (line 27)
    winreg_323301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 42), 'winreg', False)
    # Obtaining the member 'HKEY_LOCAL_MACHINE' of a type (line 27)
    HKEY_LOCAL_MACHINE_323302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 42), winreg_323301, 'HKEY_LOCAL_MACHINE')
    # Processing the call keyword arguments (line 27)
    kwargs_323303 = {}
    # Getting the type of 'winreg' (line 27)
    winreg_323298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 13), 'winreg', False)
    # Obtaining the member 'ConnectRegistry' of a type (line 27)
    ConnectRegistry_323299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 13), winreg_323298, 'ConnectRegistry')
    # Calling ConnectRegistry(args, kwargs) (line 27)
    ConnectRegistry_call_result_323304 = invoke(stypy.reporting.localization.Localization(__file__, 27, 13), ConnectRegistry_323299, *[None_323300, HKEY_LOCAL_MACHINE_323302], **kwargs_323303)
    
    # Assigning a type to the variable 'handle' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'handle', ConnectRegistry_call_result_323304)
    
    
    # SSA begins for try-except statement (line 28)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to Close(...): (line 29)
    # Processing the call keyword arguments (line 29)
    kwargs_323312 = {}
    
    # Call to OpenKey(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'handle' (line 29)
    handle_323307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 23), 'handle', False)
    # Getting the type of 'TZKEYNAMENT' (line 29)
    TZKEYNAMENT_323308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 31), 'TZKEYNAMENT', False)
    # Processing the call keyword arguments (line 29)
    kwargs_323309 = {}
    # Getting the type of 'winreg' (line 29)
    winreg_323305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'winreg', False)
    # Obtaining the member 'OpenKey' of a type (line 29)
    OpenKey_323306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), winreg_323305, 'OpenKey')
    # Calling OpenKey(args, kwargs) (line 29)
    OpenKey_call_result_323310 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), OpenKey_323306, *[handle_323307, TZKEYNAMENT_323308], **kwargs_323309)
    
    # Obtaining the member 'Close' of a type (line 29)
    Close_323311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), OpenKey_call_result_323310, 'Close')
    # Calling Close(args, kwargs) (line 29)
    Close_call_result_323313 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), Close_323311, *[], **kwargs_323312)
    
    
    # Assigning a Name to a Name (line 30):
    
    # Assigning a Name to a Name (line 30):
    # Getting the type of 'TZKEYNAMENT' (line 30)
    TZKEYNAMENT_323314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 20), 'TZKEYNAMENT')
    # Assigning a type to the variable 'TZKEYNAME' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'TZKEYNAME', TZKEYNAMENT_323314)
    # SSA branch for the except part of a try statement (line 28)
    # SSA branch for the except 'WindowsError' branch of a try statement (line 28)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 32):
    
    # Assigning a Name to a Name (line 32):
    # Getting the type of 'TZKEYNAME9X' (line 32)
    TZKEYNAME9X_323315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 20), 'TZKEYNAME9X')
    # Assigning a type to the variable 'TZKEYNAME' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'TZKEYNAME', TZKEYNAME9X_323315)
    # SSA join for try-except statement (line 28)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to Close(...): (line 33)
    # Processing the call keyword arguments (line 33)
    kwargs_323318 = {}
    # Getting the type of 'handle' (line 33)
    handle_323316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'handle', False)
    # Obtaining the member 'Close' of a type (line 33)
    Close_323317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 4), handle_323316, 'Close')
    # Calling Close(args, kwargs) (line 33)
    Close_call_result_323319 = invoke(stypy.reporting.localization.Localization(__file__, 33, 4), Close_323317, *[], **kwargs_323318)
    
    # Getting the type of 'TZKEYNAME' (line 34)
    TZKEYNAME_323320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'TZKEYNAME')
    # Assigning a type to the variable 'stypy_return_type' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type', TZKEYNAME_323320)
    
    # ################# End of '_settzkeyname(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_settzkeyname' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_323321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_323321)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_settzkeyname'
    return stypy_return_type_323321

# Assigning a type to the variable '_settzkeyname' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '_settzkeyname', _settzkeyname)

# Assigning a Call to a Name (line 37):

# Assigning a Call to a Name (line 37):

# Call to _settzkeyname(...): (line 37)
# Processing the call keyword arguments (line 37)
kwargs_323323 = {}
# Getting the type of '_settzkeyname' (line 37)
_settzkeyname_323322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), '_settzkeyname', False)
# Calling _settzkeyname(args, kwargs) (line 37)
_settzkeyname_call_result_323324 = invoke(stypy.reporting.localization.Localization(__file__, 37, 12), _settzkeyname_323322, *[], **kwargs_323323)

# Assigning a type to the variable 'TZKEYNAME' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'TZKEYNAME', _settzkeyname_call_result_323324)
# Declaration of the 'tzres' class

class tzres(object, ):
    str_323325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, (-1)), 'str', '\n    Class for accessing `tzres.dll`, which contains timezone name related\n    resources.\n\n    .. versionadded:: 2.5.0\n    ')
    
    # Assigning a Call to a Name (line 47):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_323326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 33), 'str', 'tzres.dll')
        defaults = [str_323326]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 49, 4, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzres.__init__', ['tzres_loc'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['tzres_loc'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Name (line 51):
        
        # Assigning a Call to a Name (line 51):
        
        # Call to WinDLL(...): (line 51)
        # Processing the call arguments (line 51)
        str_323329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 31), 'str', 'user32')
        # Processing the call keyword arguments (line 51)
        kwargs_323330 = {}
        # Getting the type of 'ctypes' (line 51)
        ctypes_323327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'ctypes', False)
        # Obtaining the member 'WinDLL' of a type (line 51)
        WinDLL_323328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 17), ctypes_323327, 'WinDLL')
        # Calling WinDLL(args, kwargs) (line 51)
        WinDLL_call_result_323331 = invoke(stypy.reporting.localization.Localization(__file__, 51, 17), WinDLL_323328, *[str_323329], **kwargs_323330)
        
        # Assigning a type to the variable 'user32' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'user32', WinDLL_call_result_323331)
        
        # Assigning a Tuple to a Attribute (line 54):
        
        # Assigning a Tuple to a Attribute (line 54):
        
        # Obtaining an instance of the builtin type 'tuple' (line 54)
        tuple_323332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 54)
        # Adding element type (line 54)
        # Getting the type of 'wintypes' (line 54)
        wintypes_323333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 39), 'wintypes')
        # Obtaining the member 'HINSTANCE' of a type (line 54)
        HINSTANCE_323334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 39), wintypes_323333, 'HINSTANCE')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 39), tuple_323332, HINSTANCE_323334)
        # Adding element type (line 54)
        # Getting the type of 'wintypes' (line 55)
        wintypes_323335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 39), 'wintypes')
        # Obtaining the member 'UINT' of a type (line 55)
        UINT_323336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 39), wintypes_323335, 'UINT')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 39), tuple_323332, UINT_323336)
        # Adding element type (line 54)
        # Getting the type of 'wintypes' (line 56)
        wintypes_323337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 39), 'wintypes')
        # Obtaining the member 'LPWSTR' of a type (line 56)
        LPWSTR_323338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 39), wintypes_323337, 'LPWSTR')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 39), tuple_323332, LPWSTR_323338)
        # Adding element type (line 54)
        # Getting the type of 'ctypes' (line 57)
        ctypes_323339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 39), 'ctypes')
        # Obtaining the member 'c_int' of a type (line 57)
        c_int_323340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 39), ctypes_323339, 'c_int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 39), tuple_323332, c_int_323340)
        
        # Getting the type of 'user32' (line 54)
        user32_323341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'user32')
        # Obtaining the member 'LoadStringW' of a type (line 54)
        LoadStringW_323342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), user32_323341, 'LoadStringW')
        # Setting the type of the member 'argtypes' of a type (line 54)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), LoadStringW_323342, 'argtypes', tuple_323332)
        
        # Assigning a Attribute to a Attribute (line 59):
        
        # Assigning a Attribute to a Attribute (line 59):
        # Getting the type of 'user32' (line 59)
        user32_323343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 27), 'user32')
        # Obtaining the member 'LoadStringW' of a type (line 59)
        LoadStringW_323344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 27), user32_323343, 'LoadStringW')
        # Getting the type of 'self' (line 59)
        self_323345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self')
        # Setting the type of the member 'LoadStringW' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_323345, 'LoadStringW', LoadStringW_323344)
        
        # Assigning a Call to a Attribute (line 60):
        
        # Assigning a Call to a Attribute (line 60):
        
        # Call to WinDLL(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'tzres_loc' (line 60)
        tzres_loc_323348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 36), 'tzres_loc', False)
        # Processing the call keyword arguments (line 60)
        kwargs_323349 = {}
        # Getting the type of 'ctypes' (line 60)
        ctypes_323346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 22), 'ctypes', False)
        # Obtaining the member 'WinDLL' of a type (line 60)
        WinDLL_323347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 22), ctypes_323346, 'WinDLL')
        # Calling WinDLL(args, kwargs) (line 60)
        WinDLL_call_result_323350 = invoke(stypy.reporting.localization.Localization(__file__, 60, 22), WinDLL_323347, *[tzres_loc_323348], **kwargs_323349)
        
        # Getting the type of 'self' (line 60)
        self_323351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self')
        # Setting the type of the member '_tzres' of a type (line 60)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), self_323351, '_tzres', WinDLL_call_result_323350)
        
        # Assigning a Name to a Attribute (line 61):
        
        # Assigning a Name to a Attribute (line 61):
        # Getting the type of 'tzres_loc' (line 61)
        tzres_loc_323352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'tzres_loc')
        # Getting the type of 'self' (line 61)
        self_323353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self')
        # Setting the type of the member 'tzres_loc' of a type (line 61)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_323353, 'tzres_loc', tzres_loc_323352)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def load_name(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'load_name'
        module_type_store = module_type_store.open_function_context('load_name', 63, 4, False)
        # Assigning a type to the variable 'self' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzres.load_name.__dict__.__setitem__('stypy_localization', localization)
        tzres.load_name.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzres.load_name.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzres.load_name.__dict__.__setitem__('stypy_function_name', 'tzres.load_name')
        tzres.load_name.__dict__.__setitem__('stypy_param_names_list', ['offset'])
        tzres.load_name.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzres.load_name.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzres.load_name.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzres.load_name.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzres.load_name.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzres.load_name.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzres.load_name', ['offset'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'load_name', localization, ['offset'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'load_name(...)' code ##################

        str_323354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, (-1)), 'str', "\n        Load a timezone name from a DLL offset (integer).\n\n        >>> from dateutil.tzwin import tzres\n        >>> tzr = tzres()\n        >>> print(tzr.load_name(112))\n        'Eastern Standard Time'\n\n        :param offset:\n            A positive integer value referring to a string from the tzres dll.\n\n        ..note:\n            Offsets found in the registry are generally of the form\n            `@tzres.dll,-114`. The offset in this case if 114, not -114.\n\n        ")
        
        # Assigning a Call to a Name (line 80):
        
        # Assigning a Call to a Name (line 80):
        
        # Call to p_wchar(...): (line 80)
        # Processing the call keyword arguments (line 80)
        kwargs_323357 = {}
        # Getting the type of 'self' (line 80)
        self_323355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 19), 'self', False)
        # Obtaining the member 'p_wchar' of a type (line 80)
        p_wchar_323356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 19), self_323355, 'p_wchar')
        # Calling p_wchar(args, kwargs) (line 80)
        p_wchar_call_result_323358 = invoke(stypy.reporting.localization.Localization(__file__, 80, 19), p_wchar_323356, *[], **kwargs_323357)
        
        # Assigning a type to the variable 'resource' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'resource', p_wchar_call_result_323358)
        
        # Assigning a Call to a Name (line 81):
        
        # Assigning a Call to a Name (line 81):
        
        # Call to cast(...): (line 81)
        # Processing the call arguments (line 81)
        
        # Call to byref(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'resource' (line 81)
        resource_323363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 44), 'resource', False)
        # Processing the call keyword arguments (line 81)
        kwargs_323364 = {}
        # Getting the type of 'ctypes' (line 81)
        ctypes_323361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 31), 'ctypes', False)
        # Obtaining the member 'byref' of a type (line 81)
        byref_323362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 31), ctypes_323361, 'byref')
        # Calling byref(args, kwargs) (line 81)
        byref_call_result_323365 = invoke(stypy.reporting.localization.Localization(__file__, 81, 31), byref_323362, *[resource_323363], **kwargs_323364)
        
        # Getting the type of 'wintypes' (line 81)
        wintypes_323366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 55), 'wintypes', False)
        # Obtaining the member 'LPWSTR' of a type (line 81)
        LPWSTR_323367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 55), wintypes_323366, 'LPWSTR')
        # Processing the call keyword arguments (line 81)
        kwargs_323368 = {}
        # Getting the type of 'ctypes' (line 81)
        ctypes_323359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 19), 'ctypes', False)
        # Obtaining the member 'cast' of a type (line 81)
        cast_323360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 19), ctypes_323359, 'cast')
        # Calling cast(args, kwargs) (line 81)
        cast_call_result_323369 = invoke(stypy.reporting.localization.Localization(__file__, 81, 19), cast_323360, *[byref_call_result_323365, LPWSTR_323367], **kwargs_323368)
        
        # Assigning a type to the variable 'lpBuffer' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'lpBuffer', cast_call_result_323369)
        
        # Assigning a Call to a Name (line 82):
        
        # Assigning a Call to a Name (line 82):
        
        # Call to LoadStringW(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'self' (line 82)
        self_323372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 33), 'self', False)
        # Obtaining the member '_tzres' of a type (line 82)
        _tzres_323373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 33), self_323372, '_tzres')
        # Obtaining the member '_handle' of a type (line 82)
        _handle_323374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 33), _tzres_323373, '_handle')
        # Getting the type of 'offset' (line 82)
        offset_323375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 54), 'offset', False)
        # Getting the type of 'lpBuffer' (line 82)
        lpBuffer_323376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 62), 'lpBuffer', False)
        int_323377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 72), 'int')
        # Processing the call keyword arguments (line 82)
        kwargs_323378 = {}
        # Getting the type of 'self' (line 82)
        self_323370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'self', False)
        # Obtaining the member 'LoadStringW' of a type (line 82)
        LoadStringW_323371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 16), self_323370, 'LoadStringW')
        # Calling LoadStringW(args, kwargs) (line 82)
        LoadStringW_call_result_323379 = invoke(stypy.reporting.localization.Localization(__file__, 82, 16), LoadStringW_323371, *[_handle_323374, offset_323375, lpBuffer_323376, int_323377], **kwargs_323378)
        
        # Assigning a type to the variable 'nchar' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'nchar', LoadStringW_call_result_323379)
        
        # Obtaining the type of the subscript
        # Getting the type of 'nchar' (line 83)
        nchar_323380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 25), 'nchar')
        slice_323381 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 83, 15), None, nchar_323380, None)
        # Getting the type of 'resource' (line 83)
        resource_323382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'resource')
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___323383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 15), resource_323382, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_323384 = invoke(stypy.reporting.localization.Localization(__file__, 83, 15), getitem___323383, slice_323381)
        
        # Assigning a type to the variable 'stypy_return_type' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'stypy_return_type', subscript_call_result_323384)
        
        # ################# End of 'load_name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'load_name' in the type store
        # Getting the type of 'stypy_return_type' (line 63)
        stypy_return_type_323385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_323385)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'load_name'
        return stypy_return_type_323385


    @norecursion
    def name_from_string(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'name_from_string'
        module_type_store = module_type_store.open_function_context('name_from_string', 85, 4, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzres.name_from_string.__dict__.__setitem__('stypy_localization', localization)
        tzres.name_from_string.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzres.name_from_string.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzres.name_from_string.__dict__.__setitem__('stypy_function_name', 'tzres.name_from_string')
        tzres.name_from_string.__dict__.__setitem__('stypy_param_names_list', ['tzname_str'])
        tzres.name_from_string.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzres.name_from_string.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzres.name_from_string.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzres.name_from_string.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzres.name_from_string.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzres.name_from_string.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzres.name_from_string', ['tzname_str'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'name_from_string', localization, ['tzname_str'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'name_from_string(...)' code ##################

        str_323386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, (-1)), 'str', "\n        Parse strings as returned from the Windows registry into the time zone\n        name as defined in the registry.\n\n        >>> from dateutil.tzwin import tzres\n        >>> tzr = tzres()\n        >>> print(tzr.name_from_string('@tzres.dll,-251'))\n        'Dateline Daylight Time'\n        >>> print(tzr.name_from_string('Eastern Standard Time'))\n        'Eastern Standard Time'\n\n        :param tzname_str:\n            A timezone name string as returned from a Windows registry key.\n\n        :return:\n            Returns the localized timezone string from tzres.dll if the string\n            is of the form `@tzres.dll,-offset`, else returns the input string.\n        ")
        
        
        
        # Call to startswith(...): (line 104)
        # Processing the call arguments (line 104)
        str_323389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 37), 'str', '@')
        # Processing the call keyword arguments (line 104)
        kwargs_323390 = {}
        # Getting the type of 'tzname_str' (line 104)
        tzname_str_323387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'tzname_str', False)
        # Obtaining the member 'startswith' of a type (line 104)
        startswith_323388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 15), tzname_str_323387, 'startswith')
        # Calling startswith(args, kwargs) (line 104)
        startswith_call_result_323391 = invoke(stypy.reporting.localization.Localization(__file__, 104, 15), startswith_323388, *[str_323389], **kwargs_323390)
        
        # Applying the 'not' unary operator (line 104)
        result_not__323392 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 11), 'not', startswith_call_result_323391)
        
        # Testing the type of an if condition (line 104)
        if_condition_323393 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 8), result_not__323392)
        # Assigning a type to the variable 'if_condition_323393' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'if_condition_323393', if_condition_323393)
        # SSA begins for if statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'tzname_str' (line 105)
        tzname_str_323394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'tzname_str')
        # Assigning a type to the variable 'stypy_return_type' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'stypy_return_type', tzname_str_323394)
        # SSA join for if statement (line 104)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 107):
        
        # Assigning a Call to a Name (line 107):
        
        # Call to split(...): (line 107)
        # Processing the call arguments (line 107)
        str_323397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 37), 'str', ',-')
        # Processing the call keyword arguments (line 107)
        kwargs_323398 = {}
        # Getting the type of 'tzname_str' (line 107)
        tzname_str_323395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 20), 'tzname_str', False)
        # Obtaining the member 'split' of a type (line 107)
        split_323396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 20), tzname_str_323395, 'split')
        # Calling split(args, kwargs) (line 107)
        split_call_result_323399 = invoke(stypy.reporting.localization.Localization(__file__, 107, 20), split_323396, *[str_323397], **kwargs_323398)
        
        # Assigning a type to the variable 'name_splt' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'name_splt', split_call_result_323399)
        
        
        # SSA begins for try-except statement (line 108)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 109):
        
        # Assigning a Call to a Name (line 109):
        
        # Call to int(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Obtaining the type of the subscript
        int_323401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 35), 'int')
        # Getting the type of 'name_splt' (line 109)
        name_splt_323402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 25), 'name_splt', False)
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___323403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 25), name_splt_323402, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_323404 = invoke(stypy.reporting.localization.Localization(__file__, 109, 25), getitem___323403, int_323401)
        
        # Processing the call keyword arguments (line 109)
        kwargs_323405 = {}
        # Getting the type of 'int' (line 109)
        int_323400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'int', False)
        # Calling int(args, kwargs) (line 109)
        int_call_result_323406 = invoke(stypy.reporting.localization.Localization(__file__, 109, 21), int_323400, *[subscript_call_result_323404], **kwargs_323405)
        
        # Assigning a type to the variable 'offset' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'offset', int_call_result_323406)
        # SSA branch for the except part of a try statement (line 108)
        # SSA branch for the except '<any exception>' branch of a try statement (line 108)
        module_type_store.open_ssa_branch('except')
        
        # Call to ValueError(...): (line 111)
        # Processing the call arguments (line 111)
        str_323408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 29), 'str', 'Malformed timezone string.')
        # Processing the call keyword arguments (line 111)
        kwargs_323409 = {}
        # Getting the type of 'ValueError' (line 111)
        ValueError_323407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 111)
        ValueError_call_result_323410 = invoke(stypy.reporting.localization.Localization(__file__, 111, 18), ValueError_323407, *[str_323408], **kwargs_323409)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 111, 12), ValueError_call_result_323410, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 108)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to load_name(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'offset' (line 113)
        offset_323413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 30), 'offset', False)
        # Processing the call keyword arguments (line 113)
        kwargs_323414 = {}
        # Getting the type of 'self' (line 113)
        self_323411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'self', False)
        # Obtaining the member 'load_name' of a type (line 113)
        load_name_323412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 15), self_323411, 'load_name')
        # Calling load_name(args, kwargs) (line 113)
        load_name_call_result_323415 = invoke(stypy.reporting.localization.Localization(__file__, 113, 15), load_name_323412, *[offset_323413], **kwargs_323414)
        
        # Assigning a type to the variable 'stypy_return_type' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'stypy_return_type', load_name_call_result_323415)
        
        # ################# End of 'name_from_string(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'name_from_string' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_323416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_323416)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'name_from_string'
        return stypy_return_type_323416


# Assigning a type to the variable 'tzres' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'tzres', tzres)

# Assigning a Call to a Name (line 47):

# Call to POINTER(...): (line 47)
# Processing the call arguments (line 47)
# Getting the type of 'wintypes' (line 47)
wintypes_323419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 29), 'wintypes', False)
# Obtaining the member 'WCHAR' of a type (line 47)
WCHAR_323420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 29), wintypes_323419, 'WCHAR')
# Processing the call keyword arguments (line 47)
kwargs_323421 = {}
# Getting the type of 'ctypes' (line 47)
ctypes_323417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 14), 'ctypes', False)
# Obtaining the member 'POINTER' of a type (line 47)
POINTER_323418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 14), ctypes_323417, 'POINTER')
# Calling POINTER(args, kwargs) (line 47)
POINTER_call_result_323422 = invoke(stypy.reporting.localization.Localization(__file__, 47, 14), POINTER_323418, *[WCHAR_323420], **kwargs_323421)

# Getting the type of 'tzres'
tzres_323423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'tzres')
# Setting the type of the member 'p_wchar' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), tzres_323423, 'p_wchar', POINTER_call_result_323422)
# Declaration of the 'tzwinbase' class
# Getting the type of 'tzrangebase' (line 116)
tzrangebase_323424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'tzrangebase')

class tzwinbase(tzrangebase_323424, ):
    str_323425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 4), 'str', "tzinfo class based on win32's timezones available in the registry.")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 118, 4, False)
        # Assigning a type to the variable 'self' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzwinbase.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to NotImplementedError(...): (line 119)
        # Processing the call arguments (line 119)
        str_323427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 34), 'str', 'tzwinbase is an abstract base class')
        # Processing the call keyword arguments (line 119)
        kwargs_323428 = {}
        # Getting the type of 'NotImplementedError' (line 119)
        NotImplementedError_323426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 14), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 119)
        NotImplementedError_call_result_323429 = invoke(stypy.reporting.localization.Localization(__file__, 119, 14), NotImplementedError_323426, *[str_323427], **kwargs_323428)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 119, 8), NotImplementedError_call_result_323429, 'raise parameter', BaseException)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 121, 4, False)
        # Assigning a type to the variable 'self' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzwinbase.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        tzwinbase.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzwinbase.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzwinbase.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'tzwinbase.stypy__eq__')
        tzwinbase.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        tzwinbase.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzwinbase.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzwinbase.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzwinbase.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzwinbase.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzwinbase.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzwinbase.stypy__eq__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        
        
        # Call to isinstance(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'other' (line 123)
        other_323431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 26), 'other', False)
        # Getting the type of 'tzwinbase' (line 123)
        tzwinbase_323432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 33), 'tzwinbase', False)
        # Processing the call keyword arguments (line 123)
        kwargs_323433 = {}
        # Getting the type of 'isinstance' (line 123)
        isinstance_323430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 123)
        isinstance_call_result_323434 = invoke(stypy.reporting.localization.Localization(__file__, 123, 15), isinstance_323430, *[other_323431, tzwinbase_323432], **kwargs_323433)
        
        # Applying the 'not' unary operator (line 123)
        result_not__323435 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 11), 'not', isinstance_call_result_323434)
        
        # Testing the type of an if condition (line 123)
        if_condition_323436 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 8), result_not__323435)
        # Assigning a type to the variable 'if_condition_323436' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'if_condition_323436', if_condition_323436)
        # SSA begins for if statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'NotImplemented' (line 124)
        NotImplemented_323437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'stypy_return_type', NotImplemented_323437)
        # SSA join for if statement (line 123)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 126)
        self_323438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 17), 'self')
        # Obtaining the member '_std_offset' of a type (line 126)
        _std_offset_323439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 17), self_323438, '_std_offset')
        # Getting the type of 'other' (line 126)
        other_323440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 37), 'other')
        # Obtaining the member '_std_offset' of a type (line 126)
        _std_offset_323441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 37), other_323440, '_std_offset')
        # Applying the binary operator '==' (line 126)
        result_eq_323442 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 17), '==', _std_offset_323439, _std_offset_323441)
        
        
        # Getting the type of 'self' (line 127)
        self_323443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 17), 'self')
        # Obtaining the member '_dst_offset' of a type (line 127)
        _dst_offset_323444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 17), self_323443, '_dst_offset')
        # Getting the type of 'other' (line 127)
        other_323445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 37), 'other')
        # Obtaining the member '_dst_offset' of a type (line 127)
        _dst_offset_323446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 37), other_323445, '_dst_offset')
        # Applying the binary operator '==' (line 127)
        result_eq_323447 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 17), '==', _dst_offset_323444, _dst_offset_323446)
        
        # Applying the binary operator 'and' (line 126)
        result_and_keyword_323448 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 17), 'and', result_eq_323442, result_eq_323447)
        
        # Getting the type of 'self' (line 128)
        self_323449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 17), 'self')
        # Obtaining the member '_stddayofweek' of a type (line 128)
        _stddayofweek_323450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 17), self_323449, '_stddayofweek')
        # Getting the type of 'other' (line 128)
        other_323451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 39), 'other')
        # Obtaining the member '_stddayofweek' of a type (line 128)
        _stddayofweek_323452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 39), other_323451, '_stddayofweek')
        # Applying the binary operator '==' (line 128)
        result_eq_323453 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 17), '==', _stddayofweek_323450, _stddayofweek_323452)
        
        # Applying the binary operator 'and' (line 126)
        result_and_keyword_323454 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 17), 'and', result_and_keyword_323448, result_eq_323453)
        
        # Getting the type of 'self' (line 129)
        self_323455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 17), 'self')
        # Obtaining the member '_dstdayofweek' of a type (line 129)
        _dstdayofweek_323456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 17), self_323455, '_dstdayofweek')
        # Getting the type of 'other' (line 129)
        other_323457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 39), 'other')
        # Obtaining the member '_dstdayofweek' of a type (line 129)
        _dstdayofweek_323458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 39), other_323457, '_dstdayofweek')
        # Applying the binary operator '==' (line 129)
        result_eq_323459 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 17), '==', _dstdayofweek_323456, _dstdayofweek_323458)
        
        # Applying the binary operator 'and' (line 126)
        result_and_keyword_323460 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 17), 'and', result_and_keyword_323454, result_eq_323459)
        
        # Getting the type of 'self' (line 130)
        self_323461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 17), 'self')
        # Obtaining the member '_stdweeknumber' of a type (line 130)
        _stdweeknumber_323462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 17), self_323461, '_stdweeknumber')
        # Getting the type of 'other' (line 130)
        other_323463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 40), 'other')
        # Obtaining the member '_stdweeknumber' of a type (line 130)
        _stdweeknumber_323464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 40), other_323463, '_stdweeknumber')
        # Applying the binary operator '==' (line 130)
        result_eq_323465 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 17), '==', _stdweeknumber_323462, _stdweeknumber_323464)
        
        # Applying the binary operator 'and' (line 126)
        result_and_keyword_323466 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 17), 'and', result_and_keyword_323460, result_eq_323465)
        
        # Getting the type of 'self' (line 131)
        self_323467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 17), 'self')
        # Obtaining the member '_dstweeknumber' of a type (line 131)
        _dstweeknumber_323468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 17), self_323467, '_dstweeknumber')
        # Getting the type of 'other' (line 131)
        other_323469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 40), 'other')
        # Obtaining the member '_dstweeknumber' of a type (line 131)
        _dstweeknumber_323470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 40), other_323469, '_dstweeknumber')
        # Applying the binary operator '==' (line 131)
        result_eq_323471 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 17), '==', _dstweeknumber_323468, _dstweeknumber_323470)
        
        # Applying the binary operator 'and' (line 126)
        result_and_keyword_323472 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 17), 'and', result_and_keyword_323466, result_eq_323471)
        
        # Getting the type of 'self' (line 132)
        self_323473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 17), 'self')
        # Obtaining the member '_stdhour' of a type (line 132)
        _stdhour_323474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 17), self_323473, '_stdhour')
        # Getting the type of 'other' (line 132)
        other_323475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 34), 'other')
        # Obtaining the member '_stdhour' of a type (line 132)
        _stdhour_323476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 34), other_323475, '_stdhour')
        # Applying the binary operator '==' (line 132)
        result_eq_323477 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 17), '==', _stdhour_323474, _stdhour_323476)
        
        # Applying the binary operator 'and' (line 126)
        result_and_keyword_323478 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 17), 'and', result_and_keyword_323472, result_eq_323477)
        
        # Getting the type of 'self' (line 133)
        self_323479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 17), 'self')
        # Obtaining the member '_dsthour' of a type (line 133)
        _dsthour_323480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 17), self_323479, '_dsthour')
        # Getting the type of 'other' (line 133)
        other_323481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 34), 'other')
        # Obtaining the member '_dsthour' of a type (line 133)
        _dsthour_323482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 34), other_323481, '_dsthour')
        # Applying the binary operator '==' (line 133)
        result_eq_323483 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 17), '==', _dsthour_323480, _dsthour_323482)
        
        # Applying the binary operator 'and' (line 126)
        result_and_keyword_323484 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 17), 'and', result_and_keyword_323478, result_eq_323483)
        
        # Getting the type of 'self' (line 134)
        self_323485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 17), 'self')
        # Obtaining the member '_stdminute' of a type (line 134)
        _stdminute_323486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 17), self_323485, '_stdminute')
        # Getting the type of 'other' (line 134)
        other_323487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 36), 'other')
        # Obtaining the member '_stdminute' of a type (line 134)
        _stdminute_323488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 36), other_323487, '_stdminute')
        # Applying the binary operator '==' (line 134)
        result_eq_323489 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 17), '==', _stdminute_323486, _stdminute_323488)
        
        # Applying the binary operator 'and' (line 126)
        result_and_keyword_323490 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 17), 'and', result_and_keyword_323484, result_eq_323489)
        
        # Getting the type of 'self' (line 135)
        self_323491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 17), 'self')
        # Obtaining the member '_dstminute' of a type (line 135)
        _dstminute_323492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 17), self_323491, '_dstminute')
        # Getting the type of 'other' (line 135)
        other_323493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 36), 'other')
        # Obtaining the member '_dstminute' of a type (line 135)
        _dstminute_323494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 36), other_323493, '_dstminute')
        # Applying the binary operator '==' (line 135)
        result_eq_323495 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 17), '==', _dstminute_323492, _dstminute_323494)
        
        # Applying the binary operator 'and' (line 126)
        result_and_keyword_323496 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 17), 'and', result_and_keyword_323490, result_eq_323495)
        
        # Getting the type of 'self' (line 136)
        self_323497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 17), 'self')
        # Obtaining the member '_std_abbr' of a type (line 136)
        _std_abbr_323498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 17), self_323497, '_std_abbr')
        # Getting the type of 'other' (line 136)
        other_323499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 35), 'other')
        # Obtaining the member '_std_abbr' of a type (line 136)
        _std_abbr_323500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 35), other_323499, '_std_abbr')
        # Applying the binary operator '==' (line 136)
        result_eq_323501 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 17), '==', _std_abbr_323498, _std_abbr_323500)
        
        # Applying the binary operator 'and' (line 126)
        result_and_keyword_323502 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 17), 'and', result_and_keyword_323496, result_eq_323501)
        
        # Getting the type of 'self' (line 137)
        self_323503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 17), 'self')
        # Obtaining the member '_dst_abbr' of a type (line 137)
        _dst_abbr_323504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 17), self_323503, '_dst_abbr')
        # Getting the type of 'other' (line 137)
        other_323505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 35), 'other')
        # Obtaining the member '_dst_abbr' of a type (line 137)
        _dst_abbr_323506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 35), other_323505, '_dst_abbr')
        # Applying the binary operator '==' (line 137)
        result_eq_323507 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 17), '==', _dst_abbr_323504, _dst_abbr_323506)
        
        # Applying the binary operator 'and' (line 126)
        result_and_keyword_323508 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 17), 'and', result_and_keyword_323502, result_eq_323507)
        
        # Assigning a type to the variable 'stypy_return_type' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'stypy_return_type', result_and_keyword_323508)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 121)
        stypy_return_type_323509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_323509)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_323509


    @staticmethod
    @norecursion
    def list(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'list'
        module_type_store = module_type_store.open_function_context('list', 139, 4, False)
        
        # Passed parameters checking function
        tzwinbase.list.__dict__.__setitem__('stypy_localization', localization)
        tzwinbase.list.__dict__.__setitem__('stypy_type_of_self', None)
        tzwinbase.list.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzwinbase.list.__dict__.__setitem__('stypy_function_name', 'list')
        tzwinbase.list.__dict__.__setitem__('stypy_param_names_list', [])
        tzwinbase.list.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzwinbase.list.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzwinbase.list.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzwinbase.list.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzwinbase.list.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzwinbase.list.__dict__.__setitem__('stypy_declared_arg_number', 0)
        arguments = process_argument_values(localization, None, module_type_store, 'list', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'list', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'list(...)' code ##################

        str_323510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 8), 'str', 'Return a list of all time zones known to the system.')
        
        # Call to ConnectRegistry(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'None' (line 142)
        None_323513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 36), 'None', False)
        # Getting the type of 'winreg' (line 142)
        winreg_323514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 42), 'winreg', False)
        # Obtaining the member 'HKEY_LOCAL_MACHINE' of a type (line 142)
        HKEY_LOCAL_MACHINE_323515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 42), winreg_323514, 'HKEY_LOCAL_MACHINE')
        # Processing the call keyword arguments (line 142)
        kwargs_323516 = {}
        # Getting the type of 'winreg' (line 142)
        winreg_323511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 13), 'winreg', False)
        # Obtaining the member 'ConnectRegistry' of a type (line 142)
        ConnectRegistry_323512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 13), winreg_323511, 'ConnectRegistry')
        # Calling ConnectRegistry(args, kwargs) (line 142)
        ConnectRegistry_call_result_323517 = invoke(stypy.reporting.localization.Localization(__file__, 142, 13), ConnectRegistry_323512, *[None_323513, HKEY_LOCAL_MACHINE_323515], **kwargs_323516)
        
        with_323518 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 142, 13), ConnectRegistry_call_result_323517, 'with parameter', '__enter__', '__exit__')

        if with_323518:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 142)
            enter___323519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 13), ConnectRegistry_call_result_323517, '__enter__')
            with_enter_323520 = invoke(stypy.reporting.localization.Localization(__file__, 142, 13), enter___323519)
            # Assigning a type to the variable 'handle' (line 142)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 13), 'handle', with_enter_323520)
            
            # Call to OpenKey(...): (line 143)
            # Processing the call arguments (line 143)
            # Getting the type of 'handle' (line 143)
            handle_323523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 32), 'handle', False)
            # Getting the type of 'TZKEYNAME' (line 143)
            TZKEYNAME_323524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 40), 'TZKEYNAME', False)
            # Processing the call keyword arguments (line 143)
            kwargs_323525 = {}
            # Getting the type of 'winreg' (line 143)
            winreg_323521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 17), 'winreg', False)
            # Obtaining the member 'OpenKey' of a type (line 143)
            OpenKey_323522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 17), winreg_323521, 'OpenKey')
            # Calling OpenKey(args, kwargs) (line 143)
            OpenKey_call_result_323526 = invoke(stypy.reporting.localization.Localization(__file__, 143, 17), OpenKey_323522, *[handle_323523, TZKEYNAME_323524], **kwargs_323525)
            
            with_323527 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 143, 17), OpenKey_call_result_323526, 'with parameter', '__enter__', '__exit__')

            if with_323527:
                # Calling the __enter__ method to initiate a with section
                # Obtaining the member '__enter__' of a type (line 143)
                enter___323528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 17), OpenKey_call_result_323526, '__enter__')
                with_enter_323529 = invoke(stypy.reporting.localization.Localization(__file__, 143, 17), enter___323528)
                # Assigning a type to the variable 'tzkey' (line 143)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 17), 'tzkey', with_enter_323529)
                
                # Assigning a ListComp to a Name (line 144):
                
                # Assigning a ListComp to a Name (line 144):
                # Calculating list comprehension
                # Calculating comprehension expression
                
                # Call to range(...): (line 145)
                # Processing the call arguments (line 145)
                
                # Obtaining the type of the subscript
                int_323537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 68), 'int')
                
                # Call to QueryInfoKey(...): (line 145)
                # Processing the call arguments (line 145)
                # Getting the type of 'tzkey' (line 145)
                tzkey_323540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 61), 'tzkey', False)
                # Processing the call keyword arguments (line 145)
                kwargs_323541 = {}
                # Getting the type of 'winreg' (line 145)
                winreg_323538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 41), 'winreg', False)
                # Obtaining the member 'QueryInfoKey' of a type (line 145)
                QueryInfoKey_323539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 41), winreg_323538, 'QueryInfoKey')
                # Calling QueryInfoKey(args, kwargs) (line 145)
                QueryInfoKey_call_result_323542 = invoke(stypy.reporting.localization.Localization(__file__, 145, 41), QueryInfoKey_323539, *[tzkey_323540], **kwargs_323541)
                
                # Obtaining the member '__getitem__' of a type (line 145)
                getitem___323543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 41), QueryInfoKey_call_result_323542, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 145)
                subscript_call_result_323544 = invoke(stypy.reporting.localization.Localization(__file__, 145, 41), getitem___323543, int_323537)
                
                # Processing the call keyword arguments (line 145)
                kwargs_323545 = {}
                # Getting the type of 'range' (line 145)
                range_323536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 35), 'range', False)
                # Calling range(args, kwargs) (line 145)
                range_call_result_323546 = invoke(stypy.reporting.localization.Localization(__file__, 145, 35), range_323536, *[subscript_call_result_323544], **kwargs_323545)
                
                comprehension_323547 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 26), range_call_result_323546)
                # Assigning a type to the variable 'i' (line 144)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 26), 'i', comprehension_323547)
                
                # Call to EnumKey(...): (line 144)
                # Processing the call arguments (line 144)
                # Getting the type of 'tzkey' (line 144)
                tzkey_323532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 41), 'tzkey', False)
                # Getting the type of 'i' (line 144)
                i_323533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 48), 'i', False)
                # Processing the call keyword arguments (line 144)
                kwargs_323534 = {}
                # Getting the type of 'winreg' (line 144)
                winreg_323530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 26), 'winreg', False)
                # Obtaining the member 'EnumKey' of a type (line 144)
                EnumKey_323531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 26), winreg_323530, 'EnumKey')
                # Calling EnumKey(args, kwargs) (line 144)
                EnumKey_call_result_323535 = invoke(stypy.reporting.localization.Localization(__file__, 144, 26), EnumKey_323531, *[tzkey_323532, i_323533], **kwargs_323534)
                
                list_323548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 26), 'list')
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 26), list_323548, EnumKey_call_result_323535)
                # Assigning a type to the variable 'result' (line 144)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'result', list_323548)
                # Calling the __exit__ method to finish a with section
                # Obtaining the member '__exit__' of a type (line 143)
                exit___323549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 17), OpenKey_call_result_323526, '__exit__')
                with_exit_323550 = invoke(stypy.reporting.localization.Localization(__file__, 143, 17), exit___323549, None, None, None)

            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 142)
            exit___323551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 13), ConnectRegistry_call_result_323517, '__exit__')
            with_exit_323552 = invoke(stypy.reporting.localization.Localization(__file__, 142, 13), exit___323551, None, None, None)

        # Getting the type of 'result' (line 146)
        result_323553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'stypy_return_type', result_323553)
        
        # ################# End of 'list(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'list' in the type store
        # Getting the type of 'stypy_return_type' (line 139)
        stypy_return_type_323554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_323554)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'list'
        return stypy_return_type_323554


    @norecursion
    def display(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'display'
        module_type_store = module_type_store.open_function_context('display', 148, 4, False)
        # Assigning a type to the variable 'self' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzwinbase.display.__dict__.__setitem__('stypy_localization', localization)
        tzwinbase.display.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzwinbase.display.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzwinbase.display.__dict__.__setitem__('stypy_function_name', 'tzwinbase.display')
        tzwinbase.display.__dict__.__setitem__('stypy_param_names_list', [])
        tzwinbase.display.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzwinbase.display.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzwinbase.display.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzwinbase.display.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzwinbase.display.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzwinbase.display.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzwinbase.display', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'display', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'display(...)' code ##################

        # Getting the type of 'self' (line 149)
        self_323555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'self')
        # Obtaining the member '_display' of a type (line 149)
        _display_323556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 15), self_323555, '_display')
        # Assigning a type to the variable 'stypy_return_type' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'stypy_return_type', _display_323556)
        
        # ################# End of 'display(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'display' in the type store
        # Getting the type of 'stypy_return_type' (line 148)
        stypy_return_type_323557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_323557)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'display'
        return stypy_return_type_323557


    @norecursion
    def transitions(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'transitions'
        module_type_store = module_type_store.open_function_context('transitions', 151, 4, False)
        # Assigning a type to the variable 'self' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzwinbase.transitions.__dict__.__setitem__('stypy_localization', localization)
        tzwinbase.transitions.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzwinbase.transitions.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzwinbase.transitions.__dict__.__setitem__('stypy_function_name', 'tzwinbase.transitions')
        tzwinbase.transitions.__dict__.__setitem__('stypy_param_names_list', ['year'])
        tzwinbase.transitions.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzwinbase.transitions.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzwinbase.transitions.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzwinbase.transitions.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzwinbase.transitions.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzwinbase.transitions.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzwinbase.transitions', ['year'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'transitions', localization, ['year'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'transitions(...)' code ##################

        str_323558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, (-1)), 'str', '\n        For a given year, get the DST on and off transition times, expressed\n        always on the standard time side. For zones with no transitions, this\n        function returns ``None``.\n\n        :param year:\n            The year whose transitions you would like to query.\n\n        :return:\n            Returns a :class:`tuple` of :class:`datetime.datetime` objects,\n            ``(dston, dstoff)`` for zones with an annual DST transition, or\n            ``None`` for fixed offset zones.\n        ')
        
        
        # Getting the type of 'self' (line 166)
        self_323559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 15), 'self')
        # Obtaining the member 'hasdst' of a type (line 166)
        hasdst_323560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 15), self_323559, 'hasdst')
        # Applying the 'not' unary operator (line 166)
        result_not__323561 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 11), 'not', hasdst_323560)
        
        # Testing the type of an if condition (line 166)
        if_condition_323562 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 166, 8), result_not__323561)
        # Assigning a type to the variable 'if_condition_323562' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'if_condition_323562', if_condition_323562)
        # SSA begins for if statement (line 166)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 167)
        None_323563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'stypy_return_type', None_323563)
        # SSA join for if statement (line 166)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 169):
        
        # Assigning a Call to a Name (line 169):
        
        # Call to picknthweekday(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'year' (line 169)
        year_323565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 31), 'year', False)
        # Getting the type of 'self' (line 169)
        self_323566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 37), 'self', False)
        # Obtaining the member '_dstmonth' of a type (line 169)
        _dstmonth_323567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 37), self_323566, '_dstmonth')
        # Getting the type of 'self' (line 169)
        self_323568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 53), 'self', False)
        # Obtaining the member '_dstdayofweek' of a type (line 169)
        _dstdayofweek_323569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 53), self_323568, '_dstdayofweek')
        # Getting the type of 'self' (line 170)
        self_323570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 31), 'self', False)
        # Obtaining the member '_dsthour' of a type (line 170)
        _dsthour_323571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 31), self_323570, '_dsthour')
        # Getting the type of 'self' (line 170)
        self_323572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 46), 'self', False)
        # Obtaining the member '_dstminute' of a type (line 170)
        _dstminute_323573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 46), self_323572, '_dstminute')
        # Getting the type of 'self' (line 171)
        self_323574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 31), 'self', False)
        # Obtaining the member '_dstweeknumber' of a type (line 171)
        _dstweeknumber_323575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 31), self_323574, '_dstweeknumber')
        # Processing the call keyword arguments (line 169)
        kwargs_323576 = {}
        # Getting the type of 'picknthweekday' (line 169)
        picknthweekday_323564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'picknthweekday', False)
        # Calling picknthweekday(args, kwargs) (line 169)
        picknthweekday_call_result_323577 = invoke(stypy.reporting.localization.Localization(__file__, 169, 16), picknthweekday_323564, *[year_323565, _dstmonth_323567, _dstdayofweek_323569, _dsthour_323571, _dstminute_323573, _dstweeknumber_323575], **kwargs_323576)
        
        # Assigning a type to the variable 'dston' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'dston', picknthweekday_call_result_323577)
        
        # Assigning a Call to a Name (line 173):
        
        # Assigning a Call to a Name (line 173):
        
        # Call to picknthweekday(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'year' (line 173)
        year_323579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 32), 'year', False)
        # Getting the type of 'self' (line 173)
        self_323580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 38), 'self', False)
        # Obtaining the member '_stdmonth' of a type (line 173)
        _stdmonth_323581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 38), self_323580, '_stdmonth')
        # Getting the type of 'self' (line 173)
        self_323582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 54), 'self', False)
        # Obtaining the member '_stddayofweek' of a type (line 173)
        _stddayofweek_323583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 54), self_323582, '_stddayofweek')
        # Getting the type of 'self' (line 174)
        self_323584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 32), 'self', False)
        # Obtaining the member '_stdhour' of a type (line 174)
        _stdhour_323585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 32), self_323584, '_stdhour')
        # Getting the type of 'self' (line 174)
        self_323586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 47), 'self', False)
        # Obtaining the member '_stdminute' of a type (line 174)
        _stdminute_323587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 47), self_323586, '_stdminute')
        # Getting the type of 'self' (line 175)
        self_323588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 32), 'self', False)
        # Obtaining the member '_stdweeknumber' of a type (line 175)
        _stdweeknumber_323589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 32), self_323588, '_stdweeknumber')
        # Processing the call keyword arguments (line 173)
        kwargs_323590 = {}
        # Getting the type of 'picknthweekday' (line 173)
        picknthweekday_323578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 17), 'picknthweekday', False)
        # Calling picknthweekday(args, kwargs) (line 173)
        picknthweekday_call_result_323591 = invoke(stypy.reporting.localization.Localization(__file__, 173, 17), picknthweekday_323578, *[year_323579, _stdmonth_323581, _stddayofweek_323583, _stdhour_323585, _stdminute_323587, _stdweeknumber_323589], **kwargs_323590)
        
        # Assigning a type to the variable 'dstoff' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'dstoff', picknthweekday_call_result_323591)
        
        # Getting the type of 'dstoff' (line 178)
        dstoff_323592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'dstoff')
        # Getting the type of 'self' (line 178)
        self_323593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 18), 'self')
        # Obtaining the member '_dst_base_offset' of a type (line 178)
        _dst_base_offset_323594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 18), self_323593, '_dst_base_offset')
        # Applying the binary operator '-=' (line 178)
        result_isub_323595 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 8), '-=', dstoff_323592, _dst_base_offset_323594)
        # Assigning a type to the variable 'dstoff' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'dstoff', result_isub_323595)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 180)
        tuple_323596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 180)
        # Adding element type (line 180)
        # Getting the type of 'dston' (line 180)
        dston_323597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 15), 'dston')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 15), tuple_323596, dston_323597)
        # Adding element type (line 180)
        # Getting the type of 'dstoff' (line 180)
        dstoff_323598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 22), 'dstoff')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 15), tuple_323596, dstoff_323598)
        
        # Assigning a type to the variable 'stypy_return_type' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'stypy_return_type', tuple_323596)
        
        # ################# End of 'transitions(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'transitions' in the type store
        # Getting the type of 'stypy_return_type' (line 151)
        stypy_return_type_323599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_323599)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'transitions'
        return stypy_return_type_323599


    @norecursion
    def _get_hasdst(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_hasdst'
        module_type_store = module_type_store.open_function_context('_get_hasdst', 182, 4, False)
        # Assigning a type to the variable 'self' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzwinbase._get_hasdst.__dict__.__setitem__('stypy_localization', localization)
        tzwinbase._get_hasdst.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzwinbase._get_hasdst.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzwinbase._get_hasdst.__dict__.__setitem__('stypy_function_name', 'tzwinbase._get_hasdst')
        tzwinbase._get_hasdst.__dict__.__setitem__('stypy_param_names_list', [])
        tzwinbase._get_hasdst.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzwinbase._get_hasdst.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzwinbase._get_hasdst.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzwinbase._get_hasdst.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzwinbase._get_hasdst.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzwinbase._get_hasdst.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzwinbase._get_hasdst', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_hasdst', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_hasdst(...)' code ##################

        
        # Getting the type of 'self' (line 183)
        self_323600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 15), 'self')
        # Obtaining the member '_dstmonth' of a type (line 183)
        _dstmonth_323601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 15), self_323600, '_dstmonth')
        int_323602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 33), 'int')
        # Applying the binary operator '!=' (line 183)
        result_ne_323603 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 15), '!=', _dstmonth_323601, int_323602)
        
        # Assigning a type to the variable 'stypy_return_type' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'stypy_return_type', result_ne_323603)
        
        # ################# End of '_get_hasdst(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_hasdst' in the type store
        # Getting the type of 'stypy_return_type' (line 182)
        stypy_return_type_323604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_323604)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_hasdst'
        return stypy_return_type_323604


    @norecursion
    def _dst_base_offset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_dst_base_offset'
        module_type_store = module_type_store.open_function_context('_dst_base_offset', 185, 4, False)
        # Assigning a type to the variable 'self' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzwinbase._dst_base_offset.__dict__.__setitem__('stypy_localization', localization)
        tzwinbase._dst_base_offset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzwinbase._dst_base_offset.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzwinbase._dst_base_offset.__dict__.__setitem__('stypy_function_name', 'tzwinbase._dst_base_offset')
        tzwinbase._dst_base_offset.__dict__.__setitem__('stypy_param_names_list', [])
        tzwinbase._dst_base_offset.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzwinbase._dst_base_offset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzwinbase._dst_base_offset.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzwinbase._dst_base_offset.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzwinbase._dst_base_offset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzwinbase._dst_base_offset.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzwinbase._dst_base_offset', [], None, None, defaults, varargs, kwargs)

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

        # Getting the type of 'self' (line 187)
        self_323605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 15), 'self')
        # Obtaining the member '_dst_base_offset_' of a type (line 187)
        _dst_base_offset__323606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 15), self_323605, '_dst_base_offset_')
        # Assigning a type to the variable 'stypy_return_type' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'stypy_return_type', _dst_base_offset__323606)
        
        # ################# End of '_dst_base_offset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_dst_base_offset' in the type store
        # Getting the type of 'stypy_return_type' (line 185)
        stypy_return_type_323607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_323607)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_dst_base_offset'
        return stypy_return_type_323607


# Assigning a type to the variable 'tzwinbase' (line 116)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'tzwinbase', tzwinbase)
# Declaration of the 'tzwin' class
# Getting the type of 'tzwinbase' (line 190)
tzwinbase_323608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'tzwinbase')

class tzwin(tzwinbase_323608, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 192, 4, False)
        # Assigning a type to the variable 'self' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzwin.__init__', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 193):
        
        # Assigning a Name to a Attribute (line 193):
        # Getting the type of 'name' (line 193)
        name_323609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 21), 'name')
        # Getting the type of 'self' (line 193)
        self_323610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'self')
        # Setting the type of the member '_name' of a type (line 193)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 8), self_323610, '_name', name_323609)
        
        # Call to ConnectRegistry(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'None' (line 196)
        None_323613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 36), 'None', False)
        # Getting the type of 'winreg' (line 196)
        winreg_323614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 42), 'winreg', False)
        # Obtaining the member 'HKEY_LOCAL_MACHINE' of a type (line 196)
        HKEY_LOCAL_MACHINE_323615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 42), winreg_323614, 'HKEY_LOCAL_MACHINE')
        # Processing the call keyword arguments (line 196)
        kwargs_323616 = {}
        # Getting the type of 'winreg' (line 196)
        winreg_323611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 13), 'winreg', False)
        # Obtaining the member 'ConnectRegistry' of a type (line 196)
        ConnectRegistry_323612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 13), winreg_323611, 'ConnectRegistry')
        # Calling ConnectRegistry(args, kwargs) (line 196)
        ConnectRegistry_call_result_323617 = invoke(stypy.reporting.localization.Localization(__file__, 196, 13), ConnectRegistry_323612, *[None_323613, HKEY_LOCAL_MACHINE_323615], **kwargs_323616)
        
        with_323618 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 196, 13), ConnectRegistry_call_result_323617, 'with parameter', '__enter__', '__exit__')

        if with_323618:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 196)
            enter___323619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 13), ConnectRegistry_call_result_323617, '__enter__')
            with_enter_323620 = invoke(stypy.reporting.localization.Localization(__file__, 196, 13), enter___323619)
            # Assigning a type to the variable 'handle' (line 196)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 13), 'handle', with_enter_323620)
            
            # Assigning a Call to a Name (line 197):
            
            # Assigning a Call to a Name (line 197):
            
            # Call to format(...): (line 197)
            # Processing the call keyword arguments (line 197)
            # Getting the type of 'TZKEYNAME' (line 197)
            TZKEYNAME_323626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 60), 'TZKEYNAME', False)
            keyword_323627 = TZKEYNAME_323626
            # Getting the type of 'name' (line 197)
            name_323628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 76), 'name', False)
            keyword_323629 = name_323628
            kwargs_323630 = {'kn': keyword_323627, 'name': keyword_323629}
            
            # Call to text_type(...): (line 197)
            # Processing the call arguments (line 197)
            str_323622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 34), 'str', '{kn}\\{name}')
            # Processing the call keyword arguments (line 197)
            kwargs_323623 = {}
            # Getting the type of 'text_type' (line 197)
            text_type_323621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 24), 'text_type', False)
            # Calling text_type(args, kwargs) (line 197)
            text_type_call_result_323624 = invoke(stypy.reporting.localization.Localization(__file__, 197, 24), text_type_323621, *[str_323622], **kwargs_323623)
            
            # Obtaining the member 'format' of a type (line 197)
            format_323625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 24), text_type_call_result_323624, 'format')
            # Calling format(args, kwargs) (line 197)
            format_call_result_323631 = invoke(stypy.reporting.localization.Localization(__file__, 197, 24), format_323625, *[], **kwargs_323630)
            
            # Assigning a type to the variable 'tzkeyname' (line 197)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'tzkeyname', format_call_result_323631)
            
            # Call to OpenKey(...): (line 198)
            # Processing the call arguments (line 198)
            # Getting the type of 'handle' (line 198)
            handle_323634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 32), 'handle', False)
            # Getting the type of 'tzkeyname' (line 198)
            tzkeyname_323635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 40), 'tzkeyname', False)
            # Processing the call keyword arguments (line 198)
            kwargs_323636 = {}
            # Getting the type of 'winreg' (line 198)
            winreg_323632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 17), 'winreg', False)
            # Obtaining the member 'OpenKey' of a type (line 198)
            OpenKey_323633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 17), winreg_323632, 'OpenKey')
            # Calling OpenKey(args, kwargs) (line 198)
            OpenKey_call_result_323637 = invoke(stypy.reporting.localization.Localization(__file__, 198, 17), OpenKey_323633, *[handle_323634, tzkeyname_323635], **kwargs_323636)
            
            with_323638 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 198, 17), OpenKey_call_result_323637, 'with parameter', '__enter__', '__exit__')

            if with_323638:
                # Calling the __enter__ method to initiate a with section
                # Obtaining the member '__enter__' of a type (line 198)
                enter___323639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 17), OpenKey_call_result_323637, '__enter__')
                with_enter_323640 = invoke(stypy.reporting.localization.Localization(__file__, 198, 17), enter___323639)
                # Assigning a type to the variable 'tzkey' (line 198)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 17), 'tzkey', with_enter_323640)
                
                # Assigning a Call to a Name (line 199):
                
                # Assigning a Call to a Name (line 199):
                
                # Call to valuestodict(...): (line 199)
                # Processing the call arguments (line 199)
                # Getting the type of 'tzkey' (line 199)
                tzkey_323642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 39), 'tzkey', False)
                # Processing the call keyword arguments (line 199)
                kwargs_323643 = {}
                # Getting the type of 'valuestodict' (line 199)
                valuestodict_323641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 26), 'valuestodict', False)
                # Calling valuestodict(args, kwargs) (line 199)
                valuestodict_call_result_323644 = invoke(stypy.reporting.localization.Localization(__file__, 199, 26), valuestodict_323641, *[tzkey_323642], **kwargs_323643)
                
                # Assigning a type to the variable 'keydict' (line 199)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'keydict', valuestodict_call_result_323644)
                # Calling the __exit__ method to finish a with section
                # Obtaining the member '__exit__' of a type (line 198)
                exit___323645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 17), OpenKey_call_result_323637, '__exit__')
                with_exit_323646 = invoke(stypy.reporting.localization.Localization(__file__, 198, 17), exit___323645, None, None, None)

            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 196)
            exit___323647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 13), ConnectRegistry_call_result_323617, '__exit__')
            with_exit_323648 = invoke(stypy.reporting.localization.Localization(__file__, 196, 13), exit___323647, None, None, None)

        
        # Assigning a Subscript to a Attribute (line 201):
        
        # Assigning a Subscript to a Attribute (line 201):
        
        # Obtaining the type of the subscript
        str_323649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 33), 'str', 'Std')
        # Getting the type of 'keydict' (line 201)
        keydict_323650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 25), 'keydict')
        # Obtaining the member '__getitem__' of a type (line 201)
        getitem___323651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 25), keydict_323650, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 201)
        subscript_call_result_323652 = invoke(stypy.reporting.localization.Localization(__file__, 201, 25), getitem___323651, str_323649)
        
        # Getting the type of 'self' (line 201)
        self_323653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'self')
        # Setting the type of the member '_std_abbr' of a type (line 201)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), self_323653, '_std_abbr', subscript_call_result_323652)
        
        # Assigning a Subscript to a Attribute (line 202):
        
        # Assigning a Subscript to a Attribute (line 202):
        
        # Obtaining the type of the subscript
        str_323654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 33), 'str', 'Dlt')
        # Getting the type of 'keydict' (line 202)
        keydict_323655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 25), 'keydict')
        # Obtaining the member '__getitem__' of a type (line 202)
        getitem___323656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 25), keydict_323655, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 202)
        subscript_call_result_323657 = invoke(stypy.reporting.localization.Localization(__file__, 202, 25), getitem___323656, str_323654)
        
        # Getting the type of 'self' (line 202)
        self_323658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'self')
        # Setting the type of the member '_dst_abbr' of a type (line 202)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), self_323658, '_dst_abbr', subscript_call_result_323657)
        
        # Assigning a Subscript to a Attribute (line 204):
        
        # Assigning a Subscript to a Attribute (line 204):
        
        # Obtaining the type of the subscript
        str_323659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 32), 'str', 'Display')
        # Getting the type of 'keydict' (line 204)
        keydict_323660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 24), 'keydict')
        # Obtaining the member '__getitem__' of a type (line 204)
        getitem___323661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 24), keydict_323660, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 204)
        subscript_call_result_323662 = invoke(stypy.reporting.localization.Localization(__file__, 204, 24), getitem___323661, str_323659)
        
        # Getting the type of 'self' (line 204)
        self_323663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'self')
        # Setting the type of the member '_display' of a type (line 204)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), self_323663, '_display', subscript_call_result_323662)
        
        # Assigning a Call to a Name (line 207):
        
        # Assigning a Call to a Name (line 207):
        
        # Call to unpack(...): (line 207)
        # Processing the call arguments (line 207)
        str_323666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 28), 'str', '=3l16h')
        
        # Obtaining the type of the subscript
        str_323667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 46), 'str', 'TZI')
        # Getting the type of 'keydict' (line 207)
        keydict_323668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 38), 'keydict', False)
        # Obtaining the member '__getitem__' of a type (line 207)
        getitem___323669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 38), keydict_323668, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 207)
        subscript_call_result_323670 = invoke(stypy.reporting.localization.Localization(__file__, 207, 38), getitem___323669, str_323667)
        
        # Processing the call keyword arguments (line 207)
        kwargs_323671 = {}
        # Getting the type of 'struct' (line 207)
        struct_323664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 14), 'struct', False)
        # Obtaining the member 'unpack' of a type (line 207)
        unpack_323665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 14), struct_323664, 'unpack')
        # Calling unpack(args, kwargs) (line 207)
        unpack_call_result_323672 = invoke(stypy.reporting.localization.Localization(__file__, 207, 14), unpack_323665, *[str_323666, subscript_call_result_323670], **kwargs_323671)
        
        # Assigning a type to the variable 'tup' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'tup', unpack_call_result_323672)
        
        # Assigning a BinOp to a Name (line 208):
        
        # Assigning a BinOp to a Name (line 208):
        
        
        # Obtaining the type of the subscript
        int_323673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 25), 'int')
        # Getting the type of 'tup' (line 208)
        tup_323674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 21), 'tup')
        # Obtaining the member '__getitem__' of a type (line 208)
        getitem___323675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 21), tup_323674, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 208)
        subscript_call_result_323676 = invoke(stypy.reporting.localization.Localization(__file__, 208, 21), getitem___323675, int_323673)
        
        # Applying the 'usub' unary operator (line 208)
        result___neg___323677 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 20), 'usub', subscript_call_result_323676)
        
        
        # Obtaining the type of the subscript
        int_323678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 32), 'int')
        # Getting the type of 'tup' (line 208)
        tup_323679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 28), 'tup')
        # Obtaining the member '__getitem__' of a type (line 208)
        getitem___323680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 28), tup_323679, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 208)
        subscript_call_result_323681 = invoke(stypy.reporting.localization.Localization(__file__, 208, 28), getitem___323680, int_323678)
        
        # Applying the binary operator '-' (line 208)
        result_sub_323682 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 20), '-', result___neg___323677, subscript_call_result_323681)
        
        # Assigning a type to the variable 'stdoffset' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'stdoffset', result_sub_323682)
        
        # Assigning a BinOp to a Name (line 209):
        
        # Assigning a BinOp to a Name (line 209):
        # Getting the type of 'stdoffset' (line 209)
        stdoffset_323683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 20), 'stdoffset')
        
        # Obtaining the type of the subscript
        int_323684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 34), 'int')
        # Getting the type of 'tup' (line 209)
        tup_323685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 30), 'tup')
        # Obtaining the member '__getitem__' of a type (line 209)
        getitem___323686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 30), tup_323685, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 209)
        subscript_call_result_323687 = invoke(stypy.reporting.localization.Localization(__file__, 209, 30), getitem___323686, int_323684)
        
        # Applying the binary operator '-' (line 209)
        result_sub_323688 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 20), '-', stdoffset_323683, subscript_call_result_323687)
        
        # Assigning a type to the variable 'dstoffset' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'dstoffset', result_sub_323688)
        
        # Assigning a Call to a Attribute (line 210):
        
        # Assigning a Call to a Attribute (line 210):
        
        # Call to timedelta(...): (line 210)
        # Processing the call keyword arguments (line 210)
        # Getting the type of 'stdoffset' (line 210)
        stdoffset_323691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 54), 'stdoffset', False)
        keyword_323692 = stdoffset_323691
        kwargs_323693 = {'minutes': keyword_323692}
        # Getting the type of 'datetime' (line 210)
        datetime_323689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 27), 'datetime', False)
        # Obtaining the member 'timedelta' of a type (line 210)
        timedelta_323690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 27), datetime_323689, 'timedelta')
        # Calling timedelta(args, kwargs) (line 210)
        timedelta_call_result_323694 = invoke(stypy.reporting.localization.Localization(__file__, 210, 27), timedelta_323690, *[], **kwargs_323693)
        
        # Getting the type of 'self' (line 210)
        self_323695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'self')
        # Setting the type of the member '_std_offset' of a type (line 210)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), self_323695, '_std_offset', timedelta_call_result_323694)
        
        # Assigning a Call to a Attribute (line 211):
        
        # Assigning a Call to a Attribute (line 211):
        
        # Call to timedelta(...): (line 211)
        # Processing the call keyword arguments (line 211)
        # Getting the type of 'dstoffset' (line 211)
        dstoffset_323698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 54), 'dstoffset', False)
        keyword_323699 = dstoffset_323698
        kwargs_323700 = {'minutes': keyword_323699}
        # Getting the type of 'datetime' (line 211)
        datetime_323696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 27), 'datetime', False)
        # Obtaining the member 'timedelta' of a type (line 211)
        timedelta_323697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 27), datetime_323696, 'timedelta')
        # Calling timedelta(args, kwargs) (line 211)
        timedelta_call_result_323701 = invoke(stypy.reporting.localization.Localization(__file__, 211, 27), timedelta_323697, *[], **kwargs_323700)
        
        # Getting the type of 'self' (line 211)
        self_323702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'self')
        # Setting the type of the member '_dst_offset' of a type (line 211)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), self_323702, '_dst_offset', timedelta_call_result_323701)
        
        # Assigning a Subscript to a Tuple (line 215):
        
        # Assigning a Subscript to a Name (line 215):
        
        # Obtaining the type of the subscript
        int_323703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 8), 'int')
        
        # Obtaining the type of the subscript
        int_323704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 32), 'int')
        int_323705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 34), 'int')
        slice_323706 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 219, 28), int_323704, int_323705, None)
        # Getting the type of 'tup' (line 219)
        tup_323707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 28), 'tup')
        # Obtaining the member '__getitem__' of a type (line 219)
        getitem___323708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 28), tup_323707, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 219)
        subscript_call_result_323709 = invoke(stypy.reporting.localization.Localization(__file__, 219, 28), getitem___323708, slice_323706)
        
        # Obtaining the member '__getitem__' of a type (line 215)
        getitem___323710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), subscript_call_result_323709, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 215)
        subscript_call_result_323711 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), getitem___323710, int_323703)
        
        # Assigning a type to the variable 'tuple_var_assignment_323254' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'tuple_var_assignment_323254', subscript_call_result_323711)
        
        # Assigning a Subscript to a Name (line 215):
        
        # Obtaining the type of the subscript
        int_323712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 8), 'int')
        
        # Obtaining the type of the subscript
        int_323713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 32), 'int')
        int_323714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 34), 'int')
        slice_323715 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 219, 28), int_323713, int_323714, None)
        # Getting the type of 'tup' (line 219)
        tup_323716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 28), 'tup')
        # Obtaining the member '__getitem__' of a type (line 219)
        getitem___323717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 28), tup_323716, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 219)
        subscript_call_result_323718 = invoke(stypy.reporting.localization.Localization(__file__, 219, 28), getitem___323717, slice_323715)
        
        # Obtaining the member '__getitem__' of a type (line 215)
        getitem___323719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), subscript_call_result_323718, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 215)
        subscript_call_result_323720 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), getitem___323719, int_323712)
        
        # Assigning a type to the variable 'tuple_var_assignment_323255' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'tuple_var_assignment_323255', subscript_call_result_323720)
        
        # Assigning a Subscript to a Name (line 215):
        
        # Obtaining the type of the subscript
        int_323721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 8), 'int')
        
        # Obtaining the type of the subscript
        int_323722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 32), 'int')
        int_323723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 34), 'int')
        slice_323724 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 219, 28), int_323722, int_323723, None)
        # Getting the type of 'tup' (line 219)
        tup_323725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 28), 'tup')
        # Obtaining the member '__getitem__' of a type (line 219)
        getitem___323726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 28), tup_323725, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 219)
        subscript_call_result_323727 = invoke(stypy.reporting.localization.Localization(__file__, 219, 28), getitem___323726, slice_323724)
        
        # Obtaining the member '__getitem__' of a type (line 215)
        getitem___323728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), subscript_call_result_323727, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 215)
        subscript_call_result_323729 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), getitem___323728, int_323721)
        
        # Assigning a type to the variable 'tuple_var_assignment_323256' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'tuple_var_assignment_323256', subscript_call_result_323729)
        
        # Assigning a Subscript to a Name (line 215):
        
        # Obtaining the type of the subscript
        int_323730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 8), 'int')
        
        # Obtaining the type of the subscript
        int_323731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 32), 'int')
        int_323732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 34), 'int')
        slice_323733 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 219, 28), int_323731, int_323732, None)
        # Getting the type of 'tup' (line 219)
        tup_323734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 28), 'tup')
        # Obtaining the member '__getitem__' of a type (line 219)
        getitem___323735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 28), tup_323734, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 219)
        subscript_call_result_323736 = invoke(stypy.reporting.localization.Localization(__file__, 219, 28), getitem___323735, slice_323733)
        
        # Obtaining the member '__getitem__' of a type (line 215)
        getitem___323737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), subscript_call_result_323736, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 215)
        subscript_call_result_323738 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), getitem___323737, int_323730)
        
        # Assigning a type to the variable 'tuple_var_assignment_323257' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'tuple_var_assignment_323257', subscript_call_result_323738)
        
        # Assigning a Subscript to a Name (line 215):
        
        # Obtaining the type of the subscript
        int_323739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 8), 'int')
        
        # Obtaining the type of the subscript
        int_323740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 32), 'int')
        int_323741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 34), 'int')
        slice_323742 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 219, 28), int_323740, int_323741, None)
        # Getting the type of 'tup' (line 219)
        tup_323743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 28), 'tup')
        # Obtaining the member '__getitem__' of a type (line 219)
        getitem___323744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 28), tup_323743, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 219)
        subscript_call_result_323745 = invoke(stypy.reporting.localization.Localization(__file__, 219, 28), getitem___323744, slice_323742)
        
        # Obtaining the member '__getitem__' of a type (line 215)
        getitem___323746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), subscript_call_result_323745, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 215)
        subscript_call_result_323747 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), getitem___323746, int_323739)
        
        # Assigning a type to the variable 'tuple_var_assignment_323258' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'tuple_var_assignment_323258', subscript_call_result_323747)
        
        # Assigning a Name to a Attribute (line 215):
        # Getting the type of 'tuple_var_assignment_323254' (line 215)
        tuple_var_assignment_323254_323748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'tuple_var_assignment_323254')
        # Getting the type of 'self' (line 215)
        self_323749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 9), 'self')
        # Setting the type of the member '_stdmonth' of a type (line 215)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 9), self_323749, '_stdmonth', tuple_var_assignment_323254_323748)
        
        # Assigning a Name to a Attribute (line 215):
        # Getting the type of 'tuple_var_assignment_323255' (line 215)
        tuple_var_assignment_323255_323750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'tuple_var_assignment_323255')
        # Getting the type of 'self' (line 216)
        self_323751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 9), 'self')
        # Setting the type of the member '_stddayofweek' of a type (line 216)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 9), self_323751, '_stddayofweek', tuple_var_assignment_323255_323750)
        
        # Assigning a Name to a Attribute (line 215):
        # Getting the type of 'tuple_var_assignment_323256' (line 215)
        tuple_var_assignment_323256_323752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'tuple_var_assignment_323256')
        # Getting the type of 'self' (line 217)
        self_323753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 9), 'self')
        # Setting the type of the member '_stdweeknumber' of a type (line 217)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 9), self_323753, '_stdweeknumber', tuple_var_assignment_323256_323752)
        
        # Assigning a Name to a Attribute (line 215):
        # Getting the type of 'tuple_var_assignment_323257' (line 215)
        tuple_var_assignment_323257_323754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'tuple_var_assignment_323257')
        # Getting the type of 'self' (line 218)
        self_323755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 9), 'self')
        # Setting the type of the member '_stdhour' of a type (line 218)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 9), self_323755, '_stdhour', tuple_var_assignment_323257_323754)
        
        # Assigning a Name to a Attribute (line 215):
        # Getting the type of 'tuple_var_assignment_323258' (line 215)
        tuple_var_assignment_323258_323756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'tuple_var_assignment_323258')
        # Getting the type of 'self' (line 219)
        self_323757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 9), 'self')
        # Setting the type of the member '_stdminute' of a type (line 219)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 9), self_323757, '_stdminute', tuple_var_assignment_323258_323756)
        
        # Assigning a Subscript to a Tuple (line 221):
        
        # Assigning a Subscript to a Name (line 221):
        
        # Obtaining the type of the subscript
        int_323758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 8), 'int')
        
        # Obtaining the type of the subscript
        int_323759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 32), 'int')
        int_323760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 35), 'int')
        slice_323761 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 225, 28), int_323759, int_323760, None)
        # Getting the type of 'tup' (line 225)
        tup_323762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 28), 'tup')
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___323763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 28), tup_323762, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 225)
        subscript_call_result_323764 = invoke(stypy.reporting.localization.Localization(__file__, 225, 28), getitem___323763, slice_323761)
        
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___323765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), subscript_call_result_323764, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_323766 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), getitem___323765, int_323758)
        
        # Assigning a type to the variable 'tuple_var_assignment_323259' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'tuple_var_assignment_323259', subscript_call_result_323766)
        
        # Assigning a Subscript to a Name (line 221):
        
        # Obtaining the type of the subscript
        int_323767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 8), 'int')
        
        # Obtaining the type of the subscript
        int_323768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 32), 'int')
        int_323769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 35), 'int')
        slice_323770 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 225, 28), int_323768, int_323769, None)
        # Getting the type of 'tup' (line 225)
        tup_323771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 28), 'tup')
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___323772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 28), tup_323771, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 225)
        subscript_call_result_323773 = invoke(stypy.reporting.localization.Localization(__file__, 225, 28), getitem___323772, slice_323770)
        
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___323774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), subscript_call_result_323773, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_323775 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), getitem___323774, int_323767)
        
        # Assigning a type to the variable 'tuple_var_assignment_323260' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'tuple_var_assignment_323260', subscript_call_result_323775)
        
        # Assigning a Subscript to a Name (line 221):
        
        # Obtaining the type of the subscript
        int_323776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 8), 'int')
        
        # Obtaining the type of the subscript
        int_323777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 32), 'int')
        int_323778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 35), 'int')
        slice_323779 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 225, 28), int_323777, int_323778, None)
        # Getting the type of 'tup' (line 225)
        tup_323780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 28), 'tup')
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___323781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 28), tup_323780, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 225)
        subscript_call_result_323782 = invoke(stypy.reporting.localization.Localization(__file__, 225, 28), getitem___323781, slice_323779)
        
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___323783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), subscript_call_result_323782, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_323784 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), getitem___323783, int_323776)
        
        # Assigning a type to the variable 'tuple_var_assignment_323261' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'tuple_var_assignment_323261', subscript_call_result_323784)
        
        # Assigning a Subscript to a Name (line 221):
        
        # Obtaining the type of the subscript
        int_323785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 8), 'int')
        
        # Obtaining the type of the subscript
        int_323786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 32), 'int')
        int_323787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 35), 'int')
        slice_323788 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 225, 28), int_323786, int_323787, None)
        # Getting the type of 'tup' (line 225)
        tup_323789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 28), 'tup')
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___323790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 28), tup_323789, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 225)
        subscript_call_result_323791 = invoke(stypy.reporting.localization.Localization(__file__, 225, 28), getitem___323790, slice_323788)
        
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___323792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), subscript_call_result_323791, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_323793 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), getitem___323792, int_323785)
        
        # Assigning a type to the variable 'tuple_var_assignment_323262' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'tuple_var_assignment_323262', subscript_call_result_323793)
        
        # Assigning a Subscript to a Name (line 221):
        
        # Obtaining the type of the subscript
        int_323794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 8), 'int')
        
        # Obtaining the type of the subscript
        int_323795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 32), 'int')
        int_323796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 35), 'int')
        slice_323797 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 225, 28), int_323795, int_323796, None)
        # Getting the type of 'tup' (line 225)
        tup_323798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 28), 'tup')
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___323799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 28), tup_323798, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 225)
        subscript_call_result_323800 = invoke(stypy.reporting.localization.Localization(__file__, 225, 28), getitem___323799, slice_323797)
        
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___323801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), subscript_call_result_323800, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_323802 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), getitem___323801, int_323794)
        
        # Assigning a type to the variable 'tuple_var_assignment_323263' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'tuple_var_assignment_323263', subscript_call_result_323802)
        
        # Assigning a Name to a Attribute (line 221):
        # Getting the type of 'tuple_var_assignment_323259' (line 221)
        tuple_var_assignment_323259_323803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'tuple_var_assignment_323259')
        # Getting the type of 'self' (line 221)
        self_323804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 9), 'self')
        # Setting the type of the member '_dstmonth' of a type (line 221)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 9), self_323804, '_dstmonth', tuple_var_assignment_323259_323803)
        
        # Assigning a Name to a Attribute (line 221):
        # Getting the type of 'tuple_var_assignment_323260' (line 221)
        tuple_var_assignment_323260_323805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'tuple_var_assignment_323260')
        # Getting the type of 'self' (line 222)
        self_323806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 9), 'self')
        # Setting the type of the member '_dstdayofweek' of a type (line 222)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 9), self_323806, '_dstdayofweek', tuple_var_assignment_323260_323805)
        
        # Assigning a Name to a Attribute (line 221):
        # Getting the type of 'tuple_var_assignment_323261' (line 221)
        tuple_var_assignment_323261_323807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'tuple_var_assignment_323261')
        # Getting the type of 'self' (line 223)
        self_323808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 9), 'self')
        # Setting the type of the member '_dstweeknumber' of a type (line 223)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 9), self_323808, '_dstweeknumber', tuple_var_assignment_323261_323807)
        
        # Assigning a Name to a Attribute (line 221):
        # Getting the type of 'tuple_var_assignment_323262' (line 221)
        tuple_var_assignment_323262_323809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'tuple_var_assignment_323262')
        # Getting the type of 'self' (line 224)
        self_323810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 9), 'self')
        # Setting the type of the member '_dsthour' of a type (line 224)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 9), self_323810, '_dsthour', tuple_var_assignment_323262_323809)
        
        # Assigning a Name to a Attribute (line 221):
        # Getting the type of 'tuple_var_assignment_323263' (line 221)
        tuple_var_assignment_323263_323811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'tuple_var_assignment_323263')
        # Getting the type of 'self' (line 225)
        self_323812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 9), 'self')
        # Setting the type of the member '_dstminute' of a type (line 225)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 9), self_323812, '_dstminute', tuple_var_assignment_323263_323811)
        
        # Assigning a BinOp to a Attribute (line 227):
        
        # Assigning a BinOp to a Attribute (line 227):
        # Getting the type of 'self' (line 227)
        self_323813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 33), 'self')
        # Obtaining the member '_dst_offset' of a type (line 227)
        _dst_offset_323814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 33), self_323813, '_dst_offset')
        # Getting the type of 'self' (line 227)
        self_323815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 52), 'self')
        # Obtaining the member '_std_offset' of a type (line 227)
        _std_offset_323816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 52), self_323815, '_std_offset')
        # Applying the binary operator '-' (line 227)
        result_sub_323817 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 33), '-', _dst_offset_323814, _std_offset_323816)
        
        # Getting the type of 'self' (line 227)
        self_323818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'self')
        # Setting the type of the member '_dst_base_offset_' of a type (line 227)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), self_323818, '_dst_base_offset_', result_sub_323817)
        
        # Assigning a Call to a Attribute (line 228):
        
        # Assigning a Call to a Attribute (line 228):
        
        # Call to _get_hasdst(...): (line 228)
        # Processing the call keyword arguments (line 228)
        kwargs_323821 = {}
        # Getting the type of 'self' (line 228)
        self_323819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 22), 'self', False)
        # Obtaining the member '_get_hasdst' of a type (line 228)
        _get_hasdst_323820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 22), self_323819, '_get_hasdst')
        # Calling _get_hasdst(args, kwargs) (line 228)
        _get_hasdst_call_result_323822 = invoke(stypy.reporting.localization.Localization(__file__, 228, 22), _get_hasdst_323820, *[], **kwargs_323821)
        
        # Getting the type of 'self' (line 228)
        self_323823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'self')
        # Setting the type of the member 'hasdst' of a type (line 228)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), self_323823, 'hasdst', _get_hasdst_call_result_323822)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 230, 4, False)
        # Assigning a type to the variable 'self' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzwin.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        tzwin.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzwin.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzwin.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'tzwin.stypy__repr__')
        tzwin.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        tzwin.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzwin.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzwin.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzwin.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzwin.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzwin.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzwin.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        str_323824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 15), 'str', 'tzwin(%s)')
        
        # Call to repr(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'self' (line 231)
        self_323826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 34), 'self', False)
        # Obtaining the member '_name' of a type (line 231)
        _name_323827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 34), self_323826, '_name')
        # Processing the call keyword arguments (line 231)
        kwargs_323828 = {}
        # Getting the type of 'repr' (line 231)
        repr_323825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 29), 'repr', False)
        # Calling repr(args, kwargs) (line 231)
        repr_call_result_323829 = invoke(stypy.reporting.localization.Localization(__file__, 231, 29), repr_323825, *[_name_323827], **kwargs_323828)
        
        # Applying the binary operator '%' (line 231)
        result_mod_323830 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 15), '%', str_323824, repr_call_result_323829)
        
        # Assigning a type to the variable 'stypy_return_type' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'stypy_return_type', result_mod_323830)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 230)
        stypy_return_type_323831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_323831)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_323831


    @norecursion
    def __reduce__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__reduce__'
        module_type_store = module_type_store.open_function_context('__reduce__', 233, 4, False)
        # Assigning a type to the variable 'self' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzwin.__reduce__.__dict__.__setitem__('stypy_localization', localization)
        tzwin.__reduce__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzwin.__reduce__.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzwin.__reduce__.__dict__.__setitem__('stypy_function_name', 'tzwin.__reduce__')
        tzwin.__reduce__.__dict__.__setitem__('stypy_param_names_list', [])
        tzwin.__reduce__.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzwin.__reduce__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzwin.__reduce__.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzwin.__reduce__.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzwin.__reduce__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzwin.__reduce__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzwin.__reduce__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__reduce__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__reduce__(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 234)
        tuple_323832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 234)
        # Adding element type (line 234)
        # Getting the type of 'self' (line 234)
        self_323833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 16), 'self')
        # Obtaining the member '__class__' of a type (line 234)
        class___323834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 16), self_323833, '__class__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 16), tuple_323832, class___323834)
        # Adding element type (line 234)
        
        # Obtaining an instance of the builtin type 'tuple' (line 234)
        tuple_323835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 234)
        # Adding element type (line 234)
        # Getting the type of 'self' (line 234)
        self_323836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 33), 'self')
        # Obtaining the member '_name' of a type (line 234)
        _name_323837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 33), self_323836, '_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 33), tuple_323835, _name_323837)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 16), tuple_323832, tuple_323835)
        
        # Assigning a type to the variable 'stypy_return_type' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'stypy_return_type', tuple_323832)
        
        # ################# End of '__reduce__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__reduce__' in the type store
        # Getting the type of 'stypy_return_type' (line 233)
        stypy_return_type_323838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_323838)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__reduce__'
        return stypy_return_type_323838


# Assigning a type to the variable 'tzwin' (line 190)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'tzwin', tzwin)
# Declaration of the 'tzwinlocal' class
# Getting the type of 'tzwinbase' (line 237)
tzwinbase_323839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 17), 'tzwinbase')

class tzwinlocal(tzwinbase_323839, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 238, 4, False)
        # Assigning a type to the variable 'self' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzwinlocal.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to ConnectRegistry(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'None' (line 239)
        None_323842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 36), 'None', False)
        # Getting the type of 'winreg' (line 239)
        winreg_323843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 42), 'winreg', False)
        # Obtaining the member 'HKEY_LOCAL_MACHINE' of a type (line 239)
        HKEY_LOCAL_MACHINE_323844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 42), winreg_323843, 'HKEY_LOCAL_MACHINE')
        # Processing the call keyword arguments (line 239)
        kwargs_323845 = {}
        # Getting the type of 'winreg' (line 239)
        winreg_323840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 13), 'winreg', False)
        # Obtaining the member 'ConnectRegistry' of a type (line 239)
        ConnectRegistry_323841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 13), winreg_323840, 'ConnectRegistry')
        # Calling ConnectRegistry(args, kwargs) (line 239)
        ConnectRegistry_call_result_323846 = invoke(stypy.reporting.localization.Localization(__file__, 239, 13), ConnectRegistry_323841, *[None_323842, HKEY_LOCAL_MACHINE_323844], **kwargs_323845)
        
        with_323847 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 239, 13), ConnectRegistry_call_result_323846, 'with parameter', '__enter__', '__exit__')

        if with_323847:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 239)
            enter___323848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 13), ConnectRegistry_call_result_323846, '__enter__')
            with_enter_323849 = invoke(stypy.reporting.localization.Localization(__file__, 239, 13), enter___323848)
            # Assigning a type to the variable 'handle' (line 239)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 13), 'handle', with_enter_323849)
            
            # Call to OpenKey(...): (line 240)
            # Processing the call arguments (line 240)
            # Getting the type of 'handle' (line 240)
            handle_323852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 32), 'handle', False)
            # Getting the type of 'TZLOCALKEYNAME' (line 240)
            TZLOCALKEYNAME_323853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 40), 'TZLOCALKEYNAME', False)
            # Processing the call keyword arguments (line 240)
            kwargs_323854 = {}
            # Getting the type of 'winreg' (line 240)
            winreg_323850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 17), 'winreg', False)
            # Obtaining the member 'OpenKey' of a type (line 240)
            OpenKey_323851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 17), winreg_323850, 'OpenKey')
            # Calling OpenKey(args, kwargs) (line 240)
            OpenKey_call_result_323855 = invoke(stypy.reporting.localization.Localization(__file__, 240, 17), OpenKey_323851, *[handle_323852, TZLOCALKEYNAME_323853], **kwargs_323854)
            
            with_323856 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 240, 17), OpenKey_call_result_323855, 'with parameter', '__enter__', '__exit__')

            if with_323856:
                # Calling the __enter__ method to initiate a with section
                # Obtaining the member '__enter__' of a type (line 240)
                enter___323857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 17), OpenKey_call_result_323855, '__enter__')
                with_enter_323858 = invoke(stypy.reporting.localization.Localization(__file__, 240, 17), enter___323857)
                # Assigning a type to the variable 'tzlocalkey' (line 240)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 17), 'tzlocalkey', with_enter_323858)
                
                # Assigning a Call to a Name (line 241):
                
                # Assigning a Call to a Name (line 241):
                
                # Call to valuestodict(...): (line 241)
                # Processing the call arguments (line 241)
                # Getting the type of 'tzlocalkey' (line 241)
                tzlocalkey_323860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 39), 'tzlocalkey', False)
                # Processing the call keyword arguments (line 241)
                kwargs_323861 = {}
                # Getting the type of 'valuestodict' (line 241)
                valuestodict_323859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 26), 'valuestodict', False)
                # Calling valuestodict(args, kwargs) (line 241)
                valuestodict_call_result_323862 = invoke(stypy.reporting.localization.Localization(__file__, 241, 26), valuestodict_323859, *[tzlocalkey_323860], **kwargs_323861)
                
                # Assigning a type to the variable 'keydict' (line 241)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 16), 'keydict', valuestodict_call_result_323862)
                # Calling the __exit__ method to finish a with section
                # Obtaining the member '__exit__' of a type (line 240)
                exit___323863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 17), OpenKey_call_result_323855, '__exit__')
                with_exit_323864 = invoke(stypy.reporting.localization.Localization(__file__, 240, 17), exit___323863, None, None, None)

            
            # Assigning a Subscript to a Attribute (line 243):
            
            # Assigning a Subscript to a Attribute (line 243):
            
            # Obtaining the type of the subscript
            str_323865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 37), 'str', 'StandardName')
            # Getting the type of 'keydict' (line 243)
            keydict_323866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 29), 'keydict')
            # Obtaining the member '__getitem__' of a type (line 243)
            getitem___323867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 29), keydict_323866, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 243)
            subscript_call_result_323868 = invoke(stypy.reporting.localization.Localization(__file__, 243, 29), getitem___323867, str_323865)
            
            # Getting the type of 'self' (line 243)
            self_323869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'self')
            # Setting the type of the member '_std_abbr' of a type (line 243)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 12), self_323869, '_std_abbr', subscript_call_result_323868)
            
            # Assigning a Subscript to a Attribute (line 244):
            
            # Assigning a Subscript to a Attribute (line 244):
            
            # Obtaining the type of the subscript
            str_323870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 37), 'str', 'DaylightName')
            # Getting the type of 'keydict' (line 244)
            keydict_323871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 29), 'keydict')
            # Obtaining the member '__getitem__' of a type (line 244)
            getitem___323872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 29), keydict_323871, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 244)
            subscript_call_result_323873 = invoke(stypy.reporting.localization.Localization(__file__, 244, 29), getitem___323872, str_323870)
            
            # Getting the type of 'self' (line 244)
            self_323874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'self')
            # Setting the type of the member '_dst_abbr' of a type (line 244)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 12), self_323874, '_dst_abbr', subscript_call_result_323873)
            
            
            # SSA begins for try-except statement (line 246)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 247):
            
            # Assigning a Call to a Name (line 247):
            
            # Call to format(...): (line 247)
            # Processing the call keyword arguments (line 247)
            # Getting the type of 'TZKEYNAME' (line 247)
            TZKEYNAME_323880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 62), 'TZKEYNAME', False)
            keyword_323881 = TZKEYNAME_323880
            # Getting the type of 'self' (line 248)
            self_323882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 61), 'self', False)
            # Obtaining the member '_std_abbr' of a type (line 248)
            _std_abbr_323883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 61), self_323882, '_std_abbr')
            keyword_323884 = _std_abbr_323883
            kwargs_323885 = {'kn': keyword_323881, 'sn': keyword_323884}
            
            # Call to text_type(...): (line 247)
            # Processing the call arguments (line 247)
            str_323876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 38), 'str', '{kn}\\{sn}')
            # Processing the call keyword arguments (line 247)
            kwargs_323877 = {}
            # Getting the type of 'text_type' (line 247)
            text_type_323875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 28), 'text_type', False)
            # Calling text_type(args, kwargs) (line 247)
            text_type_call_result_323878 = invoke(stypy.reporting.localization.Localization(__file__, 247, 28), text_type_323875, *[str_323876], **kwargs_323877)
            
            # Obtaining the member 'format' of a type (line 247)
            format_323879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 28), text_type_call_result_323878, 'format')
            # Calling format(args, kwargs) (line 247)
            format_call_result_323886 = invoke(stypy.reporting.localization.Localization(__file__, 247, 28), format_323879, *[], **kwargs_323885)
            
            # Assigning a type to the variable 'tzkeyname' (line 247)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'tzkeyname', format_call_result_323886)
            
            # Call to OpenKey(...): (line 249)
            # Processing the call arguments (line 249)
            # Getting the type of 'handle' (line 249)
            handle_323889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 36), 'handle', False)
            # Getting the type of 'tzkeyname' (line 249)
            tzkeyname_323890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 44), 'tzkeyname', False)
            # Processing the call keyword arguments (line 249)
            kwargs_323891 = {}
            # Getting the type of 'winreg' (line 249)
            winreg_323887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 21), 'winreg', False)
            # Obtaining the member 'OpenKey' of a type (line 249)
            OpenKey_323888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 21), winreg_323887, 'OpenKey')
            # Calling OpenKey(args, kwargs) (line 249)
            OpenKey_call_result_323892 = invoke(stypy.reporting.localization.Localization(__file__, 249, 21), OpenKey_323888, *[handle_323889, tzkeyname_323890], **kwargs_323891)
            
            with_323893 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 249, 21), OpenKey_call_result_323892, 'with parameter', '__enter__', '__exit__')

            if with_323893:
                # Calling the __enter__ method to initiate a with section
                # Obtaining the member '__enter__' of a type (line 249)
                enter___323894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 21), OpenKey_call_result_323892, '__enter__')
                with_enter_323895 = invoke(stypy.reporting.localization.Localization(__file__, 249, 21), enter___323894)
                # Assigning a type to the variable 'tzkey' (line 249)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 21), 'tzkey', with_enter_323895)
                
                # Assigning a Call to a Name (line 250):
                
                # Assigning a Call to a Name (line 250):
                
                # Call to valuestodict(...): (line 250)
                # Processing the call arguments (line 250)
                # Getting the type of 'tzkey' (line 250)
                tzkey_323897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 44), 'tzkey', False)
                # Processing the call keyword arguments (line 250)
                kwargs_323898 = {}
                # Getting the type of 'valuestodict' (line 250)
                valuestodict_323896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 31), 'valuestodict', False)
                # Calling valuestodict(args, kwargs) (line 250)
                valuestodict_call_result_323899 = invoke(stypy.reporting.localization.Localization(__file__, 250, 31), valuestodict_323896, *[tzkey_323897], **kwargs_323898)
                
                # Assigning a type to the variable '_keydict' (line 250)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 20), '_keydict', valuestodict_call_result_323899)
                
                # Assigning a Subscript to a Attribute (line 251):
                
                # Assigning a Subscript to a Attribute (line 251):
                
                # Obtaining the type of the subscript
                str_323900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 45), 'str', 'Display')
                # Getting the type of '_keydict' (line 251)
                _keydict_323901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 36), '_keydict')
                # Obtaining the member '__getitem__' of a type (line 251)
                getitem___323902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 36), _keydict_323901, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 251)
                subscript_call_result_323903 = invoke(stypy.reporting.localization.Localization(__file__, 251, 36), getitem___323902, str_323900)
                
                # Getting the type of 'self' (line 251)
                self_323904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 20), 'self')
                # Setting the type of the member '_display' of a type (line 251)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 20), self_323904, '_display', subscript_call_result_323903)
                # Calling the __exit__ method to finish a with section
                # Obtaining the member '__exit__' of a type (line 249)
                exit___323905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 21), OpenKey_call_result_323892, '__exit__')
                with_exit_323906 = invoke(stypy.reporting.localization.Localization(__file__, 249, 21), exit___323905, None, None, None)

            # SSA branch for the except part of a try statement (line 246)
            # SSA branch for the except 'OSError' branch of a try statement (line 246)
            module_type_store.open_ssa_branch('except')
            
            # Assigning a Name to a Attribute (line 253):
            
            # Assigning a Name to a Attribute (line 253):
            # Getting the type of 'None' (line 253)
            None_323907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 32), 'None')
            # Getting the type of 'self' (line 253)
            self_323908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'self')
            # Setting the type of the member '_display' of a type (line 253)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 16), self_323908, '_display', None_323907)
            # SSA join for try-except statement (line 246)
            module_type_store = module_type_store.join_ssa_context()
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 239)
            exit___323909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 13), ConnectRegistry_call_result_323846, '__exit__')
            with_exit_323910 = invoke(stypy.reporting.localization.Localization(__file__, 239, 13), exit___323909, None, None, None)

        
        # Assigning a BinOp to a Name (line 255):
        
        # Assigning a BinOp to a Name (line 255):
        
        
        # Obtaining the type of the subscript
        str_323911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 29), 'str', 'Bias')
        # Getting the type of 'keydict' (line 255)
        keydict_323912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 21), 'keydict')
        # Obtaining the member '__getitem__' of a type (line 255)
        getitem___323913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 21), keydict_323912, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 255)
        subscript_call_result_323914 = invoke(stypy.reporting.localization.Localization(__file__, 255, 21), getitem___323913, str_323911)
        
        # Applying the 'usub' unary operator (line 255)
        result___neg___323915 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 20), 'usub', subscript_call_result_323914)
        
        
        # Obtaining the type of the subscript
        str_323916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 45), 'str', 'StandardBias')
        # Getting the type of 'keydict' (line 255)
        keydict_323917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 37), 'keydict')
        # Obtaining the member '__getitem__' of a type (line 255)
        getitem___323918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 37), keydict_323917, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 255)
        subscript_call_result_323919 = invoke(stypy.reporting.localization.Localization(__file__, 255, 37), getitem___323918, str_323916)
        
        # Applying the binary operator '-' (line 255)
        result_sub_323920 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 20), '-', result___neg___323915, subscript_call_result_323919)
        
        # Assigning a type to the variable 'stdoffset' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'stdoffset', result_sub_323920)
        
        # Assigning a BinOp to a Name (line 256):
        
        # Assigning a BinOp to a Name (line 256):
        # Getting the type of 'stdoffset' (line 256)
        stdoffset_323921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 20), 'stdoffset')
        
        # Obtaining the type of the subscript
        str_323922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 38), 'str', 'DaylightBias')
        # Getting the type of 'keydict' (line 256)
        keydict_323923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 30), 'keydict')
        # Obtaining the member '__getitem__' of a type (line 256)
        getitem___323924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 30), keydict_323923, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 256)
        subscript_call_result_323925 = invoke(stypy.reporting.localization.Localization(__file__, 256, 30), getitem___323924, str_323922)
        
        # Applying the binary operator '-' (line 256)
        result_sub_323926 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 20), '-', stdoffset_323921, subscript_call_result_323925)
        
        # Assigning a type to the variable 'dstoffset' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'dstoffset', result_sub_323926)
        
        # Assigning a Call to a Attribute (line 258):
        
        # Assigning a Call to a Attribute (line 258):
        
        # Call to timedelta(...): (line 258)
        # Processing the call keyword arguments (line 258)
        # Getting the type of 'stdoffset' (line 258)
        stdoffset_323929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 54), 'stdoffset', False)
        keyword_323930 = stdoffset_323929
        kwargs_323931 = {'minutes': keyword_323930}
        # Getting the type of 'datetime' (line 258)
        datetime_323927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 27), 'datetime', False)
        # Obtaining the member 'timedelta' of a type (line 258)
        timedelta_323928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 27), datetime_323927, 'timedelta')
        # Calling timedelta(args, kwargs) (line 258)
        timedelta_call_result_323932 = invoke(stypy.reporting.localization.Localization(__file__, 258, 27), timedelta_323928, *[], **kwargs_323931)
        
        # Getting the type of 'self' (line 258)
        self_323933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'self')
        # Setting the type of the member '_std_offset' of a type (line 258)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 8), self_323933, '_std_offset', timedelta_call_result_323932)
        
        # Assigning a Call to a Attribute (line 259):
        
        # Assigning a Call to a Attribute (line 259):
        
        # Call to timedelta(...): (line 259)
        # Processing the call keyword arguments (line 259)
        # Getting the type of 'dstoffset' (line 259)
        dstoffset_323936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 54), 'dstoffset', False)
        keyword_323937 = dstoffset_323936
        kwargs_323938 = {'minutes': keyword_323937}
        # Getting the type of 'datetime' (line 259)
        datetime_323934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 27), 'datetime', False)
        # Obtaining the member 'timedelta' of a type (line 259)
        timedelta_323935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 27), datetime_323934, 'timedelta')
        # Calling timedelta(args, kwargs) (line 259)
        timedelta_call_result_323939 = invoke(stypy.reporting.localization.Localization(__file__, 259, 27), timedelta_323935, *[], **kwargs_323938)
        
        # Getting the type of 'self' (line 259)
        self_323940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'self')
        # Setting the type of the member '_dst_offset' of a type (line 259)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 8), self_323940, '_dst_offset', timedelta_call_result_323939)
        
        # Assigning a Call to a Name (line 263):
        
        # Assigning a Call to a Name (line 263):
        
        # Call to unpack(...): (line 263)
        # Processing the call arguments (line 263)
        str_323943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 28), 'str', '=8h')
        
        # Obtaining the type of the subscript
        str_323944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 43), 'str', 'StandardStart')
        # Getting the type of 'keydict' (line 263)
        keydict_323945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 35), 'keydict', False)
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___323946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 35), keydict_323945, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_323947 = invoke(stypy.reporting.localization.Localization(__file__, 263, 35), getitem___323946, str_323944)
        
        # Processing the call keyword arguments (line 263)
        kwargs_323948 = {}
        # Getting the type of 'struct' (line 263)
        struct_323941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 14), 'struct', False)
        # Obtaining the member 'unpack' of a type (line 263)
        unpack_323942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 14), struct_323941, 'unpack')
        # Calling unpack(args, kwargs) (line 263)
        unpack_call_result_323949 = invoke(stypy.reporting.localization.Localization(__file__, 263, 14), unpack_323942, *[str_323943, subscript_call_result_323947], **kwargs_323948)
        
        # Assigning a type to the variable 'tup' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'tup', unpack_call_result_323949)
        
        # Assigning a Subscript to a Tuple (line 265):
        
        # Assigning a Subscript to a Name (line 265):
        
        # Obtaining the type of the subscript
        int_323950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 8), 'int')
        
        # Obtaining the type of the subscript
        int_323951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 32), 'int')
        int_323952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 34), 'int')
        slice_323953 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 268, 28), int_323951, int_323952, None)
        # Getting the type of 'tup' (line 268)
        tup_323954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 28), 'tup')
        # Obtaining the member '__getitem__' of a type (line 268)
        getitem___323955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 28), tup_323954, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 268)
        subscript_call_result_323956 = invoke(stypy.reporting.localization.Localization(__file__, 268, 28), getitem___323955, slice_323953)
        
        # Obtaining the member '__getitem__' of a type (line 265)
        getitem___323957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), subscript_call_result_323956, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 265)
        subscript_call_result_323958 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), getitem___323957, int_323950)
        
        # Assigning a type to the variable 'tuple_var_assignment_323264' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'tuple_var_assignment_323264', subscript_call_result_323958)
        
        # Assigning a Subscript to a Name (line 265):
        
        # Obtaining the type of the subscript
        int_323959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 8), 'int')
        
        # Obtaining the type of the subscript
        int_323960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 32), 'int')
        int_323961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 34), 'int')
        slice_323962 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 268, 28), int_323960, int_323961, None)
        # Getting the type of 'tup' (line 268)
        tup_323963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 28), 'tup')
        # Obtaining the member '__getitem__' of a type (line 268)
        getitem___323964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 28), tup_323963, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 268)
        subscript_call_result_323965 = invoke(stypy.reporting.localization.Localization(__file__, 268, 28), getitem___323964, slice_323962)
        
        # Obtaining the member '__getitem__' of a type (line 265)
        getitem___323966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), subscript_call_result_323965, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 265)
        subscript_call_result_323967 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), getitem___323966, int_323959)
        
        # Assigning a type to the variable 'tuple_var_assignment_323265' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'tuple_var_assignment_323265', subscript_call_result_323967)
        
        # Assigning a Subscript to a Name (line 265):
        
        # Obtaining the type of the subscript
        int_323968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 8), 'int')
        
        # Obtaining the type of the subscript
        int_323969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 32), 'int')
        int_323970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 34), 'int')
        slice_323971 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 268, 28), int_323969, int_323970, None)
        # Getting the type of 'tup' (line 268)
        tup_323972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 28), 'tup')
        # Obtaining the member '__getitem__' of a type (line 268)
        getitem___323973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 28), tup_323972, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 268)
        subscript_call_result_323974 = invoke(stypy.reporting.localization.Localization(__file__, 268, 28), getitem___323973, slice_323971)
        
        # Obtaining the member '__getitem__' of a type (line 265)
        getitem___323975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), subscript_call_result_323974, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 265)
        subscript_call_result_323976 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), getitem___323975, int_323968)
        
        # Assigning a type to the variable 'tuple_var_assignment_323266' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'tuple_var_assignment_323266', subscript_call_result_323976)
        
        # Assigning a Subscript to a Name (line 265):
        
        # Obtaining the type of the subscript
        int_323977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 8), 'int')
        
        # Obtaining the type of the subscript
        int_323978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 32), 'int')
        int_323979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 34), 'int')
        slice_323980 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 268, 28), int_323978, int_323979, None)
        # Getting the type of 'tup' (line 268)
        tup_323981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 28), 'tup')
        # Obtaining the member '__getitem__' of a type (line 268)
        getitem___323982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 28), tup_323981, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 268)
        subscript_call_result_323983 = invoke(stypy.reporting.localization.Localization(__file__, 268, 28), getitem___323982, slice_323980)
        
        # Obtaining the member '__getitem__' of a type (line 265)
        getitem___323984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), subscript_call_result_323983, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 265)
        subscript_call_result_323985 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), getitem___323984, int_323977)
        
        # Assigning a type to the variable 'tuple_var_assignment_323267' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'tuple_var_assignment_323267', subscript_call_result_323985)
        
        # Assigning a Name to a Attribute (line 265):
        # Getting the type of 'tuple_var_assignment_323264' (line 265)
        tuple_var_assignment_323264_323986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'tuple_var_assignment_323264')
        # Getting the type of 'self' (line 265)
        self_323987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 9), 'self')
        # Setting the type of the member '_stdmonth' of a type (line 265)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 9), self_323987, '_stdmonth', tuple_var_assignment_323264_323986)
        
        # Assigning a Name to a Attribute (line 265):
        # Getting the type of 'tuple_var_assignment_323265' (line 265)
        tuple_var_assignment_323265_323988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'tuple_var_assignment_323265')
        # Getting the type of 'self' (line 266)
        self_323989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 9), 'self')
        # Setting the type of the member '_stdweeknumber' of a type (line 266)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 9), self_323989, '_stdweeknumber', tuple_var_assignment_323265_323988)
        
        # Assigning a Name to a Attribute (line 265):
        # Getting the type of 'tuple_var_assignment_323266' (line 265)
        tuple_var_assignment_323266_323990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'tuple_var_assignment_323266')
        # Getting the type of 'self' (line 267)
        self_323991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 9), 'self')
        # Setting the type of the member '_stdhour' of a type (line 267)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 9), self_323991, '_stdhour', tuple_var_assignment_323266_323990)
        
        # Assigning a Name to a Attribute (line 265):
        # Getting the type of 'tuple_var_assignment_323267' (line 265)
        tuple_var_assignment_323267_323992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'tuple_var_assignment_323267')
        # Getting the type of 'self' (line 268)
        self_323993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 9), 'self')
        # Setting the type of the member '_stdminute' of a type (line 268)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 9), self_323993, '_stdminute', tuple_var_assignment_323267_323992)
        
        # Assigning a Subscript to a Attribute (line 270):
        
        # Assigning a Subscript to a Attribute (line 270):
        
        # Obtaining the type of the subscript
        int_323994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 33), 'int')
        # Getting the type of 'tup' (line 270)
        tup_323995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 29), 'tup')
        # Obtaining the member '__getitem__' of a type (line 270)
        getitem___323996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 29), tup_323995, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 270)
        subscript_call_result_323997 = invoke(stypy.reporting.localization.Localization(__file__, 270, 29), getitem___323996, int_323994)
        
        # Getting the type of 'self' (line 270)
        self_323998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'self')
        # Setting the type of the member '_stddayofweek' of a type (line 270)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 8), self_323998, '_stddayofweek', subscript_call_result_323997)
        
        # Assigning a Call to a Name (line 272):
        
        # Assigning a Call to a Name (line 272):
        
        # Call to unpack(...): (line 272)
        # Processing the call arguments (line 272)
        str_324001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 28), 'str', '=8h')
        
        # Obtaining the type of the subscript
        str_324002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 43), 'str', 'DaylightStart')
        # Getting the type of 'keydict' (line 272)
        keydict_324003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 35), 'keydict', False)
        # Obtaining the member '__getitem__' of a type (line 272)
        getitem___324004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 35), keydict_324003, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 272)
        subscript_call_result_324005 = invoke(stypy.reporting.localization.Localization(__file__, 272, 35), getitem___324004, str_324002)
        
        # Processing the call keyword arguments (line 272)
        kwargs_324006 = {}
        # Getting the type of 'struct' (line 272)
        struct_323999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 14), 'struct', False)
        # Obtaining the member 'unpack' of a type (line 272)
        unpack_324000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 14), struct_323999, 'unpack')
        # Calling unpack(args, kwargs) (line 272)
        unpack_call_result_324007 = invoke(stypy.reporting.localization.Localization(__file__, 272, 14), unpack_324000, *[str_324001, subscript_call_result_324005], **kwargs_324006)
        
        # Assigning a type to the variable 'tup' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'tup', unpack_call_result_324007)
        
        # Assigning a Subscript to a Tuple (line 274):
        
        # Assigning a Subscript to a Name (line 274):
        
        # Obtaining the type of the subscript
        int_324008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 8), 'int')
        
        # Obtaining the type of the subscript
        int_324009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 32), 'int')
        int_324010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 34), 'int')
        slice_324011 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 277, 28), int_324009, int_324010, None)
        # Getting the type of 'tup' (line 277)
        tup_324012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 28), 'tup')
        # Obtaining the member '__getitem__' of a type (line 277)
        getitem___324013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 28), tup_324012, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 277)
        subscript_call_result_324014 = invoke(stypy.reporting.localization.Localization(__file__, 277, 28), getitem___324013, slice_324011)
        
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___324015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), subscript_call_result_324014, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 274)
        subscript_call_result_324016 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), getitem___324015, int_324008)
        
        # Assigning a type to the variable 'tuple_var_assignment_323268' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'tuple_var_assignment_323268', subscript_call_result_324016)
        
        # Assigning a Subscript to a Name (line 274):
        
        # Obtaining the type of the subscript
        int_324017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 8), 'int')
        
        # Obtaining the type of the subscript
        int_324018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 32), 'int')
        int_324019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 34), 'int')
        slice_324020 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 277, 28), int_324018, int_324019, None)
        # Getting the type of 'tup' (line 277)
        tup_324021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 28), 'tup')
        # Obtaining the member '__getitem__' of a type (line 277)
        getitem___324022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 28), tup_324021, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 277)
        subscript_call_result_324023 = invoke(stypy.reporting.localization.Localization(__file__, 277, 28), getitem___324022, slice_324020)
        
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___324024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), subscript_call_result_324023, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 274)
        subscript_call_result_324025 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), getitem___324024, int_324017)
        
        # Assigning a type to the variable 'tuple_var_assignment_323269' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'tuple_var_assignment_323269', subscript_call_result_324025)
        
        # Assigning a Subscript to a Name (line 274):
        
        # Obtaining the type of the subscript
        int_324026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 8), 'int')
        
        # Obtaining the type of the subscript
        int_324027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 32), 'int')
        int_324028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 34), 'int')
        slice_324029 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 277, 28), int_324027, int_324028, None)
        # Getting the type of 'tup' (line 277)
        tup_324030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 28), 'tup')
        # Obtaining the member '__getitem__' of a type (line 277)
        getitem___324031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 28), tup_324030, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 277)
        subscript_call_result_324032 = invoke(stypy.reporting.localization.Localization(__file__, 277, 28), getitem___324031, slice_324029)
        
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___324033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), subscript_call_result_324032, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 274)
        subscript_call_result_324034 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), getitem___324033, int_324026)
        
        # Assigning a type to the variable 'tuple_var_assignment_323270' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'tuple_var_assignment_323270', subscript_call_result_324034)
        
        # Assigning a Subscript to a Name (line 274):
        
        # Obtaining the type of the subscript
        int_324035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 8), 'int')
        
        # Obtaining the type of the subscript
        int_324036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 32), 'int')
        int_324037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 34), 'int')
        slice_324038 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 277, 28), int_324036, int_324037, None)
        # Getting the type of 'tup' (line 277)
        tup_324039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 28), 'tup')
        # Obtaining the member '__getitem__' of a type (line 277)
        getitem___324040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 28), tup_324039, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 277)
        subscript_call_result_324041 = invoke(stypy.reporting.localization.Localization(__file__, 277, 28), getitem___324040, slice_324038)
        
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___324042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), subscript_call_result_324041, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 274)
        subscript_call_result_324043 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), getitem___324042, int_324035)
        
        # Assigning a type to the variable 'tuple_var_assignment_323271' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'tuple_var_assignment_323271', subscript_call_result_324043)
        
        # Assigning a Name to a Attribute (line 274):
        # Getting the type of 'tuple_var_assignment_323268' (line 274)
        tuple_var_assignment_323268_324044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'tuple_var_assignment_323268')
        # Getting the type of 'self' (line 274)
        self_324045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 9), 'self')
        # Setting the type of the member '_dstmonth' of a type (line 274)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 9), self_324045, '_dstmonth', tuple_var_assignment_323268_324044)
        
        # Assigning a Name to a Attribute (line 274):
        # Getting the type of 'tuple_var_assignment_323269' (line 274)
        tuple_var_assignment_323269_324046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'tuple_var_assignment_323269')
        # Getting the type of 'self' (line 275)
        self_324047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 9), 'self')
        # Setting the type of the member '_dstweeknumber' of a type (line 275)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 9), self_324047, '_dstweeknumber', tuple_var_assignment_323269_324046)
        
        # Assigning a Name to a Attribute (line 274):
        # Getting the type of 'tuple_var_assignment_323270' (line 274)
        tuple_var_assignment_323270_324048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'tuple_var_assignment_323270')
        # Getting the type of 'self' (line 276)
        self_324049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 9), 'self')
        # Setting the type of the member '_dsthour' of a type (line 276)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 9), self_324049, '_dsthour', tuple_var_assignment_323270_324048)
        
        # Assigning a Name to a Attribute (line 274):
        # Getting the type of 'tuple_var_assignment_323271' (line 274)
        tuple_var_assignment_323271_324050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'tuple_var_assignment_323271')
        # Getting the type of 'self' (line 277)
        self_324051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 9), 'self')
        # Setting the type of the member '_dstminute' of a type (line 277)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 9), self_324051, '_dstminute', tuple_var_assignment_323271_324050)
        
        # Assigning a Subscript to a Attribute (line 279):
        
        # Assigning a Subscript to a Attribute (line 279):
        
        # Obtaining the type of the subscript
        int_324052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 33), 'int')
        # Getting the type of 'tup' (line 279)
        tup_324053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 29), 'tup')
        # Obtaining the member '__getitem__' of a type (line 279)
        getitem___324054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 29), tup_324053, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 279)
        subscript_call_result_324055 = invoke(stypy.reporting.localization.Localization(__file__, 279, 29), getitem___324054, int_324052)
        
        # Getting the type of 'self' (line 279)
        self_324056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'self')
        # Setting the type of the member '_dstdayofweek' of a type (line 279)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 8), self_324056, '_dstdayofweek', subscript_call_result_324055)
        
        # Assigning a BinOp to a Attribute (line 281):
        
        # Assigning a BinOp to a Attribute (line 281):
        # Getting the type of 'self' (line 281)
        self_324057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 33), 'self')
        # Obtaining the member '_dst_offset' of a type (line 281)
        _dst_offset_324058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 33), self_324057, '_dst_offset')
        # Getting the type of 'self' (line 281)
        self_324059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 52), 'self')
        # Obtaining the member '_std_offset' of a type (line 281)
        _std_offset_324060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 52), self_324059, '_std_offset')
        # Applying the binary operator '-' (line 281)
        result_sub_324061 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 33), '-', _dst_offset_324058, _std_offset_324060)
        
        # Getting the type of 'self' (line 281)
        self_324062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'self')
        # Setting the type of the member '_dst_base_offset_' of a type (line 281)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 8), self_324062, '_dst_base_offset_', result_sub_324061)
        
        # Assigning a Call to a Attribute (line 282):
        
        # Assigning a Call to a Attribute (line 282):
        
        # Call to _get_hasdst(...): (line 282)
        # Processing the call keyword arguments (line 282)
        kwargs_324065 = {}
        # Getting the type of 'self' (line 282)
        self_324063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 22), 'self', False)
        # Obtaining the member '_get_hasdst' of a type (line 282)
        _get_hasdst_324064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 22), self_324063, '_get_hasdst')
        # Calling _get_hasdst(args, kwargs) (line 282)
        _get_hasdst_call_result_324066 = invoke(stypy.reporting.localization.Localization(__file__, 282, 22), _get_hasdst_324064, *[], **kwargs_324065)
        
        # Getting the type of 'self' (line 282)
        self_324067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'self')
        # Setting the type of the member 'hasdst' of a type (line 282)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 8), self_324067, 'hasdst', _get_hasdst_call_result_324066)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 284, 4, False)
        # Assigning a type to the variable 'self' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzwinlocal.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        tzwinlocal.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzwinlocal.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzwinlocal.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'tzwinlocal.stypy__repr__')
        tzwinlocal.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        tzwinlocal.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzwinlocal.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzwinlocal.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzwinlocal.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzwinlocal.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzwinlocal.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzwinlocal.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        str_324068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 15), 'str', 'tzwinlocal()')
        # Assigning a type to the variable 'stypy_return_type' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'stypy_return_type', str_324068)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 284)
        stypy_return_type_324069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324069)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_324069


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 287, 4, False)
        # Assigning a type to the variable 'self' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzwinlocal.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        tzwinlocal.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzwinlocal.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzwinlocal.stypy__str__.__dict__.__setitem__('stypy_function_name', 'tzwinlocal.stypy__str__')
        tzwinlocal.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        tzwinlocal.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzwinlocal.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzwinlocal.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzwinlocal.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzwinlocal.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzwinlocal.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzwinlocal.stypy__str__', [], None, None, defaults, varargs, kwargs)

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

        str_324070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 15), 'str', 'tzwinlocal(%s)')
        
        # Call to repr(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'self' (line 289)
        self_324072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 39), 'self', False)
        # Obtaining the member '_std_abbr' of a type (line 289)
        _std_abbr_324073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 39), self_324072, '_std_abbr')
        # Processing the call keyword arguments (line 289)
        kwargs_324074 = {}
        # Getting the type of 'repr' (line 289)
        repr_324071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 34), 'repr', False)
        # Calling repr(args, kwargs) (line 289)
        repr_call_result_324075 = invoke(stypy.reporting.localization.Localization(__file__, 289, 34), repr_324071, *[_std_abbr_324073], **kwargs_324074)
        
        # Applying the binary operator '%' (line 289)
        result_mod_324076 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 15), '%', str_324070, repr_call_result_324075)
        
        # Assigning a type to the variable 'stypy_return_type' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'stypy_return_type', result_mod_324076)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 287)
        stypy_return_type_324077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324077)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_324077


    @norecursion
    def __reduce__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__reduce__'
        module_type_store = module_type_store.open_function_context('__reduce__', 291, 4, False)
        # Assigning a type to the variable 'self' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzwinlocal.__reduce__.__dict__.__setitem__('stypy_localization', localization)
        tzwinlocal.__reduce__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzwinlocal.__reduce__.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzwinlocal.__reduce__.__dict__.__setitem__('stypy_function_name', 'tzwinlocal.__reduce__')
        tzwinlocal.__reduce__.__dict__.__setitem__('stypy_param_names_list', [])
        tzwinlocal.__reduce__.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzwinlocal.__reduce__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzwinlocal.__reduce__.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzwinlocal.__reduce__.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzwinlocal.__reduce__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzwinlocal.__reduce__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzwinlocal.__reduce__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__reduce__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__reduce__(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 292)
        tuple_324078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 292)
        # Adding element type (line 292)
        # Getting the type of 'self' (line 292)
        self_324079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 16), 'self')
        # Obtaining the member '__class__' of a type (line 292)
        class___324080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 16), self_324079, '__class__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 16), tuple_324078, class___324080)
        # Adding element type (line 292)
        
        # Obtaining an instance of the builtin type 'tuple' (line 292)
        tuple_324081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 292)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 16), tuple_324078, tuple_324081)
        
        # Assigning a type to the variable 'stypy_return_type' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'stypy_return_type', tuple_324078)
        
        # ################# End of '__reduce__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__reduce__' in the type store
        # Getting the type of 'stypy_return_type' (line 291)
        stypy_return_type_324082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324082)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__reduce__'
        return stypy_return_type_324082


# Assigning a type to the variable 'tzwinlocal' (line 237)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 0), 'tzwinlocal', tzwinlocal)

@norecursion
def picknthweekday(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'picknthweekday'
    module_type_store = module_type_store.open_function_context('picknthweekday', 295, 0, False)
    
    # Passed parameters checking function
    picknthweekday.stypy_localization = localization
    picknthweekday.stypy_type_of_self = None
    picknthweekday.stypy_type_store = module_type_store
    picknthweekday.stypy_function_name = 'picknthweekday'
    picknthweekday.stypy_param_names_list = ['year', 'month', 'dayofweek', 'hour', 'minute', 'whichweek']
    picknthweekday.stypy_varargs_param_name = None
    picknthweekday.stypy_kwargs_param_name = None
    picknthweekday.stypy_call_defaults = defaults
    picknthweekday.stypy_call_varargs = varargs
    picknthweekday.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'picknthweekday', ['year', 'month', 'dayofweek', 'hour', 'minute', 'whichweek'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'picknthweekday', localization, ['year', 'month', 'dayofweek', 'hour', 'minute', 'whichweek'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'picknthweekday(...)' code ##################

    str_324083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 4), 'str', ' dayofweek == 0 means Sunday, whichweek 5 means last instance ')
    
    # Assigning a Call to a Name (line 297):
    
    # Assigning a Call to a Name (line 297):
    
    # Call to datetime(...): (line 297)
    # Processing the call arguments (line 297)
    # Getting the type of 'year' (line 297)
    year_324086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 30), 'year', False)
    # Getting the type of 'month' (line 297)
    month_324087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 36), 'month', False)
    int_324088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 43), 'int')
    # Getting the type of 'hour' (line 297)
    hour_324089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 46), 'hour', False)
    # Getting the type of 'minute' (line 297)
    minute_324090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 52), 'minute', False)
    # Processing the call keyword arguments (line 297)
    kwargs_324091 = {}
    # Getting the type of 'datetime' (line 297)
    datetime_324084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'datetime', False)
    # Obtaining the member 'datetime' of a type (line 297)
    datetime_324085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 12), datetime_324084, 'datetime')
    # Calling datetime(args, kwargs) (line 297)
    datetime_call_result_324092 = invoke(stypy.reporting.localization.Localization(__file__, 297, 12), datetime_324085, *[year_324086, month_324087, int_324088, hour_324089, minute_324090], **kwargs_324091)
    
    # Assigning a type to the variable 'first' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'first', datetime_call_result_324092)
    
    # Assigning a Call to a Name (line 301):
    
    # Assigning a Call to a Name (line 301):
    
    # Call to replace(...): (line 301)
    # Processing the call keyword arguments (line 301)
    # Getting the type of 'dayofweek' (line 301)
    dayofweek_324095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 37), 'dayofweek', False)
    
    # Call to isoweekday(...): (line 301)
    # Processing the call keyword arguments (line 301)
    kwargs_324098 = {}
    # Getting the type of 'first' (line 301)
    first_324096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 49), 'first', False)
    # Obtaining the member 'isoweekday' of a type (line 301)
    isoweekday_324097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 49), first_324096, 'isoweekday')
    # Calling isoweekday(args, kwargs) (line 301)
    isoweekday_call_result_324099 = invoke(stypy.reporting.localization.Localization(__file__, 301, 49), isoweekday_324097, *[], **kwargs_324098)
    
    # Applying the binary operator '-' (line 301)
    result_sub_324100 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 37), '-', dayofweek_324095, isoweekday_call_result_324099)
    
    int_324101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 71), 'int')
    # Applying the binary operator '%' (line 301)
    result_mod_324102 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 36), '%', result_sub_324100, int_324101)
    
    int_324103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 76), 'int')
    # Applying the binary operator '+' (line 301)
    result_add_324104 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 35), '+', result_mod_324102, int_324103)
    
    keyword_324105 = result_add_324104
    kwargs_324106 = {'day': keyword_324105}
    # Getting the type of 'first' (line 301)
    first_324093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 17), 'first', False)
    # Obtaining the member 'replace' of a type (line 301)
    replace_324094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 17), first_324093, 'replace')
    # Calling replace(args, kwargs) (line 301)
    replace_call_result_324107 = invoke(stypy.reporting.localization.Localization(__file__, 301, 17), replace_324094, *[], **kwargs_324106)
    
    # Assigning a type to the variable 'weekdayone' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'weekdayone', replace_call_result_324107)
    
    # Assigning a BinOp to a Name (line 302):
    
    # Assigning a BinOp to a Name (line 302):
    # Getting the type of 'weekdayone' (line 302)
    weekdayone_324108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 9), 'weekdayone')
    # Getting the type of 'whichweek' (line 302)
    whichweek_324109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 24), 'whichweek')
    int_324110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 36), 'int')
    # Applying the binary operator '-' (line 302)
    result_sub_324111 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 24), '-', whichweek_324109, int_324110)
    
    # Getting the type of 'ONEWEEK' (line 302)
    ONEWEEK_324112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 41), 'ONEWEEK')
    # Applying the binary operator '*' (line 302)
    result_mul_324113 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 23), '*', result_sub_324111, ONEWEEK_324112)
    
    # Applying the binary operator '+' (line 302)
    result_add_324114 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 9), '+', weekdayone_324108, result_mul_324113)
    
    # Assigning a type to the variable 'wd' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'wd', result_add_324114)
    
    
    # Getting the type of 'wd' (line 303)
    wd_324115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'wd')
    # Obtaining the member 'month' of a type (line 303)
    month_324116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 8), wd_324115, 'month')
    # Getting the type of 'month' (line 303)
    month_324117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 20), 'month')
    # Applying the binary operator '!=' (line 303)
    result_ne_324118 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 8), '!=', month_324116, month_324117)
    
    # Testing the type of an if condition (line 303)
    if_condition_324119 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 303, 4), result_ne_324118)
    # Assigning a type to the variable 'if_condition_324119' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'if_condition_324119', if_condition_324119)
    # SSA begins for if statement (line 303)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'wd' (line 304)
    wd_324120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'wd')
    # Getting the type of 'ONEWEEK' (line 304)
    ONEWEEK_324121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 14), 'ONEWEEK')
    # Applying the binary operator '-=' (line 304)
    result_isub_324122 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 8), '-=', wd_324120, ONEWEEK_324121)
    # Assigning a type to the variable 'wd' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'wd', result_isub_324122)
    
    # SSA join for if statement (line 303)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'wd' (line 306)
    wd_324123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 11), 'wd')
    # Assigning a type to the variable 'stypy_return_type' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'stypy_return_type', wd_324123)
    
    # ################# End of 'picknthweekday(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'picknthweekday' in the type store
    # Getting the type of 'stypy_return_type' (line 295)
    stypy_return_type_324124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_324124)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'picknthweekday'
    return stypy_return_type_324124

# Assigning a type to the variable 'picknthweekday' (line 295)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 0), 'picknthweekday', picknthweekday)

@norecursion
def valuestodict(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'valuestodict'
    module_type_store = module_type_store.open_function_context('valuestodict', 309, 0, False)
    
    # Passed parameters checking function
    valuestodict.stypy_localization = localization
    valuestodict.stypy_type_of_self = None
    valuestodict.stypy_type_store = module_type_store
    valuestodict.stypy_function_name = 'valuestodict'
    valuestodict.stypy_param_names_list = ['key']
    valuestodict.stypy_varargs_param_name = None
    valuestodict.stypy_kwargs_param_name = None
    valuestodict.stypy_call_defaults = defaults
    valuestodict.stypy_call_varargs = varargs
    valuestodict.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'valuestodict', ['key'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'valuestodict', localization, ['key'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'valuestodict(...)' code ##################

    str_324125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 4), 'str', "Convert a registry key's values to a dictionary.")
    
    # Assigning a Dict to a Name (line 311):
    
    # Assigning a Dict to a Name (line 311):
    
    # Obtaining an instance of the builtin type 'dict' (line 311)
    dict_324126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 11), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 311)
    
    # Assigning a type to the variable 'dout' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'dout', dict_324126)
    
    # Assigning a Subscript to a Name (line 312):
    
    # Assigning a Subscript to a Name (line 312):
    
    # Obtaining the type of the subscript
    int_324127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 36), 'int')
    
    # Call to QueryInfoKey(...): (line 312)
    # Processing the call arguments (line 312)
    # Getting the type of 'key' (line 312)
    key_324130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 31), 'key', False)
    # Processing the call keyword arguments (line 312)
    kwargs_324131 = {}
    # Getting the type of 'winreg' (line 312)
    winreg_324128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 11), 'winreg', False)
    # Obtaining the member 'QueryInfoKey' of a type (line 312)
    QueryInfoKey_324129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 11), winreg_324128, 'QueryInfoKey')
    # Calling QueryInfoKey(args, kwargs) (line 312)
    QueryInfoKey_call_result_324132 = invoke(stypy.reporting.localization.Localization(__file__, 312, 11), QueryInfoKey_324129, *[key_324130], **kwargs_324131)
    
    # Obtaining the member '__getitem__' of a type (line 312)
    getitem___324133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 11), QueryInfoKey_call_result_324132, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 312)
    subscript_call_result_324134 = invoke(stypy.reporting.localization.Localization(__file__, 312, 11), getitem___324133, int_324127)
    
    # Assigning a type to the variable 'size' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'size', subscript_call_result_324134)
    
    # Assigning a Name to a Name (line 313):
    
    # Assigning a Name to a Name (line 313):
    # Getting the type of 'None' (line 313)
    None_324135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 13), 'None')
    # Assigning a type to the variable 'tz_res' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'tz_res', None_324135)
    
    
    # Call to range(...): (line 315)
    # Processing the call arguments (line 315)
    # Getting the type of 'size' (line 315)
    size_324137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 19), 'size', False)
    # Processing the call keyword arguments (line 315)
    kwargs_324138 = {}
    # Getting the type of 'range' (line 315)
    range_324136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 13), 'range', False)
    # Calling range(args, kwargs) (line 315)
    range_call_result_324139 = invoke(stypy.reporting.localization.Localization(__file__, 315, 13), range_324136, *[size_324137], **kwargs_324138)
    
    # Testing the type of a for loop iterable (line 315)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 315, 4), range_call_result_324139)
    # Getting the type of the for loop variable (line 315)
    for_loop_var_324140 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 315, 4), range_call_result_324139)
    # Assigning a type to the variable 'i' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'i', for_loop_var_324140)
    # SSA begins for a for statement (line 315)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 316):
    
    # Assigning a Call to a Name:
    
    # Call to EnumValue(...): (line 316)
    # Processing the call arguments (line 316)
    # Getting the type of 'key' (line 316)
    key_324143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 50), 'key', False)
    # Getting the type of 'i' (line 316)
    i_324144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 55), 'i', False)
    # Processing the call keyword arguments (line 316)
    kwargs_324145 = {}
    # Getting the type of 'winreg' (line 316)
    winreg_324141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 33), 'winreg', False)
    # Obtaining the member 'EnumValue' of a type (line 316)
    EnumValue_324142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 33), winreg_324141, 'EnumValue')
    # Calling EnumValue(args, kwargs) (line 316)
    EnumValue_call_result_324146 = invoke(stypy.reporting.localization.Localization(__file__, 316, 33), EnumValue_324142, *[key_324143, i_324144], **kwargs_324145)
    
    # Assigning a type to the variable 'call_assignment_323272' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'call_assignment_323272', EnumValue_call_result_324146)
    
    # Assigning a Call to a Name (line 316):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_324149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 8), 'int')
    # Processing the call keyword arguments
    kwargs_324150 = {}
    # Getting the type of 'call_assignment_323272' (line 316)
    call_assignment_323272_324147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'call_assignment_323272', False)
    # Obtaining the member '__getitem__' of a type (line 316)
    getitem___324148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), call_assignment_323272_324147, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_324151 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___324148, *[int_324149], **kwargs_324150)
    
    # Assigning a type to the variable 'call_assignment_323273' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'call_assignment_323273', getitem___call_result_324151)
    
    # Assigning a Name to a Name (line 316):
    # Getting the type of 'call_assignment_323273' (line 316)
    call_assignment_323273_324152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'call_assignment_323273')
    # Assigning a type to the variable 'key_name' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'key_name', call_assignment_323273_324152)
    
    # Assigning a Call to a Name (line 316):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_324155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 8), 'int')
    # Processing the call keyword arguments
    kwargs_324156 = {}
    # Getting the type of 'call_assignment_323272' (line 316)
    call_assignment_323272_324153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'call_assignment_323272', False)
    # Obtaining the member '__getitem__' of a type (line 316)
    getitem___324154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), call_assignment_323272_324153, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_324157 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___324154, *[int_324155], **kwargs_324156)
    
    # Assigning a type to the variable 'call_assignment_323274' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'call_assignment_323274', getitem___call_result_324157)
    
    # Assigning a Name to a Name (line 316):
    # Getting the type of 'call_assignment_323274' (line 316)
    call_assignment_323274_324158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'call_assignment_323274')
    # Assigning a type to the variable 'value' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 18), 'value', call_assignment_323274_324158)
    
    # Assigning a Call to a Name (line 316):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_324161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 8), 'int')
    # Processing the call keyword arguments
    kwargs_324162 = {}
    # Getting the type of 'call_assignment_323272' (line 316)
    call_assignment_323272_324159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'call_assignment_323272', False)
    # Obtaining the member '__getitem__' of a type (line 316)
    getitem___324160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), call_assignment_323272_324159, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_324163 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___324160, *[int_324161], **kwargs_324162)
    
    # Assigning a type to the variable 'call_assignment_323275' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'call_assignment_323275', getitem___call_result_324163)
    
    # Assigning a Name to a Name (line 316):
    # Getting the type of 'call_assignment_323275' (line 316)
    call_assignment_323275_324164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'call_assignment_323275')
    # Assigning a type to the variable 'dtype' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 25), 'dtype', call_assignment_323275_324164)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'dtype' (line 317)
    dtype_324165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 11), 'dtype')
    # Getting the type of 'winreg' (line 317)
    winreg_324166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 20), 'winreg')
    # Obtaining the member 'REG_DWORD' of a type (line 317)
    REG_DWORD_324167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 20), winreg_324166, 'REG_DWORD')
    # Applying the binary operator '==' (line 317)
    result_eq_324168 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 11), '==', dtype_324165, REG_DWORD_324167)
    
    
    # Getting the type of 'dtype' (line 317)
    dtype_324169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 40), 'dtype')
    # Getting the type of 'winreg' (line 317)
    winreg_324170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 49), 'winreg')
    # Obtaining the member 'REG_DWORD_LITTLE_ENDIAN' of a type (line 317)
    REG_DWORD_LITTLE_ENDIAN_324171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 49), winreg_324170, 'REG_DWORD_LITTLE_ENDIAN')
    # Applying the binary operator '==' (line 317)
    result_eq_324172 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 40), '==', dtype_324169, REG_DWORD_LITTLE_ENDIAN_324171)
    
    # Applying the binary operator 'or' (line 317)
    result_or_keyword_324173 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 11), 'or', result_eq_324168, result_eq_324172)
    
    # Testing the type of an if condition (line 317)
    if_condition_324174 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 317, 8), result_or_keyword_324173)
    # Assigning a type to the variable 'if_condition_324174' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'if_condition_324174', if_condition_324174)
    # SSA begins for if statement (line 317)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'value' (line 320)
    value_324175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 15), 'value')
    int_324176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 24), 'int')
    int_324177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 29), 'int')
    # Applying the binary operator '<<' (line 320)
    result_lshift_324178 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 24), '<<', int_324176, int_324177)
    
    # Applying the binary operator '&' (line 320)
    result_and__324179 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 15), '&', value_324175, result_lshift_324178)
    
    # Testing the type of an if condition (line 320)
    if_condition_324180 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 320, 12), result_and__324179)
    # Assigning a type to the variable 'if_condition_324180' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'if_condition_324180', if_condition_324180)
    # SSA begins for if statement (line 320)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 321):
    
    # Assigning a BinOp to a Name (line 321):
    # Getting the type of 'value' (line 321)
    value_324181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 24), 'value')
    int_324182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 33), 'int')
    int_324183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 38), 'int')
    # Applying the binary operator '<<' (line 321)
    result_lshift_324184 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 33), '<<', int_324182, int_324183)
    
    # Applying the binary operator '-' (line 321)
    result_sub_324185 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 24), '-', value_324181, result_lshift_324184)
    
    # Assigning a type to the variable 'value' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 16), 'value', result_sub_324185)
    # SSA join for if statement (line 320)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 317)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'dtype' (line 322)
    dtype_324186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 13), 'dtype')
    # Getting the type of 'winreg' (line 322)
    winreg_324187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 22), 'winreg')
    # Obtaining the member 'REG_SZ' of a type (line 322)
    REG_SZ_324188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 22), winreg_324187, 'REG_SZ')
    # Applying the binary operator '==' (line 322)
    result_eq_324189 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 13), '==', dtype_324186, REG_SZ_324188)
    
    # Testing the type of an if condition (line 322)
    if_condition_324190 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 13), result_eq_324189)
    # Assigning a type to the variable 'if_condition_324190' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 13), 'if_condition_324190', if_condition_324190)
    # SSA begins for if statement (line 322)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to startswith(...): (line 324)
    # Processing the call arguments (line 324)
    str_324193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 32), 'str', '@tzres')
    # Processing the call keyword arguments (line 324)
    kwargs_324194 = {}
    # Getting the type of 'value' (line 324)
    value_324191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 15), 'value', False)
    # Obtaining the member 'startswith' of a type (line 324)
    startswith_324192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 15), value_324191, 'startswith')
    # Calling startswith(args, kwargs) (line 324)
    startswith_call_result_324195 = invoke(stypy.reporting.localization.Localization(__file__, 324, 15), startswith_324192, *[str_324193], **kwargs_324194)
    
    # Testing the type of an if condition (line 324)
    if_condition_324196 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 324, 12), startswith_call_result_324195)
    # Assigning a type to the variable 'if_condition_324196' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'if_condition_324196', if_condition_324196)
    # SSA begins for if statement (line 324)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BoolOp to a Name (line 325):
    
    # Assigning a BoolOp to a Name (line 325):
    
    # Evaluating a boolean operation
    # Getting the type of 'tz_res' (line 325)
    tz_res_324197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 25), 'tz_res')
    
    # Call to tzres(...): (line 325)
    # Processing the call keyword arguments (line 325)
    kwargs_324199 = {}
    # Getting the type of 'tzres' (line 325)
    tzres_324198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 35), 'tzres', False)
    # Calling tzres(args, kwargs) (line 325)
    tzres_call_result_324200 = invoke(stypy.reporting.localization.Localization(__file__, 325, 35), tzres_324198, *[], **kwargs_324199)
    
    # Applying the binary operator 'or' (line 325)
    result_or_keyword_324201 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 25), 'or', tz_res_324197, tzres_call_result_324200)
    
    # Assigning a type to the variable 'tz_res' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 16), 'tz_res', result_or_keyword_324201)
    
    # Assigning a Call to a Name (line 326):
    
    # Assigning a Call to a Name (line 326):
    
    # Call to name_from_string(...): (line 326)
    # Processing the call arguments (line 326)
    # Getting the type of 'value' (line 326)
    value_324204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 48), 'value', False)
    # Processing the call keyword arguments (line 326)
    kwargs_324205 = {}
    # Getting the type of 'tz_res' (line 326)
    tz_res_324202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 24), 'tz_res', False)
    # Obtaining the member 'name_from_string' of a type (line 326)
    name_from_string_324203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 24), tz_res_324202, 'name_from_string')
    # Calling name_from_string(args, kwargs) (line 326)
    name_from_string_call_result_324206 = invoke(stypy.reporting.localization.Localization(__file__, 326, 24), name_from_string_324203, *[value_324204], **kwargs_324205)
    
    # Assigning a type to the variable 'value' (line 326)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 16), 'value', name_from_string_call_result_324206)
    # SSA join for if statement (line 324)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 328):
    
    # Assigning a Call to a Name (line 328):
    
    # Call to rstrip(...): (line 328)
    # Processing the call arguments (line 328)
    str_324209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 33), 'str', '\x00')
    # Processing the call keyword arguments (line 328)
    kwargs_324210 = {}
    # Getting the type of 'value' (line 328)
    value_324207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 20), 'value', False)
    # Obtaining the member 'rstrip' of a type (line 328)
    rstrip_324208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 20), value_324207, 'rstrip')
    # Calling rstrip(args, kwargs) (line 328)
    rstrip_call_result_324211 = invoke(stypy.reporting.localization.Localization(__file__, 328, 20), rstrip_324208, *[str_324209], **kwargs_324210)
    
    # Assigning a type to the variable 'value' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'value', rstrip_call_result_324211)
    # SSA join for if statement (line 322)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 317)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 330):
    
    # Assigning a Name to a Subscript (line 330):
    # Getting the type of 'value' (line 330)
    value_324212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 25), 'value')
    # Getting the type of 'dout' (line 330)
    dout_324213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'dout')
    # Getting the type of 'key_name' (line 330)
    key_name_324214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 13), 'key_name')
    # Storing an element on a container (line 330)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 8), dout_324213, (key_name_324214, value_324212))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'dout' (line 332)
    dout_324215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 11), 'dout')
    # Assigning a type to the variable 'stypy_return_type' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'stypy_return_type', dout_324215)
    
    # ################# End of 'valuestodict(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'valuestodict' in the type store
    # Getting the type of 'stypy_return_type' (line 309)
    stypy_return_type_324216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_324216)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'valuestodict'
    return stypy_return_type_324216

# Assigning a type to the variable 'valuestodict' (line 309)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 0), 'valuestodict', valuestodict)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
