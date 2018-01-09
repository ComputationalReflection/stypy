
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # -*- coding: utf-8 -*-
2: import warnings
3: import json
4: 
5: from tarfile import TarFile
6: from pkgutil import get_data
7: from io import BytesIO
8: from contextlib import closing
9: 
10: from dateutil.tz import tzfile
11: 
12: __all__ = ["get_zonefile_instance", "gettz", "gettz_db_metadata", "rebuild"]
13: 
14: ZONEFILENAME = "dateutil-zoneinfo.tar.gz"
15: METADATA_FN = 'METADATA'
16: 
17: # python2.6 compatability. Note that TarFile.__exit__ != TarFile.close, but
18: # it's close enough for python2.6
19: tar_open = TarFile.open
20: if not hasattr(TarFile, '__exit__'):
21:     def tar_open(*args, **kwargs):
22:         return closing(TarFile.open(*args, **kwargs))
23: 
24: 
25: class tzfile(tzfile):
26:     def __reduce__(self):
27:         return (gettz, (self._filename,))
28: 
29: 
30: def getzoneinfofile_stream():
31:     try:
32:         return BytesIO(get_data(__name__, ZONEFILENAME))
33:     except IOError as e:  # TODO  switch to FileNotFoundError?
34:         warnings.warn("I/O error({0}): {1}".format(e.errno, e.strerror))
35:         return None
36: 
37: 
38: class ZoneInfoFile(object):
39:     def __init__(self, zonefile_stream=None):
40:         if zonefile_stream is not None:
41:             with tar_open(fileobj=zonefile_stream, mode='r') as tf:
42:                 # dict comprehension does not work on python2.6
43:                 # TODO: get back to the nicer syntax when we ditch python2.6
44:                 # self.zones = {zf.name: tzfile(tf.extractfile(zf),
45:                 #               filename = zf.name)
46:                 #              for zf in tf.getmembers() if zf.isfile()}
47:                 self.zones = dict((zf.name, tzfile(tf.extractfile(zf),
48:                                                    filename=zf.name))
49:                                   for zf in tf.getmembers()
50:                                   if zf.isfile() and zf.name != METADATA_FN)
51:                 # deal with links: They'll point to their parent object. Less
52:                 # waste of memory
53:                 # links = {zl.name: self.zones[zl.linkname]
54:                 #        for zl in tf.getmembers() if zl.islnk() or zl.issym()}
55:                 links = dict((zl.name, self.zones[zl.linkname])
56:                              for zl in tf.getmembers() if
57:                              zl.islnk() or zl.issym())
58:                 self.zones.update(links)
59:                 try:
60:                     metadata_json = tf.extractfile(tf.getmember(METADATA_FN))
61:                     metadata_str = metadata_json.read().decode('UTF-8')
62:                     self.metadata = json.loads(metadata_str)
63:                 except KeyError:
64:                     # no metadata in tar file
65:                     self.metadata = None
66:         else:
67:             self.zones = dict()
68:             self.metadata = None
69: 
70:     def get(self, name, default=None):
71:         '''
72:         Wrapper for :func:`ZoneInfoFile.zones.get`. This is a convenience method
73:         for retrieving zones from the zone dictionary.
74: 
75:         :param name:
76:             The name of the zone to retrieve. (Generally IANA zone names)
77: 
78:         :param default:
79:             The value to return in the event of a missing key.
80: 
81:         .. versionadded:: 2.6.0
82: 
83:         '''
84:         return self.zones.get(name, default)
85: 
86: 
87: # The current API has gettz as a module function, although in fact it taps into
88: # a stateful class. So as a workaround for now, without changing the API, we
89: # will create a new "global" class instance the first time a user requests a
90: # timezone. Ugly, but adheres to the api.
91: #
92: # TODO: Remove after deprecation period.
93: _CLASS_ZONE_INSTANCE = list()
94: 
95: 
96: def get_zonefile_instance(new_instance=False):
97:     '''
98:     This is a convenience function which provides a :class:`ZoneInfoFile`
99:     instance using the data provided by the ``dateutil`` package. By default, it
100:     caches a single instance of the ZoneInfoFile object and returns that.
101: 
102:     :param new_instance:
103:         If ``True``, a new instance of :class:`ZoneInfoFile` is instantiated and
104:         used as the cached instance for the next call. Otherwise, new instances
105:         are created only as necessary.
106: 
107:     :return:
108:         Returns a :class:`ZoneInfoFile` object.
109: 
110:     .. versionadded:: 2.6
111:     '''
112:     if new_instance:
113:         zif = None
114:     else:
115:         zif = getattr(get_zonefile_instance, '_cached_instance', None)
116: 
117:     if zif is None:
118:         zif = ZoneInfoFile(getzoneinfofile_stream())
119: 
120:         get_zonefile_instance._cached_instance = zif
121: 
122:     return zif
123: 
124: 
125: def gettz(name):
126:     '''
127:     This retrieves a time zone from the local zoneinfo tarball that is packaged
128:     with dateutil.
129: 
130:     :param name:
131:         An IANA-style time zone name, as found in the zoneinfo file.
132: 
133:     :return:
134:         Returns a :class:`dateutil.tz.tzfile` time zone object.
135: 
136:     .. warning::
137:         It is generally inadvisable to use this function, and it is only
138:         provided for API compatibility with earlier versions. This is *not*
139:         equivalent to ``dateutil.tz.gettz()``, which selects an appropriate
140:         time zone based on the inputs, favoring system zoneinfo. This is ONLY
141:         for accessing the dateutil-specific zoneinfo (which may be out of
142:         date compared to the system zoneinfo).
143: 
144:     .. deprecated:: 2.6
145:         If you need to use a specific zoneinfofile over the system zoneinfo,
146:         instantiate a :class:`dateutil.zoneinfo.ZoneInfoFile` object and call
147:         :func:`dateutil.zoneinfo.ZoneInfoFile.get(name)` instead.
148: 
149:         Use :func:`get_zonefile_instance` to retrieve an instance of the
150:         dateutil-provided zoneinfo.
151:     '''
152:     warnings.warn("zoneinfo.gettz() will be removed in future versions, "
153:                   "to use the dateutil-provided zoneinfo files, instantiate a "
154:                   "ZoneInfoFile object and use ZoneInfoFile.zones.get() "
155:                   "instead. See the documentation for details.",
156:                   DeprecationWarning)
157: 
158:     if len(_CLASS_ZONE_INSTANCE) == 0:
159:         _CLASS_ZONE_INSTANCE.append(ZoneInfoFile(getzoneinfofile_stream()))
160:     return _CLASS_ZONE_INSTANCE[0].zones.get(name)
161: 
162: 
163: def gettz_db_metadata():
164:     ''' Get the zonefile metadata
165: 
166:     See `zonefile_metadata`_
167: 
168:     :returns:
169:         A dictionary with the database metadata
170: 
171:     .. deprecated:: 2.6
172:         See deprecation warning in :func:`zoneinfo.gettz`. To get metadata,
173:         query the attribute ``zoneinfo.ZoneInfoFile.metadata``.
174:     '''
175:     warnings.warn("zoneinfo.gettz_db_metadata() will be removed in future "
176:                   "versions, to use the dateutil-provided zoneinfo files, "
177:                   "ZoneInfoFile object and query the 'metadata' attribute "
178:                   "instead. See the documentation for details.",
179:                   DeprecationWarning)
180: 
181:     if len(_CLASS_ZONE_INSTANCE) == 0:
182:         _CLASS_ZONE_INSTANCE.append(ZoneInfoFile(getzoneinfofile_stream()))
183:     return _CLASS_ZONE_INSTANCE[0].metadata
184: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import warnings' statement (line 2)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import json' statement (line 3)
import json

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'json', json, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from tarfile import TarFile' statement (line 5)
try:
    from tarfile import TarFile

except:
    TarFile = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'tarfile', None, module_type_store, ['TarFile'], [TarFile])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from pkgutil import get_data' statement (line 6)
try:
    from pkgutil import get_data

except:
    get_data = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pkgutil', None, module_type_store, ['get_data'], [get_data])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from io import BytesIO' statement (line 7)
try:
    from io import BytesIO

except:
    BytesIO = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'io', None, module_type_store, ['BytesIO'], [BytesIO])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from contextlib import closing' statement (line 8)
try:
    from contextlib import closing

except:
    closing = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'contextlib', None, module_type_store, ['closing'], [closing])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from dateutil.tz import tzfile' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/dateutil/zoneinfo/')
import_324973 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'dateutil.tz')

if (type(import_324973) is not StypyTypeError):

    if (import_324973 != 'pyd_module'):
        __import__(import_324973)
        sys_modules_324974 = sys.modules[import_324973]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'dateutil.tz', sys_modules_324974.module_type_store, module_type_store, ['tzfile'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_324974, sys_modules_324974.module_type_store, module_type_store)
    else:
        from dateutil.tz import tzfile

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'dateutil.tz', None, module_type_store, ['tzfile'], [tzfile])

else:
    # Assigning a type to the variable 'dateutil.tz' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'dateutil.tz', import_324973)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/dateutil/zoneinfo/')


# Assigning a List to a Name (line 12):
__all__ = ['get_zonefile_instance', 'gettz', 'gettz_db_metadata', 'rebuild']
module_type_store.set_exportable_members(['get_zonefile_instance', 'gettz', 'gettz_db_metadata', 'rebuild'])

# Obtaining an instance of the builtin type 'list' (line 12)
list_324975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
str_324976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 11), 'str', 'get_zonefile_instance')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_324975, str_324976)
# Adding element type (line 12)
str_324977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 36), 'str', 'gettz')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_324975, str_324977)
# Adding element type (line 12)
str_324978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 45), 'str', 'gettz_db_metadata')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_324975, str_324978)
# Adding element type (line 12)
str_324979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 66), 'str', 'rebuild')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_324975, str_324979)

# Assigning a type to the variable '__all__' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), '__all__', list_324975)

# Assigning a Str to a Name (line 14):
str_324980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'str', 'dateutil-zoneinfo.tar.gz')
# Assigning a type to the variable 'ZONEFILENAME' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'ZONEFILENAME', str_324980)

# Assigning a Str to a Name (line 15):
str_324981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 14), 'str', 'METADATA')
# Assigning a type to the variable 'METADATA_FN' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'METADATA_FN', str_324981)

# Assigning a Attribute to a Name (line 19):
# Getting the type of 'TarFile' (line 19)
TarFile_324982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'TarFile')
# Obtaining the member 'open' of a type (line 19)
open_324983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 11), TarFile_324982, 'open')
# Assigning a type to the variable 'tar_open' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'tar_open', open_324983)

# Type idiom detected: calculating its left and rigth part (line 20)
str_324984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 24), 'str', '__exit__')
# Getting the type of 'TarFile' (line 20)
TarFile_324985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 15), 'TarFile')

(may_be_324986, more_types_in_union_324987) = may_not_provide_member(str_324984, TarFile_324985)

if may_be_324986:

    if more_types_in_union_324987:
        # Runtime conditional SSA (line 20)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    # Assigning a type to the variable 'TarFile' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'TarFile', remove_member_provider_from_union(TarFile_324985, '__exit__'))

    @norecursion
    def tar_open(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tar_open'
        module_type_store = module_type_store.open_function_context('tar_open', 21, 4, False)
        
        # Passed parameters checking function
        tar_open.stypy_localization = localization
        tar_open.stypy_type_of_self = None
        tar_open.stypy_type_store = module_type_store
        tar_open.stypy_function_name = 'tar_open'
        tar_open.stypy_param_names_list = []
        tar_open.stypy_varargs_param_name = 'args'
        tar_open.stypy_kwargs_param_name = 'kwargs'
        tar_open.stypy_call_defaults = defaults
        tar_open.stypy_call_varargs = varargs
        tar_open.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'tar_open', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tar_open', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tar_open(...)' code ##################

        
        # Call to closing(...): (line 22)
        # Processing the call arguments (line 22)
        
        # Call to open(...): (line 22)
        # Getting the type of 'args' (line 22)
        args_324991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 37), 'args', False)
        # Processing the call keyword arguments (line 22)
        # Getting the type of 'kwargs' (line 22)
        kwargs_324992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 45), 'kwargs', False)
        kwargs_324993 = {'kwargs_324992': kwargs_324992}
        # Getting the type of 'TarFile' (line 22)
        TarFile_324989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 23), 'TarFile', False)
        # Obtaining the member 'open' of a type (line 22)
        open_324990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 23), TarFile_324989, 'open')
        # Calling open(args, kwargs) (line 22)
        open_call_result_324994 = invoke(stypy.reporting.localization.Localization(__file__, 22, 23), open_324990, *[args_324991], **kwargs_324993)
        
        # Processing the call keyword arguments (line 22)
        kwargs_324995 = {}
        # Getting the type of 'closing' (line 22)
        closing_324988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 15), 'closing', False)
        # Calling closing(args, kwargs) (line 22)
        closing_call_result_324996 = invoke(stypy.reporting.localization.Localization(__file__, 22, 15), closing_324988, *[open_call_result_324994], **kwargs_324995)
        
        # Assigning a type to the variable 'stypy_return_type' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'stypy_return_type', closing_call_result_324996)
        
        # ################# End of 'tar_open(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tar_open' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_324997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324997)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tar_open'
        return stypy_return_type_324997

    # Assigning a type to the variable 'tar_open' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'tar_open', tar_open)

    if more_types_in_union_324987:
        # SSA join for if statement (line 20)
        module_type_store = module_type_store.join_ssa_context()



# Declaration of the 'tzfile' class
# Getting the type of 'tzfile' (line 25)
tzfile_324998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 13), 'tzfile')

class tzfile(tzfile_324998, ):

    @norecursion
    def __reduce__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__reduce__'
        module_type_store = module_type_store.open_function_context('__reduce__', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        tzfile.__reduce__.__dict__.__setitem__('stypy_localization', localization)
        tzfile.__reduce__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        tzfile.__reduce__.__dict__.__setitem__('stypy_type_store', module_type_store)
        tzfile.__reduce__.__dict__.__setitem__('stypy_function_name', 'tzfile.__reduce__')
        tzfile.__reduce__.__dict__.__setitem__('stypy_param_names_list', [])
        tzfile.__reduce__.__dict__.__setitem__('stypy_varargs_param_name', None)
        tzfile.__reduce__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        tzfile.__reduce__.__dict__.__setitem__('stypy_call_defaults', defaults)
        tzfile.__reduce__.__dict__.__setitem__('stypy_call_varargs', varargs)
        tzfile.__reduce__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        tzfile.__reduce__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzfile.__reduce__', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'tuple' (line 27)
        tuple_324999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 27)
        # Adding element type (line 27)
        # Getting the type of 'gettz' (line 27)
        gettz_325000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 16), 'gettz')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 16), tuple_324999, gettz_325000)
        # Adding element type (line 27)
        
        # Obtaining an instance of the builtin type 'tuple' (line 27)
        tuple_325001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 27)
        # Adding element type (line 27)
        # Getting the type of 'self' (line 27)
        self_325002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 24), 'self')
        # Obtaining the member '_filename' of a type (line 27)
        _filename_325003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 24), self_325002, '_filename')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 24), tuple_325001, _filename_325003)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 16), tuple_324999, tuple_325001)
        
        # Assigning a type to the variable 'stypy_return_type' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'stypy_return_type', tuple_324999)
        
        # ################# End of '__reduce__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__reduce__' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_325004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_325004)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__reduce__'
        return stypy_return_type_325004


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 25, 0, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'tzfile.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'tzfile' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'tzfile', tzfile)

@norecursion
def getzoneinfofile_stream(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getzoneinfofile_stream'
    module_type_store = module_type_store.open_function_context('getzoneinfofile_stream', 30, 0, False)
    
    # Passed parameters checking function
    getzoneinfofile_stream.stypy_localization = localization
    getzoneinfofile_stream.stypy_type_of_self = None
    getzoneinfofile_stream.stypy_type_store = module_type_store
    getzoneinfofile_stream.stypy_function_name = 'getzoneinfofile_stream'
    getzoneinfofile_stream.stypy_param_names_list = []
    getzoneinfofile_stream.stypy_varargs_param_name = None
    getzoneinfofile_stream.stypy_kwargs_param_name = None
    getzoneinfofile_stream.stypy_call_defaults = defaults
    getzoneinfofile_stream.stypy_call_varargs = varargs
    getzoneinfofile_stream.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getzoneinfofile_stream', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getzoneinfofile_stream', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getzoneinfofile_stream(...)' code ##################

    
    
    # SSA begins for try-except statement (line 31)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to BytesIO(...): (line 32)
    # Processing the call arguments (line 32)
    
    # Call to get_data(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of '__name__' (line 32)
    name___325007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 32), '__name__', False)
    # Getting the type of 'ZONEFILENAME' (line 32)
    ZONEFILENAME_325008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 42), 'ZONEFILENAME', False)
    # Processing the call keyword arguments (line 32)
    kwargs_325009 = {}
    # Getting the type of 'get_data' (line 32)
    get_data_325006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 23), 'get_data', False)
    # Calling get_data(args, kwargs) (line 32)
    get_data_call_result_325010 = invoke(stypy.reporting.localization.Localization(__file__, 32, 23), get_data_325006, *[name___325007, ZONEFILENAME_325008], **kwargs_325009)
    
    # Processing the call keyword arguments (line 32)
    kwargs_325011 = {}
    # Getting the type of 'BytesIO' (line 32)
    BytesIO_325005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'BytesIO', False)
    # Calling BytesIO(args, kwargs) (line 32)
    BytesIO_call_result_325012 = invoke(stypy.reporting.localization.Localization(__file__, 32, 15), BytesIO_325005, *[get_data_call_result_325010], **kwargs_325011)
    
    # Assigning a type to the variable 'stypy_return_type' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'stypy_return_type', BytesIO_call_result_325012)
    # SSA branch for the except part of a try statement (line 31)
    # SSA branch for the except 'IOError' branch of a try statement (line 31)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'IOError' (line 33)
    IOError_325013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 11), 'IOError')
    # Assigning a type to the variable 'e' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'e', IOError_325013)
    
    # Call to warn(...): (line 34)
    # Processing the call arguments (line 34)
    
    # Call to format(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'e' (line 34)
    e_325018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 51), 'e', False)
    # Obtaining the member 'errno' of a type (line 34)
    errno_325019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 51), e_325018, 'errno')
    # Getting the type of 'e' (line 34)
    e_325020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 60), 'e', False)
    # Obtaining the member 'strerror' of a type (line 34)
    strerror_325021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 60), e_325020, 'strerror')
    # Processing the call keyword arguments (line 34)
    kwargs_325022 = {}
    str_325016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 22), 'str', 'I/O error({0}): {1}')
    # Obtaining the member 'format' of a type (line 34)
    format_325017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 22), str_325016, 'format')
    # Calling format(args, kwargs) (line 34)
    format_call_result_325023 = invoke(stypy.reporting.localization.Localization(__file__, 34, 22), format_325017, *[errno_325019, strerror_325021], **kwargs_325022)
    
    # Processing the call keyword arguments (line 34)
    kwargs_325024 = {}
    # Getting the type of 'warnings' (line 34)
    warnings_325014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 34)
    warn_325015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), warnings_325014, 'warn')
    # Calling warn(args, kwargs) (line 34)
    warn_call_result_325025 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), warn_325015, *[format_call_result_325023], **kwargs_325024)
    
    # Getting the type of 'None' (line 35)
    None_325026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'stypy_return_type', None_325026)
    # SSA join for try-except statement (line 31)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'getzoneinfofile_stream(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getzoneinfofile_stream' in the type store
    # Getting the type of 'stypy_return_type' (line 30)
    stypy_return_type_325027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_325027)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getzoneinfofile_stream'
    return stypy_return_type_325027

# Assigning a type to the variable 'getzoneinfofile_stream' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'getzoneinfofile_stream', getzoneinfofile_stream)
# Declaration of the 'ZoneInfoFile' class

class ZoneInfoFile(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 39)
        None_325028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 39), 'None')
        defaults = [None_325028]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ZoneInfoFile.__init__', ['zonefile_stream'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['zonefile_stream'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 40)
        # Getting the type of 'zonefile_stream' (line 40)
        zonefile_stream_325029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'zonefile_stream')
        # Getting the type of 'None' (line 40)
        None_325030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 34), 'None')
        
        (may_be_325031, more_types_in_union_325032) = may_not_be_none(zonefile_stream_325029, None_325030)

        if may_be_325031:

            if more_types_in_union_325032:
                # Runtime conditional SSA (line 40)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to tar_open(...): (line 41)
            # Processing the call keyword arguments (line 41)
            # Getting the type of 'zonefile_stream' (line 41)
            zonefile_stream_325034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 34), 'zonefile_stream', False)
            keyword_325035 = zonefile_stream_325034
            str_325036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 56), 'str', 'r')
            keyword_325037 = str_325036
            kwargs_325038 = {'mode': keyword_325037, 'fileobj': keyword_325035}
            # Getting the type of 'tar_open' (line 41)
            tar_open_325033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 17), 'tar_open', False)
            # Calling tar_open(args, kwargs) (line 41)
            tar_open_call_result_325039 = invoke(stypy.reporting.localization.Localization(__file__, 41, 17), tar_open_325033, *[], **kwargs_325038)
            
            with_325040 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 41, 17), tar_open_call_result_325039, 'with parameter', '__enter__', '__exit__')

            if with_325040:
                # Calling the __enter__ method to initiate a with section
                # Obtaining the member '__enter__' of a type (line 41)
                enter___325041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 17), tar_open_call_result_325039, '__enter__')
                with_enter_325042 = invoke(stypy.reporting.localization.Localization(__file__, 41, 17), enter___325041)
                # Assigning a type to the variable 'tf' (line 41)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 17), 'tf', with_enter_325042)
                
                # Assigning a Call to a Attribute (line 47):
                
                # Call to dict(...): (line 47)
                # Processing the call arguments (line 47)
                # Calculating generator expression
                module_type_store = module_type_store.open_function_context('list comprehension expression', 47, 34, True)
                # Calculating comprehension expression
                
                # Call to getmembers(...): (line 49)
                # Processing the call keyword arguments (line 49)
                kwargs_325069 = {}
                # Getting the type of 'tf' (line 49)
                tf_325067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 44), 'tf', False)
                # Obtaining the member 'getmembers' of a type (line 49)
                getmembers_325068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 44), tf_325067, 'getmembers')
                # Calling getmembers(args, kwargs) (line 49)
                getmembers_call_result_325070 = invoke(stypy.reporting.localization.Localization(__file__, 49, 44), getmembers_325068, *[], **kwargs_325069)
                
                comprehension_325071 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 34), getmembers_call_result_325070)
                # Assigning a type to the variable 'zf' (line 47)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 34), 'zf', comprehension_325071)
                
                # Evaluating a boolean operation
                
                # Call to isfile(...): (line 50)
                # Processing the call keyword arguments (line 50)
                kwargs_325060 = {}
                # Getting the type of 'zf' (line 50)
                zf_325058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 37), 'zf', False)
                # Obtaining the member 'isfile' of a type (line 50)
                isfile_325059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 37), zf_325058, 'isfile')
                # Calling isfile(args, kwargs) (line 50)
                isfile_call_result_325061 = invoke(stypy.reporting.localization.Localization(__file__, 50, 37), isfile_325059, *[], **kwargs_325060)
                
                
                # Getting the type of 'zf' (line 50)
                zf_325062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 53), 'zf', False)
                # Obtaining the member 'name' of a type (line 50)
                name_325063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 53), zf_325062, 'name')
                # Getting the type of 'METADATA_FN' (line 50)
                METADATA_FN_325064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 64), 'METADATA_FN', False)
                # Applying the binary operator '!=' (line 50)
                result_ne_325065 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 53), '!=', name_325063, METADATA_FN_325064)
                
                # Applying the binary operator 'and' (line 50)
                result_and_keyword_325066 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 37), 'and', isfile_call_result_325061, result_ne_325065)
                
                
                # Obtaining an instance of the builtin type 'tuple' (line 47)
                tuple_325044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 35), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 47)
                # Adding element type (line 47)
                # Getting the type of 'zf' (line 47)
                zf_325045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 35), 'zf', False)
                # Obtaining the member 'name' of a type (line 47)
                name_325046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 35), zf_325045, 'name')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 35), tuple_325044, name_325046)
                # Adding element type (line 47)
                
                # Call to tzfile(...): (line 47)
                # Processing the call arguments (line 47)
                
                # Call to extractfile(...): (line 47)
                # Processing the call arguments (line 47)
                # Getting the type of 'zf' (line 47)
                zf_325050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 66), 'zf', False)
                # Processing the call keyword arguments (line 47)
                kwargs_325051 = {}
                # Getting the type of 'tf' (line 47)
                tf_325048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 51), 'tf', False)
                # Obtaining the member 'extractfile' of a type (line 47)
                extractfile_325049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 51), tf_325048, 'extractfile')
                # Calling extractfile(args, kwargs) (line 47)
                extractfile_call_result_325052 = invoke(stypy.reporting.localization.Localization(__file__, 47, 51), extractfile_325049, *[zf_325050], **kwargs_325051)
                
                # Processing the call keyword arguments (line 47)
                # Getting the type of 'zf' (line 48)
                zf_325053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 60), 'zf', False)
                # Obtaining the member 'name' of a type (line 48)
                name_325054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 60), zf_325053, 'name')
                keyword_325055 = name_325054
                kwargs_325056 = {'filename': keyword_325055}
                # Getting the type of 'tzfile' (line 47)
                tzfile_325047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 44), 'tzfile', False)
                # Calling tzfile(args, kwargs) (line 47)
                tzfile_call_result_325057 = invoke(stypy.reporting.localization.Localization(__file__, 47, 44), tzfile_325047, *[extractfile_call_result_325052], **kwargs_325056)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 35), tuple_325044, tzfile_call_result_325057)
                
                list_325072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 34), 'list')
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 34), list_325072, tuple_325044)
                # Processing the call keyword arguments (line 47)
                kwargs_325073 = {}
                # Getting the type of 'dict' (line 47)
                dict_325043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 29), 'dict', False)
                # Calling dict(args, kwargs) (line 47)
                dict_call_result_325074 = invoke(stypy.reporting.localization.Localization(__file__, 47, 29), dict_325043, *[list_325072], **kwargs_325073)
                
                # Getting the type of 'self' (line 47)
                self_325075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'self')
                # Setting the type of the member 'zones' of a type (line 47)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 16), self_325075, 'zones', dict_call_result_325074)
                
                # Assigning a Call to a Name (line 55):
                
                # Call to dict(...): (line 55)
                # Processing the call arguments (line 55)
                # Calculating generator expression
                module_type_store = module_type_store.open_function_context('list comprehension expression', 55, 29, True)
                # Calculating comprehension expression
                
                # Call to getmembers(...): (line 56)
                # Processing the call keyword arguments (line 56)
                kwargs_325097 = {}
                # Getting the type of 'tf' (line 56)
                tf_325095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 39), 'tf', False)
                # Obtaining the member 'getmembers' of a type (line 56)
                getmembers_325096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 39), tf_325095, 'getmembers')
                # Calling getmembers(args, kwargs) (line 56)
                getmembers_call_result_325098 = invoke(stypy.reporting.localization.Localization(__file__, 56, 39), getmembers_325096, *[], **kwargs_325097)
                
                comprehension_325099 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 29), getmembers_call_result_325098)
                # Assigning a type to the variable 'zl' (line 55)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 29), 'zl', comprehension_325099)
                
                # Evaluating a boolean operation
                
                # Call to islnk(...): (line 57)
                # Processing the call keyword arguments (line 57)
                kwargs_325088 = {}
                # Getting the type of 'zl' (line 57)
                zl_325086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 29), 'zl', False)
                # Obtaining the member 'islnk' of a type (line 57)
                islnk_325087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 29), zl_325086, 'islnk')
                # Calling islnk(args, kwargs) (line 57)
                islnk_call_result_325089 = invoke(stypy.reporting.localization.Localization(__file__, 57, 29), islnk_325087, *[], **kwargs_325088)
                
                
                # Call to issym(...): (line 57)
                # Processing the call keyword arguments (line 57)
                kwargs_325092 = {}
                # Getting the type of 'zl' (line 57)
                zl_325090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 43), 'zl', False)
                # Obtaining the member 'issym' of a type (line 57)
                issym_325091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 43), zl_325090, 'issym')
                # Calling issym(args, kwargs) (line 57)
                issym_call_result_325093 = invoke(stypy.reporting.localization.Localization(__file__, 57, 43), issym_325091, *[], **kwargs_325092)
                
                # Applying the binary operator 'or' (line 57)
                result_or_keyword_325094 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 29), 'or', islnk_call_result_325089, issym_call_result_325093)
                
                
                # Obtaining an instance of the builtin type 'tuple' (line 55)
                tuple_325077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 30), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 55)
                # Adding element type (line 55)
                # Getting the type of 'zl' (line 55)
                zl_325078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 30), 'zl', False)
                # Obtaining the member 'name' of a type (line 55)
                name_325079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 30), zl_325078, 'name')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 30), tuple_325077, name_325079)
                # Adding element type (line 55)
                
                # Obtaining the type of the subscript
                # Getting the type of 'zl' (line 55)
                zl_325080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 50), 'zl', False)
                # Obtaining the member 'linkname' of a type (line 55)
                linkname_325081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 50), zl_325080, 'linkname')
                # Getting the type of 'self' (line 55)
                self_325082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 39), 'self', False)
                # Obtaining the member 'zones' of a type (line 55)
                zones_325083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 39), self_325082, 'zones')
                # Obtaining the member '__getitem__' of a type (line 55)
                getitem___325084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 39), zones_325083, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 55)
                subscript_call_result_325085 = invoke(stypy.reporting.localization.Localization(__file__, 55, 39), getitem___325084, linkname_325081)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 30), tuple_325077, subscript_call_result_325085)
                
                list_325100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 29), 'list')
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 29), list_325100, tuple_325077)
                # Processing the call keyword arguments (line 55)
                kwargs_325101 = {}
                # Getting the type of 'dict' (line 55)
                dict_325076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 24), 'dict', False)
                # Calling dict(args, kwargs) (line 55)
                dict_call_result_325102 = invoke(stypy.reporting.localization.Localization(__file__, 55, 24), dict_325076, *[list_325100], **kwargs_325101)
                
                # Assigning a type to the variable 'links' (line 55)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'links', dict_call_result_325102)
                
                # Call to update(...): (line 58)
                # Processing the call arguments (line 58)
                # Getting the type of 'links' (line 58)
                links_325106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 34), 'links', False)
                # Processing the call keyword arguments (line 58)
                kwargs_325107 = {}
                # Getting the type of 'self' (line 58)
                self_325103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'self', False)
                # Obtaining the member 'zones' of a type (line 58)
                zones_325104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 16), self_325103, 'zones')
                # Obtaining the member 'update' of a type (line 58)
                update_325105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 16), zones_325104, 'update')
                # Calling update(args, kwargs) (line 58)
                update_call_result_325108 = invoke(stypy.reporting.localization.Localization(__file__, 58, 16), update_325105, *[links_325106], **kwargs_325107)
                
                
                
                # SSA begins for try-except statement (line 59)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                
                # Assigning a Call to a Name (line 60):
                
                # Call to extractfile(...): (line 60)
                # Processing the call arguments (line 60)
                
                # Call to getmember(...): (line 60)
                # Processing the call arguments (line 60)
                # Getting the type of 'METADATA_FN' (line 60)
                METADATA_FN_325113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 64), 'METADATA_FN', False)
                # Processing the call keyword arguments (line 60)
                kwargs_325114 = {}
                # Getting the type of 'tf' (line 60)
                tf_325111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 51), 'tf', False)
                # Obtaining the member 'getmember' of a type (line 60)
                getmember_325112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 51), tf_325111, 'getmember')
                # Calling getmember(args, kwargs) (line 60)
                getmember_call_result_325115 = invoke(stypy.reporting.localization.Localization(__file__, 60, 51), getmember_325112, *[METADATA_FN_325113], **kwargs_325114)
                
                # Processing the call keyword arguments (line 60)
                kwargs_325116 = {}
                # Getting the type of 'tf' (line 60)
                tf_325109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 36), 'tf', False)
                # Obtaining the member 'extractfile' of a type (line 60)
                extractfile_325110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 36), tf_325109, 'extractfile')
                # Calling extractfile(args, kwargs) (line 60)
                extractfile_call_result_325117 = invoke(stypy.reporting.localization.Localization(__file__, 60, 36), extractfile_325110, *[getmember_call_result_325115], **kwargs_325116)
                
                # Assigning a type to the variable 'metadata_json' (line 60)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'metadata_json', extractfile_call_result_325117)
                
                # Assigning a Call to a Name (line 61):
                
                # Call to decode(...): (line 61)
                # Processing the call arguments (line 61)
                str_325123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 63), 'str', 'UTF-8')
                # Processing the call keyword arguments (line 61)
                kwargs_325124 = {}
                
                # Call to read(...): (line 61)
                # Processing the call keyword arguments (line 61)
                kwargs_325120 = {}
                # Getting the type of 'metadata_json' (line 61)
                metadata_json_325118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 35), 'metadata_json', False)
                # Obtaining the member 'read' of a type (line 61)
                read_325119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 35), metadata_json_325118, 'read')
                # Calling read(args, kwargs) (line 61)
                read_call_result_325121 = invoke(stypy.reporting.localization.Localization(__file__, 61, 35), read_325119, *[], **kwargs_325120)
                
                # Obtaining the member 'decode' of a type (line 61)
                decode_325122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 35), read_call_result_325121, 'decode')
                # Calling decode(args, kwargs) (line 61)
                decode_call_result_325125 = invoke(stypy.reporting.localization.Localization(__file__, 61, 35), decode_325122, *[str_325123], **kwargs_325124)
                
                # Assigning a type to the variable 'metadata_str' (line 61)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'metadata_str', decode_call_result_325125)
                
                # Assigning a Call to a Attribute (line 62):
                
                # Call to loads(...): (line 62)
                # Processing the call arguments (line 62)
                # Getting the type of 'metadata_str' (line 62)
                metadata_str_325128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 47), 'metadata_str', False)
                # Processing the call keyword arguments (line 62)
                kwargs_325129 = {}
                # Getting the type of 'json' (line 62)
                json_325126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 36), 'json', False)
                # Obtaining the member 'loads' of a type (line 62)
                loads_325127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 36), json_325126, 'loads')
                # Calling loads(args, kwargs) (line 62)
                loads_call_result_325130 = invoke(stypy.reporting.localization.Localization(__file__, 62, 36), loads_325127, *[metadata_str_325128], **kwargs_325129)
                
                # Getting the type of 'self' (line 62)
                self_325131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'self')
                # Setting the type of the member 'metadata' of a type (line 62)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 20), self_325131, 'metadata', loads_call_result_325130)
                # SSA branch for the except part of a try statement (line 59)
                # SSA branch for the except 'KeyError' branch of a try statement (line 59)
                module_type_store.open_ssa_branch('except')
                
                # Assigning a Name to a Attribute (line 65):
                # Getting the type of 'None' (line 65)
                None_325132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 36), 'None')
                # Getting the type of 'self' (line 65)
                self_325133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'self')
                # Setting the type of the member 'metadata' of a type (line 65)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 20), self_325133, 'metadata', None_325132)
                # SSA join for try-except statement (line 59)
                module_type_store = module_type_store.join_ssa_context()
                
                # Calling the __exit__ method to finish a with section
                # Obtaining the member '__exit__' of a type (line 41)
                exit___325134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 17), tar_open_call_result_325039, '__exit__')
                with_exit_325135 = invoke(stypy.reporting.localization.Localization(__file__, 41, 17), exit___325134, None, None, None)


            if more_types_in_union_325032:
                # Runtime conditional SSA for else branch (line 40)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_325031) or more_types_in_union_325032):
            
            # Assigning a Call to a Attribute (line 67):
            
            # Call to dict(...): (line 67)
            # Processing the call keyword arguments (line 67)
            kwargs_325137 = {}
            # Getting the type of 'dict' (line 67)
            dict_325136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 25), 'dict', False)
            # Calling dict(args, kwargs) (line 67)
            dict_call_result_325138 = invoke(stypy.reporting.localization.Localization(__file__, 67, 25), dict_325136, *[], **kwargs_325137)
            
            # Getting the type of 'self' (line 67)
            self_325139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'self')
            # Setting the type of the member 'zones' of a type (line 67)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 12), self_325139, 'zones', dict_call_result_325138)
            
            # Assigning a Name to a Attribute (line 68):
            # Getting the type of 'None' (line 68)
            None_325140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 28), 'None')
            # Getting the type of 'self' (line 68)
            self_325141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'self')
            # Setting the type of the member 'metadata' of a type (line 68)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), self_325141, 'metadata', None_325140)

            if (may_be_325031 and more_types_in_union_325032):
                # SSA join for if statement (line 40)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 70)
        None_325142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 32), 'None')
        defaults = [None_325142]
        # Create a new context for function 'get'
        module_type_store = module_type_store.open_function_context('get', 70, 4, False)
        # Assigning a type to the variable 'self' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ZoneInfoFile.get.__dict__.__setitem__('stypy_localization', localization)
        ZoneInfoFile.get.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ZoneInfoFile.get.__dict__.__setitem__('stypy_type_store', module_type_store)
        ZoneInfoFile.get.__dict__.__setitem__('stypy_function_name', 'ZoneInfoFile.get')
        ZoneInfoFile.get.__dict__.__setitem__('stypy_param_names_list', ['name', 'default'])
        ZoneInfoFile.get.__dict__.__setitem__('stypy_varargs_param_name', None)
        ZoneInfoFile.get.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ZoneInfoFile.get.__dict__.__setitem__('stypy_call_defaults', defaults)
        ZoneInfoFile.get.__dict__.__setitem__('stypy_call_varargs', varargs)
        ZoneInfoFile.get.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ZoneInfoFile.get.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ZoneInfoFile.get', ['name', 'default'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get', localization, ['name', 'default'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get(...)' code ##################

        str_325143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, (-1)), 'str', '\n        Wrapper for :func:`ZoneInfoFile.zones.get`. This is a convenience method\n        for retrieving zones from the zone dictionary.\n\n        :param name:\n            The name of the zone to retrieve. (Generally IANA zone names)\n\n        :param default:\n            The value to return in the event of a missing key.\n\n        .. versionadded:: 2.6.0\n\n        ')
        
        # Call to get(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'name' (line 84)
        name_325147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 30), 'name', False)
        # Getting the type of 'default' (line 84)
        default_325148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 36), 'default', False)
        # Processing the call keyword arguments (line 84)
        kwargs_325149 = {}
        # Getting the type of 'self' (line 84)
        self_325144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 15), 'self', False)
        # Obtaining the member 'zones' of a type (line 84)
        zones_325145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 15), self_325144, 'zones')
        # Obtaining the member 'get' of a type (line 84)
        get_325146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 15), zones_325145, 'get')
        # Calling get(args, kwargs) (line 84)
        get_call_result_325150 = invoke(stypy.reporting.localization.Localization(__file__, 84, 15), get_325146, *[name_325147, default_325148], **kwargs_325149)
        
        # Assigning a type to the variable 'stypy_return_type' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'stypy_return_type', get_call_result_325150)
        
        # ################# End of 'get(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get' in the type store
        # Getting the type of 'stypy_return_type' (line 70)
        stypy_return_type_325151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_325151)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get'
        return stypy_return_type_325151


# Assigning a type to the variable 'ZoneInfoFile' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'ZoneInfoFile', ZoneInfoFile)

# Assigning a Call to a Name (line 93):

# Call to list(...): (line 93)
# Processing the call keyword arguments (line 93)
kwargs_325153 = {}
# Getting the type of 'list' (line 93)
list_325152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 23), 'list', False)
# Calling list(args, kwargs) (line 93)
list_call_result_325154 = invoke(stypy.reporting.localization.Localization(__file__, 93, 23), list_325152, *[], **kwargs_325153)

# Assigning a type to the variable '_CLASS_ZONE_INSTANCE' (line 93)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), '_CLASS_ZONE_INSTANCE', list_call_result_325154)

@norecursion
def get_zonefile_instance(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 96)
    False_325155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 39), 'False')
    defaults = [False_325155]
    # Create a new context for function 'get_zonefile_instance'
    module_type_store = module_type_store.open_function_context('get_zonefile_instance', 96, 0, False)
    
    # Passed parameters checking function
    get_zonefile_instance.stypy_localization = localization
    get_zonefile_instance.stypy_type_of_self = None
    get_zonefile_instance.stypy_type_store = module_type_store
    get_zonefile_instance.stypy_function_name = 'get_zonefile_instance'
    get_zonefile_instance.stypy_param_names_list = ['new_instance']
    get_zonefile_instance.stypy_varargs_param_name = None
    get_zonefile_instance.stypy_kwargs_param_name = None
    get_zonefile_instance.stypy_call_defaults = defaults
    get_zonefile_instance.stypy_call_varargs = varargs
    get_zonefile_instance.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_zonefile_instance', ['new_instance'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_zonefile_instance', localization, ['new_instance'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_zonefile_instance(...)' code ##################

    str_325156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, (-1)), 'str', '\n    This is a convenience function which provides a :class:`ZoneInfoFile`\n    instance using the data provided by the ``dateutil`` package. By default, it\n    caches a single instance of the ZoneInfoFile object and returns that.\n\n    :param new_instance:\n        If ``True``, a new instance of :class:`ZoneInfoFile` is instantiated and\n        used as the cached instance for the next call. Otherwise, new instances\n        are created only as necessary.\n\n    :return:\n        Returns a :class:`ZoneInfoFile` object.\n\n    .. versionadded:: 2.6\n    ')
    
    # Getting the type of 'new_instance' (line 112)
    new_instance_325157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 7), 'new_instance')
    # Testing the type of an if condition (line 112)
    if_condition_325158 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 4), new_instance_325157)
    # Assigning a type to the variable 'if_condition_325158' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'if_condition_325158', if_condition_325158)
    # SSA begins for if statement (line 112)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 113):
    # Getting the type of 'None' (line 113)
    None_325159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 14), 'None')
    # Assigning a type to the variable 'zif' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'zif', None_325159)
    # SSA branch for the else part of an if statement (line 112)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 115):
    
    # Call to getattr(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'get_zonefile_instance' (line 115)
    get_zonefile_instance_325161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 22), 'get_zonefile_instance', False)
    str_325162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 45), 'str', '_cached_instance')
    # Getting the type of 'None' (line 115)
    None_325163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 65), 'None', False)
    # Processing the call keyword arguments (line 115)
    kwargs_325164 = {}
    # Getting the type of 'getattr' (line 115)
    getattr_325160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 14), 'getattr', False)
    # Calling getattr(args, kwargs) (line 115)
    getattr_call_result_325165 = invoke(stypy.reporting.localization.Localization(__file__, 115, 14), getattr_325160, *[get_zonefile_instance_325161, str_325162, None_325163], **kwargs_325164)
    
    # Assigning a type to the variable 'zif' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'zif', getattr_call_result_325165)
    # SSA join for if statement (line 112)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 117)
    # Getting the type of 'zif' (line 117)
    zif_325166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 7), 'zif')
    # Getting the type of 'None' (line 117)
    None_325167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 14), 'None')
    
    (may_be_325168, more_types_in_union_325169) = may_be_none(zif_325166, None_325167)

    if may_be_325168:

        if more_types_in_union_325169:
            # Runtime conditional SSA (line 117)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 118):
        
        # Call to ZoneInfoFile(...): (line 118)
        # Processing the call arguments (line 118)
        
        # Call to getzoneinfofile_stream(...): (line 118)
        # Processing the call keyword arguments (line 118)
        kwargs_325172 = {}
        # Getting the type of 'getzoneinfofile_stream' (line 118)
        getzoneinfofile_stream_325171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 27), 'getzoneinfofile_stream', False)
        # Calling getzoneinfofile_stream(args, kwargs) (line 118)
        getzoneinfofile_stream_call_result_325173 = invoke(stypy.reporting.localization.Localization(__file__, 118, 27), getzoneinfofile_stream_325171, *[], **kwargs_325172)
        
        # Processing the call keyword arguments (line 118)
        kwargs_325174 = {}
        # Getting the type of 'ZoneInfoFile' (line 118)
        ZoneInfoFile_325170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 14), 'ZoneInfoFile', False)
        # Calling ZoneInfoFile(args, kwargs) (line 118)
        ZoneInfoFile_call_result_325175 = invoke(stypy.reporting.localization.Localization(__file__, 118, 14), ZoneInfoFile_325170, *[getzoneinfofile_stream_call_result_325173], **kwargs_325174)
        
        # Assigning a type to the variable 'zif' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'zif', ZoneInfoFile_call_result_325175)
        
        # Assigning a Name to a Attribute (line 120):
        # Getting the type of 'zif' (line 120)
        zif_325176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 49), 'zif')
        # Getting the type of 'get_zonefile_instance' (line 120)
        get_zonefile_instance_325177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'get_zonefile_instance')
        # Setting the type of the member '_cached_instance' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), get_zonefile_instance_325177, '_cached_instance', zif_325176)

        if more_types_in_union_325169:
            # SSA join for if statement (line 117)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'zif' (line 122)
    zif_325178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 11), 'zif')
    # Assigning a type to the variable 'stypy_return_type' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type', zif_325178)
    
    # ################# End of 'get_zonefile_instance(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_zonefile_instance' in the type store
    # Getting the type of 'stypy_return_type' (line 96)
    stypy_return_type_325179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_325179)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_zonefile_instance'
    return stypy_return_type_325179

# Assigning a type to the variable 'get_zonefile_instance' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'get_zonefile_instance', get_zonefile_instance)

@norecursion
def gettz(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'gettz'
    module_type_store = module_type_store.open_function_context('gettz', 125, 0, False)
    
    # Passed parameters checking function
    gettz.stypy_localization = localization
    gettz.stypy_type_of_self = None
    gettz.stypy_type_store = module_type_store
    gettz.stypy_function_name = 'gettz'
    gettz.stypy_param_names_list = ['name']
    gettz.stypy_varargs_param_name = None
    gettz.stypy_kwargs_param_name = None
    gettz.stypy_call_defaults = defaults
    gettz.stypy_call_varargs = varargs
    gettz.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gettz', ['name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gettz', localization, ['name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gettz(...)' code ##################

    str_325180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, (-1)), 'str', '\n    This retrieves a time zone from the local zoneinfo tarball that is packaged\n    with dateutil.\n\n    :param name:\n        An IANA-style time zone name, as found in the zoneinfo file.\n\n    :return:\n        Returns a :class:`dateutil.tz.tzfile` time zone object.\n\n    .. warning::\n        It is generally inadvisable to use this function, and it is only\n        provided for API compatibility with earlier versions. This is *not*\n        equivalent to ``dateutil.tz.gettz()``, which selects an appropriate\n        time zone based on the inputs, favoring system zoneinfo. This is ONLY\n        for accessing the dateutil-specific zoneinfo (which may be out of\n        date compared to the system zoneinfo).\n\n    .. deprecated:: 2.6\n        If you need to use a specific zoneinfofile over the system zoneinfo,\n        instantiate a :class:`dateutil.zoneinfo.ZoneInfoFile` object and call\n        :func:`dateutil.zoneinfo.ZoneInfoFile.get(name)` instead.\n\n        Use :func:`get_zonefile_instance` to retrieve an instance of the\n        dateutil-provided zoneinfo.\n    ')
    
    # Call to warn(...): (line 152)
    # Processing the call arguments (line 152)
    str_325183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 18), 'str', 'zoneinfo.gettz() will be removed in future versions, to use the dateutil-provided zoneinfo files, instantiate a ZoneInfoFile object and use ZoneInfoFile.zones.get() instead. See the documentation for details.')
    # Getting the type of 'DeprecationWarning' (line 156)
    DeprecationWarning_325184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 18), 'DeprecationWarning', False)
    # Processing the call keyword arguments (line 152)
    kwargs_325185 = {}
    # Getting the type of 'warnings' (line 152)
    warnings_325181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 152)
    warn_325182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 4), warnings_325181, 'warn')
    # Calling warn(args, kwargs) (line 152)
    warn_call_result_325186 = invoke(stypy.reporting.localization.Localization(__file__, 152, 4), warn_325182, *[str_325183, DeprecationWarning_325184], **kwargs_325185)
    
    
    
    
    # Call to len(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of '_CLASS_ZONE_INSTANCE' (line 158)
    _CLASS_ZONE_INSTANCE_325188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), '_CLASS_ZONE_INSTANCE', False)
    # Processing the call keyword arguments (line 158)
    kwargs_325189 = {}
    # Getting the type of 'len' (line 158)
    len_325187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 7), 'len', False)
    # Calling len(args, kwargs) (line 158)
    len_call_result_325190 = invoke(stypy.reporting.localization.Localization(__file__, 158, 7), len_325187, *[_CLASS_ZONE_INSTANCE_325188], **kwargs_325189)
    
    int_325191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 36), 'int')
    # Applying the binary operator '==' (line 158)
    result_eq_325192 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 7), '==', len_call_result_325190, int_325191)
    
    # Testing the type of an if condition (line 158)
    if_condition_325193 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 4), result_eq_325192)
    # Assigning a type to the variable 'if_condition_325193' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'if_condition_325193', if_condition_325193)
    # SSA begins for if statement (line 158)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 159)
    # Processing the call arguments (line 159)
    
    # Call to ZoneInfoFile(...): (line 159)
    # Processing the call arguments (line 159)
    
    # Call to getzoneinfofile_stream(...): (line 159)
    # Processing the call keyword arguments (line 159)
    kwargs_325198 = {}
    # Getting the type of 'getzoneinfofile_stream' (line 159)
    getzoneinfofile_stream_325197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 49), 'getzoneinfofile_stream', False)
    # Calling getzoneinfofile_stream(args, kwargs) (line 159)
    getzoneinfofile_stream_call_result_325199 = invoke(stypy.reporting.localization.Localization(__file__, 159, 49), getzoneinfofile_stream_325197, *[], **kwargs_325198)
    
    # Processing the call keyword arguments (line 159)
    kwargs_325200 = {}
    # Getting the type of 'ZoneInfoFile' (line 159)
    ZoneInfoFile_325196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 36), 'ZoneInfoFile', False)
    # Calling ZoneInfoFile(args, kwargs) (line 159)
    ZoneInfoFile_call_result_325201 = invoke(stypy.reporting.localization.Localization(__file__, 159, 36), ZoneInfoFile_325196, *[getzoneinfofile_stream_call_result_325199], **kwargs_325200)
    
    # Processing the call keyword arguments (line 159)
    kwargs_325202 = {}
    # Getting the type of '_CLASS_ZONE_INSTANCE' (line 159)
    _CLASS_ZONE_INSTANCE_325194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), '_CLASS_ZONE_INSTANCE', False)
    # Obtaining the member 'append' of a type (line 159)
    append_325195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), _CLASS_ZONE_INSTANCE_325194, 'append')
    # Calling append(args, kwargs) (line 159)
    append_call_result_325203 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), append_325195, *[ZoneInfoFile_call_result_325201], **kwargs_325202)
    
    # SSA join for if statement (line 158)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to get(...): (line 160)
    # Processing the call arguments (line 160)
    # Getting the type of 'name' (line 160)
    name_325210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 45), 'name', False)
    # Processing the call keyword arguments (line 160)
    kwargs_325211 = {}
    
    # Obtaining the type of the subscript
    int_325204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 32), 'int')
    # Getting the type of '_CLASS_ZONE_INSTANCE' (line 160)
    _CLASS_ZONE_INSTANCE_325205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 11), '_CLASS_ZONE_INSTANCE', False)
    # Obtaining the member '__getitem__' of a type (line 160)
    getitem___325206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 11), _CLASS_ZONE_INSTANCE_325205, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 160)
    subscript_call_result_325207 = invoke(stypy.reporting.localization.Localization(__file__, 160, 11), getitem___325206, int_325204)
    
    # Obtaining the member 'zones' of a type (line 160)
    zones_325208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 11), subscript_call_result_325207, 'zones')
    # Obtaining the member 'get' of a type (line 160)
    get_325209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 11), zones_325208, 'get')
    # Calling get(args, kwargs) (line 160)
    get_call_result_325212 = invoke(stypy.reporting.localization.Localization(__file__, 160, 11), get_325209, *[name_325210], **kwargs_325211)
    
    # Assigning a type to the variable 'stypy_return_type' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'stypy_return_type', get_call_result_325212)
    
    # ################# End of 'gettz(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gettz' in the type store
    # Getting the type of 'stypy_return_type' (line 125)
    stypy_return_type_325213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_325213)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gettz'
    return stypy_return_type_325213

# Assigning a type to the variable 'gettz' (line 125)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 0), 'gettz', gettz)

@norecursion
def gettz_db_metadata(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'gettz_db_metadata'
    module_type_store = module_type_store.open_function_context('gettz_db_metadata', 163, 0, False)
    
    # Passed parameters checking function
    gettz_db_metadata.stypy_localization = localization
    gettz_db_metadata.stypy_type_of_self = None
    gettz_db_metadata.stypy_type_store = module_type_store
    gettz_db_metadata.stypy_function_name = 'gettz_db_metadata'
    gettz_db_metadata.stypy_param_names_list = []
    gettz_db_metadata.stypy_varargs_param_name = None
    gettz_db_metadata.stypy_kwargs_param_name = None
    gettz_db_metadata.stypy_call_defaults = defaults
    gettz_db_metadata.stypy_call_varargs = varargs
    gettz_db_metadata.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gettz_db_metadata', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gettz_db_metadata', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gettz_db_metadata(...)' code ##################

    str_325214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, (-1)), 'str', ' Get the zonefile metadata\n\n    See `zonefile_metadata`_\n\n    :returns:\n        A dictionary with the database metadata\n\n    .. deprecated:: 2.6\n        See deprecation warning in :func:`zoneinfo.gettz`. To get metadata,\n        query the attribute ``zoneinfo.ZoneInfoFile.metadata``.\n    ')
    
    # Call to warn(...): (line 175)
    # Processing the call arguments (line 175)
    str_325217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 18), 'str', "zoneinfo.gettz_db_metadata() will be removed in future versions, to use the dateutil-provided zoneinfo files, ZoneInfoFile object and query the 'metadata' attribute instead. See the documentation for details.")
    # Getting the type of 'DeprecationWarning' (line 179)
    DeprecationWarning_325218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 18), 'DeprecationWarning', False)
    # Processing the call keyword arguments (line 175)
    kwargs_325219 = {}
    # Getting the type of 'warnings' (line 175)
    warnings_325215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 175)
    warn_325216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 4), warnings_325215, 'warn')
    # Calling warn(args, kwargs) (line 175)
    warn_call_result_325220 = invoke(stypy.reporting.localization.Localization(__file__, 175, 4), warn_325216, *[str_325217, DeprecationWarning_325218], **kwargs_325219)
    
    
    
    
    # Call to len(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of '_CLASS_ZONE_INSTANCE' (line 181)
    _CLASS_ZONE_INSTANCE_325222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 11), '_CLASS_ZONE_INSTANCE', False)
    # Processing the call keyword arguments (line 181)
    kwargs_325223 = {}
    # Getting the type of 'len' (line 181)
    len_325221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 7), 'len', False)
    # Calling len(args, kwargs) (line 181)
    len_call_result_325224 = invoke(stypy.reporting.localization.Localization(__file__, 181, 7), len_325221, *[_CLASS_ZONE_INSTANCE_325222], **kwargs_325223)
    
    int_325225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 36), 'int')
    # Applying the binary operator '==' (line 181)
    result_eq_325226 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 7), '==', len_call_result_325224, int_325225)
    
    # Testing the type of an if condition (line 181)
    if_condition_325227 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 4), result_eq_325226)
    # Assigning a type to the variable 'if_condition_325227' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'if_condition_325227', if_condition_325227)
    # SSA begins for if statement (line 181)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 182)
    # Processing the call arguments (line 182)
    
    # Call to ZoneInfoFile(...): (line 182)
    # Processing the call arguments (line 182)
    
    # Call to getzoneinfofile_stream(...): (line 182)
    # Processing the call keyword arguments (line 182)
    kwargs_325232 = {}
    # Getting the type of 'getzoneinfofile_stream' (line 182)
    getzoneinfofile_stream_325231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 49), 'getzoneinfofile_stream', False)
    # Calling getzoneinfofile_stream(args, kwargs) (line 182)
    getzoneinfofile_stream_call_result_325233 = invoke(stypy.reporting.localization.Localization(__file__, 182, 49), getzoneinfofile_stream_325231, *[], **kwargs_325232)
    
    # Processing the call keyword arguments (line 182)
    kwargs_325234 = {}
    # Getting the type of 'ZoneInfoFile' (line 182)
    ZoneInfoFile_325230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 36), 'ZoneInfoFile', False)
    # Calling ZoneInfoFile(args, kwargs) (line 182)
    ZoneInfoFile_call_result_325235 = invoke(stypy.reporting.localization.Localization(__file__, 182, 36), ZoneInfoFile_325230, *[getzoneinfofile_stream_call_result_325233], **kwargs_325234)
    
    # Processing the call keyword arguments (line 182)
    kwargs_325236 = {}
    # Getting the type of '_CLASS_ZONE_INSTANCE' (line 182)
    _CLASS_ZONE_INSTANCE_325228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), '_CLASS_ZONE_INSTANCE', False)
    # Obtaining the member 'append' of a type (line 182)
    append_325229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), _CLASS_ZONE_INSTANCE_325228, 'append')
    # Calling append(args, kwargs) (line 182)
    append_call_result_325237 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), append_325229, *[ZoneInfoFile_call_result_325235], **kwargs_325236)
    
    # SSA join for if statement (line 181)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    int_325238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 32), 'int')
    # Getting the type of '_CLASS_ZONE_INSTANCE' (line 183)
    _CLASS_ZONE_INSTANCE_325239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 11), '_CLASS_ZONE_INSTANCE')
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___325240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 11), _CLASS_ZONE_INSTANCE_325239, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 183)
    subscript_call_result_325241 = invoke(stypy.reporting.localization.Localization(__file__, 183, 11), getitem___325240, int_325238)
    
    # Obtaining the member 'metadata' of a type (line 183)
    metadata_325242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 11), subscript_call_result_325241, 'metadata')
    # Assigning a type to the variable 'stypy_return_type' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'stypy_return_type', metadata_325242)
    
    # ################# End of 'gettz_db_metadata(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gettz_db_metadata' in the type store
    # Getting the type of 'stypy_return_type' (line 163)
    stypy_return_type_325243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_325243)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gettz_db_metadata'
    return stypy_return_type_325243

# Assigning a type to the variable 'gettz_db_metadata' (line 163)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), 'gettz_db_metadata', gettz_db_metadata)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
