
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: 
6: '''
7: Core functions and attributes for the matplotlib style library:
8: 
9: ``use``
10:     Select style sheet to override the current matplotlib settings.
11: ``context``
12:     Context manager to use a style sheet temporarily.
13: ``available``
14:     List available style sheets.
15: ``library``
16:     A dictionary of style names and matplotlib settings.
17: '''
18: import os
19: import re
20: import contextlib
21: import warnings
22: 
23: import matplotlib as mpl
24: from matplotlib import cbook
25: from matplotlib import rc_params_from_file, rcParamsDefault
26: 
27: 
28: __all__ = ['use', 'context', 'available', 'library', 'reload_library']
29: 
30: 
31: BASE_LIBRARY_PATH = os.path.join(mpl.get_data_path(), 'stylelib')
32: # Users may want multiple library paths, so store a list of paths.
33: USER_LIBRARY_PATHS = [os.path.join(mpl._get_configdir(), 'stylelib')]
34: STYLE_EXTENSION = 'mplstyle'
35: STYLE_FILE_PATTERN = re.compile(r'([\S]+).%s$' % STYLE_EXTENSION)
36: 
37: 
38: # A list of rcParams that should not be applied from styles
39: STYLE_BLACKLIST = {
40:     'interactive', 'backend', 'backend.qt4', 'webagg.port',
41:     'webagg.port_retries', 'webagg.open_in_browser', 'backend_fallback',
42:     'toolbar', 'timezone', 'datapath', 'figure.max_open_warning',
43:     'savefig.directory', 'tk.window_focus', 'docstring.hardcopy'}
44: 
45: 
46: def _remove_blacklisted_style_params(d, warn=True):
47:     o = {}
48:     for key, val in d.items():
49:         if key in STYLE_BLACKLIST:
50:             if warn:
51:                 warnings.warn(
52:                     "Style includes a parameter, '{0}', that is not related "
53:                     "to style.  Ignoring".format(key))
54:         else:
55:             o[key] = val
56:     return o
57: 
58: 
59: def is_style_file(filename):
60:     '''Return True if the filename looks like a style file.'''
61:     return STYLE_FILE_PATTERN.match(filename) is not None
62: 
63: 
64: def _apply_style(d, warn=True):
65:     mpl.rcParams.update(_remove_blacklisted_style_params(d, warn=warn))
66: 
67: 
68: def use(style):
69:     '''Use matplotlib style settings from a style specification.
70: 
71:     The style name of 'default' is reserved for reverting back to
72:     the default style settings.
73: 
74:     Parameters
75:     ----------
76:     style : str, dict, or list
77:         A style specification. Valid options are:
78: 
79:         +------+-------------------------------------------------------------+
80:         | str  | The name of a style or a path/URL to a style file. For a    |
81:         |      | list of available style names, see `style.available`.       |
82:         +------+-------------------------------------------------------------+
83:         | dict | Dictionary with valid key/value pairs for                   |
84:         |      | `matplotlib.rcParams`.                                      |
85:         +------+-------------------------------------------------------------+
86:         | list | A list of style specifiers (str or dict) applied from first |
87:         |      | to last in the list.                                        |
88:         +------+-------------------------------------------------------------+
89: 
90: 
91:     '''
92:     style_alias = {'mpl20': 'default',
93:                    'mpl15': 'classic'}
94:     if isinstance(style, six.string_types) or hasattr(style, 'keys'):
95:         # If name is a single str or dict, make it a single element list.
96:         styles = [style]
97:     else:
98:         styles = style
99: 
100:     styles = (style_alias.get(s, s)
101:               if isinstance(s, six.string_types)
102:               else s
103:               for s in styles)
104:     for style in styles:
105:         if not isinstance(style, six.string_types):
106:             _apply_style(style)
107:         elif style == 'default':
108:             _apply_style(rcParamsDefault, warn=False)
109:         elif style in library:
110:             _apply_style(library[style])
111:         else:
112:             try:
113:                 rc = rc_params_from_file(style, use_default_template=False)
114:                 _apply_style(rc)
115:             except IOError:
116:                 msg = ("'%s' not found in the style library and input is "
117:                        "not a valid URL or path. See `style.available` for "
118:                        "list of available styles.")
119:                 raise IOError(msg % style)
120: 
121: 
122: @contextlib.contextmanager
123: def context(style, after_reset=False):
124:     '''Context manager for using style settings temporarily.
125: 
126:     Parameters
127:     ----------
128:     style : str, dict, or list
129:         A style specification. Valid options are:
130: 
131:         +------+-------------------------------------------------------------+
132:         | str  | The name of a style or a path/URL to a style file. For a    |
133:         |      | list of available style names, see `style.available`.       |
134:         +------+-------------------------------------------------------------+
135:         | dict | Dictionary with valid key/value pairs for                   |
136:         |      | `matplotlib.rcParams`.                                      |
137:         +------+-------------------------------------------------------------+
138:         | list | A list of style specifiers (str or dict) applied from first |
139:         |      | to last in the list.                                        |
140:         +------+-------------------------------------------------------------+
141: 
142:     after_reset : bool
143:         If True, apply style after resetting settings to their defaults;
144:         otherwise, apply style on top of the current settings.
145:     '''
146:     initial_settings = mpl.rcParams.copy()
147:     if after_reset:
148:         mpl.rcdefaults()
149:     try:
150:         use(style)
151:     except:
152:         # Restore original settings before raising errors during the update.
153:         mpl.rcParams.update(initial_settings)
154:         raise
155:     else:
156:         yield
157:     finally:
158:         mpl.rcParams.update(initial_settings)
159: 
160: 
161: def load_base_library():
162:     '''Load style library defined in this package.'''
163:     library = dict()
164:     library.update(read_style_directory(BASE_LIBRARY_PATH))
165:     return library
166: 
167: 
168: def iter_user_libraries():
169:     for stylelib_path in USER_LIBRARY_PATHS:
170:         stylelib_path = os.path.expanduser(stylelib_path)
171:         if os.path.exists(stylelib_path) and os.path.isdir(stylelib_path):
172:             yield stylelib_path
173: 
174: 
175: def update_user_library(library):
176:     '''Update style library with user-defined rc files'''
177:     for stylelib_path in iter_user_libraries():
178:         styles = read_style_directory(stylelib_path)
179:         update_nested_dict(library, styles)
180:     return library
181: 
182: 
183: def iter_style_files(style_dir):
184:     '''Yield file path and name of styles in the given directory.'''
185:     for path in os.listdir(style_dir):
186:         filename = os.path.basename(path)
187:         if is_style_file(filename):
188:             match = STYLE_FILE_PATTERN.match(filename)
189:             path = os.path.abspath(os.path.join(style_dir, path))
190:             yield path, match.groups()[0]
191: 
192: 
193: def read_style_directory(style_dir):
194:     '''Return dictionary of styles defined in `style_dir`.'''
195:     styles = dict()
196:     for path, name in iter_style_files(style_dir):
197:         with warnings.catch_warnings(record=True) as warns:
198:             styles[name] = rc_params_from_file(path,
199:                                                use_default_template=False)
200: 
201:         for w in warns:
202:             message = 'In %s: %s' % (path, w.message)
203:             warnings.warn(message)
204: 
205:     return styles
206: 
207: 
208: def update_nested_dict(main_dict, new_dict):
209:     '''Update nested dict (only level of nesting) with new values.
210: 
211:     Unlike dict.update, this assumes that the values of the parent dict are
212:     dicts (or dict-like), so you shouldn't replace the nested dict if it
213:     already exists. Instead you should update the sub-dict.
214:     '''
215:     # update named styles specified by user
216:     for name, rc_dict in six.iteritems(new_dict):
217:         if name in main_dict:
218:             main_dict[name].update(rc_dict)
219:         else:
220:             main_dict[name] = rc_dict
221:     return main_dict
222: 
223: 
224: # Load style library
225: # ==================
226: _base_library = load_base_library()
227: 
228: library = None
229: available = []
230: 
231: 
232: def reload_library():
233:     '''Reload style library.'''
234:     global library
235:     available[:] = library = update_user_library(_base_library)
236: reload_library()
237: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/style/')
import_287907 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_287907) is not StypyTypeError):

    if (import_287907 != 'pyd_module'):
        __import__(import_287907)
        sys_modules_287908 = sys.modules[import_287907]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_287908.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_287907)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/style/')

unicode_287909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'unicode', u'\nCore functions and attributes for the matplotlib style library:\n\n``use``\n    Select style sheet to override the current matplotlib settings.\n``context``\n    Context manager to use a style sheet temporarily.\n``available``\n    List available style sheets.\n``library``\n    A dictionary of style names and matplotlib settings.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import os' statement (line 18)
import os

import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import re' statement (line 19)
import re

import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import contextlib' statement (line 20)
import contextlib

import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'contextlib', contextlib, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'import warnings' statement (line 21)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'import matplotlib' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/style/')
import_287910 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'matplotlib')

if (type(import_287910) is not StypyTypeError):

    if (import_287910 != 'pyd_module'):
        __import__(import_287910)
        sys_modules_287911 = sys.modules[import_287910]
        import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'mpl', sys_modules_287911.module_type_store, module_type_store)
    else:
        import matplotlib as mpl

        import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'mpl', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'matplotlib', import_287910)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/style/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from matplotlib import cbook' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/style/')
import_287912 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib')

if (type(import_287912) is not StypyTypeError):

    if (import_287912 != 'pyd_module'):
        __import__(import_287912)
        sys_modules_287913 = sys.modules[import_287912]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib', sys_modules_287913.module_type_store, module_type_store, ['cbook'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_287913, sys_modules_287913.module_type_store, module_type_store)
    else:
        from matplotlib import cbook

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib', None, module_type_store, ['cbook'], [cbook])

else:
    # Assigning a type to the variable 'matplotlib' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib', import_287912)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/style/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from matplotlib import rc_params_from_file, rcParamsDefault' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/style/')
import_287914 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib')

if (type(import_287914) is not StypyTypeError):

    if (import_287914 != 'pyd_module'):
        __import__(import_287914)
        sys_modules_287915 = sys.modules[import_287914]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib', sys_modules_287915.module_type_store, module_type_store, ['rc_params_from_file', 'rcParamsDefault'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_287915, sys_modules_287915.module_type_store, module_type_store)
    else:
        from matplotlib import rc_params_from_file, rcParamsDefault

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib', None, module_type_store, ['rc_params_from_file', 'rcParamsDefault'], [rc_params_from_file, rcParamsDefault])

else:
    # Assigning a type to the variable 'matplotlib' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib', import_287914)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/style/')


# Assigning a List to a Name (line 28):
__all__ = [u'use', u'context', u'available', u'library', u'reload_library']
module_type_store.set_exportable_members([u'use', u'context', u'available', u'library', u'reload_library'])

# Obtaining an instance of the builtin type 'list' (line 28)
list_287916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 28)
# Adding element type (line 28)
unicode_287917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 11), 'unicode', u'use')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_287916, unicode_287917)
# Adding element type (line 28)
unicode_287918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 18), 'unicode', u'context')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_287916, unicode_287918)
# Adding element type (line 28)
unicode_287919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 29), 'unicode', u'available')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_287916, unicode_287919)
# Adding element type (line 28)
unicode_287920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 42), 'unicode', u'library')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_287916, unicode_287920)
# Adding element type (line 28)
unicode_287921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 53), 'unicode', u'reload_library')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_287916, unicode_287921)

# Assigning a type to the variable '__all__' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), '__all__', list_287916)

# Assigning a Call to a Name (line 31):

# Call to join(...): (line 31)
# Processing the call arguments (line 31)

# Call to get_data_path(...): (line 31)
# Processing the call keyword arguments (line 31)
kwargs_287927 = {}
# Getting the type of 'mpl' (line 31)
mpl_287925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 33), 'mpl', False)
# Obtaining the member 'get_data_path' of a type (line 31)
get_data_path_287926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 33), mpl_287925, 'get_data_path')
# Calling get_data_path(args, kwargs) (line 31)
get_data_path_call_result_287928 = invoke(stypy.reporting.localization.Localization(__file__, 31, 33), get_data_path_287926, *[], **kwargs_287927)

unicode_287929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 54), 'unicode', u'stylelib')
# Processing the call keyword arguments (line 31)
kwargs_287930 = {}
# Getting the type of 'os' (line 31)
os_287922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'os', False)
# Obtaining the member 'path' of a type (line 31)
path_287923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 20), os_287922, 'path')
# Obtaining the member 'join' of a type (line 31)
join_287924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 20), path_287923, 'join')
# Calling join(args, kwargs) (line 31)
join_call_result_287931 = invoke(stypy.reporting.localization.Localization(__file__, 31, 20), join_287924, *[get_data_path_call_result_287928, unicode_287929], **kwargs_287930)

# Assigning a type to the variable 'BASE_LIBRARY_PATH' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'BASE_LIBRARY_PATH', join_call_result_287931)

# Assigning a List to a Name (line 33):

# Obtaining an instance of the builtin type 'list' (line 33)
list_287932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 33)
# Adding element type (line 33)

# Call to join(...): (line 33)
# Processing the call arguments (line 33)

# Call to _get_configdir(...): (line 33)
# Processing the call keyword arguments (line 33)
kwargs_287938 = {}
# Getting the type of 'mpl' (line 33)
mpl_287936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 35), 'mpl', False)
# Obtaining the member '_get_configdir' of a type (line 33)
_get_configdir_287937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 35), mpl_287936, '_get_configdir')
# Calling _get_configdir(args, kwargs) (line 33)
_get_configdir_call_result_287939 = invoke(stypy.reporting.localization.Localization(__file__, 33, 35), _get_configdir_287937, *[], **kwargs_287938)

unicode_287940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 57), 'unicode', u'stylelib')
# Processing the call keyword arguments (line 33)
kwargs_287941 = {}
# Getting the type of 'os' (line 33)
os_287933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 22), 'os', False)
# Obtaining the member 'path' of a type (line 33)
path_287934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 22), os_287933, 'path')
# Obtaining the member 'join' of a type (line 33)
join_287935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 22), path_287934, 'join')
# Calling join(args, kwargs) (line 33)
join_call_result_287942 = invoke(stypy.reporting.localization.Localization(__file__, 33, 22), join_287935, *[_get_configdir_call_result_287939, unicode_287940], **kwargs_287941)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), list_287932, join_call_result_287942)

# Assigning a type to the variable 'USER_LIBRARY_PATHS' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'USER_LIBRARY_PATHS', list_287932)

# Assigning a Str to a Name (line 34):
unicode_287943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 18), 'unicode', u'mplstyle')
# Assigning a type to the variable 'STYLE_EXTENSION' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'STYLE_EXTENSION', unicode_287943)

# Assigning a Call to a Name (line 35):

# Call to compile(...): (line 35)
# Processing the call arguments (line 35)
unicode_287946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 32), 'unicode', u'([\\S]+).%s$')
# Getting the type of 'STYLE_EXTENSION' (line 35)
STYLE_EXTENSION_287947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 49), 'STYLE_EXTENSION', False)
# Applying the binary operator '%' (line 35)
result_mod_287948 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 32), '%', unicode_287946, STYLE_EXTENSION_287947)

# Processing the call keyword arguments (line 35)
kwargs_287949 = {}
# Getting the type of 're' (line 35)
re_287944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 21), 're', False)
# Obtaining the member 'compile' of a type (line 35)
compile_287945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 21), re_287944, 'compile')
# Calling compile(args, kwargs) (line 35)
compile_call_result_287950 = invoke(stypy.reporting.localization.Localization(__file__, 35, 21), compile_287945, *[result_mod_287948], **kwargs_287949)

# Assigning a type to the variable 'STYLE_FILE_PATTERN' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'STYLE_FILE_PATTERN', compile_call_result_287950)

# Assigning a Set to a Name (line 39):

# Obtaining an instance of the builtin type 'set' (line 39)
set_287951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 18), 'set')
# Adding type elements to the builtin type 'set' instance (line 39)
# Adding element type (line 39)
unicode_287952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 4), 'unicode', u'interactive')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 18), set_287951, unicode_287952)
# Adding element type (line 39)
unicode_287953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 19), 'unicode', u'backend')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 18), set_287951, unicode_287953)
# Adding element type (line 39)
unicode_287954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 30), 'unicode', u'backend.qt4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 18), set_287951, unicode_287954)
# Adding element type (line 39)
unicode_287955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 45), 'unicode', u'webagg.port')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 18), set_287951, unicode_287955)
# Adding element type (line 39)
unicode_287956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 4), 'unicode', u'webagg.port_retries')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 18), set_287951, unicode_287956)
# Adding element type (line 39)
unicode_287957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 27), 'unicode', u'webagg.open_in_browser')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 18), set_287951, unicode_287957)
# Adding element type (line 39)
unicode_287958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 53), 'unicode', u'backend_fallback')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 18), set_287951, unicode_287958)
# Adding element type (line 39)
unicode_287959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 4), 'unicode', u'toolbar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 18), set_287951, unicode_287959)
# Adding element type (line 39)
unicode_287960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 15), 'unicode', u'timezone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 18), set_287951, unicode_287960)
# Adding element type (line 39)
unicode_287961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 27), 'unicode', u'datapath')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 18), set_287951, unicode_287961)
# Adding element type (line 39)
unicode_287962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 39), 'unicode', u'figure.max_open_warning')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 18), set_287951, unicode_287962)
# Adding element type (line 39)
unicode_287963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 4), 'unicode', u'savefig.directory')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 18), set_287951, unicode_287963)
# Adding element type (line 39)
unicode_287964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 25), 'unicode', u'tk.window_focus')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 18), set_287951, unicode_287964)
# Adding element type (line 39)
unicode_287965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 44), 'unicode', u'docstring.hardcopy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 18), set_287951, unicode_287965)

# Assigning a type to the variable 'STYLE_BLACKLIST' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'STYLE_BLACKLIST', set_287951)

@norecursion
def _remove_blacklisted_style_params(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 46)
    True_287966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 45), 'True')
    defaults = [True_287966]
    # Create a new context for function '_remove_blacklisted_style_params'
    module_type_store = module_type_store.open_function_context('_remove_blacklisted_style_params', 46, 0, False)
    
    # Passed parameters checking function
    _remove_blacklisted_style_params.stypy_localization = localization
    _remove_blacklisted_style_params.stypy_type_of_self = None
    _remove_blacklisted_style_params.stypy_type_store = module_type_store
    _remove_blacklisted_style_params.stypy_function_name = '_remove_blacklisted_style_params'
    _remove_blacklisted_style_params.stypy_param_names_list = ['d', 'warn']
    _remove_blacklisted_style_params.stypy_varargs_param_name = None
    _remove_blacklisted_style_params.stypy_kwargs_param_name = None
    _remove_blacklisted_style_params.stypy_call_defaults = defaults
    _remove_blacklisted_style_params.stypy_call_varargs = varargs
    _remove_blacklisted_style_params.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_remove_blacklisted_style_params', ['d', 'warn'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_remove_blacklisted_style_params', localization, ['d', 'warn'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_remove_blacklisted_style_params(...)' code ##################

    
    # Assigning a Dict to a Name (line 47):
    
    # Obtaining an instance of the builtin type 'dict' (line 47)
    dict_287967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 47)
    
    # Assigning a type to the variable 'o' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'o', dict_287967)
    
    
    # Call to items(...): (line 48)
    # Processing the call keyword arguments (line 48)
    kwargs_287970 = {}
    # Getting the type of 'd' (line 48)
    d_287968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 20), 'd', False)
    # Obtaining the member 'items' of a type (line 48)
    items_287969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 20), d_287968, 'items')
    # Calling items(args, kwargs) (line 48)
    items_call_result_287971 = invoke(stypy.reporting.localization.Localization(__file__, 48, 20), items_287969, *[], **kwargs_287970)
    
    # Testing the type of a for loop iterable (line 48)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 48, 4), items_call_result_287971)
    # Getting the type of the for loop variable (line 48)
    for_loop_var_287972 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 48, 4), items_call_result_287971)
    # Assigning a type to the variable 'key' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 4), for_loop_var_287972))
    # Assigning a type to the variable 'val' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 4), for_loop_var_287972))
    # SSA begins for a for statement (line 48)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'key' (line 49)
    key_287973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'key')
    # Getting the type of 'STYLE_BLACKLIST' (line 49)
    STYLE_BLACKLIST_287974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 18), 'STYLE_BLACKLIST')
    # Applying the binary operator 'in' (line 49)
    result_contains_287975 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 11), 'in', key_287973, STYLE_BLACKLIST_287974)
    
    # Testing the type of an if condition (line 49)
    if_condition_287976 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 8), result_contains_287975)
    # Assigning a type to the variable 'if_condition_287976' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'if_condition_287976', if_condition_287976)
    # SSA begins for if statement (line 49)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'warn' (line 50)
    warn_287977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'warn')
    # Testing the type of an if condition (line 50)
    if_condition_287978 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 12), warn_287977)
    # Assigning a type to the variable 'if_condition_287978' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'if_condition_287978', if_condition_287978)
    # SSA begins for if statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 51)
    # Processing the call arguments (line 51)
    
    # Call to format(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'key' (line 53)
    key_287983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 49), 'key', False)
    # Processing the call keyword arguments (line 52)
    kwargs_287984 = {}
    unicode_287981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 20), 'unicode', u"Style includes a parameter, '{0}', that is not related to style.  Ignoring")
    # Obtaining the member 'format' of a type (line 52)
    format_287982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 20), unicode_287981, 'format')
    # Calling format(args, kwargs) (line 52)
    format_call_result_287985 = invoke(stypy.reporting.localization.Localization(__file__, 52, 20), format_287982, *[key_287983], **kwargs_287984)
    
    # Processing the call keyword arguments (line 51)
    kwargs_287986 = {}
    # Getting the type of 'warnings' (line 51)
    warnings_287979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 51)
    warn_287980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 16), warnings_287979, 'warn')
    # Calling warn(args, kwargs) (line 51)
    warn_call_result_287987 = invoke(stypy.reporting.localization.Localization(__file__, 51, 16), warn_287980, *[format_call_result_287985], **kwargs_287986)
    
    # SSA join for if statement (line 50)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 49)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Subscript (line 55):
    # Getting the type of 'val' (line 55)
    val_287988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'val')
    # Getting the type of 'o' (line 55)
    o_287989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'o')
    # Getting the type of 'key' (line 55)
    key_287990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 14), 'key')
    # Storing an element on a container (line 55)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 12), o_287989, (key_287990, val_287988))
    # SSA join for if statement (line 49)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'o' (line 56)
    o_287991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'o')
    # Assigning a type to the variable 'stypy_return_type' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type', o_287991)
    
    # ################# End of '_remove_blacklisted_style_params(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_remove_blacklisted_style_params' in the type store
    # Getting the type of 'stypy_return_type' (line 46)
    stypy_return_type_287992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_287992)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_remove_blacklisted_style_params'
    return stypy_return_type_287992

# Assigning a type to the variable '_remove_blacklisted_style_params' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), '_remove_blacklisted_style_params', _remove_blacklisted_style_params)

@norecursion
def is_style_file(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_style_file'
    module_type_store = module_type_store.open_function_context('is_style_file', 59, 0, False)
    
    # Passed parameters checking function
    is_style_file.stypy_localization = localization
    is_style_file.stypy_type_of_self = None
    is_style_file.stypy_type_store = module_type_store
    is_style_file.stypy_function_name = 'is_style_file'
    is_style_file.stypy_param_names_list = ['filename']
    is_style_file.stypy_varargs_param_name = None
    is_style_file.stypy_kwargs_param_name = None
    is_style_file.stypy_call_defaults = defaults
    is_style_file.stypy_call_varargs = varargs
    is_style_file.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_style_file', ['filename'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_style_file', localization, ['filename'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_style_file(...)' code ##################

    unicode_287993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 4), 'unicode', u'Return True if the filename looks like a style file.')
    
    
    # Call to match(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'filename' (line 61)
    filename_287996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 36), 'filename', False)
    # Processing the call keyword arguments (line 61)
    kwargs_287997 = {}
    # Getting the type of 'STYLE_FILE_PATTERN' (line 61)
    STYLE_FILE_PATTERN_287994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'STYLE_FILE_PATTERN', False)
    # Obtaining the member 'match' of a type (line 61)
    match_287995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 11), STYLE_FILE_PATTERN_287994, 'match')
    # Calling match(args, kwargs) (line 61)
    match_call_result_287998 = invoke(stypy.reporting.localization.Localization(__file__, 61, 11), match_287995, *[filename_287996], **kwargs_287997)
    
    # Getting the type of 'None' (line 61)
    None_287999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 53), 'None')
    # Applying the binary operator 'isnot' (line 61)
    result_is_not_288000 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 11), 'isnot', match_call_result_287998, None_287999)
    
    # Assigning a type to the variable 'stypy_return_type' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type', result_is_not_288000)
    
    # ################# End of 'is_style_file(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_style_file' in the type store
    # Getting the type of 'stypy_return_type' (line 59)
    stypy_return_type_288001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288001)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_style_file'
    return stypy_return_type_288001

# Assigning a type to the variable 'is_style_file' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'is_style_file', is_style_file)

@norecursion
def _apply_style(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 64)
    True_288002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 25), 'True')
    defaults = [True_288002]
    # Create a new context for function '_apply_style'
    module_type_store = module_type_store.open_function_context('_apply_style', 64, 0, False)
    
    # Passed parameters checking function
    _apply_style.stypy_localization = localization
    _apply_style.stypy_type_of_self = None
    _apply_style.stypy_type_store = module_type_store
    _apply_style.stypy_function_name = '_apply_style'
    _apply_style.stypy_param_names_list = ['d', 'warn']
    _apply_style.stypy_varargs_param_name = None
    _apply_style.stypy_kwargs_param_name = None
    _apply_style.stypy_call_defaults = defaults
    _apply_style.stypy_call_varargs = varargs
    _apply_style.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_apply_style', ['d', 'warn'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_apply_style', localization, ['d', 'warn'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_apply_style(...)' code ##################

    
    # Call to update(...): (line 65)
    # Processing the call arguments (line 65)
    
    # Call to _remove_blacklisted_style_params(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'd' (line 65)
    d_288007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 57), 'd', False)
    # Processing the call keyword arguments (line 65)
    # Getting the type of 'warn' (line 65)
    warn_288008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 65), 'warn', False)
    keyword_288009 = warn_288008
    kwargs_288010 = {'warn': keyword_288009}
    # Getting the type of '_remove_blacklisted_style_params' (line 65)
    _remove_blacklisted_style_params_288006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 24), '_remove_blacklisted_style_params', False)
    # Calling _remove_blacklisted_style_params(args, kwargs) (line 65)
    _remove_blacklisted_style_params_call_result_288011 = invoke(stypy.reporting.localization.Localization(__file__, 65, 24), _remove_blacklisted_style_params_288006, *[d_288007], **kwargs_288010)
    
    # Processing the call keyword arguments (line 65)
    kwargs_288012 = {}
    # Getting the type of 'mpl' (line 65)
    mpl_288003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'mpl', False)
    # Obtaining the member 'rcParams' of a type (line 65)
    rcParams_288004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 4), mpl_288003, 'rcParams')
    # Obtaining the member 'update' of a type (line 65)
    update_288005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 4), rcParams_288004, 'update')
    # Calling update(args, kwargs) (line 65)
    update_call_result_288013 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), update_288005, *[_remove_blacklisted_style_params_call_result_288011], **kwargs_288012)
    
    
    # ################# End of '_apply_style(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_apply_style' in the type store
    # Getting the type of 'stypy_return_type' (line 64)
    stypy_return_type_288014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288014)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_apply_style'
    return stypy_return_type_288014

# Assigning a type to the variable '_apply_style' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), '_apply_style', _apply_style)

@norecursion
def use(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'use'
    module_type_store = module_type_store.open_function_context('use', 68, 0, False)
    
    # Passed parameters checking function
    use.stypy_localization = localization
    use.stypy_type_of_self = None
    use.stypy_type_store = module_type_store
    use.stypy_function_name = 'use'
    use.stypy_param_names_list = ['style']
    use.stypy_varargs_param_name = None
    use.stypy_kwargs_param_name = None
    use.stypy_call_defaults = defaults
    use.stypy_call_varargs = varargs
    use.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'use', ['style'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'use', localization, ['style'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'use(...)' code ##################

    unicode_288015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, (-1)), 'unicode', u"Use matplotlib style settings from a style specification.\n\n    The style name of 'default' is reserved for reverting back to\n    the default style settings.\n\n    Parameters\n    ----------\n    style : str, dict, or list\n        A style specification. Valid options are:\n\n        +------+-------------------------------------------------------------+\n        | str  | The name of a style or a path/URL to a style file. For a    |\n        |      | list of available style names, see `style.available`.       |\n        +------+-------------------------------------------------------------+\n        | dict | Dictionary with valid key/value pairs for                   |\n        |      | `matplotlib.rcParams`.                                      |\n        +------+-------------------------------------------------------------+\n        | list | A list of style specifiers (str or dict) applied from first |\n        |      | to last in the list.                                        |\n        +------+-------------------------------------------------------------+\n\n\n    ")
    
    # Assigning a Dict to a Name (line 92):
    
    # Obtaining an instance of the builtin type 'dict' (line 92)
    dict_288016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 18), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 92)
    # Adding element type (key, value) (line 92)
    unicode_288017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 19), 'unicode', u'mpl20')
    unicode_288018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 28), 'unicode', u'default')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 18), dict_288016, (unicode_288017, unicode_288018))
    # Adding element type (key, value) (line 92)
    unicode_288019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 19), 'unicode', u'mpl15')
    unicode_288020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 28), 'unicode', u'classic')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 18), dict_288016, (unicode_288019, unicode_288020))
    
    # Assigning a type to the variable 'style_alias' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'style_alias', dict_288016)
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'style' (line 94)
    style_288022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 18), 'style', False)
    # Getting the type of 'six' (line 94)
    six_288023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 25), 'six', False)
    # Obtaining the member 'string_types' of a type (line 94)
    string_types_288024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 25), six_288023, 'string_types')
    # Processing the call keyword arguments (line 94)
    kwargs_288025 = {}
    # Getting the type of 'isinstance' (line 94)
    isinstance_288021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 94)
    isinstance_call_result_288026 = invoke(stypy.reporting.localization.Localization(__file__, 94, 7), isinstance_288021, *[style_288022, string_types_288024], **kwargs_288025)
    
    
    # Call to hasattr(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'style' (line 94)
    style_288028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 54), 'style', False)
    unicode_288029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 61), 'unicode', u'keys')
    # Processing the call keyword arguments (line 94)
    kwargs_288030 = {}
    # Getting the type of 'hasattr' (line 94)
    hasattr_288027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 46), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 94)
    hasattr_call_result_288031 = invoke(stypy.reporting.localization.Localization(__file__, 94, 46), hasattr_288027, *[style_288028, unicode_288029], **kwargs_288030)
    
    # Applying the binary operator 'or' (line 94)
    result_or_keyword_288032 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 7), 'or', isinstance_call_result_288026, hasattr_call_result_288031)
    
    # Testing the type of an if condition (line 94)
    if_condition_288033 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 4), result_or_keyword_288032)
    # Assigning a type to the variable 'if_condition_288033' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'if_condition_288033', if_condition_288033)
    # SSA begins for if statement (line 94)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 96):
    
    # Obtaining an instance of the builtin type 'list' (line 96)
    list_288034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 96)
    # Adding element type (line 96)
    # Getting the type of 'style' (line 96)
    style_288035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 18), 'style')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 17), list_288034, style_288035)
    
    # Assigning a type to the variable 'styles' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'styles', list_288034)
    # SSA branch for the else part of an if statement (line 94)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 98):
    # Getting the type of 'style' (line 98)
    style_288036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 17), 'style')
    # Assigning a type to the variable 'styles' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'styles', style_288036)
    # SSA join for if statement (line 94)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a GeneratorExp to a Name (line 100):
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 100, 14, True)
    # Calculating comprehension expression
    # Getting the type of 'styles' (line 103)
    styles_288051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 23), 'styles')
    comprehension_288052 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 14), styles_288051)
    # Assigning a type to the variable 's' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 14), 's', comprehension_288052)
    
    
    # Call to isinstance(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 's' (line 101)
    s_288038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 28), 's', False)
    # Getting the type of 'six' (line 101)
    six_288039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'six', False)
    # Obtaining the member 'string_types' of a type (line 101)
    string_types_288040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 31), six_288039, 'string_types')
    # Processing the call keyword arguments (line 101)
    kwargs_288041 = {}
    # Getting the type of 'isinstance' (line 101)
    isinstance_288037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 101)
    isinstance_call_result_288042 = invoke(stypy.reporting.localization.Localization(__file__, 101, 17), isinstance_288037, *[s_288038, string_types_288040], **kwargs_288041)
    
    # Testing the type of an if expression (line 100)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 14), isinstance_call_result_288042)
    # SSA begins for if expression (line 100)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to get(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 's' (line 100)
    s_288045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 30), 's', False)
    # Getting the type of 's' (line 100)
    s_288046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 33), 's', False)
    # Processing the call keyword arguments (line 100)
    kwargs_288047 = {}
    # Getting the type of 'style_alias' (line 100)
    style_alias_288043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 14), 'style_alias', False)
    # Obtaining the member 'get' of a type (line 100)
    get_288044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 14), style_alias_288043, 'get')
    # Calling get(args, kwargs) (line 100)
    get_call_result_288048 = invoke(stypy.reporting.localization.Localization(__file__, 100, 14), get_288044, *[s_288045, s_288046], **kwargs_288047)
    
    # SSA branch for the else part of an if expression (line 100)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 's' (line 102)
    s_288049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 19), 's')
    # SSA join for if expression (line 100)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_288050 = union_type.UnionType.add(get_call_result_288048, s_288049)
    
    list_288053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 14), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 14), list_288053, if_exp_288050)
    # Assigning a type to the variable 'styles' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'styles', list_288053)
    
    # Getting the type of 'styles' (line 104)
    styles_288054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 17), 'styles')
    # Testing the type of a for loop iterable (line 104)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 104, 4), styles_288054)
    # Getting the type of the for loop variable (line 104)
    for_loop_var_288055 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 104, 4), styles_288054)
    # Assigning a type to the variable 'style' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'style', for_loop_var_288055)
    # SSA begins for a for statement (line 104)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Call to isinstance(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'style' (line 105)
    style_288057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 26), 'style', False)
    # Getting the type of 'six' (line 105)
    six_288058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 33), 'six', False)
    # Obtaining the member 'string_types' of a type (line 105)
    string_types_288059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 33), six_288058, 'string_types')
    # Processing the call keyword arguments (line 105)
    kwargs_288060 = {}
    # Getting the type of 'isinstance' (line 105)
    isinstance_288056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 105)
    isinstance_call_result_288061 = invoke(stypy.reporting.localization.Localization(__file__, 105, 15), isinstance_288056, *[style_288057, string_types_288059], **kwargs_288060)
    
    # Applying the 'not' unary operator (line 105)
    result_not__288062 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 11), 'not', isinstance_call_result_288061)
    
    # Testing the type of an if condition (line 105)
    if_condition_288063 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 8), result_not__288062)
    # Assigning a type to the variable 'if_condition_288063' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'if_condition_288063', if_condition_288063)
    # SSA begins for if statement (line 105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _apply_style(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'style' (line 106)
    style_288065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 25), 'style', False)
    # Processing the call keyword arguments (line 106)
    kwargs_288066 = {}
    # Getting the type of '_apply_style' (line 106)
    _apply_style_288064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), '_apply_style', False)
    # Calling _apply_style(args, kwargs) (line 106)
    _apply_style_call_result_288067 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), _apply_style_288064, *[style_288065], **kwargs_288066)
    
    # SSA branch for the else part of an if statement (line 105)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'style' (line 107)
    style_288068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 13), 'style')
    unicode_288069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 22), 'unicode', u'default')
    # Applying the binary operator '==' (line 107)
    result_eq_288070 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 13), '==', style_288068, unicode_288069)
    
    # Testing the type of an if condition (line 107)
    if_condition_288071 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 13), result_eq_288070)
    # Assigning a type to the variable 'if_condition_288071' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 13), 'if_condition_288071', if_condition_288071)
    # SSA begins for if statement (line 107)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _apply_style(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'rcParamsDefault' (line 108)
    rcParamsDefault_288073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 25), 'rcParamsDefault', False)
    # Processing the call keyword arguments (line 108)
    # Getting the type of 'False' (line 108)
    False_288074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 47), 'False', False)
    keyword_288075 = False_288074
    kwargs_288076 = {'warn': keyword_288075}
    # Getting the type of '_apply_style' (line 108)
    _apply_style_288072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), '_apply_style', False)
    # Calling _apply_style(args, kwargs) (line 108)
    _apply_style_call_result_288077 = invoke(stypy.reporting.localization.Localization(__file__, 108, 12), _apply_style_288072, *[rcParamsDefault_288073], **kwargs_288076)
    
    # SSA branch for the else part of an if statement (line 107)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'style' (line 109)
    style_288078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 13), 'style')
    # Getting the type of 'library' (line 109)
    library_288079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 22), 'library')
    # Applying the binary operator 'in' (line 109)
    result_contains_288080 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 13), 'in', style_288078, library_288079)
    
    # Testing the type of an if condition (line 109)
    if_condition_288081 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 13), result_contains_288080)
    # Assigning a type to the variable 'if_condition_288081' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 13), 'if_condition_288081', if_condition_288081)
    # SSA begins for if statement (line 109)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _apply_style(...): (line 110)
    # Processing the call arguments (line 110)
    
    # Obtaining the type of the subscript
    # Getting the type of 'style' (line 110)
    style_288083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 33), 'style', False)
    # Getting the type of 'library' (line 110)
    library_288084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 25), 'library', False)
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___288085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 25), library_288084, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_288086 = invoke(stypy.reporting.localization.Localization(__file__, 110, 25), getitem___288085, style_288083)
    
    # Processing the call keyword arguments (line 110)
    kwargs_288087 = {}
    # Getting the type of '_apply_style' (line 110)
    _apply_style_288082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), '_apply_style', False)
    # Calling _apply_style(args, kwargs) (line 110)
    _apply_style_call_result_288088 = invoke(stypy.reporting.localization.Localization(__file__, 110, 12), _apply_style_288082, *[subscript_call_result_288086], **kwargs_288087)
    
    # SSA branch for the else part of an if statement (line 109)
    module_type_store.open_ssa_branch('else')
    
    
    # SSA begins for try-except statement (line 112)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 113):
    
    # Call to rc_params_from_file(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'style' (line 113)
    style_288090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 41), 'style', False)
    # Processing the call keyword arguments (line 113)
    # Getting the type of 'False' (line 113)
    False_288091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 69), 'False', False)
    keyword_288092 = False_288091
    kwargs_288093 = {'use_default_template': keyword_288092}
    # Getting the type of 'rc_params_from_file' (line 113)
    rc_params_from_file_288089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 21), 'rc_params_from_file', False)
    # Calling rc_params_from_file(args, kwargs) (line 113)
    rc_params_from_file_call_result_288094 = invoke(stypy.reporting.localization.Localization(__file__, 113, 21), rc_params_from_file_288089, *[style_288090], **kwargs_288093)
    
    # Assigning a type to the variable 'rc' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'rc', rc_params_from_file_call_result_288094)
    
    # Call to _apply_style(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'rc' (line 114)
    rc_288096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 29), 'rc', False)
    # Processing the call keyword arguments (line 114)
    kwargs_288097 = {}
    # Getting the type of '_apply_style' (line 114)
    _apply_style_288095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), '_apply_style', False)
    # Calling _apply_style(args, kwargs) (line 114)
    _apply_style_call_result_288098 = invoke(stypy.reporting.localization.Localization(__file__, 114, 16), _apply_style_288095, *[rc_288096], **kwargs_288097)
    
    # SSA branch for the except part of a try statement (line 112)
    # SSA branch for the except 'IOError' branch of a try statement (line 112)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Str to a Name (line 116):
    unicode_288099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 23), 'unicode', u"'%s' not found in the style library and input is not a valid URL or path. See `style.available` for list of available styles.")
    # Assigning a type to the variable 'msg' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'msg', unicode_288099)
    
    # Call to IOError(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'msg' (line 119)
    msg_288101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 30), 'msg', False)
    # Getting the type of 'style' (line 119)
    style_288102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 36), 'style', False)
    # Applying the binary operator '%' (line 119)
    result_mod_288103 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 30), '%', msg_288101, style_288102)
    
    # Processing the call keyword arguments (line 119)
    kwargs_288104 = {}
    # Getting the type of 'IOError' (line 119)
    IOError_288100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 22), 'IOError', False)
    # Calling IOError(args, kwargs) (line 119)
    IOError_call_result_288105 = invoke(stypy.reporting.localization.Localization(__file__, 119, 22), IOError_288100, *[result_mod_288103], **kwargs_288104)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 119, 16), IOError_call_result_288105, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 112)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 109)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 107)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 105)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'use(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'use' in the type store
    # Getting the type of 'stypy_return_type' (line 68)
    stypy_return_type_288106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288106)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'use'
    return stypy_return_type_288106

# Assigning a type to the variable 'use' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'use', use)

@norecursion
def context(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 123)
    False_288107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 31), 'False')
    defaults = [False_288107]
    # Create a new context for function 'context'
    module_type_store = module_type_store.open_function_context('context', 122, 0, False)
    
    # Passed parameters checking function
    context.stypy_localization = localization
    context.stypy_type_of_self = None
    context.stypy_type_store = module_type_store
    context.stypy_function_name = 'context'
    context.stypy_param_names_list = ['style', 'after_reset']
    context.stypy_varargs_param_name = None
    context.stypy_kwargs_param_name = None
    context.stypy_call_defaults = defaults
    context.stypy_call_varargs = varargs
    context.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'context', ['style', 'after_reset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'context', localization, ['style', 'after_reset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'context(...)' code ##################

    unicode_288108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, (-1)), 'unicode', u'Context manager for using style settings temporarily.\n\n    Parameters\n    ----------\n    style : str, dict, or list\n        A style specification. Valid options are:\n\n        +------+-------------------------------------------------------------+\n        | str  | The name of a style or a path/URL to a style file. For a    |\n        |      | list of available style names, see `style.available`.       |\n        +------+-------------------------------------------------------------+\n        | dict | Dictionary with valid key/value pairs for                   |\n        |      | `matplotlib.rcParams`.                                      |\n        +------+-------------------------------------------------------------+\n        | list | A list of style specifiers (str or dict) applied from first |\n        |      | to last in the list.                                        |\n        +------+-------------------------------------------------------------+\n\n    after_reset : bool\n        If True, apply style after resetting settings to their defaults;\n        otherwise, apply style on top of the current settings.\n    ')
    
    # Assigning a Call to a Name (line 146):
    
    # Call to copy(...): (line 146)
    # Processing the call keyword arguments (line 146)
    kwargs_288112 = {}
    # Getting the type of 'mpl' (line 146)
    mpl_288109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 23), 'mpl', False)
    # Obtaining the member 'rcParams' of a type (line 146)
    rcParams_288110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 23), mpl_288109, 'rcParams')
    # Obtaining the member 'copy' of a type (line 146)
    copy_288111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 23), rcParams_288110, 'copy')
    # Calling copy(args, kwargs) (line 146)
    copy_call_result_288113 = invoke(stypy.reporting.localization.Localization(__file__, 146, 23), copy_288111, *[], **kwargs_288112)
    
    # Assigning a type to the variable 'initial_settings' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'initial_settings', copy_call_result_288113)
    
    # Getting the type of 'after_reset' (line 147)
    after_reset_288114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 7), 'after_reset')
    # Testing the type of an if condition (line 147)
    if_condition_288115 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 4), after_reset_288114)
    # Assigning a type to the variable 'if_condition_288115' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'if_condition_288115', if_condition_288115)
    # SSA begins for if statement (line 147)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to rcdefaults(...): (line 148)
    # Processing the call keyword arguments (line 148)
    kwargs_288118 = {}
    # Getting the type of 'mpl' (line 148)
    mpl_288116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'mpl', False)
    # Obtaining the member 'rcdefaults' of a type (line 148)
    rcdefaults_288117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), mpl_288116, 'rcdefaults')
    # Calling rcdefaults(args, kwargs) (line 148)
    rcdefaults_call_result_288119 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), rcdefaults_288117, *[], **kwargs_288118)
    
    # SSA join for if statement (line 147)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Try-finally block (line 149)
    
    
    # SSA begins for try-except statement (line 149)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to use(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'style' (line 150)
    style_288121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'style', False)
    # Processing the call keyword arguments (line 150)
    kwargs_288122 = {}
    # Getting the type of 'use' (line 150)
    use_288120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'use', False)
    # Calling use(args, kwargs) (line 150)
    use_call_result_288123 = invoke(stypy.reporting.localization.Localization(__file__, 150, 8), use_288120, *[style_288121], **kwargs_288122)
    
    # SSA branch for the except part of a try statement (line 149)
    # SSA branch for the except '<any exception>' branch of a try statement (line 149)
    module_type_store.open_ssa_branch('except')
    
    # Call to update(...): (line 153)
    # Processing the call arguments (line 153)
    # Getting the type of 'initial_settings' (line 153)
    initial_settings_288127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 28), 'initial_settings', False)
    # Processing the call keyword arguments (line 153)
    kwargs_288128 = {}
    # Getting the type of 'mpl' (line 153)
    mpl_288124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'mpl', False)
    # Obtaining the member 'rcParams' of a type (line 153)
    rcParams_288125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), mpl_288124, 'rcParams')
    # Obtaining the member 'update' of a type (line 153)
    update_288126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), rcParams_288125, 'update')
    # Calling update(args, kwargs) (line 153)
    update_call_result_288129 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), update_288126, *[initial_settings_288127], **kwargs_288128)
    
    # SSA branch for the else branch of a try statement (line 149)
    module_type_store.open_ssa_branch('except else')
    # Creating a generator
    GeneratorType_288130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 8), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 8), GeneratorType_288130, None)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'stypy_return_type', GeneratorType_288130)
    # SSA join for try-except statement (line 149)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # finally branch of the try-finally block (line 149)
    
    # Call to update(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'initial_settings' (line 158)
    initial_settings_288134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 28), 'initial_settings', False)
    # Processing the call keyword arguments (line 158)
    kwargs_288135 = {}
    # Getting the type of 'mpl' (line 158)
    mpl_288131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'mpl', False)
    # Obtaining the member 'rcParams' of a type (line 158)
    rcParams_288132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), mpl_288131, 'rcParams')
    # Obtaining the member 'update' of a type (line 158)
    update_288133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), rcParams_288132, 'update')
    # Calling update(args, kwargs) (line 158)
    update_call_result_288136 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), update_288133, *[initial_settings_288134], **kwargs_288135)
    
    
    
    # ################# End of 'context(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'context' in the type store
    # Getting the type of 'stypy_return_type' (line 122)
    stypy_return_type_288137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288137)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'context'
    return stypy_return_type_288137

# Assigning a type to the variable 'context' (line 122)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'context', context)

@norecursion
def load_base_library(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'load_base_library'
    module_type_store = module_type_store.open_function_context('load_base_library', 161, 0, False)
    
    # Passed parameters checking function
    load_base_library.stypy_localization = localization
    load_base_library.stypy_type_of_self = None
    load_base_library.stypy_type_store = module_type_store
    load_base_library.stypy_function_name = 'load_base_library'
    load_base_library.stypy_param_names_list = []
    load_base_library.stypy_varargs_param_name = None
    load_base_library.stypy_kwargs_param_name = None
    load_base_library.stypy_call_defaults = defaults
    load_base_library.stypy_call_varargs = varargs
    load_base_library.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'load_base_library', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'load_base_library', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'load_base_library(...)' code ##################

    unicode_288138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 4), 'unicode', u'Load style library defined in this package.')
    
    # Assigning a Call to a Name (line 163):
    
    # Call to dict(...): (line 163)
    # Processing the call keyword arguments (line 163)
    kwargs_288140 = {}
    # Getting the type of 'dict' (line 163)
    dict_288139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 14), 'dict', False)
    # Calling dict(args, kwargs) (line 163)
    dict_call_result_288141 = invoke(stypy.reporting.localization.Localization(__file__, 163, 14), dict_288139, *[], **kwargs_288140)
    
    # Assigning a type to the variable 'library' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'library', dict_call_result_288141)
    
    # Call to update(...): (line 164)
    # Processing the call arguments (line 164)
    
    # Call to read_style_directory(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'BASE_LIBRARY_PATH' (line 164)
    BASE_LIBRARY_PATH_288145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 40), 'BASE_LIBRARY_PATH', False)
    # Processing the call keyword arguments (line 164)
    kwargs_288146 = {}
    # Getting the type of 'read_style_directory' (line 164)
    read_style_directory_288144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 19), 'read_style_directory', False)
    # Calling read_style_directory(args, kwargs) (line 164)
    read_style_directory_call_result_288147 = invoke(stypy.reporting.localization.Localization(__file__, 164, 19), read_style_directory_288144, *[BASE_LIBRARY_PATH_288145], **kwargs_288146)
    
    # Processing the call keyword arguments (line 164)
    kwargs_288148 = {}
    # Getting the type of 'library' (line 164)
    library_288142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'library', False)
    # Obtaining the member 'update' of a type (line 164)
    update_288143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 4), library_288142, 'update')
    # Calling update(args, kwargs) (line 164)
    update_call_result_288149 = invoke(stypy.reporting.localization.Localization(__file__, 164, 4), update_288143, *[read_style_directory_call_result_288147], **kwargs_288148)
    
    # Getting the type of 'library' (line 165)
    library_288150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 11), 'library')
    # Assigning a type to the variable 'stypy_return_type' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'stypy_return_type', library_288150)
    
    # ################# End of 'load_base_library(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'load_base_library' in the type store
    # Getting the type of 'stypy_return_type' (line 161)
    stypy_return_type_288151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288151)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'load_base_library'
    return stypy_return_type_288151

# Assigning a type to the variable 'load_base_library' (line 161)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'load_base_library', load_base_library)

@norecursion
def iter_user_libraries(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iter_user_libraries'
    module_type_store = module_type_store.open_function_context('iter_user_libraries', 168, 0, False)
    
    # Passed parameters checking function
    iter_user_libraries.stypy_localization = localization
    iter_user_libraries.stypy_type_of_self = None
    iter_user_libraries.stypy_type_store = module_type_store
    iter_user_libraries.stypy_function_name = 'iter_user_libraries'
    iter_user_libraries.stypy_param_names_list = []
    iter_user_libraries.stypy_varargs_param_name = None
    iter_user_libraries.stypy_kwargs_param_name = None
    iter_user_libraries.stypy_call_defaults = defaults
    iter_user_libraries.stypy_call_varargs = varargs
    iter_user_libraries.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iter_user_libraries', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iter_user_libraries', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iter_user_libraries(...)' code ##################

    
    # Getting the type of 'USER_LIBRARY_PATHS' (line 169)
    USER_LIBRARY_PATHS_288152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 25), 'USER_LIBRARY_PATHS')
    # Testing the type of a for loop iterable (line 169)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 169, 4), USER_LIBRARY_PATHS_288152)
    # Getting the type of the for loop variable (line 169)
    for_loop_var_288153 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 169, 4), USER_LIBRARY_PATHS_288152)
    # Assigning a type to the variable 'stylelib_path' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'stylelib_path', for_loop_var_288153)
    # SSA begins for a for statement (line 169)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 170):
    
    # Call to expanduser(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'stylelib_path' (line 170)
    stylelib_path_288157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 43), 'stylelib_path', False)
    # Processing the call keyword arguments (line 170)
    kwargs_288158 = {}
    # Getting the type of 'os' (line 170)
    os_288154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 170)
    path_288155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 24), os_288154, 'path')
    # Obtaining the member 'expanduser' of a type (line 170)
    expanduser_288156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 24), path_288155, 'expanduser')
    # Calling expanduser(args, kwargs) (line 170)
    expanduser_call_result_288159 = invoke(stypy.reporting.localization.Localization(__file__, 170, 24), expanduser_288156, *[stylelib_path_288157], **kwargs_288158)
    
    # Assigning a type to the variable 'stylelib_path' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'stylelib_path', expanduser_call_result_288159)
    
    
    # Evaluating a boolean operation
    
    # Call to exists(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'stylelib_path' (line 171)
    stylelib_path_288163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 26), 'stylelib_path', False)
    # Processing the call keyword arguments (line 171)
    kwargs_288164 = {}
    # Getting the type of 'os' (line 171)
    os_288160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 171)
    path_288161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 11), os_288160, 'path')
    # Obtaining the member 'exists' of a type (line 171)
    exists_288162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 11), path_288161, 'exists')
    # Calling exists(args, kwargs) (line 171)
    exists_call_result_288165 = invoke(stypy.reporting.localization.Localization(__file__, 171, 11), exists_288162, *[stylelib_path_288163], **kwargs_288164)
    
    
    # Call to isdir(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'stylelib_path' (line 171)
    stylelib_path_288169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 59), 'stylelib_path', False)
    # Processing the call keyword arguments (line 171)
    kwargs_288170 = {}
    # Getting the type of 'os' (line 171)
    os_288166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 45), 'os', False)
    # Obtaining the member 'path' of a type (line 171)
    path_288167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 45), os_288166, 'path')
    # Obtaining the member 'isdir' of a type (line 171)
    isdir_288168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 45), path_288167, 'isdir')
    # Calling isdir(args, kwargs) (line 171)
    isdir_call_result_288171 = invoke(stypy.reporting.localization.Localization(__file__, 171, 45), isdir_288168, *[stylelib_path_288169], **kwargs_288170)
    
    # Applying the binary operator 'and' (line 171)
    result_and_keyword_288172 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 11), 'and', exists_call_result_288165, isdir_call_result_288171)
    
    # Testing the type of an if condition (line 171)
    if_condition_288173 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 8), result_and_keyword_288172)
    # Assigning a type to the variable 'if_condition_288173' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'if_condition_288173', if_condition_288173)
    # SSA begins for if statement (line 171)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Creating a generator
    # Getting the type of 'stylelib_path' (line 172)
    stylelib_path_288174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 18), 'stylelib_path')
    GeneratorType_288175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 12), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 12), GeneratorType_288175, stylelib_path_288174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'stypy_return_type', GeneratorType_288175)
    # SSA join for if statement (line 171)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'iter_user_libraries(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iter_user_libraries' in the type store
    # Getting the type of 'stypy_return_type' (line 168)
    stypy_return_type_288176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288176)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iter_user_libraries'
    return stypy_return_type_288176

# Assigning a type to the variable 'iter_user_libraries' (line 168)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), 'iter_user_libraries', iter_user_libraries)

@norecursion
def update_user_library(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'update_user_library'
    module_type_store = module_type_store.open_function_context('update_user_library', 175, 0, False)
    
    # Passed parameters checking function
    update_user_library.stypy_localization = localization
    update_user_library.stypy_type_of_self = None
    update_user_library.stypy_type_store = module_type_store
    update_user_library.stypy_function_name = 'update_user_library'
    update_user_library.stypy_param_names_list = ['library']
    update_user_library.stypy_varargs_param_name = None
    update_user_library.stypy_kwargs_param_name = None
    update_user_library.stypy_call_defaults = defaults
    update_user_library.stypy_call_varargs = varargs
    update_user_library.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'update_user_library', ['library'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'update_user_library', localization, ['library'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'update_user_library(...)' code ##################

    unicode_288177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 4), 'unicode', u'Update style library with user-defined rc files')
    
    
    # Call to iter_user_libraries(...): (line 177)
    # Processing the call keyword arguments (line 177)
    kwargs_288179 = {}
    # Getting the type of 'iter_user_libraries' (line 177)
    iter_user_libraries_288178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 25), 'iter_user_libraries', False)
    # Calling iter_user_libraries(args, kwargs) (line 177)
    iter_user_libraries_call_result_288180 = invoke(stypy.reporting.localization.Localization(__file__, 177, 25), iter_user_libraries_288178, *[], **kwargs_288179)
    
    # Testing the type of a for loop iterable (line 177)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 177, 4), iter_user_libraries_call_result_288180)
    # Getting the type of the for loop variable (line 177)
    for_loop_var_288181 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 177, 4), iter_user_libraries_call_result_288180)
    # Assigning a type to the variable 'stylelib_path' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stylelib_path', for_loop_var_288181)
    # SSA begins for a for statement (line 177)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 178):
    
    # Call to read_style_directory(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'stylelib_path' (line 178)
    stylelib_path_288183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 38), 'stylelib_path', False)
    # Processing the call keyword arguments (line 178)
    kwargs_288184 = {}
    # Getting the type of 'read_style_directory' (line 178)
    read_style_directory_288182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 17), 'read_style_directory', False)
    # Calling read_style_directory(args, kwargs) (line 178)
    read_style_directory_call_result_288185 = invoke(stypy.reporting.localization.Localization(__file__, 178, 17), read_style_directory_288182, *[stylelib_path_288183], **kwargs_288184)
    
    # Assigning a type to the variable 'styles' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'styles', read_style_directory_call_result_288185)
    
    # Call to update_nested_dict(...): (line 179)
    # Processing the call arguments (line 179)
    # Getting the type of 'library' (line 179)
    library_288187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 27), 'library', False)
    # Getting the type of 'styles' (line 179)
    styles_288188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 36), 'styles', False)
    # Processing the call keyword arguments (line 179)
    kwargs_288189 = {}
    # Getting the type of 'update_nested_dict' (line 179)
    update_nested_dict_288186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'update_nested_dict', False)
    # Calling update_nested_dict(args, kwargs) (line 179)
    update_nested_dict_call_result_288190 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), update_nested_dict_288186, *[library_288187, styles_288188], **kwargs_288189)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'library' (line 180)
    library_288191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 11), 'library')
    # Assigning a type to the variable 'stypy_return_type' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type', library_288191)
    
    # ################# End of 'update_user_library(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'update_user_library' in the type store
    # Getting the type of 'stypy_return_type' (line 175)
    stypy_return_type_288192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288192)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'update_user_library'
    return stypy_return_type_288192

# Assigning a type to the variable 'update_user_library' (line 175)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'update_user_library', update_user_library)

@norecursion
def iter_style_files(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iter_style_files'
    module_type_store = module_type_store.open_function_context('iter_style_files', 183, 0, False)
    
    # Passed parameters checking function
    iter_style_files.stypy_localization = localization
    iter_style_files.stypy_type_of_self = None
    iter_style_files.stypy_type_store = module_type_store
    iter_style_files.stypy_function_name = 'iter_style_files'
    iter_style_files.stypy_param_names_list = ['style_dir']
    iter_style_files.stypy_varargs_param_name = None
    iter_style_files.stypy_kwargs_param_name = None
    iter_style_files.stypy_call_defaults = defaults
    iter_style_files.stypy_call_varargs = varargs
    iter_style_files.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iter_style_files', ['style_dir'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iter_style_files', localization, ['style_dir'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iter_style_files(...)' code ##################

    unicode_288193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 4), 'unicode', u'Yield file path and name of styles in the given directory.')
    
    
    # Call to listdir(...): (line 185)
    # Processing the call arguments (line 185)
    # Getting the type of 'style_dir' (line 185)
    style_dir_288196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 27), 'style_dir', False)
    # Processing the call keyword arguments (line 185)
    kwargs_288197 = {}
    # Getting the type of 'os' (line 185)
    os_288194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 16), 'os', False)
    # Obtaining the member 'listdir' of a type (line 185)
    listdir_288195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 16), os_288194, 'listdir')
    # Calling listdir(args, kwargs) (line 185)
    listdir_call_result_288198 = invoke(stypy.reporting.localization.Localization(__file__, 185, 16), listdir_288195, *[style_dir_288196], **kwargs_288197)
    
    # Testing the type of a for loop iterable (line 185)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 185, 4), listdir_call_result_288198)
    # Getting the type of the for loop variable (line 185)
    for_loop_var_288199 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 185, 4), listdir_call_result_288198)
    # Assigning a type to the variable 'path' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'path', for_loop_var_288199)
    # SSA begins for a for statement (line 185)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 186):
    
    # Call to basename(...): (line 186)
    # Processing the call arguments (line 186)
    # Getting the type of 'path' (line 186)
    path_288203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 36), 'path', False)
    # Processing the call keyword arguments (line 186)
    kwargs_288204 = {}
    # Getting the type of 'os' (line 186)
    os_288200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 186)
    path_288201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 19), os_288200, 'path')
    # Obtaining the member 'basename' of a type (line 186)
    basename_288202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 19), path_288201, 'basename')
    # Calling basename(args, kwargs) (line 186)
    basename_call_result_288205 = invoke(stypy.reporting.localization.Localization(__file__, 186, 19), basename_288202, *[path_288203], **kwargs_288204)
    
    # Assigning a type to the variable 'filename' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'filename', basename_call_result_288205)
    
    
    # Call to is_style_file(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'filename' (line 187)
    filename_288207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 25), 'filename', False)
    # Processing the call keyword arguments (line 187)
    kwargs_288208 = {}
    # Getting the type of 'is_style_file' (line 187)
    is_style_file_288206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 11), 'is_style_file', False)
    # Calling is_style_file(args, kwargs) (line 187)
    is_style_file_call_result_288209 = invoke(stypy.reporting.localization.Localization(__file__, 187, 11), is_style_file_288206, *[filename_288207], **kwargs_288208)
    
    # Testing the type of an if condition (line 187)
    if_condition_288210 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 187, 8), is_style_file_call_result_288209)
    # Assigning a type to the variable 'if_condition_288210' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'if_condition_288210', if_condition_288210)
    # SSA begins for if statement (line 187)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 188):
    
    # Call to match(...): (line 188)
    # Processing the call arguments (line 188)
    # Getting the type of 'filename' (line 188)
    filename_288213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 45), 'filename', False)
    # Processing the call keyword arguments (line 188)
    kwargs_288214 = {}
    # Getting the type of 'STYLE_FILE_PATTERN' (line 188)
    STYLE_FILE_PATTERN_288211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 20), 'STYLE_FILE_PATTERN', False)
    # Obtaining the member 'match' of a type (line 188)
    match_288212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 20), STYLE_FILE_PATTERN_288211, 'match')
    # Calling match(args, kwargs) (line 188)
    match_call_result_288215 = invoke(stypy.reporting.localization.Localization(__file__, 188, 20), match_288212, *[filename_288213], **kwargs_288214)
    
    # Assigning a type to the variable 'match' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'match', match_call_result_288215)
    
    # Assigning a Call to a Name (line 189):
    
    # Call to abspath(...): (line 189)
    # Processing the call arguments (line 189)
    
    # Call to join(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of 'style_dir' (line 189)
    style_dir_288222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 48), 'style_dir', False)
    # Getting the type of 'path' (line 189)
    path_288223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 59), 'path', False)
    # Processing the call keyword arguments (line 189)
    kwargs_288224 = {}
    # Getting the type of 'os' (line 189)
    os_288219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 35), 'os', False)
    # Obtaining the member 'path' of a type (line 189)
    path_288220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 35), os_288219, 'path')
    # Obtaining the member 'join' of a type (line 189)
    join_288221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 35), path_288220, 'join')
    # Calling join(args, kwargs) (line 189)
    join_call_result_288225 = invoke(stypy.reporting.localization.Localization(__file__, 189, 35), join_288221, *[style_dir_288222, path_288223], **kwargs_288224)
    
    # Processing the call keyword arguments (line 189)
    kwargs_288226 = {}
    # Getting the type of 'os' (line 189)
    os_288216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 189)
    path_288217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 19), os_288216, 'path')
    # Obtaining the member 'abspath' of a type (line 189)
    abspath_288218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 19), path_288217, 'abspath')
    # Calling abspath(args, kwargs) (line 189)
    abspath_call_result_288227 = invoke(stypy.reporting.localization.Localization(__file__, 189, 19), abspath_288218, *[join_call_result_288225], **kwargs_288226)
    
    # Assigning a type to the variable 'path' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'path', abspath_call_result_288227)
    # Creating a generator
    
    # Obtaining an instance of the builtin type 'tuple' (line 190)
    tuple_288228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 190)
    # Adding element type (line 190)
    # Getting the type of 'path' (line 190)
    path_288229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 18), 'path')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 18), tuple_288228, path_288229)
    # Adding element type (line 190)
    
    # Obtaining the type of the subscript
    int_288230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 39), 'int')
    
    # Call to groups(...): (line 190)
    # Processing the call keyword arguments (line 190)
    kwargs_288233 = {}
    # Getting the type of 'match' (line 190)
    match_288231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 24), 'match', False)
    # Obtaining the member 'groups' of a type (line 190)
    groups_288232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 24), match_288231, 'groups')
    # Calling groups(args, kwargs) (line 190)
    groups_call_result_288234 = invoke(stypy.reporting.localization.Localization(__file__, 190, 24), groups_288232, *[], **kwargs_288233)
    
    # Obtaining the member '__getitem__' of a type (line 190)
    getitem___288235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 24), groups_call_result_288234, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 190)
    subscript_call_result_288236 = invoke(stypy.reporting.localization.Localization(__file__, 190, 24), getitem___288235, int_288230)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 18), tuple_288228, subscript_call_result_288236)
    
    GeneratorType_288237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 12), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 12), GeneratorType_288237, tuple_288228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'stypy_return_type', GeneratorType_288237)
    # SSA join for if statement (line 187)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'iter_style_files(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iter_style_files' in the type store
    # Getting the type of 'stypy_return_type' (line 183)
    stypy_return_type_288238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288238)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iter_style_files'
    return stypy_return_type_288238

# Assigning a type to the variable 'iter_style_files' (line 183)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'iter_style_files', iter_style_files)

@norecursion
def read_style_directory(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'read_style_directory'
    module_type_store = module_type_store.open_function_context('read_style_directory', 193, 0, False)
    
    # Passed parameters checking function
    read_style_directory.stypy_localization = localization
    read_style_directory.stypy_type_of_self = None
    read_style_directory.stypy_type_store = module_type_store
    read_style_directory.stypy_function_name = 'read_style_directory'
    read_style_directory.stypy_param_names_list = ['style_dir']
    read_style_directory.stypy_varargs_param_name = None
    read_style_directory.stypy_kwargs_param_name = None
    read_style_directory.stypy_call_defaults = defaults
    read_style_directory.stypy_call_varargs = varargs
    read_style_directory.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'read_style_directory', ['style_dir'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'read_style_directory', localization, ['style_dir'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'read_style_directory(...)' code ##################

    unicode_288239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 4), 'unicode', u'Return dictionary of styles defined in `style_dir`.')
    
    # Assigning a Call to a Name (line 195):
    
    # Call to dict(...): (line 195)
    # Processing the call keyword arguments (line 195)
    kwargs_288241 = {}
    # Getting the type of 'dict' (line 195)
    dict_288240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 13), 'dict', False)
    # Calling dict(args, kwargs) (line 195)
    dict_call_result_288242 = invoke(stypy.reporting.localization.Localization(__file__, 195, 13), dict_288240, *[], **kwargs_288241)
    
    # Assigning a type to the variable 'styles' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'styles', dict_call_result_288242)
    
    
    # Call to iter_style_files(...): (line 196)
    # Processing the call arguments (line 196)
    # Getting the type of 'style_dir' (line 196)
    style_dir_288244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 39), 'style_dir', False)
    # Processing the call keyword arguments (line 196)
    kwargs_288245 = {}
    # Getting the type of 'iter_style_files' (line 196)
    iter_style_files_288243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 22), 'iter_style_files', False)
    # Calling iter_style_files(args, kwargs) (line 196)
    iter_style_files_call_result_288246 = invoke(stypy.reporting.localization.Localization(__file__, 196, 22), iter_style_files_288243, *[style_dir_288244], **kwargs_288245)
    
    # Testing the type of a for loop iterable (line 196)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 196, 4), iter_style_files_call_result_288246)
    # Getting the type of the for loop variable (line 196)
    for_loop_var_288247 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 196, 4), iter_style_files_call_result_288246)
    # Assigning a type to the variable 'path' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'path', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 4), for_loop_var_288247))
    # Assigning a type to the variable 'name' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 4), for_loop_var_288247))
    # SSA begins for a for statement (line 196)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to catch_warnings(...): (line 197)
    # Processing the call keyword arguments (line 197)
    # Getting the type of 'True' (line 197)
    True_288250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 44), 'True', False)
    keyword_288251 = True_288250
    kwargs_288252 = {'record': keyword_288251}
    # Getting the type of 'warnings' (line 197)
    warnings_288248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 13), 'warnings', False)
    # Obtaining the member 'catch_warnings' of a type (line 197)
    catch_warnings_288249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 13), warnings_288248, 'catch_warnings')
    # Calling catch_warnings(args, kwargs) (line 197)
    catch_warnings_call_result_288253 = invoke(stypy.reporting.localization.Localization(__file__, 197, 13), catch_warnings_288249, *[], **kwargs_288252)
    
    with_288254 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 197, 13), catch_warnings_call_result_288253, 'with parameter', '__enter__', '__exit__')

    if with_288254:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 197)
        enter___288255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 13), catch_warnings_call_result_288253, '__enter__')
        with_enter_288256 = invoke(stypy.reporting.localization.Localization(__file__, 197, 13), enter___288255)
        # Assigning a type to the variable 'warns' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 13), 'warns', with_enter_288256)
        
        # Assigning a Call to a Subscript (line 198):
        
        # Call to rc_params_from_file(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'path' (line 198)
        path_288258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 47), 'path', False)
        # Processing the call keyword arguments (line 198)
        # Getting the type of 'False' (line 199)
        False_288259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 68), 'False', False)
        keyword_288260 = False_288259
        kwargs_288261 = {'use_default_template': keyword_288260}
        # Getting the type of 'rc_params_from_file' (line 198)
        rc_params_from_file_288257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 27), 'rc_params_from_file', False)
        # Calling rc_params_from_file(args, kwargs) (line 198)
        rc_params_from_file_call_result_288262 = invoke(stypy.reporting.localization.Localization(__file__, 198, 27), rc_params_from_file_288257, *[path_288258], **kwargs_288261)
        
        # Getting the type of 'styles' (line 198)
        styles_288263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'styles')
        # Getting the type of 'name' (line 198)
        name_288264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 19), 'name')
        # Storing an element on a container (line 198)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 12), styles_288263, (name_288264, rc_params_from_file_call_result_288262))
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 197)
        exit___288265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 13), catch_warnings_call_result_288253, '__exit__')
        with_exit_288266 = invoke(stypy.reporting.localization.Localization(__file__, 197, 13), exit___288265, None, None, None)

    
    # Getting the type of 'warns' (line 201)
    warns_288267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 17), 'warns')
    # Testing the type of a for loop iterable (line 201)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 201, 8), warns_288267)
    # Getting the type of the for loop variable (line 201)
    for_loop_var_288268 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 201, 8), warns_288267)
    # Assigning a type to the variable 'w' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'w', for_loop_var_288268)
    # SSA begins for a for statement (line 201)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 202):
    unicode_288269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 22), 'unicode', u'In %s: %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 202)
    tuple_288270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 202)
    # Adding element type (line 202)
    # Getting the type of 'path' (line 202)
    path_288271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 37), 'path')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 37), tuple_288270, path_288271)
    # Adding element type (line 202)
    # Getting the type of 'w' (line 202)
    w_288272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 43), 'w')
    # Obtaining the member 'message' of a type (line 202)
    message_288273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 43), w_288272, 'message')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 37), tuple_288270, message_288273)
    
    # Applying the binary operator '%' (line 202)
    result_mod_288274 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 22), '%', unicode_288269, tuple_288270)
    
    # Assigning a type to the variable 'message' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'message', result_mod_288274)
    
    # Call to warn(...): (line 203)
    # Processing the call arguments (line 203)
    # Getting the type of 'message' (line 203)
    message_288277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 26), 'message', False)
    # Processing the call keyword arguments (line 203)
    kwargs_288278 = {}
    # Getting the type of 'warnings' (line 203)
    warnings_288275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 203)
    warn_288276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 12), warnings_288275, 'warn')
    # Calling warn(args, kwargs) (line 203)
    warn_call_result_288279 = invoke(stypy.reporting.localization.Localization(__file__, 203, 12), warn_288276, *[message_288277], **kwargs_288278)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'styles' (line 205)
    styles_288280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 11), 'styles')
    # Assigning a type to the variable 'stypy_return_type' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'stypy_return_type', styles_288280)
    
    # ################# End of 'read_style_directory(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'read_style_directory' in the type store
    # Getting the type of 'stypy_return_type' (line 193)
    stypy_return_type_288281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288281)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'read_style_directory'
    return stypy_return_type_288281

# Assigning a type to the variable 'read_style_directory' (line 193)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), 'read_style_directory', read_style_directory)

@norecursion
def update_nested_dict(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'update_nested_dict'
    module_type_store = module_type_store.open_function_context('update_nested_dict', 208, 0, False)
    
    # Passed parameters checking function
    update_nested_dict.stypy_localization = localization
    update_nested_dict.stypy_type_of_self = None
    update_nested_dict.stypy_type_store = module_type_store
    update_nested_dict.stypy_function_name = 'update_nested_dict'
    update_nested_dict.stypy_param_names_list = ['main_dict', 'new_dict']
    update_nested_dict.stypy_varargs_param_name = None
    update_nested_dict.stypy_kwargs_param_name = None
    update_nested_dict.stypy_call_defaults = defaults
    update_nested_dict.stypy_call_varargs = varargs
    update_nested_dict.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'update_nested_dict', ['main_dict', 'new_dict'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'update_nested_dict', localization, ['main_dict', 'new_dict'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'update_nested_dict(...)' code ##################

    unicode_288282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, (-1)), 'unicode', u"Update nested dict (only level of nesting) with new values.\n\n    Unlike dict.update, this assumes that the values of the parent dict are\n    dicts (or dict-like), so you shouldn't replace the nested dict if it\n    already exists. Instead you should update the sub-dict.\n    ")
    
    
    # Call to iteritems(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'new_dict' (line 216)
    new_dict_288285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 39), 'new_dict', False)
    # Processing the call keyword arguments (line 216)
    kwargs_288286 = {}
    # Getting the type of 'six' (line 216)
    six_288283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 25), 'six', False)
    # Obtaining the member 'iteritems' of a type (line 216)
    iteritems_288284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 25), six_288283, 'iteritems')
    # Calling iteritems(args, kwargs) (line 216)
    iteritems_call_result_288287 = invoke(stypy.reporting.localization.Localization(__file__, 216, 25), iteritems_288284, *[new_dict_288285], **kwargs_288286)
    
    # Testing the type of a for loop iterable (line 216)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 216, 4), iteritems_call_result_288287)
    # Getting the type of the for loop variable (line 216)
    for_loop_var_288288 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 216, 4), iteritems_call_result_288287)
    # Assigning a type to the variable 'name' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 4), for_loop_var_288288))
    # Assigning a type to the variable 'rc_dict' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'rc_dict', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 4), for_loop_var_288288))
    # SSA begins for a for statement (line 216)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'name' (line 217)
    name_288289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 11), 'name')
    # Getting the type of 'main_dict' (line 217)
    main_dict_288290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 19), 'main_dict')
    # Applying the binary operator 'in' (line 217)
    result_contains_288291 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 11), 'in', name_288289, main_dict_288290)
    
    # Testing the type of an if condition (line 217)
    if_condition_288292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 8), result_contains_288291)
    # Assigning a type to the variable 'if_condition_288292' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'if_condition_288292', if_condition_288292)
    # SSA begins for if statement (line 217)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to update(...): (line 218)
    # Processing the call arguments (line 218)
    # Getting the type of 'rc_dict' (line 218)
    rc_dict_288298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 35), 'rc_dict', False)
    # Processing the call keyword arguments (line 218)
    kwargs_288299 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 218)
    name_288293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 22), 'name', False)
    # Getting the type of 'main_dict' (line 218)
    main_dict_288294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'main_dict', False)
    # Obtaining the member '__getitem__' of a type (line 218)
    getitem___288295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 12), main_dict_288294, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 218)
    subscript_call_result_288296 = invoke(stypy.reporting.localization.Localization(__file__, 218, 12), getitem___288295, name_288293)
    
    # Obtaining the member 'update' of a type (line 218)
    update_288297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 12), subscript_call_result_288296, 'update')
    # Calling update(args, kwargs) (line 218)
    update_call_result_288300 = invoke(stypy.reporting.localization.Localization(__file__, 218, 12), update_288297, *[rc_dict_288298], **kwargs_288299)
    
    # SSA branch for the else part of an if statement (line 217)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Subscript (line 220):
    # Getting the type of 'rc_dict' (line 220)
    rc_dict_288301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 30), 'rc_dict')
    # Getting the type of 'main_dict' (line 220)
    main_dict_288302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'main_dict')
    # Getting the type of 'name' (line 220)
    name_288303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 22), 'name')
    # Storing an element on a container (line 220)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 12), main_dict_288302, (name_288303, rc_dict_288301))
    # SSA join for if statement (line 217)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'main_dict' (line 221)
    main_dict_288304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 11), 'main_dict')
    # Assigning a type to the variable 'stypy_return_type' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'stypy_return_type', main_dict_288304)
    
    # ################# End of 'update_nested_dict(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'update_nested_dict' in the type store
    # Getting the type of 'stypy_return_type' (line 208)
    stypy_return_type_288305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288305)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'update_nested_dict'
    return stypy_return_type_288305

# Assigning a type to the variable 'update_nested_dict' (line 208)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 0), 'update_nested_dict', update_nested_dict)

# Assigning a Call to a Name (line 226):

# Call to load_base_library(...): (line 226)
# Processing the call keyword arguments (line 226)
kwargs_288307 = {}
# Getting the type of 'load_base_library' (line 226)
load_base_library_288306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 16), 'load_base_library', False)
# Calling load_base_library(args, kwargs) (line 226)
load_base_library_call_result_288308 = invoke(stypy.reporting.localization.Localization(__file__, 226, 16), load_base_library_288306, *[], **kwargs_288307)

# Assigning a type to the variable '_base_library' (line 226)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 0), '_base_library', load_base_library_call_result_288308)

# Assigning a Name to a Name (line 228):
# Getting the type of 'None' (line 228)
None_288309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 10), 'None')
# Assigning a type to the variable 'library' (line 228)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 0), 'library', None_288309)

# Assigning a List to a Name (line 229):

# Obtaining an instance of the builtin type 'list' (line 229)
list_288310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 229)

# Assigning a type to the variable 'available' (line 229)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 0), 'available', list_288310)

@norecursion
def reload_library(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'reload_library'
    module_type_store = module_type_store.open_function_context('reload_library', 232, 0, False)
    
    # Passed parameters checking function
    reload_library.stypy_localization = localization
    reload_library.stypy_type_of_self = None
    reload_library.stypy_type_store = module_type_store
    reload_library.stypy_function_name = 'reload_library'
    reload_library.stypy_param_names_list = []
    reload_library.stypy_varargs_param_name = None
    reload_library.stypy_kwargs_param_name = None
    reload_library.stypy_call_defaults = defaults
    reload_library.stypy_call_varargs = varargs
    reload_library.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'reload_library', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'reload_library', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'reload_library(...)' code ##################

    unicode_288311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 4), 'unicode', u'Reload style library.')
    # Marking variables as global (line 234)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 234, 4), 'library')
    
    # Multiple assignment of 2 elements.
    
    # Call to update_user_library(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of '_base_library' (line 235)
    _base_library_288313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 49), '_base_library', False)
    # Processing the call keyword arguments (line 235)
    kwargs_288314 = {}
    # Getting the type of 'update_user_library' (line 235)
    update_user_library_288312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 29), 'update_user_library', False)
    # Calling update_user_library(args, kwargs) (line 235)
    update_user_library_call_result_288315 = invoke(stypy.reporting.localization.Localization(__file__, 235, 29), update_user_library_288312, *[_base_library_288313], **kwargs_288314)
    
    # Assigning a type to the variable 'library' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 19), 'library', update_user_library_call_result_288315)
    # Getting the type of 'library' (line 235)
    library_288316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 19), 'library')
    # Getting the type of 'available' (line 235)
    available_288317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'available')
    slice_288318 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 235, 4), None, None, None)
    # Storing an element on a container (line 235)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 4), available_288317, (slice_288318, library_288316))
    
    # ################# End of 'reload_library(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'reload_library' in the type store
    # Getting the type of 'stypy_return_type' (line 232)
    stypy_return_type_288319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288319)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'reload_library'
    return stypy_return_type_288319

# Assigning a type to the variable 'reload_library' (line 232)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 0), 'reload_library', reload_library)

# Call to reload_library(...): (line 236)
# Processing the call keyword arguments (line 236)
kwargs_288321 = {}
# Getting the type of 'reload_library' (line 236)
reload_library_288320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 0), 'reload_library', False)
# Calling reload_library(args, kwargs) (line 236)
reload_library_call_result_288322 = invoke(stypy.reporting.localization.Localization(__file__, 236, 0), reload_library_288320, *[], **kwargs_288321)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
