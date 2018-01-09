
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: A module for parsing and generating fontconfig patterns.
3: 
4: See the `fontconfig pattern specification
5: <https://www.freedesktop.org/software/fontconfig/fontconfig-user.html>`_ for
6: more information.
7: '''
8: 
9: # This class is defined here because it must be available in:
10: #   - The old-style config framework (:file:`rcsetup.py`)
11: #   - The traits-based config framework (:file:`mpltraits.py`)
12: #   - The font manager (:file:`font_manager.py`)
13: 
14: # It probably logically belongs in :file:`font_manager.py`, but
15: # placing it in any of these places would have created cyclical
16: # dependency problems, or an undesired dependency on traits even
17: # when the traits-based config framework is not used.
18: 
19: from __future__ import (absolute_import, division, print_function,
20:                         unicode_literals)
21: 
22: import six
23: 
24: import re
25: import sys
26: from pyparsing import (Literal, ZeroOrMore, Optional, Regex, StringEnd,
27:                        ParseException, Suppress)
28: 
29: try:
30:     from functools import lru_cache
31: except ImportError:
32:     from backports.functools_lru_cache import lru_cache
33: 
34: family_punc = r'\\\-:,'
35: family_unescape = re.compile(r'\\([%s])' % family_punc).sub
36: family_escape = re.compile(r'([%s])' % family_punc).sub
37: 
38: value_punc = r'\\=_:,'
39: value_unescape = re.compile(r'\\([%s])' % value_punc).sub
40: value_escape = re.compile(r'([%s])' % value_punc).sub
41: 
42: class FontconfigPatternParser(object):
43:     '''A simple pyparsing-based parser for fontconfig-style patterns.
44: 
45:     See the `fontconfig pattern specification
46:     <https://www.freedesktop.org/software/fontconfig/fontconfig-user.html>`_
47:     for more information.
48:     '''
49: 
50:     _constants = {
51:         'thin'           : ('weight', 'light'),
52:         'extralight'     : ('weight', 'light'),
53:         'ultralight'     : ('weight', 'light'),
54:         'light'          : ('weight', 'light'),
55:         'book'           : ('weight', 'book'),
56:         'regular'        : ('weight', 'regular'),
57:         'normal'         : ('weight', 'normal'),
58:         'medium'         : ('weight', 'medium'),
59:         'demibold'       : ('weight', 'demibold'),
60:         'semibold'       : ('weight', 'semibold'),
61:         'bold'           : ('weight', 'bold'),
62:         'extrabold'      : ('weight', 'extra bold'),
63:         'black'          : ('weight', 'black'),
64:         'heavy'          : ('weight', 'heavy'),
65:         'roman'          : ('slant', 'normal'),
66:         'italic'         : ('slant', 'italic'),
67:         'oblique'        : ('slant', 'oblique'),
68:         'ultracondensed' : ('width', 'ultra-condensed'),
69:         'extracondensed' : ('width', 'extra-condensed'),
70:         'condensed'      : ('width', 'condensed'),
71:         'semicondensed'  : ('width', 'semi-condensed'),
72:         'expanded'       : ('width', 'expanded'),
73:         'extraexpanded'  : ('width', 'extra-expanded'),
74:         'ultraexpanded'  : ('width', 'ultra-expanded')
75:         }
76: 
77:     def __init__(self):
78:         family      = Regex(r'([^%s]|(\\[%s]))*' %
79:                             (family_punc, family_punc)) \
80:                       .setParseAction(self._family)
81:         size        = Regex(r"([0-9]+\.?[0-9]*|\.[0-9]+)") \
82:                       .setParseAction(self._size)
83:         name        = Regex(r'[a-z]+') \
84:                       .setParseAction(self._name)
85:         value       = Regex(r'([^%s]|(\\[%s]))*' %
86:                             (value_punc, value_punc)) \
87:                       .setParseAction(self._value)
88: 
89:         families    =(family
90:                     + ZeroOrMore(
91:                         Literal(',')
92:                       + family)
93:                     ).setParseAction(self._families)
94: 
95:         point_sizes =(size
96:                     + ZeroOrMore(
97:                         Literal(',')
98:                       + size)
99:                     ).setParseAction(self._point_sizes)
100: 
101:         property    =( (name
102:                       + Suppress(Literal('='))
103:                       + value
104:                       + ZeroOrMore(
105:                           Suppress(Literal(','))
106:                         + value)
107:                       )
108:                      |  name
109:                     ).setParseAction(self._property)
110: 
111:         pattern     =(Optional(
112:                         families)
113:                     + Optional(
114:                         Literal('-')
115:                       + point_sizes)
116:                     + ZeroOrMore(
117:                         Literal(':')
118:                       + property)
119:                     + StringEnd()
120:                     )
121: 
122:         self._parser = pattern
123:         self.ParseException = ParseException
124: 
125:     def parse(self, pattern):
126:         '''
127:         Parse the given fontconfig *pattern* and return a dictionary
128:         of key/value pairs useful for initializing a
129:         :class:`font_manager.FontProperties` object.
130:         '''
131:         props = self._properties = {}
132:         try:
133:             self._parser.parseString(pattern)
134:         except self.ParseException as e:
135:             raise ValueError(
136:                 "Could not parse font string: '%s'\n%s" % (pattern, e))
137: 
138:         self._properties = None
139: 
140:         self._parser.resetCache()
141: 
142:         return props
143: 
144:     def _family(self, s, loc, tokens):
145:         return [family_unescape(r'\1', str(tokens[0]))]
146: 
147:     def _size(self, s, loc, tokens):
148:         return [float(tokens[0])]
149: 
150:     def _name(self, s, loc, tokens):
151:         return [str(tokens[0])]
152: 
153:     def _value(self, s, loc, tokens):
154:         return [value_unescape(r'\1', str(tokens[0]))]
155: 
156:     def _families(self, s, loc, tokens):
157:         self._properties['family'] = [str(x) for x in tokens]
158:         return []
159: 
160:     def _point_sizes(self, s, loc, tokens):
161:         self._properties['size'] = [str(x) for x in tokens]
162:         return []
163: 
164:     def _property(self, s, loc, tokens):
165:         if len(tokens) == 1:
166:             if tokens[0] in self._constants:
167:                 key, val = self._constants[tokens[0]]
168:                 self._properties.setdefault(key, []).append(val)
169:         else:
170:             key = tokens[0]
171:             val = tokens[1:]
172:             self._properties.setdefault(key, []).extend(val)
173:         return []
174: 
175: 
176: # `parse_fontconfig_pattern` is a bottleneck during the tests because it is
177: # repeatedly called when the rcParams are reset (to validate the default
178: # fonts).  In practice, the cache size doesn't grow beyond a few dozen entries
179: # during the test suite.
180: parse_fontconfig_pattern = lru_cache()(FontconfigPatternParser().parse)
181: 
182: 
183: def generate_fontconfig_pattern(d):
184:     '''
185:     Given a dictionary of key/value pairs, generates a fontconfig
186:     pattern string.
187:     '''
188:     props = []
189:     families = ''
190:     size = ''
191:     for key in 'family style variant weight stretch file size'.split():
192:         val = getattr(d, 'get_' + key)()
193:         if val is not None and val != []:
194:             if type(val) == list:
195:                 val = [value_escape(r'\\\1', str(x)) for x in val
196:                        if x is not None]
197:                 if val != []:
198:                     val = ','.join(val)
199:             props.append(":%s=%s" % (key, val))
200:     return ''.join(props)
201: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'unicode', u'\nA module for parsing and generating fontconfig patterns.\n\nSee the `fontconfig pattern specification\n<https://www.freedesktop.org/software/fontconfig/fontconfig-user.html>`_ for\nmore information.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'import six' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_4 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'six')

if (type(import_4) is not StypyTypeError):

    if (import_4 != 'pyd_module'):
        __import__(import_4)
        sys_modules_5 = sys.modules[import_4]
        import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'six', sys_modules_5.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'six', import_4)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'import re' statement (line 24)
import re

import_module(stypy.reporting.localization.Localization(__file__, 24, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'import sys' statement (line 25)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from pyparsing import Literal, ZeroOrMore, Optional, Regex, StringEnd, ParseException, Suppress' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_6 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'pyparsing')

if (type(import_6) is not StypyTypeError):

    if (import_6 != 'pyd_module'):
        __import__(import_6)
        sys_modules_7 = sys.modules[import_6]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'pyparsing', sys_modules_7.module_type_store, module_type_store, ['Literal', 'ZeroOrMore', 'Optional', 'Regex', 'StringEnd', 'ParseException', 'Suppress'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_7, sys_modules_7.module_type_store, module_type_store)
    else:
        from pyparsing import Literal, ZeroOrMore, Optional, Regex, StringEnd, ParseException, Suppress

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'pyparsing', None, module_type_store, ['Literal', 'ZeroOrMore', 'Optional', 'Regex', 'StringEnd', 'ParseException', 'Suppress'], [Literal, ZeroOrMore, Optional, Regex, StringEnd, ParseException, Suppress])

else:
    # Assigning a type to the variable 'pyparsing' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'pyparsing', import_6)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')



# SSA begins for try-except statement (line 29)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 4))

# 'from functools import lru_cache' statement (line 30)
try:
    from functools import lru_cache

except:
    lru_cache = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 30, 4), 'functools', None, module_type_store, ['lru_cache'], [lru_cache])

# SSA branch for the except part of a try statement (line 29)
# SSA branch for the except 'ImportError' branch of a try statement (line 29)
module_type_store.open_ssa_branch('except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 4))

# 'from backports.functools_lru_cache import lru_cache' statement (line 32)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_8 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 32, 4), 'backports.functools_lru_cache')

if (type(import_8) is not StypyTypeError):

    if (import_8 != 'pyd_module'):
        __import__(import_8)
        sys_modules_9 = sys.modules[import_8]
        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 4), 'backports.functools_lru_cache', sys_modules_9.module_type_store, module_type_store, ['lru_cache'])
        nest_module(stypy.reporting.localization.Localization(__file__, 32, 4), __file__, sys_modules_9, sys_modules_9.module_type_store, module_type_store)
    else:
        from backports.functools_lru_cache import lru_cache

        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 4), 'backports.functools_lru_cache', None, module_type_store, ['lru_cache'], [lru_cache])

else:
    # Assigning a type to the variable 'backports.functools_lru_cache' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'backports.functools_lru_cache', import_8)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

# SSA join for try-except statement (line 29)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Str to a Name (line 34):

# Assigning a Str to a Name (line 34):
unicode_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 14), 'unicode', u'\\\\\\-:,')
# Assigning a type to the variable 'family_punc' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'family_punc', unicode_10)

# Assigning a Attribute to a Name (line 35):

# Assigning a Attribute to a Name (line 35):

# Call to compile(...): (line 35)
# Processing the call arguments (line 35)
unicode_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 29), 'unicode', u'\\\\([%s])')
# Getting the type of 'family_punc' (line 35)
family_punc_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 43), 'family_punc', False)
# Applying the binary operator '%' (line 35)
result_mod_15 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 29), '%', unicode_13, family_punc_14)

# Processing the call keyword arguments (line 35)
kwargs_16 = {}
# Getting the type of 're' (line 35)
re_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 18), 're', False)
# Obtaining the member 'compile' of a type (line 35)
compile_12 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 18), re_11, 'compile')
# Calling compile(args, kwargs) (line 35)
compile_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 35, 18), compile_12, *[result_mod_15], **kwargs_16)

# Obtaining the member 'sub' of a type (line 35)
sub_18 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 18), compile_call_result_17, 'sub')
# Assigning a type to the variable 'family_unescape' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'family_unescape', sub_18)

# Assigning a Attribute to a Name (line 36):

# Assigning a Attribute to a Name (line 36):

# Call to compile(...): (line 36)
# Processing the call arguments (line 36)
unicode_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 27), 'unicode', u'([%s])')
# Getting the type of 'family_punc' (line 36)
family_punc_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 39), 'family_punc', False)
# Applying the binary operator '%' (line 36)
result_mod_23 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 27), '%', unicode_21, family_punc_22)

# Processing the call keyword arguments (line 36)
kwargs_24 = {}
# Getting the type of 're' (line 36)
re_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 're', False)
# Obtaining the member 'compile' of a type (line 36)
compile_20 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 16), re_19, 'compile')
# Calling compile(args, kwargs) (line 36)
compile_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 36, 16), compile_20, *[result_mod_23], **kwargs_24)

# Obtaining the member 'sub' of a type (line 36)
sub_26 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 16), compile_call_result_25, 'sub')
# Assigning a type to the variable 'family_escape' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'family_escape', sub_26)

# Assigning a Str to a Name (line 38):

# Assigning a Str to a Name (line 38):
unicode_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 13), 'unicode', u'\\\\=_:,')
# Assigning a type to the variable 'value_punc' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'value_punc', unicode_27)

# Assigning a Attribute to a Name (line 39):

# Assigning a Attribute to a Name (line 39):

# Call to compile(...): (line 39)
# Processing the call arguments (line 39)
unicode_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 28), 'unicode', u'\\\\([%s])')
# Getting the type of 'value_punc' (line 39)
value_punc_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 42), 'value_punc', False)
# Applying the binary operator '%' (line 39)
result_mod_32 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 28), '%', unicode_30, value_punc_31)

# Processing the call keyword arguments (line 39)
kwargs_33 = {}
# Getting the type of 're' (line 39)
re_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 17), 're', False)
# Obtaining the member 'compile' of a type (line 39)
compile_29 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 17), re_28, 'compile')
# Calling compile(args, kwargs) (line 39)
compile_call_result_34 = invoke(stypy.reporting.localization.Localization(__file__, 39, 17), compile_29, *[result_mod_32], **kwargs_33)

# Obtaining the member 'sub' of a type (line 39)
sub_35 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 17), compile_call_result_34, 'sub')
# Assigning a type to the variable 'value_unescape' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'value_unescape', sub_35)

# Assigning a Attribute to a Name (line 40):

# Assigning a Attribute to a Name (line 40):

# Call to compile(...): (line 40)
# Processing the call arguments (line 40)
unicode_38 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 26), 'unicode', u'([%s])')
# Getting the type of 'value_punc' (line 40)
value_punc_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 38), 'value_punc', False)
# Applying the binary operator '%' (line 40)
result_mod_40 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 26), '%', unicode_38, value_punc_39)

# Processing the call keyword arguments (line 40)
kwargs_41 = {}
# Getting the type of 're' (line 40)
re_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 're', False)
# Obtaining the member 'compile' of a type (line 40)
compile_37 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 15), re_36, 'compile')
# Calling compile(args, kwargs) (line 40)
compile_call_result_42 = invoke(stypy.reporting.localization.Localization(__file__, 40, 15), compile_37, *[result_mod_40], **kwargs_41)

# Obtaining the member 'sub' of a type (line 40)
sub_43 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 15), compile_call_result_42, 'sub')
# Assigning a type to the variable 'value_escape' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'value_escape', sub_43)
# Declaration of the 'FontconfigPatternParser' class

class FontconfigPatternParser(object, ):
    unicode_44 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, (-1)), 'unicode', u'A simple pyparsing-based parser for fontconfig-style patterns.\n\n    See the `fontconfig pattern specification\n    <https://www.freedesktop.org/software/fontconfig/fontconfig-user.html>`_\n    for more information.\n    ')
    
    # Assigning a Dict to a Name (line 50):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 77, 4, False)
        # Assigning a type to the variable 'self' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontconfigPatternParser.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 78):
        
        # Assigning a Call to a Name (line 78):
        
        # Call to setParseAction(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'self' (line 80)
        self_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 38), 'self', False)
        # Obtaining the member '_family' of a type (line 80)
        _family_55 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 38), self_54, '_family')
        # Processing the call keyword arguments (line 78)
        kwargs_56 = {}
        
        # Call to Regex(...): (line 78)
        # Processing the call arguments (line 78)
        unicode_46 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 28), 'unicode', u'([^%s]|(\\\\[%s]))*')
        
        # Obtaining an instance of the builtin type 'tuple' (line 79)
        tuple_47 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 79)
        # Adding element type (line 79)
        # Getting the type of 'family_punc' (line 79)
        family_punc_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 29), 'family_punc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 29), tuple_47, family_punc_48)
        # Adding element type (line 79)
        # Getting the type of 'family_punc' (line 79)
        family_punc_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 42), 'family_punc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 29), tuple_47, family_punc_49)
        
        # Applying the binary operator '%' (line 78)
        result_mod_50 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 28), '%', unicode_46, tuple_47)
        
        # Processing the call keyword arguments (line 78)
        kwargs_51 = {}
        # Getting the type of 'Regex' (line 78)
        Regex_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 22), 'Regex', False)
        # Calling Regex(args, kwargs) (line 78)
        Regex_call_result_52 = invoke(stypy.reporting.localization.Localization(__file__, 78, 22), Regex_45, *[result_mod_50], **kwargs_51)
        
        # Obtaining the member 'setParseAction' of a type (line 78)
        setParseAction_53 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 22), Regex_call_result_52, 'setParseAction')
        # Calling setParseAction(args, kwargs) (line 78)
        setParseAction_call_result_57 = invoke(stypy.reporting.localization.Localization(__file__, 78, 22), setParseAction_53, *[_family_55], **kwargs_56)
        
        # Assigning a type to the variable 'family' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'family', setParseAction_call_result_57)
        
        # Assigning a Call to a Name (line 81):
        
        # Assigning a Call to a Name (line 81):
        
        # Call to setParseAction(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'self' (line 82)
        self_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 38), 'self', False)
        # Obtaining the member '_size' of a type (line 82)
        _size_64 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 38), self_63, '_size')
        # Processing the call keyword arguments (line 81)
        kwargs_65 = {}
        
        # Call to Regex(...): (line 81)
        # Processing the call arguments (line 81)
        unicode_59 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 28), 'unicode', u'([0-9]+\\.?[0-9]*|\\.[0-9]+)')
        # Processing the call keyword arguments (line 81)
        kwargs_60 = {}
        # Getting the type of 'Regex' (line 81)
        Regex_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 22), 'Regex', False)
        # Calling Regex(args, kwargs) (line 81)
        Regex_call_result_61 = invoke(stypy.reporting.localization.Localization(__file__, 81, 22), Regex_58, *[unicode_59], **kwargs_60)
        
        # Obtaining the member 'setParseAction' of a type (line 81)
        setParseAction_62 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 22), Regex_call_result_61, 'setParseAction')
        # Calling setParseAction(args, kwargs) (line 81)
        setParseAction_call_result_66 = invoke(stypy.reporting.localization.Localization(__file__, 81, 22), setParseAction_62, *[_size_64], **kwargs_65)
        
        # Assigning a type to the variable 'size' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'size', setParseAction_call_result_66)
        
        # Assigning a Call to a Name (line 83):
        
        # Assigning a Call to a Name (line 83):
        
        # Call to setParseAction(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'self' (line 84)
        self_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 38), 'self', False)
        # Obtaining the member '_name' of a type (line 84)
        _name_73 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 38), self_72, '_name')
        # Processing the call keyword arguments (line 83)
        kwargs_74 = {}
        
        # Call to Regex(...): (line 83)
        # Processing the call arguments (line 83)
        unicode_68 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 28), 'unicode', u'[a-z]+')
        # Processing the call keyword arguments (line 83)
        kwargs_69 = {}
        # Getting the type of 'Regex' (line 83)
        Regex_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 22), 'Regex', False)
        # Calling Regex(args, kwargs) (line 83)
        Regex_call_result_70 = invoke(stypy.reporting.localization.Localization(__file__, 83, 22), Regex_67, *[unicode_68], **kwargs_69)
        
        # Obtaining the member 'setParseAction' of a type (line 83)
        setParseAction_71 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 22), Regex_call_result_70, 'setParseAction')
        # Calling setParseAction(args, kwargs) (line 83)
        setParseAction_call_result_75 = invoke(stypy.reporting.localization.Localization(__file__, 83, 22), setParseAction_71, *[_name_73], **kwargs_74)
        
        # Assigning a type to the variable 'name' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'name', setParseAction_call_result_75)
        
        # Assigning a Call to a Name (line 85):
        
        # Assigning a Call to a Name (line 85):
        
        # Call to setParseAction(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'self' (line 87)
        self_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 38), 'self', False)
        # Obtaining the member '_value' of a type (line 87)
        _value_86 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 38), self_85, '_value')
        # Processing the call keyword arguments (line 85)
        kwargs_87 = {}
        
        # Call to Regex(...): (line 85)
        # Processing the call arguments (line 85)
        unicode_77 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 28), 'unicode', u'([^%s]|(\\\\[%s]))*')
        
        # Obtaining an instance of the builtin type 'tuple' (line 86)
        tuple_78 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 86)
        # Adding element type (line 86)
        # Getting the type of 'value_punc' (line 86)
        value_punc_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 29), 'value_punc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 29), tuple_78, value_punc_79)
        # Adding element type (line 86)
        # Getting the type of 'value_punc' (line 86)
        value_punc_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 41), 'value_punc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 29), tuple_78, value_punc_80)
        
        # Applying the binary operator '%' (line 85)
        result_mod_81 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 28), '%', unicode_77, tuple_78)
        
        # Processing the call keyword arguments (line 85)
        kwargs_82 = {}
        # Getting the type of 'Regex' (line 85)
        Regex_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 22), 'Regex', False)
        # Calling Regex(args, kwargs) (line 85)
        Regex_call_result_83 = invoke(stypy.reporting.localization.Localization(__file__, 85, 22), Regex_76, *[result_mod_81], **kwargs_82)
        
        # Obtaining the member 'setParseAction' of a type (line 85)
        setParseAction_84 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 22), Regex_call_result_83, 'setParseAction')
        # Calling setParseAction(args, kwargs) (line 85)
        setParseAction_call_result_88 = invoke(stypy.reporting.localization.Localization(__file__, 85, 22), setParseAction_84, *[_value_86], **kwargs_87)
        
        # Assigning a type to the variable 'value' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'value', setParseAction_call_result_88)
        
        # Assigning a Call to a Name (line 89):
        
        # Assigning a Call to a Name (line 89):
        
        # Call to setParseAction(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'self' (line 93)
        self_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 37), 'self', False)
        # Obtaining the member '_families' of a type (line 93)
        _families_102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 37), self_101, '_families')
        # Processing the call keyword arguments (line 89)
        kwargs_103 = {}
        # Getting the type of 'family' (line 89)
        family_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 22), 'family', False)
        
        # Call to ZeroOrMore(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Call to Literal(...): (line 91)
        # Processing the call arguments (line 91)
        unicode_92 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 32), 'unicode', u',')
        # Processing the call keyword arguments (line 91)
        kwargs_93 = {}
        # Getting the type of 'Literal' (line 91)
        Literal_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 24), 'Literal', False)
        # Calling Literal(args, kwargs) (line 91)
        Literal_call_result_94 = invoke(stypy.reporting.localization.Localization(__file__, 91, 24), Literal_91, *[unicode_92], **kwargs_93)
        
        # Getting the type of 'family' (line 92)
        family_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'family', False)
        # Applying the binary operator '+' (line 91)
        result_add_96 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 24), '+', Literal_call_result_94, family_95)
        
        # Processing the call keyword arguments (line 90)
        kwargs_97 = {}
        # Getting the type of 'ZeroOrMore' (line 90)
        ZeroOrMore_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 22), 'ZeroOrMore', False)
        # Calling ZeroOrMore(args, kwargs) (line 90)
        ZeroOrMore_call_result_98 = invoke(stypy.reporting.localization.Localization(__file__, 90, 22), ZeroOrMore_90, *[result_add_96], **kwargs_97)
        
        # Applying the binary operator '+' (line 89)
        result_add_99 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 22), '+', family_89, ZeroOrMore_call_result_98)
        
        # Obtaining the member 'setParseAction' of a type (line 89)
        setParseAction_100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 22), result_add_99, 'setParseAction')
        # Calling setParseAction(args, kwargs) (line 89)
        setParseAction_call_result_104 = invoke(stypy.reporting.localization.Localization(__file__, 89, 22), setParseAction_100, *[_families_102], **kwargs_103)
        
        # Assigning a type to the variable 'families' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'families', setParseAction_call_result_104)
        
        # Assigning a Call to a Name (line 95):
        
        # Assigning a Call to a Name (line 95):
        
        # Call to setParseAction(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'self' (line 99)
        self_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 37), 'self', False)
        # Obtaining the member '_point_sizes' of a type (line 99)
        _point_sizes_118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 37), self_117, '_point_sizes')
        # Processing the call keyword arguments (line 95)
        kwargs_119 = {}
        # Getting the type of 'size' (line 95)
        size_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 22), 'size', False)
        
        # Call to ZeroOrMore(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Call to Literal(...): (line 97)
        # Processing the call arguments (line 97)
        unicode_108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 32), 'unicode', u',')
        # Processing the call keyword arguments (line 97)
        kwargs_109 = {}
        # Getting the type of 'Literal' (line 97)
        Literal_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), 'Literal', False)
        # Calling Literal(args, kwargs) (line 97)
        Literal_call_result_110 = invoke(stypy.reporting.localization.Localization(__file__, 97, 24), Literal_107, *[unicode_108], **kwargs_109)
        
        # Getting the type of 'size' (line 98)
        size_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 24), 'size', False)
        # Applying the binary operator '+' (line 97)
        result_add_112 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 24), '+', Literal_call_result_110, size_111)
        
        # Processing the call keyword arguments (line 96)
        kwargs_113 = {}
        # Getting the type of 'ZeroOrMore' (line 96)
        ZeroOrMore_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 22), 'ZeroOrMore', False)
        # Calling ZeroOrMore(args, kwargs) (line 96)
        ZeroOrMore_call_result_114 = invoke(stypy.reporting.localization.Localization(__file__, 96, 22), ZeroOrMore_106, *[result_add_112], **kwargs_113)
        
        # Applying the binary operator '+' (line 95)
        result_add_115 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 22), '+', size_105, ZeroOrMore_call_result_114)
        
        # Obtaining the member 'setParseAction' of a type (line 95)
        setParseAction_116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 22), result_add_115, 'setParseAction')
        # Calling setParseAction(args, kwargs) (line 95)
        setParseAction_call_result_120 = invoke(stypy.reporting.localization.Localization(__file__, 95, 22), setParseAction_116, *[_point_sizes_118], **kwargs_119)
        
        # Assigning a type to the variable 'point_sizes' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'point_sizes', setParseAction_call_result_120)
        
        # Assigning a Call to a Name (line 101):
        
        # Assigning a Call to a Name (line 101):
        
        # Call to setParseAction(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'self' (line 109)
        self_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 37), 'self', False)
        # Obtaining the member '_property' of a type (line 109)
        _property_149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 37), self_148, '_property')
        # Processing the call keyword arguments (line 101)
        kwargs_150 = {}
        # Getting the type of 'name' (line 101)
        name_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'name', False)
        
        # Call to Suppress(...): (line 102)
        # Processing the call arguments (line 102)
        
        # Call to Literal(...): (line 102)
        # Processing the call arguments (line 102)
        unicode_124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 41), 'unicode', u'=')
        # Processing the call keyword arguments (line 102)
        kwargs_125 = {}
        # Getting the type of 'Literal' (line 102)
        Literal_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 33), 'Literal', False)
        # Calling Literal(args, kwargs) (line 102)
        Literal_call_result_126 = invoke(stypy.reporting.localization.Localization(__file__, 102, 33), Literal_123, *[unicode_124], **kwargs_125)
        
        # Processing the call keyword arguments (line 102)
        kwargs_127 = {}
        # Getting the type of 'Suppress' (line 102)
        Suppress_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), 'Suppress', False)
        # Calling Suppress(args, kwargs) (line 102)
        Suppress_call_result_128 = invoke(stypy.reporting.localization.Localization(__file__, 102, 24), Suppress_122, *[Literal_call_result_126], **kwargs_127)
        
        # Applying the binary operator '+' (line 101)
        result_add_129 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 24), '+', name_121, Suppress_call_result_128)
        
        # Getting the type of 'value' (line 103)
        value_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 24), 'value', False)
        # Applying the binary operator '+' (line 103)
        result_add_131 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 22), '+', result_add_129, value_130)
        
        
        # Call to ZeroOrMore(...): (line 104)
        # Processing the call arguments (line 104)
        
        # Call to Suppress(...): (line 105)
        # Processing the call arguments (line 105)
        
        # Call to Literal(...): (line 105)
        # Processing the call arguments (line 105)
        unicode_135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 43), 'unicode', u',')
        # Processing the call keyword arguments (line 105)
        kwargs_136 = {}
        # Getting the type of 'Literal' (line 105)
        Literal_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 35), 'Literal', False)
        # Calling Literal(args, kwargs) (line 105)
        Literal_call_result_137 = invoke(stypy.reporting.localization.Localization(__file__, 105, 35), Literal_134, *[unicode_135], **kwargs_136)
        
        # Processing the call keyword arguments (line 105)
        kwargs_138 = {}
        # Getting the type of 'Suppress' (line 105)
        Suppress_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 26), 'Suppress', False)
        # Calling Suppress(args, kwargs) (line 105)
        Suppress_call_result_139 = invoke(stypy.reporting.localization.Localization(__file__, 105, 26), Suppress_133, *[Literal_call_result_137], **kwargs_138)
        
        # Getting the type of 'value' (line 106)
        value_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 26), 'value', False)
        # Applying the binary operator '+' (line 105)
        result_add_141 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 26), '+', Suppress_call_result_139, value_140)
        
        # Processing the call keyword arguments (line 104)
        kwargs_142 = {}
        # Getting the type of 'ZeroOrMore' (line 104)
        ZeroOrMore_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 24), 'ZeroOrMore', False)
        # Calling ZeroOrMore(args, kwargs) (line 104)
        ZeroOrMore_call_result_143 = invoke(stypy.reporting.localization.Localization(__file__, 104, 24), ZeroOrMore_132, *[result_add_141], **kwargs_142)
        
        # Applying the binary operator '+' (line 104)
        result_add_144 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 22), '+', result_add_131, ZeroOrMore_call_result_143)
        
        # Getting the type of 'name' (line 108)
        name_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), 'name', False)
        # Applying the binary operator '|' (line 101)
        result_or__146 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 23), '|', result_add_144, name_145)
        
        # Obtaining the member 'setParseAction' of a type (line 101)
        setParseAction_147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 23), result_or__146, 'setParseAction')
        # Calling setParseAction(args, kwargs) (line 101)
        setParseAction_call_result_151 = invoke(stypy.reporting.localization.Localization(__file__, 101, 23), setParseAction_147, *[_property_149], **kwargs_150)
        
        # Assigning a type to the variable 'property' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'property', setParseAction_call_result_151)
        
        # Assigning a BinOp to a Name (line 111):
        
        # Assigning a BinOp to a Name (line 111):
        
        # Call to Optional(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'families' (line 112)
        families_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 24), 'families', False)
        # Processing the call keyword arguments (line 111)
        kwargs_154 = {}
        # Getting the type of 'Optional' (line 111)
        Optional_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 22), 'Optional', False)
        # Calling Optional(args, kwargs) (line 111)
        Optional_call_result_155 = invoke(stypy.reporting.localization.Localization(__file__, 111, 22), Optional_152, *[families_153], **kwargs_154)
        
        
        # Call to Optional(...): (line 113)
        # Processing the call arguments (line 113)
        
        # Call to Literal(...): (line 114)
        # Processing the call arguments (line 114)
        unicode_158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 32), 'unicode', u'-')
        # Processing the call keyword arguments (line 114)
        kwargs_159 = {}
        # Getting the type of 'Literal' (line 114)
        Literal_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 24), 'Literal', False)
        # Calling Literal(args, kwargs) (line 114)
        Literal_call_result_160 = invoke(stypy.reporting.localization.Localization(__file__, 114, 24), Literal_157, *[unicode_158], **kwargs_159)
        
        # Getting the type of 'point_sizes' (line 115)
        point_sizes_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 24), 'point_sizes', False)
        # Applying the binary operator '+' (line 114)
        result_add_162 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 24), '+', Literal_call_result_160, point_sizes_161)
        
        # Processing the call keyword arguments (line 113)
        kwargs_163 = {}
        # Getting the type of 'Optional' (line 113)
        Optional_156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 22), 'Optional', False)
        # Calling Optional(args, kwargs) (line 113)
        Optional_call_result_164 = invoke(stypy.reporting.localization.Localization(__file__, 113, 22), Optional_156, *[result_add_162], **kwargs_163)
        
        # Applying the binary operator '+' (line 111)
        result_add_165 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 22), '+', Optional_call_result_155, Optional_call_result_164)
        
        
        # Call to ZeroOrMore(...): (line 116)
        # Processing the call arguments (line 116)
        
        # Call to Literal(...): (line 117)
        # Processing the call arguments (line 117)
        unicode_168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 32), 'unicode', u':')
        # Processing the call keyword arguments (line 117)
        kwargs_169 = {}
        # Getting the type of 'Literal' (line 117)
        Literal_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 24), 'Literal', False)
        # Calling Literal(args, kwargs) (line 117)
        Literal_call_result_170 = invoke(stypy.reporting.localization.Localization(__file__, 117, 24), Literal_167, *[unicode_168], **kwargs_169)
        
        # Getting the type of 'property' (line 118)
        property_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 24), 'property', False)
        # Applying the binary operator '+' (line 117)
        result_add_172 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 24), '+', Literal_call_result_170, property_171)
        
        # Processing the call keyword arguments (line 116)
        kwargs_173 = {}
        # Getting the type of 'ZeroOrMore' (line 116)
        ZeroOrMore_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 22), 'ZeroOrMore', False)
        # Calling ZeroOrMore(args, kwargs) (line 116)
        ZeroOrMore_call_result_174 = invoke(stypy.reporting.localization.Localization(__file__, 116, 22), ZeroOrMore_166, *[result_add_172], **kwargs_173)
        
        # Applying the binary operator '+' (line 116)
        result_add_175 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 20), '+', result_add_165, ZeroOrMore_call_result_174)
        
        
        # Call to StringEnd(...): (line 119)
        # Processing the call keyword arguments (line 119)
        kwargs_177 = {}
        # Getting the type of 'StringEnd' (line 119)
        StringEnd_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 22), 'StringEnd', False)
        # Calling StringEnd(args, kwargs) (line 119)
        StringEnd_call_result_178 = invoke(stypy.reporting.localization.Localization(__file__, 119, 22), StringEnd_176, *[], **kwargs_177)
        
        # Applying the binary operator '+' (line 119)
        result_add_179 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 20), '+', result_add_175, StringEnd_call_result_178)
        
        # Assigning a type to the variable 'pattern' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'pattern', result_add_179)
        
        # Assigning a Name to a Attribute (line 122):
        
        # Assigning a Name to a Attribute (line 122):
        # Getting the type of 'pattern' (line 122)
        pattern_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 23), 'pattern')
        # Getting the type of 'self' (line 122)
        self_181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'self')
        # Setting the type of the member '_parser' of a type (line 122)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), self_181, '_parser', pattern_180)
        
        # Assigning a Name to a Attribute (line 123):
        
        # Assigning a Name to a Attribute (line 123):
        # Getting the type of 'ParseException' (line 123)
        ParseException_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 30), 'ParseException')
        # Getting the type of 'self' (line 123)
        self_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'self')
        # Setting the type of the member 'ParseException' of a type (line 123)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), self_183, 'ParseException', ParseException_182)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def parse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'parse'
        module_type_store = module_type_store.open_function_context('parse', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontconfigPatternParser.parse.__dict__.__setitem__('stypy_localization', localization)
        FontconfigPatternParser.parse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontconfigPatternParser.parse.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontconfigPatternParser.parse.__dict__.__setitem__('stypy_function_name', 'FontconfigPatternParser.parse')
        FontconfigPatternParser.parse.__dict__.__setitem__('stypy_param_names_list', ['pattern'])
        FontconfigPatternParser.parse.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontconfigPatternParser.parse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontconfigPatternParser.parse.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontconfigPatternParser.parse.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontconfigPatternParser.parse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontconfigPatternParser.parse.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontconfigPatternParser.parse', ['pattern'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'parse', localization, ['pattern'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'parse(...)' code ##################

        unicode_184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, (-1)), 'unicode', u'\n        Parse the given fontconfig *pattern* and return a dictionary\n        of key/value pairs useful for initializing a\n        :class:`font_manager.FontProperties` object.\n        ')
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Dict to a Attribute (line 131):
        
        # Obtaining an instance of the builtin type 'dict' (line 131)
        dict_185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 35), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 131)
        
        # Getting the type of 'self' (line 131)
        self_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'self')
        # Setting the type of the member '_properties' of a type (line 131)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 16), self_186, '_properties', dict_185)
        
        # Assigning a Attribute to a Name (line 131):
        # Getting the type of 'self' (line 131)
        self_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'self')
        # Obtaining the member '_properties' of a type (line 131)
        _properties_188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 16), self_187, '_properties')
        # Assigning a type to the variable 'props' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'props', _properties_188)
        
        
        # SSA begins for try-except statement (line 132)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to parseString(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'pattern' (line 133)
        pattern_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 37), 'pattern', False)
        # Processing the call keyword arguments (line 133)
        kwargs_193 = {}
        # Getting the type of 'self' (line 133)
        self_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'self', False)
        # Obtaining the member '_parser' of a type (line 133)
        _parser_190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 12), self_189, '_parser')
        # Obtaining the member 'parseString' of a type (line 133)
        parseString_191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 12), _parser_190, 'parseString')
        # Calling parseString(args, kwargs) (line 133)
        parseString_call_result_194 = invoke(stypy.reporting.localization.Localization(__file__, 133, 12), parseString_191, *[pattern_192], **kwargs_193)
        
        # SSA branch for the except part of a try statement (line 132)
        # SSA branch for the except 'Attribute' branch of a try statement (line 132)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'self' (line 134)
        self_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'self')
        # Obtaining the member 'ParseException' of a type (line 134)
        ParseException_196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 15), self_195, 'ParseException')
        # Assigning a type to the variable 'e' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'e', ParseException_196)
        
        # Call to ValueError(...): (line 135)
        # Processing the call arguments (line 135)
        unicode_198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 16), 'unicode', u"Could not parse font string: '%s'\n%s")
        
        # Obtaining an instance of the builtin type 'tuple' (line 136)
        tuple_199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 59), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 136)
        # Adding element type (line 136)
        # Getting the type of 'pattern' (line 136)
        pattern_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 59), 'pattern', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 59), tuple_199, pattern_200)
        # Adding element type (line 136)
        # Getting the type of 'e' (line 136)
        e_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 68), 'e', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 59), tuple_199, e_201)
        
        # Applying the binary operator '%' (line 136)
        result_mod_202 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 16), '%', unicode_198, tuple_199)
        
        # Processing the call keyword arguments (line 135)
        kwargs_203 = {}
        # Getting the type of 'ValueError' (line 135)
        ValueError_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 135)
        ValueError_call_result_204 = invoke(stypy.reporting.localization.Localization(__file__, 135, 18), ValueError_197, *[result_mod_202], **kwargs_203)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 135, 12), ValueError_call_result_204, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 132)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 138):
        
        # Assigning a Name to a Attribute (line 138):
        # Getting the type of 'None' (line 138)
        None_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 27), 'None')
        # Getting the type of 'self' (line 138)
        self_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'self')
        # Setting the type of the member '_properties' of a type (line 138)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), self_206, '_properties', None_205)
        
        # Call to resetCache(...): (line 140)
        # Processing the call keyword arguments (line 140)
        kwargs_210 = {}
        # Getting the type of 'self' (line 140)
        self_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'self', False)
        # Obtaining the member '_parser' of a type (line 140)
        _parser_208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), self_207, '_parser')
        # Obtaining the member 'resetCache' of a type (line 140)
        resetCache_209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), _parser_208, 'resetCache')
        # Calling resetCache(args, kwargs) (line 140)
        resetCache_call_result_211 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), resetCache_209, *[], **kwargs_210)
        
        # Getting the type of 'props' (line 142)
        props_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 15), 'props')
        # Assigning a type to the variable 'stypy_return_type' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'stypy_return_type', props_212)
        
        # ################# End of 'parse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'parse' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_213)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'parse'
        return stypy_return_type_213


    @norecursion
    def _family(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_family'
        module_type_store = module_type_store.open_function_context('_family', 144, 4, False)
        # Assigning a type to the variable 'self' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontconfigPatternParser._family.__dict__.__setitem__('stypy_localization', localization)
        FontconfigPatternParser._family.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontconfigPatternParser._family.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontconfigPatternParser._family.__dict__.__setitem__('stypy_function_name', 'FontconfigPatternParser._family')
        FontconfigPatternParser._family.__dict__.__setitem__('stypy_param_names_list', ['s', 'loc', 'tokens'])
        FontconfigPatternParser._family.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontconfigPatternParser._family.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontconfigPatternParser._family.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontconfigPatternParser._family.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontconfigPatternParser._family.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontconfigPatternParser._family.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontconfigPatternParser._family', ['s', 'loc', 'tokens'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_family', localization, ['s', 'loc', 'tokens'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_family(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 145)
        list_214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 145)
        # Adding element type (line 145)
        
        # Call to family_unescape(...): (line 145)
        # Processing the call arguments (line 145)
        unicode_216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 32), 'unicode', u'\\1')
        
        # Call to str(...): (line 145)
        # Processing the call arguments (line 145)
        
        # Obtaining the type of the subscript
        int_218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 50), 'int')
        # Getting the type of 'tokens' (line 145)
        tokens_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 43), 'tokens', False)
        # Obtaining the member '__getitem__' of a type (line 145)
        getitem___220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 43), tokens_219, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 145)
        subscript_call_result_221 = invoke(stypy.reporting.localization.Localization(__file__, 145, 43), getitem___220, int_218)
        
        # Processing the call keyword arguments (line 145)
        kwargs_222 = {}
        # Getting the type of 'str' (line 145)
        str_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 39), 'str', False)
        # Calling str(args, kwargs) (line 145)
        str_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 145, 39), str_217, *[subscript_call_result_221], **kwargs_222)
        
        # Processing the call keyword arguments (line 145)
        kwargs_224 = {}
        # Getting the type of 'family_unescape' (line 145)
        family_unescape_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'family_unescape', False)
        # Calling family_unescape(args, kwargs) (line 145)
        family_unescape_call_result_225 = invoke(stypy.reporting.localization.Localization(__file__, 145, 16), family_unescape_215, *[unicode_216, str_call_result_223], **kwargs_224)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 15), list_214, family_unescape_call_result_225)
        
        # Assigning a type to the variable 'stypy_return_type' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'stypy_return_type', list_214)
        
        # ################# End of '_family(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_family' in the type store
        # Getting the type of 'stypy_return_type' (line 144)
        stypy_return_type_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_226)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_family'
        return stypy_return_type_226


    @norecursion
    def _size(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_size'
        module_type_store = module_type_store.open_function_context('_size', 147, 4, False)
        # Assigning a type to the variable 'self' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontconfigPatternParser._size.__dict__.__setitem__('stypy_localization', localization)
        FontconfigPatternParser._size.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontconfigPatternParser._size.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontconfigPatternParser._size.__dict__.__setitem__('stypy_function_name', 'FontconfigPatternParser._size')
        FontconfigPatternParser._size.__dict__.__setitem__('stypy_param_names_list', ['s', 'loc', 'tokens'])
        FontconfigPatternParser._size.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontconfigPatternParser._size.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontconfigPatternParser._size.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontconfigPatternParser._size.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontconfigPatternParser._size.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontconfigPatternParser._size.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontconfigPatternParser._size', ['s', 'loc', 'tokens'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_size', localization, ['s', 'loc', 'tokens'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_size(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 148)
        list_227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 148)
        # Adding element type (line 148)
        
        # Call to float(...): (line 148)
        # Processing the call arguments (line 148)
        
        # Obtaining the type of the subscript
        int_229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 29), 'int')
        # Getting the type of 'tokens' (line 148)
        tokens_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 22), 'tokens', False)
        # Obtaining the member '__getitem__' of a type (line 148)
        getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 22), tokens_230, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 148)
        subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 148, 22), getitem___231, int_229)
        
        # Processing the call keyword arguments (line 148)
        kwargs_233 = {}
        # Getting the type of 'float' (line 148)
        float_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'float', False)
        # Calling float(args, kwargs) (line 148)
        float_call_result_234 = invoke(stypy.reporting.localization.Localization(__file__, 148, 16), float_228, *[subscript_call_result_232], **kwargs_233)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 15), list_227, float_call_result_234)
        
        # Assigning a type to the variable 'stypy_return_type' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'stypy_return_type', list_227)
        
        # ################# End of '_size(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_size' in the type store
        # Getting the type of 'stypy_return_type' (line 147)
        stypy_return_type_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_235)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_size'
        return stypy_return_type_235


    @norecursion
    def _name(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_name'
        module_type_store = module_type_store.open_function_context('_name', 150, 4, False)
        # Assigning a type to the variable 'self' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontconfigPatternParser._name.__dict__.__setitem__('stypy_localization', localization)
        FontconfigPatternParser._name.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontconfigPatternParser._name.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontconfigPatternParser._name.__dict__.__setitem__('stypy_function_name', 'FontconfigPatternParser._name')
        FontconfigPatternParser._name.__dict__.__setitem__('stypy_param_names_list', ['s', 'loc', 'tokens'])
        FontconfigPatternParser._name.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontconfigPatternParser._name.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontconfigPatternParser._name.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontconfigPatternParser._name.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontconfigPatternParser._name.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontconfigPatternParser._name.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontconfigPatternParser._name', ['s', 'loc', 'tokens'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_name', localization, ['s', 'loc', 'tokens'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_name(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 151)
        list_236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 151)
        # Adding element type (line 151)
        
        # Call to str(...): (line 151)
        # Processing the call arguments (line 151)
        
        # Obtaining the type of the subscript
        int_238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 27), 'int')
        # Getting the type of 'tokens' (line 151)
        tokens_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 20), 'tokens', False)
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 20), tokens_239, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 151)
        subscript_call_result_241 = invoke(stypy.reporting.localization.Localization(__file__, 151, 20), getitem___240, int_238)
        
        # Processing the call keyword arguments (line 151)
        kwargs_242 = {}
        # Getting the type of 'str' (line 151)
        str_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'str', False)
        # Calling str(args, kwargs) (line 151)
        str_call_result_243 = invoke(stypy.reporting.localization.Localization(__file__, 151, 16), str_237, *[subscript_call_result_241], **kwargs_242)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 15), list_236, str_call_result_243)
        
        # Assigning a type to the variable 'stypy_return_type' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'stypy_return_type', list_236)
        
        # ################# End of '_name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_name' in the type store
        # Getting the type of 'stypy_return_type' (line 150)
        stypy_return_type_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_244)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_name'
        return stypy_return_type_244


    @norecursion
    def _value(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_value'
        module_type_store = module_type_store.open_function_context('_value', 153, 4, False)
        # Assigning a type to the variable 'self' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontconfigPatternParser._value.__dict__.__setitem__('stypy_localization', localization)
        FontconfigPatternParser._value.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontconfigPatternParser._value.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontconfigPatternParser._value.__dict__.__setitem__('stypy_function_name', 'FontconfigPatternParser._value')
        FontconfigPatternParser._value.__dict__.__setitem__('stypy_param_names_list', ['s', 'loc', 'tokens'])
        FontconfigPatternParser._value.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontconfigPatternParser._value.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontconfigPatternParser._value.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontconfigPatternParser._value.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontconfigPatternParser._value.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontconfigPatternParser._value.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontconfigPatternParser._value', ['s', 'loc', 'tokens'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_value', localization, ['s', 'loc', 'tokens'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_value(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 154)
        list_245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 154)
        # Adding element type (line 154)
        
        # Call to value_unescape(...): (line 154)
        # Processing the call arguments (line 154)
        unicode_247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 31), 'unicode', u'\\1')
        
        # Call to str(...): (line 154)
        # Processing the call arguments (line 154)
        
        # Obtaining the type of the subscript
        int_249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 49), 'int')
        # Getting the type of 'tokens' (line 154)
        tokens_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 42), 'tokens', False)
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 42), tokens_250, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_252 = invoke(stypy.reporting.localization.Localization(__file__, 154, 42), getitem___251, int_249)
        
        # Processing the call keyword arguments (line 154)
        kwargs_253 = {}
        # Getting the type of 'str' (line 154)
        str_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 38), 'str', False)
        # Calling str(args, kwargs) (line 154)
        str_call_result_254 = invoke(stypy.reporting.localization.Localization(__file__, 154, 38), str_248, *[subscript_call_result_252], **kwargs_253)
        
        # Processing the call keyword arguments (line 154)
        kwargs_255 = {}
        # Getting the type of 'value_unescape' (line 154)
        value_unescape_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'value_unescape', False)
        # Calling value_unescape(args, kwargs) (line 154)
        value_unescape_call_result_256 = invoke(stypy.reporting.localization.Localization(__file__, 154, 16), value_unescape_246, *[unicode_247, str_call_result_254], **kwargs_255)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 15), list_245, value_unescape_call_result_256)
        
        # Assigning a type to the variable 'stypy_return_type' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'stypy_return_type', list_245)
        
        # ################# End of '_value(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_value' in the type store
        # Getting the type of 'stypy_return_type' (line 153)
        stypy_return_type_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_257)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_value'
        return stypy_return_type_257


    @norecursion
    def _families(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_families'
        module_type_store = module_type_store.open_function_context('_families', 156, 4, False)
        # Assigning a type to the variable 'self' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontconfigPatternParser._families.__dict__.__setitem__('stypy_localization', localization)
        FontconfigPatternParser._families.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontconfigPatternParser._families.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontconfigPatternParser._families.__dict__.__setitem__('stypy_function_name', 'FontconfigPatternParser._families')
        FontconfigPatternParser._families.__dict__.__setitem__('stypy_param_names_list', ['s', 'loc', 'tokens'])
        FontconfigPatternParser._families.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontconfigPatternParser._families.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontconfigPatternParser._families.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontconfigPatternParser._families.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontconfigPatternParser._families.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontconfigPatternParser._families.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontconfigPatternParser._families', ['s', 'loc', 'tokens'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_families', localization, ['s', 'loc', 'tokens'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_families(...)' code ##################

        
        # Assigning a ListComp to a Subscript (line 157):
        
        # Assigning a ListComp to a Subscript (line 157):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'tokens' (line 157)
        tokens_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 54), 'tokens')
        comprehension_263 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 38), tokens_262)
        # Assigning a type to the variable 'x' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 38), 'x', comprehension_263)
        
        # Call to str(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'x' (line 157)
        x_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 42), 'x', False)
        # Processing the call keyword arguments (line 157)
        kwargs_260 = {}
        # Getting the type of 'str' (line 157)
        str_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 38), 'str', False)
        # Calling str(args, kwargs) (line 157)
        str_call_result_261 = invoke(stypy.reporting.localization.Localization(__file__, 157, 38), str_258, *[x_259], **kwargs_260)
        
        list_264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 38), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 38), list_264, str_call_result_261)
        # Getting the type of 'self' (line 157)
        self_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'self')
        # Obtaining the member '_properties' of a type (line 157)
        _properties_266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), self_265, '_properties')
        unicode_267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 25), 'unicode', u'family')
        # Storing an element on a container (line 157)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 8), _properties_266, (unicode_267, list_264))
        
        # Obtaining an instance of the builtin type 'list' (line 158)
        list_268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 158)
        
        # Assigning a type to the variable 'stypy_return_type' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'stypy_return_type', list_268)
        
        # ################# End of '_families(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_families' in the type store
        # Getting the type of 'stypy_return_type' (line 156)
        stypy_return_type_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_269)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_families'
        return stypy_return_type_269


    @norecursion
    def _point_sizes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_point_sizes'
        module_type_store = module_type_store.open_function_context('_point_sizes', 160, 4, False)
        # Assigning a type to the variable 'self' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontconfigPatternParser._point_sizes.__dict__.__setitem__('stypy_localization', localization)
        FontconfigPatternParser._point_sizes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontconfigPatternParser._point_sizes.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontconfigPatternParser._point_sizes.__dict__.__setitem__('stypy_function_name', 'FontconfigPatternParser._point_sizes')
        FontconfigPatternParser._point_sizes.__dict__.__setitem__('stypy_param_names_list', ['s', 'loc', 'tokens'])
        FontconfigPatternParser._point_sizes.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontconfigPatternParser._point_sizes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontconfigPatternParser._point_sizes.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontconfigPatternParser._point_sizes.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontconfigPatternParser._point_sizes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontconfigPatternParser._point_sizes.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontconfigPatternParser._point_sizes', ['s', 'loc', 'tokens'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_point_sizes', localization, ['s', 'loc', 'tokens'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_point_sizes(...)' code ##################

        
        # Assigning a ListComp to a Subscript (line 161):
        
        # Assigning a ListComp to a Subscript (line 161):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'tokens' (line 161)
        tokens_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 52), 'tokens')
        comprehension_275 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 36), tokens_274)
        # Assigning a type to the variable 'x' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 36), 'x', comprehension_275)
        
        # Call to str(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'x' (line 161)
        x_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 40), 'x', False)
        # Processing the call keyword arguments (line 161)
        kwargs_272 = {}
        # Getting the type of 'str' (line 161)
        str_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 36), 'str', False)
        # Calling str(args, kwargs) (line 161)
        str_call_result_273 = invoke(stypy.reporting.localization.Localization(__file__, 161, 36), str_270, *[x_271], **kwargs_272)
        
        list_276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 36), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 36), list_276, str_call_result_273)
        # Getting the type of 'self' (line 161)
        self_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'self')
        # Obtaining the member '_properties' of a type (line 161)
        _properties_278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 8), self_277, '_properties')
        unicode_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 25), 'unicode', u'size')
        # Storing an element on a container (line 161)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 8), _properties_278, (unicode_279, list_276))
        
        # Obtaining an instance of the builtin type 'list' (line 162)
        list_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 162)
        
        # Assigning a type to the variable 'stypy_return_type' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'stypy_return_type', list_280)
        
        # ################# End of '_point_sizes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_point_sizes' in the type store
        # Getting the type of 'stypy_return_type' (line 160)
        stypy_return_type_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_281)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_point_sizes'
        return stypy_return_type_281


    @norecursion
    def _property(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_property'
        module_type_store = module_type_store.open_function_context('_property', 164, 4, False)
        # Assigning a type to the variable 'self' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontconfigPatternParser._property.__dict__.__setitem__('stypy_localization', localization)
        FontconfigPatternParser._property.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontconfigPatternParser._property.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontconfigPatternParser._property.__dict__.__setitem__('stypy_function_name', 'FontconfigPatternParser._property')
        FontconfigPatternParser._property.__dict__.__setitem__('stypy_param_names_list', ['s', 'loc', 'tokens'])
        FontconfigPatternParser._property.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontconfigPatternParser._property.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontconfigPatternParser._property.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontconfigPatternParser._property.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontconfigPatternParser._property.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontconfigPatternParser._property.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontconfigPatternParser._property', ['s', 'loc', 'tokens'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_property', localization, ['s', 'loc', 'tokens'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_property(...)' code ##################

        
        
        
        # Call to len(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'tokens' (line 165)
        tokens_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 15), 'tokens', False)
        # Processing the call keyword arguments (line 165)
        kwargs_284 = {}
        # Getting the type of 'len' (line 165)
        len_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 11), 'len', False)
        # Calling len(args, kwargs) (line 165)
        len_call_result_285 = invoke(stypy.reporting.localization.Localization(__file__, 165, 11), len_282, *[tokens_283], **kwargs_284)
        
        int_286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 26), 'int')
        # Applying the binary operator '==' (line 165)
        result_eq_287 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 11), '==', len_call_result_285, int_286)
        
        # Testing the type of an if condition (line 165)
        if_condition_288 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 8), result_eq_287)
        # Assigning a type to the variable 'if_condition_288' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'if_condition_288', if_condition_288)
        # SSA begins for if statement (line 165)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Obtaining the type of the subscript
        int_289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 22), 'int')
        # Getting the type of 'tokens' (line 166)
        tokens_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 15), 'tokens')
        # Obtaining the member '__getitem__' of a type (line 166)
        getitem___291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 15), tokens_290, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 166)
        subscript_call_result_292 = invoke(stypy.reporting.localization.Localization(__file__, 166, 15), getitem___291, int_289)
        
        # Getting the type of 'self' (line 166)
        self_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 28), 'self')
        # Obtaining the member '_constants' of a type (line 166)
        _constants_294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 28), self_293, '_constants')
        # Applying the binary operator 'in' (line 166)
        result_contains_295 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 15), 'in', subscript_call_result_292, _constants_294)
        
        # Testing the type of an if condition (line 166)
        if_condition_296 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 166, 12), result_contains_295)
        # Assigning a type to the variable 'if_condition_296' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'if_condition_296', if_condition_296)
        # SSA begins for if statement (line 166)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Tuple (line 167):
        
        # Assigning a Subscript to a Name (line 167):
        
        # Obtaining the type of the subscript
        int_297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 16), 'int')
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 50), 'int')
        # Getting the type of 'tokens' (line 167)
        tokens_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 43), 'tokens')
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 43), tokens_299, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_301 = invoke(stypy.reporting.localization.Localization(__file__, 167, 43), getitem___300, int_298)
        
        # Getting the type of 'self' (line 167)
        self_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 27), 'self')
        # Obtaining the member '_constants' of a type (line 167)
        _constants_303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 27), self_302, '_constants')
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 27), _constants_303, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_305 = invoke(stypy.reporting.localization.Localization(__file__, 167, 27), getitem___304, subscript_call_result_301)
        
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 16), subscript_call_result_305, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_307 = invoke(stypy.reporting.localization.Localization(__file__, 167, 16), getitem___306, int_297)
        
        # Assigning a type to the variable 'tuple_var_assignment_1' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'tuple_var_assignment_1', subscript_call_result_307)
        
        # Assigning a Subscript to a Name (line 167):
        
        # Obtaining the type of the subscript
        int_308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 16), 'int')
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 50), 'int')
        # Getting the type of 'tokens' (line 167)
        tokens_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 43), 'tokens')
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 43), tokens_310, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_312 = invoke(stypy.reporting.localization.Localization(__file__, 167, 43), getitem___311, int_309)
        
        # Getting the type of 'self' (line 167)
        self_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 27), 'self')
        # Obtaining the member '_constants' of a type (line 167)
        _constants_314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 27), self_313, '_constants')
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 27), _constants_314, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_316 = invoke(stypy.reporting.localization.Localization(__file__, 167, 27), getitem___315, subscript_call_result_312)
        
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 16), subscript_call_result_316, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_318 = invoke(stypy.reporting.localization.Localization(__file__, 167, 16), getitem___317, int_308)
        
        # Assigning a type to the variable 'tuple_var_assignment_2' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'tuple_var_assignment_2', subscript_call_result_318)
        
        # Assigning a Name to a Name (line 167):
        # Getting the type of 'tuple_var_assignment_1' (line 167)
        tuple_var_assignment_1_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'tuple_var_assignment_1')
        # Assigning a type to the variable 'key' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'key', tuple_var_assignment_1_319)
        
        # Assigning a Name to a Name (line 167):
        # Getting the type of 'tuple_var_assignment_2' (line 167)
        tuple_var_assignment_2_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'tuple_var_assignment_2')
        # Assigning a type to the variable 'val' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 21), 'val', tuple_var_assignment_2_320)
        
        # Call to append(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'val' (line 168)
        val_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 60), 'val', False)
        # Processing the call keyword arguments (line 168)
        kwargs_330 = {}
        
        # Call to setdefault(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'key' (line 168)
        key_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 44), 'key', False)
        
        # Obtaining an instance of the builtin type 'list' (line 168)
        list_325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 168)
        
        # Processing the call keyword arguments (line 168)
        kwargs_326 = {}
        # Getting the type of 'self' (line 168)
        self_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'self', False)
        # Obtaining the member '_properties' of a type (line 168)
        _properties_322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 16), self_321, '_properties')
        # Obtaining the member 'setdefault' of a type (line 168)
        setdefault_323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 16), _properties_322, 'setdefault')
        # Calling setdefault(args, kwargs) (line 168)
        setdefault_call_result_327 = invoke(stypy.reporting.localization.Localization(__file__, 168, 16), setdefault_323, *[key_324, list_325], **kwargs_326)
        
        # Obtaining the member 'append' of a type (line 168)
        append_328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 16), setdefault_call_result_327, 'append')
        # Calling append(args, kwargs) (line 168)
        append_call_result_331 = invoke(stypy.reporting.localization.Localization(__file__, 168, 16), append_328, *[val_329], **kwargs_330)
        
        # SSA join for if statement (line 166)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 165)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 170):
        
        # Assigning a Subscript to a Name (line 170):
        
        # Obtaining the type of the subscript
        int_332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 25), 'int')
        # Getting the type of 'tokens' (line 170)
        tokens_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 18), 'tokens')
        # Obtaining the member '__getitem__' of a type (line 170)
        getitem___334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 18), tokens_333, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 170)
        subscript_call_result_335 = invoke(stypy.reporting.localization.Localization(__file__, 170, 18), getitem___334, int_332)
        
        # Assigning a type to the variable 'key' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'key', subscript_call_result_335)
        
        # Assigning a Subscript to a Name (line 171):
        
        # Assigning a Subscript to a Name (line 171):
        
        # Obtaining the type of the subscript
        int_336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 25), 'int')
        slice_337 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 171, 18), int_336, None, None)
        # Getting the type of 'tokens' (line 171)
        tokens_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 18), 'tokens')
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 18), tokens_338, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 171)
        subscript_call_result_340 = invoke(stypy.reporting.localization.Localization(__file__, 171, 18), getitem___339, slice_337)
        
        # Assigning a type to the variable 'val' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'val', subscript_call_result_340)
        
        # Call to extend(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'val' (line 172)
        val_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 56), 'val', False)
        # Processing the call keyword arguments (line 172)
        kwargs_350 = {}
        
        # Call to setdefault(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'key' (line 172)
        key_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 40), 'key', False)
        
        # Obtaining an instance of the builtin type 'list' (line 172)
        list_345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 172)
        
        # Processing the call keyword arguments (line 172)
        kwargs_346 = {}
        # Getting the type of 'self' (line 172)
        self_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'self', False)
        # Obtaining the member '_properties' of a type (line 172)
        _properties_342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), self_341, '_properties')
        # Obtaining the member 'setdefault' of a type (line 172)
        setdefault_343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), _properties_342, 'setdefault')
        # Calling setdefault(args, kwargs) (line 172)
        setdefault_call_result_347 = invoke(stypy.reporting.localization.Localization(__file__, 172, 12), setdefault_343, *[key_344, list_345], **kwargs_346)
        
        # Obtaining the member 'extend' of a type (line 172)
        extend_348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), setdefault_call_result_347, 'extend')
        # Calling extend(args, kwargs) (line 172)
        extend_call_result_351 = invoke(stypy.reporting.localization.Localization(__file__, 172, 12), extend_348, *[val_349], **kwargs_350)
        
        # SSA join for if statement (line 165)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'list' (line 173)
        list_352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 173)
        
        # Assigning a type to the variable 'stypy_return_type' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'stypy_return_type', list_352)
        
        # ################# End of '_property(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_property' in the type store
        # Getting the type of 'stypy_return_type' (line 164)
        stypy_return_type_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_353)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_property'
        return stypy_return_type_353


# Assigning a type to the variable 'FontconfigPatternParser' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'FontconfigPatternParser', FontconfigPatternParser)

# Assigning a Dict to a Name (line 50):

# Obtaining an instance of the builtin type 'dict' (line 50)
dict_354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 50)
# Adding element type (key, value) (line 50)
unicode_355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 8), 'unicode', u'thin')

# Obtaining an instance of the builtin type 'tuple' (line 51)
tuple_356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 51)
# Adding element type (line 51)
unicode_357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 28), 'unicode', u'weight')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 28), tuple_356, unicode_357)
# Adding element type (line 51)
unicode_358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 38), 'unicode', u'light')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 28), tuple_356, unicode_358)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_355, tuple_356))
# Adding element type (key, value) (line 50)
unicode_359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 8), 'unicode', u'extralight')

# Obtaining an instance of the builtin type 'tuple' (line 52)
tuple_360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 52)
# Adding element type (line 52)
unicode_361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 28), 'unicode', u'weight')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 28), tuple_360, unicode_361)
# Adding element type (line 52)
unicode_362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 38), 'unicode', u'light')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 28), tuple_360, unicode_362)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_359, tuple_360))
# Adding element type (key, value) (line 50)
unicode_363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 8), 'unicode', u'ultralight')

# Obtaining an instance of the builtin type 'tuple' (line 53)
tuple_364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 53)
# Adding element type (line 53)
unicode_365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 28), 'unicode', u'weight')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 28), tuple_364, unicode_365)
# Adding element type (line 53)
unicode_366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 38), 'unicode', u'light')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 28), tuple_364, unicode_366)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_363, tuple_364))
# Adding element type (key, value) (line 50)
unicode_367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 8), 'unicode', u'light')

# Obtaining an instance of the builtin type 'tuple' (line 54)
tuple_368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 54)
# Adding element type (line 54)
unicode_369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 28), 'unicode', u'weight')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 28), tuple_368, unicode_369)
# Adding element type (line 54)
unicode_370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 38), 'unicode', u'light')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 28), tuple_368, unicode_370)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_367, tuple_368))
# Adding element type (key, value) (line 50)
unicode_371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 8), 'unicode', u'book')

# Obtaining an instance of the builtin type 'tuple' (line 55)
tuple_372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 55)
# Adding element type (line 55)
unicode_373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 28), 'unicode', u'weight')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 28), tuple_372, unicode_373)
# Adding element type (line 55)
unicode_374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 38), 'unicode', u'book')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 28), tuple_372, unicode_374)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_371, tuple_372))
# Adding element type (key, value) (line 50)
unicode_375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 8), 'unicode', u'regular')

# Obtaining an instance of the builtin type 'tuple' (line 56)
tuple_376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 56)
# Adding element type (line 56)
unicode_377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 28), 'unicode', u'weight')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 28), tuple_376, unicode_377)
# Adding element type (line 56)
unicode_378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 38), 'unicode', u'regular')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 28), tuple_376, unicode_378)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_375, tuple_376))
# Adding element type (key, value) (line 50)
unicode_379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 8), 'unicode', u'normal')

# Obtaining an instance of the builtin type 'tuple' (line 57)
tuple_380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 57)
# Adding element type (line 57)
unicode_381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 28), 'unicode', u'weight')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 28), tuple_380, unicode_381)
# Adding element type (line 57)
unicode_382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 38), 'unicode', u'normal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 28), tuple_380, unicode_382)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_379, tuple_380))
# Adding element type (key, value) (line 50)
unicode_383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 8), 'unicode', u'medium')

# Obtaining an instance of the builtin type 'tuple' (line 58)
tuple_384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 58)
# Adding element type (line 58)
unicode_385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 28), 'unicode', u'weight')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 28), tuple_384, unicode_385)
# Adding element type (line 58)
unicode_386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 38), 'unicode', u'medium')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 28), tuple_384, unicode_386)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_383, tuple_384))
# Adding element type (key, value) (line 50)
unicode_387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 8), 'unicode', u'demibold')

# Obtaining an instance of the builtin type 'tuple' (line 59)
tuple_388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 59)
# Adding element type (line 59)
unicode_389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 28), 'unicode', u'weight')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 28), tuple_388, unicode_389)
# Adding element type (line 59)
unicode_390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 38), 'unicode', u'demibold')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 28), tuple_388, unicode_390)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_387, tuple_388))
# Adding element type (key, value) (line 50)
unicode_391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'unicode', u'semibold')

# Obtaining an instance of the builtin type 'tuple' (line 60)
tuple_392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 60)
# Adding element type (line 60)
unicode_393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 28), 'unicode', u'weight')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 28), tuple_392, unicode_393)
# Adding element type (line 60)
unicode_394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 38), 'unicode', u'semibold')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 28), tuple_392, unicode_394)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_391, tuple_392))
# Adding element type (key, value) (line 50)
unicode_395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 8), 'unicode', u'bold')

# Obtaining an instance of the builtin type 'tuple' (line 61)
tuple_396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 61)
# Adding element type (line 61)
unicode_397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 28), 'unicode', u'weight')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 28), tuple_396, unicode_397)
# Adding element type (line 61)
unicode_398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 38), 'unicode', u'bold')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 28), tuple_396, unicode_398)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_395, tuple_396))
# Adding element type (key, value) (line 50)
unicode_399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 8), 'unicode', u'extrabold')

# Obtaining an instance of the builtin type 'tuple' (line 62)
tuple_400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 62)
# Adding element type (line 62)
unicode_401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 28), 'unicode', u'weight')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 28), tuple_400, unicode_401)
# Adding element type (line 62)
unicode_402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 38), 'unicode', u'extra bold')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 28), tuple_400, unicode_402)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_399, tuple_400))
# Adding element type (key, value) (line 50)
unicode_403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 8), 'unicode', u'black')

# Obtaining an instance of the builtin type 'tuple' (line 63)
tuple_404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 63)
# Adding element type (line 63)
unicode_405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 28), 'unicode', u'weight')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 28), tuple_404, unicode_405)
# Adding element type (line 63)
unicode_406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 38), 'unicode', u'black')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 28), tuple_404, unicode_406)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_403, tuple_404))
# Adding element type (key, value) (line 50)
unicode_407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'unicode', u'heavy')

# Obtaining an instance of the builtin type 'tuple' (line 64)
tuple_408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 64)
# Adding element type (line 64)
unicode_409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 28), 'unicode', u'weight')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 28), tuple_408, unicode_409)
# Adding element type (line 64)
unicode_410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 38), 'unicode', u'heavy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 28), tuple_408, unicode_410)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_407, tuple_408))
# Adding element type (key, value) (line 50)
unicode_411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 8), 'unicode', u'roman')

# Obtaining an instance of the builtin type 'tuple' (line 65)
tuple_412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 65)
# Adding element type (line 65)
unicode_413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 28), 'unicode', u'slant')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 28), tuple_412, unicode_413)
# Adding element type (line 65)
unicode_414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 37), 'unicode', u'normal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 28), tuple_412, unicode_414)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_411, tuple_412))
# Adding element type (key, value) (line 50)
unicode_415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 8), 'unicode', u'italic')

# Obtaining an instance of the builtin type 'tuple' (line 66)
tuple_416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 66)
# Adding element type (line 66)
unicode_417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 28), 'unicode', u'slant')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 28), tuple_416, unicode_417)
# Adding element type (line 66)
unicode_418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 37), 'unicode', u'italic')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 28), tuple_416, unicode_418)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_415, tuple_416))
# Adding element type (key, value) (line 50)
unicode_419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 8), 'unicode', u'oblique')

# Obtaining an instance of the builtin type 'tuple' (line 67)
tuple_420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 67)
# Adding element type (line 67)
unicode_421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 28), 'unicode', u'slant')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 28), tuple_420, unicode_421)
# Adding element type (line 67)
unicode_422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 37), 'unicode', u'oblique')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 28), tuple_420, unicode_422)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_419, tuple_420))
# Adding element type (key, value) (line 50)
unicode_423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 8), 'unicode', u'ultracondensed')

# Obtaining an instance of the builtin type 'tuple' (line 68)
tuple_424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 68)
# Adding element type (line 68)
unicode_425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 28), 'unicode', u'width')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 28), tuple_424, unicode_425)
# Adding element type (line 68)
unicode_426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 37), 'unicode', u'ultra-condensed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 28), tuple_424, unicode_426)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_423, tuple_424))
# Adding element type (key, value) (line 50)
unicode_427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 8), 'unicode', u'extracondensed')

# Obtaining an instance of the builtin type 'tuple' (line 69)
tuple_428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 69)
# Adding element type (line 69)
unicode_429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 28), 'unicode', u'width')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 28), tuple_428, unicode_429)
# Adding element type (line 69)
unicode_430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 37), 'unicode', u'extra-condensed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 28), tuple_428, unicode_430)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_427, tuple_428))
# Adding element type (key, value) (line 50)
unicode_431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 8), 'unicode', u'condensed')

# Obtaining an instance of the builtin type 'tuple' (line 70)
tuple_432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 70)
# Adding element type (line 70)
unicode_433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 28), 'unicode', u'width')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 28), tuple_432, unicode_433)
# Adding element type (line 70)
unicode_434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 37), 'unicode', u'condensed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 28), tuple_432, unicode_434)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_431, tuple_432))
# Adding element type (key, value) (line 50)
unicode_435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 8), 'unicode', u'semicondensed')

# Obtaining an instance of the builtin type 'tuple' (line 71)
tuple_436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 71)
# Adding element type (line 71)
unicode_437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 28), 'unicode', u'width')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 28), tuple_436, unicode_437)
# Adding element type (line 71)
unicode_438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 37), 'unicode', u'semi-condensed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 28), tuple_436, unicode_438)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_435, tuple_436))
# Adding element type (key, value) (line 50)
unicode_439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 8), 'unicode', u'expanded')

# Obtaining an instance of the builtin type 'tuple' (line 72)
tuple_440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 72)
# Adding element type (line 72)
unicode_441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 28), 'unicode', u'width')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 28), tuple_440, unicode_441)
# Adding element type (line 72)
unicode_442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 37), 'unicode', u'expanded')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 28), tuple_440, unicode_442)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_439, tuple_440))
# Adding element type (key, value) (line 50)
unicode_443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 8), 'unicode', u'extraexpanded')

# Obtaining an instance of the builtin type 'tuple' (line 73)
tuple_444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 73)
# Adding element type (line 73)
unicode_445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 28), 'unicode', u'width')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 28), tuple_444, unicode_445)
# Adding element type (line 73)
unicode_446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 37), 'unicode', u'extra-expanded')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 28), tuple_444, unicode_446)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_443, tuple_444))
# Adding element type (key, value) (line 50)
unicode_447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 8), 'unicode', u'ultraexpanded')

# Obtaining an instance of the builtin type 'tuple' (line 74)
tuple_448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 74)
# Adding element type (line 74)
unicode_449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 28), 'unicode', u'width')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 28), tuple_448, unicode_449)
# Adding element type (line 74)
unicode_450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 37), 'unicode', u'ultra-expanded')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 28), tuple_448, unicode_450)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), dict_354, (unicode_447, tuple_448))

# Getting the type of 'FontconfigPatternParser'
FontconfigPatternParser_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FontconfigPatternParser')
# Setting the type of the member '_constants' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FontconfigPatternParser_451, '_constants', dict_354)

# Assigning a Call to a Name (line 180):

# Assigning a Call to a Name (line 180):

# Call to (...): (line 180)
# Processing the call arguments (line 180)

# Call to FontconfigPatternParser(...): (line 180)
# Processing the call keyword arguments (line 180)
kwargs_456 = {}
# Getting the type of 'FontconfigPatternParser' (line 180)
FontconfigPatternParser_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 39), 'FontconfigPatternParser', False)
# Calling FontconfigPatternParser(args, kwargs) (line 180)
FontconfigPatternParser_call_result_457 = invoke(stypy.reporting.localization.Localization(__file__, 180, 39), FontconfigPatternParser_455, *[], **kwargs_456)

# Obtaining the member 'parse' of a type (line 180)
parse_458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 39), FontconfigPatternParser_call_result_457, 'parse')
# Processing the call keyword arguments (line 180)
kwargs_459 = {}

# Call to lru_cache(...): (line 180)
# Processing the call keyword arguments (line 180)
kwargs_453 = {}
# Getting the type of 'lru_cache' (line 180)
lru_cache_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 27), 'lru_cache', False)
# Calling lru_cache(args, kwargs) (line 180)
lru_cache_call_result_454 = invoke(stypy.reporting.localization.Localization(__file__, 180, 27), lru_cache_452, *[], **kwargs_453)

# Calling (args, kwargs) (line 180)
_call_result_460 = invoke(stypy.reporting.localization.Localization(__file__, 180, 27), lru_cache_call_result_454, *[parse_458], **kwargs_459)

# Assigning a type to the variable 'parse_fontconfig_pattern' (line 180)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 0), 'parse_fontconfig_pattern', _call_result_460)

@norecursion
def generate_fontconfig_pattern(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'generate_fontconfig_pattern'
    module_type_store = module_type_store.open_function_context('generate_fontconfig_pattern', 183, 0, False)
    
    # Passed parameters checking function
    generate_fontconfig_pattern.stypy_localization = localization
    generate_fontconfig_pattern.stypy_type_of_self = None
    generate_fontconfig_pattern.stypy_type_store = module_type_store
    generate_fontconfig_pattern.stypy_function_name = 'generate_fontconfig_pattern'
    generate_fontconfig_pattern.stypy_param_names_list = ['d']
    generate_fontconfig_pattern.stypy_varargs_param_name = None
    generate_fontconfig_pattern.stypy_kwargs_param_name = None
    generate_fontconfig_pattern.stypy_call_defaults = defaults
    generate_fontconfig_pattern.stypy_call_varargs = varargs
    generate_fontconfig_pattern.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'generate_fontconfig_pattern', ['d'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'generate_fontconfig_pattern', localization, ['d'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'generate_fontconfig_pattern(...)' code ##################

    unicode_461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, (-1)), 'unicode', u'\n    Given a dictionary of key/value pairs, generates a fontconfig\n    pattern string.\n    ')
    
    # Assigning a List to a Name (line 188):
    
    # Assigning a List to a Name (line 188):
    
    # Obtaining an instance of the builtin type 'list' (line 188)
    list_462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 188)
    
    # Assigning a type to the variable 'props' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'props', list_462)
    
    # Assigning a Str to a Name (line 189):
    
    # Assigning a Str to a Name (line 189):
    unicode_463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 15), 'unicode', u'')
    # Assigning a type to the variable 'families' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'families', unicode_463)
    
    # Assigning a Str to a Name (line 190):
    
    # Assigning a Str to a Name (line 190):
    unicode_464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 11), 'unicode', u'')
    # Assigning a type to the variable 'size' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'size', unicode_464)
    
    
    # Call to split(...): (line 191)
    # Processing the call keyword arguments (line 191)
    kwargs_467 = {}
    unicode_465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 15), 'unicode', u'family style variant weight stretch file size')
    # Obtaining the member 'split' of a type (line 191)
    split_466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 15), unicode_465, 'split')
    # Calling split(args, kwargs) (line 191)
    split_call_result_468 = invoke(stypy.reporting.localization.Localization(__file__, 191, 15), split_466, *[], **kwargs_467)
    
    # Testing the type of a for loop iterable (line 191)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 191, 4), split_call_result_468)
    # Getting the type of the for loop variable (line 191)
    for_loop_var_469 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 191, 4), split_call_result_468)
    # Assigning a type to the variable 'key' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'key', for_loop_var_469)
    # SSA begins for a for statement (line 191)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 192):
    
    # Assigning a Call to a Name (line 192):
    
    # Call to (...): (line 192)
    # Processing the call keyword arguments (line 192)
    kwargs_477 = {}
    
    # Call to getattr(...): (line 192)
    # Processing the call arguments (line 192)
    # Getting the type of 'd' (line 192)
    d_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 22), 'd', False)
    unicode_472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 25), 'unicode', u'get_')
    # Getting the type of 'key' (line 192)
    key_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 34), 'key', False)
    # Applying the binary operator '+' (line 192)
    result_add_474 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 25), '+', unicode_472, key_473)
    
    # Processing the call keyword arguments (line 192)
    kwargs_475 = {}
    # Getting the type of 'getattr' (line 192)
    getattr_470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 14), 'getattr', False)
    # Calling getattr(args, kwargs) (line 192)
    getattr_call_result_476 = invoke(stypy.reporting.localization.Localization(__file__, 192, 14), getattr_470, *[d_471, result_add_474], **kwargs_475)
    
    # Calling (args, kwargs) (line 192)
    _call_result_478 = invoke(stypy.reporting.localization.Localization(__file__, 192, 14), getattr_call_result_476, *[], **kwargs_477)
    
    # Assigning a type to the variable 'val' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'val', _call_result_478)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'val' (line 193)
    val_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 11), 'val')
    # Getting the type of 'None' (line 193)
    None_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 22), 'None')
    # Applying the binary operator 'isnot' (line 193)
    result_is_not_481 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 11), 'isnot', val_479, None_480)
    
    
    # Getting the type of 'val' (line 193)
    val_482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 31), 'val')
    
    # Obtaining an instance of the builtin type 'list' (line 193)
    list_483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 193)
    
    # Applying the binary operator '!=' (line 193)
    result_ne_484 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 31), '!=', val_482, list_483)
    
    # Applying the binary operator 'and' (line 193)
    result_and_keyword_485 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 11), 'and', result_is_not_481, result_ne_484)
    
    # Testing the type of an if condition (line 193)
    if_condition_486 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 193, 8), result_and_keyword_485)
    # Assigning a type to the variable 'if_condition_486' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'if_condition_486', if_condition_486)
    # SSA begins for if statement (line 193)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 194)
    # Getting the type of 'val' (line 194)
    val_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 20), 'val')
    # Getting the type of 'list' (line 194)
    list_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 28), 'list')
    
    (may_be_489, more_types_in_union_490) = may_be_type(val_487, list_488)

    if may_be_489:

        if more_types_in_union_490:
            # Runtime conditional SSA (line 194)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'val' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'val', list_488())
        
        # Assigning a ListComp to a Name (line 195):
        
        # Assigning a ListComp to a Name (line 195):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'val' (line 195)
        val_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 62), 'val')
        comprehension_503 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 23), val_502)
        # Assigning a type to the variable 'x' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 23), 'x', comprehension_503)
        
        # Getting the type of 'x' (line 196)
        x_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 26), 'x')
        # Getting the type of 'None' (line 196)
        None_500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 35), 'None')
        # Applying the binary operator 'isnot' (line 196)
        result_is_not_501 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 26), 'isnot', x_499, None_500)
        
        
        # Call to value_escape(...): (line 195)
        # Processing the call arguments (line 195)
        unicode_492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 36), 'unicode', u'\\\\\\1')
        
        # Call to str(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'x' (line 195)
        x_494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 49), 'x', False)
        # Processing the call keyword arguments (line 195)
        kwargs_495 = {}
        # Getting the type of 'str' (line 195)
        str_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 45), 'str', False)
        # Calling str(args, kwargs) (line 195)
        str_call_result_496 = invoke(stypy.reporting.localization.Localization(__file__, 195, 45), str_493, *[x_494], **kwargs_495)
        
        # Processing the call keyword arguments (line 195)
        kwargs_497 = {}
        # Getting the type of 'value_escape' (line 195)
        value_escape_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 23), 'value_escape', False)
        # Calling value_escape(args, kwargs) (line 195)
        value_escape_call_result_498 = invoke(stypy.reporting.localization.Localization(__file__, 195, 23), value_escape_491, *[unicode_492, str_call_result_496], **kwargs_497)
        
        list_504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 23), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 23), list_504, value_escape_call_result_498)
        # Assigning a type to the variable 'val' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 16), 'val', list_504)
        
        
        # Getting the type of 'val' (line 197)
        val_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'val')
        
        # Obtaining an instance of the builtin type 'list' (line 197)
        list_506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 197)
        
        # Applying the binary operator '!=' (line 197)
        result_ne_507 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 19), '!=', val_505, list_506)
        
        # Testing the type of an if condition (line 197)
        if_condition_508 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 16), result_ne_507)
        # Assigning a type to the variable 'if_condition_508' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'if_condition_508', if_condition_508)
        # SSA begins for if statement (line 197)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 198):
        
        # Assigning a Call to a Name (line 198):
        
        # Call to join(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'val' (line 198)
        val_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 35), 'val', False)
        # Processing the call keyword arguments (line 198)
        kwargs_512 = {}
        unicode_509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 26), 'unicode', u',')
        # Obtaining the member 'join' of a type (line 198)
        join_510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 26), unicode_509, 'join')
        # Calling join(args, kwargs) (line 198)
        join_call_result_513 = invoke(stypy.reporting.localization.Localization(__file__, 198, 26), join_510, *[val_511], **kwargs_512)
        
        # Assigning a type to the variable 'val' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 20), 'val', join_call_result_513)
        # SSA join for if statement (line 197)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_490:
            # SSA join for if statement (line 194)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to append(...): (line 199)
    # Processing the call arguments (line 199)
    unicode_516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 25), 'unicode', u':%s=%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 199)
    tuple_517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 199)
    # Adding element type (line 199)
    # Getting the type of 'key' (line 199)
    key_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 37), 'key', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 37), tuple_517, key_518)
    # Adding element type (line 199)
    # Getting the type of 'val' (line 199)
    val_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 42), 'val', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 37), tuple_517, val_519)
    
    # Applying the binary operator '%' (line 199)
    result_mod_520 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 25), '%', unicode_516, tuple_517)
    
    # Processing the call keyword arguments (line 199)
    kwargs_521 = {}
    # Getting the type of 'props' (line 199)
    props_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'props', False)
    # Obtaining the member 'append' of a type (line 199)
    append_515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 12), props_514, 'append')
    # Calling append(args, kwargs) (line 199)
    append_call_result_522 = invoke(stypy.reporting.localization.Localization(__file__, 199, 12), append_515, *[result_mod_520], **kwargs_521)
    
    # SSA join for if statement (line 193)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to join(...): (line 200)
    # Processing the call arguments (line 200)
    # Getting the type of 'props' (line 200)
    props_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 19), 'props', False)
    # Processing the call keyword arguments (line 200)
    kwargs_526 = {}
    unicode_523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 11), 'unicode', u'')
    # Obtaining the member 'join' of a type (line 200)
    join_524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 11), unicode_523, 'join')
    # Calling join(args, kwargs) (line 200)
    join_call_result_527 = invoke(stypy.reporting.localization.Localization(__file__, 200, 11), join_524, *[props_525], **kwargs_526)
    
    # Assigning a type to the variable 'stypy_return_type' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'stypy_return_type', join_call_result_527)
    
    # ################# End of 'generate_fontconfig_pattern(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'generate_fontconfig_pattern' in the type store
    # Getting the type of 'stypy_return_type' (line 183)
    stypy_return_type_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_528)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'generate_fontconfig_pattern'
    return stypy_return_type_528

# Assigning a type to the variable 'generate_fontconfig_pattern' (line 183)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'generate_fontconfig_pattern', generate_fontconfig_pattern)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
