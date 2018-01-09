
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #
2: # distutils/version.py
3: #
4: # Implements multiple version numbering conventions for the
5: # Python Module Distribution Utilities.
6: #
7: # $Id$
8: #
9: 
10: '''Provides classes to represent module version numbers (one class for
11: each style of version numbering).  There are currently two such classes
12: implemented: StrictVersion and LooseVersion.
13: 
14: Every version number class implements the following interface:
15:   * the 'parse' method takes a string and parses it to some internal
16:     representation; if the string is an invalid version number,
17:     'parse' raises a ValueError exception
18:   * the class constructor takes an optional string argument which,
19:     if supplied, is passed to 'parse'
20:   * __str__ reconstructs the string that was passed to 'parse' (or
21:     an equivalent string -- ie. one that will generate an equivalent
22:     version number instance)
23:   * __repr__ generates Python code to recreate the version number instance
24:   * __cmp__ compares the current instance with either another instance
25:     of the same class or a string (which will be parsed to an instance
26:     of the same class, thus must follow the same rules)
27: '''
28: 
29: import string, re
30: from types import StringType
31: 
32: class Version:
33:     '''Abstract base class for version numbering classes.  Just provides
34:     constructor (__init__) and reproducer (__repr__), because those
35:     seem to be the same for all version numbering classes.
36:     '''
37: 
38:     def __init__ (self, vstring=None):
39:         if vstring:
40:             self.parse(vstring)
41: 
42:     def __repr__ (self):
43:         return "%s ('%s')" % (self.__class__.__name__, str(self))
44: 
45: 
46: # Interface for version-number classes -- must be implemented
47: # by the following classes (the concrete ones -- Version should
48: # be treated as an abstract class).
49: #    __init__ (string) - create and take same action as 'parse'
50: #                        (string parameter is optional)
51: #    parse (string)    - convert a string representation to whatever
52: #                        internal representation is appropriate for
53: #                        this style of version numbering
54: #    __str__ (self)    - convert back to a string; should be very similar
55: #                        (if not identical to) the string supplied to parse
56: #    __repr__ (self)   - generate Python code to recreate
57: #                        the instance
58: #    __cmp__ (self, other) - compare two version numbers ('other' may
59: #                        be an unparsed version string, or another
60: #                        instance of your version class)
61: 
62: 
63: class StrictVersion (Version):
64: 
65:     '''Version numbering for anal retentives and software idealists.
66:     Implements the standard interface for version number classes as
67:     described above.  A version number consists of two or three
68:     dot-separated numeric components, with an optional "pre-release" tag
69:     on the end.  The pre-release tag consists of the letter 'a' or 'b'
70:     followed by a number.  If the numeric components of two version
71:     numbers are equal, then one with a pre-release tag will always
72:     be deemed earlier (lesser) than one without.
73: 
74:     The following are valid version numbers (shown in the order that
75:     would be obtained by sorting according to the supplied cmp function):
76: 
77:         0.4       0.4.0  (these two are equivalent)
78:         0.4.1
79:         0.5a1
80:         0.5b3
81:         0.5
82:         0.9.6
83:         1.0
84:         1.0.4a3
85:         1.0.4b1
86:         1.0.4
87: 
88:     The following are examples of invalid version numbers:
89: 
90:         1
91:         2.7.2.2
92:         1.3.a4
93:         1.3pl1
94:         1.3c4
95: 
96:     The rationale for this version numbering system will be explained
97:     in the distutils documentation.
98:     '''
99: 
100:     version_re = re.compile(r'^(\d+) \. (\d+) (\. (\d+))? ([ab](\d+))?$',
101:                             re.VERBOSE)
102: 
103: 
104:     def parse (self, vstring):
105:         match = self.version_re.match(vstring)
106:         if not match:
107:             raise ValueError, "invalid version number '%s'" % vstring
108: 
109:         (major, minor, patch, prerelease, prerelease_num) = \
110:             match.group(1, 2, 4, 5, 6)
111: 
112:         if patch:
113:             self.version = tuple(map(string.atoi, [major, minor, patch]))
114:         else:
115:             self.version = tuple(map(string.atoi, [major, minor]) + [0])
116: 
117:         if prerelease:
118:             self.prerelease = (prerelease[0], string.atoi(prerelease_num))
119:         else:
120:             self.prerelease = None
121: 
122: 
123:     def __str__ (self):
124: 
125:         if self.version[2] == 0:
126:             vstring = string.join(map(str, self.version[0:2]), '.')
127:         else:
128:             vstring = string.join(map(str, self.version), '.')
129: 
130:         if self.prerelease:
131:             vstring = vstring + self.prerelease[0] + str(self.prerelease[1])
132: 
133:         return vstring
134: 
135: 
136:     def __cmp__ (self, other):
137:         if isinstance(other, StringType):
138:             other = StrictVersion(other)
139: 
140:         compare = cmp(self.version, other.version)
141:         if (compare == 0):              # have to compare prerelease
142: 
143:             # case 1: neither has prerelease; they're equal
144:             # case 2: self has prerelease, other doesn't; other is greater
145:             # case 3: self doesn't have prerelease, other does: self is greater
146:             # case 4: both have prerelease: must compare them!
147: 
148:             if (not self.prerelease and not other.prerelease):
149:                 return 0
150:             elif (self.prerelease and not other.prerelease):
151:                 return -1
152:             elif (not self.prerelease and other.prerelease):
153:                 return 1
154:             elif (self.prerelease and other.prerelease):
155:                 return cmp(self.prerelease, other.prerelease)
156: 
157:         else:                           # numeric versions don't match --
158:             return compare              # prerelease stuff doesn't matter
159: 
160: 
161: # end class StrictVersion
162: 
163: 
164: # The rules according to Greg Stein:
165: # 1) a version number has 1 or more numbers separated by a period or by
166: #    sequences of letters. If only periods, then these are compared
167: #    left-to-right to determine an ordering.
168: # 2) sequences of letters are part of the tuple for comparison and are
169: #    compared lexicographically
170: # 3) recognize the numeric components may have leading zeroes
171: #
172: # The LooseVersion class below implements these rules: a version number
173: # string is split up into a tuple of integer and string components, and
174: # comparison is a simple tuple comparison.  This means that version
175: # numbers behave in a predictable and obvious way, but a way that might
176: # not necessarily be how people *want* version numbers to behave.  There
177: # wouldn't be a problem if people could stick to purely numeric version
178: # numbers: just split on period and compare the numbers as tuples.
179: # However, people insist on putting letters into their version numbers;
180: # the most common purpose seems to be:
181: #   - indicating a "pre-release" version
182: #     ('alpha', 'beta', 'a', 'b', 'pre', 'p')
183: #   - indicating a post-release patch ('p', 'pl', 'patch')
184: # but of course this can't cover all version number schemes, and there's
185: # no way to know what a programmer means without asking him.
186: #
187: # The problem is what to do with letters (and other non-numeric
188: # characters) in a version number.  The current implementation does the
189: # obvious and predictable thing: keep them as strings and compare
190: # lexically within a tuple comparison.  This has the desired effect if
191: # an appended letter sequence implies something "post-release":
192: # eg. "0.99" < "0.99pl14" < "1.0", and "5.001" < "5.001m" < "5.002".
193: #
194: # However, if letters in a version number imply a pre-release version,
195: # the "obvious" thing isn't correct.  Eg. you would expect that
196: # "1.5.1" < "1.5.2a2" < "1.5.2", but under the tuple/lexical comparison
197: # implemented here, this just isn't so.
198: #
199: # Two possible solutions come to mind.  The first is to tie the
200: # comparison algorithm to a particular set of semantic rules, as has
201: # been done in the StrictVersion class above.  This works great as long
202: # as everyone can go along with bondage and discipline.  Hopefully a
203: # (large) subset of Python module programmers will agree that the
204: # particular flavour of bondage and discipline provided by StrictVersion
205: # provides enough benefit to be worth using, and will submit their
206: # version numbering scheme to its domination.  The free-thinking
207: # anarchists in the lot will never give in, though, and something needs
208: # to be done to accommodate them.
209: #
210: # Perhaps a "moderately strict" version class could be implemented that
211: # lets almost anything slide (syntactically), and makes some heuristic
212: # assumptions about non-digits in version number strings.  This could
213: # sink into special-case-hell, though; if I was as talented and
214: # idiosyncratic as Larry Wall, I'd go ahead and implement a class that
215: # somehow knows that "1.2.1" < "1.2.2a2" < "1.2.2" < "1.2.2pl3", and is
216: # just as happy dealing with things like "2g6" and "1.13++".  I don't
217: # think I'm smart enough to do it right though.
218: #
219: # In any case, I've coded the test suite for this module (see
220: # ../test/test_version.py) specifically to fail on things like comparing
221: # "1.2a2" and "1.2".  That's not because the *code* is doing anything
222: # wrong, it's because the simple, obvious design doesn't match my
223: # complicated, hairy expectations for real-world version numbers.  It
224: # would be a snap to fix the test suite to say, "Yep, LooseVersion does
225: # the Right Thing" (ie. the code matches the conception).  But I'd rather
226: # have a conception that matches common notions about version numbers.
227: 
228: class LooseVersion (Version):
229: 
230:     '''Version numbering for anarchists and software realists.
231:     Implements the standard interface for version number classes as
232:     described above.  A version number consists of a series of numbers,
233:     separated by either periods or strings of letters.  When comparing
234:     version numbers, the numeric components will be compared
235:     numerically, and the alphabetic components lexically.  The following
236:     are all valid version numbers, in no particular order:
237: 
238:         1.5.1
239:         1.5.2b2
240:         161
241:         3.10a
242:         8.02
243:         3.4j
244:         1996.07.12
245:         3.2.pl0
246:         3.1.1.6
247:         2g6
248:         11g
249:         0.960923
250:         2.2beta29
251:         1.13++
252:         5.5.kw
253:         2.0b1pl0
254: 
255:     In fact, there is no such thing as an invalid version number under
256:     this scheme; the rules for comparison are simple and predictable,
257:     but may not always give the results you want (for some definition
258:     of "want").
259:     '''
260: 
261:     component_re = re.compile(r'(\d+ | [a-z]+ | \.)', re.VERBOSE)
262: 
263:     def __init__ (self, vstring=None):
264:         if vstring:
265:             self.parse(vstring)
266: 
267: 
268:     def parse (self, vstring):
269:         # I've given up on thinking I can reconstruct the version string
270:         # from the parsed tuple -- so I just store the string here for
271:         # use by __str__
272:         self.vstring = vstring
273:         components = filter(lambda x: x and x != '.',
274:                             self.component_re.split(vstring))
275:         for i in range(len(components)):
276:             try:
277:                 components[i] = int(components[i])
278:             except ValueError:
279:                 pass
280: 
281:         self.version = components
282: 
283: 
284:     def __str__ (self):
285:         return self.vstring
286: 
287: 
288:     def __repr__ (self):
289:         return "LooseVersion ('%s')" % str(self)
290: 
291: 
292:     def __cmp__ (self, other):
293:         if isinstance(other, StringType):
294:             other = LooseVersion(other)
295: 
296:         return cmp(self.version, other.version)
297: 
298: 
299: # end class LooseVersion
300: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_11037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, (-1)), 'str', "Provides classes to represent module version numbers (one class for\neach style of version numbering).  There are currently two such classes\nimplemented: StrictVersion and LooseVersion.\n\nEvery version number class implements the following interface:\n  * the 'parse' method takes a string and parses it to some internal\n    representation; if the string is an invalid version number,\n    'parse' raises a ValueError exception\n  * the class constructor takes an optional string argument which,\n    if supplied, is passed to 'parse'\n  * __str__ reconstructs the string that was passed to 'parse' (or\n    an equivalent string -- ie. one that will generate an equivalent\n    version number instance)\n  * __repr__ generates Python code to recreate the version number instance\n  * __cmp__ compares the current instance with either another instance\n    of the same class or a string (which will be parsed to an instance\n    of the same class, thus must follow the same rules)\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# Multiple import statement. import string (1/2) (line 29)
import string

import_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'string', string, module_type_store)
# Multiple import statement. import re (2/2) (line 29)
import re

import_module(stypy.reporting.localization.Localization(__file__, 29, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'from types import StringType' statement (line 30)
try:
    from types import StringType

except:
    StringType = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'types', None, module_type_store, ['StringType'], [StringType])

# Declaration of the 'Version' class

class Version:
    str_11038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, (-1)), 'str', 'Abstract base class for version numbering classes.  Just provides\n    constructor (__init__) and reproducer (__repr__), because those\n    seem to be the same for all version numbering classes.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 38)
        None_11039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 32), 'None')
        defaults = [None_11039]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Version.__init__', ['vstring'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['vstring'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Getting the type of 'vstring' (line 39)
        vstring_11040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 11), 'vstring')
        # Testing the type of an if condition (line 39)
        if_condition_11041 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 8), vstring_11040)
        # Assigning a type to the variable 'if_condition_11041' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'if_condition_11041', if_condition_11041)
        # SSA begins for if statement (line 39)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to parse(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'vstring' (line 40)
        vstring_11044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 23), 'vstring', False)
        # Processing the call keyword arguments (line 40)
        kwargs_11045 = {}
        # Getting the type of 'self' (line 40)
        self_11042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'self', False)
        # Obtaining the member 'parse' of a type (line 40)
        parse_11043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 12), self_11042, 'parse')
        # Calling parse(args, kwargs) (line 40)
        parse_call_result_11046 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), parse_11043, *[vstring_11044], **kwargs_11045)
        
        # SSA join for if statement (line 39)
        module_type_store = module_type_store.join_ssa_context()
        
        
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
        module_type_store = module_type_store.open_function_context('__repr__', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Version.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        Version.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Version.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Version.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Version.stypy__repr__')
        Version.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        Version.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Version.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Version.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Version.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Version.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Version.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Version.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        str_11047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 15), 'str', "%s ('%s')")
        
        # Obtaining an instance of the builtin type 'tuple' (line 43)
        tuple_11048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 43)
        # Adding element type (line 43)
        # Getting the type of 'self' (line 43)
        self_11049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 30), 'self')
        # Obtaining the member '__class__' of a type (line 43)
        class___11050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 30), self_11049, '__class__')
        # Obtaining the member '__name__' of a type (line 43)
        name___11051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 30), class___11050, '__name__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 30), tuple_11048, name___11051)
        # Adding element type (line 43)
        
        # Call to str(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'self' (line 43)
        self_11053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 59), 'self', False)
        # Processing the call keyword arguments (line 43)
        kwargs_11054 = {}
        # Getting the type of 'str' (line 43)
        str_11052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 55), 'str', False)
        # Calling str(args, kwargs) (line 43)
        str_call_result_11055 = invoke(stypy.reporting.localization.Localization(__file__, 43, 55), str_11052, *[self_11053], **kwargs_11054)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 30), tuple_11048, str_call_result_11055)
        
        # Applying the binary operator '%' (line 43)
        result_mod_11056 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 15), '%', str_11047, tuple_11048)
        
        # Assigning a type to the variable 'stypy_return_type' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'stypy_return_type', result_mod_11056)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_11057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11057)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_11057


# Assigning a type to the variable 'Version' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'Version', Version)
# Declaration of the 'StrictVersion' class
# Getting the type of 'Version' (line 63)
Version_11058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 21), 'Version')

class StrictVersion(Version_11058, ):
    str_11059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, (-1)), 'str', 'Version numbering for anal retentives and software idealists.\n    Implements the standard interface for version number classes as\n    described above.  A version number consists of two or three\n    dot-separated numeric components, with an optional "pre-release" tag\n    on the end.  The pre-release tag consists of the letter \'a\' or \'b\'\n    followed by a number.  If the numeric components of two version\n    numbers are equal, then one with a pre-release tag will always\n    be deemed earlier (lesser) than one without.\n\n    The following are valid version numbers (shown in the order that\n    would be obtained by sorting according to the supplied cmp function):\n\n        0.4       0.4.0  (these two are equivalent)\n        0.4.1\n        0.5a1\n        0.5b3\n        0.5\n        0.9.6\n        1.0\n        1.0.4a3\n        1.0.4b1\n        1.0.4\n\n    The following are examples of invalid version numbers:\n\n        1\n        2.7.2.2\n        1.3.a4\n        1.3pl1\n        1.3c4\n\n    The rationale for this version numbering system will be explained\n    in the distutils documentation.\n    ')
    
    # Assigning a Call to a Name (line 100):

    @norecursion
    def parse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'parse'
        module_type_store = module_type_store.open_function_context('parse', 104, 4, False)
        # Assigning a type to the variable 'self' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StrictVersion.parse.__dict__.__setitem__('stypy_localization', localization)
        StrictVersion.parse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StrictVersion.parse.__dict__.__setitem__('stypy_type_store', module_type_store)
        StrictVersion.parse.__dict__.__setitem__('stypy_function_name', 'StrictVersion.parse')
        StrictVersion.parse.__dict__.__setitem__('stypy_param_names_list', ['vstring'])
        StrictVersion.parse.__dict__.__setitem__('stypy_varargs_param_name', None)
        StrictVersion.parse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StrictVersion.parse.__dict__.__setitem__('stypy_call_defaults', defaults)
        StrictVersion.parse.__dict__.__setitem__('stypy_call_varargs', varargs)
        StrictVersion.parse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StrictVersion.parse.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StrictVersion.parse', ['vstring'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'parse', localization, ['vstring'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'parse(...)' code ##################

        
        # Assigning a Call to a Name (line 105):
        
        # Assigning a Call to a Name (line 105):
        
        # Call to match(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'vstring' (line 105)
        vstring_11063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 38), 'vstring', False)
        # Processing the call keyword arguments (line 105)
        kwargs_11064 = {}
        # Getting the type of 'self' (line 105)
        self_11060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'self', False)
        # Obtaining the member 'version_re' of a type (line 105)
        version_re_11061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 16), self_11060, 'version_re')
        # Obtaining the member 'match' of a type (line 105)
        match_11062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 16), version_re_11061, 'match')
        # Calling match(args, kwargs) (line 105)
        match_call_result_11065 = invoke(stypy.reporting.localization.Localization(__file__, 105, 16), match_11062, *[vstring_11063], **kwargs_11064)
        
        # Assigning a type to the variable 'match' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'match', match_call_result_11065)
        
        
        # Getting the type of 'match' (line 106)
        match_11066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 15), 'match')
        # Applying the 'not' unary operator (line 106)
        result_not__11067 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 11), 'not', match_11066)
        
        # Testing the type of an if condition (line 106)
        if_condition_11068 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 8), result_not__11067)
        # Assigning a type to the variable 'if_condition_11068' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'if_condition_11068', if_condition_11068)
        # SSA begins for if statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'ValueError' (line 107)
        ValueError_11069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 18), 'ValueError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 107, 12), ValueError_11069, 'raise parameter', BaseException)
        # SSA join for if statement (line 106)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 109):
        
        # Assigning a Subscript to a Name (line 109):
        
        # Obtaining the type of the subscript
        int_11070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 8), 'int')
        
        # Call to group(...): (line 110)
        # Processing the call arguments (line 110)
        int_11073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 24), 'int')
        int_11074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 27), 'int')
        int_11075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 30), 'int')
        int_11076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 33), 'int')
        int_11077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 36), 'int')
        # Processing the call keyword arguments (line 110)
        kwargs_11078 = {}
        # Getting the type of 'match' (line 110)
        match_11071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'match', False)
        # Obtaining the member 'group' of a type (line 110)
        group_11072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), match_11071, 'group')
        # Calling group(args, kwargs) (line 110)
        group_call_result_11079 = invoke(stypy.reporting.localization.Localization(__file__, 110, 12), group_11072, *[int_11073, int_11074, int_11075, int_11076, int_11077], **kwargs_11078)
        
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___11080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), group_call_result_11079, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_11081 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), getitem___11080, int_11070)
        
        # Assigning a type to the variable 'tuple_var_assignment_11032' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'tuple_var_assignment_11032', subscript_call_result_11081)
        
        # Assigning a Subscript to a Name (line 109):
        
        # Obtaining the type of the subscript
        int_11082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 8), 'int')
        
        # Call to group(...): (line 110)
        # Processing the call arguments (line 110)
        int_11085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 24), 'int')
        int_11086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 27), 'int')
        int_11087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 30), 'int')
        int_11088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 33), 'int')
        int_11089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 36), 'int')
        # Processing the call keyword arguments (line 110)
        kwargs_11090 = {}
        # Getting the type of 'match' (line 110)
        match_11083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'match', False)
        # Obtaining the member 'group' of a type (line 110)
        group_11084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), match_11083, 'group')
        # Calling group(args, kwargs) (line 110)
        group_call_result_11091 = invoke(stypy.reporting.localization.Localization(__file__, 110, 12), group_11084, *[int_11085, int_11086, int_11087, int_11088, int_11089], **kwargs_11090)
        
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___11092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), group_call_result_11091, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_11093 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), getitem___11092, int_11082)
        
        # Assigning a type to the variable 'tuple_var_assignment_11033' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'tuple_var_assignment_11033', subscript_call_result_11093)
        
        # Assigning a Subscript to a Name (line 109):
        
        # Obtaining the type of the subscript
        int_11094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 8), 'int')
        
        # Call to group(...): (line 110)
        # Processing the call arguments (line 110)
        int_11097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 24), 'int')
        int_11098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 27), 'int')
        int_11099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 30), 'int')
        int_11100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 33), 'int')
        int_11101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 36), 'int')
        # Processing the call keyword arguments (line 110)
        kwargs_11102 = {}
        # Getting the type of 'match' (line 110)
        match_11095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'match', False)
        # Obtaining the member 'group' of a type (line 110)
        group_11096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), match_11095, 'group')
        # Calling group(args, kwargs) (line 110)
        group_call_result_11103 = invoke(stypy.reporting.localization.Localization(__file__, 110, 12), group_11096, *[int_11097, int_11098, int_11099, int_11100, int_11101], **kwargs_11102)
        
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___11104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), group_call_result_11103, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_11105 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), getitem___11104, int_11094)
        
        # Assigning a type to the variable 'tuple_var_assignment_11034' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'tuple_var_assignment_11034', subscript_call_result_11105)
        
        # Assigning a Subscript to a Name (line 109):
        
        # Obtaining the type of the subscript
        int_11106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 8), 'int')
        
        # Call to group(...): (line 110)
        # Processing the call arguments (line 110)
        int_11109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 24), 'int')
        int_11110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 27), 'int')
        int_11111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 30), 'int')
        int_11112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 33), 'int')
        int_11113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 36), 'int')
        # Processing the call keyword arguments (line 110)
        kwargs_11114 = {}
        # Getting the type of 'match' (line 110)
        match_11107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'match', False)
        # Obtaining the member 'group' of a type (line 110)
        group_11108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), match_11107, 'group')
        # Calling group(args, kwargs) (line 110)
        group_call_result_11115 = invoke(stypy.reporting.localization.Localization(__file__, 110, 12), group_11108, *[int_11109, int_11110, int_11111, int_11112, int_11113], **kwargs_11114)
        
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___11116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), group_call_result_11115, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_11117 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), getitem___11116, int_11106)
        
        # Assigning a type to the variable 'tuple_var_assignment_11035' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'tuple_var_assignment_11035', subscript_call_result_11117)
        
        # Assigning a Subscript to a Name (line 109):
        
        # Obtaining the type of the subscript
        int_11118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 8), 'int')
        
        # Call to group(...): (line 110)
        # Processing the call arguments (line 110)
        int_11121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 24), 'int')
        int_11122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 27), 'int')
        int_11123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 30), 'int')
        int_11124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 33), 'int')
        int_11125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 36), 'int')
        # Processing the call keyword arguments (line 110)
        kwargs_11126 = {}
        # Getting the type of 'match' (line 110)
        match_11119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'match', False)
        # Obtaining the member 'group' of a type (line 110)
        group_11120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), match_11119, 'group')
        # Calling group(args, kwargs) (line 110)
        group_call_result_11127 = invoke(stypy.reporting.localization.Localization(__file__, 110, 12), group_11120, *[int_11121, int_11122, int_11123, int_11124, int_11125], **kwargs_11126)
        
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___11128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), group_call_result_11127, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_11129 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), getitem___11128, int_11118)
        
        # Assigning a type to the variable 'tuple_var_assignment_11036' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'tuple_var_assignment_11036', subscript_call_result_11129)
        
        # Assigning a Name to a Name (line 109):
        # Getting the type of 'tuple_var_assignment_11032' (line 109)
        tuple_var_assignment_11032_11130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'tuple_var_assignment_11032')
        # Assigning a type to the variable 'major' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 9), 'major', tuple_var_assignment_11032_11130)
        
        # Assigning a Name to a Name (line 109):
        # Getting the type of 'tuple_var_assignment_11033' (line 109)
        tuple_var_assignment_11033_11131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'tuple_var_assignment_11033')
        # Assigning a type to the variable 'minor' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'minor', tuple_var_assignment_11033_11131)
        
        # Assigning a Name to a Name (line 109):
        # Getting the type of 'tuple_var_assignment_11034' (line 109)
        tuple_var_assignment_11034_11132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'tuple_var_assignment_11034')
        # Assigning a type to the variable 'patch' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 23), 'patch', tuple_var_assignment_11034_11132)
        
        # Assigning a Name to a Name (line 109):
        # Getting the type of 'tuple_var_assignment_11035' (line 109)
        tuple_var_assignment_11035_11133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'tuple_var_assignment_11035')
        # Assigning a type to the variable 'prerelease' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 30), 'prerelease', tuple_var_assignment_11035_11133)
        
        # Assigning a Name to a Name (line 109):
        # Getting the type of 'tuple_var_assignment_11036' (line 109)
        tuple_var_assignment_11036_11134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'tuple_var_assignment_11036')
        # Assigning a type to the variable 'prerelease_num' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 42), 'prerelease_num', tuple_var_assignment_11036_11134)
        
        # Getting the type of 'patch' (line 112)
        patch_11135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'patch')
        # Testing the type of an if condition (line 112)
        if_condition_11136 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 8), patch_11135)
        # Assigning a type to the variable 'if_condition_11136' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'if_condition_11136', if_condition_11136)
        # SSA begins for if statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 113):
        
        # Assigning a Call to a Attribute (line 113):
        
        # Call to tuple(...): (line 113)
        # Processing the call arguments (line 113)
        
        # Call to map(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'string' (line 113)
        string_11139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 37), 'string', False)
        # Obtaining the member 'atoi' of a type (line 113)
        atoi_11140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 37), string_11139, 'atoi')
        
        # Obtaining an instance of the builtin type 'list' (line 113)
        list_11141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 113)
        # Adding element type (line 113)
        # Getting the type of 'major' (line 113)
        major_11142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 51), 'major', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 50), list_11141, major_11142)
        # Adding element type (line 113)
        # Getting the type of 'minor' (line 113)
        minor_11143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 58), 'minor', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 50), list_11141, minor_11143)
        # Adding element type (line 113)
        # Getting the type of 'patch' (line 113)
        patch_11144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 65), 'patch', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 50), list_11141, patch_11144)
        
        # Processing the call keyword arguments (line 113)
        kwargs_11145 = {}
        # Getting the type of 'map' (line 113)
        map_11138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 33), 'map', False)
        # Calling map(args, kwargs) (line 113)
        map_call_result_11146 = invoke(stypy.reporting.localization.Localization(__file__, 113, 33), map_11138, *[atoi_11140, list_11141], **kwargs_11145)
        
        # Processing the call keyword arguments (line 113)
        kwargs_11147 = {}
        # Getting the type of 'tuple' (line 113)
        tuple_11137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 27), 'tuple', False)
        # Calling tuple(args, kwargs) (line 113)
        tuple_call_result_11148 = invoke(stypy.reporting.localization.Localization(__file__, 113, 27), tuple_11137, *[map_call_result_11146], **kwargs_11147)
        
        # Getting the type of 'self' (line 113)
        self_11149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'self')
        # Setting the type of the member 'version' of a type (line 113)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 12), self_11149, 'version', tuple_call_result_11148)
        # SSA branch for the else part of an if statement (line 112)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Attribute (line 115):
        
        # Assigning a Call to a Attribute (line 115):
        
        # Call to tuple(...): (line 115)
        # Processing the call arguments (line 115)
        
        # Call to map(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'string' (line 115)
        string_11152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 37), 'string', False)
        # Obtaining the member 'atoi' of a type (line 115)
        atoi_11153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 37), string_11152, 'atoi')
        
        # Obtaining an instance of the builtin type 'list' (line 115)
        list_11154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 115)
        # Adding element type (line 115)
        # Getting the type of 'major' (line 115)
        major_11155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 51), 'major', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 50), list_11154, major_11155)
        # Adding element type (line 115)
        # Getting the type of 'minor' (line 115)
        minor_11156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 58), 'minor', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 50), list_11154, minor_11156)
        
        # Processing the call keyword arguments (line 115)
        kwargs_11157 = {}
        # Getting the type of 'map' (line 115)
        map_11151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 33), 'map', False)
        # Calling map(args, kwargs) (line 115)
        map_call_result_11158 = invoke(stypy.reporting.localization.Localization(__file__, 115, 33), map_11151, *[atoi_11153, list_11154], **kwargs_11157)
        
        
        # Obtaining an instance of the builtin type 'list' (line 115)
        list_11159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 68), 'list')
        # Adding type elements to the builtin type 'list' instance (line 115)
        # Adding element type (line 115)
        int_11160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 69), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 68), list_11159, int_11160)
        
        # Applying the binary operator '+' (line 115)
        result_add_11161 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 33), '+', map_call_result_11158, list_11159)
        
        # Processing the call keyword arguments (line 115)
        kwargs_11162 = {}
        # Getting the type of 'tuple' (line 115)
        tuple_11150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 27), 'tuple', False)
        # Calling tuple(args, kwargs) (line 115)
        tuple_call_result_11163 = invoke(stypy.reporting.localization.Localization(__file__, 115, 27), tuple_11150, *[result_add_11161], **kwargs_11162)
        
        # Getting the type of 'self' (line 115)
        self_11164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'self')
        # Setting the type of the member 'version' of a type (line 115)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), self_11164, 'version', tuple_call_result_11163)
        # SSA join for if statement (line 112)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'prerelease' (line 117)
        prerelease_11165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 11), 'prerelease')
        # Testing the type of an if condition (line 117)
        if_condition_11166 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 117, 8), prerelease_11165)
        # Assigning a type to the variable 'if_condition_11166' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'if_condition_11166', if_condition_11166)
        # SSA begins for if statement (line 117)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Attribute (line 118):
        
        # Assigning a Tuple to a Attribute (line 118):
        
        # Obtaining an instance of the builtin type 'tuple' (line 118)
        tuple_11167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 118)
        # Adding element type (line 118)
        
        # Obtaining the type of the subscript
        int_11168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 42), 'int')
        # Getting the type of 'prerelease' (line 118)
        prerelease_11169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 31), 'prerelease')
        # Obtaining the member '__getitem__' of a type (line 118)
        getitem___11170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 31), prerelease_11169, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 118)
        subscript_call_result_11171 = invoke(stypy.reporting.localization.Localization(__file__, 118, 31), getitem___11170, int_11168)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 31), tuple_11167, subscript_call_result_11171)
        # Adding element type (line 118)
        
        # Call to atoi(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'prerelease_num' (line 118)
        prerelease_num_11174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 58), 'prerelease_num', False)
        # Processing the call keyword arguments (line 118)
        kwargs_11175 = {}
        # Getting the type of 'string' (line 118)
        string_11172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 46), 'string', False)
        # Obtaining the member 'atoi' of a type (line 118)
        atoi_11173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 46), string_11172, 'atoi')
        # Calling atoi(args, kwargs) (line 118)
        atoi_call_result_11176 = invoke(stypy.reporting.localization.Localization(__file__, 118, 46), atoi_11173, *[prerelease_num_11174], **kwargs_11175)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 31), tuple_11167, atoi_call_result_11176)
        
        # Getting the type of 'self' (line 118)
        self_11177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'self')
        # Setting the type of the member 'prerelease' of a type (line 118)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 12), self_11177, 'prerelease', tuple_11167)
        # SSA branch for the else part of an if statement (line 117)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 120):
        
        # Assigning a Name to a Attribute (line 120):
        # Getting the type of 'None' (line 120)
        None_11178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 30), 'None')
        # Getting the type of 'self' (line 120)
        self_11179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'self')
        # Setting the type of the member 'prerelease' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), self_11179, 'prerelease', None_11178)
        # SSA join for if statement (line 117)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'parse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'parse' in the type store
        # Getting the type of 'stypy_return_type' (line 104)
        stypy_return_type_11180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11180)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'parse'
        return stypy_return_type_11180


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 123, 4, False)
        # Assigning a type to the variable 'self' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StrictVersion.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        StrictVersion.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StrictVersion.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        StrictVersion.stypy__str__.__dict__.__setitem__('stypy_function_name', 'StrictVersion.stypy__str__')
        StrictVersion.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        StrictVersion.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        StrictVersion.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StrictVersion.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        StrictVersion.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        StrictVersion.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StrictVersion.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StrictVersion.stypy__str__', [], None, None, defaults, varargs, kwargs)

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

        
        
        
        # Obtaining the type of the subscript
        int_11181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 24), 'int')
        # Getting the type of 'self' (line 125)
        self_11182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 11), 'self')
        # Obtaining the member 'version' of a type (line 125)
        version_11183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 11), self_11182, 'version')
        # Obtaining the member '__getitem__' of a type (line 125)
        getitem___11184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 11), version_11183, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 125)
        subscript_call_result_11185 = invoke(stypy.reporting.localization.Localization(__file__, 125, 11), getitem___11184, int_11181)
        
        int_11186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 30), 'int')
        # Applying the binary operator '==' (line 125)
        result_eq_11187 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 11), '==', subscript_call_result_11185, int_11186)
        
        # Testing the type of an if condition (line 125)
        if_condition_11188 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 8), result_eq_11187)
        # Assigning a type to the variable 'if_condition_11188' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'if_condition_11188', if_condition_11188)
        # SSA begins for if statement (line 125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 126):
        
        # Assigning a Call to a Name (line 126):
        
        # Call to join(...): (line 126)
        # Processing the call arguments (line 126)
        
        # Call to map(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'str' (line 126)
        str_11192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 38), 'str', False)
        
        # Obtaining the type of the subscript
        int_11193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 56), 'int')
        int_11194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 58), 'int')
        slice_11195 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 126, 43), int_11193, int_11194, None)
        # Getting the type of 'self' (line 126)
        self_11196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 43), 'self', False)
        # Obtaining the member 'version' of a type (line 126)
        version_11197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 43), self_11196, 'version')
        # Obtaining the member '__getitem__' of a type (line 126)
        getitem___11198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 43), version_11197, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 126)
        subscript_call_result_11199 = invoke(stypy.reporting.localization.Localization(__file__, 126, 43), getitem___11198, slice_11195)
        
        # Processing the call keyword arguments (line 126)
        kwargs_11200 = {}
        # Getting the type of 'map' (line 126)
        map_11191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 34), 'map', False)
        # Calling map(args, kwargs) (line 126)
        map_call_result_11201 = invoke(stypy.reporting.localization.Localization(__file__, 126, 34), map_11191, *[str_11192, subscript_call_result_11199], **kwargs_11200)
        
        str_11202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 63), 'str', '.')
        # Processing the call keyword arguments (line 126)
        kwargs_11203 = {}
        # Getting the type of 'string' (line 126)
        string_11189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 22), 'string', False)
        # Obtaining the member 'join' of a type (line 126)
        join_11190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 22), string_11189, 'join')
        # Calling join(args, kwargs) (line 126)
        join_call_result_11204 = invoke(stypy.reporting.localization.Localization(__file__, 126, 22), join_11190, *[map_call_result_11201, str_11202], **kwargs_11203)
        
        # Assigning a type to the variable 'vstring' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'vstring', join_call_result_11204)
        # SSA branch for the else part of an if statement (line 125)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 128):
        
        # Assigning a Call to a Name (line 128):
        
        # Call to join(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Call to map(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'str' (line 128)
        str_11208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 38), 'str', False)
        # Getting the type of 'self' (line 128)
        self_11209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 43), 'self', False)
        # Obtaining the member 'version' of a type (line 128)
        version_11210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 43), self_11209, 'version')
        # Processing the call keyword arguments (line 128)
        kwargs_11211 = {}
        # Getting the type of 'map' (line 128)
        map_11207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 34), 'map', False)
        # Calling map(args, kwargs) (line 128)
        map_call_result_11212 = invoke(stypy.reporting.localization.Localization(__file__, 128, 34), map_11207, *[str_11208, version_11210], **kwargs_11211)
        
        str_11213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 58), 'str', '.')
        # Processing the call keyword arguments (line 128)
        kwargs_11214 = {}
        # Getting the type of 'string' (line 128)
        string_11205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 22), 'string', False)
        # Obtaining the member 'join' of a type (line 128)
        join_11206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 22), string_11205, 'join')
        # Calling join(args, kwargs) (line 128)
        join_call_result_11215 = invoke(stypy.reporting.localization.Localization(__file__, 128, 22), join_11206, *[map_call_result_11212, str_11213], **kwargs_11214)
        
        # Assigning a type to the variable 'vstring' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'vstring', join_call_result_11215)
        # SSA join for if statement (line 125)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 130)
        self_11216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 11), 'self')
        # Obtaining the member 'prerelease' of a type (line 130)
        prerelease_11217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 11), self_11216, 'prerelease')
        # Testing the type of an if condition (line 130)
        if_condition_11218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 8), prerelease_11217)
        # Assigning a type to the variable 'if_condition_11218' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'if_condition_11218', if_condition_11218)
        # SSA begins for if statement (line 130)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 131):
        
        # Assigning a BinOp to a Name (line 131):
        # Getting the type of 'vstring' (line 131)
        vstring_11219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 22), 'vstring')
        
        # Obtaining the type of the subscript
        int_11220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 48), 'int')
        # Getting the type of 'self' (line 131)
        self_11221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 32), 'self')
        # Obtaining the member 'prerelease' of a type (line 131)
        prerelease_11222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 32), self_11221, 'prerelease')
        # Obtaining the member '__getitem__' of a type (line 131)
        getitem___11223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 32), prerelease_11222, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 131)
        subscript_call_result_11224 = invoke(stypy.reporting.localization.Localization(__file__, 131, 32), getitem___11223, int_11220)
        
        # Applying the binary operator '+' (line 131)
        result_add_11225 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 22), '+', vstring_11219, subscript_call_result_11224)
        
        
        # Call to str(...): (line 131)
        # Processing the call arguments (line 131)
        
        # Obtaining the type of the subscript
        int_11227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 73), 'int')
        # Getting the type of 'self' (line 131)
        self_11228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 57), 'self', False)
        # Obtaining the member 'prerelease' of a type (line 131)
        prerelease_11229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 57), self_11228, 'prerelease')
        # Obtaining the member '__getitem__' of a type (line 131)
        getitem___11230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 57), prerelease_11229, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 131)
        subscript_call_result_11231 = invoke(stypy.reporting.localization.Localization(__file__, 131, 57), getitem___11230, int_11227)
        
        # Processing the call keyword arguments (line 131)
        kwargs_11232 = {}
        # Getting the type of 'str' (line 131)
        str_11226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 53), 'str', False)
        # Calling str(args, kwargs) (line 131)
        str_call_result_11233 = invoke(stypy.reporting.localization.Localization(__file__, 131, 53), str_11226, *[subscript_call_result_11231], **kwargs_11232)
        
        # Applying the binary operator '+' (line 131)
        result_add_11234 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 51), '+', result_add_11225, str_call_result_11233)
        
        # Assigning a type to the variable 'vstring' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'vstring', result_add_11234)
        # SSA join for if statement (line 130)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'vstring' (line 133)
        vstring_11235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'vstring')
        # Assigning a type to the variable 'stypy_return_type' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'stypy_return_type', vstring_11235)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 123)
        stypy_return_type_11236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11236)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_11236


    @norecursion
    def stypy__cmp__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__cmp__'
        module_type_store = module_type_store.open_function_context('__cmp__', 136, 4, False)
        # Assigning a type to the variable 'self' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StrictVersion.stypy__cmp__.__dict__.__setitem__('stypy_localization', localization)
        StrictVersion.stypy__cmp__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StrictVersion.stypy__cmp__.__dict__.__setitem__('stypy_type_store', module_type_store)
        StrictVersion.stypy__cmp__.__dict__.__setitem__('stypy_function_name', 'StrictVersion.stypy__cmp__')
        StrictVersion.stypy__cmp__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        StrictVersion.stypy__cmp__.__dict__.__setitem__('stypy_varargs_param_name', None)
        StrictVersion.stypy__cmp__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StrictVersion.stypy__cmp__.__dict__.__setitem__('stypy_call_defaults', defaults)
        StrictVersion.stypy__cmp__.__dict__.__setitem__('stypy_call_varargs', varargs)
        StrictVersion.stypy__cmp__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StrictVersion.stypy__cmp__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StrictVersion.stypy__cmp__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__cmp__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__cmp__(...)' code ##################

        
        
        # Call to isinstance(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'other' (line 137)
        other_11238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 22), 'other', False)
        # Getting the type of 'StringType' (line 137)
        StringType_11239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 29), 'StringType', False)
        # Processing the call keyword arguments (line 137)
        kwargs_11240 = {}
        # Getting the type of 'isinstance' (line 137)
        isinstance_11237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 137)
        isinstance_call_result_11241 = invoke(stypy.reporting.localization.Localization(__file__, 137, 11), isinstance_11237, *[other_11238, StringType_11239], **kwargs_11240)
        
        # Testing the type of an if condition (line 137)
        if_condition_11242 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 8), isinstance_call_result_11241)
        # Assigning a type to the variable 'if_condition_11242' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'if_condition_11242', if_condition_11242)
        # SSA begins for if statement (line 137)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 138):
        
        # Assigning a Call to a Name (line 138):
        
        # Call to StrictVersion(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'other' (line 138)
        other_11244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 34), 'other', False)
        # Processing the call keyword arguments (line 138)
        kwargs_11245 = {}
        # Getting the type of 'StrictVersion' (line 138)
        StrictVersion_11243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 20), 'StrictVersion', False)
        # Calling StrictVersion(args, kwargs) (line 138)
        StrictVersion_call_result_11246 = invoke(stypy.reporting.localization.Localization(__file__, 138, 20), StrictVersion_11243, *[other_11244], **kwargs_11245)
        
        # Assigning a type to the variable 'other' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'other', StrictVersion_call_result_11246)
        # SSA join for if statement (line 137)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to cmp(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'self' (line 140)
        self_11248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 22), 'self', False)
        # Obtaining the member 'version' of a type (line 140)
        version_11249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 22), self_11248, 'version')
        # Getting the type of 'other' (line 140)
        other_11250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 36), 'other', False)
        # Obtaining the member 'version' of a type (line 140)
        version_11251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 36), other_11250, 'version')
        # Processing the call keyword arguments (line 140)
        kwargs_11252 = {}
        # Getting the type of 'cmp' (line 140)
        cmp_11247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 18), 'cmp', False)
        # Calling cmp(args, kwargs) (line 140)
        cmp_call_result_11253 = invoke(stypy.reporting.localization.Localization(__file__, 140, 18), cmp_11247, *[version_11249, version_11251], **kwargs_11252)
        
        # Assigning a type to the variable 'compare' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'compare', cmp_call_result_11253)
        
        
        # Getting the type of 'compare' (line 141)
        compare_11254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'compare')
        int_11255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 23), 'int')
        # Applying the binary operator '==' (line 141)
        result_eq_11256 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 12), '==', compare_11254, int_11255)
        
        # Testing the type of an if condition (line 141)
        if_condition_11257 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 8), result_eq_11256)
        # Assigning a type to the variable 'if_condition_11257' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'if_condition_11257', if_condition_11257)
        # SSA begins for if statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 148)
        self_11258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'self')
        # Obtaining the member 'prerelease' of a type (line 148)
        prerelease_11259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 20), self_11258, 'prerelease')
        # Applying the 'not' unary operator (line 148)
        result_not__11260 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 16), 'not', prerelease_11259)
        
        
        # Getting the type of 'other' (line 148)
        other_11261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 44), 'other')
        # Obtaining the member 'prerelease' of a type (line 148)
        prerelease_11262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 44), other_11261, 'prerelease')
        # Applying the 'not' unary operator (line 148)
        result_not__11263 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 40), 'not', prerelease_11262)
        
        # Applying the binary operator 'and' (line 148)
        result_and_keyword_11264 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 16), 'and', result_not__11260, result_not__11263)
        
        # Testing the type of an if condition (line 148)
        if_condition_11265 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 12), result_and_keyword_11264)
        # Assigning a type to the variable 'if_condition_11265' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'if_condition_11265', if_condition_11265)
        # SSA begins for if statement (line 148)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_11266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 23), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'stypy_return_type', int_11266)
        # SSA branch for the else part of an if statement (line 148)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 150)
        self_11267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 18), 'self')
        # Obtaining the member 'prerelease' of a type (line 150)
        prerelease_11268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 18), self_11267, 'prerelease')
        
        # Getting the type of 'other' (line 150)
        other_11269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 42), 'other')
        # Obtaining the member 'prerelease' of a type (line 150)
        prerelease_11270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 42), other_11269, 'prerelease')
        # Applying the 'not' unary operator (line 150)
        result_not__11271 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 38), 'not', prerelease_11270)
        
        # Applying the binary operator 'and' (line 150)
        result_and_keyword_11272 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 18), 'and', prerelease_11268, result_not__11271)
        
        # Testing the type of an if condition (line 150)
        if_condition_11273 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 17), result_and_keyword_11272)
        # Assigning a type to the variable 'if_condition_11273' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 17), 'if_condition_11273', if_condition_11273)
        # SSA begins for if statement (line 150)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_11274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 23), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'stypy_return_type', int_11274)
        # SSA branch for the else part of an if statement (line 150)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 152)
        self_11275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 22), 'self')
        # Obtaining the member 'prerelease' of a type (line 152)
        prerelease_11276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 22), self_11275, 'prerelease')
        # Applying the 'not' unary operator (line 152)
        result_not__11277 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 18), 'not', prerelease_11276)
        
        # Getting the type of 'other' (line 152)
        other_11278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 42), 'other')
        # Obtaining the member 'prerelease' of a type (line 152)
        prerelease_11279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 42), other_11278, 'prerelease')
        # Applying the binary operator 'and' (line 152)
        result_and_keyword_11280 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 18), 'and', result_not__11277, prerelease_11279)
        
        # Testing the type of an if condition (line 152)
        if_condition_11281 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 17), result_and_keyword_11280)
        # Assigning a type to the variable 'if_condition_11281' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 17), 'if_condition_11281', if_condition_11281)
        # SSA begins for if statement (line 152)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_11282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 23), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'stypy_return_type', int_11282)
        # SSA branch for the else part of an if statement (line 152)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 154)
        self_11283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 18), 'self')
        # Obtaining the member 'prerelease' of a type (line 154)
        prerelease_11284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 18), self_11283, 'prerelease')
        # Getting the type of 'other' (line 154)
        other_11285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 38), 'other')
        # Obtaining the member 'prerelease' of a type (line 154)
        prerelease_11286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 38), other_11285, 'prerelease')
        # Applying the binary operator 'and' (line 154)
        result_and_keyword_11287 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 18), 'and', prerelease_11284, prerelease_11286)
        
        # Testing the type of an if condition (line 154)
        if_condition_11288 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 17), result_and_keyword_11287)
        # Assigning a type to the variable 'if_condition_11288' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 17), 'if_condition_11288', if_condition_11288)
        # SSA begins for if statement (line 154)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to cmp(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'self' (line 155)
        self_11290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 27), 'self', False)
        # Obtaining the member 'prerelease' of a type (line 155)
        prerelease_11291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 27), self_11290, 'prerelease')
        # Getting the type of 'other' (line 155)
        other_11292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 44), 'other', False)
        # Obtaining the member 'prerelease' of a type (line 155)
        prerelease_11293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 44), other_11292, 'prerelease')
        # Processing the call keyword arguments (line 155)
        kwargs_11294 = {}
        # Getting the type of 'cmp' (line 155)
        cmp_11289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 23), 'cmp', False)
        # Calling cmp(args, kwargs) (line 155)
        cmp_call_result_11295 = invoke(stypy.reporting.localization.Localization(__file__, 155, 23), cmp_11289, *[prerelease_11291, prerelease_11293], **kwargs_11294)
        
        # Assigning a type to the variable 'stypy_return_type' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'stypy_return_type', cmp_call_result_11295)
        # SSA join for if statement (line 154)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 152)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 150)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 148)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 141)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'compare' (line 158)
        compare_11296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 19), 'compare')
        # Assigning a type to the variable 'stypy_return_type' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'stypy_return_type', compare_11296)
        # SSA join for if statement (line 141)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__cmp__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__cmp__' in the type store
        # Getting the type of 'stypy_return_type' (line 136)
        stypy_return_type_11297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11297)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__cmp__'
        return stypy_return_type_11297


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 63, 0, False)
        # Assigning a type to the variable 'self' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StrictVersion.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'StrictVersion' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'StrictVersion', StrictVersion)

# Assigning a Call to a Name (line 100):

# Call to compile(...): (line 100)
# Processing the call arguments (line 100)
str_11300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 28), 'str', '^(\\d+) \\. (\\d+) (\\. (\\d+))? ([ab](\\d+))?$')
# Getting the type of 're' (line 101)
re_11301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 28), 're', False)
# Obtaining the member 'VERBOSE' of a type (line 101)
VERBOSE_11302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 28), re_11301, 'VERBOSE')
# Processing the call keyword arguments (line 100)
kwargs_11303 = {}
# Getting the type of 're' (line 100)
re_11298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 17), 're', False)
# Obtaining the member 'compile' of a type (line 100)
compile_11299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 17), re_11298, 'compile')
# Calling compile(args, kwargs) (line 100)
compile_call_result_11304 = invoke(stypy.reporting.localization.Localization(__file__, 100, 17), compile_11299, *[str_11300, VERBOSE_11302], **kwargs_11303)

# Getting the type of 'StrictVersion'
StrictVersion_11305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StrictVersion')
# Setting the type of the member 'version_re' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StrictVersion_11305, 'version_re', compile_call_result_11304)
# Declaration of the 'LooseVersion' class
# Getting the type of 'Version' (line 228)
Version_11306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 20), 'Version')

class LooseVersion(Version_11306, ):
    str_11307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, (-1)), 'str', 'Version numbering for anarchists and software realists.\n    Implements the standard interface for version number classes as\n    described above.  A version number consists of a series of numbers,\n    separated by either periods or strings of letters.  When comparing\n    version numbers, the numeric components will be compared\n    numerically, and the alphabetic components lexically.  The following\n    are all valid version numbers, in no particular order:\n\n        1.5.1\n        1.5.2b2\n        161\n        3.10a\n        8.02\n        3.4j\n        1996.07.12\n        3.2.pl0\n        3.1.1.6\n        2g6\n        11g\n        0.960923\n        2.2beta29\n        1.13++\n        5.5.kw\n        2.0b1pl0\n\n    In fact, there is no such thing as an invalid version number under\n    this scheme; the rules for comparison are simple and predictable,\n    but may not always give the results you want (for some definition\n    of "want").\n    ')
    
    # Assigning a Call to a Name (line 261):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 263)
        None_11308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 32), 'None')
        defaults = [None_11308]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 263, 4, False)
        # Assigning a type to the variable 'self' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LooseVersion.__init__', ['vstring'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['vstring'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Getting the type of 'vstring' (line 264)
        vstring_11309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 11), 'vstring')
        # Testing the type of an if condition (line 264)
        if_condition_11310 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 264, 8), vstring_11309)
        # Assigning a type to the variable 'if_condition_11310' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'if_condition_11310', if_condition_11310)
        # SSA begins for if statement (line 264)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to parse(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'vstring' (line 265)
        vstring_11313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 23), 'vstring', False)
        # Processing the call keyword arguments (line 265)
        kwargs_11314 = {}
        # Getting the type of 'self' (line 265)
        self_11311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'self', False)
        # Obtaining the member 'parse' of a type (line 265)
        parse_11312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 12), self_11311, 'parse')
        # Calling parse(args, kwargs) (line 265)
        parse_call_result_11315 = invoke(stypy.reporting.localization.Localization(__file__, 265, 12), parse_11312, *[vstring_11313], **kwargs_11314)
        
        # SSA join for if statement (line 264)
        module_type_store = module_type_store.join_ssa_context()
        
        
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
        module_type_store = module_type_store.open_function_context('parse', 268, 4, False)
        # Assigning a type to the variable 'self' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LooseVersion.parse.__dict__.__setitem__('stypy_localization', localization)
        LooseVersion.parse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LooseVersion.parse.__dict__.__setitem__('stypy_type_store', module_type_store)
        LooseVersion.parse.__dict__.__setitem__('stypy_function_name', 'LooseVersion.parse')
        LooseVersion.parse.__dict__.__setitem__('stypy_param_names_list', ['vstring'])
        LooseVersion.parse.__dict__.__setitem__('stypy_varargs_param_name', None)
        LooseVersion.parse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LooseVersion.parse.__dict__.__setitem__('stypy_call_defaults', defaults)
        LooseVersion.parse.__dict__.__setitem__('stypy_call_varargs', varargs)
        LooseVersion.parse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LooseVersion.parse.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LooseVersion.parse', ['vstring'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'parse', localization, ['vstring'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'parse(...)' code ##################

        
        # Assigning a Name to a Attribute (line 272):
        
        # Assigning a Name to a Attribute (line 272):
        # Getting the type of 'vstring' (line 272)
        vstring_11316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 23), 'vstring')
        # Getting the type of 'self' (line 272)
        self_11317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'self')
        # Setting the type of the member 'vstring' of a type (line 272)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), self_11317, 'vstring', vstring_11316)
        
        # Assigning a Call to a Name (line 273):
        
        # Assigning a Call to a Name (line 273):
        
        # Call to filter(...): (line 273)
        # Processing the call arguments (line 273)

        @norecursion
        def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_1'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 273, 28, True)
            # Passed parameters checking function
            _stypy_temp_lambda_1.stypy_localization = localization
            _stypy_temp_lambda_1.stypy_type_of_self = None
            _stypy_temp_lambda_1.stypy_type_store = module_type_store
            _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
            _stypy_temp_lambda_1.stypy_param_names_list = ['x']
            _stypy_temp_lambda_1.stypy_varargs_param_name = None
            _stypy_temp_lambda_1.stypy_kwargs_param_name = None
            _stypy_temp_lambda_1.stypy_call_defaults = defaults
            _stypy_temp_lambda_1.stypy_call_varargs = varargs
            _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_1', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Evaluating a boolean operation
            # Getting the type of 'x' (line 273)
            x_11319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 38), 'x', False)
            
            # Getting the type of 'x' (line 273)
            x_11320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 44), 'x', False)
            str_11321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 49), 'str', '.')
            # Applying the binary operator '!=' (line 273)
            result_ne_11322 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 44), '!=', x_11320, str_11321)
            
            # Applying the binary operator 'and' (line 273)
            result_and_keyword_11323 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 38), 'and', x_11319, result_ne_11322)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 273)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 28), 'stypy_return_type', result_and_keyword_11323)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_1' in the type store
            # Getting the type of 'stypy_return_type' (line 273)
            stypy_return_type_11324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 28), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_11324)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_1'
            return stypy_return_type_11324

        # Assigning a type to the variable '_stypy_temp_lambda_1' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 28), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
        # Getting the type of '_stypy_temp_lambda_1' (line 273)
        _stypy_temp_lambda_1_11325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 28), '_stypy_temp_lambda_1')
        
        # Call to split(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'vstring' (line 274)
        vstring_11329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 52), 'vstring', False)
        # Processing the call keyword arguments (line 274)
        kwargs_11330 = {}
        # Getting the type of 'self' (line 274)
        self_11326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 28), 'self', False)
        # Obtaining the member 'component_re' of a type (line 274)
        component_re_11327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 28), self_11326, 'component_re')
        # Obtaining the member 'split' of a type (line 274)
        split_11328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 28), component_re_11327, 'split')
        # Calling split(args, kwargs) (line 274)
        split_call_result_11331 = invoke(stypy.reporting.localization.Localization(__file__, 274, 28), split_11328, *[vstring_11329], **kwargs_11330)
        
        # Processing the call keyword arguments (line 273)
        kwargs_11332 = {}
        # Getting the type of 'filter' (line 273)
        filter_11318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 21), 'filter', False)
        # Calling filter(args, kwargs) (line 273)
        filter_call_result_11333 = invoke(stypy.reporting.localization.Localization(__file__, 273, 21), filter_11318, *[_stypy_temp_lambda_1_11325, split_call_result_11331], **kwargs_11332)
        
        # Assigning a type to the variable 'components' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'components', filter_call_result_11333)
        
        
        # Call to range(...): (line 275)
        # Processing the call arguments (line 275)
        
        # Call to len(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'components' (line 275)
        components_11336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 27), 'components', False)
        # Processing the call keyword arguments (line 275)
        kwargs_11337 = {}
        # Getting the type of 'len' (line 275)
        len_11335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 23), 'len', False)
        # Calling len(args, kwargs) (line 275)
        len_call_result_11338 = invoke(stypy.reporting.localization.Localization(__file__, 275, 23), len_11335, *[components_11336], **kwargs_11337)
        
        # Processing the call keyword arguments (line 275)
        kwargs_11339 = {}
        # Getting the type of 'range' (line 275)
        range_11334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 17), 'range', False)
        # Calling range(args, kwargs) (line 275)
        range_call_result_11340 = invoke(stypy.reporting.localization.Localization(__file__, 275, 17), range_11334, *[len_call_result_11338], **kwargs_11339)
        
        # Testing the type of a for loop iterable (line 275)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 275, 8), range_call_result_11340)
        # Getting the type of the for loop variable (line 275)
        for_loop_var_11341 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 275, 8), range_call_result_11340)
        # Assigning a type to the variable 'i' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'i', for_loop_var_11341)
        # SSA begins for a for statement (line 275)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # SSA begins for try-except statement (line 276)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Subscript (line 277):
        
        # Assigning a Call to a Subscript (line 277):
        
        # Call to int(...): (line 277)
        # Processing the call arguments (line 277)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 277)
        i_11343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 47), 'i', False)
        # Getting the type of 'components' (line 277)
        components_11344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 36), 'components', False)
        # Obtaining the member '__getitem__' of a type (line 277)
        getitem___11345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 36), components_11344, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 277)
        subscript_call_result_11346 = invoke(stypy.reporting.localization.Localization(__file__, 277, 36), getitem___11345, i_11343)
        
        # Processing the call keyword arguments (line 277)
        kwargs_11347 = {}
        # Getting the type of 'int' (line 277)
        int_11342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 32), 'int', False)
        # Calling int(args, kwargs) (line 277)
        int_call_result_11348 = invoke(stypy.reporting.localization.Localization(__file__, 277, 32), int_11342, *[subscript_call_result_11346], **kwargs_11347)
        
        # Getting the type of 'components' (line 277)
        components_11349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 16), 'components')
        # Getting the type of 'i' (line 277)
        i_11350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 27), 'i')
        # Storing an element on a container (line 277)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 16), components_11349, (i_11350, int_call_result_11348))
        # SSA branch for the except part of a try statement (line 276)
        # SSA branch for the except 'ValueError' branch of a try statement (line 276)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 276)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 281):
        
        # Assigning a Name to a Attribute (line 281):
        # Getting the type of 'components' (line 281)
        components_11351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 23), 'components')
        # Getting the type of 'self' (line 281)
        self_11352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'self')
        # Setting the type of the member 'version' of a type (line 281)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 8), self_11352, 'version', components_11351)
        
        # ################# End of 'parse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'parse' in the type store
        # Getting the type of 'stypy_return_type' (line 268)
        stypy_return_type_11353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11353)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'parse'
        return stypy_return_type_11353


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 284, 4, False)
        # Assigning a type to the variable 'self' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LooseVersion.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        LooseVersion.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LooseVersion.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        LooseVersion.stypy__str__.__dict__.__setitem__('stypy_function_name', 'LooseVersion.stypy__str__')
        LooseVersion.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        LooseVersion.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        LooseVersion.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LooseVersion.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        LooseVersion.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        LooseVersion.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LooseVersion.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LooseVersion.stypy__str__', [], None, None, defaults, varargs, kwargs)

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

        # Getting the type of 'self' (line 285)
        self_11354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 15), 'self')
        # Obtaining the member 'vstring' of a type (line 285)
        vstring_11355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 15), self_11354, 'vstring')
        # Assigning a type to the variable 'stypy_return_type' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'stypy_return_type', vstring_11355)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 284)
        stypy_return_type_11356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11356)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_11356


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 288, 4, False)
        # Assigning a type to the variable 'self' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LooseVersion.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        LooseVersion.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LooseVersion.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        LooseVersion.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'LooseVersion.stypy__repr__')
        LooseVersion.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        LooseVersion.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        LooseVersion.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LooseVersion.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        LooseVersion.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        LooseVersion.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LooseVersion.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LooseVersion.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        str_11357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 15), 'str', "LooseVersion ('%s')")
        
        # Call to str(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'self' (line 289)
        self_11359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 43), 'self', False)
        # Processing the call keyword arguments (line 289)
        kwargs_11360 = {}
        # Getting the type of 'str' (line 289)
        str_11358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 39), 'str', False)
        # Calling str(args, kwargs) (line 289)
        str_call_result_11361 = invoke(stypy.reporting.localization.Localization(__file__, 289, 39), str_11358, *[self_11359], **kwargs_11360)
        
        # Applying the binary operator '%' (line 289)
        result_mod_11362 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 15), '%', str_11357, str_call_result_11361)
        
        # Assigning a type to the variable 'stypy_return_type' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'stypy_return_type', result_mod_11362)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 288)
        stypy_return_type_11363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11363)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_11363


    @norecursion
    def stypy__cmp__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__cmp__'
        module_type_store = module_type_store.open_function_context('__cmp__', 292, 4, False)
        # Assigning a type to the variable 'self' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LooseVersion.stypy__cmp__.__dict__.__setitem__('stypy_localization', localization)
        LooseVersion.stypy__cmp__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LooseVersion.stypy__cmp__.__dict__.__setitem__('stypy_type_store', module_type_store)
        LooseVersion.stypy__cmp__.__dict__.__setitem__('stypy_function_name', 'LooseVersion.stypy__cmp__')
        LooseVersion.stypy__cmp__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        LooseVersion.stypy__cmp__.__dict__.__setitem__('stypy_varargs_param_name', None)
        LooseVersion.stypy__cmp__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LooseVersion.stypy__cmp__.__dict__.__setitem__('stypy_call_defaults', defaults)
        LooseVersion.stypy__cmp__.__dict__.__setitem__('stypy_call_varargs', varargs)
        LooseVersion.stypy__cmp__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LooseVersion.stypy__cmp__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LooseVersion.stypy__cmp__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__cmp__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__cmp__(...)' code ##################

        
        
        # Call to isinstance(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'other' (line 293)
        other_11365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 22), 'other', False)
        # Getting the type of 'StringType' (line 293)
        StringType_11366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 29), 'StringType', False)
        # Processing the call keyword arguments (line 293)
        kwargs_11367 = {}
        # Getting the type of 'isinstance' (line 293)
        isinstance_11364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 293)
        isinstance_call_result_11368 = invoke(stypy.reporting.localization.Localization(__file__, 293, 11), isinstance_11364, *[other_11365, StringType_11366], **kwargs_11367)
        
        # Testing the type of an if condition (line 293)
        if_condition_11369 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 293, 8), isinstance_call_result_11368)
        # Assigning a type to the variable 'if_condition_11369' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'if_condition_11369', if_condition_11369)
        # SSA begins for if statement (line 293)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 294):
        
        # Assigning a Call to a Name (line 294):
        
        # Call to LooseVersion(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'other' (line 294)
        other_11371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 33), 'other', False)
        # Processing the call keyword arguments (line 294)
        kwargs_11372 = {}
        # Getting the type of 'LooseVersion' (line 294)
        LooseVersion_11370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 20), 'LooseVersion', False)
        # Calling LooseVersion(args, kwargs) (line 294)
        LooseVersion_call_result_11373 = invoke(stypy.reporting.localization.Localization(__file__, 294, 20), LooseVersion_11370, *[other_11371], **kwargs_11372)
        
        # Assigning a type to the variable 'other' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'other', LooseVersion_call_result_11373)
        # SSA join for if statement (line 293)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to cmp(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 'self' (line 296)
        self_11375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 19), 'self', False)
        # Obtaining the member 'version' of a type (line 296)
        version_11376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 19), self_11375, 'version')
        # Getting the type of 'other' (line 296)
        other_11377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 33), 'other', False)
        # Obtaining the member 'version' of a type (line 296)
        version_11378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 33), other_11377, 'version')
        # Processing the call keyword arguments (line 296)
        kwargs_11379 = {}
        # Getting the type of 'cmp' (line 296)
        cmp_11374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 15), 'cmp', False)
        # Calling cmp(args, kwargs) (line 296)
        cmp_call_result_11380 = invoke(stypy.reporting.localization.Localization(__file__, 296, 15), cmp_11374, *[version_11376, version_11378], **kwargs_11379)
        
        # Assigning a type to the variable 'stypy_return_type' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'stypy_return_type', cmp_call_result_11380)
        
        # ################# End of '__cmp__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__cmp__' in the type store
        # Getting the type of 'stypy_return_type' (line 292)
        stypy_return_type_11381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11381)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__cmp__'
        return stypy_return_type_11381


# Assigning a type to the variable 'LooseVersion' (line 228)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 0), 'LooseVersion', LooseVersion)

# Assigning a Call to a Name (line 261):

# Call to compile(...): (line 261)
# Processing the call arguments (line 261)
str_11384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 30), 'str', '(\\d+ | [a-z]+ | \\.)')
# Getting the type of 're' (line 261)
re_11385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 54), 're', False)
# Obtaining the member 'VERBOSE' of a type (line 261)
VERBOSE_11386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 54), re_11385, 'VERBOSE')
# Processing the call keyword arguments (line 261)
kwargs_11387 = {}
# Getting the type of 're' (line 261)
re_11382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 19), 're', False)
# Obtaining the member 'compile' of a type (line 261)
compile_11383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 19), re_11382, 'compile')
# Calling compile(args, kwargs) (line 261)
compile_call_result_11388 = invoke(stypy.reporting.localization.Localization(__file__, 261, 19), compile_11383, *[str_11384, VERBOSE_11386], **kwargs_11387)

# Getting the type of 'LooseVersion'
LooseVersion_11389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LooseVersion')
# Setting the type of the member 'component_re' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LooseVersion_11389, 'component_re', compile_call_result_11388)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
