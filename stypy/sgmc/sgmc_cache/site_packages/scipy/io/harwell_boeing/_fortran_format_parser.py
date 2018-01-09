
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Preliminary module to handle fortran formats for IO. Does not use this outside
3: scipy.sparse io for now, until the API is deemed reasonable.
4: 
5: The *Format classes handle conversion between fortran and python format, and
6: FortranFormatParser can create *Format instances from raw fortran format
7: strings (e.g. '(3I4)', '(10I3)', etc...)
8: '''
9: from __future__ import division, print_function, absolute_import
10: 
11: import re
12: import warnings
13: 
14: import numpy as np
15: 
16: 
17: __all__ = ["BadFortranFormat", "FortranFormatParser", "IntFormat", "ExpFormat"]
18: 
19: 
20: TOKENS = {
21:     "LPAR": r"\(",
22:     "RPAR": r"\)",
23:     "INT_ID": r"I",
24:     "EXP_ID": r"E",
25:     "INT": r"\d+",
26:     "DOT": r"\.",
27: }
28: 
29: 
30: class BadFortranFormat(SyntaxError):
31:     pass
32: 
33: 
34: def number_digits(n):
35:     return int(np.floor(np.log10(np.abs(n))) + 1)
36: 
37: 
38: class IntFormat(object):
39:     @classmethod
40:     def from_number(cls, n, min=None):
41:         '''Given an integer, returns a "reasonable" IntFormat instance to represent
42:         any number between 0 and n if n > 0, -n and n if n < 0
43: 
44:         Parameters
45:         ----------
46:         n : int
47:             max number one wants to be able to represent
48:         min : int
49:             minimum number of characters to use for the format
50: 
51:         Returns
52:         -------
53:         res : IntFormat
54:             IntFormat instance with reasonable (see Notes) computed width
55: 
56:         Notes
57:         -----
58:         Reasonable should be understood as the minimal string length necessary
59:         without losing precision. For example, IntFormat.from_number(1) will
60:         return an IntFormat instance of width 2, so that any 0 and 1 may be
61:         represented as 1-character strings without loss of information.
62:         '''
63:         width = number_digits(n) + 1
64:         if n < 0:
65:             width += 1
66:         repeat = 80 // width
67:         return cls(width, min, repeat=repeat)
68: 
69:     def __init__(self, width, min=None, repeat=None):
70:         self.width = width
71:         self.repeat = repeat
72:         self.min = min
73: 
74:     def __repr__(self):
75:         r = "IntFormat("
76:         if self.repeat:
77:             r += "%d" % self.repeat
78:         r += "I%d" % self.width
79:         if self.min:
80:             r += ".%d" % self.min
81:         return r + ")"
82: 
83:     @property
84:     def fortran_format(self):
85:         r = "("
86:         if self.repeat:
87:             r += "%d" % self.repeat
88:         r += "I%d" % self.width
89:         if self.min:
90:             r += ".%d" % self.min
91:         return r + ")"
92: 
93:     @property
94:     def python_format(self):
95:         return "%" + str(self.width) + "d"
96: 
97: 
98: class ExpFormat(object):
99:     @classmethod
100:     def from_number(cls, n, min=None):
101:         '''Given a float number, returns a "reasonable" ExpFormat instance to
102:         represent any number between -n and n.
103: 
104:         Parameters
105:         ----------
106:         n : float
107:             max number one wants to be able to represent
108:         min : int
109:             minimum number of characters to use for the format
110: 
111:         Returns
112:         -------
113:         res : ExpFormat
114:             ExpFormat instance with reasonable (see Notes) computed width
115: 
116:         Notes
117:         -----
118:         Reasonable should be understood as the minimal string length necessary
119:         to avoid losing precision.
120:         '''
121:         # len of one number in exp format: sign + 1|0 + "." +
122:         # number of digit for fractional part + 'E' + sign of exponent +
123:         # len of exponent
124:         finfo = np.finfo(n.dtype)
125:         # Number of digits for fractional part
126:         n_prec = finfo.precision + 1
127:         # Number of digits for exponential part
128:         n_exp = number_digits(np.max(np.abs([finfo.maxexp, finfo.minexp])))
129:         width = 1 + 1 + n_prec + 1 + n_exp + 1
130:         if n < 0:
131:             width += 1
132:         repeat = int(np.floor(80 / width))
133:         return cls(width, n_prec, min, repeat=repeat)
134: 
135:     def __init__(self, width, significand, min=None, repeat=None):
136:         '''\
137:         Parameters
138:         ----------
139:         width : int
140:             number of characters taken by the string (includes space).
141:         '''
142:         self.width = width
143:         self.significand = significand
144:         self.repeat = repeat
145:         self.min = min
146: 
147:     def __repr__(self):
148:         r = "ExpFormat("
149:         if self.repeat:
150:             r += "%d" % self.repeat
151:         r += "E%d.%d" % (self.width, self.significand)
152:         if self.min:
153:             r += "E%d" % self.min
154:         return r + ")"
155: 
156:     @property
157:     def fortran_format(self):
158:         r = "("
159:         if self.repeat:
160:             r += "%d" % self.repeat
161:         r += "E%d.%d" % (self.width, self.significand)
162:         if self.min:
163:             r += "E%d" % self.min
164:         return r + ")"
165: 
166:     @property
167:     def python_format(self):
168:         return "%" + str(self.width-1) + "." + str(self.significand) + "E"
169: 
170: 
171: class Token(object):
172:     def __init__(self, type, value, pos):
173:         self.type = type
174:         self.value = value
175:         self.pos = pos
176: 
177:     def __str__(self):
178:         return '''Token('%s', "%s")''' % (self.type, self.value)
179: 
180:     def __repr__(self):
181:         return self.__str__()
182: 
183: 
184: class Tokenizer(object):
185:     def __init__(self):
186:         self.tokens = list(TOKENS.keys())
187:         self.res = [re.compile(TOKENS[i]) for i in self.tokens]
188: 
189:     def input(self, s):
190:         self.data = s
191:         self.curpos = 0
192:         self.len = len(s)
193: 
194:     def next_token(self):
195:         curpos = self.curpos
196:         tokens = self.tokens
197: 
198:         while curpos < self.len:
199:             for i, r in enumerate(self.res):
200:                 m = r.match(self.data, curpos)
201:                 if m is None:
202:                     continue
203:                 else:
204:                     self.curpos = m.end()
205:                     return Token(self.tokens[i], m.group(), self.curpos)
206:             raise SyntaxError("Unknown character at position %d (%s)"
207:                               % (self.curpos, self.data[curpos]))
208: 
209: 
210: # Grammar for fortran format:
211: # format            : LPAR format_string RPAR
212: # format_string     : repeated | simple
213: # repeated          : repeat simple
214: # simple            : int_fmt | exp_fmt
215: # int_fmt           : INT_ID width
216: # exp_fmt           : simple_exp_fmt
217: # simple_exp_fmt    : EXP_ID width DOT significand
218: # extended_exp_fmt  : EXP_ID width DOT significand EXP_ID ndigits
219: # repeat            : INT
220: # width             : INT
221: # significand       : INT
222: # ndigits           : INT
223: 
224: # Naive fortran formatter - parser is hand-made
225: class FortranFormatParser(object):
226:     '''Parser for fortran format strings. The parse method returns a *Format
227:     instance.
228: 
229:     Notes
230:     -----
231:     Only ExpFormat (exponential format for floating values) and IntFormat
232:     (integer format) for now.
233:     '''
234:     def __init__(self):
235:         self.tokenizer = Tokenizer()
236: 
237:     def parse(self, s):
238:         self.tokenizer.input(s)
239: 
240:         tokens = []
241: 
242:         try:
243:             while True:
244:                 t = self.tokenizer.next_token()
245:                 if t is None:
246:                     break
247:                 else:
248:                     tokens.append(t)
249:             return self._parse_format(tokens)
250:         except SyntaxError as e:
251:             raise BadFortranFormat(str(e))
252: 
253:     def _get_min(self, tokens):
254:         next = tokens.pop(0)
255:         if not next.type == "DOT":
256:             raise SyntaxError()
257:         next = tokens.pop(0)
258:         return next.value
259: 
260:     def _expect(self, token, tp):
261:         if not token.type == tp:
262:             raise SyntaxError()
263: 
264:     def _parse_format(self, tokens):
265:         if not tokens[0].type == "LPAR":
266:             raise SyntaxError("Expected left parenthesis at position "
267:                               "%d (got '%s')" % (0, tokens[0].value))
268:         elif not tokens[-1].type == "RPAR":
269:             raise SyntaxError("Expected right parenthesis at position "
270:                               "%d (got '%s')" % (len(tokens), tokens[-1].value))
271: 
272:         tokens = tokens[1:-1]
273:         types = [t.type for t in tokens]
274:         if types[0] == "INT":
275:             repeat = int(tokens.pop(0).value)
276:         else:
277:             repeat = None
278: 
279:         next = tokens.pop(0)
280:         if next.type == "INT_ID":
281:             next = self._next(tokens, "INT")
282:             width = int(next.value)
283:             if tokens:
284:                 min = int(self._get_min(tokens))
285:             else:
286:                 min = None
287:             return IntFormat(width, min, repeat)
288:         elif next.type == "EXP_ID":
289:             next = self._next(tokens, "INT")
290:             width = int(next.value)
291: 
292:             next = self._next(tokens, "DOT")
293: 
294:             next = self._next(tokens, "INT")
295:             significand = int(next.value)
296: 
297:             if tokens:
298:                 next = self._next(tokens, "EXP_ID")
299: 
300:                 next = self._next(tokens, "INT")
301:                 min = int(next.value)
302:             else:
303:                 min = None
304:             return ExpFormat(width, significand, min, repeat)
305:         else:
306:             raise SyntaxError("Invalid formater type %s" % next.value)
307: 
308:     def _next(self, tokens, tp):
309:         if not len(tokens) > 0:
310:             raise SyntaxError()
311:         next = tokens.pop(0)
312:         self._expect(next, tp)
313:         return next
314: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_132219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, (-1)), 'str', "\nPreliminary module to handle fortran formats for IO. Does not use this outside\nscipy.sparse io for now, until the API is deemed reasonable.\n\nThe *Format classes handle conversion between fortran and python format, and\nFortranFormatParser can create *Format instances from raw fortran format\nstrings (e.g. '(3I4)', '(10I3)', etc...)\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import re' statement (line 11)
import re

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import warnings' statement (line 12)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import numpy' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/')
import_132220 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy')

if (type(import_132220) is not StypyTypeError):

    if (import_132220 != 'pyd_module'):
        __import__(import_132220)
        sys_modules_132221 = sys.modules[import_132220]
        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'np', sys_modules_132221.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy', import_132220)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/')


# Assigning a List to a Name (line 17):
__all__ = ['BadFortranFormat', 'FortranFormatParser', 'IntFormat', 'ExpFormat']
module_type_store.set_exportable_members(['BadFortranFormat', 'FortranFormatParser', 'IntFormat', 'ExpFormat'])

# Obtaining an instance of the builtin type 'list' (line 17)
list_132222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
str_132223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 11), 'str', 'BadFortranFormat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_132222, str_132223)
# Adding element type (line 17)
str_132224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 31), 'str', 'FortranFormatParser')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_132222, str_132224)
# Adding element type (line 17)
str_132225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 54), 'str', 'IntFormat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_132222, str_132225)
# Adding element type (line 17)
str_132226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 67), 'str', 'ExpFormat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_132222, str_132226)

# Assigning a type to the variable '__all__' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), '__all__', list_132222)

# Assigning a Dict to a Name (line 20):

# Obtaining an instance of the builtin type 'dict' (line 20)
dict_132227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 20)
# Adding element type (key, value) (line 20)
str_132228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 4), 'str', 'LPAR')
str_132229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 12), 'str', '\\(')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 9), dict_132227, (str_132228, str_132229))
# Adding element type (key, value) (line 20)
str_132230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 4), 'str', 'RPAR')
str_132231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 12), 'str', '\\)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 9), dict_132227, (str_132230, str_132231))
# Adding element type (key, value) (line 20)
str_132232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 4), 'str', 'INT_ID')
str_132233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 14), 'str', 'I')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 9), dict_132227, (str_132232, str_132233))
# Adding element type (key, value) (line 20)
str_132234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 4), 'str', 'EXP_ID')
str_132235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 14), 'str', 'E')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 9), dict_132227, (str_132234, str_132235))
# Adding element type (key, value) (line 20)
str_132236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 4), 'str', 'INT')
str_132237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 11), 'str', '\\d+')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 9), dict_132227, (str_132236, str_132237))
# Adding element type (key, value) (line 20)
str_132238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 4), 'str', 'DOT')
str_132239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 11), 'str', '\\.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 9), dict_132227, (str_132238, str_132239))

# Assigning a type to the variable 'TOKENS' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'TOKENS', dict_132227)
# Declaration of the 'BadFortranFormat' class
# Getting the type of 'SyntaxError' (line 30)
SyntaxError_132240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 23), 'SyntaxError')

class BadFortranFormat(SyntaxError_132240, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 30, 0, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BadFortranFormat.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BadFortranFormat' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'BadFortranFormat', BadFortranFormat)

@norecursion
def number_digits(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'number_digits'
    module_type_store = module_type_store.open_function_context('number_digits', 34, 0, False)
    
    # Passed parameters checking function
    number_digits.stypy_localization = localization
    number_digits.stypy_type_of_self = None
    number_digits.stypy_type_store = module_type_store
    number_digits.stypy_function_name = 'number_digits'
    number_digits.stypy_param_names_list = ['n']
    number_digits.stypy_varargs_param_name = None
    number_digits.stypy_kwargs_param_name = None
    number_digits.stypy_call_defaults = defaults
    number_digits.stypy_call_varargs = varargs
    number_digits.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'number_digits', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'number_digits', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'number_digits(...)' code ##################

    
    # Call to int(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Call to floor(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Call to log10(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Call to abs(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'n' (line 35)
    n_132248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 40), 'n', False)
    # Processing the call keyword arguments (line 35)
    kwargs_132249 = {}
    # Getting the type of 'np' (line 35)
    np_132246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 33), 'np', False)
    # Obtaining the member 'abs' of a type (line 35)
    abs_132247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 33), np_132246, 'abs')
    # Calling abs(args, kwargs) (line 35)
    abs_call_result_132250 = invoke(stypy.reporting.localization.Localization(__file__, 35, 33), abs_132247, *[n_132248], **kwargs_132249)
    
    # Processing the call keyword arguments (line 35)
    kwargs_132251 = {}
    # Getting the type of 'np' (line 35)
    np_132244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 24), 'np', False)
    # Obtaining the member 'log10' of a type (line 35)
    log10_132245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 24), np_132244, 'log10')
    # Calling log10(args, kwargs) (line 35)
    log10_call_result_132252 = invoke(stypy.reporting.localization.Localization(__file__, 35, 24), log10_132245, *[abs_call_result_132250], **kwargs_132251)
    
    # Processing the call keyword arguments (line 35)
    kwargs_132253 = {}
    # Getting the type of 'np' (line 35)
    np_132242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'np', False)
    # Obtaining the member 'floor' of a type (line 35)
    floor_132243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 15), np_132242, 'floor')
    # Calling floor(args, kwargs) (line 35)
    floor_call_result_132254 = invoke(stypy.reporting.localization.Localization(__file__, 35, 15), floor_132243, *[log10_call_result_132252], **kwargs_132253)
    
    int_132255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 47), 'int')
    # Applying the binary operator '+' (line 35)
    result_add_132256 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 15), '+', floor_call_result_132254, int_132255)
    
    # Processing the call keyword arguments (line 35)
    kwargs_132257 = {}
    # Getting the type of 'int' (line 35)
    int_132241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 'int', False)
    # Calling int(args, kwargs) (line 35)
    int_call_result_132258 = invoke(stypy.reporting.localization.Localization(__file__, 35, 11), int_132241, *[result_add_132256], **kwargs_132257)
    
    # Assigning a type to the variable 'stypy_return_type' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type', int_call_result_132258)
    
    # ################# End of 'number_digits(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'number_digits' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_132259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_132259)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'number_digits'
    return stypy_return_type_132259

# Assigning a type to the variable 'number_digits' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'number_digits', number_digits)
# Declaration of the 'IntFormat' class

class IntFormat(object, ):

    @norecursion
    def from_number(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 40)
        None_132260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 32), 'None')
        defaults = [None_132260]
        # Create a new context for function 'from_number'
        module_type_store = module_type_store.open_function_context('from_number', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntFormat.from_number.__dict__.__setitem__('stypy_localization', localization)
        IntFormat.from_number.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntFormat.from_number.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntFormat.from_number.__dict__.__setitem__('stypy_function_name', 'IntFormat.from_number')
        IntFormat.from_number.__dict__.__setitem__('stypy_param_names_list', ['n', 'min'])
        IntFormat.from_number.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntFormat.from_number.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntFormat.from_number.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntFormat.from_number.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntFormat.from_number.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntFormat.from_number.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntFormat.from_number', ['n', 'min'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'from_number', localization, ['n', 'min'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'from_number(...)' code ##################

        str_132261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, (-1)), 'str', 'Given an integer, returns a "reasonable" IntFormat instance to represent\n        any number between 0 and n if n > 0, -n and n if n < 0\n\n        Parameters\n        ----------\n        n : int\n            max number one wants to be able to represent\n        min : int\n            minimum number of characters to use for the format\n\n        Returns\n        -------\n        res : IntFormat\n            IntFormat instance with reasonable (see Notes) computed width\n\n        Notes\n        -----\n        Reasonable should be understood as the minimal string length necessary\n        without losing precision. For example, IntFormat.from_number(1) will\n        return an IntFormat instance of width 2, so that any 0 and 1 may be\n        represented as 1-character strings without loss of information.\n        ')
        
        # Assigning a BinOp to a Name (line 63):
        
        # Call to number_digits(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'n' (line 63)
        n_132263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 30), 'n', False)
        # Processing the call keyword arguments (line 63)
        kwargs_132264 = {}
        # Getting the type of 'number_digits' (line 63)
        number_digits_132262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'number_digits', False)
        # Calling number_digits(args, kwargs) (line 63)
        number_digits_call_result_132265 = invoke(stypy.reporting.localization.Localization(__file__, 63, 16), number_digits_132262, *[n_132263], **kwargs_132264)
        
        int_132266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 35), 'int')
        # Applying the binary operator '+' (line 63)
        result_add_132267 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 16), '+', number_digits_call_result_132265, int_132266)
        
        # Assigning a type to the variable 'width' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'width', result_add_132267)
        
        
        # Getting the type of 'n' (line 64)
        n_132268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'n')
        int_132269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 15), 'int')
        # Applying the binary operator '<' (line 64)
        result_lt_132270 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 11), '<', n_132268, int_132269)
        
        # Testing the type of an if condition (line 64)
        if_condition_132271 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 8), result_lt_132270)
        # Assigning a type to the variable 'if_condition_132271' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'if_condition_132271', if_condition_132271)
        # SSA begins for if statement (line 64)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'width' (line 65)
        width_132272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'width')
        int_132273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 21), 'int')
        # Applying the binary operator '+=' (line 65)
        result_iadd_132274 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 12), '+=', width_132272, int_132273)
        # Assigning a type to the variable 'width' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'width', result_iadd_132274)
        
        # SSA join for if statement (line 64)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 66):
        int_132275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 17), 'int')
        # Getting the type of 'width' (line 66)
        width_132276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 23), 'width')
        # Applying the binary operator '//' (line 66)
        result_floordiv_132277 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 17), '//', int_132275, width_132276)
        
        # Assigning a type to the variable 'repeat' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'repeat', result_floordiv_132277)
        
        # Call to cls(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'width' (line 67)
        width_132279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 19), 'width', False)
        # Getting the type of 'min' (line 67)
        min_132280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 26), 'min', False)
        # Processing the call keyword arguments (line 67)
        # Getting the type of 'repeat' (line 67)
        repeat_132281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 38), 'repeat', False)
        keyword_132282 = repeat_132281
        kwargs_132283 = {'repeat': keyword_132282}
        # Getting the type of 'cls' (line 67)
        cls_132278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'cls', False)
        # Calling cls(args, kwargs) (line 67)
        cls_call_result_132284 = invoke(stypy.reporting.localization.Localization(__file__, 67, 15), cls_132278, *[width_132279, min_132280], **kwargs_132283)
        
        # Assigning a type to the variable 'stypy_return_type' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'stypy_return_type', cls_call_result_132284)
        
        # ################# End of 'from_number(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'from_number' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_132285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132285)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'from_number'
        return stypy_return_type_132285


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 69)
        None_132286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 34), 'None')
        # Getting the type of 'None' (line 69)
        None_132287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 47), 'None')
        defaults = [None_132286, None_132287]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 69, 4, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntFormat.__init__', ['width', 'min', 'repeat'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['width', 'min', 'repeat'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 70):
        # Getting the type of 'width' (line 70)
        width_132288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 21), 'width')
        # Getting the type of 'self' (line 70)
        self_132289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'self')
        # Setting the type of the member 'width' of a type (line 70)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), self_132289, 'width', width_132288)
        
        # Assigning a Name to a Attribute (line 71):
        # Getting the type of 'repeat' (line 71)
        repeat_132290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'repeat')
        # Getting the type of 'self' (line 71)
        self_132291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'self')
        # Setting the type of the member 'repeat' of a type (line 71)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), self_132291, 'repeat', repeat_132290)
        
        # Assigning a Name to a Attribute (line 72):
        # Getting the type of 'min' (line 72)
        min_132292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'min')
        # Getting the type of 'self' (line 72)
        self_132293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'self')
        # Setting the type of the member 'min' of a type (line 72)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), self_132293, 'min', min_132292)
        
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
        module_type_store = module_type_store.open_function_context('__repr__', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntFormat.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        IntFormat.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntFormat.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntFormat.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'IntFormat.stypy__repr__')
        IntFormat.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        IntFormat.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntFormat.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntFormat.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntFormat.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntFormat.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntFormat.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntFormat.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Name (line 75):
        str_132294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 12), 'str', 'IntFormat(')
        # Assigning a type to the variable 'r' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'r', str_132294)
        
        # Getting the type of 'self' (line 76)
        self_132295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'self')
        # Obtaining the member 'repeat' of a type (line 76)
        repeat_132296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 11), self_132295, 'repeat')
        # Testing the type of an if condition (line 76)
        if_condition_132297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 8), repeat_132296)
        # Assigning a type to the variable 'if_condition_132297' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'if_condition_132297', if_condition_132297)
        # SSA begins for if statement (line 76)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'r' (line 77)
        r_132298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'r')
        str_132299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 17), 'str', '%d')
        # Getting the type of 'self' (line 77)
        self_132300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 24), 'self')
        # Obtaining the member 'repeat' of a type (line 77)
        repeat_132301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 24), self_132300, 'repeat')
        # Applying the binary operator '%' (line 77)
        result_mod_132302 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 17), '%', str_132299, repeat_132301)
        
        # Applying the binary operator '+=' (line 77)
        result_iadd_132303 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 12), '+=', r_132298, result_mod_132302)
        # Assigning a type to the variable 'r' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'r', result_iadd_132303)
        
        # SSA join for if statement (line 76)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'r' (line 78)
        r_132304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'r')
        str_132305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 13), 'str', 'I%d')
        # Getting the type of 'self' (line 78)
        self_132306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 21), 'self')
        # Obtaining the member 'width' of a type (line 78)
        width_132307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 21), self_132306, 'width')
        # Applying the binary operator '%' (line 78)
        result_mod_132308 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 13), '%', str_132305, width_132307)
        
        # Applying the binary operator '+=' (line 78)
        result_iadd_132309 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 8), '+=', r_132304, result_mod_132308)
        # Assigning a type to the variable 'r' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'r', result_iadd_132309)
        
        
        # Getting the type of 'self' (line 79)
        self_132310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'self')
        # Obtaining the member 'min' of a type (line 79)
        min_132311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 11), self_132310, 'min')
        # Testing the type of an if condition (line 79)
        if_condition_132312 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 8), min_132311)
        # Assigning a type to the variable 'if_condition_132312' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'if_condition_132312', if_condition_132312)
        # SSA begins for if statement (line 79)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'r' (line 80)
        r_132313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'r')
        str_132314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 17), 'str', '.%d')
        # Getting the type of 'self' (line 80)
        self_132315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 25), 'self')
        # Obtaining the member 'min' of a type (line 80)
        min_132316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 25), self_132315, 'min')
        # Applying the binary operator '%' (line 80)
        result_mod_132317 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 17), '%', str_132314, min_132316)
        
        # Applying the binary operator '+=' (line 80)
        result_iadd_132318 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 12), '+=', r_132313, result_mod_132317)
        # Assigning a type to the variable 'r' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'r', result_iadd_132318)
        
        # SSA join for if statement (line 79)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'r' (line 81)
        r_132319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'r')
        str_132320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 19), 'str', ')')
        # Applying the binary operator '+' (line 81)
        result_add_132321 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 15), '+', r_132319, str_132320)
        
        # Assigning a type to the variable 'stypy_return_type' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'stypy_return_type', result_add_132321)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_132322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132322)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_132322


    @norecursion
    def fortran_format(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fortran_format'
        module_type_store = module_type_store.open_function_context('fortran_format', 83, 4, False)
        # Assigning a type to the variable 'self' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntFormat.fortran_format.__dict__.__setitem__('stypy_localization', localization)
        IntFormat.fortran_format.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntFormat.fortran_format.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntFormat.fortran_format.__dict__.__setitem__('stypy_function_name', 'IntFormat.fortran_format')
        IntFormat.fortran_format.__dict__.__setitem__('stypy_param_names_list', [])
        IntFormat.fortran_format.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntFormat.fortran_format.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntFormat.fortran_format.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntFormat.fortran_format.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntFormat.fortran_format.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntFormat.fortran_format.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntFormat.fortran_format', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fortran_format', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fortran_format(...)' code ##################

        
        # Assigning a Str to a Name (line 85):
        str_132323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 12), 'str', '(')
        # Assigning a type to the variable 'r' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'r', str_132323)
        
        # Getting the type of 'self' (line 86)
        self_132324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 11), 'self')
        # Obtaining the member 'repeat' of a type (line 86)
        repeat_132325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 11), self_132324, 'repeat')
        # Testing the type of an if condition (line 86)
        if_condition_132326 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 8), repeat_132325)
        # Assigning a type to the variable 'if_condition_132326' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'if_condition_132326', if_condition_132326)
        # SSA begins for if statement (line 86)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'r' (line 87)
        r_132327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'r')
        str_132328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 17), 'str', '%d')
        # Getting the type of 'self' (line 87)
        self_132329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 24), 'self')
        # Obtaining the member 'repeat' of a type (line 87)
        repeat_132330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 24), self_132329, 'repeat')
        # Applying the binary operator '%' (line 87)
        result_mod_132331 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 17), '%', str_132328, repeat_132330)
        
        # Applying the binary operator '+=' (line 87)
        result_iadd_132332 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 12), '+=', r_132327, result_mod_132331)
        # Assigning a type to the variable 'r' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'r', result_iadd_132332)
        
        # SSA join for if statement (line 86)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'r' (line 88)
        r_132333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'r')
        str_132334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 13), 'str', 'I%d')
        # Getting the type of 'self' (line 88)
        self_132335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 21), 'self')
        # Obtaining the member 'width' of a type (line 88)
        width_132336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 21), self_132335, 'width')
        # Applying the binary operator '%' (line 88)
        result_mod_132337 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 13), '%', str_132334, width_132336)
        
        # Applying the binary operator '+=' (line 88)
        result_iadd_132338 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 8), '+=', r_132333, result_mod_132337)
        # Assigning a type to the variable 'r' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'r', result_iadd_132338)
        
        
        # Getting the type of 'self' (line 89)
        self_132339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'self')
        # Obtaining the member 'min' of a type (line 89)
        min_132340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 11), self_132339, 'min')
        # Testing the type of an if condition (line 89)
        if_condition_132341 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 8), min_132340)
        # Assigning a type to the variable 'if_condition_132341' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'if_condition_132341', if_condition_132341)
        # SSA begins for if statement (line 89)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'r' (line 90)
        r_132342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'r')
        str_132343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 17), 'str', '.%d')
        # Getting the type of 'self' (line 90)
        self_132344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'self')
        # Obtaining the member 'min' of a type (line 90)
        min_132345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 25), self_132344, 'min')
        # Applying the binary operator '%' (line 90)
        result_mod_132346 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 17), '%', str_132343, min_132345)
        
        # Applying the binary operator '+=' (line 90)
        result_iadd_132347 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 12), '+=', r_132342, result_mod_132346)
        # Assigning a type to the variable 'r' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'r', result_iadd_132347)
        
        # SSA join for if statement (line 89)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'r' (line 91)
        r_132348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 15), 'r')
        str_132349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 19), 'str', ')')
        # Applying the binary operator '+' (line 91)
        result_add_132350 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 15), '+', r_132348, str_132349)
        
        # Assigning a type to the variable 'stypy_return_type' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'stypy_return_type', result_add_132350)
        
        # ################# End of 'fortran_format(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fortran_format' in the type store
        # Getting the type of 'stypy_return_type' (line 83)
        stypy_return_type_132351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132351)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fortran_format'
        return stypy_return_type_132351


    @norecursion
    def python_format(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'python_format'
        module_type_store = module_type_store.open_function_context('python_format', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IntFormat.python_format.__dict__.__setitem__('stypy_localization', localization)
        IntFormat.python_format.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IntFormat.python_format.__dict__.__setitem__('stypy_type_store', module_type_store)
        IntFormat.python_format.__dict__.__setitem__('stypy_function_name', 'IntFormat.python_format')
        IntFormat.python_format.__dict__.__setitem__('stypy_param_names_list', [])
        IntFormat.python_format.__dict__.__setitem__('stypy_varargs_param_name', None)
        IntFormat.python_format.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IntFormat.python_format.__dict__.__setitem__('stypy_call_defaults', defaults)
        IntFormat.python_format.__dict__.__setitem__('stypy_call_varargs', varargs)
        IntFormat.python_format.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IntFormat.python_format.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IntFormat.python_format', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'python_format', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'python_format(...)' code ##################

        str_132352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 15), 'str', '%')
        
        # Call to str(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'self' (line 95)
        self_132354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 25), 'self', False)
        # Obtaining the member 'width' of a type (line 95)
        width_132355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 25), self_132354, 'width')
        # Processing the call keyword arguments (line 95)
        kwargs_132356 = {}
        # Getting the type of 'str' (line 95)
        str_132353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 21), 'str', False)
        # Calling str(args, kwargs) (line 95)
        str_call_result_132357 = invoke(stypy.reporting.localization.Localization(__file__, 95, 21), str_132353, *[width_132355], **kwargs_132356)
        
        # Applying the binary operator '+' (line 95)
        result_add_132358 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 15), '+', str_132352, str_call_result_132357)
        
        str_132359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 39), 'str', 'd')
        # Applying the binary operator '+' (line 95)
        result_add_132360 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 37), '+', result_add_132358, str_132359)
        
        # Assigning a type to the variable 'stypy_return_type' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'stypy_return_type', result_add_132360)
        
        # ################# End of 'python_format(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'python_format' in the type store
        # Getting the type of 'stypy_return_type' (line 93)
        stypy_return_type_132361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132361)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'python_format'
        return stypy_return_type_132361


# Assigning a type to the variable 'IntFormat' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'IntFormat', IntFormat)
# Declaration of the 'ExpFormat' class

class ExpFormat(object, ):

    @norecursion
    def from_number(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 100)
        None_132362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 32), 'None')
        defaults = [None_132362]
        # Create a new context for function 'from_number'
        module_type_store = module_type_store.open_function_context('from_number', 99, 4, False)
        # Assigning a type to the variable 'self' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ExpFormat.from_number.__dict__.__setitem__('stypy_localization', localization)
        ExpFormat.from_number.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ExpFormat.from_number.__dict__.__setitem__('stypy_type_store', module_type_store)
        ExpFormat.from_number.__dict__.__setitem__('stypy_function_name', 'ExpFormat.from_number')
        ExpFormat.from_number.__dict__.__setitem__('stypy_param_names_list', ['n', 'min'])
        ExpFormat.from_number.__dict__.__setitem__('stypy_varargs_param_name', None)
        ExpFormat.from_number.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ExpFormat.from_number.__dict__.__setitem__('stypy_call_defaults', defaults)
        ExpFormat.from_number.__dict__.__setitem__('stypy_call_varargs', varargs)
        ExpFormat.from_number.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ExpFormat.from_number.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ExpFormat.from_number', ['n', 'min'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'from_number', localization, ['n', 'min'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'from_number(...)' code ##################

        str_132363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, (-1)), 'str', 'Given a float number, returns a "reasonable" ExpFormat instance to\n        represent any number between -n and n.\n\n        Parameters\n        ----------\n        n : float\n            max number one wants to be able to represent\n        min : int\n            minimum number of characters to use for the format\n\n        Returns\n        -------\n        res : ExpFormat\n            ExpFormat instance with reasonable (see Notes) computed width\n\n        Notes\n        -----\n        Reasonable should be understood as the minimal string length necessary\n        to avoid losing precision.\n        ')
        
        # Assigning a Call to a Name (line 124):
        
        # Call to finfo(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'n' (line 124)
        n_132366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 25), 'n', False)
        # Obtaining the member 'dtype' of a type (line 124)
        dtype_132367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 25), n_132366, 'dtype')
        # Processing the call keyword arguments (line 124)
        kwargs_132368 = {}
        # Getting the type of 'np' (line 124)
        np_132364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'np', False)
        # Obtaining the member 'finfo' of a type (line 124)
        finfo_132365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 16), np_132364, 'finfo')
        # Calling finfo(args, kwargs) (line 124)
        finfo_call_result_132369 = invoke(stypy.reporting.localization.Localization(__file__, 124, 16), finfo_132365, *[dtype_132367], **kwargs_132368)
        
        # Assigning a type to the variable 'finfo' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'finfo', finfo_call_result_132369)
        
        # Assigning a BinOp to a Name (line 126):
        # Getting the type of 'finfo' (line 126)
        finfo_132370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 17), 'finfo')
        # Obtaining the member 'precision' of a type (line 126)
        precision_132371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 17), finfo_132370, 'precision')
        int_132372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 35), 'int')
        # Applying the binary operator '+' (line 126)
        result_add_132373 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 17), '+', precision_132371, int_132372)
        
        # Assigning a type to the variable 'n_prec' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'n_prec', result_add_132373)
        
        # Assigning a Call to a Name (line 128):
        
        # Call to number_digits(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Call to max(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Call to abs(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Obtaining an instance of the builtin type 'list' (line 128)
        list_132379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 128)
        # Adding element type (line 128)
        # Getting the type of 'finfo' (line 128)
        finfo_132380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 45), 'finfo', False)
        # Obtaining the member 'maxexp' of a type (line 128)
        maxexp_132381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 45), finfo_132380, 'maxexp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 44), list_132379, maxexp_132381)
        # Adding element type (line 128)
        # Getting the type of 'finfo' (line 128)
        finfo_132382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 59), 'finfo', False)
        # Obtaining the member 'minexp' of a type (line 128)
        minexp_132383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 59), finfo_132382, 'minexp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 44), list_132379, minexp_132383)
        
        # Processing the call keyword arguments (line 128)
        kwargs_132384 = {}
        # Getting the type of 'np' (line 128)
        np_132377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 37), 'np', False)
        # Obtaining the member 'abs' of a type (line 128)
        abs_132378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 37), np_132377, 'abs')
        # Calling abs(args, kwargs) (line 128)
        abs_call_result_132385 = invoke(stypy.reporting.localization.Localization(__file__, 128, 37), abs_132378, *[list_132379], **kwargs_132384)
        
        # Processing the call keyword arguments (line 128)
        kwargs_132386 = {}
        # Getting the type of 'np' (line 128)
        np_132375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 30), 'np', False)
        # Obtaining the member 'max' of a type (line 128)
        max_132376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 30), np_132375, 'max')
        # Calling max(args, kwargs) (line 128)
        max_call_result_132387 = invoke(stypy.reporting.localization.Localization(__file__, 128, 30), max_132376, *[abs_call_result_132385], **kwargs_132386)
        
        # Processing the call keyword arguments (line 128)
        kwargs_132388 = {}
        # Getting the type of 'number_digits' (line 128)
        number_digits_132374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'number_digits', False)
        # Calling number_digits(args, kwargs) (line 128)
        number_digits_call_result_132389 = invoke(stypy.reporting.localization.Localization(__file__, 128, 16), number_digits_132374, *[max_call_result_132387], **kwargs_132388)
        
        # Assigning a type to the variable 'n_exp' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'n_exp', number_digits_call_result_132389)
        
        # Assigning a BinOp to a Name (line 129):
        int_132390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 16), 'int')
        int_132391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 20), 'int')
        # Applying the binary operator '+' (line 129)
        result_add_132392 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 16), '+', int_132390, int_132391)
        
        # Getting the type of 'n_prec' (line 129)
        n_prec_132393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 24), 'n_prec')
        # Applying the binary operator '+' (line 129)
        result_add_132394 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 22), '+', result_add_132392, n_prec_132393)
        
        int_132395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 33), 'int')
        # Applying the binary operator '+' (line 129)
        result_add_132396 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 31), '+', result_add_132394, int_132395)
        
        # Getting the type of 'n_exp' (line 129)
        n_exp_132397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 37), 'n_exp')
        # Applying the binary operator '+' (line 129)
        result_add_132398 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 35), '+', result_add_132396, n_exp_132397)
        
        int_132399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 45), 'int')
        # Applying the binary operator '+' (line 129)
        result_add_132400 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 43), '+', result_add_132398, int_132399)
        
        # Assigning a type to the variable 'width' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'width', result_add_132400)
        
        
        # Getting the type of 'n' (line 130)
        n_132401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 11), 'n')
        int_132402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 15), 'int')
        # Applying the binary operator '<' (line 130)
        result_lt_132403 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 11), '<', n_132401, int_132402)
        
        # Testing the type of an if condition (line 130)
        if_condition_132404 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 8), result_lt_132403)
        # Assigning a type to the variable 'if_condition_132404' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'if_condition_132404', if_condition_132404)
        # SSA begins for if statement (line 130)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'width' (line 131)
        width_132405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'width')
        int_132406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 21), 'int')
        # Applying the binary operator '+=' (line 131)
        result_iadd_132407 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 12), '+=', width_132405, int_132406)
        # Assigning a type to the variable 'width' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'width', result_iadd_132407)
        
        # SSA join for if statement (line 130)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 132):
        
        # Call to int(...): (line 132)
        # Processing the call arguments (line 132)
        
        # Call to floor(...): (line 132)
        # Processing the call arguments (line 132)
        int_132411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 30), 'int')
        # Getting the type of 'width' (line 132)
        width_132412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 35), 'width', False)
        # Applying the binary operator 'div' (line 132)
        result_div_132413 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 30), 'div', int_132411, width_132412)
        
        # Processing the call keyword arguments (line 132)
        kwargs_132414 = {}
        # Getting the type of 'np' (line 132)
        np_132409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 21), 'np', False)
        # Obtaining the member 'floor' of a type (line 132)
        floor_132410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 21), np_132409, 'floor')
        # Calling floor(args, kwargs) (line 132)
        floor_call_result_132415 = invoke(stypy.reporting.localization.Localization(__file__, 132, 21), floor_132410, *[result_div_132413], **kwargs_132414)
        
        # Processing the call keyword arguments (line 132)
        kwargs_132416 = {}
        # Getting the type of 'int' (line 132)
        int_132408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 17), 'int', False)
        # Calling int(args, kwargs) (line 132)
        int_call_result_132417 = invoke(stypy.reporting.localization.Localization(__file__, 132, 17), int_132408, *[floor_call_result_132415], **kwargs_132416)
        
        # Assigning a type to the variable 'repeat' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'repeat', int_call_result_132417)
        
        # Call to cls(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'width' (line 133)
        width_132419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'width', False)
        # Getting the type of 'n_prec' (line 133)
        n_prec_132420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 26), 'n_prec', False)
        # Getting the type of 'min' (line 133)
        min_132421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 34), 'min', False)
        # Processing the call keyword arguments (line 133)
        # Getting the type of 'repeat' (line 133)
        repeat_132422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 46), 'repeat', False)
        keyword_132423 = repeat_132422
        kwargs_132424 = {'repeat': keyword_132423}
        # Getting the type of 'cls' (line 133)
        cls_132418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'cls', False)
        # Calling cls(args, kwargs) (line 133)
        cls_call_result_132425 = invoke(stypy.reporting.localization.Localization(__file__, 133, 15), cls_132418, *[width_132419, n_prec_132420, min_132421], **kwargs_132424)
        
        # Assigning a type to the variable 'stypy_return_type' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'stypy_return_type', cls_call_result_132425)
        
        # ################# End of 'from_number(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'from_number' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_132426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132426)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'from_number'
        return stypy_return_type_132426


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 135)
        None_132427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 47), 'None')
        # Getting the type of 'None' (line 135)
        None_132428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 60), 'None')
        defaults = [None_132427, None_132428]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 135, 4, False)
        # Assigning a type to the variable 'self' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ExpFormat.__init__', ['width', 'significand', 'min', 'repeat'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['width', 'significand', 'min', 'repeat'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_132429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, (-1)), 'str', '        Parameters\n        ----------\n        width : int\n            number of characters taken by the string (includes space).\n        ')
        
        # Assigning a Name to a Attribute (line 142):
        # Getting the type of 'width' (line 142)
        width_132430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 21), 'width')
        # Getting the type of 'self' (line 142)
        self_132431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'self')
        # Setting the type of the member 'width' of a type (line 142)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), self_132431, 'width', width_132430)
        
        # Assigning a Name to a Attribute (line 143):
        # Getting the type of 'significand' (line 143)
        significand_132432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 27), 'significand')
        # Getting the type of 'self' (line 143)
        self_132433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'self')
        # Setting the type of the member 'significand' of a type (line 143)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), self_132433, 'significand', significand_132432)
        
        # Assigning a Name to a Attribute (line 144):
        # Getting the type of 'repeat' (line 144)
        repeat_132434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 22), 'repeat')
        # Getting the type of 'self' (line 144)
        self_132435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'self')
        # Setting the type of the member 'repeat' of a type (line 144)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), self_132435, 'repeat', repeat_132434)
        
        # Assigning a Name to a Attribute (line 145):
        # Getting the type of 'min' (line 145)
        min_132436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 19), 'min')
        # Getting the type of 'self' (line 145)
        self_132437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'self')
        # Setting the type of the member 'min' of a type (line 145)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), self_132437, 'min', min_132436)
        
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
        module_type_store = module_type_store.open_function_context('__repr__', 147, 4, False)
        # Assigning a type to the variable 'self' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ExpFormat.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        ExpFormat.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ExpFormat.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ExpFormat.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'ExpFormat.stypy__repr__')
        ExpFormat.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        ExpFormat.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ExpFormat.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ExpFormat.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ExpFormat.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ExpFormat.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ExpFormat.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ExpFormat.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Name (line 148):
        str_132438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 12), 'str', 'ExpFormat(')
        # Assigning a type to the variable 'r' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'r', str_132438)
        
        # Getting the type of 'self' (line 149)
        self_132439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 11), 'self')
        # Obtaining the member 'repeat' of a type (line 149)
        repeat_132440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 11), self_132439, 'repeat')
        # Testing the type of an if condition (line 149)
        if_condition_132441 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 8), repeat_132440)
        # Assigning a type to the variable 'if_condition_132441' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'if_condition_132441', if_condition_132441)
        # SSA begins for if statement (line 149)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'r' (line 150)
        r_132442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'r')
        str_132443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 17), 'str', '%d')
        # Getting the type of 'self' (line 150)
        self_132444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 24), 'self')
        # Obtaining the member 'repeat' of a type (line 150)
        repeat_132445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 24), self_132444, 'repeat')
        # Applying the binary operator '%' (line 150)
        result_mod_132446 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 17), '%', str_132443, repeat_132445)
        
        # Applying the binary operator '+=' (line 150)
        result_iadd_132447 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 12), '+=', r_132442, result_mod_132446)
        # Assigning a type to the variable 'r' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'r', result_iadd_132447)
        
        # SSA join for if statement (line 149)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'r' (line 151)
        r_132448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'r')
        str_132449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 13), 'str', 'E%d.%d')
        
        # Obtaining an instance of the builtin type 'tuple' (line 151)
        tuple_132450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 151)
        # Adding element type (line 151)
        # Getting the type of 'self' (line 151)
        self_132451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 25), 'self')
        # Obtaining the member 'width' of a type (line 151)
        width_132452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 25), self_132451, 'width')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 25), tuple_132450, width_132452)
        # Adding element type (line 151)
        # Getting the type of 'self' (line 151)
        self_132453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 37), 'self')
        # Obtaining the member 'significand' of a type (line 151)
        significand_132454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 37), self_132453, 'significand')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 25), tuple_132450, significand_132454)
        
        # Applying the binary operator '%' (line 151)
        result_mod_132455 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 13), '%', str_132449, tuple_132450)
        
        # Applying the binary operator '+=' (line 151)
        result_iadd_132456 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 8), '+=', r_132448, result_mod_132455)
        # Assigning a type to the variable 'r' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'r', result_iadd_132456)
        
        
        # Getting the type of 'self' (line 152)
        self_132457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 11), 'self')
        # Obtaining the member 'min' of a type (line 152)
        min_132458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 11), self_132457, 'min')
        # Testing the type of an if condition (line 152)
        if_condition_132459 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 8), min_132458)
        # Assigning a type to the variable 'if_condition_132459' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'if_condition_132459', if_condition_132459)
        # SSA begins for if statement (line 152)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'r' (line 153)
        r_132460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'r')
        str_132461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 17), 'str', 'E%d')
        # Getting the type of 'self' (line 153)
        self_132462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 25), 'self')
        # Obtaining the member 'min' of a type (line 153)
        min_132463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 25), self_132462, 'min')
        # Applying the binary operator '%' (line 153)
        result_mod_132464 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 17), '%', str_132461, min_132463)
        
        # Applying the binary operator '+=' (line 153)
        result_iadd_132465 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 12), '+=', r_132460, result_mod_132464)
        # Assigning a type to the variable 'r' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'r', result_iadd_132465)
        
        # SSA join for if statement (line 152)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'r' (line 154)
        r_132466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'r')
        str_132467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 19), 'str', ')')
        # Applying the binary operator '+' (line 154)
        result_add_132468 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 15), '+', r_132466, str_132467)
        
        # Assigning a type to the variable 'stypy_return_type' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'stypy_return_type', result_add_132468)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 147)
        stypy_return_type_132469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132469)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_132469


    @norecursion
    def fortran_format(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fortran_format'
        module_type_store = module_type_store.open_function_context('fortran_format', 156, 4, False)
        # Assigning a type to the variable 'self' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ExpFormat.fortran_format.__dict__.__setitem__('stypy_localization', localization)
        ExpFormat.fortran_format.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ExpFormat.fortran_format.__dict__.__setitem__('stypy_type_store', module_type_store)
        ExpFormat.fortran_format.__dict__.__setitem__('stypy_function_name', 'ExpFormat.fortran_format')
        ExpFormat.fortran_format.__dict__.__setitem__('stypy_param_names_list', [])
        ExpFormat.fortran_format.__dict__.__setitem__('stypy_varargs_param_name', None)
        ExpFormat.fortran_format.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ExpFormat.fortran_format.__dict__.__setitem__('stypy_call_defaults', defaults)
        ExpFormat.fortran_format.__dict__.__setitem__('stypy_call_varargs', varargs)
        ExpFormat.fortran_format.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ExpFormat.fortran_format.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ExpFormat.fortran_format', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fortran_format', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fortran_format(...)' code ##################

        
        # Assigning a Str to a Name (line 158):
        str_132470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 12), 'str', '(')
        # Assigning a type to the variable 'r' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'r', str_132470)
        
        # Getting the type of 'self' (line 159)
        self_132471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 11), 'self')
        # Obtaining the member 'repeat' of a type (line 159)
        repeat_132472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 11), self_132471, 'repeat')
        # Testing the type of an if condition (line 159)
        if_condition_132473 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 8), repeat_132472)
        # Assigning a type to the variable 'if_condition_132473' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'if_condition_132473', if_condition_132473)
        # SSA begins for if statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'r' (line 160)
        r_132474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'r')
        str_132475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 17), 'str', '%d')
        # Getting the type of 'self' (line 160)
        self_132476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'self')
        # Obtaining the member 'repeat' of a type (line 160)
        repeat_132477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 24), self_132476, 'repeat')
        # Applying the binary operator '%' (line 160)
        result_mod_132478 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 17), '%', str_132475, repeat_132477)
        
        # Applying the binary operator '+=' (line 160)
        result_iadd_132479 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 12), '+=', r_132474, result_mod_132478)
        # Assigning a type to the variable 'r' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'r', result_iadd_132479)
        
        # SSA join for if statement (line 159)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'r' (line 161)
        r_132480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'r')
        str_132481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 13), 'str', 'E%d.%d')
        
        # Obtaining an instance of the builtin type 'tuple' (line 161)
        tuple_132482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 161)
        # Adding element type (line 161)
        # Getting the type of 'self' (line 161)
        self_132483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 25), 'self')
        # Obtaining the member 'width' of a type (line 161)
        width_132484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 25), self_132483, 'width')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 25), tuple_132482, width_132484)
        # Adding element type (line 161)
        # Getting the type of 'self' (line 161)
        self_132485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 37), 'self')
        # Obtaining the member 'significand' of a type (line 161)
        significand_132486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 37), self_132485, 'significand')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 25), tuple_132482, significand_132486)
        
        # Applying the binary operator '%' (line 161)
        result_mod_132487 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 13), '%', str_132481, tuple_132482)
        
        # Applying the binary operator '+=' (line 161)
        result_iadd_132488 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 8), '+=', r_132480, result_mod_132487)
        # Assigning a type to the variable 'r' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'r', result_iadd_132488)
        
        
        # Getting the type of 'self' (line 162)
        self_132489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 11), 'self')
        # Obtaining the member 'min' of a type (line 162)
        min_132490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 11), self_132489, 'min')
        # Testing the type of an if condition (line 162)
        if_condition_132491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 8), min_132490)
        # Assigning a type to the variable 'if_condition_132491' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'if_condition_132491', if_condition_132491)
        # SSA begins for if statement (line 162)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'r' (line 163)
        r_132492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'r')
        str_132493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 17), 'str', 'E%d')
        # Getting the type of 'self' (line 163)
        self_132494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 25), 'self')
        # Obtaining the member 'min' of a type (line 163)
        min_132495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 25), self_132494, 'min')
        # Applying the binary operator '%' (line 163)
        result_mod_132496 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 17), '%', str_132493, min_132495)
        
        # Applying the binary operator '+=' (line 163)
        result_iadd_132497 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 12), '+=', r_132492, result_mod_132496)
        # Assigning a type to the variable 'r' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'r', result_iadd_132497)
        
        # SSA join for if statement (line 162)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'r' (line 164)
        r_132498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 15), 'r')
        str_132499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 19), 'str', ')')
        # Applying the binary operator '+' (line 164)
        result_add_132500 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 15), '+', r_132498, str_132499)
        
        # Assigning a type to the variable 'stypy_return_type' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'stypy_return_type', result_add_132500)
        
        # ################# End of 'fortran_format(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fortran_format' in the type store
        # Getting the type of 'stypy_return_type' (line 156)
        stypy_return_type_132501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132501)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fortran_format'
        return stypy_return_type_132501


    @norecursion
    def python_format(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'python_format'
        module_type_store = module_type_store.open_function_context('python_format', 166, 4, False)
        # Assigning a type to the variable 'self' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ExpFormat.python_format.__dict__.__setitem__('stypy_localization', localization)
        ExpFormat.python_format.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ExpFormat.python_format.__dict__.__setitem__('stypy_type_store', module_type_store)
        ExpFormat.python_format.__dict__.__setitem__('stypy_function_name', 'ExpFormat.python_format')
        ExpFormat.python_format.__dict__.__setitem__('stypy_param_names_list', [])
        ExpFormat.python_format.__dict__.__setitem__('stypy_varargs_param_name', None)
        ExpFormat.python_format.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ExpFormat.python_format.__dict__.__setitem__('stypy_call_defaults', defaults)
        ExpFormat.python_format.__dict__.__setitem__('stypy_call_varargs', varargs)
        ExpFormat.python_format.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ExpFormat.python_format.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ExpFormat.python_format', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'python_format', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'python_format(...)' code ##################

        str_132502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 15), 'str', '%')
        
        # Call to str(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'self' (line 168)
        self_132504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 25), 'self', False)
        # Obtaining the member 'width' of a type (line 168)
        width_132505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 25), self_132504, 'width')
        int_132506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 36), 'int')
        # Applying the binary operator '-' (line 168)
        result_sub_132507 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 25), '-', width_132505, int_132506)
        
        # Processing the call keyword arguments (line 168)
        kwargs_132508 = {}
        # Getting the type of 'str' (line 168)
        str_132503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 21), 'str', False)
        # Calling str(args, kwargs) (line 168)
        str_call_result_132509 = invoke(stypy.reporting.localization.Localization(__file__, 168, 21), str_132503, *[result_sub_132507], **kwargs_132508)
        
        # Applying the binary operator '+' (line 168)
        result_add_132510 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 15), '+', str_132502, str_call_result_132509)
        
        str_132511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 41), 'str', '.')
        # Applying the binary operator '+' (line 168)
        result_add_132512 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 39), '+', result_add_132510, str_132511)
        
        
        # Call to str(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'self' (line 168)
        self_132514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 51), 'self', False)
        # Obtaining the member 'significand' of a type (line 168)
        significand_132515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 51), self_132514, 'significand')
        # Processing the call keyword arguments (line 168)
        kwargs_132516 = {}
        # Getting the type of 'str' (line 168)
        str_132513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 47), 'str', False)
        # Calling str(args, kwargs) (line 168)
        str_call_result_132517 = invoke(stypy.reporting.localization.Localization(__file__, 168, 47), str_132513, *[significand_132515], **kwargs_132516)
        
        # Applying the binary operator '+' (line 168)
        result_add_132518 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 45), '+', result_add_132512, str_call_result_132517)
        
        str_132519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 71), 'str', 'E')
        # Applying the binary operator '+' (line 168)
        result_add_132520 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 69), '+', result_add_132518, str_132519)
        
        # Assigning a type to the variable 'stypy_return_type' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'stypy_return_type', result_add_132520)
        
        # ################# End of 'python_format(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'python_format' in the type store
        # Getting the type of 'stypy_return_type' (line 166)
        stypy_return_type_132521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132521)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'python_format'
        return stypy_return_type_132521


# Assigning a type to the variable 'ExpFormat' (line 98)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'ExpFormat', ExpFormat)
# Declaration of the 'Token' class

class Token(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 172, 4, False)
        # Assigning a type to the variable 'self' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Token.__init__', ['type', 'value', 'pos'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['type', 'value', 'pos'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 173):
        # Getting the type of 'type' (line 173)
        type_132522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 20), 'type')
        # Getting the type of 'self' (line 173)
        self_132523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'self')
        # Setting the type of the member 'type' of a type (line 173)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), self_132523, 'type', type_132522)
        
        # Assigning a Name to a Attribute (line 174):
        # Getting the type of 'value' (line 174)
        value_132524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 21), 'value')
        # Getting the type of 'self' (line 174)
        self_132525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'self')
        # Setting the type of the member 'value' of a type (line 174)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), self_132525, 'value', value_132524)
        
        # Assigning a Name to a Attribute (line 175):
        # Getting the type of 'pos' (line 175)
        pos_132526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 19), 'pos')
        # Getting the type of 'self' (line 175)
        self_132527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'self')
        # Setting the type of the member 'pos' of a type (line 175)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), self_132527, 'pos', pos_132526)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 177, 4, False)
        # Assigning a type to the variable 'self' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Token.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        Token.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Token.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Token.stypy__str__.__dict__.__setitem__('stypy_function_name', 'Token.stypy__str__')
        Token.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        Token.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Token.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Token.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Token.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Token.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Token.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Token.stypy__str__', [], None, None, defaults, varargs, kwargs)

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

        str_132528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 15), 'str', 'Token(\'%s\', "%s")')
        
        # Obtaining an instance of the builtin type 'tuple' (line 178)
        tuple_132529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 178)
        # Adding element type (line 178)
        # Getting the type of 'self' (line 178)
        self_132530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 42), 'self')
        # Obtaining the member 'type' of a type (line 178)
        type_132531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 42), self_132530, 'type')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 42), tuple_132529, type_132531)
        # Adding element type (line 178)
        # Getting the type of 'self' (line 178)
        self_132532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 53), 'self')
        # Obtaining the member 'value' of a type (line 178)
        value_132533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 53), self_132532, 'value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 42), tuple_132529, value_132533)
        
        # Applying the binary operator '%' (line 178)
        result_mod_132534 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 15), '%', str_132528, tuple_132529)
        
        # Assigning a type to the variable 'stypy_return_type' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'stypy_return_type', result_mod_132534)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 177)
        stypy_return_type_132535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132535)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_132535


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 180, 4, False)
        # Assigning a type to the variable 'self' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Token.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        Token.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Token.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Token.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Token.stypy__repr__')
        Token.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        Token.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Token.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Token.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Token.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Token.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Token.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Token.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to __str__(...): (line 181)
        # Processing the call keyword arguments (line 181)
        kwargs_132538 = {}
        # Getting the type of 'self' (line 181)
        self_132536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 15), 'self', False)
        # Obtaining the member '__str__' of a type (line 181)
        str___132537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 15), self_132536, '__str__')
        # Calling __str__(args, kwargs) (line 181)
        str___call_result_132539 = invoke(stypy.reporting.localization.Localization(__file__, 181, 15), str___132537, *[], **kwargs_132538)
        
        # Assigning a type to the variable 'stypy_return_type' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'stypy_return_type', str___call_result_132539)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 180)
        stypy_return_type_132540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132540)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_132540


# Assigning a type to the variable 'Token' (line 171)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'Token', Token)
# Declaration of the 'Tokenizer' class

class Tokenizer(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 185, 4, False)
        # Assigning a type to the variable 'self' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Tokenizer.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 186):
        
        # Call to list(...): (line 186)
        # Processing the call arguments (line 186)
        
        # Call to keys(...): (line 186)
        # Processing the call keyword arguments (line 186)
        kwargs_132544 = {}
        # Getting the type of 'TOKENS' (line 186)
        TOKENS_132542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 27), 'TOKENS', False)
        # Obtaining the member 'keys' of a type (line 186)
        keys_132543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 27), TOKENS_132542, 'keys')
        # Calling keys(args, kwargs) (line 186)
        keys_call_result_132545 = invoke(stypy.reporting.localization.Localization(__file__, 186, 27), keys_132543, *[], **kwargs_132544)
        
        # Processing the call keyword arguments (line 186)
        kwargs_132546 = {}
        # Getting the type of 'list' (line 186)
        list_132541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 22), 'list', False)
        # Calling list(args, kwargs) (line 186)
        list_call_result_132547 = invoke(stypy.reporting.localization.Localization(__file__, 186, 22), list_132541, *[keys_call_result_132545], **kwargs_132546)
        
        # Getting the type of 'self' (line 186)
        self_132548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'self')
        # Setting the type of the member 'tokens' of a type (line 186)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), self_132548, 'tokens', list_call_result_132547)
        
        # Assigning a ListComp to a Attribute (line 187):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 187)
        self_132557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 51), 'self')
        # Obtaining the member 'tokens' of a type (line 187)
        tokens_132558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 51), self_132557, 'tokens')
        comprehension_132559 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 20), tokens_132558)
        # Assigning a type to the variable 'i' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 20), 'i', comprehension_132559)
        
        # Call to compile(...): (line 187)
        # Processing the call arguments (line 187)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 187)
        i_132551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 38), 'i', False)
        # Getting the type of 'TOKENS' (line 187)
        TOKENS_132552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 31), 'TOKENS', False)
        # Obtaining the member '__getitem__' of a type (line 187)
        getitem___132553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 31), TOKENS_132552, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 187)
        subscript_call_result_132554 = invoke(stypy.reporting.localization.Localization(__file__, 187, 31), getitem___132553, i_132551)
        
        # Processing the call keyword arguments (line 187)
        kwargs_132555 = {}
        # Getting the type of 're' (line 187)
        re_132549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 20), 're', False)
        # Obtaining the member 'compile' of a type (line 187)
        compile_132550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 20), re_132549, 'compile')
        # Calling compile(args, kwargs) (line 187)
        compile_call_result_132556 = invoke(stypy.reporting.localization.Localization(__file__, 187, 20), compile_132550, *[subscript_call_result_132554], **kwargs_132555)
        
        list_132560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 20), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 20), list_132560, compile_call_result_132556)
        # Getting the type of 'self' (line 187)
        self_132561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'self')
        # Setting the type of the member 'res' of a type (line 187)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), self_132561, 'res', list_132560)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def input(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'input'
        module_type_store = module_type_store.open_function_context('input', 189, 4, False)
        # Assigning a type to the variable 'self' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Tokenizer.input.__dict__.__setitem__('stypy_localization', localization)
        Tokenizer.input.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Tokenizer.input.__dict__.__setitem__('stypy_type_store', module_type_store)
        Tokenizer.input.__dict__.__setitem__('stypy_function_name', 'Tokenizer.input')
        Tokenizer.input.__dict__.__setitem__('stypy_param_names_list', ['s'])
        Tokenizer.input.__dict__.__setitem__('stypy_varargs_param_name', None)
        Tokenizer.input.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Tokenizer.input.__dict__.__setitem__('stypy_call_defaults', defaults)
        Tokenizer.input.__dict__.__setitem__('stypy_call_varargs', varargs)
        Tokenizer.input.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Tokenizer.input.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Tokenizer.input', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'input', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'input(...)' code ##################

        
        # Assigning a Name to a Attribute (line 190):
        # Getting the type of 's' (line 190)
        s_132562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 20), 's')
        # Getting the type of 'self' (line 190)
        self_132563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'self')
        # Setting the type of the member 'data' of a type (line 190)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), self_132563, 'data', s_132562)
        
        # Assigning a Num to a Attribute (line 191):
        int_132564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 22), 'int')
        # Getting the type of 'self' (line 191)
        self_132565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'self')
        # Setting the type of the member 'curpos' of a type (line 191)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), self_132565, 'curpos', int_132564)
        
        # Assigning a Call to a Attribute (line 192):
        
        # Call to len(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 's' (line 192)
        s_132567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 23), 's', False)
        # Processing the call keyword arguments (line 192)
        kwargs_132568 = {}
        # Getting the type of 'len' (line 192)
        len_132566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 19), 'len', False)
        # Calling len(args, kwargs) (line 192)
        len_call_result_132569 = invoke(stypy.reporting.localization.Localization(__file__, 192, 19), len_132566, *[s_132567], **kwargs_132568)
        
        # Getting the type of 'self' (line 192)
        self_132570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'self')
        # Setting the type of the member 'len' of a type (line 192)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), self_132570, 'len', len_call_result_132569)
        
        # ################# End of 'input(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'input' in the type store
        # Getting the type of 'stypy_return_type' (line 189)
        stypy_return_type_132571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132571)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'input'
        return stypy_return_type_132571


    @norecursion
    def next_token(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'next_token'
        module_type_store = module_type_store.open_function_context('next_token', 194, 4, False)
        # Assigning a type to the variable 'self' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Tokenizer.next_token.__dict__.__setitem__('stypy_localization', localization)
        Tokenizer.next_token.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Tokenizer.next_token.__dict__.__setitem__('stypy_type_store', module_type_store)
        Tokenizer.next_token.__dict__.__setitem__('stypy_function_name', 'Tokenizer.next_token')
        Tokenizer.next_token.__dict__.__setitem__('stypy_param_names_list', [])
        Tokenizer.next_token.__dict__.__setitem__('stypy_varargs_param_name', None)
        Tokenizer.next_token.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Tokenizer.next_token.__dict__.__setitem__('stypy_call_defaults', defaults)
        Tokenizer.next_token.__dict__.__setitem__('stypy_call_varargs', varargs)
        Tokenizer.next_token.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Tokenizer.next_token.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Tokenizer.next_token', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'next_token', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'next_token(...)' code ##################

        
        # Assigning a Attribute to a Name (line 195):
        # Getting the type of 'self' (line 195)
        self_132572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 17), 'self')
        # Obtaining the member 'curpos' of a type (line 195)
        curpos_132573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 17), self_132572, 'curpos')
        # Assigning a type to the variable 'curpos' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'curpos', curpos_132573)
        
        # Assigning a Attribute to a Name (line 196):
        # Getting the type of 'self' (line 196)
        self_132574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 17), 'self')
        # Obtaining the member 'tokens' of a type (line 196)
        tokens_132575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 17), self_132574, 'tokens')
        # Assigning a type to the variable 'tokens' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'tokens', tokens_132575)
        
        
        # Getting the type of 'curpos' (line 198)
        curpos_132576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 14), 'curpos')
        # Getting the type of 'self' (line 198)
        self_132577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 23), 'self')
        # Obtaining the member 'len' of a type (line 198)
        len_132578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 23), self_132577, 'len')
        # Applying the binary operator '<' (line 198)
        result_lt_132579 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 14), '<', curpos_132576, len_132578)
        
        # Testing the type of an if condition (line 198)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 8), result_lt_132579)
        # SSA begins for while statement (line 198)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        
        # Call to enumerate(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'self' (line 199)
        self_132581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 34), 'self', False)
        # Obtaining the member 'res' of a type (line 199)
        res_132582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 34), self_132581, 'res')
        # Processing the call keyword arguments (line 199)
        kwargs_132583 = {}
        # Getting the type of 'enumerate' (line 199)
        enumerate_132580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 24), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 199)
        enumerate_call_result_132584 = invoke(stypy.reporting.localization.Localization(__file__, 199, 24), enumerate_132580, *[res_132582], **kwargs_132583)
        
        # Testing the type of a for loop iterable (line 199)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 199, 12), enumerate_call_result_132584)
        # Getting the type of the for loop variable (line 199)
        for_loop_var_132585 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 199, 12), enumerate_call_result_132584)
        # Assigning a type to the variable 'i' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 12), for_loop_var_132585))
        # Assigning a type to the variable 'r' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'r', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 12), for_loop_var_132585))
        # SSA begins for a for statement (line 199)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 200):
        
        # Call to match(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'self' (line 200)
        self_132588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 28), 'self', False)
        # Obtaining the member 'data' of a type (line 200)
        data_132589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 28), self_132588, 'data')
        # Getting the type of 'curpos' (line 200)
        curpos_132590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 39), 'curpos', False)
        # Processing the call keyword arguments (line 200)
        kwargs_132591 = {}
        # Getting the type of 'r' (line 200)
        r_132586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 20), 'r', False)
        # Obtaining the member 'match' of a type (line 200)
        match_132587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 20), r_132586, 'match')
        # Calling match(args, kwargs) (line 200)
        match_call_result_132592 = invoke(stypy.reporting.localization.Localization(__file__, 200, 20), match_132587, *[data_132589, curpos_132590], **kwargs_132591)
        
        # Assigning a type to the variable 'm' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 16), 'm', match_call_result_132592)
        
        # Type idiom detected: calculating its left and rigth part (line 201)
        # Getting the type of 'm' (line 201)
        m_132593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 19), 'm')
        # Getting the type of 'None' (line 201)
        None_132594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 24), 'None')
        
        (may_be_132595, more_types_in_union_132596) = may_be_none(m_132593, None_132594)

        if may_be_132595:

            if more_types_in_union_132596:
                # Runtime conditional SSA (line 201)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store


            if more_types_in_union_132596:
                # Runtime conditional SSA for else branch (line 201)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_132595) or more_types_in_union_132596):
            
            # Assigning a Call to a Attribute (line 204):
            
            # Call to end(...): (line 204)
            # Processing the call keyword arguments (line 204)
            kwargs_132599 = {}
            # Getting the type of 'm' (line 204)
            m_132597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 34), 'm', False)
            # Obtaining the member 'end' of a type (line 204)
            end_132598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 34), m_132597, 'end')
            # Calling end(args, kwargs) (line 204)
            end_call_result_132600 = invoke(stypy.reporting.localization.Localization(__file__, 204, 34), end_132598, *[], **kwargs_132599)
            
            # Getting the type of 'self' (line 204)
            self_132601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 20), 'self')
            # Setting the type of the member 'curpos' of a type (line 204)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 20), self_132601, 'curpos', end_call_result_132600)
            
            # Call to Token(...): (line 205)
            # Processing the call arguments (line 205)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 205)
            i_132603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 45), 'i', False)
            # Getting the type of 'self' (line 205)
            self_132604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 33), 'self', False)
            # Obtaining the member 'tokens' of a type (line 205)
            tokens_132605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 33), self_132604, 'tokens')
            # Obtaining the member '__getitem__' of a type (line 205)
            getitem___132606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 33), tokens_132605, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 205)
            subscript_call_result_132607 = invoke(stypy.reporting.localization.Localization(__file__, 205, 33), getitem___132606, i_132603)
            
            
            # Call to group(...): (line 205)
            # Processing the call keyword arguments (line 205)
            kwargs_132610 = {}
            # Getting the type of 'm' (line 205)
            m_132608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 49), 'm', False)
            # Obtaining the member 'group' of a type (line 205)
            group_132609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 49), m_132608, 'group')
            # Calling group(args, kwargs) (line 205)
            group_call_result_132611 = invoke(stypy.reporting.localization.Localization(__file__, 205, 49), group_132609, *[], **kwargs_132610)
            
            # Getting the type of 'self' (line 205)
            self_132612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 60), 'self', False)
            # Obtaining the member 'curpos' of a type (line 205)
            curpos_132613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 60), self_132612, 'curpos')
            # Processing the call keyword arguments (line 205)
            kwargs_132614 = {}
            # Getting the type of 'Token' (line 205)
            Token_132602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 27), 'Token', False)
            # Calling Token(args, kwargs) (line 205)
            Token_call_result_132615 = invoke(stypy.reporting.localization.Localization(__file__, 205, 27), Token_132602, *[subscript_call_result_132607, group_call_result_132611, curpos_132613], **kwargs_132614)
            
            # Assigning a type to the variable 'stypy_return_type' (line 205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'stypy_return_type', Token_call_result_132615)

            if (may_be_132595 and more_types_in_union_132596):
                # SSA join for if statement (line 201)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to SyntaxError(...): (line 206)
        # Processing the call arguments (line 206)
        str_132617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 30), 'str', 'Unknown character at position %d (%s)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 207)
        tuple_132618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 207)
        # Adding element type (line 207)
        # Getting the type of 'self' (line 207)
        self_132619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 33), 'self', False)
        # Obtaining the member 'curpos' of a type (line 207)
        curpos_132620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 33), self_132619, 'curpos')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 33), tuple_132618, curpos_132620)
        # Adding element type (line 207)
        
        # Obtaining the type of the subscript
        # Getting the type of 'curpos' (line 207)
        curpos_132621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 56), 'curpos', False)
        # Getting the type of 'self' (line 207)
        self_132622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 46), 'self', False)
        # Obtaining the member 'data' of a type (line 207)
        data_132623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 46), self_132622, 'data')
        # Obtaining the member '__getitem__' of a type (line 207)
        getitem___132624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 46), data_132623, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 207)
        subscript_call_result_132625 = invoke(stypy.reporting.localization.Localization(__file__, 207, 46), getitem___132624, curpos_132621)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 33), tuple_132618, subscript_call_result_132625)
        
        # Applying the binary operator '%' (line 206)
        result_mod_132626 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 30), '%', str_132617, tuple_132618)
        
        # Processing the call keyword arguments (line 206)
        kwargs_132627 = {}
        # Getting the type of 'SyntaxError' (line 206)
        SyntaxError_132616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 18), 'SyntaxError', False)
        # Calling SyntaxError(args, kwargs) (line 206)
        SyntaxError_call_result_132628 = invoke(stypy.reporting.localization.Localization(__file__, 206, 18), SyntaxError_132616, *[result_mod_132626], **kwargs_132627)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 206, 12), SyntaxError_call_result_132628, 'raise parameter', BaseException)
        # SSA join for while statement (line 198)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'next_token(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'next_token' in the type store
        # Getting the type of 'stypy_return_type' (line 194)
        stypy_return_type_132629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132629)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'next_token'
        return stypy_return_type_132629


# Assigning a type to the variable 'Tokenizer' (line 184)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 0), 'Tokenizer', Tokenizer)
# Declaration of the 'FortranFormatParser' class

class FortranFormatParser(object, ):
    str_132630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, (-1)), 'str', 'Parser for fortran format strings. The parse method returns a *Format\n    instance.\n\n    Notes\n    -----\n    Only ExpFormat (exponential format for floating values) and IntFormat\n    (integer format) for now.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 234, 4, False)
        # Assigning a type to the variable 'self' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FortranFormatParser.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 235):
        
        # Call to Tokenizer(...): (line 235)
        # Processing the call keyword arguments (line 235)
        kwargs_132632 = {}
        # Getting the type of 'Tokenizer' (line 235)
        Tokenizer_132631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 25), 'Tokenizer', False)
        # Calling Tokenizer(args, kwargs) (line 235)
        Tokenizer_call_result_132633 = invoke(stypy.reporting.localization.Localization(__file__, 235, 25), Tokenizer_132631, *[], **kwargs_132632)
        
        # Getting the type of 'self' (line 235)
        self_132634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'self')
        # Setting the type of the member 'tokenizer' of a type (line 235)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), self_132634, 'tokenizer', Tokenizer_call_result_132633)
        
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
        module_type_store = module_type_store.open_function_context('parse', 237, 4, False)
        # Assigning a type to the variable 'self' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FortranFormatParser.parse.__dict__.__setitem__('stypy_localization', localization)
        FortranFormatParser.parse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FortranFormatParser.parse.__dict__.__setitem__('stypy_type_store', module_type_store)
        FortranFormatParser.parse.__dict__.__setitem__('stypy_function_name', 'FortranFormatParser.parse')
        FortranFormatParser.parse.__dict__.__setitem__('stypy_param_names_list', ['s'])
        FortranFormatParser.parse.__dict__.__setitem__('stypy_varargs_param_name', None)
        FortranFormatParser.parse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FortranFormatParser.parse.__dict__.__setitem__('stypy_call_defaults', defaults)
        FortranFormatParser.parse.__dict__.__setitem__('stypy_call_varargs', varargs)
        FortranFormatParser.parse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FortranFormatParser.parse.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FortranFormatParser.parse', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'parse', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'parse(...)' code ##################

        
        # Call to input(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 's' (line 238)
        s_132638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 29), 's', False)
        # Processing the call keyword arguments (line 238)
        kwargs_132639 = {}
        # Getting the type of 'self' (line 238)
        self_132635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'self', False)
        # Obtaining the member 'tokenizer' of a type (line 238)
        tokenizer_132636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), self_132635, 'tokenizer')
        # Obtaining the member 'input' of a type (line 238)
        input_132637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), tokenizer_132636, 'input')
        # Calling input(args, kwargs) (line 238)
        input_call_result_132640 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), input_132637, *[s_132638], **kwargs_132639)
        
        
        # Assigning a List to a Name (line 240):
        
        # Obtaining an instance of the builtin type 'list' (line 240)
        list_132641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 240)
        
        # Assigning a type to the variable 'tokens' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'tokens', list_132641)
        
        
        # SSA begins for try-except statement (line 242)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Getting the type of 'True' (line 243)
        True_132642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 18), 'True')
        # Testing the type of an if condition (line 243)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 243, 12), True_132642)
        # SSA begins for while statement (line 243)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 244):
        
        # Call to next_token(...): (line 244)
        # Processing the call keyword arguments (line 244)
        kwargs_132646 = {}
        # Getting the type of 'self' (line 244)
        self_132643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 20), 'self', False)
        # Obtaining the member 'tokenizer' of a type (line 244)
        tokenizer_132644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 20), self_132643, 'tokenizer')
        # Obtaining the member 'next_token' of a type (line 244)
        next_token_132645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 20), tokenizer_132644, 'next_token')
        # Calling next_token(args, kwargs) (line 244)
        next_token_call_result_132647 = invoke(stypy.reporting.localization.Localization(__file__, 244, 20), next_token_132645, *[], **kwargs_132646)
        
        # Assigning a type to the variable 't' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 16), 't', next_token_call_result_132647)
        
        # Type idiom detected: calculating its left and rigth part (line 245)
        # Getting the type of 't' (line 245)
        t_132648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 19), 't')
        # Getting the type of 'None' (line 245)
        None_132649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 24), 'None')
        
        (may_be_132650, more_types_in_union_132651) = may_be_none(t_132648, None_132649)

        if may_be_132650:

            if more_types_in_union_132651:
                # Runtime conditional SSA (line 245)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store


            if more_types_in_union_132651:
                # Runtime conditional SSA for else branch (line 245)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_132650) or more_types_in_union_132651):
            
            # Call to append(...): (line 248)
            # Processing the call arguments (line 248)
            # Getting the type of 't' (line 248)
            t_132654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 34), 't', False)
            # Processing the call keyword arguments (line 248)
            kwargs_132655 = {}
            # Getting the type of 'tokens' (line 248)
            tokens_132652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 20), 'tokens', False)
            # Obtaining the member 'append' of a type (line 248)
            append_132653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 20), tokens_132652, 'append')
            # Calling append(args, kwargs) (line 248)
            append_call_result_132656 = invoke(stypy.reporting.localization.Localization(__file__, 248, 20), append_132653, *[t_132654], **kwargs_132655)
            

            if (may_be_132650 and more_types_in_union_132651):
                # SSA join for if statement (line 245)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for while statement (line 243)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _parse_format(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'tokens' (line 249)
        tokens_132659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 38), 'tokens', False)
        # Processing the call keyword arguments (line 249)
        kwargs_132660 = {}
        # Getting the type of 'self' (line 249)
        self_132657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 19), 'self', False)
        # Obtaining the member '_parse_format' of a type (line 249)
        _parse_format_132658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 19), self_132657, '_parse_format')
        # Calling _parse_format(args, kwargs) (line 249)
        _parse_format_call_result_132661 = invoke(stypy.reporting.localization.Localization(__file__, 249, 19), _parse_format_132658, *[tokens_132659], **kwargs_132660)
        
        # Assigning a type to the variable 'stypy_return_type' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'stypy_return_type', _parse_format_call_result_132661)
        # SSA branch for the except part of a try statement (line 242)
        # SSA branch for the except 'SyntaxError' branch of a try statement (line 242)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'SyntaxError' (line 250)
        SyntaxError_132662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 15), 'SyntaxError')
        # Assigning a type to the variable 'e' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'e', SyntaxError_132662)
        
        # Call to BadFortranFormat(...): (line 251)
        # Processing the call arguments (line 251)
        
        # Call to str(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'e' (line 251)
        e_132665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 39), 'e', False)
        # Processing the call keyword arguments (line 251)
        kwargs_132666 = {}
        # Getting the type of 'str' (line 251)
        str_132664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 35), 'str', False)
        # Calling str(args, kwargs) (line 251)
        str_call_result_132667 = invoke(stypy.reporting.localization.Localization(__file__, 251, 35), str_132664, *[e_132665], **kwargs_132666)
        
        # Processing the call keyword arguments (line 251)
        kwargs_132668 = {}
        # Getting the type of 'BadFortranFormat' (line 251)
        BadFortranFormat_132663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 18), 'BadFortranFormat', False)
        # Calling BadFortranFormat(args, kwargs) (line 251)
        BadFortranFormat_call_result_132669 = invoke(stypy.reporting.localization.Localization(__file__, 251, 18), BadFortranFormat_132663, *[str_call_result_132667], **kwargs_132668)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 251, 12), BadFortranFormat_call_result_132669, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 242)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'parse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'parse' in the type store
        # Getting the type of 'stypy_return_type' (line 237)
        stypy_return_type_132670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132670)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'parse'
        return stypy_return_type_132670


    @norecursion
    def _get_min(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_min'
        module_type_store = module_type_store.open_function_context('_get_min', 253, 4, False)
        # Assigning a type to the variable 'self' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FortranFormatParser._get_min.__dict__.__setitem__('stypy_localization', localization)
        FortranFormatParser._get_min.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FortranFormatParser._get_min.__dict__.__setitem__('stypy_type_store', module_type_store)
        FortranFormatParser._get_min.__dict__.__setitem__('stypy_function_name', 'FortranFormatParser._get_min')
        FortranFormatParser._get_min.__dict__.__setitem__('stypy_param_names_list', ['tokens'])
        FortranFormatParser._get_min.__dict__.__setitem__('stypy_varargs_param_name', None)
        FortranFormatParser._get_min.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FortranFormatParser._get_min.__dict__.__setitem__('stypy_call_defaults', defaults)
        FortranFormatParser._get_min.__dict__.__setitem__('stypy_call_varargs', varargs)
        FortranFormatParser._get_min.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FortranFormatParser._get_min.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FortranFormatParser._get_min', ['tokens'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_min', localization, ['tokens'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_min(...)' code ##################

        
        # Assigning a Call to a Name (line 254):
        
        # Call to pop(...): (line 254)
        # Processing the call arguments (line 254)
        int_132673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 26), 'int')
        # Processing the call keyword arguments (line 254)
        kwargs_132674 = {}
        # Getting the type of 'tokens' (line 254)
        tokens_132671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 15), 'tokens', False)
        # Obtaining the member 'pop' of a type (line 254)
        pop_132672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 15), tokens_132671, 'pop')
        # Calling pop(args, kwargs) (line 254)
        pop_call_result_132675 = invoke(stypy.reporting.localization.Localization(__file__, 254, 15), pop_132672, *[int_132673], **kwargs_132674)
        
        # Assigning a type to the variable 'next' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'next', pop_call_result_132675)
        
        
        
        # Getting the type of 'next' (line 255)
        next_132676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 15), 'next')
        # Obtaining the member 'type' of a type (line 255)
        type_132677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 15), next_132676, 'type')
        str_132678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 28), 'str', 'DOT')
        # Applying the binary operator '==' (line 255)
        result_eq_132679 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 15), '==', type_132677, str_132678)
        
        # Applying the 'not' unary operator (line 255)
        result_not__132680 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 11), 'not', result_eq_132679)
        
        # Testing the type of an if condition (line 255)
        if_condition_132681 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 8), result_not__132680)
        # Assigning a type to the variable 'if_condition_132681' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'if_condition_132681', if_condition_132681)
        # SSA begins for if statement (line 255)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to SyntaxError(...): (line 256)
        # Processing the call keyword arguments (line 256)
        kwargs_132683 = {}
        # Getting the type of 'SyntaxError' (line 256)
        SyntaxError_132682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 18), 'SyntaxError', False)
        # Calling SyntaxError(args, kwargs) (line 256)
        SyntaxError_call_result_132684 = invoke(stypy.reporting.localization.Localization(__file__, 256, 18), SyntaxError_132682, *[], **kwargs_132683)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 256, 12), SyntaxError_call_result_132684, 'raise parameter', BaseException)
        # SSA join for if statement (line 255)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 257):
        
        # Call to pop(...): (line 257)
        # Processing the call arguments (line 257)
        int_132687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 26), 'int')
        # Processing the call keyword arguments (line 257)
        kwargs_132688 = {}
        # Getting the type of 'tokens' (line 257)
        tokens_132685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 15), 'tokens', False)
        # Obtaining the member 'pop' of a type (line 257)
        pop_132686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 15), tokens_132685, 'pop')
        # Calling pop(args, kwargs) (line 257)
        pop_call_result_132689 = invoke(stypy.reporting.localization.Localization(__file__, 257, 15), pop_132686, *[int_132687], **kwargs_132688)
        
        # Assigning a type to the variable 'next' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'next', pop_call_result_132689)
        # Getting the type of 'next' (line 258)
        next_132690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 15), 'next')
        # Obtaining the member 'value' of a type (line 258)
        value_132691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 15), next_132690, 'value')
        # Assigning a type to the variable 'stypy_return_type' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'stypy_return_type', value_132691)
        
        # ################# End of '_get_min(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_min' in the type store
        # Getting the type of 'stypy_return_type' (line 253)
        stypy_return_type_132692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132692)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_min'
        return stypy_return_type_132692


    @norecursion
    def _expect(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_expect'
        module_type_store = module_type_store.open_function_context('_expect', 260, 4, False)
        # Assigning a type to the variable 'self' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FortranFormatParser._expect.__dict__.__setitem__('stypy_localization', localization)
        FortranFormatParser._expect.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FortranFormatParser._expect.__dict__.__setitem__('stypy_type_store', module_type_store)
        FortranFormatParser._expect.__dict__.__setitem__('stypy_function_name', 'FortranFormatParser._expect')
        FortranFormatParser._expect.__dict__.__setitem__('stypy_param_names_list', ['token', 'tp'])
        FortranFormatParser._expect.__dict__.__setitem__('stypy_varargs_param_name', None)
        FortranFormatParser._expect.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FortranFormatParser._expect.__dict__.__setitem__('stypy_call_defaults', defaults)
        FortranFormatParser._expect.__dict__.__setitem__('stypy_call_varargs', varargs)
        FortranFormatParser._expect.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FortranFormatParser._expect.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FortranFormatParser._expect', ['token', 'tp'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_expect', localization, ['token', 'tp'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_expect(...)' code ##################

        
        
        
        # Getting the type of 'token' (line 261)
        token_132693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 15), 'token')
        # Obtaining the member 'type' of a type (line 261)
        type_132694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 15), token_132693, 'type')
        # Getting the type of 'tp' (line 261)
        tp_132695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 29), 'tp')
        # Applying the binary operator '==' (line 261)
        result_eq_132696 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 15), '==', type_132694, tp_132695)
        
        # Applying the 'not' unary operator (line 261)
        result_not__132697 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 11), 'not', result_eq_132696)
        
        # Testing the type of an if condition (line 261)
        if_condition_132698 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 8), result_not__132697)
        # Assigning a type to the variable 'if_condition_132698' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'if_condition_132698', if_condition_132698)
        # SSA begins for if statement (line 261)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to SyntaxError(...): (line 262)
        # Processing the call keyword arguments (line 262)
        kwargs_132700 = {}
        # Getting the type of 'SyntaxError' (line 262)
        SyntaxError_132699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 18), 'SyntaxError', False)
        # Calling SyntaxError(args, kwargs) (line 262)
        SyntaxError_call_result_132701 = invoke(stypy.reporting.localization.Localization(__file__, 262, 18), SyntaxError_132699, *[], **kwargs_132700)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 262, 12), SyntaxError_call_result_132701, 'raise parameter', BaseException)
        # SSA join for if statement (line 261)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_expect(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_expect' in the type store
        # Getting the type of 'stypy_return_type' (line 260)
        stypy_return_type_132702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132702)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_expect'
        return stypy_return_type_132702


    @norecursion
    def _parse_format(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_parse_format'
        module_type_store = module_type_store.open_function_context('_parse_format', 264, 4, False)
        # Assigning a type to the variable 'self' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FortranFormatParser._parse_format.__dict__.__setitem__('stypy_localization', localization)
        FortranFormatParser._parse_format.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FortranFormatParser._parse_format.__dict__.__setitem__('stypy_type_store', module_type_store)
        FortranFormatParser._parse_format.__dict__.__setitem__('stypy_function_name', 'FortranFormatParser._parse_format')
        FortranFormatParser._parse_format.__dict__.__setitem__('stypy_param_names_list', ['tokens'])
        FortranFormatParser._parse_format.__dict__.__setitem__('stypy_varargs_param_name', None)
        FortranFormatParser._parse_format.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FortranFormatParser._parse_format.__dict__.__setitem__('stypy_call_defaults', defaults)
        FortranFormatParser._parse_format.__dict__.__setitem__('stypy_call_varargs', varargs)
        FortranFormatParser._parse_format.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FortranFormatParser._parse_format.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FortranFormatParser._parse_format', ['tokens'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_parse_format', localization, ['tokens'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_parse_format(...)' code ##################

        
        
        
        
        # Obtaining the type of the subscript
        int_132703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 22), 'int')
        # Getting the type of 'tokens' (line 265)
        tokens_132704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 15), 'tokens')
        # Obtaining the member '__getitem__' of a type (line 265)
        getitem___132705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 15), tokens_132704, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 265)
        subscript_call_result_132706 = invoke(stypy.reporting.localization.Localization(__file__, 265, 15), getitem___132705, int_132703)
        
        # Obtaining the member 'type' of a type (line 265)
        type_132707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 15), subscript_call_result_132706, 'type')
        str_132708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 33), 'str', 'LPAR')
        # Applying the binary operator '==' (line 265)
        result_eq_132709 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 15), '==', type_132707, str_132708)
        
        # Applying the 'not' unary operator (line 265)
        result_not__132710 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 11), 'not', result_eq_132709)
        
        # Testing the type of an if condition (line 265)
        if_condition_132711 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 8), result_not__132710)
        # Assigning a type to the variable 'if_condition_132711' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'if_condition_132711', if_condition_132711)
        # SSA begins for if statement (line 265)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to SyntaxError(...): (line 266)
        # Processing the call arguments (line 266)
        str_132713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 30), 'str', "Expected left parenthesis at position %d (got '%s')")
        
        # Obtaining an instance of the builtin type 'tuple' (line 267)
        tuple_132714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 267)
        # Adding element type (line 267)
        int_132715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 49), tuple_132714, int_132715)
        # Adding element type (line 267)
        
        # Obtaining the type of the subscript
        int_132716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 59), 'int')
        # Getting the type of 'tokens' (line 267)
        tokens_132717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 52), 'tokens', False)
        # Obtaining the member '__getitem__' of a type (line 267)
        getitem___132718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 52), tokens_132717, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 267)
        subscript_call_result_132719 = invoke(stypy.reporting.localization.Localization(__file__, 267, 52), getitem___132718, int_132716)
        
        # Obtaining the member 'value' of a type (line 267)
        value_132720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 52), subscript_call_result_132719, 'value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 49), tuple_132714, value_132720)
        
        # Applying the binary operator '%' (line 266)
        result_mod_132721 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 30), '%', str_132713, tuple_132714)
        
        # Processing the call keyword arguments (line 266)
        kwargs_132722 = {}
        # Getting the type of 'SyntaxError' (line 266)
        SyntaxError_132712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 18), 'SyntaxError', False)
        # Calling SyntaxError(args, kwargs) (line 266)
        SyntaxError_call_result_132723 = invoke(stypy.reporting.localization.Localization(__file__, 266, 18), SyntaxError_132712, *[result_mod_132721], **kwargs_132722)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 266, 12), SyntaxError_call_result_132723, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 265)
        module_type_store.open_ssa_branch('else')
        
        
        
        
        # Obtaining the type of the subscript
        int_132724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 24), 'int')
        # Getting the type of 'tokens' (line 268)
        tokens_132725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 17), 'tokens')
        # Obtaining the member '__getitem__' of a type (line 268)
        getitem___132726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 17), tokens_132725, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 268)
        subscript_call_result_132727 = invoke(stypy.reporting.localization.Localization(__file__, 268, 17), getitem___132726, int_132724)
        
        # Obtaining the member 'type' of a type (line 268)
        type_132728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 17), subscript_call_result_132727, 'type')
        str_132729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 36), 'str', 'RPAR')
        # Applying the binary operator '==' (line 268)
        result_eq_132730 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 17), '==', type_132728, str_132729)
        
        # Applying the 'not' unary operator (line 268)
        result_not__132731 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 13), 'not', result_eq_132730)
        
        # Testing the type of an if condition (line 268)
        if_condition_132732 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 268, 13), result_not__132731)
        # Assigning a type to the variable 'if_condition_132732' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 13), 'if_condition_132732', if_condition_132732)
        # SSA begins for if statement (line 268)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to SyntaxError(...): (line 269)
        # Processing the call arguments (line 269)
        str_132734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 30), 'str', "Expected right parenthesis at position %d (got '%s')")
        
        # Obtaining an instance of the builtin type 'tuple' (line 270)
        tuple_132735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 270)
        # Adding element type (line 270)
        
        # Call to len(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'tokens' (line 270)
        tokens_132737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 53), 'tokens', False)
        # Processing the call keyword arguments (line 270)
        kwargs_132738 = {}
        # Getting the type of 'len' (line 270)
        len_132736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 49), 'len', False)
        # Calling len(args, kwargs) (line 270)
        len_call_result_132739 = invoke(stypy.reporting.localization.Localization(__file__, 270, 49), len_132736, *[tokens_132737], **kwargs_132738)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 49), tuple_132735, len_call_result_132739)
        # Adding element type (line 270)
        
        # Obtaining the type of the subscript
        int_132740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 69), 'int')
        # Getting the type of 'tokens' (line 270)
        tokens_132741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 62), 'tokens', False)
        # Obtaining the member '__getitem__' of a type (line 270)
        getitem___132742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 62), tokens_132741, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 270)
        subscript_call_result_132743 = invoke(stypy.reporting.localization.Localization(__file__, 270, 62), getitem___132742, int_132740)
        
        # Obtaining the member 'value' of a type (line 270)
        value_132744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 62), subscript_call_result_132743, 'value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 49), tuple_132735, value_132744)
        
        # Applying the binary operator '%' (line 269)
        result_mod_132745 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 30), '%', str_132734, tuple_132735)
        
        # Processing the call keyword arguments (line 269)
        kwargs_132746 = {}
        # Getting the type of 'SyntaxError' (line 269)
        SyntaxError_132733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 18), 'SyntaxError', False)
        # Calling SyntaxError(args, kwargs) (line 269)
        SyntaxError_call_result_132747 = invoke(stypy.reporting.localization.Localization(__file__, 269, 18), SyntaxError_132733, *[result_mod_132745], **kwargs_132746)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 269, 12), SyntaxError_call_result_132747, 'raise parameter', BaseException)
        # SSA join for if statement (line 268)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 265)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 272):
        
        # Obtaining the type of the subscript
        int_132748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 24), 'int')
        int_132749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 26), 'int')
        slice_132750 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 272, 17), int_132748, int_132749, None)
        # Getting the type of 'tokens' (line 272)
        tokens_132751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 17), 'tokens')
        # Obtaining the member '__getitem__' of a type (line 272)
        getitem___132752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 17), tokens_132751, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 272)
        subscript_call_result_132753 = invoke(stypy.reporting.localization.Localization(__file__, 272, 17), getitem___132752, slice_132750)
        
        # Assigning a type to the variable 'tokens' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'tokens', subscript_call_result_132753)
        
        # Assigning a ListComp to a Name (line 273):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'tokens' (line 273)
        tokens_132756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 33), 'tokens')
        comprehension_132757 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 17), tokens_132756)
        # Assigning a type to the variable 't' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 17), 't', comprehension_132757)
        # Getting the type of 't' (line 273)
        t_132754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 17), 't')
        # Obtaining the member 'type' of a type (line 273)
        type_132755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 17), t_132754, 'type')
        list_132758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 17), list_132758, type_132755)
        # Assigning a type to the variable 'types' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'types', list_132758)
        
        
        
        # Obtaining the type of the subscript
        int_132759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 17), 'int')
        # Getting the type of 'types' (line 274)
        types_132760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 11), 'types')
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___132761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 11), types_132760, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 274)
        subscript_call_result_132762 = invoke(stypy.reporting.localization.Localization(__file__, 274, 11), getitem___132761, int_132759)
        
        str_132763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 23), 'str', 'INT')
        # Applying the binary operator '==' (line 274)
        result_eq_132764 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 11), '==', subscript_call_result_132762, str_132763)
        
        # Testing the type of an if condition (line 274)
        if_condition_132765 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 274, 8), result_eq_132764)
        # Assigning a type to the variable 'if_condition_132765' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'if_condition_132765', if_condition_132765)
        # SSA begins for if statement (line 274)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 275):
        
        # Call to int(...): (line 275)
        # Processing the call arguments (line 275)
        
        # Call to pop(...): (line 275)
        # Processing the call arguments (line 275)
        int_132769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 36), 'int')
        # Processing the call keyword arguments (line 275)
        kwargs_132770 = {}
        # Getting the type of 'tokens' (line 275)
        tokens_132767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 25), 'tokens', False)
        # Obtaining the member 'pop' of a type (line 275)
        pop_132768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 25), tokens_132767, 'pop')
        # Calling pop(args, kwargs) (line 275)
        pop_call_result_132771 = invoke(stypy.reporting.localization.Localization(__file__, 275, 25), pop_132768, *[int_132769], **kwargs_132770)
        
        # Obtaining the member 'value' of a type (line 275)
        value_132772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 25), pop_call_result_132771, 'value')
        # Processing the call keyword arguments (line 275)
        kwargs_132773 = {}
        # Getting the type of 'int' (line 275)
        int_132766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 21), 'int', False)
        # Calling int(args, kwargs) (line 275)
        int_call_result_132774 = invoke(stypy.reporting.localization.Localization(__file__, 275, 21), int_132766, *[value_132772], **kwargs_132773)
        
        # Assigning a type to the variable 'repeat' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'repeat', int_call_result_132774)
        # SSA branch for the else part of an if statement (line 274)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 277):
        # Getting the type of 'None' (line 277)
        None_132775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 21), 'None')
        # Assigning a type to the variable 'repeat' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'repeat', None_132775)
        # SSA join for if statement (line 274)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 279):
        
        # Call to pop(...): (line 279)
        # Processing the call arguments (line 279)
        int_132778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 26), 'int')
        # Processing the call keyword arguments (line 279)
        kwargs_132779 = {}
        # Getting the type of 'tokens' (line 279)
        tokens_132776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 15), 'tokens', False)
        # Obtaining the member 'pop' of a type (line 279)
        pop_132777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 15), tokens_132776, 'pop')
        # Calling pop(args, kwargs) (line 279)
        pop_call_result_132780 = invoke(stypy.reporting.localization.Localization(__file__, 279, 15), pop_132777, *[int_132778], **kwargs_132779)
        
        # Assigning a type to the variable 'next' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'next', pop_call_result_132780)
        
        
        # Getting the type of 'next' (line 280)
        next_132781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 11), 'next')
        # Obtaining the member 'type' of a type (line 280)
        type_132782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 11), next_132781, 'type')
        str_132783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 24), 'str', 'INT_ID')
        # Applying the binary operator '==' (line 280)
        result_eq_132784 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 11), '==', type_132782, str_132783)
        
        # Testing the type of an if condition (line 280)
        if_condition_132785 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 8), result_eq_132784)
        # Assigning a type to the variable 'if_condition_132785' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'if_condition_132785', if_condition_132785)
        # SSA begins for if statement (line 280)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 281):
        
        # Call to _next(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'tokens' (line 281)
        tokens_132788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 30), 'tokens', False)
        str_132789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 38), 'str', 'INT')
        # Processing the call keyword arguments (line 281)
        kwargs_132790 = {}
        # Getting the type of 'self' (line 281)
        self_132786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 19), 'self', False)
        # Obtaining the member '_next' of a type (line 281)
        _next_132787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 19), self_132786, '_next')
        # Calling _next(args, kwargs) (line 281)
        _next_call_result_132791 = invoke(stypy.reporting.localization.Localization(__file__, 281, 19), _next_132787, *[tokens_132788, str_132789], **kwargs_132790)
        
        # Assigning a type to the variable 'next' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'next', _next_call_result_132791)
        
        # Assigning a Call to a Name (line 282):
        
        # Call to int(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'next' (line 282)
        next_132793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 24), 'next', False)
        # Obtaining the member 'value' of a type (line 282)
        value_132794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 24), next_132793, 'value')
        # Processing the call keyword arguments (line 282)
        kwargs_132795 = {}
        # Getting the type of 'int' (line 282)
        int_132792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 20), 'int', False)
        # Calling int(args, kwargs) (line 282)
        int_call_result_132796 = invoke(stypy.reporting.localization.Localization(__file__, 282, 20), int_132792, *[value_132794], **kwargs_132795)
        
        # Assigning a type to the variable 'width' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'width', int_call_result_132796)
        
        # Getting the type of 'tokens' (line 283)
        tokens_132797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 15), 'tokens')
        # Testing the type of an if condition (line 283)
        if_condition_132798 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 283, 12), tokens_132797)
        # Assigning a type to the variable 'if_condition_132798' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'if_condition_132798', if_condition_132798)
        # SSA begins for if statement (line 283)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 284):
        
        # Call to int(...): (line 284)
        # Processing the call arguments (line 284)
        
        # Call to _get_min(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'tokens' (line 284)
        tokens_132802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 40), 'tokens', False)
        # Processing the call keyword arguments (line 284)
        kwargs_132803 = {}
        # Getting the type of 'self' (line 284)
        self_132800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 26), 'self', False)
        # Obtaining the member '_get_min' of a type (line 284)
        _get_min_132801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 26), self_132800, '_get_min')
        # Calling _get_min(args, kwargs) (line 284)
        _get_min_call_result_132804 = invoke(stypy.reporting.localization.Localization(__file__, 284, 26), _get_min_132801, *[tokens_132802], **kwargs_132803)
        
        # Processing the call keyword arguments (line 284)
        kwargs_132805 = {}
        # Getting the type of 'int' (line 284)
        int_132799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 22), 'int', False)
        # Calling int(args, kwargs) (line 284)
        int_call_result_132806 = invoke(stypy.reporting.localization.Localization(__file__, 284, 22), int_132799, *[_get_min_call_result_132804], **kwargs_132805)
        
        # Assigning a type to the variable 'min' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'min', int_call_result_132806)
        # SSA branch for the else part of an if statement (line 283)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 286):
        # Getting the type of 'None' (line 286)
        None_132807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 22), 'None')
        # Assigning a type to the variable 'min' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 16), 'min', None_132807)
        # SSA join for if statement (line 283)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to IntFormat(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'width' (line 287)
        width_132809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 29), 'width', False)
        # Getting the type of 'min' (line 287)
        min_132810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 36), 'min', False)
        # Getting the type of 'repeat' (line 287)
        repeat_132811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 41), 'repeat', False)
        # Processing the call keyword arguments (line 287)
        kwargs_132812 = {}
        # Getting the type of 'IntFormat' (line 287)
        IntFormat_132808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 19), 'IntFormat', False)
        # Calling IntFormat(args, kwargs) (line 287)
        IntFormat_call_result_132813 = invoke(stypy.reporting.localization.Localization(__file__, 287, 19), IntFormat_132808, *[width_132809, min_132810, repeat_132811], **kwargs_132812)
        
        # Assigning a type to the variable 'stypy_return_type' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'stypy_return_type', IntFormat_call_result_132813)
        # SSA branch for the else part of an if statement (line 280)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'next' (line 288)
        next_132814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 13), 'next')
        # Obtaining the member 'type' of a type (line 288)
        type_132815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 13), next_132814, 'type')
        str_132816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 26), 'str', 'EXP_ID')
        # Applying the binary operator '==' (line 288)
        result_eq_132817 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 13), '==', type_132815, str_132816)
        
        # Testing the type of an if condition (line 288)
        if_condition_132818 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 13), result_eq_132817)
        # Assigning a type to the variable 'if_condition_132818' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 13), 'if_condition_132818', if_condition_132818)
        # SSA begins for if statement (line 288)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 289):
        
        # Call to _next(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'tokens' (line 289)
        tokens_132821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 30), 'tokens', False)
        str_132822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 38), 'str', 'INT')
        # Processing the call keyword arguments (line 289)
        kwargs_132823 = {}
        # Getting the type of 'self' (line 289)
        self_132819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 19), 'self', False)
        # Obtaining the member '_next' of a type (line 289)
        _next_132820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 19), self_132819, '_next')
        # Calling _next(args, kwargs) (line 289)
        _next_call_result_132824 = invoke(stypy.reporting.localization.Localization(__file__, 289, 19), _next_132820, *[tokens_132821, str_132822], **kwargs_132823)
        
        # Assigning a type to the variable 'next' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'next', _next_call_result_132824)
        
        # Assigning a Call to a Name (line 290):
        
        # Call to int(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'next' (line 290)
        next_132826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 24), 'next', False)
        # Obtaining the member 'value' of a type (line 290)
        value_132827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 24), next_132826, 'value')
        # Processing the call keyword arguments (line 290)
        kwargs_132828 = {}
        # Getting the type of 'int' (line 290)
        int_132825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 20), 'int', False)
        # Calling int(args, kwargs) (line 290)
        int_call_result_132829 = invoke(stypy.reporting.localization.Localization(__file__, 290, 20), int_132825, *[value_132827], **kwargs_132828)
        
        # Assigning a type to the variable 'width' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'width', int_call_result_132829)
        
        # Assigning a Call to a Name (line 292):
        
        # Call to _next(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'tokens' (line 292)
        tokens_132832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 30), 'tokens', False)
        str_132833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 38), 'str', 'DOT')
        # Processing the call keyword arguments (line 292)
        kwargs_132834 = {}
        # Getting the type of 'self' (line 292)
        self_132830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 19), 'self', False)
        # Obtaining the member '_next' of a type (line 292)
        _next_132831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 19), self_132830, '_next')
        # Calling _next(args, kwargs) (line 292)
        _next_call_result_132835 = invoke(stypy.reporting.localization.Localization(__file__, 292, 19), _next_132831, *[tokens_132832, str_132833], **kwargs_132834)
        
        # Assigning a type to the variable 'next' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'next', _next_call_result_132835)
        
        # Assigning a Call to a Name (line 294):
        
        # Call to _next(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'tokens' (line 294)
        tokens_132838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 30), 'tokens', False)
        str_132839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 38), 'str', 'INT')
        # Processing the call keyword arguments (line 294)
        kwargs_132840 = {}
        # Getting the type of 'self' (line 294)
        self_132836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 19), 'self', False)
        # Obtaining the member '_next' of a type (line 294)
        _next_132837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 19), self_132836, '_next')
        # Calling _next(args, kwargs) (line 294)
        _next_call_result_132841 = invoke(stypy.reporting.localization.Localization(__file__, 294, 19), _next_132837, *[tokens_132838, str_132839], **kwargs_132840)
        
        # Assigning a type to the variable 'next' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'next', _next_call_result_132841)
        
        # Assigning a Call to a Name (line 295):
        
        # Call to int(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'next' (line 295)
        next_132843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 30), 'next', False)
        # Obtaining the member 'value' of a type (line 295)
        value_132844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 30), next_132843, 'value')
        # Processing the call keyword arguments (line 295)
        kwargs_132845 = {}
        # Getting the type of 'int' (line 295)
        int_132842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 26), 'int', False)
        # Calling int(args, kwargs) (line 295)
        int_call_result_132846 = invoke(stypy.reporting.localization.Localization(__file__, 295, 26), int_132842, *[value_132844], **kwargs_132845)
        
        # Assigning a type to the variable 'significand' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'significand', int_call_result_132846)
        
        # Getting the type of 'tokens' (line 297)
        tokens_132847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 15), 'tokens')
        # Testing the type of an if condition (line 297)
        if_condition_132848 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 297, 12), tokens_132847)
        # Assigning a type to the variable 'if_condition_132848' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'if_condition_132848', if_condition_132848)
        # SSA begins for if statement (line 297)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 298):
        
        # Call to _next(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'tokens' (line 298)
        tokens_132851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 34), 'tokens', False)
        str_132852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 42), 'str', 'EXP_ID')
        # Processing the call keyword arguments (line 298)
        kwargs_132853 = {}
        # Getting the type of 'self' (line 298)
        self_132849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 23), 'self', False)
        # Obtaining the member '_next' of a type (line 298)
        _next_132850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 23), self_132849, '_next')
        # Calling _next(args, kwargs) (line 298)
        _next_call_result_132854 = invoke(stypy.reporting.localization.Localization(__file__, 298, 23), _next_132850, *[tokens_132851, str_132852], **kwargs_132853)
        
        # Assigning a type to the variable 'next' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 16), 'next', _next_call_result_132854)
        
        # Assigning a Call to a Name (line 300):
        
        # Call to _next(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'tokens' (line 300)
        tokens_132857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 34), 'tokens', False)
        str_132858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 42), 'str', 'INT')
        # Processing the call keyword arguments (line 300)
        kwargs_132859 = {}
        # Getting the type of 'self' (line 300)
        self_132855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 23), 'self', False)
        # Obtaining the member '_next' of a type (line 300)
        _next_132856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 23), self_132855, '_next')
        # Calling _next(args, kwargs) (line 300)
        _next_call_result_132860 = invoke(stypy.reporting.localization.Localization(__file__, 300, 23), _next_132856, *[tokens_132857, str_132858], **kwargs_132859)
        
        # Assigning a type to the variable 'next' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'next', _next_call_result_132860)
        
        # Assigning a Call to a Name (line 301):
        
        # Call to int(...): (line 301)
        # Processing the call arguments (line 301)
        # Getting the type of 'next' (line 301)
        next_132862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 26), 'next', False)
        # Obtaining the member 'value' of a type (line 301)
        value_132863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 26), next_132862, 'value')
        # Processing the call keyword arguments (line 301)
        kwargs_132864 = {}
        # Getting the type of 'int' (line 301)
        int_132861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 22), 'int', False)
        # Calling int(args, kwargs) (line 301)
        int_call_result_132865 = invoke(stypy.reporting.localization.Localization(__file__, 301, 22), int_132861, *[value_132863], **kwargs_132864)
        
        # Assigning a type to the variable 'min' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'min', int_call_result_132865)
        # SSA branch for the else part of an if statement (line 297)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 303):
        # Getting the type of 'None' (line 303)
        None_132866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 22), 'None')
        # Assigning a type to the variable 'min' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'min', None_132866)
        # SSA join for if statement (line 297)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to ExpFormat(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'width' (line 304)
        width_132868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 29), 'width', False)
        # Getting the type of 'significand' (line 304)
        significand_132869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 36), 'significand', False)
        # Getting the type of 'min' (line 304)
        min_132870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 49), 'min', False)
        # Getting the type of 'repeat' (line 304)
        repeat_132871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 54), 'repeat', False)
        # Processing the call keyword arguments (line 304)
        kwargs_132872 = {}
        # Getting the type of 'ExpFormat' (line 304)
        ExpFormat_132867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 19), 'ExpFormat', False)
        # Calling ExpFormat(args, kwargs) (line 304)
        ExpFormat_call_result_132873 = invoke(stypy.reporting.localization.Localization(__file__, 304, 19), ExpFormat_132867, *[width_132868, significand_132869, min_132870, repeat_132871], **kwargs_132872)
        
        # Assigning a type to the variable 'stypy_return_type' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'stypy_return_type', ExpFormat_call_result_132873)
        # SSA branch for the else part of an if statement (line 288)
        module_type_store.open_ssa_branch('else')
        
        # Call to SyntaxError(...): (line 306)
        # Processing the call arguments (line 306)
        str_132875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 30), 'str', 'Invalid formater type %s')
        # Getting the type of 'next' (line 306)
        next_132876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 59), 'next', False)
        # Obtaining the member 'value' of a type (line 306)
        value_132877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 59), next_132876, 'value')
        # Applying the binary operator '%' (line 306)
        result_mod_132878 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 30), '%', str_132875, value_132877)
        
        # Processing the call keyword arguments (line 306)
        kwargs_132879 = {}
        # Getting the type of 'SyntaxError' (line 306)
        SyntaxError_132874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 18), 'SyntaxError', False)
        # Calling SyntaxError(args, kwargs) (line 306)
        SyntaxError_call_result_132880 = invoke(stypy.reporting.localization.Localization(__file__, 306, 18), SyntaxError_132874, *[result_mod_132878], **kwargs_132879)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 306, 12), SyntaxError_call_result_132880, 'raise parameter', BaseException)
        # SSA join for if statement (line 288)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 280)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_parse_format(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_parse_format' in the type store
        # Getting the type of 'stypy_return_type' (line 264)
        stypy_return_type_132881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132881)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_parse_format'
        return stypy_return_type_132881


    @norecursion
    def _next(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_next'
        module_type_store = module_type_store.open_function_context('_next', 308, 4, False)
        # Assigning a type to the variable 'self' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FortranFormatParser._next.__dict__.__setitem__('stypy_localization', localization)
        FortranFormatParser._next.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FortranFormatParser._next.__dict__.__setitem__('stypy_type_store', module_type_store)
        FortranFormatParser._next.__dict__.__setitem__('stypy_function_name', 'FortranFormatParser._next')
        FortranFormatParser._next.__dict__.__setitem__('stypy_param_names_list', ['tokens', 'tp'])
        FortranFormatParser._next.__dict__.__setitem__('stypy_varargs_param_name', None)
        FortranFormatParser._next.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FortranFormatParser._next.__dict__.__setitem__('stypy_call_defaults', defaults)
        FortranFormatParser._next.__dict__.__setitem__('stypy_call_varargs', varargs)
        FortranFormatParser._next.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FortranFormatParser._next.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FortranFormatParser._next', ['tokens', 'tp'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_next', localization, ['tokens', 'tp'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_next(...)' code ##################

        
        
        
        
        # Call to len(...): (line 309)
        # Processing the call arguments (line 309)
        # Getting the type of 'tokens' (line 309)
        tokens_132883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 19), 'tokens', False)
        # Processing the call keyword arguments (line 309)
        kwargs_132884 = {}
        # Getting the type of 'len' (line 309)
        len_132882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 15), 'len', False)
        # Calling len(args, kwargs) (line 309)
        len_call_result_132885 = invoke(stypy.reporting.localization.Localization(__file__, 309, 15), len_132882, *[tokens_132883], **kwargs_132884)
        
        int_132886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 29), 'int')
        # Applying the binary operator '>' (line 309)
        result_gt_132887 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 15), '>', len_call_result_132885, int_132886)
        
        # Applying the 'not' unary operator (line 309)
        result_not__132888 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 11), 'not', result_gt_132887)
        
        # Testing the type of an if condition (line 309)
        if_condition_132889 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 309, 8), result_not__132888)
        # Assigning a type to the variable 'if_condition_132889' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'if_condition_132889', if_condition_132889)
        # SSA begins for if statement (line 309)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to SyntaxError(...): (line 310)
        # Processing the call keyword arguments (line 310)
        kwargs_132891 = {}
        # Getting the type of 'SyntaxError' (line 310)
        SyntaxError_132890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 18), 'SyntaxError', False)
        # Calling SyntaxError(args, kwargs) (line 310)
        SyntaxError_call_result_132892 = invoke(stypy.reporting.localization.Localization(__file__, 310, 18), SyntaxError_132890, *[], **kwargs_132891)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 310, 12), SyntaxError_call_result_132892, 'raise parameter', BaseException)
        # SSA join for if statement (line 309)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 311):
        
        # Call to pop(...): (line 311)
        # Processing the call arguments (line 311)
        int_132895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 26), 'int')
        # Processing the call keyword arguments (line 311)
        kwargs_132896 = {}
        # Getting the type of 'tokens' (line 311)
        tokens_132893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 15), 'tokens', False)
        # Obtaining the member 'pop' of a type (line 311)
        pop_132894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 15), tokens_132893, 'pop')
        # Calling pop(args, kwargs) (line 311)
        pop_call_result_132897 = invoke(stypy.reporting.localization.Localization(__file__, 311, 15), pop_132894, *[int_132895], **kwargs_132896)
        
        # Assigning a type to the variable 'next' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'next', pop_call_result_132897)
        
        # Call to _expect(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'next' (line 312)
        next_132900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 21), 'next', False)
        # Getting the type of 'tp' (line 312)
        tp_132901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 27), 'tp', False)
        # Processing the call keyword arguments (line 312)
        kwargs_132902 = {}
        # Getting the type of 'self' (line 312)
        self_132898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'self', False)
        # Obtaining the member '_expect' of a type (line 312)
        _expect_132899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 8), self_132898, '_expect')
        # Calling _expect(args, kwargs) (line 312)
        _expect_call_result_132903 = invoke(stypy.reporting.localization.Localization(__file__, 312, 8), _expect_132899, *[next_132900, tp_132901], **kwargs_132902)
        
        # Getting the type of 'next' (line 313)
        next_132904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 15), 'next')
        # Assigning a type to the variable 'stypy_return_type' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'stypy_return_type', next_132904)
        
        # ################# End of '_next(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_next' in the type store
        # Getting the type of 'stypy_return_type' (line 308)
        stypy_return_type_132905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132905)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_next'
        return stypy_return_type_132905


# Assigning a type to the variable 'FortranFormatParser' (line 225)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'FortranFormatParser', FortranFormatParser)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
