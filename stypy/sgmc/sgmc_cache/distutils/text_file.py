
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''text_file
2: 
3: provides the TextFile class, which gives an interface to text files
4: that (optionally) takes care of stripping comments, ignoring blank
5: lines, and joining lines with backslashes.'''
6: 
7: __revision__ = "$Id$"
8: 
9: import sys
10: 
11: 
12: class TextFile:
13: 
14:     '''Provides a file-like object that takes care of all the things you
15:        commonly want to do when processing a text file that has some
16:        line-by-line syntax: strip comments (as long as "#" is your
17:        comment character), skip blank lines, join adjacent lines by
18:        escaping the newline (ie. backslash at end of line), strip
19:        leading and/or trailing whitespace.  All of these are optional
20:        and independently controllable.
21: 
22:        Provides a 'warn()' method so you can generate warning messages that
23:        report physical line number, even if the logical line in question
24:        spans multiple physical lines.  Also provides 'unreadline()' for
25:        implementing line-at-a-time lookahead.
26: 
27:        Constructor is called as:
28: 
29:            TextFile (filename=None, file=None, **options)
30: 
31:        It bombs (RuntimeError) if both 'filename' and 'file' are None;
32:        'filename' should be a string, and 'file' a file object (or
33:        something that provides 'readline()' and 'close()' methods).  It is
34:        recommended that you supply at least 'filename', so that TextFile
35:        can include it in warning messages.  If 'file' is not supplied,
36:        TextFile creates its own using the 'open()' builtin.
37: 
38:        The options are all boolean, and affect the value returned by
39:        'readline()':
40:          strip_comments [default: true]
41:            strip from "#" to end-of-line, as well as any whitespace
42:            leading up to the "#" -- unless it is escaped by a backslash
43:          lstrip_ws [default: false]
44:            strip leading whitespace from each line before returning it
45:          rstrip_ws [default: true]
46:            strip trailing whitespace (including line terminator!) from
47:            each line before returning it
48:          skip_blanks [default: true}
49:            skip lines that are empty *after* stripping comments and
50:            whitespace.  (If both lstrip_ws and rstrip_ws are false,
51:            then some lines may consist of solely whitespace: these will
52:            *not* be skipped, even if 'skip_blanks' is true.)
53:          join_lines [default: false]
54:            if a backslash is the last non-newline character on a line
55:            after stripping comments and whitespace, join the following line
56:            to it to form one "logical line"; if N consecutive lines end
57:            with a backslash, then N+1 physical lines will be joined to
58:            form one logical line.
59:          collapse_join [default: false]
60:            strip leading whitespace from lines that are joined to their
61:            predecessor; only matters if (join_lines and not lstrip_ws)
62: 
63:        Note that since 'rstrip_ws' can strip the trailing newline, the
64:        semantics of 'readline()' must differ from those of the builtin file
65:        object's 'readline()' method!  In particular, 'readline()' returns
66:        None for end-of-file: an empty string might just be a blank line (or
67:        an all-whitespace line), if 'rstrip_ws' is true but 'skip_blanks' is
68:        not.'''
69: 
70:     default_options = { 'strip_comments': 1,
71:                         'skip_blanks':    1,
72:                         'lstrip_ws':      0,
73:                         'rstrip_ws':      1,
74:                         'join_lines':     0,
75:                         'collapse_join':  0,
76:                       }
77: 
78:     def __init__ (self, filename=None, file=None, **options):
79:         '''Construct a new TextFile object.  At least one of 'filename'
80:            (a string) and 'file' (a file-like object) must be supplied.
81:            They keyword argument options are described above and affect
82:            the values returned by 'readline()'.'''
83: 
84:         if filename is None and file is None:
85:             raise RuntimeError, \
86:                   "you must supply either or both of 'filename' and 'file'"
87: 
88:         # set values for all options -- either from client option hash
89:         # or fallback to default_options
90:         for opt in self.default_options.keys():
91:             if opt in options:
92:                 setattr (self, opt, options[opt])
93: 
94:             else:
95:                 setattr (self, opt, self.default_options[opt])
96: 
97:         # sanity check client option hash
98:         for opt in options.keys():
99:             if opt not in self.default_options:
100:                 raise KeyError, "invalid TextFile option '%s'" % opt
101: 
102:         if file is None:
103:             self.open (filename)
104:         else:
105:             self.filename = filename
106:             self.file = file
107:             self.current_line = 0       # assuming that file is at BOF!
108: 
109:         # 'linebuf' is a stack of lines that will be emptied before we
110:         # actually read from the file; it's only populated by an
111:         # 'unreadline()' operation
112:         self.linebuf = []
113: 
114: 
115:     def open (self, filename):
116:         '''Open a new file named 'filename'.  This overrides both the
117:            'filename' and 'file' arguments to the constructor.'''
118: 
119:         self.filename = filename
120:         self.file = open (self.filename, 'r')
121:         self.current_line = 0
122: 
123: 
124:     def close (self):
125:         '''Close the current file and forget everything we know about it
126:            (filename, current line number).'''
127:         file = self.file
128:         self.file = None
129:         self.filename = None
130:         self.current_line = None
131:         file.close()
132: 
133: 
134:     def gen_error (self, msg, line=None):
135:         outmsg = []
136:         if line is None:
137:             line = self.current_line
138:         outmsg.append(self.filename + ", ")
139:         if isinstance(line, (list, tuple)):
140:             outmsg.append("lines %d-%d: " % tuple (line))
141:         else:
142:             outmsg.append("line %d: " % line)
143:         outmsg.append(str(msg))
144:         return ''.join(outmsg)
145: 
146: 
147:     def error (self, msg, line=None):
148:         raise ValueError, "error: " + self.gen_error(msg, line)
149: 
150:     def warn (self, msg, line=None):
151:         '''Print (to stderr) a warning message tied to the current logical
152:            line in the current file.  If the current logical line in the
153:            file spans multiple physical lines, the warning refers to the
154:            whole range, eg. "lines 3-5".  If 'line' supplied, it overrides
155:            the current line number; it may be a list or tuple to indicate a
156:            range of physical lines, or an integer for a single physical
157:            line.'''
158:         sys.stderr.write("warning: " + self.gen_error(msg, line) + "\n")
159: 
160: 
161:     def readline (self):
162:         '''Read and return a single logical line from the current file (or
163:            from an internal buffer if lines have previously been "unread"
164:            with 'unreadline()').  If the 'join_lines' option is true, this
165:            may involve reading multiple physical lines concatenated into a
166:            single string.  Updates the current line number, so calling
167:            'warn()' after 'readline()' emits a warning about the physical
168:            line(s) just read.  Returns None on end-of-file, since the empty
169:            string can occur if 'rstrip_ws' is true but 'strip_blanks' is
170:            not.'''
171: 
172:         # If any "unread" lines waiting in 'linebuf', return the top
173:         # one.  (We don't actually buffer read-ahead data -- lines only
174:         # get put in 'linebuf' if the client explicitly does an
175:         # 'unreadline()'.
176:         if self.linebuf:
177:             line = self.linebuf[-1]
178:             del self.linebuf[-1]
179:             return line
180: 
181:         buildup_line = ''
182: 
183:         while 1:
184:             # read the line, make it None if EOF
185:             line = self.file.readline()
186:             if line == '': line = None
187: 
188:             if self.strip_comments and line:
189: 
190:                 # Look for the first "#" in the line.  If none, never
191:                 # mind.  If we find one and it's the first character, or
192:                 # is not preceded by "\", then it starts a comment --
193:                 # strip the comment, strip whitespace before it, and
194:                 # carry on.  Otherwise, it's just an escaped "#", so
195:                 # unescape it (and any other escaped "#"'s that might be
196:                 # lurking in there) and otherwise leave the line alone.
197: 
198:                 pos = line.find("#")
199:                 if pos == -1:           # no "#" -- no comments
200:                     pass
201: 
202:                 # It's definitely a comment -- either "#" is the first
203:                 # character, or it's elsewhere and unescaped.
204:                 elif pos == 0 or line[pos-1] != "\\":
205:                     # Have to preserve the trailing newline, because it's
206:                     # the job of a later step (rstrip_ws) to remove it --
207:                     # and if rstrip_ws is false, we'd better preserve it!
208:                     # (NB. this means that if the final line is all comment
209:                     # and has no trailing newline, we will think that it's
210:                     # EOF; I think that's OK.)
211:                     eol = (line[-1] == '\n') and '\n' or ''
212:                     line = line[0:pos] + eol
213: 
214:                     # If all that's left is whitespace, then skip line
215:                     # *now*, before we try to join it to 'buildup_line' --
216:                     # that way constructs like
217:                     #   hello \\
218:                     #   # comment that should be ignored
219:                     #   there
220:                     # result in "hello there".
221:                     if line.strip() == "":
222:                         continue
223: 
224:                 else:                   # it's an escaped "#"
225:                     line = line.replace("\\#", "#")
226: 
227: 
228:             # did previous line end with a backslash? then accumulate
229:             if self.join_lines and buildup_line:
230:                 # oops: end of file
231:                 if line is None:
232:                     self.warn ("continuation line immediately precedes "
233:                                "end-of-file")
234:                     return buildup_line
235: 
236:                 if self.collapse_join:
237:                     line = line.lstrip()
238:                 line = buildup_line + line
239: 
240:                 # careful: pay attention to line number when incrementing it
241:                 if isinstance(self.current_line, list):
242:                     self.current_line[1] = self.current_line[1] + 1
243:                 else:
244:                     self.current_line = [self.current_line,
245:                                          self.current_line+1]
246:             # just an ordinary line, read it as usual
247:             else:
248:                 if line is None:        # eof
249:                     return None
250: 
251:                 # still have to be careful about incrementing the line number!
252:                 if isinstance(self.current_line, list):
253:                     self.current_line = self.current_line[1] + 1
254:                 else:
255:                     self.current_line = self.current_line + 1
256: 
257: 
258:             # strip whitespace however the client wants (leading and
259:             # trailing, or one or the other, or neither)
260:             if self.lstrip_ws and self.rstrip_ws:
261:                 line = line.strip()
262:             elif self.lstrip_ws:
263:                 line = line.lstrip()
264:             elif self.rstrip_ws:
265:                 line = line.rstrip()
266: 
267:             # blank line (whether we rstrip'ed or not)? skip to next line
268:             # if appropriate
269:             if (line == '' or line == '\n') and self.skip_blanks:
270:                 continue
271: 
272:             if self.join_lines:
273:                 if line[-1] == '\\':
274:                     buildup_line = line[:-1]
275:                     continue
276: 
277:                 if line[-2:] == '\\\n':
278:                     buildup_line = line[0:-2] + '\n'
279:                     continue
280: 
281:             # well, I guess there's some actual content there: return it
282:             return line
283: 
284:     # readline ()
285: 
286: 
287:     def readlines (self):
288:         '''Read and return the list of all logical lines remaining in the
289:            current file.'''
290: 
291:         lines = []
292:         while 1:
293:             line = self.readline()
294:             if line is None:
295:                 return lines
296:             lines.append (line)
297: 
298: 
299:     def unreadline (self, line):
300:         '''Push 'line' (a string) onto an internal buffer that will be
301:            checked by future 'readline()' calls.  Handy for implementing
302:            a parser with line-at-a-time lookahead.'''
303: 
304:         self.linebuf.append (line)
305: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_8592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', 'text_file\n\nprovides the TextFile class, which gives an interface to text files\nthat (optionally) takes care of stripping comments, ignoring blank\nlines, and joining lines with backslashes.')

# Assigning a Str to a Name (line 7):
str_8593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__revision__', str_8593)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import sys' statement (line 9)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'sys', sys, module_type_store)

# Declaration of the 'TextFile' class

class TextFile:
    str_8594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, (-1)), 'str', 'Provides a file-like object that takes care of all the things you\n       commonly want to do when processing a text file that has some\n       line-by-line syntax: strip comments (as long as "#" is your\n       comment character), skip blank lines, join adjacent lines by\n       escaping the newline (ie. backslash at end of line), strip\n       leading and/or trailing whitespace.  All of these are optional\n       and independently controllable.\n\n       Provides a \'warn()\' method so you can generate warning messages that\n       report physical line number, even if the logical line in question\n       spans multiple physical lines.  Also provides \'unreadline()\' for\n       implementing line-at-a-time lookahead.\n\n       Constructor is called as:\n\n           TextFile (filename=None, file=None, **options)\n\n       It bombs (RuntimeError) if both \'filename\' and \'file\' are None;\n       \'filename\' should be a string, and \'file\' a file object (or\n       something that provides \'readline()\' and \'close()\' methods).  It is\n       recommended that you supply at least \'filename\', so that TextFile\n       can include it in warning messages.  If \'file\' is not supplied,\n       TextFile creates its own using the \'open()\' builtin.\n\n       The options are all boolean, and affect the value returned by\n       \'readline()\':\n         strip_comments [default: true]\n           strip from "#" to end-of-line, as well as any whitespace\n           leading up to the "#" -- unless it is escaped by a backslash\n         lstrip_ws [default: false]\n           strip leading whitespace from each line before returning it\n         rstrip_ws [default: true]\n           strip trailing whitespace (including line terminator!) from\n           each line before returning it\n         skip_blanks [default: true}\n           skip lines that are empty *after* stripping comments and\n           whitespace.  (If both lstrip_ws and rstrip_ws are false,\n           then some lines may consist of solely whitespace: these will\n           *not* be skipped, even if \'skip_blanks\' is true.)\n         join_lines [default: false]\n           if a backslash is the last non-newline character on a line\n           after stripping comments and whitespace, join the following line\n           to it to form one "logical line"; if N consecutive lines end\n           with a backslash, then N+1 physical lines will be joined to\n           form one logical line.\n         collapse_join [default: false]\n           strip leading whitespace from lines that are joined to their\n           predecessor; only matters if (join_lines and not lstrip_ws)\n\n       Note that since \'rstrip_ws\' can strip the trailing newline, the\n       semantics of \'readline()\' must differ from those of the builtin file\n       object\'s \'readline()\' method!  In particular, \'readline()\' returns\n       None for end-of-file: an empty string might just be a blank line (or\n       an all-whitespace line), if \'rstrip_ws\' is true but \'skip_blanks\' is\n       not.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 78)
        None_8595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 33), 'None')
        # Getting the type of 'None' (line 78)
        None_8596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'None')
        defaults = [None_8595, None_8596]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextFile.__init__', ['filename', 'file'], None, 'options', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['filename', 'file'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_8597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, (-1)), 'str', "Construct a new TextFile object.  At least one of 'filename'\n           (a string) and 'file' (a file-like object) must be supplied.\n           They keyword argument options are described above and affect\n           the values returned by 'readline()'.")
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'filename' (line 84)
        filename_8598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 11), 'filename')
        # Getting the type of 'None' (line 84)
        None_8599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 23), 'None')
        # Applying the binary operator 'is' (line 84)
        result_is__8600 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 11), 'is', filename_8598, None_8599)
        
        
        # Getting the type of 'file' (line 84)
        file_8601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 32), 'file')
        # Getting the type of 'None' (line 84)
        None_8602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 40), 'None')
        # Applying the binary operator 'is' (line 84)
        result_is__8603 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 32), 'is', file_8601, None_8602)
        
        # Applying the binary operator 'and' (line 84)
        result_and_keyword_8604 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 11), 'and', result_is__8600, result_is__8603)
        
        # Testing the type of an if condition (line 84)
        if_condition_8605 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 8), result_and_keyword_8604)
        # Assigning a type to the variable 'if_condition_8605' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'if_condition_8605', if_condition_8605)
        # SSA begins for if statement (line 84)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'RuntimeError' (line 85)
        RuntimeError_8606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 18), 'RuntimeError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 85, 12), RuntimeError_8606, 'raise parameter', BaseException)
        # SSA join for if statement (line 84)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to keys(...): (line 90)
        # Processing the call keyword arguments (line 90)
        kwargs_8610 = {}
        # Getting the type of 'self' (line 90)
        self_8607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'self', False)
        # Obtaining the member 'default_options' of a type (line 90)
        default_options_8608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 19), self_8607, 'default_options')
        # Obtaining the member 'keys' of a type (line 90)
        keys_8609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 19), default_options_8608, 'keys')
        # Calling keys(args, kwargs) (line 90)
        keys_call_result_8611 = invoke(stypy.reporting.localization.Localization(__file__, 90, 19), keys_8609, *[], **kwargs_8610)
        
        # Testing the type of a for loop iterable (line 90)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 90, 8), keys_call_result_8611)
        # Getting the type of the for loop variable (line 90)
        for_loop_var_8612 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 90, 8), keys_call_result_8611)
        # Assigning a type to the variable 'opt' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'opt', for_loop_var_8612)
        # SSA begins for a for statement (line 90)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'opt' (line 91)
        opt_8613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 15), 'opt')
        # Getting the type of 'options' (line 91)
        options_8614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 22), 'options')
        # Applying the binary operator 'in' (line 91)
        result_contains_8615 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 15), 'in', opt_8613, options_8614)
        
        # Testing the type of an if condition (line 91)
        if_condition_8616 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 12), result_contains_8615)
        # Assigning a type to the variable 'if_condition_8616' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'if_condition_8616', if_condition_8616)
        # SSA begins for if statement (line 91)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setattr(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'self' (line 92)
        self_8618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'self', False)
        # Getting the type of 'opt' (line 92)
        opt_8619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 31), 'opt', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'opt' (line 92)
        opt_8620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 44), 'opt', False)
        # Getting the type of 'options' (line 92)
        options_8621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 36), 'options', False)
        # Obtaining the member '__getitem__' of a type (line 92)
        getitem___8622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 36), options_8621, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 92)
        subscript_call_result_8623 = invoke(stypy.reporting.localization.Localization(__file__, 92, 36), getitem___8622, opt_8620)
        
        # Processing the call keyword arguments (line 92)
        kwargs_8624 = {}
        # Getting the type of 'setattr' (line 92)
        setattr_8617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'setattr', False)
        # Calling setattr(args, kwargs) (line 92)
        setattr_call_result_8625 = invoke(stypy.reporting.localization.Localization(__file__, 92, 16), setattr_8617, *[self_8618, opt_8619, subscript_call_result_8623], **kwargs_8624)
        
        # SSA branch for the else part of an if statement (line 91)
        module_type_store.open_ssa_branch('else')
        
        # Call to setattr(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'self' (line 95)
        self_8627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 25), 'self', False)
        # Getting the type of 'opt' (line 95)
        opt_8628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 31), 'opt', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'opt' (line 95)
        opt_8629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 57), 'opt', False)
        # Getting the type of 'self' (line 95)
        self_8630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 36), 'self', False)
        # Obtaining the member 'default_options' of a type (line 95)
        default_options_8631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 36), self_8630, 'default_options')
        # Obtaining the member '__getitem__' of a type (line 95)
        getitem___8632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 36), default_options_8631, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 95)
        subscript_call_result_8633 = invoke(stypy.reporting.localization.Localization(__file__, 95, 36), getitem___8632, opt_8629)
        
        # Processing the call keyword arguments (line 95)
        kwargs_8634 = {}
        # Getting the type of 'setattr' (line 95)
        setattr_8626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'setattr', False)
        # Calling setattr(args, kwargs) (line 95)
        setattr_call_result_8635 = invoke(stypy.reporting.localization.Localization(__file__, 95, 16), setattr_8626, *[self_8627, opt_8628, subscript_call_result_8633], **kwargs_8634)
        
        # SSA join for if statement (line 91)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to keys(...): (line 98)
        # Processing the call keyword arguments (line 98)
        kwargs_8638 = {}
        # Getting the type of 'options' (line 98)
        options_8636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 19), 'options', False)
        # Obtaining the member 'keys' of a type (line 98)
        keys_8637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 19), options_8636, 'keys')
        # Calling keys(args, kwargs) (line 98)
        keys_call_result_8639 = invoke(stypy.reporting.localization.Localization(__file__, 98, 19), keys_8637, *[], **kwargs_8638)
        
        # Testing the type of a for loop iterable (line 98)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 98, 8), keys_call_result_8639)
        # Getting the type of the for loop variable (line 98)
        for_loop_var_8640 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 98, 8), keys_call_result_8639)
        # Assigning a type to the variable 'opt' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'opt', for_loop_var_8640)
        # SSA begins for a for statement (line 98)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'opt' (line 99)
        opt_8641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 15), 'opt')
        # Getting the type of 'self' (line 99)
        self_8642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 26), 'self')
        # Obtaining the member 'default_options' of a type (line 99)
        default_options_8643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 26), self_8642, 'default_options')
        # Applying the binary operator 'notin' (line 99)
        result_contains_8644 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 15), 'notin', opt_8641, default_options_8643)
        
        # Testing the type of an if condition (line 99)
        if_condition_8645 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 12), result_contains_8644)
        # Assigning a type to the variable 'if_condition_8645' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'if_condition_8645', if_condition_8645)
        # SSA begins for if statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'KeyError' (line 100)
        KeyError_8646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 22), 'KeyError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 100, 16), KeyError_8646, 'raise parameter', BaseException)
        # SSA join for if statement (line 99)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 102)
        # Getting the type of 'file' (line 102)
        file_8647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 11), 'file')
        # Getting the type of 'None' (line 102)
        None_8648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 19), 'None')
        
        (may_be_8649, more_types_in_union_8650) = may_be_none(file_8647, None_8648)

        if may_be_8649:

            if more_types_in_union_8650:
                # Runtime conditional SSA (line 102)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to open(...): (line 103)
            # Processing the call arguments (line 103)
            # Getting the type of 'filename' (line 103)
            filename_8653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 23), 'filename', False)
            # Processing the call keyword arguments (line 103)
            kwargs_8654 = {}
            # Getting the type of 'self' (line 103)
            self_8651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'self', False)
            # Obtaining the member 'open' of a type (line 103)
            open_8652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), self_8651, 'open')
            # Calling open(args, kwargs) (line 103)
            open_call_result_8655 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), open_8652, *[filename_8653], **kwargs_8654)
            

            if more_types_in_union_8650:
                # Runtime conditional SSA for else branch (line 102)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_8649) or more_types_in_union_8650):
            
            # Assigning a Name to a Attribute (line 105):
            # Getting the type of 'filename' (line 105)
            filename_8656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 28), 'filename')
            # Getting the type of 'self' (line 105)
            self_8657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'self')
            # Setting the type of the member 'filename' of a type (line 105)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), self_8657, 'filename', filename_8656)
            
            # Assigning a Name to a Attribute (line 106):
            # Getting the type of 'file' (line 106)
            file_8658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 24), 'file')
            # Getting the type of 'self' (line 106)
            self_8659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'self')
            # Setting the type of the member 'file' of a type (line 106)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), self_8659, 'file', file_8658)
            
            # Assigning a Num to a Attribute (line 107):
            int_8660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 32), 'int')
            # Getting the type of 'self' (line 107)
            self_8661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'self')
            # Setting the type of the member 'current_line' of a type (line 107)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 12), self_8661, 'current_line', int_8660)

            if (may_be_8649 and more_types_in_union_8650):
                # SSA join for if statement (line 102)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a List to a Attribute (line 112):
        
        # Obtaining an instance of the builtin type 'list' (line 112)
        list_8662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 112)
        
        # Getting the type of 'self' (line 112)
        self_8663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'self')
        # Setting the type of the member 'linebuf' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), self_8663, 'linebuf', list_8662)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def open(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'open'
        module_type_store = module_type_store.open_function_context('open', 115, 4, False)
        # Assigning a type to the variable 'self' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextFile.open.__dict__.__setitem__('stypy_localization', localization)
        TextFile.open.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextFile.open.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextFile.open.__dict__.__setitem__('stypy_function_name', 'TextFile.open')
        TextFile.open.__dict__.__setitem__('stypy_param_names_list', ['filename'])
        TextFile.open.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextFile.open.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextFile.open.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextFile.open.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextFile.open.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextFile.open.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextFile.open', ['filename'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'open', localization, ['filename'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'open(...)' code ##################

        str_8664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, (-1)), 'str', "Open a new file named 'filename'.  This overrides both the\n           'filename' and 'file' arguments to the constructor.")
        
        # Assigning a Name to a Attribute (line 119):
        # Getting the type of 'filename' (line 119)
        filename_8665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 24), 'filename')
        # Getting the type of 'self' (line 119)
        self_8666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'self')
        # Setting the type of the member 'filename' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), self_8666, 'filename', filename_8665)
        
        # Assigning a Call to a Attribute (line 120):
        
        # Call to open(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'self' (line 120)
        self_8668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 26), 'self', False)
        # Obtaining the member 'filename' of a type (line 120)
        filename_8669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 26), self_8668, 'filename')
        str_8670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 41), 'str', 'r')
        # Processing the call keyword arguments (line 120)
        kwargs_8671 = {}
        # Getting the type of 'open' (line 120)
        open_8667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 20), 'open', False)
        # Calling open(args, kwargs) (line 120)
        open_call_result_8672 = invoke(stypy.reporting.localization.Localization(__file__, 120, 20), open_8667, *[filename_8669, str_8670], **kwargs_8671)
        
        # Getting the type of 'self' (line 120)
        self_8673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'self')
        # Setting the type of the member 'file' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), self_8673, 'file', open_call_result_8672)
        
        # Assigning a Num to a Attribute (line 121):
        int_8674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 28), 'int')
        # Getting the type of 'self' (line 121)
        self_8675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'self')
        # Setting the type of the member 'current_line' of a type (line 121)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), self_8675, 'current_line', int_8674)
        
        # ################# End of 'open(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'open' in the type store
        # Getting the type of 'stypy_return_type' (line 115)
        stypy_return_type_8676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8676)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'open'
        return stypy_return_type_8676


    @norecursion
    def close(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'close'
        module_type_store = module_type_store.open_function_context('close', 124, 4, False)
        # Assigning a type to the variable 'self' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextFile.close.__dict__.__setitem__('stypy_localization', localization)
        TextFile.close.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextFile.close.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextFile.close.__dict__.__setitem__('stypy_function_name', 'TextFile.close')
        TextFile.close.__dict__.__setitem__('stypy_param_names_list', [])
        TextFile.close.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextFile.close.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextFile.close.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextFile.close.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextFile.close.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextFile.close.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextFile.close', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'close', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'close(...)' code ##################

        str_8677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, (-1)), 'str', 'Close the current file and forget everything we know about it\n           (filename, current line number).')
        
        # Assigning a Attribute to a Name (line 127):
        # Getting the type of 'self' (line 127)
        self_8678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'self')
        # Obtaining the member 'file' of a type (line 127)
        file_8679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 15), self_8678, 'file')
        # Assigning a type to the variable 'file' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'file', file_8679)
        
        # Assigning a Name to a Attribute (line 128):
        # Getting the type of 'None' (line 128)
        None_8680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 20), 'None')
        # Getting the type of 'self' (line 128)
        self_8681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'self')
        # Setting the type of the member 'file' of a type (line 128)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), self_8681, 'file', None_8680)
        
        # Assigning a Name to a Attribute (line 129):
        # Getting the type of 'None' (line 129)
        None_8682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 24), 'None')
        # Getting the type of 'self' (line 129)
        self_8683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self')
        # Setting the type of the member 'filename' of a type (line 129)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_8683, 'filename', None_8682)
        
        # Assigning a Name to a Attribute (line 130):
        # Getting the type of 'None' (line 130)
        None_8684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 28), 'None')
        # Getting the type of 'self' (line 130)
        self_8685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'self')
        # Setting the type of the member 'current_line' of a type (line 130)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), self_8685, 'current_line', None_8684)
        
        # Call to close(...): (line 131)
        # Processing the call keyword arguments (line 131)
        kwargs_8688 = {}
        # Getting the type of 'file' (line 131)
        file_8686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'file', False)
        # Obtaining the member 'close' of a type (line 131)
        close_8687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), file_8686, 'close')
        # Calling close(args, kwargs) (line 131)
        close_call_result_8689 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), close_8687, *[], **kwargs_8688)
        
        
        # ################# End of 'close(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'close' in the type store
        # Getting the type of 'stypy_return_type' (line 124)
        stypy_return_type_8690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8690)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'close'
        return stypy_return_type_8690


    @norecursion
    def gen_error(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 134)
        None_8691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 35), 'None')
        defaults = [None_8691]
        # Create a new context for function 'gen_error'
        module_type_store = module_type_store.open_function_context('gen_error', 134, 4, False)
        # Assigning a type to the variable 'self' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextFile.gen_error.__dict__.__setitem__('stypy_localization', localization)
        TextFile.gen_error.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextFile.gen_error.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextFile.gen_error.__dict__.__setitem__('stypy_function_name', 'TextFile.gen_error')
        TextFile.gen_error.__dict__.__setitem__('stypy_param_names_list', ['msg', 'line'])
        TextFile.gen_error.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextFile.gen_error.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextFile.gen_error.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextFile.gen_error.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextFile.gen_error.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextFile.gen_error.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextFile.gen_error', ['msg', 'line'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'gen_error', localization, ['msg', 'line'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'gen_error(...)' code ##################

        
        # Assigning a List to a Name (line 135):
        
        # Obtaining an instance of the builtin type 'list' (line 135)
        list_8692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 135)
        
        # Assigning a type to the variable 'outmsg' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'outmsg', list_8692)
        
        # Type idiom detected: calculating its left and rigth part (line 136)
        # Getting the type of 'line' (line 136)
        line_8693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 11), 'line')
        # Getting the type of 'None' (line 136)
        None_8694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), 'None')
        
        (may_be_8695, more_types_in_union_8696) = may_be_none(line_8693, None_8694)

        if may_be_8695:

            if more_types_in_union_8696:
                # Runtime conditional SSA (line 136)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 137):
            # Getting the type of 'self' (line 137)
            self_8697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 19), 'self')
            # Obtaining the member 'current_line' of a type (line 137)
            current_line_8698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 19), self_8697, 'current_line')
            # Assigning a type to the variable 'line' (line 137)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'line', current_line_8698)

            if more_types_in_union_8696:
                # SSA join for if statement (line 136)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to append(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'self' (line 138)
        self_8701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 22), 'self', False)
        # Obtaining the member 'filename' of a type (line 138)
        filename_8702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 22), self_8701, 'filename')
        str_8703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 38), 'str', ', ')
        # Applying the binary operator '+' (line 138)
        result_add_8704 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 22), '+', filename_8702, str_8703)
        
        # Processing the call keyword arguments (line 138)
        kwargs_8705 = {}
        # Getting the type of 'outmsg' (line 138)
        outmsg_8699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'outmsg', False)
        # Obtaining the member 'append' of a type (line 138)
        append_8700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), outmsg_8699, 'append')
        # Calling append(args, kwargs) (line 138)
        append_call_result_8706 = invoke(stypy.reporting.localization.Localization(__file__, 138, 8), append_8700, *[result_add_8704], **kwargs_8705)
        
        
        
        # Call to isinstance(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'line' (line 139)
        line_8708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 22), 'line', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 139)
        tuple_8709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 139)
        # Adding element type (line 139)
        # Getting the type of 'list' (line 139)
        list_8710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 29), 'list', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 29), tuple_8709, list_8710)
        # Adding element type (line 139)
        # Getting the type of 'tuple' (line 139)
        tuple_8711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 35), 'tuple', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 29), tuple_8709, tuple_8711)
        
        # Processing the call keyword arguments (line 139)
        kwargs_8712 = {}
        # Getting the type of 'isinstance' (line 139)
        isinstance_8707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 139)
        isinstance_call_result_8713 = invoke(stypy.reporting.localization.Localization(__file__, 139, 11), isinstance_8707, *[line_8708, tuple_8709], **kwargs_8712)
        
        # Testing the type of an if condition (line 139)
        if_condition_8714 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 8), isinstance_call_result_8713)
        # Assigning a type to the variable 'if_condition_8714' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'if_condition_8714', if_condition_8714)
        # SSA begins for if statement (line 139)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 140)
        # Processing the call arguments (line 140)
        str_8717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 26), 'str', 'lines %d-%d: ')
        
        # Call to tuple(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'line' (line 140)
        line_8719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 51), 'line', False)
        # Processing the call keyword arguments (line 140)
        kwargs_8720 = {}
        # Getting the type of 'tuple' (line 140)
        tuple_8718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 44), 'tuple', False)
        # Calling tuple(args, kwargs) (line 140)
        tuple_call_result_8721 = invoke(stypy.reporting.localization.Localization(__file__, 140, 44), tuple_8718, *[line_8719], **kwargs_8720)
        
        # Applying the binary operator '%' (line 140)
        result_mod_8722 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 26), '%', str_8717, tuple_call_result_8721)
        
        # Processing the call keyword arguments (line 140)
        kwargs_8723 = {}
        # Getting the type of 'outmsg' (line 140)
        outmsg_8715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'outmsg', False)
        # Obtaining the member 'append' of a type (line 140)
        append_8716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 12), outmsg_8715, 'append')
        # Calling append(args, kwargs) (line 140)
        append_call_result_8724 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), append_8716, *[result_mod_8722], **kwargs_8723)
        
        # SSA branch for the else part of an if statement (line 139)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 142)
        # Processing the call arguments (line 142)
        str_8727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 26), 'str', 'line %d: ')
        # Getting the type of 'line' (line 142)
        line_8728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 40), 'line', False)
        # Applying the binary operator '%' (line 142)
        result_mod_8729 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 26), '%', str_8727, line_8728)
        
        # Processing the call keyword arguments (line 142)
        kwargs_8730 = {}
        # Getting the type of 'outmsg' (line 142)
        outmsg_8725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'outmsg', False)
        # Obtaining the member 'append' of a type (line 142)
        append_8726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 12), outmsg_8725, 'append')
        # Calling append(args, kwargs) (line 142)
        append_call_result_8731 = invoke(stypy.reporting.localization.Localization(__file__, 142, 12), append_8726, *[result_mod_8729], **kwargs_8730)
        
        # SSA join for if statement (line 139)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 143)
        # Processing the call arguments (line 143)
        
        # Call to str(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'msg' (line 143)
        msg_8735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 26), 'msg', False)
        # Processing the call keyword arguments (line 143)
        kwargs_8736 = {}
        # Getting the type of 'str' (line 143)
        str_8734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 22), 'str', False)
        # Calling str(args, kwargs) (line 143)
        str_call_result_8737 = invoke(stypy.reporting.localization.Localization(__file__, 143, 22), str_8734, *[msg_8735], **kwargs_8736)
        
        # Processing the call keyword arguments (line 143)
        kwargs_8738 = {}
        # Getting the type of 'outmsg' (line 143)
        outmsg_8732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'outmsg', False)
        # Obtaining the member 'append' of a type (line 143)
        append_8733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), outmsg_8732, 'append')
        # Calling append(args, kwargs) (line 143)
        append_call_result_8739 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), append_8733, *[str_call_result_8737], **kwargs_8738)
        
        
        # Call to join(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'outmsg' (line 144)
        outmsg_8742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 23), 'outmsg', False)
        # Processing the call keyword arguments (line 144)
        kwargs_8743 = {}
        str_8740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 15), 'str', '')
        # Obtaining the member 'join' of a type (line 144)
        join_8741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 15), str_8740, 'join')
        # Calling join(args, kwargs) (line 144)
        join_call_result_8744 = invoke(stypy.reporting.localization.Localization(__file__, 144, 15), join_8741, *[outmsg_8742], **kwargs_8743)
        
        # Assigning a type to the variable 'stypy_return_type' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'stypy_return_type', join_call_result_8744)
        
        # ################# End of 'gen_error(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'gen_error' in the type store
        # Getting the type of 'stypy_return_type' (line 134)
        stypy_return_type_8745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8745)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'gen_error'
        return stypy_return_type_8745


    @norecursion
    def error(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 147)
        None_8746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 31), 'None')
        defaults = [None_8746]
        # Create a new context for function 'error'
        module_type_store = module_type_store.open_function_context('error', 147, 4, False)
        # Assigning a type to the variable 'self' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextFile.error.__dict__.__setitem__('stypy_localization', localization)
        TextFile.error.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextFile.error.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextFile.error.__dict__.__setitem__('stypy_function_name', 'TextFile.error')
        TextFile.error.__dict__.__setitem__('stypy_param_names_list', ['msg', 'line'])
        TextFile.error.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextFile.error.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextFile.error.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextFile.error.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextFile.error.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextFile.error.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextFile.error', ['msg', 'line'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'error', localization, ['msg', 'line'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'error(...)' code ##################

        # Getting the type of 'ValueError' (line 148)
        ValueError_8747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 14), 'ValueError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 148, 8), ValueError_8747, 'raise parameter', BaseException)
        
        # ################# End of 'error(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'error' in the type store
        # Getting the type of 'stypy_return_type' (line 147)
        stypy_return_type_8748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8748)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'error'
        return stypy_return_type_8748


    @norecursion
    def warn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 150)
        None_8749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 30), 'None')
        defaults = [None_8749]
        # Create a new context for function 'warn'
        module_type_store = module_type_store.open_function_context('warn', 150, 4, False)
        # Assigning a type to the variable 'self' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextFile.warn.__dict__.__setitem__('stypy_localization', localization)
        TextFile.warn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextFile.warn.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextFile.warn.__dict__.__setitem__('stypy_function_name', 'TextFile.warn')
        TextFile.warn.__dict__.__setitem__('stypy_param_names_list', ['msg', 'line'])
        TextFile.warn.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextFile.warn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextFile.warn.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextFile.warn.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextFile.warn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextFile.warn.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextFile.warn', ['msg', 'line'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'warn', localization, ['msg', 'line'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'warn(...)' code ##################

        str_8750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, (-1)), 'str', 'Print (to stderr) a warning message tied to the current logical\n           line in the current file.  If the current logical line in the\n           file spans multiple physical lines, the warning refers to the\n           whole range, eg. "lines 3-5".  If \'line\' supplied, it overrides\n           the current line number; it may be a list or tuple to indicate a\n           range of physical lines, or an integer for a single physical\n           line.')
        
        # Call to write(...): (line 158)
        # Processing the call arguments (line 158)
        str_8754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 25), 'str', 'warning: ')
        
        # Call to gen_error(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'msg' (line 158)
        msg_8757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 54), 'msg', False)
        # Getting the type of 'line' (line 158)
        line_8758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 59), 'line', False)
        # Processing the call keyword arguments (line 158)
        kwargs_8759 = {}
        # Getting the type of 'self' (line 158)
        self_8755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 39), 'self', False)
        # Obtaining the member 'gen_error' of a type (line 158)
        gen_error_8756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 39), self_8755, 'gen_error')
        # Calling gen_error(args, kwargs) (line 158)
        gen_error_call_result_8760 = invoke(stypy.reporting.localization.Localization(__file__, 158, 39), gen_error_8756, *[msg_8757, line_8758], **kwargs_8759)
        
        # Applying the binary operator '+' (line 158)
        result_add_8761 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 25), '+', str_8754, gen_error_call_result_8760)
        
        str_8762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 67), 'str', '\n')
        # Applying the binary operator '+' (line 158)
        result_add_8763 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 65), '+', result_add_8761, str_8762)
        
        # Processing the call keyword arguments (line 158)
        kwargs_8764 = {}
        # Getting the type of 'sys' (line 158)
        sys_8751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 158)
        stderr_8752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), sys_8751, 'stderr')
        # Obtaining the member 'write' of a type (line 158)
        write_8753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), stderr_8752, 'write')
        # Calling write(args, kwargs) (line 158)
        write_call_result_8765 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), write_8753, *[result_add_8763], **kwargs_8764)
        
        
        # ################# End of 'warn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'warn' in the type store
        # Getting the type of 'stypy_return_type' (line 150)
        stypy_return_type_8766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8766)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'warn'
        return stypy_return_type_8766


    @norecursion
    def readline(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'readline'
        module_type_store = module_type_store.open_function_context('readline', 161, 4, False)
        # Assigning a type to the variable 'self' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextFile.readline.__dict__.__setitem__('stypy_localization', localization)
        TextFile.readline.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextFile.readline.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextFile.readline.__dict__.__setitem__('stypy_function_name', 'TextFile.readline')
        TextFile.readline.__dict__.__setitem__('stypy_param_names_list', [])
        TextFile.readline.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextFile.readline.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextFile.readline.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextFile.readline.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextFile.readline.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextFile.readline.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextFile.readline', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'readline', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'readline(...)' code ##################

        str_8767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, (-1)), 'str', 'Read and return a single logical line from the current file (or\n           from an internal buffer if lines have previously been "unread"\n           with \'unreadline()\').  If the \'join_lines\' option is true, this\n           may involve reading multiple physical lines concatenated into a\n           single string.  Updates the current line number, so calling\n           \'warn()\' after \'readline()\' emits a warning about the physical\n           line(s) just read.  Returns None on end-of-file, since the empty\n           string can occur if \'rstrip_ws\' is true but \'strip_blanks\' is\n           not.')
        
        # Getting the type of 'self' (line 176)
        self_8768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 11), 'self')
        # Obtaining the member 'linebuf' of a type (line 176)
        linebuf_8769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 11), self_8768, 'linebuf')
        # Testing the type of an if condition (line 176)
        if_condition_8770 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 176, 8), linebuf_8769)
        # Assigning a type to the variable 'if_condition_8770' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'if_condition_8770', if_condition_8770)
        # SSA begins for if statement (line 176)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 177):
        
        # Obtaining the type of the subscript
        int_8771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 32), 'int')
        # Getting the type of 'self' (line 177)
        self_8772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 19), 'self')
        # Obtaining the member 'linebuf' of a type (line 177)
        linebuf_8773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 19), self_8772, 'linebuf')
        # Obtaining the member '__getitem__' of a type (line 177)
        getitem___8774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 19), linebuf_8773, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 177)
        subscript_call_result_8775 = invoke(stypy.reporting.localization.Localization(__file__, 177, 19), getitem___8774, int_8771)
        
        # Assigning a type to the variable 'line' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'line', subscript_call_result_8775)
        # Deleting a member
        # Getting the type of 'self' (line 178)
        self_8776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'self')
        # Obtaining the member 'linebuf' of a type (line 178)
        linebuf_8777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 16), self_8776, 'linebuf')
        
        # Obtaining the type of the subscript
        int_8778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 29), 'int')
        # Getting the type of 'self' (line 178)
        self_8779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'self')
        # Obtaining the member 'linebuf' of a type (line 178)
        linebuf_8780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 16), self_8779, 'linebuf')
        # Obtaining the member '__getitem__' of a type (line 178)
        getitem___8781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 16), linebuf_8780, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 178)
        subscript_call_result_8782 = invoke(stypy.reporting.localization.Localization(__file__, 178, 16), getitem___8781, int_8778)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 12), linebuf_8777, subscript_call_result_8782)
        # Getting the type of 'line' (line 179)
        line_8783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 19), 'line')
        # Assigning a type to the variable 'stypy_return_type' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'stypy_return_type', line_8783)
        # SSA join for if statement (line 176)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Str to a Name (line 181):
        str_8784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 23), 'str', '')
        # Assigning a type to the variable 'buildup_line' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'buildup_line', str_8784)
        
        int_8785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 14), 'int')
        # Testing the type of an if condition (line 183)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 183, 8), int_8785)
        # SSA begins for while statement (line 183)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 185):
        
        # Call to readline(...): (line 185)
        # Processing the call keyword arguments (line 185)
        kwargs_8789 = {}
        # Getting the type of 'self' (line 185)
        self_8786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 19), 'self', False)
        # Obtaining the member 'file' of a type (line 185)
        file_8787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 19), self_8786, 'file')
        # Obtaining the member 'readline' of a type (line 185)
        readline_8788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 19), file_8787, 'readline')
        # Calling readline(args, kwargs) (line 185)
        readline_call_result_8790 = invoke(stypy.reporting.localization.Localization(__file__, 185, 19), readline_8788, *[], **kwargs_8789)
        
        # Assigning a type to the variable 'line' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'line', readline_call_result_8790)
        
        
        # Getting the type of 'line' (line 186)
        line_8791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 15), 'line')
        str_8792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 23), 'str', '')
        # Applying the binary operator '==' (line 186)
        result_eq_8793 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 15), '==', line_8791, str_8792)
        
        # Testing the type of an if condition (line 186)
        if_condition_8794 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 12), result_eq_8793)
        # Assigning a type to the variable 'if_condition_8794' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'if_condition_8794', if_condition_8794)
        # SSA begins for if statement (line 186)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 186):
        # Getting the type of 'None' (line 186)
        None_8795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 34), 'None')
        # Assigning a type to the variable 'line' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 27), 'line', None_8795)
        # SSA join for if statement (line 186)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 188)
        self_8796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), 'self')
        # Obtaining the member 'strip_comments' of a type (line 188)
        strip_comments_8797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 15), self_8796, 'strip_comments')
        # Getting the type of 'line' (line 188)
        line_8798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 39), 'line')
        # Applying the binary operator 'and' (line 188)
        result_and_keyword_8799 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 15), 'and', strip_comments_8797, line_8798)
        
        # Testing the type of an if condition (line 188)
        if_condition_8800 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 12), result_and_keyword_8799)
        # Assigning a type to the variable 'if_condition_8800' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'if_condition_8800', if_condition_8800)
        # SSA begins for if statement (line 188)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 198):
        
        # Call to find(...): (line 198)
        # Processing the call arguments (line 198)
        str_8803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 32), 'str', '#')
        # Processing the call keyword arguments (line 198)
        kwargs_8804 = {}
        # Getting the type of 'line' (line 198)
        line_8801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 22), 'line', False)
        # Obtaining the member 'find' of a type (line 198)
        find_8802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 22), line_8801, 'find')
        # Calling find(args, kwargs) (line 198)
        find_call_result_8805 = invoke(stypy.reporting.localization.Localization(__file__, 198, 22), find_8802, *[str_8803], **kwargs_8804)
        
        # Assigning a type to the variable 'pos' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'pos', find_call_result_8805)
        
        
        # Getting the type of 'pos' (line 199)
        pos_8806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), 'pos')
        int_8807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 26), 'int')
        # Applying the binary operator '==' (line 199)
        result_eq_8808 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 19), '==', pos_8806, int_8807)
        
        # Testing the type of an if condition (line 199)
        if_condition_8809 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 16), result_eq_8808)
        # Assigning a type to the variable 'if_condition_8809' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'if_condition_8809', if_condition_8809)
        # SSA begins for if statement (line 199)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA branch for the else part of an if statement (line 199)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'pos' (line 204)
        pos_8810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 21), 'pos')
        int_8811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 28), 'int')
        # Applying the binary operator '==' (line 204)
        result_eq_8812 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 21), '==', pos_8810, int_8811)
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'pos' (line 204)
        pos_8813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 38), 'pos')
        int_8814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 42), 'int')
        # Applying the binary operator '-' (line 204)
        result_sub_8815 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 38), '-', pos_8813, int_8814)
        
        # Getting the type of 'line' (line 204)
        line_8816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 33), 'line')
        # Obtaining the member '__getitem__' of a type (line 204)
        getitem___8817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 33), line_8816, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 204)
        subscript_call_result_8818 = invoke(stypy.reporting.localization.Localization(__file__, 204, 33), getitem___8817, result_sub_8815)
        
        str_8819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 48), 'str', '\\')
        # Applying the binary operator '!=' (line 204)
        result_ne_8820 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 33), '!=', subscript_call_result_8818, str_8819)
        
        # Applying the binary operator 'or' (line 204)
        result_or_keyword_8821 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 21), 'or', result_eq_8812, result_ne_8820)
        
        # Testing the type of an if condition (line 204)
        if_condition_8822 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 204, 21), result_or_keyword_8821)
        # Assigning a type to the variable 'if_condition_8822' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 21), 'if_condition_8822', if_condition_8822)
        # SSA begins for if statement (line 204)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BoolOp to a Name (line 211):
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        
        
        # Obtaining the type of the subscript
        int_8823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 32), 'int')
        # Getting the type of 'line' (line 211)
        line_8824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 27), 'line')
        # Obtaining the member '__getitem__' of a type (line 211)
        getitem___8825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 27), line_8824, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 211)
        subscript_call_result_8826 = invoke(stypy.reporting.localization.Localization(__file__, 211, 27), getitem___8825, int_8823)
        
        str_8827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 39), 'str', '\n')
        # Applying the binary operator '==' (line 211)
        result_eq_8828 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 27), '==', subscript_call_result_8826, str_8827)
        
        str_8829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 49), 'str', '\n')
        # Applying the binary operator 'and' (line 211)
        result_and_keyword_8830 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 26), 'and', result_eq_8828, str_8829)
        
        str_8831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 57), 'str', '')
        # Applying the binary operator 'or' (line 211)
        result_or_keyword_8832 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 26), 'or', result_and_keyword_8830, str_8831)
        
        # Assigning a type to the variable 'eol' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 20), 'eol', result_or_keyword_8832)
        
        # Assigning a BinOp to a Name (line 212):
        
        # Obtaining the type of the subscript
        int_8833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 32), 'int')
        # Getting the type of 'pos' (line 212)
        pos_8834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 34), 'pos')
        slice_8835 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 212, 27), int_8833, pos_8834, None)
        # Getting the type of 'line' (line 212)
        line_8836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 27), 'line')
        # Obtaining the member '__getitem__' of a type (line 212)
        getitem___8837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 27), line_8836, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 212)
        subscript_call_result_8838 = invoke(stypy.reporting.localization.Localization(__file__, 212, 27), getitem___8837, slice_8835)
        
        # Getting the type of 'eol' (line 212)
        eol_8839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 41), 'eol')
        # Applying the binary operator '+' (line 212)
        result_add_8840 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 27), '+', subscript_call_result_8838, eol_8839)
        
        # Assigning a type to the variable 'line' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), 'line', result_add_8840)
        
        
        
        # Call to strip(...): (line 221)
        # Processing the call keyword arguments (line 221)
        kwargs_8843 = {}
        # Getting the type of 'line' (line 221)
        line_8841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 23), 'line', False)
        # Obtaining the member 'strip' of a type (line 221)
        strip_8842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 23), line_8841, 'strip')
        # Calling strip(args, kwargs) (line 221)
        strip_call_result_8844 = invoke(stypy.reporting.localization.Localization(__file__, 221, 23), strip_8842, *[], **kwargs_8843)
        
        str_8845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 39), 'str', '')
        # Applying the binary operator '==' (line 221)
        result_eq_8846 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 23), '==', strip_call_result_8844, str_8845)
        
        # Testing the type of an if condition (line 221)
        if_condition_8847 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 20), result_eq_8846)
        # Assigning a type to the variable 'if_condition_8847' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'if_condition_8847', if_condition_8847)
        # SSA begins for if statement (line 221)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 221)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 204)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 225):
        
        # Call to replace(...): (line 225)
        # Processing the call arguments (line 225)
        str_8850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 40), 'str', '\\#')
        str_8851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 47), 'str', '#')
        # Processing the call keyword arguments (line 225)
        kwargs_8852 = {}
        # Getting the type of 'line' (line 225)
        line_8848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 27), 'line', False)
        # Obtaining the member 'replace' of a type (line 225)
        replace_8849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 27), line_8848, 'replace')
        # Calling replace(args, kwargs) (line 225)
        replace_call_result_8853 = invoke(stypy.reporting.localization.Localization(__file__, 225, 27), replace_8849, *[str_8850, str_8851], **kwargs_8852)
        
        # Assigning a type to the variable 'line' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 20), 'line', replace_call_result_8853)
        # SSA join for if statement (line 204)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 199)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 188)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 229)
        self_8854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'self')
        # Obtaining the member 'join_lines' of a type (line 229)
        join_lines_8855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 15), self_8854, 'join_lines')
        # Getting the type of 'buildup_line' (line 229)
        buildup_line_8856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 35), 'buildup_line')
        # Applying the binary operator 'and' (line 229)
        result_and_keyword_8857 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 15), 'and', join_lines_8855, buildup_line_8856)
        
        # Testing the type of an if condition (line 229)
        if_condition_8858 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 12), result_and_keyword_8857)
        # Assigning a type to the variable 'if_condition_8858' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'if_condition_8858', if_condition_8858)
        # SSA begins for if statement (line 229)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 231)
        # Getting the type of 'line' (line 231)
        line_8859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 19), 'line')
        # Getting the type of 'None' (line 231)
        None_8860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 27), 'None')
        
        (may_be_8861, more_types_in_union_8862) = may_be_none(line_8859, None_8860)

        if may_be_8861:

            if more_types_in_union_8862:
                # Runtime conditional SSA (line 231)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to warn(...): (line 232)
            # Processing the call arguments (line 232)
            str_8865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 31), 'str', 'continuation line immediately precedes end-of-file')
            # Processing the call keyword arguments (line 232)
            kwargs_8866 = {}
            # Getting the type of 'self' (line 232)
            self_8863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 20), 'self', False)
            # Obtaining the member 'warn' of a type (line 232)
            warn_8864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 20), self_8863, 'warn')
            # Calling warn(args, kwargs) (line 232)
            warn_call_result_8867 = invoke(stypy.reporting.localization.Localization(__file__, 232, 20), warn_8864, *[str_8865], **kwargs_8866)
            
            # Getting the type of 'buildup_line' (line 234)
            buildup_line_8868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 27), 'buildup_line')
            # Assigning a type to the variable 'stypy_return_type' (line 234)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 20), 'stypy_return_type', buildup_line_8868)

            if more_types_in_union_8862:
                # SSA join for if statement (line 231)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'self' (line 236)
        self_8869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 19), 'self')
        # Obtaining the member 'collapse_join' of a type (line 236)
        collapse_join_8870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 19), self_8869, 'collapse_join')
        # Testing the type of an if condition (line 236)
        if_condition_8871 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 16), collapse_join_8870)
        # Assigning a type to the variable 'if_condition_8871' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 16), 'if_condition_8871', if_condition_8871)
        # SSA begins for if statement (line 236)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 237):
        
        # Call to lstrip(...): (line 237)
        # Processing the call keyword arguments (line 237)
        kwargs_8874 = {}
        # Getting the type of 'line' (line 237)
        line_8872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 27), 'line', False)
        # Obtaining the member 'lstrip' of a type (line 237)
        lstrip_8873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 27), line_8872, 'lstrip')
        # Calling lstrip(args, kwargs) (line 237)
        lstrip_call_result_8875 = invoke(stypy.reporting.localization.Localization(__file__, 237, 27), lstrip_8873, *[], **kwargs_8874)
        
        # Assigning a type to the variable 'line' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 20), 'line', lstrip_call_result_8875)
        # SSA join for if statement (line 236)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 238):
        # Getting the type of 'buildup_line' (line 238)
        buildup_line_8876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 23), 'buildup_line')
        # Getting the type of 'line' (line 238)
        line_8877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 38), 'line')
        # Applying the binary operator '+' (line 238)
        result_add_8878 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 23), '+', buildup_line_8876, line_8877)
        
        # Assigning a type to the variable 'line' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'line', result_add_8878)
        
        # Type idiom detected: calculating its left and rigth part (line 241)
        # Getting the type of 'list' (line 241)
        list_8879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 49), 'list')
        # Getting the type of 'self' (line 241)
        self_8880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 30), 'self')
        # Obtaining the member 'current_line' of a type (line 241)
        current_line_8881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 30), self_8880, 'current_line')
        
        (may_be_8882, more_types_in_union_8883) = may_be_subtype(list_8879, current_line_8881)

        if may_be_8882:

            if more_types_in_union_8883:
                # Runtime conditional SSA (line 241)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 241)
            self_8884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 16), 'self')
            # Obtaining the member 'current_line' of a type (line 241)
            current_line_8885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 16), self_8884, 'current_line')
            # Setting the type of the member 'current_line' of a type (line 241)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 16), self_8884, 'current_line', remove_not_subtype_from_union(current_line_8881, list))
            
            # Assigning a BinOp to a Subscript (line 242):
            
            # Obtaining the type of the subscript
            int_8886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 61), 'int')
            # Getting the type of 'self' (line 242)
            self_8887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 43), 'self')
            # Obtaining the member 'current_line' of a type (line 242)
            current_line_8888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 43), self_8887, 'current_line')
            # Obtaining the member '__getitem__' of a type (line 242)
            getitem___8889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 43), current_line_8888, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 242)
            subscript_call_result_8890 = invoke(stypy.reporting.localization.Localization(__file__, 242, 43), getitem___8889, int_8886)
            
            int_8891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 66), 'int')
            # Applying the binary operator '+' (line 242)
            result_add_8892 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 43), '+', subscript_call_result_8890, int_8891)
            
            # Getting the type of 'self' (line 242)
            self_8893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 20), 'self')
            # Obtaining the member 'current_line' of a type (line 242)
            current_line_8894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 20), self_8893, 'current_line')
            int_8895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 38), 'int')
            # Storing an element on a container (line 242)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 20), current_line_8894, (int_8895, result_add_8892))

            if more_types_in_union_8883:
                # Runtime conditional SSA for else branch (line 241)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_8882) or more_types_in_union_8883):
            # Getting the type of 'self' (line 241)
            self_8896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 16), 'self')
            # Obtaining the member 'current_line' of a type (line 241)
            current_line_8897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 16), self_8896, 'current_line')
            # Setting the type of the member 'current_line' of a type (line 241)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 16), self_8896, 'current_line', remove_subtype_from_union(current_line_8881, list))
            
            # Assigning a List to a Attribute (line 244):
            
            # Obtaining an instance of the builtin type 'list' (line 244)
            list_8898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 40), 'list')
            # Adding type elements to the builtin type 'list' instance (line 244)
            # Adding element type (line 244)
            # Getting the type of 'self' (line 244)
            self_8899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 41), 'self')
            # Obtaining the member 'current_line' of a type (line 244)
            current_line_8900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 41), self_8899, 'current_line')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 40), list_8898, current_line_8900)
            # Adding element type (line 244)
            # Getting the type of 'self' (line 245)
            self_8901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 41), 'self')
            # Obtaining the member 'current_line' of a type (line 245)
            current_line_8902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 41), self_8901, 'current_line')
            int_8903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 59), 'int')
            # Applying the binary operator '+' (line 245)
            result_add_8904 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 41), '+', current_line_8902, int_8903)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 40), list_8898, result_add_8904)
            
            # Getting the type of 'self' (line 244)
            self_8905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 20), 'self')
            # Setting the type of the member 'current_line' of a type (line 244)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 20), self_8905, 'current_line', list_8898)

            if (may_be_8882 and more_types_in_union_8883):
                # SSA join for if statement (line 241)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA branch for the else part of an if statement (line 229)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 248)
        # Getting the type of 'line' (line 248)
        line_8906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 19), 'line')
        # Getting the type of 'None' (line 248)
        None_8907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 27), 'None')
        
        (may_be_8908, more_types_in_union_8909) = may_be_none(line_8906, None_8907)

        if may_be_8908:

            if more_types_in_union_8909:
                # Runtime conditional SSA (line 248)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'None' (line 249)
            None_8910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 27), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 249)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 20), 'stypy_return_type', None_8910)

            if more_types_in_union_8909:
                # SSA join for if statement (line 248)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 252)
        # Getting the type of 'list' (line 252)
        list_8911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 49), 'list')
        # Getting the type of 'self' (line 252)
        self_8912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 30), 'self')
        # Obtaining the member 'current_line' of a type (line 252)
        current_line_8913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 30), self_8912, 'current_line')
        
        (may_be_8914, more_types_in_union_8915) = may_be_subtype(list_8911, current_line_8913)

        if may_be_8914:

            if more_types_in_union_8915:
                # Runtime conditional SSA (line 252)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 252)
            self_8916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'self')
            # Obtaining the member 'current_line' of a type (line 252)
            current_line_8917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 16), self_8916, 'current_line')
            # Setting the type of the member 'current_line' of a type (line 252)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 16), self_8916, 'current_line', remove_not_subtype_from_union(current_line_8913, list))
            
            # Assigning a BinOp to a Attribute (line 253):
            
            # Obtaining the type of the subscript
            int_8918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 58), 'int')
            # Getting the type of 'self' (line 253)
            self_8919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 40), 'self')
            # Obtaining the member 'current_line' of a type (line 253)
            current_line_8920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 40), self_8919, 'current_line')
            # Obtaining the member '__getitem__' of a type (line 253)
            getitem___8921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 40), current_line_8920, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 253)
            subscript_call_result_8922 = invoke(stypy.reporting.localization.Localization(__file__, 253, 40), getitem___8921, int_8918)
            
            int_8923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 63), 'int')
            # Applying the binary operator '+' (line 253)
            result_add_8924 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 40), '+', subscript_call_result_8922, int_8923)
            
            # Getting the type of 'self' (line 253)
            self_8925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 20), 'self')
            # Setting the type of the member 'current_line' of a type (line 253)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 20), self_8925, 'current_line', result_add_8924)

            if more_types_in_union_8915:
                # Runtime conditional SSA for else branch (line 252)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_8914) or more_types_in_union_8915):
            # Getting the type of 'self' (line 252)
            self_8926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'self')
            # Obtaining the member 'current_line' of a type (line 252)
            current_line_8927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 16), self_8926, 'current_line')
            # Setting the type of the member 'current_line' of a type (line 252)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 16), self_8926, 'current_line', remove_subtype_from_union(current_line_8913, list))
            
            # Assigning a BinOp to a Attribute (line 255):
            # Getting the type of 'self' (line 255)
            self_8928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 40), 'self')
            # Obtaining the member 'current_line' of a type (line 255)
            current_line_8929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 40), self_8928, 'current_line')
            int_8930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 60), 'int')
            # Applying the binary operator '+' (line 255)
            result_add_8931 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 40), '+', current_line_8929, int_8930)
            
            # Getting the type of 'self' (line 255)
            self_8932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 20), 'self')
            # Setting the type of the member 'current_line' of a type (line 255)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 20), self_8932, 'current_line', result_add_8931)

            if (may_be_8914 and more_types_in_union_8915):
                # SSA join for if statement (line 252)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 229)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 260)
        self_8933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 15), 'self')
        # Obtaining the member 'lstrip_ws' of a type (line 260)
        lstrip_ws_8934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 15), self_8933, 'lstrip_ws')
        # Getting the type of 'self' (line 260)
        self_8935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 34), 'self')
        # Obtaining the member 'rstrip_ws' of a type (line 260)
        rstrip_ws_8936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 34), self_8935, 'rstrip_ws')
        # Applying the binary operator 'and' (line 260)
        result_and_keyword_8937 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 15), 'and', lstrip_ws_8934, rstrip_ws_8936)
        
        # Testing the type of an if condition (line 260)
        if_condition_8938 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 12), result_and_keyword_8937)
        # Assigning a type to the variable 'if_condition_8938' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'if_condition_8938', if_condition_8938)
        # SSA begins for if statement (line 260)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 261):
        
        # Call to strip(...): (line 261)
        # Processing the call keyword arguments (line 261)
        kwargs_8941 = {}
        # Getting the type of 'line' (line 261)
        line_8939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 23), 'line', False)
        # Obtaining the member 'strip' of a type (line 261)
        strip_8940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 23), line_8939, 'strip')
        # Calling strip(args, kwargs) (line 261)
        strip_call_result_8942 = invoke(stypy.reporting.localization.Localization(__file__, 261, 23), strip_8940, *[], **kwargs_8941)
        
        # Assigning a type to the variable 'line' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 16), 'line', strip_call_result_8942)
        # SSA branch for the else part of an if statement (line 260)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'self' (line 262)
        self_8943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 17), 'self')
        # Obtaining the member 'lstrip_ws' of a type (line 262)
        lstrip_ws_8944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 17), self_8943, 'lstrip_ws')
        # Testing the type of an if condition (line 262)
        if_condition_8945 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 262, 17), lstrip_ws_8944)
        # Assigning a type to the variable 'if_condition_8945' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 17), 'if_condition_8945', if_condition_8945)
        # SSA begins for if statement (line 262)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 263):
        
        # Call to lstrip(...): (line 263)
        # Processing the call keyword arguments (line 263)
        kwargs_8948 = {}
        # Getting the type of 'line' (line 263)
        line_8946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 23), 'line', False)
        # Obtaining the member 'lstrip' of a type (line 263)
        lstrip_8947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 23), line_8946, 'lstrip')
        # Calling lstrip(args, kwargs) (line 263)
        lstrip_call_result_8949 = invoke(stypy.reporting.localization.Localization(__file__, 263, 23), lstrip_8947, *[], **kwargs_8948)
        
        # Assigning a type to the variable 'line' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 16), 'line', lstrip_call_result_8949)
        # SSA branch for the else part of an if statement (line 262)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'self' (line 264)
        self_8950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 17), 'self')
        # Obtaining the member 'rstrip_ws' of a type (line 264)
        rstrip_ws_8951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 17), self_8950, 'rstrip_ws')
        # Testing the type of an if condition (line 264)
        if_condition_8952 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 264, 17), rstrip_ws_8951)
        # Assigning a type to the variable 'if_condition_8952' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 17), 'if_condition_8952', if_condition_8952)
        # SSA begins for if statement (line 264)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 265):
        
        # Call to rstrip(...): (line 265)
        # Processing the call keyword arguments (line 265)
        kwargs_8955 = {}
        # Getting the type of 'line' (line 265)
        line_8953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 23), 'line', False)
        # Obtaining the member 'rstrip' of a type (line 265)
        rstrip_8954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 23), line_8953, 'rstrip')
        # Calling rstrip(args, kwargs) (line 265)
        rstrip_call_result_8956 = invoke(stypy.reporting.localization.Localization(__file__, 265, 23), rstrip_8954, *[], **kwargs_8955)
        
        # Assigning a type to the variable 'line' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 16), 'line', rstrip_call_result_8956)
        # SSA join for if statement (line 264)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 262)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 260)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        
        # Getting the type of 'line' (line 269)
        line_8957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 16), 'line')
        str_8958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 24), 'str', '')
        # Applying the binary operator '==' (line 269)
        result_eq_8959 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 16), '==', line_8957, str_8958)
        
        
        # Getting the type of 'line' (line 269)
        line_8960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 30), 'line')
        str_8961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 38), 'str', '\n')
        # Applying the binary operator '==' (line 269)
        result_eq_8962 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 30), '==', line_8960, str_8961)
        
        # Applying the binary operator 'or' (line 269)
        result_or_keyword_8963 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 16), 'or', result_eq_8959, result_eq_8962)
        
        # Getting the type of 'self' (line 269)
        self_8964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 48), 'self')
        # Obtaining the member 'skip_blanks' of a type (line 269)
        skip_blanks_8965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 48), self_8964, 'skip_blanks')
        # Applying the binary operator 'and' (line 269)
        result_and_keyword_8966 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 15), 'and', result_or_keyword_8963, skip_blanks_8965)
        
        # Testing the type of an if condition (line 269)
        if_condition_8967 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 269, 12), result_and_keyword_8966)
        # Assigning a type to the variable 'if_condition_8967' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'if_condition_8967', if_condition_8967)
        # SSA begins for if statement (line 269)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 269)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 272)
        self_8968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 15), 'self')
        # Obtaining the member 'join_lines' of a type (line 272)
        join_lines_8969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 15), self_8968, 'join_lines')
        # Testing the type of an if condition (line 272)
        if_condition_8970 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 12), join_lines_8969)
        # Assigning a type to the variable 'if_condition_8970' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'if_condition_8970', if_condition_8970)
        # SSA begins for if statement (line 272)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Obtaining the type of the subscript
        int_8971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 24), 'int')
        # Getting the type of 'line' (line 273)
        line_8972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 19), 'line')
        # Obtaining the member '__getitem__' of a type (line 273)
        getitem___8973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 19), line_8972, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 273)
        subscript_call_result_8974 = invoke(stypy.reporting.localization.Localization(__file__, 273, 19), getitem___8973, int_8971)
        
        str_8975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 31), 'str', '\\')
        # Applying the binary operator '==' (line 273)
        result_eq_8976 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 19), '==', subscript_call_result_8974, str_8975)
        
        # Testing the type of an if condition (line 273)
        if_condition_8977 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 16), result_eq_8976)
        # Assigning a type to the variable 'if_condition_8977' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'if_condition_8977', if_condition_8977)
        # SSA begins for if statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 274):
        
        # Obtaining the type of the subscript
        int_8978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 41), 'int')
        slice_8979 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 274, 35), None, int_8978, None)
        # Getting the type of 'line' (line 274)
        line_8980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 35), 'line')
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___8981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 35), line_8980, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 274)
        subscript_call_result_8982 = invoke(stypy.reporting.localization.Localization(__file__, 274, 35), getitem___8981, slice_8979)
        
        # Assigning a type to the variable 'buildup_line' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 20), 'buildup_line', subscript_call_result_8982)
        # SSA join for if statement (line 273)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        int_8983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 24), 'int')
        slice_8984 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 277, 19), int_8983, None, None)
        # Getting the type of 'line' (line 277)
        line_8985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 19), 'line')
        # Obtaining the member '__getitem__' of a type (line 277)
        getitem___8986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 19), line_8985, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 277)
        subscript_call_result_8987 = invoke(stypy.reporting.localization.Localization(__file__, 277, 19), getitem___8986, slice_8984)
        
        str_8988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 32), 'str', '\\\n')
        # Applying the binary operator '==' (line 277)
        result_eq_8989 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 19), '==', subscript_call_result_8987, str_8988)
        
        # Testing the type of an if condition (line 277)
        if_condition_8990 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 277, 16), result_eq_8989)
        # Assigning a type to the variable 'if_condition_8990' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 16), 'if_condition_8990', if_condition_8990)
        # SSA begins for if statement (line 277)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 278):
        
        # Obtaining the type of the subscript
        int_8991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 40), 'int')
        int_8992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 42), 'int')
        slice_8993 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 278, 35), int_8991, int_8992, None)
        # Getting the type of 'line' (line 278)
        line_8994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 35), 'line')
        # Obtaining the member '__getitem__' of a type (line 278)
        getitem___8995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 35), line_8994, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 278)
        subscript_call_result_8996 = invoke(stypy.reporting.localization.Localization(__file__, 278, 35), getitem___8995, slice_8993)
        
        str_8997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 48), 'str', '\n')
        # Applying the binary operator '+' (line 278)
        result_add_8998 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 35), '+', subscript_call_result_8996, str_8997)
        
        # Assigning a type to the variable 'buildup_line' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 20), 'buildup_line', result_add_8998)
        # SSA join for if statement (line 277)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 272)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'line' (line 282)
        line_8999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 19), 'line')
        # Assigning a type to the variable 'stypy_return_type' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'stypy_return_type', line_8999)
        # SSA join for while statement (line 183)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'readline(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'readline' in the type store
        # Getting the type of 'stypy_return_type' (line 161)
        stypy_return_type_9000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9000)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'readline'
        return stypy_return_type_9000


    @norecursion
    def readlines(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'readlines'
        module_type_store = module_type_store.open_function_context('readlines', 287, 4, False)
        # Assigning a type to the variable 'self' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextFile.readlines.__dict__.__setitem__('stypy_localization', localization)
        TextFile.readlines.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextFile.readlines.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextFile.readlines.__dict__.__setitem__('stypy_function_name', 'TextFile.readlines')
        TextFile.readlines.__dict__.__setitem__('stypy_param_names_list', [])
        TextFile.readlines.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextFile.readlines.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextFile.readlines.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextFile.readlines.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextFile.readlines.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextFile.readlines.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextFile.readlines', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'readlines', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'readlines(...)' code ##################

        str_9001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, (-1)), 'str', 'Read and return the list of all logical lines remaining in the\n           current file.')
        
        # Assigning a List to a Name (line 291):
        
        # Obtaining an instance of the builtin type 'list' (line 291)
        list_9002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 291)
        
        # Assigning a type to the variable 'lines' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'lines', list_9002)
        
        int_9003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 14), 'int')
        # Testing the type of an if condition (line 292)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 8), int_9003)
        # SSA begins for while statement (line 292)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 293):
        
        # Call to readline(...): (line 293)
        # Processing the call keyword arguments (line 293)
        kwargs_9006 = {}
        # Getting the type of 'self' (line 293)
        self_9004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 19), 'self', False)
        # Obtaining the member 'readline' of a type (line 293)
        readline_9005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 19), self_9004, 'readline')
        # Calling readline(args, kwargs) (line 293)
        readline_call_result_9007 = invoke(stypy.reporting.localization.Localization(__file__, 293, 19), readline_9005, *[], **kwargs_9006)
        
        # Assigning a type to the variable 'line' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'line', readline_call_result_9007)
        
        # Type idiom detected: calculating its left and rigth part (line 294)
        # Getting the type of 'line' (line 294)
        line_9008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 15), 'line')
        # Getting the type of 'None' (line 294)
        None_9009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 23), 'None')
        
        (may_be_9010, more_types_in_union_9011) = may_be_none(line_9008, None_9009)

        if may_be_9010:

            if more_types_in_union_9011:
                # Runtime conditional SSA (line 294)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'lines' (line 295)
            lines_9012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 23), 'lines')
            # Assigning a type to the variable 'stypy_return_type' (line 295)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'stypy_return_type', lines_9012)

            if more_types_in_union_9011:
                # SSA join for if statement (line 294)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to append(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 'line' (line 296)
        line_9015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 26), 'line', False)
        # Processing the call keyword arguments (line 296)
        kwargs_9016 = {}
        # Getting the type of 'lines' (line 296)
        lines_9013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'lines', False)
        # Obtaining the member 'append' of a type (line 296)
        append_9014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 12), lines_9013, 'append')
        # Calling append(args, kwargs) (line 296)
        append_call_result_9017 = invoke(stypy.reporting.localization.Localization(__file__, 296, 12), append_9014, *[line_9015], **kwargs_9016)
        
        # SSA join for while statement (line 292)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'readlines(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'readlines' in the type store
        # Getting the type of 'stypy_return_type' (line 287)
        stypy_return_type_9018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9018)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'readlines'
        return stypy_return_type_9018


    @norecursion
    def unreadline(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'unreadline'
        module_type_store = module_type_store.open_function_context('unreadline', 299, 4, False)
        # Assigning a type to the variable 'self' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextFile.unreadline.__dict__.__setitem__('stypy_localization', localization)
        TextFile.unreadline.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextFile.unreadline.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextFile.unreadline.__dict__.__setitem__('stypy_function_name', 'TextFile.unreadline')
        TextFile.unreadline.__dict__.__setitem__('stypy_param_names_list', ['line'])
        TextFile.unreadline.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextFile.unreadline.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextFile.unreadline.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextFile.unreadline.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextFile.unreadline.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextFile.unreadline.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextFile.unreadline', ['line'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'unreadline', localization, ['line'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'unreadline(...)' code ##################

        str_9019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, (-1)), 'str', "Push 'line' (a string) onto an internal buffer that will be\n           checked by future 'readline()' calls.  Handy for implementing\n           a parser with line-at-a-time lookahead.")
        
        # Call to append(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'line' (line 304)
        line_9023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 29), 'line', False)
        # Processing the call keyword arguments (line 304)
        kwargs_9024 = {}
        # Getting the type of 'self' (line 304)
        self_9020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'self', False)
        # Obtaining the member 'linebuf' of a type (line 304)
        linebuf_9021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), self_9020, 'linebuf')
        # Obtaining the member 'append' of a type (line 304)
        append_9022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), linebuf_9021, 'append')
        # Calling append(args, kwargs) (line 304)
        append_call_result_9025 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), append_9022, *[line_9023], **kwargs_9024)
        
        
        # ################# End of 'unreadline(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'unreadline' in the type store
        # Getting the type of 'stypy_return_type' (line 299)
        stypy_return_type_9026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9026)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'unreadline'
        return stypy_return_type_9026


# Assigning a type to the variable 'TextFile' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'TextFile', TextFile)

# Assigning a Dict to a Name (line 70):

# Obtaining an instance of the builtin type 'dict' (line 70)
dict_9027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 70)
# Adding element type (key, value) (line 70)
str_9028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 24), 'str', 'strip_comments')
int_9029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 42), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 22), dict_9027, (str_9028, int_9029))
# Adding element type (key, value) (line 70)
str_9030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 24), 'str', 'skip_blanks')
int_9031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 42), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 22), dict_9027, (str_9030, int_9031))
# Adding element type (key, value) (line 70)
str_9032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 24), 'str', 'lstrip_ws')
int_9033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 42), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 22), dict_9027, (str_9032, int_9033))
# Adding element type (key, value) (line 70)
str_9034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 24), 'str', 'rstrip_ws')
int_9035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 42), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 22), dict_9027, (str_9034, int_9035))
# Adding element type (key, value) (line 70)
str_9036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 24), 'str', 'join_lines')
int_9037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 42), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 22), dict_9027, (str_9036, int_9037))
# Adding element type (key, value) (line 70)
str_9038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 24), 'str', 'collapse_join')
int_9039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 42), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 22), dict_9027, (str_9038, int_9039))

# Getting the type of 'TextFile'
TextFile_9040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TextFile')
# Setting the type of the member 'default_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TextFile_9040, 'default_options', dict_9027)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
