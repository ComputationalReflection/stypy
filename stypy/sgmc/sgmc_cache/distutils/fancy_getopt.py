
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.fancy_getopt
2: 
3: Wrapper around the standard getopt module that provides the following
4: additional features:
5:   * short and long options are tied together
6:   * options have help strings, so fancy_getopt could potentially
7:     create a complete usage summary
8:   * options set attributes of a passed-in object
9: '''
10: 
11: __revision__ = "$Id$"
12: 
13: import sys
14: import string
15: import re
16: import getopt
17: from distutils.errors import DistutilsGetoptError, DistutilsArgError
18: 
19: # Much like command_re in distutils.core, this is close to but not quite
20: # the same as a Python NAME -- except, in the spirit of most GNU
21: # utilities, we use '-' in place of '_'.  (The spirit of LISP lives on!)
22: # The similarities to NAME are again not a coincidence...
23: longopt_pat = r'[a-zA-Z](?:[a-zA-Z0-9-]*)'
24: longopt_re = re.compile(r'^%s$' % longopt_pat)
25: 
26: # For recognizing "negative alias" options, eg. "quiet=!verbose"
27: neg_alias_re = re.compile("^(%s)=!(%s)$" % (longopt_pat, longopt_pat))
28: 
29: # This is used to translate long options to legitimate Python identifiers
30: # (for use as attributes of some object).
31: longopt_xlate = string.maketrans('-', '_')
32: 
33: class FancyGetopt:
34:     '''Wrapper around the standard 'getopt()' module that provides some
35:     handy extra functionality:
36:       * short and long options are tied together
37:       * options have help strings, and help text can be assembled
38:         from them
39:       * options set attributes of a passed-in object
40:       * boolean options can have "negative aliases" -- eg. if
41:         --quiet is the "negative alias" of --verbose, then "--quiet"
42:         on the command line sets 'verbose' to false
43:     '''
44: 
45:     def __init__ (self, option_table=None):
46: 
47:         # The option table is (currently) a list of tuples.  The
48:         # tuples may have 3 or four values:
49:         #   (long_option, short_option, help_string [, repeatable])
50:         # if an option takes an argument, its long_option should have '='
51:         # appended; short_option should just be a single character, no ':'
52:         # in any case.  If a long_option doesn't have a corresponding
53:         # short_option, short_option should be None.  All option tuples
54:         # must have long options.
55:         self.option_table = option_table
56: 
57:         # 'option_index' maps long option names to entries in the option
58:         # table (ie. those 3-tuples).
59:         self.option_index = {}
60:         if self.option_table:
61:             self._build_index()
62: 
63:         # 'alias' records (duh) alias options; {'foo': 'bar'} means
64:         # --foo is an alias for --bar
65:         self.alias = {}
66: 
67:         # 'negative_alias' keeps track of options that are the boolean
68:         # opposite of some other option
69:         self.negative_alias = {}
70: 
71:         # These keep track of the information in the option table.  We
72:         # don't actually populate these structures until we're ready to
73:         # parse the command-line, since the 'option_table' passed in here
74:         # isn't necessarily the final word.
75:         self.short_opts = []
76:         self.long_opts = []
77:         self.short2long = {}
78:         self.attr_name = {}
79:         self.takes_arg = {}
80: 
81:         # And 'option_order' is filled up in 'getopt()'; it records the
82:         # original order of options (and their values) on the command-line,
83:         # but expands short options, converts aliases, etc.
84:         self.option_order = []
85: 
86:     # __init__ ()
87: 
88: 
89:     def _build_index (self):
90:         self.option_index.clear()
91:         for option in self.option_table:
92:             self.option_index[option[0]] = option
93: 
94:     def set_option_table (self, option_table):
95:         self.option_table = option_table
96:         self._build_index()
97: 
98:     def add_option (self, long_option, short_option=None, help_string=None):
99:         if long_option in self.option_index:
100:             raise DistutilsGetoptError, \
101:                   "option conflict: already an option '%s'" % long_option
102:         else:
103:             option = (long_option, short_option, help_string)
104:             self.option_table.append(option)
105:             self.option_index[long_option] = option
106: 
107: 
108:     def has_option (self, long_option):
109:         '''Return true if the option table for this parser has an
110:         option with long name 'long_option'.'''
111:         return long_option in self.option_index
112: 
113:     def get_attr_name (self, long_option):
114:         '''Translate long option name 'long_option' to the form it
115:         has as an attribute of some object: ie., translate hyphens
116:         to underscores.'''
117:         return string.translate(long_option, longopt_xlate)
118: 
119: 
120:     def _check_alias_dict (self, aliases, what):
121:         assert isinstance(aliases, dict)
122:         for (alias, opt) in aliases.items():
123:             if alias not in self.option_index:
124:                 raise DistutilsGetoptError, \
125:                       ("invalid %s '%s': "
126:                        "option '%s' not defined") % (what, alias, alias)
127:             if opt not in self.option_index:
128:                 raise DistutilsGetoptError, \
129:                       ("invalid %s '%s': "
130:                        "aliased option '%s' not defined") % (what, alias, opt)
131: 
132:     def set_aliases (self, alias):
133:         '''Set the aliases for this option parser.'''
134:         self._check_alias_dict(alias, "alias")
135:         self.alias = alias
136: 
137:     def set_negative_aliases (self, negative_alias):
138:         '''Set the negative aliases for this option parser.
139:         'negative_alias' should be a dictionary mapping option names to
140:         option names, both the key and value must already be defined
141:         in the option table.'''
142:         self._check_alias_dict(negative_alias, "negative alias")
143:         self.negative_alias = negative_alias
144: 
145: 
146:     def _grok_option_table (self):
147:         '''Populate the various data structures that keep tabs on the
148:         option table.  Called by 'getopt()' before it can do anything
149:         worthwhile.
150:         '''
151:         self.long_opts = []
152:         self.short_opts = []
153:         self.short2long.clear()
154:         self.repeat = {}
155: 
156:         for option in self.option_table:
157:             if len(option) == 3:
158:                 long, short, help = option
159:                 repeat = 0
160:             elif len(option) == 4:
161:                 long, short, help, repeat = option
162:             else:
163:                 # the option table is part of the code, so simply
164:                 # assert that it is correct
165:                 raise ValueError, "invalid option tuple: %r" % (option,)
166: 
167:             # Type- and value-check the option names
168:             if not isinstance(long, str) or len(long) < 2:
169:                 raise DistutilsGetoptError, \
170:                       ("invalid long option '%s': "
171:                        "must be a string of length >= 2") % long
172: 
173:             if (not ((short is None) or
174:                      (isinstance(short, str) and len(short) == 1))):
175:                 raise DistutilsGetoptError, \
176:                       ("invalid short option '%s': "
177:                        "must a single character or None") % short
178: 
179:             self.repeat[long] = repeat
180:             self.long_opts.append(long)
181: 
182:             if long[-1] == '=':             # option takes an argument?
183:                 if short: short = short + ':'
184:                 long = long[0:-1]
185:                 self.takes_arg[long] = 1
186:             else:
187: 
188:                 # Is option is a "negative alias" for some other option (eg.
189:                 # "quiet" == "!verbose")?
190:                 alias_to = self.negative_alias.get(long)
191:                 if alias_to is not None:
192:                     if self.takes_arg[alias_to]:
193:                         raise DistutilsGetoptError, \
194:                               ("invalid negative alias '%s': "
195:                                "aliased option '%s' takes a value") % \
196:                                (long, alias_to)
197: 
198:                     self.long_opts[-1] = long # XXX redundant?!
199:                     self.takes_arg[long] = 0
200: 
201:                 else:
202:                     self.takes_arg[long] = 0
203: 
204:             # If this is an alias option, make sure its "takes arg" flag is
205:             # the same as the option it's aliased to.
206:             alias_to = self.alias.get(long)
207:             if alias_to is not None:
208:                 if self.takes_arg[long] != self.takes_arg[alias_to]:
209:                     raise DistutilsGetoptError, \
210:                           ("invalid alias '%s': inconsistent with "
211:                            "aliased option '%s' (one of them takes a value, "
212:                            "the other doesn't") % (long, alias_to)
213: 
214: 
215:             # Now enforce some bondage on the long option name, so we can
216:             # later translate it to an attribute name on some object.  Have
217:             # to do this a bit late to make sure we've removed any trailing
218:             # '='.
219:             if not longopt_re.match(long):
220:                 raise DistutilsGetoptError, \
221:                       ("invalid long option name '%s' " +
222:                        "(must be letters, numbers, hyphens only") % long
223: 
224:             self.attr_name[long] = self.get_attr_name(long)
225:             if short:
226:                 self.short_opts.append(short)
227:                 self.short2long[short[0]] = long
228: 
229:         # for option_table
230: 
231:     # _grok_option_table()
232: 
233: 
234:     def getopt (self, args=None, object=None):
235:         '''Parse command-line options in args. Store as attributes on object.
236: 
237:         If 'args' is None or not supplied, uses 'sys.argv[1:]'.  If
238:         'object' is None or not supplied, creates a new OptionDummy
239:         object, stores option values there, and returns a tuple (args,
240:         object).  If 'object' is supplied, it is modified in place and
241:         'getopt()' just returns 'args'; in both cases, the returned
242:         'args' is a modified copy of the passed-in 'args' list, which
243:         is left untouched.
244:         '''
245:         if args is None:
246:             args = sys.argv[1:]
247:         if object is None:
248:             object = OptionDummy()
249:             created_object = 1
250:         else:
251:             created_object = 0
252: 
253:         self._grok_option_table()
254: 
255:         short_opts = string.join(self.short_opts)
256:         try:
257:             opts, args = getopt.getopt(args, short_opts, self.long_opts)
258:         except getopt.error, msg:
259:             raise DistutilsArgError, msg
260: 
261:         for opt, val in opts:
262:             if len(opt) == 2 and opt[0] == '-': # it's a short option
263:                 opt = self.short2long[opt[1]]
264:             else:
265:                 assert len(opt) > 2 and opt[:2] == '--'
266:                 opt = opt[2:]
267: 
268:             alias = self.alias.get(opt)
269:             if alias:
270:                 opt = alias
271: 
272:             if not self.takes_arg[opt]:     # boolean option?
273:                 assert val == '', "boolean option can't have value"
274:                 alias = self.negative_alias.get(opt)
275:                 if alias:
276:                     opt = alias
277:                     val = 0
278:                 else:
279:                     val = 1
280: 
281:             attr = self.attr_name[opt]
282:             # The only repeating option at the moment is 'verbose'.
283:             # It has a negative option -q quiet, which should set verbose = 0.
284:             if val and self.repeat.get(attr) is not None:
285:                 val = getattr(object, attr, 0) + 1
286:             setattr(object, attr, val)
287:             self.option_order.append((opt, val))
288: 
289:         # for opts
290:         if created_object:
291:             return args, object
292:         else:
293:             return args
294: 
295:     # getopt()
296: 
297: 
298:     def get_option_order (self):
299:         '''Returns the list of (option, value) tuples processed by the
300:         previous run of 'getopt()'.  Raises RuntimeError if
301:         'getopt()' hasn't been called yet.
302:         '''
303:         if self.option_order is None:
304:             raise RuntimeError, "'getopt()' hasn't been called yet"
305:         else:
306:             return self.option_order
307: 
308: 
309:     def generate_help (self, header=None):
310:         '''Generate help text (a list of strings, one per suggested line of
311:         output) from the option table for this FancyGetopt object.
312:         '''
313:         # Blithely assume the option table is good: probably wouldn't call
314:         # 'generate_help()' unless you've already called 'getopt()'.
315: 
316:         # First pass: determine maximum length of long option names
317:         max_opt = 0
318:         for option in self.option_table:
319:             long = option[0]
320:             short = option[1]
321:             l = len(long)
322:             if long[-1] == '=':
323:                 l = l - 1
324:             if short is not None:
325:                 l = l + 5                   # " (-x)" where short == 'x'
326:             if l > max_opt:
327:                 max_opt = l
328: 
329:         opt_width = max_opt + 2 + 2 + 2     # room for indent + dashes + gutter
330: 
331:         # Typical help block looks like this:
332:         #   --foo       controls foonabulation
333:         # Help block for longest option looks like this:
334:         #   --flimflam  set the flim-flam level
335:         # and with wrapped text:
336:         #   --flimflam  set the flim-flam level (must be between
337:         #               0 and 100, except on Tuesdays)
338:         # Options with short names will have the short name shown (but
339:         # it doesn't contribute to max_opt):
340:         #   --foo (-f)  controls foonabulation
341:         # If adding the short option would make the left column too wide,
342:         # we push the explanation off to the next line
343:         #   --flimflam (-l)
344:         #               set the flim-flam level
345:         # Important parameters:
346:         #   - 2 spaces before option block start lines
347:         #   - 2 dashes for each long option name
348:         #   - min. 2 spaces between option and explanation (gutter)
349:         #   - 5 characters (incl. space) for short option name
350: 
351:         # Now generate lines of help text.  (If 80 columns were good enough
352:         # for Jesus, then 78 columns are good enough for me!)
353:         line_width = 78
354:         text_width = line_width - opt_width
355:         big_indent = ' ' * opt_width
356:         if header:
357:             lines = [header]
358:         else:
359:             lines = ['Option summary:']
360: 
361:         for option in self.option_table:
362:             long, short, help = option[:3]
363:             text = wrap_text(help, text_width)
364:             if long[-1] == '=':
365:                 long = long[0:-1]
366: 
367:             # Case 1: no short option at all (makes life easy)
368:             if short is None:
369:                 if text:
370:                     lines.append("  --%-*s  %s" % (max_opt, long, text[0]))
371:                 else:
372:                     lines.append("  --%-*s  " % (max_opt, long))
373: 
374:             # Case 2: we have a short option, so we have to include it
375:             # just after the long option
376:             else:
377:                 opt_names = "%s (-%s)" % (long, short)
378:                 if text:
379:                     lines.append("  --%-*s  %s" %
380:                                  (max_opt, opt_names, text[0]))
381:                 else:
382:                     lines.append("  --%-*s" % opt_names)
383: 
384:             for l in text[1:]:
385:                 lines.append(big_indent + l)
386: 
387:         # for self.option_table
388: 
389:         return lines
390: 
391:     # generate_help ()
392: 
393:     def print_help (self, header=None, file=None):
394:         if file is None:
395:             file = sys.stdout
396:         for line in self.generate_help(header):
397:             file.write(line + "\n")
398: 
399: # class FancyGetopt
400: 
401: 
402: def fancy_getopt (options, negative_opt, object, args):
403:     parser = FancyGetopt(options)
404:     parser.set_negative_aliases(negative_opt)
405:     return parser.getopt(args, object)
406: 
407: 
408: WS_TRANS = string.maketrans(string.whitespace, ' ' * len(string.whitespace))
409: 
410: def wrap_text (text, width):
411:     '''wrap_text(text : string, width : int) -> [string]
412: 
413:     Split 'text' into multiple lines of no more than 'width' characters
414:     each, and return the list of strings that results.
415:     '''
416: 
417:     if text is None:
418:         return []
419:     if len(text) <= width:
420:         return [text]
421: 
422:     text = string.expandtabs(text)
423:     text = string.translate(text, WS_TRANS)
424:     chunks = re.split(r'( +|-+)', text)
425:     chunks = filter(None, chunks)      # ' - ' results in empty strings
426:     lines = []
427: 
428:     while chunks:
429: 
430:         cur_line = []                   # list of chunks (to-be-joined)
431:         cur_len = 0                     # length of current line
432: 
433:         while chunks:
434:             l = len(chunks[0])
435:             if cur_len + l <= width:    # can squeeze (at least) this chunk in
436:                 cur_line.append(chunks[0])
437:                 del chunks[0]
438:                 cur_len = cur_len + l
439:             else:                       # this line is full
440:                 # drop last chunk if all space
441:                 if cur_line and cur_line[-1][0] == ' ':
442:                     del cur_line[-1]
443:                 break
444: 
445:         if chunks:                      # any chunks left to process?
446: 
447:             # if the current line is still empty, then we had a single
448:             # chunk that's too big too fit on a line -- so we break
449:             # down and break it up at the line width
450:             if cur_len == 0:
451:                 cur_line.append(chunks[0][0:width])
452:                 chunks[0] = chunks[0][width:]
453: 
454:             # all-whitespace chunks at the end of a line can be discarded
455:             # (and we know from the re.split above that if a chunk has
456:             # *any* whitespace, it is *all* whitespace)
457:             if chunks[0][0] == ' ':
458:                 del chunks[0]
459: 
460:         # and store this line in the list-of-all-lines -- as a single
461:         # string, of course!
462:         lines.append(string.join(cur_line, ''))
463: 
464:     # while chunks
465: 
466:     return lines
467: 
468: 
469: def translate_longopt(opt):
470:     '''Convert a long option name to a valid Python identifier by
471:     changing "-" to "_".
472:     '''
473:     return string.translate(opt, longopt_xlate)
474: 
475: 
476: class OptionDummy:
477:     '''Dummy class just used as a place to hold command-line option
478:     values as instance attributes.'''
479: 
480:     def __init__ (self, options=[]):
481:         '''Create a new OptionDummy instance.  The attributes listed in
482:         'options' will be initialized to None.'''
483:         for opt in options:
484:             setattr(self, opt, None)
485: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, (-1)), 'str', 'distutils.fancy_getopt\n\nWrapper around the standard getopt module that provides the following\nadditional features:\n  * short and long options are tied together\n  * options have help strings, so fancy_getopt could potentially\n    create a complete usage summary\n  * options set attributes of a passed-in object\n')

# Assigning a Str to a Name (line 11):

# Assigning a Str to a Name (line 11):
str_474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), '__revision__', str_474)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import sys' statement (line 13)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import string' statement (line 14)
import string

import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'string', string, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'import re' statement (line 15)
import re

import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import getopt' statement (line 16)
import getopt

import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'getopt', getopt, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from distutils.errors import DistutilsGetoptError, DistutilsArgError' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_475 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.errors')

if (type(import_475) is not StypyTypeError):

    if (import_475 != 'pyd_module'):
        __import__(import_475)
        sys_modules_476 = sys.modules[import_475]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.errors', sys_modules_476.module_type_store, module_type_store, ['DistutilsGetoptError', 'DistutilsArgError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_476, sys_modules_476.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsGetoptError, DistutilsArgError

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.errors', None, module_type_store, ['DistutilsGetoptError', 'DistutilsArgError'], [DistutilsGetoptError, DistutilsArgError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.errors', import_475)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')


# Assigning a Str to a Name (line 23):

# Assigning a Str to a Name (line 23):
str_477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 14), 'str', '[a-zA-Z](?:[a-zA-Z0-9-]*)')
# Assigning a type to the variable 'longopt_pat' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'longopt_pat', str_477)

# Assigning a Call to a Name (line 24):

# Assigning a Call to a Name (line 24):

# Call to compile(...): (line 24)
# Processing the call arguments (line 24)
str_480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 24), 'str', '^%s$')
# Getting the type of 'longopt_pat' (line 24)
longopt_pat_481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 34), 'longopt_pat', False)
# Applying the binary operator '%' (line 24)
result_mod_482 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 24), '%', str_480, longopt_pat_481)

# Processing the call keyword arguments (line 24)
kwargs_483 = {}
# Getting the type of 're' (line 24)
re_478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 13), 're', False)
# Obtaining the member 'compile' of a type (line 24)
compile_479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 13), re_478, 'compile')
# Calling compile(args, kwargs) (line 24)
compile_call_result_484 = invoke(stypy.reporting.localization.Localization(__file__, 24, 13), compile_479, *[result_mod_482], **kwargs_483)

# Assigning a type to the variable 'longopt_re' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'longopt_re', compile_call_result_484)

# Assigning a Call to a Name (line 27):

# Assigning a Call to a Name (line 27):

# Call to compile(...): (line 27)
# Processing the call arguments (line 27)
str_487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 26), 'str', '^(%s)=!(%s)$')

# Obtaining an instance of the builtin type 'tuple' (line 27)
tuple_488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 44), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 27)
# Adding element type (line 27)
# Getting the type of 'longopt_pat' (line 27)
longopt_pat_489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 44), 'longopt_pat', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 44), tuple_488, longopt_pat_489)
# Adding element type (line 27)
# Getting the type of 'longopt_pat' (line 27)
longopt_pat_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 57), 'longopt_pat', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 44), tuple_488, longopt_pat_490)

# Applying the binary operator '%' (line 27)
result_mod_491 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 26), '%', str_487, tuple_488)

# Processing the call keyword arguments (line 27)
kwargs_492 = {}
# Getting the type of 're' (line 27)
re_485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 're', False)
# Obtaining the member 'compile' of a type (line 27)
compile_486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 15), re_485, 'compile')
# Calling compile(args, kwargs) (line 27)
compile_call_result_493 = invoke(stypy.reporting.localization.Localization(__file__, 27, 15), compile_486, *[result_mod_491], **kwargs_492)

# Assigning a type to the variable 'neg_alias_re' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'neg_alias_re', compile_call_result_493)

# Assigning a Call to a Name (line 31):

# Assigning a Call to a Name (line 31):

# Call to maketrans(...): (line 31)
# Processing the call arguments (line 31)
str_496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 33), 'str', '-')
str_497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 38), 'str', '_')
# Processing the call keyword arguments (line 31)
kwargs_498 = {}
# Getting the type of 'string' (line 31)
string_494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'string', False)
# Obtaining the member 'maketrans' of a type (line 31)
maketrans_495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 16), string_494, 'maketrans')
# Calling maketrans(args, kwargs) (line 31)
maketrans_call_result_499 = invoke(stypy.reporting.localization.Localization(__file__, 31, 16), maketrans_495, *[str_496, str_497], **kwargs_498)

# Assigning a type to the variable 'longopt_xlate' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'longopt_xlate', maketrans_call_result_499)
# Declaration of the 'FancyGetopt' class

class FancyGetopt:
    str_500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'str', 'Wrapper around the standard \'getopt()\' module that provides some\n    handy extra functionality:\n      * short and long options are tied together\n      * options have help strings, and help text can be assembled\n        from them\n      * options set attributes of a passed-in object\n      * boolean options can have "negative aliases" -- eg. if\n        --quiet is the "negative alias" of --verbose, then "--quiet"\n        on the command line sets \'verbose\' to false\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 45)
        None_501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 37), 'None')
        defaults = [None_501]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FancyGetopt.__init__', ['option_table'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['option_table'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 55):
        
        # Assigning a Name to a Attribute (line 55):
        # Getting the type of 'option_table' (line 55)
        option_table_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 28), 'option_table')
        # Getting the type of 'self' (line 55)
        self_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'self')
        # Setting the type of the member 'option_table' of a type (line 55)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), self_503, 'option_table', option_table_502)
        
        # Assigning a Dict to a Attribute (line 59):
        
        # Assigning a Dict to a Attribute (line 59):
        
        # Obtaining an instance of the builtin type 'dict' (line 59)
        dict_504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 59)
        
        # Getting the type of 'self' (line 59)
        self_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self')
        # Setting the type of the member 'option_index' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_505, 'option_index', dict_504)
        
        # Getting the type of 'self' (line 60)
        self_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'self')
        # Obtaining the member 'option_table' of a type (line 60)
        option_table_507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 11), self_506, 'option_table')
        # Testing the type of an if condition (line 60)
        if_condition_508 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 8), option_table_507)
        # Assigning a type to the variable 'if_condition_508' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'if_condition_508', if_condition_508)
        # SSA begins for if statement (line 60)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _build_index(...): (line 61)
        # Processing the call keyword arguments (line 61)
        kwargs_511 = {}
        # Getting the type of 'self' (line 61)
        self_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'self', False)
        # Obtaining the member '_build_index' of a type (line 61)
        _build_index_510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 12), self_509, '_build_index')
        # Calling _build_index(args, kwargs) (line 61)
        _build_index_call_result_512 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), _build_index_510, *[], **kwargs_511)
        
        # SSA join for if statement (line 60)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Dict to a Attribute (line 65):
        
        # Assigning a Dict to a Attribute (line 65):
        
        # Obtaining an instance of the builtin type 'dict' (line 65)
        dict_513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 21), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 65)
        
        # Getting the type of 'self' (line 65)
        self_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self')
        # Setting the type of the member 'alias' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), self_514, 'alias', dict_513)
        
        # Assigning a Dict to a Attribute (line 69):
        
        # Assigning a Dict to a Attribute (line 69):
        
        # Obtaining an instance of the builtin type 'dict' (line 69)
        dict_515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 30), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 69)
        
        # Getting the type of 'self' (line 69)
        self_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'self')
        # Setting the type of the member 'negative_alias' of a type (line 69)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), self_516, 'negative_alias', dict_515)
        
        # Assigning a List to a Attribute (line 75):
        
        # Assigning a List to a Attribute (line 75):
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        
        # Getting the type of 'self' (line 75)
        self_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'self')
        # Setting the type of the member 'short_opts' of a type (line 75)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), self_518, 'short_opts', list_517)
        
        # Assigning a List to a Attribute (line 76):
        
        # Assigning a List to a Attribute (line 76):
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        
        # Getting the type of 'self' (line 76)
        self_520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'self')
        # Setting the type of the member 'long_opts' of a type (line 76)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), self_520, 'long_opts', list_519)
        
        # Assigning a Dict to a Attribute (line 77):
        
        # Assigning a Dict to a Attribute (line 77):
        
        # Obtaining an instance of the builtin type 'dict' (line 77)
        dict_521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 26), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 77)
        
        # Getting the type of 'self' (line 77)
        self_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'self')
        # Setting the type of the member 'short2long' of a type (line 77)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), self_522, 'short2long', dict_521)
        
        # Assigning a Dict to a Attribute (line 78):
        
        # Assigning a Dict to a Attribute (line 78):
        
        # Obtaining an instance of the builtin type 'dict' (line 78)
        dict_523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 25), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 78)
        
        # Getting the type of 'self' (line 78)
        self_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self')
        # Setting the type of the member 'attr_name' of a type (line 78)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_524, 'attr_name', dict_523)
        
        # Assigning a Dict to a Attribute (line 79):
        
        # Assigning a Dict to a Attribute (line 79):
        
        # Obtaining an instance of the builtin type 'dict' (line 79)
        dict_525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 25), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 79)
        
        # Getting the type of 'self' (line 79)
        self_526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'self')
        # Setting the type of the member 'takes_arg' of a type (line 79)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), self_526, 'takes_arg', dict_525)
        
        # Assigning a List to a Attribute (line 84):
        
        # Assigning a List to a Attribute (line 84):
        
        # Obtaining an instance of the builtin type 'list' (line 84)
        list_527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 84)
        
        # Getting the type of 'self' (line 84)
        self_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'self')
        # Setting the type of the member 'option_order' of a type (line 84)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), self_528, 'option_order', list_527)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _build_index(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_build_index'
        module_type_store = module_type_store.open_function_context('_build_index', 89, 4, False)
        # Assigning a type to the variable 'self' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FancyGetopt._build_index.__dict__.__setitem__('stypy_localization', localization)
        FancyGetopt._build_index.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FancyGetopt._build_index.__dict__.__setitem__('stypy_type_store', module_type_store)
        FancyGetopt._build_index.__dict__.__setitem__('stypy_function_name', 'FancyGetopt._build_index')
        FancyGetopt._build_index.__dict__.__setitem__('stypy_param_names_list', [])
        FancyGetopt._build_index.__dict__.__setitem__('stypy_varargs_param_name', None)
        FancyGetopt._build_index.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FancyGetopt._build_index.__dict__.__setitem__('stypy_call_defaults', defaults)
        FancyGetopt._build_index.__dict__.__setitem__('stypy_call_varargs', varargs)
        FancyGetopt._build_index.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FancyGetopt._build_index.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FancyGetopt._build_index', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_build_index', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_build_index(...)' code ##################

        
        # Call to clear(...): (line 90)
        # Processing the call keyword arguments (line 90)
        kwargs_532 = {}
        # Getting the type of 'self' (line 90)
        self_529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self', False)
        # Obtaining the member 'option_index' of a type (line 90)
        option_index_530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_529, 'option_index')
        # Obtaining the member 'clear' of a type (line 90)
        clear_531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), option_index_530, 'clear')
        # Calling clear(args, kwargs) (line 90)
        clear_call_result_533 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), clear_531, *[], **kwargs_532)
        
        
        # Getting the type of 'self' (line 91)
        self_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 22), 'self')
        # Obtaining the member 'option_table' of a type (line 91)
        option_table_535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 22), self_534, 'option_table')
        # Testing the type of a for loop iterable (line 91)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 91, 8), option_table_535)
        # Getting the type of the for loop variable (line 91)
        for_loop_var_536 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 91, 8), option_table_535)
        # Assigning a type to the variable 'option' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'option', for_loop_var_536)
        # SSA begins for a for statement (line 91)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Subscript (line 92):
        
        # Assigning a Name to a Subscript (line 92):
        # Getting the type of 'option' (line 92)
        option_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 43), 'option')
        # Getting the type of 'self' (line 92)
        self_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'self')
        # Obtaining the member 'option_index' of a type (line 92)
        option_index_539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), self_538, 'option_index')
        
        # Obtaining the type of the subscript
        int_540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 37), 'int')
        # Getting the type of 'option' (line 92)
        option_541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 30), 'option')
        # Obtaining the member '__getitem__' of a type (line 92)
        getitem___542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 30), option_541, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 92)
        subscript_call_result_543 = invoke(stypy.reporting.localization.Localization(__file__, 92, 30), getitem___542, int_540)
        
        # Storing an element on a container (line 92)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 12), option_index_539, (subscript_call_result_543, option_537))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_build_index(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_build_index' in the type store
        # Getting the type of 'stypy_return_type' (line 89)
        stypy_return_type_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_544)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_build_index'
        return stypy_return_type_544


    @norecursion
    def set_option_table(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_option_table'
        module_type_store = module_type_store.open_function_context('set_option_table', 94, 4, False)
        # Assigning a type to the variable 'self' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FancyGetopt.set_option_table.__dict__.__setitem__('stypy_localization', localization)
        FancyGetopt.set_option_table.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FancyGetopt.set_option_table.__dict__.__setitem__('stypy_type_store', module_type_store)
        FancyGetopt.set_option_table.__dict__.__setitem__('stypy_function_name', 'FancyGetopt.set_option_table')
        FancyGetopt.set_option_table.__dict__.__setitem__('stypy_param_names_list', ['option_table'])
        FancyGetopt.set_option_table.__dict__.__setitem__('stypy_varargs_param_name', None)
        FancyGetopt.set_option_table.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FancyGetopt.set_option_table.__dict__.__setitem__('stypy_call_defaults', defaults)
        FancyGetopt.set_option_table.__dict__.__setitem__('stypy_call_varargs', varargs)
        FancyGetopt.set_option_table.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FancyGetopt.set_option_table.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FancyGetopt.set_option_table', ['option_table'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_option_table', localization, ['option_table'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_option_table(...)' code ##################

        
        # Assigning a Name to a Attribute (line 95):
        
        # Assigning a Name to a Attribute (line 95):
        # Getting the type of 'option_table' (line 95)
        option_table_545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 28), 'option_table')
        # Getting the type of 'self' (line 95)
        self_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'self')
        # Setting the type of the member 'option_table' of a type (line 95)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), self_546, 'option_table', option_table_545)
        
        # Call to _build_index(...): (line 96)
        # Processing the call keyword arguments (line 96)
        kwargs_549 = {}
        # Getting the type of 'self' (line 96)
        self_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'self', False)
        # Obtaining the member '_build_index' of a type (line 96)
        _build_index_548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), self_547, '_build_index')
        # Calling _build_index(args, kwargs) (line 96)
        _build_index_call_result_550 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), _build_index_548, *[], **kwargs_549)
        
        
        # ################# End of 'set_option_table(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_option_table' in the type store
        # Getting the type of 'stypy_return_type' (line 94)
        stypy_return_type_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_551)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_option_table'
        return stypy_return_type_551


    @norecursion
    def add_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 98)
        None_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 52), 'None')
        # Getting the type of 'None' (line 98)
        None_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 70), 'None')
        defaults = [None_552, None_553]
        # Create a new context for function 'add_option'
        module_type_store = module_type_store.open_function_context('add_option', 98, 4, False)
        # Assigning a type to the variable 'self' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FancyGetopt.add_option.__dict__.__setitem__('stypy_localization', localization)
        FancyGetopt.add_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FancyGetopt.add_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        FancyGetopt.add_option.__dict__.__setitem__('stypy_function_name', 'FancyGetopt.add_option')
        FancyGetopt.add_option.__dict__.__setitem__('stypy_param_names_list', ['long_option', 'short_option', 'help_string'])
        FancyGetopt.add_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        FancyGetopt.add_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FancyGetopt.add_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        FancyGetopt.add_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        FancyGetopt.add_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FancyGetopt.add_option.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FancyGetopt.add_option', ['long_option', 'short_option', 'help_string'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_option', localization, ['long_option', 'short_option', 'help_string'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_option(...)' code ##################

        
        
        # Getting the type of 'long_option' (line 99)
        long_option_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 11), 'long_option')
        # Getting the type of 'self' (line 99)
        self_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 26), 'self')
        # Obtaining the member 'option_index' of a type (line 99)
        option_index_556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 26), self_555, 'option_index')
        # Applying the binary operator 'in' (line 99)
        result_contains_557 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 11), 'in', long_option_554, option_index_556)
        
        # Testing the type of an if condition (line 99)
        if_condition_558 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 8), result_contains_557)
        # Assigning a type to the variable 'if_condition_558' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'if_condition_558', if_condition_558)
        # SSA begins for if statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsGetoptError' (line 100)
        DistutilsGetoptError_559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 18), 'DistutilsGetoptError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 100, 12), DistutilsGetoptError_559, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 99)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Tuple to a Name (line 103):
        
        # Assigning a Tuple to a Name (line 103):
        
        # Obtaining an instance of the builtin type 'tuple' (line 103)
        tuple_560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 103)
        # Adding element type (line 103)
        # Getting the type of 'long_option' (line 103)
        long_option_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 22), 'long_option')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 22), tuple_560, long_option_561)
        # Adding element type (line 103)
        # Getting the type of 'short_option' (line 103)
        short_option_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 35), 'short_option')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 22), tuple_560, short_option_562)
        # Adding element type (line 103)
        # Getting the type of 'help_string' (line 103)
        help_string_563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 49), 'help_string')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 22), tuple_560, help_string_563)
        
        # Assigning a type to the variable 'option' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'option', tuple_560)
        
        # Call to append(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'option' (line 104)
        option_567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 37), 'option', False)
        # Processing the call keyword arguments (line 104)
        kwargs_568 = {}
        # Getting the type of 'self' (line 104)
        self_564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'self', False)
        # Obtaining the member 'option_table' of a type (line 104)
        option_table_565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 12), self_564, 'option_table')
        # Obtaining the member 'append' of a type (line 104)
        append_566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 12), option_table_565, 'append')
        # Calling append(args, kwargs) (line 104)
        append_call_result_569 = invoke(stypy.reporting.localization.Localization(__file__, 104, 12), append_566, *[option_567], **kwargs_568)
        
        
        # Assigning a Name to a Subscript (line 105):
        
        # Assigning a Name to a Subscript (line 105):
        # Getting the type of 'option' (line 105)
        option_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 45), 'option')
        # Getting the type of 'self' (line 105)
        self_571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'self')
        # Obtaining the member 'option_index' of a type (line 105)
        option_index_572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), self_571, 'option_index')
        # Getting the type of 'long_option' (line 105)
        long_option_573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 30), 'long_option')
        # Storing an element on a container (line 105)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), option_index_572, (long_option_573, option_570))
        # SSA join for if statement (line 99)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'add_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_option' in the type store
        # Getting the type of 'stypy_return_type' (line 98)
        stypy_return_type_574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_574)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_option'
        return stypy_return_type_574


    @norecursion
    def has_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_option'
        module_type_store = module_type_store.open_function_context('has_option', 108, 4, False)
        # Assigning a type to the variable 'self' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FancyGetopt.has_option.__dict__.__setitem__('stypy_localization', localization)
        FancyGetopt.has_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FancyGetopt.has_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        FancyGetopt.has_option.__dict__.__setitem__('stypy_function_name', 'FancyGetopt.has_option')
        FancyGetopt.has_option.__dict__.__setitem__('stypy_param_names_list', ['long_option'])
        FancyGetopt.has_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        FancyGetopt.has_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FancyGetopt.has_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        FancyGetopt.has_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        FancyGetopt.has_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FancyGetopt.has_option.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FancyGetopt.has_option', ['long_option'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_option', localization, ['long_option'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_option(...)' code ##################

        str_575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, (-1)), 'str', "Return true if the option table for this parser has an\n        option with long name 'long_option'.")
        
        # Getting the type of 'long_option' (line 111)
        long_option_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'long_option')
        # Getting the type of 'self' (line 111)
        self_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 30), 'self')
        # Obtaining the member 'option_index' of a type (line 111)
        option_index_578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 30), self_577, 'option_index')
        # Applying the binary operator 'in' (line 111)
        result_contains_579 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 15), 'in', long_option_576, option_index_578)
        
        # Assigning a type to the variable 'stypy_return_type' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'stypy_return_type', result_contains_579)
        
        # ################# End of 'has_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_option' in the type store
        # Getting the type of 'stypy_return_type' (line 108)
        stypy_return_type_580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_580)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_option'
        return stypy_return_type_580


    @norecursion
    def get_attr_name(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_attr_name'
        module_type_store = module_type_store.open_function_context('get_attr_name', 113, 4, False)
        # Assigning a type to the variable 'self' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FancyGetopt.get_attr_name.__dict__.__setitem__('stypy_localization', localization)
        FancyGetopt.get_attr_name.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FancyGetopt.get_attr_name.__dict__.__setitem__('stypy_type_store', module_type_store)
        FancyGetopt.get_attr_name.__dict__.__setitem__('stypy_function_name', 'FancyGetopt.get_attr_name')
        FancyGetopt.get_attr_name.__dict__.__setitem__('stypy_param_names_list', ['long_option'])
        FancyGetopt.get_attr_name.__dict__.__setitem__('stypy_varargs_param_name', None)
        FancyGetopt.get_attr_name.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FancyGetopt.get_attr_name.__dict__.__setitem__('stypy_call_defaults', defaults)
        FancyGetopt.get_attr_name.__dict__.__setitem__('stypy_call_varargs', varargs)
        FancyGetopt.get_attr_name.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FancyGetopt.get_attr_name.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FancyGetopt.get_attr_name', ['long_option'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_attr_name', localization, ['long_option'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_attr_name(...)' code ##################

        str_581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, (-1)), 'str', "Translate long option name 'long_option' to the form it\n        has as an attribute of some object: ie., translate hyphens\n        to underscores.")
        
        # Call to translate(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'long_option' (line 117)
        long_option_584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 32), 'long_option', False)
        # Getting the type of 'longopt_xlate' (line 117)
        longopt_xlate_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 45), 'longopt_xlate', False)
        # Processing the call keyword arguments (line 117)
        kwargs_586 = {}
        # Getting the type of 'string' (line 117)
        string_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'string', False)
        # Obtaining the member 'translate' of a type (line 117)
        translate_583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 15), string_582, 'translate')
        # Calling translate(args, kwargs) (line 117)
        translate_call_result_587 = invoke(stypy.reporting.localization.Localization(__file__, 117, 15), translate_583, *[long_option_584, longopt_xlate_585], **kwargs_586)
        
        # Assigning a type to the variable 'stypy_return_type' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'stypy_return_type', translate_call_result_587)
        
        # ################# End of 'get_attr_name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_attr_name' in the type store
        # Getting the type of 'stypy_return_type' (line 113)
        stypy_return_type_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_588)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_attr_name'
        return stypy_return_type_588


    @norecursion
    def _check_alias_dict(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check_alias_dict'
        module_type_store = module_type_store.open_function_context('_check_alias_dict', 120, 4, False)
        # Assigning a type to the variable 'self' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FancyGetopt._check_alias_dict.__dict__.__setitem__('stypy_localization', localization)
        FancyGetopt._check_alias_dict.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FancyGetopt._check_alias_dict.__dict__.__setitem__('stypy_type_store', module_type_store)
        FancyGetopt._check_alias_dict.__dict__.__setitem__('stypy_function_name', 'FancyGetopt._check_alias_dict')
        FancyGetopt._check_alias_dict.__dict__.__setitem__('stypy_param_names_list', ['aliases', 'what'])
        FancyGetopt._check_alias_dict.__dict__.__setitem__('stypy_varargs_param_name', None)
        FancyGetopt._check_alias_dict.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FancyGetopt._check_alias_dict.__dict__.__setitem__('stypy_call_defaults', defaults)
        FancyGetopt._check_alias_dict.__dict__.__setitem__('stypy_call_varargs', varargs)
        FancyGetopt._check_alias_dict.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FancyGetopt._check_alias_dict.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FancyGetopt._check_alias_dict', ['aliases', 'what'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_alias_dict', localization, ['aliases', 'what'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_alias_dict(...)' code ##################

        # Evaluating assert statement condition
        
        # Call to isinstance(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'aliases' (line 121)
        aliases_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 26), 'aliases', False)
        # Getting the type of 'dict' (line 121)
        dict_591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 35), 'dict', False)
        # Processing the call keyword arguments (line 121)
        kwargs_592 = {}
        # Getting the type of 'isinstance' (line 121)
        isinstance_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 121)
        isinstance_call_result_593 = invoke(stypy.reporting.localization.Localization(__file__, 121, 15), isinstance_589, *[aliases_590, dict_591], **kwargs_592)
        
        
        
        # Call to items(...): (line 122)
        # Processing the call keyword arguments (line 122)
        kwargs_596 = {}
        # Getting the type of 'aliases' (line 122)
        aliases_594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 28), 'aliases', False)
        # Obtaining the member 'items' of a type (line 122)
        items_595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 28), aliases_594, 'items')
        # Calling items(args, kwargs) (line 122)
        items_call_result_597 = invoke(stypy.reporting.localization.Localization(__file__, 122, 28), items_595, *[], **kwargs_596)
        
        # Testing the type of a for loop iterable (line 122)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 122, 8), items_call_result_597)
        # Getting the type of the for loop variable (line 122)
        for_loop_var_598 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 122, 8), items_call_result_597)
        # Assigning a type to the variable 'alias' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'alias', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 8), for_loop_var_598))
        # Assigning a type to the variable 'opt' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'opt', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 8), for_loop_var_598))
        # SSA begins for a for statement (line 122)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'alias' (line 123)
        alias_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 'alias')
        # Getting the type of 'self' (line 123)
        self_600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 28), 'self')
        # Obtaining the member 'option_index' of a type (line 123)
        option_index_601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 28), self_600, 'option_index')
        # Applying the binary operator 'notin' (line 123)
        result_contains_602 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 15), 'notin', alias_599, option_index_601)
        
        # Testing the type of an if condition (line 123)
        if_condition_603 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 12), result_contains_602)
        # Assigning a type to the variable 'if_condition_603' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'if_condition_603', if_condition_603)
        # SSA begins for if statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsGetoptError' (line 124)
        DistutilsGetoptError_604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 22), 'DistutilsGetoptError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 124, 16), DistutilsGetoptError_604, 'raise parameter', BaseException)
        # SSA join for if statement (line 123)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'opt' (line 127)
        opt_605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'opt')
        # Getting the type of 'self' (line 127)
        self_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 26), 'self')
        # Obtaining the member 'option_index' of a type (line 127)
        option_index_607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 26), self_606, 'option_index')
        # Applying the binary operator 'notin' (line 127)
        result_contains_608 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 15), 'notin', opt_605, option_index_607)
        
        # Testing the type of an if condition (line 127)
        if_condition_609 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 12), result_contains_608)
        # Assigning a type to the variable 'if_condition_609' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'if_condition_609', if_condition_609)
        # SSA begins for if statement (line 127)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsGetoptError' (line 128)
        DistutilsGetoptError_610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 22), 'DistutilsGetoptError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 128, 16), DistutilsGetoptError_610, 'raise parameter', BaseException)
        # SSA join for if statement (line 127)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_check_alias_dict(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_alias_dict' in the type store
        # Getting the type of 'stypy_return_type' (line 120)
        stypy_return_type_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_611)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_alias_dict'
        return stypy_return_type_611


    @norecursion
    def set_aliases(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_aliases'
        module_type_store = module_type_store.open_function_context('set_aliases', 132, 4, False)
        # Assigning a type to the variable 'self' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FancyGetopt.set_aliases.__dict__.__setitem__('stypy_localization', localization)
        FancyGetopt.set_aliases.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FancyGetopt.set_aliases.__dict__.__setitem__('stypy_type_store', module_type_store)
        FancyGetopt.set_aliases.__dict__.__setitem__('stypy_function_name', 'FancyGetopt.set_aliases')
        FancyGetopt.set_aliases.__dict__.__setitem__('stypy_param_names_list', ['alias'])
        FancyGetopt.set_aliases.__dict__.__setitem__('stypy_varargs_param_name', None)
        FancyGetopt.set_aliases.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FancyGetopt.set_aliases.__dict__.__setitem__('stypy_call_defaults', defaults)
        FancyGetopt.set_aliases.__dict__.__setitem__('stypy_call_varargs', varargs)
        FancyGetopt.set_aliases.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FancyGetopt.set_aliases.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FancyGetopt.set_aliases', ['alias'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_aliases', localization, ['alias'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_aliases(...)' code ##################

        str_612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 8), 'str', 'Set the aliases for this option parser.')
        
        # Call to _check_alias_dict(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'alias' (line 134)
        alias_615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 31), 'alias', False)
        str_616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 38), 'str', 'alias')
        # Processing the call keyword arguments (line 134)
        kwargs_617 = {}
        # Getting the type of 'self' (line 134)
        self_613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'self', False)
        # Obtaining the member '_check_alias_dict' of a type (line 134)
        _check_alias_dict_614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), self_613, '_check_alias_dict')
        # Calling _check_alias_dict(args, kwargs) (line 134)
        _check_alias_dict_call_result_618 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), _check_alias_dict_614, *[alias_615, str_616], **kwargs_617)
        
        
        # Assigning a Name to a Attribute (line 135):
        
        # Assigning a Name to a Attribute (line 135):
        # Getting the type of 'alias' (line 135)
        alias_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 21), 'alias')
        # Getting the type of 'self' (line 135)
        self_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'self')
        # Setting the type of the member 'alias' of a type (line 135)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), self_620, 'alias', alias_619)
        
        # ################# End of 'set_aliases(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_aliases' in the type store
        # Getting the type of 'stypy_return_type' (line 132)
        stypy_return_type_621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_621)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_aliases'
        return stypy_return_type_621


    @norecursion
    def set_negative_aliases(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_negative_aliases'
        module_type_store = module_type_store.open_function_context('set_negative_aliases', 137, 4, False)
        # Assigning a type to the variable 'self' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FancyGetopt.set_negative_aliases.__dict__.__setitem__('stypy_localization', localization)
        FancyGetopt.set_negative_aliases.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FancyGetopt.set_negative_aliases.__dict__.__setitem__('stypy_type_store', module_type_store)
        FancyGetopt.set_negative_aliases.__dict__.__setitem__('stypy_function_name', 'FancyGetopt.set_negative_aliases')
        FancyGetopt.set_negative_aliases.__dict__.__setitem__('stypy_param_names_list', ['negative_alias'])
        FancyGetopt.set_negative_aliases.__dict__.__setitem__('stypy_varargs_param_name', None)
        FancyGetopt.set_negative_aliases.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FancyGetopt.set_negative_aliases.__dict__.__setitem__('stypy_call_defaults', defaults)
        FancyGetopt.set_negative_aliases.__dict__.__setitem__('stypy_call_varargs', varargs)
        FancyGetopt.set_negative_aliases.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FancyGetopt.set_negative_aliases.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FancyGetopt.set_negative_aliases', ['negative_alias'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_negative_aliases', localization, ['negative_alias'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_negative_aliases(...)' code ##################

        str_622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, (-1)), 'str', "Set the negative aliases for this option parser.\n        'negative_alias' should be a dictionary mapping option names to\n        option names, both the key and value must already be defined\n        in the option table.")
        
        # Call to _check_alias_dict(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'negative_alias' (line 142)
        negative_alias_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 31), 'negative_alias', False)
        str_626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 47), 'str', 'negative alias')
        # Processing the call keyword arguments (line 142)
        kwargs_627 = {}
        # Getting the type of 'self' (line 142)
        self_623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'self', False)
        # Obtaining the member '_check_alias_dict' of a type (line 142)
        _check_alias_dict_624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), self_623, '_check_alias_dict')
        # Calling _check_alias_dict(args, kwargs) (line 142)
        _check_alias_dict_call_result_628 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), _check_alias_dict_624, *[negative_alias_625, str_626], **kwargs_627)
        
        
        # Assigning a Name to a Attribute (line 143):
        
        # Assigning a Name to a Attribute (line 143):
        # Getting the type of 'negative_alias' (line 143)
        negative_alias_629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 30), 'negative_alias')
        # Getting the type of 'self' (line 143)
        self_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'self')
        # Setting the type of the member 'negative_alias' of a type (line 143)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), self_630, 'negative_alias', negative_alias_629)
        
        # ################# End of 'set_negative_aliases(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_negative_aliases' in the type store
        # Getting the type of 'stypy_return_type' (line 137)
        stypy_return_type_631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_631)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_negative_aliases'
        return stypy_return_type_631


    @norecursion
    def _grok_option_table(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_grok_option_table'
        module_type_store = module_type_store.open_function_context('_grok_option_table', 146, 4, False)
        # Assigning a type to the variable 'self' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FancyGetopt._grok_option_table.__dict__.__setitem__('stypy_localization', localization)
        FancyGetopt._grok_option_table.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FancyGetopt._grok_option_table.__dict__.__setitem__('stypy_type_store', module_type_store)
        FancyGetopt._grok_option_table.__dict__.__setitem__('stypy_function_name', 'FancyGetopt._grok_option_table')
        FancyGetopt._grok_option_table.__dict__.__setitem__('stypy_param_names_list', [])
        FancyGetopt._grok_option_table.__dict__.__setitem__('stypy_varargs_param_name', None)
        FancyGetopt._grok_option_table.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FancyGetopt._grok_option_table.__dict__.__setitem__('stypy_call_defaults', defaults)
        FancyGetopt._grok_option_table.__dict__.__setitem__('stypy_call_varargs', varargs)
        FancyGetopt._grok_option_table.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FancyGetopt._grok_option_table.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FancyGetopt._grok_option_table', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_grok_option_table', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_grok_option_table(...)' code ##################

        str_632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, (-1)), 'str', "Populate the various data structures that keep tabs on the\n        option table.  Called by 'getopt()' before it can do anything\n        worthwhile.\n        ")
        
        # Assigning a List to a Attribute (line 151):
        
        # Assigning a List to a Attribute (line 151):
        
        # Obtaining an instance of the builtin type 'list' (line 151)
        list_633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 151)
        
        # Getting the type of 'self' (line 151)
        self_634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'self')
        # Setting the type of the member 'long_opts' of a type (line 151)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), self_634, 'long_opts', list_633)
        
        # Assigning a List to a Attribute (line 152):
        
        # Assigning a List to a Attribute (line 152):
        
        # Obtaining an instance of the builtin type 'list' (line 152)
        list_635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 152)
        
        # Getting the type of 'self' (line 152)
        self_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'self')
        # Setting the type of the member 'short_opts' of a type (line 152)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 8), self_636, 'short_opts', list_635)
        
        # Call to clear(...): (line 153)
        # Processing the call keyword arguments (line 153)
        kwargs_640 = {}
        # Getting the type of 'self' (line 153)
        self_637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'self', False)
        # Obtaining the member 'short2long' of a type (line 153)
        short2long_638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), self_637, 'short2long')
        # Obtaining the member 'clear' of a type (line 153)
        clear_639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), short2long_638, 'clear')
        # Calling clear(args, kwargs) (line 153)
        clear_call_result_641 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), clear_639, *[], **kwargs_640)
        
        
        # Assigning a Dict to a Attribute (line 154):
        
        # Assigning a Dict to a Attribute (line 154):
        
        # Obtaining an instance of the builtin type 'dict' (line 154)
        dict_642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 22), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 154)
        
        # Getting the type of 'self' (line 154)
        self_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'self')
        # Setting the type of the member 'repeat' of a type (line 154)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), self_643, 'repeat', dict_642)
        
        # Getting the type of 'self' (line 156)
        self_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 22), 'self')
        # Obtaining the member 'option_table' of a type (line 156)
        option_table_645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 22), self_644, 'option_table')
        # Testing the type of a for loop iterable (line 156)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 156, 8), option_table_645)
        # Getting the type of the for loop variable (line 156)
        for_loop_var_646 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 156, 8), option_table_645)
        # Assigning a type to the variable 'option' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'option', for_loop_var_646)
        # SSA begins for a for statement (line 156)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to len(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'option' (line 157)
        option_648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 19), 'option', False)
        # Processing the call keyword arguments (line 157)
        kwargs_649 = {}
        # Getting the type of 'len' (line 157)
        len_647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 15), 'len', False)
        # Calling len(args, kwargs) (line 157)
        len_call_result_650 = invoke(stypy.reporting.localization.Localization(__file__, 157, 15), len_647, *[option_648], **kwargs_649)
        
        int_651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 30), 'int')
        # Applying the binary operator '==' (line 157)
        result_eq_652 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 15), '==', len_call_result_650, int_651)
        
        # Testing the type of an if condition (line 157)
        if_condition_653 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 12), result_eq_652)
        # Assigning a type to the variable 'if_condition_653' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'if_condition_653', if_condition_653)
        # SSA begins for if statement (line 157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Tuple (line 158):
        
        # Assigning a Subscript to a Name (line 158):
        
        # Obtaining the type of the subscript
        int_654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 16), 'int')
        # Getting the type of 'option' (line 158)
        option_655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 36), 'option')
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 16), option_655, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_657 = invoke(stypy.reporting.localization.Localization(__file__, 158, 16), getitem___656, int_654)
        
        # Assigning a type to the variable 'tuple_var_assignment_461' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'tuple_var_assignment_461', subscript_call_result_657)
        
        # Assigning a Subscript to a Name (line 158):
        
        # Obtaining the type of the subscript
        int_658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 16), 'int')
        # Getting the type of 'option' (line 158)
        option_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 36), 'option')
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 16), option_659, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_661 = invoke(stypy.reporting.localization.Localization(__file__, 158, 16), getitem___660, int_658)
        
        # Assigning a type to the variable 'tuple_var_assignment_462' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'tuple_var_assignment_462', subscript_call_result_661)
        
        # Assigning a Subscript to a Name (line 158):
        
        # Obtaining the type of the subscript
        int_662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 16), 'int')
        # Getting the type of 'option' (line 158)
        option_663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 36), 'option')
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 16), option_663, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_665 = invoke(stypy.reporting.localization.Localization(__file__, 158, 16), getitem___664, int_662)
        
        # Assigning a type to the variable 'tuple_var_assignment_463' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'tuple_var_assignment_463', subscript_call_result_665)
        
        # Assigning a Name to a Name (line 158):
        # Getting the type of 'tuple_var_assignment_461' (line 158)
        tuple_var_assignment_461_666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'tuple_var_assignment_461')
        # Assigning a type to the variable 'long' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'long', tuple_var_assignment_461_666)
        
        # Assigning a Name to a Name (line 158):
        # Getting the type of 'tuple_var_assignment_462' (line 158)
        tuple_var_assignment_462_667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'tuple_var_assignment_462')
        # Assigning a type to the variable 'short' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 22), 'short', tuple_var_assignment_462_667)
        
        # Assigning a Name to a Name (line 158):
        # Getting the type of 'tuple_var_assignment_463' (line 158)
        tuple_var_assignment_463_668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'tuple_var_assignment_463')
        # Assigning a type to the variable 'help' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 29), 'help', tuple_var_assignment_463_668)
        
        # Assigning a Num to a Name (line 159):
        
        # Assigning a Num to a Name (line 159):
        int_669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 25), 'int')
        # Assigning a type to the variable 'repeat' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'repeat', int_669)
        # SSA branch for the else part of an if statement (line 157)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to len(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'option' (line 160)
        option_671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 21), 'option', False)
        # Processing the call keyword arguments (line 160)
        kwargs_672 = {}
        # Getting the type of 'len' (line 160)
        len_670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 17), 'len', False)
        # Calling len(args, kwargs) (line 160)
        len_call_result_673 = invoke(stypy.reporting.localization.Localization(__file__, 160, 17), len_670, *[option_671], **kwargs_672)
        
        int_674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 32), 'int')
        # Applying the binary operator '==' (line 160)
        result_eq_675 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 17), '==', len_call_result_673, int_674)
        
        # Testing the type of an if condition (line 160)
        if_condition_676 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 17), result_eq_675)
        # Assigning a type to the variable 'if_condition_676' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 17), 'if_condition_676', if_condition_676)
        # SSA begins for if statement (line 160)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Tuple (line 161):
        
        # Assigning a Subscript to a Name (line 161):
        
        # Obtaining the type of the subscript
        int_677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 16), 'int')
        # Getting the type of 'option' (line 161)
        option_678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 44), 'option')
        # Obtaining the member '__getitem__' of a type (line 161)
        getitem___679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 16), option_678, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 161)
        subscript_call_result_680 = invoke(stypy.reporting.localization.Localization(__file__, 161, 16), getitem___679, int_677)
        
        # Assigning a type to the variable 'tuple_var_assignment_464' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'tuple_var_assignment_464', subscript_call_result_680)
        
        # Assigning a Subscript to a Name (line 161):
        
        # Obtaining the type of the subscript
        int_681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 16), 'int')
        # Getting the type of 'option' (line 161)
        option_682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 44), 'option')
        # Obtaining the member '__getitem__' of a type (line 161)
        getitem___683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 16), option_682, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 161)
        subscript_call_result_684 = invoke(stypy.reporting.localization.Localization(__file__, 161, 16), getitem___683, int_681)
        
        # Assigning a type to the variable 'tuple_var_assignment_465' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'tuple_var_assignment_465', subscript_call_result_684)
        
        # Assigning a Subscript to a Name (line 161):
        
        # Obtaining the type of the subscript
        int_685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 16), 'int')
        # Getting the type of 'option' (line 161)
        option_686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 44), 'option')
        # Obtaining the member '__getitem__' of a type (line 161)
        getitem___687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 16), option_686, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 161)
        subscript_call_result_688 = invoke(stypy.reporting.localization.Localization(__file__, 161, 16), getitem___687, int_685)
        
        # Assigning a type to the variable 'tuple_var_assignment_466' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'tuple_var_assignment_466', subscript_call_result_688)
        
        # Assigning a Subscript to a Name (line 161):
        
        # Obtaining the type of the subscript
        int_689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 16), 'int')
        # Getting the type of 'option' (line 161)
        option_690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 44), 'option')
        # Obtaining the member '__getitem__' of a type (line 161)
        getitem___691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 16), option_690, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 161)
        subscript_call_result_692 = invoke(stypy.reporting.localization.Localization(__file__, 161, 16), getitem___691, int_689)
        
        # Assigning a type to the variable 'tuple_var_assignment_467' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'tuple_var_assignment_467', subscript_call_result_692)
        
        # Assigning a Name to a Name (line 161):
        # Getting the type of 'tuple_var_assignment_464' (line 161)
        tuple_var_assignment_464_693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'tuple_var_assignment_464')
        # Assigning a type to the variable 'long' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'long', tuple_var_assignment_464_693)
        
        # Assigning a Name to a Name (line 161):
        # Getting the type of 'tuple_var_assignment_465' (line 161)
        tuple_var_assignment_465_694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'tuple_var_assignment_465')
        # Assigning a type to the variable 'short' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 22), 'short', tuple_var_assignment_465_694)
        
        # Assigning a Name to a Name (line 161):
        # Getting the type of 'tuple_var_assignment_466' (line 161)
        tuple_var_assignment_466_695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'tuple_var_assignment_466')
        # Assigning a type to the variable 'help' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 29), 'help', tuple_var_assignment_466_695)
        
        # Assigning a Name to a Name (line 161):
        # Getting the type of 'tuple_var_assignment_467' (line 161)
        tuple_var_assignment_467_696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'tuple_var_assignment_467')
        # Assigning a type to the variable 'repeat' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 35), 'repeat', tuple_var_assignment_467_696)
        # SSA branch for the else part of an if statement (line 160)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'ValueError' (line 165)
        ValueError_697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 22), 'ValueError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 165, 16), ValueError_697, 'raise parameter', BaseException)
        # SSA join for if statement (line 160)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 157)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        
        # Call to isinstance(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'long' (line 168)
        long_699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 30), 'long', False)
        # Getting the type of 'str' (line 168)
        str_700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 36), 'str', False)
        # Processing the call keyword arguments (line 168)
        kwargs_701 = {}
        # Getting the type of 'isinstance' (line 168)
        isinstance_698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 19), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 168)
        isinstance_call_result_702 = invoke(stypy.reporting.localization.Localization(__file__, 168, 19), isinstance_698, *[long_699, str_700], **kwargs_701)
        
        # Applying the 'not' unary operator (line 168)
        result_not__703 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 15), 'not', isinstance_call_result_702)
        
        
        
        # Call to len(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'long' (line 168)
        long_705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 48), 'long', False)
        # Processing the call keyword arguments (line 168)
        kwargs_706 = {}
        # Getting the type of 'len' (line 168)
        len_704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 44), 'len', False)
        # Calling len(args, kwargs) (line 168)
        len_call_result_707 = invoke(stypy.reporting.localization.Localization(__file__, 168, 44), len_704, *[long_705], **kwargs_706)
        
        int_708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 56), 'int')
        # Applying the binary operator '<' (line 168)
        result_lt_709 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 44), '<', len_call_result_707, int_708)
        
        # Applying the binary operator 'or' (line 168)
        result_or_keyword_710 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 15), 'or', result_not__703, result_lt_709)
        
        # Testing the type of an if condition (line 168)
        if_condition_711 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 12), result_or_keyword_710)
        # Assigning a type to the variable 'if_condition_711' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'if_condition_711', if_condition_711)
        # SSA begins for if statement (line 168)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsGetoptError' (line 169)
        DistutilsGetoptError_712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 22), 'DistutilsGetoptError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 169, 16), DistutilsGetoptError_712, 'raise parameter', BaseException)
        # SSA join for if statement (line 168)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'short' (line 173)
        short_713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 22), 'short')
        # Getting the type of 'None' (line 173)
        None_714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 31), 'None')
        # Applying the binary operator 'is' (line 173)
        result_is__715 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 22), 'is', short_713, None_714)
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'short' (line 174)
        short_717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 33), 'short', False)
        # Getting the type of 'str' (line 174)
        str_718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 40), 'str', False)
        # Processing the call keyword arguments (line 174)
        kwargs_719 = {}
        # Getting the type of 'isinstance' (line 174)
        isinstance_716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 22), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 174)
        isinstance_call_result_720 = invoke(stypy.reporting.localization.Localization(__file__, 174, 22), isinstance_716, *[short_717, str_718], **kwargs_719)
        
        
        
        # Call to len(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'short' (line 174)
        short_722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 53), 'short', False)
        # Processing the call keyword arguments (line 174)
        kwargs_723 = {}
        # Getting the type of 'len' (line 174)
        len_721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 49), 'len', False)
        # Calling len(args, kwargs) (line 174)
        len_call_result_724 = invoke(stypy.reporting.localization.Localization(__file__, 174, 49), len_721, *[short_722], **kwargs_723)
        
        int_725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 63), 'int')
        # Applying the binary operator '==' (line 174)
        result_eq_726 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 49), '==', len_call_result_724, int_725)
        
        # Applying the binary operator 'and' (line 174)
        result_and_keyword_727 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 22), 'and', isinstance_call_result_720, result_eq_726)
        
        # Applying the binary operator 'or' (line 173)
        result_or_keyword_728 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 21), 'or', result_is__715, result_and_keyword_727)
        
        # Applying the 'not' unary operator (line 173)
        result_not__729 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 16), 'not', result_or_keyword_728)
        
        # Testing the type of an if condition (line 173)
        if_condition_730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 12), result_not__729)
        # Assigning a type to the variable 'if_condition_730' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'if_condition_730', if_condition_730)
        # SSA begins for if statement (line 173)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsGetoptError' (line 175)
        DistutilsGetoptError_731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 22), 'DistutilsGetoptError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 175, 16), DistutilsGetoptError_731, 'raise parameter', BaseException)
        # SSA join for if statement (line 173)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 179):
        
        # Assigning a Name to a Subscript (line 179):
        # Getting the type of 'repeat' (line 179)
        repeat_732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 32), 'repeat')
        # Getting the type of 'self' (line 179)
        self_733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'self')
        # Obtaining the member 'repeat' of a type (line 179)
        repeat_734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 12), self_733, 'repeat')
        # Getting the type of 'long' (line 179)
        long_735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 24), 'long')
        # Storing an element on a container (line 179)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 12), repeat_734, (long_735, repeat_732))
        
        # Call to append(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'long' (line 180)
        long_739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 34), 'long', False)
        # Processing the call keyword arguments (line 180)
        kwargs_740 = {}
        # Getting the type of 'self' (line 180)
        self_736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'self', False)
        # Obtaining the member 'long_opts' of a type (line 180)
        long_opts_737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 12), self_736, 'long_opts')
        # Obtaining the member 'append' of a type (line 180)
        append_738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 12), long_opts_737, 'append')
        # Calling append(args, kwargs) (line 180)
        append_call_result_741 = invoke(stypy.reporting.localization.Localization(__file__, 180, 12), append_738, *[long_739], **kwargs_740)
        
        
        
        
        # Obtaining the type of the subscript
        int_742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 20), 'int')
        # Getting the type of 'long' (line 182)
        long_743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'long')
        # Obtaining the member '__getitem__' of a type (line 182)
        getitem___744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 15), long_743, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 182)
        subscript_call_result_745 = invoke(stypy.reporting.localization.Localization(__file__, 182, 15), getitem___744, int_742)
        
        str_746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 27), 'str', '=')
        # Applying the binary operator '==' (line 182)
        result_eq_747 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 15), '==', subscript_call_result_745, str_746)
        
        # Testing the type of an if condition (line 182)
        if_condition_748 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 12), result_eq_747)
        # Assigning a type to the variable 'if_condition_748' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'if_condition_748', if_condition_748)
        # SSA begins for if statement (line 182)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'short' (line 183)
        short_749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 19), 'short')
        # Testing the type of an if condition (line 183)
        if_condition_750 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 183, 16), short_749)
        # Assigning a type to the variable 'if_condition_750' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'if_condition_750', if_condition_750)
        # SSA begins for if statement (line 183)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 183):
        
        # Assigning a BinOp to a Name (line 183):
        # Getting the type of 'short' (line 183)
        short_751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 34), 'short')
        str_752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 42), 'str', ':')
        # Applying the binary operator '+' (line 183)
        result_add_753 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 34), '+', short_751, str_752)
        
        # Assigning a type to the variable 'short' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 26), 'short', result_add_753)
        # SSA join for if statement (line 183)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 184):
        
        # Assigning a Subscript to a Name (line 184):
        
        # Obtaining the type of the subscript
        int_754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 28), 'int')
        int_755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 30), 'int')
        slice_756 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 184, 23), int_754, int_755, None)
        # Getting the type of 'long' (line 184)
        long_757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 23), 'long')
        # Obtaining the member '__getitem__' of a type (line 184)
        getitem___758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 23), long_757, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 184)
        subscript_call_result_759 = invoke(stypy.reporting.localization.Localization(__file__, 184, 23), getitem___758, slice_756)
        
        # Assigning a type to the variable 'long' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'long', subscript_call_result_759)
        
        # Assigning a Num to a Subscript (line 185):
        
        # Assigning a Num to a Subscript (line 185):
        int_760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 39), 'int')
        # Getting the type of 'self' (line 185)
        self_761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 16), 'self')
        # Obtaining the member 'takes_arg' of a type (line 185)
        takes_arg_762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 16), self_761, 'takes_arg')
        # Getting the type of 'long' (line 185)
        long_763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 31), 'long')
        # Storing an element on a container (line 185)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 16), takes_arg_762, (long_763, int_760))
        # SSA branch for the else part of an if statement (line 182)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 190):
        
        # Assigning a Call to a Name (line 190):
        
        # Call to get(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'long' (line 190)
        long_767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 51), 'long', False)
        # Processing the call keyword arguments (line 190)
        kwargs_768 = {}
        # Getting the type of 'self' (line 190)
        self_764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 27), 'self', False)
        # Obtaining the member 'negative_alias' of a type (line 190)
        negative_alias_765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 27), self_764, 'negative_alias')
        # Obtaining the member 'get' of a type (line 190)
        get_766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 27), negative_alias_765, 'get')
        # Calling get(args, kwargs) (line 190)
        get_call_result_769 = invoke(stypy.reporting.localization.Localization(__file__, 190, 27), get_766, *[long_767], **kwargs_768)
        
        # Assigning a type to the variable 'alias_to' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'alias_to', get_call_result_769)
        
        # Type idiom detected: calculating its left and rigth part (line 191)
        # Getting the type of 'alias_to' (line 191)
        alias_to_770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'alias_to')
        # Getting the type of 'None' (line 191)
        None_771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 35), 'None')
        
        (may_be_772, more_types_in_union_773) = may_not_be_none(alias_to_770, None_771)

        if may_be_772:

            if more_types_in_union_773:
                # Runtime conditional SSA (line 191)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Obtaining the type of the subscript
            # Getting the type of 'alias_to' (line 192)
            alias_to_774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 38), 'alias_to')
            # Getting the type of 'self' (line 192)
            self_775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 23), 'self')
            # Obtaining the member 'takes_arg' of a type (line 192)
            takes_arg_776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 23), self_775, 'takes_arg')
            # Obtaining the member '__getitem__' of a type (line 192)
            getitem___777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 23), takes_arg_776, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 192)
            subscript_call_result_778 = invoke(stypy.reporting.localization.Localization(__file__, 192, 23), getitem___777, alias_to_774)
            
            # Testing the type of an if condition (line 192)
            if_condition_779 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 20), subscript_call_result_778)
            # Assigning a type to the variable 'if_condition_779' (line 192)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 20), 'if_condition_779', if_condition_779)
            # SSA begins for if statement (line 192)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'DistutilsGetoptError' (line 193)
            DistutilsGetoptError_780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 30), 'DistutilsGetoptError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 193, 24), DistutilsGetoptError_780, 'raise parameter', BaseException)
            # SSA join for if statement (line 192)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Name to a Subscript (line 198):
            
            # Assigning a Name to a Subscript (line 198):
            # Getting the type of 'long' (line 198)
            long_781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 41), 'long')
            # Getting the type of 'self' (line 198)
            self_782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 20), 'self')
            # Obtaining the member 'long_opts' of a type (line 198)
            long_opts_783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 20), self_782, 'long_opts')
            int_784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 35), 'int')
            # Storing an element on a container (line 198)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 20), long_opts_783, (int_784, long_781))
            
            # Assigning a Num to a Subscript (line 199):
            
            # Assigning a Num to a Subscript (line 199):
            int_785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 43), 'int')
            # Getting the type of 'self' (line 199)
            self_786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'self')
            # Obtaining the member 'takes_arg' of a type (line 199)
            takes_arg_787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 20), self_786, 'takes_arg')
            # Getting the type of 'long' (line 199)
            long_788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 35), 'long')
            # Storing an element on a container (line 199)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 20), takes_arg_787, (long_788, int_785))

            if more_types_in_union_773:
                # Runtime conditional SSA for else branch (line 191)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_772) or more_types_in_union_773):
            
            # Assigning a Num to a Subscript (line 202):
            
            # Assigning a Num to a Subscript (line 202):
            int_789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 43), 'int')
            # Getting the type of 'self' (line 202)
            self_790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 20), 'self')
            # Obtaining the member 'takes_arg' of a type (line 202)
            takes_arg_791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 20), self_790, 'takes_arg')
            # Getting the type of 'long' (line 202)
            long_792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 35), 'long')
            # Storing an element on a container (line 202)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 20), takes_arg_791, (long_792, int_789))

            if (may_be_772 and more_types_in_union_773):
                # SSA join for if statement (line 191)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 182)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 206):
        
        # Assigning a Call to a Name (line 206):
        
        # Call to get(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'long' (line 206)
        long_796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 38), 'long', False)
        # Processing the call keyword arguments (line 206)
        kwargs_797 = {}
        # Getting the type of 'self' (line 206)
        self_793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 23), 'self', False)
        # Obtaining the member 'alias' of a type (line 206)
        alias_794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 23), self_793, 'alias')
        # Obtaining the member 'get' of a type (line 206)
        get_795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 23), alias_794, 'get')
        # Calling get(args, kwargs) (line 206)
        get_call_result_798 = invoke(stypy.reporting.localization.Localization(__file__, 206, 23), get_795, *[long_796], **kwargs_797)
        
        # Assigning a type to the variable 'alias_to' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'alias_to', get_call_result_798)
        
        # Type idiom detected: calculating its left and rigth part (line 207)
        # Getting the type of 'alias_to' (line 207)
        alias_to_799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'alias_to')
        # Getting the type of 'None' (line 207)
        None_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 31), 'None')
        
        (may_be_801, more_types_in_union_802) = may_not_be_none(alias_to_799, None_800)

        if may_be_801:

            if more_types_in_union_802:
                # Runtime conditional SSA (line 207)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'long' (line 208)
            long_803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 34), 'long')
            # Getting the type of 'self' (line 208)
            self_804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 19), 'self')
            # Obtaining the member 'takes_arg' of a type (line 208)
            takes_arg_805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 19), self_804, 'takes_arg')
            # Obtaining the member '__getitem__' of a type (line 208)
            getitem___806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 19), takes_arg_805, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 208)
            subscript_call_result_807 = invoke(stypy.reporting.localization.Localization(__file__, 208, 19), getitem___806, long_803)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'alias_to' (line 208)
            alias_to_808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 58), 'alias_to')
            # Getting the type of 'self' (line 208)
            self_809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 43), 'self')
            # Obtaining the member 'takes_arg' of a type (line 208)
            takes_arg_810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 43), self_809, 'takes_arg')
            # Obtaining the member '__getitem__' of a type (line 208)
            getitem___811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 43), takes_arg_810, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 208)
            subscript_call_result_812 = invoke(stypy.reporting.localization.Localization(__file__, 208, 43), getitem___811, alias_to_808)
            
            # Applying the binary operator '!=' (line 208)
            result_ne_813 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 19), '!=', subscript_call_result_807, subscript_call_result_812)
            
            # Testing the type of an if condition (line 208)
            if_condition_814 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 208, 16), result_ne_813)
            # Assigning a type to the variable 'if_condition_814' (line 208)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 16), 'if_condition_814', if_condition_814)
            # SSA begins for if statement (line 208)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'DistutilsGetoptError' (line 209)
            DistutilsGetoptError_815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 26), 'DistutilsGetoptError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 209, 20), DistutilsGetoptError_815, 'raise parameter', BaseException)
            # SSA join for if statement (line 208)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_802:
                # SSA join for if statement (line 207)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        
        # Call to match(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'long' (line 219)
        long_818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 36), 'long', False)
        # Processing the call keyword arguments (line 219)
        kwargs_819 = {}
        # Getting the type of 'longopt_re' (line 219)
        longopt_re_816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 19), 'longopt_re', False)
        # Obtaining the member 'match' of a type (line 219)
        match_817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 19), longopt_re_816, 'match')
        # Calling match(args, kwargs) (line 219)
        match_call_result_820 = invoke(stypy.reporting.localization.Localization(__file__, 219, 19), match_817, *[long_818], **kwargs_819)
        
        # Applying the 'not' unary operator (line 219)
        result_not__821 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 15), 'not', match_call_result_820)
        
        # Testing the type of an if condition (line 219)
        if_condition_822 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 219, 12), result_not__821)
        # Assigning a type to the variable 'if_condition_822' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'if_condition_822', if_condition_822)
        # SSA begins for if statement (line 219)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsGetoptError' (line 220)
        DistutilsGetoptError_823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 22), 'DistutilsGetoptError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 220, 16), DistutilsGetoptError_823, 'raise parameter', BaseException)
        # SSA join for if statement (line 219)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Subscript (line 224):
        
        # Assigning a Call to a Subscript (line 224):
        
        # Call to get_attr_name(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'long' (line 224)
        long_826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 54), 'long', False)
        # Processing the call keyword arguments (line 224)
        kwargs_827 = {}
        # Getting the type of 'self' (line 224)
        self_824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 35), 'self', False)
        # Obtaining the member 'get_attr_name' of a type (line 224)
        get_attr_name_825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 35), self_824, 'get_attr_name')
        # Calling get_attr_name(args, kwargs) (line 224)
        get_attr_name_call_result_828 = invoke(stypy.reporting.localization.Localization(__file__, 224, 35), get_attr_name_825, *[long_826], **kwargs_827)
        
        # Getting the type of 'self' (line 224)
        self_829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'self')
        # Obtaining the member 'attr_name' of a type (line 224)
        attr_name_830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 12), self_829, 'attr_name')
        # Getting the type of 'long' (line 224)
        long_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 27), 'long')
        # Storing an element on a container (line 224)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 12), attr_name_830, (long_831, get_attr_name_call_result_828))
        
        # Getting the type of 'short' (line 225)
        short_832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 15), 'short')
        # Testing the type of an if condition (line 225)
        if_condition_833 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 225, 12), short_832)
        # Assigning a type to the variable 'if_condition_833' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'if_condition_833', if_condition_833)
        # SSA begins for if statement (line 225)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'short' (line 226)
        short_837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 39), 'short', False)
        # Processing the call keyword arguments (line 226)
        kwargs_838 = {}
        # Getting the type of 'self' (line 226)
        self_834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 16), 'self', False)
        # Obtaining the member 'short_opts' of a type (line 226)
        short_opts_835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 16), self_834, 'short_opts')
        # Obtaining the member 'append' of a type (line 226)
        append_836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 16), short_opts_835, 'append')
        # Calling append(args, kwargs) (line 226)
        append_call_result_839 = invoke(stypy.reporting.localization.Localization(__file__, 226, 16), append_836, *[short_837], **kwargs_838)
        
        
        # Assigning a Name to a Subscript (line 227):
        
        # Assigning a Name to a Subscript (line 227):
        # Getting the type of 'long' (line 227)
        long_840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 44), 'long')
        # Getting the type of 'self' (line 227)
        self_841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'self')
        # Obtaining the member 'short2long' of a type (line 227)
        short2long_842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), self_841, 'short2long')
        
        # Obtaining the type of the subscript
        int_843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 38), 'int')
        # Getting the type of 'short' (line 227)
        short_844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 32), 'short')
        # Obtaining the member '__getitem__' of a type (line 227)
        getitem___845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 32), short_844, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 227)
        subscript_call_result_846 = invoke(stypy.reporting.localization.Localization(__file__, 227, 32), getitem___845, int_843)
        
        # Storing an element on a container (line 227)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 16), short2long_842, (subscript_call_result_846, long_840))
        # SSA join for if statement (line 225)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_grok_option_table(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_grok_option_table' in the type store
        # Getting the type of 'stypy_return_type' (line 146)
        stypy_return_type_847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_847)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_grok_option_table'
        return stypy_return_type_847


    @norecursion
    def getopt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 234)
        None_848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 27), 'None')
        # Getting the type of 'None' (line 234)
        None_849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 40), 'None')
        defaults = [None_848, None_849]
        # Create a new context for function 'getopt'
        module_type_store = module_type_store.open_function_context('getopt', 234, 4, False)
        # Assigning a type to the variable 'self' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FancyGetopt.getopt.__dict__.__setitem__('stypy_localization', localization)
        FancyGetopt.getopt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FancyGetopt.getopt.__dict__.__setitem__('stypy_type_store', module_type_store)
        FancyGetopt.getopt.__dict__.__setitem__('stypy_function_name', 'FancyGetopt.getopt')
        FancyGetopt.getopt.__dict__.__setitem__('stypy_param_names_list', ['args', 'object'])
        FancyGetopt.getopt.__dict__.__setitem__('stypy_varargs_param_name', None)
        FancyGetopt.getopt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FancyGetopt.getopt.__dict__.__setitem__('stypy_call_defaults', defaults)
        FancyGetopt.getopt.__dict__.__setitem__('stypy_call_varargs', varargs)
        FancyGetopt.getopt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FancyGetopt.getopt.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FancyGetopt.getopt', ['args', 'object'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getopt', localization, ['args', 'object'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getopt(...)' code ##################

        str_850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, (-1)), 'str', "Parse command-line options in args. Store as attributes on object.\n\n        If 'args' is None or not supplied, uses 'sys.argv[1:]'.  If\n        'object' is None or not supplied, creates a new OptionDummy\n        object, stores option values there, and returns a tuple (args,\n        object).  If 'object' is supplied, it is modified in place and\n        'getopt()' just returns 'args'; in both cases, the returned\n        'args' is a modified copy of the passed-in 'args' list, which\n        is left untouched.\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 245)
        # Getting the type of 'args' (line 245)
        args_851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 11), 'args')
        # Getting the type of 'None' (line 245)
        None_852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 19), 'None')
        
        (may_be_853, more_types_in_union_854) = may_be_none(args_851, None_852)

        if may_be_853:

            if more_types_in_union_854:
                # Runtime conditional SSA (line 245)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 246):
            
            # Assigning a Subscript to a Name (line 246):
            
            # Obtaining the type of the subscript
            int_855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 28), 'int')
            slice_856 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 246, 19), int_855, None, None)
            # Getting the type of 'sys' (line 246)
            sys_857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 19), 'sys')
            # Obtaining the member 'argv' of a type (line 246)
            argv_858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 19), sys_857, 'argv')
            # Obtaining the member '__getitem__' of a type (line 246)
            getitem___859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 19), argv_858, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 246)
            subscript_call_result_860 = invoke(stypy.reporting.localization.Localization(__file__, 246, 19), getitem___859, slice_856)
            
            # Assigning a type to the variable 'args' (line 246)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'args', subscript_call_result_860)

            if more_types_in_union_854:
                # SSA join for if statement (line 245)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 247)
        # Getting the type of 'object' (line 247)
        object_861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 11), 'object')
        # Getting the type of 'None' (line 247)
        None_862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 21), 'None')
        
        (may_be_863, more_types_in_union_864) = may_be_none(object_861, None_862)

        if may_be_863:

            if more_types_in_union_864:
                # Runtime conditional SSA (line 247)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 248):
            
            # Assigning a Call to a Name (line 248):
            
            # Call to OptionDummy(...): (line 248)
            # Processing the call keyword arguments (line 248)
            kwargs_866 = {}
            # Getting the type of 'OptionDummy' (line 248)
            OptionDummy_865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 21), 'OptionDummy', False)
            # Calling OptionDummy(args, kwargs) (line 248)
            OptionDummy_call_result_867 = invoke(stypy.reporting.localization.Localization(__file__, 248, 21), OptionDummy_865, *[], **kwargs_866)
            
            # Assigning a type to the variable 'object' (line 248)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'object', OptionDummy_call_result_867)
            
            # Assigning a Num to a Name (line 249):
            
            # Assigning a Num to a Name (line 249):
            int_868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 29), 'int')
            # Assigning a type to the variable 'created_object' (line 249)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'created_object', int_868)

            if more_types_in_union_864:
                # Runtime conditional SSA for else branch (line 247)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_863) or more_types_in_union_864):
            
            # Assigning a Num to a Name (line 251):
            
            # Assigning a Num to a Name (line 251):
            int_869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 29), 'int')
            # Assigning a type to the variable 'created_object' (line 251)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'created_object', int_869)

            if (may_be_863 and more_types_in_union_864):
                # SSA join for if statement (line 247)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to _grok_option_table(...): (line 253)
        # Processing the call keyword arguments (line 253)
        kwargs_872 = {}
        # Getting the type of 'self' (line 253)
        self_870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'self', False)
        # Obtaining the member '_grok_option_table' of a type (line 253)
        _grok_option_table_871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), self_870, '_grok_option_table')
        # Calling _grok_option_table(args, kwargs) (line 253)
        _grok_option_table_call_result_873 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), _grok_option_table_871, *[], **kwargs_872)
        
        
        # Assigning a Call to a Name (line 255):
        
        # Assigning a Call to a Name (line 255):
        
        # Call to join(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'self' (line 255)
        self_876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 33), 'self', False)
        # Obtaining the member 'short_opts' of a type (line 255)
        short_opts_877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 33), self_876, 'short_opts')
        # Processing the call keyword arguments (line 255)
        kwargs_878 = {}
        # Getting the type of 'string' (line 255)
        string_874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 21), 'string', False)
        # Obtaining the member 'join' of a type (line 255)
        join_875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 21), string_874, 'join')
        # Calling join(args, kwargs) (line 255)
        join_call_result_879 = invoke(stypy.reporting.localization.Localization(__file__, 255, 21), join_875, *[short_opts_877], **kwargs_878)
        
        # Assigning a type to the variable 'short_opts' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'short_opts', join_call_result_879)
        
        
        # SSA begins for try-except statement (line 256)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Tuple (line 257):
        
        # Assigning a Subscript to a Name (line 257):
        
        # Obtaining the type of the subscript
        int_880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 12), 'int')
        
        # Call to getopt(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'args' (line 257)
        args_883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 39), 'args', False)
        # Getting the type of 'short_opts' (line 257)
        short_opts_884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 45), 'short_opts', False)
        # Getting the type of 'self' (line 257)
        self_885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 57), 'self', False)
        # Obtaining the member 'long_opts' of a type (line 257)
        long_opts_886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 57), self_885, 'long_opts')
        # Processing the call keyword arguments (line 257)
        kwargs_887 = {}
        # Getting the type of 'getopt' (line 257)
        getopt_881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 25), 'getopt', False)
        # Obtaining the member 'getopt' of a type (line 257)
        getopt_882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 25), getopt_881, 'getopt')
        # Calling getopt(args, kwargs) (line 257)
        getopt_call_result_888 = invoke(stypy.reporting.localization.Localization(__file__, 257, 25), getopt_882, *[args_883, short_opts_884, long_opts_886], **kwargs_887)
        
        # Obtaining the member '__getitem__' of a type (line 257)
        getitem___889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 12), getopt_call_result_888, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 257)
        subscript_call_result_890 = invoke(stypy.reporting.localization.Localization(__file__, 257, 12), getitem___889, int_880)
        
        # Assigning a type to the variable 'tuple_var_assignment_468' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'tuple_var_assignment_468', subscript_call_result_890)
        
        # Assigning a Subscript to a Name (line 257):
        
        # Obtaining the type of the subscript
        int_891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 12), 'int')
        
        # Call to getopt(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'args' (line 257)
        args_894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 39), 'args', False)
        # Getting the type of 'short_opts' (line 257)
        short_opts_895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 45), 'short_opts', False)
        # Getting the type of 'self' (line 257)
        self_896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 57), 'self', False)
        # Obtaining the member 'long_opts' of a type (line 257)
        long_opts_897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 57), self_896, 'long_opts')
        # Processing the call keyword arguments (line 257)
        kwargs_898 = {}
        # Getting the type of 'getopt' (line 257)
        getopt_892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 25), 'getopt', False)
        # Obtaining the member 'getopt' of a type (line 257)
        getopt_893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 25), getopt_892, 'getopt')
        # Calling getopt(args, kwargs) (line 257)
        getopt_call_result_899 = invoke(stypy.reporting.localization.Localization(__file__, 257, 25), getopt_893, *[args_894, short_opts_895, long_opts_897], **kwargs_898)
        
        # Obtaining the member '__getitem__' of a type (line 257)
        getitem___900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 12), getopt_call_result_899, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 257)
        subscript_call_result_901 = invoke(stypy.reporting.localization.Localization(__file__, 257, 12), getitem___900, int_891)
        
        # Assigning a type to the variable 'tuple_var_assignment_469' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'tuple_var_assignment_469', subscript_call_result_901)
        
        # Assigning a Name to a Name (line 257):
        # Getting the type of 'tuple_var_assignment_468' (line 257)
        tuple_var_assignment_468_902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'tuple_var_assignment_468')
        # Assigning a type to the variable 'opts' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'opts', tuple_var_assignment_468_902)
        
        # Assigning a Name to a Name (line 257):
        # Getting the type of 'tuple_var_assignment_469' (line 257)
        tuple_var_assignment_469_903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'tuple_var_assignment_469')
        # Assigning a type to the variable 'args' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 18), 'args', tuple_var_assignment_469_903)
        # SSA branch for the except part of a try statement (line 256)
        # SSA branch for the except 'Attribute' branch of a try statement (line 256)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'getopt' (line 258)
        getopt_904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 15), 'getopt')
        # Obtaining the member 'error' of a type (line 258)
        error_905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 15), getopt_904, 'error')
        # Assigning a type to the variable 'msg' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'msg', error_905)
        # Getting the type of 'DistutilsArgError' (line 259)
        DistutilsArgError_906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 18), 'DistutilsArgError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 259, 12), DistutilsArgError_906, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 256)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'opts' (line 261)
        opts_907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 24), 'opts')
        # Testing the type of a for loop iterable (line 261)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 261, 8), opts_907)
        # Getting the type of the for loop variable (line 261)
        for_loop_var_908 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 261, 8), opts_907)
        # Assigning a type to the variable 'opt' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'opt', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 8), for_loop_var_908))
        # Assigning a type to the variable 'val' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 8), for_loop_var_908))
        # SSA begins for a for statement (line 261)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'opt' (line 262)
        opt_910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 19), 'opt', False)
        # Processing the call keyword arguments (line 262)
        kwargs_911 = {}
        # Getting the type of 'len' (line 262)
        len_909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 15), 'len', False)
        # Calling len(args, kwargs) (line 262)
        len_call_result_912 = invoke(stypy.reporting.localization.Localization(__file__, 262, 15), len_909, *[opt_910], **kwargs_911)
        
        int_913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 27), 'int')
        # Applying the binary operator '==' (line 262)
        result_eq_914 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 15), '==', len_call_result_912, int_913)
        
        
        
        # Obtaining the type of the subscript
        int_915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 37), 'int')
        # Getting the type of 'opt' (line 262)
        opt_916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 33), 'opt')
        # Obtaining the member '__getitem__' of a type (line 262)
        getitem___917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 33), opt_916, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 262)
        subscript_call_result_918 = invoke(stypy.reporting.localization.Localization(__file__, 262, 33), getitem___917, int_915)
        
        str_919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 43), 'str', '-')
        # Applying the binary operator '==' (line 262)
        result_eq_920 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 33), '==', subscript_call_result_918, str_919)
        
        # Applying the binary operator 'and' (line 262)
        result_and_keyword_921 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 15), 'and', result_eq_914, result_eq_920)
        
        # Testing the type of an if condition (line 262)
        if_condition_922 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 262, 12), result_and_keyword_921)
        # Assigning a type to the variable 'if_condition_922' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'if_condition_922', if_condition_922)
        # SSA begins for if statement (line 262)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 263):
        
        # Assigning a Subscript to a Name (line 263):
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 42), 'int')
        # Getting the type of 'opt' (line 263)
        opt_924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 38), 'opt')
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 38), opt_924, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_926 = invoke(stypy.reporting.localization.Localization(__file__, 263, 38), getitem___925, int_923)
        
        # Getting the type of 'self' (line 263)
        self_927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 22), 'self')
        # Obtaining the member 'short2long' of a type (line 263)
        short2long_928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 22), self_927, 'short2long')
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 22), short2long_928, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_930 = invoke(stypy.reporting.localization.Localization(__file__, 263, 22), getitem___929, subscript_call_result_926)
        
        # Assigning a type to the variable 'opt' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 16), 'opt', subscript_call_result_930)
        # SSA branch for the else part of an if statement (line 262)
        module_type_store.open_ssa_branch('else')
        # Evaluating assert statement condition
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'opt' (line 265)
        opt_932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 27), 'opt', False)
        # Processing the call keyword arguments (line 265)
        kwargs_933 = {}
        # Getting the type of 'len' (line 265)
        len_931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 23), 'len', False)
        # Calling len(args, kwargs) (line 265)
        len_call_result_934 = invoke(stypy.reporting.localization.Localization(__file__, 265, 23), len_931, *[opt_932], **kwargs_933)
        
        int_935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 34), 'int')
        # Applying the binary operator '>' (line 265)
        result_gt_936 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 23), '>', len_call_result_934, int_935)
        
        
        
        # Obtaining the type of the subscript
        int_937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 45), 'int')
        slice_938 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 265, 40), None, int_937, None)
        # Getting the type of 'opt' (line 265)
        opt_939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 40), 'opt')
        # Obtaining the member '__getitem__' of a type (line 265)
        getitem___940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 40), opt_939, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 265)
        subscript_call_result_941 = invoke(stypy.reporting.localization.Localization(__file__, 265, 40), getitem___940, slice_938)
        
        str_942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 51), 'str', '--')
        # Applying the binary operator '==' (line 265)
        result_eq_943 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 40), '==', subscript_call_result_941, str_942)
        
        # Applying the binary operator 'and' (line 265)
        result_and_keyword_944 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 23), 'and', result_gt_936, result_eq_943)
        
        
        # Assigning a Subscript to a Name (line 266):
        
        # Assigning a Subscript to a Name (line 266):
        
        # Obtaining the type of the subscript
        int_945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 26), 'int')
        slice_946 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 266, 22), int_945, None, None)
        # Getting the type of 'opt' (line 266)
        opt_947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 22), 'opt')
        # Obtaining the member '__getitem__' of a type (line 266)
        getitem___948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 22), opt_947, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 266)
        subscript_call_result_949 = invoke(stypy.reporting.localization.Localization(__file__, 266, 22), getitem___948, slice_946)
        
        # Assigning a type to the variable 'opt' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'opt', subscript_call_result_949)
        # SSA join for if statement (line 262)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 268):
        
        # Assigning a Call to a Name (line 268):
        
        # Call to get(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'opt' (line 268)
        opt_953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 35), 'opt', False)
        # Processing the call keyword arguments (line 268)
        kwargs_954 = {}
        # Getting the type of 'self' (line 268)
        self_950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 20), 'self', False)
        # Obtaining the member 'alias' of a type (line 268)
        alias_951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 20), self_950, 'alias')
        # Obtaining the member 'get' of a type (line 268)
        get_952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 20), alias_951, 'get')
        # Calling get(args, kwargs) (line 268)
        get_call_result_955 = invoke(stypy.reporting.localization.Localization(__file__, 268, 20), get_952, *[opt_953], **kwargs_954)
        
        # Assigning a type to the variable 'alias' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'alias', get_call_result_955)
        
        # Getting the type of 'alias' (line 269)
        alias_956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 15), 'alias')
        # Testing the type of an if condition (line 269)
        if_condition_957 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 269, 12), alias_956)
        # Assigning a type to the variable 'if_condition_957' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'if_condition_957', if_condition_957)
        # SSA begins for if statement (line 269)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 270):
        
        # Assigning a Name to a Name (line 270):
        # Getting the type of 'alias' (line 270)
        alias_958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 22), 'alias')
        # Assigning a type to the variable 'opt' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 16), 'opt', alias_958)
        # SSA join for if statement (line 269)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'opt' (line 272)
        opt_959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 34), 'opt')
        # Getting the type of 'self' (line 272)
        self_960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 19), 'self')
        # Obtaining the member 'takes_arg' of a type (line 272)
        takes_arg_961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 19), self_960, 'takes_arg')
        # Obtaining the member '__getitem__' of a type (line 272)
        getitem___962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 19), takes_arg_961, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 272)
        subscript_call_result_963 = invoke(stypy.reporting.localization.Localization(__file__, 272, 19), getitem___962, opt_959)
        
        # Applying the 'not' unary operator (line 272)
        result_not__964 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 15), 'not', subscript_call_result_963)
        
        # Testing the type of an if condition (line 272)
        if_condition_965 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 12), result_not__964)
        # Assigning a type to the variable 'if_condition_965' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'if_condition_965', if_condition_965)
        # SSA begins for if statement (line 272)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Evaluating assert statement condition
        
        # Getting the type of 'val' (line 273)
        val_966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 23), 'val')
        str_967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 30), 'str', '')
        # Applying the binary operator '==' (line 273)
        result_eq_968 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 23), '==', val_966, str_967)
        
        
        # Assigning a Call to a Name (line 274):
        
        # Assigning a Call to a Name (line 274):
        
        # Call to get(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'opt' (line 274)
        opt_972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 48), 'opt', False)
        # Processing the call keyword arguments (line 274)
        kwargs_973 = {}
        # Getting the type of 'self' (line 274)
        self_969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 24), 'self', False)
        # Obtaining the member 'negative_alias' of a type (line 274)
        negative_alias_970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 24), self_969, 'negative_alias')
        # Obtaining the member 'get' of a type (line 274)
        get_971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 24), negative_alias_970, 'get')
        # Calling get(args, kwargs) (line 274)
        get_call_result_974 = invoke(stypy.reporting.localization.Localization(__file__, 274, 24), get_971, *[opt_972], **kwargs_973)
        
        # Assigning a type to the variable 'alias' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'alias', get_call_result_974)
        
        # Getting the type of 'alias' (line 275)
        alias_975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 19), 'alias')
        # Testing the type of an if condition (line 275)
        if_condition_976 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 16), alias_975)
        # Assigning a type to the variable 'if_condition_976' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'if_condition_976', if_condition_976)
        # SSA begins for if statement (line 275)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 276):
        
        # Assigning a Name to a Name (line 276):
        # Getting the type of 'alias' (line 276)
        alias_977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 26), 'alias')
        # Assigning a type to the variable 'opt' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 20), 'opt', alias_977)
        
        # Assigning a Num to a Name (line 277):
        
        # Assigning a Num to a Name (line 277):
        int_978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 26), 'int')
        # Assigning a type to the variable 'val' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 20), 'val', int_978)
        # SSA branch for the else part of an if statement (line 275)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 279):
        
        # Assigning a Num to a Name (line 279):
        int_979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 26), 'int')
        # Assigning a type to the variable 'val' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 20), 'val', int_979)
        # SSA join for if statement (line 275)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 272)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 281):
        
        # Assigning a Subscript to a Name (line 281):
        
        # Obtaining the type of the subscript
        # Getting the type of 'opt' (line 281)
        opt_980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 34), 'opt')
        # Getting the type of 'self' (line 281)
        self_981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 19), 'self')
        # Obtaining the member 'attr_name' of a type (line 281)
        attr_name_982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 19), self_981, 'attr_name')
        # Obtaining the member '__getitem__' of a type (line 281)
        getitem___983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 19), attr_name_982, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 281)
        subscript_call_result_984 = invoke(stypy.reporting.localization.Localization(__file__, 281, 19), getitem___983, opt_980)
        
        # Assigning a type to the variable 'attr' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'attr', subscript_call_result_984)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'val' (line 284)
        val_985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 15), 'val')
        
        
        # Call to get(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'attr' (line 284)
        attr_989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 39), 'attr', False)
        # Processing the call keyword arguments (line 284)
        kwargs_990 = {}
        # Getting the type of 'self' (line 284)
        self_986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 23), 'self', False)
        # Obtaining the member 'repeat' of a type (line 284)
        repeat_987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 23), self_986, 'repeat')
        # Obtaining the member 'get' of a type (line 284)
        get_988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 23), repeat_987, 'get')
        # Calling get(args, kwargs) (line 284)
        get_call_result_991 = invoke(stypy.reporting.localization.Localization(__file__, 284, 23), get_988, *[attr_989], **kwargs_990)
        
        # Getting the type of 'None' (line 284)
        None_992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 52), 'None')
        # Applying the binary operator 'isnot' (line 284)
        result_is_not_993 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 23), 'isnot', get_call_result_991, None_992)
        
        # Applying the binary operator 'and' (line 284)
        result_and_keyword_994 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 15), 'and', val_985, result_is_not_993)
        
        # Testing the type of an if condition (line 284)
        if_condition_995 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 12), result_and_keyword_994)
        # Assigning a type to the variable 'if_condition_995' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'if_condition_995', if_condition_995)
        # SSA begins for if statement (line 284)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 285):
        
        # Assigning a BinOp to a Name (line 285):
        
        # Call to getattr(...): (line 285)
        # Processing the call arguments (line 285)
        # Getting the type of 'object' (line 285)
        object_997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 30), 'object', False)
        # Getting the type of 'attr' (line 285)
        attr_998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 38), 'attr', False)
        int_999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 44), 'int')
        # Processing the call keyword arguments (line 285)
        kwargs_1000 = {}
        # Getting the type of 'getattr' (line 285)
        getattr_996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 22), 'getattr', False)
        # Calling getattr(args, kwargs) (line 285)
        getattr_call_result_1001 = invoke(stypy.reporting.localization.Localization(__file__, 285, 22), getattr_996, *[object_997, attr_998, int_999], **kwargs_1000)
        
        int_1002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 49), 'int')
        # Applying the binary operator '+' (line 285)
        result_add_1003 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 22), '+', getattr_call_result_1001, int_1002)
        
        # Assigning a type to the variable 'val' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 16), 'val', result_add_1003)
        # SSA join for if statement (line 284)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to setattr(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 'object' (line 286)
        object_1005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 20), 'object', False)
        # Getting the type of 'attr' (line 286)
        attr_1006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 28), 'attr', False)
        # Getting the type of 'val' (line 286)
        val_1007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 34), 'val', False)
        # Processing the call keyword arguments (line 286)
        kwargs_1008 = {}
        # Getting the type of 'setattr' (line 286)
        setattr_1004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'setattr', False)
        # Calling setattr(args, kwargs) (line 286)
        setattr_call_result_1009 = invoke(stypy.reporting.localization.Localization(__file__, 286, 12), setattr_1004, *[object_1005, attr_1006, val_1007], **kwargs_1008)
        
        
        # Call to append(...): (line 287)
        # Processing the call arguments (line 287)
        
        # Obtaining an instance of the builtin type 'tuple' (line 287)
        tuple_1013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 287)
        # Adding element type (line 287)
        # Getting the type of 'opt' (line 287)
        opt_1014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 38), 'opt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 38), tuple_1013, opt_1014)
        # Adding element type (line 287)
        # Getting the type of 'val' (line 287)
        val_1015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 43), 'val', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 38), tuple_1013, val_1015)
        
        # Processing the call keyword arguments (line 287)
        kwargs_1016 = {}
        # Getting the type of 'self' (line 287)
        self_1010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'self', False)
        # Obtaining the member 'option_order' of a type (line 287)
        option_order_1011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 12), self_1010, 'option_order')
        # Obtaining the member 'append' of a type (line 287)
        append_1012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 12), option_order_1011, 'append')
        # Calling append(args, kwargs) (line 287)
        append_call_result_1017 = invoke(stypy.reporting.localization.Localization(__file__, 287, 12), append_1012, *[tuple_1013], **kwargs_1016)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'created_object' (line 290)
        created_object_1018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 11), 'created_object')
        # Testing the type of an if condition (line 290)
        if_condition_1019 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 290, 8), created_object_1018)
        # Assigning a type to the variable 'if_condition_1019' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'if_condition_1019', if_condition_1019)
        # SSA begins for if statement (line 290)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 291)
        tuple_1020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 291)
        # Adding element type (line 291)
        # Getting the type of 'args' (line 291)
        args_1021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 19), 'args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 19), tuple_1020, args_1021)
        # Adding element type (line 291)
        # Getting the type of 'object' (line 291)
        object_1022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 25), 'object')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 19), tuple_1020, object_1022)
        
        # Assigning a type to the variable 'stypy_return_type' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'stypy_return_type', tuple_1020)
        # SSA branch for the else part of an if statement (line 290)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'args' (line 293)
        args_1023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 19), 'args')
        # Assigning a type to the variable 'stypy_return_type' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'stypy_return_type', args_1023)
        # SSA join for if statement (line 290)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'getopt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getopt' in the type store
        # Getting the type of 'stypy_return_type' (line 234)
        stypy_return_type_1024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1024)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getopt'
        return stypy_return_type_1024


    @norecursion
    def get_option_order(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_option_order'
        module_type_store = module_type_store.open_function_context('get_option_order', 298, 4, False)
        # Assigning a type to the variable 'self' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FancyGetopt.get_option_order.__dict__.__setitem__('stypy_localization', localization)
        FancyGetopt.get_option_order.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FancyGetopt.get_option_order.__dict__.__setitem__('stypy_type_store', module_type_store)
        FancyGetopt.get_option_order.__dict__.__setitem__('stypy_function_name', 'FancyGetopt.get_option_order')
        FancyGetopt.get_option_order.__dict__.__setitem__('stypy_param_names_list', [])
        FancyGetopt.get_option_order.__dict__.__setitem__('stypy_varargs_param_name', None)
        FancyGetopt.get_option_order.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FancyGetopt.get_option_order.__dict__.__setitem__('stypy_call_defaults', defaults)
        FancyGetopt.get_option_order.__dict__.__setitem__('stypy_call_varargs', varargs)
        FancyGetopt.get_option_order.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FancyGetopt.get_option_order.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FancyGetopt.get_option_order', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_option_order', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_option_order(...)' code ##################

        str_1025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, (-1)), 'str', "Returns the list of (option, value) tuples processed by the\n        previous run of 'getopt()'.  Raises RuntimeError if\n        'getopt()' hasn't been called yet.\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 303)
        # Getting the type of 'self' (line 303)
        self_1026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 11), 'self')
        # Obtaining the member 'option_order' of a type (line 303)
        option_order_1027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 11), self_1026, 'option_order')
        # Getting the type of 'None' (line 303)
        None_1028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 32), 'None')
        
        (may_be_1029, more_types_in_union_1030) = may_be_none(option_order_1027, None_1028)

        if may_be_1029:

            if more_types_in_union_1030:
                # Runtime conditional SSA (line 303)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'RuntimeError' (line 304)
            RuntimeError_1031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 18), 'RuntimeError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 304, 12), RuntimeError_1031, 'raise parameter', BaseException)

            if more_types_in_union_1030:
                # Runtime conditional SSA for else branch (line 303)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_1029) or more_types_in_union_1030):
            # Getting the type of 'self' (line 306)
            self_1032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 19), 'self')
            # Obtaining the member 'option_order' of a type (line 306)
            option_order_1033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 19), self_1032, 'option_order')
            # Assigning a type to the variable 'stypy_return_type' (line 306)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'stypy_return_type', option_order_1033)

            if (may_be_1029 and more_types_in_union_1030):
                # SSA join for if statement (line 303)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'get_option_order(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_option_order' in the type store
        # Getting the type of 'stypy_return_type' (line 298)
        stypy_return_type_1034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1034)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_option_order'
        return stypy_return_type_1034


    @norecursion
    def generate_help(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 309)
        None_1035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 36), 'None')
        defaults = [None_1035]
        # Create a new context for function 'generate_help'
        module_type_store = module_type_store.open_function_context('generate_help', 309, 4, False)
        # Assigning a type to the variable 'self' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FancyGetopt.generate_help.__dict__.__setitem__('stypy_localization', localization)
        FancyGetopt.generate_help.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FancyGetopt.generate_help.__dict__.__setitem__('stypy_type_store', module_type_store)
        FancyGetopt.generate_help.__dict__.__setitem__('stypy_function_name', 'FancyGetopt.generate_help')
        FancyGetopt.generate_help.__dict__.__setitem__('stypy_param_names_list', ['header'])
        FancyGetopt.generate_help.__dict__.__setitem__('stypy_varargs_param_name', None)
        FancyGetopt.generate_help.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FancyGetopt.generate_help.__dict__.__setitem__('stypy_call_defaults', defaults)
        FancyGetopt.generate_help.__dict__.__setitem__('stypy_call_varargs', varargs)
        FancyGetopt.generate_help.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FancyGetopt.generate_help.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FancyGetopt.generate_help', ['header'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'generate_help', localization, ['header'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'generate_help(...)' code ##################

        str_1036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, (-1)), 'str', 'Generate help text (a list of strings, one per suggested line of\n        output) from the option table for this FancyGetopt object.\n        ')
        
        # Assigning a Num to a Name (line 317):
        
        # Assigning a Num to a Name (line 317):
        int_1037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 18), 'int')
        # Assigning a type to the variable 'max_opt' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'max_opt', int_1037)
        
        # Getting the type of 'self' (line 318)
        self_1038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 22), 'self')
        # Obtaining the member 'option_table' of a type (line 318)
        option_table_1039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 22), self_1038, 'option_table')
        # Testing the type of a for loop iterable (line 318)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 318, 8), option_table_1039)
        # Getting the type of the for loop variable (line 318)
        for_loop_var_1040 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 318, 8), option_table_1039)
        # Assigning a type to the variable 'option' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'option', for_loop_var_1040)
        # SSA begins for a for statement (line 318)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 319):
        
        # Assigning a Subscript to a Name (line 319):
        
        # Obtaining the type of the subscript
        int_1041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 26), 'int')
        # Getting the type of 'option' (line 319)
        option_1042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 19), 'option')
        # Obtaining the member '__getitem__' of a type (line 319)
        getitem___1043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 19), option_1042, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 319)
        subscript_call_result_1044 = invoke(stypy.reporting.localization.Localization(__file__, 319, 19), getitem___1043, int_1041)
        
        # Assigning a type to the variable 'long' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'long', subscript_call_result_1044)
        
        # Assigning a Subscript to a Name (line 320):
        
        # Assigning a Subscript to a Name (line 320):
        
        # Obtaining the type of the subscript
        int_1045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 27), 'int')
        # Getting the type of 'option' (line 320)
        option_1046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 20), 'option')
        # Obtaining the member '__getitem__' of a type (line 320)
        getitem___1047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 20), option_1046, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 320)
        subscript_call_result_1048 = invoke(stypy.reporting.localization.Localization(__file__, 320, 20), getitem___1047, int_1045)
        
        # Assigning a type to the variable 'short' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'short', subscript_call_result_1048)
        
        # Assigning a Call to a Name (line 321):
        
        # Assigning a Call to a Name (line 321):
        
        # Call to len(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'long' (line 321)
        long_1050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 20), 'long', False)
        # Processing the call keyword arguments (line 321)
        kwargs_1051 = {}
        # Getting the type of 'len' (line 321)
        len_1049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 16), 'len', False)
        # Calling len(args, kwargs) (line 321)
        len_call_result_1052 = invoke(stypy.reporting.localization.Localization(__file__, 321, 16), len_1049, *[long_1050], **kwargs_1051)
        
        # Assigning a type to the variable 'l' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'l', len_call_result_1052)
        
        
        
        # Obtaining the type of the subscript
        int_1053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 20), 'int')
        # Getting the type of 'long' (line 322)
        long_1054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 15), 'long')
        # Obtaining the member '__getitem__' of a type (line 322)
        getitem___1055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 15), long_1054, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 322)
        subscript_call_result_1056 = invoke(stypy.reporting.localization.Localization(__file__, 322, 15), getitem___1055, int_1053)
        
        str_1057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 27), 'str', '=')
        # Applying the binary operator '==' (line 322)
        result_eq_1058 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 15), '==', subscript_call_result_1056, str_1057)
        
        # Testing the type of an if condition (line 322)
        if_condition_1059 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 12), result_eq_1058)
        # Assigning a type to the variable 'if_condition_1059' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'if_condition_1059', if_condition_1059)
        # SSA begins for if statement (line 322)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 323):
        
        # Assigning a BinOp to a Name (line 323):
        # Getting the type of 'l' (line 323)
        l_1060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 20), 'l')
        int_1061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 24), 'int')
        # Applying the binary operator '-' (line 323)
        result_sub_1062 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 20), '-', l_1060, int_1061)
        
        # Assigning a type to the variable 'l' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 16), 'l', result_sub_1062)
        # SSA join for if statement (line 322)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 324)
        # Getting the type of 'short' (line 324)
        short_1063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'short')
        # Getting the type of 'None' (line 324)
        None_1064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 28), 'None')
        
        (may_be_1065, more_types_in_union_1066) = may_not_be_none(short_1063, None_1064)

        if may_be_1065:

            if more_types_in_union_1066:
                # Runtime conditional SSA (line 324)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 325):
            
            # Assigning a BinOp to a Name (line 325):
            # Getting the type of 'l' (line 325)
            l_1067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 20), 'l')
            int_1068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 24), 'int')
            # Applying the binary operator '+' (line 325)
            result_add_1069 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 20), '+', l_1067, int_1068)
            
            # Assigning a type to the variable 'l' (line 325)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 16), 'l', result_add_1069)

            if more_types_in_union_1066:
                # SSA join for if statement (line 324)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'l' (line 326)
        l_1070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 15), 'l')
        # Getting the type of 'max_opt' (line 326)
        max_opt_1071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 19), 'max_opt')
        # Applying the binary operator '>' (line 326)
        result_gt_1072 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 15), '>', l_1070, max_opt_1071)
        
        # Testing the type of an if condition (line 326)
        if_condition_1073 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 326, 12), result_gt_1072)
        # Assigning a type to the variable 'if_condition_1073' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'if_condition_1073', if_condition_1073)
        # SSA begins for if statement (line 326)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 327):
        
        # Assigning a Name to a Name (line 327):
        # Getting the type of 'l' (line 327)
        l_1074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 26), 'l')
        # Assigning a type to the variable 'max_opt' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'max_opt', l_1074)
        # SSA join for if statement (line 326)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 329):
        
        # Assigning a BinOp to a Name (line 329):
        # Getting the type of 'max_opt' (line 329)
        max_opt_1075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 20), 'max_opt')
        int_1076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 30), 'int')
        # Applying the binary operator '+' (line 329)
        result_add_1077 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 20), '+', max_opt_1075, int_1076)
        
        int_1078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 34), 'int')
        # Applying the binary operator '+' (line 329)
        result_add_1079 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 32), '+', result_add_1077, int_1078)
        
        int_1080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 38), 'int')
        # Applying the binary operator '+' (line 329)
        result_add_1081 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 36), '+', result_add_1079, int_1080)
        
        # Assigning a type to the variable 'opt_width' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'opt_width', result_add_1081)
        
        # Assigning a Num to a Name (line 353):
        
        # Assigning a Num to a Name (line 353):
        int_1082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 21), 'int')
        # Assigning a type to the variable 'line_width' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'line_width', int_1082)
        
        # Assigning a BinOp to a Name (line 354):
        
        # Assigning a BinOp to a Name (line 354):
        # Getting the type of 'line_width' (line 354)
        line_width_1083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 21), 'line_width')
        # Getting the type of 'opt_width' (line 354)
        opt_width_1084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 34), 'opt_width')
        # Applying the binary operator '-' (line 354)
        result_sub_1085 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 21), '-', line_width_1083, opt_width_1084)
        
        # Assigning a type to the variable 'text_width' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'text_width', result_sub_1085)
        
        # Assigning a BinOp to a Name (line 355):
        
        # Assigning a BinOp to a Name (line 355):
        str_1086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 21), 'str', ' ')
        # Getting the type of 'opt_width' (line 355)
        opt_width_1087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 27), 'opt_width')
        # Applying the binary operator '*' (line 355)
        result_mul_1088 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 21), '*', str_1086, opt_width_1087)
        
        # Assigning a type to the variable 'big_indent' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'big_indent', result_mul_1088)
        
        # Getting the type of 'header' (line 356)
        header_1089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 11), 'header')
        # Testing the type of an if condition (line 356)
        if_condition_1090 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 356, 8), header_1089)
        # Assigning a type to the variable 'if_condition_1090' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'if_condition_1090', if_condition_1090)
        # SSA begins for if statement (line 356)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 357):
        
        # Assigning a List to a Name (line 357):
        
        # Obtaining an instance of the builtin type 'list' (line 357)
        list_1091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 357)
        # Adding element type (line 357)
        # Getting the type of 'header' (line 357)
        header_1092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 21), 'header')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 20), list_1091, header_1092)
        
        # Assigning a type to the variable 'lines' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'lines', list_1091)
        # SSA branch for the else part of an if statement (line 356)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 359):
        
        # Assigning a List to a Name (line 359):
        
        # Obtaining an instance of the builtin type 'list' (line 359)
        list_1093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 359)
        # Adding element type (line 359)
        str_1094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 21), 'str', 'Option summary:')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 20), list_1093, str_1094)
        
        # Assigning a type to the variable 'lines' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'lines', list_1093)
        # SSA join for if statement (line 356)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 361)
        self_1095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 22), 'self')
        # Obtaining the member 'option_table' of a type (line 361)
        option_table_1096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 22), self_1095, 'option_table')
        # Testing the type of a for loop iterable (line 361)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 361, 8), option_table_1096)
        # Getting the type of the for loop variable (line 361)
        for_loop_var_1097 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 361, 8), option_table_1096)
        # Assigning a type to the variable 'option' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'option', for_loop_var_1097)
        # SSA begins for a for statement (line 361)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Tuple (line 362):
        
        # Assigning a Subscript to a Name (line 362):
        
        # Obtaining the type of the subscript
        int_1098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 12), 'int')
        
        # Obtaining the type of the subscript
        int_1099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 40), 'int')
        slice_1100 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 362, 32), None, int_1099, None)
        # Getting the type of 'option' (line 362)
        option_1101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 32), 'option')
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___1102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 32), option_1101, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_1103 = invoke(stypy.reporting.localization.Localization(__file__, 362, 32), getitem___1102, slice_1100)
        
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___1104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 12), subscript_call_result_1103, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_1105 = invoke(stypy.reporting.localization.Localization(__file__, 362, 12), getitem___1104, int_1098)
        
        # Assigning a type to the variable 'tuple_var_assignment_470' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'tuple_var_assignment_470', subscript_call_result_1105)
        
        # Assigning a Subscript to a Name (line 362):
        
        # Obtaining the type of the subscript
        int_1106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 12), 'int')
        
        # Obtaining the type of the subscript
        int_1107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 40), 'int')
        slice_1108 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 362, 32), None, int_1107, None)
        # Getting the type of 'option' (line 362)
        option_1109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 32), 'option')
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___1110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 32), option_1109, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_1111 = invoke(stypy.reporting.localization.Localization(__file__, 362, 32), getitem___1110, slice_1108)
        
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___1112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 12), subscript_call_result_1111, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_1113 = invoke(stypy.reporting.localization.Localization(__file__, 362, 12), getitem___1112, int_1106)
        
        # Assigning a type to the variable 'tuple_var_assignment_471' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'tuple_var_assignment_471', subscript_call_result_1113)
        
        # Assigning a Subscript to a Name (line 362):
        
        # Obtaining the type of the subscript
        int_1114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 12), 'int')
        
        # Obtaining the type of the subscript
        int_1115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 40), 'int')
        slice_1116 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 362, 32), None, int_1115, None)
        # Getting the type of 'option' (line 362)
        option_1117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 32), 'option')
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___1118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 32), option_1117, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_1119 = invoke(stypy.reporting.localization.Localization(__file__, 362, 32), getitem___1118, slice_1116)
        
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___1120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 12), subscript_call_result_1119, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_1121 = invoke(stypy.reporting.localization.Localization(__file__, 362, 12), getitem___1120, int_1114)
        
        # Assigning a type to the variable 'tuple_var_assignment_472' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'tuple_var_assignment_472', subscript_call_result_1121)
        
        # Assigning a Name to a Name (line 362):
        # Getting the type of 'tuple_var_assignment_470' (line 362)
        tuple_var_assignment_470_1122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'tuple_var_assignment_470')
        # Assigning a type to the variable 'long' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'long', tuple_var_assignment_470_1122)
        
        # Assigning a Name to a Name (line 362):
        # Getting the type of 'tuple_var_assignment_471' (line 362)
        tuple_var_assignment_471_1123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'tuple_var_assignment_471')
        # Assigning a type to the variable 'short' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 18), 'short', tuple_var_assignment_471_1123)
        
        # Assigning a Name to a Name (line 362):
        # Getting the type of 'tuple_var_assignment_472' (line 362)
        tuple_var_assignment_472_1124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'tuple_var_assignment_472')
        # Assigning a type to the variable 'help' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 25), 'help', tuple_var_assignment_472_1124)
        
        # Assigning a Call to a Name (line 363):
        
        # Assigning a Call to a Name (line 363):
        
        # Call to wrap_text(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'help' (line 363)
        help_1126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 29), 'help', False)
        # Getting the type of 'text_width' (line 363)
        text_width_1127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 35), 'text_width', False)
        # Processing the call keyword arguments (line 363)
        kwargs_1128 = {}
        # Getting the type of 'wrap_text' (line 363)
        wrap_text_1125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 19), 'wrap_text', False)
        # Calling wrap_text(args, kwargs) (line 363)
        wrap_text_call_result_1129 = invoke(stypy.reporting.localization.Localization(__file__, 363, 19), wrap_text_1125, *[help_1126, text_width_1127], **kwargs_1128)
        
        # Assigning a type to the variable 'text' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'text', wrap_text_call_result_1129)
        
        
        
        # Obtaining the type of the subscript
        int_1130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 20), 'int')
        # Getting the type of 'long' (line 364)
        long_1131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 15), 'long')
        # Obtaining the member '__getitem__' of a type (line 364)
        getitem___1132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 15), long_1131, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 364)
        subscript_call_result_1133 = invoke(stypy.reporting.localization.Localization(__file__, 364, 15), getitem___1132, int_1130)
        
        str_1134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 27), 'str', '=')
        # Applying the binary operator '==' (line 364)
        result_eq_1135 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 15), '==', subscript_call_result_1133, str_1134)
        
        # Testing the type of an if condition (line 364)
        if_condition_1136 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 364, 12), result_eq_1135)
        # Assigning a type to the variable 'if_condition_1136' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'if_condition_1136', if_condition_1136)
        # SSA begins for if statement (line 364)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 365):
        
        # Assigning a Subscript to a Name (line 365):
        
        # Obtaining the type of the subscript
        int_1137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 28), 'int')
        int_1138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 30), 'int')
        slice_1139 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 365, 23), int_1137, int_1138, None)
        # Getting the type of 'long' (line 365)
        long_1140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 23), 'long')
        # Obtaining the member '__getitem__' of a type (line 365)
        getitem___1141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 23), long_1140, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 365)
        subscript_call_result_1142 = invoke(stypy.reporting.localization.Localization(__file__, 365, 23), getitem___1141, slice_1139)
        
        # Assigning a type to the variable 'long' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 16), 'long', subscript_call_result_1142)
        # SSA join for if statement (line 364)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 368)
        # Getting the type of 'short' (line 368)
        short_1143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 15), 'short')
        # Getting the type of 'None' (line 368)
        None_1144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 24), 'None')
        
        (may_be_1145, more_types_in_union_1146) = may_be_none(short_1143, None_1144)

        if may_be_1145:

            if more_types_in_union_1146:
                # Runtime conditional SSA (line 368)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 'text' (line 369)
            text_1147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 19), 'text')
            # Testing the type of an if condition (line 369)
            if_condition_1148 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 369, 16), text_1147)
            # Assigning a type to the variable 'if_condition_1148' (line 369)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 16), 'if_condition_1148', if_condition_1148)
            # SSA begins for if statement (line 369)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 370)
            # Processing the call arguments (line 370)
            str_1151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 33), 'str', '  --%-*s  %s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 370)
            tuple_1152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 51), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 370)
            # Adding element type (line 370)
            # Getting the type of 'max_opt' (line 370)
            max_opt_1153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 51), 'max_opt', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 51), tuple_1152, max_opt_1153)
            # Adding element type (line 370)
            # Getting the type of 'long' (line 370)
            long_1154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 60), 'long', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 51), tuple_1152, long_1154)
            # Adding element type (line 370)
            
            # Obtaining the type of the subscript
            int_1155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 71), 'int')
            # Getting the type of 'text' (line 370)
            text_1156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 66), 'text', False)
            # Obtaining the member '__getitem__' of a type (line 370)
            getitem___1157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 66), text_1156, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 370)
            subscript_call_result_1158 = invoke(stypy.reporting.localization.Localization(__file__, 370, 66), getitem___1157, int_1155)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 51), tuple_1152, subscript_call_result_1158)
            
            # Applying the binary operator '%' (line 370)
            result_mod_1159 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 33), '%', str_1151, tuple_1152)
            
            # Processing the call keyword arguments (line 370)
            kwargs_1160 = {}
            # Getting the type of 'lines' (line 370)
            lines_1149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 20), 'lines', False)
            # Obtaining the member 'append' of a type (line 370)
            append_1150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 20), lines_1149, 'append')
            # Calling append(args, kwargs) (line 370)
            append_call_result_1161 = invoke(stypy.reporting.localization.Localization(__file__, 370, 20), append_1150, *[result_mod_1159], **kwargs_1160)
            
            # SSA branch for the else part of an if statement (line 369)
            module_type_store.open_ssa_branch('else')
            
            # Call to append(...): (line 372)
            # Processing the call arguments (line 372)
            str_1164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 33), 'str', '  --%-*s  ')
            
            # Obtaining an instance of the builtin type 'tuple' (line 372)
            tuple_1165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 49), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 372)
            # Adding element type (line 372)
            # Getting the type of 'max_opt' (line 372)
            max_opt_1166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 49), 'max_opt', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 49), tuple_1165, max_opt_1166)
            # Adding element type (line 372)
            # Getting the type of 'long' (line 372)
            long_1167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 58), 'long', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 49), tuple_1165, long_1167)
            
            # Applying the binary operator '%' (line 372)
            result_mod_1168 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 33), '%', str_1164, tuple_1165)
            
            # Processing the call keyword arguments (line 372)
            kwargs_1169 = {}
            # Getting the type of 'lines' (line 372)
            lines_1162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 20), 'lines', False)
            # Obtaining the member 'append' of a type (line 372)
            append_1163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 20), lines_1162, 'append')
            # Calling append(args, kwargs) (line 372)
            append_call_result_1170 = invoke(stypy.reporting.localization.Localization(__file__, 372, 20), append_1163, *[result_mod_1168], **kwargs_1169)
            
            # SSA join for if statement (line 369)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_1146:
                # Runtime conditional SSA for else branch (line 368)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_1145) or more_types_in_union_1146):
            
            # Assigning a BinOp to a Name (line 377):
            
            # Assigning a BinOp to a Name (line 377):
            str_1171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 28), 'str', '%s (-%s)')
            
            # Obtaining an instance of the builtin type 'tuple' (line 377)
            tuple_1172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 42), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 377)
            # Adding element type (line 377)
            # Getting the type of 'long' (line 377)
            long_1173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 42), 'long')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 42), tuple_1172, long_1173)
            # Adding element type (line 377)
            # Getting the type of 'short' (line 377)
            short_1174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 48), 'short')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 42), tuple_1172, short_1174)
            
            # Applying the binary operator '%' (line 377)
            result_mod_1175 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 28), '%', str_1171, tuple_1172)
            
            # Assigning a type to the variable 'opt_names' (line 377)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 16), 'opt_names', result_mod_1175)
            
            # Getting the type of 'text' (line 378)
            text_1176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 19), 'text')
            # Testing the type of an if condition (line 378)
            if_condition_1177 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 378, 16), text_1176)
            # Assigning a type to the variable 'if_condition_1177' (line 378)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 16), 'if_condition_1177', if_condition_1177)
            # SSA begins for if statement (line 378)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 379)
            # Processing the call arguments (line 379)
            str_1180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 33), 'str', '  --%-*s  %s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 380)
            tuple_1181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 34), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 380)
            # Adding element type (line 380)
            # Getting the type of 'max_opt' (line 380)
            max_opt_1182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 34), 'max_opt', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 34), tuple_1181, max_opt_1182)
            # Adding element type (line 380)
            # Getting the type of 'opt_names' (line 380)
            opt_names_1183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 43), 'opt_names', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 34), tuple_1181, opt_names_1183)
            # Adding element type (line 380)
            
            # Obtaining the type of the subscript
            int_1184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 59), 'int')
            # Getting the type of 'text' (line 380)
            text_1185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 54), 'text', False)
            # Obtaining the member '__getitem__' of a type (line 380)
            getitem___1186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 54), text_1185, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 380)
            subscript_call_result_1187 = invoke(stypy.reporting.localization.Localization(__file__, 380, 54), getitem___1186, int_1184)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 34), tuple_1181, subscript_call_result_1187)
            
            # Applying the binary operator '%' (line 379)
            result_mod_1188 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 33), '%', str_1180, tuple_1181)
            
            # Processing the call keyword arguments (line 379)
            kwargs_1189 = {}
            # Getting the type of 'lines' (line 379)
            lines_1178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 20), 'lines', False)
            # Obtaining the member 'append' of a type (line 379)
            append_1179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 20), lines_1178, 'append')
            # Calling append(args, kwargs) (line 379)
            append_call_result_1190 = invoke(stypy.reporting.localization.Localization(__file__, 379, 20), append_1179, *[result_mod_1188], **kwargs_1189)
            
            # SSA branch for the else part of an if statement (line 378)
            module_type_store.open_ssa_branch('else')
            
            # Call to append(...): (line 382)
            # Processing the call arguments (line 382)
            str_1193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 33), 'str', '  --%-*s')
            # Getting the type of 'opt_names' (line 382)
            opt_names_1194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 46), 'opt_names', False)
            # Applying the binary operator '%' (line 382)
            result_mod_1195 = python_operator(stypy.reporting.localization.Localization(__file__, 382, 33), '%', str_1193, opt_names_1194)
            
            # Processing the call keyword arguments (line 382)
            kwargs_1196 = {}
            # Getting the type of 'lines' (line 382)
            lines_1191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 20), 'lines', False)
            # Obtaining the member 'append' of a type (line 382)
            append_1192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 20), lines_1191, 'append')
            # Calling append(args, kwargs) (line 382)
            append_call_result_1197 = invoke(stypy.reporting.localization.Localization(__file__, 382, 20), append_1192, *[result_mod_1195], **kwargs_1196)
            
            # SSA join for if statement (line 378)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_1145 and more_types_in_union_1146):
                # SSA join for if statement (line 368)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Obtaining the type of the subscript
        int_1198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 26), 'int')
        slice_1199 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 384, 21), int_1198, None, None)
        # Getting the type of 'text' (line 384)
        text_1200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 21), 'text')
        # Obtaining the member '__getitem__' of a type (line 384)
        getitem___1201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 21), text_1200, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 384)
        subscript_call_result_1202 = invoke(stypy.reporting.localization.Localization(__file__, 384, 21), getitem___1201, slice_1199)
        
        # Testing the type of a for loop iterable (line 384)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 384, 12), subscript_call_result_1202)
        # Getting the type of the for loop variable (line 384)
        for_loop_var_1203 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 384, 12), subscript_call_result_1202)
        # Assigning a type to the variable 'l' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'l', for_loop_var_1203)
        # SSA begins for a for statement (line 384)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 385)
        # Processing the call arguments (line 385)
        # Getting the type of 'big_indent' (line 385)
        big_indent_1206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 29), 'big_indent', False)
        # Getting the type of 'l' (line 385)
        l_1207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 42), 'l', False)
        # Applying the binary operator '+' (line 385)
        result_add_1208 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 29), '+', big_indent_1206, l_1207)
        
        # Processing the call keyword arguments (line 385)
        kwargs_1209 = {}
        # Getting the type of 'lines' (line 385)
        lines_1204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 16), 'lines', False)
        # Obtaining the member 'append' of a type (line 385)
        append_1205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 16), lines_1204, 'append')
        # Calling append(args, kwargs) (line 385)
        append_call_result_1210 = invoke(stypy.reporting.localization.Localization(__file__, 385, 16), append_1205, *[result_add_1208], **kwargs_1209)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'lines' (line 389)
        lines_1211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 15), 'lines')
        # Assigning a type to the variable 'stypy_return_type' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'stypy_return_type', lines_1211)
        
        # ################# End of 'generate_help(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'generate_help' in the type store
        # Getting the type of 'stypy_return_type' (line 309)
        stypy_return_type_1212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1212)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'generate_help'
        return stypy_return_type_1212


    @norecursion
    def print_help(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 393)
        None_1213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 33), 'None')
        # Getting the type of 'None' (line 393)
        None_1214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 44), 'None')
        defaults = [None_1213, None_1214]
        # Create a new context for function 'print_help'
        module_type_store = module_type_store.open_function_context('print_help', 393, 4, False)
        # Assigning a type to the variable 'self' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FancyGetopt.print_help.__dict__.__setitem__('stypy_localization', localization)
        FancyGetopt.print_help.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FancyGetopt.print_help.__dict__.__setitem__('stypy_type_store', module_type_store)
        FancyGetopt.print_help.__dict__.__setitem__('stypy_function_name', 'FancyGetopt.print_help')
        FancyGetopt.print_help.__dict__.__setitem__('stypy_param_names_list', ['header', 'file'])
        FancyGetopt.print_help.__dict__.__setitem__('stypy_varargs_param_name', None)
        FancyGetopt.print_help.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FancyGetopt.print_help.__dict__.__setitem__('stypy_call_defaults', defaults)
        FancyGetopt.print_help.__dict__.__setitem__('stypy_call_varargs', varargs)
        FancyGetopt.print_help.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FancyGetopt.print_help.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FancyGetopt.print_help', ['header', 'file'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'print_help', localization, ['header', 'file'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'print_help(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 394)
        # Getting the type of 'file' (line 394)
        file_1215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 11), 'file')
        # Getting the type of 'None' (line 394)
        None_1216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 19), 'None')
        
        (may_be_1217, more_types_in_union_1218) = may_be_none(file_1215, None_1216)

        if may_be_1217:

            if more_types_in_union_1218:
                # Runtime conditional SSA (line 394)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 395):
            
            # Assigning a Attribute to a Name (line 395):
            # Getting the type of 'sys' (line 395)
            sys_1219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 19), 'sys')
            # Obtaining the member 'stdout' of a type (line 395)
            stdout_1220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 19), sys_1219, 'stdout')
            # Assigning a type to the variable 'file' (line 395)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'file', stdout_1220)

            if more_types_in_union_1218:
                # SSA join for if statement (line 394)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Call to generate_help(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 'header' (line 396)
        header_1223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 39), 'header', False)
        # Processing the call keyword arguments (line 396)
        kwargs_1224 = {}
        # Getting the type of 'self' (line 396)
        self_1221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 20), 'self', False)
        # Obtaining the member 'generate_help' of a type (line 396)
        generate_help_1222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 20), self_1221, 'generate_help')
        # Calling generate_help(args, kwargs) (line 396)
        generate_help_call_result_1225 = invoke(stypy.reporting.localization.Localization(__file__, 396, 20), generate_help_1222, *[header_1223], **kwargs_1224)
        
        # Testing the type of a for loop iterable (line 396)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 396, 8), generate_help_call_result_1225)
        # Getting the type of the for loop variable (line 396)
        for_loop_var_1226 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 396, 8), generate_help_call_result_1225)
        # Assigning a type to the variable 'line' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'line', for_loop_var_1226)
        # SSA begins for a for statement (line 396)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to write(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 'line' (line 397)
        line_1229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 23), 'line', False)
        str_1230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 30), 'str', '\n')
        # Applying the binary operator '+' (line 397)
        result_add_1231 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 23), '+', line_1229, str_1230)
        
        # Processing the call keyword arguments (line 397)
        kwargs_1232 = {}
        # Getting the type of 'file' (line 397)
        file_1227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'file', False)
        # Obtaining the member 'write' of a type (line 397)
        write_1228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 12), file_1227, 'write')
        # Calling write(args, kwargs) (line 397)
        write_call_result_1233 = invoke(stypy.reporting.localization.Localization(__file__, 397, 12), write_1228, *[result_add_1231], **kwargs_1232)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'print_help(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'print_help' in the type store
        # Getting the type of 'stypy_return_type' (line 393)
        stypy_return_type_1234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1234)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'print_help'
        return stypy_return_type_1234


# Assigning a type to the variable 'FancyGetopt' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'FancyGetopt', FancyGetopt)

@norecursion
def fancy_getopt(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fancy_getopt'
    module_type_store = module_type_store.open_function_context('fancy_getopt', 402, 0, False)
    
    # Passed parameters checking function
    fancy_getopt.stypy_localization = localization
    fancy_getopt.stypy_type_of_self = None
    fancy_getopt.stypy_type_store = module_type_store
    fancy_getopt.stypy_function_name = 'fancy_getopt'
    fancy_getopt.stypy_param_names_list = ['options', 'negative_opt', 'object', 'args']
    fancy_getopt.stypy_varargs_param_name = None
    fancy_getopt.stypy_kwargs_param_name = None
    fancy_getopt.stypy_call_defaults = defaults
    fancy_getopt.stypy_call_varargs = varargs
    fancy_getopt.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fancy_getopt', ['options', 'negative_opt', 'object', 'args'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fancy_getopt', localization, ['options', 'negative_opt', 'object', 'args'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fancy_getopt(...)' code ##################

    
    # Assigning a Call to a Name (line 403):
    
    # Assigning a Call to a Name (line 403):
    
    # Call to FancyGetopt(...): (line 403)
    # Processing the call arguments (line 403)
    # Getting the type of 'options' (line 403)
    options_1236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 25), 'options', False)
    # Processing the call keyword arguments (line 403)
    kwargs_1237 = {}
    # Getting the type of 'FancyGetopt' (line 403)
    FancyGetopt_1235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 13), 'FancyGetopt', False)
    # Calling FancyGetopt(args, kwargs) (line 403)
    FancyGetopt_call_result_1238 = invoke(stypy.reporting.localization.Localization(__file__, 403, 13), FancyGetopt_1235, *[options_1236], **kwargs_1237)
    
    # Assigning a type to the variable 'parser' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'parser', FancyGetopt_call_result_1238)
    
    # Call to set_negative_aliases(...): (line 404)
    # Processing the call arguments (line 404)
    # Getting the type of 'negative_opt' (line 404)
    negative_opt_1241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 32), 'negative_opt', False)
    # Processing the call keyword arguments (line 404)
    kwargs_1242 = {}
    # Getting the type of 'parser' (line 404)
    parser_1239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'parser', False)
    # Obtaining the member 'set_negative_aliases' of a type (line 404)
    set_negative_aliases_1240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 4), parser_1239, 'set_negative_aliases')
    # Calling set_negative_aliases(args, kwargs) (line 404)
    set_negative_aliases_call_result_1243 = invoke(stypy.reporting.localization.Localization(__file__, 404, 4), set_negative_aliases_1240, *[negative_opt_1241], **kwargs_1242)
    
    
    # Call to getopt(...): (line 405)
    # Processing the call arguments (line 405)
    # Getting the type of 'args' (line 405)
    args_1246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 25), 'args', False)
    # Getting the type of 'object' (line 405)
    object_1247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 31), 'object', False)
    # Processing the call keyword arguments (line 405)
    kwargs_1248 = {}
    # Getting the type of 'parser' (line 405)
    parser_1244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 11), 'parser', False)
    # Obtaining the member 'getopt' of a type (line 405)
    getopt_1245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 11), parser_1244, 'getopt')
    # Calling getopt(args, kwargs) (line 405)
    getopt_call_result_1249 = invoke(stypy.reporting.localization.Localization(__file__, 405, 11), getopt_1245, *[args_1246, object_1247], **kwargs_1248)
    
    # Assigning a type to the variable 'stypy_return_type' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'stypy_return_type', getopt_call_result_1249)
    
    # ################# End of 'fancy_getopt(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fancy_getopt' in the type store
    # Getting the type of 'stypy_return_type' (line 402)
    stypy_return_type_1250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1250)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fancy_getopt'
    return stypy_return_type_1250

# Assigning a type to the variable 'fancy_getopt' (line 402)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 0), 'fancy_getopt', fancy_getopt)

# Assigning a Call to a Name (line 408):

# Assigning a Call to a Name (line 408):

# Call to maketrans(...): (line 408)
# Processing the call arguments (line 408)
# Getting the type of 'string' (line 408)
string_1253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 28), 'string', False)
# Obtaining the member 'whitespace' of a type (line 408)
whitespace_1254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 28), string_1253, 'whitespace')
str_1255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 47), 'str', ' ')

# Call to len(...): (line 408)
# Processing the call arguments (line 408)
# Getting the type of 'string' (line 408)
string_1257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 57), 'string', False)
# Obtaining the member 'whitespace' of a type (line 408)
whitespace_1258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 57), string_1257, 'whitespace')
# Processing the call keyword arguments (line 408)
kwargs_1259 = {}
# Getting the type of 'len' (line 408)
len_1256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 53), 'len', False)
# Calling len(args, kwargs) (line 408)
len_call_result_1260 = invoke(stypy.reporting.localization.Localization(__file__, 408, 53), len_1256, *[whitespace_1258], **kwargs_1259)

# Applying the binary operator '*' (line 408)
result_mul_1261 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 47), '*', str_1255, len_call_result_1260)

# Processing the call keyword arguments (line 408)
kwargs_1262 = {}
# Getting the type of 'string' (line 408)
string_1251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 11), 'string', False)
# Obtaining the member 'maketrans' of a type (line 408)
maketrans_1252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 11), string_1251, 'maketrans')
# Calling maketrans(args, kwargs) (line 408)
maketrans_call_result_1263 = invoke(stypy.reporting.localization.Localization(__file__, 408, 11), maketrans_1252, *[whitespace_1254, result_mul_1261], **kwargs_1262)

# Assigning a type to the variable 'WS_TRANS' (line 408)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 0), 'WS_TRANS', maketrans_call_result_1263)

@norecursion
def wrap_text(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'wrap_text'
    module_type_store = module_type_store.open_function_context('wrap_text', 410, 0, False)
    
    # Passed parameters checking function
    wrap_text.stypy_localization = localization
    wrap_text.stypy_type_of_self = None
    wrap_text.stypy_type_store = module_type_store
    wrap_text.stypy_function_name = 'wrap_text'
    wrap_text.stypy_param_names_list = ['text', 'width']
    wrap_text.stypy_varargs_param_name = None
    wrap_text.stypy_kwargs_param_name = None
    wrap_text.stypy_call_defaults = defaults
    wrap_text.stypy_call_varargs = varargs
    wrap_text.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'wrap_text', ['text', 'width'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'wrap_text', localization, ['text', 'width'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'wrap_text(...)' code ##################

    str_1264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, (-1)), 'str', "wrap_text(text : string, width : int) -> [string]\n\n    Split 'text' into multiple lines of no more than 'width' characters\n    each, and return the list of strings that results.\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 417)
    # Getting the type of 'text' (line 417)
    text_1265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 7), 'text')
    # Getting the type of 'None' (line 417)
    None_1266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 15), 'None')
    
    (may_be_1267, more_types_in_union_1268) = may_be_none(text_1265, None_1266)

    if may_be_1267:

        if more_types_in_union_1268:
            # Runtime conditional SSA (line 417)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Obtaining an instance of the builtin type 'list' (line 418)
        list_1269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 418)
        
        # Assigning a type to the variable 'stypy_return_type' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'stypy_return_type', list_1269)

        if more_types_in_union_1268:
            # SSA join for if statement (line 417)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    # Call to len(...): (line 419)
    # Processing the call arguments (line 419)
    # Getting the type of 'text' (line 419)
    text_1271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 11), 'text', False)
    # Processing the call keyword arguments (line 419)
    kwargs_1272 = {}
    # Getting the type of 'len' (line 419)
    len_1270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 7), 'len', False)
    # Calling len(args, kwargs) (line 419)
    len_call_result_1273 = invoke(stypy.reporting.localization.Localization(__file__, 419, 7), len_1270, *[text_1271], **kwargs_1272)
    
    # Getting the type of 'width' (line 419)
    width_1274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 20), 'width')
    # Applying the binary operator '<=' (line 419)
    result_le_1275 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 7), '<=', len_call_result_1273, width_1274)
    
    # Testing the type of an if condition (line 419)
    if_condition_1276 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 419, 4), result_le_1275)
    # Assigning a type to the variable 'if_condition_1276' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'if_condition_1276', if_condition_1276)
    # SSA begins for if statement (line 419)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'list' (line 420)
    list_1277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 420)
    # Adding element type (line 420)
    # Getting the type of 'text' (line 420)
    text_1278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 16), 'text')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 15), list_1277, text_1278)
    
    # Assigning a type to the variable 'stypy_return_type' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'stypy_return_type', list_1277)
    # SSA join for if statement (line 419)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 422):
    
    # Assigning a Call to a Name (line 422):
    
    # Call to expandtabs(...): (line 422)
    # Processing the call arguments (line 422)
    # Getting the type of 'text' (line 422)
    text_1281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 29), 'text', False)
    # Processing the call keyword arguments (line 422)
    kwargs_1282 = {}
    # Getting the type of 'string' (line 422)
    string_1279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 11), 'string', False)
    # Obtaining the member 'expandtabs' of a type (line 422)
    expandtabs_1280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 11), string_1279, 'expandtabs')
    # Calling expandtabs(args, kwargs) (line 422)
    expandtabs_call_result_1283 = invoke(stypy.reporting.localization.Localization(__file__, 422, 11), expandtabs_1280, *[text_1281], **kwargs_1282)
    
    # Assigning a type to the variable 'text' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'text', expandtabs_call_result_1283)
    
    # Assigning a Call to a Name (line 423):
    
    # Assigning a Call to a Name (line 423):
    
    # Call to translate(...): (line 423)
    # Processing the call arguments (line 423)
    # Getting the type of 'text' (line 423)
    text_1286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 28), 'text', False)
    # Getting the type of 'WS_TRANS' (line 423)
    WS_TRANS_1287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 34), 'WS_TRANS', False)
    # Processing the call keyword arguments (line 423)
    kwargs_1288 = {}
    # Getting the type of 'string' (line 423)
    string_1284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 11), 'string', False)
    # Obtaining the member 'translate' of a type (line 423)
    translate_1285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 11), string_1284, 'translate')
    # Calling translate(args, kwargs) (line 423)
    translate_call_result_1289 = invoke(stypy.reporting.localization.Localization(__file__, 423, 11), translate_1285, *[text_1286, WS_TRANS_1287], **kwargs_1288)
    
    # Assigning a type to the variable 'text' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'text', translate_call_result_1289)
    
    # Assigning a Call to a Name (line 424):
    
    # Assigning a Call to a Name (line 424):
    
    # Call to split(...): (line 424)
    # Processing the call arguments (line 424)
    str_1292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 22), 'str', '( +|-+)')
    # Getting the type of 'text' (line 424)
    text_1293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 34), 'text', False)
    # Processing the call keyword arguments (line 424)
    kwargs_1294 = {}
    # Getting the type of 're' (line 424)
    re_1290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 13), 're', False)
    # Obtaining the member 'split' of a type (line 424)
    split_1291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 13), re_1290, 'split')
    # Calling split(args, kwargs) (line 424)
    split_call_result_1295 = invoke(stypy.reporting.localization.Localization(__file__, 424, 13), split_1291, *[str_1292, text_1293], **kwargs_1294)
    
    # Assigning a type to the variable 'chunks' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'chunks', split_call_result_1295)
    
    # Assigning a Call to a Name (line 425):
    
    # Assigning a Call to a Name (line 425):
    
    # Call to filter(...): (line 425)
    # Processing the call arguments (line 425)
    # Getting the type of 'None' (line 425)
    None_1297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 20), 'None', False)
    # Getting the type of 'chunks' (line 425)
    chunks_1298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 26), 'chunks', False)
    # Processing the call keyword arguments (line 425)
    kwargs_1299 = {}
    # Getting the type of 'filter' (line 425)
    filter_1296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 13), 'filter', False)
    # Calling filter(args, kwargs) (line 425)
    filter_call_result_1300 = invoke(stypy.reporting.localization.Localization(__file__, 425, 13), filter_1296, *[None_1297, chunks_1298], **kwargs_1299)
    
    # Assigning a type to the variable 'chunks' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'chunks', filter_call_result_1300)
    
    # Assigning a List to a Name (line 426):
    
    # Assigning a List to a Name (line 426):
    
    # Obtaining an instance of the builtin type 'list' (line 426)
    list_1301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 426)
    
    # Assigning a type to the variable 'lines' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'lines', list_1301)
    
    # Getting the type of 'chunks' (line 428)
    chunks_1302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 10), 'chunks')
    # Testing the type of an if condition (line 428)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 428, 4), chunks_1302)
    # SSA begins for while statement (line 428)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a List to a Name (line 430):
    
    # Assigning a List to a Name (line 430):
    
    # Obtaining an instance of the builtin type 'list' (line 430)
    list_1303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 430)
    
    # Assigning a type to the variable 'cur_line' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'cur_line', list_1303)
    
    # Assigning a Num to a Name (line 431):
    
    # Assigning a Num to a Name (line 431):
    int_1304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 18), 'int')
    # Assigning a type to the variable 'cur_len' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'cur_len', int_1304)
    
    # Getting the type of 'chunks' (line 433)
    chunks_1305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 14), 'chunks')
    # Testing the type of an if condition (line 433)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 433, 8), chunks_1305)
    # SSA begins for while statement (line 433)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 434):
    
    # Assigning a Call to a Name (line 434):
    
    # Call to len(...): (line 434)
    # Processing the call arguments (line 434)
    
    # Obtaining the type of the subscript
    int_1307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 27), 'int')
    # Getting the type of 'chunks' (line 434)
    chunks_1308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 20), 'chunks', False)
    # Obtaining the member '__getitem__' of a type (line 434)
    getitem___1309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 20), chunks_1308, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 434)
    subscript_call_result_1310 = invoke(stypy.reporting.localization.Localization(__file__, 434, 20), getitem___1309, int_1307)
    
    # Processing the call keyword arguments (line 434)
    kwargs_1311 = {}
    # Getting the type of 'len' (line 434)
    len_1306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 16), 'len', False)
    # Calling len(args, kwargs) (line 434)
    len_call_result_1312 = invoke(stypy.reporting.localization.Localization(__file__, 434, 16), len_1306, *[subscript_call_result_1310], **kwargs_1311)
    
    # Assigning a type to the variable 'l' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'l', len_call_result_1312)
    
    
    # Getting the type of 'cur_len' (line 435)
    cur_len_1313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 15), 'cur_len')
    # Getting the type of 'l' (line 435)
    l_1314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 25), 'l')
    # Applying the binary operator '+' (line 435)
    result_add_1315 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 15), '+', cur_len_1313, l_1314)
    
    # Getting the type of 'width' (line 435)
    width_1316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 30), 'width')
    # Applying the binary operator '<=' (line 435)
    result_le_1317 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 15), '<=', result_add_1315, width_1316)
    
    # Testing the type of an if condition (line 435)
    if_condition_1318 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 435, 12), result_le_1317)
    # Assigning a type to the variable 'if_condition_1318' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'if_condition_1318', if_condition_1318)
    # SSA begins for if statement (line 435)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 436)
    # Processing the call arguments (line 436)
    
    # Obtaining the type of the subscript
    int_1321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 39), 'int')
    # Getting the type of 'chunks' (line 436)
    chunks_1322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 32), 'chunks', False)
    # Obtaining the member '__getitem__' of a type (line 436)
    getitem___1323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 32), chunks_1322, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 436)
    subscript_call_result_1324 = invoke(stypy.reporting.localization.Localization(__file__, 436, 32), getitem___1323, int_1321)
    
    # Processing the call keyword arguments (line 436)
    kwargs_1325 = {}
    # Getting the type of 'cur_line' (line 436)
    cur_line_1319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 'cur_line', False)
    # Obtaining the member 'append' of a type (line 436)
    append_1320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 16), cur_line_1319, 'append')
    # Calling append(args, kwargs) (line 436)
    append_call_result_1326 = invoke(stypy.reporting.localization.Localization(__file__, 436, 16), append_1320, *[subscript_call_result_1324], **kwargs_1325)
    
    # Deleting a member
    # Getting the type of 'chunks' (line 437)
    chunks_1327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 20), 'chunks')
    
    # Obtaining the type of the subscript
    int_1328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 27), 'int')
    # Getting the type of 'chunks' (line 437)
    chunks_1329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 20), 'chunks')
    # Obtaining the member '__getitem__' of a type (line 437)
    getitem___1330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 20), chunks_1329, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 437)
    subscript_call_result_1331 = invoke(stypy.reporting.localization.Localization(__file__, 437, 20), getitem___1330, int_1328)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 16), chunks_1327, subscript_call_result_1331)
    
    # Assigning a BinOp to a Name (line 438):
    
    # Assigning a BinOp to a Name (line 438):
    # Getting the type of 'cur_len' (line 438)
    cur_len_1332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 26), 'cur_len')
    # Getting the type of 'l' (line 438)
    l_1333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 36), 'l')
    # Applying the binary operator '+' (line 438)
    result_add_1334 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 26), '+', cur_len_1332, l_1333)
    
    # Assigning a type to the variable 'cur_len' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 'cur_len', result_add_1334)
    # SSA branch for the else part of an if statement (line 435)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    # Getting the type of 'cur_line' (line 441)
    cur_line_1335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 19), 'cur_line')
    
    
    # Obtaining the type of the subscript
    int_1336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 45), 'int')
    
    # Obtaining the type of the subscript
    int_1337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 41), 'int')
    # Getting the type of 'cur_line' (line 441)
    cur_line_1338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 32), 'cur_line')
    # Obtaining the member '__getitem__' of a type (line 441)
    getitem___1339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 32), cur_line_1338, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 441)
    subscript_call_result_1340 = invoke(stypy.reporting.localization.Localization(__file__, 441, 32), getitem___1339, int_1337)
    
    # Obtaining the member '__getitem__' of a type (line 441)
    getitem___1341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 32), subscript_call_result_1340, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 441)
    subscript_call_result_1342 = invoke(stypy.reporting.localization.Localization(__file__, 441, 32), getitem___1341, int_1336)
    
    str_1343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 51), 'str', ' ')
    # Applying the binary operator '==' (line 441)
    result_eq_1344 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 32), '==', subscript_call_result_1342, str_1343)
    
    # Applying the binary operator 'and' (line 441)
    result_and_keyword_1345 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 19), 'and', cur_line_1335, result_eq_1344)
    
    # Testing the type of an if condition (line 441)
    if_condition_1346 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 441, 16), result_and_keyword_1345)
    # Assigning a type to the variable 'if_condition_1346' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 16), 'if_condition_1346', if_condition_1346)
    # SSA begins for if statement (line 441)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Deleting a member
    # Getting the type of 'cur_line' (line 442)
    cur_line_1347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 24), 'cur_line')
    
    # Obtaining the type of the subscript
    int_1348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 33), 'int')
    # Getting the type of 'cur_line' (line 442)
    cur_line_1349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 24), 'cur_line')
    # Obtaining the member '__getitem__' of a type (line 442)
    getitem___1350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 24), cur_line_1349, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 442)
    subscript_call_result_1351 = invoke(stypy.reporting.localization.Localization(__file__, 442, 24), getitem___1350, int_1348)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 20), cur_line_1347, subscript_call_result_1351)
    # SSA join for if statement (line 441)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 435)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 433)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'chunks' (line 445)
    chunks_1352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 11), 'chunks')
    # Testing the type of an if condition (line 445)
    if_condition_1353 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 445, 8), chunks_1352)
    # Assigning a type to the variable 'if_condition_1353' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'if_condition_1353', if_condition_1353)
    # SSA begins for if statement (line 445)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'cur_len' (line 450)
    cur_len_1354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 15), 'cur_len')
    int_1355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 26), 'int')
    # Applying the binary operator '==' (line 450)
    result_eq_1356 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 15), '==', cur_len_1354, int_1355)
    
    # Testing the type of an if condition (line 450)
    if_condition_1357 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 450, 12), result_eq_1356)
    # Assigning a type to the variable 'if_condition_1357' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 12), 'if_condition_1357', if_condition_1357)
    # SSA begins for if statement (line 450)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 451)
    # Processing the call arguments (line 451)
    
    # Obtaining the type of the subscript
    int_1360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 42), 'int')
    # Getting the type of 'width' (line 451)
    width_1361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 44), 'width', False)
    slice_1362 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 451, 32), int_1360, width_1361, None)
    
    # Obtaining the type of the subscript
    int_1363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 39), 'int')
    # Getting the type of 'chunks' (line 451)
    chunks_1364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 32), 'chunks', False)
    # Obtaining the member '__getitem__' of a type (line 451)
    getitem___1365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 32), chunks_1364, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 451)
    subscript_call_result_1366 = invoke(stypy.reporting.localization.Localization(__file__, 451, 32), getitem___1365, int_1363)
    
    # Obtaining the member '__getitem__' of a type (line 451)
    getitem___1367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 32), subscript_call_result_1366, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 451)
    subscript_call_result_1368 = invoke(stypy.reporting.localization.Localization(__file__, 451, 32), getitem___1367, slice_1362)
    
    # Processing the call keyword arguments (line 451)
    kwargs_1369 = {}
    # Getting the type of 'cur_line' (line 451)
    cur_line_1358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 16), 'cur_line', False)
    # Obtaining the member 'append' of a type (line 451)
    append_1359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 16), cur_line_1358, 'append')
    # Calling append(args, kwargs) (line 451)
    append_call_result_1370 = invoke(stypy.reporting.localization.Localization(__file__, 451, 16), append_1359, *[subscript_call_result_1368], **kwargs_1369)
    
    
    # Assigning a Subscript to a Subscript (line 452):
    
    # Assigning a Subscript to a Subscript (line 452):
    
    # Obtaining the type of the subscript
    # Getting the type of 'width' (line 452)
    width_1371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 38), 'width')
    slice_1372 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 452, 28), width_1371, None, None)
    
    # Obtaining the type of the subscript
    int_1373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 35), 'int')
    # Getting the type of 'chunks' (line 452)
    chunks_1374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 28), 'chunks')
    # Obtaining the member '__getitem__' of a type (line 452)
    getitem___1375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 28), chunks_1374, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 452)
    subscript_call_result_1376 = invoke(stypy.reporting.localization.Localization(__file__, 452, 28), getitem___1375, int_1373)
    
    # Obtaining the member '__getitem__' of a type (line 452)
    getitem___1377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 28), subscript_call_result_1376, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 452)
    subscript_call_result_1378 = invoke(stypy.reporting.localization.Localization(__file__, 452, 28), getitem___1377, slice_1372)
    
    # Getting the type of 'chunks' (line 452)
    chunks_1379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 16), 'chunks')
    int_1380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 23), 'int')
    # Storing an element on a container (line 452)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 16), chunks_1379, (int_1380, subscript_call_result_1378))
    # SSA join for if statement (line 450)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_1381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 25), 'int')
    
    # Obtaining the type of the subscript
    int_1382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 22), 'int')
    # Getting the type of 'chunks' (line 457)
    chunks_1383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 15), 'chunks')
    # Obtaining the member '__getitem__' of a type (line 457)
    getitem___1384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 15), chunks_1383, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 457)
    subscript_call_result_1385 = invoke(stypy.reporting.localization.Localization(__file__, 457, 15), getitem___1384, int_1382)
    
    # Obtaining the member '__getitem__' of a type (line 457)
    getitem___1386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 15), subscript_call_result_1385, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 457)
    subscript_call_result_1387 = invoke(stypy.reporting.localization.Localization(__file__, 457, 15), getitem___1386, int_1381)
    
    str_1388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 31), 'str', ' ')
    # Applying the binary operator '==' (line 457)
    result_eq_1389 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 15), '==', subscript_call_result_1387, str_1388)
    
    # Testing the type of an if condition (line 457)
    if_condition_1390 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 457, 12), result_eq_1389)
    # Assigning a type to the variable 'if_condition_1390' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'if_condition_1390', if_condition_1390)
    # SSA begins for if statement (line 457)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Deleting a member
    # Getting the type of 'chunks' (line 458)
    chunks_1391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 20), 'chunks')
    
    # Obtaining the type of the subscript
    int_1392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 27), 'int')
    # Getting the type of 'chunks' (line 458)
    chunks_1393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 20), 'chunks')
    # Obtaining the member '__getitem__' of a type (line 458)
    getitem___1394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 20), chunks_1393, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 458)
    subscript_call_result_1395 = invoke(stypy.reporting.localization.Localization(__file__, 458, 20), getitem___1394, int_1392)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 16), chunks_1391, subscript_call_result_1395)
    # SSA join for if statement (line 457)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 445)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 462)
    # Processing the call arguments (line 462)
    
    # Call to join(...): (line 462)
    # Processing the call arguments (line 462)
    # Getting the type of 'cur_line' (line 462)
    cur_line_1400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 33), 'cur_line', False)
    str_1401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 43), 'str', '')
    # Processing the call keyword arguments (line 462)
    kwargs_1402 = {}
    # Getting the type of 'string' (line 462)
    string_1398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 21), 'string', False)
    # Obtaining the member 'join' of a type (line 462)
    join_1399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 21), string_1398, 'join')
    # Calling join(args, kwargs) (line 462)
    join_call_result_1403 = invoke(stypy.reporting.localization.Localization(__file__, 462, 21), join_1399, *[cur_line_1400, str_1401], **kwargs_1402)
    
    # Processing the call keyword arguments (line 462)
    kwargs_1404 = {}
    # Getting the type of 'lines' (line 462)
    lines_1396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'lines', False)
    # Obtaining the member 'append' of a type (line 462)
    append_1397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 8), lines_1396, 'append')
    # Calling append(args, kwargs) (line 462)
    append_call_result_1405 = invoke(stypy.reporting.localization.Localization(__file__, 462, 8), append_1397, *[join_call_result_1403], **kwargs_1404)
    
    # SSA join for while statement (line 428)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'lines' (line 466)
    lines_1406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 11), 'lines')
    # Assigning a type to the variable 'stypy_return_type' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'stypy_return_type', lines_1406)
    
    # ################# End of 'wrap_text(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'wrap_text' in the type store
    # Getting the type of 'stypy_return_type' (line 410)
    stypy_return_type_1407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1407)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'wrap_text'
    return stypy_return_type_1407

# Assigning a type to the variable 'wrap_text' (line 410)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 0), 'wrap_text', wrap_text)

@norecursion
def translate_longopt(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'translate_longopt'
    module_type_store = module_type_store.open_function_context('translate_longopt', 469, 0, False)
    
    # Passed parameters checking function
    translate_longopt.stypy_localization = localization
    translate_longopt.stypy_type_of_self = None
    translate_longopt.stypy_type_store = module_type_store
    translate_longopt.stypy_function_name = 'translate_longopt'
    translate_longopt.stypy_param_names_list = ['opt']
    translate_longopt.stypy_varargs_param_name = None
    translate_longopt.stypy_kwargs_param_name = None
    translate_longopt.stypy_call_defaults = defaults
    translate_longopt.stypy_call_varargs = varargs
    translate_longopt.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'translate_longopt', ['opt'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'translate_longopt', localization, ['opt'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'translate_longopt(...)' code ##################

    str_1408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, (-1)), 'str', 'Convert a long option name to a valid Python identifier by\n    changing "-" to "_".\n    ')
    
    # Call to translate(...): (line 473)
    # Processing the call arguments (line 473)
    # Getting the type of 'opt' (line 473)
    opt_1411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 28), 'opt', False)
    # Getting the type of 'longopt_xlate' (line 473)
    longopt_xlate_1412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 33), 'longopt_xlate', False)
    # Processing the call keyword arguments (line 473)
    kwargs_1413 = {}
    # Getting the type of 'string' (line 473)
    string_1409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 11), 'string', False)
    # Obtaining the member 'translate' of a type (line 473)
    translate_1410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 11), string_1409, 'translate')
    # Calling translate(args, kwargs) (line 473)
    translate_call_result_1414 = invoke(stypy.reporting.localization.Localization(__file__, 473, 11), translate_1410, *[opt_1411, longopt_xlate_1412], **kwargs_1413)
    
    # Assigning a type to the variable 'stypy_return_type' (line 473)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 4), 'stypy_return_type', translate_call_result_1414)
    
    # ################# End of 'translate_longopt(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'translate_longopt' in the type store
    # Getting the type of 'stypy_return_type' (line 469)
    stypy_return_type_1415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1415)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'translate_longopt'
    return stypy_return_type_1415

# Assigning a type to the variable 'translate_longopt' (line 469)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 0), 'translate_longopt', translate_longopt)
# Declaration of the 'OptionDummy' class

class OptionDummy:
    str_1416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, (-1)), 'str', 'Dummy class just used as a place to hold command-line option\n    values as instance attributes.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Obtaining an instance of the builtin type 'list' (line 480)
        list_1417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 480)
        
        defaults = [list_1417]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 480, 4, False)
        # Assigning a type to the variable 'self' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'OptionDummy.__init__', ['options'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['options'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_1418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, (-1)), 'str', "Create a new OptionDummy instance.  The attributes listed in\n        'options' will be initialized to None.")
        
        # Getting the type of 'options' (line 483)
        options_1419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 19), 'options')
        # Testing the type of a for loop iterable (line 483)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 483, 8), options_1419)
        # Getting the type of the for loop variable (line 483)
        for_loop_var_1420 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 483, 8), options_1419)
        # Assigning a type to the variable 'opt' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'opt', for_loop_var_1420)
        # SSA begins for a for statement (line 483)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to setattr(...): (line 484)
        # Processing the call arguments (line 484)
        # Getting the type of 'self' (line 484)
        self_1422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 20), 'self', False)
        # Getting the type of 'opt' (line 484)
        opt_1423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 26), 'opt', False)
        # Getting the type of 'None' (line 484)
        None_1424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 31), 'None', False)
        # Processing the call keyword arguments (line 484)
        kwargs_1425 = {}
        # Getting the type of 'setattr' (line 484)
        setattr_1421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 12), 'setattr', False)
        # Calling setattr(args, kwargs) (line 484)
        setattr_call_result_1426 = invoke(stypy.reporting.localization.Localization(__file__, 484, 12), setattr_1421, *[self_1422, opt_1423, None_1424], **kwargs_1425)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'OptionDummy' (line 476)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 0), 'OptionDummy', OptionDummy)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
