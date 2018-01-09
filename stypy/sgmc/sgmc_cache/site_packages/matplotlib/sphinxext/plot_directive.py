
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: A directive for including a matplotlib plot in a Sphinx document.
3: 
4: By default, in HTML output, `plot` will include a .png file with a
5: link to a high-res .png and .pdf.  In LaTeX output, it will include a
6: .pdf.
7: 
8: The source code for the plot may be included in one of three ways:
9: 
10:   1. **A path to a source file** as the argument to the directive::
11: 
12:        .. plot:: path/to/plot.py
13: 
14:      When a path to a source file is given, the content of the
15:      directive may optionally contain a caption for the plot::
16: 
17:        .. plot:: path/to/plot.py
18: 
19:           This is the caption for the plot
20: 
21:      Additionally, one may specify the name of a function to call (with
22:      no arguments) immediately after importing the module::
23: 
24:        .. plot:: path/to/plot.py plot_function1
25: 
26:   2. Included as **inline content** to the directive::
27: 
28:        .. plot::
29: 
30:           import matplotlib.pyplot as plt
31:           import matplotlib.image as mpimg
32:           import numpy as np
33:           img = mpimg.imread('_static/stinkbug.png')
34:           imgplot = plt.imshow(img)
35: 
36:   3. Using **doctest** syntax::
37: 
38:        .. plot::
39:           A plotting example:
40:           >>> import matplotlib.pyplot as plt
41:           >>> plt.plot([1,2,3], [4,5,6])
42: 
43: Options
44: -------
45: 
46: The ``plot`` directive supports the following options:
47: 
48:     format : {'python', 'doctest'}
49:         Specify the format of the input
50: 
51:     include-source : bool
52:         Whether to display the source code. The default can be changed
53:         using the `plot_include_source` variable in conf.py
54: 
55:     encoding : str
56:         If this source file is in a non-UTF8 or non-ASCII encoding,
57:         the encoding must be specified using the `:encoding:` option.
58:         The encoding will not be inferred using the ``-*- coding -*-``
59:         metacomment.
60: 
61:     context : bool or str
62:         If provided, the code will be run in the context of all
63:         previous plot directives for which the `:context:` option was
64:         specified.  This only applies to inline code plot directives,
65:         not those run from files. If the ``:context: reset`` option is
66:         specified, the context is reset for this and future plots, and
67:         previous figures are closed prior to running the code.
68:         ``:context:close-figs`` keeps the context but closes previous figures
69:         before running the code.
70: 
71:     nofigs : bool
72:         If specified, the code block will be run, but no figures will
73:         be inserted.  This is usually useful with the ``:context:``
74:         option.
75: 
76: Additionally, this directive supports all of the options of the
77: `image` directive, except for `target` (since plot will add its own
78: target).  These include `alt`, `height`, `width`, `scale`, `align` and
79: `class`.
80: 
81: Configuration options
82: ---------------------
83: 
84: The plot directive has the following configuration options:
85: 
86:     plot_include_source
87:         Default value for the include-source option
88: 
89:     plot_html_show_source_link
90:         Whether to show a link to the source in HTML.
91: 
92:     plot_pre_code
93:         Code that should be executed before each plot.
94: 
95:     plot_basedir
96:         Base directory, to which ``plot::`` file names are relative
97:         to.  (If None or empty, file names are relative to the
98:         directory where the file containing the directive is.)
99: 
100:     plot_formats
101:         File formats to generate. List of tuples or strings::
102: 
103:             [(suffix, dpi), suffix, ...]
104: 
105:         that determine the file format and the DPI. For entries whose
106:         DPI was omitted, sensible defaults are chosen. When passing from
107:         the command line through sphinx_build the list should be passed as
108:         suffix:dpi,suffix:dpi, ....
109: 
110:     plot_html_show_formats
111:         Whether to show links to the files in HTML.
112: 
113:     plot_rcparams
114:         A dictionary containing any non-standard rcParams that should
115:         be applied before each plot.
116: 
117:     plot_apply_rcparams
118:         By default, rcParams are applied when `context` option is not used in
119:         a plot directive.  This configuration option overrides this behavior
120:         and applies rcParams before each plot.
121: 
122:     plot_working_directory
123:         By default, the working directory will be changed to the directory of
124:         the example, so the code can get at its data files, if any.  Also its
125:         path will be added to `sys.path` so it can import any helper modules
126:         sitting beside it.  This configuration option can be used to specify
127:         a central directory (also added to `sys.path`) where data files and
128:         helper modules for all code are located.
129: 
130:     plot_template
131:         Provide a customized template for preparing restructured text.
132: '''
133: from __future__ import (absolute_import, division, print_function,
134:                         unicode_literals)
135: 
136: import six
137: from six.moves import xrange
138: 
139: import sys, os, shutil, io, re, textwrap
140: from os.path import relpath
141: import traceback
142: import warnings
143: 
144: if not six.PY3:
145:     import cStringIO
146: 
147: from docutils.parsers.rst import directives
148: from docutils.parsers.rst.directives.images import Image
149: align = Image.align
150: import sphinx
151: 
152: sphinx_version = sphinx.__version__.split(".")
153: # The split is necessary for sphinx beta versions where the string is
154: # '6b1'
155: sphinx_version = tuple([int(re.split('[^0-9]', x)[0])
156:                         for x in sphinx_version[:2]])
157: 
158: try:
159:     # Sphinx depends on either Jinja or Jinja2
160:     import jinja2
161:     def format_template(template, **kw):
162:         return jinja2.Template(template).render(**kw)
163: except ImportError:
164:     import jinja
165:     def format_template(template, **kw):
166:         return jinja.from_string(template, **kw)
167: 
168: import matplotlib
169: import matplotlib.cbook as cbook
170: try:
171:     with warnings.catch_warnings(record=True):
172:         warnings.simplefilter("error", UserWarning)
173:         matplotlib.use('Agg')
174: except UserWarning:
175:     import matplotlib.pyplot as plt
176:     plt.switch_backend("Agg")
177: else:
178:     import matplotlib.pyplot as plt
179: from matplotlib import _pylab_helpers
180: 
181: __version__ = 2
182: 
183: #------------------------------------------------------------------------------
184: # Registration hook
185: #------------------------------------------------------------------------------
186: 
187: def plot_directive(name, arguments, options, content, lineno,
188:                    content_offset, block_text, state, state_machine):
189:     return run(arguments, content, options, state_machine, state, lineno)
190: plot_directive.__doc__ = __doc__
191: 
192: 
193: def _option_boolean(arg):
194:     if not arg or not arg.strip():
195:         # no argument given, assume used as a flag
196:         return True
197:     elif arg.strip().lower() in ('no', '0', 'false'):
198:         return False
199:     elif arg.strip().lower() in ('yes', '1', 'true'):
200:         return True
201:     else:
202:         raise ValueError('"%s" unknown boolean' % arg)
203: 
204: 
205: def _option_context(arg):
206:     if arg in [None, 'reset', 'close-figs']:
207:         return arg
208:     raise ValueError("argument should be None or 'reset' or 'close-figs'")
209: 
210: 
211: def _option_format(arg):
212:     return directives.choice(arg, ('python', 'doctest'))
213: 
214: 
215: def _option_align(arg):
216:     return directives.choice(arg, ("top", "middle", "bottom", "left", "center",
217:                                    "right"))
218: 
219: 
220: def mark_plot_labels(app, document):
221:     '''
222:     To make plots referenceable, we need to move the reference from
223:     the "htmlonly" (or "latexonly") node to the actual figure node
224:     itself.
225:     '''
226:     for name, explicit in six.iteritems(document.nametypes):
227:         if not explicit:
228:             continue
229:         labelid = document.nameids[name]
230:         if labelid is None:
231:             continue
232:         node = document.ids[labelid]
233:         if node.tagname in ('html_only', 'latex_only'):
234:             for n in node:
235:                 if n.tagname == 'figure':
236:                     sectname = name
237:                     for c in n:
238:                         if c.tagname == 'caption':
239:                             sectname = c.astext()
240:                             break
241: 
242:                     node['ids'].remove(labelid)
243:                     node['names'].remove(name)
244:                     n['ids'].append(labelid)
245:                     n['names'].append(name)
246:                     document.settings.env.labels[name] = \
247:                         document.settings.env.docname, labelid, sectname
248:                     break
249: 
250: 
251: def setup(app):
252:     setup.app = app
253:     setup.config = app.config
254:     setup.confdir = app.confdir
255: 
256:     options = {'alt': directives.unchanged,
257:                'height': directives.length_or_unitless,
258:                'width': directives.length_or_percentage_or_unitless,
259:                'scale': directives.nonnegative_int,
260:                'align': _option_align,
261:                'class': directives.class_option,
262:                'include-source': _option_boolean,
263:                'format': _option_format,
264:                'context': _option_context,
265:                'nofigs': directives.flag,
266:                'encoding': directives.encoding
267:                }
268: 
269:     app.add_directive('plot', plot_directive, True, (0, 2, False), **options)
270:     app.add_config_value('plot_pre_code', None, True)
271:     app.add_config_value('plot_include_source', False, True)
272:     app.add_config_value('plot_html_show_source_link', True, True)
273:     app.add_config_value('plot_formats', ['png', 'hires.png', 'pdf'], True)
274:     app.add_config_value('plot_basedir', None, True)
275:     app.add_config_value('plot_html_show_formats', True, True)
276:     app.add_config_value('plot_rcparams', {}, True)
277:     app.add_config_value('plot_apply_rcparams', False, True)
278:     app.add_config_value('plot_working_directory', None, True)
279:     app.add_config_value('plot_template', None, True)
280: 
281:     app.connect(str('doctree-read'), mark_plot_labels)
282: 
283:     metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
284:     return metadata
285: 
286: #------------------------------------------------------------------------------
287: # Doctest handling
288: #------------------------------------------------------------------------------
289: 
290: def contains_doctest(text):
291:     try:
292:         # check if it's valid Python as-is
293:         compile(text, '<string>', 'exec')
294:         return False
295:     except SyntaxError:
296:         pass
297:     r = re.compile(r'^\s*>>>', re.M)
298:     m = r.search(text)
299:     return bool(m)
300: 
301: 
302: def unescape_doctest(text):
303:     '''
304:     Extract code from a piece of text, which contains either Python code
305:     or doctests.
306: 
307:     '''
308:     if not contains_doctest(text):
309:         return text
310: 
311:     code = ""
312:     for line in text.split("\n"):
313:         m = re.match(r'^\s*(>>>|\.\.\.) (.*)$', line)
314:         if m:
315:             code += m.group(2) + "\n"
316:         elif line.strip():
317:             code += "# " + line.strip() + "\n"
318:         else:
319:             code += "\n"
320:     return code
321: 
322: 
323: def split_code_at_show(text):
324:     '''
325:     Split code at plt.show()
326: 
327:     '''
328: 
329:     parts = []
330:     is_doctest = contains_doctest(text)
331: 
332:     part = []
333:     for line in text.split("\n"):
334:         if (not is_doctest and line.strip() == 'plt.show()') or \
335:                (is_doctest and line.strip() == '>>> plt.show()'):
336:             part.append(line)
337:             parts.append("\n".join(part))
338:             part = []
339:         else:
340:             part.append(line)
341:     if "\n".join(part).strip():
342:         parts.append("\n".join(part))
343:     return parts
344: 
345: 
346: def remove_coding(text):
347:     r'''
348:     Remove the coding comment, which six.exec\_ doesn't like.
349:     '''
350:     sub_re = re.compile("^#\s*-\*-\s*coding:\s*.*-\*-$", flags=re.MULTILINE)
351:     return sub_re.sub("", text)
352: 
353: #------------------------------------------------------------------------------
354: # Template
355: #------------------------------------------------------------------------------
356: 
357: 
358: TEMPLATE = '''
359: {{ source_code }}
360: 
361: {{ only_html }}
362: 
363:    {% if source_link or (html_show_formats and not multi_image) %}
364:    (
365:    {%- if source_link -%}
366:    `Source code <{{ source_link }}>`__
367:    {%- endif -%}
368:    {%- if html_show_formats and not multi_image -%}
369:      {%- for img in images -%}
370:        {%- for fmt in img.formats -%}
371:          {%- if source_link or not loop.first -%}, {% endif -%}
372:          `{{ fmt }} <{{ dest_dir }}/{{ img.basename }}.{{ fmt }}>`__
373:        {%- endfor -%}
374:      {%- endfor -%}
375:    {%- endif -%}
376:    )
377:    {% endif %}
378: 
379:    {% for img in images %}
380:    .. figure:: {{ build_dir }}/{{ img.basename }}.{{ default_fmt }}
381:       {% for option in options -%}
382:       {{ option }}
383:       {% endfor %}
384: 
385:       {% if html_show_formats and multi_image -%}
386:         (
387:         {%- for fmt in img.formats -%}
388:         {%- if not loop.first -%}, {% endif -%}
389:         `{{ fmt }} <{{ dest_dir }}/{{ img.basename }}.{{ fmt }}>`__
390:         {%- endfor -%}
391:         )
392:       {%- endif -%}
393: 
394:       {{ caption }}
395:    {% endfor %}
396: 
397: {{ only_latex }}
398: 
399:    {% for img in images %}
400:    {% if 'pdf' in img.formats -%}
401:    .. figure:: {{ build_dir }}/{{ img.basename }}.pdf
402:       {% for option in options -%}
403:       {{ option }}
404:       {% endfor %}
405: 
406:       {{ caption }}
407:    {% endif -%}
408:    {% endfor %}
409: 
410: {{ only_texinfo }}
411: 
412:    {% for img in images %}
413:    .. image:: {{ build_dir }}/{{ img.basename }}.png
414:       {% for option in options -%}
415:       {{ option }}
416:       {% endfor %}
417: 
418:    {% endfor %}
419: 
420: '''
421: 
422: exception_template = '''
423: .. htmlonly::
424: 
425:    [`source code <%(linkdir)s/%(basename)s.py>`__]
426: 
427: Exception occurred rendering plot.
428: 
429: '''
430: 
431: # the context of the plot for all directives specified with the
432: # :context: option
433: plot_context = dict()
434: 
435: class ImageFile(object):
436:     def __init__(self, basename, dirname):
437:         self.basename = basename
438:         self.dirname = dirname
439:         self.formats = []
440: 
441:     def filename(self, format):
442:         return os.path.join(self.dirname, "%s.%s" % (self.basename, format))
443: 
444:     def filenames(self):
445:         return [self.filename(fmt) for fmt in self.formats]
446: 
447: 
448: def out_of_date(original, derived):
449:     '''
450:     Returns True if derivative is out-of-date wrt original,
451:     both of which are full file paths.
452:     '''
453:     return (not os.path.exists(derived) or
454:             (os.path.exists(original) and
455:              os.stat(derived).st_mtime < os.stat(original).st_mtime))
456: 
457: 
458: class PlotError(RuntimeError):
459:     pass
460: 
461: 
462: def run_code(code, code_path, ns=None, function_name=None):
463:     '''
464:     Import a Python module from a path, and run the function given by
465:     name, if function_name is not None.
466:     '''
467: 
468:     # Change the working directory to the directory of the example, so
469:     # it can get at its data files, if any.  Add its path to sys.path
470:     # so it can import any helper modules sitting beside it.
471:     if six.PY2:
472:         pwd = os.getcwdu()
473:     else:
474:         pwd = os.getcwd()
475:     old_sys_path = list(sys.path)
476:     if setup.config.plot_working_directory is not None:
477:         try:
478:             os.chdir(setup.config.plot_working_directory)
479:         except OSError as err:
480:             raise OSError(str(err) + '\n`plot_working_directory` option in'
481:                           'Sphinx configuration file must be a valid '
482:                           'directory path')
483:         except TypeError as err:
484:             raise TypeError(str(err) + '\n`plot_working_directory` option in '
485:                             'Sphinx configuration file must be a string or '
486:                             'None')
487:         sys.path.insert(0, setup.config.plot_working_directory)
488:     elif code_path is not None:
489:         dirname = os.path.abspath(os.path.dirname(code_path))
490:         os.chdir(dirname)
491:         sys.path.insert(0, dirname)
492: 
493:     # Reset sys.argv
494:     old_sys_argv = sys.argv
495:     sys.argv = [code_path]
496: 
497:     # Redirect stdout
498:     stdout = sys.stdout
499:     if six.PY3:
500:         sys.stdout = io.StringIO()
501:     else:
502:         sys.stdout = cStringIO.StringIO()
503: 
504:     # Assign a do-nothing print function to the namespace.  There
505:     # doesn't seem to be any other way to provide a way to (not) print
506:     # that works correctly across Python 2 and 3.
507:     def _dummy_print(*arg, **kwarg):
508:         pass
509: 
510:     try:
511:         try:
512:             code = unescape_doctest(code)
513:             if ns is None:
514:                 ns = {}
515:             if not ns:
516:                 if setup.config.plot_pre_code is None:
517:                     six.exec_(six.text_type("import numpy as np\n" +
518:                     "from matplotlib import pyplot as plt\n"), ns)
519:                 else:
520:                     six.exec_(six.text_type(setup.config.plot_pre_code), ns)
521:             ns['print'] = _dummy_print
522:             if "__main__" in code:
523:                 six.exec_("__name__ = '__main__'", ns)
524:             code = remove_coding(code)
525:             six.exec_(code, ns)
526:             if function_name is not None:
527:                 six.exec_(function_name + "()", ns)
528:         except (Exception, SystemExit) as err:
529:             raise PlotError(traceback.format_exc())
530:     finally:
531:         os.chdir(pwd)
532:         sys.argv = old_sys_argv
533:         sys.path[:] = old_sys_path
534:         sys.stdout = stdout
535:     return ns
536: 
537: 
538: def clear_state(plot_rcparams, close=True):
539:     if close:
540:         plt.close('all')
541:     matplotlib.rc_file_defaults()
542:     matplotlib.rcParams.update(plot_rcparams)
543: 
544: 
545: def get_plot_formats(config):
546:     default_dpi = {'png': 80, 'hires.png': 200, 'pdf': 200}
547:     formats = []
548:     plot_formats = config.plot_formats
549:     if isinstance(plot_formats, six.string_types):
550:         # String Sphinx < 1.3, Split on , to mimic
551:         # Sphinx 1.3 and later. Sphinx 1.3 always
552:         # returns a list.
553:         plot_formats = plot_formats.split(',')
554:     for fmt in plot_formats:
555:         if isinstance(fmt, six.string_types):
556:             if ':' in fmt:
557:                 suffix, dpi = fmt.split(':')
558:                 formats.append((str(suffix), int(dpi)))
559:             else:
560:                 formats.append((fmt, default_dpi.get(fmt, 80)))
561:         elif type(fmt) in (tuple, list) and len(fmt) == 2:
562:             formats.append((str(fmt[0]), int(fmt[1])))
563:         else:
564:             raise PlotError('invalid image format "%r" in plot_formats' % fmt)
565:     return formats
566: 
567: 
568: def render_figures(code, code_path, output_dir, output_base, context,
569:                    function_name, config, context_reset=False,
570:                    close_figs=False):
571:     '''
572:     Run a pyplot script and save the images in *output_dir*.
573: 
574:     Save the images under *output_dir* with file names derived from
575:     *output_base*
576:     '''
577:     formats = get_plot_formats(config)
578: 
579:     # -- Try to determine if all images already exist
580: 
581:     code_pieces = split_code_at_show(code)
582: 
583:     # Look for single-figure output files first
584:     all_exists = True
585:     img = ImageFile(output_base, output_dir)
586:     for format, dpi in formats:
587:         if out_of_date(code_path, img.filename(format)):
588:             all_exists = False
589:             break
590:         img.formats.append(format)
591: 
592:     if all_exists:
593:         return [(code, [img])]
594: 
595:     # Then look for multi-figure output files
596:     results = []
597:     all_exists = True
598:     for i, code_piece in enumerate(code_pieces):
599:         images = []
600:         for j in xrange(1000):
601:             if len(code_pieces) > 1:
602:                 img = ImageFile('%s_%02d_%02d' % (output_base, i, j), output_dir)
603:             else:
604:                 img = ImageFile('%s_%02d' % (output_base, j), output_dir)
605:             for format, dpi in formats:
606:                 if out_of_date(code_path, img.filename(format)):
607:                     all_exists = False
608:                     break
609:                 img.formats.append(format)
610: 
611:             # assume that if we have one, we have them all
612:             if not all_exists:
613:                 all_exists = (j > 0)
614:                 break
615:             images.append(img)
616:         if not all_exists:
617:             break
618:         results.append((code_piece, images))
619: 
620:     if all_exists:
621:         return results
622: 
623:     # We didn't find the files, so build them
624: 
625:     results = []
626:     if context:
627:         ns = plot_context
628:     else:
629:         ns = {}
630: 
631:     if context_reset:
632:         clear_state(config.plot_rcparams)
633:         plot_context.clear()
634: 
635:     close_figs = not context or close_figs
636: 
637:     for i, code_piece in enumerate(code_pieces):
638: 
639:         if not context or config.plot_apply_rcparams:
640:             clear_state(config.plot_rcparams, close_figs)
641:         elif close_figs:
642:             plt.close('all')
643: 
644:         run_code(code_piece, code_path, ns, function_name)
645: 
646:         images = []
647:         fig_managers = _pylab_helpers.Gcf.get_all_fig_managers()
648:         for j, figman in enumerate(fig_managers):
649:             if len(fig_managers) == 1 and len(code_pieces) == 1:
650:                 img = ImageFile(output_base, output_dir)
651:             elif len(code_pieces) == 1:
652:                 img = ImageFile("%s_%02d" % (output_base, j), output_dir)
653:             else:
654:                 img = ImageFile("%s_%02d_%02d" % (output_base, i, j),
655:                                 output_dir)
656:             images.append(img)
657:             for format, dpi in formats:
658:                 try:
659:                     figman.canvas.figure.savefig(img.filename(format), dpi=dpi)
660:                 except Exception as err:
661:                     raise PlotError(traceback.format_exc())
662:                 img.formats.append(format)
663: 
664:         results.append((code_piece, images))
665: 
666:     if not context or config.plot_apply_rcparams:
667:         clear_state(config.plot_rcparams, close=not context)
668: 
669:     return results
670: 
671: 
672: def run(arguments, content, options, state_machine, state, lineno):
673:     document = state_machine.document
674:     config = document.settings.env.config
675:     nofigs = 'nofigs' in options
676: 
677:     formats = get_plot_formats(config)
678:     default_fmt = formats[0][0]
679: 
680:     options.setdefault('include-source', config.plot_include_source)
681:     keep_context = 'context' in options
682:     context_opt = None if not keep_context else options['context']
683: 
684:     rst_file = document.attributes['source']
685:     rst_dir = os.path.dirname(rst_file)
686: 
687:     if len(arguments):
688:         if not config.plot_basedir:
689:             source_file_name = os.path.join(setup.app.builder.srcdir,
690:                                             directives.uri(arguments[0]))
691:         else:
692:             source_file_name = os.path.join(setup.confdir, config.plot_basedir,
693:                                             directives.uri(arguments[0]))
694: 
695:         # If there is content, it will be passed as a caption.
696:         caption = '\n'.join(content)
697: 
698:         # If the optional function name is provided, use it
699:         if len(arguments) == 2:
700:             function_name = arguments[1]
701:         else:
702:             function_name = None
703: 
704:         with io.open(source_file_name, 'r', encoding='utf-8') as fd:
705:             code = fd.read()
706:         output_base = os.path.basename(source_file_name)
707:     else:
708:         source_file_name = rst_file
709:         code = textwrap.dedent("\n".join(map(six.text_type, content)))
710:         counter = document.attributes.get('_plot_counter', 0) + 1
711:         document.attributes['_plot_counter'] = counter
712:         base, ext = os.path.splitext(os.path.basename(source_file_name))
713:         output_base = '%s-%d.py' % (base, counter)
714:         function_name = None
715:         caption = ''
716: 
717:     base, source_ext = os.path.splitext(output_base)
718:     if source_ext in ('.py', '.rst', '.txt'):
719:         output_base = base
720:     else:
721:         source_ext = ''
722: 
723:     # ensure that LaTeX includegraphics doesn't choke in foo.bar.pdf filenames
724:     output_base = output_base.replace('.', '-')
725: 
726:     # is it in doctest format?
727:     is_doctest = contains_doctest(code)
728:     if 'format' in options:
729:         if options['format'] == 'python':
730:             is_doctest = False
731:         else:
732:             is_doctest = True
733: 
734:     # determine output directory name fragment
735:     source_rel_name = relpath(source_file_name, setup.confdir)
736:     source_rel_dir = os.path.dirname(source_rel_name)
737:     while source_rel_dir.startswith(os.path.sep):
738:         source_rel_dir = source_rel_dir[1:]
739: 
740:     # build_dir: where to place output files (temporarily)
741:     build_dir = os.path.join(os.path.dirname(setup.app.doctreedir),
742:                              'plot_directive',
743:                              source_rel_dir)
744:     # get rid of .. in paths, also changes pathsep
745:     # see note in Python docs for warning about symbolic links on Windows.
746:     # need to compare source and dest paths at end
747:     build_dir = os.path.normpath(build_dir)
748: 
749:     if not os.path.exists(build_dir):
750:         os.makedirs(build_dir)
751: 
752:     # output_dir: final location in the builder's directory
753:     dest_dir = os.path.abspath(os.path.join(setup.app.builder.outdir,
754:                                             source_rel_dir))
755:     if not os.path.exists(dest_dir):
756:         os.makedirs(dest_dir) # no problem here for me, but just use built-ins
757: 
758:     # how to link to files from the RST file
759:     dest_dir_link = os.path.join(relpath(setup.confdir, rst_dir),
760:                                  source_rel_dir).replace(os.path.sep, '/')
761:     try:
762:         build_dir_link = relpath(build_dir, rst_dir).replace(os.path.sep, '/')
763:     except ValueError:
764:         # on Windows, relpath raises ValueError when path and start are on
765:         # different mounts/drives
766:         build_dir_link = build_dir
767:     source_link = dest_dir_link + '/' + output_base + source_ext
768: 
769:     # make figures
770:     try:
771:         results = render_figures(code,
772:                                  source_file_name,
773:                                  build_dir,
774:                                  output_base,
775:                                  keep_context,
776:                                  function_name,
777:                                  config,
778:                                  context_reset=context_opt == 'reset',
779:                                  close_figs=context_opt == 'close-figs')
780:         errors = []
781:     except PlotError as err:
782:         reporter = state.memo.reporter
783:         sm = reporter.system_message(
784:             2, "Exception occurred in plotting %s\n from %s:\n%s" % (output_base,
785:                                                 source_file_name, err),
786:             line=lineno)
787:         results = [(code, [])]
788:         errors = [sm]
789: 
790:     # Properly indent the caption
791:     caption = '\n'.join('      ' + line.strip()
792:                         for line in caption.split('\n'))
793: 
794:     # generate output restructuredtext
795:     total_lines = []
796:     for j, (code_piece, images) in enumerate(results):
797:         if options['include-source']:
798:             if is_doctest:
799:                 lines = ['']
800:                 lines += [row.rstrip() for row in code_piece.split('\n')]
801:             else:
802:                 lines = ['.. code-block:: python', '']
803:                 lines += ['    %s' % row.rstrip()
804:                           for row in code_piece.split('\n')]
805:             source_code = "\n".join(lines)
806:         else:
807:             source_code = ""
808: 
809:         if nofigs:
810:             images = []
811: 
812:         opts = [':%s: %s' % (key, val) for key, val in six.iteritems(options)
813:                 if key in ('alt', 'height', 'width', 'scale', 'align', 'class')]
814: 
815:         only_html = ".. only:: html"
816:         only_latex = ".. only:: latex"
817:         only_texinfo = ".. only:: texinfo"
818: 
819:         # Not-None src_link signals the need for a source link in the generated
820:         # html
821:         if j == 0 and config.plot_html_show_source_link:
822:             src_link = source_link
823:         else:
824:             src_link = None
825: 
826:         result = format_template(
827:             config.plot_template or TEMPLATE,
828:             default_fmt=default_fmt,
829:             dest_dir=dest_dir_link,
830:             build_dir=build_dir_link,
831:             source_link=src_link,
832:             multi_image=len(images) > 1,
833:             only_html=only_html,
834:             only_latex=only_latex,
835:             only_texinfo=only_texinfo,
836:             options=opts,
837:             images=images,
838:             source_code=source_code,
839:             html_show_formats=config.plot_html_show_formats and len(images),
840:             caption=caption)
841: 
842:         total_lines.extend(result.split("\n"))
843:         total_lines.extend("\n")
844: 
845:     if total_lines:
846:         state_machine.insert_input(total_lines, source=source_file_name)
847: 
848:     # copy image files to builder's output directory, if necessary
849:     if not os.path.exists(dest_dir):
850:         cbook.mkdirs(dest_dir)
851: 
852:     for code_piece, images in results:
853:         for img in images:
854:             for fn in img.filenames():
855:                 destimg = os.path.join(dest_dir, os.path.basename(fn))
856:                 if fn != destimg:
857:                     shutil.copyfile(fn, destimg)
858: 
859:     # copy script (if necessary)
860:     target_name = os.path.join(dest_dir, output_base + source_ext)
861:     with io.open(target_name, 'w', encoding="utf-8") as f:
862:         if source_file_name == rst_file:
863:             code_escaped = unescape_doctest(code)
864:         else:
865:             code_escaped = code
866:         f.write(code_escaped)
867: 
868:     return errors
869: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_285944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, (-1)), 'unicode', u"\nA directive for including a matplotlib plot in a Sphinx document.\n\nBy default, in HTML output, `plot` will include a .png file with a\nlink to a high-res .png and .pdf.  In LaTeX output, it will include a\n.pdf.\n\nThe source code for the plot may be included in one of three ways:\n\n  1. **A path to a source file** as the argument to the directive::\n\n       .. plot:: path/to/plot.py\n\n     When a path to a source file is given, the content of the\n     directive may optionally contain a caption for the plot::\n\n       .. plot:: path/to/plot.py\n\n          This is the caption for the plot\n\n     Additionally, one may specify the name of a function to call (with\n     no arguments) immediately after importing the module::\n\n       .. plot:: path/to/plot.py plot_function1\n\n  2. Included as **inline content** to the directive::\n\n       .. plot::\n\n          import matplotlib.pyplot as plt\n          import matplotlib.image as mpimg\n          import numpy as np\n          img = mpimg.imread('_static/stinkbug.png')\n          imgplot = plt.imshow(img)\n\n  3. Using **doctest** syntax::\n\n       .. plot::\n          A plotting example:\n          >>> import matplotlib.pyplot as plt\n          >>> plt.plot([1,2,3], [4,5,6])\n\nOptions\n-------\n\nThe ``plot`` directive supports the following options:\n\n    format : {'python', 'doctest'}\n        Specify the format of the input\n\n    include-source : bool\n        Whether to display the source code. The default can be changed\n        using the `plot_include_source` variable in conf.py\n\n    encoding : str\n        If this source file is in a non-UTF8 or non-ASCII encoding,\n        the encoding must be specified using the `:encoding:` option.\n        The encoding will not be inferred using the ``-*- coding -*-``\n        metacomment.\n\n    context : bool or str\n        If provided, the code will be run in the context of all\n        previous plot directives for which the `:context:` option was\n        specified.  This only applies to inline code plot directives,\n        not those run from files. If the ``:context: reset`` option is\n        specified, the context is reset for this and future plots, and\n        previous figures are closed prior to running the code.\n        ``:context:close-figs`` keeps the context but closes previous figures\n        before running the code.\n\n    nofigs : bool\n        If specified, the code block will be run, but no figures will\n        be inserted.  This is usually useful with the ``:context:``\n        option.\n\nAdditionally, this directive supports all of the options of the\n`image` directive, except for `target` (since plot will add its own\ntarget).  These include `alt`, `height`, `width`, `scale`, `align` and\n`class`.\n\nConfiguration options\n---------------------\n\nThe plot directive has the following configuration options:\n\n    plot_include_source\n        Default value for the include-source option\n\n    plot_html_show_source_link\n        Whether to show a link to the source in HTML.\n\n    plot_pre_code\n        Code that should be executed before each plot.\n\n    plot_basedir\n        Base directory, to which ``plot::`` file names are relative\n        to.  (If None or empty, file names are relative to the\n        directory where the file containing the directive is.)\n\n    plot_formats\n        File formats to generate. List of tuples or strings::\n\n            [(suffix, dpi), suffix, ...]\n\n        that determine the file format and the DPI. For entries whose\n        DPI was omitted, sensible defaults are chosen. When passing from\n        the command line through sphinx_build the list should be passed as\n        suffix:dpi,suffix:dpi, ....\n\n    plot_html_show_formats\n        Whether to show links to the files in HTML.\n\n    plot_rcparams\n        A dictionary containing any non-standard rcParams that should\n        be applied before each plot.\n\n    plot_apply_rcparams\n        By default, rcParams are applied when `context` option is not used in\n        a plot directive.  This configuration option overrides this behavior\n        and applies rcParams before each plot.\n\n    plot_working_directory\n        By default, the working directory will be changed to the directory of\n        the example, so the code can get at its data files, if any.  Also its\n        path will be added to `sys.path` so it can import any helper modules\n        sitting beside it.  This configuration option can be used to specify\n        a central directory (also added to `sys.path`) where data files and\n        helper modules for all code are located.\n\n    plot_template\n        Provide a customized template for preparing restructured text.\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 136, 0))

# 'import six' statement (line 136)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/sphinxext/')
import_285945 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 136, 0), 'six')

if (type(import_285945) is not StypyTypeError):

    if (import_285945 != 'pyd_module'):
        __import__(import_285945)
        sys_modules_285946 = sys.modules[import_285945]
        import_module(stypy.reporting.localization.Localization(__file__, 136, 0), 'six', sys_modules_285946.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 136, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'six', import_285945)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/sphinxext/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 137, 0))

# 'from six.moves import xrange' statement (line 137)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/sphinxext/')
import_285947 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 137, 0), 'six.moves')

if (type(import_285947) is not StypyTypeError):

    if (import_285947 != 'pyd_module'):
        __import__(import_285947)
        sys_modules_285948 = sys.modules[import_285947]
        import_from_module(stypy.reporting.localization.Localization(__file__, 137, 0), 'six.moves', sys_modules_285948.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 137, 0), __file__, sys_modules_285948, sys_modules_285948.module_type_store, module_type_store)
    else:
        from six.moves import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 137, 0), 'six.moves', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'six.moves' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 0), 'six.moves', import_285947)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/sphinxext/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 139, 0))

# Multiple import statement. import sys (1/6) (line 139)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 139, 0), 'sys', sys, module_type_store)
# Multiple import statement. import os (2/6) (line 139)
import os

import_module(stypy.reporting.localization.Localization(__file__, 139, 0), 'os', os, module_type_store)
# Multiple import statement. import shutil (3/6) (line 139)
import shutil

import_module(stypy.reporting.localization.Localization(__file__, 139, 0), 'shutil', shutil, module_type_store)
# Multiple import statement. import io (4/6) (line 139)
import io

import_module(stypy.reporting.localization.Localization(__file__, 139, 0), 'io', io, module_type_store)
# Multiple import statement. import re (5/6) (line 139)
import re

import_module(stypy.reporting.localization.Localization(__file__, 139, 0), 're', re, module_type_store)
# Multiple import statement. import textwrap (6/6) (line 139)
import textwrap

import_module(stypy.reporting.localization.Localization(__file__, 139, 0), 'textwrap', textwrap, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 140, 0))

# 'from os.path import relpath' statement (line 140)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/sphinxext/')
import_285949 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 140, 0), 'os.path')

if (type(import_285949) is not StypyTypeError):

    if (import_285949 != 'pyd_module'):
        __import__(import_285949)
        sys_modules_285950 = sys.modules[import_285949]
        import_from_module(stypy.reporting.localization.Localization(__file__, 140, 0), 'os.path', sys_modules_285950.module_type_store, module_type_store, ['relpath'])
        nest_module(stypy.reporting.localization.Localization(__file__, 140, 0), __file__, sys_modules_285950, sys_modules_285950.module_type_store, module_type_store)
    else:
        from os.path import relpath

        import_from_module(stypy.reporting.localization.Localization(__file__, 140, 0), 'os.path', None, module_type_store, ['relpath'], [relpath])

else:
    # Assigning a type to the variable 'os.path' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), 'os.path', import_285949)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/sphinxext/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 141, 0))

# 'import traceback' statement (line 141)
import traceback

import_module(stypy.reporting.localization.Localization(__file__, 141, 0), 'traceback', traceback, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 142, 0))

# 'import warnings' statement (line 142)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 142, 0), 'warnings', warnings, module_type_store)



# Getting the type of 'six' (line 144)
six_285951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 7), 'six')
# Obtaining the member 'PY3' of a type (line 144)
PY3_285952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 7), six_285951, 'PY3')
# Applying the 'not' unary operator (line 144)
result_not__285953 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 3), 'not', PY3_285952)

# Testing the type of an if condition (line 144)
if_condition_285954 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 0), result_not__285953)
# Assigning a type to the variable 'if_condition_285954' (line 144)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 0), 'if_condition_285954', if_condition_285954)
# SSA begins for if statement (line 144)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 145, 4))

# 'import cStringIO' statement (line 145)
import cStringIO

import_module(stypy.reporting.localization.Localization(__file__, 145, 4), 'cStringIO', cStringIO, module_type_store)

# SSA join for if statement (line 144)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 147, 0))

# 'from docutils.parsers.rst import directives' statement (line 147)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/sphinxext/')
import_285955 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 147, 0), 'docutils.parsers.rst')

if (type(import_285955) is not StypyTypeError):

    if (import_285955 != 'pyd_module'):
        __import__(import_285955)
        sys_modules_285956 = sys.modules[import_285955]
        import_from_module(stypy.reporting.localization.Localization(__file__, 147, 0), 'docutils.parsers.rst', sys_modules_285956.module_type_store, module_type_store, ['directives'])
        nest_module(stypy.reporting.localization.Localization(__file__, 147, 0), __file__, sys_modules_285956, sys_modules_285956.module_type_store, module_type_store)
    else:
        from docutils.parsers.rst import directives

        import_from_module(stypy.reporting.localization.Localization(__file__, 147, 0), 'docutils.parsers.rst', None, module_type_store, ['directives'], [directives])

else:
    # Assigning a type to the variable 'docutils.parsers.rst' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 0), 'docutils.parsers.rst', import_285955)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/sphinxext/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 148, 0))

# 'from docutils.parsers.rst.directives.images import Image' statement (line 148)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/sphinxext/')
import_285957 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 148, 0), 'docutils.parsers.rst.directives.images')

if (type(import_285957) is not StypyTypeError):

    if (import_285957 != 'pyd_module'):
        __import__(import_285957)
        sys_modules_285958 = sys.modules[import_285957]
        import_from_module(stypy.reporting.localization.Localization(__file__, 148, 0), 'docutils.parsers.rst.directives.images', sys_modules_285958.module_type_store, module_type_store, ['Image'])
        nest_module(stypy.reporting.localization.Localization(__file__, 148, 0), __file__, sys_modules_285958, sys_modules_285958.module_type_store, module_type_store)
    else:
        from docutils.parsers.rst.directives.images import Image

        import_from_module(stypy.reporting.localization.Localization(__file__, 148, 0), 'docutils.parsers.rst.directives.images', None, module_type_store, ['Image'], [Image])

else:
    # Assigning a type to the variable 'docutils.parsers.rst.directives.images' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 0), 'docutils.parsers.rst.directives.images', import_285957)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/sphinxext/')


# Assigning a Attribute to a Name (line 149):

# Assigning a Attribute to a Name (line 149):
# Getting the type of 'Image' (line 149)
Image_285959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'Image')
# Obtaining the member 'align' of a type (line 149)
align_285960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), Image_285959, 'align')
# Assigning a type to the variable 'align' (line 149)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 0), 'align', align_285960)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 150, 0))

# 'import sphinx' statement (line 150)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/sphinxext/')
import_285961 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 150, 0), 'sphinx')

if (type(import_285961) is not StypyTypeError):

    if (import_285961 != 'pyd_module'):
        __import__(import_285961)
        sys_modules_285962 = sys.modules[import_285961]
        import_module(stypy.reporting.localization.Localization(__file__, 150, 0), 'sphinx', sys_modules_285962.module_type_store, module_type_store)
    else:
        import sphinx

        import_module(stypy.reporting.localization.Localization(__file__, 150, 0), 'sphinx', sphinx, module_type_store)

else:
    # Assigning a type to the variable 'sphinx' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 0), 'sphinx', import_285961)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/sphinxext/')


# Assigning a Call to a Name (line 152):

# Assigning a Call to a Name (line 152):

# Call to split(...): (line 152)
# Processing the call arguments (line 152)
unicode_285966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 42), 'unicode', u'.')
# Processing the call keyword arguments (line 152)
kwargs_285967 = {}
# Getting the type of 'sphinx' (line 152)
sphinx_285963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 17), 'sphinx', False)
# Obtaining the member '__version__' of a type (line 152)
version___285964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 17), sphinx_285963, '__version__')
# Obtaining the member 'split' of a type (line 152)
split_285965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 17), version___285964, 'split')
# Calling split(args, kwargs) (line 152)
split_call_result_285968 = invoke(stypy.reporting.localization.Localization(__file__, 152, 17), split_285965, *[unicode_285966], **kwargs_285967)

# Assigning a type to the variable 'sphinx_version' (line 152)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'sphinx_version', split_call_result_285968)

# Assigning a Call to a Name (line 155):

# Assigning a Call to a Name (line 155):

# Call to tuple(...): (line 155)
# Processing the call arguments (line 155)
# Calculating list comprehension
# Calculating comprehension expression

# Obtaining the type of the subscript
int_285982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 49), 'int')
slice_285983 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 156, 33), None, int_285982, None)
# Getting the type of 'sphinx_version' (line 156)
sphinx_version_285984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 33), 'sphinx_version', False)
# Obtaining the member '__getitem__' of a type (line 156)
getitem___285985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 33), sphinx_version_285984, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 156)
subscript_call_result_285986 = invoke(stypy.reporting.localization.Localization(__file__, 156, 33), getitem___285985, slice_285983)

comprehension_285987 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 24), subscript_call_result_285986)
# Assigning a type to the variable 'x' (line 155)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 24), 'x', comprehension_285987)

# Call to int(...): (line 155)
# Processing the call arguments (line 155)

# Obtaining the type of the subscript
int_285971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 50), 'int')

# Call to split(...): (line 155)
# Processing the call arguments (line 155)
unicode_285974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 37), 'unicode', u'[^0-9]')
# Getting the type of 'x' (line 155)
x_285975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 47), 'x', False)
# Processing the call keyword arguments (line 155)
kwargs_285976 = {}
# Getting the type of 're' (line 155)
re_285972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 28), 're', False)
# Obtaining the member 'split' of a type (line 155)
split_285973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 28), re_285972, 'split')
# Calling split(args, kwargs) (line 155)
split_call_result_285977 = invoke(stypy.reporting.localization.Localization(__file__, 155, 28), split_285973, *[unicode_285974, x_285975], **kwargs_285976)

# Obtaining the member '__getitem__' of a type (line 155)
getitem___285978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 28), split_call_result_285977, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 155)
subscript_call_result_285979 = invoke(stypy.reporting.localization.Localization(__file__, 155, 28), getitem___285978, int_285971)

# Processing the call keyword arguments (line 155)
kwargs_285980 = {}
# Getting the type of 'int' (line 155)
int_285970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 24), 'int', False)
# Calling int(args, kwargs) (line 155)
int_call_result_285981 = invoke(stypy.reporting.localization.Localization(__file__, 155, 24), int_285970, *[subscript_call_result_285979], **kwargs_285980)

list_285988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 24), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 24), list_285988, int_call_result_285981)
# Processing the call keyword arguments (line 155)
kwargs_285989 = {}
# Getting the type of 'tuple' (line 155)
tuple_285969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 17), 'tuple', False)
# Calling tuple(args, kwargs) (line 155)
tuple_call_result_285990 = invoke(stypy.reporting.localization.Localization(__file__, 155, 17), tuple_285969, *[list_285988], **kwargs_285989)

# Assigning a type to the variable 'sphinx_version' (line 155)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 0), 'sphinx_version', tuple_call_result_285990)


# SSA begins for try-except statement (line 158)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 160, 4))

# 'import jinja2' statement (line 160)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/sphinxext/')
import_285991 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 160, 4), 'jinja2')

if (type(import_285991) is not StypyTypeError):

    if (import_285991 != 'pyd_module'):
        __import__(import_285991)
        sys_modules_285992 = sys.modules[import_285991]
        import_module(stypy.reporting.localization.Localization(__file__, 160, 4), 'jinja2', sys_modules_285992.module_type_store, module_type_store)
    else:
        import jinja2

        import_module(stypy.reporting.localization.Localization(__file__, 160, 4), 'jinja2', jinja2, module_type_store)

else:
    # Assigning a type to the variable 'jinja2' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'jinja2', import_285991)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/sphinxext/')


@norecursion
def format_template(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'format_template'
    module_type_store = module_type_store.open_function_context('format_template', 161, 4, False)
    
    # Passed parameters checking function
    format_template.stypy_localization = localization
    format_template.stypy_type_of_self = None
    format_template.stypy_type_store = module_type_store
    format_template.stypy_function_name = 'format_template'
    format_template.stypy_param_names_list = ['template']
    format_template.stypy_varargs_param_name = None
    format_template.stypy_kwargs_param_name = 'kw'
    format_template.stypy_call_defaults = defaults
    format_template.stypy_call_varargs = varargs
    format_template.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'format_template', ['template'], None, 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'format_template', localization, ['template'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'format_template(...)' code ##################

    
    # Call to render(...): (line 162)
    # Processing the call keyword arguments (line 162)
    # Getting the type of 'kw' (line 162)
    kw_285999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 50), 'kw', False)
    kwargs_286000 = {'kw_285999': kw_285999}
    
    # Call to Template(...): (line 162)
    # Processing the call arguments (line 162)
    # Getting the type of 'template' (line 162)
    template_285995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 31), 'template', False)
    # Processing the call keyword arguments (line 162)
    kwargs_285996 = {}
    # Getting the type of 'jinja2' (line 162)
    jinja2_285993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 15), 'jinja2', False)
    # Obtaining the member 'Template' of a type (line 162)
    Template_285994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 15), jinja2_285993, 'Template')
    # Calling Template(args, kwargs) (line 162)
    Template_call_result_285997 = invoke(stypy.reporting.localization.Localization(__file__, 162, 15), Template_285994, *[template_285995], **kwargs_285996)
    
    # Obtaining the member 'render' of a type (line 162)
    render_285998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 15), Template_call_result_285997, 'render')
    # Calling render(args, kwargs) (line 162)
    render_call_result_286001 = invoke(stypy.reporting.localization.Localization(__file__, 162, 15), render_285998, *[], **kwargs_286000)
    
    # Assigning a type to the variable 'stypy_return_type' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'stypy_return_type', render_call_result_286001)
    
    # ################# End of 'format_template(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'format_template' in the type store
    # Getting the type of 'stypy_return_type' (line 161)
    stypy_return_type_286002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286002)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'format_template'
    return stypy_return_type_286002

# Assigning a type to the variable 'format_template' (line 161)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'format_template', format_template)
# SSA branch for the except part of a try statement (line 158)
# SSA branch for the except 'ImportError' branch of a try statement (line 158)
module_type_store.open_ssa_branch('except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 164, 4))

# 'import jinja' statement (line 164)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/sphinxext/')
import_286003 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 164, 4), 'jinja')

if (type(import_286003) is not StypyTypeError):

    if (import_286003 != 'pyd_module'):
        __import__(import_286003)
        sys_modules_286004 = sys.modules[import_286003]
        import_module(stypy.reporting.localization.Localization(__file__, 164, 4), 'jinja', sys_modules_286004.module_type_store, module_type_store)
    else:
        import jinja

        import_module(stypy.reporting.localization.Localization(__file__, 164, 4), 'jinja', jinja, module_type_store)

else:
    # Assigning a type to the variable 'jinja' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'jinja', import_286003)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/sphinxext/')


@norecursion
def format_template(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'format_template'
    module_type_store = module_type_store.open_function_context('format_template', 165, 4, False)
    
    # Passed parameters checking function
    format_template.stypy_localization = localization
    format_template.stypy_type_of_self = None
    format_template.stypy_type_store = module_type_store
    format_template.stypy_function_name = 'format_template'
    format_template.stypy_param_names_list = ['template']
    format_template.stypy_varargs_param_name = None
    format_template.stypy_kwargs_param_name = 'kw'
    format_template.stypy_call_defaults = defaults
    format_template.stypy_call_varargs = varargs
    format_template.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'format_template', ['template'], None, 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'format_template', localization, ['template'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'format_template(...)' code ##################

    
    # Call to from_string(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'template' (line 166)
    template_286007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 33), 'template', False)
    # Processing the call keyword arguments (line 166)
    # Getting the type of 'kw' (line 166)
    kw_286008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 45), 'kw', False)
    kwargs_286009 = {'kw_286008': kw_286008}
    # Getting the type of 'jinja' (line 166)
    jinja_286005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 15), 'jinja', False)
    # Obtaining the member 'from_string' of a type (line 166)
    from_string_286006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 15), jinja_286005, 'from_string')
    # Calling from_string(args, kwargs) (line 166)
    from_string_call_result_286010 = invoke(stypy.reporting.localization.Localization(__file__, 166, 15), from_string_286006, *[template_286007], **kwargs_286009)
    
    # Assigning a type to the variable 'stypy_return_type' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'stypy_return_type', from_string_call_result_286010)
    
    # ################# End of 'format_template(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'format_template' in the type store
    # Getting the type of 'stypy_return_type' (line 165)
    stypy_return_type_286011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286011)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'format_template'
    return stypy_return_type_286011

# Assigning a type to the variable 'format_template' (line 165)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'format_template', format_template)
# SSA join for try-except statement (line 158)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 168, 0))

# 'import matplotlib' statement (line 168)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/sphinxext/')
import_286012 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 168, 0), 'matplotlib')

if (type(import_286012) is not StypyTypeError):

    if (import_286012 != 'pyd_module'):
        __import__(import_286012)
        sys_modules_286013 = sys.modules[import_286012]
        import_module(stypy.reporting.localization.Localization(__file__, 168, 0), 'matplotlib', sys_modules_286013.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 168, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), 'matplotlib', import_286012)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/sphinxext/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 169, 0))

# 'import matplotlib.cbook' statement (line 169)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/sphinxext/')
import_286014 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 169, 0), 'matplotlib.cbook')

if (type(import_286014) is not StypyTypeError):

    if (import_286014 != 'pyd_module'):
        __import__(import_286014)
        sys_modules_286015 = sys.modules[import_286014]
        import_module(stypy.reporting.localization.Localization(__file__, 169, 0), 'cbook', sys_modules_286015.module_type_store, module_type_store)
    else:
        import matplotlib.cbook as cbook

        import_module(stypy.reporting.localization.Localization(__file__, 169, 0), 'cbook', matplotlib.cbook, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 0), 'matplotlib.cbook', import_286014)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/sphinxext/')



# SSA begins for try-except statement (line 170)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Call to catch_warnings(...): (line 171)
# Processing the call keyword arguments (line 171)
# Getting the type of 'True' (line 171)
True_286018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 40), 'True', False)
keyword_286019 = True_286018
kwargs_286020 = {'record': keyword_286019}
# Getting the type of 'warnings' (line 171)
warnings_286016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 9), 'warnings', False)
# Obtaining the member 'catch_warnings' of a type (line 171)
catch_warnings_286017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 9), warnings_286016, 'catch_warnings')
# Calling catch_warnings(args, kwargs) (line 171)
catch_warnings_call_result_286021 = invoke(stypy.reporting.localization.Localization(__file__, 171, 9), catch_warnings_286017, *[], **kwargs_286020)

with_286022 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 171, 9), catch_warnings_call_result_286021, 'with parameter', '__enter__', '__exit__')

if with_286022:
    # Calling the __enter__ method to initiate a with section
    # Obtaining the member '__enter__' of a type (line 171)
    enter___286023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 9), catch_warnings_call_result_286021, '__enter__')
    with_enter_286024 = invoke(stypy.reporting.localization.Localization(__file__, 171, 9), enter___286023)
    
    # Call to simplefilter(...): (line 172)
    # Processing the call arguments (line 172)
    unicode_286027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 30), 'unicode', u'error')
    # Getting the type of 'UserWarning' (line 172)
    UserWarning_286028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 39), 'UserWarning', False)
    # Processing the call keyword arguments (line 172)
    kwargs_286029 = {}
    # Getting the type of 'warnings' (line 172)
    warnings_286025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'warnings', False)
    # Obtaining the member 'simplefilter' of a type (line 172)
    simplefilter_286026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), warnings_286025, 'simplefilter')
    # Calling simplefilter(args, kwargs) (line 172)
    simplefilter_call_result_286030 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), simplefilter_286026, *[unicode_286027, UserWarning_286028], **kwargs_286029)
    
    
    # Call to use(...): (line 173)
    # Processing the call arguments (line 173)
    unicode_286033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 23), 'unicode', u'Agg')
    # Processing the call keyword arguments (line 173)
    kwargs_286034 = {}
    # Getting the type of 'matplotlib' (line 173)
    matplotlib_286031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'matplotlib', False)
    # Obtaining the member 'use' of a type (line 173)
    use_286032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), matplotlib_286031, 'use')
    # Calling use(args, kwargs) (line 173)
    use_call_result_286035 = invoke(stypy.reporting.localization.Localization(__file__, 173, 8), use_286032, *[unicode_286033], **kwargs_286034)
    
    # Calling the __exit__ method to finish a with section
    # Obtaining the member '__exit__' of a type (line 171)
    exit___286036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 9), catch_warnings_call_result_286021, '__exit__')
    with_exit_286037 = invoke(stypy.reporting.localization.Localization(__file__, 171, 9), exit___286036, None, None, None)

# SSA branch for the except part of a try statement (line 170)
# SSA branch for the except 'UserWarning' branch of a try statement (line 170)
module_type_store.open_ssa_branch('except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 175, 4))

# 'import matplotlib.pyplot' statement (line 175)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/sphinxext/')
import_286038 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 175, 4), 'matplotlib.pyplot')

if (type(import_286038) is not StypyTypeError):

    if (import_286038 != 'pyd_module'):
        __import__(import_286038)
        sys_modules_286039 = sys.modules[import_286038]
        import_module(stypy.reporting.localization.Localization(__file__, 175, 4), 'plt', sys_modules_286039.module_type_store, module_type_store)
    else:
        import matplotlib.pyplot as plt

        import_module(stypy.reporting.localization.Localization(__file__, 175, 4), 'plt', matplotlib.pyplot, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.pyplot' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'matplotlib.pyplot', import_286038)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/sphinxext/')


# Call to switch_backend(...): (line 176)
# Processing the call arguments (line 176)
unicode_286042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 23), 'unicode', u'Agg')
# Processing the call keyword arguments (line 176)
kwargs_286043 = {}
# Getting the type of 'plt' (line 176)
plt_286040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'plt', False)
# Obtaining the member 'switch_backend' of a type (line 176)
switch_backend_286041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 4), plt_286040, 'switch_backend')
# Calling switch_backend(args, kwargs) (line 176)
switch_backend_call_result_286044 = invoke(stypy.reporting.localization.Localization(__file__, 176, 4), switch_backend_286041, *[unicode_286042], **kwargs_286043)

# SSA branch for the else branch of a try statement (line 170)
module_type_store.open_ssa_branch('except else')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 178, 4))

# 'import matplotlib.pyplot' statement (line 178)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/sphinxext/')
import_286045 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 178, 4), 'matplotlib.pyplot')

if (type(import_286045) is not StypyTypeError):

    if (import_286045 != 'pyd_module'):
        __import__(import_286045)
        sys_modules_286046 = sys.modules[import_286045]
        import_module(stypy.reporting.localization.Localization(__file__, 178, 4), 'plt', sys_modules_286046.module_type_store, module_type_store)
    else:
        import matplotlib.pyplot as plt

        import_module(stypy.reporting.localization.Localization(__file__, 178, 4), 'plt', matplotlib.pyplot, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.pyplot' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'matplotlib.pyplot', import_286045)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/sphinxext/')

# SSA join for try-except statement (line 170)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 179, 0))

# 'from matplotlib import _pylab_helpers' statement (line 179)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/sphinxext/')
import_286047 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 179, 0), 'matplotlib')

if (type(import_286047) is not StypyTypeError):

    if (import_286047 != 'pyd_module'):
        __import__(import_286047)
        sys_modules_286048 = sys.modules[import_286047]
        import_from_module(stypy.reporting.localization.Localization(__file__, 179, 0), 'matplotlib', sys_modules_286048.module_type_store, module_type_store, ['_pylab_helpers'])
        nest_module(stypy.reporting.localization.Localization(__file__, 179, 0), __file__, sys_modules_286048, sys_modules_286048.module_type_store, module_type_store)
    else:
        from matplotlib import _pylab_helpers

        import_from_module(stypy.reporting.localization.Localization(__file__, 179, 0), 'matplotlib', None, module_type_store, ['_pylab_helpers'], [_pylab_helpers])

else:
    # Assigning a type to the variable 'matplotlib' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 0), 'matplotlib', import_286047)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/sphinxext/')


# Assigning a Num to a Name (line 181):

# Assigning a Num to a Name (line 181):
int_286049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 14), 'int')
# Assigning a type to the variable '__version__' (line 181)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 0), '__version__', int_286049)

@norecursion
def plot_directive(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'plot_directive'
    module_type_store = module_type_store.open_function_context('plot_directive', 187, 0, False)
    
    # Passed parameters checking function
    plot_directive.stypy_localization = localization
    plot_directive.stypy_type_of_self = None
    plot_directive.stypy_type_store = module_type_store
    plot_directive.stypy_function_name = 'plot_directive'
    plot_directive.stypy_param_names_list = ['name', 'arguments', 'options', 'content', 'lineno', 'content_offset', 'block_text', 'state', 'state_machine']
    plot_directive.stypy_varargs_param_name = None
    plot_directive.stypy_kwargs_param_name = None
    plot_directive.stypy_call_defaults = defaults
    plot_directive.stypy_call_varargs = varargs
    plot_directive.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'plot_directive', ['name', 'arguments', 'options', 'content', 'lineno', 'content_offset', 'block_text', 'state', 'state_machine'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'plot_directive', localization, ['name', 'arguments', 'options', 'content', 'lineno', 'content_offset', 'block_text', 'state', 'state_machine'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'plot_directive(...)' code ##################

    
    # Call to run(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of 'arguments' (line 189)
    arguments_286051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 15), 'arguments', False)
    # Getting the type of 'content' (line 189)
    content_286052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 26), 'content', False)
    # Getting the type of 'options' (line 189)
    options_286053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 35), 'options', False)
    # Getting the type of 'state_machine' (line 189)
    state_machine_286054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 44), 'state_machine', False)
    # Getting the type of 'state' (line 189)
    state_286055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 59), 'state', False)
    # Getting the type of 'lineno' (line 189)
    lineno_286056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 66), 'lineno', False)
    # Processing the call keyword arguments (line 189)
    kwargs_286057 = {}
    # Getting the type of 'run' (line 189)
    run_286050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 11), 'run', False)
    # Calling run(args, kwargs) (line 189)
    run_call_result_286058 = invoke(stypy.reporting.localization.Localization(__file__, 189, 11), run_286050, *[arguments_286051, content_286052, options_286053, state_machine_286054, state_286055, lineno_286056], **kwargs_286057)
    
    # Assigning a type to the variable 'stypy_return_type' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'stypy_return_type', run_call_result_286058)
    
    # ################# End of 'plot_directive(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'plot_directive' in the type store
    # Getting the type of 'stypy_return_type' (line 187)
    stypy_return_type_286059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286059)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'plot_directive'
    return stypy_return_type_286059

# Assigning a type to the variable 'plot_directive' (line 187)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'plot_directive', plot_directive)

# Assigning a Name to a Attribute (line 190):

# Assigning a Name to a Attribute (line 190):
# Getting the type of '__doc__' (line 190)
doc___286060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 25), '__doc__')
# Getting the type of 'plot_directive' (line 190)
plot_directive_286061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'plot_directive')
# Setting the type of the member '__doc__' of a type (line 190)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 0), plot_directive_286061, '__doc__', doc___286060)

@norecursion
def _option_boolean(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_option_boolean'
    module_type_store = module_type_store.open_function_context('_option_boolean', 193, 0, False)
    
    # Passed parameters checking function
    _option_boolean.stypy_localization = localization
    _option_boolean.stypy_type_of_self = None
    _option_boolean.stypy_type_store = module_type_store
    _option_boolean.stypy_function_name = '_option_boolean'
    _option_boolean.stypy_param_names_list = ['arg']
    _option_boolean.stypy_varargs_param_name = None
    _option_boolean.stypy_kwargs_param_name = None
    _option_boolean.stypy_call_defaults = defaults
    _option_boolean.stypy_call_varargs = varargs
    _option_boolean.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_option_boolean', ['arg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_option_boolean', localization, ['arg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_option_boolean(...)' code ##################

    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'arg' (line 194)
    arg_286062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 11), 'arg')
    # Applying the 'not' unary operator (line 194)
    result_not__286063 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 7), 'not', arg_286062)
    
    
    
    # Call to strip(...): (line 194)
    # Processing the call keyword arguments (line 194)
    kwargs_286066 = {}
    # Getting the type of 'arg' (line 194)
    arg_286064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 22), 'arg', False)
    # Obtaining the member 'strip' of a type (line 194)
    strip_286065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 22), arg_286064, 'strip')
    # Calling strip(args, kwargs) (line 194)
    strip_call_result_286067 = invoke(stypy.reporting.localization.Localization(__file__, 194, 22), strip_286065, *[], **kwargs_286066)
    
    # Applying the 'not' unary operator (line 194)
    result_not__286068 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 18), 'not', strip_call_result_286067)
    
    # Applying the binary operator 'or' (line 194)
    result_or_keyword_286069 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 7), 'or', result_not__286063, result_not__286068)
    
    # Testing the type of an if condition (line 194)
    if_condition_286070 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 4), result_or_keyword_286069)
    # Assigning a type to the variable 'if_condition_286070' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'if_condition_286070', if_condition_286070)
    # SSA begins for if statement (line 194)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 196)
    True_286071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 15), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'stypy_return_type', True_286071)
    # SSA branch for the else part of an if statement (line 194)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to lower(...): (line 197)
    # Processing the call keyword arguments (line 197)
    kwargs_286077 = {}
    
    # Call to strip(...): (line 197)
    # Processing the call keyword arguments (line 197)
    kwargs_286074 = {}
    # Getting the type of 'arg' (line 197)
    arg_286072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 9), 'arg', False)
    # Obtaining the member 'strip' of a type (line 197)
    strip_286073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 9), arg_286072, 'strip')
    # Calling strip(args, kwargs) (line 197)
    strip_call_result_286075 = invoke(stypy.reporting.localization.Localization(__file__, 197, 9), strip_286073, *[], **kwargs_286074)
    
    # Obtaining the member 'lower' of a type (line 197)
    lower_286076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 9), strip_call_result_286075, 'lower')
    # Calling lower(args, kwargs) (line 197)
    lower_call_result_286078 = invoke(stypy.reporting.localization.Localization(__file__, 197, 9), lower_286076, *[], **kwargs_286077)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 197)
    tuple_286079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 197)
    # Adding element type (line 197)
    unicode_286080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 33), 'unicode', u'no')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 33), tuple_286079, unicode_286080)
    # Adding element type (line 197)
    unicode_286081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 39), 'unicode', u'0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 33), tuple_286079, unicode_286081)
    # Adding element type (line 197)
    unicode_286082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 44), 'unicode', u'false')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 33), tuple_286079, unicode_286082)
    
    # Applying the binary operator 'in' (line 197)
    result_contains_286083 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 9), 'in', lower_call_result_286078, tuple_286079)
    
    # Testing the type of an if condition (line 197)
    if_condition_286084 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 9), result_contains_286083)
    # Assigning a type to the variable 'if_condition_286084' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 9), 'if_condition_286084', if_condition_286084)
    # SSA begins for if statement (line 197)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'False' (line 198)
    False_286085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'stypy_return_type', False_286085)
    # SSA branch for the else part of an if statement (line 197)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to lower(...): (line 199)
    # Processing the call keyword arguments (line 199)
    kwargs_286091 = {}
    
    # Call to strip(...): (line 199)
    # Processing the call keyword arguments (line 199)
    kwargs_286088 = {}
    # Getting the type of 'arg' (line 199)
    arg_286086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 9), 'arg', False)
    # Obtaining the member 'strip' of a type (line 199)
    strip_286087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 9), arg_286086, 'strip')
    # Calling strip(args, kwargs) (line 199)
    strip_call_result_286089 = invoke(stypy.reporting.localization.Localization(__file__, 199, 9), strip_286087, *[], **kwargs_286088)
    
    # Obtaining the member 'lower' of a type (line 199)
    lower_286090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 9), strip_call_result_286089, 'lower')
    # Calling lower(args, kwargs) (line 199)
    lower_call_result_286092 = invoke(stypy.reporting.localization.Localization(__file__, 199, 9), lower_286090, *[], **kwargs_286091)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 199)
    tuple_286093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 199)
    # Adding element type (line 199)
    unicode_286094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 33), 'unicode', u'yes')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 33), tuple_286093, unicode_286094)
    # Adding element type (line 199)
    unicode_286095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 40), 'unicode', u'1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 33), tuple_286093, unicode_286095)
    # Adding element type (line 199)
    unicode_286096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 45), 'unicode', u'true')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 33), tuple_286093, unicode_286096)
    
    # Applying the binary operator 'in' (line 199)
    result_contains_286097 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 9), 'in', lower_call_result_286092, tuple_286093)
    
    # Testing the type of an if condition (line 199)
    if_condition_286098 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 9), result_contains_286097)
    # Assigning a type to the variable 'if_condition_286098' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 9), 'if_condition_286098', if_condition_286098)
    # SSA begins for if statement (line 199)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 200)
    True_286099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'stypy_return_type', True_286099)
    # SSA branch for the else part of an if statement (line 199)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 202)
    # Processing the call arguments (line 202)
    unicode_286101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 25), 'unicode', u'"%s" unknown boolean')
    # Getting the type of 'arg' (line 202)
    arg_286102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 50), 'arg', False)
    # Applying the binary operator '%' (line 202)
    result_mod_286103 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 25), '%', unicode_286101, arg_286102)
    
    # Processing the call keyword arguments (line 202)
    kwargs_286104 = {}
    # Getting the type of 'ValueError' (line 202)
    ValueError_286100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 202)
    ValueError_call_result_286105 = invoke(stypy.reporting.localization.Localization(__file__, 202, 14), ValueError_286100, *[result_mod_286103], **kwargs_286104)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 202, 8), ValueError_call_result_286105, 'raise parameter', BaseException)
    # SSA join for if statement (line 199)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 197)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 194)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_option_boolean(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_option_boolean' in the type store
    # Getting the type of 'stypy_return_type' (line 193)
    stypy_return_type_286106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286106)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_option_boolean'
    return stypy_return_type_286106

# Assigning a type to the variable '_option_boolean' (line 193)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), '_option_boolean', _option_boolean)

@norecursion
def _option_context(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_option_context'
    module_type_store = module_type_store.open_function_context('_option_context', 205, 0, False)
    
    # Passed parameters checking function
    _option_context.stypy_localization = localization
    _option_context.stypy_type_of_self = None
    _option_context.stypy_type_store = module_type_store
    _option_context.stypy_function_name = '_option_context'
    _option_context.stypy_param_names_list = ['arg']
    _option_context.stypy_varargs_param_name = None
    _option_context.stypy_kwargs_param_name = None
    _option_context.stypy_call_defaults = defaults
    _option_context.stypy_call_varargs = varargs
    _option_context.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_option_context', ['arg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_option_context', localization, ['arg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_option_context(...)' code ##################

    
    
    # Getting the type of 'arg' (line 206)
    arg_286107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 7), 'arg')
    
    # Obtaining an instance of the builtin type 'list' (line 206)
    list_286108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 206)
    # Adding element type (line 206)
    # Getting the type of 'None' (line 206)
    None_286109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 15), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 14), list_286108, None_286109)
    # Adding element type (line 206)
    unicode_286110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 21), 'unicode', u'reset')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 14), list_286108, unicode_286110)
    # Adding element type (line 206)
    unicode_286111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 30), 'unicode', u'close-figs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 14), list_286108, unicode_286111)
    
    # Applying the binary operator 'in' (line 206)
    result_contains_286112 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 7), 'in', arg_286107, list_286108)
    
    # Testing the type of an if condition (line 206)
    if_condition_286113 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 4), result_contains_286112)
    # Assigning a type to the variable 'if_condition_286113' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'if_condition_286113', if_condition_286113)
    # SSA begins for if statement (line 206)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'arg' (line 207)
    arg_286114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 15), 'arg')
    # Assigning a type to the variable 'stypy_return_type' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'stypy_return_type', arg_286114)
    # SSA join for if statement (line 206)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to ValueError(...): (line 208)
    # Processing the call arguments (line 208)
    unicode_286116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 21), 'unicode', u"argument should be None or 'reset' or 'close-figs'")
    # Processing the call keyword arguments (line 208)
    kwargs_286117 = {}
    # Getting the type of 'ValueError' (line 208)
    ValueError_286115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 10), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 208)
    ValueError_call_result_286118 = invoke(stypy.reporting.localization.Localization(__file__, 208, 10), ValueError_286115, *[unicode_286116], **kwargs_286117)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 208, 4), ValueError_call_result_286118, 'raise parameter', BaseException)
    
    # ################# End of '_option_context(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_option_context' in the type store
    # Getting the type of 'stypy_return_type' (line 205)
    stypy_return_type_286119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286119)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_option_context'
    return stypy_return_type_286119

# Assigning a type to the variable '_option_context' (line 205)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 0), '_option_context', _option_context)

@norecursion
def _option_format(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_option_format'
    module_type_store = module_type_store.open_function_context('_option_format', 211, 0, False)
    
    # Passed parameters checking function
    _option_format.stypy_localization = localization
    _option_format.stypy_type_of_self = None
    _option_format.stypy_type_store = module_type_store
    _option_format.stypy_function_name = '_option_format'
    _option_format.stypy_param_names_list = ['arg']
    _option_format.stypy_varargs_param_name = None
    _option_format.stypy_kwargs_param_name = None
    _option_format.stypy_call_defaults = defaults
    _option_format.stypy_call_varargs = varargs
    _option_format.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_option_format', ['arg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_option_format', localization, ['arg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_option_format(...)' code ##################

    
    # Call to choice(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'arg' (line 212)
    arg_286122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 29), 'arg', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 212)
    tuple_286123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 212)
    # Adding element type (line 212)
    unicode_286124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 35), 'unicode', u'python')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 35), tuple_286123, unicode_286124)
    # Adding element type (line 212)
    unicode_286125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 45), 'unicode', u'doctest')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 35), tuple_286123, unicode_286125)
    
    # Processing the call keyword arguments (line 212)
    kwargs_286126 = {}
    # Getting the type of 'directives' (line 212)
    directives_286120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 11), 'directives', False)
    # Obtaining the member 'choice' of a type (line 212)
    choice_286121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 11), directives_286120, 'choice')
    # Calling choice(args, kwargs) (line 212)
    choice_call_result_286127 = invoke(stypy.reporting.localization.Localization(__file__, 212, 11), choice_286121, *[arg_286122, tuple_286123], **kwargs_286126)
    
    # Assigning a type to the variable 'stypy_return_type' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'stypy_return_type', choice_call_result_286127)
    
    # ################# End of '_option_format(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_option_format' in the type store
    # Getting the type of 'stypy_return_type' (line 211)
    stypy_return_type_286128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286128)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_option_format'
    return stypy_return_type_286128

# Assigning a type to the variable '_option_format' (line 211)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 0), '_option_format', _option_format)

@norecursion
def _option_align(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_option_align'
    module_type_store = module_type_store.open_function_context('_option_align', 215, 0, False)
    
    # Passed parameters checking function
    _option_align.stypy_localization = localization
    _option_align.stypy_type_of_self = None
    _option_align.stypy_type_store = module_type_store
    _option_align.stypy_function_name = '_option_align'
    _option_align.stypy_param_names_list = ['arg']
    _option_align.stypy_varargs_param_name = None
    _option_align.stypy_kwargs_param_name = None
    _option_align.stypy_call_defaults = defaults
    _option_align.stypy_call_varargs = varargs
    _option_align.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_option_align', ['arg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_option_align', localization, ['arg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_option_align(...)' code ##################

    
    # Call to choice(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'arg' (line 216)
    arg_286131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 29), 'arg', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 216)
    tuple_286132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 216)
    # Adding element type (line 216)
    unicode_286133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 35), 'unicode', u'top')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 35), tuple_286132, unicode_286133)
    # Adding element type (line 216)
    unicode_286134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 42), 'unicode', u'middle')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 35), tuple_286132, unicode_286134)
    # Adding element type (line 216)
    unicode_286135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 52), 'unicode', u'bottom')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 35), tuple_286132, unicode_286135)
    # Adding element type (line 216)
    unicode_286136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 62), 'unicode', u'left')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 35), tuple_286132, unicode_286136)
    # Adding element type (line 216)
    unicode_286137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 70), 'unicode', u'center')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 35), tuple_286132, unicode_286137)
    # Adding element type (line 216)
    unicode_286138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 35), 'unicode', u'right')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 35), tuple_286132, unicode_286138)
    
    # Processing the call keyword arguments (line 216)
    kwargs_286139 = {}
    # Getting the type of 'directives' (line 216)
    directives_286129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 11), 'directives', False)
    # Obtaining the member 'choice' of a type (line 216)
    choice_286130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 11), directives_286129, 'choice')
    # Calling choice(args, kwargs) (line 216)
    choice_call_result_286140 = invoke(stypy.reporting.localization.Localization(__file__, 216, 11), choice_286130, *[arg_286131, tuple_286132], **kwargs_286139)
    
    # Assigning a type to the variable 'stypy_return_type' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'stypy_return_type', choice_call_result_286140)
    
    # ################# End of '_option_align(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_option_align' in the type store
    # Getting the type of 'stypy_return_type' (line 215)
    stypy_return_type_286141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286141)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_option_align'
    return stypy_return_type_286141

# Assigning a type to the variable '_option_align' (line 215)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), '_option_align', _option_align)

@norecursion
def mark_plot_labels(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'mark_plot_labels'
    module_type_store = module_type_store.open_function_context('mark_plot_labels', 220, 0, False)
    
    # Passed parameters checking function
    mark_plot_labels.stypy_localization = localization
    mark_plot_labels.stypy_type_of_self = None
    mark_plot_labels.stypy_type_store = module_type_store
    mark_plot_labels.stypy_function_name = 'mark_plot_labels'
    mark_plot_labels.stypy_param_names_list = ['app', 'document']
    mark_plot_labels.stypy_varargs_param_name = None
    mark_plot_labels.stypy_kwargs_param_name = None
    mark_plot_labels.stypy_call_defaults = defaults
    mark_plot_labels.stypy_call_varargs = varargs
    mark_plot_labels.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mark_plot_labels', ['app', 'document'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mark_plot_labels', localization, ['app', 'document'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mark_plot_labels(...)' code ##################

    unicode_286142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, (-1)), 'unicode', u'\n    To make plots referenceable, we need to move the reference from\n    the "htmlonly" (or "latexonly") node to the actual figure node\n    itself.\n    ')
    
    
    # Call to iteritems(...): (line 226)
    # Processing the call arguments (line 226)
    # Getting the type of 'document' (line 226)
    document_286145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 40), 'document', False)
    # Obtaining the member 'nametypes' of a type (line 226)
    nametypes_286146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 40), document_286145, 'nametypes')
    # Processing the call keyword arguments (line 226)
    kwargs_286147 = {}
    # Getting the type of 'six' (line 226)
    six_286143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 26), 'six', False)
    # Obtaining the member 'iteritems' of a type (line 226)
    iteritems_286144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 26), six_286143, 'iteritems')
    # Calling iteritems(args, kwargs) (line 226)
    iteritems_call_result_286148 = invoke(stypy.reporting.localization.Localization(__file__, 226, 26), iteritems_286144, *[nametypes_286146], **kwargs_286147)
    
    # Testing the type of a for loop iterable (line 226)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 226, 4), iteritems_call_result_286148)
    # Getting the type of the for loop variable (line 226)
    for_loop_var_286149 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 226, 4), iteritems_call_result_286148)
    # Assigning a type to the variable 'name' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 4), for_loop_var_286149))
    # Assigning a type to the variable 'explicit' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'explicit', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 4), for_loop_var_286149))
    # SSA begins for a for statement (line 226)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'explicit' (line 227)
    explicit_286150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 15), 'explicit')
    # Applying the 'not' unary operator (line 227)
    result_not__286151 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 11), 'not', explicit_286150)
    
    # Testing the type of an if condition (line 227)
    if_condition_286152 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 8), result_not__286151)
    # Assigning a type to the variable 'if_condition_286152' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'if_condition_286152', if_condition_286152)
    # SSA begins for if statement (line 227)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 227)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 229):
    
    # Assigning a Subscript to a Name (line 229):
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 229)
    name_286153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 35), 'name')
    # Getting the type of 'document' (line 229)
    document_286154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 18), 'document')
    # Obtaining the member 'nameids' of a type (line 229)
    nameids_286155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 18), document_286154, 'nameids')
    # Obtaining the member '__getitem__' of a type (line 229)
    getitem___286156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 18), nameids_286155, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 229)
    subscript_call_result_286157 = invoke(stypy.reporting.localization.Localization(__file__, 229, 18), getitem___286156, name_286153)
    
    # Assigning a type to the variable 'labelid' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'labelid', subscript_call_result_286157)
    
    # Type idiom detected: calculating its left and rigth part (line 230)
    # Getting the type of 'labelid' (line 230)
    labelid_286158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 11), 'labelid')
    # Getting the type of 'None' (line 230)
    None_286159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 22), 'None')
    
    (may_be_286160, more_types_in_union_286161) = may_be_none(labelid_286158, None_286159)

    if may_be_286160:

        if more_types_in_union_286161:
            # Runtime conditional SSA (line 230)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        if more_types_in_union_286161:
            # SSA join for if statement (line 230)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Subscript to a Name (line 232):
    
    # Assigning a Subscript to a Name (line 232):
    
    # Obtaining the type of the subscript
    # Getting the type of 'labelid' (line 232)
    labelid_286162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 28), 'labelid')
    # Getting the type of 'document' (line 232)
    document_286163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 15), 'document')
    # Obtaining the member 'ids' of a type (line 232)
    ids_286164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 15), document_286163, 'ids')
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___286165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 15), ids_286164, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_286166 = invoke(stypy.reporting.localization.Localization(__file__, 232, 15), getitem___286165, labelid_286162)
    
    # Assigning a type to the variable 'node' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'node', subscript_call_result_286166)
    
    
    # Getting the type of 'node' (line 233)
    node_286167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 11), 'node')
    # Obtaining the member 'tagname' of a type (line 233)
    tagname_286168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 11), node_286167, 'tagname')
    
    # Obtaining an instance of the builtin type 'tuple' (line 233)
    tuple_286169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 233)
    # Adding element type (line 233)
    unicode_286170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 28), 'unicode', u'html_only')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 28), tuple_286169, unicode_286170)
    # Adding element type (line 233)
    unicode_286171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 41), 'unicode', u'latex_only')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 28), tuple_286169, unicode_286171)
    
    # Applying the binary operator 'in' (line 233)
    result_contains_286172 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 11), 'in', tagname_286168, tuple_286169)
    
    # Testing the type of an if condition (line 233)
    if_condition_286173 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 233, 8), result_contains_286172)
    # Assigning a type to the variable 'if_condition_286173' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'if_condition_286173', if_condition_286173)
    # SSA begins for if statement (line 233)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'node' (line 234)
    node_286174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 21), 'node')
    # Testing the type of a for loop iterable (line 234)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 234, 12), node_286174)
    # Getting the type of the for loop variable (line 234)
    for_loop_var_286175 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 234, 12), node_286174)
    # Assigning a type to the variable 'n' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'n', for_loop_var_286175)
    # SSA begins for a for statement (line 234)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'n' (line 235)
    n_286176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 19), 'n')
    # Obtaining the member 'tagname' of a type (line 235)
    tagname_286177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 19), n_286176, 'tagname')
    unicode_286178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 32), 'unicode', u'figure')
    # Applying the binary operator '==' (line 235)
    result_eq_286179 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 19), '==', tagname_286177, unicode_286178)
    
    # Testing the type of an if condition (line 235)
    if_condition_286180 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 16), result_eq_286179)
    # Assigning a type to the variable 'if_condition_286180' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'if_condition_286180', if_condition_286180)
    # SSA begins for if statement (line 235)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 236):
    
    # Assigning a Name to a Name (line 236):
    # Getting the type of 'name' (line 236)
    name_286181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 31), 'name')
    # Assigning a type to the variable 'sectname' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 20), 'sectname', name_286181)
    
    # Getting the type of 'n' (line 237)
    n_286182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 29), 'n')
    # Testing the type of a for loop iterable (line 237)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 237, 20), n_286182)
    # Getting the type of the for loop variable (line 237)
    for_loop_var_286183 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 237, 20), n_286182)
    # Assigning a type to the variable 'c' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 20), 'c', for_loop_var_286183)
    # SSA begins for a for statement (line 237)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'c' (line 238)
    c_286184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 27), 'c')
    # Obtaining the member 'tagname' of a type (line 238)
    tagname_286185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 27), c_286184, 'tagname')
    unicode_286186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 40), 'unicode', u'caption')
    # Applying the binary operator '==' (line 238)
    result_eq_286187 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 27), '==', tagname_286185, unicode_286186)
    
    # Testing the type of an if condition (line 238)
    if_condition_286188 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 238, 24), result_eq_286187)
    # Assigning a type to the variable 'if_condition_286188' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 24), 'if_condition_286188', if_condition_286188)
    # SSA begins for if statement (line 238)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 239):
    
    # Assigning a Call to a Name (line 239):
    
    # Call to astext(...): (line 239)
    # Processing the call keyword arguments (line 239)
    kwargs_286191 = {}
    # Getting the type of 'c' (line 239)
    c_286189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 39), 'c', False)
    # Obtaining the member 'astext' of a type (line 239)
    astext_286190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 39), c_286189, 'astext')
    # Calling astext(args, kwargs) (line 239)
    astext_call_result_286192 = invoke(stypy.reporting.localization.Localization(__file__, 239, 39), astext_286190, *[], **kwargs_286191)
    
    # Assigning a type to the variable 'sectname' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 28), 'sectname', astext_call_result_286192)
    # SSA join for if statement (line 238)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to remove(...): (line 242)
    # Processing the call arguments (line 242)
    # Getting the type of 'labelid' (line 242)
    labelid_286198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 39), 'labelid', False)
    # Processing the call keyword arguments (line 242)
    kwargs_286199 = {}
    
    # Obtaining the type of the subscript
    unicode_286193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 25), 'unicode', u'ids')
    # Getting the type of 'node' (line 242)
    node_286194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 20), 'node', False)
    # Obtaining the member '__getitem__' of a type (line 242)
    getitem___286195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 20), node_286194, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 242)
    subscript_call_result_286196 = invoke(stypy.reporting.localization.Localization(__file__, 242, 20), getitem___286195, unicode_286193)
    
    # Obtaining the member 'remove' of a type (line 242)
    remove_286197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 20), subscript_call_result_286196, 'remove')
    # Calling remove(args, kwargs) (line 242)
    remove_call_result_286200 = invoke(stypy.reporting.localization.Localization(__file__, 242, 20), remove_286197, *[labelid_286198], **kwargs_286199)
    
    
    # Call to remove(...): (line 243)
    # Processing the call arguments (line 243)
    # Getting the type of 'name' (line 243)
    name_286206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 41), 'name', False)
    # Processing the call keyword arguments (line 243)
    kwargs_286207 = {}
    
    # Obtaining the type of the subscript
    unicode_286201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 25), 'unicode', u'names')
    # Getting the type of 'node' (line 243)
    node_286202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 20), 'node', False)
    # Obtaining the member '__getitem__' of a type (line 243)
    getitem___286203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 20), node_286202, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 243)
    subscript_call_result_286204 = invoke(stypy.reporting.localization.Localization(__file__, 243, 20), getitem___286203, unicode_286201)
    
    # Obtaining the member 'remove' of a type (line 243)
    remove_286205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 20), subscript_call_result_286204, 'remove')
    # Calling remove(args, kwargs) (line 243)
    remove_call_result_286208 = invoke(stypy.reporting.localization.Localization(__file__, 243, 20), remove_286205, *[name_286206], **kwargs_286207)
    
    
    # Call to append(...): (line 244)
    # Processing the call arguments (line 244)
    # Getting the type of 'labelid' (line 244)
    labelid_286214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 36), 'labelid', False)
    # Processing the call keyword arguments (line 244)
    kwargs_286215 = {}
    
    # Obtaining the type of the subscript
    unicode_286209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 22), 'unicode', u'ids')
    # Getting the type of 'n' (line 244)
    n_286210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 20), 'n', False)
    # Obtaining the member '__getitem__' of a type (line 244)
    getitem___286211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 20), n_286210, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 244)
    subscript_call_result_286212 = invoke(stypy.reporting.localization.Localization(__file__, 244, 20), getitem___286211, unicode_286209)
    
    # Obtaining the member 'append' of a type (line 244)
    append_286213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 20), subscript_call_result_286212, 'append')
    # Calling append(args, kwargs) (line 244)
    append_call_result_286216 = invoke(stypy.reporting.localization.Localization(__file__, 244, 20), append_286213, *[labelid_286214], **kwargs_286215)
    
    
    # Call to append(...): (line 245)
    # Processing the call arguments (line 245)
    # Getting the type of 'name' (line 245)
    name_286222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 38), 'name', False)
    # Processing the call keyword arguments (line 245)
    kwargs_286223 = {}
    
    # Obtaining the type of the subscript
    unicode_286217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 22), 'unicode', u'names')
    # Getting the type of 'n' (line 245)
    n_286218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 20), 'n', False)
    # Obtaining the member '__getitem__' of a type (line 245)
    getitem___286219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 20), n_286218, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 245)
    subscript_call_result_286220 = invoke(stypy.reporting.localization.Localization(__file__, 245, 20), getitem___286219, unicode_286217)
    
    # Obtaining the member 'append' of a type (line 245)
    append_286221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 20), subscript_call_result_286220, 'append')
    # Calling append(args, kwargs) (line 245)
    append_call_result_286224 = invoke(stypy.reporting.localization.Localization(__file__, 245, 20), append_286221, *[name_286222], **kwargs_286223)
    
    
    # Assigning a Tuple to a Subscript (line 246):
    
    # Assigning a Tuple to a Subscript (line 246):
    
    # Obtaining an instance of the builtin type 'tuple' (line 247)
    tuple_286225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 247)
    # Adding element type (line 247)
    # Getting the type of 'document' (line 247)
    document_286226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 24), 'document')
    # Obtaining the member 'settings' of a type (line 247)
    settings_286227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 24), document_286226, 'settings')
    # Obtaining the member 'env' of a type (line 247)
    env_286228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 24), settings_286227, 'env')
    # Obtaining the member 'docname' of a type (line 247)
    docname_286229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 24), env_286228, 'docname')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 24), tuple_286225, docname_286229)
    # Adding element type (line 247)
    # Getting the type of 'labelid' (line 247)
    labelid_286230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 55), 'labelid')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 24), tuple_286225, labelid_286230)
    # Adding element type (line 247)
    # Getting the type of 'sectname' (line 247)
    sectname_286231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 64), 'sectname')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 24), tuple_286225, sectname_286231)
    
    # Getting the type of 'document' (line 246)
    document_286232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 20), 'document')
    # Obtaining the member 'settings' of a type (line 246)
    settings_286233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 20), document_286232, 'settings')
    # Obtaining the member 'env' of a type (line 246)
    env_286234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 20), settings_286233, 'env')
    # Obtaining the member 'labels' of a type (line 246)
    labels_286235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 20), env_286234, 'labels')
    # Getting the type of 'name' (line 246)
    name_286236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 49), 'name')
    # Storing an element on a container (line 246)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 20), labels_286235, (name_286236, tuple_286225))
    # SSA join for if statement (line 235)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 233)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'mark_plot_labels(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mark_plot_labels' in the type store
    # Getting the type of 'stypy_return_type' (line 220)
    stypy_return_type_286237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286237)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mark_plot_labels'
    return stypy_return_type_286237

# Assigning a type to the variable 'mark_plot_labels' (line 220)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 0), 'mark_plot_labels', mark_plot_labels)

@norecursion
def setup(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'setup'
    module_type_store = module_type_store.open_function_context('setup', 251, 0, False)
    
    # Passed parameters checking function
    setup.stypy_localization = localization
    setup.stypy_type_of_self = None
    setup.stypy_type_store = module_type_store
    setup.stypy_function_name = 'setup'
    setup.stypy_param_names_list = ['app']
    setup.stypy_varargs_param_name = None
    setup.stypy_kwargs_param_name = None
    setup.stypy_call_defaults = defaults
    setup.stypy_call_varargs = varargs
    setup.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'setup', ['app'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'setup', localization, ['app'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'setup(...)' code ##################

    
    # Assigning a Name to a Attribute (line 252):
    
    # Assigning a Name to a Attribute (line 252):
    # Getting the type of 'app' (line 252)
    app_286238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'app')
    # Getting the type of 'setup' (line 252)
    setup_286239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'setup')
    # Setting the type of the member 'app' of a type (line 252)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 4), setup_286239, 'app', app_286238)
    
    # Assigning a Attribute to a Attribute (line 253):
    
    # Assigning a Attribute to a Attribute (line 253):
    # Getting the type of 'app' (line 253)
    app_286240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 19), 'app')
    # Obtaining the member 'config' of a type (line 253)
    config_286241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 19), app_286240, 'config')
    # Getting the type of 'setup' (line 253)
    setup_286242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'setup')
    # Setting the type of the member 'config' of a type (line 253)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 4), setup_286242, 'config', config_286241)
    
    # Assigning a Attribute to a Attribute (line 254):
    
    # Assigning a Attribute to a Attribute (line 254):
    # Getting the type of 'app' (line 254)
    app_286243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 20), 'app')
    # Obtaining the member 'confdir' of a type (line 254)
    confdir_286244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 20), app_286243, 'confdir')
    # Getting the type of 'setup' (line 254)
    setup_286245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'setup')
    # Setting the type of the member 'confdir' of a type (line 254)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 4), setup_286245, 'confdir', confdir_286244)
    
    # Assigning a Dict to a Name (line 256):
    
    # Assigning a Dict to a Name (line 256):
    
    # Obtaining an instance of the builtin type 'dict' (line 256)
    dict_286246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 14), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 256)
    # Adding element type (key, value) (line 256)
    unicode_286247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 15), 'unicode', u'alt')
    # Getting the type of 'directives' (line 256)
    directives_286248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 22), 'directives')
    # Obtaining the member 'unchanged' of a type (line 256)
    unchanged_286249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 22), directives_286248, 'unchanged')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 14), dict_286246, (unicode_286247, unchanged_286249))
    # Adding element type (key, value) (line 256)
    unicode_286250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 15), 'unicode', u'height')
    # Getting the type of 'directives' (line 257)
    directives_286251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 25), 'directives')
    # Obtaining the member 'length_or_unitless' of a type (line 257)
    length_or_unitless_286252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 25), directives_286251, 'length_or_unitless')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 14), dict_286246, (unicode_286250, length_or_unitless_286252))
    # Adding element type (key, value) (line 256)
    unicode_286253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 15), 'unicode', u'width')
    # Getting the type of 'directives' (line 258)
    directives_286254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 24), 'directives')
    # Obtaining the member 'length_or_percentage_or_unitless' of a type (line 258)
    length_or_percentage_or_unitless_286255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 24), directives_286254, 'length_or_percentage_or_unitless')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 14), dict_286246, (unicode_286253, length_or_percentage_or_unitless_286255))
    # Adding element type (key, value) (line 256)
    unicode_286256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 15), 'unicode', u'scale')
    # Getting the type of 'directives' (line 259)
    directives_286257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 24), 'directives')
    # Obtaining the member 'nonnegative_int' of a type (line 259)
    nonnegative_int_286258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 24), directives_286257, 'nonnegative_int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 14), dict_286246, (unicode_286256, nonnegative_int_286258))
    # Adding element type (key, value) (line 256)
    unicode_286259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 15), 'unicode', u'align')
    # Getting the type of '_option_align' (line 260)
    _option_align_286260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 24), '_option_align')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 14), dict_286246, (unicode_286259, _option_align_286260))
    # Adding element type (key, value) (line 256)
    unicode_286261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 15), 'unicode', u'class')
    # Getting the type of 'directives' (line 261)
    directives_286262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 24), 'directives')
    # Obtaining the member 'class_option' of a type (line 261)
    class_option_286263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 24), directives_286262, 'class_option')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 14), dict_286246, (unicode_286261, class_option_286263))
    # Adding element type (key, value) (line 256)
    unicode_286264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 15), 'unicode', u'include-source')
    # Getting the type of '_option_boolean' (line 262)
    _option_boolean_286265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 33), '_option_boolean')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 14), dict_286246, (unicode_286264, _option_boolean_286265))
    # Adding element type (key, value) (line 256)
    unicode_286266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 15), 'unicode', u'format')
    # Getting the type of '_option_format' (line 263)
    _option_format_286267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 25), '_option_format')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 14), dict_286246, (unicode_286266, _option_format_286267))
    # Adding element type (key, value) (line 256)
    unicode_286268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 15), 'unicode', u'context')
    # Getting the type of '_option_context' (line 264)
    _option_context_286269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 26), '_option_context')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 14), dict_286246, (unicode_286268, _option_context_286269))
    # Adding element type (key, value) (line 256)
    unicode_286270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 15), 'unicode', u'nofigs')
    # Getting the type of 'directives' (line 265)
    directives_286271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 25), 'directives')
    # Obtaining the member 'flag' of a type (line 265)
    flag_286272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 25), directives_286271, 'flag')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 14), dict_286246, (unicode_286270, flag_286272))
    # Adding element type (key, value) (line 256)
    unicode_286273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 15), 'unicode', u'encoding')
    # Getting the type of 'directives' (line 266)
    directives_286274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 27), 'directives')
    # Obtaining the member 'encoding' of a type (line 266)
    encoding_286275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 27), directives_286274, 'encoding')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 14), dict_286246, (unicode_286273, encoding_286275))
    
    # Assigning a type to the variable 'options' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'options', dict_286246)
    
    # Call to add_directive(...): (line 269)
    # Processing the call arguments (line 269)
    unicode_286278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 22), 'unicode', u'plot')
    # Getting the type of 'plot_directive' (line 269)
    plot_directive_286279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 30), 'plot_directive', False)
    # Getting the type of 'True' (line 269)
    True_286280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 46), 'True', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 269)
    tuple_286281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 269)
    # Adding element type (line 269)
    int_286282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 53), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 53), tuple_286281, int_286282)
    # Adding element type (line 269)
    int_286283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 56), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 53), tuple_286281, int_286283)
    # Adding element type (line 269)
    # Getting the type of 'False' (line 269)
    False_286284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 59), 'False', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 53), tuple_286281, False_286284)
    
    # Processing the call keyword arguments (line 269)
    # Getting the type of 'options' (line 269)
    options_286285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 69), 'options', False)
    kwargs_286286 = {'options_286285': options_286285}
    # Getting the type of 'app' (line 269)
    app_286276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'app', False)
    # Obtaining the member 'add_directive' of a type (line 269)
    add_directive_286277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 4), app_286276, 'add_directive')
    # Calling add_directive(args, kwargs) (line 269)
    add_directive_call_result_286287 = invoke(stypy.reporting.localization.Localization(__file__, 269, 4), add_directive_286277, *[unicode_286278, plot_directive_286279, True_286280, tuple_286281], **kwargs_286286)
    
    
    # Call to add_config_value(...): (line 270)
    # Processing the call arguments (line 270)
    unicode_286290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 25), 'unicode', u'plot_pre_code')
    # Getting the type of 'None' (line 270)
    None_286291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 42), 'None', False)
    # Getting the type of 'True' (line 270)
    True_286292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 48), 'True', False)
    # Processing the call keyword arguments (line 270)
    kwargs_286293 = {}
    # Getting the type of 'app' (line 270)
    app_286288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'app', False)
    # Obtaining the member 'add_config_value' of a type (line 270)
    add_config_value_286289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 4), app_286288, 'add_config_value')
    # Calling add_config_value(args, kwargs) (line 270)
    add_config_value_call_result_286294 = invoke(stypy.reporting.localization.Localization(__file__, 270, 4), add_config_value_286289, *[unicode_286290, None_286291, True_286292], **kwargs_286293)
    
    
    # Call to add_config_value(...): (line 271)
    # Processing the call arguments (line 271)
    unicode_286297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 25), 'unicode', u'plot_include_source')
    # Getting the type of 'False' (line 271)
    False_286298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 48), 'False', False)
    # Getting the type of 'True' (line 271)
    True_286299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 55), 'True', False)
    # Processing the call keyword arguments (line 271)
    kwargs_286300 = {}
    # Getting the type of 'app' (line 271)
    app_286295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'app', False)
    # Obtaining the member 'add_config_value' of a type (line 271)
    add_config_value_286296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 4), app_286295, 'add_config_value')
    # Calling add_config_value(args, kwargs) (line 271)
    add_config_value_call_result_286301 = invoke(stypy.reporting.localization.Localization(__file__, 271, 4), add_config_value_286296, *[unicode_286297, False_286298, True_286299], **kwargs_286300)
    
    
    # Call to add_config_value(...): (line 272)
    # Processing the call arguments (line 272)
    unicode_286304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 25), 'unicode', u'plot_html_show_source_link')
    # Getting the type of 'True' (line 272)
    True_286305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 55), 'True', False)
    # Getting the type of 'True' (line 272)
    True_286306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 61), 'True', False)
    # Processing the call keyword arguments (line 272)
    kwargs_286307 = {}
    # Getting the type of 'app' (line 272)
    app_286302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'app', False)
    # Obtaining the member 'add_config_value' of a type (line 272)
    add_config_value_286303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 4), app_286302, 'add_config_value')
    # Calling add_config_value(args, kwargs) (line 272)
    add_config_value_call_result_286308 = invoke(stypy.reporting.localization.Localization(__file__, 272, 4), add_config_value_286303, *[unicode_286304, True_286305, True_286306], **kwargs_286307)
    
    
    # Call to add_config_value(...): (line 273)
    # Processing the call arguments (line 273)
    unicode_286311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 25), 'unicode', u'plot_formats')
    
    # Obtaining an instance of the builtin type 'list' (line 273)
    list_286312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 273)
    # Adding element type (line 273)
    unicode_286313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 42), 'unicode', u'png')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 41), list_286312, unicode_286313)
    # Adding element type (line 273)
    unicode_286314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 49), 'unicode', u'hires.png')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 41), list_286312, unicode_286314)
    # Adding element type (line 273)
    unicode_286315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 62), 'unicode', u'pdf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 41), list_286312, unicode_286315)
    
    # Getting the type of 'True' (line 273)
    True_286316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 70), 'True', False)
    # Processing the call keyword arguments (line 273)
    kwargs_286317 = {}
    # Getting the type of 'app' (line 273)
    app_286309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'app', False)
    # Obtaining the member 'add_config_value' of a type (line 273)
    add_config_value_286310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 4), app_286309, 'add_config_value')
    # Calling add_config_value(args, kwargs) (line 273)
    add_config_value_call_result_286318 = invoke(stypy.reporting.localization.Localization(__file__, 273, 4), add_config_value_286310, *[unicode_286311, list_286312, True_286316], **kwargs_286317)
    
    
    # Call to add_config_value(...): (line 274)
    # Processing the call arguments (line 274)
    unicode_286321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 25), 'unicode', u'plot_basedir')
    # Getting the type of 'None' (line 274)
    None_286322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 41), 'None', False)
    # Getting the type of 'True' (line 274)
    True_286323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 47), 'True', False)
    # Processing the call keyword arguments (line 274)
    kwargs_286324 = {}
    # Getting the type of 'app' (line 274)
    app_286319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'app', False)
    # Obtaining the member 'add_config_value' of a type (line 274)
    add_config_value_286320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 4), app_286319, 'add_config_value')
    # Calling add_config_value(args, kwargs) (line 274)
    add_config_value_call_result_286325 = invoke(stypy.reporting.localization.Localization(__file__, 274, 4), add_config_value_286320, *[unicode_286321, None_286322, True_286323], **kwargs_286324)
    
    
    # Call to add_config_value(...): (line 275)
    # Processing the call arguments (line 275)
    unicode_286328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 25), 'unicode', u'plot_html_show_formats')
    # Getting the type of 'True' (line 275)
    True_286329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 51), 'True', False)
    # Getting the type of 'True' (line 275)
    True_286330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 57), 'True', False)
    # Processing the call keyword arguments (line 275)
    kwargs_286331 = {}
    # Getting the type of 'app' (line 275)
    app_286326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'app', False)
    # Obtaining the member 'add_config_value' of a type (line 275)
    add_config_value_286327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 4), app_286326, 'add_config_value')
    # Calling add_config_value(args, kwargs) (line 275)
    add_config_value_call_result_286332 = invoke(stypy.reporting.localization.Localization(__file__, 275, 4), add_config_value_286327, *[unicode_286328, True_286329, True_286330], **kwargs_286331)
    
    
    # Call to add_config_value(...): (line 276)
    # Processing the call arguments (line 276)
    unicode_286335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 25), 'unicode', u'plot_rcparams')
    
    # Obtaining an instance of the builtin type 'dict' (line 276)
    dict_286336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 42), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 276)
    
    # Getting the type of 'True' (line 276)
    True_286337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 46), 'True', False)
    # Processing the call keyword arguments (line 276)
    kwargs_286338 = {}
    # Getting the type of 'app' (line 276)
    app_286333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'app', False)
    # Obtaining the member 'add_config_value' of a type (line 276)
    add_config_value_286334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 4), app_286333, 'add_config_value')
    # Calling add_config_value(args, kwargs) (line 276)
    add_config_value_call_result_286339 = invoke(stypy.reporting.localization.Localization(__file__, 276, 4), add_config_value_286334, *[unicode_286335, dict_286336, True_286337], **kwargs_286338)
    
    
    # Call to add_config_value(...): (line 277)
    # Processing the call arguments (line 277)
    unicode_286342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 25), 'unicode', u'plot_apply_rcparams')
    # Getting the type of 'False' (line 277)
    False_286343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 48), 'False', False)
    # Getting the type of 'True' (line 277)
    True_286344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 55), 'True', False)
    # Processing the call keyword arguments (line 277)
    kwargs_286345 = {}
    # Getting the type of 'app' (line 277)
    app_286340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'app', False)
    # Obtaining the member 'add_config_value' of a type (line 277)
    add_config_value_286341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 4), app_286340, 'add_config_value')
    # Calling add_config_value(args, kwargs) (line 277)
    add_config_value_call_result_286346 = invoke(stypy.reporting.localization.Localization(__file__, 277, 4), add_config_value_286341, *[unicode_286342, False_286343, True_286344], **kwargs_286345)
    
    
    # Call to add_config_value(...): (line 278)
    # Processing the call arguments (line 278)
    unicode_286349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 25), 'unicode', u'plot_working_directory')
    # Getting the type of 'None' (line 278)
    None_286350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 51), 'None', False)
    # Getting the type of 'True' (line 278)
    True_286351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 57), 'True', False)
    # Processing the call keyword arguments (line 278)
    kwargs_286352 = {}
    # Getting the type of 'app' (line 278)
    app_286347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'app', False)
    # Obtaining the member 'add_config_value' of a type (line 278)
    add_config_value_286348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 4), app_286347, 'add_config_value')
    # Calling add_config_value(args, kwargs) (line 278)
    add_config_value_call_result_286353 = invoke(stypy.reporting.localization.Localization(__file__, 278, 4), add_config_value_286348, *[unicode_286349, None_286350, True_286351], **kwargs_286352)
    
    
    # Call to add_config_value(...): (line 279)
    # Processing the call arguments (line 279)
    unicode_286356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 25), 'unicode', u'plot_template')
    # Getting the type of 'None' (line 279)
    None_286357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 42), 'None', False)
    # Getting the type of 'True' (line 279)
    True_286358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 48), 'True', False)
    # Processing the call keyword arguments (line 279)
    kwargs_286359 = {}
    # Getting the type of 'app' (line 279)
    app_286354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'app', False)
    # Obtaining the member 'add_config_value' of a type (line 279)
    add_config_value_286355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 4), app_286354, 'add_config_value')
    # Calling add_config_value(args, kwargs) (line 279)
    add_config_value_call_result_286360 = invoke(stypy.reporting.localization.Localization(__file__, 279, 4), add_config_value_286355, *[unicode_286356, None_286357, True_286358], **kwargs_286359)
    
    
    # Call to connect(...): (line 281)
    # Processing the call arguments (line 281)
    
    # Call to str(...): (line 281)
    # Processing the call arguments (line 281)
    unicode_286364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 20), 'unicode', u'doctree-read')
    # Processing the call keyword arguments (line 281)
    kwargs_286365 = {}
    # Getting the type of 'str' (line 281)
    str_286363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'str', False)
    # Calling str(args, kwargs) (line 281)
    str_call_result_286366 = invoke(stypy.reporting.localization.Localization(__file__, 281, 16), str_286363, *[unicode_286364], **kwargs_286365)
    
    # Getting the type of 'mark_plot_labels' (line 281)
    mark_plot_labels_286367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 37), 'mark_plot_labels', False)
    # Processing the call keyword arguments (line 281)
    kwargs_286368 = {}
    # Getting the type of 'app' (line 281)
    app_286361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'app', False)
    # Obtaining the member 'connect' of a type (line 281)
    connect_286362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 4), app_286361, 'connect')
    # Calling connect(args, kwargs) (line 281)
    connect_call_result_286369 = invoke(stypy.reporting.localization.Localization(__file__, 281, 4), connect_286362, *[str_call_result_286366, mark_plot_labels_286367], **kwargs_286368)
    
    
    # Assigning a Dict to a Name (line 283):
    
    # Assigning a Dict to a Name (line 283):
    
    # Obtaining an instance of the builtin type 'dict' (line 283)
    dict_286370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 283)
    # Adding element type (key, value) (line 283)
    unicode_286371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 16), 'unicode', u'parallel_read_safe')
    # Getting the type of 'True' (line 283)
    True_286372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 38), 'True')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 15), dict_286370, (unicode_286371, True_286372))
    # Adding element type (key, value) (line 283)
    unicode_286373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 44), 'unicode', u'parallel_write_safe')
    # Getting the type of 'True' (line 283)
    True_286374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 67), 'True')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 15), dict_286370, (unicode_286373, True_286374))
    
    # Assigning a type to the variable 'metadata' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'metadata', dict_286370)
    # Getting the type of 'metadata' (line 284)
    metadata_286375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 11), 'metadata')
    # Assigning a type to the variable 'stypy_return_type' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'stypy_return_type', metadata_286375)
    
    # ################# End of 'setup(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'setup' in the type store
    # Getting the type of 'stypy_return_type' (line 251)
    stypy_return_type_286376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286376)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'setup'
    return stypy_return_type_286376

# Assigning a type to the variable 'setup' (line 251)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 0), 'setup', setup)

@norecursion
def contains_doctest(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'contains_doctest'
    module_type_store = module_type_store.open_function_context('contains_doctest', 290, 0, False)
    
    # Passed parameters checking function
    contains_doctest.stypy_localization = localization
    contains_doctest.stypy_type_of_self = None
    contains_doctest.stypy_type_store = module_type_store
    contains_doctest.stypy_function_name = 'contains_doctest'
    contains_doctest.stypy_param_names_list = ['text']
    contains_doctest.stypy_varargs_param_name = None
    contains_doctest.stypy_kwargs_param_name = None
    contains_doctest.stypy_call_defaults = defaults
    contains_doctest.stypy_call_varargs = varargs
    contains_doctest.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'contains_doctest', ['text'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'contains_doctest', localization, ['text'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'contains_doctest(...)' code ##################

    
    
    # SSA begins for try-except statement (line 291)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to compile(...): (line 293)
    # Processing the call arguments (line 293)
    # Getting the type of 'text' (line 293)
    text_286378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 16), 'text', False)
    unicode_286379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 22), 'unicode', u'<string>')
    unicode_286380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 34), 'unicode', u'exec')
    # Processing the call keyword arguments (line 293)
    kwargs_286381 = {}
    # Getting the type of 'compile' (line 293)
    compile_286377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'compile', False)
    # Calling compile(args, kwargs) (line 293)
    compile_call_result_286382 = invoke(stypy.reporting.localization.Localization(__file__, 293, 8), compile_286377, *[text_286378, unicode_286379, unicode_286380], **kwargs_286381)
    
    # Getting the type of 'False' (line 294)
    False_286383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'stypy_return_type', False_286383)
    # SSA branch for the except part of a try statement (line 291)
    # SSA branch for the except 'SyntaxError' branch of a try statement (line 291)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 291)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 297):
    
    # Assigning a Call to a Name (line 297):
    
    # Call to compile(...): (line 297)
    # Processing the call arguments (line 297)
    unicode_286386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 19), 'unicode', u'^\\s*>>>')
    # Getting the type of 're' (line 297)
    re_286387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 31), 're', False)
    # Obtaining the member 'M' of a type (line 297)
    M_286388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 31), re_286387, 'M')
    # Processing the call keyword arguments (line 297)
    kwargs_286389 = {}
    # Getting the type of 're' (line 297)
    re_286384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 're', False)
    # Obtaining the member 'compile' of a type (line 297)
    compile_286385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 8), re_286384, 'compile')
    # Calling compile(args, kwargs) (line 297)
    compile_call_result_286390 = invoke(stypy.reporting.localization.Localization(__file__, 297, 8), compile_286385, *[unicode_286386, M_286388], **kwargs_286389)
    
    # Assigning a type to the variable 'r' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'r', compile_call_result_286390)
    
    # Assigning a Call to a Name (line 298):
    
    # Assigning a Call to a Name (line 298):
    
    # Call to search(...): (line 298)
    # Processing the call arguments (line 298)
    # Getting the type of 'text' (line 298)
    text_286393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 17), 'text', False)
    # Processing the call keyword arguments (line 298)
    kwargs_286394 = {}
    # Getting the type of 'r' (line 298)
    r_286391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'r', False)
    # Obtaining the member 'search' of a type (line 298)
    search_286392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), r_286391, 'search')
    # Calling search(args, kwargs) (line 298)
    search_call_result_286395 = invoke(stypy.reporting.localization.Localization(__file__, 298, 8), search_286392, *[text_286393], **kwargs_286394)
    
    # Assigning a type to the variable 'm' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'm', search_call_result_286395)
    
    # Call to bool(...): (line 299)
    # Processing the call arguments (line 299)
    # Getting the type of 'm' (line 299)
    m_286397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 16), 'm', False)
    # Processing the call keyword arguments (line 299)
    kwargs_286398 = {}
    # Getting the type of 'bool' (line 299)
    bool_286396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 11), 'bool', False)
    # Calling bool(args, kwargs) (line 299)
    bool_call_result_286399 = invoke(stypy.reporting.localization.Localization(__file__, 299, 11), bool_286396, *[m_286397], **kwargs_286398)
    
    # Assigning a type to the variable 'stypy_return_type' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'stypy_return_type', bool_call_result_286399)
    
    # ################# End of 'contains_doctest(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'contains_doctest' in the type store
    # Getting the type of 'stypy_return_type' (line 290)
    stypy_return_type_286400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286400)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'contains_doctest'
    return stypy_return_type_286400

# Assigning a type to the variable 'contains_doctest' (line 290)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 0), 'contains_doctest', contains_doctest)

@norecursion
def unescape_doctest(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'unescape_doctest'
    module_type_store = module_type_store.open_function_context('unescape_doctest', 302, 0, False)
    
    # Passed parameters checking function
    unescape_doctest.stypy_localization = localization
    unescape_doctest.stypy_type_of_self = None
    unescape_doctest.stypy_type_store = module_type_store
    unescape_doctest.stypy_function_name = 'unescape_doctest'
    unescape_doctest.stypy_param_names_list = ['text']
    unescape_doctest.stypy_varargs_param_name = None
    unescape_doctest.stypy_kwargs_param_name = None
    unescape_doctest.stypy_call_defaults = defaults
    unescape_doctest.stypy_call_varargs = varargs
    unescape_doctest.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'unescape_doctest', ['text'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'unescape_doctest', localization, ['text'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'unescape_doctest(...)' code ##################

    unicode_286401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, (-1)), 'unicode', u'\n    Extract code from a piece of text, which contains either Python code\n    or doctests.\n\n    ')
    
    
    
    # Call to contains_doctest(...): (line 308)
    # Processing the call arguments (line 308)
    # Getting the type of 'text' (line 308)
    text_286403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 28), 'text', False)
    # Processing the call keyword arguments (line 308)
    kwargs_286404 = {}
    # Getting the type of 'contains_doctest' (line 308)
    contains_doctest_286402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 11), 'contains_doctest', False)
    # Calling contains_doctest(args, kwargs) (line 308)
    contains_doctest_call_result_286405 = invoke(stypy.reporting.localization.Localization(__file__, 308, 11), contains_doctest_286402, *[text_286403], **kwargs_286404)
    
    # Applying the 'not' unary operator (line 308)
    result_not__286406 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 7), 'not', contains_doctest_call_result_286405)
    
    # Testing the type of an if condition (line 308)
    if_condition_286407 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 308, 4), result_not__286406)
    # Assigning a type to the variable 'if_condition_286407' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'if_condition_286407', if_condition_286407)
    # SSA begins for if statement (line 308)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'text' (line 309)
    text_286408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 15), 'text')
    # Assigning a type to the variable 'stypy_return_type' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'stypy_return_type', text_286408)
    # SSA join for if statement (line 308)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Str to a Name (line 311):
    
    # Assigning a Str to a Name (line 311):
    unicode_286409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 11), 'unicode', u'')
    # Assigning a type to the variable 'code' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'code', unicode_286409)
    
    
    # Call to split(...): (line 312)
    # Processing the call arguments (line 312)
    unicode_286412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 27), 'unicode', u'\n')
    # Processing the call keyword arguments (line 312)
    kwargs_286413 = {}
    # Getting the type of 'text' (line 312)
    text_286410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 16), 'text', False)
    # Obtaining the member 'split' of a type (line 312)
    split_286411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 16), text_286410, 'split')
    # Calling split(args, kwargs) (line 312)
    split_call_result_286414 = invoke(stypy.reporting.localization.Localization(__file__, 312, 16), split_286411, *[unicode_286412], **kwargs_286413)
    
    # Testing the type of a for loop iterable (line 312)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 312, 4), split_call_result_286414)
    # Getting the type of the for loop variable (line 312)
    for_loop_var_286415 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 312, 4), split_call_result_286414)
    # Assigning a type to the variable 'line' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'line', for_loop_var_286415)
    # SSA begins for a for statement (line 312)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 313):
    
    # Assigning a Call to a Name (line 313):
    
    # Call to match(...): (line 313)
    # Processing the call arguments (line 313)
    unicode_286418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 21), 'unicode', u'^\\s*(>>>|\\.\\.\\.) (.*)$')
    # Getting the type of 'line' (line 313)
    line_286419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 48), 'line', False)
    # Processing the call keyword arguments (line 313)
    kwargs_286420 = {}
    # Getting the type of 're' (line 313)
    re_286416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 're', False)
    # Obtaining the member 'match' of a type (line 313)
    match_286417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 12), re_286416, 'match')
    # Calling match(args, kwargs) (line 313)
    match_call_result_286421 = invoke(stypy.reporting.localization.Localization(__file__, 313, 12), match_286417, *[unicode_286418, line_286419], **kwargs_286420)
    
    # Assigning a type to the variable 'm' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'm', match_call_result_286421)
    
    # Getting the type of 'm' (line 314)
    m_286422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 11), 'm')
    # Testing the type of an if condition (line 314)
    if_condition_286423 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 314, 8), m_286422)
    # Assigning a type to the variable 'if_condition_286423' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'if_condition_286423', if_condition_286423)
    # SSA begins for if statement (line 314)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'code' (line 315)
    code_286424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'code')
    
    # Call to group(...): (line 315)
    # Processing the call arguments (line 315)
    int_286427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 28), 'int')
    # Processing the call keyword arguments (line 315)
    kwargs_286428 = {}
    # Getting the type of 'm' (line 315)
    m_286425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 20), 'm', False)
    # Obtaining the member 'group' of a type (line 315)
    group_286426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 20), m_286425, 'group')
    # Calling group(args, kwargs) (line 315)
    group_call_result_286429 = invoke(stypy.reporting.localization.Localization(__file__, 315, 20), group_286426, *[int_286427], **kwargs_286428)
    
    unicode_286430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 33), 'unicode', u'\n')
    # Applying the binary operator '+' (line 315)
    result_add_286431 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 20), '+', group_call_result_286429, unicode_286430)
    
    # Applying the binary operator '+=' (line 315)
    result_iadd_286432 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 12), '+=', code_286424, result_add_286431)
    # Assigning a type to the variable 'code' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'code', result_iadd_286432)
    
    # SSA branch for the else part of an if statement (line 314)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to strip(...): (line 316)
    # Processing the call keyword arguments (line 316)
    kwargs_286435 = {}
    # Getting the type of 'line' (line 316)
    line_286433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 13), 'line', False)
    # Obtaining the member 'strip' of a type (line 316)
    strip_286434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 13), line_286433, 'strip')
    # Calling strip(args, kwargs) (line 316)
    strip_call_result_286436 = invoke(stypy.reporting.localization.Localization(__file__, 316, 13), strip_286434, *[], **kwargs_286435)
    
    # Testing the type of an if condition (line 316)
    if_condition_286437 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 316, 13), strip_call_result_286436)
    # Assigning a type to the variable 'if_condition_286437' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 13), 'if_condition_286437', if_condition_286437)
    # SSA begins for if statement (line 316)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'code' (line 317)
    code_286438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'code')
    unicode_286439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 20), 'unicode', u'# ')
    
    # Call to strip(...): (line 317)
    # Processing the call keyword arguments (line 317)
    kwargs_286442 = {}
    # Getting the type of 'line' (line 317)
    line_286440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 27), 'line', False)
    # Obtaining the member 'strip' of a type (line 317)
    strip_286441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 27), line_286440, 'strip')
    # Calling strip(args, kwargs) (line 317)
    strip_call_result_286443 = invoke(stypy.reporting.localization.Localization(__file__, 317, 27), strip_286441, *[], **kwargs_286442)
    
    # Applying the binary operator '+' (line 317)
    result_add_286444 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 20), '+', unicode_286439, strip_call_result_286443)
    
    unicode_286445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 42), 'unicode', u'\n')
    # Applying the binary operator '+' (line 317)
    result_add_286446 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 40), '+', result_add_286444, unicode_286445)
    
    # Applying the binary operator '+=' (line 317)
    result_iadd_286447 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 12), '+=', code_286438, result_add_286446)
    # Assigning a type to the variable 'code' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'code', result_iadd_286447)
    
    # SSA branch for the else part of an if statement (line 316)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'code' (line 319)
    code_286448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'code')
    unicode_286449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 20), 'unicode', u'\n')
    # Applying the binary operator '+=' (line 319)
    result_iadd_286450 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 12), '+=', code_286448, unicode_286449)
    # Assigning a type to the variable 'code' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'code', result_iadd_286450)
    
    # SSA join for if statement (line 316)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 314)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'code' (line 320)
    code_286451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 11), 'code')
    # Assigning a type to the variable 'stypy_return_type' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'stypy_return_type', code_286451)
    
    # ################# End of 'unescape_doctest(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'unescape_doctest' in the type store
    # Getting the type of 'stypy_return_type' (line 302)
    stypy_return_type_286452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286452)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'unescape_doctest'
    return stypy_return_type_286452

# Assigning a type to the variable 'unescape_doctest' (line 302)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 0), 'unescape_doctest', unescape_doctest)

@norecursion
def split_code_at_show(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'split_code_at_show'
    module_type_store = module_type_store.open_function_context('split_code_at_show', 323, 0, False)
    
    # Passed parameters checking function
    split_code_at_show.stypy_localization = localization
    split_code_at_show.stypy_type_of_self = None
    split_code_at_show.stypy_type_store = module_type_store
    split_code_at_show.stypy_function_name = 'split_code_at_show'
    split_code_at_show.stypy_param_names_list = ['text']
    split_code_at_show.stypy_varargs_param_name = None
    split_code_at_show.stypy_kwargs_param_name = None
    split_code_at_show.stypy_call_defaults = defaults
    split_code_at_show.stypy_call_varargs = varargs
    split_code_at_show.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'split_code_at_show', ['text'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'split_code_at_show', localization, ['text'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'split_code_at_show(...)' code ##################

    unicode_286453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, (-1)), 'unicode', u'\n    Split code at plt.show()\n\n    ')
    
    # Assigning a List to a Name (line 329):
    
    # Assigning a List to a Name (line 329):
    
    # Obtaining an instance of the builtin type 'list' (line 329)
    list_286454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 329)
    
    # Assigning a type to the variable 'parts' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'parts', list_286454)
    
    # Assigning a Call to a Name (line 330):
    
    # Assigning a Call to a Name (line 330):
    
    # Call to contains_doctest(...): (line 330)
    # Processing the call arguments (line 330)
    # Getting the type of 'text' (line 330)
    text_286456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 34), 'text', False)
    # Processing the call keyword arguments (line 330)
    kwargs_286457 = {}
    # Getting the type of 'contains_doctest' (line 330)
    contains_doctest_286455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 17), 'contains_doctest', False)
    # Calling contains_doctest(args, kwargs) (line 330)
    contains_doctest_call_result_286458 = invoke(stypy.reporting.localization.Localization(__file__, 330, 17), contains_doctest_286455, *[text_286456], **kwargs_286457)
    
    # Assigning a type to the variable 'is_doctest' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'is_doctest', contains_doctest_call_result_286458)
    
    # Assigning a List to a Name (line 332):
    
    # Assigning a List to a Name (line 332):
    
    # Obtaining an instance of the builtin type 'list' (line 332)
    list_286459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 332)
    
    # Assigning a type to the variable 'part' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'part', list_286459)
    
    
    # Call to split(...): (line 333)
    # Processing the call arguments (line 333)
    unicode_286462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 27), 'unicode', u'\n')
    # Processing the call keyword arguments (line 333)
    kwargs_286463 = {}
    # Getting the type of 'text' (line 333)
    text_286460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 16), 'text', False)
    # Obtaining the member 'split' of a type (line 333)
    split_286461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 16), text_286460, 'split')
    # Calling split(args, kwargs) (line 333)
    split_call_result_286464 = invoke(stypy.reporting.localization.Localization(__file__, 333, 16), split_286461, *[unicode_286462], **kwargs_286463)
    
    # Testing the type of a for loop iterable (line 333)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 333, 4), split_call_result_286464)
    # Getting the type of the for loop variable (line 333)
    for_loop_var_286465 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 333, 4), split_call_result_286464)
    # Assigning a type to the variable 'line' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'line', for_loop_var_286465)
    # SSA begins for a for statement (line 333)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    # Getting the type of 'is_doctest' (line 334)
    is_doctest_286466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'is_doctest')
    # Applying the 'not' unary operator (line 334)
    result_not__286467 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 12), 'not', is_doctest_286466)
    
    
    
    # Call to strip(...): (line 334)
    # Processing the call keyword arguments (line 334)
    kwargs_286470 = {}
    # Getting the type of 'line' (line 334)
    line_286468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 31), 'line', False)
    # Obtaining the member 'strip' of a type (line 334)
    strip_286469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 31), line_286468, 'strip')
    # Calling strip(args, kwargs) (line 334)
    strip_call_result_286471 = invoke(stypy.reporting.localization.Localization(__file__, 334, 31), strip_286469, *[], **kwargs_286470)
    
    unicode_286472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 47), 'unicode', u'plt.show()')
    # Applying the binary operator '==' (line 334)
    result_eq_286473 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 31), '==', strip_call_result_286471, unicode_286472)
    
    # Applying the binary operator 'and' (line 334)
    result_and_keyword_286474 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 12), 'and', result_not__286467, result_eq_286473)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'is_doctest' (line 335)
    is_doctest_286475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 16), 'is_doctest')
    
    
    # Call to strip(...): (line 335)
    # Processing the call keyword arguments (line 335)
    kwargs_286478 = {}
    # Getting the type of 'line' (line 335)
    line_286476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 31), 'line', False)
    # Obtaining the member 'strip' of a type (line 335)
    strip_286477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 31), line_286476, 'strip')
    # Calling strip(args, kwargs) (line 335)
    strip_call_result_286479 = invoke(stypy.reporting.localization.Localization(__file__, 335, 31), strip_286477, *[], **kwargs_286478)
    
    unicode_286480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 47), 'unicode', u'>>> plt.show()')
    # Applying the binary operator '==' (line 335)
    result_eq_286481 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 31), '==', strip_call_result_286479, unicode_286480)
    
    # Applying the binary operator 'and' (line 335)
    result_and_keyword_286482 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 16), 'and', is_doctest_286475, result_eq_286481)
    
    # Applying the binary operator 'or' (line 334)
    result_or_keyword_286483 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 11), 'or', result_and_keyword_286474, result_and_keyword_286482)
    
    # Testing the type of an if condition (line 334)
    if_condition_286484 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 8), result_or_keyword_286483)
    # Assigning a type to the variable 'if_condition_286484' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'if_condition_286484', if_condition_286484)
    # SSA begins for if statement (line 334)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 336)
    # Processing the call arguments (line 336)
    # Getting the type of 'line' (line 336)
    line_286487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 24), 'line', False)
    # Processing the call keyword arguments (line 336)
    kwargs_286488 = {}
    # Getting the type of 'part' (line 336)
    part_286485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'part', False)
    # Obtaining the member 'append' of a type (line 336)
    append_286486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 12), part_286485, 'append')
    # Calling append(args, kwargs) (line 336)
    append_call_result_286489 = invoke(stypy.reporting.localization.Localization(__file__, 336, 12), append_286486, *[line_286487], **kwargs_286488)
    
    
    # Call to append(...): (line 337)
    # Processing the call arguments (line 337)
    
    # Call to join(...): (line 337)
    # Processing the call arguments (line 337)
    # Getting the type of 'part' (line 337)
    part_286494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 35), 'part', False)
    # Processing the call keyword arguments (line 337)
    kwargs_286495 = {}
    unicode_286492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 25), 'unicode', u'\n')
    # Obtaining the member 'join' of a type (line 337)
    join_286493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 25), unicode_286492, 'join')
    # Calling join(args, kwargs) (line 337)
    join_call_result_286496 = invoke(stypy.reporting.localization.Localization(__file__, 337, 25), join_286493, *[part_286494], **kwargs_286495)
    
    # Processing the call keyword arguments (line 337)
    kwargs_286497 = {}
    # Getting the type of 'parts' (line 337)
    parts_286490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'parts', False)
    # Obtaining the member 'append' of a type (line 337)
    append_286491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 12), parts_286490, 'append')
    # Calling append(args, kwargs) (line 337)
    append_call_result_286498 = invoke(stypy.reporting.localization.Localization(__file__, 337, 12), append_286491, *[join_call_result_286496], **kwargs_286497)
    
    
    # Assigning a List to a Name (line 338):
    
    # Assigning a List to a Name (line 338):
    
    # Obtaining an instance of the builtin type 'list' (line 338)
    list_286499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 338)
    
    # Assigning a type to the variable 'part' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'part', list_286499)
    # SSA branch for the else part of an if statement (line 334)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 340)
    # Processing the call arguments (line 340)
    # Getting the type of 'line' (line 340)
    line_286502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 24), 'line', False)
    # Processing the call keyword arguments (line 340)
    kwargs_286503 = {}
    # Getting the type of 'part' (line 340)
    part_286500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 12), 'part', False)
    # Obtaining the member 'append' of a type (line 340)
    append_286501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 12), part_286500, 'append')
    # Calling append(args, kwargs) (line 340)
    append_call_result_286504 = invoke(stypy.reporting.localization.Localization(__file__, 340, 12), append_286501, *[line_286502], **kwargs_286503)
    
    # SSA join for if statement (line 334)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to strip(...): (line 341)
    # Processing the call keyword arguments (line 341)
    kwargs_286511 = {}
    
    # Call to join(...): (line 341)
    # Processing the call arguments (line 341)
    # Getting the type of 'part' (line 341)
    part_286507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 17), 'part', False)
    # Processing the call keyword arguments (line 341)
    kwargs_286508 = {}
    unicode_286505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 7), 'unicode', u'\n')
    # Obtaining the member 'join' of a type (line 341)
    join_286506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 7), unicode_286505, 'join')
    # Calling join(args, kwargs) (line 341)
    join_call_result_286509 = invoke(stypy.reporting.localization.Localization(__file__, 341, 7), join_286506, *[part_286507], **kwargs_286508)
    
    # Obtaining the member 'strip' of a type (line 341)
    strip_286510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 7), join_call_result_286509, 'strip')
    # Calling strip(args, kwargs) (line 341)
    strip_call_result_286512 = invoke(stypy.reporting.localization.Localization(__file__, 341, 7), strip_286510, *[], **kwargs_286511)
    
    # Testing the type of an if condition (line 341)
    if_condition_286513 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 341, 4), strip_call_result_286512)
    # Assigning a type to the variable 'if_condition_286513' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'if_condition_286513', if_condition_286513)
    # SSA begins for if statement (line 341)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 342)
    # Processing the call arguments (line 342)
    
    # Call to join(...): (line 342)
    # Processing the call arguments (line 342)
    # Getting the type of 'part' (line 342)
    part_286518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 31), 'part', False)
    # Processing the call keyword arguments (line 342)
    kwargs_286519 = {}
    unicode_286516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 21), 'unicode', u'\n')
    # Obtaining the member 'join' of a type (line 342)
    join_286517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 21), unicode_286516, 'join')
    # Calling join(args, kwargs) (line 342)
    join_call_result_286520 = invoke(stypy.reporting.localization.Localization(__file__, 342, 21), join_286517, *[part_286518], **kwargs_286519)
    
    # Processing the call keyword arguments (line 342)
    kwargs_286521 = {}
    # Getting the type of 'parts' (line 342)
    parts_286514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'parts', False)
    # Obtaining the member 'append' of a type (line 342)
    append_286515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 8), parts_286514, 'append')
    # Calling append(args, kwargs) (line 342)
    append_call_result_286522 = invoke(stypy.reporting.localization.Localization(__file__, 342, 8), append_286515, *[join_call_result_286520], **kwargs_286521)
    
    # SSA join for if statement (line 341)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'parts' (line 343)
    parts_286523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 11), 'parts')
    # Assigning a type to the variable 'stypy_return_type' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'stypy_return_type', parts_286523)
    
    # ################# End of 'split_code_at_show(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'split_code_at_show' in the type store
    # Getting the type of 'stypy_return_type' (line 323)
    stypy_return_type_286524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286524)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'split_code_at_show'
    return stypy_return_type_286524

# Assigning a type to the variable 'split_code_at_show' (line 323)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 0), 'split_code_at_show', split_code_at_show)

@norecursion
def remove_coding(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'remove_coding'
    module_type_store = module_type_store.open_function_context('remove_coding', 346, 0, False)
    
    # Passed parameters checking function
    remove_coding.stypy_localization = localization
    remove_coding.stypy_type_of_self = None
    remove_coding.stypy_type_store = module_type_store
    remove_coding.stypy_function_name = 'remove_coding'
    remove_coding.stypy_param_names_list = ['text']
    remove_coding.stypy_varargs_param_name = None
    remove_coding.stypy_kwargs_param_name = None
    remove_coding.stypy_call_defaults = defaults
    remove_coding.stypy_call_varargs = varargs
    remove_coding.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'remove_coding', ['text'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'remove_coding', localization, ['text'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'remove_coding(...)' code ##################

    unicode_286525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, (-1)), 'unicode', u"\n    Remove the coding comment, which six.exec\\_ doesn't like.\n    ")
    
    # Assigning a Call to a Name (line 350):
    
    # Assigning a Call to a Name (line 350):
    
    # Call to compile(...): (line 350)
    # Processing the call arguments (line 350)
    unicode_286528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 24), 'unicode', u'^#\\s*-\\*-\\s*coding:\\s*.*-\\*-$')
    # Processing the call keyword arguments (line 350)
    # Getting the type of 're' (line 350)
    re_286529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 63), 're', False)
    # Obtaining the member 'MULTILINE' of a type (line 350)
    MULTILINE_286530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 63), re_286529, 'MULTILINE')
    keyword_286531 = MULTILINE_286530
    kwargs_286532 = {'flags': keyword_286531}
    # Getting the type of 're' (line 350)
    re_286526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 13), 're', False)
    # Obtaining the member 'compile' of a type (line 350)
    compile_286527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 13), re_286526, 'compile')
    # Calling compile(args, kwargs) (line 350)
    compile_call_result_286533 = invoke(stypy.reporting.localization.Localization(__file__, 350, 13), compile_286527, *[unicode_286528], **kwargs_286532)
    
    # Assigning a type to the variable 'sub_re' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'sub_re', compile_call_result_286533)
    
    # Call to sub(...): (line 351)
    # Processing the call arguments (line 351)
    unicode_286536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 22), 'unicode', u'')
    # Getting the type of 'text' (line 351)
    text_286537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 26), 'text', False)
    # Processing the call keyword arguments (line 351)
    kwargs_286538 = {}
    # Getting the type of 'sub_re' (line 351)
    sub_re_286534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 11), 'sub_re', False)
    # Obtaining the member 'sub' of a type (line 351)
    sub_286535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 11), sub_re_286534, 'sub')
    # Calling sub(args, kwargs) (line 351)
    sub_call_result_286539 = invoke(stypy.reporting.localization.Localization(__file__, 351, 11), sub_286535, *[unicode_286536, text_286537], **kwargs_286538)
    
    # Assigning a type to the variable 'stypy_return_type' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'stypy_return_type', sub_call_result_286539)
    
    # ################# End of 'remove_coding(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'remove_coding' in the type store
    # Getting the type of 'stypy_return_type' (line 346)
    stypy_return_type_286540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286540)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'remove_coding'
    return stypy_return_type_286540

# Assigning a type to the variable 'remove_coding' (line 346)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 0), 'remove_coding', remove_coding)

# Assigning a Str to a Name (line 358):

# Assigning a Str to a Name (line 358):
unicode_286541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, (-1)), 'unicode', u"\n{{ source_code }}\n\n{{ only_html }}\n\n   {% if source_link or (html_show_formats and not multi_image) %}\n   (\n   {%- if source_link -%}\n   `Source code <{{ source_link }}>`__\n   {%- endif -%}\n   {%- if html_show_formats and not multi_image -%}\n     {%- for img in images -%}\n       {%- for fmt in img.formats -%}\n         {%- if source_link or not loop.first -%}, {% endif -%}\n         `{{ fmt }} <{{ dest_dir }}/{{ img.basename }}.{{ fmt }}>`__\n       {%- endfor -%}\n     {%- endfor -%}\n   {%- endif -%}\n   )\n   {% endif %}\n\n   {% for img in images %}\n   .. figure:: {{ build_dir }}/{{ img.basename }}.{{ default_fmt }}\n      {% for option in options -%}\n      {{ option }}\n      {% endfor %}\n\n      {% if html_show_formats and multi_image -%}\n        (\n        {%- for fmt in img.formats -%}\n        {%- if not loop.first -%}, {% endif -%}\n        `{{ fmt }} <{{ dest_dir }}/{{ img.basename }}.{{ fmt }}>`__\n        {%- endfor -%}\n        )\n      {%- endif -%}\n\n      {{ caption }}\n   {% endfor %}\n\n{{ only_latex }}\n\n   {% for img in images %}\n   {% if 'pdf' in img.formats -%}\n   .. figure:: {{ build_dir }}/{{ img.basename }}.pdf\n      {% for option in options -%}\n      {{ option }}\n      {% endfor %}\n\n      {{ caption }}\n   {% endif -%}\n   {% endfor %}\n\n{{ only_texinfo }}\n\n   {% for img in images %}\n   .. image:: {{ build_dir }}/{{ img.basename }}.png\n      {% for option in options -%}\n      {{ option }}\n      {% endfor %}\n\n   {% endfor %}\n\n")
# Assigning a type to the variable 'TEMPLATE' (line 358)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 0), 'TEMPLATE', unicode_286541)

# Assigning a Str to a Name (line 422):

# Assigning a Str to a Name (line 422):
unicode_286542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, (-1)), 'unicode', u'\n.. htmlonly::\n\n   [`source code <%(linkdir)s/%(basename)s.py>`__]\n\nException occurred rendering plot.\n\n')
# Assigning a type to the variable 'exception_template' (line 422)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 0), 'exception_template', unicode_286542)

# Assigning a Call to a Name (line 433):

# Assigning a Call to a Name (line 433):

# Call to dict(...): (line 433)
# Processing the call keyword arguments (line 433)
kwargs_286544 = {}
# Getting the type of 'dict' (line 433)
dict_286543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 15), 'dict', False)
# Calling dict(args, kwargs) (line 433)
dict_call_result_286545 = invoke(stypy.reporting.localization.Localization(__file__, 433, 15), dict_286543, *[], **kwargs_286544)

# Assigning a type to the variable 'plot_context' (line 433)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 0), 'plot_context', dict_call_result_286545)
# Declaration of the 'ImageFile' class

class ImageFile(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 436, 4, False)
        # Assigning a type to the variable 'self' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ImageFile.__init__', ['basename', 'dirname'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['basename', 'dirname'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 437):
        
        # Assigning a Name to a Attribute (line 437):
        # Getting the type of 'basename' (line 437)
        basename_286546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 24), 'basename')
        # Getting the type of 'self' (line 437)
        self_286547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'self')
        # Setting the type of the member 'basename' of a type (line 437)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), self_286547, 'basename', basename_286546)
        
        # Assigning a Name to a Attribute (line 438):
        
        # Assigning a Name to a Attribute (line 438):
        # Getting the type of 'dirname' (line 438)
        dirname_286548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 23), 'dirname')
        # Getting the type of 'self' (line 438)
        self_286549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'self')
        # Setting the type of the member 'dirname' of a type (line 438)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 8), self_286549, 'dirname', dirname_286548)
        
        # Assigning a List to a Attribute (line 439):
        
        # Assigning a List to a Attribute (line 439):
        
        # Obtaining an instance of the builtin type 'list' (line 439)
        list_286550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 439)
        
        # Getting the type of 'self' (line 439)
        self_286551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'self')
        # Setting the type of the member 'formats' of a type (line 439)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 8), self_286551, 'formats', list_286550)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def filename(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'filename'
        module_type_store = module_type_store.open_function_context('filename', 441, 4, False)
        # Assigning a type to the variable 'self' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ImageFile.filename.__dict__.__setitem__('stypy_localization', localization)
        ImageFile.filename.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ImageFile.filename.__dict__.__setitem__('stypy_type_store', module_type_store)
        ImageFile.filename.__dict__.__setitem__('stypy_function_name', 'ImageFile.filename')
        ImageFile.filename.__dict__.__setitem__('stypy_param_names_list', ['format'])
        ImageFile.filename.__dict__.__setitem__('stypy_varargs_param_name', None)
        ImageFile.filename.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ImageFile.filename.__dict__.__setitem__('stypy_call_defaults', defaults)
        ImageFile.filename.__dict__.__setitem__('stypy_call_varargs', varargs)
        ImageFile.filename.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ImageFile.filename.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ImageFile.filename', ['format'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'filename', localization, ['format'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'filename(...)' code ##################

        
        # Call to join(...): (line 442)
        # Processing the call arguments (line 442)
        # Getting the type of 'self' (line 442)
        self_286555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 28), 'self', False)
        # Obtaining the member 'dirname' of a type (line 442)
        dirname_286556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 28), self_286555, 'dirname')
        unicode_286557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 42), 'unicode', u'%s.%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 442)
        tuple_286558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 53), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 442)
        # Adding element type (line 442)
        # Getting the type of 'self' (line 442)
        self_286559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 53), 'self', False)
        # Obtaining the member 'basename' of a type (line 442)
        basename_286560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 53), self_286559, 'basename')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 53), tuple_286558, basename_286560)
        # Adding element type (line 442)
        # Getting the type of 'format' (line 442)
        format_286561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 68), 'format', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 53), tuple_286558, format_286561)
        
        # Applying the binary operator '%' (line 442)
        result_mod_286562 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 42), '%', unicode_286557, tuple_286558)
        
        # Processing the call keyword arguments (line 442)
        kwargs_286563 = {}
        # Getting the type of 'os' (line 442)
        os_286552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 442)
        path_286553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 15), os_286552, 'path')
        # Obtaining the member 'join' of a type (line 442)
        join_286554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 15), path_286553, 'join')
        # Calling join(args, kwargs) (line 442)
        join_call_result_286564 = invoke(stypy.reporting.localization.Localization(__file__, 442, 15), join_286554, *[dirname_286556, result_mod_286562], **kwargs_286563)
        
        # Assigning a type to the variable 'stypy_return_type' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'stypy_return_type', join_call_result_286564)
        
        # ################# End of 'filename(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'filename' in the type store
        # Getting the type of 'stypy_return_type' (line 441)
        stypy_return_type_286565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_286565)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'filename'
        return stypy_return_type_286565


    @norecursion
    def filenames(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'filenames'
        module_type_store = module_type_store.open_function_context('filenames', 444, 4, False)
        # Assigning a type to the variable 'self' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ImageFile.filenames.__dict__.__setitem__('stypy_localization', localization)
        ImageFile.filenames.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ImageFile.filenames.__dict__.__setitem__('stypy_type_store', module_type_store)
        ImageFile.filenames.__dict__.__setitem__('stypy_function_name', 'ImageFile.filenames')
        ImageFile.filenames.__dict__.__setitem__('stypy_param_names_list', [])
        ImageFile.filenames.__dict__.__setitem__('stypy_varargs_param_name', None)
        ImageFile.filenames.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ImageFile.filenames.__dict__.__setitem__('stypy_call_defaults', defaults)
        ImageFile.filenames.__dict__.__setitem__('stypy_call_varargs', varargs)
        ImageFile.filenames.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ImageFile.filenames.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ImageFile.filenames', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'filenames', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'filenames(...)' code ##################

        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 445)
        self_286571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 46), 'self')
        # Obtaining the member 'formats' of a type (line 445)
        formats_286572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 46), self_286571, 'formats')
        comprehension_286573 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 16), formats_286572)
        # Assigning a type to the variable 'fmt' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 16), 'fmt', comprehension_286573)
        
        # Call to filename(...): (line 445)
        # Processing the call arguments (line 445)
        # Getting the type of 'fmt' (line 445)
        fmt_286568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 30), 'fmt', False)
        # Processing the call keyword arguments (line 445)
        kwargs_286569 = {}
        # Getting the type of 'self' (line 445)
        self_286566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 16), 'self', False)
        # Obtaining the member 'filename' of a type (line 445)
        filename_286567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 16), self_286566, 'filename')
        # Calling filename(args, kwargs) (line 445)
        filename_call_result_286570 = invoke(stypy.reporting.localization.Localization(__file__, 445, 16), filename_286567, *[fmt_286568], **kwargs_286569)
        
        list_286574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 16), list_286574, filename_call_result_286570)
        # Assigning a type to the variable 'stypy_return_type' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'stypy_return_type', list_286574)
        
        # ################# End of 'filenames(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'filenames' in the type store
        # Getting the type of 'stypy_return_type' (line 444)
        stypy_return_type_286575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_286575)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'filenames'
        return stypy_return_type_286575


# Assigning a type to the variable 'ImageFile' (line 435)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 0), 'ImageFile', ImageFile)

@norecursion
def out_of_date(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'out_of_date'
    module_type_store = module_type_store.open_function_context('out_of_date', 448, 0, False)
    
    # Passed parameters checking function
    out_of_date.stypy_localization = localization
    out_of_date.stypy_type_of_self = None
    out_of_date.stypy_type_store = module_type_store
    out_of_date.stypy_function_name = 'out_of_date'
    out_of_date.stypy_param_names_list = ['original', 'derived']
    out_of_date.stypy_varargs_param_name = None
    out_of_date.stypy_kwargs_param_name = None
    out_of_date.stypy_call_defaults = defaults
    out_of_date.stypy_call_varargs = varargs
    out_of_date.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'out_of_date', ['original', 'derived'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'out_of_date', localization, ['original', 'derived'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'out_of_date(...)' code ##################

    unicode_286576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, (-1)), 'unicode', u'\n    Returns True if derivative is out-of-date wrt original,\n    both of which are full file paths.\n    ')
    
    # Evaluating a boolean operation
    
    
    # Call to exists(...): (line 453)
    # Processing the call arguments (line 453)
    # Getting the type of 'derived' (line 453)
    derived_286580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 31), 'derived', False)
    # Processing the call keyword arguments (line 453)
    kwargs_286581 = {}
    # Getting the type of 'os' (line 453)
    os_286577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 16), 'os', False)
    # Obtaining the member 'path' of a type (line 453)
    path_286578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 16), os_286577, 'path')
    # Obtaining the member 'exists' of a type (line 453)
    exists_286579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 16), path_286578, 'exists')
    # Calling exists(args, kwargs) (line 453)
    exists_call_result_286582 = invoke(stypy.reporting.localization.Localization(__file__, 453, 16), exists_286579, *[derived_286580], **kwargs_286581)
    
    # Applying the 'not' unary operator (line 453)
    result_not__286583 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 12), 'not', exists_call_result_286582)
    
    
    # Evaluating a boolean operation
    
    # Call to exists(...): (line 454)
    # Processing the call arguments (line 454)
    # Getting the type of 'original' (line 454)
    original_286587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 28), 'original', False)
    # Processing the call keyword arguments (line 454)
    kwargs_286588 = {}
    # Getting the type of 'os' (line 454)
    os_286584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 13), 'os', False)
    # Obtaining the member 'path' of a type (line 454)
    path_286585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 13), os_286584, 'path')
    # Obtaining the member 'exists' of a type (line 454)
    exists_286586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 13), path_286585, 'exists')
    # Calling exists(args, kwargs) (line 454)
    exists_call_result_286589 = invoke(stypy.reporting.localization.Localization(__file__, 454, 13), exists_286586, *[original_286587], **kwargs_286588)
    
    
    
    # Call to stat(...): (line 455)
    # Processing the call arguments (line 455)
    # Getting the type of 'derived' (line 455)
    derived_286592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 21), 'derived', False)
    # Processing the call keyword arguments (line 455)
    kwargs_286593 = {}
    # Getting the type of 'os' (line 455)
    os_286590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 13), 'os', False)
    # Obtaining the member 'stat' of a type (line 455)
    stat_286591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 13), os_286590, 'stat')
    # Calling stat(args, kwargs) (line 455)
    stat_call_result_286594 = invoke(stypy.reporting.localization.Localization(__file__, 455, 13), stat_286591, *[derived_286592], **kwargs_286593)
    
    # Obtaining the member 'st_mtime' of a type (line 455)
    st_mtime_286595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 13), stat_call_result_286594, 'st_mtime')
    
    # Call to stat(...): (line 455)
    # Processing the call arguments (line 455)
    # Getting the type of 'original' (line 455)
    original_286598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 49), 'original', False)
    # Processing the call keyword arguments (line 455)
    kwargs_286599 = {}
    # Getting the type of 'os' (line 455)
    os_286596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 41), 'os', False)
    # Obtaining the member 'stat' of a type (line 455)
    stat_286597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 41), os_286596, 'stat')
    # Calling stat(args, kwargs) (line 455)
    stat_call_result_286600 = invoke(stypy.reporting.localization.Localization(__file__, 455, 41), stat_286597, *[original_286598], **kwargs_286599)
    
    # Obtaining the member 'st_mtime' of a type (line 455)
    st_mtime_286601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 41), stat_call_result_286600, 'st_mtime')
    # Applying the binary operator '<' (line 455)
    result_lt_286602 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 13), '<', st_mtime_286595, st_mtime_286601)
    
    # Applying the binary operator 'and' (line 454)
    result_and_keyword_286603 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 13), 'and', exists_call_result_286589, result_lt_286602)
    
    # Applying the binary operator 'or' (line 453)
    result_or_keyword_286604 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 12), 'or', result_not__286583, result_and_keyword_286603)
    
    # Assigning a type to the variable 'stypy_return_type' (line 453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'stypy_return_type', result_or_keyword_286604)
    
    # ################# End of 'out_of_date(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'out_of_date' in the type store
    # Getting the type of 'stypy_return_type' (line 448)
    stypy_return_type_286605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286605)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'out_of_date'
    return stypy_return_type_286605

# Assigning a type to the variable 'out_of_date' (line 448)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 0), 'out_of_date', out_of_date)
# Declaration of the 'PlotError' class
# Getting the type of 'RuntimeError' (line 458)
RuntimeError_286606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 16), 'RuntimeError')

class PlotError(RuntimeError_286606, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 458, 0, False)
        # Assigning a type to the variable 'self' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PlotError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'PlotError' (line 458)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 0), 'PlotError', PlotError)

@norecursion
def run_code(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 462)
    None_286607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 33), 'None')
    # Getting the type of 'None' (line 462)
    None_286608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 53), 'None')
    defaults = [None_286607, None_286608]
    # Create a new context for function 'run_code'
    module_type_store = module_type_store.open_function_context('run_code', 462, 0, False)
    
    # Passed parameters checking function
    run_code.stypy_localization = localization
    run_code.stypy_type_of_self = None
    run_code.stypy_type_store = module_type_store
    run_code.stypy_function_name = 'run_code'
    run_code.stypy_param_names_list = ['code', 'code_path', 'ns', 'function_name']
    run_code.stypy_varargs_param_name = None
    run_code.stypy_kwargs_param_name = None
    run_code.stypy_call_defaults = defaults
    run_code.stypy_call_varargs = varargs
    run_code.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'run_code', ['code', 'code_path', 'ns', 'function_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'run_code', localization, ['code', 'code_path', 'ns', 'function_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'run_code(...)' code ##################

    unicode_286609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, (-1)), 'unicode', u'\n    Import a Python module from a path, and run the function given by\n    name, if function_name is not None.\n    ')
    
    # Getting the type of 'six' (line 471)
    six_286610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 7), 'six')
    # Obtaining the member 'PY2' of a type (line 471)
    PY2_286611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 7), six_286610, 'PY2')
    # Testing the type of an if condition (line 471)
    if_condition_286612 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 471, 4), PY2_286611)
    # Assigning a type to the variable 'if_condition_286612' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'if_condition_286612', if_condition_286612)
    # SSA begins for if statement (line 471)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 472):
    
    # Assigning a Call to a Name (line 472):
    
    # Call to getcwdu(...): (line 472)
    # Processing the call keyword arguments (line 472)
    kwargs_286615 = {}
    # Getting the type of 'os' (line 472)
    os_286613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 14), 'os', False)
    # Obtaining the member 'getcwdu' of a type (line 472)
    getcwdu_286614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 14), os_286613, 'getcwdu')
    # Calling getcwdu(args, kwargs) (line 472)
    getcwdu_call_result_286616 = invoke(stypy.reporting.localization.Localization(__file__, 472, 14), getcwdu_286614, *[], **kwargs_286615)
    
    # Assigning a type to the variable 'pwd' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'pwd', getcwdu_call_result_286616)
    # SSA branch for the else part of an if statement (line 471)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 474):
    
    # Assigning a Call to a Name (line 474):
    
    # Call to getcwd(...): (line 474)
    # Processing the call keyword arguments (line 474)
    kwargs_286619 = {}
    # Getting the type of 'os' (line 474)
    os_286617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 14), 'os', False)
    # Obtaining the member 'getcwd' of a type (line 474)
    getcwd_286618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 14), os_286617, 'getcwd')
    # Calling getcwd(args, kwargs) (line 474)
    getcwd_call_result_286620 = invoke(stypy.reporting.localization.Localization(__file__, 474, 14), getcwd_286618, *[], **kwargs_286619)
    
    # Assigning a type to the variable 'pwd' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'pwd', getcwd_call_result_286620)
    # SSA join for if statement (line 471)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 475):
    
    # Assigning a Call to a Name (line 475):
    
    # Call to list(...): (line 475)
    # Processing the call arguments (line 475)
    # Getting the type of 'sys' (line 475)
    sys_286622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 24), 'sys', False)
    # Obtaining the member 'path' of a type (line 475)
    path_286623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 24), sys_286622, 'path')
    # Processing the call keyword arguments (line 475)
    kwargs_286624 = {}
    # Getting the type of 'list' (line 475)
    list_286621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 19), 'list', False)
    # Calling list(args, kwargs) (line 475)
    list_call_result_286625 = invoke(stypy.reporting.localization.Localization(__file__, 475, 19), list_286621, *[path_286623], **kwargs_286624)
    
    # Assigning a type to the variable 'old_sys_path' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'old_sys_path', list_call_result_286625)
    
    
    # Getting the type of 'setup' (line 476)
    setup_286626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 7), 'setup')
    # Obtaining the member 'config' of a type (line 476)
    config_286627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 7), setup_286626, 'config')
    # Obtaining the member 'plot_working_directory' of a type (line 476)
    plot_working_directory_286628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 7), config_286627, 'plot_working_directory')
    # Getting the type of 'None' (line 476)
    None_286629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 50), 'None')
    # Applying the binary operator 'isnot' (line 476)
    result_is_not_286630 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 7), 'isnot', plot_working_directory_286628, None_286629)
    
    # Testing the type of an if condition (line 476)
    if_condition_286631 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 476, 4), result_is_not_286630)
    # Assigning a type to the variable 'if_condition_286631' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'if_condition_286631', if_condition_286631)
    # SSA begins for if statement (line 476)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 477)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to chdir(...): (line 478)
    # Processing the call arguments (line 478)
    # Getting the type of 'setup' (line 478)
    setup_286634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 21), 'setup', False)
    # Obtaining the member 'config' of a type (line 478)
    config_286635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 21), setup_286634, 'config')
    # Obtaining the member 'plot_working_directory' of a type (line 478)
    plot_working_directory_286636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 21), config_286635, 'plot_working_directory')
    # Processing the call keyword arguments (line 478)
    kwargs_286637 = {}
    # Getting the type of 'os' (line 478)
    os_286632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'os', False)
    # Obtaining the member 'chdir' of a type (line 478)
    chdir_286633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 12), os_286632, 'chdir')
    # Calling chdir(args, kwargs) (line 478)
    chdir_call_result_286638 = invoke(stypy.reporting.localization.Localization(__file__, 478, 12), chdir_286633, *[plot_working_directory_286636], **kwargs_286637)
    
    # SSA branch for the except part of a try statement (line 477)
    # SSA branch for the except 'OSError' branch of a try statement (line 477)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'OSError' (line 479)
    OSError_286639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 15), 'OSError')
    # Assigning a type to the variable 'err' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'err', OSError_286639)
    
    # Call to OSError(...): (line 480)
    # Processing the call arguments (line 480)
    
    # Call to str(...): (line 480)
    # Processing the call arguments (line 480)
    # Getting the type of 'err' (line 480)
    err_286642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 30), 'err', False)
    # Processing the call keyword arguments (line 480)
    kwargs_286643 = {}
    # Getting the type of 'str' (line 480)
    str_286641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 26), 'str', False)
    # Calling str(args, kwargs) (line 480)
    str_call_result_286644 = invoke(stypy.reporting.localization.Localization(__file__, 480, 26), str_286641, *[err_286642], **kwargs_286643)
    
    unicode_286645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 37), 'unicode', u'\n`plot_working_directory` option inSphinx configuration file must be a valid directory path')
    # Applying the binary operator '+' (line 480)
    result_add_286646 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 26), '+', str_call_result_286644, unicode_286645)
    
    # Processing the call keyword arguments (line 480)
    kwargs_286647 = {}
    # Getting the type of 'OSError' (line 480)
    OSError_286640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 18), 'OSError', False)
    # Calling OSError(args, kwargs) (line 480)
    OSError_call_result_286648 = invoke(stypy.reporting.localization.Localization(__file__, 480, 18), OSError_286640, *[result_add_286646], **kwargs_286647)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 480, 12), OSError_call_result_286648, 'raise parameter', BaseException)
    # SSA branch for the except 'TypeError' branch of a try statement (line 477)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'TypeError' (line 483)
    TypeError_286649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 15), 'TypeError')
    # Assigning a type to the variable 'err' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'err', TypeError_286649)
    
    # Call to TypeError(...): (line 484)
    # Processing the call arguments (line 484)
    
    # Call to str(...): (line 484)
    # Processing the call arguments (line 484)
    # Getting the type of 'err' (line 484)
    err_286652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 32), 'err', False)
    # Processing the call keyword arguments (line 484)
    kwargs_286653 = {}
    # Getting the type of 'str' (line 484)
    str_286651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 28), 'str', False)
    # Calling str(args, kwargs) (line 484)
    str_call_result_286654 = invoke(stypy.reporting.localization.Localization(__file__, 484, 28), str_286651, *[err_286652], **kwargs_286653)
    
    unicode_286655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 39), 'unicode', u'\n`plot_working_directory` option in Sphinx configuration file must be a string or None')
    # Applying the binary operator '+' (line 484)
    result_add_286656 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 28), '+', str_call_result_286654, unicode_286655)
    
    # Processing the call keyword arguments (line 484)
    kwargs_286657 = {}
    # Getting the type of 'TypeError' (line 484)
    TypeError_286650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 18), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 484)
    TypeError_call_result_286658 = invoke(stypy.reporting.localization.Localization(__file__, 484, 18), TypeError_286650, *[result_add_286656], **kwargs_286657)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 484, 12), TypeError_call_result_286658, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 477)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to insert(...): (line 487)
    # Processing the call arguments (line 487)
    int_286662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 24), 'int')
    # Getting the type of 'setup' (line 487)
    setup_286663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 27), 'setup', False)
    # Obtaining the member 'config' of a type (line 487)
    config_286664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 27), setup_286663, 'config')
    # Obtaining the member 'plot_working_directory' of a type (line 487)
    plot_working_directory_286665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 27), config_286664, 'plot_working_directory')
    # Processing the call keyword arguments (line 487)
    kwargs_286666 = {}
    # Getting the type of 'sys' (line 487)
    sys_286659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'sys', False)
    # Obtaining the member 'path' of a type (line 487)
    path_286660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 8), sys_286659, 'path')
    # Obtaining the member 'insert' of a type (line 487)
    insert_286661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 8), path_286660, 'insert')
    # Calling insert(args, kwargs) (line 487)
    insert_call_result_286667 = invoke(stypy.reporting.localization.Localization(__file__, 487, 8), insert_286661, *[int_286662, plot_working_directory_286665], **kwargs_286666)
    
    # SSA branch for the else part of an if statement (line 476)
    module_type_store.open_ssa_branch('else')
    
    # Type idiom detected: calculating its left and rigth part (line 488)
    # Getting the type of 'code_path' (line 488)
    code_path_286668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 9), 'code_path')
    # Getting the type of 'None' (line 488)
    None_286669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 26), 'None')
    
    (may_be_286670, more_types_in_union_286671) = may_not_be_none(code_path_286668, None_286669)

    if may_be_286670:

        if more_types_in_union_286671:
            # Runtime conditional SSA (line 488)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 489):
        
        # Assigning a Call to a Name (line 489):
        
        # Call to abspath(...): (line 489)
        # Processing the call arguments (line 489)
        
        # Call to dirname(...): (line 489)
        # Processing the call arguments (line 489)
        # Getting the type of 'code_path' (line 489)
        code_path_286678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 50), 'code_path', False)
        # Processing the call keyword arguments (line 489)
        kwargs_286679 = {}
        # Getting the type of 'os' (line 489)
        os_286675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 489)
        path_286676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 34), os_286675, 'path')
        # Obtaining the member 'dirname' of a type (line 489)
        dirname_286677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 34), path_286676, 'dirname')
        # Calling dirname(args, kwargs) (line 489)
        dirname_call_result_286680 = invoke(stypy.reporting.localization.Localization(__file__, 489, 34), dirname_286677, *[code_path_286678], **kwargs_286679)
        
        # Processing the call keyword arguments (line 489)
        kwargs_286681 = {}
        # Getting the type of 'os' (line 489)
        os_286672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 489)
        path_286673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 18), os_286672, 'path')
        # Obtaining the member 'abspath' of a type (line 489)
        abspath_286674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 18), path_286673, 'abspath')
        # Calling abspath(args, kwargs) (line 489)
        abspath_call_result_286682 = invoke(stypy.reporting.localization.Localization(__file__, 489, 18), abspath_286674, *[dirname_call_result_286680], **kwargs_286681)
        
        # Assigning a type to the variable 'dirname' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'dirname', abspath_call_result_286682)
        
        # Call to chdir(...): (line 490)
        # Processing the call arguments (line 490)
        # Getting the type of 'dirname' (line 490)
        dirname_286685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 17), 'dirname', False)
        # Processing the call keyword arguments (line 490)
        kwargs_286686 = {}
        # Getting the type of 'os' (line 490)
        os_286683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 490)
        chdir_286684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 8), os_286683, 'chdir')
        # Calling chdir(args, kwargs) (line 490)
        chdir_call_result_286687 = invoke(stypy.reporting.localization.Localization(__file__, 490, 8), chdir_286684, *[dirname_286685], **kwargs_286686)
        
        
        # Call to insert(...): (line 491)
        # Processing the call arguments (line 491)
        int_286691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 24), 'int')
        # Getting the type of 'dirname' (line 491)
        dirname_286692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 27), 'dirname', False)
        # Processing the call keyword arguments (line 491)
        kwargs_286693 = {}
        # Getting the type of 'sys' (line 491)
        sys_286688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'sys', False)
        # Obtaining the member 'path' of a type (line 491)
        path_286689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 8), sys_286688, 'path')
        # Obtaining the member 'insert' of a type (line 491)
        insert_286690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 8), path_286689, 'insert')
        # Calling insert(args, kwargs) (line 491)
        insert_call_result_286694 = invoke(stypy.reporting.localization.Localization(__file__, 491, 8), insert_286690, *[int_286691, dirname_286692], **kwargs_286693)
        

        if more_types_in_union_286671:
            # SSA join for if statement (line 488)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 476)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 494):
    
    # Assigning a Attribute to a Name (line 494):
    # Getting the type of 'sys' (line 494)
    sys_286695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 19), 'sys')
    # Obtaining the member 'argv' of a type (line 494)
    argv_286696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 19), sys_286695, 'argv')
    # Assigning a type to the variable 'old_sys_argv' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 4), 'old_sys_argv', argv_286696)
    
    # Assigning a List to a Attribute (line 495):
    
    # Assigning a List to a Attribute (line 495):
    
    # Obtaining an instance of the builtin type 'list' (line 495)
    list_286697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 495)
    # Adding element type (line 495)
    # Getting the type of 'code_path' (line 495)
    code_path_286698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 16), 'code_path')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 15), list_286697, code_path_286698)
    
    # Getting the type of 'sys' (line 495)
    sys_286699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 4), 'sys')
    # Setting the type of the member 'argv' of a type (line 495)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 4), sys_286699, 'argv', list_286697)
    
    # Assigning a Attribute to a Name (line 498):
    
    # Assigning a Attribute to a Name (line 498):
    # Getting the type of 'sys' (line 498)
    sys_286700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 13), 'sys')
    # Obtaining the member 'stdout' of a type (line 498)
    stdout_286701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 13), sys_286700, 'stdout')
    # Assigning a type to the variable 'stdout' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 4), 'stdout', stdout_286701)
    
    # Getting the type of 'six' (line 499)
    six_286702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 7), 'six')
    # Obtaining the member 'PY3' of a type (line 499)
    PY3_286703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 7), six_286702, 'PY3')
    # Testing the type of an if condition (line 499)
    if_condition_286704 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 499, 4), PY3_286703)
    # Assigning a type to the variable 'if_condition_286704' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 4), 'if_condition_286704', if_condition_286704)
    # SSA begins for if statement (line 499)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Attribute (line 500):
    
    # Assigning a Call to a Attribute (line 500):
    
    # Call to StringIO(...): (line 500)
    # Processing the call keyword arguments (line 500)
    kwargs_286707 = {}
    # Getting the type of 'io' (line 500)
    io_286705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 21), 'io', False)
    # Obtaining the member 'StringIO' of a type (line 500)
    StringIO_286706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 21), io_286705, 'StringIO')
    # Calling StringIO(args, kwargs) (line 500)
    StringIO_call_result_286708 = invoke(stypy.reporting.localization.Localization(__file__, 500, 21), StringIO_286706, *[], **kwargs_286707)
    
    # Getting the type of 'sys' (line 500)
    sys_286709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'sys')
    # Setting the type of the member 'stdout' of a type (line 500)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 8), sys_286709, 'stdout', StringIO_call_result_286708)
    # SSA branch for the else part of an if statement (line 499)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Attribute (line 502):
    
    # Assigning a Call to a Attribute (line 502):
    
    # Call to StringIO(...): (line 502)
    # Processing the call keyword arguments (line 502)
    kwargs_286712 = {}
    # Getting the type of 'cStringIO' (line 502)
    cStringIO_286710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 21), 'cStringIO', False)
    # Obtaining the member 'StringIO' of a type (line 502)
    StringIO_286711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 21), cStringIO_286710, 'StringIO')
    # Calling StringIO(args, kwargs) (line 502)
    StringIO_call_result_286713 = invoke(stypy.reporting.localization.Localization(__file__, 502, 21), StringIO_286711, *[], **kwargs_286712)
    
    # Getting the type of 'sys' (line 502)
    sys_286714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'sys')
    # Setting the type of the member 'stdout' of a type (line 502)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 8), sys_286714, 'stdout', StringIO_call_result_286713)
    # SSA join for if statement (line 499)
    module_type_store = module_type_store.join_ssa_context()
    

    @norecursion
    def _dummy_print(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_dummy_print'
        module_type_store = module_type_store.open_function_context('_dummy_print', 507, 4, False)
        
        # Passed parameters checking function
        _dummy_print.stypy_localization = localization
        _dummy_print.stypy_type_of_self = None
        _dummy_print.stypy_type_store = module_type_store
        _dummy_print.stypy_function_name = '_dummy_print'
        _dummy_print.stypy_param_names_list = []
        _dummy_print.stypy_varargs_param_name = 'arg'
        _dummy_print.stypy_kwargs_param_name = 'kwarg'
        _dummy_print.stypy_call_defaults = defaults
        _dummy_print.stypy_call_varargs = varargs
        _dummy_print.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_dummy_print', [], 'arg', 'kwarg', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_dummy_print', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_dummy_print(...)' code ##################

        pass
        
        # ################# End of '_dummy_print(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_dummy_print' in the type store
        # Getting the type of 'stypy_return_type' (line 507)
        stypy_return_type_286715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_286715)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_dummy_print'
        return stypy_return_type_286715

    # Assigning a type to the variable '_dummy_print' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), '_dummy_print', _dummy_print)
    
    # Try-finally block (line 510)
    
    
    # SSA begins for try-except statement (line 511)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 512):
    
    # Assigning a Call to a Name (line 512):
    
    # Call to unescape_doctest(...): (line 512)
    # Processing the call arguments (line 512)
    # Getting the type of 'code' (line 512)
    code_286717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 36), 'code', False)
    # Processing the call keyword arguments (line 512)
    kwargs_286718 = {}
    # Getting the type of 'unescape_doctest' (line 512)
    unescape_doctest_286716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 19), 'unescape_doctest', False)
    # Calling unescape_doctest(args, kwargs) (line 512)
    unescape_doctest_call_result_286719 = invoke(stypy.reporting.localization.Localization(__file__, 512, 19), unescape_doctest_286716, *[code_286717], **kwargs_286718)
    
    # Assigning a type to the variable 'code' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'code', unescape_doctest_call_result_286719)
    
    # Type idiom detected: calculating its left and rigth part (line 513)
    # Getting the type of 'ns' (line 513)
    ns_286720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 15), 'ns')
    # Getting the type of 'None' (line 513)
    None_286721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 21), 'None')
    
    (may_be_286722, more_types_in_union_286723) = may_be_none(ns_286720, None_286721)

    if may_be_286722:

        if more_types_in_union_286723:
            # Runtime conditional SSA (line 513)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Dict to a Name (line 514):
        
        # Assigning a Dict to a Name (line 514):
        
        # Obtaining an instance of the builtin type 'dict' (line 514)
        dict_286724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 21), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 514)
        
        # Assigning a type to the variable 'ns' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 16), 'ns', dict_286724)

        if more_types_in_union_286723:
            # SSA join for if statement (line 513)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'ns' (line 515)
    ns_286725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 19), 'ns')
    # Applying the 'not' unary operator (line 515)
    result_not__286726 = python_operator(stypy.reporting.localization.Localization(__file__, 515, 15), 'not', ns_286725)
    
    # Testing the type of an if condition (line 515)
    if_condition_286727 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 515, 12), result_not__286726)
    # Assigning a type to the variable 'if_condition_286727' (line 515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 12), 'if_condition_286727', if_condition_286727)
    # SSA begins for if statement (line 515)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 516)
    # Getting the type of 'setup' (line 516)
    setup_286728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 19), 'setup')
    # Obtaining the member 'config' of a type (line 516)
    config_286729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 19), setup_286728, 'config')
    # Obtaining the member 'plot_pre_code' of a type (line 516)
    plot_pre_code_286730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 19), config_286729, 'plot_pre_code')
    # Getting the type of 'None' (line 516)
    None_286731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 49), 'None')
    
    (may_be_286732, more_types_in_union_286733) = may_be_none(plot_pre_code_286730, None_286731)

    if may_be_286732:

        if more_types_in_union_286733:
            # Runtime conditional SSA (line 516)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to exec_(...): (line 517)
        # Processing the call arguments (line 517)
        
        # Call to text_type(...): (line 517)
        # Processing the call arguments (line 517)
        unicode_286738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 44), 'unicode', u'import numpy as np\n')
        unicode_286739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 20), 'unicode', u'from matplotlib import pyplot as plt\n')
        # Applying the binary operator '+' (line 517)
        result_add_286740 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 44), '+', unicode_286738, unicode_286739)
        
        # Processing the call keyword arguments (line 517)
        kwargs_286741 = {}
        # Getting the type of 'six' (line 517)
        six_286736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 30), 'six', False)
        # Obtaining the member 'text_type' of a type (line 517)
        text_type_286737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 30), six_286736, 'text_type')
        # Calling text_type(args, kwargs) (line 517)
        text_type_call_result_286742 = invoke(stypy.reporting.localization.Localization(__file__, 517, 30), text_type_286737, *[result_add_286740], **kwargs_286741)
        
        # Getting the type of 'ns' (line 518)
        ns_286743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 63), 'ns', False)
        # Processing the call keyword arguments (line 517)
        kwargs_286744 = {}
        # Getting the type of 'six' (line 517)
        six_286734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 20), 'six', False)
        # Obtaining the member 'exec_' of a type (line 517)
        exec__286735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 20), six_286734, 'exec_')
        # Calling exec_(args, kwargs) (line 517)
        exec__call_result_286745 = invoke(stypy.reporting.localization.Localization(__file__, 517, 20), exec__286735, *[text_type_call_result_286742, ns_286743], **kwargs_286744)
        

        if more_types_in_union_286733:
            # Runtime conditional SSA for else branch (line 516)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_286732) or more_types_in_union_286733):
        
        # Call to exec_(...): (line 520)
        # Processing the call arguments (line 520)
        
        # Call to text_type(...): (line 520)
        # Processing the call arguments (line 520)
        # Getting the type of 'setup' (line 520)
        setup_286750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 44), 'setup', False)
        # Obtaining the member 'config' of a type (line 520)
        config_286751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 44), setup_286750, 'config')
        # Obtaining the member 'plot_pre_code' of a type (line 520)
        plot_pre_code_286752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 44), config_286751, 'plot_pre_code')
        # Processing the call keyword arguments (line 520)
        kwargs_286753 = {}
        # Getting the type of 'six' (line 520)
        six_286748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 30), 'six', False)
        # Obtaining the member 'text_type' of a type (line 520)
        text_type_286749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 30), six_286748, 'text_type')
        # Calling text_type(args, kwargs) (line 520)
        text_type_call_result_286754 = invoke(stypy.reporting.localization.Localization(__file__, 520, 30), text_type_286749, *[plot_pre_code_286752], **kwargs_286753)
        
        # Getting the type of 'ns' (line 520)
        ns_286755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 73), 'ns', False)
        # Processing the call keyword arguments (line 520)
        kwargs_286756 = {}
        # Getting the type of 'six' (line 520)
        six_286746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 20), 'six', False)
        # Obtaining the member 'exec_' of a type (line 520)
        exec__286747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 20), six_286746, 'exec_')
        # Calling exec_(args, kwargs) (line 520)
        exec__call_result_286757 = invoke(stypy.reporting.localization.Localization(__file__, 520, 20), exec__286747, *[text_type_call_result_286754, ns_286755], **kwargs_286756)
        

        if (may_be_286732 and more_types_in_union_286733):
            # SSA join for if statement (line 516)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 515)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 521):
    
    # Assigning a Name to a Subscript (line 521):
    # Getting the type of '_dummy_print' (line 521)
    _dummy_print_286758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 26), '_dummy_print')
    # Getting the type of 'ns' (line 521)
    ns_286759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'ns')
    unicode_286760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 15), 'unicode', u'print')
    # Storing an element on a container (line 521)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 12), ns_286759, (unicode_286760, _dummy_print_286758))
    
    
    unicode_286761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 15), 'unicode', u'__main__')
    # Getting the type of 'code' (line 522)
    code_286762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 29), 'code')
    # Applying the binary operator 'in' (line 522)
    result_contains_286763 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 15), 'in', unicode_286761, code_286762)
    
    # Testing the type of an if condition (line 522)
    if_condition_286764 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 522, 12), result_contains_286763)
    # Assigning a type to the variable 'if_condition_286764' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'if_condition_286764', if_condition_286764)
    # SSA begins for if statement (line 522)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to exec_(...): (line 523)
    # Processing the call arguments (line 523)
    unicode_286767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 26), 'unicode', u"__name__ = '__main__'")
    # Getting the type of 'ns' (line 523)
    ns_286768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 51), 'ns', False)
    # Processing the call keyword arguments (line 523)
    kwargs_286769 = {}
    # Getting the type of 'six' (line 523)
    six_286765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 16), 'six', False)
    # Obtaining the member 'exec_' of a type (line 523)
    exec__286766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 16), six_286765, 'exec_')
    # Calling exec_(args, kwargs) (line 523)
    exec__call_result_286770 = invoke(stypy.reporting.localization.Localization(__file__, 523, 16), exec__286766, *[unicode_286767, ns_286768], **kwargs_286769)
    
    # SSA join for if statement (line 522)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 524):
    
    # Assigning a Call to a Name (line 524):
    
    # Call to remove_coding(...): (line 524)
    # Processing the call arguments (line 524)
    # Getting the type of 'code' (line 524)
    code_286772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 33), 'code', False)
    # Processing the call keyword arguments (line 524)
    kwargs_286773 = {}
    # Getting the type of 'remove_coding' (line 524)
    remove_coding_286771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 19), 'remove_coding', False)
    # Calling remove_coding(args, kwargs) (line 524)
    remove_coding_call_result_286774 = invoke(stypy.reporting.localization.Localization(__file__, 524, 19), remove_coding_286771, *[code_286772], **kwargs_286773)
    
    # Assigning a type to the variable 'code' (line 524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 12), 'code', remove_coding_call_result_286774)
    
    # Call to exec_(...): (line 525)
    # Processing the call arguments (line 525)
    # Getting the type of 'code' (line 525)
    code_286777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 22), 'code', False)
    # Getting the type of 'ns' (line 525)
    ns_286778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 28), 'ns', False)
    # Processing the call keyword arguments (line 525)
    kwargs_286779 = {}
    # Getting the type of 'six' (line 525)
    six_286775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 12), 'six', False)
    # Obtaining the member 'exec_' of a type (line 525)
    exec__286776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 12), six_286775, 'exec_')
    # Calling exec_(args, kwargs) (line 525)
    exec__call_result_286780 = invoke(stypy.reporting.localization.Localization(__file__, 525, 12), exec__286776, *[code_286777, ns_286778], **kwargs_286779)
    
    
    # Type idiom detected: calculating its left and rigth part (line 526)
    # Getting the type of 'function_name' (line 526)
    function_name_286781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 12), 'function_name')
    # Getting the type of 'None' (line 526)
    None_286782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 36), 'None')
    
    (may_be_286783, more_types_in_union_286784) = may_not_be_none(function_name_286781, None_286782)

    if may_be_286783:

        if more_types_in_union_286784:
            # Runtime conditional SSA (line 526)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to exec_(...): (line 527)
        # Processing the call arguments (line 527)
        # Getting the type of 'function_name' (line 527)
        function_name_286787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 26), 'function_name', False)
        unicode_286788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 42), 'unicode', u'()')
        # Applying the binary operator '+' (line 527)
        result_add_286789 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 26), '+', function_name_286787, unicode_286788)
        
        # Getting the type of 'ns' (line 527)
        ns_286790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 48), 'ns', False)
        # Processing the call keyword arguments (line 527)
        kwargs_286791 = {}
        # Getting the type of 'six' (line 527)
        six_286785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 16), 'six', False)
        # Obtaining the member 'exec_' of a type (line 527)
        exec__286786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 16), six_286785, 'exec_')
        # Calling exec_(args, kwargs) (line 527)
        exec__call_result_286792 = invoke(stypy.reporting.localization.Localization(__file__, 527, 16), exec__286786, *[result_add_286789, ns_286790], **kwargs_286791)
        

        if more_types_in_union_286784:
            # SSA join for if statement (line 526)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA branch for the except part of a try statement (line 511)
    # SSA branch for the except 'Tuple' branch of a try statement (line 511)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    
    # Obtaining an instance of the builtin type 'tuple' (line 528)
    tuple_286793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 528)
    # Adding element type (line 528)
    # Getting the type of 'Exception' (line 528)
    Exception_286794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 16), 'Exception')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 16), tuple_286793, Exception_286794)
    # Adding element type (line 528)
    # Getting the type of 'SystemExit' (line 528)
    SystemExit_286795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 27), 'SystemExit')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 16), tuple_286793, SystemExit_286795)
    
    # Assigning a type to the variable 'err' (line 528)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'err', tuple_286793)
    
    # Call to PlotError(...): (line 529)
    # Processing the call arguments (line 529)
    
    # Call to format_exc(...): (line 529)
    # Processing the call keyword arguments (line 529)
    kwargs_286799 = {}
    # Getting the type of 'traceback' (line 529)
    traceback_286797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 28), 'traceback', False)
    # Obtaining the member 'format_exc' of a type (line 529)
    format_exc_286798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 28), traceback_286797, 'format_exc')
    # Calling format_exc(args, kwargs) (line 529)
    format_exc_call_result_286800 = invoke(stypy.reporting.localization.Localization(__file__, 529, 28), format_exc_286798, *[], **kwargs_286799)
    
    # Processing the call keyword arguments (line 529)
    kwargs_286801 = {}
    # Getting the type of 'PlotError' (line 529)
    PlotError_286796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 18), 'PlotError', False)
    # Calling PlotError(args, kwargs) (line 529)
    PlotError_call_result_286802 = invoke(stypy.reporting.localization.Localization(__file__, 529, 18), PlotError_286796, *[format_exc_call_result_286800], **kwargs_286801)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 529, 12), PlotError_call_result_286802, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 511)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # finally branch of the try-finally block (line 510)
    
    # Call to chdir(...): (line 531)
    # Processing the call arguments (line 531)
    # Getting the type of 'pwd' (line 531)
    pwd_286805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 17), 'pwd', False)
    # Processing the call keyword arguments (line 531)
    kwargs_286806 = {}
    # Getting the type of 'os' (line 531)
    os_286803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 8), 'os', False)
    # Obtaining the member 'chdir' of a type (line 531)
    chdir_286804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 8), os_286803, 'chdir')
    # Calling chdir(args, kwargs) (line 531)
    chdir_call_result_286807 = invoke(stypy.reporting.localization.Localization(__file__, 531, 8), chdir_286804, *[pwd_286805], **kwargs_286806)
    
    
    # Assigning a Name to a Attribute (line 532):
    
    # Assigning a Name to a Attribute (line 532):
    # Getting the type of 'old_sys_argv' (line 532)
    old_sys_argv_286808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 19), 'old_sys_argv')
    # Getting the type of 'sys' (line 532)
    sys_286809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'sys')
    # Setting the type of the member 'argv' of a type (line 532)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 8), sys_286809, 'argv', old_sys_argv_286808)
    
    # Assigning a Name to a Subscript (line 533):
    
    # Assigning a Name to a Subscript (line 533):
    # Getting the type of 'old_sys_path' (line 533)
    old_sys_path_286810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 22), 'old_sys_path')
    # Getting the type of 'sys' (line 533)
    sys_286811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'sys')
    # Obtaining the member 'path' of a type (line 533)
    path_286812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 8), sys_286811, 'path')
    slice_286813 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 533, 8), None, None, None)
    # Storing an element on a container (line 533)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 8), path_286812, (slice_286813, old_sys_path_286810))
    
    # Assigning a Name to a Attribute (line 534):
    
    # Assigning a Name to a Attribute (line 534):
    # Getting the type of 'stdout' (line 534)
    stdout_286814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 21), 'stdout')
    # Getting the type of 'sys' (line 534)
    sys_286815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 8), 'sys')
    # Setting the type of the member 'stdout' of a type (line 534)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 8), sys_286815, 'stdout', stdout_286814)
    
    # Getting the type of 'ns' (line 535)
    ns_286816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 11), 'ns')
    # Assigning a type to the variable 'stypy_return_type' (line 535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'stypy_return_type', ns_286816)
    
    # ################# End of 'run_code(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run_code' in the type store
    # Getting the type of 'stypy_return_type' (line 462)
    stypy_return_type_286817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286817)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run_code'
    return stypy_return_type_286817

# Assigning a type to the variable 'run_code' (line 462)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 0), 'run_code', run_code)

@norecursion
def clear_state(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 538)
    True_286818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 37), 'True')
    defaults = [True_286818]
    # Create a new context for function 'clear_state'
    module_type_store = module_type_store.open_function_context('clear_state', 538, 0, False)
    
    # Passed parameters checking function
    clear_state.stypy_localization = localization
    clear_state.stypy_type_of_self = None
    clear_state.stypy_type_store = module_type_store
    clear_state.stypy_function_name = 'clear_state'
    clear_state.stypy_param_names_list = ['plot_rcparams', 'close']
    clear_state.stypy_varargs_param_name = None
    clear_state.stypy_kwargs_param_name = None
    clear_state.stypy_call_defaults = defaults
    clear_state.stypy_call_varargs = varargs
    clear_state.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'clear_state', ['plot_rcparams', 'close'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'clear_state', localization, ['plot_rcparams', 'close'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'clear_state(...)' code ##################

    
    # Getting the type of 'close' (line 539)
    close_286819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 7), 'close')
    # Testing the type of an if condition (line 539)
    if_condition_286820 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 539, 4), close_286819)
    # Assigning a type to the variable 'if_condition_286820' (line 539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'if_condition_286820', if_condition_286820)
    # SSA begins for if statement (line 539)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to close(...): (line 540)
    # Processing the call arguments (line 540)
    unicode_286823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 18), 'unicode', u'all')
    # Processing the call keyword arguments (line 540)
    kwargs_286824 = {}
    # Getting the type of 'plt' (line 540)
    plt_286821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'plt', False)
    # Obtaining the member 'close' of a type (line 540)
    close_286822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 8), plt_286821, 'close')
    # Calling close(args, kwargs) (line 540)
    close_call_result_286825 = invoke(stypy.reporting.localization.Localization(__file__, 540, 8), close_286822, *[unicode_286823], **kwargs_286824)
    
    # SSA join for if statement (line 539)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to rc_file_defaults(...): (line 541)
    # Processing the call keyword arguments (line 541)
    kwargs_286828 = {}
    # Getting the type of 'matplotlib' (line 541)
    matplotlib_286826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'matplotlib', False)
    # Obtaining the member 'rc_file_defaults' of a type (line 541)
    rc_file_defaults_286827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 4), matplotlib_286826, 'rc_file_defaults')
    # Calling rc_file_defaults(args, kwargs) (line 541)
    rc_file_defaults_call_result_286829 = invoke(stypy.reporting.localization.Localization(__file__, 541, 4), rc_file_defaults_286827, *[], **kwargs_286828)
    
    
    # Call to update(...): (line 542)
    # Processing the call arguments (line 542)
    # Getting the type of 'plot_rcparams' (line 542)
    plot_rcparams_286833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 31), 'plot_rcparams', False)
    # Processing the call keyword arguments (line 542)
    kwargs_286834 = {}
    # Getting the type of 'matplotlib' (line 542)
    matplotlib_286830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 4), 'matplotlib', False)
    # Obtaining the member 'rcParams' of a type (line 542)
    rcParams_286831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 4), matplotlib_286830, 'rcParams')
    # Obtaining the member 'update' of a type (line 542)
    update_286832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 4), rcParams_286831, 'update')
    # Calling update(args, kwargs) (line 542)
    update_call_result_286835 = invoke(stypy.reporting.localization.Localization(__file__, 542, 4), update_286832, *[plot_rcparams_286833], **kwargs_286834)
    
    
    # ################# End of 'clear_state(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'clear_state' in the type store
    # Getting the type of 'stypy_return_type' (line 538)
    stypy_return_type_286836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286836)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'clear_state'
    return stypy_return_type_286836

# Assigning a type to the variable 'clear_state' (line 538)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 0), 'clear_state', clear_state)

@norecursion
def get_plot_formats(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_plot_formats'
    module_type_store = module_type_store.open_function_context('get_plot_formats', 545, 0, False)
    
    # Passed parameters checking function
    get_plot_formats.stypy_localization = localization
    get_plot_formats.stypy_type_of_self = None
    get_plot_formats.stypy_type_store = module_type_store
    get_plot_formats.stypy_function_name = 'get_plot_formats'
    get_plot_formats.stypy_param_names_list = ['config']
    get_plot_formats.stypy_varargs_param_name = None
    get_plot_formats.stypy_kwargs_param_name = None
    get_plot_formats.stypy_call_defaults = defaults
    get_plot_formats.stypy_call_varargs = varargs
    get_plot_formats.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_plot_formats', ['config'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_plot_formats', localization, ['config'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_plot_formats(...)' code ##################

    
    # Assigning a Dict to a Name (line 546):
    
    # Assigning a Dict to a Name (line 546):
    
    # Obtaining an instance of the builtin type 'dict' (line 546)
    dict_286837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 18), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 546)
    # Adding element type (key, value) (line 546)
    unicode_286838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 19), 'unicode', u'png')
    int_286839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 26), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 18), dict_286837, (unicode_286838, int_286839))
    # Adding element type (key, value) (line 546)
    unicode_286840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 30), 'unicode', u'hires.png')
    int_286841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 43), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 18), dict_286837, (unicode_286840, int_286841))
    # Adding element type (key, value) (line 546)
    unicode_286842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 48), 'unicode', u'pdf')
    int_286843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 55), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 18), dict_286837, (unicode_286842, int_286843))
    
    # Assigning a type to the variable 'default_dpi' (line 546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'default_dpi', dict_286837)
    
    # Assigning a List to a Name (line 547):
    
    # Assigning a List to a Name (line 547):
    
    # Obtaining an instance of the builtin type 'list' (line 547)
    list_286844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 547)
    
    # Assigning a type to the variable 'formats' (line 547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 4), 'formats', list_286844)
    
    # Assigning a Attribute to a Name (line 548):
    
    # Assigning a Attribute to a Name (line 548):
    # Getting the type of 'config' (line 548)
    config_286845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 19), 'config')
    # Obtaining the member 'plot_formats' of a type (line 548)
    plot_formats_286846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 19), config_286845, 'plot_formats')
    # Assigning a type to the variable 'plot_formats' (line 548)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 4), 'plot_formats', plot_formats_286846)
    
    
    # Call to isinstance(...): (line 549)
    # Processing the call arguments (line 549)
    # Getting the type of 'plot_formats' (line 549)
    plot_formats_286848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 18), 'plot_formats', False)
    # Getting the type of 'six' (line 549)
    six_286849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 32), 'six', False)
    # Obtaining the member 'string_types' of a type (line 549)
    string_types_286850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 32), six_286849, 'string_types')
    # Processing the call keyword arguments (line 549)
    kwargs_286851 = {}
    # Getting the type of 'isinstance' (line 549)
    isinstance_286847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 549)
    isinstance_call_result_286852 = invoke(stypy.reporting.localization.Localization(__file__, 549, 7), isinstance_286847, *[plot_formats_286848, string_types_286850], **kwargs_286851)
    
    # Testing the type of an if condition (line 549)
    if_condition_286853 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 549, 4), isinstance_call_result_286852)
    # Assigning a type to the variable 'if_condition_286853' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'if_condition_286853', if_condition_286853)
    # SSA begins for if statement (line 549)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 553):
    
    # Assigning a Call to a Name (line 553):
    
    # Call to split(...): (line 553)
    # Processing the call arguments (line 553)
    unicode_286856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 42), 'unicode', u',')
    # Processing the call keyword arguments (line 553)
    kwargs_286857 = {}
    # Getting the type of 'plot_formats' (line 553)
    plot_formats_286854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 23), 'plot_formats', False)
    # Obtaining the member 'split' of a type (line 553)
    split_286855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 23), plot_formats_286854, 'split')
    # Calling split(args, kwargs) (line 553)
    split_call_result_286858 = invoke(stypy.reporting.localization.Localization(__file__, 553, 23), split_286855, *[unicode_286856], **kwargs_286857)
    
    # Assigning a type to the variable 'plot_formats' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'plot_formats', split_call_result_286858)
    # SSA join for if statement (line 549)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'plot_formats' (line 554)
    plot_formats_286859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 15), 'plot_formats')
    # Testing the type of a for loop iterable (line 554)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 554, 4), plot_formats_286859)
    # Getting the type of the for loop variable (line 554)
    for_loop_var_286860 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 554, 4), plot_formats_286859)
    # Assigning a type to the variable 'fmt' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'fmt', for_loop_var_286860)
    # SSA begins for a for statement (line 554)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to isinstance(...): (line 555)
    # Processing the call arguments (line 555)
    # Getting the type of 'fmt' (line 555)
    fmt_286862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 22), 'fmt', False)
    # Getting the type of 'six' (line 555)
    six_286863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 27), 'six', False)
    # Obtaining the member 'string_types' of a type (line 555)
    string_types_286864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 27), six_286863, 'string_types')
    # Processing the call keyword arguments (line 555)
    kwargs_286865 = {}
    # Getting the type of 'isinstance' (line 555)
    isinstance_286861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 555)
    isinstance_call_result_286866 = invoke(stypy.reporting.localization.Localization(__file__, 555, 11), isinstance_286861, *[fmt_286862, string_types_286864], **kwargs_286865)
    
    # Testing the type of an if condition (line 555)
    if_condition_286867 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 555, 8), isinstance_call_result_286866)
    # Assigning a type to the variable 'if_condition_286867' (line 555)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'if_condition_286867', if_condition_286867)
    # SSA begins for if statement (line 555)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    unicode_286868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 15), 'unicode', u':')
    # Getting the type of 'fmt' (line 556)
    fmt_286869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 22), 'fmt')
    # Applying the binary operator 'in' (line 556)
    result_contains_286870 = python_operator(stypy.reporting.localization.Localization(__file__, 556, 15), 'in', unicode_286868, fmt_286869)
    
    # Testing the type of an if condition (line 556)
    if_condition_286871 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 556, 12), result_contains_286870)
    # Assigning a type to the variable 'if_condition_286871' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 12), 'if_condition_286871', if_condition_286871)
    # SSA begins for if statement (line 556)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 557):
    
    # Assigning a Call to a Name:
    
    # Call to split(...): (line 557)
    # Processing the call arguments (line 557)
    unicode_286874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 40), 'unicode', u':')
    # Processing the call keyword arguments (line 557)
    kwargs_286875 = {}
    # Getting the type of 'fmt' (line 557)
    fmt_286872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 30), 'fmt', False)
    # Obtaining the member 'split' of a type (line 557)
    split_286873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 30), fmt_286872, 'split')
    # Calling split(args, kwargs) (line 557)
    split_call_result_286876 = invoke(stypy.reporting.localization.Localization(__file__, 557, 30), split_286873, *[unicode_286874], **kwargs_286875)
    
    # Assigning a type to the variable 'call_assignment_285935' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 16), 'call_assignment_285935', split_call_result_286876)
    
    # Assigning a Call to a Name (line 557):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_286879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 16), 'int')
    # Processing the call keyword arguments
    kwargs_286880 = {}
    # Getting the type of 'call_assignment_285935' (line 557)
    call_assignment_285935_286877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 16), 'call_assignment_285935', False)
    # Obtaining the member '__getitem__' of a type (line 557)
    getitem___286878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 16), call_assignment_285935_286877, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_286881 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___286878, *[int_286879], **kwargs_286880)
    
    # Assigning a type to the variable 'call_assignment_285936' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 16), 'call_assignment_285936', getitem___call_result_286881)
    
    # Assigning a Name to a Name (line 557):
    # Getting the type of 'call_assignment_285936' (line 557)
    call_assignment_285936_286882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 16), 'call_assignment_285936')
    # Assigning a type to the variable 'suffix' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 16), 'suffix', call_assignment_285936_286882)
    
    # Assigning a Call to a Name (line 557):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_286885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 16), 'int')
    # Processing the call keyword arguments
    kwargs_286886 = {}
    # Getting the type of 'call_assignment_285935' (line 557)
    call_assignment_285935_286883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 16), 'call_assignment_285935', False)
    # Obtaining the member '__getitem__' of a type (line 557)
    getitem___286884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 16), call_assignment_285935_286883, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_286887 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___286884, *[int_286885], **kwargs_286886)
    
    # Assigning a type to the variable 'call_assignment_285937' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 16), 'call_assignment_285937', getitem___call_result_286887)
    
    # Assigning a Name to a Name (line 557):
    # Getting the type of 'call_assignment_285937' (line 557)
    call_assignment_285937_286888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 16), 'call_assignment_285937')
    # Assigning a type to the variable 'dpi' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 24), 'dpi', call_assignment_285937_286888)
    
    # Call to append(...): (line 558)
    # Processing the call arguments (line 558)
    
    # Obtaining an instance of the builtin type 'tuple' (line 558)
    tuple_286891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 558)
    # Adding element type (line 558)
    
    # Call to str(...): (line 558)
    # Processing the call arguments (line 558)
    # Getting the type of 'suffix' (line 558)
    suffix_286893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 36), 'suffix', False)
    # Processing the call keyword arguments (line 558)
    kwargs_286894 = {}
    # Getting the type of 'str' (line 558)
    str_286892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 32), 'str', False)
    # Calling str(args, kwargs) (line 558)
    str_call_result_286895 = invoke(stypy.reporting.localization.Localization(__file__, 558, 32), str_286892, *[suffix_286893], **kwargs_286894)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 32), tuple_286891, str_call_result_286895)
    # Adding element type (line 558)
    
    # Call to int(...): (line 558)
    # Processing the call arguments (line 558)
    # Getting the type of 'dpi' (line 558)
    dpi_286897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 49), 'dpi', False)
    # Processing the call keyword arguments (line 558)
    kwargs_286898 = {}
    # Getting the type of 'int' (line 558)
    int_286896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 45), 'int', False)
    # Calling int(args, kwargs) (line 558)
    int_call_result_286899 = invoke(stypy.reporting.localization.Localization(__file__, 558, 45), int_286896, *[dpi_286897], **kwargs_286898)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 32), tuple_286891, int_call_result_286899)
    
    # Processing the call keyword arguments (line 558)
    kwargs_286900 = {}
    # Getting the type of 'formats' (line 558)
    formats_286889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 16), 'formats', False)
    # Obtaining the member 'append' of a type (line 558)
    append_286890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 16), formats_286889, 'append')
    # Calling append(args, kwargs) (line 558)
    append_call_result_286901 = invoke(stypy.reporting.localization.Localization(__file__, 558, 16), append_286890, *[tuple_286891], **kwargs_286900)
    
    # SSA branch for the else part of an if statement (line 556)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 560)
    # Processing the call arguments (line 560)
    
    # Obtaining an instance of the builtin type 'tuple' (line 560)
    tuple_286904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 560)
    # Adding element type (line 560)
    # Getting the type of 'fmt' (line 560)
    fmt_286905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 32), 'fmt', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 32), tuple_286904, fmt_286905)
    # Adding element type (line 560)
    
    # Call to get(...): (line 560)
    # Processing the call arguments (line 560)
    # Getting the type of 'fmt' (line 560)
    fmt_286908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 53), 'fmt', False)
    int_286909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 58), 'int')
    # Processing the call keyword arguments (line 560)
    kwargs_286910 = {}
    # Getting the type of 'default_dpi' (line 560)
    default_dpi_286906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 37), 'default_dpi', False)
    # Obtaining the member 'get' of a type (line 560)
    get_286907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 37), default_dpi_286906, 'get')
    # Calling get(args, kwargs) (line 560)
    get_call_result_286911 = invoke(stypy.reporting.localization.Localization(__file__, 560, 37), get_286907, *[fmt_286908, int_286909], **kwargs_286910)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 32), tuple_286904, get_call_result_286911)
    
    # Processing the call keyword arguments (line 560)
    kwargs_286912 = {}
    # Getting the type of 'formats' (line 560)
    formats_286902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 16), 'formats', False)
    # Obtaining the member 'append' of a type (line 560)
    append_286903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 16), formats_286902, 'append')
    # Calling append(args, kwargs) (line 560)
    append_call_result_286913 = invoke(stypy.reporting.localization.Localization(__file__, 560, 16), append_286903, *[tuple_286904], **kwargs_286912)
    
    # SSA join for if statement (line 556)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 555)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    
    # Call to type(...): (line 561)
    # Processing the call arguments (line 561)
    # Getting the type of 'fmt' (line 561)
    fmt_286915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 18), 'fmt', False)
    # Processing the call keyword arguments (line 561)
    kwargs_286916 = {}
    # Getting the type of 'type' (line 561)
    type_286914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 13), 'type', False)
    # Calling type(args, kwargs) (line 561)
    type_call_result_286917 = invoke(stypy.reporting.localization.Localization(__file__, 561, 13), type_286914, *[fmt_286915], **kwargs_286916)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 561)
    tuple_286918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 561)
    # Adding element type (line 561)
    # Getting the type of 'tuple' (line 561)
    tuple_286919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 27), 'tuple')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 27), tuple_286918, tuple_286919)
    # Adding element type (line 561)
    # Getting the type of 'list' (line 561)
    list_286920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 34), 'list')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 27), tuple_286918, list_286920)
    
    # Applying the binary operator 'in' (line 561)
    result_contains_286921 = python_operator(stypy.reporting.localization.Localization(__file__, 561, 13), 'in', type_call_result_286917, tuple_286918)
    
    
    
    # Call to len(...): (line 561)
    # Processing the call arguments (line 561)
    # Getting the type of 'fmt' (line 561)
    fmt_286923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 48), 'fmt', False)
    # Processing the call keyword arguments (line 561)
    kwargs_286924 = {}
    # Getting the type of 'len' (line 561)
    len_286922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 44), 'len', False)
    # Calling len(args, kwargs) (line 561)
    len_call_result_286925 = invoke(stypy.reporting.localization.Localization(__file__, 561, 44), len_286922, *[fmt_286923], **kwargs_286924)
    
    int_286926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 56), 'int')
    # Applying the binary operator '==' (line 561)
    result_eq_286927 = python_operator(stypy.reporting.localization.Localization(__file__, 561, 44), '==', len_call_result_286925, int_286926)
    
    # Applying the binary operator 'and' (line 561)
    result_and_keyword_286928 = python_operator(stypy.reporting.localization.Localization(__file__, 561, 13), 'and', result_contains_286921, result_eq_286927)
    
    # Testing the type of an if condition (line 561)
    if_condition_286929 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 561, 13), result_and_keyword_286928)
    # Assigning a type to the variable 'if_condition_286929' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 13), 'if_condition_286929', if_condition_286929)
    # SSA begins for if statement (line 561)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 562)
    # Processing the call arguments (line 562)
    
    # Obtaining an instance of the builtin type 'tuple' (line 562)
    tuple_286932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 562)
    # Adding element type (line 562)
    
    # Call to str(...): (line 562)
    # Processing the call arguments (line 562)
    
    # Obtaining the type of the subscript
    int_286934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 36), 'int')
    # Getting the type of 'fmt' (line 562)
    fmt_286935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 32), 'fmt', False)
    # Obtaining the member '__getitem__' of a type (line 562)
    getitem___286936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 32), fmt_286935, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 562)
    subscript_call_result_286937 = invoke(stypy.reporting.localization.Localization(__file__, 562, 32), getitem___286936, int_286934)
    
    # Processing the call keyword arguments (line 562)
    kwargs_286938 = {}
    # Getting the type of 'str' (line 562)
    str_286933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 28), 'str', False)
    # Calling str(args, kwargs) (line 562)
    str_call_result_286939 = invoke(stypy.reporting.localization.Localization(__file__, 562, 28), str_286933, *[subscript_call_result_286937], **kwargs_286938)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 562, 28), tuple_286932, str_call_result_286939)
    # Adding element type (line 562)
    
    # Call to int(...): (line 562)
    # Processing the call arguments (line 562)
    
    # Obtaining the type of the subscript
    int_286941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 49), 'int')
    # Getting the type of 'fmt' (line 562)
    fmt_286942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 45), 'fmt', False)
    # Obtaining the member '__getitem__' of a type (line 562)
    getitem___286943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 45), fmt_286942, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 562)
    subscript_call_result_286944 = invoke(stypy.reporting.localization.Localization(__file__, 562, 45), getitem___286943, int_286941)
    
    # Processing the call keyword arguments (line 562)
    kwargs_286945 = {}
    # Getting the type of 'int' (line 562)
    int_286940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 41), 'int', False)
    # Calling int(args, kwargs) (line 562)
    int_call_result_286946 = invoke(stypy.reporting.localization.Localization(__file__, 562, 41), int_286940, *[subscript_call_result_286944], **kwargs_286945)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 562, 28), tuple_286932, int_call_result_286946)
    
    # Processing the call keyword arguments (line 562)
    kwargs_286947 = {}
    # Getting the type of 'formats' (line 562)
    formats_286930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 12), 'formats', False)
    # Obtaining the member 'append' of a type (line 562)
    append_286931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 12), formats_286930, 'append')
    # Calling append(args, kwargs) (line 562)
    append_call_result_286948 = invoke(stypy.reporting.localization.Localization(__file__, 562, 12), append_286931, *[tuple_286932], **kwargs_286947)
    
    # SSA branch for the else part of an if statement (line 561)
    module_type_store.open_ssa_branch('else')
    
    # Call to PlotError(...): (line 564)
    # Processing the call arguments (line 564)
    unicode_286950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 28), 'unicode', u'invalid image format "%r" in plot_formats')
    # Getting the type of 'fmt' (line 564)
    fmt_286951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 74), 'fmt', False)
    # Applying the binary operator '%' (line 564)
    result_mod_286952 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 28), '%', unicode_286950, fmt_286951)
    
    # Processing the call keyword arguments (line 564)
    kwargs_286953 = {}
    # Getting the type of 'PlotError' (line 564)
    PlotError_286949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 18), 'PlotError', False)
    # Calling PlotError(args, kwargs) (line 564)
    PlotError_call_result_286954 = invoke(stypy.reporting.localization.Localization(__file__, 564, 18), PlotError_286949, *[result_mod_286952], **kwargs_286953)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 564, 12), PlotError_call_result_286954, 'raise parameter', BaseException)
    # SSA join for if statement (line 561)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 555)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'formats' (line 565)
    formats_286955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 11), 'formats')
    # Assigning a type to the variable 'stypy_return_type' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'stypy_return_type', formats_286955)
    
    # ################# End of 'get_plot_formats(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_plot_formats' in the type store
    # Getting the type of 'stypy_return_type' (line 545)
    stypy_return_type_286956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286956)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_plot_formats'
    return stypy_return_type_286956

# Assigning a type to the variable 'get_plot_formats' (line 545)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 0), 'get_plot_formats', get_plot_formats)

@norecursion
def render_figures(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 569)
    False_286957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 56), 'False')
    # Getting the type of 'False' (line 570)
    False_286958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 30), 'False')
    defaults = [False_286957, False_286958]
    # Create a new context for function 'render_figures'
    module_type_store = module_type_store.open_function_context('render_figures', 568, 0, False)
    
    # Passed parameters checking function
    render_figures.stypy_localization = localization
    render_figures.stypy_type_of_self = None
    render_figures.stypy_type_store = module_type_store
    render_figures.stypy_function_name = 'render_figures'
    render_figures.stypy_param_names_list = ['code', 'code_path', 'output_dir', 'output_base', 'context', 'function_name', 'config', 'context_reset', 'close_figs']
    render_figures.stypy_varargs_param_name = None
    render_figures.stypy_kwargs_param_name = None
    render_figures.stypy_call_defaults = defaults
    render_figures.stypy_call_varargs = varargs
    render_figures.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'render_figures', ['code', 'code_path', 'output_dir', 'output_base', 'context', 'function_name', 'config', 'context_reset', 'close_figs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'render_figures', localization, ['code', 'code_path', 'output_dir', 'output_base', 'context', 'function_name', 'config', 'context_reset', 'close_figs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'render_figures(...)' code ##################

    unicode_286959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, (-1)), 'unicode', u'\n    Run a pyplot script and save the images in *output_dir*.\n\n    Save the images under *output_dir* with file names derived from\n    *output_base*\n    ')
    
    # Assigning a Call to a Name (line 577):
    
    # Assigning a Call to a Name (line 577):
    
    # Call to get_plot_formats(...): (line 577)
    # Processing the call arguments (line 577)
    # Getting the type of 'config' (line 577)
    config_286961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 31), 'config', False)
    # Processing the call keyword arguments (line 577)
    kwargs_286962 = {}
    # Getting the type of 'get_plot_formats' (line 577)
    get_plot_formats_286960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 14), 'get_plot_formats', False)
    # Calling get_plot_formats(args, kwargs) (line 577)
    get_plot_formats_call_result_286963 = invoke(stypy.reporting.localization.Localization(__file__, 577, 14), get_plot_formats_286960, *[config_286961], **kwargs_286962)
    
    # Assigning a type to the variable 'formats' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 4), 'formats', get_plot_formats_call_result_286963)
    
    # Assigning a Call to a Name (line 581):
    
    # Assigning a Call to a Name (line 581):
    
    # Call to split_code_at_show(...): (line 581)
    # Processing the call arguments (line 581)
    # Getting the type of 'code' (line 581)
    code_286965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 37), 'code', False)
    # Processing the call keyword arguments (line 581)
    kwargs_286966 = {}
    # Getting the type of 'split_code_at_show' (line 581)
    split_code_at_show_286964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 18), 'split_code_at_show', False)
    # Calling split_code_at_show(args, kwargs) (line 581)
    split_code_at_show_call_result_286967 = invoke(stypy.reporting.localization.Localization(__file__, 581, 18), split_code_at_show_286964, *[code_286965], **kwargs_286966)
    
    # Assigning a type to the variable 'code_pieces' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 4), 'code_pieces', split_code_at_show_call_result_286967)
    
    # Assigning a Name to a Name (line 584):
    
    # Assigning a Name to a Name (line 584):
    # Getting the type of 'True' (line 584)
    True_286968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 17), 'True')
    # Assigning a type to the variable 'all_exists' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'all_exists', True_286968)
    
    # Assigning a Call to a Name (line 585):
    
    # Assigning a Call to a Name (line 585):
    
    # Call to ImageFile(...): (line 585)
    # Processing the call arguments (line 585)
    # Getting the type of 'output_base' (line 585)
    output_base_286970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 20), 'output_base', False)
    # Getting the type of 'output_dir' (line 585)
    output_dir_286971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 33), 'output_dir', False)
    # Processing the call keyword arguments (line 585)
    kwargs_286972 = {}
    # Getting the type of 'ImageFile' (line 585)
    ImageFile_286969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 10), 'ImageFile', False)
    # Calling ImageFile(args, kwargs) (line 585)
    ImageFile_call_result_286973 = invoke(stypy.reporting.localization.Localization(__file__, 585, 10), ImageFile_286969, *[output_base_286970, output_dir_286971], **kwargs_286972)
    
    # Assigning a type to the variable 'img' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'img', ImageFile_call_result_286973)
    
    # Getting the type of 'formats' (line 586)
    formats_286974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 23), 'formats')
    # Testing the type of a for loop iterable (line 586)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 586, 4), formats_286974)
    # Getting the type of the for loop variable (line 586)
    for_loop_var_286975 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 586, 4), formats_286974)
    # Assigning a type to the variable 'format' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'format', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 586, 4), for_loop_var_286975))
    # Assigning a type to the variable 'dpi' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'dpi', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 586, 4), for_loop_var_286975))
    # SSA begins for a for statement (line 586)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to out_of_date(...): (line 587)
    # Processing the call arguments (line 587)
    # Getting the type of 'code_path' (line 587)
    code_path_286977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 23), 'code_path', False)
    
    # Call to filename(...): (line 587)
    # Processing the call arguments (line 587)
    # Getting the type of 'format' (line 587)
    format_286980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 47), 'format', False)
    # Processing the call keyword arguments (line 587)
    kwargs_286981 = {}
    # Getting the type of 'img' (line 587)
    img_286978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 34), 'img', False)
    # Obtaining the member 'filename' of a type (line 587)
    filename_286979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 34), img_286978, 'filename')
    # Calling filename(args, kwargs) (line 587)
    filename_call_result_286982 = invoke(stypy.reporting.localization.Localization(__file__, 587, 34), filename_286979, *[format_286980], **kwargs_286981)
    
    # Processing the call keyword arguments (line 587)
    kwargs_286983 = {}
    # Getting the type of 'out_of_date' (line 587)
    out_of_date_286976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 11), 'out_of_date', False)
    # Calling out_of_date(args, kwargs) (line 587)
    out_of_date_call_result_286984 = invoke(stypy.reporting.localization.Localization(__file__, 587, 11), out_of_date_286976, *[code_path_286977, filename_call_result_286982], **kwargs_286983)
    
    # Testing the type of an if condition (line 587)
    if_condition_286985 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 587, 8), out_of_date_call_result_286984)
    # Assigning a type to the variable 'if_condition_286985' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'if_condition_286985', if_condition_286985)
    # SSA begins for if statement (line 587)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 588):
    
    # Assigning a Name to a Name (line 588):
    # Getting the type of 'False' (line 588)
    False_286986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 25), 'False')
    # Assigning a type to the variable 'all_exists' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 12), 'all_exists', False_286986)
    # SSA join for if statement (line 587)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 590)
    # Processing the call arguments (line 590)
    # Getting the type of 'format' (line 590)
    format_286990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 27), 'format', False)
    # Processing the call keyword arguments (line 590)
    kwargs_286991 = {}
    # Getting the type of 'img' (line 590)
    img_286987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 8), 'img', False)
    # Obtaining the member 'formats' of a type (line 590)
    formats_286988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 8), img_286987, 'formats')
    # Obtaining the member 'append' of a type (line 590)
    append_286989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 8), formats_286988, 'append')
    # Calling append(args, kwargs) (line 590)
    append_call_result_286992 = invoke(stypy.reporting.localization.Localization(__file__, 590, 8), append_286989, *[format_286990], **kwargs_286991)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'all_exists' (line 592)
    all_exists_286993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 7), 'all_exists')
    # Testing the type of an if condition (line 592)
    if_condition_286994 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 592, 4), all_exists_286993)
    # Assigning a type to the variable 'if_condition_286994' (line 592)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 4), 'if_condition_286994', if_condition_286994)
    # SSA begins for if statement (line 592)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'list' (line 593)
    list_286995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 593)
    # Adding element type (line 593)
    
    # Obtaining an instance of the builtin type 'tuple' (line 593)
    tuple_286996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 593)
    # Adding element type (line 593)
    # Getting the type of 'code' (line 593)
    code_286997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 17), 'code')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 593, 17), tuple_286996, code_286997)
    # Adding element type (line 593)
    
    # Obtaining an instance of the builtin type 'list' (line 593)
    list_286998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 593)
    # Adding element type (line 593)
    # Getting the type of 'img' (line 593)
    img_286999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 24), 'img')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 593, 23), list_286998, img_286999)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 593, 17), tuple_286996, list_286998)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 593, 15), list_286995, tuple_286996)
    
    # Assigning a type to the variable 'stypy_return_type' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'stypy_return_type', list_286995)
    # SSA join for if statement (line 592)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 596):
    
    # Assigning a List to a Name (line 596):
    
    # Obtaining an instance of the builtin type 'list' (line 596)
    list_287000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 596)
    
    # Assigning a type to the variable 'results' (line 596)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'results', list_287000)
    
    # Assigning a Name to a Name (line 597):
    
    # Assigning a Name to a Name (line 597):
    # Getting the type of 'True' (line 597)
    True_287001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 17), 'True')
    # Assigning a type to the variable 'all_exists' (line 597)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 4), 'all_exists', True_287001)
    
    
    # Call to enumerate(...): (line 598)
    # Processing the call arguments (line 598)
    # Getting the type of 'code_pieces' (line 598)
    code_pieces_287003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 35), 'code_pieces', False)
    # Processing the call keyword arguments (line 598)
    kwargs_287004 = {}
    # Getting the type of 'enumerate' (line 598)
    enumerate_287002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 25), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 598)
    enumerate_call_result_287005 = invoke(stypy.reporting.localization.Localization(__file__, 598, 25), enumerate_287002, *[code_pieces_287003], **kwargs_287004)
    
    # Testing the type of a for loop iterable (line 598)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 598, 4), enumerate_call_result_287005)
    # Getting the type of the for loop variable (line 598)
    for_loop_var_287006 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 598, 4), enumerate_call_result_287005)
    # Assigning a type to the variable 'i' (line 598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 4), for_loop_var_287006))
    # Assigning a type to the variable 'code_piece' (line 598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'code_piece', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 4), for_loop_var_287006))
    # SSA begins for a for statement (line 598)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a List to a Name (line 599):
    
    # Assigning a List to a Name (line 599):
    
    # Obtaining an instance of the builtin type 'list' (line 599)
    list_287007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 599)
    
    # Assigning a type to the variable 'images' (line 599)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'images', list_287007)
    
    
    # Call to xrange(...): (line 600)
    # Processing the call arguments (line 600)
    int_287009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 24), 'int')
    # Processing the call keyword arguments (line 600)
    kwargs_287010 = {}
    # Getting the type of 'xrange' (line 600)
    xrange_287008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 17), 'xrange', False)
    # Calling xrange(args, kwargs) (line 600)
    xrange_call_result_287011 = invoke(stypy.reporting.localization.Localization(__file__, 600, 17), xrange_287008, *[int_287009], **kwargs_287010)
    
    # Testing the type of a for loop iterable (line 600)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 600, 8), xrange_call_result_287011)
    # Getting the type of the for loop variable (line 600)
    for_loop_var_287012 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 600, 8), xrange_call_result_287011)
    # Assigning a type to the variable 'j' (line 600)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'j', for_loop_var_287012)
    # SSA begins for a for statement (line 600)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Call to len(...): (line 601)
    # Processing the call arguments (line 601)
    # Getting the type of 'code_pieces' (line 601)
    code_pieces_287014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 19), 'code_pieces', False)
    # Processing the call keyword arguments (line 601)
    kwargs_287015 = {}
    # Getting the type of 'len' (line 601)
    len_287013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 15), 'len', False)
    # Calling len(args, kwargs) (line 601)
    len_call_result_287016 = invoke(stypy.reporting.localization.Localization(__file__, 601, 15), len_287013, *[code_pieces_287014], **kwargs_287015)
    
    int_287017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 34), 'int')
    # Applying the binary operator '>' (line 601)
    result_gt_287018 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 15), '>', len_call_result_287016, int_287017)
    
    # Testing the type of an if condition (line 601)
    if_condition_287019 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 601, 12), result_gt_287018)
    # Assigning a type to the variable 'if_condition_287019' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 12), 'if_condition_287019', if_condition_287019)
    # SSA begins for if statement (line 601)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 602):
    
    # Assigning a Call to a Name (line 602):
    
    # Call to ImageFile(...): (line 602)
    # Processing the call arguments (line 602)
    unicode_287021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 32), 'unicode', u'%s_%02d_%02d')
    
    # Obtaining an instance of the builtin type 'tuple' (line 602)
    tuple_287022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 602)
    # Adding element type (line 602)
    # Getting the type of 'output_base' (line 602)
    output_base_287023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 50), 'output_base', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 602, 50), tuple_287022, output_base_287023)
    # Adding element type (line 602)
    # Getting the type of 'i' (line 602)
    i_287024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 63), 'i', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 602, 50), tuple_287022, i_287024)
    # Adding element type (line 602)
    # Getting the type of 'j' (line 602)
    j_287025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 66), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 602, 50), tuple_287022, j_287025)
    
    # Applying the binary operator '%' (line 602)
    result_mod_287026 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 32), '%', unicode_287021, tuple_287022)
    
    # Getting the type of 'output_dir' (line 602)
    output_dir_287027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 70), 'output_dir', False)
    # Processing the call keyword arguments (line 602)
    kwargs_287028 = {}
    # Getting the type of 'ImageFile' (line 602)
    ImageFile_287020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 22), 'ImageFile', False)
    # Calling ImageFile(args, kwargs) (line 602)
    ImageFile_call_result_287029 = invoke(stypy.reporting.localization.Localization(__file__, 602, 22), ImageFile_287020, *[result_mod_287026, output_dir_287027], **kwargs_287028)
    
    # Assigning a type to the variable 'img' (line 602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 16), 'img', ImageFile_call_result_287029)
    # SSA branch for the else part of an if statement (line 601)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 604):
    
    # Assigning a Call to a Name (line 604):
    
    # Call to ImageFile(...): (line 604)
    # Processing the call arguments (line 604)
    unicode_287031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 32), 'unicode', u'%s_%02d')
    
    # Obtaining an instance of the builtin type 'tuple' (line 604)
    tuple_287032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 604)
    # Adding element type (line 604)
    # Getting the type of 'output_base' (line 604)
    output_base_287033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 45), 'output_base', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 604, 45), tuple_287032, output_base_287033)
    # Adding element type (line 604)
    # Getting the type of 'j' (line 604)
    j_287034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 58), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 604, 45), tuple_287032, j_287034)
    
    # Applying the binary operator '%' (line 604)
    result_mod_287035 = python_operator(stypy.reporting.localization.Localization(__file__, 604, 32), '%', unicode_287031, tuple_287032)
    
    # Getting the type of 'output_dir' (line 604)
    output_dir_287036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 62), 'output_dir', False)
    # Processing the call keyword arguments (line 604)
    kwargs_287037 = {}
    # Getting the type of 'ImageFile' (line 604)
    ImageFile_287030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 22), 'ImageFile', False)
    # Calling ImageFile(args, kwargs) (line 604)
    ImageFile_call_result_287038 = invoke(stypy.reporting.localization.Localization(__file__, 604, 22), ImageFile_287030, *[result_mod_287035, output_dir_287036], **kwargs_287037)
    
    # Assigning a type to the variable 'img' (line 604)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 16), 'img', ImageFile_call_result_287038)
    # SSA join for if statement (line 601)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'formats' (line 605)
    formats_287039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 31), 'formats')
    # Testing the type of a for loop iterable (line 605)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 605, 12), formats_287039)
    # Getting the type of the for loop variable (line 605)
    for_loop_var_287040 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 605, 12), formats_287039)
    # Assigning a type to the variable 'format' (line 605)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 12), 'format', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 12), for_loop_var_287040))
    # Assigning a type to the variable 'dpi' (line 605)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 12), 'dpi', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 12), for_loop_var_287040))
    # SSA begins for a for statement (line 605)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to out_of_date(...): (line 606)
    # Processing the call arguments (line 606)
    # Getting the type of 'code_path' (line 606)
    code_path_287042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 31), 'code_path', False)
    
    # Call to filename(...): (line 606)
    # Processing the call arguments (line 606)
    # Getting the type of 'format' (line 606)
    format_287045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 55), 'format', False)
    # Processing the call keyword arguments (line 606)
    kwargs_287046 = {}
    # Getting the type of 'img' (line 606)
    img_287043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 42), 'img', False)
    # Obtaining the member 'filename' of a type (line 606)
    filename_287044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 42), img_287043, 'filename')
    # Calling filename(args, kwargs) (line 606)
    filename_call_result_287047 = invoke(stypy.reporting.localization.Localization(__file__, 606, 42), filename_287044, *[format_287045], **kwargs_287046)
    
    # Processing the call keyword arguments (line 606)
    kwargs_287048 = {}
    # Getting the type of 'out_of_date' (line 606)
    out_of_date_287041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 19), 'out_of_date', False)
    # Calling out_of_date(args, kwargs) (line 606)
    out_of_date_call_result_287049 = invoke(stypy.reporting.localization.Localization(__file__, 606, 19), out_of_date_287041, *[code_path_287042, filename_call_result_287047], **kwargs_287048)
    
    # Testing the type of an if condition (line 606)
    if_condition_287050 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 606, 16), out_of_date_call_result_287049)
    # Assigning a type to the variable 'if_condition_287050' (line 606)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 16), 'if_condition_287050', if_condition_287050)
    # SSA begins for if statement (line 606)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 607):
    
    # Assigning a Name to a Name (line 607):
    # Getting the type of 'False' (line 607)
    False_287051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 33), 'False')
    # Assigning a type to the variable 'all_exists' (line 607)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 20), 'all_exists', False_287051)
    # SSA join for if statement (line 606)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 609)
    # Processing the call arguments (line 609)
    # Getting the type of 'format' (line 609)
    format_287055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 35), 'format', False)
    # Processing the call keyword arguments (line 609)
    kwargs_287056 = {}
    # Getting the type of 'img' (line 609)
    img_287052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 16), 'img', False)
    # Obtaining the member 'formats' of a type (line 609)
    formats_287053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 16), img_287052, 'formats')
    # Obtaining the member 'append' of a type (line 609)
    append_287054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 16), formats_287053, 'append')
    # Calling append(args, kwargs) (line 609)
    append_call_result_287057 = invoke(stypy.reporting.localization.Localization(__file__, 609, 16), append_287054, *[format_287055], **kwargs_287056)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'all_exists' (line 612)
    all_exists_287058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 19), 'all_exists')
    # Applying the 'not' unary operator (line 612)
    result_not__287059 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 15), 'not', all_exists_287058)
    
    # Testing the type of an if condition (line 612)
    if_condition_287060 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 612, 12), result_not__287059)
    # Assigning a type to the variable 'if_condition_287060' (line 612)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 12), 'if_condition_287060', if_condition_287060)
    # SSA begins for if statement (line 612)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Compare to a Name (line 613):
    
    # Assigning a Compare to a Name (line 613):
    
    # Getting the type of 'j' (line 613)
    j_287061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 30), 'j')
    int_287062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 34), 'int')
    # Applying the binary operator '>' (line 613)
    result_gt_287063 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 30), '>', j_287061, int_287062)
    
    # Assigning a type to the variable 'all_exists' (line 613)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 16), 'all_exists', result_gt_287063)
    # SSA join for if statement (line 612)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 615)
    # Processing the call arguments (line 615)
    # Getting the type of 'img' (line 615)
    img_287066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 26), 'img', False)
    # Processing the call keyword arguments (line 615)
    kwargs_287067 = {}
    # Getting the type of 'images' (line 615)
    images_287064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 12), 'images', False)
    # Obtaining the member 'append' of a type (line 615)
    append_287065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 12), images_287064, 'append')
    # Calling append(args, kwargs) (line 615)
    append_call_result_287068 = invoke(stypy.reporting.localization.Localization(__file__, 615, 12), append_287065, *[img_287066], **kwargs_287067)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'all_exists' (line 616)
    all_exists_287069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 15), 'all_exists')
    # Applying the 'not' unary operator (line 616)
    result_not__287070 = python_operator(stypy.reporting.localization.Localization(__file__, 616, 11), 'not', all_exists_287069)
    
    # Testing the type of an if condition (line 616)
    if_condition_287071 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 616, 8), result_not__287070)
    # Assigning a type to the variable 'if_condition_287071' (line 616)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 8), 'if_condition_287071', if_condition_287071)
    # SSA begins for if statement (line 616)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 616)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 618)
    # Processing the call arguments (line 618)
    
    # Obtaining an instance of the builtin type 'tuple' (line 618)
    tuple_287074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 618)
    # Adding element type (line 618)
    # Getting the type of 'code_piece' (line 618)
    code_piece_287075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 24), 'code_piece', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 618, 24), tuple_287074, code_piece_287075)
    # Adding element type (line 618)
    # Getting the type of 'images' (line 618)
    images_287076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 36), 'images', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 618, 24), tuple_287074, images_287076)
    
    # Processing the call keyword arguments (line 618)
    kwargs_287077 = {}
    # Getting the type of 'results' (line 618)
    results_287072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'results', False)
    # Obtaining the member 'append' of a type (line 618)
    append_287073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 8), results_287072, 'append')
    # Calling append(args, kwargs) (line 618)
    append_call_result_287078 = invoke(stypy.reporting.localization.Localization(__file__, 618, 8), append_287073, *[tuple_287074], **kwargs_287077)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'all_exists' (line 620)
    all_exists_287079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 7), 'all_exists')
    # Testing the type of an if condition (line 620)
    if_condition_287080 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 620, 4), all_exists_287079)
    # Assigning a type to the variable 'if_condition_287080' (line 620)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 4), 'if_condition_287080', if_condition_287080)
    # SSA begins for if statement (line 620)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'results' (line 621)
    results_287081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 15), 'results')
    # Assigning a type to the variable 'stypy_return_type' (line 621)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 8), 'stypy_return_type', results_287081)
    # SSA join for if statement (line 620)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 625):
    
    # Assigning a List to a Name (line 625):
    
    # Obtaining an instance of the builtin type 'list' (line 625)
    list_287082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 625)
    
    # Assigning a type to the variable 'results' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'results', list_287082)
    
    # Getting the type of 'context' (line 626)
    context_287083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 7), 'context')
    # Testing the type of an if condition (line 626)
    if_condition_287084 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 626, 4), context_287083)
    # Assigning a type to the variable 'if_condition_287084' (line 626)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 4), 'if_condition_287084', if_condition_287084)
    # SSA begins for if statement (line 626)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 627):
    
    # Assigning a Name to a Name (line 627):
    # Getting the type of 'plot_context' (line 627)
    plot_context_287085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 13), 'plot_context')
    # Assigning a type to the variable 'ns' (line 627)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'ns', plot_context_287085)
    # SSA branch for the else part of an if statement (line 626)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Dict to a Name (line 629):
    
    # Assigning a Dict to a Name (line 629):
    
    # Obtaining an instance of the builtin type 'dict' (line 629)
    dict_287086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 629)
    
    # Assigning a type to the variable 'ns' (line 629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'ns', dict_287086)
    # SSA join for if statement (line 626)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'context_reset' (line 631)
    context_reset_287087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 7), 'context_reset')
    # Testing the type of an if condition (line 631)
    if_condition_287088 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 631, 4), context_reset_287087)
    # Assigning a type to the variable 'if_condition_287088' (line 631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 4), 'if_condition_287088', if_condition_287088)
    # SSA begins for if statement (line 631)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to clear_state(...): (line 632)
    # Processing the call arguments (line 632)
    # Getting the type of 'config' (line 632)
    config_287090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 20), 'config', False)
    # Obtaining the member 'plot_rcparams' of a type (line 632)
    plot_rcparams_287091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 20), config_287090, 'plot_rcparams')
    # Processing the call keyword arguments (line 632)
    kwargs_287092 = {}
    # Getting the type of 'clear_state' (line 632)
    clear_state_287089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 8), 'clear_state', False)
    # Calling clear_state(args, kwargs) (line 632)
    clear_state_call_result_287093 = invoke(stypy.reporting.localization.Localization(__file__, 632, 8), clear_state_287089, *[plot_rcparams_287091], **kwargs_287092)
    
    
    # Call to clear(...): (line 633)
    # Processing the call keyword arguments (line 633)
    kwargs_287096 = {}
    # Getting the type of 'plot_context' (line 633)
    plot_context_287094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 8), 'plot_context', False)
    # Obtaining the member 'clear' of a type (line 633)
    clear_287095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 8), plot_context_287094, 'clear')
    # Calling clear(args, kwargs) (line 633)
    clear_call_result_287097 = invoke(stypy.reporting.localization.Localization(__file__, 633, 8), clear_287095, *[], **kwargs_287096)
    
    # SSA join for if statement (line 631)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 635):
    
    # Assigning a BoolOp to a Name (line 635):
    
    # Evaluating a boolean operation
    
    # Getting the type of 'context' (line 635)
    context_287098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 21), 'context')
    # Applying the 'not' unary operator (line 635)
    result_not__287099 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 17), 'not', context_287098)
    
    # Getting the type of 'close_figs' (line 635)
    close_figs_287100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 32), 'close_figs')
    # Applying the binary operator 'or' (line 635)
    result_or_keyword_287101 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 17), 'or', result_not__287099, close_figs_287100)
    
    # Assigning a type to the variable 'close_figs' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'close_figs', result_or_keyword_287101)
    
    
    # Call to enumerate(...): (line 637)
    # Processing the call arguments (line 637)
    # Getting the type of 'code_pieces' (line 637)
    code_pieces_287103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 35), 'code_pieces', False)
    # Processing the call keyword arguments (line 637)
    kwargs_287104 = {}
    # Getting the type of 'enumerate' (line 637)
    enumerate_287102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 25), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 637)
    enumerate_call_result_287105 = invoke(stypy.reporting.localization.Localization(__file__, 637, 25), enumerate_287102, *[code_pieces_287103], **kwargs_287104)
    
    # Testing the type of a for loop iterable (line 637)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 637, 4), enumerate_call_result_287105)
    # Getting the type of the for loop variable (line 637)
    for_loop_var_287106 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 637, 4), enumerate_call_result_287105)
    # Assigning a type to the variable 'i' (line 637)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 4), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 637, 4), for_loop_var_287106))
    # Assigning a type to the variable 'code_piece' (line 637)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 4), 'code_piece', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 637, 4), for_loop_var_287106))
    # SSA begins for a for statement (line 637)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'context' (line 639)
    context_287107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 15), 'context')
    # Applying the 'not' unary operator (line 639)
    result_not__287108 = python_operator(stypy.reporting.localization.Localization(__file__, 639, 11), 'not', context_287107)
    
    # Getting the type of 'config' (line 639)
    config_287109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 26), 'config')
    # Obtaining the member 'plot_apply_rcparams' of a type (line 639)
    plot_apply_rcparams_287110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 26), config_287109, 'plot_apply_rcparams')
    # Applying the binary operator 'or' (line 639)
    result_or_keyword_287111 = python_operator(stypy.reporting.localization.Localization(__file__, 639, 11), 'or', result_not__287108, plot_apply_rcparams_287110)
    
    # Testing the type of an if condition (line 639)
    if_condition_287112 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 639, 8), result_or_keyword_287111)
    # Assigning a type to the variable 'if_condition_287112' (line 639)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 8), 'if_condition_287112', if_condition_287112)
    # SSA begins for if statement (line 639)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to clear_state(...): (line 640)
    # Processing the call arguments (line 640)
    # Getting the type of 'config' (line 640)
    config_287114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 24), 'config', False)
    # Obtaining the member 'plot_rcparams' of a type (line 640)
    plot_rcparams_287115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 24), config_287114, 'plot_rcparams')
    # Getting the type of 'close_figs' (line 640)
    close_figs_287116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 46), 'close_figs', False)
    # Processing the call keyword arguments (line 640)
    kwargs_287117 = {}
    # Getting the type of 'clear_state' (line 640)
    clear_state_287113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 12), 'clear_state', False)
    # Calling clear_state(args, kwargs) (line 640)
    clear_state_call_result_287118 = invoke(stypy.reporting.localization.Localization(__file__, 640, 12), clear_state_287113, *[plot_rcparams_287115, close_figs_287116], **kwargs_287117)
    
    # SSA branch for the else part of an if statement (line 639)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'close_figs' (line 641)
    close_figs_287119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 13), 'close_figs')
    # Testing the type of an if condition (line 641)
    if_condition_287120 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 641, 13), close_figs_287119)
    # Assigning a type to the variable 'if_condition_287120' (line 641)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 13), 'if_condition_287120', if_condition_287120)
    # SSA begins for if statement (line 641)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to close(...): (line 642)
    # Processing the call arguments (line 642)
    unicode_287123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 22), 'unicode', u'all')
    # Processing the call keyword arguments (line 642)
    kwargs_287124 = {}
    # Getting the type of 'plt' (line 642)
    plt_287121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 12), 'plt', False)
    # Obtaining the member 'close' of a type (line 642)
    close_287122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 12), plt_287121, 'close')
    # Calling close(args, kwargs) (line 642)
    close_call_result_287125 = invoke(stypy.reporting.localization.Localization(__file__, 642, 12), close_287122, *[unicode_287123], **kwargs_287124)
    
    # SSA join for if statement (line 641)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 639)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to run_code(...): (line 644)
    # Processing the call arguments (line 644)
    # Getting the type of 'code_piece' (line 644)
    code_piece_287127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 17), 'code_piece', False)
    # Getting the type of 'code_path' (line 644)
    code_path_287128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 29), 'code_path', False)
    # Getting the type of 'ns' (line 644)
    ns_287129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 40), 'ns', False)
    # Getting the type of 'function_name' (line 644)
    function_name_287130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 44), 'function_name', False)
    # Processing the call keyword arguments (line 644)
    kwargs_287131 = {}
    # Getting the type of 'run_code' (line 644)
    run_code_287126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'run_code', False)
    # Calling run_code(args, kwargs) (line 644)
    run_code_call_result_287132 = invoke(stypy.reporting.localization.Localization(__file__, 644, 8), run_code_287126, *[code_piece_287127, code_path_287128, ns_287129, function_name_287130], **kwargs_287131)
    
    
    # Assigning a List to a Name (line 646):
    
    # Assigning a List to a Name (line 646):
    
    # Obtaining an instance of the builtin type 'list' (line 646)
    list_287133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 646)
    
    # Assigning a type to the variable 'images' (line 646)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 8), 'images', list_287133)
    
    # Assigning a Call to a Name (line 647):
    
    # Assigning a Call to a Name (line 647):
    
    # Call to get_all_fig_managers(...): (line 647)
    # Processing the call keyword arguments (line 647)
    kwargs_287137 = {}
    # Getting the type of '_pylab_helpers' (line 647)
    _pylab_helpers_287134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 23), '_pylab_helpers', False)
    # Obtaining the member 'Gcf' of a type (line 647)
    Gcf_287135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 23), _pylab_helpers_287134, 'Gcf')
    # Obtaining the member 'get_all_fig_managers' of a type (line 647)
    get_all_fig_managers_287136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 23), Gcf_287135, 'get_all_fig_managers')
    # Calling get_all_fig_managers(args, kwargs) (line 647)
    get_all_fig_managers_call_result_287138 = invoke(stypy.reporting.localization.Localization(__file__, 647, 23), get_all_fig_managers_287136, *[], **kwargs_287137)
    
    # Assigning a type to the variable 'fig_managers' (line 647)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'fig_managers', get_all_fig_managers_call_result_287138)
    
    
    # Call to enumerate(...): (line 648)
    # Processing the call arguments (line 648)
    # Getting the type of 'fig_managers' (line 648)
    fig_managers_287140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 35), 'fig_managers', False)
    # Processing the call keyword arguments (line 648)
    kwargs_287141 = {}
    # Getting the type of 'enumerate' (line 648)
    enumerate_287139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 25), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 648)
    enumerate_call_result_287142 = invoke(stypy.reporting.localization.Localization(__file__, 648, 25), enumerate_287139, *[fig_managers_287140], **kwargs_287141)
    
    # Testing the type of a for loop iterable (line 648)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 648, 8), enumerate_call_result_287142)
    # Getting the type of the for loop variable (line 648)
    for_loop_var_287143 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 648, 8), enumerate_call_result_287142)
    # Assigning a type to the variable 'j' (line 648)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 8), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 648, 8), for_loop_var_287143))
    # Assigning a type to the variable 'figman' (line 648)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 8), 'figman', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 648, 8), for_loop_var_287143))
    # SSA begins for a for statement (line 648)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 649)
    # Processing the call arguments (line 649)
    # Getting the type of 'fig_managers' (line 649)
    fig_managers_287145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 19), 'fig_managers', False)
    # Processing the call keyword arguments (line 649)
    kwargs_287146 = {}
    # Getting the type of 'len' (line 649)
    len_287144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 15), 'len', False)
    # Calling len(args, kwargs) (line 649)
    len_call_result_287147 = invoke(stypy.reporting.localization.Localization(__file__, 649, 15), len_287144, *[fig_managers_287145], **kwargs_287146)
    
    int_287148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 36), 'int')
    # Applying the binary operator '==' (line 649)
    result_eq_287149 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 15), '==', len_call_result_287147, int_287148)
    
    
    
    # Call to len(...): (line 649)
    # Processing the call arguments (line 649)
    # Getting the type of 'code_pieces' (line 649)
    code_pieces_287151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 46), 'code_pieces', False)
    # Processing the call keyword arguments (line 649)
    kwargs_287152 = {}
    # Getting the type of 'len' (line 649)
    len_287150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 42), 'len', False)
    # Calling len(args, kwargs) (line 649)
    len_call_result_287153 = invoke(stypy.reporting.localization.Localization(__file__, 649, 42), len_287150, *[code_pieces_287151], **kwargs_287152)
    
    int_287154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 62), 'int')
    # Applying the binary operator '==' (line 649)
    result_eq_287155 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 42), '==', len_call_result_287153, int_287154)
    
    # Applying the binary operator 'and' (line 649)
    result_and_keyword_287156 = python_operator(stypy.reporting.localization.Localization(__file__, 649, 15), 'and', result_eq_287149, result_eq_287155)
    
    # Testing the type of an if condition (line 649)
    if_condition_287157 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 649, 12), result_and_keyword_287156)
    # Assigning a type to the variable 'if_condition_287157' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 12), 'if_condition_287157', if_condition_287157)
    # SSA begins for if statement (line 649)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 650):
    
    # Assigning a Call to a Name (line 650):
    
    # Call to ImageFile(...): (line 650)
    # Processing the call arguments (line 650)
    # Getting the type of 'output_base' (line 650)
    output_base_287159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 32), 'output_base', False)
    # Getting the type of 'output_dir' (line 650)
    output_dir_287160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 45), 'output_dir', False)
    # Processing the call keyword arguments (line 650)
    kwargs_287161 = {}
    # Getting the type of 'ImageFile' (line 650)
    ImageFile_287158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 22), 'ImageFile', False)
    # Calling ImageFile(args, kwargs) (line 650)
    ImageFile_call_result_287162 = invoke(stypy.reporting.localization.Localization(__file__, 650, 22), ImageFile_287158, *[output_base_287159, output_dir_287160], **kwargs_287161)
    
    # Assigning a type to the variable 'img' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 16), 'img', ImageFile_call_result_287162)
    # SSA branch for the else part of an if statement (line 649)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 651)
    # Processing the call arguments (line 651)
    # Getting the type of 'code_pieces' (line 651)
    code_pieces_287164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 21), 'code_pieces', False)
    # Processing the call keyword arguments (line 651)
    kwargs_287165 = {}
    # Getting the type of 'len' (line 651)
    len_287163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 17), 'len', False)
    # Calling len(args, kwargs) (line 651)
    len_call_result_287166 = invoke(stypy.reporting.localization.Localization(__file__, 651, 17), len_287163, *[code_pieces_287164], **kwargs_287165)
    
    int_287167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 37), 'int')
    # Applying the binary operator '==' (line 651)
    result_eq_287168 = python_operator(stypy.reporting.localization.Localization(__file__, 651, 17), '==', len_call_result_287166, int_287167)
    
    # Testing the type of an if condition (line 651)
    if_condition_287169 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 651, 17), result_eq_287168)
    # Assigning a type to the variable 'if_condition_287169' (line 651)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 17), 'if_condition_287169', if_condition_287169)
    # SSA begins for if statement (line 651)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 652):
    
    # Assigning a Call to a Name (line 652):
    
    # Call to ImageFile(...): (line 652)
    # Processing the call arguments (line 652)
    unicode_287171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 32), 'unicode', u'%s_%02d')
    
    # Obtaining an instance of the builtin type 'tuple' (line 652)
    tuple_287172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 652)
    # Adding element type (line 652)
    # Getting the type of 'output_base' (line 652)
    output_base_287173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 45), 'output_base', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 652, 45), tuple_287172, output_base_287173)
    # Adding element type (line 652)
    # Getting the type of 'j' (line 652)
    j_287174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 58), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 652, 45), tuple_287172, j_287174)
    
    # Applying the binary operator '%' (line 652)
    result_mod_287175 = python_operator(stypy.reporting.localization.Localization(__file__, 652, 32), '%', unicode_287171, tuple_287172)
    
    # Getting the type of 'output_dir' (line 652)
    output_dir_287176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 62), 'output_dir', False)
    # Processing the call keyword arguments (line 652)
    kwargs_287177 = {}
    # Getting the type of 'ImageFile' (line 652)
    ImageFile_287170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 22), 'ImageFile', False)
    # Calling ImageFile(args, kwargs) (line 652)
    ImageFile_call_result_287178 = invoke(stypy.reporting.localization.Localization(__file__, 652, 22), ImageFile_287170, *[result_mod_287175, output_dir_287176], **kwargs_287177)
    
    # Assigning a type to the variable 'img' (line 652)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 16), 'img', ImageFile_call_result_287178)
    # SSA branch for the else part of an if statement (line 651)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 654):
    
    # Assigning a Call to a Name (line 654):
    
    # Call to ImageFile(...): (line 654)
    # Processing the call arguments (line 654)
    unicode_287180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 32), 'unicode', u'%s_%02d_%02d')
    
    # Obtaining an instance of the builtin type 'tuple' (line 654)
    tuple_287181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 654)
    # Adding element type (line 654)
    # Getting the type of 'output_base' (line 654)
    output_base_287182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 50), 'output_base', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 50), tuple_287181, output_base_287182)
    # Adding element type (line 654)
    # Getting the type of 'i' (line 654)
    i_287183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 63), 'i', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 50), tuple_287181, i_287183)
    # Adding element type (line 654)
    # Getting the type of 'j' (line 654)
    j_287184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 66), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 50), tuple_287181, j_287184)
    
    # Applying the binary operator '%' (line 654)
    result_mod_287185 = python_operator(stypy.reporting.localization.Localization(__file__, 654, 32), '%', unicode_287180, tuple_287181)
    
    # Getting the type of 'output_dir' (line 655)
    output_dir_287186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 32), 'output_dir', False)
    # Processing the call keyword arguments (line 654)
    kwargs_287187 = {}
    # Getting the type of 'ImageFile' (line 654)
    ImageFile_287179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 22), 'ImageFile', False)
    # Calling ImageFile(args, kwargs) (line 654)
    ImageFile_call_result_287188 = invoke(stypy.reporting.localization.Localization(__file__, 654, 22), ImageFile_287179, *[result_mod_287185, output_dir_287186], **kwargs_287187)
    
    # Assigning a type to the variable 'img' (line 654)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 16), 'img', ImageFile_call_result_287188)
    # SSA join for if statement (line 651)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 649)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 656)
    # Processing the call arguments (line 656)
    # Getting the type of 'img' (line 656)
    img_287191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 26), 'img', False)
    # Processing the call keyword arguments (line 656)
    kwargs_287192 = {}
    # Getting the type of 'images' (line 656)
    images_287189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 12), 'images', False)
    # Obtaining the member 'append' of a type (line 656)
    append_287190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 12), images_287189, 'append')
    # Calling append(args, kwargs) (line 656)
    append_call_result_287193 = invoke(stypy.reporting.localization.Localization(__file__, 656, 12), append_287190, *[img_287191], **kwargs_287192)
    
    
    # Getting the type of 'formats' (line 657)
    formats_287194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 31), 'formats')
    # Testing the type of a for loop iterable (line 657)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 657, 12), formats_287194)
    # Getting the type of the for loop variable (line 657)
    for_loop_var_287195 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 657, 12), formats_287194)
    # Assigning a type to the variable 'format' (line 657)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 12), 'format', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 657, 12), for_loop_var_287195))
    # Assigning a type to the variable 'dpi' (line 657)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 12), 'dpi', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 657, 12), for_loop_var_287195))
    # SSA begins for a for statement (line 657)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # SSA begins for try-except statement (line 658)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to savefig(...): (line 659)
    # Processing the call arguments (line 659)
    
    # Call to filename(...): (line 659)
    # Processing the call arguments (line 659)
    # Getting the type of 'format' (line 659)
    format_287202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 62), 'format', False)
    # Processing the call keyword arguments (line 659)
    kwargs_287203 = {}
    # Getting the type of 'img' (line 659)
    img_287200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 49), 'img', False)
    # Obtaining the member 'filename' of a type (line 659)
    filename_287201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 49), img_287200, 'filename')
    # Calling filename(args, kwargs) (line 659)
    filename_call_result_287204 = invoke(stypy.reporting.localization.Localization(__file__, 659, 49), filename_287201, *[format_287202], **kwargs_287203)
    
    # Processing the call keyword arguments (line 659)
    # Getting the type of 'dpi' (line 659)
    dpi_287205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 75), 'dpi', False)
    keyword_287206 = dpi_287205
    kwargs_287207 = {'dpi': keyword_287206}
    # Getting the type of 'figman' (line 659)
    figman_287196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 20), 'figman', False)
    # Obtaining the member 'canvas' of a type (line 659)
    canvas_287197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 20), figman_287196, 'canvas')
    # Obtaining the member 'figure' of a type (line 659)
    figure_287198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 20), canvas_287197, 'figure')
    # Obtaining the member 'savefig' of a type (line 659)
    savefig_287199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 20), figure_287198, 'savefig')
    # Calling savefig(args, kwargs) (line 659)
    savefig_call_result_287208 = invoke(stypy.reporting.localization.Localization(__file__, 659, 20), savefig_287199, *[filename_call_result_287204], **kwargs_287207)
    
    # SSA branch for the except part of a try statement (line 658)
    # SSA branch for the except 'Exception' branch of a try statement (line 658)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 660)
    Exception_287209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 23), 'Exception')
    # Assigning a type to the variable 'err' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 16), 'err', Exception_287209)
    
    # Call to PlotError(...): (line 661)
    # Processing the call arguments (line 661)
    
    # Call to format_exc(...): (line 661)
    # Processing the call keyword arguments (line 661)
    kwargs_287213 = {}
    # Getting the type of 'traceback' (line 661)
    traceback_287211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 36), 'traceback', False)
    # Obtaining the member 'format_exc' of a type (line 661)
    format_exc_287212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 36), traceback_287211, 'format_exc')
    # Calling format_exc(args, kwargs) (line 661)
    format_exc_call_result_287214 = invoke(stypy.reporting.localization.Localization(__file__, 661, 36), format_exc_287212, *[], **kwargs_287213)
    
    # Processing the call keyword arguments (line 661)
    kwargs_287215 = {}
    # Getting the type of 'PlotError' (line 661)
    PlotError_287210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 26), 'PlotError', False)
    # Calling PlotError(args, kwargs) (line 661)
    PlotError_call_result_287216 = invoke(stypy.reporting.localization.Localization(__file__, 661, 26), PlotError_287210, *[format_exc_call_result_287214], **kwargs_287215)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 661, 20), PlotError_call_result_287216, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 658)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 662)
    # Processing the call arguments (line 662)
    # Getting the type of 'format' (line 662)
    format_287220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 35), 'format', False)
    # Processing the call keyword arguments (line 662)
    kwargs_287221 = {}
    # Getting the type of 'img' (line 662)
    img_287217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 16), 'img', False)
    # Obtaining the member 'formats' of a type (line 662)
    formats_287218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 16), img_287217, 'formats')
    # Obtaining the member 'append' of a type (line 662)
    append_287219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 16), formats_287218, 'append')
    # Calling append(args, kwargs) (line 662)
    append_call_result_287222 = invoke(stypy.reporting.localization.Localization(__file__, 662, 16), append_287219, *[format_287220], **kwargs_287221)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 664)
    # Processing the call arguments (line 664)
    
    # Obtaining an instance of the builtin type 'tuple' (line 664)
    tuple_287225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 664)
    # Adding element type (line 664)
    # Getting the type of 'code_piece' (line 664)
    code_piece_287226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 24), 'code_piece', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 664, 24), tuple_287225, code_piece_287226)
    # Adding element type (line 664)
    # Getting the type of 'images' (line 664)
    images_287227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 36), 'images', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 664, 24), tuple_287225, images_287227)
    
    # Processing the call keyword arguments (line 664)
    kwargs_287228 = {}
    # Getting the type of 'results' (line 664)
    results_287223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 8), 'results', False)
    # Obtaining the member 'append' of a type (line 664)
    append_287224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 8), results_287223, 'append')
    # Calling append(args, kwargs) (line 664)
    append_call_result_287229 = invoke(stypy.reporting.localization.Localization(__file__, 664, 8), append_287224, *[tuple_287225], **kwargs_287228)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'context' (line 666)
    context_287230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 11), 'context')
    # Applying the 'not' unary operator (line 666)
    result_not__287231 = python_operator(stypy.reporting.localization.Localization(__file__, 666, 7), 'not', context_287230)
    
    # Getting the type of 'config' (line 666)
    config_287232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 22), 'config')
    # Obtaining the member 'plot_apply_rcparams' of a type (line 666)
    plot_apply_rcparams_287233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 22), config_287232, 'plot_apply_rcparams')
    # Applying the binary operator 'or' (line 666)
    result_or_keyword_287234 = python_operator(stypy.reporting.localization.Localization(__file__, 666, 7), 'or', result_not__287231, plot_apply_rcparams_287233)
    
    # Testing the type of an if condition (line 666)
    if_condition_287235 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 666, 4), result_or_keyword_287234)
    # Assigning a type to the variable 'if_condition_287235' (line 666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 4), 'if_condition_287235', if_condition_287235)
    # SSA begins for if statement (line 666)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to clear_state(...): (line 667)
    # Processing the call arguments (line 667)
    # Getting the type of 'config' (line 667)
    config_287237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 20), 'config', False)
    # Obtaining the member 'plot_rcparams' of a type (line 667)
    plot_rcparams_287238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 20), config_287237, 'plot_rcparams')
    # Processing the call keyword arguments (line 667)
    
    # Getting the type of 'context' (line 667)
    context_287239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 52), 'context', False)
    # Applying the 'not' unary operator (line 667)
    result_not__287240 = python_operator(stypy.reporting.localization.Localization(__file__, 667, 48), 'not', context_287239)
    
    keyword_287241 = result_not__287240
    kwargs_287242 = {'close': keyword_287241}
    # Getting the type of 'clear_state' (line 667)
    clear_state_287236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 8), 'clear_state', False)
    # Calling clear_state(args, kwargs) (line 667)
    clear_state_call_result_287243 = invoke(stypy.reporting.localization.Localization(__file__, 667, 8), clear_state_287236, *[plot_rcparams_287238], **kwargs_287242)
    
    # SSA join for if statement (line 666)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'results' (line 669)
    results_287244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 11), 'results')
    # Assigning a type to the variable 'stypy_return_type' (line 669)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 4), 'stypy_return_type', results_287244)
    
    # ################# End of 'render_figures(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'render_figures' in the type store
    # Getting the type of 'stypy_return_type' (line 568)
    stypy_return_type_287245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_287245)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'render_figures'
    return stypy_return_type_287245

# Assigning a type to the variable 'render_figures' (line 568)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 0), 'render_figures', render_figures)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 672, 0, False)
    
    # Passed parameters checking function
    run.stypy_localization = localization
    run.stypy_type_of_self = None
    run.stypy_type_store = module_type_store
    run.stypy_function_name = 'run'
    run.stypy_param_names_list = ['arguments', 'content', 'options', 'state_machine', 'state', 'lineno']
    run.stypy_varargs_param_name = None
    run.stypy_kwargs_param_name = None
    run.stypy_call_defaults = defaults
    run.stypy_call_varargs = varargs
    run.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'run', ['arguments', 'content', 'options', 'state_machine', 'state', 'lineno'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'run', localization, ['arguments', 'content', 'options', 'state_machine', 'state', 'lineno'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'run(...)' code ##################

    
    # Assigning a Attribute to a Name (line 673):
    
    # Assigning a Attribute to a Name (line 673):
    # Getting the type of 'state_machine' (line 673)
    state_machine_287246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 15), 'state_machine')
    # Obtaining the member 'document' of a type (line 673)
    document_287247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 15), state_machine_287246, 'document')
    # Assigning a type to the variable 'document' (line 673)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 4), 'document', document_287247)
    
    # Assigning a Attribute to a Name (line 674):
    
    # Assigning a Attribute to a Name (line 674):
    # Getting the type of 'document' (line 674)
    document_287248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 13), 'document')
    # Obtaining the member 'settings' of a type (line 674)
    settings_287249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 13), document_287248, 'settings')
    # Obtaining the member 'env' of a type (line 674)
    env_287250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 13), settings_287249, 'env')
    # Obtaining the member 'config' of a type (line 674)
    config_287251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 13), env_287250, 'config')
    # Assigning a type to the variable 'config' (line 674)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 4), 'config', config_287251)
    
    # Assigning a Compare to a Name (line 675):
    
    # Assigning a Compare to a Name (line 675):
    
    unicode_287252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 13), 'unicode', u'nofigs')
    # Getting the type of 'options' (line 675)
    options_287253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 25), 'options')
    # Applying the binary operator 'in' (line 675)
    result_contains_287254 = python_operator(stypy.reporting.localization.Localization(__file__, 675, 13), 'in', unicode_287252, options_287253)
    
    # Assigning a type to the variable 'nofigs' (line 675)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 4), 'nofigs', result_contains_287254)
    
    # Assigning a Call to a Name (line 677):
    
    # Assigning a Call to a Name (line 677):
    
    # Call to get_plot_formats(...): (line 677)
    # Processing the call arguments (line 677)
    # Getting the type of 'config' (line 677)
    config_287256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 31), 'config', False)
    # Processing the call keyword arguments (line 677)
    kwargs_287257 = {}
    # Getting the type of 'get_plot_formats' (line 677)
    get_plot_formats_287255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 14), 'get_plot_formats', False)
    # Calling get_plot_formats(args, kwargs) (line 677)
    get_plot_formats_call_result_287258 = invoke(stypy.reporting.localization.Localization(__file__, 677, 14), get_plot_formats_287255, *[config_287256], **kwargs_287257)
    
    # Assigning a type to the variable 'formats' (line 677)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 4), 'formats', get_plot_formats_call_result_287258)
    
    # Assigning a Subscript to a Name (line 678):
    
    # Assigning a Subscript to a Name (line 678):
    
    # Obtaining the type of the subscript
    int_287259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 29), 'int')
    
    # Obtaining the type of the subscript
    int_287260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 26), 'int')
    # Getting the type of 'formats' (line 678)
    formats_287261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 18), 'formats')
    # Obtaining the member '__getitem__' of a type (line 678)
    getitem___287262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 18), formats_287261, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 678)
    subscript_call_result_287263 = invoke(stypy.reporting.localization.Localization(__file__, 678, 18), getitem___287262, int_287260)
    
    # Obtaining the member '__getitem__' of a type (line 678)
    getitem___287264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 18), subscript_call_result_287263, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 678)
    subscript_call_result_287265 = invoke(stypy.reporting.localization.Localization(__file__, 678, 18), getitem___287264, int_287259)
    
    # Assigning a type to the variable 'default_fmt' (line 678)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 4), 'default_fmt', subscript_call_result_287265)
    
    # Call to setdefault(...): (line 680)
    # Processing the call arguments (line 680)
    unicode_287268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 23), 'unicode', u'include-source')
    # Getting the type of 'config' (line 680)
    config_287269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 41), 'config', False)
    # Obtaining the member 'plot_include_source' of a type (line 680)
    plot_include_source_287270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 41), config_287269, 'plot_include_source')
    # Processing the call keyword arguments (line 680)
    kwargs_287271 = {}
    # Getting the type of 'options' (line 680)
    options_287266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 4), 'options', False)
    # Obtaining the member 'setdefault' of a type (line 680)
    setdefault_287267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 4), options_287266, 'setdefault')
    # Calling setdefault(args, kwargs) (line 680)
    setdefault_call_result_287272 = invoke(stypy.reporting.localization.Localization(__file__, 680, 4), setdefault_287267, *[unicode_287268, plot_include_source_287270], **kwargs_287271)
    
    
    # Assigning a Compare to a Name (line 681):
    
    # Assigning a Compare to a Name (line 681):
    
    unicode_287273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 19), 'unicode', u'context')
    # Getting the type of 'options' (line 681)
    options_287274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 32), 'options')
    # Applying the binary operator 'in' (line 681)
    result_contains_287275 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 19), 'in', unicode_287273, options_287274)
    
    # Assigning a type to the variable 'keep_context' (line 681)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 4), 'keep_context', result_contains_287275)
    
    # Assigning a IfExp to a Name (line 682):
    
    # Assigning a IfExp to a Name (line 682):
    
    
    # Getting the type of 'keep_context' (line 682)
    keep_context_287276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 30), 'keep_context')
    # Applying the 'not' unary operator (line 682)
    result_not__287277 = python_operator(stypy.reporting.localization.Localization(__file__, 682, 26), 'not', keep_context_287276)
    
    # Testing the type of an if expression (line 682)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 682, 18), result_not__287277)
    # SSA begins for if expression (line 682)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'None' (line 682)
    None_287278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 18), 'None')
    # SSA branch for the else part of an if expression (line 682)
    module_type_store.open_ssa_branch('if expression else')
    
    # Obtaining the type of the subscript
    unicode_287279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 56), 'unicode', u'context')
    # Getting the type of 'options' (line 682)
    options_287280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 48), 'options')
    # Obtaining the member '__getitem__' of a type (line 682)
    getitem___287281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 48), options_287280, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 682)
    subscript_call_result_287282 = invoke(stypy.reporting.localization.Localization(__file__, 682, 48), getitem___287281, unicode_287279)
    
    # SSA join for if expression (line 682)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_287283 = union_type.UnionType.add(None_287278, subscript_call_result_287282)
    
    # Assigning a type to the variable 'context_opt' (line 682)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 4), 'context_opt', if_exp_287283)
    
    # Assigning a Subscript to a Name (line 684):
    
    # Assigning a Subscript to a Name (line 684):
    
    # Obtaining the type of the subscript
    unicode_287284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 35), 'unicode', u'source')
    # Getting the type of 'document' (line 684)
    document_287285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 15), 'document')
    # Obtaining the member 'attributes' of a type (line 684)
    attributes_287286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 15), document_287285, 'attributes')
    # Obtaining the member '__getitem__' of a type (line 684)
    getitem___287287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 15), attributes_287286, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 684)
    subscript_call_result_287288 = invoke(stypy.reporting.localization.Localization(__file__, 684, 15), getitem___287287, unicode_287284)
    
    # Assigning a type to the variable 'rst_file' (line 684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 4), 'rst_file', subscript_call_result_287288)
    
    # Assigning a Call to a Name (line 685):
    
    # Assigning a Call to a Name (line 685):
    
    # Call to dirname(...): (line 685)
    # Processing the call arguments (line 685)
    # Getting the type of 'rst_file' (line 685)
    rst_file_287292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 30), 'rst_file', False)
    # Processing the call keyword arguments (line 685)
    kwargs_287293 = {}
    # Getting the type of 'os' (line 685)
    os_287289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 14), 'os', False)
    # Obtaining the member 'path' of a type (line 685)
    path_287290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 685, 14), os_287289, 'path')
    # Obtaining the member 'dirname' of a type (line 685)
    dirname_287291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 685, 14), path_287290, 'dirname')
    # Calling dirname(args, kwargs) (line 685)
    dirname_call_result_287294 = invoke(stypy.reporting.localization.Localization(__file__, 685, 14), dirname_287291, *[rst_file_287292], **kwargs_287293)
    
    # Assigning a type to the variable 'rst_dir' (line 685)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 4), 'rst_dir', dirname_call_result_287294)
    
    
    # Call to len(...): (line 687)
    # Processing the call arguments (line 687)
    # Getting the type of 'arguments' (line 687)
    arguments_287296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 11), 'arguments', False)
    # Processing the call keyword arguments (line 687)
    kwargs_287297 = {}
    # Getting the type of 'len' (line 687)
    len_287295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 7), 'len', False)
    # Calling len(args, kwargs) (line 687)
    len_call_result_287298 = invoke(stypy.reporting.localization.Localization(__file__, 687, 7), len_287295, *[arguments_287296], **kwargs_287297)
    
    # Testing the type of an if condition (line 687)
    if_condition_287299 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 687, 4), len_call_result_287298)
    # Assigning a type to the variable 'if_condition_287299' (line 687)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 4), 'if_condition_287299', if_condition_287299)
    # SSA begins for if statement (line 687)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'config' (line 688)
    config_287300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 15), 'config')
    # Obtaining the member 'plot_basedir' of a type (line 688)
    plot_basedir_287301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 15), config_287300, 'plot_basedir')
    # Applying the 'not' unary operator (line 688)
    result_not__287302 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 11), 'not', plot_basedir_287301)
    
    # Testing the type of an if condition (line 688)
    if_condition_287303 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 688, 8), result_not__287302)
    # Assigning a type to the variable 'if_condition_287303' (line 688)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 8), 'if_condition_287303', if_condition_287303)
    # SSA begins for if statement (line 688)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 689):
    
    # Assigning a Call to a Name (line 689):
    
    # Call to join(...): (line 689)
    # Processing the call arguments (line 689)
    # Getting the type of 'setup' (line 689)
    setup_287307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 44), 'setup', False)
    # Obtaining the member 'app' of a type (line 689)
    app_287308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 44), setup_287307, 'app')
    # Obtaining the member 'builder' of a type (line 689)
    builder_287309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 44), app_287308, 'builder')
    # Obtaining the member 'srcdir' of a type (line 689)
    srcdir_287310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 44), builder_287309, 'srcdir')
    
    # Call to uri(...): (line 690)
    # Processing the call arguments (line 690)
    
    # Obtaining the type of the subscript
    int_287313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 69), 'int')
    # Getting the type of 'arguments' (line 690)
    arguments_287314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 59), 'arguments', False)
    # Obtaining the member '__getitem__' of a type (line 690)
    getitem___287315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 59), arguments_287314, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 690)
    subscript_call_result_287316 = invoke(stypy.reporting.localization.Localization(__file__, 690, 59), getitem___287315, int_287313)
    
    # Processing the call keyword arguments (line 690)
    kwargs_287317 = {}
    # Getting the type of 'directives' (line 690)
    directives_287311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 44), 'directives', False)
    # Obtaining the member 'uri' of a type (line 690)
    uri_287312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 44), directives_287311, 'uri')
    # Calling uri(args, kwargs) (line 690)
    uri_call_result_287318 = invoke(stypy.reporting.localization.Localization(__file__, 690, 44), uri_287312, *[subscript_call_result_287316], **kwargs_287317)
    
    # Processing the call keyword arguments (line 689)
    kwargs_287319 = {}
    # Getting the type of 'os' (line 689)
    os_287304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 31), 'os', False)
    # Obtaining the member 'path' of a type (line 689)
    path_287305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 31), os_287304, 'path')
    # Obtaining the member 'join' of a type (line 689)
    join_287306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 31), path_287305, 'join')
    # Calling join(args, kwargs) (line 689)
    join_call_result_287320 = invoke(stypy.reporting.localization.Localization(__file__, 689, 31), join_287306, *[srcdir_287310, uri_call_result_287318], **kwargs_287319)
    
    # Assigning a type to the variable 'source_file_name' (line 689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 12), 'source_file_name', join_call_result_287320)
    # SSA branch for the else part of an if statement (line 688)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 692):
    
    # Assigning a Call to a Name (line 692):
    
    # Call to join(...): (line 692)
    # Processing the call arguments (line 692)
    # Getting the type of 'setup' (line 692)
    setup_287324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 44), 'setup', False)
    # Obtaining the member 'confdir' of a type (line 692)
    confdir_287325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 44), setup_287324, 'confdir')
    # Getting the type of 'config' (line 692)
    config_287326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 59), 'config', False)
    # Obtaining the member 'plot_basedir' of a type (line 692)
    plot_basedir_287327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 59), config_287326, 'plot_basedir')
    
    # Call to uri(...): (line 693)
    # Processing the call arguments (line 693)
    
    # Obtaining the type of the subscript
    int_287330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 69), 'int')
    # Getting the type of 'arguments' (line 693)
    arguments_287331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 59), 'arguments', False)
    # Obtaining the member '__getitem__' of a type (line 693)
    getitem___287332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 59), arguments_287331, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 693)
    subscript_call_result_287333 = invoke(stypy.reporting.localization.Localization(__file__, 693, 59), getitem___287332, int_287330)
    
    # Processing the call keyword arguments (line 693)
    kwargs_287334 = {}
    # Getting the type of 'directives' (line 693)
    directives_287328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 44), 'directives', False)
    # Obtaining the member 'uri' of a type (line 693)
    uri_287329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 44), directives_287328, 'uri')
    # Calling uri(args, kwargs) (line 693)
    uri_call_result_287335 = invoke(stypy.reporting.localization.Localization(__file__, 693, 44), uri_287329, *[subscript_call_result_287333], **kwargs_287334)
    
    # Processing the call keyword arguments (line 692)
    kwargs_287336 = {}
    # Getting the type of 'os' (line 692)
    os_287321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 31), 'os', False)
    # Obtaining the member 'path' of a type (line 692)
    path_287322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 31), os_287321, 'path')
    # Obtaining the member 'join' of a type (line 692)
    join_287323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 31), path_287322, 'join')
    # Calling join(args, kwargs) (line 692)
    join_call_result_287337 = invoke(stypy.reporting.localization.Localization(__file__, 692, 31), join_287323, *[confdir_287325, plot_basedir_287327, uri_call_result_287335], **kwargs_287336)
    
    # Assigning a type to the variable 'source_file_name' (line 692)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 12), 'source_file_name', join_call_result_287337)
    # SSA join for if statement (line 688)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 696):
    
    # Assigning a Call to a Name (line 696):
    
    # Call to join(...): (line 696)
    # Processing the call arguments (line 696)
    # Getting the type of 'content' (line 696)
    content_287340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 28), 'content', False)
    # Processing the call keyword arguments (line 696)
    kwargs_287341 = {}
    unicode_287338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 18), 'unicode', u'\n')
    # Obtaining the member 'join' of a type (line 696)
    join_287339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 696, 18), unicode_287338, 'join')
    # Calling join(args, kwargs) (line 696)
    join_call_result_287342 = invoke(stypy.reporting.localization.Localization(__file__, 696, 18), join_287339, *[content_287340], **kwargs_287341)
    
    # Assigning a type to the variable 'caption' (line 696)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 8), 'caption', join_call_result_287342)
    
    
    
    # Call to len(...): (line 699)
    # Processing the call arguments (line 699)
    # Getting the type of 'arguments' (line 699)
    arguments_287344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 15), 'arguments', False)
    # Processing the call keyword arguments (line 699)
    kwargs_287345 = {}
    # Getting the type of 'len' (line 699)
    len_287343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 11), 'len', False)
    # Calling len(args, kwargs) (line 699)
    len_call_result_287346 = invoke(stypy.reporting.localization.Localization(__file__, 699, 11), len_287343, *[arguments_287344], **kwargs_287345)
    
    int_287347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 29), 'int')
    # Applying the binary operator '==' (line 699)
    result_eq_287348 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 11), '==', len_call_result_287346, int_287347)
    
    # Testing the type of an if condition (line 699)
    if_condition_287349 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 699, 8), result_eq_287348)
    # Assigning a type to the variable 'if_condition_287349' (line 699)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 8), 'if_condition_287349', if_condition_287349)
    # SSA begins for if statement (line 699)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 700):
    
    # Assigning a Subscript to a Name (line 700):
    
    # Obtaining the type of the subscript
    int_287350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 38), 'int')
    # Getting the type of 'arguments' (line 700)
    arguments_287351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 28), 'arguments')
    # Obtaining the member '__getitem__' of a type (line 700)
    getitem___287352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 28), arguments_287351, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 700)
    subscript_call_result_287353 = invoke(stypy.reporting.localization.Localization(__file__, 700, 28), getitem___287352, int_287350)
    
    # Assigning a type to the variable 'function_name' (line 700)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 12), 'function_name', subscript_call_result_287353)
    # SSA branch for the else part of an if statement (line 699)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 702):
    
    # Assigning a Name to a Name (line 702):
    # Getting the type of 'None' (line 702)
    None_287354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 28), 'None')
    # Assigning a type to the variable 'function_name' (line 702)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 12), 'function_name', None_287354)
    # SSA join for if statement (line 699)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to open(...): (line 704)
    # Processing the call arguments (line 704)
    # Getting the type of 'source_file_name' (line 704)
    source_file_name_287357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 21), 'source_file_name', False)
    unicode_287358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 39), 'unicode', u'r')
    # Processing the call keyword arguments (line 704)
    unicode_287359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 53), 'unicode', u'utf-8')
    keyword_287360 = unicode_287359
    kwargs_287361 = {'encoding': keyword_287360}
    # Getting the type of 'io' (line 704)
    io_287355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 13), 'io', False)
    # Obtaining the member 'open' of a type (line 704)
    open_287356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 13), io_287355, 'open')
    # Calling open(args, kwargs) (line 704)
    open_call_result_287362 = invoke(stypy.reporting.localization.Localization(__file__, 704, 13), open_287356, *[source_file_name_287357, unicode_287358], **kwargs_287361)
    
    with_287363 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 704, 13), open_call_result_287362, 'with parameter', '__enter__', '__exit__')

    if with_287363:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 704)
        enter___287364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 13), open_call_result_287362, '__enter__')
        with_enter_287365 = invoke(stypy.reporting.localization.Localization(__file__, 704, 13), enter___287364)
        # Assigning a type to the variable 'fd' (line 704)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 13), 'fd', with_enter_287365)
        
        # Assigning a Call to a Name (line 705):
        
        # Assigning a Call to a Name (line 705):
        
        # Call to read(...): (line 705)
        # Processing the call keyword arguments (line 705)
        kwargs_287368 = {}
        # Getting the type of 'fd' (line 705)
        fd_287366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 19), 'fd', False)
        # Obtaining the member 'read' of a type (line 705)
        read_287367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 19), fd_287366, 'read')
        # Calling read(args, kwargs) (line 705)
        read_call_result_287369 = invoke(stypy.reporting.localization.Localization(__file__, 705, 19), read_287367, *[], **kwargs_287368)
        
        # Assigning a type to the variable 'code' (line 705)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 12), 'code', read_call_result_287369)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 704)
        exit___287370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 13), open_call_result_287362, '__exit__')
        with_exit_287371 = invoke(stypy.reporting.localization.Localization(__file__, 704, 13), exit___287370, None, None, None)

    
    # Assigning a Call to a Name (line 706):
    
    # Assigning a Call to a Name (line 706):
    
    # Call to basename(...): (line 706)
    # Processing the call arguments (line 706)
    # Getting the type of 'source_file_name' (line 706)
    source_file_name_287375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 39), 'source_file_name', False)
    # Processing the call keyword arguments (line 706)
    kwargs_287376 = {}
    # Getting the type of 'os' (line 706)
    os_287372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 22), 'os', False)
    # Obtaining the member 'path' of a type (line 706)
    path_287373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 22), os_287372, 'path')
    # Obtaining the member 'basename' of a type (line 706)
    basename_287374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 22), path_287373, 'basename')
    # Calling basename(args, kwargs) (line 706)
    basename_call_result_287377 = invoke(stypy.reporting.localization.Localization(__file__, 706, 22), basename_287374, *[source_file_name_287375], **kwargs_287376)
    
    # Assigning a type to the variable 'output_base' (line 706)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'output_base', basename_call_result_287377)
    # SSA branch for the else part of an if statement (line 687)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 708):
    
    # Assigning a Name to a Name (line 708):
    # Getting the type of 'rst_file' (line 708)
    rst_file_287378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 27), 'rst_file')
    # Assigning a type to the variable 'source_file_name' (line 708)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 8), 'source_file_name', rst_file_287378)
    
    # Assigning a Call to a Name (line 709):
    
    # Assigning a Call to a Name (line 709):
    
    # Call to dedent(...): (line 709)
    # Processing the call arguments (line 709)
    
    # Call to join(...): (line 709)
    # Processing the call arguments (line 709)
    
    # Call to map(...): (line 709)
    # Processing the call arguments (line 709)
    # Getting the type of 'six' (line 709)
    six_287384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 45), 'six', False)
    # Obtaining the member 'text_type' of a type (line 709)
    text_type_287385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 45), six_287384, 'text_type')
    # Getting the type of 'content' (line 709)
    content_287386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 60), 'content', False)
    # Processing the call keyword arguments (line 709)
    kwargs_287387 = {}
    # Getting the type of 'map' (line 709)
    map_287383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 41), 'map', False)
    # Calling map(args, kwargs) (line 709)
    map_call_result_287388 = invoke(stypy.reporting.localization.Localization(__file__, 709, 41), map_287383, *[text_type_287385, content_287386], **kwargs_287387)
    
    # Processing the call keyword arguments (line 709)
    kwargs_287389 = {}
    unicode_287381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 31), 'unicode', u'\n')
    # Obtaining the member 'join' of a type (line 709)
    join_287382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 31), unicode_287381, 'join')
    # Calling join(args, kwargs) (line 709)
    join_call_result_287390 = invoke(stypy.reporting.localization.Localization(__file__, 709, 31), join_287382, *[map_call_result_287388], **kwargs_287389)
    
    # Processing the call keyword arguments (line 709)
    kwargs_287391 = {}
    # Getting the type of 'textwrap' (line 709)
    textwrap_287379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 15), 'textwrap', False)
    # Obtaining the member 'dedent' of a type (line 709)
    dedent_287380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 15), textwrap_287379, 'dedent')
    # Calling dedent(args, kwargs) (line 709)
    dedent_call_result_287392 = invoke(stypy.reporting.localization.Localization(__file__, 709, 15), dedent_287380, *[join_call_result_287390], **kwargs_287391)
    
    # Assigning a type to the variable 'code' (line 709)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 8), 'code', dedent_call_result_287392)
    
    # Assigning a BinOp to a Name (line 710):
    
    # Assigning a BinOp to a Name (line 710):
    
    # Call to get(...): (line 710)
    # Processing the call arguments (line 710)
    unicode_287396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 42), 'unicode', u'_plot_counter')
    int_287397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 59), 'int')
    # Processing the call keyword arguments (line 710)
    kwargs_287398 = {}
    # Getting the type of 'document' (line 710)
    document_287393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 18), 'document', False)
    # Obtaining the member 'attributes' of a type (line 710)
    attributes_287394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 18), document_287393, 'attributes')
    # Obtaining the member 'get' of a type (line 710)
    get_287395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 18), attributes_287394, 'get')
    # Calling get(args, kwargs) (line 710)
    get_call_result_287399 = invoke(stypy.reporting.localization.Localization(__file__, 710, 18), get_287395, *[unicode_287396, int_287397], **kwargs_287398)
    
    int_287400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 64), 'int')
    # Applying the binary operator '+' (line 710)
    result_add_287401 = python_operator(stypy.reporting.localization.Localization(__file__, 710, 18), '+', get_call_result_287399, int_287400)
    
    # Assigning a type to the variable 'counter' (line 710)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 710, 8), 'counter', result_add_287401)
    
    # Assigning a Name to a Subscript (line 711):
    
    # Assigning a Name to a Subscript (line 711):
    # Getting the type of 'counter' (line 711)
    counter_287402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 47), 'counter')
    # Getting the type of 'document' (line 711)
    document_287403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 8), 'document')
    # Obtaining the member 'attributes' of a type (line 711)
    attributes_287404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 8), document_287403, 'attributes')
    unicode_287405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 28), 'unicode', u'_plot_counter')
    # Storing an element on a container (line 711)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 8), attributes_287404, (unicode_287405, counter_287402))
    
    # Assigning a Call to a Tuple (line 712):
    
    # Assigning a Call to a Name:
    
    # Call to splitext(...): (line 712)
    # Processing the call arguments (line 712)
    
    # Call to basename(...): (line 712)
    # Processing the call arguments (line 712)
    # Getting the type of 'source_file_name' (line 712)
    source_file_name_287412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 54), 'source_file_name', False)
    # Processing the call keyword arguments (line 712)
    kwargs_287413 = {}
    # Getting the type of 'os' (line 712)
    os_287409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 37), 'os', False)
    # Obtaining the member 'path' of a type (line 712)
    path_287410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 37), os_287409, 'path')
    # Obtaining the member 'basename' of a type (line 712)
    basename_287411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 37), path_287410, 'basename')
    # Calling basename(args, kwargs) (line 712)
    basename_call_result_287414 = invoke(stypy.reporting.localization.Localization(__file__, 712, 37), basename_287411, *[source_file_name_287412], **kwargs_287413)
    
    # Processing the call keyword arguments (line 712)
    kwargs_287415 = {}
    # Getting the type of 'os' (line 712)
    os_287406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 20), 'os', False)
    # Obtaining the member 'path' of a type (line 712)
    path_287407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 20), os_287406, 'path')
    # Obtaining the member 'splitext' of a type (line 712)
    splitext_287408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 20), path_287407, 'splitext')
    # Calling splitext(args, kwargs) (line 712)
    splitext_call_result_287416 = invoke(stypy.reporting.localization.Localization(__file__, 712, 20), splitext_287408, *[basename_call_result_287414], **kwargs_287415)
    
    # Assigning a type to the variable 'call_assignment_285938' (line 712)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'call_assignment_285938', splitext_call_result_287416)
    
    # Assigning a Call to a Name (line 712):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_287419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 8), 'int')
    # Processing the call keyword arguments
    kwargs_287420 = {}
    # Getting the type of 'call_assignment_285938' (line 712)
    call_assignment_285938_287417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'call_assignment_285938', False)
    # Obtaining the member '__getitem__' of a type (line 712)
    getitem___287418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 8), call_assignment_285938_287417, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_287421 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___287418, *[int_287419], **kwargs_287420)
    
    # Assigning a type to the variable 'call_assignment_285939' (line 712)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'call_assignment_285939', getitem___call_result_287421)
    
    # Assigning a Name to a Name (line 712):
    # Getting the type of 'call_assignment_285939' (line 712)
    call_assignment_285939_287422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'call_assignment_285939')
    # Assigning a type to the variable 'base' (line 712)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'base', call_assignment_285939_287422)
    
    # Assigning a Call to a Name (line 712):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_287425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 8), 'int')
    # Processing the call keyword arguments
    kwargs_287426 = {}
    # Getting the type of 'call_assignment_285938' (line 712)
    call_assignment_285938_287423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'call_assignment_285938', False)
    # Obtaining the member '__getitem__' of a type (line 712)
    getitem___287424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 8), call_assignment_285938_287423, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_287427 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___287424, *[int_287425], **kwargs_287426)
    
    # Assigning a type to the variable 'call_assignment_285940' (line 712)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'call_assignment_285940', getitem___call_result_287427)
    
    # Assigning a Name to a Name (line 712):
    # Getting the type of 'call_assignment_285940' (line 712)
    call_assignment_285940_287428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'call_assignment_285940')
    # Assigning a type to the variable 'ext' (line 712)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 14), 'ext', call_assignment_285940_287428)
    
    # Assigning a BinOp to a Name (line 713):
    
    # Assigning a BinOp to a Name (line 713):
    unicode_287429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 22), 'unicode', u'%s-%d.py')
    
    # Obtaining an instance of the builtin type 'tuple' (line 713)
    tuple_287430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 713)
    # Adding element type (line 713)
    # Getting the type of 'base' (line 713)
    base_287431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 36), 'base')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 713, 36), tuple_287430, base_287431)
    # Adding element type (line 713)
    # Getting the type of 'counter' (line 713)
    counter_287432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 42), 'counter')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 713, 36), tuple_287430, counter_287432)
    
    # Applying the binary operator '%' (line 713)
    result_mod_287433 = python_operator(stypy.reporting.localization.Localization(__file__, 713, 22), '%', unicode_287429, tuple_287430)
    
    # Assigning a type to the variable 'output_base' (line 713)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 8), 'output_base', result_mod_287433)
    
    # Assigning a Name to a Name (line 714):
    
    # Assigning a Name to a Name (line 714):
    # Getting the type of 'None' (line 714)
    None_287434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 24), 'None')
    # Assigning a type to the variable 'function_name' (line 714)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 8), 'function_name', None_287434)
    
    # Assigning a Str to a Name (line 715):
    
    # Assigning a Str to a Name (line 715):
    unicode_287435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 18), 'unicode', u'')
    # Assigning a type to the variable 'caption' (line 715)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 8), 'caption', unicode_287435)
    # SSA join for if statement (line 687)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 717):
    
    # Assigning a Call to a Name:
    
    # Call to splitext(...): (line 717)
    # Processing the call arguments (line 717)
    # Getting the type of 'output_base' (line 717)
    output_base_287439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 40), 'output_base', False)
    # Processing the call keyword arguments (line 717)
    kwargs_287440 = {}
    # Getting the type of 'os' (line 717)
    os_287436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 23), 'os', False)
    # Obtaining the member 'path' of a type (line 717)
    path_287437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 23), os_287436, 'path')
    # Obtaining the member 'splitext' of a type (line 717)
    splitext_287438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 23), path_287437, 'splitext')
    # Calling splitext(args, kwargs) (line 717)
    splitext_call_result_287441 = invoke(stypy.reporting.localization.Localization(__file__, 717, 23), splitext_287438, *[output_base_287439], **kwargs_287440)
    
    # Assigning a type to the variable 'call_assignment_285941' (line 717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 4), 'call_assignment_285941', splitext_call_result_287441)
    
    # Assigning a Call to a Name (line 717):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_287444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 4), 'int')
    # Processing the call keyword arguments
    kwargs_287445 = {}
    # Getting the type of 'call_assignment_285941' (line 717)
    call_assignment_285941_287442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 4), 'call_assignment_285941', False)
    # Obtaining the member '__getitem__' of a type (line 717)
    getitem___287443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 4), call_assignment_285941_287442, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_287446 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___287443, *[int_287444], **kwargs_287445)
    
    # Assigning a type to the variable 'call_assignment_285942' (line 717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 4), 'call_assignment_285942', getitem___call_result_287446)
    
    # Assigning a Name to a Name (line 717):
    # Getting the type of 'call_assignment_285942' (line 717)
    call_assignment_285942_287447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 4), 'call_assignment_285942')
    # Assigning a type to the variable 'base' (line 717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 4), 'base', call_assignment_285942_287447)
    
    # Assigning a Call to a Name (line 717):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_287450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 4), 'int')
    # Processing the call keyword arguments
    kwargs_287451 = {}
    # Getting the type of 'call_assignment_285941' (line 717)
    call_assignment_285941_287448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 4), 'call_assignment_285941', False)
    # Obtaining the member '__getitem__' of a type (line 717)
    getitem___287449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 4), call_assignment_285941_287448, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_287452 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___287449, *[int_287450], **kwargs_287451)
    
    # Assigning a type to the variable 'call_assignment_285943' (line 717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 4), 'call_assignment_285943', getitem___call_result_287452)
    
    # Assigning a Name to a Name (line 717):
    # Getting the type of 'call_assignment_285943' (line 717)
    call_assignment_285943_287453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 4), 'call_assignment_285943')
    # Assigning a type to the variable 'source_ext' (line 717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 10), 'source_ext', call_assignment_285943_287453)
    
    
    # Getting the type of 'source_ext' (line 718)
    source_ext_287454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 7), 'source_ext')
    
    # Obtaining an instance of the builtin type 'tuple' (line 718)
    tuple_287455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 718)
    # Adding element type (line 718)
    unicode_287456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 22), 'unicode', u'.py')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), tuple_287455, unicode_287456)
    # Adding element type (line 718)
    unicode_287457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 29), 'unicode', u'.rst')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), tuple_287455, unicode_287457)
    # Adding element type (line 718)
    unicode_287458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 37), 'unicode', u'.txt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), tuple_287455, unicode_287458)
    
    # Applying the binary operator 'in' (line 718)
    result_contains_287459 = python_operator(stypy.reporting.localization.Localization(__file__, 718, 7), 'in', source_ext_287454, tuple_287455)
    
    # Testing the type of an if condition (line 718)
    if_condition_287460 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 718, 4), result_contains_287459)
    # Assigning a type to the variable 'if_condition_287460' (line 718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 4), 'if_condition_287460', if_condition_287460)
    # SSA begins for if statement (line 718)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 719):
    
    # Assigning a Name to a Name (line 719):
    # Getting the type of 'base' (line 719)
    base_287461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 22), 'base')
    # Assigning a type to the variable 'output_base' (line 719)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 8), 'output_base', base_287461)
    # SSA branch for the else part of an if statement (line 718)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 721):
    
    # Assigning a Str to a Name (line 721):
    unicode_287462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 21), 'unicode', u'')
    # Assigning a type to the variable 'source_ext' (line 721)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 8), 'source_ext', unicode_287462)
    # SSA join for if statement (line 718)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 724):
    
    # Assigning a Call to a Name (line 724):
    
    # Call to replace(...): (line 724)
    # Processing the call arguments (line 724)
    unicode_287465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 38), 'unicode', u'.')
    unicode_287466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 43), 'unicode', u'-')
    # Processing the call keyword arguments (line 724)
    kwargs_287467 = {}
    # Getting the type of 'output_base' (line 724)
    output_base_287463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 18), 'output_base', False)
    # Obtaining the member 'replace' of a type (line 724)
    replace_287464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 18), output_base_287463, 'replace')
    # Calling replace(args, kwargs) (line 724)
    replace_call_result_287468 = invoke(stypy.reporting.localization.Localization(__file__, 724, 18), replace_287464, *[unicode_287465, unicode_287466], **kwargs_287467)
    
    # Assigning a type to the variable 'output_base' (line 724)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 4), 'output_base', replace_call_result_287468)
    
    # Assigning a Call to a Name (line 727):
    
    # Assigning a Call to a Name (line 727):
    
    # Call to contains_doctest(...): (line 727)
    # Processing the call arguments (line 727)
    # Getting the type of 'code' (line 727)
    code_287470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 34), 'code', False)
    # Processing the call keyword arguments (line 727)
    kwargs_287471 = {}
    # Getting the type of 'contains_doctest' (line 727)
    contains_doctest_287469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 17), 'contains_doctest', False)
    # Calling contains_doctest(args, kwargs) (line 727)
    contains_doctest_call_result_287472 = invoke(stypy.reporting.localization.Localization(__file__, 727, 17), contains_doctest_287469, *[code_287470], **kwargs_287471)
    
    # Assigning a type to the variable 'is_doctest' (line 727)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 4), 'is_doctest', contains_doctest_call_result_287472)
    
    
    unicode_287473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 7), 'unicode', u'format')
    # Getting the type of 'options' (line 728)
    options_287474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 19), 'options')
    # Applying the binary operator 'in' (line 728)
    result_contains_287475 = python_operator(stypy.reporting.localization.Localization(__file__, 728, 7), 'in', unicode_287473, options_287474)
    
    # Testing the type of an if condition (line 728)
    if_condition_287476 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 728, 4), result_contains_287475)
    # Assigning a type to the variable 'if_condition_287476' (line 728)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 4), 'if_condition_287476', if_condition_287476)
    # SSA begins for if statement (line 728)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    unicode_287477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 19), 'unicode', u'format')
    # Getting the type of 'options' (line 729)
    options_287478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 11), 'options')
    # Obtaining the member '__getitem__' of a type (line 729)
    getitem___287479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 11), options_287478, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 729)
    subscript_call_result_287480 = invoke(stypy.reporting.localization.Localization(__file__, 729, 11), getitem___287479, unicode_287477)
    
    unicode_287481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 32), 'unicode', u'python')
    # Applying the binary operator '==' (line 729)
    result_eq_287482 = python_operator(stypy.reporting.localization.Localization(__file__, 729, 11), '==', subscript_call_result_287480, unicode_287481)
    
    # Testing the type of an if condition (line 729)
    if_condition_287483 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 729, 8), result_eq_287482)
    # Assigning a type to the variable 'if_condition_287483' (line 729)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 8), 'if_condition_287483', if_condition_287483)
    # SSA begins for if statement (line 729)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 730):
    
    # Assigning a Name to a Name (line 730):
    # Getting the type of 'False' (line 730)
    False_287484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 25), 'False')
    # Assigning a type to the variable 'is_doctest' (line 730)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 12), 'is_doctest', False_287484)
    # SSA branch for the else part of an if statement (line 729)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 732):
    
    # Assigning a Name to a Name (line 732):
    # Getting the type of 'True' (line 732)
    True_287485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 25), 'True')
    # Assigning a type to the variable 'is_doctest' (line 732)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 12), 'is_doctest', True_287485)
    # SSA join for if statement (line 729)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 728)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 735):
    
    # Assigning a Call to a Name (line 735):
    
    # Call to relpath(...): (line 735)
    # Processing the call arguments (line 735)
    # Getting the type of 'source_file_name' (line 735)
    source_file_name_287487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 30), 'source_file_name', False)
    # Getting the type of 'setup' (line 735)
    setup_287488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 48), 'setup', False)
    # Obtaining the member 'confdir' of a type (line 735)
    confdir_287489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 48), setup_287488, 'confdir')
    # Processing the call keyword arguments (line 735)
    kwargs_287490 = {}
    # Getting the type of 'relpath' (line 735)
    relpath_287486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 22), 'relpath', False)
    # Calling relpath(args, kwargs) (line 735)
    relpath_call_result_287491 = invoke(stypy.reporting.localization.Localization(__file__, 735, 22), relpath_287486, *[source_file_name_287487, confdir_287489], **kwargs_287490)
    
    # Assigning a type to the variable 'source_rel_name' (line 735)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'source_rel_name', relpath_call_result_287491)
    
    # Assigning a Call to a Name (line 736):
    
    # Assigning a Call to a Name (line 736):
    
    # Call to dirname(...): (line 736)
    # Processing the call arguments (line 736)
    # Getting the type of 'source_rel_name' (line 736)
    source_rel_name_287495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 37), 'source_rel_name', False)
    # Processing the call keyword arguments (line 736)
    kwargs_287496 = {}
    # Getting the type of 'os' (line 736)
    os_287492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 21), 'os', False)
    # Obtaining the member 'path' of a type (line 736)
    path_287493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 21), os_287492, 'path')
    # Obtaining the member 'dirname' of a type (line 736)
    dirname_287494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 21), path_287493, 'dirname')
    # Calling dirname(args, kwargs) (line 736)
    dirname_call_result_287497 = invoke(stypy.reporting.localization.Localization(__file__, 736, 21), dirname_287494, *[source_rel_name_287495], **kwargs_287496)
    
    # Assigning a type to the variable 'source_rel_dir' (line 736)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 4), 'source_rel_dir', dirname_call_result_287497)
    
    
    # Call to startswith(...): (line 737)
    # Processing the call arguments (line 737)
    # Getting the type of 'os' (line 737)
    os_287500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 36), 'os', False)
    # Obtaining the member 'path' of a type (line 737)
    path_287501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 36), os_287500, 'path')
    # Obtaining the member 'sep' of a type (line 737)
    sep_287502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 36), path_287501, 'sep')
    # Processing the call keyword arguments (line 737)
    kwargs_287503 = {}
    # Getting the type of 'source_rel_dir' (line 737)
    source_rel_dir_287498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 10), 'source_rel_dir', False)
    # Obtaining the member 'startswith' of a type (line 737)
    startswith_287499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 10), source_rel_dir_287498, 'startswith')
    # Calling startswith(args, kwargs) (line 737)
    startswith_call_result_287504 = invoke(stypy.reporting.localization.Localization(__file__, 737, 10), startswith_287499, *[sep_287502], **kwargs_287503)
    
    # Testing the type of an if condition (line 737)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 737, 4), startswith_call_result_287504)
    # SSA begins for while statement (line 737)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Subscript to a Name (line 738):
    
    # Assigning a Subscript to a Name (line 738):
    
    # Obtaining the type of the subscript
    int_287505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 40), 'int')
    slice_287506 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 738, 25), int_287505, None, None)
    # Getting the type of 'source_rel_dir' (line 738)
    source_rel_dir_287507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 25), 'source_rel_dir')
    # Obtaining the member '__getitem__' of a type (line 738)
    getitem___287508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 25), source_rel_dir_287507, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 738)
    subscript_call_result_287509 = invoke(stypy.reporting.localization.Localization(__file__, 738, 25), getitem___287508, slice_287506)
    
    # Assigning a type to the variable 'source_rel_dir' (line 738)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 8), 'source_rel_dir', subscript_call_result_287509)
    # SSA join for while statement (line 737)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 741):
    
    # Assigning a Call to a Name (line 741):
    
    # Call to join(...): (line 741)
    # Processing the call arguments (line 741)
    
    # Call to dirname(...): (line 741)
    # Processing the call arguments (line 741)
    # Getting the type of 'setup' (line 741)
    setup_287516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 45), 'setup', False)
    # Obtaining the member 'app' of a type (line 741)
    app_287517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 45), setup_287516, 'app')
    # Obtaining the member 'doctreedir' of a type (line 741)
    doctreedir_287518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 45), app_287517, 'doctreedir')
    # Processing the call keyword arguments (line 741)
    kwargs_287519 = {}
    # Getting the type of 'os' (line 741)
    os_287513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 29), 'os', False)
    # Obtaining the member 'path' of a type (line 741)
    path_287514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 29), os_287513, 'path')
    # Obtaining the member 'dirname' of a type (line 741)
    dirname_287515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 29), path_287514, 'dirname')
    # Calling dirname(args, kwargs) (line 741)
    dirname_call_result_287520 = invoke(stypy.reporting.localization.Localization(__file__, 741, 29), dirname_287515, *[doctreedir_287518], **kwargs_287519)
    
    unicode_287521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 29), 'unicode', u'plot_directive')
    # Getting the type of 'source_rel_dir' (line 743)
    source_rel_dir_287522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 29), 'source_rel_dir', False)
    # Processing the call keyword arguments (line 741)
    kwargs_287523 = {}
    # Getting the type of 'os' (line 741)
    os_287510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 16), 'os', False)
    # Obtaining the member 'path' of a type (line 741)
    path_287511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 16), os_287510, 'path')
    # Obtaining the member 'join' of a type (line 741)
    join_287512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 16), path_287511, 'join')
    # Calling join(args, kwargs) (line 741)
    join_call_result_287524 = invoke(stypy.reporting.localization.Localization(__file__, 741, 16), join_287512, *[dirname_call_result_287520, unicode_287521, source_rel_dir_287522], **kwargs_287523)
    
    # Assigning a type to the variable 'build_dir' (line 741)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 4), 'build_dir', join_call_result_287524)
    
    # Assigning a Call to a Name (line 747):
    
    # Assigning a Call to a Name (line 747):
    
    # Call to normpath(...): (line 747)
    # Processing the call arguments (line 747)
    # Getting the type of 'build_dir' (line 747)
    build_dir_287528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 33), 'build_dir', False)
    # Processing the call keyword arguments (line 747)
    kwargs_287529 = {}
    # Getting the type of 'os' (line 747)
    os_287525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 16), 'os', False)
    # Obtaining the member 'path' of a type (line 747)
    path_287526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 16), os_287525, 'path')
    # Obtaining the member 'normpath' of a type (line 747)
    normpath_287527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 16), path_287526, 'normpath')
    # Calling normpath(args, kwargs) (line 747)
    normpath_call_result_287530 = invoke(stypy.reporting.localization.Localization(__file__, 747, 16), normpath_287527, *[build_dir_287528], **kwargs_287529)
    
    # Assigning a type to the variable 'build_dir' (line 747)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 4), 'build_dir', normpath_call_result_287530)
    
    
    
    # Call to exists(...): (line 749)
    # Processing the call arguments (line 749)
    # Getting the type of 'build_dir' (line 749)
    build_dir_287534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 26), 'build_dir', False)
    # Processing the call keyword arguments (line 749)
    kwargs_287535 = {}
    # Getting the type of 'os' (line 749)
    os_287531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 749)
    path_287532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 749, 11), os_287531, 'path')
    # Obtaining the member 'exists' of a type (line 749)
    exists_287533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 749, 11), path_287532, 'exists')
    # Calling exists(args, kwargs) (line 749)
    exists_call_result_287536 = invoke(stypy.reporting.localization.Localization(__file__, 749, 11), exists_287533, *[build_dir_287534], **kwargs_287535)
    
    # Applying the 'not' unary operator (line 749)
    result_not__287537 = python_operator(stypy.reporting.localization.Localization(__file__, 749, 7), 'not', exists_call_result_287536)
    
    # Testing the type of an if condition (line 749)
    if_condition_287538 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 749, 4), result_not__287537)
    # Assigning a type to the variable 'if_condition_287538' (line 749)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 4), 'if_condition_287538', if_condition_287538)
    # SSA begins for if statement (line 749)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to makedirs(...): (line 750)
    # Processing the call arguments (line 750)
    # Getting the type of 'build_dir' (line 750)
    build_dir_287541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 20), 'build_dir', False)
    # Processing the call keyword arguments (line 750)
    kwargs_287542 = {}
    # Getting the type of 'os' (line 750)
    os_287539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 8), 'os', False)
    # Obtaining the member 'makedirs' of a type (line 750)
    makedirs_287540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 8), os_287539, 'makedirs')
    # Calling makedirs(args, kwargs) (line 750)
    makedirs_call_result_287543 = invoke(stypy.reporting.localization.Localization(__file__, 750, 8), makedirs_287540, *[build_dir_287541], **kwargs_287542)
    
    # SSA join for if statement (line 749)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 753):
    
    # Assigning a Call to a Name (line 753):
    
    # Call to abspath(...): (line 753)
    # Processing the call arguments (line 753)
    
    # Call to join(...): (line 753)
    # Processing the call arguments (line 753)
    # Getting the type of 'setup' (line 753)
    setup_287550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 44), 'setup', False)
    # Obtaining the member 'app' of a type (line 753)
    app_287551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 753, 44), setup_287550, 'app')
    # Obtaining the member 'builder' of a type (line 753)
    builder_287552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 753, 44), app_287551, 'builder')
    # Obtaining the member 'outdir' of a type (line 753)
    outdir_287553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 753, 44), builder_287552, 'outdir')
    # Getting the type of 'source_rel_dir' (line 754)
    source_rel_dir_287554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 44), 'source_rel_dir', False)
    # Processing the call keyword arguments (line 753)
    kwargs_287555 = {}
    # Getting the type of 'os' (line 753)
    os_287547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 31), 'os', False)
    # Obtaining the member 'path' of a type (line 753)
    path_287548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 753, 31), os_287547, 'path')
    # Obtaining the member 'join' of a type (line 753)
    join_287549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 753, 31), path_287548, 'join')
    # Calling join(args, kwargs) (line 753)
    join_call_result_287556 = invoke(stypy.reporting.localization.Localization(__file__, 753, 31), join_287549, *[outdir_287553, source_rel_dir_287554], **kwargs_287555)
    
    # Processing the call keyword arguments (line 753)
    kwargs_287557 = {}
    # Getting the type of 'os' (line 753)
    os_287544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 753)
    path_287545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 753, 15), os_287544, 'path')
    # Obtaining the member 'abspath' of a type (line 753)
    abspath_287546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 753, 15), path_287545, 'abspath')
    # Calling abspath(args, kwargs) (line 753)
    abspath_call_result_287558 = invoke(stypy.reporting.localization.Localization(__file__, 753, 15), abspath_287546, *[join_call_result_287556], **kwargs_287557)
    
    # Assigning a type to the variable 'dest_dir' (line 753)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 753, 4), 'dest_dir', abspath_call_result_287558)
    
    
    
    # Call to exists(...): (line 755)
    # Processing the call arguments (line 755)
    # Getting the type of 'dest_dir' (line 755)
    dest_dir_287562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 26), 'dest_dir', False)
    # Processing the call keyword arguments (line 755)
    kwargs_287563 = {}
    # Getting the type of 'os' (line 755)
    os_287559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 755)
    path_287560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 11), os_287559, 'path')
    # Obtaining the member 'exists' of a type (line 755)
    exists_287561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 11), path_287560, 'exists')
    # Calling exists(args, kwargs) (line 755)
    exists_call_result_287564 = invoke(stypy.reporting.localization.Localization(__file__, 755, 11), exists_287561, *[dest_dir_287562], **kwargs_287563)
    
    # Applying the 'not' unary operator (line 755)
    result_not__287565 = python_operator(stypy.reporting.localization.Localization(__file__, 755, 7), 'not', exists_call_result_287564)
    
    # Testing the type of an if condition (line 755)
    if_condition_287566 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 755, 4), result_not__287565)
    # Assigning a type to the variable 'if_condition_287566' (line 755)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 4), 'if_condition_287566', if_condition_287566)
    # SSA begins for if statement (line 755)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to makedirs(...): (line 756)
    # Processing the call arguments (line 756)
    # Getting the type of 'dest_dir' (line 756)
    dest_dir_287569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 20), 'dest_dir', False)
    # Processing the call keyword arguments (line 756)
    kwargs_287570 = {}
    # Getting the type of 'os' (line 756)
    os_287567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 8), 'os', False)
    # Obtaining the member 'makedirs' of a type (line 756)
    makedirs_287568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 8), os_287567, 'makedirs')
    # Calling makedirs(args, kwargs) (line 756)
    makedirs_call_result_287571 = invoke(stypy.reporting.localization.Localization(__file__, 756, 8), makedirs_287568, *[dest_dir_287569], **kwargs_287570)
    
    # SSA join for if statement (line 755)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 759):
    
    # Assigning a Call to a Name (line 759):
    
    # Call to replace(...): (line 759)
    # Processing the call arguments (line 759)
    # Getting the type of 'os' (line 760)
    os_287585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 57), 'os', False)
    # Obtaining the member 'path' of a type (line 760)
    path_287586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 57), os_287585, 'path')
    # Obtaining the member 'sep' of a type (line 760)
    sep_287587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 57), path_287586, 'sep')
    unicode_287588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 70), 'unicode', u'/')
    # Processing the call keyword arguments (line 759)
    kwargs_287589 = {}
    
    # Call to join(...): (line 759)
    # Processing the call arguments (line 759)
    
    # Call to relpath(...): (line 759)
    # Processing the call arguments (line 759)
    # Getting the type of 'setup' (line 759)
    setup_287576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 41), 'setup', False)
    # Obtaining the member 'confdir' of a type (line 759)
    confdir_287577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 41), setup_287576, 'confdir')
    # Getting the type of 'rst_dir' (line 759)
    rst_dir_287578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 56), 'rst_dir', False)
    # Processing the call keyword arguments (line 759)
    kwargs_287579 = {}
    # Getting the type of 'relpath' (line 759)
    relpath_287575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 33), 'relpath', False)
    # Calling relpath(args, kwargs) (line 759)
    relpath_call_result_287580 = invoke(stypy.reporting.localization.Localization(__file__, 759, 33), relpath_287575, *[confdir_287577, rst_dir_287578], **kwargs_287579)
    
    # Getting the type of 'source_rel_dir' (line 760)
    source_rel_dir_287581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 33), 'source_rel_dir', False)
    # Processing the call keyword arguments (line 759)
    kwargs_287582 = {}
    # Getting the type of 'os' (line 759)
    os_287572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 20), 'os', False)
    # Obtaining the member 'path' of a type (line 759)
    path_287573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 20), os_287572, 'path')
    # Obtaining the member 'join' of a type (line 759)
    join_287574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 20), path_287573, 'join')
    # Calling join(args, kwargs) (line 759)
    join_call_result_287583 = invoke(stypy.reporting.localization.Localization(__file__, 759, 20), join_287574, *[relpath_call_result_287580, source_rel_dir_287581], **kwargs_287582)
    
    # Obtaining the member 'replace' of a type (line 759)
    replace_287584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 20), join_call_result_287583, 'replace')
    # Calling replace(args, kwargs) (line 759)
    replace_call_result_287590 = invoke(stypy.reporting.localization.Localization(__file__, 759, 20), replace_287584, *[sep_287587, unicode_287588], **kwargs_287589)
    
    # Assigning a type to the variable 'dest_dir_link' (line 759)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 4), 'dest_dir_link', replace_call_result_287590)
    
    
    # SSA begins for try-except statement (line 761)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 762):
    
    # Assigning a Call to a Name (line 762):
    
    # Call to replace(...): (line 762)
    # Processing the call arguments (line 762)
    # Getting the type of 'os' (line 762)
    os_287597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 61), 'os', False)
    # Obtaining the member 'path' of a type (line 762)
    path_287598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 61), os_287597, 'path')
    # Obtaining the member 'sep' of a type (line 762)
    sep_287599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 61), path_287598, 'sep')
    unicode_287600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 74), 'unicode', u'/')
    # Processing the call keyword arguments (line 762)
    kwargs_287601 = {}
    
    # Call to relpath(...): (line 762)
    # Processing the call arguments (line 762)
    # Getting the type of 'build_dir' (line 762)
    build_dir_287592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 33), 'build_dir', False)
    # Getting the type of 'rst_dir' (line 762)
    rst_dir_287593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 44), 'rst_dir', False)
    # Processing the call keyword arguments (line 762)
    kwargs_287594 = {}
    # Getting the type of 'relpath' (line 762)
    relpath_287591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 25), 'relpath', False)
    # Calling relpath(args, kwargs) (line 762)
    relpath_call_result_287595 = invoke(stypy.reporting.localization.Localization(__file__, 762, 25), relpath_287591, *[build_dir_287592, rst_dir_287593], **kwargs_287594)
    
    # Obtaining the member 'replace' of a type (line 762)
    replace_287596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 25), relpath_call_result_287595, 'replace')
    # Calling replace(args, kwargs) (line 762)
    replace_call_result_287602 = invoke(stypy.reporting.localization.Localization(__file__, 762, 25), replace_287596, *[sep_287599, unicode_287600], **kwargs_287601)
    
    # Assigning a type to the variable 'build_dir_link' (line 762)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 8), 'build_dir_link', replace_call_result_287602)
    # SSA branch for the except part of a try statement (line 761)
    # SSA branch for the except 'ValueError' branch of a try statement (line 761)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 766):
    
    # Assigning a Name to a Name (line 766):
    # Getting the type of 'build_dir' (line 766)
    build_dir_287603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 25), 'build_dir')
    # Assigning a type to the variable 'build_dir_link' (line 766)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 8), 'build_dir_link', build_dir_287603)
    # SSA join for try-except statement (line 761)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 767):
    
    # Assigning a BinOp to a Name (line 767):
    # Getting the type of 'dest_dir_link' (line 767)
    dest_dir_link_287604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 18), 'dest_dir_link')
    unicode_287605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 34), 'unicode', u'/')
    # Applying the binary operator '+' (line 767)
    result_add_287606 = python_operator(stypy.reporting.localization.Localization(__file__, 767, 18), '+', dest_dir_link_287604, unicode_287605)
    
    # Getting the type of 'output_base' (line 767)
    output_base_287607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 40), 'output_base')
    # Applying the binary operator '+' (line 767)
    result_add_287608 = python_operator(stypy.reporting.localization.Localization(__file__, 767, 38), '+', result_add_287606, output_base_287607)
    
    # Getting the type of 'source_ext' (line 767)
    source_ext_287609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 54), 'source_ext')
    # Applying the binary operator '+' (line 767)
    result_add_287610 = python_operator(stypy.reporting.localization.Localization(__file__, 767, 52), '+', result_add_287608, source_ext_287609)
    
    # Assigning a type to the variable 'source_link' (line 767)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 4), 'source_link', result_add_287610)
    
    
    # SSA begins for try-except statement (line 770)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 771):
    
    # Assigning a Call to a Name (line 771):
    
    # Call to render_figures(...): (line 771)
    # Processing the call arguments (line 771)
    # Getting the type of 'code' (line 771)
    code_287612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 33), 'code', False)
    # Getting the type of 'source_file_name' (line 772)
    source_file_name_287613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 33), 'source_file_name', False)
    # Getting the type of 'build_dir' (line 773)
    build_dir_287614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 33), 'build_dir', False)
    # Getting the type of 'output_base' (line 774)
    output_base_287615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 33), 'output_base', False)
    # Getting the type of 'keep_context' (line 775)
    keep_context_287616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 33), 'keep_context', False)
    # Getting the type of 'function_name' (line 776)
    function_name_287617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 33), 'function_name', False)
    # Getting the type of 'config' (line 777)
    config_287618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 33), 'config', False)
    # Processing the call keyword arguments (line 771)
    
    # Getting the type of 'context_opt' (line 778)
    context_opt_287619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 47), 'context_opt', False)
    unicode_287620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 62), 'unicode', u'reset')
    # Applying the binary operator '==' (line 778)
    result_eq_287621 = python_operator(stypy.reporting.localization.Localization(__file__, 778, 47), '==', context_opt_287619, unicode_287620)
    
    keyword_287622 = result_eq_287621
    
    # Getting the type of 'context_opt' (line 779)
    context_opt_287623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 44), 'context_opt', False)
    unicode_287624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 59), 'unicode', u'close-figs')
    # Applying the binary operator '==' (line 779)
    result_eq_287625 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 44), '==', context_opt_287623, unicode_287624)
    
    keyword_287626 = result_eq_287625
    kwargs_287627 = {'close_figs': keyword_287626, 'context_reset': keyword_287622}
    # Getting the type of 'render_figures' (line 771)
    render_figures_287611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 18), 'render_figures', False)
    # Calling render_figures(args, kwargs) (line 771)
    render_figures_call_result_287628 = invoke(stypy.reporting.localization.Localization(__file__, 771, 18), render_figures_287611, *[code_287612, source_file_name_287613, build_dir_287614, output_base_287615, keep_context_287616, function_name_287617, config_287618], **kwargs_287627)
    
    # Assigning a type to the variable 'results' (line 771)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 8), 'results', render_figures_call_result_287628)
    
    # Assigning a List to a Name (line 780):
    
    # Assigning a List to a Name (line 780):
    
    # Obtaining an instance of the builtin type 'list' (line 780)
    list_287629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 780)
    
    # Assigning a type to the variable 'errors' (line 780)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 8), 'errors', list_287629)
    # SSA branch for the except part of a try statement (line 770)
    # SSA branch for the except 'PlotError' branch of a try statement (line 770)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'PlotError' (line 781)
    PlotError_287630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 11), 'PlotError')
    # Assigning a type to the variable 'err' (line 781)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 781, 4), 'err', PlotError_287630)
    
    # Assigning a Attribute to a Name (line 782):
    
    # Assigning a Attribute to a Name (line 782):
    # Getting the type of 'state' (line 782)
    state_287631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 19), 'state')
    # Obtaining the member 'memo' of a type (line 782)
    memo_287632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 19), state_287631, 'memo')
    # Obtaining the member 'reporter' of a type (line 782)
    reporter_287633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 19), memo_287632, 'reporter')
    # Assigning a type to the variable 'reporter' (line 782)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 782, 8), 'reporter', reporter_287633)
    
    # Assigning a Call to a Name (line 783):
    
    # Assigning a Call to a Name (line 783):
    
    # Call to system_message(...): (line 783)
    # Processing the call arguments (line 783)
    int_287636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 12), 'int')
    unicode_287637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 15), 'unicode', u'Exception occurred in plotting %s\n from %s:\n%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 784)
    tuple_287638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 69), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 784)
    # Adding element type (line 784)
    # Getting the type of 'output_base' (line 784)
    output_base_287639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 69), 'output_base', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 784, 69), tuple_287638, output_base_287639)
    # Adding element type (line 784)
    # Getting the type of 'source_file_name' (line 785)
    source_file_name_287640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 48), 'source_file_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 784, 69), tuple_287638, source_file_name_287640)
    # Adding element type (line 784)
    # Getting the type of 'err' (line 785)
    err_287641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 66), 'err', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 784, 69), tuple_287638, err_287641)
    
    # Applying the binary operator '%' (line 784)
    result_mod_287642 = python_operator(stypy.reporting.localization.Localization(__file__, 784, 15), '%', unicode_287637, tuple_287638)
    
    # Processing the call keyword arguments (line 783)
    # Getting the type of 'lineno' (line 786)
    lineno_287643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 17), 'lineno', False)
    keyword_287644 = lineno_287643
    kwargs_287645 = {'line': keyword_287644}
    # Getting the type of 'reporter' (line 783)
    reporter_287634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 13), 'reporter', False)
    # Obtaining the member 'system_message' of a type (line 783)
    system_message_287635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 13), reporter_287634, 'system_message')
    # Calling system_message(args, kwargs) (line 783)
    system_message_call_result_287646 = invoke(stypy.reporting.localization.Localization(__file__, 783, 13), system_message_287635, *[int_287636, result_mod_287642], **kwargs_287645)
    
    # Assigning a type to the variable 'sm' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 8), 'sm', system_message_call_result_287646)
    
    # Assigning a List to a Name (line 787):
    
    # Assigning a List to a Name (line 787):
    
    # Obtaining an instance of the builtin type 'list' (line 787)
    list_287647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 787)
    # Adding element type (line 787)
    
    # Obtaining an instance of the builtin type 'tuple' (line 787)
    tuple_287648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 787)
    # Adding element type (line 787)
    # Getting the type of 'code' (line 787)
    code_287649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 20), 'code')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 787, 20), tuple_287648, code_287649)
    # Adding element type (line 787)
    
    # Obtaining an instance of the builtin type 'list' (line 787)
    list_287650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 787)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 787, 20), tuple_287648, list_287650)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 787, 18), list_287647, tuple_287648)
    
    # Assigning a type to the variable 'results' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 8), 'results', list_287647)
    
    # Assigning a List to a Name (line 788):
    
    # Assigning a List to a Name (line 788):
    
    # Obtaining an instance of the builtin type 'list' (line 788)
    list_287651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 788)
    # Adding element type (line 788)
    # Getting the type of 'sm' (line 788)
    sm_287652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 18), 'sm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 788, 17), list_287651, sm_287652)
    
    # Assigning a type to the variable 'errors' (line 788)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 8), 'errors', list_287651)
    # SSA join for try-except statement (line 770)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 791):
    
    # Assigning a Call to a Name (line 791):
    
    # Call to join(...): (line 791)
    # Processing the call arguments (line 791)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 791, 24, True)
    # Calculating comprehension expression
    
    # Call to split(...): (line 792)
    # Processing the call arguments (line 792)
    unicode_287663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 50), 'unicode', u'\n')
    # Processing the call keyword arguments (line 792)
    kwargs_287664 = {}
    # Getting the type of 'caption' (line 792)
    caption_287661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 36), 'caption', False)
    # Obtaining the member 'split' of a type (line 792)
    split_287662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 36), caption_287661, 'split')
    # Calling split(args, kwargs) (line 792)
    split_call_result_287665 = invoke(stypy.reporting.localization.Localization(__file__, 792, 36), split_287662, *[unicode_287663], **kwargs_287664)
    
    comprehension_287666 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 791, 24), split_call_result_287665)
    # Assigning a type to the variable 'line' (line 791)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 791, 24), 'line', comprehension_287666)
    unicode_287655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, 24), 'unicode', u'      ')
    
    # Call to strip(...): (line 791)
    # Processing the call keyword arguments (line 791)
    kwargs_287658 = {}
    # Getting the type of 'line' (line 791)
    line_287656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 35), 'line', False)
    # Obtaining the member 'strip' of a type (line 791)
    strip_287657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 35), line_287656, 'strip')
    # Calling strip(args, kwargs) (line 791)
    strip_call_result_287659 = invoke(stypy.reporting.localization.Localization(__file__, 791, 35), strip_287657, *[], **kwargs_287658)
    
    # Applying the binary operator '+' (line 791)
    result_add_287660 = python_operator(stypy.reporting.localization.Localization(__file__, 791, 24), '+', unicode_287655, strip_call_result_287659)
    
    list_287667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, 24), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 791, 24), list_287667, result_add_287660)
    # Processing the call keyword arguments (line 791)
    kwargs_287668 = {}
    unicode_287653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, 14), 'unicode', u'\n')
    # Obtaining the member 'join' of a type (line 791)
    join_287654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 14), unicode_287653, 'join')
    # Calling join(args, kwargs) (line 791)
    join_call_result_287669 = invoke(stypy.reporting.localization.Localization(__file__, 791, 14), join_287654, *[list_287667], **kwargs_287668)
    
    # Assigning a type to the variable 'caption' (line 791)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 791, 4), 'caption', join_call_result_287669)
    
    # Assigning a List to a Name (line 795):
    
    # Assigning a List to a Name (line 795):
    
    # Obtaining an instance of the builtin type 'list' (line 795)
    list_287670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 795)
    
    # Assigning a type to the variable 'total_lines' (line 795)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 795, 4), 'total_lines', list_287670)
    
    
    # Call to enumerate(...): (line 796)
    # Processing the call arguments (line 796)
    # Getting the type of 'results' (line 796)
    results_287672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 45), 'results', False)
    # Processing the call keyword arguments (line 796)
    kwargs_287673 = {}
    # Getting the type of 'enumerate' (line 796)
    enumerate_287671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 35), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 796)
    enumerate_call_result_287674 = invoke(stypy.reporting.localization.Localization(__file__, 796, 35), enumerate_287671, *[results_287672], **kwargs_287673)
    
    # Testing the type of a for loop iterable (line 796)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 796, 4), enumerate_call_result_287674)
    # Getting the type of the for loop variable (line 796)
    for_loop_var_287675 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 796, 4), enumerate_call_result_287674)
    # Assigning a type to the variable 'j' (line 796)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 796, 4), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 796, 4), for_loop_var_287675))
    # Assigning a type to the variable 'code_piece' (line 796)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 796, 4), 'code_piece', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 796, 4), for_loop_var_287675))
    # Assigning a type to the variable 'images' (line 796)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 796, 4), 'images', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 796, 4), for_loop_var_287675))
    # SSA begins for a for statement (line 796)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining the type of the subscript
    unicode_287676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 797, 19), 'unicode', u'include-source')
    # Getting the type of 'options' (line 797)
    options_287677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 11), 'options')
    # Obtaining the member '__getitem__' of a type (line 797)
    getitem___287678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 11), options_287677, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 797)
    subscript_call_result_287679 = invoke(stypy.reporting.localization.Localization(__file__, 797, 11), getitem___287678, unicode_287676)
    
    # Testing the type of an if condition (line 797)
    if_condition_287680 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 797, 8), subscript_call_result_287679)
    # Assigning a type to the variable 'if_condition_287680' (line 797)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 8), 'if_condition_287680', if_condition_287680)
    # SSA begins for if statement (line 797)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'is_doctest' (line 798)
    is_doctest_287681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 15), 'is_doctest')
    # Testing the type of an if condition (line 798)
    if_condition_287682 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 798, 12), is_doctest_287681)
    # Assigning a type to the variable 'if_condition_287682' (line 798)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 798, 12), 'if_condition_287682', if_condition_287682)
    # SSA begins for if statement (line 798)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 799):
    
    # Assigning a List to a Name (line 799):
    
    # Obtaining an instance of the builtin type 'list' (line 799)
    list_287683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 799, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 799)
    # Adding element type (line 799)
    unicode_287684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 799, 25), 'unicode', u'')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 799, 24), list_287683, unicode_287684)
    
    # Assigning a type to the variable 'lines' (line 799)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 799, 16), 'lines', list_287683)
    
    # Getting the type of 'lines' (line 800)
    lines_287685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 16), 'lines')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to split(...): (line 800)
    # Processing the call arguments (line 800)
    unicode_287692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 67), 'unicode', u'\n')
    # Processing the call keyword arguments (line 800)
    kwargs_287693 = {}
    # Getting the type of 'code_piece' (line 800)
    code_piece_287690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 50), 'code_piece', False)
    # Obtaining the member 'split' of a type (line 800)
    split_287691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 800, 50), code_piece_287690, 'split')
    # Calling split(args, kwargs) (line 800)
    split_call_result_287694 = invoke(stypy.reporting.localization.Localization(__file__, 800, 50), split_287691, *[unicode_287692], **kwargs_287693)
    
    comprehension_287695 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 800, 26), split_call_result_287694)
    # Assigning a type to the variable 'row' (line 800)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 800, 26), 'row', comprehension_287695)
    
    # Call to rstrip(...): (line 800)
    # Processing the call keyword arguments (line 800)
    kwargs_287688 = {}
    # Getting the type of 'row' (line 800)
    row_287686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 26), 'row', False)
    # Obtaining the member 'rstrip' of a type (line 800)
    rstrip_287687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 800, 26), row_287686, 'rstrip')
    # Calling rstrip(args, kwargs) (line 800)
    rstrip_call_result_287689 = invoke(stypy.reporting.localization.Localization(__file__, 800, 26), rstrip_287687, *[], **kwargs_287688)
    
    list_287696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 26), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 800, 26), list_287696, rstrip_call_result_287689)
    # Applying the binary operator '+=' (line 800)
    result_iadd_287697 = python_operator(stypy.reporting.localization.Localization(__file__, 800, 16), '+=', lines_287685, list_287696)
    # Assigning a type to the variable 'lines' (line 800)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 800, 16), 'lines', result_iadd_287697)
    
    # SSA branch for the else part of an if statement (line 798)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a List to a Name (line 802):
    
    # Assigning a List to a Name (line 802):
    
    # Obtaining an instance of the builtin type 'list' (line 802)
    list_287698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 802, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 802)
    # Adding element type (line 802)
    unicode_287699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 802, 25), 'unicode', u'.. code-block:: python')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 802, 24), list_287698, unicode_287699)
    # Adding element type (line 802)
    unicode_287700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 802, 51), 'unicode', u'')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 802, 24), list_287698, unicode_287700)
    
    # Assigning a type to the variable 'lines' (line 802)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 802, 16), 'lines', list_287698)
    
    # Getting the type of 'lines' (line 803)
    lines_287701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 16), 'lines')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to split(...): (line 804)
    # Processing the call arguments (line 804)
    unicode_287710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 54), 'unicode', u'\n')
    # Processing the call keyword arguments (line 804)
    kwargs_287711 = {}
    # Getting the type of 'code_piece' (line 804)
    code_piece_287708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 37), 'code_piece', False)
    # Obtaining the member 'split' of a type (line 804)
    split_287709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 804, 37), code_piece_287708, 'split')
    # Calling split(args, kwargs) (line 804)
    split_call_result_287712 = invoke(stypy.reporting.localization.Localization(__file__, 804, 37), split_287709, *[unicode_287710], **kwargs_287711)
    
    comprehension_287713 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 803, 26), split_call_result_287712)
    # Assigning a type to the variable 'row' (line 803)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 803, 26), 'row', comprehension_287713)
    unicode_287702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 803, 26), 'unicode', u'    %s')
    
    # Call to rstrip(...): (line 803)
    # Processing the call keyword arguments (line 803)
    kwargs_287705 = {}
    # Getting the type of 'row' (line 803)
    row_287703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 37), 'row', False)
    # Obtaining the member 'rstrip' of a type (line 803)
    rstrip_287704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 803, 37), row_287703, 'rstrip')
    # Calling rstrip(args, kwargs) (line 803)
    rstrip_call_result_287706 = invoke(stypy.reporting.localization.Localization(__file__, 803, 37), rstrip_287704, *[], **kwargs_287705)
    
    # Applying the binary operator '%' (line 803)
    result_mod_287707 = python_operator(stypy.reporting.localization.Localization(__file__, 803, 26), '%', unicode_287702, rstrip_call_result_287706)
    
    list_287714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 803, 26), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 803, 26), list_287714, result_mod_287707)
    # Applying the binary operator '+=' (line 803)
    result_iadd_287715 = python_operator(stypy.reporting.localization.Localization(__file__, 803, 16), '+=', lines_287701, list_287714)
    # Assigning a type to the variable 'lines' (line 803)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 803, 16), 'lines', result_iadd_287715)
    
    # SSA join for if statement (line 798)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 805):
    
    # Assigning a Call to a Name (line 805):
    
    # Call to join(...): (line 805)
    # Processing the call arguments (line 805)
    # Getting the type of 'lines' (line 805)
    lines_287718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 36), 'lines', False)
    # Processing the call keyword arguments (line 805)
    kwargs_287719 = {}
    unicode_287716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 26), 'unicode', u'\n')
    # Obtaining the member 'join' of a type (line 805)
    join_287717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 26), unicode_287716, 'join')
    # Calling join(args, kwargs) (line 805)
    join_call_result_287720 = invoke(stypy.reporting.localization.Localization(__file__, 805, 26), join_287717, *[lines_287718], **kwargs_287719)
    
    # Assigning a type to the variable 'source_code' (line 805)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 805, 12), 'source_code', join_call_result_287720)
    # SSA branch for the else part of an if statement (line 797)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 807):
    
    # Assigning a Str to a Name (line 807):
    unicode_287721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 26), 'unicode', u'')
    # Assigning a type to the variable 'source_code' (line 807)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 807, 12), 'source_code', unicode_287721)
    # SSA join for if statement (line 797)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'nofigs' (line 809)
    nofigs_287722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 11), 'nofigs')
    # Testing the type of an if condition (line 809)
    if_condition_287723 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 809, 8), nofigs_287722)
    # Assigning a type to the variable 'if_condition_287723' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 8), 'if_condition_287723', if_condition_287723)
    # SSA begins for if statement (line 809)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 810):
    
    # Assigning a List to a Name (line 810):
    
    # Obtaining an instance of the builtin type 'list' (line 810)
    list_287724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 810)
    
    # Assigning a type to the variable 'images' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 12), 'images', list_287724)
    # SSA join for if statement (line 809)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Name (line 812):
    
    # Assigning a ListComp to a Name (line 812):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to iteritems(...): (line 812)
    # Processing the call arguments (line 812)
    # Getting the type of 'options' (line 812)
    options_287741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 69), 'options', False)
    # Processing the call keyword arguments (line 812)
    kwargs_287742 = {}
    # Getting the type of 'six' (line 812)
    six_287739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 55), 'six', False)
    # Obtaining the member 'iteritems' of a type (line 812)
    iteritems_287740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 55), six_287739, 'iteritems')
    # Calling iteritems(args, kwargs) (line 812)
    iteritems_call_result_287743 = invoke(stypy.reporting.localization.Localization(__file__, 812, 55), iteritems_287740, *[options_287741], **kwargs_287742)
    
    comprehension_287744 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 812, 16), iteritems_call_result_287743)
    # Assigning a type to the variable 'key' (line 812)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 16), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 812, 16), comprehension_287744))
    # Assigning a type to the variable 'val' (line 812)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 16), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 812, 16), comprehension_287744))
    
    # Getting the type of 'key' (line 813)
    key_287730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 19), 'key')
    
    # Obtaining an instance of the builtin type 'tuple' (line 813)
    tuple_287731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 813)
    # Adding element type (line 813)
    unicode_287732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 27), 'unicode', u'alt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 813, 27), tuple_287731, unicode_287732)
    # Adding element type (line 813)
    unicode_287733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 34), 'unicode', u'height')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 813, 27), tuple_287731, unicode_287733)
    # Adding element type (line 813)
    unicode_287734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 44), 'unicode', u'width')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 813, 27), tuple_287731, unicode_287734)
    # Adding element type (line 813)
    unicode_287735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 53), 'unicode', u'scale')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 813, 27), tuple_287731, unicode_287735)
    # Adding element type (line 813)
    unicode_287736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 62), 'unicode', u'align')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 813, 27), tuple_287731, unicode_287736)
    # Adding element type (line 813)
    unicode_287737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 71), 'unicode', u'class')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 813, 27), tuple_287731, unicode_287737)
    
    # Applying the binary operator 'in' (line 813)
    result_contains_287738 = python_operator(stypy.reporting.localization.Localization(__file__, 813, 19), 'in', key_287730, tuple_287731)
    
    unicode_287725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 812, 16), 'unicode', u':%s: %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 812)
    tuple_287726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 812, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 812)
    # Adding element type (line 812)
    # Getting the type of 'key' (line 812)
    key_287727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 29), 'key')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 812, 29), tuple_287726, key_287727)
    # Adding element type (line 812)
    # Getting the type of 'val' (line 812)
    val_287728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 34), 'val')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 812, 29), tuple_287726, val_287728)
    
    # Applying the binary operator '%' (line 812)
    result_mod_287729 = python_operator(stypy.reporting.localization.Localization(__file__, 812, 16), '%', unicode_287725, tuple_287726)
    
    list_287745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 812, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 812, 16), list_287745, result_mod_287729)
    # Assigning a type to the variable 'opts' (line 812)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 8), 'opts', list_287745)
    
    # Assigning a Str to a Name (line 815):
    
    # Assigning a Str to a Name (line 815):
    unicode_287746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 20), 'unicode', u'.. only:: html')
    # Assigning a type to the variable 'only_html' (line 815)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 8), 'only_html', unicode_287746)
    
    # Assigning a Str to a Name (line 816):
    
    # Assigning a Str to a Name (line 816):
    unicode_287747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 816, 21), 'unicode', u'.. only:: latex')
    # Assigning a type to the variable 'only_latex' (line 816)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 816, 8), 'only_latex', unicode_287747)
    
    # Assigning a Str to a Name (line 817):
    
    # Assigning a Str to a Name (line 817):
    unicode_287748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 23), 'unicode', u'.. only:: texinfo')
    # Assigning a type to the variable 'only_texinfo' (line 817)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 8), 'only_texinfo', unicode_287748)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'j' (line 821)
    j_287749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 11), 'j')
    int_287750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 16), 'int')
    # Applying the binary operator '==' (line 821)
    result_eq_287751 = python_operator(stypy.reporting.localization.Localization(__file__, 821, 11), '==', j_287749, int_287750)
    
    # Getting the type of 'config' (line 821)
    config_287752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 22), 'config')
    # Obtaining the member 'plot_html_show_source_link' of a type (line 821)
    plot_html_show_source_link_287753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 821, 22), config_287752, 'plot_html_show_source_link')
    # Applying the binary operator 'and' (line 821)
    result_and_keyword_287754 = python_operator(stypy.reporting.localization.Localization(__file__, 821, 11), 'and', result_eq_287751, plot_html_show_source_link_287753)
    
    # Testing the type of an if condition (line 821)
    if_condition_287755 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 821, 8), result_and_keyword_287754)
    # Assigning a type to the variable 'if_condition_287755' (line 821)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 8), 'if_condition_287755', if_condition_287755)
    # SSA begins for if statement (line 821)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 822):
    
    # Assigning a Name to a Name (line 822):
    # Getting the type of 'source_link' (line 822)
    source_link_287756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 23), 'source_link')
    # Assigning a type to the variable 'src_link' (line 822)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 822, 12), 'src_link', source_link_287756)
    # SSA branch for the else part of an if statement (line 821)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 824):
    
    # Assigning a Name to a Name (line 824):
    # Getting the type of 'None' (line 824)
    None_287757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 23), 'None')
    # Assigning a type to the variable 'src_link' (line 824)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 12), 'src_link', None_287757)
    # SSA join for if statement (line 821)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 826):
    
    # Assigning a Call to a Name (line 826):
    
    # Call to format_template(...): (line 826)
    # Processing the call arguments (line 826)
    
    # Evaluating a boolean operation
    # Getting the type of 'config' (line 827)
    config_287759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 12), 'config', False)
    # Obtaining the member 'plot_template' of a type (line 827)
    plot_template_287760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 827, 12), config_287759, 'plot_template')
    # Getting the type of 'TEMPLATE' (line 827)
    TEMPLATE_287761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 36), 'TEMPLATE', False)
    # Applying the binary operator 'or' (line 827)
    result_or_keyword_287762 = python_operator(stypy.reporting.localization.Localization(__file__, 827, 12), 'or', plot_template_287760, TEMPLATE_287761)
    
    # Processing the call keyword arguments (line 826)
    # Getting the type of 'default_fmt' (line 828)
    default_fmt_287763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 24), 'default_fmt', False)
    keyword_287764 = default_fmt_287763
    # Getting the type of 'dest_dir_link' (line 829)
    dest_dir_link_287765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 21), 'dest_dir_link', False)
    keyword_287766 = dest_dir_link_287765
    # Getting the type of 'build_dir_link' (line 830)
    build_dir_link_287767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 22), 'build_dir_link', False)
    keyword_287768 = build_dir_link_287767
    # Getting the type of 'src_link' (line 831)
    src_link_287769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 24), 'src_link', False)
    keyword_287770 = src_link_287769
    
    
    # Call to len(...): (line 832)
    # Processing the call arguments (line 832)
    # Getting the type of 'images' (line 832)
    images_287772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 28), 'images', False)
    # Processing the call keyword arguments (line 832)
    kwargs_287773 = {}
    # Getting the type of 'len' (line 832)
    len_287771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 24), 'len', False)
    # Calling len(args, kwargs) (line 832)
    len_call_result_287774 = invoke(stypy.reporting.localization.Localization(__file__, 832, 24), len_287771, *[images_287772], **kwargs_287773)
    
    int_287775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 832, 38), 'int')
    # Applying the binary operator '>' (line 832)
    result_gt_287776 = python_operator(stypy.reporting.localization.Localization(__file__, 832, 24), '>', len_call_result_287774, int_287775)
    
    keyword_287777 = result_gt_287776
    # Getting the type of 'only_html' (line 833)
    only_html_287778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 22), 'only_html', False)
    keyword_287779 = only_html_287778
    # Getting the type of 'only_latex' (line 834)
    only_latex_287780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 23), 'only_latex', False)
    keyword_287781 = only_latex_287780
    # Getting the type of 'only_texinfo' (line 835)
    only_texinfo_287782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 25), 'only_texinfo', False)
    keyword_287783 = only_texinfo_287782
    # Getting the type of 'opts' (line 836)
    opts_287784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 20), 'opts', False)
    keyword_287785 = opts_287784
    # Getting the type of 'images' (line 837)
    images_287786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 19), 'images', False)
    keyword_287787 = images_287786
    # Getting the type of 'source_code' (line 838)
    source_code_287788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 24), 'source_code', False)
    keyword_287789 = source_code_287788
    
    # Evaluating a boolean operation
    # Getting the type of 'config' (line 839)
    config_287790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 30), 'config', False)
    # Obtaining the member 'plot_html_show_formats' of a type (line 839)
    plot_html_show_formats_287791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 30), config_287790, 'plot_html_show_formats')
    
    # Call to len(...): (line 839)
    # Processing the call arguments (line 839)
    # Getting the type of 'images' (line 839)
    images_287793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 68), 'images', False)
    # Processing the call keyword arguments (line 839)
    kwargs_287794 = {}
    # Getting the type of 'len' (line 839)
    len_287792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 64), 'len', False)
    # Calling len(args, kwargs) (line 839)
    len_call_result_287795 = invoke(stypy.reporting.localization.Localization(__file__, 839, 64), len_287792, *[images_287793], **kwargs_287794)
    
    # Applying the binary operator 'and' (line 839)
    result_and_keyword_287796 = python_operator(stypy.reporting.localization.Localization(__file__, 839, 30), 'and', plot_html_show_formats_287791, len_call_result_287795)
    
    keyword_287797 = result_and_keyword_287796
    # Getting the type of 'caption' (line 840)
    caption_287798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 20), 'caption', False)
    keyword_287799 = caption_287798
    kwargs_287800 = {'only_texinfo': keyword_287783, 'html_show_formats': keyword_287797, 'only_html': keyword_287779, 'default_fmt': keyword_287764, 'source_link': keyword_287770, 'multi_image': keyword_287777, 'build_dir': keyword_287768, 'images': keyword_287787, 'only_latex': keyword_287781, 'dest_dir': keyword_287766, 'source_code': keyword_287789, 'caption': keyword_287799, 'options': keyword_287785}
    # Getting the type of 'format_template' (line 826)
    format_template_287758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 17), 'format_template', False)
    # Calling format_template(args, kwargs) (line 826)
    format_template_call_result_287801 = invoke(stypy.reporting.localization.Localization(__file__, 826, 17), format_template_287758, *[result_or_keyword_287762], **kwargs_287800)
    
    # Assigning a type to the variable 'result' (line 826)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 826, 8), 'result', format_template_call_result_287801)
    
    # Call to extend(...): (line 842)
    # Processing the call arguments (line 842)
    
    # Call to split(...): (line 842)
    # Processing the call arguments (line 842)
    unicode_287806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 40), 'unicode', u'\n')
    # Processing the call keyword arguments (line 842)
    kwargs_287807 = {}
    # Getting the type of 'result' (line 842)
    result_287804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 27), 'result', False)
    # Obtaining the member 'split' of a type (line 842)
    split_287805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 842, 27), result_287804, 'split')
    # Calling split(args, kwargs) (line 842)
    split_call_result_287808 = invoke(stypy.reporting.localization.Localization(__file__, 842, 27), split_287805, *[unicode_287806], **kwargs_287807)
    
    # Processing the call keyword arguments (line 842)
    kwargs_287809 = {}
    # Getting the type of 'total_lines' (line 842)
    total_lines_287802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 8), 'total_lines', False)
    # Obtaining the member 'extend' of a type (line 842)
    extend_287803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 842, 8), total_lines_287802, 'extend')
    # Calling extend(args, kwargs) (line 842)
    extend_call_result_287810 = invoke(stypy.reporting.localization.Localization(__file__, 842, 8), extend_287803, *[split_call_result_287808], **kwargs_287809)
    
    
    # Call to extend(...): (line 843)
    # Processing the call arguments (line 843)
    unicode_287813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 843, 27), 'unicode', u'\n')
    # Processing the call keyword arguments (line 843)
    kwargs_287814 = {}
    # Getting the type of 'total_lines' (line 843)
    total_lines_287811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 8), 'total_lines', False)
    # Obtaining the member 'extend' of a type (line 843)
    extend_287812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 8), total_lines_287811, 'extend')
    # Calling extend(args, kwargs) (line 843)
    extend_call_result_287815 = invoke(stypy.reporting.localization.Localization(__file__, 843, 8), extend_287812, *[unicode_287813], **kwargs_287814)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'total_lines' (line 845)
    total_lines_287816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 7), 'total_lines')
    # Testing the type of an if condition (line 845)
    if_condition_287817 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 845, 4), total_lines_287816)
    # Assigning a type to the variable 'if_condition_287817' (line 845)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 845, 4), 'if_condition_287817', if_condition_287817)
    # SSA begins for if statement (line 845)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to insert_input(...): (line 846)
    # Processing the call arguments (line 846)
    # Getting the type of 'total_lines' (line 846)
    total_lines_287820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 35), 'total_lines', False)
    # Processing the call keyword arguments (line 846)
    # Getting the type of 'source_file_name' (line 846)
    source_file_name_287821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 55), 'source_file_name', False)
    keyword_287822 = source_file_name_287821
    kwargs_287823 = {'source': keyword_287822}
    # Getting the type of 'state_machine' (line 846)
    state_machine_287818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 8), 'state_machine', False)
    # Obtaining the member 'insert_input' of a type (line 846)
    insert_input_287819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 8), state_machine_287818, 'insert_input')
    # Calling insert_input(args, kwargs) (line 846)
    insert_input_call_result_287824 = invoke(stypy.reporting.localization.Localization(__file__, 846, 8), insert_input_287819, *[total_lines_287820], **kwargs_287823)
    
    # SSA join for if statement (line 845)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to exists(...): (line 849)
    # Processing the call arguments (line 849)
    # Getting the type of 'dest_dir' (line 849)
    dest_dir_287828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 26), 'dest_dir', False)
    # Processing the call keyword arguments (line 849)
    kwargs_287829 = {}
    # Getting the type of 'os' (line 849)
    os_287825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 849)
    path_287826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 849, 11), os_287825, 'path')
    # Obtaining the member 'exists' of a type (line 849)
    exists_287827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 849, 11), path_287826, 'exists')
    # Calling exists(args, kwargs) (line 849)
    exists_call_result_287830 = invoke(stypy.reporting.localization.Localization(__file__, 849, 11), exists_287827, *[dest_dir_287828], **kwargs_287829)
    
    # Applying the 'not' unary operator (line 849)
    result_not__287831 = python_operator(stypy.reporting.localization.Localization(__file__, 849, 7), 'not', exists_call_result_287830)
    
    # Testing the type of an if condition (line 849)
    if_condition_287832 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 849, 4), result_not__287831)
    # Assigning a type to the variable 'if_condition_287832' (line 849)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 849, 4), 'if_condition_287832', if_condition_287832)
    # SSA begins for if statement (line 849)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to mkdirs(...): (line 850)
    # Processing the call arguments (line 850)
    # Getting the type of 'dest_dir' (line 850)
    dest_dir_287835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 21), 'dest_dir', False)
    # Processing the call keyword arguments (line 850)
    kwargs_287836 = {}
    # Getting the type of 'cbook' (line 850)
    cbook_287833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 8), 'cbook', False)
    # Obtaining the member 'mkdirs' of a type (line 850)
    mkdirs_287834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 850, 8), cbook_287833, 'mkdirs')
    # Calling mkdirs(args, kwargs) (line 850)
    mkdirs_call_result_287837 = invoke(stypy.reporting.localization.Localization(__file__, 850, 8), mkdirs_287834, *[dest_dir_287835], **kwargs_287836)
    
    # SSA join for if statement (line 849)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'results' (line 852)
    results_287838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 30), 'results')
    # Testing the type of a for loop iterable (line 852)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 852, 4), results_287838)
    # Getting the type of the for loop variable (line 852)
    for_loop_var_287839 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 852, 4), results_287838)
    # Assigning a type to the variable 'code_piece' (line 852)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 852, 4), 'code_piece', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 852, 4), for_loop_var_287839))
    # Assigning a type to the variable 'images' (line 852)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 852, 4), 'images', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 852, 4), for_loop_var_287839))
    # SSA begins for a for statement (line 852)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'images' (line 853)
    images_287840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 19), 'images')
    # Testing the type of a for loop iterable (line 853)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 853, 8), images_287840)
    # Getting the type of the for loop variable (line 853)
    for_loop_var_287841 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 853, 8), images_287840)
    # Assigning a type to the variable 'img' (line 853)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 853, 8), 'img', for_loop_var_287841)
    # SSA begins for a for statement (line 853)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to filenames(...): (line 854)
    # Processing the call keyword arguments (line 854)
    kwargs_287844 = {}
    # Getting the type of 'img' (line 854)
    img_287842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 22), 'img', False)
    # Obtaining the member 'filenames' of a type (line 854)
    filenames_287843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 854, 22), img_287842, 'filenames')
    # Calling filenames(args, kwargs) (line 854)
    filenames_call_result_287845 = invoke(stypy.reporting.localization.Localization(__file__, 854, 22), filenames_287843, *[], **kwargs_287844)
    
    # Testing the type of a for loop iterable (line 854)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 854, 12), filenames_call_result_287845)
    # Getting the type of the for loop variable (line 854)
    for_loop_var_287846 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 854, 12), filenames_call_result_287845)
    # Assigning a type to the variable 'fn' (line 854)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 854, 12), 'fn', for_loop_var_287846)
    # SSA begins for a for statement (line 854)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 855):
    
    # Assigning a Call to a Name (line 855):
    
    # Call to join(...): (line 855)
    # Processing the call arguments (line 855)
    # Getting the type of 'dest_dir' (line 855)
    dest_dir_287850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 39), 'dest_dir', False)
    
    # Call to basename(...): (line 855)
    # Processing the call arguments (line 855)
    # Getting the type of 'fn' (line 855)
    fn_287854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 66), 'fn', False)
    # Processing the call keyword arguments (line 855)
    kwargs_287855 = {}
    # Getting the type of 'os' (line 855)
    os_287851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 49), 'os', False)
    # Obtaining the member 'path' of a type (line 855)
    path_287852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 855, 49), os_287851, 'path')
    # Obtaining the member 'basename' of a type (line 855)
    basename_287853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 855, 49), path_287852, 'basename')
    # Calling basename(args, kwargs) (line 855)
    basename_call_result_287856 = invoke(stypy.reporting.localization.Localization(__file__, 855, 49), basename_287853, *[fn_287854], **kwargs_287855)
    
    # Processing the call keyword arguments (line 855)
    kwargs_287857 = {}
    # Getting the type of 'os' (line 855)
    os_287847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 26), 'os', False)
    # Obtaining the member 'path' of a type (line 855)
    path_287848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 855, 26), os_287847, 'path')
    # Obtaining the member 'join' of a type (line 855)
    join_287849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 855, 26), path_287848, 'join')
    # Calling join(args, kwargs) (line 855)
    join_call_result_287858 = invoke(stypy.reporting.localization.Localization(__file__, 855, 26), join_287849, *[dest_dir_287850, basename_call_result_287856], **kwargs_287857)
    
    # Assigning a type to the variable 'destimg' (line 855)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 855, 16), 'destimg', join_call_result_287858)
    
    
    # Getting the type of 'fn' (line 856)
    fn_287859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 19), 'fn')
    # Getting the type of 'destimg' (line 856)
    destimg_287860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 25), 'destimg')
    # Applying the binary operator '!=' (line 856)
    result_ne_287861 = python_operator(stypy.reporting.localization.Localization(__file__, 856, 19), '!=', fn_287859, destimg_287860)
    
    # Testing the type of an if condition (line 856)
    if_condition_287862 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 856, 16), result_ne_287861)
    # Assigning a type to the variable 'if_condition_287862' (line 856)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 856, 16), 'if_condition_287862', if_condition_287862)
    # SSA begins for if statement (line 856)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to copyfile(...): (line 857)
    # Processing the call arguments (line 857)
    # Getting the type of 'fn' (line 857)
    fn_287865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 36), 'fn', False)
    # Getting the type of 'destimg' (line 857)
    destimg_287866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 40), 'destimg', False)
    # Processing the call keyword arguments (line 857)
    kwargs_287867 = {}
    # Getting the type of 'shutil' (line 857)
    shutil_287863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 20), 'shutil', False)
    # Obtaining the member 'copyfile' of a type (line 857)
    copyfile_287864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 857, 20), shutil_287863, 'copyfile')
    # Calling copyfile(args, kwargs) (line 857)
    copyfile_call_result_287868 = invoke(stypy.reporting.localization.Localization(__file__, 857, 20), copyfile_287864, *[fn_287865, destimg_287866], **kwargs_287867)
    
    # SSA join for if statement (line 856)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 860):
    
    # Assigning a Call to a Name (line 860):
    
    # Call to join(...): (line 860)
    # Processing the call arguments (line 860)
    # Getting the type of 'dest_dir' (line 860)
    dest_dir_287872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 31), 'dest_dir', False)
    # Getting the type of 'output_base' (line 860)
    output_base_287873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 41), 'output_base', False)
    # Getting the type of 'source_ext' (line 860)
    source_ext_287874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 55), 'source_ext', False)
    # Applying the binary operator '+' (line 860)
    result_add_287875 = python_operator(stypy.reporting.localization.Localization(__file__, 860, 41), '+', output_base_287873, source_ext_287874)
    
    # Processing the call keyword arguments (line 860)
    kwargs_287876 = {}
    # Getting the type of 'os' (line 860)
    os_287869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 18), 'os', False)
    # Obtaining the member 'path' of a type (line 860)
    path_287870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 18), os_287869, 'path')
    # Obtaining the member 'join' of a type (line 860)
    join_287871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 18), path_287870, 'join')
    # Calling join(args, kwargs) (line 860)
    join_call_result_287877 = invoke(stypy.reporting.localization.Localization(__file__, 860, 18), join_287871, *[dest_dir_287872, result_add_287875], **kwargs_287876)
    
    # Assigning a type to the variable 'target_name' (line 860)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 860, 4), 'target_name', join_call_result_287877)
    
    # Call to open(...): (line 861)
    # Processing the call arguments (line 861)
    # Getting the type of 'target_name' (line 861)
    target_name_287880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 17), 'target_name', False)
    unicode_287881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 861, 30), 'unicode', u'w')
    # Processing the call keyword arguments (line 861)
    unicode_287882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 861, 44), 'unicode', u'utf-8')
    keyword_287883 = unicode_287882
    kwargs_287884 = {'encoding': keyword_287883}
    # Getting the type of 'io' (line 861)
    io_287878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 9), 'io', False)
    # Obtaining the member 'open' of a type (line 861)
    open_287879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 861, 9), io_287878, 'open')
    # Calling open(args, kwargs) (line 861)
    open_call_result_287885 = invoke(stypy.reporting.localization.Localization(__file__, 861, 9), open_287879, *[target_name_287880, unicode_287881], **kwargs_287884)
    
    with_287886 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 861, 9), open_call_result_287885, 'with parameter', '__enter__', '__exit__')

    if with_287886:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 861)
        enter___287887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 861, 9), open_call_result_287885, '__enter__')
        with_enter_287888 = invoke(stypy.reporting.localization.Localization(__file__, 861, 9), enter___287887)
        # Assigning a type to the variable 'f' (line 861)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 861, 9), 'f', with_enter_287888)
        
        
        # Getting the type of 'source_file_name' (line 862)
        source_file_name_287889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 11), 'source_file_name')
        # Getting the type of 'rst_file' (line 862)
        rst_file_287890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 31), 'rst_file')
        # Applying the binary operator '==' (line 862)
        result_eq_287891 = python_operator(stypy.reporting.localization.Localization(__file__, 862, 11), '==', source_file_name_287889, rst_file_287890)
        
        # Testing the type of an if condition (line 862)
        if_condition_287892 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 862, 8), result_eq_287891)
        # Assigning a type to the variable 'if_condition_287892' (line 862)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 862, 8), 'if_condition_287892', if_condition_287892)
        # SSA begins for if statement (line 862)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 863):
        
        # Assigning a Call to a Name (line 863):
        
        # Call to unescape_doctest(...): (line 863)
        # Processing the call arguments (line 863)
        # Getting the type of 'code' (line 863)
        code_287894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 44), 'code', False)
        # Processing the call keyword arguments (line 863)
        kwargs_287895 = {}
        # Getting the type of 'unescape_doctest' (line 863)
        unescape_doctest_287893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 27), 'unescape_doctest', False)
        # Calling unescape_doctest(args, kwargs) (line 863)
        unescape_doctest_call_result_287896 = invoke(stypy.reporting.localization.Localization(__file__, 863, 27), unescape_doctest_287893, *[code_287894], **kwargs_287895)
        
        # Assigning a type to the variable 'code_escaped' (line 863)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 863, 12), 'code_escaped', unescape_doctest_call_result_287896)
        # SSA branch for the else part of an if statement (line 862)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 865):
        
        # Assigning a Name to a Name (line 865):
        # Getting the type of 'code' (line 865)
        code_287897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 27), 'code')
        # Assigning a type to the variable 'code_escaped' (line 865)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 865, 12), 'code_escaped', code_287897)
        # SSA join for if statement (line 862)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write(...): (line 866)
        # Processing the call arguments (line 866)
        # Getting the type of 'code_escaped' (line 866)
        code_escaped_287900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 16), 'code_escaped', False)
        # Processing the call keyword arguments (line 866)
        kwargs_287901 = {}
        # Getting the type of 'f' (line 866)
        f_287898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 8), 'f', False)
        # Obtaining the member 'write' of a type (line 866)
        write_287899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 866, 8), f_287898, 'write')
        # Calling write(args, kwargs) (line 866)
        write_call_result_287902 = invoke(stypy.reporting.localization.Localization(__file__, 866, 8), write_287899, *[code_escaped_287900], **kwargs_287901)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 861)
        exit___287903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 861, 9), open_call_result_287885, '__exit__')
        with_exit_287904 = invoke(stypy.reporting.localization.Localization(__file__, 861, 9), exit___287903, None, None, None)

    # Getting the type of 'errors' (line 868)
    errors_287905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 11), 'errors')
    # Assigning a type to the variable 'stypy_return_type' (line 868)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 4), 'stypy_return_type', errors_287905)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 672)
    stypy_return_type_287906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_287906)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_287906

# Assigning a type to the variable 'run' (line 672)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 0), 'run', run)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
