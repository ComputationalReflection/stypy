
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # -*- coding: utf-8 -*-
2: '''
3: formlayout
4: ==========
5: 
6: Module creating Qt form dialogs/layouts to edit various type of parameters
7: 
8: 
9: formlayout License Agreement (MIT License)
10: ------------------------------------------
11: 
12: Copyright (c) 2009 Pierre Raybaut
13: 
14: Permission is hereby granted, free of charge, to any person
15: obtaining a copy of this software and associated documentation
16: files (the "Software"), to deal in the Software without
17: restriction, including without limitation the rights to use,
18: copy, modify, merge, publish, distribute, sublicense, and/or sell
19: copies of the Software, and to permit persons to whom the
20: Software is furnished to do so, subject to the following
21: conditions:
22: 
23: The above copyright notice and this permission notice shall be
24: included in all copies or substantial portions of the Software.
25: 
26: THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
27: EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
28: OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
29: NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
30: HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
31: WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
32: FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
33: OTHER DEALINGS IN THE SOFTWARE.
34: '''
35: 
36: # History:
37: # 1.0.10: added float validator (disable "Ok" and "Apply" button when not valid)
38: # 1.0.7: added support for "Apply" button
39: # 1.0.6: code cleaning
40: 
41: from __future__ import (absolute_import, division, print_function,
42:                         unicode_literals)
43: 
44: __version__ = '1.0.10'
45: __license__ = __doc__
46: 
47: DEBUG = False
48: 
49: import copy
50: import datetime
51: import warnings
52: 
53: import six
54: 
55: from matplotlib import colors as mcolors
56: from matplotlib.backends.qt_compat import QtGui, QtWidgets, QtCore
57: 
58: 
59: BLACKLIST = {"title", "label"}
60: 
61: 
62: class ColorButton(QtWidgets.QPushButton):
63:     '''
64:     Color choosing push button
65:     '''
66:     colorChanged = QtCore.Signal(QtGui.QColor)
67: 
68:     def __init__(self, parent=None):
69:         QtWidgets.QPushButton.__init__(self, parent)
70:         self.setFixedSize(20, 20)
71:         self.setIconSize(QtCore.QSize(12, 12))
72:         self.clicked.connect(self.choose_color)
73:         self._color = QtGui.QColor()
74: 
75:     def choose_color(self):
76:         color = QtWidgets.QColorDialog.getColor(
77:             self._color, self.parentWidget(), "",
78:             QtWidgets.QColorDialog.ShowAlphaChannel)
79:         if color.isValid():
80:             self.set_color(color)
81: 
82:     def get_color(self):
83:         return self._color
84: 
85:     @QtCore.Slot(QtGui.QColor)
86:     def set_color(self, color):
87:         if color != self._color:
88:             self._color = color
89:             self.colorChanged.emit(self._color)
90:             pixmap = QtGui.QPixmap(self.iconSize())
91:             pixmap.fill(color)
92:             self.setIcon(QtGui.QIcon(pixmap))
93: 
94:     color = QtCore.Property(QtGui.QColor, get_color, set_color)
95: 
96: 
97: def to_qcolor(color):
98:     '''Create a QColor from a matplotlib color'''
99:     qcolor = QtGui.QColor()
100:     try:
101:         rgba = mcolors.to_rgba(color)
102:     except ValueError:
103:         warnings.warn('Ignoring invalid color %r' % color)
104:         return qcolor  # return invalid QColor
105:     qcolor.setRgbF(*rgba)
106:     return qcolor
107: 
108: 
109: class ColorLayout(QtWidgets.QHBoxLayout):
110:     '''Color-specialized QLineEdit layout'''
111:     def __init__(self, color, parent=None):
112:         QtWidgets.QHBoxLayout.__init__(self)
113:         assert isinstance(color, QtGui.QColor)
114:         self.lineedit = QtWidgets.QLineEdit(
115:             mcolors.to_hex(color.getRgbF(), keep_alpha=True), parent)
116:         self.lineedit.editingFinished.connect(self.update_color)
117:         self.addWidget(self.lineedit)
118:         self.colorbtn = ColorButton(parent)
119:         self.colorbtn.color = color
120:         self.colorbtn.colorChanged.connect(self.update_text)
121:         self.addWidget(self.colorbtn)
122: 
123:     def update_color(self):
124:         color = self.text()
125:         qcolor = to_qcolor(color)
126:         self.colorbtn.color = qcolor  # defaults to black if not qcolor.isValid()
127: 
128:     def update_text(self, color):
129:         self.lineedit.setText(mcolors.to_hex(color.getRgbF(), keep_alpha=True))
130: 
131:     def text(self):
132:         return self.lineedit.text()
133: 
134: 
135: def font_is_installed(font):
136:     '''Check if font is installed'''
137:     return [fam for fam in QtGui.QFontDatabase().families()
138:             if six.text_type(fam) == font]
139: 
140: 
141: def tuple_to_qfont(tup):
142:     '''
143:     Create a QFont from tuple:
144:         (family [string], size [int], italic [bool], bold [bool])
145:     '''
146:     if not (isinstance(tup, tuple) and len(tup) == 4
147:             and font_is_installed(tup[0])
148:             and isinstance(tup[1], int)
149:             and isinstance(tup[2], bool)
150:             and isinstance(tup[3], bool)):
151:         return None
152:     font = QtGui.QFont()
153:     family, size, italic, bold = tup
154:     font.setFamily(family)
155:     font.setPointSize(size)
156:     font.setItalic(italic)
157:     font.setBold(bold)
158:     return font
159: 
160: 
161: def qfont_to_tuple(font):
162:     return (six.text_type(font.family()), int(font.pointSize()),
163:             font.italic(), font.bold())
164: 
165: 
166: class FontLayout(QtWidgets.QGridLayout):
167:     '''Font selection'''
168:     def __init__(self, value, parent=None):
169:         QtWidgets.QGridLayout.__init__(self)
170:         font = tuple_to_qfont(value)
171:         assert font is not None
172: 
173:         # Font family
174:         self.family = QtWidgets.QFontComboBox(parent)
175:         self.family.setCurrentFont(font)
176:         self.addWidget(self.family, 0, 0, 1, -1)
177: 
178:         # Font size
179:         self.size = QtWidgets.QComboBox(parent)
180:         self.size.setEditable(True)
181:         sizelist = list(range(6, 12)) + list(range(12, 30, 2)) + [36, 48, 72]
182:         size = font.pointSize()
183:         if size not in sizelist:
184:             sizelist.append(size)
185:             sizelist.sort()
186:         self.size.addItems([str(s) for s in sizelist])
187:         self.size.setCurrentIndex(sizelist.index(size))
188:         self.addWidget(self.size, 1, 0)
189: 
190:         # Italic or not
191:         self.italic = QtWidgets.QCheckBox(self.tr("Italic"), parent)
192:         self.italic.setChecked(font.italic())
193:         self.addWidget(self.italic, 1, 1)
194: 
195:         # Bold or not
196:         self.bold = QtWidgets.QCheckBox(self.tr("Bold"), parent)
197:         self.bold.setChecked(font.bold())
198:         self.addWidget(self.bold, 1, 2)
199: 
200:     def get_font(self):
201:         font = self.family.currentFont()
202:         font.setItalic(self.italic.isChecked())
203:         font.setBold(self.bold.isChecked())
204:         font.setPointSize(int(self.size.currentText()))
205:         return qfont_to_tuple(font)
206: 
207: 
208: def is_edit_valid(edit):
209:     text = edit.text()
210:     state = edit.validator().validate(text, 0)[0]
211: 
212:     return state == QtGui.QDoubleValidator.Acceptable
213: 
214: 
215: class FormWidget(QtWidgets.QWidget):
216:     update_buttons = QtCore.Signal()
217:     def __init__(self, data, comment="", parent=None):
218:         QtWidgets.QWidget.__init__(self, parent)
219:         self.data = copy.deepcopy(data)
220:         self.widgets = []
221:         self.formlayout = QtWidgets.QFormLayout(self)
222:         if comment:
223:             self.formlayout.addRow(QtWidgets.QLabel(comment))
224:             self.formlayout.addRow(QtWidgets.QLabel(" "))
225:         if DEBUG:
226:             print("\n"+("*"*80))
227:             print("DATA:", self.data)
228:             print("*"*80)
229:             print("COMMENT:", comment)
230:             print("*"*80)
231: 
232:     def get_dialog(self):
233:         '''Return FormDialog instance'''
234:         dialog = self.parent()
235:         while not isinstance(dialog, QtWidgets.QDialog):
236:             dialog = dialog.parent()
237:         return dialog
238: 
239:     def setup(self):
240:         for label, value in self.data:
241:             if DEBUG:
242:                 print("value:", value)
243:             if label is None and value is None:
244:                 # Separator: (None, None)
245:                 self.formlayout.addRow(QtWidgets.QLabel(" "), QtWidgets.QLabel(" "))
246:                 self.widgets.append(None)
247:                 continue
248:             elif label is None:
249:                 # Comment
250:                 self.formlayout.addRow(QtWidgets.QLabel(value))
251:                 self.widgets.append(None)
252:                 continue
253:             elif tuple_to_qfont(value) is not None:
254:                 field = FontLayout(value, self)
255:             elif (label.lower() not in BLACKLIST
256:                   and mcolors.is_color_like(value)):
257:                 field = ColorLayout(to_qcolor(value), self)
258:             elif isinstance(value, six.string_types):
259:                 field = QtWidgets.QLineEdit(value, self)
260:             elif isinstance(value, (list, tuple)):
261:                 if isinstance(value, tuple):
262:                     value = list(value)
263:                 selindex = value.pop(0)
264:                 field = QtWidgets.QComboBox(self)
265:                 if isinstance(value[0], (list, tuple)):
266:                     keys = [key for key, _val in value]
267:                     value = [val for _key, val in value]
268:                 else:
269:                     keys = value
270:                 field.addItems(value)
271:                 if selindex in value:
272:                     selindex = value.index(selindex)
273:                 elif selindex in keys:
274:                     selindex = keys.index(selindex)
275:                 elif not isinstance(selindex, int):
276:                     warnings.warn(
277:                         "index '%s' is invalid (label: %s, value: %s)" %
278:                         (selindex, label, value))
279:                     selindex = 0
280:                 field.setCurrentIndex(selindex)
281:             elif isinstance(value, bool):
282:                 field = QtWidgets.QCheckBox(self)
283:                 if value:
284:                     field.setCheckState(QtCore.Qt.Checked)
285:                 else:
286:                     field.setCheckState(QtCore.Qt.Unchecked)
287:             elif isinstance(value, float):
288:                 field = QtWidgets.QLineEdit(repr(value), self)
289:                 field.setCursorPosition(0)
290:                 field.setValidator(QtGui.QDoubleValidator(field))
291:                 field.validator().setLocale(QtCore.QLocale("C"))
292:                 dialog = self.get_dialog()
293:                 dialog.register_float_field(field)
294:                 field.textChanged.connect(lambda text: dialog.update_buttons())
295:             elif isinstance(value, int):
296:                 field = QtWidgets.QSpinBox(self)
297:                 field.setRange(-1e9, 1e9)
298:                 field.setValue(value)
299:             elif isinstance(value, datetime.datetime):
300:                 field = QtWidgets.QDateTimeEdit(self)
301:                 field.setDateTime(value)
302:             elif isinstance(value, datetime.date):
303:                 field = QtWidgets.QDateEdit(self)
304:                 field.setDate(value)
305:             else:
306:                 field = QtWidgets.QLineEdit(repr(value), self)
307:             self.formlayout.addRow(label, field)
308:             self.widgets.append(field)
309: 
310:     def get(self):
311:         valuelist = []
312:         for index, (label, value) in enumerate(self.data):
313:             field = self.widgets[index]
314:             if label is None:
315:                 # Separator / Comment
316:                 continue
317:             elif tuple_to_qfont(value) is not None:
318:                 value = field.get_font()
319:             elif (isinstance(value, six.string_types)
320:                   or mcolors.is_color_like(value)):
321:                 value = six.text_type(field.text())
322:             elif isinstance(value, (list, tuple)):
323:                 index = int(field.currentIndex())
324:                 if isinstance(value[0], (list, tuple)):
325:                     value = value[index][0]
326:                 else:
327:                     value = value[index]
328:             elif isinstance(value, bool):
329:                 value = field.checkState() == QtCore.Qt.Checked
330:             elif isinstance(value, float):
331:                 value = float(str(field.text()))
332:             elif isinstance(value, int):
333:                 value = int(field.value())
334:             elif isinstance(value, datetime.datetime):
335:                 value = field.dateTime().toPyDateTime()
336:             elif isinstance(value, datetime.date):
337:                 value = field.date().toPyDate()
338:             else:
339:                 value = eval(str(field.text()))
340:             valuelist.append(value)
341:         return valuelist
342: 
343: 
344: class FormComboWidget(QtWidgets.QWidget):
345:     update_buttons = QtCore.Signal()
346: 
347:     def __init__(self, datalist, comment="", parent=None):
348:         QtWidgets.QWidget.__init__(self, parent)
349:         layout = QtWidgets.QVBoxLayout()
350:         self.setLayout(layout)
351:         self.combobox = QtWidgets.QComboBox()
352:         layout.addWidget(self.combobox)
353: 
354:         self.stackwidget = QtWidgets.QStackedWidget(self)
355:         layout.addWidget(self.stackwidget)
356:         self.combobox.currentIndexChanged.connect(self.stackwidget.setCurrentIndex)
357: 
358:         self.widgetlist = []
359:         for data, title, comment in datalist:
360:             self.combobox.addItem(title)
361:             widget = FormWidget(data, comment=comment, parent=self)
362:             self.stackwidget.addWidget(widget)
363:             self.widgetlist.append(widget)
364: 
365:     def setup(self):
366:         for widget in self.widgetlist:
367:             widget.setup()
368: 
369:     def get(self):
370:         return [widget.get() for widget in self.widgetlist]
371: 
372: 
373: class FormTabWidget(QtWidgets.QWidget):
374:     update_buttons = QtCore.Signal()
375: 
376:     def __init__(self, datalist, comment="", parent=None):
377:         QtWidgets.QWidget.__init__(self, parent)
378:         layout = QtWidgets.QVBoxLayout()
379:         self.tabwidget = QtWidgets.QTabWidget()
380:         layout.addWidget(self.tabwidget)
381:         self.setLayout(layout)
382:         self.widgetlist = []
383:         for data, title, comment in datalist:
384:             if len(data[0]) == 3:
385:                 widget = FormComboWidget(data, comment=comment, parent=self)
386:             else:
387:                 widget = FormWidget(data, comment=comment, parent=self)
388:             index = self.tabwidget.addTab(widget, title)
389:             self.tabwidget.setTabToolTip(index, comment)
390:             self.widgetlist.append(widget)
391: 
392:     def setup(self):
393:         for widget in self.widgetlist:
394:             widget.setup()
395: 
396:     def get(self):
397:         return [widget.get() for widget in self.widgetlist]
398: 
399: 
400: class FormDialog(QtWidgets.QDialog):
401:     '''Form Dialog'''
402:     def __init__(self, data, title="", comment="",
403:                  icon=None, parent=None, apply=None):
404:         QtWidgets.QDialog.__init__(self, parent)
405: 
406:         self.apply_callback = apply
407: 
408:         # Form
409:         if isinstance(data[0][0], (list, tuple)):
410:             self.formwidget = FormTabWidget(data, comment=comment,
411:                                             parent=self)
412:         elif len(data[0]) == 3:
413:             self.formwidget = FormComboWidget(data, comment=comment,
414:                                               parent=self)
415:         else:
416:             self.formwidget = FormWidget(data, comment=comment,
417:                                          parent=self)
418:         layout = QtWidgets.QVBoxLayout()
419:         layout.addWidget(self.formwidget)
420: 
421:         self.float_fields = []
422:         self.formwidget.setup()
423: 
424:         # Button box
425:         self.bbox = bbox = QtWidgets.QDialogButtonBox(
426:             QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
427:         self.formwidget.update_buttons.connect(self.update_buttons)
428:         if self.apply_callback is not None:
429:             apply_btn = bbox.addButton(QtWidgets.QDialogButtonBox.Apply)
430:             apply_btn.clicked.connect(self.apply)
431: 
432:         bbox.accepted.connect(self.accept)
433:         bbox.rejected.connect(self.reject)
434:         layout.addWidget(bbox)
435: 
436:         self.setLayout(layout)
437: 
438:         self.setWindowTitle(title)
439:         if not isinstance(icon, QtGui.QIcon):
440:             icon = QtWidgets.QWidget().style().standardIcon(QtWidgets.QStyle.SP_MessageBoxQuestion)
441:         self.setWindowIcon(icon)
442: 
443:     def register_float_field(self, field):
444:         self.float_fields.append(field)
445: 
446:     def update_buttons(self):
447:         valid = True
448:         for field in self.float_fields:
449:             if not is_edit_valid(field):
450:                 valid = False
451:         for btn_type in (QtWidgets.QDialogButtonBox.Ok,
452:                          QtWidgets.QDialogButtonBox.Apply):
453:             btn = self.bbox.button(btn_type)
454:             if btn is not None:
455:                 btn.setEnabled(valid)
456: 
457:     def accept(self):
458:         self.data = self.formwidget.get()
459:         QtWidgets.QDialog.accept(self)
460: 
461:     def reject(self):
462:         self.data = None
463:         QtWidgets.QDialog.reject(self)
464: 
465:     def apply(self):
466:         self.apply_callback(self.formwidget.get())
467: 
468:     def get(self):
469:         '''Return form result'''
470:         return self.data
471: 
472: 
473: def fedit(data, title="", comment="", icon=None, parent=None, apply=None):
474:     '''
475:     Create form dialog and return result
476:     (if Cancel button is pressed, return None)
477: 
478:     data: datalist, datagroup
479:     title: string
480:     comment: string
481:     icon: QIcon instance
482:     parent: parent QWidget
483:     apply: apply callback (function)
484: 
485:     datalist: list/tuple of (field_name, field_value)
486:     datagroup: list/tuple of (datalist *or* datagroup, title, comment)
487: 
488:     -> one field for each member of a datalist
489:     -> one tab for each member of a top-level datagroup
490:     -> one page (of a multipage widget, each page can be selected with a combo
491:        box) for each member of a datagroup inside a datagroup
492: 
493:     Supported types for field_value:
494:       - int, float, str, unicode, bool
495:       - colors: in Qt-compatible text form, i.e. in hex format or name (red,...)
496:                 (automatically detected from a string)
497:       - list/tuple:
498:           * the first element will be the selected index (or value)
499:           * the other elements can be couples (key, value) or only values
500:     '''
501: 
502:     # Create a QApplication instance if no instance currently exists
503:     # (e.g., if the module is used directly from the interpreter)
504:     if QtWidgets.QApplication.startingUp():
505:         _app = QtWidgets.QApplication([])
506:     dialog = FormDialog(data, title, comment, icon, parent, apply)
507:     if dialog.exec_():
508:         return dialog.get()
509: 
510: 
511: if __name__ == "__main__":
512: 
513:     def create_datalist_example():
514:         return [('str', 'this is a string'),
515:                 ('list', [0, '1', '3', '4']),
516:                 ('list2', ['--', ('none', 'None'), ('--', 'Dashed'),
517:                            ('-.', 'DashDot'), ('-', 'Solid'),
518:                            ('steps', 'Steps'), (':', 'Dotted')]),
519:                 ('float', 1.2),
520:                 (None, 'Other:'),
521:                 ('int', 12),
522:                 ('font', ('Arial', 10, False, True)),
523:                 ('color', '#123409'),
524:                 ('bool', True),
525:                 ('date', datetime.date(2010, 10, 10)),
526:                 ('datetime', datetime.datetime(2010, 10, 10)),
527:                 ]
528: 
529:     def create_datagroup_example():
530:         datalist = create_datalist_example()
531:         return ((datalist, "Category 1", "Category 1 comment"),
532:                 (datalist, "Category 2", "Category 2 comment"),
533:                 (datalist, "Category 3", "Category 3 comment"))
534: 
535:     #--------- datalist example
536:     datalist = create_datalist_example()
537: 
538:     def apply_test(data):
539:         print("data:", data)
540:     print("result:", fedit(datalist, title="Example",
541:                            comment="This is just an <b>example</b>.",
542:                            apply=apply_test))
543: 
544:     #--------- datagroup example
545:     datagroup = create_datagroup_example()
546:     print("result:", fedit(datagroup, "Global title"))
547: 
548:     #--------- datagroup inside a datagroup example
549:     datalist = create_datalist_example()
550:     datagroup = create_datagroup_example()
551:     print("result:", fedit(((datagroup, "Title 1", "Tab 1 comment"),
552:                             (datalist, "Title 2", "Tab 2 comment"),
553:                             (datalist, "Title 3", "Tab 3 comment")),
554:                             "Global title"))
555: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_271004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, (-1)), 'unicode', u'\nformlayout\n==========\n\nModule creating Qt form dialogs/layouts to edit various type of parameters\n\n\nformlayout License Agreement (MIT License)\n------------------------------------------\n\nCopyright (c) 2009 Pierre Raybaut\n\nPermission is hereby granted, free of charge, to any person\nobtaining a copy of this software and associated documentation\nfiles (the "Software"), to deal in the Software without\nrestriction, including without limitation the rights to use,\ncopy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the\nSoftware is furnished to do so, subject to the following\nconditions:\n\nThe above copyright notice and this permission notice shall be\nincluded in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,\nEXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES\nOF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND\nNONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT\nHOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,\nWHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING\nFROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR\nOTHER DEALINGS IN THE SOFTWARE.\n')

# Assigning a Str to a Name (line 44):

# Assigning a Str to a Name (line 44):
unicode_271005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 14), 'unicode', u'1.0.10')
# Assigning a type to the variable '__version__' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), '__version__', unicode_271005)

# Assigning a Name to a Name (line 45):

# Assigning a Name to a Name (line 45):
# Getting the type of '__doc__' (line 45)
doc___271006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 14), '__doc__')
# Assigning a type to the variable '__license__' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), '__license__', doc___271006)

# Assigning a Name to a Name (line 47):

# Assigning a Name to a Name (line 47):
# Getting the type of 'False' (line 47)
False_271007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'False')
# Assigning a type to the variable 'DEBUG' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'DEBUG', False_271007)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 49, 0))

# 'import copy' statement (line 49)
import copy

import_module(stypy.reporting.localization.Localization(__file__, 49, 0), 'copy', copy, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 50, 0))

# 'import datetime' statement (line 50)
import datetime

import_module(stypy.reporting.localization.Localization(__file__, 50, 0), 'datetime', datetime, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 51, 0))

# 'import warnings' statement (line 51)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 51, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 53, 0))

# 'import six' statement (line 53)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/qt_editor/')
import_271008 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 53, 0), 'six')

if (type(import_271008) is not StypyTypeError):

    if (import_271008 != 'pyd_module'):
        __import__(import_271008)
        sys_modules_271009 = sys.modules[import_271008]
        import_module(stypy.reporting.localization.Localization(__file__, 53, 0), 'six', sys_modules_271009.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 53, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'six', import_271008)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/qt_editor/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 55, 0))

# 'from matplotlib import mcolors' statement (line 55)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/qt_editor/')
import_271010 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 55, 0), 'matplotlib')

if (type(import_271010) is not StypyTypeError):

    if (import_271010 != 'pyd_module'):
        __import__(import_271010)
        sys_modules_271011 = sys.modules[import_271010]
        import_from_module(stypy.reporting.localization.Localization(__file__, 55, 0), 'matplotlib', sys_modules_271011.module_type_store, module_type_store, ['colors'])
        nest_module(stypy.reporting.localization.Localization(__file__, 55, 0), __file__, sys_modules_271011, sys_modules_271011.module_type_store, module_type_store)
    else:
        from matplotlib import colors as mcolors

        import_from_module(stypy.reporting.localization.Localization(__file__, 55, 0), 'matplotlib', None, module_type_store, ['colors'], [mcolors])

else:
    # Assigning a type to the variable 'matplotlib' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'matplotlib', import_271010)

# Adding an alias
module_type_store.add_alias('mcolors', 'colors')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/qt_editor/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 56, 0))

# 'from matplotlib.backends.qt_compat import QtGui, QtWidgets, QtCore' statement (line 56)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/qt_editor/')
import_271012 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 56, 0), 'matplotlib.backends.qt_compat')

if (type(import_271012) is not StypyTypeError):

    if (import_271012 != 'pyd_module'):
        __import__(import_271012)
        sys_modules_271013 = sys.modules[import_271012]
        import_from_module(stypy.reporting.localization.Localization(__file__, 56, 0), 'matplotlib.backends.qt_compat', sys_modules_271013.module_type_store, module_type_store, ['QtGui', 'QtWidgets', 'QtCore'])
        nest_module(stypy.reporting.localization.Localization(__file__, 56, 0), __file__, sys_modules_271013, sys_modules_271013.module_type_store, module_type_store)
    else:
        from matplotlib.backends.qt_compat import QtGui, QtWidgets, QtCore

        import_from_module(stypy.reporting.localization.Localization(__file__, 56, 0), 'matplotlib.backends.qt_compat', None, module_type_store, ['QtGui', 'QtWidgets', 'QtCore'], [QtGui, QtWidgets, QtCore])

else:
    # Assigning a type to the variable 'matplotlib.backends.qt_compat' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'matplotlib.backends.qt_compat', import_271012)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/qt_editor/')


# Assigning a Set to a Name (line 59):

# Assigning a Set to a Name (line 59):

# Obtaining an instance of the builtin type 'set' (line 59)
set_271014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 12), 'set')
# Adding type elements to the builtin type 'set' instance (line 59)
# Adding element type (line 59)
unicode_271015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 13), 'unicode', u'title')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 12), set_271014, unicode_271015)
# Adding element type (line 59)
unicode_271016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 22), 'unicode', u'label')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 12), set_271014, unicode_271016)

# Assigning a type to the variable 'BLACKLIST' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'BLACKLIST', set_271014)
# Declaration of the 'ColorButton' class
# Getting the type of 'QtWidgets' (line 62)
QtWidgets_271017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 18), 'QtWidgets')
# Obtaining the member 'QPushButton' of a type (line 62)
QPushButton_271018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 18), QtWidgets_271017, 'QPushButton')

class ColorButton(QPushButton_271018, ):
    unicode_271019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, (-1)), 'unicode', u'\n    Color choosing push button\n    ')
    
    # Assigning a Call to a Name (line 66):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 68)
        None_271020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 30), 'None')
        defaults = [None_271020]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 68, 4, False)
        # Assigning a type to the variable 'self' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ColorButton.__init__', ['parent'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['parent'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'self' (line 69)
        self_271024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 39), 'self', False)
        # Getting the type of 'parent' (line 69)
        parent_271025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 45), 'parent', False)
        # Processing the call keyword arguments (line 69)
        kwargs_271026 = {}
        # Getting the type of 'QtWidgets' (line 69)
        QtWidgets_271021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'QtWidgets', False)
        # Obtaining the member 'QPushButton' of a type (line 69)
        QPushButton_271022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), QtWidgets_271021, 'QPushButton')
        # Obtaining the member '__init__' of a type (line 69)
        init___271023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), QPushButton_271022, '__init__')
        # Calling __init__(args, kwargs) (line 69)
        init___call_result_271027 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), init___271023, *[self_271024, parent_271025], **kwargs_271026)
        
        
        # Call to setFixedSize(...): (line 70)
        # Processing the call arguments (line 70)
        int_271030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 26), 'int')
        int_271031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 30), 'int')
        # Processing the call keyword arguments (line 70)
        kwargs_271032 = {}
        # Getting the type of 'self' (line 70)
        self_271028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'self', False)
        # Obtaining the member 'setFixedSize' of a type (line 70)
        setFixedSize_271029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), self_271028, 'setFixedSize')
        # Calling setFixedSize(args, kwargs) (line 70)
        setFixedSize_call_result_271033 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), setFixedSize_271029, *[int_271030, int_271031], **kwargs_271032)
        
        
        # Call to setIconSize(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Call to QSize(...): (line 71)
        # Processing the call arguments (line 71)
        int_271038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 38), 'int')
        int_271039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 42), 'int')
        # Processing the call keyword arguments (line 71)
        kwargs_271040 = {}
        # Getting the type of 'QtCore' (line 71)
        QtCore_271036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 25), 'QtCore', False)
        # Obtaining the member 'QSize' of a type (line 71)
        QSize_271037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 25), QtCore_271036, 'QSize')
        # Calling QSize(args, kwargs) (line 71)
        QSize_call_result_271041 = invoke(stypy.reporting.localization.Localization(__file__, 71, 25), QSize_271037, *[int_271038, int_271039], **kwargs_271040)
        
        # Processing the call keyword arguments (line 71)
        kwargs_271042 = {}
        # Getting the type of 'self' (line 71)
        self_271034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'self', False)
        # Obtaining the member 'setIconSize' of a type (line 71)
        setIconSize_271035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), self_271034, 'setIconSize')
        # Calling setIconSize(args, kwargs) (line 71)
        setIconSize_call_result_271043 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), setIconSize_271035, *[QSize_call_result_271041], **kwargs_271042)
        
        
        # Call to connect(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'self' (line 72)
        self_271047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 29), 'self', False)
        # Obtaining the member 'choose_color' of a type (line 72)
        choose_color_271048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 29), self_271047, 'choose_color')
        # Processing the call keyword arguments (line 72)
        kwargs_271049 = {}
        # Getting the type of 'self' (line 72)
        self_271044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'self', False)
        # Obtaining the member 'clicked' of a type (line 72)
        clicked_271045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), self_271044, 'clicked')
        # Obtaining the member 'connect' of a type (line 72)
        connect_271046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), clicked_271045, 'connect')
        # Calling connect(args, kwargs) (line 72)
        connect_call_result_271050 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), connect_271046, *[choose_color_271048], **kwargs_271049)
        
        
        # Assigning a Call to a Attribute (line 73):
        
        # Assigning a Call to a Attribute (line 73):
        
        # Call to QColor(...): (line 73)
        # Processing the call keyword arguments (line 73)
        kwargs_271053 = {}
        # Getting the type of 'QtGui' (line 73)
        QtGui_271051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 22), 'QtGui', False)
        # Obtaining the member 'QColor' of a type (line 73)
        QColor_271052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 22), QtGui_271051, 'QColor')
        # Calling QColor(args, kwargs) (line 73)
        QColor_call_result_271054 = invoke(stypy.reporting.localization.Localization(__file__, 73, 22), QColor_271052, *[], **kwargs_271053)
        
        # Getting the type of 'self' (line 73)
        self_271055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self')
        # Setting the type of the member '_color' of a type (line 73)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_271055, '_color', QColor_call_result_271054)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def choose_color(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'choose_color'
        module_type_store = module_type_store.open_function_context('choose_color', 75, 4, False)
        # Assigning a type to the variable 'self' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ColorButton.choose_color.__dict__.__setitem__('stypy_localization', localization)
        ColorButton.choose_color.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ColorButton.choose_color.__dict__.__setitem__('stypy_type_store', module_type_store)
        ColorButton.choose_color.__dict__.__setitem__('stypy_function_name', 'ColorButton.choose_color')
        ColorButton.choose_color.__dict__.__setitem__('stypy_param_names_list', [])
        ColorButton.choose_color.__dict__.__setitem__('stypy_varargs_param_name', None)
        ColorButton.choose_color.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ColorButton.choose_color.__dict__.__setitem__('stypy_call_defaults', defaults)
        ColorButton.choose_color.__dict__.__setitem__('stypy_call_varargs', varargs)
        ColorButton.choose_color.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ColorButton.choose_color.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ColorButton.choose_color', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'choose_color', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'choose_color(...)' code ##################

        
        # Assigning a Call to a Name (line 76):
        
        # Assigning a Call to a Name (line 76):
        
        # Call to getColor(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'self' (line 77)
        self_271059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'self', False)
        # Obtaining the member '_color' of a type (line 77)
        _color_271060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 12), self_271059, '_color')
        
        # Call to parentWidget(...): (line 77)
        # Processing the call keyword arguments (line 77)
        kwargs_271063 = {}
        # Getting the type of 'self' (line 77)
        self_271061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 25), 'self', False)
        # Obtaining the member 'parentWidget' of a type (line 77)
        parentWidget_271062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 25), self_271061, 'parentWidget')
        # Calling parentWidget(args, kwargs) (line 77)
        parentWidget_call_result_271064 = invoke(stypy.reporting.localization.Localization(__file__, 77, 25), parentWidget_271062, *[], **kwargs_271063)
        
        unicode_271065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 46), 'unicode', u'')
        # Getting the type of 'QtWidgets' (line 78)
        QtWidgets_271066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'QtWidgets', False)
        # Obtaining the member 'QColorDialog' of a type (line 78)
        QColorDialog_271067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), QtWidgets_271066, 'QColorDialog')
        # Obtaining the member 'ShowAlphaChannel' of a type (line 78)
        ShowAlphaChannel_271068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), QColorDialog_271067, 'ShowAlphaChannel')
        # Processing the call keyword arguments (line 76)
        kwargs_271069 = {}
        # Getting the type of 'QtWidgets' (line 76)
        QtWidgets_271056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'QtWidgets', False)
        # Obtaining the member 'QColorDialog' of a type (line 76)
        QColorDialog_271057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 16), QtWidgets_271056, 'QColorDialog')
        # Obtaining the member 'getColor' of a type (line 76)
        getColor_271058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 16), QColorDialog_271057, 'getColor')
        # Calling getColor(args, kwargs) (line 76)
        getColor_call_result_271070 = invoke(stypy.reporting.localization.Localization(__file__, 76, 16), getColor_271058, *[_color_271060, parentWidget_call_result_271064, unicode_271065, ShowAlphaChannel_271068], **kwargs_271069)
        
        # Assigning a type to the variable 'color' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'color', getColor_call_result_271070)
        
        
        # Call to isValid(...): (line 79)
        # Processing the call keyword arguments (line 79)
        kwargs_271073 = {}
        # Getting the type of 'color' (line 79)
        color_271071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'color', False)
        # Obtaining the member 'isValid' of a type (line 79)
        isValid_271072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 11), color_271071, 'isValid')
        # Calling isValid(args, kwargs) (line 79)
        isValid_call_result_271074 = invoke(stypy.reporting.localization.Localization(__file__, 79, 11), isValid_271072, *[], **kwargs_271073)
        
        # Testing the type of an if condition (line 79)
        if_condition_271075 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 8), isValid_call_result_271074)
        # Assigning a type to the variable 'if_condition_271075' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'if_condition_271075', if_condition_271075)
        # SSA begins for if statement (line 79)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_color(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'color' (line 80)
        color_271078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 27), 'color', False)
        # Processing the call keyword arguments (line 80)
        kwargs_271079 = {}
        # Getting the type of 'self' (line 80)
        self_271076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'self', False)
        # Obtaining the member 'set_color' of a type (line 80)
        set_color_271077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), self_271076, 'set_color')
        # Calling set_color(args, kwargs) (line 80)
        set_color_call_result_271080 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), set_color_271077, *[color_271078], **kwargs_271079)
        
        # SSA join for if statement (line 79)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'choose_color(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'choose_color' in the type store
        # Getting the type of 'stypy_return_type' (line 75)
        stypy_return_type_271081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_271081)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'choose_color'
        return stypy_return_type_271081


    @norecursion
    def get_color(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_color'
        module_type_store = module_type_store.open_function_context('get_color', 82, 4, False)
        # Assigning a type to the variable 'self' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ColorButton.get_color.__dict__.__setitem__('stypy_localization', localization)
        ColorButton.get_color.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ColorButton.get_color.__dict__.__setitem__('stypy_type_store', module_type_store)
        ColorButton.get_color.__dict__.__setitem__('stypy_function_name', 'ColorButton.get_color')
        ColorButton.get_color.__dict__.__setitem__('stypy_param_names_list', [])
        ColorButton.get_color.__dict__.__setitem__('stypy_varargs_param_name', None)
        ColorButton.get_color.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ColorButton.get_color.__dict__.__setitem__('stypy_call_defaults', defaults)
        ColorButton.get_color.__dict__.__setitem__('stypy_call_varargs', varargs)
        ColorButton.get_color.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ColorButton.get_color.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ColorButton.get_color', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_color', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_color(...)' code ##################

        # Getting the type of 'self' (line 83)
        self_271082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'self')
        # Obtaining the member '_color' of a type (line 83)
        _color_271083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 15), self_271082, '_color')
        # Assigning a type to the variable 'stypy_return_type' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'stypy_return_type', _color_271083)
        
        # ################# End of 'get_color(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_color' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_271084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_271084)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_color'
        return stypy_return_type_271084


    @norecursion
    def set_color(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_color'
        module_type_store = module_type_store.open_function_context('set_color', 85, 4, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ColorButton.set_color.__dict__.__setitem__('stypy_localization', localization)
        ColorButton.set_color.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ColorButton.set_color.__dict__.__setitem__('stypy_type_store', module_type_store)
        ColorButton.set_color.__dict__.__setitem__('stypy_function_name', 'ColorButton.set_color')
        ColorButton.set_color.__dict__.__setitem__('stypy_param_names_list', ['color'])
        ColorButton.set_color.__dict__.__setitem__('stypy_varargs_param_name', None)
        ColorButton.set_color.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ColorButton.set_color.__dict__.__setitem__('stypy_call_defaults', defaults)
        ColorButton.set_color.__dict__.__setitem__('stypy_call_varargs', varargs)
        ColorButton.set_color.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ColorButton.set_color.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ColorButton.set_color', ['color'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_color', localization, ['color'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_color(...)' code ##################

        
        
        # Getting the type of 'color' (line 87)
        color_271085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 11), 'color')
        # Getting the type of 'self' (line 87)
        self_271086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'self')
        # Obtaining the member '_color' of a type (line 87)
        _color_271087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 20), self_271086, '_color')
        # Applying the binary operator '!=' (line 87)
        result_ne_271088 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 11), '!=', color_271085, _color_271087)
        
        # Testing the type of an if condition (line 87)
        if_condition_271089 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 8), result_ne_271088)
        # Assigning a type to the variable 'if_condition_271089' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'if_condition_271089', if_condition_271089)
        # SSA begins for if statement (line 87)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 88):
        
        # Assigning a Name to a Attribute (line 88):
        # Getting the type of 'color' (line 88)
        color_271090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 26), 'color')
        # Getting the type of 'self' (line 88)
        self_271091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'self')
        # Setting the type of the member '_color' of a type (line 88)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 12), self_271091, '_color', color_271090)
        
        # Call to emit(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'self' (line 89)
        self_271095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 35), 'self', False)
        # Obtaining the member '_color' of a type (line 89)
        _color_271096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 35), self_271095, '_color')
        # Processing the call keyword arguments (line 89)
        kwargs_271097 = {}
        # Getting the type of 'self' (line 89)
        self_271092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'self', False)
        # Obtaining the member 'colorChanged' of a type (line 89)
        colorChanged_271093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), self_271092, 'colorChanged')
        # Obtaining the member 'emit' of a type (line 89)
        emit_271094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), colorChanged_271093, 'emit')
        # Calling emit(args, kwargs) (line 89)
        emit_call_result_271098 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), emit_271094, *[_color_271096], **kwargs_271097)
        
        
        # Assigning a Call to a Name (line 90):
        
        # Assigning a Call to a Name (line 90):
        
        # Call to QPixmap(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Call to iconSize(...): (line 90)
        # Processing the call keyword arguments (line 90)
        kwargs_271103 = {}
        # Getting the type of 'self' (line 90)
        self_271101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 35), 'self', False)
        # Obtaining the member 'iconSize' of a type (line 90)
        iconSize_271102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 35), self_271101, 'iconSize')
        # Calling iconSize(args, kwargs) (line 90)
        iconSize_call_result_271104 = invoke(stypy.reporting.localization.Localization(__file__, 90, 35), iconSize_271102, *[], **kwargs_271103)
        
        # Processing the call keyword arguments (line 90)
        kwargs_271105 = {}
        # Getting the type of 'QtGui' (line 90)
        QtGui_271099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 21), 'QtGui', False)
        # Obtaining the member 'QPixmap' of a type (line 90)
        QPixmap_271100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 21), QtGui_271099, 'QPixmap')
        # Calling QPixmap(args, kwargs) (line 90)
        QPixmap_call_result_271106 = invoke(stypy.reporting.localization.Localization(__file__, 90, 21), QPixmap_271100, *[iconSize_call_result_271104], **kwargs_271105)
        
        # Assigning a type to the variable 'pixmap' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'pixmap', QPixmap_call_result_271106)
        
        # Call to fill(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'color' (line 91)
        color_271109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 24), 'color', False)
        # Processing the call keyword arguments (line 91)
        kwargs_271110 = {}
        # Getting the type of 'pixmap' (line 91)
        pixmap_271107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'pixmap', False)
        # Obtaining the member 'fill' of a type (line 91)
        fill_271108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), pixmap_271107, 'fill')
        # Calling fill(args, kwargs) (line 91)
        fill_call_result_271111 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), fill_271108, *[color_271109], **kwargs_271110)
        
        
        # Call to setIcon(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Call to QIcon(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'pixmap' (line 92)
        pixmap_271116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 37), 'pixmap', False)
        # Processing the call keyword arguments (line 92)
        kwargs_271117 = {}
        # Getting the type of 'QtGui' (line 92)
        QtGui_271114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'QtGui', False)
        # Obtaining the member 'QIcon' of a type (line 92)
        QIcon_271115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 25), QtGui_271114, 'QIcon')
        # Calling QIcon(args, kwargs) (line 92)
        QIcon_call_result_271118 = invoke(stypy.reporting.localization.Localization(__file__, 92, 25), QIcon_271115, *[pixmap_271116], **kwargs_271117)
        
        # Processing the call keyword arguments (line 92)
        kwargs_271119 = {}
        # Getting the type of 'self' (line 92)
        self_271112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'self', False)
        # Obtaining the member 'setIcon' of a type (line 92)
        setIcon_271113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), self_271112, 'setIcon')
        # Calling setIcon(args, kwargs) (line 92)
        setIcon_call_result_271120 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), setIcon_271113, *[QIcon_call_result_271118], **kwargs_271119)
        
        # SSA join for if statement (line 87)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_color(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_color' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_271121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_271121)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_color'
        return stypy_return_type_271121

    
    # Assigning a Call to a Name (line 94):

# Assigning a type to the variable 'ColorButton' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'ColorButton', ColorButton)

# Assigning a Call to a Name (line 66):

# Call to Signal(...): (line 66)
# Processing the call arguments (line 66)
# Getting the type of 'QtGui' (line 66)
QtGui_271124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 33), 'QtGui', False)
# Obtaining the member 'QColor' of a type (line 66)
QColor_271125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 33), QtGui_271124, 'QColor')
# Processing the call keyword arguments (line 66)
kwargs_271126 = {}
# Getting the type of 'QtCore' (line 66)
QtCore_271122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 19), 'QtCore', False)
# Obtaining the member 'Signal' of a type (line 66)
Signal_271123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 19), QtCore_271122, 'Signal')
# Calling Signal(args, kwargs) (line 66)
Signal_call_result_271127 = invoke(stypy.reporting.localization.Localization(__file__, 66, 19), Signal_271123, *[QColor_271125], **kwargs_271126)

# Getting the type of 'ColorButton'
ColorButton_271128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ColorButton')
# Setting the type of the member 'colorChanged' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ColorButton_271128, 'colorChanged', Signal_call_result_271127)

# Assigning a Call to a Name (line 94):

# Call to Property(...): (line 94)
# Processing the call arguments (line 94)
# Getting the type of 'QtGui' (line 94)
QtGui_271131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 28), 'QtGui', False)
# Obtaining the member 'QColor' of a type (line 94)
QColor_271132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 28), QtGui_271131, 'QColor')
# Getting the type of 'ColorButton'
ColorButton_271133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ColorButton', False)
# Obtaining the member 'get_color' of a type
get_color_271134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ColorButton_271133, 'get_color')
# Getting the type of 'ColorButton'
ColorButton_271135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ColorButton', False)
# Obtaining the member 'set_color' of a type
set_color_271136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ColorButton_271135, 'set_color')
# Processing the call keyword arguments (line 94)
kwargs_271137 = {}
# Getting the type of 'QtCore' (line 94)
QtCore_271129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'QtCore', False)
# Obtaining the member 'Property' of a type (line 94)
Property_271130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), QtCore_271129, 'Property')
# Calling Property(args, kwargs) (line 94)
Property_call_result_271138 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), Property_271130, *[QColor_271132, get_color_271134, set_color_271136], **kwargs_271137)

# Getting the type of 'ColorButton'
ColorButton_271139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ColorButton')
# Setting the type of the member 'color' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ColorButton_271139, 'color', Property_call_result_271138)

@norecursion
def to_qcolor(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'to_qcolor'
    module_type_store = module_type_store.open_function_context('to_qcolor', 97, 0, False)
    
    # Passed parameters checking function
    to_qcolor.stypy_localization = localization
    to_qcolor.stypy_type_of_self = None
    to_qcolor.stypy_type_store = module_type_store
    to_qcolor.stypy_function_name = 'to_qcolor'
    to_qcolor.stypy_param_names_list = ['color']
    to_qcolor.stypy_varargs_param_name = None
    to_qcolor.stypy_kwargs_param_name = None
    to_qcolor.stypy_call_defaults = defaults
    to_qcolor.stypy_call_varargs = varargs
    to_qcolor.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'to_qcolor', ['color'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'to_qcolor', localization, ['color'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'to_qcolor(...)' code ##################

    unicode_271140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'unicode', u'Create a QColor from a matplotlib color')
    
    # Assigning a Call to a Name (line 99):
    
    # Assigning a Call to a Name (line 99):
    
    # Call to QColor(...): (line 99)
    # Processing the call keyword arguments (line 99)
    kwargs_271143 = {}
    # Getting the type of 'QtGui' (line 99)
    QtGui_271141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 13), 'QtGui', False)
    # Obtaining the member 'QColor' of a type (line 99)
    QColor_271142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 13), QtGui_271141, 'QColor')
    # Calling QColor(args, kwargs) (line 99)
    QColor_call_result_271144 = invoke(stypy.reporting.localization.Localization(__file__, 99, 13), QColor_271142, *[], **kwargs_271143)
    
    # Assigning a type to the variable 'qcolor' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'qcolor', QColor_call_result_271144)
    
    
    # SSA begins for try-except statement (line 100)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 101):
    
    # Assigning a Call to a Name (line 101):
    
    # Call to to_rgba(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'color' (line 101)
    color_271147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'color', False)
    # Processing the call keyword arguments (line 101)
    kwargs_271148 = {}
    # Getting the type of 'mcolors' (line 101)
    mcolors_271145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'mcolors', False)
    # Obtaining the member 'to_rgba' of a type (line 101)
    to_rgba_271146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 15), mcolors_271145, 'to_rgba')
    # Calling to_rgba(args, kwargs) (line 101)
    to_rgba_call_result_271149 = invoke(stypy.reporting.localization.Localization(__file__, 101, 15), to_rgba_271146, *[color_271147], **kwargs_271148)
    
    # Assigning a type to the variable 'rgba' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'rgba', to_rgba_call_result_271149)
    # SSA branch for the except part of a try statement (line 100)
    # SSA branch for the except 'ValueError' branch of a try statement (line 100)
    module_type_store.open_ssa_branch('except')
    
    # Call to warn(...): (line 103)
    # Processing the call arguments (line 103)
    unicode_271152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 22), 'unicode', u'Ignoring invalid color %r')
    # Getting the type of 'color' (line 103)
    color_271153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 52), 'color', False)
    # Applying the binary operator '%' (line 103)
    result_mod_271154 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 22), '%', unicode_271152, color_271153)
    
    # Processing the call keyword arguments (line 103)
    kwargs_271155 = {}
    # Getting the type of 'warnings' (line 103)
    warnings_271150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 103)
    warn_271151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), warnings_271150, 'warn')
    # Calling warn(args, kwargs) (line 103)
    warn_call_result_271156 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), warn_271151, *[result_mod_271154], **kwargs_271155)
    
    # Getting the type of 'qcolor' (line 104)
    qcolor_271157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'qcolor')
    # Assigning a type to the variable 'stypy_return_type' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'stypy_return_type', qcolor_271157)
    # SSA join for try-except statement (line 100)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to setRgbF(...): (line 105)
    # Getting the type of 'rgba' (line 105)
    rgba_271160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 20), 'rgba', False)
    # Processing the call keyword arguments (line 105)
    kwargs_271161 = {}
    # Getting the type of 'qcolor' (line 105)
    qcolor_271158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'qcolor', False)
    # Obtaining the member 'setRgbF' of a type (line 105)
    setRgbF_271159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 4), qcolor_271158, 'setRgbF')
    # Calling setRgbF(args, kwargs) (line 105)
    setRgbF_call_result_271162 = invoke(stypy.reporting.localization.Localization(__file__, 105, 4), setRgbF_271159, *[rgba_271160], **kwargs_271161)
    
    # Getting the type of 'qcolor' (line 106)
    qcolor_271163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'qcolor')
    # Assigning a type to the variable 'stypy_return_type' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'stypy_return_type', qcolor_271163)
    
    # ################# End of 'to_qcolor(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'to_qcolor' in the type store
    # Getting the type of 'stypy_return_type' (line 97)
    stypy_return_type_271164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_271164)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'to_qcolor'
    return stypy_return_type_271164

# Assigning a type to the variable 'to_qcolor' (line 97)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'to_qcolor', to_qcolor)
# Declaration of the 'ColorLayout' class
# Getting the type of 'QtWidgets' (line 109)
QtWidgets_271165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 18), 'QtWidgets')
# Obtaining the member 'QHBoxLayout' of a type (line 109)
QHBoxLayout_271166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 18), QtWidgets_271165, 'QHBoxLayout')

class ColorLayout(QHBoxLayout_271166, ):
    unicode_271167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 4), 'unicode', u'Color-specialized QLineEdit layout')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 111)
        None_271168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 37), 'None')
        defaults = [None_271168]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 111, 4, False)
        # Assigning a type to the variable 'self' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ColorLayout.__init__', ['color', 'parent'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['color', 'parent'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'self' (line 112)
        self_271172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 39), 'self', False)
        # Processing the call keyword arguments (line 112)
        kwargs_271173 = {}
        # Getting the type of 'QtWidgets' (line 112)
        QtWidgets_271169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'QtWidgets', False)
        # Obtaining the member 'QHBoxLayout' of a type (line 112)
        QHBoxLayout_271170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), QtWidgets_271169, 'QHBoxLayout')
        # Obtaining the member '__init__' of a type (line 112)
        init___271171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), QHBoxLayout_271170, '__init__')
        # Calling __init__(args, kwargs) (line 112)
        init___call_result_271174 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), init___271171, *[self_271172], **kwargs_271173)
        
        # Evaluating assert statement condition
        
        # Call to isinstance(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'color' (line 113)
        color_271176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 26), 'color', False)
        # Getting the type of 'QtGui' (line 113)
        QtGui_271177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 33), 'QtGui', False)
        # Obtaining the member 'QColor' of a type (line 113)
        QColor_271178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 33), QtGui_271177, 'QColor')
        # Processing the call keyword arguments (line 113)
        kwargs_271179 = {}
        # Getting the type of 'isinstance' (line 113)
        isinstance_271175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 113)
        isinstance_call_result_271180 = invoke(stypy.reporting.localization.Localization(__file__, 113, 15), isinstance_271175, *[color_271176, QColor_271178], **kwargs_271179)
        
        
        # Assigning a Call to a Attribute (line 114):
        
        # Assigning a Call to a Attribute (line 114):
        
        # Call to QLineEdit(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Call to to_hex(...): (line 115)
        # Processing the call arguments (line 115)
        
        # Call to getRgbF(...): (line 115)
        # Processing the call keyword arguments (line 115)
        kwargs_271187 = {}
        # Getting the type of 'color' (line 115)
        color_271185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 27), 'color', False)
        # Obtaining the member 'getRgbF' of a type (line 115)
        getRgbF_271186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 27), color_271185, 'getRgbF')
        # Calling getRgbF(args, kwargs) (line 115)
        getRgbF_call_result_271188 = invoke(stypy.reporting.localization.Localization(__file__, 115, 27), getRgbF_271186, *[], **kwargs_271187)
        
        # Processing the call keyword arguments (line 115)
        # Getting the type of 'True' (line 115)
        True_271189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 55), 'True', False)
        keyword_271190 = True_271189
        kwargs_271191 = {'keep_alpha': keyword_271190}
        # Getting the type of 'mcolors' (line 115)
        mcolors_271183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'mcolors', False)
        # Obtaining the member 'to_hex' of a type (line 115)
        to_hex_271184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), mcolors_271183, 'to_hex')
        # Calling to_hex(args, kwargs) (line 115)
        to_hex_call_result_271192 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), to_hex_271184, *[getRgbF_call_result_271188], **kwargs_271191)
        
        # Getting the type of 'parent' (line 115)
        parent_271193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 62), 'parent', False)
        # Processing the call keyword arguments (line 114)
        kwargs_271194 = {}
        # Getting the type of 'QtWidgets' (line 114)
        QtWidgets_271181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 24), 'QtWidgets', False)
        # Obtaining the member 'QLineEdit' of a type (line 114)
        QLineEdit_271182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 24), QtWidgets_271181, 'QLineEdit')
        # Calling QLineEdit(args, kwargs) (line 114)
        QLineEdit_call_result_271195 = invoke(stypy.reporting.localization.Localization(__file__, 114, 24), QLineEdit_271182, *[to_hex_call_result_271192, parent_271193], **kwargs_271194)
        
        # Getting the type of 'self' (line 114)
        self_271196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'self')
        # Setting the type of the member 'lineedit' of a type (line 114)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), self_271196, 'lineedit', QLineEdit_call_result_271195)
        
        # Call to connect(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'self' (line 116)
        self_271201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 46), 'self', False)
        # Obtaining the member 'update_color' of a type (line 116)
        update_color_271202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 46), self_271201, 'update_color')
        # Processing the call keyword arguments (line 116)
        kwargs_271203 = {}
        # Getting the type of 'self' (line 116)
        self_271197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'self', False)
        # Obtaining the member 'lineedit' of a type (line 116)
        lineedit_271198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), self_271197, 'lineedit')
        # Obtaining the member 'editingFinished' of a type (line 116)
        editingFinished_271199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), lineedit_271198, 'editingFinished')
        # Obtaining the member 'connect' of a type (line 116)
        connect_271200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), editingFinished_271199, 'connect')
        # Calling connect(args, kwargs) (line 116)
        connect_call_result_271204 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), connect_271200, *[update_color_271202], **kwargs_271203)
        
        
        # Call to addWidget(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'self' (line 117)
        self_271207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 23), 'self', False)
        # Obtaining the member 'lineedit' of a type (line 117)
        lineedit_271208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 23), self_271207, 'lineedit')
        # Processing the call keyword arguments (line 117)
        kwargs_271209 = {}
        # Getting the type of 'self' (line 117)
        self_271205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'self', False)
        # Obtaining the member 'addWidget' of a type (line 117)
        addWidget_271206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), self_271205, 'addWidget')
        # Calling addWidget(args, kwargs) (line 117)
        addWidget_call_result_271210 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), addWidget_271206, *[lineedit_271208], **kwargs_271209)
        
        
        # Assigning a Call to a Attribute (line 118):
        
        # Assigning a Call to a Attribute (line 118):
        
        # Call to ColorButton(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'parent' (line 118)
        parent_271212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 36), 'parent', False)
        # Processing the call keyword arguments (line 118)
        kwargs_271213 = {}
        # Getting the type of 'ColorButton' (line 118)
        ColorButton_271211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 24), 'ColorButton', False)
        # Calling ColorButton(args, kwargs) (line 118)
        ColorButton_call_result_271214 = invoke(stypy.reporting.localization.Localization(__file__, 118, 24), ColorButton_271211, *[parent_271212], **kwargs_271213)
        
        # Getting the type of 'self' (line 118)
        self_271215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'self')
        # Setting the type of the member 'colorbtn' of a type (line 118)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), self_271215, 'colorbtn', ColorButton_call_result_271214)
        
        # Assigning a Name to a Attribute (line 119):
        
        # Assigning a Name to a Attribute (line 119):
        # Getting the type of 'color' (line 119)
        color_271216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 30), 'color')
        # Getting the type of 'self' (line 119)
        self_271217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'self')
        # Obtaining the member 'colorbtn' of a type (line 119)
        colorbtn_271218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), self_271217, 'colorbtn')
        # Setting the type of the member 'color' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), colorbtn_271218, 'color', color_271216)
        
        # Call to connect(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'self' (line 120)
        self_271223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 43), 'self', False)
        # Obtaining the member 'update_text' of a type (line 120)
        update_text_271224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 43), self_271223, 'update_text')
        # Processing the call keyword arguments (line 120)
        kwargs_271225 = {}
        # Getting the type of 'self' (line 120)
        self_271219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'self', False)
        # Obtaining the member 'colorbtn' of a type (line 120)
        colorbtn_271220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), self_271219, 'colorbtn')
        # Obtaining the member 'colorChanged' of a type (line 120)
        colorChanged_271221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), colorbtn_271220, 'colorChanged')
        # Obtaining the member 'connect' of a type (line 120)
        connect_271222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), colorChanged_271221, 'connect')
        # Calling connect(args, kwargs) (line 120)
        connect_call_result_271226 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), connect_271222, *[update_text_271224], **kwargs_271225)
        
        
        # Call to addWidget(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'self' (line 121)
        self_271229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'self', False)
        # Obtaining the member 'colorbtn' of a type (line 121)
        colorbtn_271230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 23), self_271229, 'colorbtn')
        # Processing the call keyword arguments (line 121)
        kwargs_271231 = {}
        # Getting the type of 'self' (line 121)
        self_271227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'self', False)
        # Obtaining the member 'addWidget' of a type (line 121)
        addWidget_271228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), self_271227, 'addWidget')
        # Calling addWidget(args, kwargs) (line 121)
        addWidget_call_result_271232 = invoke(stypy.reporting.localization.Localization(__file__, 121, 8), addWidget_271228, *[colorbtn_271230], **kwargs_271231)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def update_color(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update_color'
        module_type_store = module_type_store.open_function_context('update_color', 123, 4, False)
        # Assigning a type to the variable 'self' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ColorLayout.update_color.__dict__.__setitem__('stypy_localization', localization)
        ColorLayout.update_color.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ColorLayout.update_color.__dict__.__setitem__('stypy_type_store', module_type_store)
        ColorLayout.update_color.__dict__.__setitem__('stypy_function_name', 'ColorLayout.update_color')
        ColorLayout.update_color.__dict__.__setitem__('stypy_param_names_list', [])
        ColorLayout.update_color.__dict__.__setitem__('stypy_varargs_param_name', None)
        ColorLayout.update_color.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ColorLayout.update_color.__dict__.__setitem__('stypy_call_defaults', defaults)
        ColorLayout.update_color.__dict__.__setitem__('stypy_call_varargs', varargs)
        ColorLayout.update_color.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ColorLayout.update_color.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ColorLayout.update_color', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update_color', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update_color(...)' code ##################

        
        # Assigning a Call to a Name (line 124):
        
        # Assigning a Call to a Name (line 124):
        
        # Call to text(...): (line 124)
        # Processing the call keyword arguments (line 124)
        kwargs_271235 = {}
        # Getting the type of 'self' (line 124)
        self_271233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'self', False)
        # Obtaining the member 'text' of a type (line 124)
        text_271234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 16), self_271233, 'text')
        # Calling text(args, kwargs) (line 124)
        text_call_result_271236 = invoke(stypy.reporting.localization.Localization(__file__, 124, 16), text_271234, *[], **kwargs_271235)
        
        # Assigning a type to the variable 'color' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'color', text_call_result_271236)
        
        # Assigning a Call to a Name (line 125):
        
        # Assigning a Call to a Name (line 125):
        
        # Call to to_qcolor(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'color' (line 125)
        color_271238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 27), 'color', False)
        # Processing the call keyword arguments (line 125)
        kwargs_271239 = {}
        # Getting the type of 'to_qcolor' (line 125)
        to_qcolor_271237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 17), 'to_qcolor', False)
        # Calling to_qcolor(args, kwargs) (line 125)
        to_qcolor_call_result_271240 = invoke(stypy.reporting.localization.Localization(__file__, 125, 17), to_qcolor_271237, *[color_271238], **kwargs_271239)
        
        # Assigning a type to the variable 'qcolor' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'qcolor', to_qcolor_call_result_271240)
        
        # Assigning a Name to a Attribute (line 126):
        
        # Assigning a Name to a Attribute (line 126):
        # Getting the type of 'qcolor' (line 126)
        qcolor_271241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 30), 'qcolor')
        # Getting the type of 'self' (line 126)
        self_271242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'self')
        # Obtaining the member 'colorbtn' of a type (line 126)
        colorbtn_271243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), self_271242, 'colorbtn')
        # Setting the type of the member 'color' of a type (line 126)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), colorbtn_271243, 'color', qcolor_271241)
        
        # ################# End of 'update_color(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_color' in the type store
        # Getting the type of 'stypy_return_type' (line 123)
        stypy_return_type_271244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_271244)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_color'
        return stypy_return_type_271244


    @norecursion
    def update_text(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update_text'
        module_type_store = module_type_store.open_function_context('update_text', 128, 4, False)
        # Assigning a type to the variable 'self' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ColorLayout.update_text.__dict__.__setitem__('stypy_localization', localization)
        ColorLayout.update_text.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ColorLayout.update_text.__dict__.__setitem__('stypy_type_store', module_type_store)
        ColorLayout.update_text.__dict__.__setitem__('stypy_function_name', 'ColorLayout.update_text')
        ColorLayout.update_text.__dict__.__setitem__('stypy_param_names_list', ['color'])
        ColorLayout.update_text.__dict__.__setitem__('stypy_varargs_param_name', None)
        ColorLayout.update_text.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ColorLayout.update_text.__dict__.__setitem__('stypy_call_defaults', defaults)
        ColorLayout.update_text.__dict__.__setitem__('stypy_call_varargs', varargs)
        ColorLayout.update_text.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ColorLayout.update_text.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ColorLayout.update_text', ['color'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update_text', localization, ['color'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update_text(...)' code ##################

        
        # Call to setText(...): (line 129)
        # Processing the call arguments (line 129)
        
        # Call to to_hex(...): (line 129)
        # Processing the call arguments (line 129)
        
        # Call to getRgbF(...): (line 129)
        # Processing the call keyword arguments (line 129)
        kwargs_271252 = {}
        # Getting the type of 'color' (line 129)
        color_271250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 45), 'color', False)
        # Obtaining the member 'getRgbF' of a type (line 129)
        getRgbF_271251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 45), color_271250, 'getRgbF')
        # Calling getRgbF(args, kwargs) (line 129)
        getRgbF_call_result_271253 = invoke(stypy.reporting.localization.Localization(__file__, 129, 45), getRgbF_271251, *[], **kwargs_271252)
        
        # Processing the call keyword arguments (line 129)
        # Getting the type of 'True' (line 129)
        True_271254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 73), 'True', False)
        keyword_271255 = True_271254
        kwargs_271256 = {'keep_alpha': keyword_271255}
        # Getting the type of 'mcolors' (line 129)
        mcolors_271248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 30), 'mcolors', False)
        # Obtaining the member 'to_hex' of a type (line 129)
        to_hex_271249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 30), mcolors_271248, 'to_hex')
        # Calling to_hex(args, kwargs) (line 129)
        to_hex_call_result_271257 = invoke(stypy.reporting.localization.Localization(__file__, 129, 30), to_hex_271249, *[getRgbF_call_result_271253], **kwargs_271256)
        
        # Processing the call keyword arguments (line 129)
        kwargs_271258 = {}
        # Getting the type of 'self' (line 129)
        self_271245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self', False)
        # Obtaining the member 'lineedit' of a type (line 129)
        lineedit_271246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_271245, 'lineedit')
        # Obtaining the member 'setText' of a type (line 129)
        setText_271247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), lineedit_271246, 'setText')
        # Calling setText(args, kwargs) (line 129)
        setText_call_result_271259 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), setText_271247, *[to_hex_call_result_271257], **kwargs_271258)
        
        
        # ################# End of 'update_text(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_text' in the type store
        # Getting the type of 'stypy_return_type' (line 128)
        stypy_return_type_271260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_271260)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_text'
        return stypy_return_type_271260


    @norecursion
    def text(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'text'
        module_type_store = module_type_store.open_function_context('text', 131, 4, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ColorLayout.text.__dict__.__setitem__('stypy_localization', localization)
        ColorLayout.text.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ColorLayout.text.__dict__.__setitem__('stypy_type_store', module_type_store)
        ColorLayout.text.__dict__.__setitem__('stypy_function_name', 'ColorLayout.text')
        ColorLayout.text.__dict__.__setitem__('stypy_param_names_list', [])
        ColorLayout.text.__dict__.__setitem__('stypy_varargs_param_name', None)
        ColorLayout.text.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ColorLayout.text.__dict__.__setitem__('stypy_call_defaults', defaults)
        ColorLayout.text.__dict__.__setitem__('stypy_call_varargs', varargs)
        ColorLayout.text.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ColorLayout.text.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ColorLayout.text', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'text', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'text(...)' code ##################

        
        # Call to text(...): (line 132)
        # Processing the call keyword arguments (line 132)
        kwargs_271264 = {}
        # Getting the type of 'self' (line 132)
        self_271261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 15), 'self', False)
        # Obtaining the member 'lineedit' of a type (line 132)
        lineedit_271262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 15), self_271261, 'lineedit')
        # Obtaining the member 'text' of a type (line 132)
        text_271263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 15), lineedit_271262, 'text')
        # Calling text(args, kwargs) (line 132)
        text_call_result_271265 = invoke(stypy.reporting.localization.Localization(__file__, 132, 15), text_271263, *[], **kwargs_271264)
        
        # Assigning a type to the variable 'stypy_return_type' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'stypy_return_type', text_call_result_271265)
        
        # ################# End of 'text(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'text' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_271266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_271266)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'text'
        return stypy_return_type_271266


# Assigning a type to the variable 'ColorLayout' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'ColorLayout', ColorLayout)

@norecursion
def font_is_installed(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'font_is_installed'
    module_type_store = module_type_store.open_function_context('font_is_installed', 135, 0, False)
    
    # Passed parameters checking function
    font_is_installed.stypy_localization = localization
    font_is_installed.stypy_type_of_self = None
    font_is_installed.stypy_type_store = module_type_store
    font_is_installed.stypy_function_name = 'font_is_installed'
    font_is_installed.stypy_param_names_list = ['font']
    font_is_installed.stypy_varargs_param_name = None
    font_is_installed.stypy_kwargs_param_name = None
    font_is_installed.stypy_call_defaults = defaults
    font_is_installed.stypy_call_varargs = varargs
    font_is_installed.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'font_is_installed', ['font'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'font_is_installed', localization, ['font'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'font_is_installed(...)' code ##################

    unicode_271267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 4), 'unicode', u'Check if font is installed')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to families(...): (line 137)
    # Processing the call keyword arguments (line 137)
    kwargs_271281 = {}
    
    # Call to QFontDatabase(...): (line 137)
    # Processing the call keyword arguments (line 137)
    kwargs_271278 = {}
    # Getting the type of 'QtGui' (line 137)
    QtGui_271276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 27), 'QtGui', False)
    # Obtaining the member 'QFontDatabase' of a type (line 137)
    QFontDatabase_271277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 27), QtGui_271276, 'QFontDatabase')
    # Calling QFontDatabase(args, kwargs) (line 137)
    QFontDatabase_call_result_271279 = invoke(stypy.reporting.localization.Localization(__file__, 137, 27), QFontDatabase_271277, *[], **kwargs_271278)
    
    # Obtaining the member 'families' of a type (line 137)
    families_271280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 27), QFontDatabase_call_result_271279, 'families')
    # Calling families(args, kwargs) (line 137)
    families_call_result_271282 = invoke(stypy.reporting.localization.Localization(__file__, 137, 27), families_271280, *[], **kwargs_271281)
    
    comprehension_271283 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 12), families_call_result_271282)
    # Assigning a type to the variable 'fam' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'fam', comprehension_271283)
    
    
    # Call to text_type(...): (line 138)
    # Processing the call arguments (line 138)
    # Getting the type of 'fam' (line 138)
    fam_271271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 29), 'fam', False)
    # Processing the call keyword arguments (line 138)
    kwargs_271272 = {}
    # Getting the type of 'six' (line 138)
    six_271269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'six', False)
    # Obtaining the member 'text_type' of a type (line 138)
    text_type_271270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 15), six_271269, 'text_type')
    # Calling text_type(args, kwargs) (line 138)
    text_type_call_result_271273 = invoke(stypy.reporting.localization.Localization(__file__, 138, 15), text_type_271270, *[fam_271271], **kwargs_271272)
    
    # Getting the type of 'font' (line 138)
    font_271274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 37), 'font')
    # Applying the binary operator '==' (line 138)
    result_eq_271275 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 15), '==', text_type_call_result_271273, font_271274)
    
    # Getting the type of 'fam' (line 137)
    fam_271268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'fam')
    list_271284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 12), list_271284, fam_271268)
    # Assigning a type to the variable 'stypy_return_type' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type', list_271284)
    
    # ################# End of 'font_is_installed(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'font_is_installed' in the type store
    # Getting the type of 'stypy_return_type' (line 135)
    stypy_return_type_271285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_271285)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'font_is_installed'
    return stypy_return_type_271285

# Assigning a type to the variable 'font_is_installed' (line 135)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'font_is_installed', font_is_installed)

@norecursion
def tuple_to_qfont(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'tuple_to_qfont'
    module_type_store = module_type_store.open_function_context('tuple_to_qfont', 141, 0, False)
    
    # Passed parameters checking function
    tuple_to_qfont.stypy_localization = localization
    tuple_to_qfont.stypy_type_of_self = None
    tuple_to_qfont.stypy_type_store = module_type_store
    tuple_to_qfont.stypy_function_name = 'tuple_to_qfont'
    tuple_to_qfont.stypy_param_names_list = ['tup']
    tuple_to_qfont.stypy_varargs_param_name = None
    tuple_to_qfont.stypy_kwargs_param_name = None
    tuple_to_qfont.stypy_call_defaults = defaults
    tuple_to_qfont.stypy_call_varargs = varargs
    tuple_to_qfont.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tuple_to_qfont', ['tup'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tuple_to_qfont', localization, ['tup'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tuple_to_qfont(...)' code ##################

    unicode_271286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, (-1)), 'unicode', u'\n    Create a QFont from tuple:\n        (family [string], size [int], italic [bool], bold [bool])\n    ')
    
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'tup' (line 146)
    tup_271288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 23), 'tup', False)
    # Getting the type of 'tuple' (line 146)
    tuple_271289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 28), 'tuple', False)
    # Processing the call keyword arguments (line 146)
    kwargs_271290 = {}
    # Getting the type of 'isinstance' (line 146)
    isinstance_271287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 146)
    isinstance_call_result_271291 = invoke(stypy.reporting.localization.Localization(__file__, 146, 12), isinstance_271287, *[tup_271288, tuple_271289], **kwargs_271290)
    
    
    
    # Call to len(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'tup' (line 146)
    tup_271293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 43), 'tup', False)
    # Processing the call keyword arguments (line 146)
    kwargs_271294 = {}
    # Getting the type of 'len' (line 146)
    len_271292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 39), 'len', False)
    # Calling len(args, kwargs) (line 146)
    len_call_result_271295 = invoke(stypy.reporting.localization.Localization(__file__, 146, 39), len_271292, *[tup_271293], **kwargs_271294)
    
    int_271296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 51), 'int')
    # Applying the binary operator '==' (line 146)
    result_eq_271297 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 39), '==', len_call_result_271295, int_271296)
    
    # Applying the binary operator 'and' (line 146)
    result_and_keyword_271298 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 12), 'and', isinstance_call_result_271291, result_eq_271297)
    
    # Call to font_is_installed(...): (line 147)
    # Processing the call arguments (line 147)
    
    # Obtaining the type of the subscript
    int_271300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 38), 'int')
    # Getting the type of 'tup' (line 147)
    tup_271301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 34), 'tup', False)
    # Obtaining the member '__getitem__' of a type (line 147)
    getitem___271302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 34), tup_271301, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 147)
    subscript_call_result_271303 = invoke(stypy.reporting.localization.Localization(__file__, 147, 34), getitem___271302, int_271300)
    
    # Processing the call keyword arguments (line 147)
    kwargs_271304 = {}
    # Getting the type of 'font_is_installed' (line 147)
    font_is_installed_271299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'font_is_installed', False)
    # Calling font_is_installed(args, kwargs) (line 147)
    font_is_installed_call_result_271305 = invoke(stypy.reporting.localization.Localization(__file__, 147, 16), font_is_installed_271299, *[subscript_call_result_271303], **kwargs_271304)
    
    # Applying the binary operator 'and' (line 146)
    result_and_keyword_271306 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 12), 'and', result_and_keyword_271298, font_is_installed_call_result_271305)
    
    # Call to isinstance(...): (line 148)
    # Processing the call arguments (line 148)
    
    # Obtaining the type of the subscript
    int_271308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 31), 'int')
    # Getting the type of 'tup' (line 148)
    tup_271309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 27), 'tup', False)
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___271310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 27), tup_271309, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_271311 = invoke(stypy.reporting.localization.Localization(__file__, 148, 27), getitem___271310, int_271308)
    
    # Getting the type of 'int' (line 148)
    int_271312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 35), 'int', False)
    # Processing the call keyword arguments (line 148)
    kwargs_271313 = {}
    # Getting the type of 'isinstance' (line 148)
    isinstance_271307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 148)
    isinstance_call_result_271314 = invoke(stypy.reporting.localization.Localization(__file__, 148, 16), isinstance_271307, *[subscript_call_result_271311, int_271312], **kwargs_271313)
    
    # Applying the binary operator 'and' (line 146)
    result_and_keyword_271315 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 12), 'and', result_and_keyword_271306, isinstance_call_result_271314)
    
    # Call to isinstance(...): (line 149)
    # Processing the call arguments (line 149)
    
    # Obtaining the type of the subscript
    int_271317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 31), 'int')
    # Getting the type of 'tup' (line 149)
    tup_271318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 27), 'tup', False)
    # Obtaining the member '__getitem__' of a type (line 149)
    getitem___271319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 27), tup_271318, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 149)
    subscript_call_result_271320 = invoke(stypy.reporting.localization.Localization(__file__, 149, 27), getitem___271319, int_271317)
    
    # Getting the type of 'bool' (line 149)
    bool_271321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 35), 'bool', False)
    # Processing the call keyword arguments (line 149)
    kwargs_271322 = {}
    # Getting the type of 'isinstance' (line 149)
    isinstance_271316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 149)
    isinstance_call_result_271323 = invoke(stypy.reporting.localization.Localization(__file__, 149, 16), isinstance_271316, *[subscript_call_result_271320, bool_271321], **kwargs_271322)
    
    # Applying the binary operator 'and' (line 146)
    result_and_keyword_271324 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 12), 'and', result_and_keyword_271315, isinstance_call_result_271323)
    
    # Call to isinstance(...): (line 150)
    # Processing the call arguments (line 150)
    
    # Obtaining the type of the subscript
    int_271326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 31), 'int')
    # Getting the type of 'tup' (line 150)
    tup_271327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 27), 'tup', False)
    # Obtaining the member '__getitem__' of a type (line 150)
    getitem___271328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 27), tup_271327, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 150)
    subscript_call_result_271329 = invoke(stypy.reporting.localization.Localization(__file__, 150, 27), getitem___271328, int_271326)
    
    # Getting the type of 'bool' (line 150)
    bool_271330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 35), 'bool', False)
    # Processing the call keyword arguments (line 150)
    kwargs_271331 = {}
    # Getting the type of 'isinstance' (line 150)
    isinstance_271325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 150)
    isinstance_call_result_271332 = invoke(stypy.reporting.localization.Localization(__file__, 150, 16), isinstance_271325, *[subscript_call_result_271329, bool_271330], **kwargs_271331)
    
    # Applying the binary operator 'and' (line 146)
    result_and_keyword_271333 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 12), 'and', result_and_keyword_271324, isinstance_call_result_271332)
    
    # Applying the 'not' unary operator (line 146)
    result_not__271334 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 7), 'not', result_and_keyword_271333)
    
    # Testing the type of an if condition (line 146)
    if_condition_271335 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 4), result_not__271334)
    # Assigning a type to the variable 'if_condition_271335' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'if_condition_271335', if_condition_271335)
    # SSA begins for if statement (line 146)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'None' (line 151)
    None_271336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'stypy_return_type', None_271336)
    # SSA join for if statement (line 146)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 152):
    
    # Assigning a Call to a Name (line 152):
    
    # Call to QFont(...): (line 152)
    # Processing the call keyword arguments (line 152)
    kwargs_271339 = {}
    # Getting the type of 'QtGui' (line 152)
    QtGui_271337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 11), 'QtGui', False)
    # Obtaining the member 'QFont' of a type (line 152)
    QFont_271338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 11), QtGui_271337, 'QFont')
    # Calling QFont(args, kwargs) (line 152)
    QFont_call_result_271340 = invoke(stypy.reporting.localization.Localization(__file__, 152, 11), QFont_271338, *[], **kwargs_271339)
    
    # Assigning a type to the variable 'font' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'font', QFont_call_result_271340)
    
    # Assigning a Name to a Tuple (line 153):
    
    # Assigning a Subscript to a Name (line 153):
    
    # Obtaining the type of the subscript
    int_271341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 4), 'int')
    # Getting the type of 'tup' (line 153)
    tup_271342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 33), 'tup')
    # Obtaining the member '__getitem__' of a type (line 153)
    getitem___271343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 4), tup_271342, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 153)
    subscript_call_result_271344 = invoke(stypy.reporting.localization.Localization(__file__, 153, 4), getitem___271343, int_271341)
    
    # Assigning a type to the variable 'tuple_var_assignment_271000' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'tuple_var_assignment_271000', subscript_call_result_271344)
    
    # Assigning a Subscript to a Name (line 153):
    
    # Obtaining the type of the subscript
    int_271345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 4), 'int')
    # Getting the type of 'tup' (line 153)
    tup_271346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 33), 'tup')
    # Obtaining the member '__getitem__' of a type (line 153)
    getitem___271347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 4), tup_271346, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 153)
    subscript_call_result_271348 = invoke(stypy.reporting.localization.Localization(__file__, 153, 4), getitem___271347, int_271345)
    
    # Assigning a type to the variable 'tuple_var_assignment_271001' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'tuple_var_assignment_271001', subscript_call_result_271348)
    
    # Assigning a Subscript to a Name (line 153):
    
    # Obtaining the type of the subscript
    int_271349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 4), 'int')
    # Getting the type of 'tup' (line 153)
    tup_271350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 33), 'tup')
    # Obtaining the member '__getitem__' of a type (line 153)
    getitem___271351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 4), tup_271350, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 153)
    subscript_call_result_271352 = invoke(stypy.reporting.localization.Localization(__file__, 153, 4), getitem___271351, int_271349)
    
    # Assigning a type to the variable 'tuple_var_assignment_271002' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'tuple_var_assignment_271002', subscript_call_result_271352)
    
    # Assigning a Subscript to a Name (line 153):
    
    # Obtaining the type of the subscript
    int_271353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 4), 'int')
    # Getting the type of 'tup' (line 153)
    tup_271354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 33), 'tup')
    # Obtaining the member '__getitem__' of a type (line 153)
    getitem___271355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 4), tup_271354, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 153)
    subscript_call_result_271356 = invoke(stypy.reporting.localization.Localization(__file__, 153, 4), getitem___271355, int_271353)
    
    # Assigning a type to the variable 'tuple_var_assignment_271003' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'tuple_var_assignment_271003', subscript_call_result_271356)
    
    # Assigning a Name to a Name (line 153):
    # Getting the type of 'tuple_var_assignment_271000' (line 153)
    tuple_var_assignment_271000_271357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'tuple_var_assignment_271000')
    # Assigning a type to the variable 'family' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'family', tuple_var_assignment_271000_271357)
    
    # Assigning a Name to a Name (line 153):
    # Getting the type of 'tuple_var_assignment_271001' (line 153)
    tuple_var_assignment_271001_271358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'tuple_var_assignment_271001')
    # Assigning a type to the variable 'size' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'size', tuple_var_assignment_271001_271358)
    
    # Assigning a Name to a Name (line 153):
    # Getting the type of 'tuple_var_assignment_271002' (line 153)
    tuple_var_assignment_271002_271359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'tuple_var_assignment_271002')
    # Assigning a type to the variable 'italic' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 18), 'italic', tuple_var_assignment_271002_271359)
    
    # Assigning a Name to a Name (line 153):
    # Getting the type of 'tuple_var_assignment_271003' (line 153)
    tuple_var_assignment_271003_271360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'tuple_var_assignment_271003')
    # Assigning a type to the variable 'bold' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 26), 'bold', tuple_var_assignment_271003_271360)
    
    # Call to setFamily(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'family' (line 154)
    family_271363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 19), 'family', False)
    # Processing the call keyword arguments (line 154)
    kwargs_271364 = {}
    # Getting the type of 'font' (line 154)
    font_271361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'font', False)
    # Obtaining the member 'setFamily' of a type (line 154)
    setFamily_271362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 4), font_271361, 'setFamily')
    # Calling setFamily(args, kwargs) (line 154)
    setFamily_call_result_271365 = invoke(stypy.reporting.localization.Localization(__file__, 154, 4), setFamily_271362, *[family_271363], **kwargs_271364)
    
    
    # Call to setPointSize(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 'size' (line 155)
    size_271368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 22), 'size', False)
    # Processing the call keyword arguments (line 155)
    kwargs_271369 = {}
    # Getting the type of 'font' (line 155)
    font_271366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'font', False)
    # Obtaining the member 'setPointSize' of a type (line 155)
    setPointSize_271367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 4), font_271366, 'setPointSize')
    # Calling setPointSize(args, kwargs) (line 155)
    setPointSize_call_result_271370 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), setPointSize_271367, *[size_271368], **kwargs_271369)
    
    
    # Call to setItalic(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'italic' (line 156)
    italic_271373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 19), 'italic', False)
    # Processing the call keyword arguments (line 156)
    kwargs_271374 = {}
    # Getting the type of 'font' (line 156)
    font_271371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'font', False)
    # Obtaining the member 'setItalic' of a type (line 156)
    setItalic_271372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 4), font_271371, 'setItalic')
    # Calling setItalic(args, kwargs) (line 156)
    setItalic_call_result_271375 = invoke(stypy.reporting.localization.Localization(__file__, 156, 4), setItalic_271372, *[italic_271373], **kwargs_271374)
    
    
    # Call to setBold(...): (line 157)
    # Processing the call arguments (line 157)
    # Getting the type of 'bold' (line 157)
    bold_271378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 17), 'bold', False)
    # Processing the call keyword arguments (line 157)
    kwargs_271379 = {}
    # Getting the type of 'font' (line 157)
    font_271376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'font', False)
    # Obtaining the member 'setBold' of a type (line 157)
    setBold_271377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 4), font_271376, 'setBold')
    # Calling setBold(args, kwargs) (line 157)
    setBold_call_result_271380 = invoke(stypy.reporting.localization.Localization(__file__, 157, 4), setBold_271377, *[bold_271378], **kwargs_271379)
    
    # Getting the type of 'font' (line 158)
    font_271381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), 'font')
    # Assigning a type to the variable 'stypy_return_type' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'stypy_return_type', font_271381)
    
    # ################# End of 'tuple_to_qfont(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tuple_to_qfont' in the type store
    # Getting the type of 'stypy_return_type' (line 141)
    stypy_return_type_271382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_271382)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tuple_to_qfont'
    return stypy_return_type_271382

# Assigning a type to the variable 'tuple_to_qfont' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'tuple_to_qfont', tuple_to_qfont)

@norecursion
def qfont_to_tuple(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'qfont_to_tuple'
    module_type_store = module_type_store.open_function_context('qfont_to_tuple', 161, 0, False)
    
    # Passed parameters checking function
    qfont_to_tuple.stypy_localization = localization
    qfont_to_tuple.stypy_type_of_self = None
    qfont_to_tuple.stypy_type_store = module_type_store
    qfont_to_tuple.stypy_function_name = 'qfont_to_tuple'
    qfont_to_tuple.stypy_param_names_list = ['font']
    qfont_to_tuple.stypy_varargs_param_name = None
    qfont_to_tuple.stypy_kwargs_param_name = None
    qfont_to_tuple.stypy_call_defaults = defaults
    qfont_to_tuple.stypy_call_varargs = varargs
    qfont_to_tuple.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'qfont_to_tuple', ['font'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'qfont_to_tuple', localization, ['font'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'qfont_to_tuple(...)' code ##################

    
    # Obtaining an instance of the builtin type 'tuple' (line 162)
    tuple_271383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 162)
    # Adding element type (line 162)
    
    # Call to text_type(...): (line 162)
    # Processing the call arguments (line 162)
    
    # Call to family(...): (line 162)
    # Processing the call keyword arguments (line 162)
    kwargs_271388 = {}
    # Getting the type of 'font' (line 162)
    font_271386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 26), 'font', False)
    # Obtaining the member 'family' of a type (line 162)
    family_271387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 26), font_271386, 'family')
    # Calling family(args, kwargs) (line 162)
    family_call_result_271389 = invoke(stypy.reporting.localization.Localization(__file__, 162, 26), family_271387, *[], **kwargs_271388)
    
    # Processing the call keyword arguments (line 162)
    kwargs_271390 = {}
    # Getting the type of 'six' (line 162)
    six_271384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'six', False)
    # Obtaining the member 'text_type' of a type (line 162)
    text_type_271385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 12), six_271384, 'text_type')
    # Calling text_type(args, kwargs) (line 162)
    text_type_call_result_271391 = invoke(stypy.reporting.localization.Localization(__file__, 162, 12), text_type_271385, *[family_call_result_271389], **kwargs_271390)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 12), tuple_271383, text_type_call_result_271391)
    # Adding element type (line 162)
    
    # Call to int(...): (line 162)
    # Processing the call arguments (line 162)
    
    # Call to pointSize(...): (line 162)
    # Processing the call keyword arguments (line 162)
    kwargs_271395 = {}
    # Getting the type of 'font' (line 162)
    font_271393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 46), 'font', False)
    # Obtaining the member 'pointSize' of a type (line 162)
    pointSize_271394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 46), font_271393, 'pointSize')
    # Calling pointSize(args, kwargs) (line 162)
    pointSize_call_result_271396 = invoke(stypy.reporting.localization.Localization(__file__, 162, 46), pointSize_271394, *[], **kwargs_271395)
    
    # Processing the call keyword arguments (line 162)
    kwargs_271397 = {}
    # Getting the type of 'int' (line 162)
    int_271392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 42), 'int', False)
    # Calling int(args, kwargs) (line 162)
    int_call_result_271398 = invoke(stypy.reporting.localization.Localization(__file__, 162, 42), int_271392, *[pointSize_call_result_271396], **kwargs_271397)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 12), tuple_271383, int_call_result_271398)
    # Adding element type (line 162)
    
    # Call to italic(...): (line 163)
    # Processing the call keyword arguments (line 163)
    kwargs_271401 = {}
    # Getting the type of 'font' (line 163)
    font_271399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'font', False)
    # Obtaining the member 'italic' of a type (line 163)
    italic_271400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 12), font_271399, 'italic')
    # Calling italic(args, kwargs) (line 163)
    italic_call_result_271402 = invoke(stypy.reporting.localization.Localization(__file__, 163, 12), italic_271400, *[], **kwargs_271401)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 12), tuple_271383, italic_call_result_271402)
    # Adding element type (line 162)
    
    # Call to bold(...): (line 163)
    # Processing the call keyword arguments (line 163)
    kwargs_271405 = {}
    # Getting the type of 'font' (line 163)
    font_271403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 27), 'font', False)
    # Obtaining the member 'bold' of a type (line 163)
    bold_271404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 27), font_271403, 'bold')
    # Calling bold(args, kwargs) (line 163)
    bold_call_result_271406 = invoke(stypy.reporting.localization.Localization(__file__, 163, 27), bold_271404, *[], **kwargs_271405)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 12), tuple_271383, bold_call_result_271406)
    
    # Assigning a type to the variable 'stypy_return_type' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'stypy_return_type', tuple_271383)
    
    # ################# End of 'qfont_to_tuple(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'qfont_to_tuple' in the type store
    # Getting the type of 'stypy_return_type' (line 161)
    stypy_return_type_271407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_271407)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'qfont_to_tuple'
    return stypy_return_type_271407

# Assigning a type to the variable 'qfont_to_tuple' (line 161)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'qfont_to_tuple', qfont_to_tuple)
# Declaration of the 'FontLayout' class
# Getting the type of 'QtWidgets' (line 166)
QtWidgets_271408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 17), 'QtWidgets')
# Obtaining the member 'QGridLayout' of a type (line 166)
QGridLayout_271409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 17), QtWidgets_271408, 'QGridLayout')

class FontLayout(QGridLayout_271409, ):
    unicode_271410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 4), 'unicode', u'Font selection')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 168)
        None_271411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 37), 'None')
        defaults = [None_271411]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 168, 4, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontLayout.__init__', ['value', 'parent'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['value', 'parent'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'self' (line 169)
        self_271415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 39), 'self', False)
        # Processing the call keyword arguments (line 169)
        kwargs_271416 = {}
        # Getting the type of 'QtWidgets' (line 169)
        QtWidgets_271412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'QtWidgets', False)
        # Obtaining the member 'QGridLayout' of a type (line 169)
        QGridLayout_271413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), QtWidgets_271412, 'QGridLayout')
        # Obtaining the member '__init__' of a type (line 169)
        init___271414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), QGridLayout_271413, '__init__')
        # Calling __init__(args, kwargs) (line 169)
        init___call_result_271417 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), init___271414, *[self_271415], **kwargs_271416)
        
        
        # Assigning a Call to a Name (line 170):
        
        # Assigning a Call to a Name (line 170):
        
        # Call to tuple_to_qfont(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'value' (line 170)
        value_271419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 30), 'value', False)
        # Processing the call keyword arguments (line 170)
        kwargs_271420 = {}
        # Getting the type of 'tuple_to_qfont' (line 170)
        tuple_to_qfont_271418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 15), 'tuple_to_qfont', False)
        # Calling tuple_to_qfont(args, kwargs) (line 170)
        tuple_to_qfont_call_result_271421 = invoke(stypy.reporting.localization.Localization(__file__, 170, 15), tuple_to_qfont_271418, *[value_271419], **kwargs_271420)
        
        # Assigning a type to the variable 'font' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'font', tuple_to_qfont_call_result_271421)
        # Evaluating assert statement condition
        
        # Getting the type of 'font' (line 171)
        font_271422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 15), 'font')
        # Getting the type of 'None' (line 171)
        None_271423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 27), 'None')
        # Applying the binary operator 'isnot' (line 171)
        result_is_not_271424 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 15), 'isnot', font_271422, None_271423)
        
        
        # Assigning a Call to a Attribute (line 174):
        
        # Assigning a Call to a Attribute (line 174):
        
        # Call to QFontComboBox(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'parent' (line 174)
        parent_271427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 46), 'parent', False)
        # Processing the call keyword arguments (line 174)
        kwargs_271428 = {}
        # Getting the type of 'QtWidgets' (line 174)
        QtWidgets_271425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 22), 'QtWidgets', False)
        # Obtaining the member 'QFontComboBox' of a type (line 174)
        QFontComboBox_271426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 22), QtWidgets_271425, 'QFontComboBox')
        # Calling QFontComboBox(args, kwargs) (line 174)
        QFontComboBox_call_result_271429 = invoke(stypy.reporting.localization.Localization(__file__, 174, 22), QFontComboBox_271426, *[parent_271427], **kwargs_271428)
        
        # Getting the type of 'self' (line 174)
        self_271430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'self')
        # Setting the type of the member 'family' of a type (line 174)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), self_271430, 'family', QFontComboBox_call_result_271429)
        
        # Call to setCurrentFont(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'font' (line 175)
        font_271434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 35), 'font', False)
        # Processing the call keyword arguments (line 175)
        kwargs_271435 = {}
        # Getting the type of 'self' (line 175)
        self_271431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'self', False)
        # Obtaining the member 'family' of a type (line 175)
        family_271432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), self_271431, 'family')
        # Obtaining the member 'setCurrentFont' of a type (line 175)
        setCurrentFont_271433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), family_271432, 'setCurrentFont')
        # Calling setCurrentFont(args, kwargs) (line 175)
        setCurrentFont_call_result_271436 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), setCurrentFont_271433, *[font_271434], **kwargs_271435)
        
        
        # Call to addWidget(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'self' (line 176)
        self_271439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 23), 'self', False)
        # Obtaining the member 'family' of a type (line 176)
        family_271440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 23), self_271439, 'family')
        int_271441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 36), 'int')
        int_271442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 39), 'int')
        int_271443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 42), 'int')
        int_271444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 45), 'int')
        # Processing the call keyword arguments (line 176)
        kwargs_271445 = {}
        # Getting the type of 'self' (line 176)
        self_271437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'self', False)
        # Obtaining the member 'addWidget' of a type (line 176)
        addWidget_271438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), self_271437, 'addWidget')
        # Calling addWidget(args, kwargs) (line 176)
        addWidget_call_result_271446 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), addWidget_271438, *[family_271440, int_271441, int_271442, int_271443, int_271444], **kwargs_271445)
        
        
        # Assigning a Call to a Attribute (line 179):
        
        # Assigning a Call to a Attribute (line 179):
        
        # Call to QComboBox(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'parent' (line 179)
        parent_271449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 40), 'parent', False)
        # Processing the call keyword arguments (line 179)
        kwargs_271450 = {}
        # Getting the type of 'QtWidgets' (line 179)
        QtWidgets_271447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 20), 'QtWidgets', False)
        # Obtaining the member 'QComboBox' of a type (line 179)
        QComboBox_271448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 20), QtWidgets_271447, 'QComboBox')
        # Calling QComboBox(args, kwargs) (line 179)
        QComboBox_call_result_271451 = invoke(stypy.reporting.localization.Localization(__file__, 179, 20), QComboBox_271448, *[parent_271449], **kwargs_271450)
        
        # Getting the type of 'self' (line 179)
        self_271452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'self')
        # Setting the type of the member 'size' of a type (line 179)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), self_271452, 'size', QComboBox_call_result_271451)
        
        # Call to setEditable(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'True' (line 180)
        True_271456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 30), 'True', False)
        # Processing the call keyword arguments (line 180)
        kwargs_271457 = {}
        # Getting the type of 'self' (line 180)
        self_271453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'self', False)
        # Obtaining the member 'size' of a type (line 180)
        size_271454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), self_271453, 'size')
        # Obtaining the member 'setEditable' of a type (line 180)
        setEditable_271455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), size_271454, 'setEditable')
        # Calling setEditable(args, kwargs) (line 180)
        setEditable_call_result_271458 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), setEditable_271455, *[True_271456], **kwargs_271457)
        
        
        # Assigning a BinOp to a Name (line 181):
        
        # Assigning a BinOp to a Name (line 181):
        
        # Call to list(...): (line 181)
        # Processing the call arguments (line 181)
        
        # Call to range(...): (line 181)
        # Processing the call arguments (line 181)
        int_271461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 30), 'int')
        int_271462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 33), 'int')
        # Processing the call keyword arguments (line 181)
        kwargs_271463 = {}
        # Getting the type of 'range' (line 181)
        range_271460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 24), 'range', False)
        # Calling range(args, kwargs) (line 181)
        range_call_result_271464 = invoke(stypy.reporting.localization.Localization(__file__, 181, 24), range_271460, *[int_271461, int_271462], **kwargs_271463)
        
        # Processing the call keyword arguments (line 181)
        kwargs_271465 = {}
        # Getting the type of 'list' (line 181)
        list_271459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 19), 'list', False)
        # Calling list(args, kwargs) (line 181)
        list_call_result_271466 = invoke(stypy.reporting.localization.Localization(__file__, 181, 19), list_271459, *[range_call_result_271464], **kwargs_271465)
        
        
        # Call to list(...): (line 181)
        # Processing the call arguments (line 181)
        
        # Call to range(...): (line 181)
        # Processing the call arguments (line 181)
        int_271469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 51), 'int')
        int_271470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 55), 'int')
        int_271471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 59), 'int')
        # Processing the call keyword arguments (line 181)
        kwargs_271472 = {}
        # Getting the type of 'range' (line 181)
        range_271468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 45), 'range', False)
        # Calling range(args, kwargs) (line 181)
        range_call_result_271473 = invoke(stypy.reporting.localization.Localization(__file__, 181, 45), range_271468, *[int_271469, int_271470, int_271471], **kwargs_271472)
        
        # Processing the call keyword arguments (line 181)
        kwargs_271474 = {}
        # Getting the type of 'list' (line 181)
        list_271467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 40), 'list', False)
        # Calling list(args, kwargs) (line 181)
        list_call_result_271475 = invoke(stypy.reporting.localization.Localization(__file__, 181, 40), list_271467, *[range_call_result_271473], **kwargs_271474)
        
        # Applying the binary operator '+' (line 181)
        result_add_271476 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 19), '+', list_call_result_271466, list_call_result_271475)
        
        
        # Obtaining an instance of the builtin type 'list' (line 181)
        list_271477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 65), 'list')
        # Adding type elements to the builtin type 'list' instance (line 181)
        # Adding element type (line 181)
        int_271478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 66), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 65), list_271477, int_271478)
        # Adding element type (line 181)
        int_271479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 70), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 65), list_271477, int_271479)
        # Adding element type (line 181)
        int_271480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 74), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 65), list_271477, int_271480)
        
        # Applying the binary operator '+' (line 181)
        result_add_271481 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 63), '+', result_add_271476, list_271477)
        
        # Assigning a type to the variable 'sizelist' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'sizelist', result_add_271481)
        
        # Assigning a Call to a Name (line 182):
        
        # Assigning a Call to a Name (line 182):
        
        # Call to pointSize(...): (line 182)
        # Processing the call keyword arguments (line 182)
        kwargs_271484 = {}
        # Getting the type of 'font' (line 182)
        font_271482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'font', False)
        # Obtaining the member 'pointSize' of a type (line 182)
        pointSize_271483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 15), font_271482, 'pointSize')
        # Calling pointSize(args, kwargs) (line 182)
        pointSize_call_result_271485 = invoke(stypy.reporting.localization.Localization(__file__, 182, 15), pointSize_271483, *[], **kwargs_271484)
        
        # Assigning a type to the variable 'size' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'size', pointSize_call_result_271485)
        
        
        # Getting the type of 'size' (line 183)
        size_271486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 11), 'size')
        # Getting the type of 'sizelist' (line 183)
        sizelist_271487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 23), 'sizelist')
        # Applying the binary operator 'notin' (line 183)
        result_contains_271488 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 11), 'notin', size_271486, sizelist_271487)
        
        # Testing the type of an if condition (line 183)
        if_condition_271489 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 183, 8), result_contains_271488)
        # Assigning a type to the variable 'if_condition_271489' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'if_condition_271489', if_condition_271489)
        # SSA begins for if statement (line 183)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'size' (line 184)
        size_271492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 28), 'size', False)
        # Processing the call keyword arguments (line 184)
        kwargs_271493 = {}
        # Getting the type of 'sizelist' (line 184)
        sizelist_271490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'sizelist', False)
        # Obtaining the member 'append' of a type (line 184)
        append_271491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 12), sizelist_271490, 'append')
        # Calling append(args, kwargs) (line 184)
        append_call_result_271494 = invoke(stypy.reporting.localization.Localization(__file__, 184, 12), append_271491, *[size_271492], **kwargs_271493)
        
        
        # Call to sort(...): (line 185)
        # Processing the call keyword arguments (line 185)
        kwargs_271497 = {}
        # Getting the type of 'sizelist' (line 185)
        sizelist_271495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'sizelist', False)
        # Obtaining the member 'sort' of a type (line 185)
        sort_271496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 12), sizelist_271495, 'sort')
        # Calling sort(args, kwargs) (line 185)
        sort_call_result_271498 = invoke(stypy.reporting.localization.Localization(__file__, 185, 12), sort_271496, *[], **kwargs_271497)
        
        # SSA join for if statement (line 183)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to addItems(...): (line 186)
        # Processing the call arguments (line 186)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'sizelist' (line 186)
        sizelist_271506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 44), 'sizelist', False)
        comprehension_271507 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 28), sizelist_271506)
        # Assigning a type to the variable 's' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 28), 's', comprehension_271507)
        
        # Call to str(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 's' (line 186)
        s_271503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 32), 's', False)
        # Processing the call keyword arguments (line 186)
        kwargs_271504 = {}
        # Getting the type of 'str' (line 186)
        str_271502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 28), 'str', False)
        # Calling str(args, kwargs) (line 186)
        str_call_result_271505 = invoke(stypy.reporting.localization.Localization(__file__, 186, 28), str_271502, *[s_271503], **kwargs_271504)
        
        list_271508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 28), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 28), list_271508, str_call_result_271505)
        # Processing the call keyword arguments (line 186)
        kwargs_271509 = {}
        # Getting the type of 'self' (line 186)
        self_271499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'self', False)
        # Obtaining the member 'size' of a type (line 186)
        size_271500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), self_271499, 'size')
        # Obtaining the member 'addItems' of a type (line 186)
        addItems_271501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), size_271500, 'addItems')
        # Calling addItems(args, kwargs) (line 186)
        addItems_call_result_271510 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), addItems_271501, *[list_271508], **kwargs_271509)
        
        
        # Call to setCurrentIndex(...): (line 187)
        # Processing the call arguments (line 187)
        
        # Call to index(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'size' (line 187)
        size_271516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 49), 'size', False)
        # Processing the call keyword arguments (line 187)
        kwargs_271517 = {}
        # Getting the type of 'sizelist' (line 187)
        sizelist_271514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 34), 'sizelist', False)
        # Obtaining the member 'index' of a type (line 187)
        index_271515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 34), sizelist_271514, 'index')
        # Calling index(args, kwargs) (line 187)
        index_call_result_271518 = invoke(stypy.reporting.localization.Localization(__file__, 187, 34), index_271515, *[size_271516], **kwargs_271517)
        
        # Processing the call keyword arguments (line 187)
        kwargs_271519 = {}
        # Getting the type of 'self' (line 187)
        self_271511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'self', False)
        # Obtaining the member 'size' of a type (line 187)
        size_271512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), self_271511, 'size')
        # Obtaining the member 'setCurrentIndex' of a type (line 187)
        setCurrentIndex_271513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), size_271512, 'setCurrentIndex')
        # Calling setCurrentIndex(args, kwargs) (line 187)
        setCurrentIndex_call_result_271520 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), setCurrentIndex_271513, *[index_call_result_271518], **kwargs_271519)
        
        
        # Call to addWidget(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'self' (line 188)
        self_271523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 23), 'self', False)
        # Obtaining the member 'size' of a type (line 188)
        size_271524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 23), self_271523, 'size')
        int_271525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 34), 'int')
        int_271526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 37), 'int')
        # Processing the call keyword arguments (line 188)
        kwargs_271527 = {}
        # Getting the type of 'self' (line 188)
        self_271521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'self', False)
        # Obtaining the member 'addWidget' of a type (line 188)
        addWidget_271522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), self_271521, 'addWidget')
        # Calling addWidget(args, kwargs) (line 188)
        addWidget_call_result_271528 = invoke(stypy.reporting.localization.Localization(__file__, 188, 8), addWidget_271522, *[size_271524, int_271525, int_271526], **kwargs_271527)
        
        
        # Assigning a Call to a Attribute (line 191):
        
        # Assigning a Call to a Attribute (line 191):
        
        # Call to QCheckBox(...): (line 191)
        # Processing the call arguments (line 191)
        
        # Call to tr(...): (line 191)
        # Processing the call arguments (line 191)
        unicode_271533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 50), 'unicode', u'Italic')
        # Processing the call keyword arguments (line 191)
        kwargs_271534 = {}
        # Getting the type of 'self' (line 191)
        self_271531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 42), 'self', False)
        # Obtaining the member 'tr' of a type (line 191)
        tr_271532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 42), self_271531, 'tr')
        # Calling tr(args, kwargs) (line 191)
        tr_call_result_271535 = invoke(stypy.reporting.localization.Localization(__file__, 191, 42), tr_271532, *[unicode_271533], **kwargs_271534)
        
        # Getting the type of 'parent' (line 191)
        parent_271536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 61), 'parent', False)
        # Processing the call keyword arguments (line 191)
        kwargs_271537 = {}
        # Getting the type of 'QtWidgets' (line 191)
        QtWidgets_271529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 22), 'QtWidgets', False)
        # Obtaining the member 'QCheckBox' of a type (line 191)
        QCheckBox_271530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 22), QtWidgets_271529, 'QCheckBox')
        # Calling QCheckBox(args, kwargs) (line 191)
        QCheckBox_call_result_271538 = invoke(stypy.reporting.localization.Localization(__file__, 191, 22), QCheckBox_271530, *[tr_call_result_271535, parent_271536], **kwargs_271537)
        
        # Getting the type of 'self' (line 191)
        self_271539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'self')
        # Setting the type of the member 'italic' of a type (line 191)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), self_271539, 'italic', QCheckBox_call_result_271538)
        
        # Call to setChecked(...): (line 192)
        # Processing the call arguments (line 192)
        
        # Call to italic(...): (line 192)
        # Processing the call keyword arguments (line 192)
        kwargs_271545 = {}
        # Getting the type of 'font' (line 192)
        font_271543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 31), 'font', False)
        # Obtaining the member 'italic' of a type (line 192)
        italic_271544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 31), font_271543, 'italic')
        # Calling italic(args, kwargs) (line 192)
        italic_call_result_271546 = invoke(stypy.reporting.localization.Localization(__file__, 192, 31), italic_271544, *[], **kwargs_271545)
        
        # Processing the call keyword arguments (line 192)
        kwargs_271547 = {}
        # Getting the type of 'self' (line 192)
        self_271540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'self', False)
        # Obtaining the member 'italic' of a type (line 192)
        italic_271541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), self_271540, 'italic')
        # Obtaining the member 'setChecked' of a type (line 192)
        setChecked_271542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), italic_271541, 'setChecked')
        # Calling setChecked(args, kwargs) (line 192)
        setChecked_call_result_271548 = invoke(stypy.reporting.localization.Localization(__file__, 192, 8), setChecked_271542, *[italic_call_result_271546], **kwargs_271547)
        
        
        # Call to addWidget(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'self' (line 193)
        self_271551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 23), 'self', False)
        # Obtaining the member 'italic' of a type (line 193)
        italic_271552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 23), self_271551, 'italic')
        int_271553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 36), 'int')
        int_271554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 39), 'int')
        # Processing the call keyword arguments (line 193)
        kwargs_271555 = {}
        # Getting the type of 'self' (line 193)
        self_271549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'self', False)
        # Obtaining the member 'addWidget' of a type (line 193)
        addWidget_271550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 8), self_271549, 'addWidget')
        # Calling addWidget(args, kwargs) (line 193)
        addWidget_call_result_271556 = invoke(stypy.reporting.localization.Localization(__file__, 193, 8), addWidget_271550, *[italic_271552, int_271553, int_271554], **kwargs_271555)
        
        
        # Assigning a Call to a Attribute (line 196):
        
        # Assigning a Call to a Attribute (line 196):
        
        # Call to QCheckBox(...): (line 196)
        # Processing the call arguments (line 196)
        
        # Call to tr(...): (line 196)
        # Processing the call arguments (line 196)
        unicode_271561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 48), 'unicode', u'Bold')
        # Processing the call keyword arguments (line 196)
        kwargs_271562 = {}
        # Getting the type of 'self' (line 196)
        self_271559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 40), 'self', False)
        # Obtaining the member 'tr' of a type (line 196)
        tr_271560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 40), self_271559, 'tr')
        # Calling tr(args, kwargs) (line 196)
        tr_call_result_271563 = invoke(stypy.reporting.localization.Localization(__file__, 196, 40), tr_271560, *[unicode_271561], **kwargs_271562)
        
        # Getting the type of 'parent' (line 196)
        parent_271564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 57), 'parent', False)
        # Processing the call keyword arguments (line 196)
        kwargs_271565 = {}
        # Getting the type of 'QtWidgets' (line 196)
        QtWidgets_271557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 20), 'QtWidgets', False)
        # Obtaining the member 'QCheckBox' of a type (line 196)
        QCheckBox_271558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 20), QtWidgets_271557, 'QCheckBox')
        # Calling QCheckBox(args, kwargs) (line 196)
        QCheckBox_call_result_271566 = invoke(stypy.reporting.localization.Localization(__file__, 196, 20), QCheckBox_271558, *[tr_call_result_271563, parent_271564], **kwargs_271565)
        
        # Getting the type of 'self' (line 196)
        self_271567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'self')
        # Setting the type of the member 'bold' of a type (line 196)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), self_271567, 'bold', QCheckBox_call_result_271566)
        
        # Call to setChecked(...): (line 197)
        # Processing the call arguments (line 197)
        
        # Call to bold(...): (line 197)
        # Processing the call keyword arguments (line 197)
        kwargs_271573 = {}
        # Getting the type of 'font' (line 197)
        font_271571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 29), 'font', False)
        # Obtaining the member 'bold' of a type (line 197)
        bold_271572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 29), font_271571, 'bold')
        # Calling bold(args, kwargs) (line 197)
        bold_call_result_271574 = invoke(stypy.reporting.localization.Localization(__file__, 197, 29), bold_271572, *[], **kwargs_271573)
        
        # Processing the call keyword arguments (line 197)
        kwargs_271575 = {}
        # Getting the type of 'self' (line 197)
        self_271568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'self', False)
        # Obtaining the member 'bold' of a type (line 197)
        bold_271569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), self_271568, 'bold')
        # Obtaining the member 'setChecked' of a type (line 197)
        setChecked_271570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), bold_271569, 'setChecked')
        # Calling setChecked(args, kwargs) (line 197)
        setChecked_call_result_271576 = invoke(stypy.reporting.localization.Localization(__file__, 197, 8), setChecked_271570, *[bold_call_result_271574], **kwargs_271575)
        
        
        # Call to addWidget(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'self' (line 198)
        self_271579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 23), 'self', False)
        # Obtaining the member 'bold' of a type (line 198)
        bold_271580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 23), self_271579, 'bold')
        int_271581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 34), 'int')
        int_271582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 37), 'int')
        # Processing the call keyword arguments (line 198)
        kwargs_271583 = {}
        # Getting the type of 'self' (line 198)
        self_271577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'self', False)
        # Obtaining the member 'addWidget' of a type (line 198)
        addWidget_271578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), self_271577, 'addWidget')
        # Calling addWidget(args, kwargs) (line 198)
        addWidget_call_result_271584 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), addWidget_271578, *[bold_271580, int_271581, int_271582], **kwargs_271583)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_font(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_font'
        module_type_store = module_type_store.open_function_context('get_font', 200, 4, False)
        # Assigning a type to the variable 'self' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FontLayout.get_font.__dict__.__setitem__('stypy_localization', localization)
        FontLayout.get_font.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FontLayout.get_font.__dict__.__setitem__('stypy_type_store', module_type_store)
        FontLayout.get_font.__dict__.__setitem__('stypy_function_name', 'FontLayout.get_font')
        FontLayout.get_font.__dict__.__setitem__('stypy_param_names_list', [])
        FontLayout.get_font.__dict__.__setitem__('stypy_varargs_param_name', None)
        FontLayout.get_font.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FontLayout.get_font.__dict__.__setitem__('stypy_call_defaults', defaults)
        FontLayout.get_font.__dict__.__setitem__('stypy_call_varargs', varargs)
        FontLayout.get_font.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FontLayout.get_font.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FontLayout.get_font', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_font', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_font(...)' code ##################

        
        # Assigning a Call to a Name (line 201):
        
        # Assigning a Call to a Name (line 201):
        
        # Call to currentFont(...): (line 201)
        # Processing the call keyword arguments (line 201)
        kwargs_271588 = {}
        # Getting the type of 'self' (line 201)
        self_271585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 15), 'self', False)
        # Obtaining the member 'family' of a type (line 201)
        family_271586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 15), self_271585, 'family')
        # Obtaining the member 'currentFont' of a type (line 201)
        currentFont_271587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 15), family_271586, 'currentFont')
        # Calling currentFont(args, kwargs) (line 201)
        currentFont_call_result_271589 = invoke(stypy.reporting.localization.Localization(__file__, 201, 15), currentFont_271587, *[], **kwargs_271588)
        
        # Assigning a type to the variable 'font' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'font', currentFont_call_result_271589)
        
        # Call to setItalic(...): (line 202)
        # Processing the call arguments (line 202)
        
        # Call to isChecked(...): (line 202)
        # Processing the call keyword arguments (line 202)
        kwargs_271595 = {}
        # Getting the type of 'self' (line 202)
        self_271592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 23), 'self', False)
        # Obtaining the member 'italic' of a type (line 202)
        italic_271593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 23), self_271592, 'italic')
        # Obtaining the member 'isChecked' of a type (line 202)
        isChecked_271594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 23), italic_271593, 'isChecked')
        # Calling isChecked(args, kwargs) (line 202)
        isChecked_call_result_271596 = invoke(stypy.reporting.localization.Localization(__file__, 202, 23), isChecked_271594, *[], **kwargs_271595)
        
        # Processing the call keyword arguments (line 202)
        kwargs_271597 = {}
        # Getting the type of 'font' (line 202)
        font_271590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'font', False)
        # Obtaining the member 'setItalic' of a type (line 202)
        setItalic_271591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), font_271590, 'setItalic')
        # Calling setItalic(args, kwargs) (line 202)
        setItalic_call_result_271598 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), setItalic_271591, *[isChecked_call_result_271596], **kwargs_271597)
        
        
        # Call to setBold(...): (line 203)
        # Processing the call arguments (line 203)
        
        # Call to isChecked(...): (line 203)
        # Processing the call keyword arguments (line 203)
        kwargs_271604 = {}
        # Getting the type of 'self' (line 203)
        self_271601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 21), 'self', False)
        # Obtaining the member 'bold' of a type (line 203)
        bold_271602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 21), self_271601, 'bold')
        # Obtaining the member 'isChecked' of a type (line 203)
        isChecked_271603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 21), bold_271602, 'isChecked')
        # Calling isChecked(args, kwargs) (line 203)
        isChecked_call_result_271605 = invoke(stypy.reporting.localization.Localization(__file__, 203, 21), isChecked_271603, *[], **kwargs_271604)
        
        # Processing the call keyword arguments (line 203)
        kwargs_271606 = {}
        # Getting the type of 'font' (line 203)
        font_271599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'font', False)
        # Obtaining the member 'setBold' of a type (line 203)
        setBold_271600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), font_271599, 'setBold')
        # Calling setBold(args, kwargs) (line 203)
        setBold_call_result_271607 = invoke(stypy.reporting.localization.Localization(__file__, 203, 8), setBold_271600, *[isChecked_call_result_271605], **kwargs_271606)
        
        
        # Call to setPointSize(...): (line 204)
        # Processing the call arguments (line 204)
        
        # Call to int(...): (line 204)
        # Processing the call arguments (line 204)
        
        # Call to currentText(...): (line 204)
        # Processing the call keyword arguments (line 204)
        kwargs_271614 = {}
        # Getting the type of 'self' (line 204)
        self_271611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 30), 'self', False)
        # Obtaining the member 'size' of a type (line 204)
        size_271612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 30), self_271611, 'size')
        # Obtaining the member 'currentText' of a type (line 204)
        currentText_271613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 30), size_271612, 'currentText')
        # Calling currentText(args, kwargs) (line 204)
        currentText_call_result_271615 = invoke(stypy.reporting.localization.Localization(__file__, 204, 30), currentText_271613, *[], **kwargs_271614)
        
        # Processing the call keyword arguments (line 204)
        kwargs_271616 = {}
        # Getting the type of 'int' (line 204)
        int_271610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 26), 'int', False)
        # Calling int(args, kwargs) (line 204)
        int_call_result_271617 = invoke(stypy.reporting.localization.Localization(__file__, 204, 26), int_271610, *[currentText_call_result_271615], **kwargs_271616)
        
        # Processing the call keyword arguments (line 204)
        kwargs_271618 = {}
        # Getting the type of 'font' (line 204)
        font_271608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'font', False)
        # Obtaining the member 'setPointSize' of a type (line 204)
        setPointSize_271609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), font_271608, 'setPointSize')
        # Calling setPointSize(args, kwargs) (line 204)
        setPointSize_call_result_271619 = invoke(stypy.reporting.localization.Localization(__file__, 204, 8), setPointSize_271609, *[int_call_result_271617], **kwargs_271618)
        
        
        # Call to qfont_to_tuple(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'font' (line 205)
        font_271621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 30), 'font', False)
        # Processing the call keyword arguments (line 205)
        kwargs_271622 = {}
        # Getting the type of 'qfont_to_tuple' (line 205)
        qfont_to_tuple_271620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 15), 'qfont_to_tuple', False)
        # Calling qfont_to_tuple(args, kwargs) (line 205)
        qfont_to_tuple_call_result_271623 = invoke(stypy.reporting.localization.Localization(__file__, 205, 15), qfont_to_tuple_271620, *[font_271621], **kwargs_271622)
        
        # Assigning a type to the variable 'stypy_return_type' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'stypy_return_type', qfont_to_tuple_call_result_271623)
        
        # ################# End of 'get_font(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_font' in the type store
        # Getting the type of 'stypy_return_type' (line 200)
        stypy_return_type_271624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_271624)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_font'
        return stypy_return_type_271624


# Assigning a type to the variable 'FontLayout' (line 166)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'FontLayout', FontLayout)

@norecursion
def is_edit_valid(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_edit_valid'
    module_type_store = module_type_store.open_function_context('is_edit_valid', 208, 0, False)
    
    # Passed parameters checking function
    is_edit_valid.stypy_localization = localization
    is_edit_valid.stypy_type_of_self = None
    is_edit_valid.stypy_type_store = module_type_store
    is_edit_valid.stypy_function_name = 'is_edit_valid'
    is_edit_valid.stypy_param_names_list = ['edit']
    is_edit_valid.stypy_varargs_param_name = None
    is_edit_valid.stypy_kwargs_param_name = None
    is_edit_valid.stypy_call_defaults = defaults
    is_edit_valid.stypy_call_varargs = varargs
    is_edit_valid.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_edit_valid', ['edit'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_edit_valid', localization, ['edit'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_edit_valid(...)' code ##################

    
    # Assigning a Call to a Name (line 209):
    
    # Assigning a Call to a Name (line 209):
    
    # Call to text(...): (line 209)
    # Processing the call keyword arguments (line 209)
    kwargs_271627 = {}
    # Getting the type of 'edit' (line 209)
    edit_271625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 11), 'edit', False)
    # Obtaining the member 'text' of a type (line 209)
    text_271626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 11), edit_271625, 'text')
    # Calling text(args, kwargs) (line 209)
    text_call_result_271628 = invoke(stypy.reporting.localization.Localization(__file__, 209, 11), text_271626, *[], **kwargs_271627)
    
    # Assigning a type to the variable 'text' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'text', text_call_result_271628)
    
    # Assigning a Subscript to a Name (line 210):
    
    # Assigning a Subscript to a Name (line 210):
    
    # Obtaining the type of the subscript
    int_271629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 47), 'int')
    
    # Call to validate(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'text' (line 210)
    text_271635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 38), 'text', False)
    int_271636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 44), 'int')
    # Processing the call keyword arguments (line 210)
    kwargs_271637 = {}
    
    # Call to validator(...): (line 210)
    # Processing the call keyword arguments (line 210)
    kwargs_271632 = {}
    # Getting the type of 'edit' (line 210)
    edit_271630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'edit', False)
    # Obtaining the member 'validator' of a type (line 210)
    validator_271631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 12), edit_271630, 'validator')
    # Calling validator(args, kwargs) (line 210)
    validator_call_result_271633 = invoke(stypy.reporting.localization.Localization(__file__, 210, 12), validator_271631, *[], **kwargs_271632)
    
    # Obtaining the member 'validate' of a type (line 210)
    validate_271634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 12), validator_call_result_271633, 'validate')
    # Calling validate(args, kwargs) (line 210)
    validate_call_result_271638 = invoke(stypy.reporting.localization.Localization(__file__, 210, 12), validate_271634, *[text_271635, int_271636], **kwargs_271637)
    
    # Obtaining the member '__getitem__' of a type (line 210)
    getitem___271639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 12), validate_call_result_271638, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 210)
    subscript_call_result_271640 = invoke(stypy.reporting.localization.Localization(__file__, 210, 12), getitem___271639, int_271629)
    
    # Assigning a type to the variable 'state' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'state', subscript_call_result_271640)
    
    # Getting the type of 'state' (line 212)
    state_271641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 11), 'state')
    # Getting the type of 'QtGui' (line 212)
    QtGui_271642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), 'QtGui')
    # Obtaining the member 'QDoubleValidator' of a type (line 212)
    QDoubleValidator_271643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 20), QtGui_271642, 'QDoubleValidator')
    # Obtaining the member 'Acceptable' of a type (line 212)
    Acceptable_271644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 20), QDoubleValidator_271643, 'Acceptable')
    # Applying the binary operator '==' (line 212)
    result_eq_271645 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 11), '==', state_271641, Acceptable_271644)
    
    # Assigning a type to the variable 'stypy_return_type' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'stypy_return_type', result_eq_271645)
    
    # ################# End of 'is_edit_valid(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_edit_valid' in the type store
    # Getting the type of 'stypy_return_type' (line 208)
    stypy_return_type_271646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_271646)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_edit_valid'
    return stypy_return_type_271646

# Assigning a type to the variable 'is_edit_valid' (line 208)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 0), 'is_edit_valid', is_edit_valid)
# Declaration of the 'FormWidget' class
# Getting the type of 'QtWidgets' (line 215)
QtWidgets_271647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 17), 'QtWidgets')
# Obtaining the member 'QWidget' of a type (line 215)
QWidget_271648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 17), QtWidgets_271647, 'QWidget')

class FormWidget(QWidget_271648, ):
    
    # Assigning a Call to a Name (line 216):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        unicode_271649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 37), 'unicode', u'')
        # Getting the type of 'None' (line 217)
        None_271650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 48), 'None')
        defaults = [unicode_271649, None_271650]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 217, 4, False)
        # Assigning a type to the variable 'self' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FormWidget.__init__', ['data', 'comment', 'parent'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['data', 'comment', 'parent'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'self' (line 218)
        self_271654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 35), 'self', False)
        # Getting the type of 'parent' (line 218)
        parent_271655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 41), 'parent', False)
        # Processing the call keyword arguments (line 218)
        kwargs_271656 = {}
        # Getting the type of 'QtWidgets' (line 218)
        QtWidgets_271651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'QtWidgets', False)
        # Obtaining the member 'QWidget' of a type (line 218)
        QWidget_271652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), QtWidgets_271651, 'QWidget')
        # Obtaining the member '__init__' of a type (line 218)
        init___271653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), QWidget_271652, '__init__')
        # Calling __init__(args, kwargs) (line 218)
        init___call_result_271657 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), init___271653, *[self_271654, parent_271655], **kwargs_271656)
        
        
        # Assigning a Call to a Attribute (line 219):
        
        # Assigning a Call to a Attribute (line 219):
        
        # Call to deepcopy(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'data' (line 219)
        data_271660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 34), 'data', False)
        # Processing the call keyword arguments (line 219)
        kwargs_271661 = {}
        # Getting the type of 'copy' (line 219)
        copy_271658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 20), 'copy', False)
        # Obtaining the member 'deepcopy' of a type (line 219)
        deepcopy_271659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 20), copy_271658, 'deepcopy')
        # Calling deepcopy(args, kwargs) (line 219)
        deepcopy_call_result_271662 = invoke(stypy.reporting.localization.Localization(__file__, 219, 20), deepcopy_271659, *[data_271660], **kwargs_271661)
        
        # Getting the type of 'self' (line 219)
        self_271663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'self')
        # Setting the type of the member 'data' of a type (line 219)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 8), self_271663, 'data', deepcopy_call_result_271662)
        
        # Assigning a List to a Attribute (line 220):
        
        # Assigning a List to a Attribute (line 220):
        
        # Obtaining an instance of the builtin type 'list' (line 220)
        list_271664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 220)
        
        # Getting the type of 'self' (line 220)
        self_271665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'self')
        # Setting the type of the member 'widgets' of a type (line 220)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 8), self_271665, 'widgets', list_271664)
        
        # Assigning a Call to a Attribute (line 221):
        
        # Assigning a Call to a Attribute (line 221):
        
        # Call to QFormLayout(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'self' (line 221)
        self_271668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 48), 'self', False)
        # Processing the call keyword arguments (line 221)
        kwargs_271669 = {}
        # Getting the type of 'QtWidgets' (line 221)
        QtWidgets_271666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 26), 'QtWidgets', False)
        # Obtaining the member 'QFormLayout' of a type (line 221)
        QFormLayout_271667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 26), QtWidgets_271666, 'QFormLayout')
        # Calling QFormLayout(args, kwargs) (line 221)
        QFormLayout_call_result_271670 = invoke(stypy.reporting.localization.Localization(__file__, 221, 26), QFormLayout_271667, *[self_271668], **kwargs_271669)
        
        # Getting the type of 'self' (line 221)
        self_271671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'self')
        # Setting the type of the member 'formlayout' of a type (line 221)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), self_271671, 'formlayout', QFormLayout_call_result_271670)
        
        # Getting the type of 'comment' (line 222)
        comment_271672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 11), 'comment')
        # Testing the type of an if condition (line 222)
        if_condition_271673 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 222, 8), comment_271672)
        # Assigning a type to the variable 'if_condition_271673' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'if_condition_271673', if_condition_271673)
        # SSA begins for if statement (line 222)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to addRow(...): (line 223)
        # Processing the call arguments (line 223)
        
        # Call to QLabel(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'comment' (line 223)
        comment_271679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 52), 'comment', False)
        # Processing the call keyword arguments (line 223)
        kwargs_271680 = {}
        # Getting the type of 'QtWidgets' (line 223)
        QtWidgets_271677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 35), 'QtWidgets', False)
        # Obtaining the member 'QLabel' of a type (line 223)
        QLabel_271678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 35), QtWidgets_271677, 'QLabel')
        # Calling QLabel(args, kwargs) (line 223)
        QLabel_call_result_271681 = invoke(stypy.reporting.localization.Localization(__file__, 223, 35), QLabel_271678, *[comment_271679], **kwargs_271680)
        
        # Processing the call keyword arguments (line 223)
        kwargs_271682 = {}
        # Getting the type of 'self' (line 223)
        self_271674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'self', False)
        # Obtaining the member 'formlayout' of a type (line 223)
        formlayout_271675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 12), self_271674, 'formlayout')
        # Obtaining the member 'addRow' of a type (line 223)
        addRow_271676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 12), formlayout_271675, 'addRow')
        # Calling addRow(args, kwargs) (line 223)
        addRow_call_result_271683 = invoke(stypy.reporting.localization.Localization(__file__, 223, 12), addRow_271676, *[QLabel_call_result_271681], **kwargs_271682)
        
        
        # Call to addRow(...): (line 224)
        # Processing the call arguments (line 224)
        
        # Call to QLabel(...): (line 224)
        # Processing the call arguments (line 224)
        unicode_271689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 52), 'unicode', u' ')
        # Processing the call keyword arguments (line 224)
        kwargs_271690 = {}
        # Getting the type of 'QtWidgets' (line 224)
        QtWidgets_271687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 35), 'QtWidgets', False)
        # Obtaining the member 'QLabel' of a type (line 224)
        QLabel_271688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 35), QtWidgets_271687, 'QLabel')
        # Calling QLabel(args, kwargs) (line 224)
        QLabel_call_result_271691 = invoke(stypy.reporting.localization.Localization(__file__, 224, 35), QLabel_271688, *[unicode_271689], **kwargs_271690)
        
        # Processing the call keyword arguments (line 224)
        kwargs_271692 = {}
        # Getting the type of 'self' (line 224)
        self_271684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'self', False)
        # Obtaining the member 'formlayout' of a type (line 224)
        formlayout_271685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 12), self_271684, 'formlayout')
        # Obtaining the member 'addRow' of a type (line 224)
        addRow_271686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 12), formlayout_271685, 'addRow')
        # Calling addRow(args, kwargs) (line 224)
        addRow_call_result_271693 = invoke(stypy.reporting.localization.Localization(__file__, 224, 12), addRow_271686, *[QLabel_call_result_271691], **kwargs_271692)
        
        # SSA join for if statement (line 222)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'DEBUG' (line 225)
        DEBUG_271694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 11), 'DEBUG')
        # Testing the type of an if condition (line 225)
        if_condition_271695 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 225, 8), DEBUG_271694)
        # Assigning a type to the variable 'if_condition_271695' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'if_condition_271695', if_condition_271695)
        # SSA begins for if statement (line 225)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 226)
        # Processing the call arguments (line 226)
        unicode_271697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 18), 'unicode', u'\n')
        unicode_271698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 24), 'unicode', u'*')
        int_271699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 28), 'int')
        # Applying the binary operator '*' (line 226)
        result_mul_271700 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 24), '*', unicode_271698, int_271699)
        
        # Applying the binary operator '+' (line 226)
        result_add_271701 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 18), '+', unicode_271697, result_mul_271700)
        
        # Processing the call keyword arguments (line 226)
        kwargs_271702 = {}
        # Getting the type of 'print' (line 226)
        print_271696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'print', False)
        # Calling print(args, kwargs) (line 226)
        print_call_result_271703 = invoke(stypy.reporting.localization.Localization(__file__, 226, 12), print_271696, *[result_add_271701], **kwargs_271702)
        
        
        # Call to print(...): (line 227)
        # Processing the call arguments (line 227)
        unicode_271705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 18), 'unicode', u'DATA:')
        # Getting the type of 'self' (line 227)
        self_271706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 27), 'self', False)
        # Obtaining the member 'data' of a type (line 227)
        data_271707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 27), self_271706, 'data')
        # Processing the call keyword arguments (line 227)
        kwargs_271708 = {}
        # Getting the type of 'print' (line 227)
        print_271704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'print', False)
        # Calling print(args, kwargs) (line 227)
        print_call_result_271709 = invoke(stypy.reporting.localization.Localization(__file__, 227, 12), print_271704, *[unicode_271705, data_271707], **kwargs_271708)
        
        
        # Call to print(...): (line 228)
        # Processing the call arguments (line 228)
        unicode_271711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 18), 'unicode', u'*')
        int_271712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 22), 'int')
        # Applying the binary operator '*' (line 228)
        result_mul_271713 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 18), '*', unicode_271711, int_271712)
        
        # Processing the call keyword arguments (line 228)
        kwargs_271714 = {}
        # Getting the type of 'print' (line 228)
        print_271710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'print', False)
        # Calling print(args, kwargs) (line 228)
        print_call_result_271715 = invoke(stypy.reporting.localization.Localization(__file__, 228, 12), print_271710, *[result_mul_271713], **kwargs_271714)
        
        
        # Call to print(...): (line 229)
        # Processing the call arguments (line 229)
        unicode_271717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 18), 'unicode', u'COMMENT:')
        # Getting the type of 'comment' (line 229)
        comment_271718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 30), 'comment', False)
        # Processing the call keyword arguments (line 229)
        kwargs_271719 = {}
        # Getting the type of 'print' (line 229)
        print_271716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'print', False)
        # Calling print(args, kwargs) (line 229)
        print_call_result_271720 = invoke(stypy.reporting.localization.Localization(__file__, 229, 12), print_271716, *[unicode_271717, comment_271718], **kwargs_271719)
        
        
        # Call to print(...): (line 230)
        # Processing the call arguments (line 230)
        unicode_271722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 18), 'unicode', u'*')
        int_271723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 22), 'int')
        # Applying the binary operator '*' (line 230)
        result_mul_271724 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 18), '*', unicode_271722, int_271723)
        
        # Processing the call keyword arguments (line 230)
        kwargs_271725 = {}
        # Getting the type of 'print' (line 230)
        print_271721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'print', False)
        # Calling print(args, kwargs) (line 230)
        print_call_result_271726 = invoke(stypy.reporting.localization.Localization(__file__, 230, 12), print_271721, *[result_mul_271724], **kwargs_271725)
        
        # SSA join for if statement (line 225)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_dialog(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_dialog'
        module_type_store = module_type_store.open_function_context('get_dialog', 232, 4, False)
        # Assigning a type to the variable 'self' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FormWidget.get_dialog.__dict__.__setitem__('stypy_localization', localization)
        FormWidget.get_dialog.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FormWidget.get_dialog.__dict__.__setitem__('stypy_type_store', module_type_store)
        FormWidget.get_dialog.__dict__.__setitem__('stypy_function_name', 'FormWidget.get_dialog')
        FormWidget.get_dialog.__dict__.__setitem__('stypy_param_names_list', [])
        FormWidget.get_dialog.__dict__.__setitem__('stypy_varargs_param_name', None)
        FormWidget.get_dialog.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FormWidget.get_dialog.__dict__.__setitem__('stypy_call_defaults', defaults)
        FormWidget.get_dialog.__dict__.__setitem__('stypy_call_varargs', varargs)
        FormWidget.get_dialog.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FormWidget.get_dialog.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FormWidget.get_dialog', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_dialog', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_dialog(...)' code ##################

        unicode_271727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 8), 'unicode', u'Return FormDialog instance')
        
        # Assigning a Call to a Name (line 234):
        
        # Assigning a Call to a Name (line 234):
        
        # Call to parent(...): (line 234)
        # Processing the call keyword arguments (line 234)
        kwargs_271730 = {}
        # Getting the type of 'self' (line 234)
        self_271728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 17), 'self', False)
        # Obtaining the member 'parent' of a type (line 234)
        parent_271729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 17), self_271728, 'parent')
        # Calling parent(args, kwargs) (line 234)
        parent_call_result_271731 = invoke(stypy.reporting.localization.Localization(__file__, 234, 17), parent_271729, *[], **kwargs_271730)
        
        # Assigning a type to the variable 'dialog' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'dialog', parent_call_result_271731)
        
        
        
        # Call to isinstance(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'dialog' (line 235)
        dialog_271733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 29), 'dialog', False)
        # Getting the type of 'QtWidgets' (line 235)
        QtWidgets_271734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 37), 'QtWidgets', False)
        # Obtaining the member 'QDialog' of a type (line 235)
        QDialog_271735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 37), QtWidgets_271734, 'QDialog')
        # Processing the call keyword arguments (line 235)
        kwargs_271736 = {}
        # Getting the type of 'isinstance' (line 235)
        isinstance_271732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 18), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 235)
        isinstance_call_result_271737 = invoke(stypy.reporting.localization.Localization(__file__, 235, 18), isinstance_271732, *[dialog_271733, QDialog_271735], **kwargs_271736)
        
        # Applying the 'not' unary operator (line 235)
        result_not__271738 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 14), 'not', isinstance_call_result_271737)
        
        # Testing the type of an if condition (line 235)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 8), result_not__271738)
        # SSA begins for while statement (line 235)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 236):
        
        # Assigning a Call to a Name (line 236):
        
        # Call to parent(...): (line 236)
        # Processing the call keyword arguments (line 236)
        kwargs_271741 = {}
        # Getting the type of 'dialog' (line 236)
        dialog_271739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 21), 'dialog', False)
        # Obtaining the member 'parent' of a type (line 236)
        parent_271740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 21), dialog_271739, 'parent')
        # Calling parent(args, kwargs) (line 236)
        parent_call_result_271742 = invoke(stypy.reporting.localization.Localization(__file__, 236, 21), parent_271740, *[], **kwargs_271741)
        
        # Assigning a type to the variable 'dialog' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'dialog', parent_call_result_271742)
        # SSA join for while statement (line 235)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'dialog' (line 237)
        dialog_271743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 15), 'dialog')
        # Assigning a type to the variable 'stypy_return_type' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'stypy_return_type', dialog_271743)
        
        # ################# End of 'get_dialog(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_dialog' in the type store
        # Getting the type of 'stypy_return_type' (line 232)
        stypy_return_type_271744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_271744)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_dialog'
        return stypy_return_type_271744


    @norecursion
    def setup(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup'
        module_type_store = module_type_store.open_function_context('setup', 239, 4, False)
        # Assigning a type to the variable 'self' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FormWidget.setup.__dict__.__setitem__('stypy_localization', localization)
        FormWidget.setup.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FormWidget.setup.__dict__.__setitem__('stypy_type_store', module_type_store)
        FormWidget.setup.__dict__.__setitem__('stypy_function_name', 'FormWidget.setup')
        FormWidget.setup.__dict__.__setitem__('stypy_param_names_list', [])
        FormWidget.setup.__dict__.__setitem__('stypy_varargs_param_name', None)
        FormWidget.setup.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FormWidget.setup.__dict__.__setitem__('stypy_call_defaults', defaults)
        FormWidget.setup.__dict__.__setitem__('stypy_call_varargs', varargs)
        FormWidget.setup.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FormWidget.setup.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FormWidget.setup', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup(...)' code ##################

        
        # Getting the type of 'self' (line 240)
        self_271745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 28), 'self')
        # Obtaining the member 'data' of a type (line 240)
        data_271746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 28), self_271745, 'data')
        # Testing the type of a for loop iterable (line 240)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 240, 8), data_271746)
        # Getting the type of the for loop variable (line 240)
        for_loop_var_271747 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 240, 8), data_271746)
        # Assigning a type to the variable 'label' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'label', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 8), for_loop_var_271747))
        # Assigning a type to the variable 'value' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 8), for_loop_var_271747))
        # SSA begins for a for statement (line 240)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'DEBUG' (line 241)
        DEBUG_271748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 15), 'DEBUG')
        # Testing the type of an if condition (line 241)
        if_condition_271749 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 12), DEBUG_271748)
        # Assigning a type to the variable 'if_condition_271749' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'if_condition_271749', if_condition_271749)
        # SSA begins for if statement (line 241)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 242)
        # Processing the call arguments (line 242)
        unicode_271751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 22), 'unicode', u'value:')
        # Getting the type of 'value' (line 242)
        value_271752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 32), 'value', False)
        # Processing the call keyword arguments (line 242)
        kwargs_271753 = {}
        # Getting the type of 'print' (line 242)
        print_271750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'print', False)
        # Calling print(args, kwargs) (line 242)
        print_call_result_271754 = invoke(stypy.reporting.localization.Localization(__file__, 242, 16), print_271750, *[unicode_271751, value_271752], **kwargs_271753)
        
        # SSA join for if statement (line 241)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'label' (line 243)
        label_271755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 15), 'label')
        # Getting the type of 'None' (line 243)
        None_271756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 24), 'None')
        # Applying the binary operator 'is' (line 243)
        result_is__271757 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 15), 'is', label_271755, None_271756)
        
        
        # Getting the type of 'value' (line 243)
        value_271758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 33), 'value')
        # Getting the type of 'None' (line 243)
        None_271759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 42), 'None')
        # Applying the binary operator 'is' (line 243)
        result_is__271760 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 33), 'is', value_271758, None_271759)
        
        # Applying the binary operator 'and' (line 243)
        result_and_keyword_271761 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 15), 'and', result_is__271757, result_is__271760)
        
        # Testing the type of an if condition (line 243)
        if_condition_271762 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 243, 12), result_and_keyword_271761)
        # Assigning a type to the variable 'if_condition_271762' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'if_condition_271762', if_condition_271762)
        # SSA begins for if statement (line 243)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to addRow(...): (line 245)
        # Processing the call arguments (line 245)
        
        # Call to QLabel(...): (line 245)
        # Processing the call arguments (line 245)
        unicode_271768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 56), 'unicode', u' ')
        # Processing the call keyword arguments (line 245)
        kwargs_271769 = {}
        # Getting the type of 'QtWidgets' (line 245)
        QtWidgets_271766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 39), 'QtWidgets', False)
        # Obtaining the member 'QLabel' of a type (line 245)
        QLabel_271767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 39), QtWidgets_271766, 'QLabel')
        # Calling QLabel(args, kwargs) (line 245)
        QLabel_call_result_271770 = invoke(stypy.reporting.localization.Localization(__file__, 245, 39), QLabel_271767, *[unicode_271768], **kwargs_271769)
        
        
        # Call to QLabel(...): (line 245)
        # Processing the call arguments (line 245)
        unicode_271773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 79), 'unicode', u' ')
        # Processing the call keyword arguments (line 245)
        kwargs_271774 = {}
        # Getting the type of 'QtWidgets' (line 245)
        QtWidgets_271771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 62), 'QtWidgets', False)
        # Obtaining the member 'QLabel' of a type (line 245)
        QLabel_271772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 62), QtWidgets_271771, 'QLabel')
        # Calling QLabel(args, kwargs) (line 245)
        QLabel_call_result_271775 = invoke(stypy.reporting.localization.Localization(__file__, 245, 62), QLabel_271772, *[unicode_271773], **kwargs_271774)
        
        # Processing the call keyword arguments (line 245)
        kwargs_271776 = {}
        # Getting the type of 'self' (line 245)
        self_271763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 16), 'self', False)
        # Obtaining the member 'formlayout' of a type (line 245)
        formlayout_271764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 16), self_271763, 'formlayout')
        # Obtaining the member 'addRow' of a type (line 245)
        addRow_271765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 16), formlayout_271764, 'addRow')
        # Calling addRow(args, kwargs) (line 245)
        addRow_call_result_271777 = invoke(stypy.reporting.localization.Localization(__file__, 245, 16), addRow_271765, *[QLabel_call_result_271770, QLabel_call_result_271775], **kwargs_271776)
        
        
        # Call to append(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'None' (line 246)
        None_271781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 36), 'None', False)
        # Processing the call keyword arguments (line 246)
        kwargs_271782 = {}
        # Getting the type of 'self' (line 246)
        self_271778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'self', False)
        # Obtaining the member 'widgets' of a type (line 246)
        widgets_271779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 16), self_271778, 'widgets')
        # Obtaining the member 'append' of a type (line 246)
        append_271780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 16), widgets_271779, 'append')
        # Calling append(args, kwargs) (line 246)
        append_call_result_271783 = invoke(stypy.reporting.localization.Localization(__file__, 246, 16), append_271780, *[None_271781], **kwargs_271782)
        
        # SSA branch for the else part of an if statement (line 243)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 248)
        # Getting the type of 'label' (line 248)
        label_271784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 17), 'label')
        # Getting the type of 'None' (line 248)
        None_271785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 26), 'None')
        
        (may_be_271786, more_types_in_union_271787) = may_be_none(label_271784, None_271785)

        if may_be_271786:

            if more_types_in_union_271787:
                # Runtime conditional SSA (line 248)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to addRow(...): (line 250)
            # Processing the call arguments (line 250)
            
            # Call to QLabel(...): (line 250)
            # Processing the call arguments (line 250)
            # Getting the type of 'value' (line 250)
            value_271793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 56), 'value', False)
            # Processing the call keyword arguments (line 250)
            kwargs_271794 = {}
            # Getting the type of 'QtWidgets' (line 250)
            QtWidgets_271791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 39), 'QtWidgets', False)
            # Obtaining the member 'QLabel' of a type (line 250)
            QLabel_271792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 39), QtWidgets_271791, 'QLabel')
            # Calling QLabel(args, kwargs) (line 250)
            QLabel_call_result_271795 = invoke(stypy.reporting.localization.Localization(__file__, 250, 39), QLabel_271792, *[value_271793], **kwargs_271794)
            
            # Processing the call keyword arguments (line 250)
            kwargs_271796 = {}
            # Getting the type of 'self' (line 250)
            self_271788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'self', False)
            # Obtaining the member 'formlayout' of a type (line 250)
            formlayout_271789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 16), self_271788, 'formlayout')
            # Obtaining the member 'addRow' of a type (line 250)
            addRow_271790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 16), formlayout_271789, 'addRow')
            # Calling addRow(args, kwargs) (line 250)
            addRow_call_result_271797 = invoke(stypy.reporting.localization.Localization(__file__, 250, 16), addRow_271790, *[QLabel_call_result_271795], **kwargs_271796)
            
            
            # Call to append(...): (line 251)
            # Processing the call arguments (line 251)
            # Getting the type of 'None' (line 251)
            None_271801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 36), 'None', False)
            # Processing the call keyword arguments (line 251)
            kwargs_271802 = {}
            # Getting the type of 'self' (line 251)
            self_271798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 16), 'self', False)
            # Obtaining the member 'widgets' of a type (line 251)
            widgets_271799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 16), self_271798, 'widgets')
            # Obtaining the member 'append' of a type (line 251)
            append_271800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 16), widgets_271799, 'append')
            # Calling append(args, kwargs) (line 251)
            append_call_result_271803 = invoke(stypy.reporting.localization.Localization(__file__, 251, 16), append_271800, *[None_271801], **kwargs_271802)
            

            if more_types_in_union_271787:
                # Runtime conditional SSA for else branch (line 248)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_271786) or more_types_in_union_271787):
            
            
            
            # Call to tuple_to_qfont(...): (line 253)
            # Processing the call arguments (line 253)
            # Getting the type of 'value' (line 253)
            value_271805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 32), 'value', False)
            # Processing the call keyword arguments (line 253)
            kwargs_271806 = {}
            # Getting the type of 'tuple_to_qfont' (line 253)
            tuple_to_qfont_271804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 17), 'tuple_to_qfont', False)
            # Calling tuple_to_qfont(args, kwargs) (line 253)
            tuple_to_qfont_call_result_271807 = invoke(stypy.reporting.localization.Localization(__file__, 253, 17), tuple_to_qfont_271804, *[value_271805], **kwargs_271806)
            
            # Getting the type of 'None' (line 253)
            None_271808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 46), 'None')
            # Applying the binary operator 'isnot' (line 253)
            result_is_not_271809 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 17), 'isnot', tuple_to_qfont_call_result_271807, None_271808)
            
            # Testing the type of an if condition (line 253)
            if_condition_271810 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 253, 17), result_is_not_271809)
            # Assigning a type to the variable 'if_condition_271810' (line 253)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 17), 'if_condition_271810', if_condition_271810)
            # SSA begins for if statement (line 253)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 254):
            
            # Assigning a Call to a Name (line 254):
            
            # Call to FontLayout(...): (line 254)
            # Processing the call arguments (line 254)
            # Getting the type of 'value' (line 254)
            value_271812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 35), 'value', False)
            # Getting the type of 'self' (line 254)
            self_271813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 42), 'self', False)
            # Processing the call keyword arguments (line 254)
            kwargs_271814 = {}
            # Getting the type of 'FontLayout' (line 254)
            FontLayout_271811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 24), 'FontLayout', False)
            # Calling FontLayout(args, kwargs) (line 254)
            FontLayout_call_result_271815 = invoke(stypy.reporting.localization.Localization(__file__, 254, 24), FontLayout_271811, *[value_271812, self_271813], **kwargs_271814)
            
            # Assigning a type to the variable 'field' (line 254)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'field', FontLayout_call_result_271815)
            # SSA branch for the else part of an if statement (line 253)
            module_type_store.open_ssa_branch('else')
            
            
            # Evaluating a boolean operation
            
            
            # Call to lower(...): (line 255)
            # Processing the call keyword arguments (line 255)
            kwargs_271818 = {}
            # Getting the type of 'label' (line 255)
            label_271816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 18), 'label', False)
            # Obtaining the member 'lower' of a type (line 255)
            lower_271817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 18), label_271816, 'lower')
            # Calling lower(args, kwargs) (line 255)
            lower_call_result_271819 = invoke(stypy.reporting.localization.Localization(__file__, 255, 18), lower_271817, *[], **kwargs_271818)
            
            # Getting the type of 'BLACKLIST' (line 255)
            BLACKLIST_271820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 39), 'BLACKLIST')
            # Applying the binary operator 'notin' (line 255)
            result_contains_271821 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 18), 'notin', lower_call_result_271819, BLACKLIST_271820)
            
            
            # Call to is_color_like(...): (line 256)
            # Processing the call arguments (line 256)
            # Getting the type of 'value' (line 256)
            value_271824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 44), 'value', False)
            # Processing the call keyword arguments (line 256)
            kwargs_271825 = {}
            # Getting the type of 'mcolors' (line 256)
            mcolors_271822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 22), 'mcolors', False)
            # Obtaining the member 'is_color_like' of a type (line 256)
            is_color_like_271823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 22), mcolors_271822, 'is_color_like')
            # Calling is_color_like(args, kwargs) (line 256)
            is_color_like_call_result_271826 = invoke(stypy.reporting.localization.Localization(__file__, 256, 22), is_color_like_271823, *[value_271824], **kwargs_271825)
            
            # Applying the binary operator 'and' (line 255)
            result_and_keyword_271827 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 18), 'and', result_contains_271821, is_color_like_call_result_271826)
            
            # Testing the type of an if condition (line 255)
            if_condition_271828 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 17), result_and_keyword_271827)
            # Assigning a type to the variable 'if_condition_271828' (line 255)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 17), 'if_condition_271828', if_condition_271828)
            # SSA begins for if statement (line 255)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 257):
            
            # Assigning a Call to a Name (line 257):
            
            # Call to ColorLayout(...): (line 257)
            # Processing the call arguments (line 257)
            
            # Call to to_qcolor(...): (line 257)
            # Processing the call arguments (line 257)
            # Getting the type of 'value' (line 257)
            value_271831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 46), 'value', False)
            # Processing the call keyword arguments (line 257)
            kwargs_271832 = {}
            # Getting the type of 'to_qcolor' (line 257)
            to_qcolor_271830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 36), 'to_qcolor', False)
            # Calling to_qcolor(args, kwargs) (line 257)
            to_qcolor_call_result_271833 = invoke(stypy.reporting.localization.Localization(__file__, 257, 36), to_qcolor_271830, *[value_271831], **kwargs_271832)
            
            # Getting the type of 'self' (line 257)
            self_271834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 54), 'self', False)
            # Processing the call keyword arguments (line 257)
            kwargs_271835 = {}
            # Getting the type of 'ColorLayout' (line 257)
            ColorLayout_271829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 24), 'ColorLayout', False)
            # Calling ColorLayout(args, kwargs) (line 257)
            ColorLayout_call_result_271836 = invoke(stypy.reporting.localization.Localization(__file__, 257, 24), ColorLayout_271829, *[to_qcolor_call_result_271833, self_271834], **kwargs_271835)
            
            # Assigning a type to the variable 'field' (line 257)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'field', ColorLayout_call_result_271836)
            # SSA branch for the else part of an if statement (line 255)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isinstance(...): (line 258)
            # Processing the call arguments (line 258)
            # Getting the type of 'value' (line 258)
            value_271838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 28), 'value', False)
            # Getting the type of 'six' (line 258)
            six_271839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 35), 'six', False)
            # Obtaining the member 'string_types' of a type (line 258)
            string_types_271840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 35), six_271839, 'string_types')
            # Processing the call keyword arguments (line 258)
            kwargs_271841 = {}
            # Getting the type of 'isinstance' (line 258)
            isinstance_271837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 17), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 258)
            isinstance_call_result_271842 = invoke(stypy.reporting.localization.Localization(__file__, 258, 17), isinstance_271837, *[value_271838, string_types_271840], **kwargs_271841)
            
            # Testing the type of an if condition (line 258)
            if_condition_271843 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 258, 17), isinstance_call_result_271842)
            # Assigning a type to the variable 'if_condition_271843' (line 258)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 17), 'if_condition_271843', if_condition_271843)
            # SSA begins for if statement (line 258)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 259):
            
            # Assigning a Call to a Name (line 259):
            
            # Call to QLineEdit(...): (line 259)
            # Processing the call arguments (line 259)
            # Getting the type of 'value' (line 259)
            value_271846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 44), 'value', False)
            # Getting the type of 'self' (line 259)
            self_271847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 51), 'self', False)
            # Processing the call keyword arguments (line 259)
            kwargs_271848 = {}
            # Getting the type of 'QtWidgets' (line 259)
            QtWidgets_271844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 24), 'QtWidgets', False)
            # Obtaining the member 'QLineEdit' of a type (line 259)
            QLineEdit_271845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 24), QtWidgets_271844, 'QLineEdit')
            # Calling QLineEdit(args, kwargs) (line 259)
            QLineEdit_call_result_271849 = invoke(stypy.reporting.localization.Localization(__file__, 259, 24), QLineEdit_271845, *[value_271846, self_271847], **kwargs_271848)
            
            # Assigning a type to the variable 'field' (line 259)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 16), 'field', QLineEdit_call_result_271849)
            # SSA branch for the else part of an if statement (line 258)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isinstance(...): (line 260)
            # Processing the call arguments (line 260)
            # Getting the type of 'value' (line 260)
            value_271851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 28), 'value', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 260)
            tuple_271852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 36), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 260)
            # Adding element type (line 260)
            # Getting the type of 'list' (line 260)
            list_271853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 36), 'list', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 36), tuple_271852, list_271853)
            # Adding element type (line 260)
            # Getting the type of 'tuple' (line 260)
            tuple_271854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 42), 'tuple', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 36), tuple_271852, tuple_271854)
            
            # Processing the call keyword arguments (line 260)
            kwargs_271855 = {}
            # Getting the type of 'isinstance' (line 260)
            isinstance_271850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 17), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 260)
            isinstance_call_result_271856 = invoke(stypy.reporting.localization.Localization(__file__, 260, 17), isinstance_271850, *[value_271851, tuple_271852], **kwargs_271855)
            
            # Testing the type of an if condition (line 260)
            if_condition_271857 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 17), isinstance_call_result_271856)
            # Assigning a type to the variable 'if_condition_271857' (line 260)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 17), 'if_condition_271857', if_condition_271857)
            # SSA begins for if statement (line 260)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Type idiom detected: calculating its left and rigth part (line 261)
            # Getting the type of 'tuple' (line 261)
            tuple_271858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 37), 'tuple')
            # Getting the type of 'value' (line 261)
            value_271859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 30), 'value')
            
            (may_be_271860, more_types_in_union_271861) = may_be_subtype(tuple_271858, value_271859)

            if may_be_271860:

                if more_types_in_union_271861:
                    # Runtime conditional SSA (line 261)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'value' (line 261)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 16), 'value', remove_not_subtype_from_union(value_271859, tuple))
                
                # Assigning a Call to a Name (line 262):
                
                # Assigning a Call to a Name (line 262):
                
                # Call to list(...): (line 262)
                # Processing the call arguments (line 262)
                # Getting the type of 'value' (line 262)
                value_271863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 33), 'value', False)
                # Processing the call keyword arguments (line 262)
                kwargs_271864 = {}
                # Getting the type of 'list' (line 262)
                list_271862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 28), 'list', False)
                # Calling list(args, kwargs) (line 262)
                list_call_result_271865 = invoke(stypy.reporting.localization.Localization(__file__, 262, 28), list_271862, *[value_271863], **kwargs_271864)
                
                # Assigning a type to the variable 'value' (line 262)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 20), 'value', list_call_result_271865)

                if more_types_in_union_271861:
                    # SSA join for if statement (line 261)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a Call to a Name (line 263):
            
            # Assigning a Call to a Name (line 263):
            
            # Call to pop(...): (line 263)
            # Processing the call arguments (line 263)
            int_271868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 37), 'int')
            # Processing the call keyword arguments (line 263)
            kwargs_271869 = {}
            # Getting the type of 'value' (line 263)
            value_271866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 27), 'value', False)
            # Obtaining the member 'pop' of a type (line 263)
            pop_271867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 27), value_271866, 'pop')
            # Calling pop(args, kwargs) (line 263)
            pop_call_result_271870 = invoke(stypy.reporting.localization.Localization(__file__, 263, 27), pop_271867, *[int_271868], **kwargs_271869)
            
            # Assigning a type to the variable 'selindex' (line 263)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 16), 'selindex', pop_call_result_271870)
            
            # Assigning a Call to a Name (line 264):
            
            # Assigning a Call to a Name (line 264):
            
            # Call to QComboBox(...): (line 264)
            # Processing the call arguments (line 264)
            # Getting the type of 'self' (line 264)
            self_271873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 44), 'self', False)
            # Processing the call keyword arguments (line 264)
            kwargs_271874 = {}
            # Getting the type of 'QtWidgets' (line 264)
            QtWidgets_271871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 24), 'QtWidgets', False)
            # Obtaining the member 'QComboBox' of a type (line 264)
            QComboBox_271872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 24), QtWidgets_271871, 'QComboBox')
            # Calling QComboBox(args, kwargs) (line 264)
            QComboBox_call_result_271875 = invoke(stypy.reporting.localization.Localization(__file__, 264, 24), QComboBox_271872, *[self_271873], **kwargs_271874)
            
            # Assigning a type to the variable 'field' (line 264)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 16), 'field', QComboBox_call_result_271875)
            
            
            # Call to isinstance(...): (line 265)
            # Processing the call arguments (line 265)
            
            # Obtaining the type of the subscript
            int_271877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 36), 'int')
            # Getting the type of 'value' (line 265)
            value_271878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 30), 'value', False)
            # Obtaining the member '__getitem__' of a type (line 265)
            getitem___271879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 30), value_271878, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 265)
            subscript_call_result_271880 = invoke(stypy.reporting.localization.Localization(__file__, 265, 30), getitem___271879, int_271877)
            
            
            # Obtaining an instance of the builtin type 'tuple' (line 265)
            tuple_271881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 41), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 265)
            # Adding element type (line 265)
            # Getting the type of 'list' (line 265)
            list_271882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 41), 'list', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 41), tuple_271881, list_271882)
            # Adding element type (line 265)
            # Getting the type of 'tuple' (line 265)
            tuple_271883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 47), 'tuple', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 41), tuple_271881, tuple_271883)
            
            # Processing the call keyword arguments (line 265)
            kwargs_271884 = {}
            # Getting the type of 'isinstance' (line 265)
            isinstance_271876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 265)
            isinstance_call_result_271885 = invoke(stypy.reporting.localization.Localization(__file__, 265, 19), isinstance_271876, *[subscript_call_result_271880, tuple_271881], **kwargs_271884)
            
            # Testing the type of an if condition (line 265)
            if_condition_271886 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 16), isinstance_call_result_271885)
            # Assigning a type to the variable 'if_condition_271886' (line 265)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 16), 'if_condition_271886', if_condition_271886)
            # SSA begins for if statement (line 265)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a ListComp to a Name (line 266):
            
            # Assigning a ListComp to a Name (line 266):
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'value' (line 266)
            value_271888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 49), 'value')
            comprehension_271889 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 28), value_271888)
            # Assigning a type to the variable 'key' (line 266)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 28), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 28), comprehension_271889))
            # Assigning a type to the variable '_val' (line 266)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 28), '_val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 28), comprehension_271889))
            # Getting the type of 'key' (line 266)
            key_271887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 28), 'key')
            list_271890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 28), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 28), list_271890, key_271887)
            # Assigning a type to the variable 'keys' (line 266)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 20), 'keys', list_271890)
            
            # Assigning a ListComp to a Name (line 267):
            
            # Assigning a ListComp to a Name (line 267):
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'value' (line 267)
            value_271892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 50), 'value')
            comprehension_271893 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 29), value_271892)
            # Assigning a type to the variable '_key' (line 267)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 29), '_key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 29), comprehension_271893))
            # Assigning a type to the variable 'val' (line 267)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 29), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 29), comprehension_271893))
            # Getting the type of 'val' (line 267)
            val_271891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 29), 'val')
            list_271894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 29), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 29), list_271894, val_271891)
            # Assigning a type to the variable 'value' (line 267)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 20), 'value', list_271894)
            # SSA branch for the else part of an if statement (line 265)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Name (line 269):
            
            # Assigning a Name to a Name (line 269):
            # Getting the type of 'value' (line 269)
            value_271895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 27), 'value')
            # Assigning a type to the variable 'keys' (line 269)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 20), 'keys', value_271895)
            # SSA join for if statement (line 265)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to addItems(...): (line 270)
            # Processing the call arguments (line 270)
            # Getting the type of 'value' (line 270)
            value_271898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 31), 'value', False)
            # Processing the call keyword arguments (line 270)
            kwargs_271899 = {}
            # Getting the type of 'field' (line 270)
            field_271896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 16), 'field', False)
            # Obtaining the member 'addItems' of a type (line 270)
            addItems_271897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 16), field_271896, 'addItems')
            # Calling addItems(args, kwargs) (line 270)
            addItems_call_result_271900 = invoke(stypy.reporting.localization.Localization(__file__, 270, 16), addItems_271897, *[value_271898], **kwargs_271899)
            
            
            
            # Getting the type of 'selindex' (line 271)
            selindex_271901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 19), 'selindex')
            # Getting the type of 'value' (line 271)
            value_271902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 31), 'value')
            # Applying the binary operator 'in' (line 271)
            result_contains_271903 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 19), 'in', selindex_271901, value_271902)
            
            # Testing the type of an if condition (line 271)
            if_condition_271904 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 271, 16), result_contains_271903)
            # Assigning a type to the variable 'if_condition_271904' (line 271)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'if_condition_271904', if_condition_271904)
            # SSA begins for if statement (line 271)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 272):
            
            # Assigning a Call to a Name (line 272):
            
            # Call to index(...): (line 272)
            # Processing the call arguments (line 272)
            # Getting the type of 'selindex' (line 272)
            selindex_271907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 43), 'selindex', False)
            # Processing the call keyword arguments (line 272)
            kwargs_271908 = {}
            # Getting the type of 'value' (line 272)
            value_271905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 31), 'value', False)
            # Obtaining the member 'index' of a type (line 272)
            index_271906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 31), value_271905, 'index')
            # Calling index(args, kwargs) (line 272)
            index_call_result_271909 = invoke(stypy.reporting.localization.Localization(__file__, 272, 31), index_271906, *[selindex_271907], **kwargs_271908)
            
            # Assigning a type to the variable 'selindex' (line 272)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 20), 'selindex', index_call_result_271909)
            # SSA branch for the else part of an if statement (line 271)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'selindex' (line 273)
            selindex_271910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 21), 'selindex')
            # Getting the type of 'keys' (line 273)
            keys_271911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 33), 'keys')
            # Applying the binary operator 'in' (line 273)
            result_contains_271912 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 21), 'in', selindex_271910, keys_271911)
            
            # Testing the type of an if condition (line 273)
            if_condition_271913 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 21), result_contains_271912)
            # Assigning a type to the variable 'if_condition_271913' (line 273)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 21), 'if_condition_271913', if_condition_271913)
            # SSA begins for if statement (line 273)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 274):
            
            # Assigning a Call to a Name (line 274):
            
            # Call to index(...): (line 274)
            # Processing the call arguments (line 274)
            # Getting the type of 'selindex' (line 274)
            selindex_271916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 42), 'selindex', False)
            # Processing the call keyword arguments (line 274)
            kwargs_271917 = {}
            # Getting the type of 'keys' (line 274)
            keys_271914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 31), 'keys', False)
            # Obtaining the member 'index' of a type (line 274)
            index_271915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 31), keys_271914, 'index')
            # Calling index(args, kwargs) (line 274)
            index_call_result_271918 = invoke(stypy.reporting.localization.Localization(__file__, 274, 31), index_271915, *[selindex_271916], **kwargs_271917)
            
            # Assigning a type to the variable 'selindex' (line 274)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 20), 'selindex', index_call_result_271918)
            # SSA branch for the else part of an if statement (line 273)
            module_type_store.open_ssa_branch('else')
            
            # Type idiom detected: calculating its left and rigth part (line 275)
            # Getting the type of 'int' (line 275)
            int_271919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 46), 'int')
            # Getting the type of 'selindex' (line 275)
            selindex_271920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 36), 'selindex')
            
            (may_be_271921, more_types_in_union_271922) = may_not_be_subtype(int_271919, selindex_271920)

            if may_be_271921:

                if more_types_in_union_271922:
                    # Runtime conditional SSA (line 275)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'selindex' (line 275)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 21), 'selindex', remove_subtype_from_union(selindex_271920, int))
                
                # Call to warn(...): (line 276)
                # Processing the call arguments (line 276)
                unicode_271925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 24), 'unicode', u"index '%s' is invalid (label: %s, value: %s)")
                
                # Obtaining an instance of the builtin type 'tuple' (line 278)
                tuple_271926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 25), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 278)
                # Adding element type (line 278)
                # Getting the type of 'selindex' (line 278)
                selindex_271927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 25), 'selindex', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 25), tuple_271926, selindex_271927)
                # Adding element type (line 278)
                # Getting the type of 'label' (line 278)
                label_271928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 35), 'label', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 25), tuple_271926, label_271928)
                # Adding element type (line 278)
                # Getting the type of 'value' (line 278)
                value_271929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 42), 'value', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 25), tuple_271926, value_271929)
                
                # Applying the binary operator '%' (line 277)
                result_mod_271930 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 24), '%', unicode_271925, tuple_271926)
                
                # Processing the call keyword arguments (line 276)
                kwargs_271931 = {}
                # Getting the type of 'warnings' (line 276)
                warnings_271923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 20), 'warnings', False)
                # Obtaining the member 'warn' of a type (line 276)
                warn_271924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 20), warnings_271923, 'warn')
                # Calling warn(args, kwargs) (line 276)
                warn_call_result_271932 = invoke(stypy.reporting.localization.Localization(__file__, 276, 20), warn_271924, *[result_mod_271930], **kwargs_271931)
                
                
                # Assigning a Num to a Name (line 279):
                
                # Assigning a Num to a Name (line 279):
                int_271933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 31), 'int')
                # Assigning a type to the variable 'selindex' (line 279)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 20), 'selindex', int_271933)

                if more_types_in_union_271922:
                    # SSA join for if statement (line 275)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 273)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 271)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to setCurrentIndex(...): (line 280)
            # Processing the call arguments (line 280)
            # Getting the type of 'selindex' (line 280)
            selindex_271936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 38), 'selindex', False)
            # Processing the call keyword arguments (line 280)
            kwargs_271937 = {}
            # Getting the type of 'field' (line 280)
            field_271934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'field', False)
            # Obtaining the member 'setCurrentIndex' of a type (line 280)
            setCurrentIndex_271935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 16), field_271934, 'setCurrentIndex')
            # Calling setCurrentIndex(args, kwargs) (line 280)
            setCurrentIndex_call_result_271938 = invoke(stypy.reporting.localization.Localization(__file__, 280, 16), setCurrentIndex_271935, *[selindex_271936], **kwargs_271937)
            
            # SSA branch for the else part of an if statement (line 260)
            module_type_store.open_ssa_branch('else')
            
            # Type idiom detected: calculating its left and rigth part (line 281)
            # Getting the type of 'bool' (line 281)
            bool_271939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 35), 'bool')
            # Getting the type of 'value' (line 281)
            value_271940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 28), 'value')
            
            (may_be_271941, more_types_in_union_271942) = may_be_subtype(bool_271939, value_271940)

            if may_be_271941:

                if more_types_in_union_271942:
                    # Runtime conditional SSA (line 281)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'value' (line 281)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 17), 'value', remove_not_subtype_from_union(value_271940, bool))
                
                # Assigning a Call to a Name (line 282):
                
                # Assigning a Call to a Name (line 282):
                
                # Call to QCheckBox(...): (line 282)
                # Processing the call arguments (line 282)
                # Getting the type of 'self' (line 282)
                self_271945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 44), 'self', False)
                # Processing the call keyword arguments (line 282)
                kwargs_271946 = {}
                # Getting the type of 'QtWidgets' (line 282)
                QtWidgets_271943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 24), 'QtWidgets', False)
                # Obtaining the member 'QCheckBox' of a type (line 282)
                QCheckBox_271944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 24), QtWidgets_271943, 'QCheckBox')
                # Calling QCheckBox(args, kwargs) (line 282)
                QCheckBox_call_result_271947 = invoke(stypy.reporting.localization.Localization(__file__, 282, 24), QCheckBox_271944, *[self_271945], **kwargs_271946)
                
                # Assigning a type to the variable 'field' (line 282)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'field', QCheckBox_call_result_271947)
                
                # Getting the type of 'value' (line 283)
                value_271948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 19), 'value')
                # Testing the type of an if condition (line 283)
                if_condition_271949 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 283, 16), value_271948)
                # Assigning a type to the variable 'if_condition_271949' (line 283)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'if_condition_271949', if_condition_271949)
                # SSA begins for if statement (line 283)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to setCheckState(...): (line 284)
                # Processing the call arguments (line 284)
                # Getting the type of 'QtCore' (line 284)
                QtCore_271952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 40), 'QtCore', False)
                # Obtaining the member 'Qt' of a type (line 284)
                Qt_271953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 40), QtCore_271952, 'Qt')
                # Obtaining the member 'Checked' of a type (line 284)
                Checked_271954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 40), Qt_271953, 'Checked')
                # Processing the call keyword arguments (line 284)
                kwargs_271955 = {}
                # Getting the type of 'field' (line 284)
                field_271950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 20), 'field', False)
                # Obtaining the member 'setCheckState' of a type (line 284)
                setCheckState_271951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 20), field_271950, 'setCheckState')
                # Calling setCheckState(args, kwargs) (line 284)
                setCheckState_call_result_271956 = invoke(stypy.reporting.localization.Localization(__file__, 284, 20), setCheckState_271951, *[Checked_271954], **kwargs_271955)
                
                # SSA branch for the else part of an if statement (line 283)
                module_type_store.open_ssa_branch('else')
                
                # Call to setCheckState(...): (line 286)
                # Processing the call arguments (line 286)
                # Getting the type of 'QtCore' (line 286)
                QtCore_271959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 40), 'QtCore', False)
                # Obtaining the member 'Qt' of a type (line 286)
                Qt_271960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 40), QtCore_271959, 'Qt')
                # Obtaining the member 'Unchecked' of a type (line 286)
                Unchecked_271961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 40), Qt_271960, 'Unchecked')
                # Processing the call keyword arguments (line 286)
                kwargs_271962 = {}
                # Getting the type of 'field' (line 286)
                field_271957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 20), 'field', False)
                # Obtaining the member 'setCheckState' of a type (line 286)
                setCheckState_271958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 20), field_271957, 'setCheckState')
                # Calling setCheckState(args, kwargs) (line 286)
                setCheckState_call_result_271963 = invoke(stypy.reporting.localization.Localization(__file__, 286, 20), setCheckState_271958, *[Unchecked_271961], **kwargs_271962)
                
                # SSA join for if statement (line 283)
                module_type_store = module_type_store.join_ssa_context()
                

                if more_types_in_union_271942:
                    # Runtime conditional SSA for else branch (line 281)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_271941) or more_types_in_union_271942):
                # Assigning a type to the variable 'value' (line 281)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 17), 'value', remove_subtype_from_union(value_271940, bool))
                
                # Type idiom detected: calculating its left and rigth part (line 287)
                # Getting the type of 'float' (line 287)
                float_271964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 35), 'float')
                # Getting the type of 'value' (line 287)
                value_271965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 28), 'value')
                
                (may_be_271966, more_types_in_union_271967) = may_be_subtype(float_271964, value_271965)

                if may_be_271966:

                    if more_types_in_union_271967:
                        # Runtime conditional SSA (line 287)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Assigning a type to the variable 'value' (line 287)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 17), 'value', remove_not_subtype_from_union(value_271965, float))
                    
                    # Assigning a Call to a Name (line 288):
                    
                    # Assigning a Call to a Name (line 288):
                    
                    # Call to QLineEdit(...): (line 288)
                    # Processing the call arguments (line 288)
                    
                    # Call to repr(...): (line 288)
                    # Processing the call arguments (line 288)
                    # Getting the type of 'value' (line 288)
                    value_271971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 49), 'value', False)
                    # Processing the call keyword arguments (line 288)
                    kwargs_271972 = {}
                    # Getting the type of 'repr' (line 288)
                    repr_271970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 44), 'repr', False)
                    # Calling repr(args, kwargs) (line 288)
                    repr_call_result_271973 = invoke(stypy.reporting.localization.Localization(__file__, 288, 44), repr_271970, *[value_271971], **kwargs_271972)
                    
                    # Getting the type of 'self' (line 288)
                    self_271974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 57), 'self', False)
                    # Processing the call keyword arguments (line 288)
                    kwargs_271975 = {}
                    # Getting the type of 'QtWidgets' (line 288)
                    QtWidgets_271968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 24), 'QtWidgets', False)
                    # Obtaining the member 'QLineEdit' of a type (line 288)
                    QLineEdit_271969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 24), QtWidgets_271968, 'QLineEdit')
                    # Calling QLineEdit(args, kwargs) (line 288)
                    QLineEdit_call_result_271976 = invoke(stypy.reporting.localization.Localization(__file__, 288, 24), QLineEdit_271969, *[repr_call_result_271973, self_271974], **kwargs_271975)
                    
                    # Assigning a type to the variable 'field' (line 288)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 16), 'field', QLineEdit_call_result_271976)
                    
                    # Call to setCursorPosition(...): (line 289)
                    # Processing the call arguments (line 289)
                    int_271979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 40), 'int')
                    # Processing the call keyword arguments (line 289)
                    kwargs_271980 = {}
                    # Getting the type of 'field' (line 289)
                    field_271977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 16), 'field', False)
                    # Obtaining the member 'setCursorPosition' of a type (line 289)
                    setCursorPosition_271978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 16), field_271977, 'setCursorPosition')
                    # Calling setCursorPosition(args, kwargs) (line 289)
                    setCursorPosition_call_result_271981 = invoke(stypy.reporting.localization.Localization(__file__, 289, 16), setCursorPosition_271978, *[int_271979], **kwargs_271980)
                    
                    
                    # Call to setValidator(...): (line 290)
                    # Processing the call arguments (line 290)
                    
                    # Call to QDoubleValidator(...): (line 290)
                    # Processing the call arguments (line 290)
                    # Getting the type of 'field' (line 290)
                    field_271986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 58), 'field', False)
                    # Processing the call keyword arguments (line 290)
                    kwargs_271987 = {}
                    # Getting the type of 'QtGui' (line 290)
                    QtGui_271984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 35), 'QtGui', False)
                    # Obtaining the member 'QDoubleValidator' of a type (line 290)
                    QDoubleValidator_271985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 35), QtGui_271984, 'QDoubleValidator')
                    # Calling QDoubleValidator(args, kwargs) (line 290)
                    QDoubleValidator_call_result_271988 = invoke(stypy.reporting.localization.Localization(__file__, 290, 35), QDoubleValidator_271985, *[field_271986], **kwargs_271987)
                    
                    # Processing the call keyword arguments (line 290)
                    kwargs_271989 = {}
                    # Getting the type of 'field' (line 290)
                    field_271982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 16), 'field', False)
                    # Obtaining the member 'setValidator' of a type (line 290)
                    setValidator_271983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 16), field_271982, 'setValidator')
                    # Calling setValidator(args, kwargs) (line 290)
                    setValidator_call_result_271990 = invoke(stypy.reporting.localization.Localization(__file__, 290, 16), setValidator_271983, *[QDoubleValidator_call_result_271988], **kwargs_271989)
                    
                    
                    # Call to setLocale(...): (line 291)
                    # Processing the call arguments (line 291)
                    
                    # Call to QLocale(...): (line 291)
                    # Processing the call arguments (line 291)
                    unicode_271998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 59), 'unicode', u'C')
                    # Processing the call keyword arguments (line 291)
                    kwargs_271999 = {}
                    # Getting the type of 'QtCore' (line 291)
                    QtCore_271996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 44), 'QtCore', False)
                    # Obtaining the member 'QLocale' of a type (line 291)
                    QLocale_271997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 44), QtCore_271996, 'QLocale')
                    # Calling QLocale(args, kwargs) (line 291)
                    QLocale_call_result_272000 = invoke(stypy.reporting.localization.Localization(__file__, 291, 44), QLocale_271997, *[unicode_271998], **kwargs_271999)
                    
                    # Processing the call keyword arguments (line 291)
                    kwargs_272001 = {}
                    
                    # Call to validator(...): (line 291)
                    # Processing the call keyword arguments (line 291)
                    kwargs_271993 = {}
                    # Getting the type of 'field' (line 291)
                    field_271991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 16), 'field', False)
                    # Obtaining the member 'validator' of a type (line 291)
                    validator_271992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 16), field_271991, 'validator')
                    # Calling validator(args, kwargs) (line 291)
                    validator_call_result_271994 = invoke(stypy.reporting.localization.Localization(__file__, 291, 16), validator_271992, *[], **kwargs_271993)
                    
                    # Obtaining the member 'setLocale' of a type (line 291)
                    setLocale_271995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 16), validator_call_result_271994, 'setLocale')
                    # Calling setLocale(args, kwargs) (line 291)
                    setLocale_call_result_272002 = invoke(stypy.reporting.localization.Localization(__file__, 291, 16), setLocale_271995, *[QLocale_call_result_272000], **kwargs_272001)
                    
                    
                    # Assigning a Call to a Name (line 292):
                    
                    # Assigning a Call to a Name (line 292):
                    
                    # Call to get_dialog(...): (line 292)
                    # Processing the call keyword arguments (line 292)
                    kwargs_272005 = {}
                    # Getting the type of 'self' (line 292)
                    self_272003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 25), 'self', False)
                    # Obtaining the member 'get_dialog' of a type (line 292)
                    get_dialog_272004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 25), self_272003, 'get_dialog')
                    # Calling get_dialog(args, kwargs) (line 292)
                    get_dialog_call_result_272006 = invoke(stypy.reporting.localization.Localization(__file__, 292, 25), get_dialog_272004, *[], **kwargs_272005)
                    
                    # Assigning a type to the variable 'dialog' (line 292)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 16), 'dialog', get_dialog_call_result_272006)
                    
                    # Call to register_float_field(...): (line 293)
                    # Processing the call arguments (line 293)
                    # Getting the type of 'field' (line 293)
                    field_272009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 44), 'field', False)
                    # Processing the call keyword arguments (line 293)
                    kwargs_272010 = {}
                    # Getting the type of 'dialog' (line 293)
                    dialog_272007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 16), 'dialog', False)
                    # Obtaining the member 'register_float_field' of a type (line 293)
                    register_float_field_272008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 16), dialog_272007, 'register_float_field')
                    # Calling register_float_field(args, kwargs) (line 293)
                    register_float_field_call_result_272011 = invoke(stypy.reporting.localization.Localization(__file__, 293, 16), register_float_field_272008, *[field_272009], **kwargs_272010)
                    
                    
                    # Call to connect(...): (line 294)
                    # Processing the call arguments (line 294)

                    @norecursion
                    def _stypy_temp_lambda_115(localization, *varargs, **kwargs):
                        global module_type_store
                        # Assign values to the parameters with defaults
                        defaults = []
                        # Create a new context for function '_stypy_temp_lambda_115'
                        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_115', 294, 42, True)
                        # Passed parameters checking function
                        _stypy_temp_lambda_115.stypy_localization = localization
                        _stypy_temp_lambda_115.stypy_type_of_self = None
                        _stypy_temp_lambda_115.stypy_type_store = module_type_store
                        _stypy_temp_lambda_115.stypy_function_name = '_stypy_temp_lambda_115'
                        _stypy_temp_lambda_115.stypy_param_names_list = ['text']
                        _stypy_temp_lambda_115.stypy_varargs_param_name = None
                        _stypy_temp_lambda_115.stypy_kwargs_param_name = None
                        _stypy_temp_lambda_115.stypy_call_defaults = defaults
                        _stypy_temp_lambda_115.stypy_call_varargs = varargs
                        _stypy_temp_lambda_115.stypy_call_kwargs = kwargs
                        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_115', ['text'], None, None, defaults, varargs, kwargs)

                        if is_error_type(arguments):
                            # Destroy the current context
                            module_type_store = module_type_store.close_function_context()
                            return arguments

                        # Stacktrace push for error reporting
                        localization.set_stack_trace('_stypy_temp_lambda_115', ['text'], arguments)
                        # Default return type storage variable (SSA)
                        # Assigning a type to the variable 'stypy_return_type'
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                        
                        
                        # ################# Begin of the lambda function code ##################

                        
                        # Call to update_buttons(...): (line 294)
                        # Processing the call keyword arguments (line 294)
                        kwargs_272017 = {}
                        # Getting the type of 'dialog' (line 294)
                        dialog_272015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 55), 'dialog', False)
                        # Obtaining the member 'update_buttons' of a type (line 294)
                        update_buttons_272016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 55), dialog_272015, 'update_buttons')
                        # Calling update_buttons(args, kwargs) (line 294)
                        update_buttons_call_result_272018 = invoke(stypy.reporting.localization.Localization(__file__, 294, 55), update_buttons_272016, *[], **kwargs_272017)
                        
                        # Assigning the return type of the lambda function
                        # Assigning a type to the variable 'stypy_return_type' (line 294)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 42), 'stypy_return_type', update_buttons_call_result_272018)
                        
                        # ################# End of the lambda function code ##################

                        # Stacktrace pop (error reporting)
                        localization.unset_stack_trace()
                        
                        # Storing the return type of function '_stypy_temp_lambda_115' in the type store
                        # Getting the type of 'stypy_return_type' (line 294)
                        stypy_return_type_272019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 42), 'stypy_return_type')
                        module_type_store.store_return_type_of_current_context(stypy_return_type_272019)
                        
                        # Destroy the current context
                        module_type_store = module_type_store.close_function_context()
                        
                        # Return type of the function '_stypy_temp_lambda_115'
                        return stypy_return_type_272019

                    # Assigning a type to the variable '_stypy_temp_lambda_115' (line 294)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 42), '_stypy_temp_lambda_115', _stypy_temp_lambda_115)
                    # Getting the type of '_stypy_temp_lambda_115' (line 294)
                    _stypy_temp_lambda_115_272020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 42), '_stypy_temp_lambda_115')
                    # Processing the call keyword arguments (line 294)
                    kwargs_272021 = {}
                    # Getting the type of 'field' (line 294)
                    field_272012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 16), 'field', False)
                    # Obtaining the member 'textChanged' of a type (line 294)
                    textChanged_272013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 16), field_272012, 'textChanged')
                    # Obtaining the member 'connect' of a type (line 294)
                    connect_272014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 16), textChanged_272013, 'connect')
                    # Calling connect(args, kwargs) (line 294)
                    connect_call_result_272022 = invoke(stypy.reporting.localization.Localization(__file__, 294, 16), connect_272014, *[_stypy_temp_lambda_115_272020], **kwargs_272021)
                    

                    if more_types_in_union_271967:
                        # Runtime conditional SSA for else branch (line 287)
                        module_type_store.open_ssa_branch('idiom else')



                if ((not may_be_271966) or more_types_in_union_271967):
                    # Assigning a type to the variable 'value' (line 287)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 17), 'value', remove_subtype_from_union(value_271965, float))
                    
                    # Type idiom detected: calculating its left and rigth part (line 295)
                    # Getting the type of 'int' (line 295)
                    int_272023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 35), 'int')
                    # Getting the type of 'value' (line 295)
                    value_272024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 28), 'value')
                    
                    (may_be_272025, more_types_in_union_272026) = may_be_subtype(int_272023, value_272024)

                    if may_be_272025:

                        if more_types_in_union_272026:
                            # Runtime conditional SSA (line 295)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                        else:
                            module_type_store = module_type_store

                        # Assigning a type to the variable 'value' (line 295)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 17), 'value', remove_not_subtype_from_union(value_272024, int))
                        
                        # Assigning a Call to a Name (line 296):
                        
                        # Assigning a Call to a Name (line 296):
                        
                        # Call to QSpinBox(...): (line 296)
                        # Processing the call arguments (line 296)
                        # Getting the type of 'self' (line 296)
                        self_272029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 43), 'self', False)
                        # Processing the call keyword arguments (line 296)
                        kwargs_272030 = {}
                        # Getting the type of 'QtWidgets' (line 296)
                        QtWidgets_272027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 24), 'QtWidgets', False)
                        # Obtaining the member 'QSpinBox' of a type (line 296)
                        QSpinBox_272028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 24), QtWidgets_272027, 'QSpinBox')
                        # Calling QSpinBox(args, kwargs) (line 296)
                        QSpinBox_call_result_272031 = invoke(stypy.reporting.localization.Localization(__file__, 296, 24), QSpinBox_272028, *[self_272029], **kwargs_272030)
                        
                        # Assigning a type to the variable 'field' (line 296)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 16), 'field', QSpinBox_call_result_272031)
                        
                        # Call to setRange(...): (line 297)
                        # Processing the call arguments (line 297)
                        float_272034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 31), 'float')
                        float_272035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 37), 'float')
                        # Processing the call keyword arguments (line 297)
                        kwargs_272036 = {}
                        # Getting the type of 'field' (line 297)
                        field_272032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 16), 'field', False)
                        # Obtaining the member 'setRange' of a type (line 297)
                        setRange_272033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 16), field_272032, 'setRange')
                        # Calling setRange(args, kwargs) (line 297)
                        setRange_call_result_272037 = invoke(stypy.reporting.localization.Localization(__file__, 297, 16), setRange_272033, *[float_272034, float_272035], **kwargs_272036)
                        
                        
                        # Call to setValue(...): (line 298)
                        # Processing the call arguments (line 298)
                        # Getting the type of 'value' (line 298)
                        value_272040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 31), 'value', False)
                        # Processing the call keyword arguments (line 298)
                        kwargs_272041 = {}
                        # Getting the type of 'field' (line 298)
                        field_272038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 16), 'field', False)
                        # Obtaining the member 'setValue' of a type (line 298)
                        setValue_272039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 16), field_272038, 'setValue')
                        # Calling setValue(args, kwargs) (line 298)
                        setValue_call_result_272042 = invoke(stypy.reporting.localization.Localization(__file__, 298, 16), setValue_272039, *[value_272040], **kwargs_272041)
                        

                        if more_types_in_union_272026:
                            # Runtime conditional SSA for else branch (line 295)
                            module_type_store.open_ssa_branch('idiom else')



                    if ((not may_be_272025) or more_types_in_union_272026):
                        # Assigning a type to the variable 'value' (line 295)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 17), 'value', remove_subtype_from_union(value_272024, int))
                        
                        
                        # Call to isinstance(...): (line 299)
                        # Processing the call arguments (line 299)
                        # Getting the type of 'value' (line 299)
                        value_272044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 28), 'value', False)
                        # Getting the type of 'datetime' (line 299)
                        datetime_272045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 35), 'datetime', False)
                        # Obtaining the member 'datetime' of a type (line 299)
                        datetime_272046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 35), datetime_272045, 'datetime')
                        # Processing the call keyword arguments (line 299)
                        kwargs_272047 = {}
                        # Getting the type of 'isinstance' (line 299)
                        isinstance_272043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 17), 'isinstance', False)
                        # Calling isinstance(args, kwargs) (line 299)
                        isinstance_call_result_272048 = invoke(stypy.reporting.localization.Localization(__file__, 299, 17), isinstance_272043, *[value_272044, datetime_272046], **kwargs_272047)
                        
                        # Testing the type of an if condition (line 299)
                        if_condition_272049 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 299, 17), isinstance_call_result_272048)
                        # Assigning a type to the variable 'if_condition_272049' (line 299)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 17), 'if_condition_272049', if_condition_272049)
                        # SSA begins for if statement (line 299)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Call to a Name (line 300):
                        
                        # Assigning a Call to a Name (line 300):
                        
                        # Call to QDateTimeEdit(...): (line 300)
                        # Processing the call arguments (line 300)
                        # Getting the type of 'self' (line 300)
                        self_272052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 48), 'self', False)
                        # Processing the call keyword arguments (line 300)
                        kwargs_272053 = {}
                        # Getting the type of 'QtWidgets' (line 300)
                        QtWidgets_272050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 24), 'QtWidgets', False)
                        # Obtaining the member 'QDateTimeEdit' of a type (line 300)
                        QDateTimeEdit_272051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 24), QtWidgets_272050, 'QDateTimeEdit')
                        # Calling QDateTimeEdit(args, kwargs) (line 300)
                        QDateTimeEdit_call_result_272054 = invoke(stypy.reporting.localization.Localization(__file__, 300, 24), QDateTimeEdit_272051, *[self_272052], **kwargs_272053)
                        
                        # Assigning a type to the variable 'field' (line 300)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'field', QDateTimeEdit_call_result_272054)
                        
                        # Call to setDateTime(...): (line 301)
                        # Processing the call arguments (line 301)
                        # Getting the type of 'value' (line 301)
                        value_272057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 34), 'value', False)
                        # Processing the call keyword arguments (line 301)
                        kwargs_272058 = {}
                        # Getting the type of 'field' (line 301)
                        field_272055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'field', False)
                        # Obtaining the member 'setDateTime' of a type (line 301)
                        setDateTime_272056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 16), field_272055, 'setDateTime')
                        # Calling setDateTime(args, kwargs) (line 301)
                        setDateTime_call_result_272059 = invoke(stypy.reporting.localization.Localization(__file__, 301, 16), setDateTime_272056, *[value_272057], **kwargs_272058)
                        
                        # SSA branch for the else part of an if statement (line 299)
                        module_type_store.open_ssa_branch('else')
                        
                        
                        # Call to isinstance(...): (line 302)
                        # Processing the call arguments (line 302)
                        # Getting the type of 'value' (line 302)
                        value_272061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 28), 'value', False)
                        # Getting the type of 'datetime' (line 302)
                        datetime_272062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 35), 'datetime', False)
                        # Obtaining the member 'date' of a type (line 302)
                        date_272063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 35), datetime_272062, 'date')
                        # Processing the call keyword arguments (line 302)
                        kwargs_272064 = {}
                        # Getting the type of 'isinstance' (line 302)
                        isinstance_272060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 17), 'isinstance', False)
                        # Calling isinstance(args, kwargs) (line 302)
                        isinstance_call_result_272065 = invoke(stypy.reporting.localization.Localization(__file__, 302, 17), isinstance_272060, *[value_272061, date_272063], **kwargs_272064)
                        
                        # Testing the type of an if condition (line 302)
                        if_condition_272066 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 302, 17), isinstance_call_result_272065)
                        # Assigning a type to the variable 'if_condition_272066' (line 302)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 17), 'if_condition_272066', if_condition_272066)
                        # SSA begins for if statement (line 302)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Call to a Name (line 303):
                        
                        # Assigning a Call to a Name (line 303):
                        
                        # Call to QDateEdit(...): (line 303)
                        # Processing the call arguments (line 303)
                        # Getting the type of 'self' (line 303)
                        self_272069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 44), 'self', False)
                        # Processing the call keyword arguments (line 303)
                        kwargs_272070 = {}
                        # Getting the type of 'QtWidgets' (line 303)
                        QtWidgets_272067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 24), 'QtWidgets', False)
                        # Obtaining the member 'QDateEdit' of a type (line 303)
                        QDateEdit_272068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 24), QtWidgets_272067, 'QDateEdit')
                        # Calling QDateEdit(args, kwargs) (line 303)
                        QDateEdit_call_result_272071 = invoke(stypy.reporting.localization.Localization(__file__, 303, 24), QDateEdit_272068, *[self_272069], **kwargs_272070)
                        
                        # Assigning a type to the variable 'field' (line 303)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'field', QDateEdit_call_result_272071)
                        
                        # Call to setDate(...): (line 304)
                        # Processing the call arguments (line 304)
                        # Getting the type of 'value' (line 304)
                        value_272074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 30), 'value', False)
                        # Processing the call keyword arguments (line 304)
                        kwargs_272075 = {}
                        # Getting the type of 'field' (line 304)
                        field_272072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'field', False)
                        # Obtaining the member 'setDate' of a type (line 304)
                        setDate_272073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 16), field_272072, 'setDate')
                        # Calling setDate(args, kwargs) (line 304)
                        setDate_call_result_272076 = invoke(stypy.reporting.localization.Localization(__file__, 304, 16), setDate_272073, *[value_272074], **kwargs_272075)
                        
                        # SSA branch for the else part of an if statement (line 302)
                        module_type_store.open_ssa_branch('else')
                        
                        # Assigning a Call to a Name (line 306):
                        
                        # Assigning a Call to a Name (line 306):
                        
                        # Call to QLineEdit(...): (line 306)
                        # Processing the call arguments (line 306)
                        
                        # Call to repr(...): (line 306)
                        # Processing the call arguments (line 306)
                        # Getting the type of 'value' (line 306)
                        value_272080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 49), 'value', False)
                        # Processing the call keyword arguments (line 306)
                        kwargs_272081 = {}
                        # Getting the type of 'repr' (line 306)
                        repr_272079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 44), 'repr', False)
                        # Calling repr(args, kwargs) (line 306)
                        repr_call_result_272082 = invoke(stypy.reporting.localization.Localization(__file__, 306, 44), repr_272079, *[value_272080], **kwargs_272081)
                        
                        # Getting the type of 'self' (line 306)
                        self_272083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 57), 'self', False)
                        # Processing the call keyword arguments (line 306)
                        kwargs_272084 = {}
                        # Getting the type of 'QtWidgets' (line 306)
                        QtWidgets_272077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 24), 'QtWidgets', False)
                        # Obtaining the member 'QLineEdit' of a type (line 306)
                        QLineEdit_272078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 24), QtWidgets_272077, 'QLineEdit')
                        # Calling QLineEdit(args, kwargs) (line 306)
                        QLineEdit_call_result_272085 = invoke(stypy.reporting.localization.Localization(__file__, 306, 24), QLineEdit_272078, *[repr_call_result_272082, self_272083], **kwargs_272084)
                        
                        # Assigning a type to the variable 'field' (line 306)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'field', QLineEdit_call_result_272085)
                        # SSA join for if statement (line 302)
                        module_type_store = module_type_store.join_ssa_context()
                        
                        # SSA join for if statement (line 299)
                        module_type_store = module_type_store.join_ssa_context()
                        

                        if (may_be_272025 and more_types_in_union_272026):
                            # SSA join for if statement (line 295)
                            module_type_store = module_type_store.join_ssa_context()


                    

                    if (may_be_271966 and more_types_in_union_271967):
                        # SSA join for if statement (line 287)
                        module_type_store = module_type_store.join_ssa_context()


                

                if (may_be_271941 and more_types_in_union_271942):
                    # SSA join for if statement (line 281)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 260)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 258)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 255)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 253)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_271786 and more_types_in_union_271787):
                # SSA join for if statement (line 248)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 243)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to addRow(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'label' (line 307)
        label_272089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 35), 'label', False)
        # Getting the type of 'field' (line 307)
        field_272090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 42), 'field', False)
        # Processing the call keyword arguments (line 307)
        kwargs_272091 = {}
        # Getting the type of 'self' (line 307)
        self_272086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'self', False)
        # Obtaining the member 'formlayout' of a type (line 307)
        formlayout_272087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 12), self_272086, 'formlayout')
        # Obtaining the member 'addRow' of a type (line 307)
        addRow_272088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 12), formlayout_272087, 'addRow')
        # Calling addRow(args, kwargs) (line 307)
        addRow_call_result_272092 = invoke(stypy.reporting.localization.Localization(__file__, 307, 12), addRow_272088, *[label_272089, field_272090], **kwargs_272091)
        
        
        # Call to append(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 'field' (line 308)
        field_272096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 32), 'field', False)
        # Processing the call keyword arguments (line 308)
        kwargs_272097 = {}
        # Getting the type of 'self' (line 308)
        self_272093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'self', False)
        # Obtaining the member 'widgets' of a type (line 308)
        widgets_272094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 12), self_272093, 'widgets')
        # Obtaining the member 'append' of a type (line 308)
        append_272095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 12), widgets_272094, 'append')
        # Calling append(args, kwargs) (line 308)
        append_call_result_272098 = invoke(stypy.reporting.localization.Localization(__file__, 308, 12), append_272095, *[field_272096], **kwargs_272097)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'setup(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup' in the type store
        # Getting the type of 'stypy_return_type' (line 239)
        stypy_return_type_272099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_272099)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup'
        return stypy_return_type_272099


    @norecursion
    def get(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get'
        module_type_store = module_type_store.open_function_context('get', 310, 4, False)
        # Assigning a type to the variable 'self' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FormWidget.get.__dict__.__setitem__('stypy_localization', localization)
        FormWidget.get.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FormWidget.get.__dict__.__setitem__('stypy_type_store', module_type_store)
        FormWidget.get.__dict__.__setitem__('stypy_function_name', 'FormWidget.get')
        FormWidget.get.__dict__.__setitem__('stypy_param_names_list', [])
        FormWidget.get.__dict__.__setitem__('stypy_varargs_param_name', None)
        FormWidget.get.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FormWidget.get.__dict__.__setitem__('stypy_call_defaults', defaults)
        FormWidget.get.__dict__.__setitem__('stypy_call_varargs', varargs)
        FormWidget.get.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FormWidget.get.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FormWidget.get', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get(...)' code ##################

        
        # Assigning a List to a Name (line 311):
        
        # Assigning a List to a Name (line 311):
        
        # Obtaining an instance of the builtin type 'list' (line 311)
        list_272100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 311)
        
        # Assigning a type to the variable 'valuelist' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'valuelist', list_272100)
        
        
        # Call to enumerate(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'self' (line 312)
        self_272102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 47), 'self', False)
        # Obtaining the member 'data' of a type (line 312)
        data_272103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 47), self_272102, 'data')
        # Processing the call keyword arguments (line 312)
        kwargs_272104 = {}
        # Getting the type of 'enumerate' (line 312)
        enumerate_272101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 37), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 312)
        enumerate_call_result_272105 = invoke(stypy.reporting.localization.Localization(__file__, 312, 37), enumerate_272101, *[data_272103], **kwargs_272104)
        
        # Testing the type of a for loop iterable (line 312)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 312, 8), enumerate_call_result_272105)
        # Getting the type of the for loop variable (line 312)
        for_loop_var_272106 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 312, 8), enumerate_call_result_272105)
        # Assigning a type to the variable 'index' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'index', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 8), for_loop_var_272106))
        # Assigning a type to the variable 'label' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'label', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 8), for_loop_var_272106))
        # Assigning a type to the variable 'value' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 8), for_loop_var_272106))
        # SSA begins for a for statement (line 312)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 313):
        
        # Assigning a Subscript to a Name (line 313):
        
        # Obtaining the type of the subscript
        # Getting the type of 'index' (line 313)
        index_272107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 33), 'index')
        # Getting the type of 'self' (line 313)
        self_272108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 20), 'self')
        # Obtaining the member 'widgets' of a type (line 313)
        widgets_272109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 20), self_272108, 'widgets')
        # Obtaining the member '__getitem__' of a type (line 313)
        getitem___272110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 20), widgets_272109, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 313)
        subscript_call_result_272111 = invoke(stypy.reporting.localization.Localization(__file__, 313, 20), getitem___272110, index_272107)
        
        # Assigning a type to the variable 'field' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'field', subscript_call_result_272111)
        
        # Type idiom detected: calculating its left and rigth part (line 314)
        # Getting the type of 'label' (line 314)
        label_272112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 15), 'label')
        # Getting the type of 'None' (line 314)
        None_272113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 24), 'None')
        
        (may_be_272114, more_types_in_union_272115) = may_be_none(label_272112, None_272113)

        if may_be_272114:

            if more_types_in_union_272115:
                # Runtime conditional SSA (line 314)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store


            if more_types_in_union_272115:
                # Runtime conditional SSA for else branch (line 314)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_272114) or more_types_in_union_272115):
            
            
            
            # Call to tuple_to_qfont(...): (line 317)
            # Processing the call arguments (line 317)
            # Getting the type of 'value' (line 317)
            value_272117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 32), 'value', False)
            # Processing the call keyword arguments (line 317)
            kwargs_272118 = {}
            # Getting the type of 'tuple_to_qfont' (line 317)
            tuple_to_qfont_272116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 17), 'tuple_to_qfont', False)
            # Calling tuple_to_qfont(args, kwargs) (line 317)
            tuple_to_qfont_call_result_272119 = invoke(stypy.reporting.localization.Localization(__file__, 317, 17), tuple_to_qfont_272116, *[value_272117], **kwargs_272118)
            
            # Getting the type of 'None' (line 317)
            None_272120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 46), 'None')
            # Applying the binary operator 'isnot' (line 317)
            result_is_not_272121 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 17), 'isnot', tuple_to_qfont_call_result_272119, None_272120)
            
            # Testing the type of an if condition (line 317)
            if_condition_272122 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 317, 17), result_is_not_272121)
            # Assigning a type to the variable 'if_condition_272122' (line 317)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 17), 'if_condition_272122', if_condition_272122)
            # SSA begins for if statement (line 317)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 318):
            
            # Assigning a Call to a Name (line 318):
            
            # Call to get_font(...): (line 318)
            # Processing the call keyword arguments (line 318)
            kwargs_272125 = {}
            # Getting the type of 'field' (line 318)
            field_272123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 24), 'field', False)
            # Obtaining the member 'get_font' of a type (line 318)
            get_font_272124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 24), field_272123, 'get_font')
            # Calling get_font(args, kwargs) (line 318)
            get_font_call_result_272126 = invoke(stypy.reporting.localization.Localization(__file__, 318, 24), get_font_272124, *[], **kwargs_272125)
            
            # Assigning a type to the variable 'value' (line 318)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 16), 'value', get_font_call_result_272126)
            # SSA branch for the else part of an if statement (line 317)
            module_type_store.open_ssa_branch('else')
            
            
            # Evaluating a boolean operation
            
            # Call to isinstance(...): (line 319)
            # Processing the call arguments (line 319)
            # Getting the type of 'value' (line 319)
            value_272128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 29), 'value', False)
            # Getting the type of 'six' (line 319)
            six_272129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 36), 'six', False)
            # Obtaining the member 'string_types' of a type (line 319)
            string_types_272130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 36), six_272129, 'string_types')
            # Processing the call keyword arguments (line 319)
            kwargs_272131 = {}
            # Getting the type of 'isinstance' (line 319)
            isinstance_272127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 18), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 319)
            isinstance_call_result_272132 = invoke(stypy.reporting.localization.Localization(__file__, 319, 18), isinstance_272127, *[value_272128, string_types_272130], **kwargs_272131)
            
            
            # Call to is_color_like(...): (line 320)
            # Processing the call arguments (line 320)
            # Getting the type of 'value' (line 320)
            value_272135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 43), 'value', False)
            # Processing the call keyword arguments (line 320)
            kwargs_272136 = {}
            # Getting the type of 'mcolors' (line 320)
            mcolors_272133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 21), 'mcolors', False)
            # Obtaining the member 'is_color_like' of a type (line 320)
            is_color_like_272134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 21), mcolors_272133, 'is_color_like')
            # Calling is_color_like(args, kwargs) (line 320)
            is_color_like_call_result_272137 = invoke(stypy.reporting.localization.Localization(__file__, 320, 21), is_color_like_272134, *[value_272135], **kwargs_272136)
            
            # Applying the binary operator 'or' (line 319)
            result_or_keyword_272138 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 18), 'or', isinstance_call_result_272132, is_color_like_call_result_272137)
            
            # Testing the type of an if condition (line 319)
            if_condition_272139 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 319, 17), result_or_keyword_272138)
            # Assigning a type to the variable 'if_condition_272139' (line 319)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 17), 'if_condition_272139', if_condition_272139)
            # SSA begins for if statement (line 319)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 321):
            
            # Assigning a Call to a Name (line 321):
            
            # Call to text_type(...): (line 321)
            # Processing the call arguments (line 321)
            
            # Call to text(...): (line 321)
            # Processing the call keyword arguments (line 321)
            kwargs_272144 = {}
            # Getting the type of 'field' (line 321)
            field_272142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 38), 'field', False)
            # Obtaining the member 'text' of a type (line 321)
            text_272143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 38), field_272142, 'text')
            # Calling text(args, kwargs) (line 321)
            text_call_result_272145 = invoke(stypy.reporting.localization.Localization(__file__, 321, 38), text_272143, *[], **kwargs_272144)
            
            # Processing the call keyword arguments (line 321)
            kwargs_272146 = {}
            # Getting the type of 'six' (line 321)
            six_272140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 24), 'six', False)
            # Obtaining the member 'text_type' of a type (line 321)
            text_type_272141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 24), six_272140, 'text_type')
            # Calling text_type(args, kwargs) (line 321)
            text_type_call_result_272147 = invoke(stypy.reporting.localization.Localization(__file__, 321, 24), text_type_272141, *[text_call_result_272145], **kwargs_272146)
            
            # Assigning a type to the variable 'value' (line 321)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 16), 'value', text_type_call_result_272147)
            # SSA branch for the else part of an if statement (line 319)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isinstance(...): (line 322)
            # Processing the call arguments (line 322)
            # Getting the type of 'value' (line 322)
            value_272149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 28), 'value', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 322)
            tuple_272150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 36), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 322)
            # Adding element type (line 322)
            # Getting the type of 'list' (line 322)
            list_272151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 36), 'list', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 36), tuple_272150, list_272151)
            # Adding element type (line 322)
            # Getting the type of 'tuple' (line 322)
            tuple_272152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 42), 'tuple', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 36), tuple_272150, tuple_272152)
            
            # Processing the call keyword arguments (line 322)
            kwargs_272153 = {}
            # Getting the type of 'isinstance' (line 322)
            isinstance_272148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 17), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 322)
            isinstance_call_result_272154 = invoke(stypy.reporting.localization.Localization(__file__, 322, 17), isinstance_272148, *[value_272149, tuple_272150], **kwargs_272153)
            
            # Testing the type of an if condition (line 322)
            if_condition_272155 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 17), isinstance_call_result_272154)
            # Assigning a type to the variable 'if_condition_272155' (line 322)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 17), 'if_condition_272155', if_condition_272155)
            # SSA begins for if statement (line 322)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 323):
            
            # Assigning a Call to a Name (line 323):
            
            # Call to int(...): (line 323)
            # Processing the call arguments (line 323)
            
            # Call to currentIndex(...): (line 323)
            # Processing the call keyword arguments (line 323)
            kwargs_272159 = {}
            # Getting the type of 'field' (line 323)
            field_272157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 28), 'field', False)
            # Obtaining the member 'currentIndex' of a type (line 323)
            currentIndex_272158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 28), field_272157, 'currentIndex')
            # Calling currentIndex(args, kwargs) (line 323)
            currentIndex_call_result_272160 = invoke(stypy.reporting.localization.Localization(__file__, 323, 28), currentIndex_272158, *[], **kwargs_272159)
            
            # Processing the call keyword arguments (line 323)
            kwargs_272161 = {}
            # Getting the type of 'int' (line 323)
            int_272156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 24), 'int', False)
            # Calling int(args, kwargs) (line 323)
            int_call_result_272162 = invoke(stypy.reporting.localization.Localization(__file__, 323, 24), int_272156, *[currentIndex_call_result_272160], **kwargs_272161)
            
            # Assigning a type to the variable 'index' (line 323)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 16), 'index', int_call_result_272162)
            
            
            # Call to isinstance(...): (line 324)
            # Processing the call arguments (line 324)
            
            # Obtaining the type of the subscript
            int_272164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 36), 'int')
            # Getting the type of 'value' (line 324)
            value_272165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 30), 'value', False)
            # Obtaining the member '__getitem__' of a type (line 324)
            getitem___272166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 30), value_272165, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 324)
            subscript_call_result_272167 = invoke(stypy.reporting.localization.Localization(__file__, 324, 30), getitem___272166, int_272164)
            
            
            # Obtaining an instance of the builtin type 'tuple' (line 324)
            tuple_272168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 41), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 324)
            # Adding element type (line 324)
            # Getting the type of 'list' (line 324)
            list_272169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 41), 'list', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 41), tuple_272168, list_272169)
            # Adding element type (line 324)
            # Getting the type of 'tuple' (line 324)
            tuple_272170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 47), 'tuple', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 41), tuple_272168, tuple_272170)
            
            # Processing the call keyword arguments (line 324)
            kwargs_272171 = {}
            # Getting the type of 'isinstance' (line 324)
            isinstance_272163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 324)
            isinstance_call_result_272172 = invoke(stypy.reporting.localization.Localization(__file__, 324, 19), isinstance_272163, *[subscript_call_result_272167, tuple_272168], **kwargs_272171)
            
            # Testing the type of an if condition (line 324)
            if_condition_272173 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 324, 16), isinstance_call_result_272172)
            # Assigning a type to the variable 'if_condition_272173' (line 324)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 16), 'if_condition_272173', if_condition_272173)
            # SSA begins for if statement (line 324)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 325):
            
            # Assigning a Subscript to a Name (line 325):
            
            # Obtaining the type of the subscript
            int_272174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 41), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'index' (line 325)
            index_272175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 34), 'index')
            # Getting the type of 'value' (line 325)
            value_272176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 28), 'value')
            # Obtaining the member '__getitem__' of a type (line 325)
            getitem___272177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 28), value_272176, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 325)
            subscript_call_result_272178 = invoke(stypy.reporting.localization.Localization(__file__, 325, 28), getitem___272177, index_272175)
            
            # Obtaining the member '__getitem__' of a type (line 325)
            getitem___272179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 28), subscript_call_result_272178, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 325)
            subscript_call_result_272180 = invoke(stypy.reporting.localization.Localization(__file__, 325, 28), getitem___272179, int_272174)
            
            # Assigning a type to the variable 'value' (line 325)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 20), 'value', subscript_call_result_272180)
            # SSA branch for the else part of an if statement (line 324)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Subscript to a Name (line 327):
            
            # Assigning a Subscript to a Name (line 327):
            
            # Obtaining the type of the subscript
            # Getting the type of 'index' (line 327)
            index_272181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 34), 'index')
            # Getting the type of 'value' (line 327)
            value_272182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 28), 'value')
            # Obtaining the member '__getitem__' of a type (line 327)
            getitem___272183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 28), value_272182, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 327)
            subscript_call_result_272184 = invoke(stypy.reporting.localization.Localization(__file__, 327, 28), getitem___272183, index_272181)
            
            # Assigning a type to the variable 'value' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 20), 'value', subscript_call_result_272184)
            # SSA join for if statement (line 324)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 322)
            module_type_store.open_ssa_branch('else')
            
            # Type idiom detected: calculating its left and rigth part (line 328)
            # Getting the type of 'bool' (line 328)
            bool_272185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 35), 'bool')
            # Getting the type of 'value' (line 328)
            value_272186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 28), 'value')
            
            (may_be_272187, more_types_in_union_272188) = may_be_subtype(bool_272185, value_272186)

            if may_be_272187:

                if more_types_in_union_272188:
                    # Runtime conditional SSA (line 328)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'value' (line 328)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 17), 'value', remove_not_subtype_from_union(value_272186, bool))
                
                # Assigning a Compare to a Name (line 329):
                
                # Assigning a Compare to a Name (line 329):
                
                
                # Call to checkState(...): (line 329)
                # Processing the call keyword arguments (line 329)
                kwargs_272191 = {}
                # Getting the type of 'field' (line 329)
                field_272189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 24), 'field', False)
                # Obtaining the member 'checkState' of a type (line 329)
                checkState_272190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 24), field_272189, 'checkState')
                # Calling checkState(args, kwargs) (line 329)
                checkState_call_result_272192 = invoke(stypy.reporting.localization.Localization(__file__, 329, 24), checkState_272190, *[], **kwargs_272191)
                
                # Getting the type of 'QtCore' (line 329)
                QtCore_272193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 46), 'QtCore')
                # Obtaining the member 'Qt' of a type (line 329)
                Qt_272194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 46), QtCore_272193, 'Qt')
                # Obtaining the member 'Checked' of a type (line 329)
                Checked_272195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 46), Qt_272194, 'Checked')
                # Applying the binary operator '==' (line 329)
                result_eq_272196 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 24), '==', checkState_call_result_272192, Checked_272195)
                
                # Assigning a type to the variable 'value' (line 329)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 16), 'value', result_eq_272196)

                if more_types_in_union_272188:
                    # Runtime conditional SSA for else branch (line 328)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_272187) or more_types_in_union_272188):
                # Assigning a type to the variable 'value' (line 328)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 17), 'value', remove_subtype_from_union(value_272186, bool))
                
                # Type idiom detected: calculating its left and rigth part (line 330)
                # Getting the type of 'float' (line 330)
                float_272197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 35), 'float')
                # Getting the type of 'value' (line 330)
                value_272198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 28), 'value')
                
                (may_be_272199, more_types_in_union_272200) = may_be_subtype(float_272197, value_272198)

                if may_be_272199:

                    if more_types_in_union_272200:
                        # Runtime conditional SSA (line 330)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Assigning a type to the variable 'value' (line 330)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 17), 'value', remove_not_subtype_from_union(value_272198, float))
                    
                    # Assigning a Call to a Name (line 331):
                    
                    # Assigning a Call to a Name (line 331):
                    
                    # Call to float(...): (line 331)
                    # Processing the call arguments (line 331)
                    
                    # Call to str(...): (line 331)
                    # Processing the call arguments (line 331)
                    
                    # Call to text(...): (line 331)
                    # Processing the call keyword arguments (line 331)
                    kwargs_272205 = {}
                    # Getting the type of 'field' (line 331)
                    field_272203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 34), 'field', False)
                    # Obtaining the member 'text' of a type (line 331)
                    text_272204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 34), field_272203, 'text')
                    # Calling text(args, kwargs) (line 331)
                    text_call_result_272206 = invoke(stypy.reporting.localization.Localization(__file__, 331, 34), text_272204, *[], **kwargs_272205)
                    
                    # Processing the call keyword arguments (line 331)
                    kwargs_272207 = {}
                    # Getting the type of 'str' (line 331)
                    str_272202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 30), 'str', False)
                    # Calling str(args, kwargs) (line 331)
                    str_call_result_272208 = invoke(stypy.reporting.localization.Localization(__file__, 331, 30), str_272202, *[text_call_result_272206], **kwargs_272207)
                    
                    # Processing the call keyword arguments (line 331)
                    kwargs_272209 = {}
                    # Getting the type of 'float' (line 331)
                    float_272201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 24), 'float', False)
                    # Calling float(args, kwargs) (line 331)
                    float_call_result_272210 = invoke(stypy.reporting.localization.Localization(__file__, 331, 24), float_272201, *[str_call_result_272208], **kwargs_272209)
                    
                    # Assigning a type to the variable 'value' (line 331)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'value', float_call_result_272210)

                    if more_types_in_union_272200:
                        # Runtime conditional SSA for else branch (line 330)
                        module_type_store.open_ssa_branch('idiom else')



                if ((not may_be_272199) or more_types_in_union_272200):
                    # Assigning a type to the variable 'value' (line 330)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 17), 'value', remove_subtype_from_union(value_272198, float))
                    
                    # Type idiom detected: calculating its left and rigth part (line 332)
                    # Getting the type of 'int' (line 332)
                    int_272211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 35), 'int')
                    # Getting the type of 'value' (line 332)
                    value_272212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 28), 'value')
                    
                    (may_be_272213, more_types_in_union_272214) = may_be_subtype(int_272211, value_272212)

                    if may_be_272213:

                        if more_types_in_union_272214:
                            # Runtime conditional SSA (line 332)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                        else:
                            module_type_store = module_type_store

                        # Assigning a type to the variable 'value' (line 332)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 17), 'value', remove_not_subtype_from_union(value_272212, int))
                        
                        # Assigning a Call to a Name (line 333):
                        
                        # Assigning a Call to a Name (line 333):
                        
                        # Call to int(...): (line 333)
                        # Processing the call arguments (line 333)
                        
                        # Call to value(...): (line 333)
                        # Processing the call keyword arguments (line 333)
                        kwargs_272218 = {}
                        # Getting the type of 'field' (line 333)
                        field_272216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 28), 'field', False)
                        # Obtaining the member 'value' of a type (line 333)
                        value_272217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 28), field_272216, 'value')
                        # Calling value(args, kwargs) (line 333)
                        value_call_result_272219 = invoke(stypy.reporting.localization.Localization(__file__, 333, 28), value_272217, *[], **kwargs_272218)
                        
                        # Processing the call keyword arguments (line 333)
                        kwargs_272220 = {}
                        # Getting the type of 'int' (line 333)
                        int_272215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 24), 'int', False)
                        # Calling int(args, kwargs) (line 333)
                        int_call_result_272221 = invoke(stypy.reporting.localization.Localization(__file__, 333, 24), int_272215, *[value_call_result_272219], **kwargs_272220)
                        
                        # Assigning a type to the variable 'value' (line 333)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 16), 'value', int_call_result_272221)

                        if more_types_in_union_272214:
                            # Runtime conditional SSA for else branch (line 332)
                            module_type_store.open_ssa_branch('idiom else')



                    if ((not may_be_272213) or more_types_in_union_272214):
                        # Assigning a type to the variable 'value' (line 332)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 17), 'value', remove_subtype_from_union(value_272212, int))
                        
                        
                        # Call to isinstance(...): (line 334)
                        # Processing the call arguments (line 334)
                        # Getting the type of 'value' (line 334)
                        value_272223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 28), 'value', False)
                        # Getting the type of 'datetime' (line 334)
                        datetime_272224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 35), 'datetime', False)
                        # Obtaining the member 'datetime' of a type (line 334)
                        datetime_272225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 35), datetime_272224, 'datetime')
                        # Processing the call keyword arguments (line 334)
                        kwargs_272226 = {}
                        # Getting the type of 'isinstance' (line 334)
                        isinstance_272222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 17), 'isinstance', False)
                        # Calling isinstance(args, kwargs) (line 334)
                        isinstance_call_result_272227 = invoke(stypy.reporting.localization.Localization(__file__, 334, 17), isinstance_272222, *[value_272223, datetime_272225], **kwargs_272226)
                        
                        # Testing the type of an if condition (line 334)
                        if_condition_272228 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 17), isinstance_call_result_272227)
                        # Assigning a type to the variable 'if_condition_272228' (line 334)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 17), 'if_condition_272228', if_condition_272228)
                        # SSA begins for if statement (line 334)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Call to a Name (line 335):
                        
                        # Assigning a Call to a Name (line 335):
                        
                        # Call to toPyDateTime(...): (line 335)
                        # Processing the call keyword arguments (line 335)
                        kwargs_272234 = {}
                        
                        # Call to dateTime(...): (line 335)
                        # Processing the call keyword arguments (line 335)
                        kwargs_272231 = {}
                        # Getting the type of 'field' (line 335)
                        field_272229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 24), 'field', False)
                        # Obtaining the member 'dateTime' of a type (line 335)
                        dateTime_272230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 24), field_272229, 'dateTime')
                        # Calling dateTime(args, kwargs) (line 335)
                        dateTime_call_result_272232 = invoke(stypy.reporting.localization.Localization(__file__, 335, 24), dateTime_272230, *[], **kwargs_272231)
                        
                        # Obtaining the member 'toPyDateTime' of a type (line 335)
                        toPyDateTime_272233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 24), dateTime_call_result_272232, 'toPyDateTime')
                        # Calling toPyDateTime(args, kwargs) (line 335)
                        toPyDateTime_call_result_272235 = invoke(stypy.reporting.localization.Localization(__file__, 335, 24), toPyDateTime_272233, *[], **kwargs_272234)
                        
                        # Assigning a type to the variable 'value' (line 335)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 16), 'value', toPyDateTime_call_result_272235)
                        # SSA branch for the else part of an if statement (line 334)
                        module_type_store.open_ssa_branch('else')
                        
                        
                        # Call to isinstance(...): (line 336)
                        # Processing the call arguments (line 336)
                        # Getting the type of 'value' (line 336)
                        value_272237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 28), 'value', False)
                        # Getting the type of 'datetime' (line 336)
                        datetime_272238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 35), 'datetime', False)
                        # Obtaining the member 'date' of a type (line 336)
                        date_272239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 35), datetime_272238, 'date')
                        # Processing the call keyword arguments (line 336)
                        kwargs_272240 = {}
                        # Getting the type of 'isinstance' (line 336)
                        isinstance_272236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 17), 'isinstance', False)
                        # Calling isinstance(args, kwargs) (line 336)
                        isinstance_call_result_272241 = invoke(stypy.reporting.localization.Localization(__file__, 336, 17), isinstance_272236, *[value_272237, date_272239], **kwargs_272240)
                        
                        # Testing the type of an if condition (line 336)
                        if_condition_272242 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 336, 17), isinstance_call_result_272241)
                        # Assigning a type to the variable 'if_condition_272242' (line 336)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 17), 'if_condition_272242', if_condition_272242)
                        # SSA begins for if statement (line 336)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Call to a Name (line 337):
                        
                        # Assigning a Call to a Name (line 337):
                        
                        # Call to toPyDate(...): (line 337)
                        # Processing the call keyword arguments (line 337)
                        kwargs_272248 = {}
                        
                        # Call to date(...): (line 337)
                        # Processing the call keyword arguments (line 337)
                        kwargs_272245 = {}
                        # Getting the type of 'field' (line 337)
                        field_272243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 24), 'field', False)
                        # Obtaining the member 'date' of a type (line 337)
                        date_272244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 24), field_272243, 'date')
                        # Calling date(args, kwargs) (line 337)
                        date_call_result_272246 = invoke(stypy.reporting.localization.Localization(__file__, 337, 24), date_272244, *[], **kwargs_272245)
                        
                        # Obtaining the member 'toPyDate' of a type (line 337)
                        toPyDate_272247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 24), date_call_result_272246, 'toPyDate')
                        # Calling toPyDate(args, kwargs) (line 337)
                        toPyDate_call_result_272249 = invoke(stypy.reporting.localization.Localization(__file__, 337, 24), toPyDate_272247, *[], **kwargs_272248)
                        
                        # Assigning a type to the variable 'value' (line 337)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 16), 'value', toPyDate_call_result_272249)
                        # SSA branch for the else part of an if statement (line 336)
                        module_type_store.open_ssa_branch('else')
                        
                        # Assigning a Call to a Name (line 339):
                        
                        # Assigning a Call to a Name (line 339):
                        
                        # Call to eval(...): (line 339)
                        # Processing the call arguments (line 339)
                        
                        # Call to str(...): (line 339)
                        # Processing the call arguments (line 339)
                        
                        # Call to text(...): (line 339)
                        # Processing the call keyword arguments (line 339)
                        kwargs_272254 = {}
                        # Getting the type of 'field' (line 339)
                        field_272252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 33), 'field', False)
                        # Obtaining the member 'text' of a type (line 339)
                        text_272253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 33), field_272252, 'text')
                        # Calling text(args, kwargs) (line 339)
                        text_call_result_272255 = invoke(stypy.reporting.localization.Localization(__file__, 339, 33), text_272253, *[], **kwargs_272254)
                        
                        # Processing the call keyword arguments (line 339)
                        kwargs_272256 = {}
                        # Getting the type of 'str' (line 339)
                        str_272251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 29), 'str', False)
                        # Calling str(args, kwargs) (line 339)
                        str_call_result_272257 = invoke(stypy.reporting.localization.Localization(__file__, 339, 29), str_272251, *[text_call_result_272255], **kwargs_272256)
                        
                        # Processing the call keyword arguments (line 339)
                        kwargs_272258 = {}
                        # Getting the type of 'eval' (line 339)
                        eval_272250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 24), 'eval', False)
                        # Calling eval(args, kwargs) (line 339)
                        eval_call_result_272259 = invoke(stypy.reporting.localization.Localization(__file__, 339, 24), eval_272250, *[str_call_result_272257], **kwargs_272258)
                        
                        # Assigning a type to the variable 'value' (line 339)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 16), 'value', eval_call_result_272259)
                        # SSA join for if statement (line 336)
                        module_type_store = module_type_store.join_ssa_context()
                        
                        # SSA join for if statement (line 334)
                        module_type_store = module_type_store.join_ssa_context()
                        

                        if (may_be_272213 and more_types_in_union_272214):
                            # SSA join for if statement (line 332)
                            module_type_store = module_type_store.join_ssa_context()


                    

                    if (may_be_272199 and more_types_in_union_272200):
                        # SSA join for if statement (line 330)
                        module_type_store = module_type_store.join_ssa_context()


                

                if (may_be_272187 and more_types_in_union_272188):
                    # SSA join for if statement (line 328)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 322)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 319)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 317)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_272114 and more_types_in_union_272115):
                # SSA join for if statement (line 314)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to append(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'value' (line 340)
        value_272262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 29), 'value', False)
        # Processing the call keyword arguments (line 340)
        kwargs_272263 = {}
        # Getting the type of 'valuelist' (line 340)
        valuelist_272260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 12), 'valuelist', False)
        # Obtaining the member 'append' of a type (line 340)
        append_272261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 12), valuelist_272260, 'append')
        # Calling append(args, kwargs) (line 340)
        append_call_result_272264 = invoke(stypy.reporting.localization.Localization(__file__, 340, 12), append_272261, *[value_272262], **kwargs_272263)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'valuelist' (line 341)
        valuelist_272265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 15), 'valuelist')
        # Assigning a type to the variable 'stypy_return_type' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'stypy_return_type', valuelist_272265)
        
        # ################# End of 'get(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get' in the type store
        # Getting the type of 'stypy_return_type' (line 310)
        stypy_return_type_272266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_272266)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get'
        return stypy_return_type_272266


# Assigning a type to the variable 'FormWidget' (line 215)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), 'FormWidget', FormWidget)

# Assigning a Call to a Name (line 216):

# Call to Signal(...): (line 216)
# Processing the call keyword arguments (line 216)
kwargs_272269 = {}
# Getting the type of 'QtCore' (line 216)
QtCore_272267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 21), 'QtCore', False)
# Obtaining the member 'Signal' of a type (line 216)
Signal_272268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 21), QtCore_272267, 'Signal')
# Calling Signal(args, kwargs) (line 216)
Signal_call_result_272270 = invoke(stypy.reporting.localization.Localization(__file__, 216, 21), Signal_272268, *[], **kwargs_272269)

# Getting the type of 'FormWidget'
FormWidget_272271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FormWidget')
# Setting the type of the member 'update_buttons' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FormWidget_272271, 'update_buttons', Signal_call_result_272270)
# Declaration of the 'FormComboWidget' class
# Getting the type of 'QtWidgets' (line 344)
QtWidgets_272272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 22), 'QtWidgets')
# Obtaining the member 'QWidget' of a type (line 344)
QWidget_272273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 22), QtWidgets_272272, 'QWidget')

class FormComboWidget(QWidget_272273, ):
    
    # Assigning a Call to a Name (line 345):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        unicode_272274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 41), 'unicode', u'')
        # Getting the type of 'None' (line 347)
        None_272275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 52), 'None')
        defaults = [unicode_272274, None_272275]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 347, 4, False)
        # Assigning a type to the variable 'self' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FormComboWidget.__init__', ['datalist', 'comment', 'parent'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['datalist', 'comment', 'parent'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 348)
        # Processing the call arguments (line 348)
        # Getting the type of 'self' (line 348)
        self_272279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 35), 'self', False)
        # Getting the type of 'parent' (line 348)
        parent_272280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 41), 'parent', False)
        # Processing the call keyword arguments (line 348)
        kwargs_272281 = {}
        # Getting the type of 'QtWidgets' (line 348)
        QtWidgets_272276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'QtWidgets', False)
        # Obtaining the member 'QWidget' of a type (line 348)
        QWidget_272277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 8), QtWidgets_272276, 'QWidget')
        # Obtaining the member '__init__' of a type (line 348)
        init___272278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 8), QWidget_272277, '__init__')
        # Calling __init__(args, kwargs) (line 348)
        init___call_result_272282 = invoke(stypy.reporting.localization.Localization(__file__, 348, 8), init___272278, *[self_272279, parent_272280], **kwargs_272281)
        
        
        # Assigning a Call to a Name (line 349):
        
        # Assigning a Call to a Name (line 349):
        
        # Call to QVBoxLayout(...): (line 349)
        # Processing the call keyword arguments (line 349)
        kwargs_272285 = {}
        # Getting the type of 'QtWidgets' (line 349)
        QtWidgets_272283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 17), 'QtWidgets', False)
        # Obtaining the member 'QVBoxLayout' of a type (line 349)
        QVBoxLayout_272284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 17), QtWidgets_272283, 'QVBoxLayout')
        # Calling QVBoxLayout(args, kwargs) (line 349)
        QVBoxLayout_call_result_272286 = invoke(stypy.reporting.localization.Localization(__file__, 349, 17), QVBoxLayout_272284, *[], **kwargs_272285)
        
        # Assigning a type to the variable 'layout' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'layout', QVBoxLayout_call_result_272286)
        
        # Call to setLayout(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'layout' (line 350)
        layout_272289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 23), 'layout', False)
        # Processing the call keyword arguments (line 350)
        kwargs_272290 = {}
        # Getting the type of 'self' (line 350)
        self_272287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'self', False)
        # Obtaining the member 'setLayout' of a type (line 350)
        setLayout_272288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), self_272287, 'setLayout')
        # Calling setLayout(args, kwargs) (line 350)
        setLayout_call_result_272291 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), setLayout_272288, *[layout_272289], **kwargs_272290)
        
        
        # Assigning a Call to a Attribute (line 351):
        
        # Assigning a Call to a Attribute (line 351):
        
        # Call to QComboBox(...): (line 351)
        # Processing the call keyword arguments (line 351)
        kwargs_272294 = {}
        # Getting the type of 'QtWidgets' (line 351)
        QtWidgets_272292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 24), 'QtWidgets', False)
        # Obtaining the member 'QComboBox' of a type (line 351)
        QComboBox_272293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 24), QtWidgets_272292, 'QComboBox')
        # Calling QComboBox(args, kwargs) (line 351)
        QComboBox_call_result_272295 = invoke(stypy.reporting.localization.Localization(__file__, 351, 24), QComboBox_272293, *[], **kwargs_272294)
        
        # Getting the type of 'self' (line 351)
        self_272296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'self')
        # Setting the type of the member 'combobox' of a type (line 351)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 8), self_272296, 'combobox', QComboBox_call_result_272295)
        
        # Call to addWidget(...): (line 352)
        # Processing the call arguments (line 352)
        # Getting the type of 'self' (line 352)
        self_272299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 25), 'self', False)
        # Obtaining the member 'combobox' of a type (line 352)
        combobox_272300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 25), self_272299, 'combobox')
        # Processing the call keyword arguments (line 352)
        kwargs_272301 = {}
        # Getting the type of 'layout' (line 352)
        layout_272297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'layout', False)
        # Obtaining the member 'addWidget' of a type (line 352)
        addWidget_272298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 8), layout_272297, 'addWidget')
        # Calling addWidget(args, kwargs) (line 352)
        addWidget_call_result_272302 = invoke(stypy.reporting.localization.Localization(__file__, 352, 8), addWidget_272298, *[combobox_272300], **kwargs_272301)
        
        
        # Assigning a Call to a Attribute (line 354):
        
        # Assigning a Call to a Attribute (line 354):
        
        # Call to QStackedWidget(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'self' (line 354)
        self_272305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 52), 'self', False)
        # Processing the call keyword arguments (line 354)
        kwargs_272306 = {}
        # Getting the type of 'QtWidgets' (line 354)
        QtWidgets_272303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 27), 'QtWidgets', False)
        # Obtaining the member 'QStackedWidget' of a type (line 354)
        QStackedWidget_272304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 27), QtWidgets_272303, 'QStackedWidget')
        # Calling QStackedWidget(args, kwargs) (line 354)
        QStackedWidget_call_result_272307 = invoke(stypy.reporting.localization.Localization(__file__, 354, 27), QStackedWidget_272304, *[self_272305], **kwargs_272306)
        
        # Getting the type of 'self' (line 354)
        self_272308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'self')
        # Setting the type of the member 'stackwidget' of a type (line 354)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 8), self_272308, 'stackwidget', QStackedWidget_call_result_272307)
        
        # Call to addWidget(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'self' (line 355)
        self_272311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 25), 'self', False)
        # Obtaining the member 'stackwidget' of a type (line 355)
        stackwidget_272312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 25), self_272311, 'stackwidget')
        # Processing the call keyword arguments (line 355)
        kwargs_272313 = {}
        # Getting the type of 'layout' (line 355)
        layout_272309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'layout', False)
        # Obtaining the member 'addWidget' of a type (line 355)
        addWidget_272310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 8), layout_272309, 'addWidget')
        # Calling addWidget(args, kwargs) (line 355)
        addWidget_call_result_272314 = invoke(stypy.reporting.localization.Localization(__file__, 355, 8), addWidget_272310, *[stackwidget_272312], **kwargs_272313)
        
        
        # Call to connect(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'self' (line 356)
        self_272319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 50), 'self', False)
        # Obtaining the member 'stackwidget' of a type (line 356)
        stackwidget_272320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 50), self_272319, 'stackwidget')
        # Obtaining the member 'setCurrentIndex' of a type (line 356)
        setCurrentIndex_272321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 50), stackwidget_272320, 'setCurrentIndex')
        # Processing the call keyword arguments (line 356)
        kwargs_272322 = {}
        # Getting the type of 'self' (line 356)
        self_272315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'self', False)
        # Obtaining the member 'combobox' of a type (line 356)
        combobox_272316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 8), self_272315, 'combobox')
        # Obtaining the member 'currentIndexChanged' of a type (line 356)
        currentIndexChanged_272317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 8), combobox_272316, 'currentIndexChanged')
        # Obtaining the member 'connect' of a type (line 356)
        connect_272318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 8), currentIndexChanged_272317, 'connect')
        # Calling connect(args, kwargs) (line 356)
        connect_call_result_272323 = invoke(stypy.reporting.localization.Localization(__file__, 356, 8), connect_272318, *[setCurrentIndex_272321], **kwargs_272322)
        
        
        # Assigning a List to a Attribute (line 358):
        
        # Assigning a List to a Attribute (line 358):
        
        # Obtaining an instance of the builtin type 'list' (line 358)
        list_272324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 358)
        
        # Getting the type of 'self' (line 358)
        self_272325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'self')
        # Setting the type of the member 'widgetlist' of a type (line 358)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 8), self_272325, 'widgetlist', list_272324)
        
        # Getting the type of 'datalist' (line 359)
        datalist_272326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 36), 'datalist')
        # Testing the type of a for loop iterable (line 359)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 359, 8), datalist_272326)
        # Getting the type of the for loop variable (line 359)
        for_loop_var_272327 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 359, 8), datalist_272326)
        # Assigning a type to the variable 'data' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'data', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 8), for_loop_var_272327))
        # Assigning a type to the variable 'title' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'title', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 8), for_loop_var_272327))
        # Assigning a type to the variable 'comment' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'comment', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 8), for_loop_var_272327))
        # SSA begins for a for statement (line 359)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to addItem(...): (line 360)
        # Processing the call arguments (line 360)
        # Getting the type of 'title' (line 360)
        title_272331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 34), 'title', False)
        # Processing the call keyword arguments (line 360)
        kwargs_272332 = {}
        # Getting the type of 'self' (line 360)
        self_272328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'self', False)
        # Obtaining the member 'combobox' of a type (line 360)
        combobox_272329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 12), self_272328, 'combobox')
        # Obtaining the member 'addItem' of a type (line 360)
        addItem_272330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 12), combobox_272329, 'addItem')
        # Calling addItem(args, kwargs) (line 360)
        addItem_call_result_272333 = invoke(stypy.reporting.localization.Localization(__file__, 360, 12), addItem_272330, *[title_272331], **kwargs_272332)
        
        
        # Assigning a Call to a Name (line 361):
        
        # Assigning a Call to a Name (line 361):
        
        # Call to FormWidget(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of 'data' (line 361)
        data_272335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 32), 'data', False)
        # Processing the call keyword arguments (line 361)
        # Getting the type of 'comment' (line 361)
        comment_272336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 46), 'comment', False)
        keyword_272337 = comment_272336
        # Getting the type of 'self' (line 361)
        self_272338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 62), 'self', False)
        keyword_272339 = self_272338
        kwargs_272340 = {'comment': keyword_272337, 'parent': keyword_272339}
        # Getting the type of 'FormWidget' (line 361)
        FormWidget_272334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 21), 'FormWidget', False)
        # Calling FormWidget(args, kwargs) (line 361)
        FormWidget_call_result_272341 = invoke(stypy.reporting.localization.Localization(__file__, 361, 21), FormWidget_272334, *[data_272335], **kwargs_272340)
        
        # Assigning a type to the variable 'widget' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'widget', FormWidget_call_result_272341)
        
        # Call to addWidget(...): (line 362)
        # Processing the call arguments (line 362)
        # Getting the type of 'widget' (line 362)
        widget_272345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 39), 'widget', False)
        # Processing the call keyword arguments (line 362)
        kwargs_272346 = {}
        # Getting the type of 'self' (line 362)
        self_272342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'self', False)
        # Obtaining the member 'stackwidget' of a type (line 362)
        stackwidget_272343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 12), self_272342, 'stackwidget')
        # Obtaining the member 'addWidget' of a type (line 362)
        addWidget_272344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 12), stackwidget_272343, 'addWidget')
        # Calling addWidget(args, kwargs) (line 362)
        addWidget_call_result_272347 = invoke(stypy.reporting.localization.Localization(__file__, 362, 12), addWidget_272344, *[widget_272345], **kwargs_272346)
        
        
        # Call to append(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'widget' (line 363)
        widget_272351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 35), 'widget', False)
        # Processing the call keyword arguments (line 363)
        kwargs_272352 = {}
        # Getting the type of 'self' (line 363)
        self_272348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'self', False)
        # Obtaining the member 'widgetlist' of a type (line 363)
        widgetlist_272349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 12), self_272348, 'widgetlist')
        # Obtaining the member 'append' of a type (line 363)
        append_272350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 12), widgetlist_272349, 'append')
        # Calling append(args, kwargs) (line 363)
        append_call_result_272353 = invoke(stypy.reporting.localization.Localization(__file__, 363, 12), append_272350, *[widget_272351], **kwargs_272352)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def setup(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup'
        module_type_store = module_type_store.open_function_context('setup', 365, 4, False)
        # Assigning a type to the variable 'self' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FormComboWidget.setup.__dict__.__setitem__('stypy_localization', localization)
        FormComboWidget.setup.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FormComboWidget.setup.__dict__.__setitem__('stypy_type_store', module_type_store)
        FormComboWidget.setup.__dict__.__setitem__('stypy_function_name', 'FormComboWidget.setup')
        FormComboWidget.setup.__dict__.__setitem__('stypy_param_names_list', [])
        FormComboWidget.setup.__dict__.__setitem__('stypy_varargs_param_name', None)
        FormComboWidget.setup.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FormComboWidget.setup.__dict__.__setitem__('stypy_call_defaults', defaults)
        FormComboWidget.setup.__dict__.__setitem__('stypy_call_varargs', varargs)
        FormComboWidget.setup.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FormComboWidget.setup.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FormComboWidget.setup', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup(...)' code ##################

        
        # Getting the type of 'self' (line 366)
        self_272354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 22), 'self')
        # Obtaining the member 'widgetlist' of a type (line 366)
        widgetlist_272355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 22), self_272354, 'widgetlist')
        # Testing the type of a for loop iterable (line 366)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 366, 8), widgetlist_272355)
        # Getting the type of the for loop variable (line 366)
        for_loop_var_272356 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 366, 8), widgetlist_272355)
        # Assigning a type to the variable 'widget' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'widget', for_loop_var_272356)
        # SSA begins for a for statement (line 366)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to setup(...): (line 367)
        # Processing the call keyword arguments (line 367)
        kwargs_272359 = {}
        # Getting the type of 'widget' (line 367)
        widget_272357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'widget', False)
        # Obtaining the member 'setup' of a type (line 367)
        setup_272358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 12), widget_272357, 'setup')
        # Calling setup(args, kwargs) (line 367)
        setup_call_result_272360 = invoke(stypy.reporting.localization.Localization(__file__, 367, 12), setup_272358, *[], **kwargs_272359)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'setup(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup' in the type store
        # Getting the type of 'stypy_return_type' (line 365)
        stypy_return_type_272361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_272361)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup'
        return stypy_return_type_272361


    @norecursion
    def get(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get'
        module_type_store = module_type_store.open_function_context('get', 369, 4, False)
        # Assigning a type to the variable 'self' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FormComboWidget.get.__dict__.__setitem__('stypy_localization', localization)
        FormComboWidget.get.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FormComboWidget.get.__dict__.__setitem__('stypy_type_store', module_type_store)
        FormComboWidget.get.__dict__.__setitem__('stypy_function_name', 'FormComboWidget.get')
        FormComboWidget.get.__dict__.__setitem__('stypy_param_names_list', [])
        FormComboWidget.get.__dict__.__setitem__('stypy_varargs_param_name', None)
        FormComboWidget.get.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FormComboWidget.get.__dict__.__setitem__('stypy_call_defaults', defaults)
        FormComboWidget.get.__dict__.__setitem__('stypy_call_varargs', varargs)
        FormComboWidget.get.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FormComboWidget.get.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FormComboWidget.get', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get(...)' code ##################

        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 370)
        self_272366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 43), 'self')
        # Obtaining the member 'widgetlist' of a type (line 370)
        widgetlist_272367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 43), self_272366, 'widgetlist')
        comprehension_272368 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 16), widgetlist_272367)
        # Assigning a type to the variable 'widget' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 16), 'widget', comprehension_272368)
        
        # Call to get(...): (line 370)
        # Processing the call keyword arguments (line 370)
        kwargs_272364 = {}
        # Getting the type of 'widget' (line 370)
        widget_272362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 16), 'widget', False)
        # Obtaining the member 'get' of a type (line 370)
        get_272363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 16), widget_272362, 'get')
        # Calling get(args, kwargs) (line 370)
        get_call_result_272365 = invoke(stypy.reporting.localization.Localization(__file__, 370, 16), get_272363, *[], **kwargs_272364)
        
        list_272369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 16), list_272369, get_call_result_272365)
        # Assigning a type to the variable 'stypy_return_type' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'stypy_return_type', list_272369)
        
        # ################# End of 'get(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get' in the type store
        # Getting the type of 'stypy_return_type' (line 369)
        stypy_return_type_272370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_272370)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get'
        return stypy_return_type_272370


# Assigning a type to the variable 'FormComboWidget' (line 344)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 0), 'FormComboWidget', FormComboWidget)

# Assigning a Call to a Name (line 345):

# Call to Signal(...): (line 345)
# Processing the call keyword arguments (line 345)
kwargs_272373 = {}
# Getting the type of 'QtCore' (line 345)
QtCore_272371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 21), 'QtCore', False)
# Obtaining the member 'Signal' of a type (line 345)
Signal_272372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 21), QtCore_272371, 'Signal')
# Calling Signal(args, kwargs) (line 345)
Signal_call_result_272374 = invoke(stypy.reporting.localization.Localization(__file__, 345, 21), Signal_272372, *[], **kwargs_272373)

# Getting the type of 'FormComboWidget'
FormComboWidget_272375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FormComboWidget')
# Setting the type of the member 'update_buttons' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FormComboWidget_272375, 'update_buttons', Signal_call_result_272374)
# Declaration of the 'FormTabWidget' class
# Getting the type of 'QtWidgets' (line 373)
QtWidgets_272376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 20), 'QtWidgets')
# Obtaining the member 'QWidget' of a type (line 373)
QWidget_272377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 20), QtWidgets_272376, 'QWidget')

class FormTabWidget(QWidget_272377, ):
    
    # Assigning a Call to a Name (line 374):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        unicode_272378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 41), 'unicode', u'')
        # Getting the type of 'None' (line 376)
        None_272379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 52), 'None')
        defaults = [unicode_272378, None_272379]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 376, 4, False)
        # Assigning a type to the variable 'self' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FormTabWidget.__init__', ['datalist', 'comment', 'parent'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['datalist', 'comment', 'parent'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 377)
        # Processing the call arguments (line 377)
        # Getting the type of 'self' (line 377)
        self_272383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 35), 'self', False)
        # Getting the type of 'parent' (line 377)
        parent_272384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 41), 'parent', False)
        # Processing the call keyword arguments (line 377)
        kwargs_272385 = {}
        # Getting the type of 'QtWidgets' (line 377)
        QtWidgets_272380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'QtWidgets', False)
        # Obtaining the member 'QWidget' of a type (line 377)
        QWidget_272381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 8), QtWidgets_272380, 'QWidget')
        # Obtaining the member '__init__' of a type (line 377)
        init___272382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 8), QWidget_272381, '__init__')
        # Calling __init__(args, kwargs) (line 377)
        init___call_result_272386 = invoke(stypy.reporting.localization.Localization(__file__, 377, 8), init___272382, *[self_272383, parent_272384], **kwargs_272385)
        
        
        # Assigning a Call to a Name (line 378):
        
        # Assigning a Call to a Name (line 378):
        
        # Call to QVBoxLayout(...): (line 378)
        # Processing the call keyword arguments (line 378)
        kwargs_272389 = {}
        # Getting the type of 'QtWidgets' (line 378)
        QtWidgets_272387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 17), 'QtWidgets', False)
        # Obtaining the member 'QVBoxLayout' of a type (line 378)
        QVBoxLayout_272388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 17), QtWidgets_272387, 'QVBoxLayout')
        # Calling QVBoxLayout(args, kwargs) (line 378)
        QVBoxLayout_call_result_272390 = invoke(stypy.reporting.localization.Localization(__file__, 378, 17), QVBoxLayout_272388, *[], **kwargs_272389)
        
        # Assigning a type to the variable 'layout' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'layout', QVBoxLayout_call_result_272390)
        
        # Assigning a Call to a Attribute (line 379):
        
        # Assigning a Call to a Attribute (line 379):
        
        # Call to QTabWidget(...): (line 379)
        # Processing the call keyword arguments (line 379)
        kwargs_272393 = {}
        # Getting the type of 'QtWidgets' (line 379)
        QtWidgets_272391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 25), 'QtWidgets', False)
        # Obtaining the member 'QTabWidget' of a type (line 379)
        QTabWidget_272392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 25), QtWidgets_272391, 'QTabWidget')
        # Calling QTabWidget(args, kwargs) (line 379)
        QTabWidget_call_result_272394 = invoke(stypy.reporting.localization.Localization(__file__, 379, 25), QTabWidget_272392, *[], **kwargs_272393)
        
        # Getting the type of 'self' (line 379)
        self_272395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'self')
        # Setting the type of the member 'tabwidget' of a type (line 379)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 8), self_272395, 'tabwidget', QTabWidget_call_result_272394)
        
        # Call to addWidget(...): (line 380)
        # Processing the call arguments (line 380)
        # Getting the type of 'self' (line 380)
        self_272398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 25), 'self', False)
        # Obtaining the member 'tabwidget' of a type (line 380)
        tabwidget_272399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 25), self_272398, 'tabwidget')
        # Processing the call keyword arguments (line 380)
        kwargs_272400 = {}
        # Getting the type of 'layout' (line 380)
        layout_272396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'layout', False)
        # Obtaining the member 'addWidget' of a type (line 380)
        addWidget_272397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 8), layout_272396, 'addWidget')
        # Calling addWidget(args, kwargs) (line 380)
        addWidget_call_result_272401 = invoke(stypy.reporting.localization.Localization(__file__, 380, 8), addWidget_272397, *[tabwidget_272399], **kwargs_272400)
        
        
        # Call to setLayout(...): (line 381)
        # Processing the call arguments (line 381)
        # Getting the type of 'layout' (line 381)
        layout_272404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 23), 'layout', False)
        # Processing the call keyword arguments (line 381)
        kwargs_272405 = {}
        # Getting the type of 'self' (line 381)
        self_272402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'self', False)
        # Obtaining the member 'setLayout' of a type (line 381)
        setLayout_272403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 8), self_272402, 'setLayout')
        # Calling setLayout(args, kwargs) (line 381)
        setLayout_call_result_272406 = invoke(stypy.reporting.localization.Localization(__file__, 381, 8), setLayout_272403, *[layout_272404], **kwargs_272405)
        
        
        # Assigning a List to a Attribute (line 382):
        
        # Assigning a List to a Attribute (line 382):
        
        # Obtaining an instance of the builtin type 'list' (line 382)
        list_272407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 382)
        
        # Getting the type of 'self' (line 382)
        self_272408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'self')
        # Setting the type of the member 'widgetlist' of a type (line 382)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 8), self_272408, 'widgetlist', list_272407)
        
        # Getting the type of 'datalist' (line 383)
        datalist_272409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 36), 'datalist')
        # Testing the type of a for loop iterable (line 383)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 383, 8), datalist_272409)
        # Getting the type of the for loop variable (line 383)
        for_loop_var_272410 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 383, 8), datalist_272409)
        # Assigning a type to the variable 'data' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'data', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 8), for_loop_var_272410))
        # Assigning a type to the variable 'title' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'title', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 8), for_loop_var_272410))
        # Assigning a type to the variable 'comment' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'comment', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 8), for_loop_var_272410))
        # SSA begins for a for statement (line 383)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to len(...): (line 384)
        # Processing the call arguments (line 384)
        
        # Obtaining the type of the subscript
        int_272412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 24), 'int')
        # Getting the type of 'data' (line 384)
        data_272413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 19), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 384)
        getitem___272414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 19), data_272413, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 384)
        subscript_call_result_272415 = invoke(stypy.reporting.localization.Localization(__file__, 384, 19), getitem___272414, int_272412)
        
        # Processing the call keyword arguments (line 384)
        kwargs_272416 = {}
        # Getting the type of 'len' (line 384)
        len_272411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 15), 'len', False)
        # Calling len(args, kwargs) (line 384)
        len_call_result_272417 = invoke(stypy.reporting.localization.Localization(__file__, 384, 15), len_272411, *[subscript_call_result_272415], **kwargs_272416)
        
        int_272418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 31), 'int')
        # Applying the binary operator '==' (line 384)
        result_eq_272419 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 15), '==', len_call_result_272417, int_272418)
        
        # Testing the type of an if condition (line 384)
        if_condition_272420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 384, 12), result_eq_272419)
        # Assigning a type to the variable 'if_condition_272420' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'if_condition_272420', if_condition_272420)
        # SSA begins for if statement (line 384)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 385):
        
        # Assigning a Call to a Name (line 385):
        
        # Call to FormComboWidget(...): (line 385)
        # Processing the call arguments (line 385)
        # Getting the type of 'data' (line 385)
        data_272422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 41), 'data', False)
        # Processing the call keyword arguments (line 385)
        # Getting the type of 'comment' (line 385)
        comment_272423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 55), 'comment', False)
        keyword_272424 = comment_272423
        # Getting the type of 'self' (line 385)
        self_272425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 71), 'self', False)
        keyword_272426 = self_272425
        kwargs_272427 = {'comment': keyword_272424, 'parent': keyword_272426}
        # Getting the type of 'FormComboWidget' (line 385)
        FormComboWidget_272421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 25), 'FormComboWidget', False)
        # Calling FormComboWidget(args, kwargs) (line 385)
        FormComboWidget_call_result_272428 = invoke(stypy.reporting.localization.Localization(__file__, 385, 25), FormComboWidget_272421, *[data_272422], **kwargs_272427)
        
        # Assigning a type to the variable 'widget' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 16), 'widget', FormComboWidget_call_result_272428)
        # SSA branch for the else part of an if statement (line 384)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 387):
        
        # Assigning a Call to a Name (line 387):
        
        # Call to FormWidget(...): (line 387)
        # Processing the call arguments (line 387)
        # Getting the type of 'data' (line 387)
        data_272430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 36), 'data', False)
        # Processing the call keyword arguments (line 387)
        # Getting the type of 'comment' (line 387)
        comment_272431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 50), 'comment', False)
        keyword_272432 = comment_272431
        # Getting the type of 'self' (line 387)
        self_272433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 66), 'self', False)
        keyword_272434 = self_272433
        kwargs_272435 = {'comment': keyword_272432, 'parent': keyword_272434}
        # Getting the type of 'FormWidget' (line 387)
        FormWidget_272429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 25), 'FormWidget', False)
        # Calling FormWidget(args, kwargs) (line 387)
        FormWidget_call_result_272436 = invoke(stypy.reporting.localization.Localization(__file__, 387, 25), FormWidget_272429, *[data_272430], **kwargs_272435)
        
        # Assigning a type to the variable 'widget' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 16), 'widget', FormWidget_call_result_272436)
        # SSA join for if statement (line 384)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 388):
        
        # Assigning a Call to a Name (line 388):
        
        # Call to addTab(...): (line 388)
        # Processing the call arguments (line 388)
        # Getting the type of 'widget' (line 388)
        widget_272440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 42), 'widget', False)
        # Getting the type of 'title' (line 388)
        title_272441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 50), 'title', False)
        # Processing the call keyword arguments (line 388)
        kwargs_272442 = {}
        # Getting the type of 'self' (line 388)
        self_272437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 20), 'self', False)
        # Obtaining the member 'tabwidget' of a type (line 388)
        tabwidget_272438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 20), self_272437, 'tabwidget')
        # Obtaining the member 'addTab' of a type (line 388)
        addTab_272439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 20), tabwidget_272438, 'addTab')
        # Calling addTab(args, kwargs) (line 388)
        addTab_call_result_272443 = invoke(stypy.reporting.localization.Localization(__file__, 388, 20), addTab_272439, *[widget_272440, title_272441], **kwargs_272442)
        
        # Assigning a type to the variable 'index' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'index', addTab_call_result_272443)
        
        # Call to setTabToolTip(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 'index' (line 389)
        index_272447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 41), 'index', False)
        # Getting the type of 'comment' (line 389)
        comment_272448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 48), 'comment', False)
        # Processing the call keyword arguments (line 389)
        kwargs_272449 = {}
        # Getting the type of 'self' (line 389)
        self_272444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'self', False)
        # Obtaining the member 'tabwidget' of a type (line 389)
        tabwidget_272445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 12), self_272444, 'tabwidget')
        # Obtaining the member 'setTabToolTip' of a type (line 389)
        setTabToolTip_272446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 12), tabwidget_272445, 'setTabToolTip')
        # Calling setTabToolTip(args, kwargs) (line 389)
        setTabToolTip_call_result_272450 = invoke(stypy.reporting.localization.Localization(__file__, 389, 12), setTabToolTip_272446, *[index_272447, comment_272448], **kwargs_272449)
        
        
        # Call to append(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 'widget' (line 390)
        widget_272454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 35), 'widget', False)
        # Processing the call keyword arguments (line 390)
        kwargs_272455 = {}
        # Getting the type of 'self' (line 390)
        self_272451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'self', False)
        # Obtaining the member 'widgetlist' of a type (line 390)
        widgetlist_272452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 12), self_272451, 'widgetlist')
        # Obtaining the member 'append' of a type (line 390)
        append_272453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 12), widgetlist_272452, 'append')
        # Calling append(args, kwargs) (line 390)
        append_call_result_272456 = invoke(stypy.reporting.localization.Localization(__file__, 390, 12), append_272453, *[widget_272454], **kwargs_272455)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def setup(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup'
        module_type_store = module_type_store.open_function_context('setup', 392, 4, False)
        # Assigning a type to the variable 'self' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FormTabWidget.setup.__dict__.__setitem__('stypy_localization', localization)
        FormTabWidget.setup.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FormTabWidget.setup.__dict__.__setitem__('stypy_type_store', module_type_store)
        FormTabWidget.setup.__dict__.__setitem__('stypy_function_name', 'FormTabWidget.setup')
        FormTabWidget.setup.__dict__.__setitem__('stypy_param_names_list', [])
        FormTabWidget.setup.__dict__.__setitem__('stypy_varargs_param_name', None)
        FormTabWidget.setup.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FormTabWidget.setup.__dict__.__setitem__('stypy_call_defaults', defaults)
        FormTabWidget.setup.__dict__.__setitem__('stypy_call_varargs', varargs)
        FormTabWidget.setup.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FormTabWidget.setup.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FormTabWidget.setup', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup(...)' code ##################

        
        # Getting the type of 'self' (line 393)
        self_272457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 22), 'self')
        # Obtaining the member 'widgetlist' of a type (line 393)
        widgetlist_272458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 22), self_272457, 'widgetlist')
        # Testing the type of a for loop iterable (line 393)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 393, 8), widgetlist_272458)
        # Getting the type of the for loop variable (line 393)
        for_loop_var_272459 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 393, 8), widgetlist_272458)
        # Assigning a type to the variable 'widget' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'widget', for_loop_var_272459)
        # SSA begins for a for statement (line 393)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to setup(...): (line 394)
        # Processing the call keyword arguments (line 394)
        kwargs_272462 = {}
        # Getting the type of 'widget' (line 394)
        widget_272460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'widget', False)
        # Obtaining the member 'setup' of a type (line 394)
        setup_272461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 12), widget_272460, 'setup')
        # Calling setup(args, kwargs) (line 394)
        setup_call_result_272463 = invoke(stypy.reporting.localization.Localization(__file__, 394, 12), setup_272461, *[], **kwargs_272462)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'setup(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup' in the type store
        # Getting the type of 'stypy_return_type' (line 392)
        stypy_return_type_272464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_272464)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup'
        return stypy_return_type_272464


    @norecursion
    def get(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get'
        module_type_store = module_type_store.open_function_context('get', 396, 4, False)
        # Assigning a type to the variable 'self' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FormTabWidget.get.__dict__.__setitem__('stypy_localization', localization)
        FormTabWidget.get.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FormTabWidget.get.__dict__.__setitem__('stypy_type_store', module_type_store)
        FormTabWidget.get.__dict__.__setitem__('stypy_function_name', 'FormTabWidget.get')
        FormTabWidget.get.__dict__.__setitem__('stypy_param_names_list', [])
        FormTabWidget.get.__dict__.__setitem__('stypy_varargs_param_name', None)
        FormTabWidget.get.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FormTabWidget.get.__dict__.__setitem__('stypy_call_defaults', defaults)
        FormTabWidget.get.__dict__.__setitem__('stypy_call_varargs', varargs)
        FormTabWidget.get.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FormTabWidget.get.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FormTabWidget.get', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get(...)' code ##################

        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 397)
        self_272469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 43), 'self')
        # Obtaining the member 'widgetlist' of a type (line 397)
        widgetlist_272470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 43), self_272469, 'widgetlist')
        comprehension_272471 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 16), widgetlist_272470)
        # Assigning a type to the variable 'widget' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 16), 'widget', comprehension_272471)
        
        # Call to get(...): (line 397)
        # Processing the call keyword arguments (line 397)
        kwargs_272467 = {}
        # Getting the type of 'widget' (line 397)
        widget_272465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 16), 'widget', False)
        # Obtaining the member 'get' of a type (line 397)
        get_272466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 16), widget_272465, 'get')
        # Calling get(args, kwargs) (line 397)
        get_call_result_272468 = invoke(stypy.reporting.localization.Localization(__file__, 397, 16), get_272466, *[], **kwargs_272467)
        
        list_272472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 16), list_272472, get_call_result_272468)
        # Assigning a type to the variable 'stypy_return_type' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'stypy_return_type', list_272472)
        
        # ################# End of 'get(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get' in the type store
        # Getting the type of 'stypy_return_type' (line 396)
        stypy_return_type_272473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_272473)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get'
        return stypy_return_type_272473


# Assigning a type to the variable 'FormTabWidget' (line 373)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 0), 'FormTabWidget', FormTabWidget)

# Assigning a Call to a Name (line 374):

# Call to Signal(...): (line 374)
# Processing the call keyword arguments (line 374)
kwargs_272476 = {}
# Getting the type of 'QtCore' (line 374)
QtCore_272474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 21), 'QtCore', False)
# Obtaining the member 'Signal' of a type (line 374)
Signal_272475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 21), QtCore_272474, 'Signal')
# Calling Signal(args, kwargs) (line 374)
Signal_call_result_272477 = invoke(stypy.reporting.localization.Localization(__file__, 374, 21), Signal_272475, *[], **kwargs_272476)

# Getting the type of 'FormTabWidget'
FormTabWidget_272478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FormTabWidget')
# Setting the type of the member 'update_buttons' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FormTabWidget_272478, 'update_buttons', Signal_call_result_272477)
# Declaration of the 'FormDialog' class
# Getting the type of 'QtWidgets' (line 400)
QtWidgets_272479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 17), 'QtWidgets')
# Obtaining the member 'QDialog' of a type (line 400)
QDialog_272480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 17), QtWidgets_272479, 'QDialog')

class FormDialog(QDialog_272480, ):
    unicode_272481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 4), 'unicode', u'Form Dialog')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        unicode_272482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 35), 'unicode', u'')
        unicode_272483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 47), 'unicode', u'')
        # Getting the type of 'None' (line 403)
        None_272484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 22), 'None')
        # Getting the type of 'None' (line 403)
        None_272485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 35), 'None')
        # Getting the type of 'None' (line 403)
        None_272486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 47), 'None')
        defaults = [unicode_272482, unicode_272483, None_272484, None_272485, None_272486]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 402, 4, False)
        # Assigning a type to the variable 'self' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FormDialog.__init__', ['data', 'title', 'comment', 'icon', 'parent', 'apply'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['data', 'title', 'comment', 'icon', 'parent', 'apply'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 404)
        # Processing the call arguments (line 404)
        # Getting the type of 'self' (line 404)
        self_272490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 35), 'self', False)
        # Getting the type of 'parent' (line 404)
        parent_272491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 41), 'parent', False)
        # Processing the call keyword arguments (line 404)
        kwargs_272492 = {}
        # Getting the type of 'QtWidgets' (line 404)
        QtWidgets_272487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'QtWidgets', False)
        # Obtaining the member 'QDialog' of a type (line 404)
        QDialog_272488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 8), QtWidgets_272487, 'QDialog')
        # Obtaining the member '__init__' of a type (line 404)
        init___272489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 8), QDialog_272488, '__init__')
        # Calling __init__(args, kwargs) (line 404)
        init___call_result_272493 = invoke(stypy.reporting.localization.Localization(__file__, 404, 8), init___272489, *[self_272490, parent_272491], **kwargs_272492)
        
        
        # Assigning a Name to a Attribute (line 406):
        
        # Assigning a Name to a Attribute (line 406):
        # Getting the type of 'apply' (line 406)
        apply_272494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 30), 'apply')
        # Getting the type of 'self' (line 406)
        self_272495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'self')
        # Setting the type of the member 'apply_callback' of a type (line 406)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 8), self_272495, 'apply_callback', apply_272494)
        
        
        # Call to isinstance(...): (line 409)
        # Processing the call arguments (line 409)
        
        # Obtaining the type of the subscript
        int_272497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 30), 'int')
        
        # Obtaining the type of the subscript
        int_272498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 27), 'int')
        # Getting the type of 'data' (line 409)
        data_272499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 22), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 409)
        getitem___272500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 22), data_272499, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 409)
        subscript_call_result_272501 = invoke(stypy.reporting.localization.Localization(__file__, 409, 22), getitem___272500, int_272498)
        
        # Obtaining the member '__getitem__' of a type (line 409)
        getitem___272502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 22), subscript_call_result_272501, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 409)
        subscript_call_result_272503 = invoke(stypy.reporting.localization.Localization(__file__, 409, 22), getitem___272502, int_272497)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 409)
        tuple_272504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 409)
        # Adding element type (line 409)
        # Getting the type of 'list' (line 409)
        list_272505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 35), 'list', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 35), tuple_272504, list_272505)
        # Adding element type (line 409)
        # Getting the type of 'tuple' (line 409)
        tuple_272506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 41), 'tuple', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 35), tuple_272504, tuple_272506)
        
        # Processing the call keyword arguments (line 409)
        kwargs_272507 = {}
        # Getting the type of 'isinstance' (line 409)
        isinstance_272496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 409)
        isinstance_call_result_272508 = invoke(stypy.reporting.localization.Localization(__file__, 409, 11), isinstance_272496, *[subscript_call_result_272503, tuple_272504], **kwargs_272507)
        
        # Testing the type of an if condition (line 409)
        if_condition_272509 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 409, 8), isinstance_call_result_272508)
        # Assigning a type to the variable 'if_condition_272509' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'if_condition_272509', if_condition_272509)
        # SSA begins for if statement (line 409)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 410):
        
        # Assigning a Call to a Attribute (line 410):
        
        # Call to FormTabWidget(...): (line 410)
        # Processing the call arguments (line 410)
        # Getting the type of 'data' (line 410)
        data_272511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 44), 'data', False)
        # Processing the call keyword arguments (line 410)
        # Getting the type of 'comment' (line 410)
        comment_272512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 58), 'comment', False)
        keyword_272513 = comment_272512
        # Getting the type of 'self' (line 411)
        self_272514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 51), 'self', False)
        keyword_272515 = self_272514
        kwargs_272516 = {'comment': keyword_272513, 'parent': keyword_272515}
        # Getting the type of 'FormTabWidget' (line 410)
        FormTabWidget_272510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 30), 'FormTabWidget', False)
        # Calling FormTabWidget(args, kwargs) (line 410)
        FormTabWidget_call_result_272517 = invoke(stypy.reporting.localization.Localization(__file__, 410, 30), FormTabWidget_272510, *[data_272511], **kwargs_272516)
        
        # Getting the type of 'self' (line 410)
        self_272518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'self')
        # Setting the type of the member 'formwidget' of a type (line 410)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 12), self_272518, 'formwidget', FormTabWidget_call_result_272517)
        # SSA branch for the else part of an if statement (line 409)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to len(...): (line 412)
        # Processing the call arguments (line 412)
        
        # Obtaining the type of the subscript
        int_272520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 22), 'int')
        # Getting the type of 'data' (line 412)
        data_272521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 17), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 412)
        getitem___272522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 17), data_272521, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 412)
        subscript_call_result_272523 = invoke(stypy.reporting.localization.Localization(__file__, 412, 17), getitem___272522, int_272520)
        
        # Processing the call keyword arguments (line 412)
        kwargs_272524 = {}
        # Getting the type of 'len' (line 412)
        len_272519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 13), 'len', False)
        # Calling len(args, kwargs) (line 412)
        len_call_result_272525 = invoke(stypy.reporting.localization.Localization(__file__, 412, 13), len_272519, *[subscript_call_result_272523], **kwargs_272524)
        
        int_272526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 29), 'int')
        # Applying the binary operator '==' (line 412)
        result_eq_272527 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 13), '==', len_call_result_272525, int_272526)
        
        # Testing the type of an if condition (line 412)
        if_condition_272528 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 412, 13), result_eq_272527)
        # Assigning a type to the variable 'if_condition_272528' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 13), 'if_condition_272528', if_condition_272528)
        # SSA begins for if statement (line 412)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 413):
        
        # Assigning a Call to a Attribute (line 413):
        
        # Call to FormComboWidget(...): (line 413)
        # Processing the call arguments (line 413)
        # Getting the type of 'data' (line 413)
        data_272530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 46), 'data', False)
        # Processing the call keyword arguments (line 413)
        # Getting the type of 'comment' (line 413)
        comment_272531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 60), 'comment', False)
        keyword_272532 = comment_272531
        # Getting the type of 'self' (line 414)
        self_272533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 53), 'self', False)
        keyword_272534 = self_272533
        kwargs_272535 = {'comment': keyword_272532, 'parent': keyword_272534}
        # Getting the type of 'FormComboWidget' (line 413)
        FormComboWidget_272529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 30), 'FormComboWidget', False)
        # Calling FormComboWidget(args, kwargs) (line 413)
        FormComboWidget_call_result_272536 = invoke(stypy.reporting.localization.Localization(__file__, 413, 30), FormComboWidget_272529, *[data_272530], **kwargs_272535)
        
        # Getting the type of 'self' (line 413)
        self_272537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'self')
        # Setting the type of the member 'formwidget' of a type (line 413)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 12), self_272537, 'formwidget', FormComboWidget_call_result_272536)
        # SSA branch for the else part of an if statement (line 412)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Attribute (line 416):
        
        # Assigning a Call to a Attribute (line 416):
        
        # Call to FormWidget(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'data' (line 416)
        data_272539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 41), 'data', False)
        # Processing the call keyword arguments (line 416)
        # Getting the type of 'comment' (line 416)
        comment_272540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 55), 'comment', False)
        keyword_272541 = comment_272540
        # Getting the type of 'self' (line 417)
        self_272542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 48), 'self', False)
        keyword_272543 = self_272542
        kwargs_272544 = {'comment': keyword_272541, 'parent': keyword_272543}
        # Getting the type of 'FormWidget' (line 416)
        FormWidget_272538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 30), 'FormWidget', False)
        # Calling FormWidget(args, kwargs) (line 416)
        FormWidget_call_result_272545 = invoke(stypy.reporting.localization.Localization(__file__, 416, 30), FormWidget_272538, *[data_272539], **kwargs_272544)
        
        # Getting the type of 'self' (line 416)
        self_272546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 12), 'self')
        # Setting the type of the member 'formwidget' of a type (line 416)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 12), self_272546, 'formwidget', FormWidget_call_result_272545)
        # SSA join for if statement (line 412)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 409)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 418):
        
        # Assigning a Call to a Name (line 418):
        
        # Call to QVBoxLayout(...): (line 418)
        # Processing the call keyword arguments (line 418)
        kwargs_272549 = {}
        # Getting the type of 'QtWidgets' (line 418)
        QtWidgets_272547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 17), 'QtWidgets', False)
        # Obtaining the member 'QVBoxLayout' of a type (line 418)
        QVBoxLayout_272548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 17), QtWidgets_272547, 'QVBoxLayout')
        # Calling QVBoxLayout(args, kwargs) (line 418)
        QVBoxLayout_call_result_272550 = invoke(stypy.reporting.localization.Localization(__file__, 418, 17), QVBoxLayout_272548, *[], **kwargs_272549)
        
        # Assigning a type to the variable 'layout' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'layout', QVBoxLayout_call_result_272550)
        
        # Call to addWidget(...): (line 419)
        # Processing the call arguments (line 419)
        # Getting the type of 'self' (line 419)
        self_272553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 25), 'self', False)
        # Obtaining the member 'formwidget' of a type (line 419)
        formwidget_272554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 25), self_272553, 'formwidget')
        # Processing the call keyword arguments (line 419)
        kwargs_272555 = {}
        # Getting the type of 'layout' (line 419)
        layout_272551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'layout', False)
        # Obtaining the member 'addWidget' of a type (line 419)
        addWidget_272552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 8), layout_272551, 'addWidget')
        # Calling addWidget(args, kwargs) (line 419)
        addWidget_call_result_272556 = invoke(stypy.reporting.localization.Localization(__file__, 419, 8), addWidget_272552, *[formwidget_272554], **kwargs_272555)
        
        
        # Assigning a List to a Attribute (line 421):
        
        # Assigning a List to a Attribute (line 421):
        
        # Obtaining an instance of the builtin type 'list' (line 421)
        list_272557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 421)
        
        # Getting the type of 'self' (line 421)
        self_272558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'self')
        # Setting the type of the member 'float_fields' of a type (line 421)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 8), self_272558, 'float_fields', list_272557)
        
        # Call to setup(...): (line 422)
        # Processing the call keyword arguments (line 422)
        kwargs_272562 = {}
        # Getting the type of 'self' (line 422)
        self_272559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'self', False)
        # Obtaining the member 'formwidget' of a type (line 422)
        formwidget_272560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 8), self_272559, 'formwidget')
        # Obtaining the member 'setup' of a type (line 422)
        setup_272561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 8), formwidget_272560, 'setup')
        # Calling setup(args, kwargs) (line 422)
        setup_call_result_272563 = invoke(stypy.reporting.localization.Localization(__file__, 422, 8), setup_272561, *[], **kwargs_272562)
        
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Call to a Name (line 425):
        
        # Call to QDialogButtonBox(...): (line 425)
        # Processing the call arguments (line 425)
        # Getting the type of 'QtWidgets' (line 426)
        QtWidgets_272566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'QtWidgets', False)
        # Obtaining the member 'QDialogButtonBox' of a type (line 426)
        QDialogButtonBox_272567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 12), QtWidgets_272566, 'QDialogButtonBox')
        # Obtaining the member 'Ok' of a type (line 426)
        Ok_272568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 12), QDialogButtonBox_272567, 'Ok')
        # Getting the type of 'QtWidgets' (line 426)
        QtWidgets_272569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 44), 'QtWidgets', False)
        # Obtaining the member 'QDialogButtonBox' of a type (line 426)
        QDialogButtonBox_272570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 44), QtWidgets_272569, 'QDialogButtonBox')
        # Obtaining the member 'Cancel' of a type (line 426)
        Cancel_272571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 44), QDialogButtonBox_272570, 'Cancel')
        # Applying the binary operator '|' (line 426)
        result_or__272572 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 12), '|', Ok_272568, Cancel_272571)
        
        # Processing the call keyword arguments (line 425)
        kwargs_272573 = {}
        # Getting the type of 'QtWidgets' (line 425)
        QtWidgets_272564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 27), 'QtWidgets', False)
        # Obtaining the member 'QDialogButtonBox' of a type (line 425)
        QDialogButtonBox_272565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 27), QtWidgets_272564, 'QDialogButtonBox')
        # Calling QDialogButtonBox(args, kwargs) (line 425)
        QDialogButtonBox_call_result_272574 = invoke(stypy.reporting.localization.Localization(__file__, 425, 27), QDialogButtonBox_272565, *[result_or__272572], **kwargs_272573)
        
        # Assigning a type to the variable 'bbox' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 20), 'bbox', QDialogButtonBox_call_result_272574)
        
        # Assigning a Name to a Attribute (line 425):
        # Getting the type of 'bbox' (line 425)
        bbox_272575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 20), 'bbox')
        # Getting the type of 'self' (line 425)
        self_272576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'self')
        # Setting the type of the member 'bbox' of a type (line 425)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 8), self_272576, 'bbox', bbox_272575)
        
        # Call to connect(...): (line 427)
        # Processing the call arguments (line 427)
        # Getting the type of 'self' (line 427)
        self_272581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 47), 'self', False)
        # Obtaining the member 'update_buttons' of a type (line 427)
        update_buttons_272582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 47), self_272581, 'update_buttons')
        # Processing the call keyword arguments (line 427)
        kwargs_272583 = {}
        # Getting the type of 'self' (line 427)
        self_272577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'self', False)
        # Obtaining the member 'formwidget' of a type (line 427)
        formwidget_272578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 8), self_272577, 'formwidget')
        # Obtaining the member 'update_buttons' of a type (line 427)
        update_buttons_272579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 8), formwidget_272578, 'update_buttons')
        # Obtaining the member 'connect' of a type (line 427)
        connect_272580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 8), update_buttons_272579, 'connect')
        # Calling connect(args, kwargs) (line 427)
        connect_call_result_272584 = invoke(stypy.reporting.localization.Localization(__file__, 427, 8), connect_272580, *[update_buttons_272582], **kwargs_272583)
        
        
        
        # Getting the type of 'self' (line 428)
        self_272585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 11), 'self')
        # Obtaining the member 'apply_callback' of a type (line 428)
        apply_callback_272586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 11), self_272585, 'apply_callback')
        # Getting the type of 'None' (line 428)
        None_272587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 38), 'None')
        # Applying the binary operator 'isnot' (line 428)
        result_is_not_272588 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 11), 'isnot', apply_callback_272586, None_272587)
        
        # Testing the type of an if condition (line 428)
        if_condition_272589 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 428, 8), result_is_not_272588)
        # Assigning a type to the variable 'if_condition_272589' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'if_condition_272589', if_condition_272589)
        # SSA begins for if statement (line 428)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 429):
        
        # Assigning a Call to a Name (line 429):
        
        # Call to addButton(...): (line 429)
        # Processing the call arguments (line 429)
        # Getting the type of 'QtWidgets' (line 429)
        QtWidgets_272592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 39), 'QtWidgets', False)
        # Obtaining the member 'QDialogButtonBox' of a type (line 429)
        QDialogButtonBox_272593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 39), QtWidgets_272592, 'QDialogButtonBox')
        # Obtaining the member 'Apply' of a type (line 429)
        Apply_272594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 39), QDialogButtonBox_272593, 'Apply')
        # Processing the call keyword arguments (line 429)
        kwargs_272595 = {}
        # Getting the type of 'bbox' (line 429)
        bbox_272590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 24), 'bbox', False)
        # Obtaining the member 'addButton' of a type (line 429)
        addButton_272591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 24), bbox_272590, 'addButton')
        # Calling addButton(args, kwargs) (line 429)
        addButton_call_result_272596 = invoke(stypy.reporting.localization.Localization(__file__, 429, 24), addButton_272591, *[Apply_272594], **kwargs_272595)
        
        # Assigning a type to the variable 'apply_btn' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'apply_btn', addButton_call_result_272596)
        
        # Call to connect(...): (line 430)
        # Processing the call arguments (line 430)
        # Getting the type of 'self' (line 430)
        self_272600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 38), 'self', False)
        # Obtaining the member 'apply' of a type (line 430)
        apply_272601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 38), self_272600, 'apply')
        # Processing the call keyword arguments (line 430)
        kwargs_272602 = {}
        # Getting the type of 'apply_btn' (line 430)
        apply_btn_272597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'apply_btn', False)
        # Obtaining the member 'clicked' of a type (line 430)
        clicked_272598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 12), apply_btn_272597, 'clicked')
        # Obtaining the member 'connect' of a type (line 430)
        connect_272599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 12), clicked_272598, 'connect')
        # Calling connect(args, kwargs) (line 430)
        connect_call_result_272603 = invoke(stypy.reporting.localization.Localization(__file__, 430, 12), connect_272599, *[apply_272601], **kwargs_272602)
        
        # SSA join for if statement (line 428)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to connect(...): (line 432)
        # Processing the call arguments (line 432)
        # Getting the type of 'self' (line 432)
        self_272607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 30), 'self', False)
        # Obtaining the member 'accept' of a type (line 432)
        accept_272608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 30), self_272607, 'accept')
        # Processing the call keyword arguments (line 432)
        kwargs_272609 = {}
        # Getting the type of 'bbox' (line 432)
        bbox_272604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'bbox', False)
        # Obtaining the member 'accepted' of a type (line 432)
        accepted_272605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 8), bbox_272604, 'accepted')
        # Obtaining the member 'connect' of a type (line 432)
        connect_272606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 8), accepted_272605, 'connect')
        # Calling connect(args, kwargs) (line 432)
        connect_call_result_272610 = invoke(stypy.reporting.localization.Localization(__file__, 432, 8), connect_272606, *[accept_272608], **kwargs_272609)
        
        
        # Call to connect(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 'self' (line 433)
        self_272614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 30), 'self', False)
        # Obtaining the member 'reject' of a type (line 433)
        reject_272615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 30), self_272614, 'reject')
        # Processing the call keyword arguments (line 433)
        kwargs_272616 = {}
        # Getting the type of 'bbox' (line 433)
        bbox_272611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'bbox', False)
        # Obtaining the member 'rejected' of a type (line 433)
        rejected_272612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 8), bbox_272611, 'rejected')
        # Obtaining the member 'connect' of a type (line 433)
        connect_272613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 8), rejected_272612, 'connect')
        # Calling connect(args, kwargs) (line 433)
        connect_call_result_272617 = invoke(stypy.reporting.localization.Localization(__file__, 433, 8), connect_272613, *[reject_272615], **kwargs_272616)
        
        
        # Call to addWidget(...): (line 434)
        # Processing the call arguments (line 434)
        # Getting the type of 'bbox' (line 434)
        bbox_272620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 25), 'bbox', False)
        # Processing the call keyword arguments (line 434)
        kwargs_272621 = {}
        # Getting the type of 'layout' (line 434)
        layout_272618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'layout', False)
        # Obtaining the member 'addWidget' of a type (line 434)
        addWidget_272619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 8), layout_272618, 'addWidget')
        # Calling addWidget(args, kwargs) (line 434)
        addWidget_call_result_272622 = invoke(stypy.reporting.localization.Localization(__file__, 434, 8), addWidget_272619, *[bbox_272620], **kwargs_272621)
        
        
        # Call to setLayout(...): (line 436)
        # Processing the call arguments (line 436)
        # Getting the type of 'layout' (line 436)
        layout_272625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 23), 'layout', False)
        # Processing the call keyword arguments (line 436)
        kwargs_272626 = {}
        # Getting the type of 'self' (line 436)
        self_272623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'self', False)
        # Obtaining the member 'setLayout' of a type (line 436)
        setLayout_272624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 8), self_272623, 'setLayout')
        # Calling setLayout(args, kwargs) (line 436)
        setLayout_call_result_272627 = invoke(stypy.reporting.localization.Localization(__file__, 436, 8), setLayout_272624, *[layout_272625], **kwargs_272626)
        
        
        # Call to setWindowTitle(...): (line 438)
        # Processing the call arguments (line 438)
        # Getting the type of 'title' (line 438)
        title_272630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 28), 'title', False)
        # Processing the call keyword arguments (line 438)
        kwargs_272631 = {}
        # Getting the type of 'self' (line 438)
        self_272628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'self', False)
        # Obtaining the member 'setWindowTitle' of a type (line 438)
        setWindowTitle_272629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 8), self_272628, 'setWindowTitle')
        # Calling setWindowTitle(args, kwargs) (line 438)
        setWindowTitle_call_result_272632 = invoke(stypy.reporting.localization.Localization(__file__, 438, 8), setWindowTitle_272629, *[title_272630], **kwargs_272631)
        
        
        
        
        # Call to isinstance(...): (line 439)
        # Processing the call arguments (line 439)
        # Getting the type of 'icon' (line 439)
        icon_272634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 26), 'icon', False)
        # Getting the type of 'QtGui' (line 439)
        QtGui_272635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 32), 'QtGui', False)
        # Obtaining the member 'QIcon' of a type (line 439)
        QIcon_272636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 32), QtGui_272635, 'QIcon')
        # Processing the call keyword arguments (line 439)
        kwargs_272637 = {}
        # Getting the type of 'isinstance' (line 439)
        isinstance_272633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 439)
        isinstance_call_result_272638 = invoke(stypy.reporting.localization.Localization(__file__, 439, 15), isinstance_272633, *[icon_272634, QIcon_272636], **kwargs_272637)
        
        # Applying the 'not' unary operator (line 439)
        result_not__272639 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 11), 'not', isinstance_call_result_272638)
        
        # Testing the type of an if condition (line 439)
        if_condition_272640 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 439, 8), result_not__272639)
        # Assigning a type to the variable 'if_condition_272640' (line 439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'if_condition_272640', if_condition_272640)
        # SSA begins for if statement (line 439)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 440):
        
        # Assigning a Call to a Name (line 440):
        
        # Call to standardIcon(...): (line 440)
        # Processing the call arguments (line 440)
        # Getting the type of 'QtWidgets' (line 440)
        QtWidgets_272649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 60), 'QtWidgets', False)
        # Obtaining the member 'QStyle' of a type (line 440)
        QStyle_272650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 60), QtWidgets_272649, 'QStyle')
        # Obtaining the member 'SP_MessageBoxQuestion' of a type (line 440)
        SP_MessageBoxQuestion_272651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 60), QStyle_272650, 'SP_MessageBoxQuestion')
        # Processing the call keyword arguments (line 440)
        kwargs_272652 = {}
        
        # Call to style(...): (line 440)
        # Processing the call keyword arguments (line 440)
        kwargs_272646 = {}
        
        # Call to QWidget(...): (line 440)
        # Processing the call keyword arguments (line 440)
        kwargs_272643 = {}
        # Getting the type of 'QtWidgets' (line 440)
        QtWidgets_272641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 19), 'QtWidgets', False)
        # Obtaining the member 'QWidget' of a type (line 440)
        QWidget_272642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 19), QtWidgets_272641, 'QWidget')
        # Calling QWidget(args, kwargs) (line 440)
        QWidget_call_result_272644 = invoke(stypy.reporting.localization.Localization(__file__, 440, 19), QWidget_272642, *[], **kwargs_272643)
        
        # Obtaining the member 'style' of a type (line 440)
        style_272645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 19), QWidget_call_result_272644, 'style')
        # Calling style(args, kwargs) (line 440)
        style_call_result_272647 = invoke(stypy.reporting.localization.Localization(__file__, 440, 19), style_272645, *[], **kwargs_272646)
        
        # Obtaining the member 'standardIcon' of a type (line 440)
        standardIcon_272648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 19), style_call_result_272647, 'standardIcon')
        # Calling standardIcon(args, kwargs) (line 440)
        standardIcon_call_result_272653 = invoke(stypy.reporting.localization.Localization(__file__, 440, 19), standardIcon_272648, *[SP_MessageBoxQuestion_272651], **kwargs_272652)
        
        # Assigning a type to the variable 'icon' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'icon', standardIcon_call_result_272653)
        # SSA join for if statement (line 439)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to setWindowIcon(...): (line 441)
        # Processing the call arguments (line 441)
        # Getting the type of 'icon' (line 441)
        icon_272656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 27), 'icon', False)
        # Processing the call keyword arguments (line 441)
        kwargs_272657 = {}
        # Getting the type of 'self' (line 441)
        self_272654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'self', False)
        # Obtaining the member 'setWindowIcon' of a type (line 441)
        setWindowIcon_272655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 8), self_272654, 'setWindowIcon')
        # Calling setWindowIcon(args, kwargs) (line 441)
        setWindowIcon_call_result_272658 = invoke(stypy.reporting.localization.Localization(__file__, 441, 8), setWindowIcon_272655, *[icon_272656], **kwargs_272657)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def register_float_field(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'register_float_field'
        module_type_store = module_type_store.open_function_context('register_float_field', 443, 4, False)
        # Assigning a type to the variable 'self' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FormDialog.register_float_field.__dict__.__setitem__('stypy_localization', localization)
        FormDialog.register_float_field.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FormDialog.register_float_field.__dict__.__setitem__('stypy_type_store', module_type_store)
        FormDialog.register_float_field.__dict__.__setitem__('stypy_function_name', 'FormDialog.register_float_field')
        FormDialog.register_float_field.__dict__.__setitem__('stypy_param_names_list', ['field'])
        FormDialog.register_float_field.__dict__.__setitem__('stypy_varargs_param_name', None)
        FormDialog.register_float_field.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FormDialog.register_float_field.__dict__.__setitem__('stypy_call_defaults', defaults)
        FormDialog.register_float_field.__dict__.__setitem__('stypy_call_varargs', varargs)
        FormDialog.register_float_field.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FormDialog.register_float_field.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FormDialog.register_float_field', ['field'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'register_float_field', localization, ['field'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'register_float_field(...)' code ##################

        
        # Call to append(...): (line 444)
        # Processing the call arguments (line 444)
        # Getting the type of 'field' (line 444)
        field_272662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 33), 'field', False)
        # Processing the call keyword arguments (line 444)
        kwargs_272663 = {}
        # Getting the type of 'self' (line 444)
        self_272659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'self', False)
        # Obtaining the member 'float_fields' of a type (line 444)
        float_fields_272660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 8), self_272659, 'float_fields')
        # Obtaining the member 'append' of a type (line 444)
        append_272661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 8), float_fields_272660, 'append')
        # Calling append(args, kwargs) (line 444)
        append_call_result_272664 = invoke(stypy.reporting.localization.Localization(__file__, 444, 8), append_272661, *[field_272662], **kwargs_272663)
        
        
        # ################# End of 'register_float_field(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'register_float_field' in the type store
        # Getting the type of 'stypy_return_type' (line 443)
        stypy_return_type_272665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_272665)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'register_float_field'
        return stypy_return_type_272665


    @norecursion
    def update_buttons(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update_buttons'
        module_type_store = module_type_store.open_function_context('update_buttons', 446, 4, False)
        # Assigning a type to the variable 'self' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FormDialog.update_buttons.__dict__.__setitem__('stypy_localization', localization)
        FormDialog.update_buttons.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FormDialog.update_buttons.__dict__.__setitem__('stypy_type_store', module_type_store)
        FormDialog.update_buttons.__dict__.__setitem__('stypy_function_name', 'FormDialog.update_buttons')
        FormDialog.update_buttons.__dict__.__setitem__('stypy_param_names_list', [])
        FormDialog.update_buttons.__dict__.__setitem__('stypy_varargs_param_name', None)
        FormDialog.update_buttons.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FormDialog.update_buttons.__dict__.__setitem__('stypy_call_defaults', defaults)
        FormDialog.update_buttons.__dict__.__setitem__('stypy_call_varargs', varargs)
        FormDialog.update_buttons.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FormDialog.update_buttons.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FormDialog.update_buttons', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update_buttons', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update_buttons(...)' code ##################

        
        # Assigning a Name to a Name (line 447):
        
        # Assigning a Name to a Name (line 447):
        # Getting the type of 'True' (line 447)
        True_272666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 16), 'True')
        # Assigning a type to the variable 'valid' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'valid', True_272666)
        
        # Getting the type of 'self' (line 448)
        self_272667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 21), 'self')
        # Obtaining the member 'float_fields' of a type (line 448)
        float_fields_272668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 21), self_272667, 'float_fields')
        # Testing the type of a for loop iterable (line 448)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 448, 8), float_fields_272668)
        # Getting the type of the for loop variable (line 448)
        for_loop_var_272669 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 448, 8), float_fields_272668)
        # Assigning a type to the variable 'field' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'field', for_loop_var_272669)
        # SSA begins for a for statement (line 448)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to is_edit_valid(...): (line 449)
        # Processing the call arguments (line 449)
        # Getting the type of 'field' (line 449)
        field_272671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 33), 'field', False)
        # Processing the call keyword arguments (line 449)
        kwargs_272672 = {}
        # Getting the type of 'is_edit_valid' (line 449)
        is_edit_valid_272670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 19), 'is_edit_valid', False)
        # Calling is_edit_valid(args, kwargs) (line 449)
        is_edit_valid_call_result_272673 = invoke(stypy.reporting.localization.Localization(__file__, 449, 19), is_edit_valid_272670, *[field_272671], **kwargs_272672)
        
        # Applying the 'not' unary operator (line 449)
        result_not__272674 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 15), 'not', is_edit_valid_call_result_272673)
        
        # Testing the type of an if condition (line 449)
        if_condition_272675 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 449, 12), result_not__272674)
        # Assigning a type to the variable 'if_condition_272675' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'if_condition_272675', if_condition_272675)
        # SSA begins for if statement (line 449)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 450):
        
        # Assigning a Name to a Name (line 450):
        # Getting the type of 'False' (line 450)
        False_272676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 24), 'False')
        # Assigning a type to the variable 'valid' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 16), 'valid', False_272676)
        # SSA join for if statement (line 449)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 451)
        tuple_272677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 451)
        # Adding element type (line 451)
        # Getting the type of 'QtWidgets' (line 451)
        QtWidgets_272678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 25), 'QtWidgets')
        # Obtaining the member 'QDialogButtonBox' of a type (line 451)
        QDialogButtonBox_272679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 25), QtWidgets_272678, 'QDialogButtonBox')
        # Obtaining the member 'Ok' of a type (line 451)
        Ok_272680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 25), QDialogButtonBox_272679, 'Ok')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 25), tuple_272677, Ok_272680)
        # Adding element type (line 451)
        # Getting the type of 'QtWidgets' (line 452)
        QtWidgets_272681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 25), 'QtWidgets')
        # Obtaining the member 'QDialogButtonBox' of a type (line 452)
        QDialogButtonBox_272682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 25), QtWidgets_272681, 'QDialogButtonBox')
        # Obtaining the member 'Apply' of a type (line 452)
        Apply_272683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 25), QDialogButtonBox_272682, 'Apply')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 25), tuple_272677, Apply_272683)
        
        # Testing the type of a for loop iterable (line 451)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 451, 8), tuple_272677)
        # Getting the type of the for loop variable (line 451)
        for_loop_var_272684 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 451, 8), tuple_272677)
        # Assigning a type to the variable 'btn_type' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'btn_type', for_loop_var_272684)
        # SSA begins for a for statement (line 451)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 453):
        
        # Assigning a Call to a Name (line 453):
        
        # Call to button(...): (line 453)
        # Processing the call arguments (line 453)
        # Getting the type of 'btn_type' (line 453)
        btn_type_272688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 35), 'btn_type', False)
        # Processing the call keyword arguments (line 453)
        kwargs_272689 = {}
        # Getting the type of 'self' (line 453)
        self_272685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 18), 'self', False)
        # Obtaining the member 'bbox' of a type (line 453)
        bbox_272686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 18), self_272685, 'bbox')
        # Obtaining the member 'button' of a type (line 453)
        button_272687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 18), bbox_272686, 'button')
        # Calling button(args, kwargs) (line 453)
        button_call_result_272690 = invoke(stypy.reporting.localization.Localization(__file__, 453, 18), button_272687, *[btn_type_272688], **kwargs_272689)
        
        # Assigning a type to the variable 'btn' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 12), 'btn', button_call_result_272690)
        
        # Type idiom detected: calculating its left and rigth part (line 454)
        # Getting the type of 'btn' (line 454)
        btn_272691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'btn')
        # Getting the type of 'None' (line 454)
        None_272692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 26), 'None')
        
        (may_be_272693, more_types_in_union_272694) = may_not_be_none(btn_272691, None_272692)

        if may_be_272693:

            if more_types_in_union_272694:
                # Runtime conditional SSA (line 454)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to setEnabled(...): (line 455)
            # Processing the call arguments (line 455)
            # Getting the type of 'valid' (line 455)
            valid_272697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 31), 'valid', False)
            # Processing the call keyword arguments (line 455)
            kwargs_272698 = {}
            # Getting the type of 'btn' (line 455)
            btn_272695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 16), 'btn', False)
            # Obtaining the member 'setEnabled' of a type (line 455)
            setEnabled_272696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 16), btn_272695, 'setEnabled')
            # Calling setEnabled(args, kwargs) (line 455)
            setEnabled_call_result_272699 = invoke(stypy.reporting.localization.Localization(__file__, 455, 16), setEnabled_272696, *[valid_272697], **kwargs_272698)
            

            if more_types_in_union_272694:
                # SSA join for if statement (line 454)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'update_buttons(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_buttons' in the type store
        # Getting the type of 'stypy_return_type' (line 446)
        stypy_return_type_272700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_272700)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_buttons'
        return stypy_return_type_272700


    @norecursion
    def accept(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'accept'
        module_type_store = module_type_store.open_function_context('accept', 457, 4, False)
        # Assigning a type to the variable 'self' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FormDialog.accept.__dict__.__setitem__('stypy_localization', localization)
        FormDialog.accept.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FormDialog.accept.__dict__.__setitem__('stypy_type_store', module_type_store)
        FormDialog.accept.__dict__.__setitem__('stypy_function_name', 'FormDialog.accept')
        FormDialog.accept.__dict__.__setitem__('stypy_param_names_list', [])
        FormDialog.accept.__dict__.__setitem__('stypy_varargs_param_name', None)
        FormDialog.accept.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FormDialog.accept.__dict__.__setitem__('stypy_call_defaults', defaults)
        FormDialog.accept.__dict__.__setitem__('stypy_call_varargs', varargs)
        FormDialog.accept.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FormDialog.accept.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FormDialog.accept', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'accept', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'accept(...)' code ##################

        
        # Assigning a Call to a Attribute (line 458):
        
        # Assigning a Call to a Attribute (line 458):
        
        # Call to get(...): (line 458)
        # Processing the call keyword arguments (line 458)
        kwargs_272704 = {}
        # Getting the type of 'self' (line 458)
        self_272701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 20), 'self', False)
        # Obtaining the member 'formwidget' of a type (line 458)
        formwidget_272702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 20), self_272701, 'formwidget')
        # Obtaining the member 'get' of a type (line 458)
        get_272703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 20), formwidget_272702, 'get')
        # Calling get(args, kwargs) (line 458)
        get_call_result_272705 = invoke(stypy.reporting.localization.Localization(__file__, 458, 20), get_272703, *[], **kwargs_272704)
        
        # Getting the type of 'self' (line 458)
        self_272706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'self')
        # Setting the type of the member 'data' of a type (line 458)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 8), self_272706, 'data', get_call_result_272705)
        
        # Call to accept(...): (line 459)
        # Processing the call arguments (line 459)
        # Getting the type of 'self' (line 459)
        self_272710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 33), 'self', False)
        # Processing the call keyword arguments (line 459)
        kwargs_272711 = {}
        # Getting the type of 'QtWidgets' (line 459)
        QtWidgets_272707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'QtWidgets', False)
        # Obtaining the member 'QDialog' of a type (line 459)
        QDialog_272708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 8), QtWidgets_272707, 'QDialog')
        # Obtaining the member 'accept' of a type (line 459)
        accept_272709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 8), QDialog_272708, 'accept')
        # Calling accept(args, kwargs) (line 459)
        accept_call_result_272712 = invoke(stypy.reporting.localization.Localization(__file__, 459, 8), accept_272709, *[self_272710], **kwargs_272711)
        
        
        # ################# End of 'accept(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'accept' in the type store
        # Getting the type of 'stypy_return_type' (line 457)
        stypy_return_type_272713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_272713)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'accept'
        return stypy_return_type_272713


    @norecursion
    def reject(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'reject'
        module_type_store = module_type_store.open_function_context('reject', 461, 4, False)
        # Assigning a type to the variable 'self' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FormDialog.reject.__dict__.__setitem__('stypy_localization', localization)
        FormDialog.reject.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FormDialog.reject.__dict__.__setitem__('stypy_type_store', module_type_store)
        FormDialog.reject.__dict__.__setitem__('stypy_function_name', 'FormDialog.reject')
        FormDialog.reject.__dict__.__setitem__('stypy_param_names_list', [])
        FormDialog.reject.__dict__.__setitem__('stypy_varargs_param_name', None)
        FormDialog.reject.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FormDialog.reject.__dict__.__setitem__('stypy_call_defaults', defaults)
        FormDialog.reject.__dict__.__setitem__('stypy_call_varargs', varargs)
        FormDialog.reject.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FormDialog.reject.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FormDialog.reject', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'reject', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'reject(...)' code ##################

        
        # Assigning a Name to a Attribute (line 462):
        
        # Assigning a Name to a Attribute (line 462):
        # Getting the type of 'None' (line 462)
        None_272714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 20), 'None')
        # Getting the type of 'self' (line 462)
        self_272715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'self')
        # Setting the type of the member 'data' of a type (line 462)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 8), self_272715, 'data', None_272714)
        
        # Call to reject(...): (line 463)
        # Processing the call arguments (line 463)
        # Getting the type of 'self' (line 463)
        self_272719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 33), 'self', False)
        # Processing the call keyword arguments (line 463)
        kwargs_272720 = {}
        # Getting the type of 'QtWidgets' (line 463)
        QtWidgets_272716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'QtWidgets', False)
        # Obtaining the member 'QDialog' of a type (line 463)
        QDialog_272717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 8), QtWidgets_272716, 'QDialog')
        # Obtaining the member 'reject' of a type (line 463)
        reject_272718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 8), QDialog_272717, 'reject')
        # Calling reject(args, kwargs) (line 463)
        reject_call_result_272721 = invoke(stypy.reporting.localization.Localization(__file__, 463, 8), reject_272718, *[self_272719], **kwargs_272720)
        
        
        # ################# End of 'reject(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'reject' in the type store
        # Getting the type of 'stypy_return_type' (line 461)
        stypy_return_type_272722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_272722)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'reject'
        return stypy_return_type_272722


    @norecursion
    def apply(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'apply'
        module_type_store = module_type_store.open_function_context('apply', 465, 4, False)
        # Assigning a type to the variable 'self' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FormDialog.apply.__dict__.__setitem__('stypy_localization', localization)
        FormDialog.apply.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FormDialog.apply.__dict__.__setitem__('stypy_type_store', module_type_store)
        FormDialog.apply.__dict__.__setitem__('stypy_function_name', 'FormDialog.apply')
        FormDialog.apply.__dict__.__setitem__('stypy_param_names_list', [])
        FormDialog.apply.__dict__.__setitem__('stypy_varargs_param_name', None)
        FormDialog.apply.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FormDialog.apply.__dict__.__setitem__('stypy_call_defaults', defaults)
        FormDialog.apply.__dict__.__setitem__('stypy_call_varargs', varargs)
        FormDialog.apply.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FormDialog.apply.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FormDialog.apply', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'apply', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'apply(...)' code ##################

        
        # Call to apply_callback(...): (line 466)
        # Processing the call arguments (line 466)
        
        # Call to get(...): (line 466)
        # Processing the call keyword arguments (line 466)
        kwargs_272728 = {}
        # Getting the type of 'self' (line 466)
        self_272725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 28), 'self', False)
        # Obtaining the member 'formwidget' of a type (line 466)
        formwidget_272726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 28), self_272725, 'formwidget')
        # Obtaining the member 'get' of a type (line 466)
        get_272727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 28), formwidget_272726, 'get')
        # Calling get(args, kwargs) (line 466)
        get_call_result_272729 = invoke(stypy.reporting.localization.Localization(__file__, 466, 28), get_272727, *[], **kwargs_272728)
        
        # Processing the call keyword arguments (line 466)
        kwargs_272730 = {}
        # Getting the type of 'self' (line 466)
        self_272723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'self', False)
        # Obtaining the member 'apply_callback' of a type (line 466)
        apply_callback_272724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 8), self_272723, 'apply_callback')
        # Calling apply_callback(args, kwargs) (line 466)
        apply_callback_call_result_272731 = invoke(stypy.reporting.localization.Localization(__file__, 466, 8), apply_callback_272724, *[get_call_result_272729], **kwargs_272730)
        
        
        # ################# End of 'apply(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'apply' in the type store
        # Getting the type of 'stypy_return_type' (line 465)
        stypy_return_type_272732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_272732)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'apply'
        return stypy_return_type_272732


    @norecursion
    def get(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get'
        module_type_store = module_type_store.open_function_context('get', 468, 4, False)
        # Assigning a type to the variable 'self' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FormDialog.get.__dict__.__setitem__('stypy_localization', localization)
        FormDialog.get.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FormDialog.get.__dict__.__setitem__('stypy_type_store', module_type_store)
        FormDialog.get.__dict__.__setitem__('stypy_function_name', 'FormDialog.get')
        FormDialog.get.__dict__.__setitem__('stypy_param_names_list', [])
        FormDialog.get.__dict__.__setitem__('stypy_varargs_param_name', None)
        FormDialog.get.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FormDialog.get.__dict__.__setitem__('stypy_call_defaults', defaults)
        FormDialog.get.__dict__.__setitem__('stypy_call_varargs', varargs)
        FormDialog.get.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FormDialog.get.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FormDialog.get', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get(...)' code ##################

        unicode_272733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 8), 'unicode', u'Return form result')
        # Getting the type of 'self' (line 470)
        self_272734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 15), 'self')
        # Obtaining the member 'data' of a type (line 470)
        data_272735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 15), self_272734, 'data')
        # Assigning a type to the variable 'stypy_return_type' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'stypy_return_type', data_272735)
        
        # ################# End of 'get(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get' in the type store
        # Getting the type of 'stypy_return_type' (line 468)
        stypy_return_type_272736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_272736)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get'
        return stypy_return_type_272736


# Assigning a type to the variable 'FormDialog' (line 400)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 0), 'FormDialog', FormDialog)

@norecursion
def fedit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    unicode_272737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 22), 'unicode', u'')
    unicode_272738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 34), 'unicode', u'')
    # Getting the type of 'None' (line 473)
    None_272739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 43), 'None')
    # Getting the type of 'None' (line 473)
    None_272740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 56), 'None')
    # Getting the type of 'None' (line 473)
    None_272741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 68), 'None')
    defaults = [unicode_272737, unicode_272738, None_272739, None_272740, None_272741]
    # Create a new context for function 'fedit'
    module_type_store = module_type_store.open_function_context('fedit', 473, 0, False)
    
    # Passed parameters checking function
    fedit.stypy_localization = localization
    fedit.stypy_type_of_self = None
    fedit.stypy_type_store = module_type_store
    fedit.stypy_function_name = 'fedit'
    fedit.stypy_param_names_list = ['data', 'title', 'comment', 'icon', 'parent', 'apply']
    fedit.stypy_varargs_param_name = None
    fedit.stypy_kwargs_param_name = None
    fedit.stypy_call_defaults = defaults
    fedit.stypy_call_varargs = varargs
    fedit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fedit', ['data', 'title', 'comment', 'icon', 'parent', 'apply'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fedit', localization, ['data', 'title', 'comment', 'icon', 'parent', 'apply'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fedit(...)' code ##################

    unicode_272742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, (-1)), 'unicode', u'\n    Create form dialog and return result\n    (if Cancel button is pressed, return None)\n\n    data: datalist, datagroup\n    title: string\n    comment: string\n    icon: QIcon instance\n    parent: parent QWidget\n    apply: apply callback (function)\n\n    datalist: list/tuple of (field_name, field_value)\n    datagroup: list/tuple of (datalist *or* datagroup, title, comment)\n\n    -> one field for each member of a datalist\n    -> one tab for each member of a top-level datagroup\n    -> one page (of a multipage widget, each page can be selected with a combo\n       box) for each member of a datagroup inside a datagroup\n\n    Supported types for field_value:\n      - int, float, str, unicode, bool\n      - colors: in Qt-compatible text form, i.e. in hex format or name (red,...)\n                (automatically detected from a string)\n      - list/tuple:\n          * the first element will be the selected index (or value)\n          * the other elements can be couples (key, value) or only values\n    ')
    
    
    # Call to startingUp(...): (line 504)
    # Processing the call keyword arguments (line 504)
    kwargs_272746 = {}
    # Getting the type of 'QtWidgets' (line 504)
    QtWidgets_272743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 7), 'QtWidgets', False)
    # Obtaining the member 'QApplication' of a type (line 504)
    QApplication_272744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 7), QtWidgets_272743, 'QApplication')
    # Obtaining the member 'startingUp' of a type (line 504)
    startingUp_272745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 7), QApplication_272744, 'startingUp')
    # Calling startingUp(args, kwargs) (line 504)
    startingUp_call_result_272747 = invoke(stypy.reporting.localization.Localization(__file__, 504, 7), startingUp_272745, *[], **kwargs_272746)
    
    # Testing the type of an if condition (line 504)
    if_condition_272748 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 504, 4), startingUp_call_result_272747)
    # Assigning a type to the variable 'if_condition_272748' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'if_condition_272748', if_condition_272748)
    # SSA begins for if statement (line 504)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 505):
    
    # Assigning a Call to a Name (line 505):
    
    # Call to QApplication(...): (line 505)
    # Processing the call arguments (line 505)
    
    # Obtaining an instance of the builtin type 'list' (line 505)
    list_272751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 505)
    
    # Processing the call keyword arguments (line 505)
    kwargs_272752 = {}
    # Getting the type of 'QtWidgets' (line 505)
    QtWidgets_272749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 15), 'QtWidgets', False)
    # Obtaining the member 'QApplication' of a type (line 505)
    QApplication_272750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 15), QtWidgets_272749, 'QApplication')
    # Calling QApplication(args, kwargs) (line 505)
    QApplication_call_result_272753 = invoke(stypy.reporting.localization.Localization(__file__, 505, 15), QApplication_272750, *[list_272751], **kwargs_272752)
    
    # Assigning a type to the variable '_app' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), '_app', QApplication_call_result_272753)
    # SSA join for if statement (line 504)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 506):
    
    # Assigning a Call to a Name (line 506):
    
    # Call to FormDialog(...): (line 506)
    # Processing the call arguments (line 506)
    # Getting the type of 'data' (line 506)
    data_272755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 24), 'data', False)
    # Getting the type of 'title' (line 506)
    title_272756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 30), 'title', False)
    # Getting the type of 'comment' (line 506)
    comment_272757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 37), 'comment', False)
    # Getting the type of 'icon' (line 506)
    icon_272758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 46), 'icon', False)
    # Getting the type of 'parent' (line 506)
    parent_272759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 52), 'parent', False)
    # Getting the type of 'apply' (line 506)
    apply_272760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 60), 'apply', False)
    # Processing the call keyword arguments (line 506)
    kwargs_272761 = {}
    # Getting the type of 'FormDialog' (line 506)
    FormDialog_272754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 13), 'FormDialog', False)
    # Calling FormDialog(args, kwargs) (line 506)
    FormDialog_call_result_272762 = invoke(stypy.reporting.localization.Localization(__file__, 506, 13), FormDialog_272754, *[data_272755, title_272756, comment_272757, icon_272758, parent_272759, apply_272760], **kwargs_272761)
    
    # Assigning a type to the variable 'dialog' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'dialog', FormDialog_call_result_272762)
    
    
    # Call to exec_(...): (line 507)
    # Processing the call keyword arguments (line 507)
    kwargs_272765 = {}
    # Getting the type of 'dialog' (line 507)
    dialog_272763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 7), 'dialog', False)
    # Obtaining the member 'exec_' of a type (line 507)
    exec__272764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 7), dialog_272763, 'exec_')
    # Calling exec_(args, kwargs) (line 507)
    exec__call_result_272766 = invoke(stypy.reporting.localization.Localization(__file__, 507, 7), exec__272764, *[], **kwargs_272765)
    
    # Testing the type of an if condition (line 507)
    if_condition_272767 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 507, 4), exec__call_result_272766)
    # Assigning a type to the variable 'if_condition_272767' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'if_condition_272767', if_condition_272767)
    # SSA begins for if statement (line 507)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to get(...): (line 508)
    # Processing the call keyword arguments (line 508)
    kwargs_272770 = {}
    # Getting the type of 'dialog' (line 508)
    dialog_272768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 15), 'dialog', False)
    # Obtaining the member 'get' of a type (line 508)
    get_272769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 15), dialog_272768, 'get')
    # Calling get(args, kwargs) (line 508)
    get_call_result_272771 = invoke(stypy.reporting.localization.Localization(__file__, 508, 15), get_272769, *[], **kwargs_272770)
    
    # Assigning a type to the variable 'stypy_return_type' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'stypy_return_type', get_call_result_272771)
    # SSA join for if statement (line 507)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'fedit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fedit' in the type store
    # Getting the type of 'stypy_return_type' (line 473)
    stypy_return_type_272772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_272772)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fedit'
    return stypy_return_type_272772

# Assigning a type to the variable 'fedit' (line 473)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 0), 'fedit', fedit)

if (__name__ == u'__main__'):

    @norecursion
    def create_datalist_example(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_datalist_example'
        module_type_store = module_type_store.open_function_context('create_datalist_example', 513, 4, False)
        
        # Passed parameters checking function
        create_datalist_example.stypy_localization = localization
        create_datalist_example.stypy_type_of_self = None
        create_datalist_example.stypy_type_store = module_type_store
        create_datalist_example.stypy_function_name = 'create_datalist_example'
        create_datalist_example.stypy_param_names_list = []
        create_datalist_example.stypy_varargs_param_name = None
        create_datalist_example.stypy_kwargs_param_name = None
        create_datalist_example.stypy_call_defaults = defaults
        create_datalist_example.stypy_call_varargs = varargs
        create_datalist_example.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'create_datalist_example', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_datalist_example', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_datalist_example(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 514)
        list_272773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 514)
        # Adding element type (line 514)
        
        # Obtaining an instance of the builtin type 'tuple' (line 514)
        tuple_272774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 514)
        # Adding element type (line 514)
        unicode_272775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 17), 'unicode', u'str')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 17), tuple_272774, unicode_272775)
        # Adding element type (line 514)
        unicode_272776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 24), 'unicode', u'this is a string')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 17), tuple_272774, unicode_272776)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 15), list_272773, tuple_272774)
        # Adding element type (line 514)
        
        # Obtaining an instance of the builtin type 'tuple' (line 515)
        tuple_272777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 515)
        # Adding element type (line 515)
        unicode_272778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 17), 'unicode', u'list')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 17), tuple_272777, unicode_272778)
        # Adding element type (line 515)
        
        # Obtaining an instance of the builtin type 'list' (line 515)
        list_272779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 515)
        # Adding element type (line 515)
        int_272780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 25), list_272779, int_272780)
        # Adding element type (line 515)
        unicode_272781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 29), 'unicode', u'1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 25), list_272779, unicode_272781)
        # Adding element type (line 515)
        unicode_272782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 34), 'unicode', u'3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 25), list_272779, unicode_272782)
        # Adding element type (line 515)
        unicode_272783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 39), 'unicode', u'4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 25), list_272779, unicode_272783)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 17), tuple_272777, list_272779)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 15), list_272773, tuple_272777)
        # Adding element type (line 514)
        
        # Obtaining an instance of the builtin type 'tuple' (line 516)
        tuple_272784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 516)
        # Adding element type (line 516)
        unicode_272785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 17), 'unicode', u'list2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 17), tuple_272784, unicode_272785)
        # Adding element type (line 516)
        
        # Obtaining an instance of the builtin type 'list' (line 516)
        list_272786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 516)
        # Adding element type (line 516)
        unicode_272787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 27), 'unicode', u'--')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 26), list_272786, unicode_272787)
        # Adding element type (line 516)
        
        # Obtaining an instance of the builtin type 'tuple' (line 516)
        tuple_272788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 516)
        # Adding element type (line 516)
        unicode_272789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 34), 'unicode', u'none')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 34), tuple_272788, unicode_272789)
        # Adding element type (line 516)
        unicode_272790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 42), 'unicode', u'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 34), tuple_272788, unicode_272790)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 26), list_272786, tuple_272788)
        # Adding element type (line 516)
        
        # Obtaining an instance of the builtin type 'tuple' (line 516)
        tuple_272791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 516)
        # Adding element type (line 516)
        unicode_272792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 52), 'unicode', u'--')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 52), tuple_272791, unicode_272792)
        # Adding element type (line 516)
        unicode_272793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 58), 'unicode', u'Dashed')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 52), tuple_272791, unicode_272793)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 26), list_272786, tuple_272791)
        # Adding element type (line 516)
        
        # Obtaining an instance of the builtin type 'tuple' (line 517)
        tuple_272794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 517)
        # Adding element type (line 517)
        unicode_272795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 28), 'unicode', u'-.')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 28), tuple_272794, unicode_272795)
        # Adding element type (line 517)
        unicode_272796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 34), 'unicode', u'DashDot')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 28), tuple_272794, unicode_272796)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 26), list_272786, tuple_272794)
        # Adding element type (line 516)
        
        # Obtaining an instance of the builtin type 'tuple' (line 517)
        tuple_272797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 517)
        # Adding element type (line 517)
        unicode_272798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 47), 'unicode', u'-')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 47), tuple_272797, unicode_272798)
        # Adding element type (line 517)
        unicode_272799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 52), 'unicode', u'Solid')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 47), tuple_272797, unicode_272799)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 26), list_272786, tuple_272797)
        # Adding element type (line 516)
        
        # Obtaining an instance of the builtin type 'tuple' (line 518)
        tuple_272800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 518)
        # Adding element type (line 518)
        unicode_272801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 28), 'unicode', u'steps')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 28), tuple_272800, unicode_272801)
        # Adding element type (line 518)
        unicode_272802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 37), 'unicode', u'Steps')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 28), tuple_272800, unicode_272802)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 26), list_272786, tuple_272800)
        # Adding element type (line 516)
        
        # Obtaining an instance of the builtin type 'tuple' (line 518)
        tuple_272803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 518)
        # Adding element type (line 518)
        unicode_272804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 48), 'unicode', u':')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 48), tuple_272803, unicode_272804)
        # Adding element type (line 518)
        unicode_272805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 53), 'unicode', u'Dotted')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 48), tuple_272803, unicode_272805)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 26), list_272786, tuple_272803)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 17), tuple_272784, list_272786)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 15), list_272773, tuple_272784)
        # Adding element type (line 514)
        
        # Obtaining an instance of the builtin type 'tuple' (line 519)
        tuple_272806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 519)
        # Adding element type (line 519)
        unicode_272807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 17), 'unicode', u'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 17), tuple_272806, unicode_272807)
        # Adding element type (line 519)
        float_272808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 17), tuple_272806, float_272808)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 15), list_272773, tuple_272806)
        # Adding element type (line 514)
        
        # Obtaining an instance of the builtin type 'tuple' (line 520)
        tuple_272809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 520)
        # Adding element type (line 520)
        # Getting the type of 'None' (line 520)
        None_272810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 17), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 17), tuple_272809, None_272810)
        # Adding element type (line 520)
        unicode_272811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 23), 'unicode', u'Other:')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 17), tuple_272809, unicode_272811)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 15), list_272773, tuple_272809)
        # Adding element type (line 514)
        
        # Obtaining an instance of the builtin type 'tuple' (line 521)
        tuple_272812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 521)
        # Adding element type (line 521)
        unicode_272813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 17), 'unicode', u'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 17), tuple_272812, unicode_272813)
        # Adding element type (line 521)
        int_272814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 17), tuple_272812, int_272814)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 15), list_272773, tuple_272812)
        # Adding element type (line 514)
        
        # Obtaining an instance of the builtin type 'tuple' (line 522)
        tuple_272815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 522)
        # Adding element type (line 522)
        unicode_272816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 17), 'unicode', u'font')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 17), tuple_272815, unicode_272816)
        # Adding element type (line 522)
        
        # Obtaining an instance of the builtin type 'tuple' (line 522)
        tuple_272817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 522)
        # Adding element type (line 522)
        unicode_272818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 26), 'unicode', u'Arial')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 26), tuple_272817, unicode_272818)
        # Adding element type (line 522)
        int_272819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 26), tuple_272817, int_272819)
        # Adding element type (line 522)
        # Getting the type of 'False' (line 522)
        False_272820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 39), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 26), tuple_272817, False_272820)
        # Adding element type (line 522)
        # Getting the type of 'True' (line 522)
        True_272821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 46), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 26), tuple_272817, True_272821)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 17), tuple_272815, tuple_272817)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 15), list_272773, tuple_272815)
        # Adding element type (line 514)
        
        # Obtaining an instance of the builtin type 'tuple' (line 523)
        tuple_272822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 523)
        # Adding element type (line 523)
        unicode_272823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 17), 'unicode', u'color')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 17), tuple_272822, unicode_272823)
        # Adding element type (line 523)
        unicode_272824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 26), 'unicode', u'#123409')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 17), tuple_272822, unicode_272824)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 15), list_272773, tuple_272822)
        # Adding element type (line 514)
        
        # Obtaining an instance of the builtin type 'tuple' (line 524)
        tuple_272825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 524)
        # Adding element type (line 524)
        unicode_272826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 17), 'unicode', u'bool')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 17), tuple_272825, unicode_272826)
        # Adding element type (line 524)
        # Getting the type of 'True' (line 524)
        True_272827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 25), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 17), tuple_272825, True_272827)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 15), list_272773, tuple_272825)
        # Adding element type (line 514)
        
        # Obtaining an instance of the builtin type 'tuple' (line 525)
        tuple_272828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 525)
        # Adding element type (line 525)
        unicode_272829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 17), 'unicode', u'date')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 17), tuple_272828, unicode_272829)
        # Adding element type (line 525)
        
        # Call to date(...): (line 525)
        # Processing the call arguments (line 525)
        int_272832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 39), 'int')
        int_272833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 45), 'int')
        int_272834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 49), 'int')
        # Processing the call keyword arguments (line 525)
        kwargs_272835 = {}
        # Getting the type of 'datetime' (line 525)
        datetime_272830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 25), 'datetime', False)
        # Obtaining the member 'date' of a type (line 525)
        date_272831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 25), datetime_272830, 'date')
        # Calling date(args, kwargs) (line 525)
        date_call_result_272836 = invoke(stypy.reporting.localization.Localization(__file__, 525, 25), date_272831, *[int_272832, int_272833, int_272834], **kwargs_272835)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 17), tuple_272828, date_call_result_272836)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 15), list_272773, tuple_272828)
        # Adding element type (line 514)
        
        # Obtaining an instance of the builtin type 'tuple' (line 526)
        tuple_272837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 526)
        # Adding element type (line 526)
        unicode_272838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 17), 'unicode', u'datetime')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 526, 17), tuple_272837, unicode_272838)
        # Adding element type (line 526)
        
        # Call to datetime(...): (line 526)
        # Processing the call arguments (line 526)
        int_272841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 47), 'int')
        int_272842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 53), 'int')
        int_272843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 57), 'int')
        # Processing the call keyword arguments (line 526)
        kwargs_272844 = {}
        # Getting the type of 'datetime' (line 526)
        datetime_272839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 29), 'datetime', False)
        # Obtaining the member 'datetime' of a type (line 526)
        datetime_272840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 29), datetime_272839, 'datetime')
        # Calling datetime(args, kwargs) (line 526)
        datetime_call_result_272845 = invoke(stypy.reporting.localization.Localization(__file__, 526, 29), datetime_272840, *[int_272841, int_272842, int_272843], **kwargs_272844)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 526, 17), tuple_272837, datetime_call_result_272845)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 15), list_272773, tuple_272837)
        
        # Assigning a type to the variable 'stypy_return_type' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'stypy_return_type', list_272773)
        
        # ################# End of 'create_datalist_example(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_datalist_example' in the type store
        # Getting the type of 'stypy_return_type' (line 513)
        stypy_return_type_272846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_272846)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_datalist_example'
        return stypy_return_type_272846

    # Assigning a type to the variable 'create_datalist_example' (line 513)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), 'create_datalist_example', create_datalist_example)

    @norecursion
    def create_datagroup_example(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_datagroup_example'
        module_type_store = module_type_store.open_function_context('create_datagroup_example', 529, 4, False)
        
        # Passed parameters checking function
        create_datagroup_example.stypy_localization = localization
        create_datagroup_example.stypy_type_of_self = None
        create_datagroup_example.stypy_type_store = module_type_store
        create_datagroup_example.stypy_function_name = 'create_datagroup_example'
        create_datagroup_example.stypy_param_names_list = []
        create_datagroup_example.stypy_varargs_param_name = None
        create_datagroup_example.stypy_kwargs_param_name = None
        create_datagroup_example.stypy_call_defaults = defaults
        create_datagroup_example.stypy_call_varargs = varargs
        create_datagroup_example.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'create_datagroup_example', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_datagroup_example', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_datagroup_example(...)' code ##################

        
        # Assigning a Call to a Name (line 530):
        
        # Assigning a Call to a Name (line 530):
        
        # Call to create_datalist_example(...): (line 530)
        # Processing the call keyword arguments (line 530)
        kwargs_272848 = {}
        # Getting the type of 'create_datalist_example' (line 530)
        create_datalist_example_272847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 19), 'create_datalist_example', False)
        # Calling create_datalist_example(args, kwargs) (line 530)
        create_datalist_example_call_result_272849 = invoke(stypy.reporting.localization.Localization(__file__, 530, 19), create_datalist_example_272847, *[], **kwargs_272848)
        
        # Assigning a type to the variable 'datalist' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'datalist', create_datalist_example_call_result_272849)
        
        # Obtaining an instance of the builtin type 'tuple' (line 531)
        tuple_272850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 531)
        # Adding element type (line 531)
        
        # Obtaining an instance of the builtin type 'tuple' (line 531)
        tuple_272851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 531)
        # Adding element type (line 531)
        # Getting the type of 'datalist' (line 531)
        datalist_272852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 17), 'datalist')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 17), tuple_272851, datalist_272852)
        # Adding element type (line 531)
        unicode_272853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 27), 'unicode', u'Category 1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 17), tuple_272851, unicode_272853)
        # Adding element type (line 531)
        unicode_272854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 41), 'unicode', u'Category 1 comment')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 17), tuple_272851, unicode_272854)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 16), tuple_272850, tuple_272851)
        # Adding element type (line 531)
        
        # Obtaining an instance of the builtin type 'tuple' (line 532)
        tuple_272855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 532)
        # Adding element type (line 532)
        # Getting the type of 'datalist' (line 532)
        datalist_272856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 17), 'datalist')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 17), tuple_272855, datalist_272856)
        # Adding element type (line 532)
        unicode_272857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 27), 'unicode', u'Category 2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 17), tuple_272855, unicode_272857)
        # Adding element type (line 532)
        unicode_272858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 41), 'unicode', u'Category 2 comment')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 17), tuple_272855, unicode_272858)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 16), tuple_272850, tuple_272855)
        # Adding element type (line 531)
        
        # Obtaining an instance of the builtin type 'tuple' (line 533)
        tuple_272859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 533)
        # Adding element type (line 533)
        # Getting the type of 'datalist' (line 533)
        datalist_272860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 17), 'datalist')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 17), tuple_272859, datalist_272860)
        # Adding element type (line 533)
        unicode_272861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 27), 'unicode', u'Category 3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 17), tuple_272859, unicode_272861)
        # Adding element type (line 533)
        unicode_272862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 41), 'unicode', u'Category 3 comment')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 17), tuple_272859, unicode_272862)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 16), tuple_272850, tuple_272859)
        
        # Assigning a type to the variable 'stypy_return_type' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 8), 'stypy_return_type', tuple_272850)
        
        # ################# End of 'create_datagroup_example(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_datagroup_example' in the type store
        # Getting the type of 'stypy_return_type' (line 529)
        stypy_return_type_272863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_272863)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_datagroup_example'
        return stypy_return_type_272863

    # Assigning a type to the variable 'create_datagroup_example' (line 529)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 4), 'create_datagroup_example', create_datagroup_example)
    
    # Assigning a Call to a Name (line 536):
    
    # Assigning a Call to a Name (line 536):
    
    # Call to create_datalist_example(...): (line 536)
    # Processing the call keyword arguments (line 536)
    kwargs_272865 = {}
    # Getting the type of 'create_datalist_example' (line 536)
    create_datalist_example_272864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 15), 'create_datalist_example', False)
    # Calling create_datalist_example(args, kwargs) (line 536)
    create_datalist_example_call_result_272866 = invoke(stypy.reporting.localization.Localization(__file__, 536, 15), create_datalist_example_272864, *[], **kwargs_272865)
    
    # Assigning a type to the variable 'datalist' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'datalist', create_datalist_example_call_result_272866)

    @norecursion
    def apply_test(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'apply_test'
        module_type_store = module_type_store.open_function_context('apply_test', 538, 4, False)
        
        # Passed parameters checking function
        apply_test.stypy_localization = localization
        apply_test.stypy_type_of_self = None
        apply_test.stypy_type_store = module_type_store
        apply_test.stypy_function_name = 'apply_test'
        apply_test.stypy_param_names_list = ['data']
        apply_test.stypy_varargs_param_name = None
        apply_test.stypy_kwargs_param_name = None
        apply_test.stypy_call_defaults = defaults
        apply_test.stypy_call_varargs = varargs
        apply_test.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'apply_test', ['data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'apply_test', localization, ['data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'apply_test(...)' code ##################

        
        # Call to print(...): (line 539)
        # Processing the call arguments (line 539)
        unicode_272868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 14), 'unicode', u'data:')
        # Getting the type of 'data' (line 539)
        data_272869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 23), 'data', False)
        # Processing the call keyword arguments (line 539)
        kwargs_272870 = {}
        # Getting the type of 'print' (line 539)
        print_272867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 8), 'print', False)
        # Calling print(args, kwargs) (line 539)
        print_call_result_272871 = invoke(stypy.reporting.localization.Localization(__file__, 539, 8), print_272867, *[unicode_272868, data_272869], **kwargs_272870)
        
        
        # ################# End of 'apply_test(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'apply_test' in the type store
        # Getting the type of 'stypy_return_type' (line 538)
        stypy_return_type_272872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_272872)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'apply_test'
        return stypy_return_type_272872

    # Assigning a type to the variable 'apply_test' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'apply_test', apply_test)
    
    # Call to print(...): (line 540)
    # Processing the call arguments (line 540)
    unicode_272874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 10), 'unicode', u'result:')
    
    # Call to fedit(...): (line 540)
    # Processing the call arguments (line 540)
    # Getting the type of 'datalist' (line 540)
    datalist_272876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 27), 'datalist', False)
    # Processing the call keyword arguments (line 540)
    unicode_272877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 43), 'unicode', u'Example')
    keyword_272878 = unicode_272877
    unicode_272879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 35), 'unicode', u'This is just an <b>example</b>.')
    keyword_272880 = unicode_272879
    # Getting the type of 'apply_test' (line 542)
    apply_test_272881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 33), 'apply_test', False)
    keyword_272882 = apply_test_272881
    kwargs_272883 = {'comment': keyword_272880, 'apply': keyword_272882, 'title': keyword_272878}
    # Getting the type of 'fedit' (line 540)
    fedit_272875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 21), 'fedit', False)
    # Calling fedit(args, kwargs) (line 540)
    fedit_call_result_272884 = invoke(stypy.reporting.localization.Localization(__file__, 540, 21), fedit_272875, *[datalist_272876], **kwargs_272883)
    
    # Processing the call keyword arguments (line 540)
    kwargs_272885 = {}
    # Getting the type of 'print' (line 540)
    print_272873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 4), 'print', False)
    # Calling print(args, kwargs) (line 540)
    print_call_result_272886 = invoke(stypy.reporting.localization.Localization(__file__, 540, 4), print_272873, *[unicode_272874, fedit_call_result_272884], **kwargs_272885)
    
    
    # Assigning a Call to a Name (line 545):
    
    # Assigning a Call to a Name (line 545):
    
    # Call to create_datagroup_example(...): (line 545)
    # Processing the call keyword arguments (line 545)
    kwargs_272888 = {}
    # Getting the type of 'create_datagroup_example' (line 545)
    create_datagroup_example_272887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 16), 'create_datagroup_example', False)
    # Calling create_datagroup_example(args, kwargs) (line 545)
    create_datagroup_example_call_result_272889 = invoke(stypy.reporting.localization.Localization(__file__, 545, 16), create_datagroup_example_272887, *[], **kwargs_272888)
    
    # Assigning a type to the variable 'datagroup' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'datagroup', create_datagroup_example_call_result_272889)
    
    # Call to print(...): (line 546)
    # Processing the call arguments (line 546)
    unicode_272891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 10), 'unicode', u'result:')
    
    # Call to fedit(...): (line 546)
    # Processing the call arguments (line 546)
    # Getting the type of 'datagroup' (line 546)
    datagroup_272893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 27), 'datagroup', False)
    unicode_272894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 38), 'unicode', u'Global title')
    # Processing the call keyword arguments (line 546)
    kwargs_272895 = {}
    # Getting the type of 'fedit' (line 546)
    fedit_272892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 21), 'fedit', False)
    # Calling fedit(args, kwargs) (line 546)
    fedit_call_result_272896 = invoke(stypy.reporting.localization.Localization(__file__, 546, 21), fedit_272892, *[datagroup_272893, unicode_272894], **kwargs_272895)
    
    # Processing the call keyword arguments (line 546)
    kwargs_272897 = {}
    # Getting the type of 'print' (line 546)
    print_272890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'print', False)
    # Calling print(args, kwargs) (line 546)
    print_call_result_272898 = invoke(stypy.reporting.localization.Localization(__file__, 546, 4), print_272890, *[unicode_272891, fedit_call_result_272896], **kwargs_272897)
    
    
    # Assigning a Call to a Name (line 549):
    
    # Assigning a Call to a Name (line 549):
    
    # Call to create_datalist_example(...): (line 549)
    # Processing the call keyword arguments (line 549)
    kwargs_272900 = {}
    # Getting the type of 'create_datalist_example' (line 549)
    create_datalist_example_272899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 15), 'create_datalist_example', False)
    # Calling create_datalist_example(args, kwargs) (line 549)
    create_datalist_example_call_result_272901 = invoke(stypy.reporting.localization.Localization(__file__, 549, 15), create_datalist_example_272899, *[], **kwargs_272900)
    
    # Assigning a type to the variable 'datalist' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'datalist', create_datalist_example_call_result_272901)
    
    # Assigning a Call to a Name (line 550):
    
    # Assigning a Call to a Name (line 550):
    
    # Call to create_datagroup_example(...): (line 550)
    # Processing the call keyword arguments (line 550)
    kwargs_272903 = {}
    # Getting the type of 'create_datagroup_example' (line 550)
    create_datagroup_example_272902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 16), 'create_datagroup_example', False)
    # Calling create_datagroup_example(args, kwargs) (line 550)
    create_datagroup_example_call_result_272904 = invoke(stypy.reporting.localization.Localization(__file__, 550, 16), create_datagroup_example_272902, *[], **kwargs_272903)
    
    # Assigning a type to the variable 'datagroup' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'datagroup', create_datagroup_example_call_result_272904)
    
    # Call to print(...): (line 551)
    # Processing the call arguments (line 551)
    unicode_272906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 10), 'unicode', u'result:')
    
    # Call to fedit(...): (line 551)
    # Processing the call arguments (line 551)
    
    # Obtaining an instance of the builtin type 'tuple' (line 551)
    tuple_272908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 551)
    # Adding element type (line 551)
    
    # Obtaining an instance of the builtin type 'tuple' (line 551)
    tuple_272909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 551)
    # Adding element type (line 551)
    # Getting the type of 'datagroup' (line 551)
    datagroup_272910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 29), 'datagroup', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 551, 29), tuple_272909, datagroup_272910)
    # Adding element type (line 551)
    unicode_272911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 40), 'unicode', u'Title 1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 551, 29), tuple_272909, unicode_272911)
    # Adding element type (line 551)
    unicode_272912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 51), 'unicode', u'Tab 1 comment')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 551, 29), tuple_272909, unicode_272912)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 551, 28), tuple_272908, tuple_272909)
    # Adding element type (line 551)
    
    # Obtaining an instance of the builtin type 'tuple' (line 552)
    tuple_272913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 552)
    # Adding element type (line 552)
    # Getting the type of 'datalist' (line 552)
    datalist_272914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 29), 'datalist', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 552, 29), tuple_272913, datalist_272914)
    # Adding element type (line 552)
    unicode_272915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 39), 'unicode', u'Title 2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 552, 29), tuple_272913, unicode_272915)
    # Adding element type (line 552)
    unicode_272916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 50), 'unicode', u'Tab 2 comment')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 552, 29), tuple_272913, unicode_272916)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 551, 28), tuple_272908, tuple_272913)
    # Adding element type (line 551)
    
    # Obtaining an instance of the builtin type 'tuple' (line 553)
    tuple_272917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 553)
    # Adding element type (line 553)
    # Getting the type of 'datalist' (line 553)
    datalist_272918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 29), 'datalist', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 553, 29), tuple_272917, datalist_272918)
    # Adding element type (line 553)
    unicode_272919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 39), 'unicode', u'Title 3')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 553, 29), tuple_272917, unicode_272919)
    # Adding element type (line 553)
    unicode_272920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 50), 'unicode', u'Tab 3 comment')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 553, 29), tuple_272917, unicode_272920)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 551, 28), tuple_272908, tuple_272917)
    
    unicode_272921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 28), 'unicode', u'Global title')
    # Processing the call keyword arguments (line 551)
    kwargs_272922 = {}
    # Getting the type of 'fedit' (line 551)
    fedit_272907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 21), 'fedit', False)
    # Calling fedit(args, kwargs) (line 551)
    fedit_call_result_272923 = invoke(stypy.reporting.localization.Localization(__file__, 551, 21), fedit_272907, *[tuple_272908, unicode_272921], **kwargs_272922)
    
    # Processing the call keyword arguments (line 551)
    kwargs_272924 = {}
    # Getting the type of 'print' (line 551)
    print_272905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 4), 'print', False)
    # Calling print(args, kwargs) (line 551)
    print_call_result_272925 = invoke(stypy.reporting.localization.Localization(__file__, 551, 4), print_272905, *[unicode_272906, fedit_call_result_272923], **kwargs_272924)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
