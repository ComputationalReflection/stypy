
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: `ToolManager`
3:     Class that makes the bridge between user interaction (key press,
4:     toolbar clicks, ..) and the actions in response to the user inputs.
5: '''
6: 
7: from __future__ import (absolute_import, division, print_function,
8:                         unicode_literals)
9: import six
10: import warnings
11: 
12: import matplotlib.cbook as cbook
13: import matplotlib.widgets as widgets
14: from matplotlib.rcsetup import validate_stringlist
15: import matplotlib.backend_tools as tools
16: 
17: 
18: class ToolEvent(object):
19:     '''Event for tool manipulation (add/remove)'''
20:     def __init__(self, name, sender, tool, data=None):
21:         self.name = name
22:         self.sender = sender
23:         self.tool = tool
24:         self.data = data
25: 
26: 
27: class ToolTriggerEvent(ToolEvent):
28:     '''Event to inform  that a tool has been triggered'''
29:     def __init__(self, name, sender, tool, canvasevent=None, data=None):
30:         ToolEvent.__init__(self, name, sender, tool, data)
31:         self.canvasevent = canvasevent
32: 
33: 
34: class ToolManagerMessageEvent(object):
35:     '''
36:     Event carrying messages from toolmanager
37: 
38:     Messages usually get displayed to the user by the toolbar
39:     '''
40:     def __init__(self, name, sender, message):
41:         self.name = name
42:         self.sender = sender
43:         self.message = message
44: 
45: 
46: class ToolManager(object):
47:     '''
48:     Helper class that groups all the user interactions for a FigureManager
49: 
50:     Attributes
51:     ----------
52:     manager: `FigureManager`
53:     keypresslock: `widgets.LockDraw`
54:         `LockDraw` object to know if the `canvas` key_press_event is locked
55:     messagelock: `widgets.LockDraw`
56:         `LockDraw` object to know if the message is available to write
57:     '''
58: 
59:     def __init__(self, figure=None):
60:         warnings.warn('Treat the new Tool classes introduced in v1.5 as ' +
61:                        'experimental for now, the API will likely change in ' +
62:                        'version 2.1 and perhaps the rcParam as well')
63: 
64:         self._key_press_handler_id = None
65: 
66:         self._tools = {}
67:         self._keys = {}
68:         self._toggled = {}
69:         self._callbacks = cbook.CallbackRegistry()
70: 
71:         # to process keypress event
72:         self.keypresslock = widgets.LockDraw()
73:         self.messagelock = widgets.LockDraw()
74: 
75:         self._figure = None
76:         self.set_figure(figure)
77: 
78:     @property
79:     def canvas(self):
80:         '''Canvas managed by FigureManager'''
81:         if not self._figure:
82:             return None
83:         return self._figure.canvas
84: 
85:     @property
86:     def figure(self):
87:         '''Figure that holds the canvas'''
88:         return self._figure
89: 
90:     @figure.setter
91:     def figure(self, figure):
92:         self.set_figure(figure)
93: 
94:     def set_figure(self, figure, update_tools=True):
95:         '''
96:         Sets the figure to interact with the tools
97: 
98:         Parameters
99:         ==========
100:         figure: `Figure`
101:         update_tools: bool
102:             Force tools to update figure
103:         '''
104:         if self._key_press_handler_id:
105:             self.canvas.mpl_disconnect(self._key_press_handler_id)
106:         self._figure = figure
107:         if figure:
108:             self._key_press_handler_id = self.canvas.mpl_connect(
109:                 'key_press_event', self._key_press)
110:         if update_tools:
111:             for tool in self._tools.values():
112:                 tool.figure = figure
113: 
114:     def toolmanager_connect(self, s, func):
115:         '''
116:         Connect event with string *s* to *func*.
117: 
118:         Parameters
119:         ----------
120:         s : String
121:             Name of the event
122: 
123:             The following events are recognized
124: 
125:             - 'tool_message_event'
126:             - 'tool_removed_event'
127:             - 'tool_added_event'
128: 
129:             For every tool added a new event is created
130: 
131:             - 'tool_trigger_TOOLNAME`
132:               Where TOOLNAME is the id of the tool.
133: 
134:         func : function
135:             Function to be called with signature
136:             def func(event)
137:         '''
138:         return self._callbacks.connect(s, func)
139: 
140:     def toolmanager_disconnect(self, cid):
141:         '''
142:         Disconnect callback id *cid*
143: 
144:         Example usage::
145: 
146:             cid = toolmanager.toolmanager_connect('tool_trigger_zoom',
147:                                                   on_press)
148:             #...later
149:             toolmanager.toolmanager_disconnect(cid)
150:         '''
151:         return self._callbacks.disconnect(cid)
152: 
153:     def message_event(self, message, sender=None):
154:         ''' Emit a `ToolManagerMessageEvent`'''
155:         if sender is None:
156:             sender = self
157: 
158:         s = 'tool_message_event'
159:         event = ToolManagerMessageEvent(s, sender, message)
160:         self._callbacks.process(s, event)
161: 
162:     @property
163:     def active_toggle(self):
164:         '''Currently toggled tools'''
165: 
166:         return self._toggled
167: 
168:     def get_tool_keymap(self, name):
169:         '''
170:         Get the keymap associated with the specified tool
171: 
172:         Parameters
173:         ----------
174:         name : string
175:             Name of the Tool
176: 
177:         Returns
178:         -------
179:         list : list of keys associated with the Tool
180:         '''
181: 
182:         keys = [k for k, i in six.iteritems(self._keys) if i == name]
183:         return keys
184: 
185:     def _remove_keys(self, name):
186:         for k in self.get_tool_keymap(name):
187:             del self._keys[k]
188: 
189:     def update_keymap(self, name, *keys):
190:         '''
191:         Set the keymap to associate with the specified tool
192: 
193:         Parameters
194:         ----------
195:         name : string
196:             Name of the Tool
197:         keys : keys to associate with the Tool
198:         '''
199: 
200:         if name not in self._tools:
201:             raise KeyError('%s not in Tools' % name)
202: 
203:         self._remove_keys(name)
204: 
205:         for key in keys:
206:             for k in validate_stringlist(key):
207:                 if k in self._keys:
208:                     warnings.warn('Key %s changed from %s to %s' %
209:                                   (k, self._keys[k], name))
210:                 self._keys[k] = name
211: 
212:     def remove_tool(self, name):
213:         '''
214:         Remove tool from `ToolManager`
215: 
216:         Parameters
217:         ----------
218:         name : string
219:             Name of the Tool
220:         '''
221: 
222:         tool = self.get_tool(name)
223:         tool.destroy()
224: 
225:         # If is a toggle tool and toggled, untoggle
226:         if getattr(tool, 'toggled', False):
227:             self.trigger_tool(tool, 'toolmanager')
228: 
229:         self._remove_keys(name)
230: 
231:         s = 'tool_removed_event'
232:         event = ToolEvent(s, self, tool)
233:         self._callbacks.process(s, event)
234: 
235:         del self._tools[name]
236: 
237:     def add_tool(self, name, tool, *args, **kwargs):
238:         '''
239:         Add *tool* to `ToolManager`
240: 
241:         If successful adds a new event `tool_trigger_name` where **name** is
242:         the **name** of the tool, this event is fired everytime
243:         the tool is triggered.
244: 
245:         Parameters
246:         ----------
247:         name : str
248:             Name of the tool, treated as the ID, has to be unique
249:         tool : class_like, i.e. str or type
250:             Reference to find the class of the Tool to added.
251: 
252:         Notes
253:         -----
254:         args and kwargs get passed directly to the tools constructor.
255: 
256:         See Also
257:         --------
258:         matplotlib.backend_tools.ToolBase : The base class for tools.
259:         '''
260: 
261:         tool_cls = self._get_cls_to_instantiate(tool)
262:         if not tool_cls:
263:             raise ValueError('Impossible to find class for %s' % str(tool))
264: 
265:         if name in self._tools:
266:             warnings.warn('A "Tool class" with the same name already exists, '
267:                           'not added')
268:             return self._tools[name]
269: 
270:         tool_obj = tool_cls(self, name, *args, **kwargs)
271:         self._tools[name] = tool_obj
272: 
273:         if tool_cls.default_keymap is not None:
274:             self.update_keymap(name, tool_cls.default_keymap)
275: 
276:         # For toggle tools init the radio_group in self._toggled
277:         if isinstance(tool_obj, tools.ToolToggleBase):
278:             # None group is not mutually exclusive, a set is used to keep track
279:             # of all toggled tools in this group
280:             if tool_obj.radio_group is None:
281:                 self._toggled.setdefault(None, set())
282:             else:
283:                 self._toggled.setdefault(tool_obj.radio_group, None)
284: 
285:             # If initially toggled
286:             if tool_obj.toggled:
287:                 self._handle_toggle(tool_obj, None, None, None)
288:         tool_obj.set_figure(self.figure)
289: 
290:         self._tool_added_event(tool_obj)
291:         return tool_obj
292: 
293:     def _tool_added_event(self, tool):
294:         s = 'tool_added_event'
295:         event = ToolEvent(s, self, tool)
296:         self._callbacks.process(s, event)
297: 
298:     def _handle_toggle(self, tool, sender, canvasevent, data):
299:         '''
300:         Toggle tools, need to untoggle prior to using other Toggle tool
301:         Called from trigger_tool
302: 
303:         Parameters
304:         ----------
305:         tool: Tool object
306:         sender: object
307:             Object that wishes to trigger the tool
308:         canvasevent : Event
309:             Original Canvas event or None
310:         data : Object
311:             Extra data to pass to the tool when triggering
312:         '''
313: 
314:         radio_group = tool.radio_group
315:         # radio_group None is not mutually exclusive
316:         # just keep track of toggled tools in this group
317:         if radio_group is None:
318:             if tool.name in self._toggled[None]:
319:                 self._toggled[None].remove(tool.name)
320:             else:
321:                 self._toggled[None].add(tool.name)
322:             return
323: 
324:         # If the tool already has a toggled state, untoggle it
325:         if self._toggled[radio_group] == tool.name:
326:             toggled = None
327:         # If no tool was toggled in the radio_group
328:         # toggle it
329:         elif self._toggled[radio_group] is None:
330:             toggled = tool.name
331:         # Other tool in the radio_group is toggled
332:         else:
333:             # Untoggle previously toggled tool
334:             self.trigger_tool(self._toggled[radio_group],
335:                               self,
336:                               canvasevent,
337:                               data)
338:             toggled = tool.name
339: 
340:         # Keep track of the toggled tool in the radio_group
341:         self._toggled[radio_group] = toggled
342: 
343:     def _get_cls_to_instantiate(self, callback_class):
344:         # Find the class that corresponds to the tool
345:         if isinstance(callback_class, six.string_types):
346:             # FIXME: make more complete searching structure
347:             if callback_class in globals():
348:                 callback_class = globals()[callback_class]
349:             else:
350:                 mod = 'backend_tools'
351:                 current_module = __import__(mod,
352:                                             globals(), locals(), [mod], 1)
353: 
354:                 callback_class = getattr(current_module, callback_class, False)
355:         if callable(callback_class):
356:             return callback_class
357:         else:
358:             return None
359: 
360:     def trigger_tool(self, name, sender=None, canvasevent=None,
361:                      data=None):
362:         '''
363:         Trigger a tool and emit the tool_trigger_[name] event
364: 
365:         Parameters
366:         ----------
367:         name : string
368:             Name of the tool
369:         sender: object
370:             Object that wishes to trigger the tool
371:         canvasevent : Event
372:             Original Canvas event or None
373:         data : Object
374:             Extra data to pass to the tool when triggering
375:         '''
376:         tool = self.get_tool(name)
377:         if tool is None:
378:             return
379: 
380:         if sender is None:
381:             sender = self
382: 
383:         self._trigger_tool(name, sender, canvasevent, data)
384: 
385:         s = 'tool_trigger_%s' % name
386:         event = ToolTriggerEvent(s, sender, tool, canvasevent, data)
387:         self._callbacks.process(s, event)
388: 
389:     def _trigger_tool(self, name, sender=None, canvasevent=None, data=None):
390:         '''
391:         Trigger on a tool
392: 
393:         Method to actually trigger the tool
394:         '''
395:         tool = self.get_tool(name)
396: 
397:         if isinstance(tool, tools.ToolToggleBase):
398:             self._handle_toggle(tool, sender, canvasevent, data)
399: 
400:         # Important!!!
401:         # This is where the Tool object gets triggered
402:         tool.trigger(sender, canvasevent, data)
403: 
404:     def _key_press(self, event):
405:         if event.key is None or self.keypresslock.locked():
406:             return
407: 
408:         name = self._keys.get(event.key, None)
409:         if name is None:
410:             return
411:         self.trigger_tool(name, canvasevent=event)
412: 
413:     @property
414:     def tools(self):
415:         '''Return the tools controlled by `ToolManager`'''
416: 
417:         return self._tools
418: 
419:     def get_tool(self, name, warn=True):
420:         '''
421:         Return the tool object, also accepts the actual tool for convenience
422: 
423:         Parameters
424:         ----------
425:         name : str, ToolBase
426:             Name of the tool, or the tool itself
427:         warn : bool, optional
428:             If this method should give warnings.
429:         '''
430:         if isinstance(name, tools.ToolBase) and name.name in self._tools:
431:             return name
432:         if name not in self._tools:
433:             if warn:
434:                 warnings.warn("ToolManager does not control tool %s" % name)
435:             return None
436:         return self._tools[name]
437: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_19441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'unicode', u'\n`ToolManager`\n    Class that makes the bridge between user interaction (key press,\n    toolbar clicks, ..) and the actions in response to the user inputs.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import six' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_19442 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'six')

if (type(import_19442) is not StypyTypeError):

    if (import_19442 != 'pyd_module'):
        __import__(import_19442)
        sys_modules_19443 = sys.modules[import_19442]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'six', sys_modules_19443.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'six', import_19442)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import warnings' statement (line 10)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import matplotlib.cbook' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_19444 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.cbook')

if (type(import_19444) is not StypyTypeError):

    if (import_19444 != 'pyd_module'):
        __import__(import_19444)
        sys_modules_19445 = sys.modules[import_19444]
        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'cbook', sys_modules_19445.module_type_store, module_type_store)
    else:
        import matplotlib.cbook as cbook

        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'cbook', matplotlib.cbook, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.cbook', import_19444)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import matplotlib.widgets' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_19446 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.widgets')

if (type(import_19446) is not StypyTypeError):

    if (import_19446 != 'pyd_module'):
        __import__(import_19446)
        sys_modules_19447 = sys.modules[import_19446]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'widgets', sys_modules_19447.module_type_store, module_type_store)
    else:
        import matplotlib.widgets as widgets

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'widgets', matplotlib.widgets, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.widgets' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'matplotlib.widgets', import_19446)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from matplotlib.rcsetup import validate_stringlist' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_19448 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.rcsetup')

if (type(import_19448) is not StypyTypeError):

    if (import_19448 != 'pyd_module'):
        __import__(import_19448)
        sys_modules_19449 = sys.modules[import_19448]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.rcsetup', sys_modules_19449.module_type_store, module_type_store, ['validate_stringlist'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_19449, sys_modules_19449.module_type_store, module_type_store)
    else:
        from matplotlib.rcsetup import validate_stringlist

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.rcsetup', None, module_type_store, ['validate_stringlist'], [validate_stringlist])

else:
    # Assigning a type to the variable 'matplotlib.rcsetup' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib.rcsetup', import_19448)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'import matplotlib.backend_tools' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_19450 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.backend_tools')

if (type(import_19450) is not StypyTypeError):

    if (import_19450 != 'pyd_module'):
        __import__(import_19450)
        sys_modules_19451 = sys.modules[import_19450]
        import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'tools', sys_modules_19451.module_type_store, module_type_store)
    else:
        import matplotlib.backend_tools as tools

        import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'tools', matplotlib.backend_tools, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.backend_tools' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.backend_tools', import_19450)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

# Declaration of the 'ToolEvent' class

class ToolEvent(object, ):
    unicode_19452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 4), 'unicode', u'Event for tool manipulation (add/remove)')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 20)
        None_19453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 48), 'None')
        defaults = [None_19453]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 20, 4, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolEvent.__init__', ['name', 'sender', 'tool', 'data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['name', 'sender', 'tool', 'data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 21):
        # Getting the type of 'name' (line 21)
        name_19454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'name')
        # Getting the type of 'self' (line 21)
        self_19455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'self')
        # Setting the type of the member 'name' of a type (line 21)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), self_19455, 'name', name_19454)
        
        # Assigning a Name to a Attribute (line 22):
        # Getting the type of 'sender' (line 22)
        sender_19456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 22), 'sender')
        # Getting the type of 'self' (line 22)
        self_19457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self')
        # Setting the type of the member 'sender' of a type (line 22)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_19457, 'sender', sender_19456)
        
        # Assigning a Name to a Attribute (line 23):
        # Getting the type of 'tool' (line 23)
        tool_19458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 20), 'tool')
        # Getting the type of 'self' (line 23)
        self_19459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self')
        # Setting the type of the member 'tool' of a type (line 23)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), self_19459, 'tool', tool_19458)
        
        # Assigning a Name to a Attribute (line 24):
        # Getting the type of 'data' (line 24)
        data_19460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 20), 'data')
        # Getting the type of 'self' (line 24)
        self_19461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'self')
        # Setting the type of the member 'data' of a type (line 24)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), self_19461, 'data', data_19460)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'ToolEvent' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'ToolEvent', ToolEvent)
# Declaration of the 'ToolTriggerEvent' class
# Getting the type of 'ToolEvent' (line 27)
ToolEvent_19462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 23), 'ToolEvent')

class ToolTriggerEvent(ToolEvent_19462, ):
    unicode_19463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 4), 'unicode', u'Event to inform  that a tool has been triggered')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 29)
        None_19464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 55), 'None')
        # Getting the type of 'None' (line 29)
        None_19465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 66), 'None')
        defaults = [None_19464, None_19465]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 29, 4, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolTriggerEvent.__init__', ['name', 'sender', 'tool', 'canvasevent', 'data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['name', 'sender', 'tool', 'canvasevent', 'data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'self' (line 30)
        self_19468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 27), 'self', False)
        # Getting the type of 'name' (line 30)
        name_19469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 33), 'name', False)
        # Getting the type of 'sender' (line 30)
        sender_19470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 39), 'sender', False)
        # Getting the type of 'tool' (line 30)
        tool_19471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 47), 'tool', False)
        # Getting the type of 'data' (line 30)
        data_19472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 53), 'data', False)
        # Processing the call keyword arguments (line 30)
        kwargs_19473 = {}
        # Getting the type of 'ToolEvent' (line 30)
        ToolEvent_19466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'ToolEvent', False)
        # Obtaining the member '__init__' of a type (line 30)
        init___19467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), ToolEvent_19466, '__init__')
        # Calling __init__(args, kwargs) (line 30)
        init___call_result_19474 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), init___19467, *[self_19468, name_19469, sender_19470, tool_19471, data_19472], **kwargs_19473)
        
        
        # Assigning a Name to a Attribute (line 31):
        # Getting the type of 'canvasevent' (line 31)
        canvasevent_19475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 27), 'canvasevent')
        # Getting the type of 'self' (line 31)
        self_19476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member 'canvasevent' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_19476, 'canvasevent', canvasevent_19475)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'ToolTriggerEvent' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'ToolTriggerEvent', ToolTriggerEvent)
# Declaration of the 'ToolManagerMessageEvent' class

class ToolManagerMessageEvent(object, ):
    unicode_19477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, (-1)), 'unicode', u'\n    Event carrying messages from toolmanager\n\n    Messages usually get displayed to the user by the toolbar\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 40, 4, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManagerMessageEvent.__init__', ['name', 'sender', 'message'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['name', 'sender', 'message'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 41):
        # Getting the type of 'name' (line 41)
        name_19478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'name')
        # Getting the type of 'self' (line 41)
        self_19479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self')
        # Setting the type of the member 'name' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_19479, 'name', name_19478)
        
        # Assigning a Name to a Attribute (line 42):
        # Getting the type of 'sender' (line 42)
        sender_19480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 22), 'sender')
        # Getting the type of 'self' (line 42)
        self_19481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self')
        # Setting the type of the member 'sender' of a type (line 42)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_19481, 'sender', sender_19480)
        
        # Assigning a Name to a Attribute (line 43):
        # Getting the type of 'message' (line 43)
        message_19482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 23), 'message')
        # Getting the type of 'self' (line 43)
        self_19483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self')
        # Setting the type of the member 'message' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_19483, 'message', message_19482)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'ToolManagerMessageEvent' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'ToolManagerMessageEvent', ToolManagerMessageEvent)
# Declaration of the 'ToolManager' class

class ToolManager(object, ):
    unicode_19484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, (-1)), 'unicode', u'\n    Helper class that groups all the user interactions for a FigureManager\n\n    Attributes\n    ----------\n    manager: `FigureManager`\n    keypresslock: `widgets.LockDraw`\n        `LockDraw` object to know if the `canvas` key_press_event is locked\n    messagelock: `widgets.LockDraw`\n        `LockDraw` object to know if the message is available to write\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 59)
        None_19485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 30), 'None')
        defaults = [None_19485]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 59, 4, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManager.__init__', ['figure'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['figure'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to warn(...): (line 60)
        # Processing the call arguments (line 60)
        unicode_19488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 22), 'unicode', u'Treat the new Tool classes introduced in v1.5 as ')
        unicode_19489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 23), 'unicode', u'experimental for now, the API will likely change in ')
        # Applying the binary operator '+' (line 60)
        result_add_19490 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 22), '+', unicode_19488, unicode_19489)
        
        unicode_19491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 23), 'unicode', u'version 2.1 and perhaps the rcParam as well')
        # Applying the binary operator '+' (line 61)
        result_add_19492 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 78), '+', result_add_19490, unicode_19491)
        
        # Processing the call keyword arguments (line 60)
        kwargs_19493 = {}
        # Getting the type of 'warnings' (line 60)
        warnings_19486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 60)
        warn_19487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), warnings_19486, 'warn')
        # Calling warn(args, kwargs) (line 60)
        warn_call_result_19494 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), warn_19487, *[result_add_19492], **kwargs_19493)
        
        
        # Assigning a Name to a Attribute (line 64):
        # Getting the type of 'None' (line 64)
        None_19495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 37), 'None')
        # Getting the type of 'self' (line 64)
        self_19496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self')
        # Setting the type of the member '_key_press_handler_id' of a type (line 64)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_19496, '_key_press_handler_id', None_19495)
        
        # Assigning a Dict to a Attribute (line 66):
        
        # Obtaining an instance of the builtin type 'dict' (line 66)
        dict_19497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 22), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 66)
        
        # Getting the type of 'self' (line 66)
        self_19498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self')
        # Setting the type of the member '_tools' of a type (line 66)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_19498, '_tools', dict_19497)
        
        # Assigning a Dict to a Attribute (line 67):
        
        # Obtaining an instance of the builtin type 'dict' (line 67)
        dict_19499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 21), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 67)
        
        # Getting the type of 'self' (line 67)
        self_19500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self')
        # Setting the type of the member '_keys' of a type (line 67)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_19500, '_keys', dict_19499)
        
        # Assigning a Dict to a Attribute (line 68):
        
        # Obtaining an instance of the builtin type 'dict' (line 68)
        dict_19501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 24), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 68)
        
        # Getting the type of 'self' (line 68)
        self_19502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self')
        # Setting the type of the member '_toggled' of a type (line 68)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_19502, '_toggled', dict_19501)
        
        # Assigning a Call to a Attribute (line 69):
        
        # Call to CallbackRegistry(...): (line 69)
        # Processing the call keyword arguments (line 69)
        kwargs_19505 = {}
        # Getting the type of 'cbook' (line 69)
        cbook_19503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 26), 'cbook', False)
        # Obtaining the member 'CallbackRegistry' of a type (line 69)
        CallbackRegistry_19504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 26), cbook_19503, 'CallbackRegistry')
        # Calling CallbackRegistry(args, kwargs) (line 69)
        CallbackRegistry_call_result_19506 = invoke(stypy.reporting.localization.Localization(__file__, 69, 26), CallbackRegistry_19504, *[], **kwargs_19505)
        
        # Getting the type of 'self' (line 69)
        self_19507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'self')
        # Setting the type of the member '_callbacks' of a type (line 69)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), self_19507, '_callbacks', CallbackRegistry_call_result_19506)
        
        # Assigning a Call to a Attribute (line 72):
        
        # Call to LockDraw(...): (line 72)
        # Processing the call keyword arguments (line 72)
        kwargs_19510 = {}
        # Getting the type of 'widgets' (line 72)
        widgets_19508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'widgets', False)
        # Obtaining the member 'LockDraw' of a type (line 72)
        LockDraw_19509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 28), widgets_19508, 'LockDraw')
        # Calling LockDraw(args, kwargs) (line 72)
        LockDraw_call_result_19511 = invoke(stypy.reporting.localization.Localization(__file__, 72, 28), LockDraw_19509, *[], **kwargs_19510)
        
        # Getting the type of 'self' (line 72)
        self_19512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'self')
        # Setting the type of the member 'keypresslock' of a type (line 72)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), self_19512, 'keypresslock', LockDraw_call_result_19511)
        
        # Assigning a Call to a Attribute (line 73):
        
        # Call to LockDraw(...): (line 73)
        # Processing the call keyword arguments (line 73)
        kwargs_19515 = {}
        # Getting the type of 'widgets' (line 73)
        widgets_19513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 27), 'widgets', False)
        # Obtaining the member 'LockDraw' of a type (line 73)
        LockDraw_19514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 27), widgets_19513, 'LockDraw')
        # Calling LockDraw(args, kwargs) (line 73)
        LockDraw_call_result_19516 = invoke(stypy.reporting.localization.Localization(__file__, 73, 27), LockDraw_19514, *[], **kwargs_19515)
        
        # Getting the type of 'self' (line 73)
        self_19517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self')
        # Setting the type of the member 'messagelock' of a type (line 73)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_19517, 'messagelock', LockDraw_call_result_19516)
        
        # Assigning a Name to a Attribute (line 75):
        # Getting the type of 'None' (line 75)
        None_19518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 23), 'None')
        # Getting the type of 'self' (line 75)
        self_19519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'self')
        # Setting the type of the member '_figure' of a type (line 75)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), self_19519, '_figure', None_19518)
        
        # Call to set_figure(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'figure' (line 76)
        figure_19522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'figure', False)
        # Processing the call keyword arguments (line 76)
        kwargs_19523 = {}
        # Getting the type of 'self' (line 76)
        self_19520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'self', False)
        # Obtaining the member 'set_figure' of a type (line 76)
        set_figure_19521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), self_19520, 'set_figure')
        # Calling set_figure(args, kwargs) (line 76)
        set_figure_call_result_19524 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), set_figure_19521, *[figure_19522], **kwargs_19523)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def canvas(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'canvas'
        module_type_store = module_type_store.open_function_context('canvas', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolManager.canvas.__dict__.__setitem__('stypy_localization', localization)
        ToolManager.canvas.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolManager.canvas.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolManager.canvas.__dict__.__setitem__('stypy_function_name', 'ToolManager.canvas')
        ToolManager.canvas.__dict__.__setitem__('stypy_param_names_list', [])
        ToolManager.canvas.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolManager.canvas.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolManager.canvas.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolManager.canvas.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolManager.canvas.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolManager.canvas.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManager.canvas', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'canvas', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'canvas(...)' code ##################

        unicode_19525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 8), 'unicode', u'Canvas managed by FigureManager')
        
        
        # Getting the type of 'self' (line 81)
        self_19526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'self')
        # Obtaining the member '_figure' of a type (line 81)
        _figure_19527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 15), self_19526, '_figure')
        # Applying the 'not' unary operator (line 81)
        result_not__19528 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 11), 'not', _figure_19527)
        
        # Testing the type of an if condition (line 81)
        if_condition_19529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 8), result_not__19528)
        # Assigning a type to the variable 'if_condition_19529' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'if_condition_19529', if_condition_19529)
        # SSA begins for if statement (line 81)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 82)
        None_19530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'stypy_return_type', None_19530)
        # SSA join for if statement (line 81)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'self' (line 83)
        self_19531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'self')
        # Obtaining the member '_figure' of a type (line 83)
        _figure_19532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 15), self_19531, '_figure')
        # Obtaining the member 'canvas' of a type (line 83)
        canvas_19533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 15), _figure_19532, 'canvas')
        # Assigning a type to the variable 'stypy_return_type' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'stypy_return_type', canvas_19533)
        
        # ################# End of 'canvas(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'canvas' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_19534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19534)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'canvas'
        return stypy_return_type_19534


    @norecursion
    def figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'figure'
        module_type_store = module_type_store.open_function_context('figure', 85, 4, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolManager.figure.__dict__.__setitem__('stypy_localization', localization)
        ToolManager.figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolManager.figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolManager.figure.__dict__.__setitem__('stypy_function_name', 'ToolManager.figure')
        ToolManager.figure.__dict__.__setitem__('stypy_param_names_list', [])
        ToolManager.figure.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolManager.figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolManager.figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolManager.figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolManager.figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolManager.figure.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManager.figure', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'figure', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'figure(...)' code ##################

        unicode_19535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 8), 'unicode', u'Figure that holds the canvas')
        # Getting the type of 'self' (line 88)
        self_19536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'self')
        # Obtaining the member '_figure' of a type (line 88)
        _figure_19537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 15), self_19536, '_figure')
        # Assigning a type to the variable 'stypy_return_type' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'stypy_return_type', _figure_19537)
        
        # ################# End of 'figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'figure' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_19538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19538)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'figure'
        return stypy_return_type_19538


    @norecursion
    def figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'figure'
        module_type_store = module_type_store.open_function_context('figure', 90, 4, False)
        # Assigning a type to the variable 'self' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolManager.figure.__dict__.__setitem__('stypy_localization', localization)
        ToolManager.figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolManager.figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolManager.figure.__dict__.__setitem__('stypy_function_name', 'ToolManager.figure')
        ToolManager.figure.__dict__.__setitem__('stypy_param_names_list', ['figure'])
        ToolManager.figure.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolManager.figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolManager.figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolManager.figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolManager.figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolManager.figure.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManager.figure', ['figure'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'figure', localization, ['figure'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'figure(...)' code ##################

        
        # Call to set_figure(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'figure' (line 92)
        figure_19541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'figure', False)
        # Processing the call keyword arguments (line 92)
        kwargs_19542 = {}
        # Getting the type of 'self' (line 92)
        self_19539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'self', False)
        # Obtaining the member 'set_figure' of a type (line 92)
        set_figure_19540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), self_19539, 'set_figure')
        # Calling set_figure(args, kwargs) (line 92)
        set_figure_call_result_19543 = invoke(stypy.reporting.localization.Localization(__file__, 92, 8), set_figure_19540, *[figure_19541], **kwargs_19542)
        
        
        # ################# End of 'figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'figure' in the type store
        # Getting the type of 'stypy_return_type' (line 90)
        stypy_return_type_19544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19544)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'figure'
        return stypy_return_type_19544


    @norecursion
    def set_figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 94)
        True_19545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 46), 'True')
        defaults = [True_19545]
        # Create a new context for function 'set_figure'
        module_type_store = module_type_store.open_function_context('set_figure', 94, 4, False)
        # Assigning a type to the variable 'self' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolManager.set_figure.__dict__.__setitem__('stypy_localization', localization)
        ToolManager.set_figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolManager.set_figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolManager.set_figure.__dict__.__setitem__('stypy_function_name', 'ToolManager.set_figure')
        ToolManager.set_figure.__dict__.__setitem__('stypy_param_names_list', ['figure', 'update_tools'])
        ToolManager.set_figure.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolManager.set_figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolManager.set_figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolManager.set_figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolManager.set_figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolManager.set_figure.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManager.set_figure', ['figure', 'update_tools'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_figure', localization, ['figure', 'update_tools'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_figure(...)' code ##################

        unicode_19546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, (-1)), 'unicode', u'\n        Sets the figure to interact with the tools\n\n        Parameters\n        ==========\n        figure: `Figure`\n        update_tools: bool\n            Force tools to update figure\n        ')
        
        # Getting the type of 'self' (line 104)
        self_19547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 11), 'self')
        # Obtaining the member '_key_press_handler_id' of a type (line 104)
        _key_press_handler_id_19548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 11), self_19547, '_key_press_handler_id')
        # Testing the type of an if condition (line 104)
        if_condition_19549 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 8), _key_press_handler_id_19548)
        # Assigning a type to the variable 'if_condition_19549' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'if_condition_19549', if_condition_19549)
        # SSA begins for if statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to mpl_disconnect(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'self' (line 105)
        self_19553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 39), 'self', False)
        # Obtaining the member '_key_press_handler_id' of a type (line 105)
        _key_press_handler_id_19554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 39), self_19553, '_key_press_handler_id')
        # Processing the call keyword arguments (line 105)
        kwargs_19555 = {}
        # Getting the type of 'self' (line 105)
        self_19550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'self', False)
        # Obtaining the member 'canvas' of a type (line 105)
        canvas_19551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), self_19550, 'canvas')
        # Obtaining the member 'mpl_disconnect' of a type (line 105)
        mpl_disconnect_19552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), canvas_19551, 'mpl_disconnect')
        # Calling mpl_disconnect(args, kwargs) (line 105)
        mpl_disconnect_call_result_19556 = invoke(stypy.reporting.localization.Localization(__file__, 105, 12), mpl_disconnect_19552, *[_key_press_handler_id_19554], **kwargs_19555)
        
        # SSA join for if statement (line 104)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 106):
        # Getting the type of 'figure' (line 106)
        figure_19557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 23), 'figure')
        # Getting the type of 'self' (line 106)
        self_19558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'self')
        # Setting the type of the member '_figure' of a type (line 106)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), self_19558, '_figure', figure_19557)
        
        # Getting the type of 'figure' (line 107)
        figure_19559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), 'figure')
        # Testing the type of an if condition (line 107)
        if_condition_19560 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 8), figure_19559)
        # Assigning a type to the variable 'if_condition_19560' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'if_condition_19560', if_condition_19560)
        # SSA begins for if statement (line 107)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 108):
        
        # Call to mpl_connect(...): (line 108)
        # Processing the call arguments (line 108)
        unicode_19564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 16), 'unicode', u'key_press_event')
        # Getting the type of 'self' (line 109)
        self_19565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 35), 'self', False)
        # Obtaining the member '_key_press' of a type (line 109)
        _key_press_19566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 35), self_19565, '_key_press')
        # Processing the call keyword arguments (line 108)
        kwargs_19567 = {}
        # Getting the type of 'self' (line 108)
        self_19561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 41), 'self', False)
        # Obtaining the member 'canvas' of a type (line 108)
        canvas_19562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 41), self_19561, 'canvas')
        # Obtaining the member 'mpl_connect' of a type (line 108)
        mpl_connect_19563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 41), canvas_19562, 'mpl_connect')
        # Calling mpl_connect(args, kwargs) (line 108)
        mpl_connect_call_result_19568 = invoke(stypy.reporting.localization.Localization(__file__, 108, 41), mpl_connect_19563, *[unicode_19564, _key_press_19566], **kwargs_19567)
        
        # Getting the type of 'self' (line 108)
        self_19569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'self')
        # Setting the type of the member '_key_press_handler_id' of a type (line 108)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), self_19569, '_key_press_handler_id', mpl_connect_call_result_19568)
        # SSA join for if statement (line 107)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'update_tools' (line 110)
        update_tools_19570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'update_tools')
        # Testing the type of an if condition (line 110)
        if_condition_19571 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 8), update_tools_19570)
        # Assigning a type to the variable 'if_condition_19571' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'if_condition_19571', if_condition_19571)
        # SSA begins for if statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to values(...): (line 111)
        # Processing the call keyword arguments (line 111)
        kwargs_19575 = {}
        # Getting the type of 'self' (line 111)
        self_19572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'self', False)
        # Obtaining the member '_tools' of a type (line 111)
        _tools_19573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 24), self_19572, '_tools')
        # Obtaining the member 'values' of a type (line 111)
        values_19574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 24), _tools_19573, 'values')
        # Calling values(args, kwargs) (line 111)
        values_call_result_19576 = invoke(stypy.reporting.localization.Localization(__file__, 111, 24), values_19574, *[], **kwargs_19575)
        
        # Testing the type of a for loop iterable (line 111)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 111, 12), values_call_result_19576)
        # Getting the type of the for loop variable (line 111)
        for_loop_var_19577 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 111, 12), values_call_result_19576)
        # Assigning a type to the variable 'tool' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'tool', for_loop_var_19577)
        # SSA begins for a for statement (line 111)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Attribute (line 112):
        # Getting the type of 'figure' (line 112)
        figure_19578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 30), 'figure')
        # Getting the type of 'tool' (line 112)
        tool_19579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'tool')
        # Setting the type of the member 'figure' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 16), tool_19579, 'figure', figure_19578)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 110)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_figure' in the type store
        # Getting the type of 'stypy_return_type' (line 94)
        stypy_return_type_19580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19580)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_figure'
        return stypy_return_type_19580


    @norecursion
    def toolmanager_connect(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'toolmanager_connect'
        module_type_store = module_type_store.open_function_context('toolmanager_connect', 114, 4, False)
        # Assigning a type to the variable 'self' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolManager.toolmanager_connect.__dict__.__setitem__('stypy_localization', localization)
        ToolManager.toolmanager_connect.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolManager.toolmanager_connect.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolManager.toolmanager_connect.__dict__.__setitem__('stypy_function_name', 'ToolManager.toolmanager_connect')
        ToolManager.toolmanager_connect.__dict__.__setitem__('stypy_param_names_list', ['s', 'func'])
        ToolManager.toolmanager_connect.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolManager.toolmanager_connect.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolManager.toolmanager_connect.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolManager.toolmanager_connect.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolManager.toolmanager_connect.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolManager.toolmanager_connect.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManager.toolmanager_connect', ['s', 'func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'toolmanager_connect', localization, ['s', 'func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'toolmanager_connect(...)' code ##################

        unicode_19581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, (-1)), 'unicode', u"\n        Connect event with string *s* to *func*.\n\n        Parameters\n        ----------\n        s : String\n            Name of the event\n\n            The following events are recognized\n\n            - 'tool_message_event'\n            - 'tool_removed_event'\n            - 'tool_added_event'\n\n            For every tool added a new event is created\n\n            - 'tool_trigger_TOOLNAME`\n              Where TOOLNAME is the id of the tool.\n\n        func : function\n            Function to be called with signature\n            def func(event)\n        ")
        
        # Call to connect(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 's' (line 138)
        s_19585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 39), 's', False)
        # Getting the type of 'func' (line 138)
        func_19586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 42), 'func', False)
        # Processing the call keyword arguments (line 138)
        kwargs_19587 = {}
        # Getting the type of 'self' (line 138)
        self_19582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'self', False)
        # Obtaining the member '_callbacks' of a type (line 138)
        _callbacks_19583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 15), self_19582, '_callbacks')
        # Obtaining the member 'connect' of a type (line 138)
        connect_19584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 15), _callbacks_19583, 'connect')
        # Calling connect(args, kwargs) (line 138)
        connect_call_result_19588 = invoke(stypy.reporting.localization.Localization(__file__, 138, 15), connect_19584, *[s_19585, func_19586], **kwargs_19587)
        
        # Assigning a type to the variable 'stypy_return_type' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'stypy_return_type', connect_call_result_19588)
        
        # ################# End of 'toolmanager_connect(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'toolmanager_connect' in the type store
        # Getting the type of 'stypy_return_type' (line 114)
        stypy_return_type_19589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19589)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'toolmanager_connect'
        return stypy_return_type_19589


    @norecursion
    def toolmanager_disconnect(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'toolmanager_disconnect'
        module_type_store = module_type_store.open_function_context('toolmanager_disconnect', 140, 4, False)
        # Assigning a type to the variable 'self' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolManager.toolmanager_disconnect.__dict__.__setitem__('stypy_localization', localization)
        ToolManager.toolmanager_disconnect.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolManager.toolmanager_disconnect.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolManager.toolmanager_disconnect.__dict__.__setitem__('stypy_function_name', 'ToolManager.toolmanager_disconnect')
        ToolManager.toolmanager_disconnect.__dict__.__setitem__('stypy_param_names_list', ['cid'])
        ToolManager.toolmanager_disconnect.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolManager.toolmanager_disconnect.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolManager.toolmanager_disconnect.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolManager.toolmanager_disconnect.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolManager.toolmanager_disconnect.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolManager.toolmanager_disconnect.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManager.toolmanager_disconnect', ['cid'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'toolmanager_disconnect', localization, ['cid'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'toolmanager_disconnect(...)' code ##################

        unicode_19590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, (-1)), 'unicode', u"\n        Disconnect callback id *cid*\n\n        Example usage::\n\n            cid = toolmanager.toolmanager_connect('tool_trigger_zoom',\n                                                  on_press)\n            #...later\n            toolmanager.toolmanager_disconnect(cid)\n        ")
        
        # Call to disconnect(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'cid' (line 151)
        cid_19594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 42), 'cid', False)
        # Processing the call keyword arguments (line 151)
        kwargs_19595 = {}
        # Getting the type of 'self' (line 151)
        self_19591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 15), 'self', False)
        # Obtaining the member '_callbacks' of a type (line 151)
        _callbacks_19592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 15), self_19591, '_callbacks')
        # Obtaining the member 'disconnect' of a type (line 151)
        disconnect_19593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 15), _callbacks_19592, 'disconnect')
        # Calling disconnect(args, kwargs) (line 151)
        disconnect_call_result_19596 = invoke(stypy.reporting.localization.Localization(__file__, 151, 15), disconnect_19593, *[cid_19594], **kwargs_19595)
        
        # Assigning a type to the variable 'stypy_return_type' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'stypy_return_type', disconnect_call_result_19596)
        
        # ################# End of 'toolmanager_disconnect(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'toolmanager_disconnect' in the type store
        # Getting the type of 'stypy_return_type' (line 140)
        stypy_return_type_19597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19597)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'toolmanager_disconnect'
        return stypy_return_type_19597


    @norecursion
    def message_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 153)
        None_19598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 44), 'None')
        defaults = [None_19598]
        # Create a new context for function 'message_event'
        module_type_store = module_type_store.open_function_context('message_event', 153, 4, False)
        # Assigning a type to the variable 'self' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolManager.message_event.__dict__.__setitem__('stypy_localization', localization)
        ToolManager.message_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolManager.message_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolManager.message_event.__dict__.__setitem__('stypy_function_name', 'ToolManager.message_event')
        ToolManager.message_event.__dict__.__setitem__('stypy_param_names_list', ['message', 'sender'])
        ToolManager.message_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolManager.message_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolManager.message_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolManager.message_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolManager.message_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolManager.message_event.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManager.message_event', ['message', 'sender'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'message_event', localization, ['message', 'sender'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'message_event(...)' code ##################

        unicode_19599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 8), 'unicode', u' Emit a `ToolManagerMessageEvent`')
        
        # Type idiom detected: calculating its left and rigth part (line 155)
        # Getting the type of 'sender' (line 155)
        sender_19600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'sender')
        # Getting the type of 'None' (line 155)
        None_19601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 21), 'None')
        
        (may_be_19602, more_types_in_union_19603) = may_be_none(sender_19600, None_19601)

        if may_be_19602:

            if more_types_in_union_19603:
                # Runtime conditional SSA (line 155)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 156):
            # Getting the type of 'self' (line 156)
            self_19604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 21), 'self')
            # Assigning a type to the variable 'sender' (line 156)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'sender', self_19604)

            if more_types_in_union_19603:
                # SSA join for if statement (line 155)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Str to a Name (line 158):
        unicode_19605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 12), 'unicode', u'tool_message_event')
        # Assigning a type to the variable 's' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 's', unicode_19605)
        
        # Assigning a Call to a Name (line 159):
        
        # Call to ToolManagerMessageEvent(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 's' (line 159)
        s_19607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 40), 's', False)
        # Getting the type of 'sender' (line 159)
        sender_19608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 43), 'sender', False)
        # Getting the type of 'message' (line 159)
        message_19609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 51), 'message', False)
        # Processing the call keyword arguments (line 159)
        kwargs_19610 = {}
        # Getting the type of 'ToolManagerMessageEvent' (line 159)
        ToolManagerMessageEvent_19606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'ToolManagerMessageEvent', False)
        # Calling ToolManagerMessageEvent(args, kwargs) (line 159)
        ToolManagerMessageEvent_call_result_19611 = invoke(stypy.reporting.localization.Localization(__file__, 159, 16), ToolManagerMessageEvent_19606, *[s_19607, sender_19608, message_19609], **kwargs_19610)
        
        # Assigning a type to the variable 'event' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'event', ToolManagerMessageEvent_call_result_19611)
        
        # Call to process(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 's' (line 160)
        s_19615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 32), 's', False)
        # Getting the type of 'event' (line 160)
        event_19616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 35), 'event', False)
        # Processing the call keyword arguments (line 160)
        kwargs_19617 = {}
        # Getting the type of 'self' (line 160)
        self_19612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'self', False)
        # Obtaining the member '_callbacks' of a type (line 160)
        _callbacks_19613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), self_19612, '_callbacks')
        # Obtaining the member 'process' of a type (line 160)
        process_19614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), _callbacks_19613, 'process')
        # Calling process(args, kwargs) (line 160)
        process_call_result_19618 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), process_19614, *[s_19615, event_19616], **kwargs_19617)
        
        
        # ################# End of 'message_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'message_event' in the type store
        # Getting the type of 'stypy_return_type' (line 153)
        stypy_return_type_19619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19619)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'message_event'
        return stypy_return_type_19619


    @norecursion
    def active_toggle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'active_toggle'
        module_type_store = module_type_store.open_function_context('active_toggle', 162, 4, False)
        # Assigning a type to the variable 'self' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolManager.active_toggle.__dict__.__setitem__('stypy_localization', localization)
        ToolManager.active_toggle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolManager.active_toggle.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolManager.active_toggle.__dict__.__setitem__('stypy_function_name', 'ToolManager.active_toggle')
        ToolManager.active_toggle.__dict__.__setitem__('stypy_param_names_list', [])
        ToolManager.active_toggle.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolManager.active_toggle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolManager.active_toggle.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolManager.active_toggle.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolManager.active_toggle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolManager.active_toggle.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManager.active_toggle', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'active_toggle', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'active_toggle(...)' code ##################

        unicode_19620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 8), 'unicode', u'Currently toggled tools')
        # Getting the type of 'self' (line 166)
        self_19621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 15), 'self')
        # Obtaining the member '_toggled' of a type (line 166)
        _toggled_19622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 15), self_19621, '_toggled')
        # Assigning a type to the variable 'stypy_return_type' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'stypy_return_type', _toggled_19622)
        
        # ################# End of 'active_toggle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'active_toggle' in the type store
        # Getting the type of 'stypy_return_type' (line 162)
        stypy_return_type_19623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19623)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'active_toggle'
        return stypy_return_type_19623


    @norecursion
    def get_tool_keymap(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_tool_keymap'
        module_type_store = module_type_store.open_function_context('get_tool_keymap', 168, 4, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolManager.get_tool_keymap.__dict__.__setitem__('stypy_localization', localization)
        ToolManager.get_tool_keymap.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolManager.get_tool_keymap.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolManager.get_tool_keymap.__dict__.__setitem__('stypy_function_name', 'ToolManager.get_tool_keymap')
        ToolManager.get_tool_keymap.__dict__.__setitem__('stypy_param_names_list', ['name'])
        ToolManager.get_tool_keymap.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolManager.get_tool_keymap.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolManager.get_tool_keymap.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolManager.get_tool_keymap.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolManager.get_tool_keymap.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolManager.get_tool_keymap.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManager.get_tool_keymap', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_tool_keymap', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_tool_keymap(...)' code ##################

        unicode_19624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, (-1)), 'unicode', u'\n        Get the keymap associated with the specified tool\n\n        Parameters\n        ----------\n        name : string\n            Name of the Tool\n\n        Returns\n        -------\n        list : list of keys associated with the Tool\n        ')
        
        # Assigning a ListComp to a Name (line 182):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to iteritems(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'self' (line 182)
        self_19631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 44), 'self', False)
        # Obtaining the member '_keys' of a type (line 182)
        _keys_19632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 44), self_19631, '_keys')
        # Processing the call keyword arguments (line 182)
        kwargs_19633 = {}
        # Getting the type of 'six' (line 182)
        six_19629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 30), 'six', False)
        # Obtaining the member 'iteritems' of a type (line 182)
        iteritems_19630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 30), six_19629, 'iteritems')
        # Calling iteritems(args, kwargs) (line 182)
        iteritems_call_result_19634 = invoke(stypy.reporting.localization.Localization(__file__, 182, 30), iteritems_19630, *[_keys_19632], **kwargs_19633)
        
        comprehension_19635 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 16), iteritems_call_result_19634)
        # Assigning a type to the variable 'k' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 16), comprehension_19635))
        # Assigning a type to the variable 'i' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 16), comprehension_19635))
        
        # Getting the type of 'i' (line 182)
        i_19626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 59), 'i')
        # Getting the type of 'name' (line 182)
        name_19627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 64), 'name')
        # Applying the binary operator '==' (line 182)
        result_eq_19628 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 59), '==', i_19626, name_19627)
        
        # Getting the type of 'k' (line 182)
        k_19625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'k')
        list_19636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 16), list_19636, k_19625)
        # Assigning a type to the variable 'keys' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'keys', list_19636)
        # Getting the type of 'keys' (line 183)
        keys_19637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 15), 'keys')
        # Assigning a type to the variable 'stypy_return_type' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'stypy_return_type', keys_19637)
        
        # ################# End of 'get_tool_keymap(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_tool_keymap' in the type store
        # Getting the type of 'stypy_return_type' (line 168)
        stypy_return_type_19638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19638)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_tool_keymap'
        return stypy_return_type_19638


    @norecursion
    def _remove_keys(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_remove_keys'
        module_type_store = module_type_store.open_function_context('_remove_keys', 185, 4, False)
        # Assigning a type to the variable 'self' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolManager._remove_keys.__dict__.__setitem__('stypy_localization', localization)
        ToolManager._remove_keys.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolManager._remove_keys.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolManager._remove_keys.__dict__.__setitem__('stypy_function_name', 'ToolManager._remove_keys')
        ToolManager._remove_keys.__dict__.__setitem__('stypy_param_names_list', ['name'])
        ToolManager._remove_keys.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolManager._remove_keys.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolManager._remove_keys.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolManager._remove_keys.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolManager._remove_keys.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolManager._remove_keys.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManager._remove_keys', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_remove_keys', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_remove_keys(...)' code ##################

        
        
        # Call to get_tool_keymap(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'name' (line 186)
        name_19641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 38), 'name', False)
        # Processing the call keyword arguments (line 186)
        kwargs_19642 = {}
        # Getting the type of 'self' (line 186)
        self_19639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 17), 'self', False)
        # Obtaining the member 'get_tool_keymap' of a type (line 186)
        get_tool_keymap_19640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 17), self_19639, 'get_tool_keymap')
        # Calling get_tool_keymap(args, kwargs) (line 186)
        get_tool_keymap_call_result_19643 = invoke(stypy.reporting.localization.Localization(__file__, 186, 17), get_tool_keymap_19640, *[name_19641], **kwargs_19642)
        
        # Testing the type of a for loop iterable (line 186)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 186, 8), get_tool_keymap_call_result_19643)
        # Getting the type of the for loop variable (line 186)
        for_loop_var_19644 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 186, 8), get_tool_keymap_call_result_19643)
        # Assigning a type to the variable 'k' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'k', for_loop_var_19644)
        # SSA begins for a for statement (line 186)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        # Deleting a member
        # Getting the type of 'self' (line 187)
        self_19645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'self')
        # Obtaining the member '_keys' of a type (line 187)
        _keys_19646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 16), self_19645, '_keys')
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 187)
        k_19647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 27), 'k')
        # Getting the type of 'self' (line 187)
        self_19648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'self')
        # Obtaining the member '_keys' of a type (line 187)
        _keys_19649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 16), self_19648, '_keys')
        # Obtaining the member '__getitem__' of a type (line 187)
        getitem___19650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 16), _keys_19649, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 187)
        subscript_call_result_19651 = invoke(stypy.reporting.localization.Localization(__file__, 187, 16), getitem___19650, k_19647)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 12), _keys_19646, subscript_call_result_19651)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_remove_keys(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_remove_keys' in the type store
        # Getting the type of 'stypy_return_type' (line 185)
        stypy_return_type_19652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19652)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_remove_keys'
        return stypy_return_type_19652


    @norecursion
    def update_keymap(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update_keymap'
        module_type_store = module_type_store.open_function_context('update_keymap', 189, 4, False)
        # Assigning a type to the variable 'self' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolManager.update_keymap.__dict__.__setitem__('stypy_localization', localization)
        ToolManager.update_keymap.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolManager.update_keymap.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolManager.update_keymap.__dict__.__setitem__('stypy_function_name', 'ToolManager.update_keymap')
        ToolManager.update_keymap.__dict__.__setitem__('stypy_param_names_list', ['name'])
        ToolManager.update_keymap.__dict__.__setitem__('stypy_varargs_param_name', 'keys')
        ToolManager.update_keymap.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolManager.update_keymap.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolManager.update_keymap.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolManager.update_keymap.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolManager.update_keymap.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManager.update_keymap', ['name'], 'keys', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update_keymap', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update_keymap(...)' code ##################

        unicode_19653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, (-1)), 'unicode', u'\n        Set the keymap to associate with the specified tool\n\n        Parameters\n        ----------\n        name : string\n            Name of the Tool\n        keys : keys to associate with the Tool\n        ')
        
        
        # Getting the type of 'name' (line 200)
        name_19654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 11), 'name')
        # Getting the type of 'self' (line 200)
        self_19655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 23), 'self')
        # Obtaining the member '_tools' of a type (line 200)
        _tools_19656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 23), self_19655, '_tools')
        # Applying the binary operator 'notin' (line 200)
        result_contains_19657 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 11), 'notin', name_19654, _tools_19656)
        
        # Testing the type of an if condition (line 200)
        if_condition_19658 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 8), result_contains_19657)
        # Assigning a type to the variable 'if_condition_19658' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'if_condition_19658', if_condition_19658)
        # SSA begins for if statement (line 200)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to KeyError(...): (line 201)
        # Processing the call arguments (line 201)
        unicode_19660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 27), 'unicode', u'%s not in Tools')
        # Getting the type of 'name' (line 201)
        name_19661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 47), 'name', False)
        # Applying the binary operator '%' (line 201)
        result_mod_19662 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 27), '%', unicode_19660, name_19661)
        
        # Processing the call keyword arguments (line 201)
        kwargs_19663 = {}
        # Getting the type of 'KeyError' (line 201)
        KeyError_19659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 18), 'KeyError', False)
        # Calling KeyError(args, kwargs) (line 201)
        KeyError_call_result_19664 = invoke(stypy.reporting.localization.Localization(__file__, 201, 18), KeyError_19659, *[result_mod_19662], **kwargs_19663)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 201, 12), KeyError_call_result_19664, 'raise parameter', BaseException)
        # SSA join for if statement (line 200)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _remove_keys(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'name' (line 203)
        name_19667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 26), 'name', False)
        # Processing the call keyword arguments (line 203)
        kwargs_19668 = {}
        # Getting the type of 'self' (line 203)
        self_19665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'self', False)
        # Obtaining the member '_remove_keys' of a type (line 203)
        _remove_keys_19666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), self_19665, '_remove_keys')
        # Calling _remove_keys(args, kwargs) (line 203)
        _remove_keys_call_result_19669 = invoke(stypy.reporting.localization.Localization(__file__, 203, 8), _remove_keys_19666, *[name_19667], **kwargs_19668)
        
        
        # Getting the type of 'keys' (line 205)
        keys_19670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 19), 'keys')
        # Testing the type of a for loop iterable (line 205)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 205, 8), keys_19670)
        # Getting the type of the for loop variable (line 205)
        for_loop_var_19671 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 205, 8), keys_19670)
        # Assigning a type to the variable 'key' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'key', for_loop_var_19671)
        # SSA begins for a for statement (line 205)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to validate_stringlist(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'key' (line 206)
        key_19673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 41), 'key', False)
        # Processing the call keyword arguments (line 206)
        kwargs_19674 = {}
        # Getting the type of 'validate_stringlist' (line 206)
        validate_stringlist_19672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 21), 'validate_stringlist', False)
        # Calling validate_stringlist(args, kwargs) (line 206)
        validate_stringlist_call_result_19675 = invoke(stypy.reporting.localization.Localization(__file__, 206, 21), validate_stringlist_19672, *[key_19673], **kwargs_19674)
        
        # Testing the type of a for loop iterable (line 206)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 206, 12), validate_stringlist_call_result_19675)
        # Getting the type of the for loop variable (line 206)
        for_loop_var_19676 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 206, 12), validate_stringlist_call_result_19675)
        # Assigning a type to the variable 'k' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'k', for_loop_var_19676)
        # SSA begins for a for statement (line 206)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'k' (line 207)
        k_19677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 19), 'k')
        # Getting the type of 'self' (line 207)
        self_19678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 24), 'self')
        # Obtaining the member '_keys' of a type (line 207)
        _keys_19679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 24), self_19678, '_keys')
        # Applying the binary operator 'in' (line 207)
        result_contains_19680 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 19), 'in', k_19677, _keys_19679)
        
        # Testing the type of an if condition (line 207)
        if_condition_19681 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 16), result_contains_19680)
        # Assigning a type to the variable 'if_condition_19681' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'if_condition_19681', if_condition_19681)
        # SSA begins for if statement (line 207)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 208)
        # Processing the call arguments (line 208)
        unicode_19684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 34), 'unicode', u'Key %s changed from %s to %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 209)
        tuple_19685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 209)
        # Adding element type (line 209)
        # Getting the type of 'k' (line 209)
        k_19686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 35), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 35), tuple_19685, k_19686)
        # Adding element type (line 209)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 209)
        k_19687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 49), 'k', False)
        # Getting the type of 'self' (line 209)
        self_19688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 38), 'self', False)
        # Obtaining the member '_keys' of a type (line 209)
        _keys_19689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 38), self_19688, '_keys')
        # Obtaining the member '__getitem__' of a type (line 209)
        getitem___19690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 38), _keys_19689, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 209)
        subscript_call_result_19691 = invoke(stypy.reporting.localization.Localization(__file__, 209, 38), getitem___19690, k_19687)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 35), tuple_19685, subscript_call_result_19691)
        # Adding element type (line 209)
        # Getting the type of 'name' (line 209)
        name_19692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 53), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 35), tuple_19685, name_19692)
        
        # Applying the binary operator '%' (line 208)
        result_mod_19693 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 34), '%', unicode_19684, tuple_19685)
        
        # Processing the call keyword arguments (line 208)
        kwargs_19694 = {}
        # Getting the type of 'warnings' (line 208)
        warnings_19682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 20), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 208)
        warn_19683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 20), warnings_19682, 'warn')
        # Calling warn(args, kwargs) (line 208)
        warn_call_result_19695 = invoke(stypy.reporting.localization.Localization(__file__, 208, 20), warn_19683, *[result_mod_19693], **kwargs_19694)
        
        # SSA join for if statement (line 207)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 210):
        # Getting the type of 'name' (line 210)
        name_19696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 32), 'name')
        # Getting the type of 'self' (line 210)
        self_19697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'self')
        # Obtaining the member '_keys' of a type (line 210)
        _keys_19698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 16), self_19697, '_keys')
        # Getting the type of 'k' (line 210)
        k_19699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 27), 'k')
        # Storing an element on a container (line 210)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 16), _keys_19698, (k_19699, name_19696))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'update_keymap(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_keymap' in the type store
        # Getting the type of 'stypy_return_type' (line 189)
        stypy_return_type_19700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19700)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_keymap'
        return stypy_return_type_19700


    @norecursion
    def remove_tool(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'remove_tool'
        module_type_store = module_type_store.open_function_context('remove_tool', 212, 4, False)
        # Assigning a type to the variable 'self' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolManager.remove_tool.__dict__.__setitem__('stypy_localization', localization)
        ToolManager.remove_tool.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolManager.remove_tool.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolManager.remove_tool.__dict__.__setitem__('stypy_function_name', 'ToolManager.remove_tool')
        ToolManager.remove_tool.__dict__.__setitem__('stypy_param_names_list', ['name'])
        ToolManager.remove_tool.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolManager.remove_tool.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolManager.remove_tool.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolManager.remove_tool.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolManager.remove_tool.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolManager.remove_tool.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManager.remove_tool', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'remove_tool', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'remove_tool(...)' code ##################

        unicode_19701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, (-1)), 'unicode', u'\n        Remove tool from `ToolManager`\n\n        Parameters\n        ----------\n        name : string\n            Name of the Tool\n        ')
        
        # Assigning a Call to a Name (line 222):
        
        # Call to get_tool(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'name' (line 222)
        name_19704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 29), 'name', False)
        # Processing the call keyword arguments (line 222)
        kwargs_19705 = {}
        # Getting the type of 'self' (line 222)
        self_19702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 15), 'self', False)
        # Obtaining the member 'get_tool' of a type (line 222)
        get_tool_19703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 15), self_19702, 'get_tool')
        # Calling get_tool(args, kwargs) (line 222)
        get_tool_call_result_19706 = invoke(stypy.reporting.localization.Localization(__file__, 222, 15), get_tool_19703, *[name_19704], **kwargs_19705)
        
        # Assigning a type to the variable 'tool' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'tool', get_tool_call_result_19706)
        
        # Call to destroy(...): (line 223)
        # Processing the call keyword arguments (line 223)
        kwargs_19709 = {}
        # Getting the type of 'tool' (line 223)
        tool_19707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tool', False)
        # Obtaining the member 'destroy' of a type (line 223)
        destroy_19708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), tool_19707, 'destroy')
        # Calling destroy(args, kwargs) (line 223)
        destroy_call_result_19710 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), destroy_19708, *[], **kwargs_19709)
        
        
        
        # Call to getattr(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'tool' (line 226)
        tool_19712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 19), 'tool', False)
        unicode_19713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 25), 'unicode', u'toggled')
        # Getting the type of 'False' (line 226)
        False_19714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 36), 'False', False)
        # Processing the call keyword arguments (line 226)
        kwargs_19715 = {}
        # Getting the type of 'getattr' (line 226)
        getattr_19711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 11), 'getattr', False)
        # Calling getattr(args, kwargs) (line 226)
        getattr_call_result_19716 = invoke(stypy.reporting.localization.Localization(__file__, 226, 11), getattr_19711, *[tool_19712, unicode_19713, False_19714], **kwargs_19715)
        
        # Testing the type of an if condition (line 226)
        if_condition_19717 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 8), getattr_call_result_19716)
        # Assigning a type to the variable 'if_condition_19717' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'if_condition_19717', if_condition_19717)
        # SSA begins for if statement (line 226)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to trigger_tool(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'tool' (line 227)
        tool_19720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 30), 'tool', False)
        unicode_19721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 36), 'unicode', u'toolmanager')
        # Processing the call keyword arguments (line 227)
        kwargs_19722 = {}
        # Getting the type of 'self' (line 227)
        self_19718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'self', False)
        # Obtaining the member 'trigger_tool' of a type (line 227)
        trigger_tool_19719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 12), self_19718, 'trigger_tool')
        # Calling trigger_tool(args, kwargs) (line 227)
        trigger_tool_call_result_19723 = invoke(stypy.reporting.localization.Localization(__file__, 227, 12), trigger_tool_19719, *[tool_19720, unicode_19721], **kwargs_19722)
        
        # SSA join for if statement (line 226)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _remove_keys(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'name' (line 229)
        name_19726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 26), 'name', False)
        # Processing the call keyword arguments (line 229)
        kwargs_19727 = {}
        # Getting the type of 'self' (line 229)
        self_19724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'self', False)
        # Obtaining the member '_remove_keys' of a type (line 229)
        _remove_keys_19725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), self_19724, '_remove_keys')
        # Calling _remove_keys(args, kwargs) (line 229)
        _remove_keys_call_result_19728 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), _remove_keys_19725, *[name_19726], **kwargs_19727)
        
        
        # Assigning a Str to a Name (line 231):
        unicode_19729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 12), 'unicode', u'tool_removed_event')
        # Assigning a type to the variable 's' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 's', unicode_19729)
        
        # Assigning a Call to a Name (line 232):
        
        # Call to ToolEvent(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 's' (line 232)
        s_19731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 26), 's', False)
        # Getting the type of 'self' (line 232)
        self_19732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 29), 'self', False)
        # Getting the type of 'tool' (line 232)
        tool_19733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 35), 'tool', False)
        # Processing the call keyword arguments (line 232)
        kwargs_19734 = {}
        # Getting the type of 'ToolEvent' (line 232)
        ToolEvent_19730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 16), 'ToolEvent', False)
        # Calling ToolEvent(args, kwargs) (line 232)
        ToolEvent_call_result_19735 = invoke(stypy.reporting.localization.Localization(__file__, 232, 16), ToolEvent_19730, *[s_19731, self_19732, tool_19733], **kwargs_19734)
        
        # Assigning a type to the variable 'event' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'event', ToolEvent_call_result_19735)
        
        # Call to process(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 's' (line 233)
        s_19739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 32), 's', False)
        # Getting the type of 'event' (line 233)
        event_19740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 35), 'event', False)
        # Processing the call keyword arguments (line 233)
        kwargs_19741 = {}
        # Getting the type of 'self' (line 233)
        self_19736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'self', False)
        # Obtaining the member '_callbacks' of a type (line 233)
        _callbacks_19737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 8), self_19736, '_callbacks')
        # Obtaining the member 'process' of a type (line 233)
        process_19738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 8), _callbacks_19737, 'process')
        # Calling process(args, kwargs) (line 233)
        process_call_result_19742 = invoke(stypy.reporting.localization.Localization(__file__, 233, 8), process_19738, *[s_19739, event_19740], **kwargs_19741)
        
        # Deleting a member
        # Getting the type of 'self' (line 235)
        self_19743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'self')
        # Obtaining the member '_tools' of a type (line 235)
        _tools_19744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), self_19743, '_tools')
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 235)
        name_19745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 24), 'name')
        # Getting the type of 'self' (line 235)
        self_19746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'self')
        # Obtaining the member '_tools' of a type (line 235)
        _tools_19747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), self_19746, '_tools')
        # Obtaining the member '__getitem__' of a type (line 235)
        getitem___19748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), _tools_19747, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 235)
        subscript_call_result_19749 = invoke(stypy.reporting.localization.Localization(__file__, 235, 12), getitem___19748, name_19745)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 8), _tools_19744, subscript_call_result_19749)
        
        # ################# End of 'remove_tool(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'remove_tool' in the type store
        # Getting the type of 'stypy_return_type' (line 212)
        stypy_return_type_19750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19750)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'remove_tool'
        return stypy_return_type_19750


    @norecursion
    def add_tool(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_tool'
        module_type_store = module_type_store.open_function_context('add_tool', 237, 4, False)
        # Assigning a type to the variable 'self' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolManager.add_tool.__dict__.__setitem__('stypy_localization', localization)
        ToolManager.add_tool.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolManager.add_tool.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolManager.add_tool.__dict__.__setitem__('stypy_function_name', 'ToolManager.add_tool')
        ToolManager.add_tool.__dict__.__setitem__('stypy_param_names_list', ['name', 'tool'])
        ToolManager.add_tool.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        ToolManager.add_tool.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        ToolManager.add_tool.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolManager.add_tool.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolManager.add_tool.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolManager.add_tool.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManager.add_tool', ['name', 'tool'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_tool', localization, ['name', 'tool'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_tool(...)' code ##################

        unicode_19751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, (-1)), 'unicode', u'\n        Add *tool* to `ToolManager`\n\n        If successful adds a new event `tool_trigger_name` where **name** is\n        the **name** of the tool, this event is fired everytime\n        the tool is triggered.\n\n        Parameters\n        ----------\n        name : str\n            Name of the tool, treated as the ID, has to be unique\n        tool : class_like, i.e. str or type\n            Reference to find the class of the Tool to added.\n\n        Notes\n        -----\n        args and kwargs get passed directly to the tools constructor.\n\n        See Also\n        --------\n        matplotlib.backend_tools.ToolBase : The base class for tools.\n        ')
        
        # Assigning a Call to a Name (line 261):
        
        # Call to _get_cls_to_instantiate(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'tool' (line 261)
        tool_19754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 48), 'tool', False)
        # Processing the call keyword arguments (line 261)
        kwargs_19755 = {}
        # Getting the type of 'self' (line 261)
        self_19752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 19), 'self', False)
        # Obtaining the member '_get_cls_to_instantiate' of a type (line 261)
        _get_cls_to_instantiate_19753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 19), self_19752, '_get_cls_to_instantiate')
        # Calling _get_cls_to_instantiate(args, kwargs) (line 261)
        _get_cls_to_instantiate_call_result_19756 = invoke(stypy.reporting.localization.Localization(__file__, 261, 19), _get_cls_to_instantiate_19753, *[tool_19754], **kwargs_19755)
        
        # Assigning a type to the variable 'tool_cls' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'tool_cls', _get_cls_to_instantiate_call_result_19756)
        
        
        # Getting the type of 'tool_cls' (line 262)
        tool_cls_19757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 15), 'tool_cls')
        # Applying the 'not' unary operator (line 262)
        result_not__19758 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 11), 'not', tool_cls_19757)
        
        # Testing the type of an if condition (line 262)
        if_condition_19759 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 262, 8), result_not__19758)
        # Assigning a type to the variable 'if_condition_19759' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'if_condition_19759', if_condition_19759)
        # SSA begins for if statement (line 262)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 263)
        # Processing the call arguments (line 263)
        unicode_19761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 29), 'unicode', u'Impossible to find class for %s')
        
        # Call to str(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'tool' (line 263)
        tool_19763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 69), 'tool', False)
        # Processing the call keyword arguments (line 263)
        kwargs_19764 = {}
        # Getting the type of 'str' (line 263)
        str_19762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 65), 'str', False)
        # Calling str(args, kwargs) (line 263)
        str_call_result_19765 = invoke(stypy.reporting.localization.Localization(__file__, 263, 65), str_19762, *[tool_19763], **kwargs_19764)
        
        # Applying the binary operator '%' (line 263)
        result_mod_19766 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 29), '%', unicode_19761, str_call_result_19765)
        
        # Processing the call keyword arguments (line 263)
        kwargs_19767 = {}
        # Getting the type of 'ValueError' (line 263)
        ValueError_19760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 263)
        ValueError_call_result_19768 = invoke(stypy.reporting.localization.Localization(__file__, 263, 18), ValueError_19760, *[result_mod_19766], **kwargs_19767)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 263, 12), ValueError_call_result_19768, 'raise parameter', BaseException)
        # SSA join for if statement (line 262)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'name' (line 265)
        name_19769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 11), 'name')
        # Getting the type of 'self' (line 265)
        self_19770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 19), 'self')
        # Obtaining the member '_tools' of a type (line 265)
        _tools_19771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 19), self_19770, '_tools')
        # Applying the binary operator 'in' (line 265)
        result_contains_19772 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 11), 'in', name_19769, _tools_19771)
        
        # Testing the type of an if condition (line 265)
        if_condition_19773 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 8), result_contains_19772)
        # Assigning a type to the variable 'if_condition_19773' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'if_condition_19773', if_condition_19773)
        # SSA begins for if statement (line 265)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 266)
        # Processing the call arguments (line 266)
        unicode_19776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 26), 'unicode', u'A "Tool class" with the same name already exists, not added')
        # Processing the call keyword arguments (line 266)
        kwargs_19777 = {}
        # Getting the type of 'warnings' (line 266)
        warnings_19774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 266)
        warn_19775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 12), warnings_19774, 'warn')
        # Calling warn(args, kwargs) (line 266)
        warn_call_result_19778 = invoke(stypy.reporting.localization.Localization(__file__, 266, 12), warn_19775, *[unicode_19776], **kwargs_19777)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 268)
        name_19779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 31), 'name')
        # Getting the type of 'self' (line 268)
        self_19780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 19), 'self')
        # Obtaining the member '_tools' of a type (line 268)
        _tools_19781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 19), self_19780, '_tools')
        # Obtaining the member '__getitem__' of a type (line 268)
        getitem___19782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 19), _tools_19781, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 268)
        subscript_call_result_19783 = invoke(stypy.reporting.localization.Localization(__file__, 268, 19), getitem___19782, name_19779)
        
        # Assigning a type to the variable 'stypy_return_type' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'stypy_return_type', subscript_call_result_19783)
        # SSA join for if statement (line 265)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 270):
        
        # Call to tool_cls(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'self' (line 270)
        self_19785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 28), 'self', False)
        # Getting the type of 'name' (line 270)
        name_19786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 34), 'name', False)
        # Getting the type of 'args' (line 270)
        args_19787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 41), 'args', False)
        # Processing the call keyword arguments (line 270)
        # Getting the type of 'kwargs' (line 270)
        kwargs_19788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 49), 'kwargs', False)
        kwargs_19789 = {'kwargs_19788': kwargs_19788}
        # Getting the type of 'tool_cls' (line 270)
        tool_cls_19784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 19), 'tool_cls', False)
        # Calling tool_cls(args, kwargs) (line 270)
        tool_cls_call_result_19790 = invoke(stypy.reporting.localization.Localization(__file__, 270, 19), tool_cls_19784, *[self_19785, name_19786, args_19787], **kwargs_19789)
        
        # Assigning a type to the variable 'tool_obj' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'tool_obj', tool_cls_call_result_19790)
        
        # Assigning a Name to a Subscript (line 271):
        # Getting the type of 'tool_obj' (line 271)
        tool_obj_19791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 28), 'tool_obj')
        # Getting the type of 'self' (line 271)
        self_19792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'self')
        # Obtaining the member '_tools' of a type (line 271)
        _tools_19793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 8), self_19792, '_tools')
        # Getting the type of 'name' (line 271)
        name_19794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 20), 'name')
        # Storing an element on a container (line 271)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 8), _tools_19793, (name_19794, tool_obj_19791))
        
        
        # Getting the type of 'tool_cls' (line 273)
        tool_cls_19795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 11), 'tool_cls')
        # Obtaining the member 'default_keymap' of a type (line 273)
        default_keymap_19796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 11), tool_cls_19795, 'default_keymap')
        # Getting the type of 'None' (line 273)
        None_19797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 42), 'None')
        # Applying the binary operator 'isnot' (line 273)
        result_is_not_19798 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 11), 'isnot', default_keymap_19796, None_19797)
        
        # Testing the type of an if condition (line 273)
        if_condition_19799 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 8), result_is_not_19798)
        # Assigning a type to the variable 'if_condition_19799' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'if_condition_19799', if_condition_19799)
        # SSA begins for if statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to update_keymap(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'name' (line 274)
        name_19802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 31), 'name', False)
        # Getting the type of 'tool_cls' (line 274)
        tool_cls_19803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 37), 'tool_cls', False)
        # Obtaining the member 'default_keymap' of a type (line 274)
        default_keymap_19804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 37), tool_cls_19803, 'default_keymap')
        # Processing the call keyword arguments (line 274)
        kwargs_19805 = {}
        # Getting the type of 'self' (line 274)
        self_19800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'self', False)
        # Obtaining the member 'update_keymap' of a type (line 274)
        update_keymap_19801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 12), self_19800, 'update_keymap')
        # Calling update_keymap(args, kwargs) (line 274)
        update_keymap_call_result_19806 = invoke(stypy.reporting.localization.Localization(__file__, 274, 12), update_keymap_19801, *[name_19802, default_keymap_19804], **kwargs_19805)
        
        # SSA join for if statement (line 273)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to isinstance(...): (line 277)
        # Processing the call arguments (line 277)
        # Getting the type of 'tool_obj' (line 277)
        tool_obj_19808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 22), 'tool_obj', False)
        # Getting the type of 'tools' (line 277)
        tools_19809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 32), 'tools', False)
        # Obtaining the member 'ToolToggleBase' of a type (line 277)
        ToolToggleBase_19810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 32), tools_19809, 'ToolToggleBase')
        # Processing the call keyword arguments (line 277)
        kwargs_19811 = {}
        # Getting the type of 'isinstance' (line 277)
        isinstance_19807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 277)
        isinstance_call_result_19812 = invoke(stypy.reporting.localization.Localization(__file__, 277, 11), isinstance_19807, *[tool_obj_19808, ToolToggleBase_19810], **kwargs_19811)
        
        # Testing the type of an if condition (line 277)
        if_condition_19813 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 277, 8), isinstance_call_result_19812)
        # Assigning a type to the variable 'if_condition_19813' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'if_condition_19813', if_condition_19813)
        # SSA begins for if statement (line 277)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 280)
        # Getting the type of 'tool_obj' (line 280)
        tool_obj_19814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 15), 'tool_obj')
        # Obtaining the member 'radio_group' of a type (line 280)
        radio_group_19815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 15), tool_obj_19814, 'radio_group')
        # Getting the type of 'None' (line 280)
        None_19816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 39), 'None')
        
        (may_be_19817, more_types_in_union_19818) = may_be_none(radio_group_19815, None_19816)

        if may_be_19817:

            if more_types_in_union_19818:
                # Runtime conditional SSA (line 280)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to setdefault(...): (line 281)
            # Processing the call arguments (line 281)
            # Getting the type of 'None' (line 281)
            None_19822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 41), 'None', False)
            
            # Call to set(...): (line 281)
            # Processing the call keyword arguments (line 281)
            kwargs_19824 = {}
            # Getting the type of 'set' (line 281)
            set_19823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 47), 'set', False)
            # Calling set(args, kwargs) (line 281)
            set_call_result_19825 = invoke(stypy.reporting.localization.Localization(__file__, 281, 47), set_19823, *[], **kwargs_19824)
            
            # Processing the call keyword arguments (line 281)
            kwargs_19826 = {}
            # Getting the type of 'self' (line 281)
            self_19819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'self', False)
            # Obtaining the member '_toggled' of a type (line 281)
            _toggled_19820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 16), self_19819, '_toggled')
            # Obtaining the member 'setdefault' of a type (line 281)
            setdefault_19821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 16), _toggled_19820, 'setdefault')
            # Calling setdefault(args, kwargs) (line 281)
            setdefault_call_result_19827 = invoke(stypy.reporting.localization.Localization(__file__, 281, 16), setdefault_19821, *[None_19822, set_call_result_19825], **kwargs_19826)
            

            if more_types_in_union_19818:
                # Runtime conditional SSA for else branch (line 280)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_19817) or more_types_in_union_19818):
            
            # Call to setdefault(...): (line 283)
            # Processing the call arguments (line 283)
            # Getting the type of 'tool_obj' (line 283)
            tool_obj_19831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 41), 'tool_obj', False)
            # Obtaining the member 'radio_group' of a type (line 283)
            radio_group_19832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 41), tool_obj_19831, 'radio_group')
            # Getting the type of 'None' (line 283)
            None_19833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 63), 'None', False)
            # Processing the call keyword arguments (line 283)
            kwargs_19834 = {}
            # Getting the type of 'self' (line 283)
            self_19828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'self', False)
            # Obtaining the member '_toggled' of a type (line 283)
            _toggled_19829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 16), self_19828, '_toggled')
            # Obtaining the member 'setdefault' of a type (line 283)
            setdefault_19830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 16), _toggled_19829, 'setdefault')
            # Calling setdefault(args, kwargs) (line 283)
            setdefault_call_result_19835 = invoke(stypy.reporting.localization.Localization(__file__, 283, 16), setdefault_19830, *[radio_group_19832, None_19833], **kwargs_19834)
            

            if (may_be_19817 and more_types_in_union_19818):
                # SSA join for if statement (line 280)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'tool_obj' (line 286)
        tool_obj_19836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 15), 'tool_obj')
        # Obtaining the member 'toggled' of a type (line 286)
        toggled_19837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 15), tool_obj_19836, 'toggled')
        # Testing the type of an if condition (line 286)
        if_condition_19838 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 286, 12), toggled_19837)
        # Assigning a type to the variable 'if_condition_19838' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'if_condition_19838', if_condition_19838)
        # SSA begins for if statement (line 286)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _handle_toggle(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'tool_obj' (line 287)
        tool_obj_19841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 36), 'tool_obj', False)
        # Getting the type of 'None' (line 287)
        None_19842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 46), 'None', False)
        # Getting the type of 'None' (line 287)
        None_19843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 52), 'None', False)
        # Getting the type of 'None' (line 287)
        None_19844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 58), 'None', False)
        # Processing the call keyword arguments (line 287)
        kwargs_19845 = {}
        # Getting the type of 'self' (line 287)
        self_19839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 16), 'self', False)
        # Obtaining the member '_handle_toggle' of a type (line 287)
        _handle_toggle_19840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 16), self_19839, '_handle_toggle')
        # Calling _handle_toggle(args, kwargs) (line 287)
        _handle_toggle_call_result_19846 = invoke(stypy.reporting.localization.Localization(__file__, 287, 16), _handle_toggle_19840, *[tool_obj_19841, None_19842, None_19843, None_19844], **kwargs_19845)
        
        # SSA join for if statement (line 286)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 277)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_figure(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 'self' (line 288)
        self_19849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 28), 'self', False)
        # Obtaining the member 'figure' of a type (line 288)
        figure_19850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 28), self_19849, 'figure')
        # Processing the call keyword arguments (line 288)
        kwargs_19851 = {}
        # Getting the type of 'tool_obj' (line 288)
        tool_obj_19847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'tool_obj', False)
        # Obtaining the member 'set_figure' of a type (line 288)
        set_figure_19848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 8), tool_obj_19847, 'set_figure')
        # Calling set_figure(args, kwargs) (line 288)
        set_figure_call_result_19852 = invoke(stypy.reporting.localization.Localization(__file__, 288, 8), set_figure_19848, *[figure_19850], **kwargs_19851)
        
        
        # Call to _tool_added_event(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'tool_obj' (line 290)
        tool_obj_19855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 31), 'tool_obj', False)
        # Processing the call keyword arguments (line 290)
        kwargs_19856 = {}
        # Getting the type of 'self' (line 290)
        self_19853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'self', False)
        # Obtaining the member '_tool_added_event' of a type (line 290)
        _tool_added_event_19854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), self_19853, '_tool_added_event')
        # Calling _tool_added_event(args, kwargs) (line 290)
        _tool_added_event_call_result_19857 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), _tool_added_event_19854, *[tool_obj_19855], **kwargs_19856)
        
        # Getting the type of 'tool_obj' (line 291)
        tool_obj_19858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 15), 'tool_obj')
        # Assigning a type to the variable 'stypy_return_type' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'stypy_return_type', tool_obj_19858)
        
        # ################# End of 'add_tool(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_tool' in the type store
        # Getting the type of 'stypy_return_type' (line 237)
        stypy_return_type_19859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19859)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_tool'
        return stypy_return_type_19859


    @norecursion
    def _tool_added_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_tool_added_event'
        module_type_store = module_type_store.open_function_context('_tool_added_event', 293, 4, False)
        # Assigning a type to the variable 'self' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolManager._tool_added_event.__dict__.__setitem__('stypy_localization', localization)
        ToolManager._tool_added_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolManager._tool_added_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolManager._tool_added_event.__dict__.__setitem__('stypy_function_name', 'ToolManager._tool_added_event')
        ToolManager._tool_added_event.__dict__.__setitem__('stypy_param_names_list', ['tool'])
        ToolManager._tool_added_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolManager._tool_added_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolManager._tool_added_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolManager._tool_added_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolManager._tool_added_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolManager._tool_added_event.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManager._tool_added_event', ['tool'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_tool_added_event', localization, ['tool'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_tool_added_event(...)' code ##################

        
        # Assigning a Str to a Name (line 294):
        unicode_19860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 12), 'unicode', u'tool_added_event')
        # Assigning a type to the variable 's' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 's', unicode_19860)
        
        # Assigning a Call to a Name (line 295):
        
        # Call to ToolEvent(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 's' (line 295)
        s_19862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 26), 's', False)
        # Getting the type of 'self' (line 295)
        self_19863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 29), 'self', False)
        # Getting the type of 'tool' (line 295)
        tool_19864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 35), 'tool', False)
        # Processing the call keyword arguments (line 295)
        kwargs_19865 = {}
        # Getting the type of 'ToolEvent' (line 295)
        ToolEvent_19861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'ToolEvent', False)
        # Calling ToolEvent(args, kwargs) (line 295)
        ToolEvent_call_result_19866 = invoke(stypy.reporting.localization.Localization(__file__, 295, 16), ToolEvent_19861, *[s_19862, self_19863, tool_19864], **kwargs_19865)
        
        # Assigning a type to the variable 'event' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'event', ToolEvent_call_result_19866)
        
        # Call to process(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 's' (line 296)
        s_19870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 32), 's', False)
        # Getting the type of 'event' (line 296)
        event_19871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 35), 'event', False)
        # Processing the call keyword arguments (line 296)
        kwargs_19872 = {}
        # Getting the type of 'self' (line 296)
        self_19867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'self', False)
        # Obtaining the member '_callbacks' of a type (line 296)
        _callbacks_19868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 8), self_19867, '_callbacks')
        # Obtaining the member 'process' of a type (line 296)
        process_19869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 8), _callbacks_19868, 'process')
        # Calling process(args, kwargs) (line 296)
        process_call_result_19873 = invoke(stypy.reporting.localization.Localization(__file__, 296, 8), process_19869, *[s_19870, event_19871], **kwargs_19872)
        
        
        # ################# End of '_tool_added_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_tool_added_event' in the type store
        # Getting the type of 'stypy_return_type' (line 293)
        stypy_return_type_19874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19874)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_tool_added_event'
        return stypy_return_type_19874


    @norecursion
    def _handle_toggle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_handle_toggle'
        module_type_store = module_type_store.open_function_context('_handle_toggle', 298, 4, False)
        # Assigning a type to the variable 'self' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolManager._handle_toggle.__dict__.__setitem__('stypy_localization', localization)
        ToolManager._handle_toggle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolManager._handle_toggle.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolManager._handle_toggle.__dict__.__setitem__('stypy_function_name', 'ToolManager._handle_toggle')
        ToolManager._handle_toggle.__dict__.__setitem__('stypy_param_names_list', ['tool', 'sender', 'canvasevent', 'data'])
        ToolManager._handle_toggle.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolManager._handle_toggle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolManager._handle_toggle.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolManager._handle_toggle.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolManager._handle_toggle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolManager._handle_toggle.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManager._handle_toggle', ['tool', 'sender', 'canvasevent', 'data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_handle_toggle', localization, ['tool', 'sender', 'canvasevent', 'data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_handle_toggle(...)' code ##################

        unicode_19875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, (-1)), 'unicode', u'\n        Toggle tools, need to untoggle prior to using other Toggle tool\n        Called from trigger_tool\n\n        Parameters\n        ----------\n        tool: Tool object\n        sender: object\n            Object that wishes to trigger the tool\n        canvasevent : Event\n            Original Canvas event or None\n        data : Object\n            Extra data to pass to the tool when triggering\n        ')
        
        # Assigning a Attribute to a Name (line 314):
        # Getting the type of 'tool' (line 314)
        tool_19876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 22), 'tool')
        # Obtaining the member 'radio_group' of a type (line 314)
        radio_group_19877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 22), tool_19876, 'radio_group')
        # Assigning a type to the variable 'radio_group' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'radio_group', radio_group_19877)
        
        # Type idiom detected: calculating its left and rigth part (line 317)
        # Getting the type of 'radio_group' (line 317)
        radio_group_19878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 11), 'radio_group')
        # Getting the type of 'None' (line 317)
        None_19879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 26), 'None')
        
        (may_be_19880, more_types_in_union_19881) = may_be_none(radio_group_19878, None_19879)

        if may_be_19880:

            if more_types_in_union_19881:
                # Runtime conditional SSA (line 317)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Getting the type of 'tool' (line 318)
            tool_19882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 15), 'tool')
            # Obtaining the member 'name' of a type (line 318)
            name_19883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 15), tool_19882, 'name')
            
            # Obtaining the type of the subscript
            # Getting the type of 'None' (line 318)
            None_19884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 42), 'None')
            # Getting the type of 'self' (line 318)
            self_19885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 28), 'self')
            # Obtaining the member '_toggled' of a type (line 318)
            _toggled_19886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 28), self_19885, '_toggled')
            # Obtaining the member '__getitem__' of a type (line 318)
            getitem___19887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 28), _toggled_19886, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 318)
            subscript_call_result_19888 = invoke(stypy.reporting.localization.Localization(__file__, 318, 28), getitem___19887, None_19884)
            
            # Applying the binary operator 'in' (line 318)
            result_contains_19889 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 15), 'in', name_19883, subscript_call_result_19888)
            
            # Testing the type of an if condition (line 318)
            if_condition_19890 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 318, 12), result_contains_19889)
            # Assigning a type to the variable 'if_condition_19890' (line 318)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'if_condition_19890', if_condition_19890)
            # SSA begins for if statement (line 318)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to remove(...): (line 319)
            # Processing the call arguments (line 319)
            # Getting the type of 'tool' (line 319)
            tool_19897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 43), 'tool', False)
            # Obtaining the member 'name' of a type (line 319)
            name_19898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 43), tool_19897, 'name')
            # Processing the call keyword arguments (line 319)
            kwargs_19899 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'None' (line 319)
            None_19891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 30), 'None', False)
            # Getting the type of 'self' (line 319)
            self_19892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 16), 'self', False)
            # Obtaining the member '_toggled' of a type (line 319)
            _toggled_19893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 16), self_19892, '_toggled')
            # Obtaining the member '__getitem__' of a type (line 319)
            getitem___19894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 16), _toggled_19893, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 319)
            subscript_call_result_19895 = invoke(stypy.reporting.localization.Localization(__file__, 319, 16), getitem___19894, None_19891)
            
            # Obtaining the member 'remove' of a type (line 319)
            remove_19896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 16), subscript_call_result_19895, 'remove')
            # Calling remove(args, kwargs) (line 319)
            remove_call_result_19900 = invoke(stypy.reporting.localization.Localization(__file__, 319, 16), remove_19896, *[name_19898], **kwargs_19899)
            
            # SSA branch for the else part of an if statement (line 318)
            module_type_store.open_ssa_branch('else')
            
            # Call to add(...): (line 321)
            # Processing the call arguments (line 321)
            # Getting the type of 'tool' (line 321)
            tool_19907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 40), 'tool', False)
            # Obtaining the member 'name' of a type (line 321)
            name_19908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 40), tool_19907, 'name')
            # Processing the call keyword arguments (line 321)
            kwargs_19909 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'None' (line 321)
            None_19901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 30), 'None', False)
            # Getting the type of 'self' (line 321)
            self_19902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 16), 'self', False)
            # Obtaining the member '_toggled' of a type (line 321)
            _toggled_19903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 16), self_19902, '_toggled')
            # Obtaining the member '__getitem__' of a type (line 321)
            getitem___19904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 16), _toggled_19903, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 321)
            subscript_call_result_19905 = invoke(stypy.reporting.localization.Localization(__file__, 321, 16), getitem___19904, None_19901)
            
            # Obtaining the member 'add' of a type (line 321)
            add_19906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 16), subscript_call_result_19905, 'add')
            # Calling add(args, kwargs) (line 321)
            add_call_result_19910 = invoke(stypy.reporting.localization.Localization(__file__, 321, 16), add_19906, *[name_19908], **kwargs_19909)
            
            # SSA join for if statement (line 318)
            module_type_store = module_type_store.join_ssa_context()
            
            # Assigning a type to the variable 'stypy_return_type' (line 322)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_19881:
                # SSA join for if statement (line 317)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'radio_group' (line 325)
        radio_group_19911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 25), 'radio_group')
        # Getting the type of 'self' (line 325)
        self_19912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 11), 'self')
        # Obtaining the member '_toggled' of a type (line 325)
        _toggled_19913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 11), self_19912, '_toggled')
        # Obtaining the member '__getitem__' of a type (line 325)
        getitem___19914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 11), _toggled_19913, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 325)
        subscript_call_result_19915 = invoke(stypy.reporting.localization.Localization(__file__, 325, 11), getitem___19914, radio_group_19911)
        
        # Getting the type of 'tool' (line 325)
        tool_19916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 41), 'tool')
        # Obtaining the member 'name' of a type (line 325)
        name_19917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 41), tool_19916, 'name')
        # Applying the binary operator '==' (line 325)
        result_eq_19918 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 11), '==', subscript_call_result_19915, name_19917)
        
        # Testing the type of an if condition (line 325)
        if_condition_19919 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 325, 8), result_eq_19918)
        # Assigning a type to the variable 'if_condition_19919' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'if_condition_19919', if_condition_19919)
        # SSA begins for if statement (line 325)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 326):
        # Getting the type of 'None' (line 326)
        None_19920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 22), 'None')
        # Assigning a type to the variable 'toggled' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'toggled', None_19920)
        # SSA branch for the else part of an if statement (line 325)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 329)
        
        # Obtaining the type of the subscript
        # Getting the type of 'radio_group' (line 329)
        radio_group_19921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 27), 'radio_group')
        # Getting the type of 'self' (line 329)
        self_19922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 13), 'self')
        # Obtaining the member '_toggled' of a type (line 329)
        _toggled_19923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 13), self_19922, '_toggled')
        # Obtaining the member '__getitem__' of a type (line 329)
        getitem___19924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 13), _toggled_19923, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 329)
        subscript_call_result_19925 = invoke(stypy.reporting.localization.Localization(__file__, 329, 13), getitem___19924, radio_group_19921)
        
        # Getting the type of 'None' (line 329)
        None_19926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 43), 'None')
        
        (may_be_19927, more_types_in_union_19928) = may_be_none(subscript_call_result_19925, None_19926)

        if may_be_19927:

            if more_types_in_union_19928:
                # Runtime conditional SSA (line 329)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 330):
            # Getting the type of 'tool' (line 330)
            tool_19929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 22), 'tool')
            # Obtaining the member 'name' of a type (line 330)
            name_19930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 22), tool_19929, 'name')
            # Assigning a type to the variable 'toggled' (line 330)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'toggled', name_19930)

            if more_types_in_union_19928:
                # Runtime conditional SSA for else branch (line 329)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_19927) or more_types_in_union_19928):
            
            # Call to trigger_tool(...): (line 334)
            # Processing the call arguments (line 334)
            
            # Obtaining the type of the subscript
            # Getting the type of 'radio_group' (line 334)
            radio_group_19933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 44), 'radio_group', False)
            # Getting the type of 'self' (line 334)
            self_19934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 30), 'self', False)
            # Obtaining the member '_toggled' of a type (line 334)
            _toggled_19935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 30), self_19934, '_toggled')
            # Obtaining the member '__getitem__' of a type (line 334)
            getitem___19936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 30), _toggled_19935, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 334)
            subscript_call_result_19937 = invoke(stypy.reporting.localization.Localization(__file__, 334, 30), getitem___19936, radio_group_19933)
            
            # Getting the type of 'self' (line 335)
            self_19938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 30), 'self', False)
            # Getting the type of 'canvasevent' (line 336)
            canvasevent_19939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 30), 'canvasevent', False)
            # Getting the type of 'data' (line 337)
            data_19940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 30), 'data', False)
            # Processing the call keyword arguments (line 334)
            kwargs_19941 = {}
            # Getting the type of 'self' (line 334)
            self_19931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'self', False)
            # Obtaining the member 'trigger_tool' of a type (line 334)
            trigger_tool_19932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 12), self_19931, 'trigger_tool')
            # Calling trigger_tool(args, kwargs) (line 334)
            trigger_tool_call_result_19942 = invoke(stypy.reporting.localization.Localization(__file__, 334, 12), trigger_tool_19932, *[subscript_call_result_19937, self_19938, canvasevent_19939, data_19940], **kwargs_19941)
            
            
            # Assigning a Attribute to a Name (line 338):
            # Getting the type of 'tool' (line 338)
            tool_19943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 22), 'tool')
            # Obtaining the member 'name' of a type (line 338)
            name_19944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 22), tool_19943, 'name')
            # Assigning a type to the variable 'toggled' (line 338)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'toggled', name_19944)

            if (may_be_19927 and more_types_in_union_19928):
                # SSA join for if statement (line 329)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 325)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 341):
        # Getting the type of 'toggled' (line 341)
        toggled_19945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 37), 'toggled')
        # Getting the type of 'self' (line 341)
        self_19946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'self')
        # Obtaining the member '_toggled' of a type (line 341)
        _toggled_19947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 8), self_19946, '_toggled')
        # Getting the type of 'radio_group' (line 341)
        radio_group_19948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 22), 'radio_group')
        # Storing an element on a container (line 341)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 8), _toggled_19947, (radio_group_19948, toggled_19945))
        
        # ################# End of '_handle_toggle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_handle_toggle' in the type store
        # Getting the type of 'stypy_return_type' (line 298)
        stypy_return_type_19949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19949)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_handle_toggle'
        return stypy_return_type_19949


    @norecursion
    def _get_cls_to_instantiate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_cls_to_instantiate'
        module_type_store = module_type_store.open_function_context('_get_cls_to_instantiate', 343, 4, False)
        # Assigning a type to the variable 'self' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolManager._get_cls_to_instantiate.__dict__.__setitem__('stypy_localization', localization)
        ToolManager._get_cls_to_instantiate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolManager._get_cls_to_instantiate.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolManager._get_cls_to_instantiate.__dict__.__setitem__('stypy_function_name', 'ToolManager._get_cls_to_instantiate')
        ToolManager._get_cls_to_instantiate.__dict__.__setitem__('stypy_param_names_list', ['callback_class'])
        ToolManager._get_cls_to_instantiate.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolManager._get_cls_to_instantiate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolManager._get_cls_to_instantiate.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolManager._get_cls_to_instantiate.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolManager._get_cls_to_instantiate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolManager._get_cls_to_instantiate.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManager._get_cls_to_instantiate', ['callback_class'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_cls_to_instantiate', localization, ['callback_class'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_cls_to_instantiate(...)' code ##################

        
        
        # Call to isinstance(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'callback_class' (line 345)
        callback_class_19951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 22), 'callback_class', False)
        # Getting the type of 'six' (line 345)
        six_19952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 38), 'six', False)
        # Obtaining the member 'string_types' of a type (line 345)
        string_types_19953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 38), six_19952, 'string_types')
        # Processing the call keyword arguments (line 345)
        kwargs_19954 = {}
        # Getting the type of 'isinstance' (line 345)
        isinstance_19950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 345)
        isinstance_call_result_19955 = invoke(stypy.reporting.localization.Localization(__file__, 345, 11), isinstance_19950, *[callback_class_19951, string_types_19953], **kwargs_19954)
        
        # Testing the type of an if condition (line 345)
        if_condition_19956 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 345, 8), isinstance_call_result_19955)
        # Assigning a type to the variable 'if_condition_19956' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'if_condition_19956', if_condition_19956)
        # SSA begins for if statement (line 345)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'callback_class' (line 347)
        callback_class_19957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 15), 'callback_class')
        
        # Call to globals(...): (line 347)
        # Processing the call keyword arguments (line 347)
        kwargs_19959 = {}
        # Getting the type of 'globals' (line 347)
        globals_19958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 33), 'globals', False)
        # Calling globals(args, kwargs) (line 347)
        globals_call_result_19960 = invoke(stypy.reporting.localization.Localization(__file__, 347, 33), globals_19958, *[], **kwargs_19959)
        
        # Applying the binary operator 'in' (line 347)
        result_contains_19961 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 15), 'in', callback_class_19957, globals_call_result_19960)
        
        # Testing the type of an if condition (line 347)
        if_condition_19962 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 347, 12), result_contains_19961)
        # Assigning a type to the variable 'if_condition_19962' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'if_condition_19962', if_condition_19962)
        # SSA begins for if statement (line 347)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 348):
        
        # Obtaining the type of the subscript
        # Getting the type of 'callback_class' (line 348)
        callback_class_19963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 43), 'callback_class')
        
        # Call to globals(...): (line 348)
        # Processing the call keyword arguments (line 348)
        kwargs_19965 = {}
        # Getting the type of 'globals' (line 348)
        globals_19964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 33), 'globals', False)
        # Calling globals(args, kwargs) (line 348)
        globals_call_result_19966 = invoke(stypy.reporting.localization.Localization(__file__, 348, 33), globals_19964, *[], **kwargs_19965)
        
        # Obtaining the member '__getitem__' of a type (line 348)
        getitem___19967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 33), globals_call_result_19966, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 348)
        subscript_call_result_19968 = invoke(stypy.reporting.localization.Localization(__file__, 348, 33), getitem___19967, callback_class_19963)
        
        # Assigning a type to the variable 'callback_class' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 16), 'callback_class', subscript_call_result_19968)
        # SSA branch for the else part of an if statement (line 347)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 350):
        unicode_19969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 22), 'unicode', u'backend_tools')
        # Assigning a type to the variable 'mod' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 16), 'mod', unicode_19969)
        
        # Assigning a Call to a Name (line 351):
        
        # Call to __import__(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'mod' (line 351)
        mod_19971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 44), 'mod', False)
        
        # Call to globals(...): (line 352)
        # Processing the call keyword arguments (line 352)
        kwargs_19973 = {}
        # Getting the type of 'globals' (line 352)
        globals_19972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 44), 'globals', False)
        # Calling globals(args, kwargs) (line 352)
        globals_call_result_19974 = invoke(stypy.reporting.localization.Localization(__file__, 352, 44), globals_19972, *[], **kwargs_19973)
        
        
        # Call to locals(...): (line 352)
        # Processing the call keyword arguments (line 352)
        kwargs_19976 = {}
        # Getting the type of 'locals' (line 352)
        locals_19975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 55), 'locals', False)
        # Calling locals(args, kwargs) (line 352)
        locals_call_result_19977 = invoke(stypy.reporting.localization.Localization(__file__, 352, 55), locals_19975, *[], **kwargs_19976)
        
        
        # Obtaining an instance of the builtin type 'list' (line 352)
        list_19978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 65), 'list')
        # Adding type elements to the builtin type 'list' instance (line 352)
        # Adding element type (line 352)
        # Getting the type of 'mod' (line 352)
        mod_19979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 66), 'mod', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 65), list_19978, mod_19979)
        
        int_19980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 72), 'int')
        # Processing the call keyword arguments (line 351)
        kwargs_19981 = {}
        # Getting the type of '__import__' (line 351)
        import___19970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 33), '__import__', False)
        # Calling __import__(args, kwargs) (line 351)
        import___call_result_19982 = invoke(stypy.reporting.localization.Localization(__file__, 351, 33), import___19970, *[mod_19971, globals_call_result_19974, locals_call_result_19977, list_19978, int_19980], **kwargs_19981)
        
        # Assigning a type to the variable 'current_module' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 16), 'current_module', import___call_result_19982)
        
        # Assigning a Call to a Name (line 354):
        
        # Call to getattr(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'current_module' (line 354)
        current_module_19984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 41), 'current_module', False)
        # Getting the type of 'callback_class' (line 354)
        callback_class_19985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 57), 'callback_class', False)
        # Getting the type of 'False' (line 354)
        False_19986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 73), 'False', False)
        # Processing the call keyword arguments (line 354)
        kwargs_19987 = {}
        # Getting the type of 'getattr' (line 354)
        getattr_19983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 33), 'getattr', False)
        # Calling getattr(args, kwargs) (line 354)
        getattr_call_result_19988 = invoke(stypy.reporting.localization.Localization(__file__, 354, 33), getattr_19983, *[current_module_19984, callback_class_19985, False_19986], **kwargs_19987)
        
        # Assigning a type to the variable 'callback_class' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 16), 'callback_class', getattr_call_result_19988)
        # SSA join for if statement (line 347)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 345)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to callable(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'callback_class' (line 355)
        callback_class_19990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 20), 'callback_class', False)
        # Processing the call keyword arguments (line 355)
        kwargs_19991 = {}
        # Getting the type of 'callable' (line 355)
        callable_19989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 11), 'callable', False)
        # Calling callable(args, kwargs) (line 355)
        callable_call_result_19992 = invoke(stypy.reporting.localization.Localization(__file__, 355, 11), callable_19989, *[callback_class_19990], **kwargs_19991)
        
        # Testing the type of an if condition (line 355)
        if_condition_19993 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 355, 8), callable_call_result_19992)
        # Assigning a type to the variable 'if_condition_19993' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'if_condition_19993', if_condition_19993)
        # SSA begins for if statement (line 355)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'callback_class' (line 356)
        callback_class_19994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 19), 'callback_class')
        # Assigning a type to the variable 'stypy_return_type' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'stypy_return_type', callback_class_19994)
        # SSA branch for the else part of an if statement (line 355)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'None' (line 358)
        None_19995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'stypy_return_type', None_19995)
        # SSA join for if statement (line 355)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_get_cls_to_instantiate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_cls_to_instantiate' in the type store
        # Getting the type of 'stypy_return_type' (line 343)
        stypy_return_type_19996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19996)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_cls_to_instantiate'
        return stypy_return_type_19996


    @norecursion
    def trigger_tool(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 360)
        None_19997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 40), 'None')
        # Getting the type of 'None' (line 360)
        None_19998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 58), 'None')
        # Getting the type of 'None' (line 361)
        None_19999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 26), 'None')
        defaults = [None_19997, None_19998, None_19999]
        # Create a new context for function 'trigger_tool'
        module_type_store = module_type_store.open_function_context('trigger_tool', 360, 4, False)
        # Assigning a type to the variable 'self' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolManager.trigger_tool.__dict__.__setitem__('stypy_localization', localization)
        ToolManager.trigger_tool.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolManager.trigger_tool.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolManager.trigger_tool.__dict__.__setitem__('stypy_function_name', 'ToolManager.trigger_tool')
        ToolManager.trigger_tool.__dict__.__setitem__('stypy_param_names_list', ['name', 'sender', 'canvasevent', 'data'])
        ToolManager.trigger_tool.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolManager.trigger_tool.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolManager.trigger_tool.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolManager.trigger_tool.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolManager.trigger_tool.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolManager.trigger_tool.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManager.trigger_tool', ['name', 'sender', 'canvasevent', 'data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'trigger_tool', localization, ['name', 'sender', 'canvasevent', 'data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'trigger_tool(...)' code ##################

        unicode_20000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, (-1)), 'unicode', u'\n        Trigger a tool and emit the tool_trigger_[name] event\n\n        Parameters\n        ----------\n        name : string\n            Name of the tool\n        sender: object\n            Object that wishes to trigger the tool\n        canvasevent : Event\n            Original Canvas event or None\n        data : Object\n            Extra data to pass to the tool when triggering\n        ')
        
        # Assigning a Call to a Name (line 376):
        
        # Call to get_tool(...): (line 376)
        # Processing the call arguments (line 376)
        # Getting the type of 'name' (line 376)
        name_20003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 29), 'name', False)
        # Processing the call keyword arguments (line 376)
        kwargs_20004 = {}
        # Getting the type of 'self' (line 376)
        self_20001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 15), 'self', False)
        # Obtaining the member 'get_tool' of a type (line 376)
        get_tool_20002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 15), self_20001, 'get_tool')
        # Calling get_tool(args, kwargs) (line 376)
        get_tool_call_result_20005 = invoke(stypy.reporting.localization.Localization(__file__, 376, 15), get_tool_20002, *[name_20003], **kwargs_20004)
        
        # Assigning a type to the variable 'tool' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'tool', get_tool_call_result_20005)
        
        # Type idiom detected: calculating its left and rigth part (line 377)
        # Getting the type of 'tool' (line 377)
        tool_20006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 11), 'tool')
        # Getting the type of 'None' (line 377)
        None_20007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 19), 'None')
        
        (may_be_20008, more_types_in_union_20009) = may_be_none(tool_20006, None_20007)

        if may_be_20008:

            if more_types_in_union_20009:
                # Runtime conditional SSA (line 377)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 378)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_20009:
                # SSA join for if statement (line 377)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 380)
        # Getting the type of 'sender' (line 380)
        sender_20010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 11), 'sender')
        # Getting the type of 'None' (line 380)
        None_20011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 21), 'None')
        
        (may_be_20012, more_types_in_union_20013) = may_be_none(sender_20010, None_20011)

        if may_be_20012:

            if more_types_in_union_20013:
                # Runtime conditional SSA (line 380)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 381):
            # Getting the type of 'self' (line 381)
            self_20014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 21), 'self')
            # Assigning a type to the variable 'sender' (line 381)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'sender', self_20014)

            if more_types_in_union_20013:
                # SSA join for if statement (line 380)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to _trigger_tool(...): (line 383)
        # Processing the call arguments (line 383)
        # Getting the type of 'name' (line 383)
        name_20017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 27), 'name', False)
        # Getting the type of 'sender' (line 383)
        sender_20018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 33), 'sender', False)
        # Getting the type of 'canvasevent' (line 383)
        canvasevent_20019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 41), 'canvasevent', False)
        # Getting the type of 'data' (line 383)
        data_20020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 54), 'data', False)
        # Processing the call keyword arguments (line 383)
        kwargs_20021 = {}
        # Getting the type of 'self' (line 383)
        self_20015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'self', False)
        # Obtaining the member '_trigger_tool' of a type (line 383)
        _trigger_tool_20016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), self_20015, '_trigger_tool')
        # Calling _trigger_tool(args, kwargs) (line 383)
        _trigger_tool_call_result_20022 = invoke(stypy.reporting.localization.Localization(__file__, 383, 8), _trigger_tool_20016, *[name_20017, sender_20018, canvasevent_20019, data_20020], **kwargs_20021)
        
        
        # Assigning a BinOp to a Name (line 385):
        unicode_20023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 12), 'unicode', u'tool_trigger_%s')
        # Getting the type of 'name' (line 385)
        name_20024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 32), 'name')
        # Applying the binary operator '%' (line 385)
        result_mod_20025 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 12), '%', unicode_20023, name_20024)
        
        # Assigning a type to the variable 's' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 's', result_mod_20025)
        
        # Assigning a Call to a Name (line 386):
        
        # Call to ToolTriggerEvent(...): (line 386)
        # Processing the call arguments (line 386)
        # Getting the type of 's' (line 386)
        s_20027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 33), 's', False)
        # Getting the type of 'sender' (line 386)
        sender_20028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 36), 'sender', False)
        # Getting the type of 'tool' (line 386)
        tool_20029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 44), 'tool', False)
        # Getting the type of 'canvasevent' (line 386)
        canvasevent_20030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 50), 'canvasevent', False)
        # Getting the type of 'data' (line 386)
        data_20031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 63), 'data', False)
        # Processing the call keyword arguments (line 386)
        kwargs_20032 = {}
        # Getting the type of 'ToolTriggerEvent' (line 386)
        ToolTriggerEvent_20026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 16), 'ToolTriggerEvent', False)
        # Calling ToolTriggerEvent(args, kwargs) (line 386)
        ToolTriggerEvent_call_result_20033 = invoke(stypy.reporting.localization.Localization(__file__, 386, 16), ToolTriggerEvent_20026, *[s_20027, sender_20028, tool_20029, canvasevent_20030, data_20031], **kwargs_20032)
        
        # Assigning a type to the variable 'event' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'event', ToolTriggerEvent_call_result_20033)
        
        # Call to process(...): (line 387)
        # Processing the call arguments (line 387)
        # Getting the type of 's' (line 387)
        s_20037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 32), 's', False)
        # Getting the type of 'event' (line 387)
        event_20038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 35), 'event', False)
        # Processing the call keyword arguments (line 387)
        kwargs_20039 = {}
        # Getting the type of 'self' (line 387)
        self_20034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'self', False)
        # Obtaining the member '_callbacks' of a type (line 387)
        _callbacks_20035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 8), self_20034, '_callbacks')
        # Obtaining the member 'process' of a type (line 387)
        process_20036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 8), _callbacks_20035, 'process')
        # Calling process(args, kwargs) (line 387)
        process_call_result_20040 = invoke(stypy.reporting.localization.Localization(__file__, 387, 8), process_20036, *[s_20037, event_20038], **kwargs_20039)
        
        
        # ################# End of 'trigger_tool(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'trigger_tool' in the type store
        # Getting the type of 'stypy_return_type' (line 360)
        stypy_return_type_20041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20041)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'trigger_tool'
        return stypy_return_type_20041


    @norecursion
    def _trigger_tool(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 389)
        None_20042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 41), 'None')
        # Getting the type of 'None' (line 389)
        None_20043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 59), 'None')
        # Getting the type of 'None' (line 389)
        None_20044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 70), 'None')
        defaults = [None_20042, None_20043, None_20044]
        # Create a new context for function '_trigger_tool'
        module_type_store = module_type_store.open_function_context('_trigger_tool', 389, 4, False)
        # Assigning a type to the variable 'self' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolManager._trigger_tool.__dict__.__setitem__('stypy_localization', localization)
        ToolManager._trigger_tool.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolManager._trigger_tool.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolManager._trigger_tool.__dict__.__setitem__('stypy_function_name', 'ToolManager._trigger_tool')
        ToolManager._trigger_tool.__dict__.__setitem__('stypy_param_names_list', ['name', 'sender', 'canvasevent', 'data'])
        ToolManager._trigger_tool.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolManager._trigger_tool.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolManager._trigger_tool.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolManager._trigger_tool.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolManager._trigger_tool.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolManager._trigger_tool.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManager._trigger_tool', ['name', 'sender', 'canvasevent', 'data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_trigger_tool', localization, ['name', 'sender', 'canvasevent', 'data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_trigger_tool(...)' code ##################

        unicode_20045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, (-1)), 'unicode', u'\n        Trigger on a tool\n\n        Method to actually trigger the tool\n        ')
        
        # Assigning a Call to a Name (line 395):
        
        # Call to get_tool(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'name' (line 395)
        name_20048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 29), 'name', False)
        # Processing the call keyword arguments (line 395)
        kwargs_20049 = {}
        # Getting the type of 'self' (line 395)
        self_20046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 15), 'self', False)
        # Obtaining the member 'get_tool' of a type (line 395)
        get_tool_20047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 15), self_20046, 'get_tool')
        # Calling get_tool(args, kwargs) (line 395)
        get_tool_call_result_20050 = invoke(stypy.reporting.localization.Localization(__file__, 395, 15), get_tool_20047, *[name_20048], **kwargs_20049)
        
        # Assigning a type to the variable 'tool' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'tool', get_tool_call_result_20050)
        
        
        # Call to isinstance(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 'tool' (line 397)
        tool_20052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 22), 'tool', False)
        # Getting the type of 'tools' (line 397)
        tools_20053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 28), 'tools', False)
        # Obtaining the member 'ToolToggleBase' of a type (line 397)
        ToolToggleBase_20054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 28), tools_20053, 'ToolToggleBase')
        # Processing the call keyword arguments (line 397)
        kwargs_20055 = {}
        # Getting the type of 'isinstance' (line 397)
        isinstance_20051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 397)
        isinstance_call_result_20056 = invoke(stypy.reporting.localization.Localization(__file__, 397, 11), isinstance_20051, *[tool_20052, ToolToggleBase_20054], **kwargs_20055)
        
        # Testing the type of an if condition (line 397)
        if_condition_20057 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 397, 8), isinstance_call_result_20056)
        # Assigning a type to the variable 'if_condition_20057' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'if_condition_20057', if_condition_20057)
        # SSA begins for if statement (line 397)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _handle_toggle(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'tool' (line 398)
        tool_20060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 32), 'tool', False)
        # Getting the type of 'sender' (line 398)
        sender_20061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 38), 'sender', False)
        # Getting the type of 'canvasevent' (line 398)
        canvasevent_20062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 46), 'canvasevent', False)
        # Getting the type of 'data' (line 398)
        data_20063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 59), 'data', False)
        # Processing the call keyword arguments (line 398)
        kwargs_20064 = {}
        # Getting the type of 'self' (line 398)
        self_20058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'self', False)
        # Obtaining the member '_handle_toggle' of a type (line 398)
        _handle_toggle_20059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 12), self_20058, '_handle_toggle')
        # Calling _handle_toggle(args, kwargs) (line 398)
        _handle_toggle_call_result_20065 = invoke(stypy.reporting.localization.Localization(__file__, 398, 12), _handle_toggle_20059, *[tool_20060, sender_20061, canvasevent_20062, data_20063], **kwargs_20064)
        
        # SSA join for if statement (line 397)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to trigger(...): (line 402)
        # Processing the call arguments (line 402)
        # Getting the type of 'sender' (line 402)
        sender_20068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 21), 'sender', False)
        # Getting the type of 'canvasevent' (line 402)
        canvasevent_20069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 29), 'canvasevent', False)
        # Getting the type of 'data' (line 402)
        data_20070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 42), 'data', False)
        # Processing the call keyword arguments (line 402)
        kwargs_20071 = {}
        # Getting the type of 'tool' (line 402)
        tool_20066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'tool', False)
        # Obtaining the member 'trigger' of a type (line 402)
        trigger_20067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 8), tool_20066, 'trigger')
        # Calling trigger(args, kwargs) (line 402)
        trigger_call_result_20072 = invoke(stypy.reporting.localization.Localization(__file__, 402, 8), trigger_20067, *[sender_20068, canvasevent_20069, data_20070], **kwargs_20071)
        
        
        # ################# End of '_trigger_tool(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_trigger_tool' in the type store
        # Getting the type of 'stypy_return_type' (line 389)
        stypy_return_type_20073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20073)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_trigger_tool'
        return stypy_return_type_20073


    @norecursion
    def _key_press(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_key_press'
        module_type_store = module_type_store.open_function_context('_key_press', 404, 4, False)
        # Assigning a type to the variable 'self' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolManager._key_press.__dict__.__setitem__('stypy_localization', localization)
        ToolManager._key_press.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolManager._key_press.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolManager._key_press.__dict__.__setitem__('stypy_function_name', 'ToolManager._key_press')
        ToolManager._key_press.__dict__.__setitem__('stypy_param_names_list', ['event'])
        ToolManager._key_press.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolManager._key_press.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolManager._key_press.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolManager._key_press.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolManager._key_press.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolManager._key_press.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManager._key_press', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_key_press', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_key_press(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'event' (line 405)
        event_20074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 11), 'event')
        # Obtaining the member 'key' of a type (line 405)
        key_20075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 11), event_20074, 'key')
        # Getting the type of 'None' (line 405)
        None_20076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 24), 'None')
        # Applying the binary operator 'is' (line 405)
        result_is__20077 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 11), 'is', key_20075, None_20076)
        
        
        # Call to locked(...): (line 405)
        # Processing the call keyword arguments (line 405)
        kwargs_20081 = {}
        # Getting the type of 'self' (line 405)
        self_20078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 32), 'self', False)
        # Obtaining the member 'keypresslock' of a type (line 405)
        keypresslock_20079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 32), self_20078, 'keypresslock')
        # Obtaining the member 'locked' of a type (line 405)
        locked_20080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 32), keypresslock_20079, 'locked')
        # Calling locked(args, kwargs) (line 405)
        locked_call_result_20082 = invoke(stypy.reporting.localization.Localization(__file__, 405, 32), locked_20080, *[], **kwargs_20081)
        
        # Applying the binary operator 'or' (line 405)
        result_or_keyword_20083 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 11), 'or', result_is__20077, locked_call_result_20082)
        
        # Testing the type of an if condition (line 405)
        if_condition_20084 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 405, 8), result_or_keyword_20083)
        # Assigning a type to the variable 'if_condition_20084' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'if_condition_20084', if_condition_20084)
        # SSA begins for if statement (line 405)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 405)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 408):
        
        # Call to get(...): (line 408)
        # Processing the call arguments (line 408)
        # Getting the type of 'event' (line 408)
        event_20088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 30), 'event', False)
        # Obtaining the member 'key' of a type (line 408)
        key_20089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 30), event_20088, 'key')
        # Getting the type of 'None' (line 408)
        None_20090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 41), 'None', False)
        # Processing the call keyword arguments (line 408)
        kwargs_20091 = {}
        # Getting the type of 'self' (line 408)
        self_20085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 15), 'self', False)
        # Obtaining the member '_keys' of a type (line 408)
        _keys_20086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 15), self_20085, '_keys')
        # Obtaining the member 'get' of a type (line 408)
        get_20087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 15), _keys_20086, 'get')
        # Calling get(args, kwargs) (line 408)
        get_call_result_20092 = invoke(stypy.reporting.localization.Localization(__file__, 408, 15), get_20087, *[key_20089, None_20090], **kwargs_20091)
        
        # Assigning a type to the variable 'name' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'name', get_call_result_20092)
        
        # Type idiom detected: calculating its left and rigth part (line 409)
        # Getting the type of 'name' (line 409)
        name_20093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 11), 'name')
        # Getting the type of 'None' (line 409)
        None_20094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 19), 'None')
        
        (may_be_20095, more_types_in_union_20096) = may_be_none(name_20093, None_20094)

        if may_be_20095:

            if more_types_in_union_20096:
                # Runtime conditional SSA (line 409)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 410)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_20096:
                # SSA join for if statement (line 409)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to trigger_tool(...): (line 411)
        # Processing the call arguments (line 411)
        # Getting the type of 'name' (line 411)
        name_20099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 26), 'name', False)
        # Processing the call keyword arguments (line 411)
        # Getting the type of 'event' (line 411)
        event_20100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 44), 'event', False)
        keyword_20101 = event_20100
        kwargs_20102 = {'canvasevent': keyword_20101}
        # Getting the type of 'self' (line 411)
        self_20097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'self', False)
        # Obtaining the member 'trigger_tool' of a type (line 411)
        trigger_tool_20098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), self_20097, 'trigger_tool')
        # Calling trigger_tool(args, kwargs) (line 411)
        trigger_tool_call_result_20103 = invoke(stypy.reporting.localization.Localization(__file__, 411, 8), trigger_tool_20098, *[name_20099], **kwargs_20102)
        
        
        # ################# End of '_key_press(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_key_press' in the type store
        # Getting the type of 'stypy_return_type' (line 404)
        stypy_return_type_20104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20104)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_key_press'
        return stypy_return_type_20104


    @norecursion
    def tools(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tools'
        module_type_store = module_type_store.open_function_context('tools', 413, 4, False)
        # Assigning a type to the variable 'self' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolManager.tools.__dict__.__setitem__('stypy_localization', localization)
        ToolManager.tools.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolManager.tools.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolManager.tools.__dict__.__setitem__('stypy_function_name', 'ToolManager.tools')
        ToolManager.tools.__dict__.__setitem__('stypy_param_names_list', [])
        ToolManager.tools.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolManager.tools.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolManager.tools.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolManager.tools.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolManager.tools.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolManager.tools.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManager.tools', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tools', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tools(...)' code ##################

        unicode_20105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 8), 'unicode', u'Return the tools controlled by `ToolManager`')
        # Getting the type of 'self' (line 417)
        self_20106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 15), 'self')
        # Obtaining the member '_tools' of a type (line 417)
        _tools_20107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 15), self_20106, '_tools')
        # Assigning a type to the variable 'stypy_return_type' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'stypy_return_type', _tools_20107)
        
        # ################# End of 'tools(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tools' in the type store
        # Getting the type of 'stypy_return_type' (line 413)
        stypy_return_type_20108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20108)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tools'
        return stypy_return_type_20108


    @norecursion
    def get_tool(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 419)
        True_20109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 34), 'True')
        defaults = [True_20109]
        # Create a new context for function 'get_tool'
        module_type_store = module_type_store.open_function_context('get_tool', 419, 4, False)
        # Assigning a type to the variable 'self' (line 420)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolManager.get_tool.__dict__.__setitem__('stypy_localization', localization)
        ToolManager.get_tool.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolManager.get_tool.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolManager.get_tool.__dict__.__setitem__('stypy_function_name', 'ToolManager.get_tool')
        ToolManager.get_tool.__dict__.__setitem__('stypy_param_names_list', ['name', 'warn'])
        ToolManager.get_tool.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolManager.get_tool.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolManager.get_tool.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolManager.get_tool.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolManager.get_tool.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolManager.get_tool.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolManager.get_tool', ['name', 'warn'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_tool', localization, ['name', 'warn'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_tool(...)' code ##################

        unicode_20110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, (-1)), 'unicode', u'\n        Return the tool object, also accepts the actual tool for convenience\n\n        Parameters\n        ----------\n        name : str, ToolBase\n            Name of the tool, or the tool itself\n        warn : bool, optional\n            If this method should give warnings.\n        ')
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 430)
        # Processing the call arguments (line 430)
        # Getting the type of 'name' (line 430)
        name_20112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 22), 'name', False)
        # Getting the type of 'tools' (line 430)
        tools_20113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 28), 'tools', False)
        # Obtaining the member 'ToolBase' of a type (line 430)
        ToolBase_20114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 28), tools_20113, 'ToolBase')
        # Processing the call keyword arguments (line 430)
        kwargs_20115 = {}
        # Getting the type of 'isinstance' (line 430)
        isinstance_20111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 430)
        isinstance_call_result_20116 = invoke(stypy.reporting.localization.Localization(__file__, 430, 11), isinstance_20111, *[name_20112, ToolBase_20114], **kwargs_20115)
        
        
        # Getting the type of 'name' (line 430)
        name_20117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 48), 'name')
        # Obtaining the member 'name' of a type (line 430)
        name_20118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 48), name_20117, 'name')
        # Getting the type of 'self' (line 430)
        self_20119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 61), 'self')
        # Obtaining the member '_tools' of a type (line 430)
        _tools_20120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 61), self_20119, '_tools')
        # Applying the binary operator 'in' (line 430)
        result_contains_20121 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 48), 'in', name_20118, _tools_20120)
        
        # Applying the binary operator 'and' (line 430)
        result_and_keyword_20122 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 11), 'and', isinstance_call_result_20116, result_contains_20121)
        
        # Testing the type of an if condition (line 430)
        if_condition_20123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 430, 8), result_and_keyword_20122)
        # Assigning a type to the variable 'if_condition_20123' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'if_condition_20123', if_condition_20123)
        # SSA begins for if statement (line 430)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'name' (line 431)
        name_20124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 19), 'name')
        # Assigning a type to the variable 'stypy_return_type' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 12), 'stypy_return_type', name_20124)
        # SSA join for if statement (line 430)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'name' (line 432)
        name_20125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 11), 'name')
        # Getting the type of 'self' (line 432)
        self_20126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 23), 'self')
        # Obtaining the member '_tools' of a type (line 432)
        _tools_20127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 23), self_20126, '_tools')
        # Applying the binary operator 'notin' (line 432)
        result_contains_20128 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 11), 'notin', name_20125, _tools_20127)
        
        # Testing the type of an if condition (line 432)
        if_condition_20129 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 432, 8), result_contains_20128)
        # Assigning a type to the variable 'if_condition_20129' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'if_condition_20129', if_condition_20129)
        # SSA begins for if statement (line 432)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'warn' (line 433)
        warn_20130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 15), 'warn')
        # Testing the type of an if condition (line 433)
        if_condition_20131 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 433, 12), warn_20130)
        # Assigning a type to the variable 'if_condition_20131' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 12), 'if_condition_20131', if_condition_20131)
        # SSA begins for if statement (line 433)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 434)
        # Processing the call arguments (line 434)
        unicode_20134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 30), 'unicode', u'ToolManager does not control tool %s')
        # Getting the type of 'name' (line 434)
        name_20135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 71), 'name', False)
        # Applying the binary operator '%' (line 434)
        result_mod_20136 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 30), '%', unicode_20134, name_20135)
        
        # Processing the call keyword arguments (line 434)
        kwargs_20137 = {}
        # Getting the type of 'warnings' (line 434)
        warnings_20132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 16), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 434)
        warn_20133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 16), warnings_20132, 'warn')
        # Calling warn(args, kwargs) (line 434)
        warn_call_result_20138 = invoke(stypy.reporting.localization.Localization(__file__, 434, 16), warn_20133, *[result_mod_20136], **kwargs_20137)
        
        # SSA join for if statement (line 433)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'None' (line 435)
        None_20139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'stypy_return_type', None_20139)
        # SSA join for if statement (line 432)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 436)
        name_20140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 27), 'name')
        # Getting the type of 'self' (line 436)
        self_20141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 15), 'self')
        # Obtaining the member '_tools' of a type (line 436)
        _tools_20142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 15), self_20141, '_tools')
        # Obtaining the member '__getitem__' of a type (line 436)
        getitem___20143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 15), _tools_20142, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 436)
        subscript_call_result_20144 = invoke(stypy.reporting.localization.Localization(__file__, 436, 15), getitem___20143, name_20140)
        
        # Assigning a type to the variable 'stypy_return_type' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'stypy_return_type', subscript_call_result_20144)
        
        # ################# End of 'get_tool(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_tool' in the type store
        # Getting the type of 'stypy_return_type' (line 419)
        stypy_return_type_20145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20145)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_tool'
        return stypy_return_type_20145


# Assigning a type to the variable 'ToolManager' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'ToolManager', ToolManager)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
