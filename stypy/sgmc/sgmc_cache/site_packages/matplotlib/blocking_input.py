
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: This provides several classes used for blocking interaction with figure
3: windows:
4: 
5: :class:`BlockingInput`
6:     creates a callable object to retrieve events in a blocking way for
7:     interactive sessions
8: 
9: :class:`BlockingKeyMouseInput`
10:     creates a callable object to retrieve key or mouse clicks in a blocking
11:     way for interactive sessions.
12:     Note: Subclass of BlockingInput. Used by waitforbuttonpress
13: 
14: :class:`BlockingMouseInput`
15:     creates a callable object to retrieve mouse clicks in a blocking way for
16:     interactive sessions.
17:     Note: Subclass of BlockingInput.  Used by ginput
18: 
19: :class:`BlockingContourLabeler`
20:     creates a callable object to retrieve mouse clicks in a blocking way that
21:     will then be used to place labels on a ContourSet
22:     Note: Subclass of BlockingMouseInput.  Used by clabel
23: '''
24: 
25: from __future__ import (absolute_import, division, print_function,
26:                         unicode_literals)
27: 
28: import six
29: from matplotlib import verbose
30: import matplotlib.lines as mlines
31: 
32: import warnings
33: 
34: 
35: class BlockingInput(object):
36:     '''
37:     Class that creates a callable object to retrieve events in a
38:     blocking way.
39:     '''
40:     def __init__(self, fig, eventslist=()):
41:         self.fig = fig
42:         self.eventslist = eventslist
43: 
44:     def on_event(self, event):
45:         '''
46:         Event handler that will be passed to the current figure to
47:         retrieve events.
48:         '''
49:         # Add a new event to list - using a separate function is
50:         # overkill for the base class, but this is consistent with
51:         # subclasses
52:         self.add_event(event)
53: 
54:         verbose.report("Event %i" % len(self.events))
55: 
56:         # This will extract info from events
57:         self.post_event()
58: 
59:         # Check if we have enough events already
60:         if len(self.events) >= self.n and self.n > 0:
61:             self.fig.canvas.stop_event_loop()
62: 
63:     def post_event(self):
64:         '''For baseclass, do nothing but collect events'''
65:         pass
66: 
67:     def cleanup(self):
68:         '''Disconnect all callbacks'''
69:         for cb in self.callbacks:
70:             self.fig.canvas.mpl_disconnect(cb)
71: 
72:         self.callbacks = []
73: 
74:     def add_event(self, event):
75:         '''For base class, this just appends an event to events.'''
76:         self.events.append(event)
77: 
78:     def pop_event(self, index=-1):
79:         '''
80:         This removes an event from the event list.  Defaults to
81:         removing last event, but an index can be supplied.  Note that
82:         this does not check that there are events, much like the
83:         normal pop method.  If not events exist, this will throw an
84:         exception.
85:         '''
86:         self.events.pop(index)
87: 
88:     def pop(self, index=-1):
89:         self.pop_event(index)
90:     pop.__doc__ = pop_event.__doc__
91: 
92:     def __call__(self, n=1, timeout=30):
93:         '''
94:         Blocking call to retrieve n events
95:         '''
96: 
97:         if not isinstance(n, int):
98:             raise ValueError("Requires an integer argument")
99:         self.n = n
100: 
101:         self.events = []
102:         self.callbacks = []
103: 
104:         if hasattr(self.fig, "manager"):
105:             # Ensure that the figure is shown, if we are managing it.
106:             self.fig.show()
107: 
108:         # connect the events to the on_event function call
109:         for n in self.eventslist:
110:             self.callbacks.append(
111:                 self.fig.canvas.mpl_connect(n, self.on_event))
112: 
113:         try:
114:             # Start event loop
115:             self.fig.canvas.start_event_loop(timeout=timeout)
116:         finally:  # Run even on exception like ctrl-c
117:             # Disconnect the callbacks
118:             self.cleanup()
119: 
120:         # Return the events in this case
121:         return self.events
122: 
123: 
124: class BlockingMouseInput(BlockingInput):
125:     '''
126:     Class that creates a callable object to retrieve mouse clicks in a
127:     blocking way.
128: 
129:     This class will also retrieve keyboard clicks and treat them like
130:     appropriate mouse clicks (delete and backspace are like mouse button 3,
131:     enter is like mouse button 2 and all others are like mouse button 1).
132:     '''
133: 
134:     button_add = 1
135:     button_pop = 3
136:     button_stop = 2
137: 
138:     def __init__(self, fig, mouse_add=1, mouse_pop=3, mouse_stop=2):
139:         BlockingInput.__init__(self, fig=fig,
140:                                eventslist=('button_press_event',
141:                                            'key_press_event'))
142:         self.button_add = mouse_add
143:         self.button_pop = mouse_pop
144:         self.button_stop = mouse_stop
145: 
146:     def post_event(self):
147:         '''
148:         This will be called to process events
149:         '''
150:         if len(self.events) == 0:
151:             warnings.warn("No events yet")
152:         elif self.events[-1].name == 'key_press_event':
153:             self.key_event()
154:         else:
155:             self.mouse_event()
156: 
157:     def mouse_event(self):
158:         '''Process a mouse click event'''
159: 
160:         event = self.events[-1]
161:         button = event.button
162: 
163:         if button == self.button_pop:
164:             self.mouse_event_pop(event)
165:         elif button == self.button_stop:
166:             self.mouse_event_stop(event)
167:         else:
168:             self.mouse_event_add(event)
169: 
170:     def key_event(self):
171:         '''
172:         Process a key click event.  This maps certain keys to appropriate
173:         mouse click events.
174:         '''
175: 
176:         event = self.events[-1]
177:         if event.key is None:
178:             # at least in mac os X gtk backend some key returns None.
179:             return
180: 
181:         key = event.key.lower()
182: 
183:         if key in ['backspace', 'delete']:
184:             self.mouse_event_pop(event)
185:         elif key in ['escape', 'enter']:
186:             # on windows XP and wxAgg, the enter key doesn't seem to register
187:             self.mouse_event_stop(event)
188:         else:
189:             self.mouse_event_add(event)
190: 
191:     def mouse_event_add(self, event):
192:         '''
193:         Will be called for any event involving a button other than
194:         button 2 or 3.  This will add a click if it is inside axes.
195:         '''
196:         if event.inaxes:
197:             self.add_click(event)
198:         else:  # If not a valid click, remove from event list
199:             BlockingInput.pop(self, -1)
200: 
201:     def mouse_event_stop(self, event):
202:         '''
203:         Will be called for any event involving button 2.
204:         Button 2 ends blocking input.
205:         '''
206: 
207:         # Remove last event just for cleanliness
208:         BlockingInput.pop(self, -1)
209: 
210:         # This will exit even if not in infinite mode.  This is
211:         # consistent with MATLAB and sometimes quite useful, but will
212:         # require the user to test how many points were actually
213:         # returned before using data.
214:         self.fig.canvas.stop_event_loop()
215: 
216:     def mouse_event_pop(self, event):
217:         '''
218:         Will be called for any event involving button 3.
219:         Button 3 removes the last click.
220:         '''
221:         # Remove this last event
222:         BlockingInput.pop(self, -1)
223: 
224:         # Now remove any existing clicks if possible
225:         if len(self.events) > 0:
226:             self.pop(event, -1)
227: 
228:     def add_click(self, event):
229:         '''
230:         This add the coordinates of an event to the list of clicks
231:         '''
232:         self.clicks.append((event.xdata, event.ydata))
233: 
234:         verbose.report("input %i: %f,%f" %
235:                        (len(self.clicks), event.xdata, event.ydata))
236: 
237:         # If desired plot up click
238:         if self.show_clicks:
239:             line = mlines.Line2D([event.xdata], [event.ydata],
240:                                  marker='+', color='r')
241:             event.inaxes.add_line(line)
242:             self.marks.append(line)
243:             self.fig.canvas.draw()
244: 
245:     def pop_click(self, event, index=-1):
246:         '''
247:         This removes a click from the list of clicks.  Defaults to
248:         removing the last click.
249:         '''
250:         self.clicks.pop(index)
251: 
252:         if self.show_clicks:
253: 
254:             mark = self.marks.pop(index)
255:             mark.remove()
256: 
257:             self.fig.canvas.draw()
258:             # NOTE: I do NOT understand why the above 3 lines does not work
259:             # for the keyboard backspace event on windows XP wxAgg.
260:             # maybe event.inaxes here is a COPY of the actual axes?
261: 
262:     def pop(self, event, index=-1):
263:         '''
264:         This removes a click and the associated event from the object.
265:         Defaults to removing the last click, but any index can be
266:         supplied.
267:         '''
268:         self.pop_click(event, index)
269:         BlockingInput.pop(self, index)
270: 
271:     def cleanup(self, event=None):
272:         # clean the figure
273:         if self.show_clicks:
274: 
275:             for mark in self.marks:
276:                 mark.remove()
277:             self.marks = []
278: 
279:             self.fig.canvas.draw()
280: 
281:         # Call base class to remove callbacks
282:         BlockingInput.cleanup(self)
283: 
284:     def __call__(self, n=1, timeout=30, show_clicks=True):
285:         '''
286:         Blocking call to retrieve n coordinate pairs through mouse
287:         clicks.
288:         '''
289:         self.show_clicks = show_clicks
290:         self.clicks = []
291:         self.marks = []
292:         BlockingInput.__call__(self, n=n, timeout=timeout)
293: 
294:         return self.clicks
295: 
296: 
297: class BlockingContourLabeler(BlockingMouseInput):
298:     '''
299:     Class that creates a callable object that uses mouse clicks or key
300:     clicks on a figure window to place contour labels.
301:     '''
302:     def __init__(self, cs):
303:         self.cs = cs
304:         BlockingMouseInput.__init__(self, fig=cs.ax.figure)
305: 
306:     def add_click(self, event):
307:         self.button1(event)
308: 
309:     def pop_click(self, event, index=-1):
310:         self.button3(event)
311: 
312:     def button1(self, event):
313:         '''
314:         This will be called if an event involving a button other than
315:         2 or 3 occcurs.  This will add a label to a contour.
316:         '''
317: 
318:         # Shorthand
319:         if event.inaxes == self.cs.ax:
320:             self.cs.add_label_near(event.x, event.y, self.inline,
321:                                    inline_spacing=self.inline_spacing,
322:                                    transform=False)
323:             self.fig.canvas.draw()
324:         else:  # Remove event if not valid
325:             BlockingInput.pop(self)
326: 
327:     def button3(self, event):
328:         '''
329:         This will be called if button 3 is clicked.  This will remove
330:         a label if not in inline mode.  Unfortunately, if one is doing
331:         inline labels, then there is currently no way to fix the
332:         broken contour - once humpty-dumpty is broken, he can't be put
333:         back together.  In inline mode, this does nothing.
334:         '''
335: 
336:         if self.inline:
337:             pass
338:         else:
339:             self.cs.pop_label()
340:             self.cs.ax.figure.canvas.draw()
341: 
342:     def __call__(self, inline, inline_spacing=5, n=-1, timeout=-1):
343:         self.inline = inline
344:         self.inline_spacing = inline_spacing
345: 
346:         BlockingMouseInput.__call__(self, n=n, timeout=timeout,
347:                                     show_clicks=False)
348: 
349: 
350: class BlockingKeyMouseInput(BlockingInput):
351:     '''
352:     Class that creates a callable object to retrieve a single mouse or
353:     keyboard click
354:     '''
355:     def __init__(self, fig):
356:         BlockingInput.__init__(self, fig=fig, eventslist=(
357:             'button_press_event', 'key_press_event'))
358: 
359:     def post_event(self):
360:         '''
361:         Determines if it is a key event
362:         '''
363:         if len(self.events) == 0:
364:             warnings.warn("No events yet")
365:         else:
366:             self.keyormouse = self.events[-1].name == 'key_press_event'
367: 
368:     def __call__(self, timeout=30):
369:         '''
370:         Blocking call to retrieve a single mouse or key click
371:         Returns True if key click, False if mouse, or None if timeout
372:         '''
373:         self.keyormouse = None
374:         BlockingInput.__call__(self, n=1, timeout=timeout)
375: 
376:         return self.keyormouse
377: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_24557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, (-1)), 'unicode', u'\nThis provides several classes used for blocking interaction with figure\nwindows:\n\n:class:`BlockingInput`\n    creates a callable object to retrieve events in a blocking way for\n    interactive sessions\n\n:class:`BlockingKeyMouseInput`\n    creates a callable object to retrieve key or mouse clicks in a blocking\n    way for interactive sessions.\n    Note: Subclass of BlockingInput. Used by waitforbuttonpress\n\n:class:`BlockingMouseInput`\n    creates a callable object to retrieve mouse clicks in a blocking way for\n    interactive sessions.\n    Note: Subclass of BlockingInput.  Used by ginput\n\n:class:`BlockingContourLabeler`\n    creates a callable object to retrieve mouse clicks in a blocking way that\n    will then be used to place labels on a ContourSet\n    Note: Subclass of BlockingMouseInput.  Used by clabel\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'import six' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_24558 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'six')

if (type(import_24558) is not StypyTypeError):

    if (import_24558 != 'pyd_module'):
        __import__(import_24558)
        sys_modules_24559 = sys.modules[import_24558]
        import_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'six', sys_modules_24559.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'six', import_24558)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'from matplotlib import verbose' statement (line 29)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_24560 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib')

if (type(import_24560) is not StypyTypeError):

    if (import_24560 != 'pyd_module'):
        __import__(import_24560)
        sys_modules_24561 = sys.modules[import_24560]
        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib', sys_modules_24561.module_type_store, module_type_store, ['verbose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 29, 0), __file__, sys_modules_24561, sys_modules_24561.module_type_store, module_type_store)
    else:
        from matplotlib import verbose

        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib', None, module_type_store, ['verbose'], [verbose])

else:
    # Assigning a type to the variable 'matplotlib' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib', import_24560)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'import matplotlib.lines' statement (line 30)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_24562 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib.lines')

if (type(import_24562) is not StypyTypeError):

    if (import_24562 != 'pyd_module'):
        __import__(import_24562)
        sys_modules_24563 = sys.modules[import_24562]
        import_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'mlines', sys_modules_24563.module_type_store, module_type_store)
    else:
        import matplotlib.lines as mlines

        import_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'mlines', matplotlib.lines, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.lines' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib.lines', import_24562)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 0))

# 'import warnings' statement (line 32)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'warnings', warnings, module_type_store)

# Declaration of the 'BlockingInput' class

class BlockingInput(object, ):
    unicode_24564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, (-1)), 'unicode', u'\n    Class that creates a callable object to retrieve events in a\n    blocking way.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Obtaining an instance of the builtin type 'tuple' (line 40)
        tuple_24565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 40)
        
        defaults = [tuple_24565]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 40, 4, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingInput.__init__', ['fig', 'eventslist'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['fig', 'eventslist'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 41):
        # Getting the type of 'fig' (line 41)
        fig_24566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 19), 'fig')
        # Getting the type of 'self' (line 41)
        self_24567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self')
        # Setting the type of the member 'fig' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_24567, 'fig', fig_24566)
        
        # Assigning a Name to a Attribute (line 42):
        # Getting the type of 'eventslist' (line 42)
        eventslist_24568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 26), 'eventslist')
        # Getting the type of 'self' (line 42)
        self_24569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self')
        # Setting the type of the member 'eventslist' of a type (line 42)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_24569, 'eventslist', eventslist_24568)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def on_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'on_event'
        module_type_store = module_type_store.open_function_context('on_event', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingInput.on_event.__dict__.__setitem__('stypy_localization', localization)
        BlockingInput.on_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingInput.on_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingInput.on_event.__dict__.__setitem__('stypy_function_name', 'BlockingInput.on_event')
        BlockingInput.on_event.__dict__.__setitem__('stypy_param_names_list', ['event'])
        BlockingInput.on_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingInput.on_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingInput.on_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingInput.on_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingInput.on_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingInput.on_event.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingInput.on_event', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'on_event', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'on_event(...)' code ##################

        unicode_24570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, (-1)), 'unicode', u'\n        Event handler that will be passed to the current figure to\n        retrieve events.\n        ')
        
        # Call to add_event(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'event' (line 52)
        event_24573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 23), 'event', False)
        # Processing the call keyword arguments (line 52)
        kwargs_24574 = {}
        # Getting the type of 'self' (line 52)
        self_24571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'self', False)
        # Obtaining the member 'add_event' of a type (line 52)
        add_event_24572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), self_24571, 'add_event')
        # Calling add_event(args, kwargs) (line 52)
        add_event_call_result_24575 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), add_event_24572, *[event_24573], **kwargs_24574)
        
        
        # Call to report(...): (line 54)
        # Processing the call arguments (line 54)
        unicode_24578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 23), 'unicode', u'Event %i')
        
        # Call to len(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'self' (line 54)
        self_24580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 40), 'self', False)
        # Obtaining the member 'events' of a type (line 54)
        events_24581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 40), self_24580, 'events')
        # Processing the call keyword arguments (line 54)
        kwargs_24582 = {}
        # Getting the type of 'len' (line 54)
        len_24579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 36), 'len', False)
        # Calling len(args, kwargs) (line 54)
        len_call_result_24583 = invoke(stypy.reporting.localization.Localization(__file__, 54, 36), len_24579, *[events_24581], **kwargs_24582)
        
        # Applying the binary operator '%' (line 54)
        result_mod_24584 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 23), '%', unicode_24578, len_call_result_24583)
        
        # Processing the call keyword arguments (line 54)
        kwargs_24585 = {}
        # Getting the type of 'verbose' (line 54)
        verbose_24576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'verbose', False)
        # Obtaining the member 'report' of a type (line 54)
        report_24577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), verbose_24576, 'report')
        # Calling report(args, kwargs) (line 54)
        report_call_result_24586 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), report_24577, *[result_mod_24584], **kwargs_24585)
        
        
        # Call to post_event(...): (line 57)
        # Processing the call keyword arguments (line 57)
        kwargs_24589 = {}
        # Getting the type of 'self' (line 57)
        self_24587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self', False)
        # Obtaining the member 'post_event' of a type (line 57)
        post_event_24588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_24587, 'post_event')
        # Calling post_event(args, kwargs) (line 57)
        post_event_call_result_24590 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), post_event_24588, *[], **kwargs_24589)
        
        
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'self' (line 60)
        self_24592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'self', False)
        # Obtaining the member 'events' of a type (line 60)
        events_24593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 15), self_24592, 'events')
        # Processing the call keyword arguments (line 60)
        kwargs_24594 = {}
        # Getting the type of 'len' (line 60)
        len_24591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'len', False)
        # Calling len(args, kwargs) (line 60)
        len_call_result_24595 = invoke(stypy.reporting.localization.Localization(__file__, 60, 11), len_24591, *[events_24593], **kwargs_24594)
        
        # Getting the type of 'self' (line 60)
        self_24596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 31), 'self')
        # Obtaining the member 'n' of a type (line 60)
        n_24597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 31), self_24596, 'n')
        # Applying the binary operator '>=' (line 60)
        result_ge_24598 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 11), '>=', len_call_result_24595, n_24597)
        
        
        # Getting the type of 'self' (line 60)
        self_24599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 42), 'self')
        # Obtaining the member 'n' of a type (line 60)
        n_24600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 42), self_24599, 'n')
        int_24601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 51), 'int')
        # Applying the binary operator '>' (line 60)
        result_gt_24602 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 42), '>', n_24600, int_24601)
        
        # Applying the binary operator 'and' (line 60)
        result_and_keyword_24603 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 11), 'and', result_ge_24598, result_gt_24602)
        
        # Testing the type of an if condition (line 60)
        if_condition_24604 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 8), result_and_keyword_24603)
        # Assigning a type to the variable 'if_condition_24604' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'if_condition_24604', if_condition_24604)
        # SSA begins for if statement (line 60)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to stop_event_loop(...): (line 61)
        # Processing the call keyword arguments (line 61)
        kwargs_24609 = {}
        # Getting the type of 'self' (line 61)
        self_24605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'self', False)
        # Obtaining the member 'fig' of a type (line 61)
        fig_24606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 12), self_24605, 'fig')
        # Obtaining the member 'canvas' of a type (line 61)
        canvas_24607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 12), fig_24606, 'canvas')
        # Obtaining the member 'stop_event_loop' of a type (line 61)
        stop_event_loop_24608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 12), canvas_24607, 'stop_event_loop')
        # Calling stop_event_loop(args, kwargs) (line 61)
        stop_event_loop_call_result_24610 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), stop_event_loop_24608, *[], **kwargs_24609)
        
        # SSA join for if statement (line 60)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'on_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'on_event' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_24611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24611)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'on_event'
        return stypy_return_type_24611


    @norecursion
    def post_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'post_event'
        module_type_store = module_type_store.open_function_context('post_event', 63, 4, False)
        # Assigning a type to the variable 'self' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingInput.post_event.__dict__.__setitem__('stypy_localization', localization)
        BlockingInput.post_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingInput.post_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingInput.post_event.__dict__.__setitem__('stypy_function_name', 'BlockingInput.post_event')
        BlockingInput.post_event.__dict__.__setitem__('stypy_param_names_list', [])
        BlockingInput.post_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingInput.post_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingInput.post_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingInput.post_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingInput.post_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingInput.post_event.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingInput.post_event', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'post_event', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'post_event(...)' code ##################

        unicode_24612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'unicode', u'For baseclass, do nothing but collect events')
        pass
        
        # ################# End of 'post_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'post_event' in the type store
        # Getting the type of 'stypy_return_type' (line 63)
        stypy_return_type_24613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24613)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'post_event'
        return stypy_return_type_24613


    @norecursion
    def cleanup(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'cleanup'
        module_type_store = module_type_store.open_function_context('cleanup', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingInput.cleanup.__dict__.__setitem__('stypy_localization', localization)
        BlockingInput.cleanup.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingInput.cleanup.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingInput.cleanup.__dict__.__setitem__('stypy_function_name', 'BlockingInput.cleanup')
        BlockingInput.cleanup.__dict__.__setitem__('stypy_param_names_list', [])
        BlockingInput.cleanup.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingInput.cleanup.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingInput.cleanup.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingInput.cleanup.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingInput.cleanup.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingInput.cleanup.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingInput.cleanup', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cleanup', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cleanup(...)' code ##################

        unicode_24614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 8), 'unicode', u'Disconnect all callbacks')
        
        # Getting the type of 'self' (line 69)
        self_24615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 18), 'self')
        # Obtaining the member 'callbacks' of a type (line 69)
        callbacks_24616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 18), self_24615, 'callbacks')
        # Testing the type of a for loop iterable (line 69)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 69, 8), callbacks_24616)
        # Getting the type of the for loop variable (line 69)
        for_loop_var_24617 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 69, 8), callbacks_24616)
        # Assigning a type to the variable 'cb' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'cb', for_loop_var_24617)
        # SSA begins for a for statement (line 69)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to mpl_disconnect(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'cb' (line 70)
        cb_24622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 43), 'cb', False)
        # Processing the call keyword arguments (line 70)
        kwargs_24623 = {}
        # Getting the type of 'self' (line 70)
        self_24618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'self', False)
        # Obtaining the member 'fig' of a type (line 70)
        fig_24619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), self_24618, 'fig')
        # Obtaining the member 'canvas' of a type (line 70)
        canvas_24620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), fig_24619, 'canvas')
        # Obtaining the member 'mpl_disconnect' of a type (line 70)
        mpl_disconnect_24621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), canvas_24620, 'mpl_disconnect')
        # Calling mpl_disconnect(args, kwargs) (line 70)
        mpl_disconnect_call_result_24624 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), mpl_disconnect_24621, *[cb_24622], **kwargs_24623)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Attribute (line 72):
        
        # Obtaining an instance of the builtin type 'list' (line 72)
        list_24625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 72)
        
        # Getting the type of 'self' (line 72)
        self_24626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'self')
        # Setting the type of the member 'callbacks' of a type (line 72)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), self_24626, 'callbacks', list_24625)
        
        # ################# End of 'cleanup(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cleanup' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_24627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24627)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cleanup'
        return stypy_return_type_24627


    @norecursion
    def add_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_event'
        module_type_store = module_type_store.open_function_context('add_event', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingInput.add_event.__dict__.__setitem__('stypy_localization', localization)
        BlockingInput.add_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingInput.add_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingInput.add_event.__dict__.__setitem__('stypy_function_name', 'BlockingInput.add_event')
        BlockingInput.add_event.__dict__.__setitem__('stypy_param_names_list', ['event'])
        BlockingInput.add_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingInput.add_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingInput.add_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingInput.add_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingInput.add_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingInput.add_event.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingInput.add_event', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_event', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_event(...)' code ##################

        unicode_24628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 8), 'unicode', u'For base class, this just appends an event to events.')
        
        # Call to append(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'event' (line 76)
        event_24632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'event', False)
        # Processing the call keyword arguments (line 76)
        kwargs_24633 = {}
        # Getting the type of 'self' (line 76)
        self_24629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'self', False)
        # Obtaining the member 'events' of a type (line 76)
        events_24630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), self_24629, 'events')
        # Obtaining the member 'append' of a type (line 76)
        append_24631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), events_24630, 'append')
        # Calling append(args, kwargs) (line 76)
        append_call_result_24634 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), append_24631, *[event_24632], **kwargs_24633)
        
        
        # ################# End of 'add_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_event' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_24635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24635)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_event'
        return stypy_return_type_24635


    @norecursion
    def pop_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_24636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 30), 'int')
        defaults = [int_24636]
        # Create a new context for function 'pop_event'
        module_type_store = module_type_store.open_function_context('pop_event', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingInput.pop_event.__dict__.__setitem__('stypy_localization', localization)
        BlockingInput.pop_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingInput.pop_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingInput.pop_event.__dict__.__setitem__('stypy_function_name', 'BlockingInput.pop_event')
        BlockingInput.pop_event.__dict__.__setitem__('stypy_param_names_list', ['index'])
        BlockingInput.pop_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingInput.pop_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingInput.pop_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingInput.pop_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingInput.pop_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingInput.pop_event.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingInput.pop_event', ['index'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pop_event', localization, ['index'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pop_event(...)' code ##################

        unicode_24637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, (-1)), 'unicode', u'\n        This removes an event from the event list.  Defaults to\n        removing last event, but an index can be supplied.  Note that\n        this does not check that there are events, much like the\n        normal pop method.  If not events exist, this will throw an\n        exception.\n        ')
        
        # Call to pop(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'index' (line 86)
        index_24641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 24), 'index', False)
        # Processing the call keyword arguments (line 86)
        kwargs_24642 = {}
        # Getting the type of 'self' (line 86)
        self_24638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'self', False)
        # Obtaining the member 'events' of a type (line 86)
        events_24639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), self_24638, 'events')
        # Obtaining the member 'pop' of a type (line 86)
        pop_24640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), events_24639, 'pop')
        # Calling pop(args, kwargs) (line 86)
        pop_call_result_24643 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), pop_24640, *[index_24641], **kwargs_24642)
        
        
        # ################# End of 'pop_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pop_event' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_24644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24644)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pop_event'
        return stypy_return_type_24644


    @norecursion
    def pop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_24645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 24), 'int')
        defaults = [int_24645]
        # Create a new context for function 'pop'
        module_type_store = module_type_store.open_function_context('pop', 88, 4, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingInput.pop.__dict__.__setitem__('stypy_localization', localization)
        BlockingInput.pop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingInput.pop.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingInput.pop.__dict__.__setitem__('stypy_function_name', 'BlockingInput.pop')
        BlockingInput.pop.__dict__.__setitem__('stypy_param_names_list', ['index'])
        BlockingInput.pop.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingInput.pop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingInput.pop.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingInput.pop.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingInput.pop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingInput.pop.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingInput.pop', ['index'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pop', localization, ['index'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pop(...)' code ##################

        
        # Call to pop_event(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'index' (line 89)
        index_24648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 23), 'index', False)
        # Processing the call keyword arguments (line 89)
        kwargs_24649 = {}
        # Getting the type of 'self' (line 89)
        self_24646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self', False)
        # Obtaining the member 'pop_event' of a type (line 89)
        pop_event_24647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_24646, 'pop_event')
        # Calling pop_event(args, kwargs) (line 89)
        pop_event_call_result_24650 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), pop_event_24647, *[index_24648], **kwargs_24649)
        
        
        # ################# End of 'pop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pop' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_24651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24651)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pop'
        return stypy_return_type_24651


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_24652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 25), 'int')
        int_24653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 36), 'int')
        defaults = [int_24652, int_24653]
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 92, 4, False)
        # Assigning a type to the variable 'self' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingInput.__call__.__dict__.__setitem__('stypy_localization', localization)
        BlockingInput.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingInput.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingInput.__call__.__dict__.__setitem__('stypy_function_name', 'BlockingInput.__call__')
        BlockingInput.__call__.__dict__.__setitem__('stypy_param_names_list', ['n', 'timeout'])
        BlockingInput.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingInput.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingInput.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingInput.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingInput.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingInput.__call__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingInput.__call__', ['n', 'timeout'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['n', 'timeout'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        unicode_24654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, (-1)), 'unicode', u'\n        Blocking call to retrieve n events\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 97)
        # Getting the type of 'int' (line 97)
        int_24655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 29), 'int')
        # Getting the type of 'n' (line 97)
        n_24656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 26), 'n')
        
        (may_be_24657, more_types_in_union_24658) = may_not_be_subtype(int_24655, n_24656)

        if may_be_24657:

            if more_types_in_union_24658:
                # Runtime conditional SSA (line 97)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'n' (line 97)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'n', remove_subtype_from_union(n_24656, int))
            
            # Call to ValueError(...): (line 98)
            # Processing the call arguments (line 98)
            unicode_24660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 29), 'unicode', u'Requires an integer argument')
            # Processing the call keyword arguments (line 98)
            kwargs_24661 = {}
            # Getting the type of 'ValueError' (line 98)
            ValueError_24659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 98)
            ValueError_call_result_24662 = invoke(stypy.reporting.localization.Localization(__file__, 98, 18), ValueError_24659, *[unicode_24660], **kwargs_24661)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 98, 12), ValueError_call_result_24662, 'raise parameter', BaseException)

            if more_types_in_union_24658:
                # SSA join for if statement (line 97)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 99):
        # Getting the type of 'n' (line 99)
        n_24663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 17), 'n')
        # Getting the type of 'self' (line 99)
        self_24664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'self')
        # Setting the type of the member 'n' of a type (line 99)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), self_24664, 'n', n_24663)
        
        # Assigning a List to a Attribute (line 101):
        
        # Obtaining an instance of the builtin type 'list' (line 101)
        list_24665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 101)
        
        # Getting the type of 'self' (line 101)
        self_24666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'self')
        # Setting the type of the member 'events' of a type (line 101)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), self_24666, 'events', list_24665)
        
        # Assigning a List to a Attribute (line 102):
        
        # Obtaining an instance of the builtin type 'list' (line 102)
        list_24667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 102)
        
        # Getting the type of 'self' (line 102)
        self_24668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'self')
        # Setting the type of the member 'callbacks' of a type (line 102)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), self_24668, 'callbacks', list_24667)
        
        # Type idiom detected: calculating its left and rigth part (line 104)
        unicode_24669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 29), 'unicode', u'manager')
        # Getting the type of 'self' (line 104)
        self_24670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 19), 'self')
        # Obtaining the member 'fig' of a type (line 104)
        fig_24671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 19), self_24670, 'fig')
        
        (may_be_24672, more_types_in_union_24673) = may_provide_member(unicode_24669, fig_24671)

        if may_be_24672:

            if more_types_in_union_24673:
                # Runtime conditional SSA (line 104)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 104)
            self_24674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'self')
            # Obtaining the member 'fig' of a type (line 104)
            fig_24675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), self_24674, 'fig')
            # Setting the type of the member 'fig' of a type (line 104)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), self_24674, 'fig', remove_not_member_provider_from_union(fig_24671, u'manager'))
            
            # Call to show(...): (line 106)
            # Processing the call keyword arguments (line 106)
            kwargs_24679 = {}
            # Getting the type of 'self' (line 106)
            self_24676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'self', False)
            # Obtaining the member 'fig' of a type (line 106)
            fig_24677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), self_24676, 'fig')
            # Obtaining the member 'show' of a type (line 106)
            show_24678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), fig_24677, 'show')
            # Calling show(args, kwargs) (line 106)
            show_call_result_24680 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), show_24678, *[], **kwargs_24679)
            

            if more_types_in_union_24673:
                # SSA join for if statement (line 104)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'self' (line 109)
        self_24681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 17), 'self')
        # Obtaining the member 'eventslist' of a type (line 109)
        eventslist_24682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 17), self_24681, 'eventslist')
        # Testing the type of a for loop iterable (line 109)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 109, 8), eventslist_24682)
        # Getting the type of the for loop variable (line 109)
        for_loop_var_24683 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 109, 8), eventslist_24682)
        # Assigning a type to the variable 'n' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'n', for_loop_var_24683)
        # SSA begins for a for statement (line 109)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 110)
        # Processing the call arguments (line 110)
        
        # Call to mpl_connect(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'n' (line 111)
        n_24691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 44), 'n', False)
        # Getting the type of 'self' (line 111)
        self_24692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 47), 'self', False)
        # Obtaining the member 'on_event' of a type (line 111)
        on_event_24693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 47), self_24692, 'on_event')
        # Processing the call keyword arguments (line 111)
        kwargs_24694 = {}
        # Getting the type of 'self' (line 111)
        self_24687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'self', False)
        # Obtaining the member 'fig' of a type (line 111)
        fig_24688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 16), self_24687, 'fig')
        # Obtaining the member 'canvas' of a type (line 111)
        canvas_24689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 16), fig_24688, 'canvas')
        # Obtaining the member 'mpl_connect' of a type (line 111)
        mpl_connect_24690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 16), canvas_24689, 'mpl_connect')
        # Calling mpl_connect(args, kwargs) (line 111)
        mpl_connect_call_result_24695 = invoke(stypy.reporting.localization.Localization(__file__, 111, 16), mpl_connect_24690, *[n_24691, on_event_24693], **kwargs_24694)
        
        # Processing the call keyword arguments (line 110)
        kwargs_24696 = {}
        # Getting the type of 'self' (line 110)
        self_24684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'self', False)
        # Obtaining the member 'callbacks' of a type (line 110)
        callbacks_24685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), self_24684, 'callbacks')
        # Obtaining the member 'append' of a type (line 110)
        append_24686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), callbacks_24685, 'append')
        # Calling append(args, kwargs) (line 110)
        append_call_result_24697 = invoke(stypy.reporting.localization.Localization(__file__, 110, 12), append_24686, *[mpl_connect_call_result_24695], **kwargs_24696)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Try-finally block (line 113)
        
        # Call to start_event_loop(...): (line 115)
        # Processing the call keyword arguments (line 115)
        # Getting the type of 'timeout' (line 115)
        timeout_24702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 53), 'timeout', False)
        keyword_24703 = timeout_24702
        kwargs_24704 = {'timeout': keyword_24703}
        # Getting the type of 'self' (line 115)
        self_24698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'self', False)
        # Obtaining the member 'fig' of a type (line 115)
        fig_24699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), self_24698, 'fig')
        # Obtaining the member 'canvas' of a type (line 115)
        canvas_24700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), fig_24699, 'canvas')
        # Obtaining the member 'start_event_loop' of a type (line 115)
        start_event_loop_24701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), canvas_24700, 'start_event_loop')
        # Calling start_event_loop(args, kwargs) (line 115)
        start_event_loop_call_result_24705 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), start_event_loop_24701, *[], **kwargs_24704)
        
        
        # finally branch of the try-finally block (line 113)
        
        # Call to cleanup(...): (line 118)
        # Processing the call keyword arguments (line 118)
        kwargs_24708 = {}
        # Getting the type of 'self' (line 118)
        self_24706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'self', False)
        # Obtaining the member 'cleanup' of a type (line 118)
        cleanup_24707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 12), self_24706, 'cleanup')
        # Calling cleanup(args, kwargs) (line 118)
        cleanup_call_result_24709 = invoke(stypy.reporting.localization.Localization(__file__, 118, 12), cleanup_24707, *[], **kwargs_24708)
        
        
        # Getting the type of 'self' (line 121)
        self_24710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'self')
        # Obtaining the member 'events' of a type (line 121)
        events_24711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 15), self_24710, 'events')
        # Assigning a type to the variable 'stypy_return_type' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'stypy_return_type', events_24711)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 92)
        stypy_return_type_24712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24712)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_24712


# Assigning a type to the variable 'BlockingInput' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'BlockingInput', BlockingInput)

# Assigning a Attribute to a Attribute (line 90):
# Getting the type of 'BlockingInput'
BlockingInput_24713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BlockingInput')
# Obtaining the member 'pop_event' of a type
pop_event_24714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BlockingInput_24713, 'pop_event')
# Obtaining the member '__doc__' of a type (line 90)
doc___24715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 18), pop_event_24714, '__doc__')
# Getting the type of 'BlockingInput'
BlockingInput_24716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BlockingInput')
# Obtaining the member 'pop' of a type
pop_24717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BlockingInput_24716, 'pop')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), pop_24717, '__doc__', doc___24715)
# Declaration of the 'BlockingMouseInput' class
# Getting the type of 'BlockingInput' (line 124)
BlockingInput_24718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 25), 'BlockingInput')

class BlockingMouseInput(BlockingInput_24718, ):
    unicode_24719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, (-1)), 'unicode', u'\n    Class that creates a callable object to retrieve mouse clicks in a\n    blocking way.\n\n    This class will also retrieve keyboard clicks and treat them like\n    appropriate mouse clicks (delete and backspace are like mouse button 3,\n    enter is like mouse button 2 and all others are like mouse button 1).\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_24720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 38), 'int')
        int_24721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 51), 'int')
        int_24722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 65), 'int')
        defaults = [int_24720, int_24721, int_24722]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 138, 4, False)
        # Assigning a type to the variable 'self' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingMouseInput.__init__', ['fig', 'mouse_add', 'mouse_pop', 'mouse_stop'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['fig', 'mouse_add', 'mouse_pop', 'mouse_stop'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'self' (line 139)
        self_24725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 31), 'self', False)
        # Processing the call keyword arguments (line 139)
        # Getting the type of 'fig' (line 139)
        fig_24726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 41), 'fig', False)
        keyword_24727 = fig_24726
        
        # Obtaining an instance of the builtin type 'tuple' (line 140)
        tuple_24728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 140)
        # Adding element type (line 140)
        unicode_24729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 43), 'unicode', u'button_press_event')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 43), tuple_24728, unicode_24729)
        # Adding element type (line 140)
        unicode_24730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 43), 'unicode', u'key_press_event')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 43), tuple_24728, unicode_24730)
        
        keyword_24731 = tuple_24728
        kwargs_24732 = {'fig': keyword_24727, 'eventslist': keyword_24731}
        # Getting the type of 'BlockingInput' (line 139)
        BlockingInput_24723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'BlockingInput', False)
        # Obtaining the member '__init__' of a type (line 139)
        init___24724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), BlockingInput_24723, '__init__')
        # Calling __init__(args, kwargs) (line 139)
        init___call_result_24733 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), init___24724, *[self_24725], **kwargs_24732)
        
        
        # Assigning a Name to a Attribute (line 142):
        # Getting the type of 'mouse_add' (line 142)
        mouse_add_24734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 26), 'mouse_add')
        # Getting the type of 'self' (line 142)
        self_24735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'self')
        # Setting the type of the member 'button_add' of a type (line 142)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), self_24735, 'button_add', mouse_add_24734)
        
        # Assigning a Name to a Attribute (line 143):
        # Getting the type of 'mouse_pop' (line 143)
        mouse_pop_24736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 26), 'mouse_pop')
        # Getting the type of 'self' (line 143)
        self_24737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'self')
        # Setting the type of the member 'button_pop' of a type (line 143)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), self_24737, 'button_pop', mouse_pop_24736)
        
        # Assigning a Name to a Attribute (line 144):
        # Getting the type of 'mouse_stop' (line 144)
        mouse_stop_24738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 27), 'mouse_stop')
        # Getting the type of 'self' (line 144)
        self_24739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'self')
        # Setting the type of the member 'button_stop' of a type (line 144)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), self_24739, 'button_stop', mouse_stop_24738)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def post_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'post_event'
        module_type_store = module_type_store.open_function_context('post_event', 146, 4, False)
        # Assigning a type to the variable 'self' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingMouseInput.post_event.__dict__.__setitem__('stypy_localization', localization)
        BlockingMouseInput.post_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingMouseInput.post_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingMouseInput.post_event.__dict__.__setitem__('stypy_function_name', 'BlockingMouseInput.post_event')
        BlockingMouseInput.post_event.__dict__.__setitem__('stypy_param_names_list', [])
        BlockingMouseInput.post_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingMouseInput.post_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingMouseInput.post_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingMouseInput.post_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingMouseInput.post_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingMouseInput.post_event.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingMouseInput.post_event', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'post_event', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'post_event(...)' code ##################

        unicode_24740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, (-1)), 'unicode', u'\n        This will be called to process events\n        ')
        
        
        
        # Call to len(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'self' (line 150)
        self_24742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 15), 'self', False)
        # Obtaining the member 'events' of a type (line 150)
        events_24743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 15), self_24742, 'events')
        # Processing the call keyword arguments (line 150)
        kwargs_24744 = {}
        # Getting the type of 'len' (line 150)
        len_24741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'len', False)
        # Calling len(args, kwargs) (line 150)
        len_call_result_24745 = invoke(stypy.reporting.localization.Localization(__file__, 150, 11), len_24741, *[events_24743], **kwargs_24744)
        
        int_24746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 31), 'int')
        # Applying the binary operator '==' (line 150)
        result_eq_24747 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 11), '==', len_call_result_24745, int_24746)
        
        # Testing the type of an if condition (line 150)
        if_condition_24748 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 8), result_eq_24747)
        # Assigning a type to the variable 'if_condition_24748' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'if_condition_24748', if_condition_24748)
        # SSA begins for if statement (line 150)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 151)
        # Processing the call arguments (line 151)
        unicode_24751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 26), 'unicode', u'No events yet')
        # Processing the call keyword arguments (line 151)
        kwargs_24752 = {}
        # Getting the type of 'warnings' (line 151)
        warnings_24749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 151)
        warn_24750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 12), warnings_24749, 'warn')
        # Calling warn(args, kwargs) (line 151)
        warn_call_result_24753 = invoke(stypy.reporting.localization.Localization(__file__, 151, 12), warn_24750, *[unicode_24751], **kwargs_24752)
        
        # SSA branch for the else part of an if statement (line 150)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        int_24754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 25), 'int')
        # Getting the type of 'self' (line 152)
        self_24755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 13), 'self')
        # Obtaining the member 'events' of a type (line 152)
        events_24756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 13), self_24755, 'events')
        # Obtaining the member '__getitem__' of a type (line 152)
        getitem___24757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 13), events_24756, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 152)
        subscript_call_result_24758 = invoke(stypy.reporting.localization.Localization(__file__, 152, 13), getitem___24757, int_24754)
        
        # Obtaining the member 'name' of a type (line 152)
        name_24759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 13), subscript_call_result_24758, 'name')
        unicode_24760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 37), 'unicode', u'key_press_event')
        # Applying the binary operator '==' (line 152)
        result_eq_24761 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 13), '==', name_24759, unicode_24760)
        
        # Testing the type of an if condition (line 152)
        if_condition_24762 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 13), result_eq_24761)
        # Assigning a type to the variable 'if_condition_24762' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 13), 'if_condition_24762', if_condition_24762)
        # SSA begins for if statement (line 152)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to key_event(...): (line 153)
        # Processing the call keyword arguments (line 153)
        kwargs_24765 = {}
        # Getting the type of 'self' (line 153)
        self_24763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'self', False)
        # Obtaining the member 'key_event' of a type (line 153)
        key_event_24764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 12), self_24763, 'key_event')
        # Calling key_event(args, kwargs) (line 153)
        key_event_call_result_24766 = invoke(stypy.reporting.localization.Localization(__file__, 153, 12), key_event_24764, *[], **kwargs_24765)
        
        # SSA branch for the else part of an if statement (line 152)
        module_type_store.open_ssa_branch('else')
        
        # Call to mouse_event(...): (line 155)
        # Processing the call keyword arguments (line 155)
        kwargs_24769 = {}
        # Getting the type of 'self' (line 155)
        self_24767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'self', False)
        # Obtaining the member 'mouse_event' of a type (line 155)
        mouse_event_24768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 12), self_24767, 'mouse_event')
        # Calling mouse_event(args, kwargs) (line 155)
        mouse_event_call_result_24770 = invoke(stypy.reporting.localization.Localization(__file__, 155, 12), mouse_event_24768, *[], **kwargs_24769)
        
        # SSA join for if statement (line 152)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 150)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'post_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'post_event' in the type store
        # Getting the type of 'stypy_return_type' (line 146)
        stypy_return_type_24771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24771)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'post_event'
        return stypy_return_type_24771


    @norecursion
    def mouse_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mouse_event'
        module_type_store = module_type_store.open_function_context('mouse_event', 157, 4, False)
        # Assigning a type to the variable 'self' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingMouseInput.mouse_event.__dict__.__setitem__('stypy_localization', localization)
        BlockingMouseInput.mouse_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingMouseInput.mouse_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingMouseInput.mouse_event.__dict__.__setitem__('stypy_function_name', 'BlockingMouseInput.mouse_event')
        BlockingMouseInput.mouse_event.__dict__.__setitem__('stypy_param_names_list', [])
        BlockingMouseInput.mouse_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingMouseInput.mouse_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingMouseInput.mouse_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingMouseInput.mouse_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingMouseInput.mouse_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingMouseInput.mouse_event.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingMouseInput.mouse_event', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mouse_event', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mouse_event(...)' code ##################

        unicode_24772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 8), 'unicode', u'Process a mouse click event')
        
        # Assigning a Subscript to a Name (line 160):
        
        # Obtaining the type of the subscript
        int_24773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 28), 'int')
        # Getting the type of 'self' (line 160)
        self_24774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 16), 'self')
        # Obtaining the member 'events' of a type (line 160)
        events_24775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 16), self_24774, 'events')
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___24776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 16), events_24775, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_24777 = invoke(stypy.reporting.localization.Localization(__file__, 160, 16), getitem___24776, int_24773)
        
        # Assigning a type to the variable 'event' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'event', subscript_call_result_24777)
        
        # Assigning a Attribute to a Name (line 161):
        # Getting the type of 'event' (line 161)
        event_24778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 17), 'event')
        # Obtaining the member 'button' of a type (line 161)
        button_24779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 17), event_24778, 'button')
        # Assigning a type to the variable 'button' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'button', button_24779)
        
        
        # Getting the type of 'button' (line 163)
        button_24780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'button')
        # Getting the type of 'self' (line 163)
        self_24781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 21), 'self')
        # Obtaining the member 'button_pop' of a type (line 163)
        button_pop_24782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 21), self_24781, 'button_pop')
        # Applying the binary operator '==' (line 163)
        result_eq_24783 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 11), '==', button_24780, button_pop_24782)
        
        # Testing the type of an if condition (line 163)
        if_condition_24784 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 8), result_eq_24783)
        # Assigning a type to the variable 'if_condition_24784' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'if_condition_24784', if_condition_24784)
        # SSA begins for if statement (line 163)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to mouse_event_pop(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'event' (line 164)
        event_24787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 33), 'event', False)
        # Processing the call keyword arguments (line 164)
        kwargs_24788 = {}
        # Getting the type of 'self' (line 164)
        self_24785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'self', False)
        # Obtaining the member 'mouse_event_pop' of a type (line 164)
        mouse_event_pop_24786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 12), self_24785, 'mouse_event_pop')
        # Calling mouse_event_pop(args, kwargs) (line 164)
        mouse_event_pop_call_result_24789 = invoke(stypy.reporting.localization.Localization(__file__, 164, 12), mouse_event_pop_24786, *[event_24787], **kwargs_24788)
        
        # SSA branch for the else part of an if statement (line 163)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'button' (line 165)
        button_24790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 13), 'button')
        # Getting the type of 'self' (line 165)
        self_24791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 23), 'self')
        # Obtaining the member 'button_stop' of a type (line 165)
        button_stop_24792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 23), self_24791, 'button_stop')
        # Applying the binary operator '==' (line 165)
        result_eq_24793 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 13), '==', button_24790, button_stop_24792)
        
        # Testing the type of an if condition (line 165)
        if_condition_24794 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 13), result_eq_24793)
        # Assigning a type to the variable 'if_condition_24794' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 13), 'if_condition_24794', if_condition_24794)
        # SSA begins for if statement (line 165)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to mouse_event_stop(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'event' (line 166)
        event_24797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 34), 'event', False)
        # Processing the call keyword arguments (line 166)
        kwargs_24798 = {}
        # Getting the type of 'self' (line 166)
        self_24795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'self', False)
        # Obtaining the member 'mouse_event_stop' of a type (line 166)
        mouse_event_stop_24796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), self_24795, 'mouse_event_stop')
        # Calling mouse_event_stop(args, kwargs) (line 166)
        mouse_event_stop_call_result_24799 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), mouse_event_stop_24796, *[event_24797], **kwargs_24798)
        
        # SSA branch for the else part of an if statement (line 165)
        module_type_store.open_ssa_branch('else')
        
        # Call to mouse_event_add(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'event' (line 168)
        event_24802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 33), 'event', False)
        # Processing the call keyword arguments (line 168)
        kwargs_24803 = {}
        # Getting the type of 'self' (line 168)
        self_24800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'self', False)
        # Obtaining the member 'mouse_event_add' of a type (line 168)
        mouse_event_add_24801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 12), self_24800, 'mouse_event_add')
        # Calling mouse_event_add(args, kwargs) (line 168)
        mouse_event_add_call_result_24804 = invoke(stypy.reporting.localization.Localization(__file__, 168, 12), mouse_event_add_24801, *[event_24802], **kwargs_24803)
        
        # SSA join for if statement (line 165)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 163)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'mouse_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mouse_event' in the type store
        # Getting the type of 'stypy_return_type' (line 157)
        stypy_return_type_24805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24805)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mouse_event'
        return stypy_return_type_24805


    @norecursion
    def key_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'key_event'
        module_type_store = module_type_store.open_function_context('key_event', 170, 4, False)
        # Assigning a type to the variable 'self' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingMouseInput.key_event.__dict__.__setitem__('stypy_localization', localization)
        BlockingMouseInput.key_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingMouseInput.key_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingMouseInput.key_event.__dict__.__setitem__('stypy_function_name', 'BlockingMouseInput.key_event')
        BlockingMouseInput.key_event.__dict__.__setitem__('stypy_param_names_list', [])
        BlockingMouseInput.key_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingMouseInput.key_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingMouseInput.key_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingMouseInput.key_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingMouseInput.key_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingMouseInput.key_event.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingMouseInput.key_event', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'key_event', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'key_event(...)' code ##################

        unicode_24806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, (-1)), 'unicode', u'\n        Process a key click event.  This maps certain keys to appropriate\n        mouse click events.\n        ')
        
        # Assigning a Subscript to a Name (line 176):
        
        # Obtaining the type of the subscript
        int_24807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 28), 'int')
        # Getting the type of 'self' (line 176)
        self_24808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'self')
        # Obtaining the member 'events' of a type (line 176)
        events_24809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 16), self_24808, 'events')
        # Obtaining the member '__getitem__' of a type (line 176)
        getitem___24810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 16), events_24809, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 176)
        subscript_call_result_24811 = invoke(stypy.reporting.localization.Localization(__file__, 176, 16), getitem___24810, int_24807)
        
        # Assigning a type to the variable 'event' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'event', subscript_call_result_24811)
        
        # Type idiom detected: calculating its left and rigth part (line 177)
        # Getting the type of 'event' (line 177)
        event_24812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 'event')
        # Obtaining the member 'key' of a type (line 177)
        key_24813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 11), event_24812, 'key')
        # Getting the type of 'None' (line 177)
        None_24814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 24), 'None')
        
        (may_be_24815, more_types_in_union_24816) = may_be_none(key_24813, None_24814)

        if may_be_24815:

            if more_types_in_union_24816:
                # Runtime conditional SSA (line 177)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 179)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_24816:
                # SSA join for if statement (line 177)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 181):
        
        # Call to lower(...): (line 181)
        # Processing the call keyword arguments (line 181)
        kwargs_24820 = {}
        # Getting the type of 'event' (line 181)
        event_24817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 14), 'event', False)
        # Obtaining the member 'key' of a type (line 181)
        key_24818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 14), event_24817, 'key')
        # Obtaining the member 'lower' of a type (line 181)
        lower_24819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 14), key_24818, 'lower')
        # Calling lower(args, kwargs) (line 181)
        lower_call_result_24821 = invoke(stypy.reporting.localization.Localization(__file__, 181, 14), lower_24819, *[], **kwargs_24820)
        
        # Assigning a type to the variable 'key' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'key', lower_call_result_24821)
        
        
        # Getting the type of 'key' (line 183)
        key_24822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 11), 'key')
        
        # Obtaining an instance of the builtin type 'list' (line 183)
        list_24823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 183)
        # Adding element type (line 183)
        unicode_24824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 19), 'unicode', u'backspace')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 18), list_24823, unicode_24824)
        # Adding element type (line 183)
        unicode_24825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 32), 'unicode', u'delete')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 18), list_24823, unicode_24825)
        
        # Applying the binary operator 'in' (line 183)
        result_contains_24826 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 11), 'in', key_24822, list_24823)
        
        # Testing the type of an if condition (line 183)
        if_condition_24827 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 183, 8), result_contains_24826)
        # Assigning a type to the variable 'if_condition_24827' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'if_condition_24827', if_condition_24827)
        # SSA begins for if statement (line 183)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to mouse_event_pop(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'event' (line 184)
        event_24830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 33), 'event', False)
        # Processing the call keyword arguments (line 184)
        kwargs_24831 = {}
        # Getting the type of 'self' (line 184)
        self_24828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'self', False)
        # Obtaining the member 'mouse_event_pop' of a type (line 184)
        mouse_event_pop_24829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 12), self_24828, 'mouse_event_pop')
        # Calling mouse_event_pop(args, kwargs) (line 184)
        mouse_event_pop_call_result_24832 = invoke(stypy.reporting.localization.Localization(__file__, 184, 12), mouse_event_pop_24829, *[event_24830], **kwargs_24831)
        
        # SSA branch for the else part of an if statement (line 183)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'key' (line 185)
        key_24833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 13), 'key')
        
        # Obtaining an instance of the builtin type 'list' (line 185)
        list_24834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 185)
        # Adding element type (line 185)
        unicode_24835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 21), 'unicode', u'escape')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 20), list_24834, unicode_24835)
        # Adding element type (line 185)
        unicode_24836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 31), 'unicode', u'enter')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 20), list_24834, unicode_24836)
        
        # Applying the binary operator 'in' (line 185)
        result_contains_24837 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 13), 'in', key_24833, list_24834)
        
        # Testing the type of an if condition (line 185)
        if_condition_24838 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 13), result_contains_24837)
        # Assigning a type to the variable 'if_condition_24838' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 13), 'if_condition_24838', if_condition_24838)
        # SSA begins for if statement (line 185)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to mouse_event_stop(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'event' (line 187)
        event_24841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 34), 'event', False)
        # Processing the call keyword arguments (line 187)
        kwargs_24842 = {}
        # Getting the type of 'self' (line 187)
        self_24839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'self', False)
        # Obtaining the member 'mouse_event_stop' of a type (line 187)
        mouse_event_stop_24840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 12), self_24839, 'mouse_event_stop')
        # Calling mouse_event_stop(args, kwargs) (line 187)
        mouse_event_stop_call_result_24843 = invoke(stypy.reporting.localization.Localization(__file__, 187, 12), mouse_event_stop_24840, *[event_24841], **kwargs_24842)
        
        # SSA branch for the else part of an if statement (line 185)
        module_type_store.open_ssa_branch('else')
        
        # Call to mouse_event_add(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'event' (line 189)
        event_24846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 33), 'event', False)
        # Processing the call keyword arguments (line 189)
        kwargs_24847 = {}
        # Getting the type of 'self' (line 189)
        self_24844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'self', False)
        # Obtaining the member 'mouse_event_add' of a type (line 189)
        mouse_event_add_24845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 12), self_24844, 'mouse_event_add')
        # Calling mouse_event_add(args, kwargs) (line 189)
        mouse_event_add_call_result_24848 = invoke(stypy.reporting.localization.Localization(__file__, 189, 12), mouse_event_add_24845, *[event_24846], **kwargs_24847)
        
        # SSA join for if statement (line 185)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 183)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'key_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'key_event' in the type store
        # Getting the type of 'stypy_return_type' (line 170)
        stypy_return_type_24849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24849)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'key_event'
        return stypy_return_type_24849


    @norecursion
    def mouse_event_add(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mouse_event_add'
        module_type_store = module_type_store.open_function_context('mouse_event_add', 191, 4, False)
        # Assigning a type to the variable 'self' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingMouseInput.mouse_event_add.__dict__.__setitem__('stypy_localization', localization)
        BlockingMouseInput.mouse_event_add.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingMouseInput.mouse_event_add.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingMouseInput.mouse_event_add.__dict__.__setitem__('stypy_function_name', 'BlockingMouseInput.mouse_event_add')
        BlockingMouseInput.mouse_event_add.__dict__.__setitem__('stypy_param_names_list', ['event'])
        BlockingMouseInput.mouse_event_add.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingMouseInput.mouse_event_add.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingMouseInput.mouse_event_add.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingMouseInput.mouse_event_add.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingMouseInput.mouse_event_add.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingMouseInput.mouse_event_add.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingMouseInput.mouse_event_add', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mouse_event_add', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mouse_event_add(...)' code ##################

        unicode_24850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, (-1)), 'unicode', u'\n        Will be called for any event involving a button other than\n        button 2 or 3.  This will add a click if it is inside axes.\n        ')
        
        # Getting the type of 'event' (line 196)
        event_24851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 11), 'event')
        # Obtaining the member 'inaxes' of a type (line 196)
        inaxes_24852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 11), event_24851, 'inaxes')
        # Testing the type of an if condition (line 196)
        if_condition_24853 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 8), inaxes_24852)
        # Assigning a type to the variable 'if_condition_24853' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'if_condition_24853', if_condition_24853)
        # SSA begins for if statement (line 196)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to add_click(...): (line 197)
        # Processing the call arguments (line 197)
        # Getting the type of 'event' (line 197)
        event_24856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 27), 'event', False)
        # Processing the call keyword arguments (line 197)
        kwargs_24857 = {}
        # Getting the type of 'self' (line 197)
        self_24854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'self', False)
        # Obtaining the member 'add_click' of a type (line 197)
        add_click_24855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 12), self_24854, 'add_click')
        # Calling add_click(args, kwargs) (line 197)
        add_click_call_result_24858 = invoke(stypy.reporting.localization.Localization(__file__, 197, 12), add_click_24855, *[event_24856], **kwargs_24857)
        
        # SSA branch for the else part of an if statement (line 196)
        module_type_store.open_ssa_branch('else')
        
        # Call to pop(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'self' (line 199)
        self_24861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 30), 'self', False)
        int_24862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 36), 'int')
        # Processing the call keyword arguments (line 199)
        kwargs_24863 = {}
        # Getting the type of 'BlockingInput' (line 199)
        BlockingInput_24859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'BlockingInput', False)
        # Obtaining the member 'pop' of a type (line 199)
        pop_24860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 12), BlockingInput_24859, 'pop')
        # Calling pop(args, kwargs) (line 199)
        pop_call_result_24864 = invoke(stypy.reporting.localization.Localization(__file__, 199, 12), pop_24860, *[self_24861, int_24862], **kwargs_24863)
        
        # SSA join for if statement (line 196)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'mouse_event_add(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mouse_event_add' in the type store
        # Getting the type of 'stypy_return_type' (line 191)
        stypy_return_type_24865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24865)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mouse_event_add'
        return stypy_return_type_24865


    @norecursion
    def mouse_event_stop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mouse_event_stop'
        module_type_store = module_type_store.open_function_context('mouse_event_stop', 201, 4, False)
        # Assigning a type to the variable 'self' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingMouseInput.mouse_event_stop.__dict__.__setitem__('stypy_localization', localization)
        BlockingMouseInput.mouse_event_stop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingMouseInput.mouse_event_stop.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingMouseInput.mouse_event_stop.__dict__.__setitem__('stypy_function_name', 'BlockingMouseInput.mouse_event_stop')
        BlockingMouseInput.mouse_event_stop.__dict__.__setitem__('stypy_param_names_list', ['event'])
        BlockingMouseInput.mouse_event_stop.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingMouseInput.mouse_event_stop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingMouseInput.mouse_event_stop.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingMouseInput.mouse_event_stop.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingMouseInput.mouse_event_stop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingMouseInput.mouse_event_stop.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingMouseInput.mouse_event_stop', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mouse_event_stop', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mouse_event_stop(...)' code ##################

        unicode_24866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, (-1)), 'unicode', u'\n        Will be called for any event involving button 2.\n        Button 2 ends blocking input.\n        ')
        
        # Call to pop(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'self' (line 208)
        self_24869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 26), 'self', False)
        int_24870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 32), 'int')
        # Processing the call keyword arguments (line 208)
        kwargs_24871 = {}
        # Getting the type of 'BlockingInput' (line 208)
        BlockingInput_24867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'BlockingInput', False)
        # Obtaining the member 'pop' of a type (line 208)
        pop_24868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), BlockingInput_24867, 'pop')
        # Calling pop(args, kwargs) (line 208)
        pop_call_result_24872 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), pop_24868, *[self_24869, int_24870], **kwargs_24871)
        
        
        # Call to stop_event_loop(...): (line 214)
        # Processing the call keyword arguments (line 214)
        kwargs_24877 = {}
        # Getting the type of 'self' (line 214)
        self_24873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'self', False)
        # Obtaining the member 'fig' of a type (line 214)
        fig_24874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), self_24873, 'fig')
        # Obtaining the member 'canvas' of a type (line 214)
        canvas_24875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), fig_24874, 'canvas')
        # Obtaining the member 'stop_event_loop' of a type (line 214)
        stop_event_loop_24876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), canvas_24875, 'stop_event_loop')
        # Calling stop_event_loop(args, kwargs) (line 214)
        stop_event_loop_call_result_24878 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), stop_event_loop_24876, *[], **kwargs_24877)
        
        
        # ################# End of 'mouse_event_stop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mouse_event_stop' in the type store
        # Getting the type of 'stypy_return_type' (line 201)
        stypy_return_type_24879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24879)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mouse_event_stop'
        return stypy_return_type_24879


    @norecursion
    def mouse_event_pop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mouse_event_pop'
        module_type_store = module_type_store.open_function_context('mouse_event_pop', 216, 4, False)
        # Assigning a type to the variable 'self' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingMouseInput.mouse_event_pop.__dict__.__setitem__('stypy_localization', localization)
        BlockingMouseInput.mouse_event_pop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingMouseInput.mouse_event_pop.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingMouseInput.mouse_event_pop.__dict__.__setitem__('stypy_function_name', 'BlockingMouseInput.mouse_event_pop')
        BlockingMouseInput.mouse_event_pop.__dict__.__setitem__('stypy_param_names_list', ['event'])
        BlockingMouseInput.mouse_event_pop.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingMouseInput.mouse_event_pop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingMouseInput.mouse_event_pop.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingMouseInput.mouse_event_pop.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingMouseInput.mouse_event_pop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingMouseInput.mouse_event_pop.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingMouseInput.mouse_event_pop', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mouse_event_pop', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mouse_event_pop(...)' code ##################

        unicode_24880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, (-1)), 'unicode', u'\n        Will be called for any event involving button 3.\n        Button 3 removes the last click.\n        ')
        
        # Call to pop(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'self' (line 222)
        self_24883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 26), 'self', False)
        int_24884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 32), 'int')
        # Processing the call keyword arguments (line 222)
        kwargs_24885 = {}
        # Getting the type of 'BlockingInput' (line 222)
        BlockingInput_24881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'BlockingInput', False)
        # Obtaining the member 'pop' of a type (line 222)
        pop_24882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), BlockingInput_24881, 'pop')
        # Calling pop(args, kwargs) (line 222)
        pop_call_result_24886 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), pop_24882, *[self_24883, int_24884], **kwargs_24885)
        
        
        
        
        # Call to len(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'self' (line 225)
        self_24888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 15), 'self', False)
        # Obtaining the member 'events' of a type (line 225)
        events_24889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 15), self_24888, 'events')
        # Processing the call keyword arguments (line 225)
        kwargs_24890 = {}
        # Getting the type of 'len' (line 225)
        len_24887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 11), 'len', False)
        # Calling len(args, kwargs) (line 225)
        len_call_result_24891 = invoke(stypy.reporting.localization.Localization(__file__, 225, 11), len_24887, *[events_24889], **kwargs_24890)
        
        int_24892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 30), 'int')
        # Applying the binary operator '>' (line 225)
        result_gt_24893 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 11), '>', len_call_result_24891, int_24892)
        
        # Testing the type of an if condition (line 225)
        if_condition_24894 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 225, 8), result_gt_24893)
        # Assigning a type to the variable 'if_condition_24894' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'if_condition_24894', if_condition_24894)
        # SSA begins for if statement (line 225)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to pop(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'event' (line 226)
        event_24897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 21), 'event', False)
        int_24898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 28), 'int')
        # Processing the call keyword arguments (line 226)
        kwargs_24899 = {}
        # Getting the type of 'self' (line 226)
        self_24895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'self', False)
        # Obtaining the member 'pop' of a type (line 226)
        pop_24896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 12), self_24895, 'pop')
        # Calling pop(args, kwargs) (line 226)
        pop_call_result_24900 = invoke(stypy.reporting.localization.Localization(__file__, 226, 12), pop_24896, *[event_24897, int_24898], **kwargs_24899)
        
        # SSA join for if statement (line 225)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'mouse_event_pop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mouse_event_pop' in the type store
        # Getting the type of 'stypy_return_type' (line 216)
        stypy_return_type_24901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24901)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mouse_event_pop'
        return stypy_return_type_24901


    @norecursion
    def add_click(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_click'
        module_type_store = module_type_store.open_function_context('add_click', 228, 4, False)
        # Assigning a type to the variable 'self' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingMouseInput.add_click.__dict__.__setitem__('stypy_localization', localization)
        BlockingMouseInput.add_click.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingMouseInput.add_click.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingMouseInput.add_click.__dict__.__setitem__('stypy_function_name', 'BlockingMouseInput.add_click')
        BlockingMouseInput.add_click.__dict__.__setitem__('stypy_param_names_list', ['event'])
        BlockingMouseInput.add_click.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingMouseInput.add_click.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingMouseInput.add_click.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingMouseInput.add_click.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingMouseInput.add_click.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingMouseInput.add_click.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingMouseInput.add_click', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_click', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_click(...)' code ##################

        unicode_24902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, (-1)), 'unicode', u'\n        This add the coordinates of an event to the list of clicks\n        ')
        
        # Call to append(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Obtaining an instance of the builtin type 'tuple' (line 232)
        tuple_24906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 232)
        # Adding element type (line 232)
        # Getting the type of 'event' (line 232)
        event_24907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 28), 'event', False)
        # Obtaining the member 'xdata' of a type (line 232)
        xdata_24908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 28), event_24907, 'xdata')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 28), tuple_24906, xdata_24908)
        # Adding element type (line 232)
        # Getting the type of 'event' (line 232)
        event_24909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 41), 'event', False)
        # Obtaining the member 'ydata' of a type (line 232)
        ydata_24910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 41), event_24909, 'ydata')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 28), tuple_24906, ydata_24910)
        
        # Processing the call keyword arguments (line 232)
        kwargs_24911 = {}
        # Getting the type of 'self' (line 232)
        self_24903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'self', False)
        # Obtaining the member 'clicks' of a type (line 232)
        clicks_24904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 8), self_24903, 'clicks')
        # Obtaining the member 'append' of a type (line 232)
        append_24905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 8), clicks_24904, 'append')
        # Calling append(args, kwargs) (line 232)
        append_call_result_24912 = invoke(stypy.reporting.localization.Localization(__file__, 232, 8), append_24905, *[tuple_24906], **kwargs_24911)
        
        
        # Call to report(...): (line 234)
        # Processing the call arguments (line 234)
        unicode_24915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 23), 'unicode', u'input %i: %f,%f')
        
        # Obtaining an instance of the builtin type 'tuple' (line 235)
        tuple_24916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 235)
        # Adding element type (line 235)
        
        # Call to len(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'self' (line 235)
        self_24918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 28), 'self', False)
        # Obtaining the member 'clicks' of a type (line 235)
        clicks_24919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 28), self_24918, 'clicks')
        # Processing the call keyword arguments (line 235)
        kwargs_24920 = {}
        # Getting the type of 'len' (line 235)
        len_24917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 24), 'len', False)
        # Calling len(args, kwargs) (line 235)
        len_call_result_24921 = invoke(stypy.reporting.localization.Localization(__file__, 235, 24), len_24917, *[clicks_24919], **kwargs_24920)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 24), tuple_24916, len_call_result_24921)
        # Adding element type (line 235)
        # Getting the type of 'event' (line 235)
        event_24922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 42), 'event', False)
        # Obtaining the member 'xdata' of a type (line 235)
        xdata_24923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 42), event_24922, 'xdata')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 24), tuple_24916, xdata_24923)
        # Adding element type (line 235)
        # Getting the type of 'event' (line 235)
        event_24924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 55), 'event', False)
        # Obtaining the member 'ydata' of a type (line 235)
        ydata_24925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 55), event_24924, 'ydata')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 24), tuple_24916, ydata_24925)
        
        # Applying the binary operator '%' (line 234)
        result_mod_24926 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 23), '%', unicode_24915, tuple_24916)
        
        # Processing the call keyword arguments (line 234)
        kwargs_24927 = {}
        # Getting the type of 'verbose' (line 234)
        verbose_24913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'verbose', False)
        # Obtaining the member 'report' of a type (line 234)
        report_24914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), verbose_24913, 'report')
        # Calling report(args, kwargs) (line 234)
        report_call_result_24928 = invoke(stypy.reporting.localization.Localization(__file__, 234, 8), report_24914, *[result_mod_24926], **kwargs_24927)
        
        
        # Getting the type of 'self' (line 238)
        self_24929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 11), 'self')
        # Obtaining the member 'show_clicks' of a type (line 238)
        show_clicks_24930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 11), self_24929, 'show_clicks')
        # Testing the type of an if condition (line 238)
        if_condition_24931 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 238, 8), show_clicks_24930)
        # Assigning a type to the variable 'if_condition_24931' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'if_condition_24931', if_condition_24931)
        # SSA begins for if statement (line 238)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 239):
        
        # Call to Line2D(...): (line 239)
        # Processing the call arguments (line 239)
        
        # Obtaining an instance of the builtin type 'list' (line 239)
        list_24934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 239)
        # Adding element type (line 239)
        # Getting the type of 'event' (line 239)
        event_24935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 34), 'event', False)
        # Obtaining the member 'xdata' of a type (line 239)
        xdata_24936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 34), event_24935, 'xdata')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 33), list_24934, xdata_24936)
        
        
        # Obtaining an instance of the builtin type 'list' (line 239)
        list_24937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 239)
        # Adding element type (line 239)
        # Getting the type of 'event' (line 239)
        event_24938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 49), 'event', False)
        # Obtaining the member 'ydata' of a type (line 239)
        ydata_24939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 49), event_24938, 'ydata')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 48), list_24937, ydata_24939)
        
        # Processing the call keyword arguments (line 239)
        unicode_24940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 40), 'unicode', u'+')
        keyword_24941 = unicode_24940
        unicode_24942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 51), 'unicode', u'r')
        keyword_24943 = unicode_24942
        kwargs_24944 = {'marker': keyword_24941, 'color': keyword_24943}
        # Getting the type of 'mlines' (line 239)
        mlines_24932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 19), 'mlines', False)
        # Obtaining the member 'Line2D' of a type (line 239)
        Line2D_24933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 19), mlines_24932, 'Line2D')
        # Calling Line2D(args, kwargs) (line 239)
        Line2D_call_result_24945 = invoke(stypy.reporting.localization.Localization(__file__, 239, 19), Line2D_24933, *[list_24934, list_24937], **kwargs_24944)
        
        # Assigning a type to the variable 'line' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'line', Line2D_call_result_24945)
        
        # Call to add_line(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'line' (line 241)
        line_24949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 34), 'line', False)
        # Processing the call keyword arguments (line 241)
        kwargs_24950 = {}
        # Getting the type of 'event' (line 241)
        event_24946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'event', False)
        # Obtaining the member 'inaxes' of a type (line 241)
        inaxes_24947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), event_24946, 'inaxes')
        # Obtaining the member 'add_line' of a type (line 241)
        add_line_24948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), inaxes_24947, 'add_line')
        # Calling add_line(args, kwargs) (line 241)
        add_line_call_result_24951 = invoke(stypy.reporting.localization.Localization(__file__, 241, 12), add_line_24948, *[line_24949], **kwargs_24950)
        
        
        # Call to append(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'line' (line 242)
        line_24955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 30), 'line', False)
        # Processing the call keyword arguments (line 242)
        kwargs_24956 = {}
        # Getting the type of 'self' (line 242)
        self_24952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'self', False)
        # Obtaining the member 'marks' of a type (line 242)
        marks_24953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 12), self_24952, 'marks')
        # Obtaining the member 'append' of a type (line 242)
        append_24954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 12), marks_24953, 'append')
        # Calling append(args, kwargs) (line 242)
        append_call_result_24957 = invoke(stypy.reporting.localization.Localization(__file__, 242, 12), append_24954, *[line_24955], **kwargs_24956)
        
        
        # Call to draw(...): (line 243)
        # Processing the call keyword arguments (line 243)
        kwargs_24962 = {}
        # Getting the type of 'self' (line 243)
        self_24958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'self', False)
        # Obtaining the member 'fig' of a type (line 243)
        fig_24959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 12), self_24958, 'fig')
        # Obtaining the member 'canvas' of a type (line 243)
        canvas_24960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 12), fig_24959, 'canvas')
        # Obtaining the member 'draw' of a type (line 243)
        draw_24961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 12), canvas_24960, 'draw')
        # Calling draw(args, kwargs) (line 243)
        draw_call_result_24963 = invoke(stypy.reporting.localization.Localization(__file__, 243, 12), draw_24961, *[], **kwargs_24962)
        
        # SSA join for if statement (line 238)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'add_click(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_click' in the type store
        # Getting the type of 'stypy_return_type' (line 228)
        stypy_return_type_24964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24964)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_click'
        return stypy_return_type_24964


    @norecursion
    def pop_click(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_24965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 37), 'int')
        defaults = [int_24965]
        # Create a new context for function 'pop_click'
        module_type_store = module_type_store.open_function_context('pop_click', 245, 4, False)
        # Assigning a type to the variable 'self' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingMouseInput.pop_click.__dict__.__setitem__('stypy_localization', localization)
        BlockingMouseInput.pop_click.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingMouseInput.pop_click.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingMouseInput.pop_click.__dict__.__setitem__('stypy_function_name', 'BlockingMouseInput.pop_click')
        BlockingMouseInput.pop_click.__dict__.__setitem__('stypy_param_names_list', ['event', 'index'])
        BlockingMouseInput.pop_click.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingMouseInput.pop_click.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingMouseInput.pop_click.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingMouseInput.pop_click.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingMouseInput.pop_click.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingMouseInput.pop_click.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingMouseInput.pop_click', ['event', 'index'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pop_click', localization, ['event', 'index'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pop_click(...)' code ##################

        unicode_24966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, (-1)), 'unicode', u'\n        This removes a click from the list of clicks.  Defaults to\n        removing the last click.\n        ')
        
        # Call to pop(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'index' (line 250)
        index_24970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 24), 'index', False)
        # Processing the call keyword arguments (line 250)
        kwargs_24971 = {}
        # Getting the type of 'self' (line 250)
        self_24967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'self', False)
        # Obtaining the member 'clicks' of a type (line 250)
        clicks_24968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), self_24967, 'clicks')
        # Obtaining the member 'pop' of a type (line 250)
        pop_24969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), clicks_24968, 'pop')
        # Calling pop(args, kwargs) (line 250)
        pop_call_result_24972 = invoke(stypy.reporting.localization.Localization(__file__, 250, 8), pop_24969, *[index_24970], **kwargs_24971)
        
        
        # Getting the type of 'self' (line 252)
        self_24973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 11), 'self')
        # Obtaining the member 'show_clicks' of a type (line 252)
        show_clicks_24974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 11), self_24973, 'show_clicks')
        # Testing the type of an if condition (line 252)
        if_condition_24975 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 252, 8), show_clicks_24974)
        # Assigning a type to the variable 'if_condition_24975' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'if_condition_24975', if_condition_24975)
        # SSA begins for if statement (line 252)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 254):
        
        # Call to pop(...): (line 254)
        # Processing the call arguments (line 254)
        # Getting the type of 'index' (line 254)
        index_24979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 34), 'index', False)
        # Processing the call keyword arguments (line 254)
        kwargs_24980 = {}
        # Getting the type of 'self' (line 254)
        self_24976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 19), 'self', False)
        # Obtaining the member 'marks' of a type (line 254)
        marks_24977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 19), self_24976, 'marks')
        # Obtaining the member 'pop' of a type (line 254)
        pop_24978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 19), marks_24977, 'pop')
        # Calling pop(args, kwargs) (line 254)
        pop_call_result_24981 = invoke(stypy.reporting.localization.Localization(__file__, 254, 19), pop_24978, *[index_24979], **kwargs_24980)
        
        # Assigning a type to the variable 'mark' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'mark', pop_call_result_24981)
        
        # Call to remove(...): (line 255)
        # Processing the call keyword arguments (line 255)
        kwargs_24984 = {}
        # Getting the type of 'mark' (line 255)
        mark_24982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'mark', False)
        # Obtaining the member 'remove' of a type (line 255)
        remove_24983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 12), mark_24982, 'remove')
        # Calling remove(args, kwargs) (line 255)
        remove_call_result_24985 = invoke(stypy.reporting.localization.Localization(__file__, 255, 12), remove_24983, *[], **kwargs_24984)
        
        
        # Call to draw(...): (line 257)
        # Processing the call keyword arguments (line 257)
        kwargs_24990 = {}
        # Getting the type of 'self' (line 257)
        self_24986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'self', False)
        # Obtaining the member 'fig' of a type (line 257)
        fig_24987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 12), self_24986, 'fig')
        # Obtaining the member 'canvas' of a type (line 257)
        canvas_24988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 12), fig_24987, 'canvas')
        # Obtaining the member 'draw' of a type (line 257)
        draw_24989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 12), canvas_24988, 'draw')
        # Calling draw(args, kwargs) (line 257)
        draw_call_result_24991 = invoke(stypy.reporting.localization.Localization(__file__, 257, 12), draw_24989, *[], **kwargs_24990)
        
        # SSA join for if statement (line 252)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'pop_click(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pop_click' in the type store
        # Getting the type of 'stypy_return_type' (line 245)
        stypy_return_type_24992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24992)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pop_click'
        return stypy_return_type_24992


    @norecursion
    def pop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_24993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 31), 'int')
        defaults = [int_24993]
        # Create a new context for function 'pop'
        module_type_store = module_type_store.open_function_context('pop', 262, 4, False)
        # Assigning a type to the variable 'self' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingMouseInput.pop.__dict__.__setitem__('stypy_localization', localization)
        BlockingMouseInput.pop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingMouseInput.pop.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingMouseInput.pop.__dict__.__setitem__('stypy_function_name', 'BlockingMouseInput.pop')
        BlockingMouseInput.pop.__dict__.__setitem__('stypy_param_names_list', ['event', 'index'])
        BlockingMouseInput.pop.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingMouseInput.pop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingMouseInput.pop.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingMouseInput.pop.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingMouseInput.pop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingMouseInput.pop.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingMouseInput.pop', ['event', 'index'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pop', localization, ['event', 'index'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pop(...)' code ##################

        unicode_24994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, (-1)), 'unicode', u'\n        This removes a click and the associated event from the object.\n        Defaults to removing the last click, but any index can be\n        supplied.\n        ')
        
        # Call to pop_click(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'event' (line 268)
        event_24997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 23), 'event', False)
        # Getting the type of 'index' (line 268)
        index_24998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 30), 'index', False)
        # Processing the call keyword arguments (line 268)
        kwargs_24999 = {}
        # Getting the type of 'self' (line 268)
        self_24995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'self', False)
        # Obtaining the member 'pop_click' of a type (line 268)
        pop_click_24996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), self_24995, 'pop_click')
        # Calling pop_click(args, kwargs) (line 268)
        pop_click_call_result_25000 = invoke(stypy.reporting.localization.Localization(__file__, 268, 8), pop_click_24996, *[event_24997, index_24998], **kwargs_24999)
        
        
        # Call to pop(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'self' (line 269)
        self_25003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 26), 'self', False)
        # Getting the type of 'index' (line 269)
        index_25004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 32), 'index', False)
        # Processing the call keyword arguments (line 269)
        kwargs_25005 = {}
        # Getting the type of 'BlockingInput' (line 269)
        BlockingInput_25001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'BlockingInput', False)
        # Obtaining the member 'pop' of a type (line 269)
        pop_25002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 8), BlockingInput_25001, 'pop')
        # Calling pop(args, kwargs) (line 269)
        pop_call_result_25006 = invoke(stypy.reporting.localization.Localization(__file__, 269, 8), pop_25002, *[self_25003, index_25004], **kwargs_25005)
        
        
        # ################# End of 'pop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pop' in the type store
        # Getting the type of 'stypy_return_type' (line 262)
        stypy_return_type_25007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25007)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pop'
        return stypy_return_type_25007


    @norecursion
    def cleanup(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 271)
        None_25008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 28), 'None')
        defaults = [None_25008]
        # Create a new context for function 'cleanup'
        module_type_store = module_type_store.open_function_context('cleanup', 271, 4, False)
        # Assigning a type to the variable 'self' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingMouseInput.cleanup.__dict__.__setitem__('stypy_localization', localization)
        BlockingMouseInput.cleanup.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingMouseInput.cleanup.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingMouseInput.cleanup.__dict__.__setitem__('stypy_function_name', 'BlockingMouseInput.cleanup')
        BlockingMouseInput.cleanup.__dict__.__setitem__('stypy_param_names_list', ['event'])
        BlockingMouseInput.cleanup.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingMouseInput.cleanup.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingMouseInput.cleanup.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingMouseInput.cleanup.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingMouseInput.cleanup.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingMouseInput.cleanup.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingMouseInput.cleanup', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cleanup', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cleanup(...)' code ##################

        
        # Getting the type of 'self' (line 273)
        self_25009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 11), 'self')
        # Obtaining the member 'show_clicks' of a type (line 273)
        show_clicks_25010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 11), self_25009, 'show_clicks')
        # Testing the type of an if condition (line 273)
        if_condition_25011 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 8), show_clicks_25010)
        # Assigning a type to the variable 'if_condition_25011' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'if_condition_25011', if_condition_25011)
        # SSA begins for if statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 275)
        self_25012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 24), 'self')
        # Obtaining the member 'marks' of a type (line 275)
        marks_25013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 24), self_25012, 'marks')
        # Testing the type of a for loop iterable (line 275)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 275, 12), marks_25013)
        # Getting the type of the for loop variable (line 275)
        for_loop_var_25014 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 275, 12), marks_25013)
        # Assigning a type to the variable 'mark' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'mark', for_loop_var_25014)
        # SSA begins for a for statement (line 275)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to remove(...): (line 276)
        # Processing the call keyword arguments (line 276)
        kwargs_25017 = {}
        # Getting the type of 'mark' (line 276)
        mark_25015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 16), 'mark', False)
        # Obtaining the member 'remove' of a type (line 276)
        remove_25016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 16), mark_25015, 'remove')
        # Calling remove(args, kwargs) (line 276)
        remove_call_result_25018 = invoke(stypy.reporting.localization.Localization(__file__, 276, 16), remove_25016, *[], **kwargs_25017)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Attribute (line 277):
        
        # Obtaining an instance of the builtin type 'list' (line 277)
        list_25019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 277)
        
        # Getting the type of 'self' (line 277)
        self_25020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'self')
        # Setting the type of the member 'marks' of a type (line 277)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 12), self_25020, 'marks', list_25019)
        
        # Call to draw(...): (line 279)
        # Processing the call keyword arguments (line 279)
        kwargs_25025 = {}
        # Getting the type of 'self' (line 279)
        self_25021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'self', False)
        # Obtaining the member 'fig' of a type (line 279)
        fig_25022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 12), self_25021, 'fig')
        # Obtaining the member 'canvas' of a type (line 279)
        canvas_25023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 12), fig_25022, 'canvas')
        # Obtaining the member 'draw' of a type (line 279)
        draw_25024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 12), canvas_25023, 'draw')
        # Calling draw(args, kwargs) (line 279)
        draw_call_result_25026 = invoke(stypy.reporting.localization.Localization(__file__, 279, 12), draw_25024, *[], **kwargs_25025)
        
        # SSA join for if statement (line 273)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to cleanup(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'self' (line 282)
        self_25029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 30), 'self', False)
        # Processing the call keyword arguments (line 282)
        kwargs_25030 = {}
        # Getting the type of 'BlockingInput' (line 282)
        BlockingInput_25027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'BlockingInput', False)
        # Obtaining the member 'cleanup' of a type (line 282)
        cleanup_25028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 8), BlockingInput_25027, 'cleanup')
        # Calling cleanup(args, kwargs) (line 282)
        cleanup_call_result_25031 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), cleanup_25028, *[self_25029], **kwargs_25030)
        
        
        # ################# End of 'cleanup(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cleanup' in the type store
        # Getting the type of 'stypy_return_type' (line 271)
        stypy_return_type_25032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25032)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cleanup'
        return stypy_return_type_25032


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_25033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 25), 'int')
        int_25034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 36), 'int')
        # Getting the type of 'True' (line 284)
        True_25035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 52), 'True')
        defaults = [int_25033, int_25034, True_25035]
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 284, 4, False)
        # Assigning a type to the variable 'self' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingMouseInput.__call__.__dict__.__setitem__('stypy_localization', localization)
        BlockingMouseInput.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingMouseInput.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingMouseInput.__call__.__dict__.__setitem__('stypy_function_name', 'BlockingMouseInput.__call__')
        BlockingMouseInput.__call__.__dict__.__setitem__('stypy_param_names_list', ['n', 'timeout', 'show_clicks'])
        BlockingMouseInput.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingMouseInput.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingMouseInput.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingMouseInput.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingMouseInput.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingMouseInput.__call__.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingMouseInput.__call__', ['n', 'timeout', 'show_clicks'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['n', 'timeout', 'show_clicks'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        unicode_25036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, (-1)), 'unicode', u'\n        Blocking call to retrieve n coordinate pairs through mouse\n        clicks.\n        ')
        
        # Assigning a Name to a Attribute (line 289):
        # Getting the type of 'show_clicks' (line 289)
        show_clicks_25037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 27), 'show_clicks')
        # Getting the type of 'self' (line 289)
        self_25038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'self')
        # Setting the type of the member 'show_clicks' of a type (line 289)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 8), self_25038, 'show_clicks', show_clicks_25037)
        
        # Assigning a List to a Attribute (line 290):
        
        # Obtaining an instance of the builtin type 'list' (line 290)
        list_25039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 290)
        
        # Getting the type of 'self' (line 290)
        self_25040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'self')
        # Setting the type of the member 'clicks' of a type (line 290)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), self_25040, 'clicks', list_25039)
        
        # Assigning a List to a Attribute (line 291):
        
        # Obtaining an instance of the builtin type 'list' (line 291)
        list_25041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 291)
        
        # Getting the type of 'self' (line 291)
        self_25042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'self')
        # Setting the type of the member 'marks' of a type (line 291)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 8), self_25042, 'marks', list_25041)
        
        # Call to __call__(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'self' (line 292)
        self_25045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 31), 'self', False)
        # Processing the call keyword arguments (line 292)
        # Getting the type of 'n' (line 292)
        n_25046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 39), 'n', False)
        keyword_25047 = n_25046
        # Getting the type of 'timeout' (line 292)
        timeout_25048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 50), 'timeout', False)
        keyword_25049 = timeout_25048
        kwargs_25050 = {'timeout': keyword_25049, 'n': keyword_25047}
        # Getting the type of 'BlockingInput' (line 292)
        BlockingInput_25043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'BlockingInput', False)
        # Obtaining the member '__call__' of a type (line 292)
        call___25044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), BlockingInput_25043, '__call__')
        # Calling __call__(args, kwargs) (line 292)
        call___call_result_25051 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), call___25044, *[self_25045], **kwargs_25050)
        
        # Getting the type of 'self' (line 294)
        self_25052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 15), 'self')
        # Obtaining the member 'clicks' of a type (line 294)
        clicks_25053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 15), self_25052, 'clicks')
        # Assigning a type to the variable 'stypy_return_type' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'stypy_return_type', clicks_25053)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 284)
        stypy_return_type_25054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25054)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_25054


# Assigning a type to the variable 'BlockingMouseInput' (line 124)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'BlockingMouseInput', BlockingMouseInput)

# Assigning a Num to a Name (line 134):
int_25055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 17), 'int')
# Getting the type of 'BlockingMouseInput'
BlockingMouseInput_25056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BlockingMouseInput')
# Setting the type of the member 'button_add' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BlockingMouseInput_25056, 'button_add', int_25055)

# Assigning a Num to a Name (line 135):
int_25057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 17), 'int')
# Getting the type of 'BlockingMouseInput'
BlockingMouseInput_25058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BlockingMouseInput')
# Setting the type of the member 'button_pop' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BlockingMouseInput_25058, 'button_pop', int_25057)

# Assigning a Num to a Name (line 136):
int_25059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 18), 'int')
# Getting the type of 'BlockingMouseInput'
BlockingMouseInput_25060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BlockingMouseInput')
# Setting the type of the member 'button_stop' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BlockingMouseInput_25060, 'button_stop', int_25059)
# Declaration of the 'BlockingContourLabeler' class
# Getting the type of 'BlockingMouseInput' (line 297)
BlockingMouseInput_25061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 29), 'BlockingMouseInput')

class BlockingContourLabeler(BlockingMouseInput_25061, ):
    unicode_25062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, (-1)), 'unicode', u'\n    Class that creates a callable object that uses mouse clicks or key\n    clicks on a figure window to place contour labels.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 302, 4, False)
        # Assigning a type to the variable 'self' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingContourLabeler.__init__', ['cs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['cs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 303):
        # Getting the type of 'cs' (line 303)
        cs_25063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 18), 'cs')
        # Getting the type of 'self' (line 303)
        self_25064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'self')
        # Setting the type of the member 'cs' of a type (line 303)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 8), self_25064, 'cs', cs_25063)
        
        # Call to __init__(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'self' (line 304)
        self_25067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 36), 'self', False)
        # Processing the call keyword arguments (line 304)
        # Getting the type of 'cs' (line 304)
        cs_25068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 46), 'cs', False)
        # Obtaining the member 'ax' of a type (line 304)
        ax_25069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 46), cs_25068, 'ax')
        # Obtaining the member 'figure' of a type (line 304)
        figure_25070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 46), ax_25069, 'figure')
        keyword_25071 = figure_25070
        kwargs_25072 = {'fig': keyword_25071}
        # Getting the type of 'BlockingMouseInput' (line 304)
        BlockingMouseInput_25065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'BlockingMouseInput', False)
        # Obtaining the member '__init__' of a type (line 304)
        init___25066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), BlockingMouseInput_25065, '__init__')
        # Calling __init__(args, kwargs) (line 304)
        init___call_result_25073 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), init___25066, *[self_25067], **kwargs_25072)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def add_click(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_click'
        module_type_store = module_type_store.open_function_context('add_click', 306, 4, False)
        # Assigning a type to the variable 'self' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingContourLabeler.add_click.__dict__.__setitem__('stypy_localization', localization)
        BlockingContourLabeler.add_click.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingContourLabeler.add_click.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingContourLabeler.add_click.__dict__.__setitem__('stypy_function_name', 'BlockingContourLabeler.add_click')
        BlockingContourLabeler.add_click.__dict__.__setitem__('stypy_param_names_list', ['event'])
        BlockingContourLabeler.add_click.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingContourLabeler.add_click.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingContourLabeler.add_click.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingContourLabeler.add_click.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingContourLabeler.add_click.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingContourLabeler.add_click.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingContourLabeler.add_click', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_click', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_click(...)' code ##################

        
        # Call to button1(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'event' (line 307)
        event_25076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 21), 'event', False)
        # Processing the call keyword arguments (line 307)
        kwargs_25077 = {}
        # Getting the type of 'self' (line 307)
        self_25074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'self', False)
        # Obtaining the member 'button1' of a type (line 307)
        button1_25075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 8), self_25074, 'button1')
        # Calling button1(args, kwargs) (line 307)
        button1_call_result_25078 = invoke(stypy.reporting.localization.Localization(__file__, 307, 8), button1_25075, *[event_25076], **kwargs_25077)
        
        
        # ################# End of 'add_click(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_click' in the type store
        # Getting the type of 'stypy_return_type' (line 306)
        stypy_return_type_25079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25079)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_click'
        return stypy_return_type_25079


    @norecursion
    def pop_click(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_25080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 37), 'int')
        defaults = [int_25080]
        # Create a new context for function 'pop_click'
        module_type_store = module_type_store.open_function_context('pop_click', 309, 4, False)
        # Assigning a type to the variable 'self' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingContourLabeler.pop_click.__dict__.__setitem__('stypy_localization', localization)
        BlockingContourLabeler.pop_click.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingContourLabeler.pop_click.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingContourLabeler.pop_click.__dict__.__setitem__('stypy_function_name', 'BlockingContourLabeler.pop_click')
        BlockingContourLabeler.pop_click.__dict__.__setitem__('stypy_param_names_list', ['event', 'index'])
        BlockingContourLabeler.pop_click.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingContourLabeler.pop_click.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingContourLabeler.pop_click.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingContourLabeler.pop_click.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingContourLabeler.pop_click.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingContourLabeler.pop_click.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingContourLabeler.pop_click', ['event', 'index'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pop_click', localization, ['event', 'index'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pop_click(...)' code ##################

        
        # Call to button3(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 'event' (line 310)
        event_25083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 21), 'event', False)
        # Processing the call keyword arguments (line 310)
        kwargs_25084 = {}
        # Getting the type of 'self' (line 310)
        self_25081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'self', False)
        # Obtaining the member 'button3' of a type (line 310)
        button3_25082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 8), self_25081, 'button3')
        # Calling button3(args, kwargs) (line 310)
        button3_call_result_25085 = invoke(stypy.reporting.localization.Localization(__file__, 310, 8), button3_25082, *[event_25083], **kwargs_25084)
        
        
        # ################# End of 'pop_click(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pop_click' in the type store
        # Getting the type of 'stypy_return_type' (line 309)
        stypy_return_type_25086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25086)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pop_click'
        return stypy_return_type_25086


    @norecursion
    def button1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'button1'
        module_type_store = module_type_store.open_function_context('button1', 312, 4, False)
        # Assigning a type to the variable 'self' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingContourLabeler.button1.__dict__.__setitem__('stypy_localization', localization)
        BlockingContourLabeler.button1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingContourLabeler.button1.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingContourLabeler.button1.__dict__.__setitem__('stypy_function_name', 'BlockingContourLabeler.button1')
        BlockingContourLabeler.button1.__dict__.__setitem__('stypy_param_names_list', ['event'])
        BlockingContourLabeler.button1.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingContourLabeler.button1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingContourLabeler.button1.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingContourLabeler.button1.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingContourLabeler.button1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingContourLabeler.button1.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingContourLabeler.button1', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'button1', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'button1(...)' code ##################

        unicode_25087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, (-1)), 'unicode', u'\n        This will be called if an event involving a button other than\n        2 or 3 occcurs.  This will add a label to a contour.\n        ')
        
        
        # Getting the type of 'event' (line 319)
        event_25088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 11), 'event')
        # Obtaining the member 'inaxes' of a type (line 319)
        inaxes_25089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 11), event_25088, 'inaxes')
        # Getting the type of 'self' (line 319)
        self_25090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 27), 'self')
        # Obtaining the member 'cs' of a type (line 319)
        cs_25091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 27), self_25090, 'cs')
        # Obtaining the member 'ax' of a type (line 319)
        ax_25092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 27), cs_25091, 'ax')
        # Applying the binary operator '==' (line 319)
        result_eq_25093 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 11), '==', inaxes_25089, ax_25092)
        
        # Testing the type of an if condition (line 319)
        if_condition_25094 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 319, 8), result_eq_25093)
        # Assigning a type to the variable 'if_condition_25094' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'if_condition_25094', if_condition_25094)
        # SSA begins for if statement (line 319)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to add_label_near(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'event' (line 320)
        event_25098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 35), 'event', False)
        # Obtaining the member 'x' of a type (line 320)
        x_25099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 35), event_25098, 'x')
        # Getting the type of 'event' (line 320)
        event_25100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 44), 'event', False)
        # Obtaining the member 'y' of a type (line 320)
        y_25101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 44), event_25100, 'y')
        # Getting the type of 'self' (line 320)
        self_25102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 53), 'self', False)
        # Obtaining the member 'inline' of a type (line 320)
        inline_25103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 53), self_25102, 'inline')
        # Processing the call keyword arguments (line 320)
        # Getting the type of 'self' (line 321)
        self_25104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 50), 'self', False)
        # Obtaining the member 'inline_spacing' of a type (line 321)
        inline_spacing_25105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 50), self_25104, 'inline_spacing')
        keyword_25106 = inline_spacing_25105
        # Getting the type of 'False' (line 322)
        False_25107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 45), 'False', False)
        keyword_25108 = False_25107
        kwargs_25109 = {'transform': keyword_25108, 'inline_spacing': keyword_25106}
        # Getting the type of 'self' (line 320)
        self_25095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'self', False)
        # Obtaining the member 'cs' of a type (line 320)
        cs_25096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 12), self_25095, 'cs')
        # Obtaining the member 'add_label_near' of a type (line 320)
        add_label_near_25097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 12), cs_25096, 'add_label_near')
        # Calling add_label_near(args, kwargs) (line 320)
        add_label_near_call_result_25110 = invoke(stypy.reporting.localization.Localization(__file__, 320, 12), add_label_near_25097, *[x_25099, y_25101, inline_25103], **kwargs_25109)
        
        
        # Call to draw(...): (line 323)
        # Processing the call keyword arguments (line 323)
        kwargs_25115 = {}
        # Getting the type of 'self' (line 323)
        self_25111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'self', False)
        # Obtaining the member 'fig' of a type (line 323)
        fig_25112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 12), self_25111, 'fig')
        # Obtaining the member 'canvas' of a type (line 323)
        canvas_25113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 12), fig_25112, 'canvas')
        # Obtaining the member 'draw' of a type (line 323)
        draw_25114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 12), canvas_25113, 'draw')
        # Calling draw(args, kwargs) (line 323)
        draw_call_result_25116 = invoke(stypy.reporting.localization.Localization(__file__, 323, 12), draw_25114, *[], **kwargs_25115)
        
        # SSA branch for the else part of an if statement (line 319)
        module_type_store.open_ssa_branch('else')
        
        # Call to pop(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'self' (line 325)
        self_25119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 30), 'self', False)
        # Processing the call keyword arguments (line 325)
        kwargs_25120 = {}
        # Getting the type of 'BlockingInput' (line 325)
        BlockingInput_25117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'BlockingInput', False)
        # Obtaining the member 'pop' of a type (line 325)
        pop_25118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 12), BlockingInput_25117, 'pop')
        # Calling pop(args, kwargs) (line 325)
        pop_call_result_25121 = invoke(stypy.reporting.localization.Localization(__file__, 325, 12), pop_25118, *[self_25119], **kwargs_25120)
        
        # SSA join for if statement (line 319)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'button1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'button1' in the type store
        # Getting the type of 'stypy_return_type' (line 312)
        stypy_return_type_25122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25122)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'button1'
        return stypy_return_type_25122


    @norecursion
    def button3(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'button3'
        module_type_store = module_type_store.open_function_context('button3', 327, 4, False)
        # Assigning a type to the variable 'self' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingContourLabeler.button3.__dict__.__setitem__('stypy_localization', localization)
        BlockingContourLabeler.button3.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingContourLabeler.button3.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingContourLabeler.button3.__dict__.__setitem__('stypy_function_name', 'BlockingContourLabeler.button3')
        BlockingContourLabeler.button3.__dict__.__setitem__('stypy_param_names_list', ['event'])
        BlockingContourLabeler.button3.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingContourLabeler.button3.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingContourLabeler.button3.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingContourLabeler.button3.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingContourLabeler.button3.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingContourLabeler.button3.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingContourLabeler.button3', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'button3', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'button3(...)' code ##################

        unicode_25123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, (-1)), 'unicode', u"\n        This will be called if button 3 is clicked.  This will remove\n        a label if not in inline mode.  Unfortunately, if one is doing\n        inline labels, then there is currently no way to fix the\n        broken contour - once humpty-dumpty is broken, he can't be put\n        back together.  In inline mode, this does nothing.\n        ")
        
        # Getting the type of 'self' (line 336)
        self_25124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 11), 'self')
        # Obtaining the member 'inline' of a type (line 336)
        inline_25125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 11), self_25124, 'inline')
        # Testing the type of an if condition (line 336)
        if_condition_25126 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 336, 8), inline_25125)
        # Assigning a type to the variable 'if_condition_25126' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'if_condition_25126', if_condition_25126)
        # SSA begins for if statement (line 336)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA branch for the else part of an if statement (line 336)
        module_type_store.open_ssa_branch('else')
        
        # Call to pop_label(...): (line 339)
        # Processing the call keyword arguments (line 339)
        kwargs_25130 = {}
        # Getting the type of 'self' (line 339)
        self_25127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'self', False)
        # Obtaining the member 'cs' of a type (line 339)
        cs_25128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 12), self_25127, 'cs')
        # Obtaining the member 'pop_label' of a type (line 339)
        pop_label_25129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 12), cs_25128, 'pop_label')
        # Calling pop_label(args, kwargs) (line 339)
        pop_label_call_result_25131 = invoke(stypy.reporting.localization.Localization(__file__, 339, 12), pop_label_25129, *[], **kwargs_25130)
        
        
        # Call to draw(...): (line 340)
        # Processing the call keyword arguments (line 340)
        kwargs_25138 = {}
        # Getting the type of 'self' (line 340)
        self_25132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 12), 'self', False)
        # Obtaining the member 'cs' of a type (line 340)
        cs_25133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 12), self_25132, 'cs')
        # Obtaining the member 'ax' of a type (line 340)
        ax_25134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 12), cs_25133, 'ax')
        # Obtaining the member 'figure' of a type (line 340)
        figure_25135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 12), ax_25134, 'figure')
        # Obtaining the member 'canvas' of a type (line 340)
        canvas_25136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 12), figure_25135, 'canvas')
        # Obtaining the member 'draw' of a type (line 340)
        draw_25137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 12), canvas_25136, 'draw')
        # Calling draw(args, kwargs) (line 340)
        draw_call_result_25139 = invoke(stypy.reporting.localization.Localization(__file__, 340, 12), draw_25137, *[], **kwargs_25138)
        
        # SSA join for if statement (line 336)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'button3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'button3' in the type store
        # Getting the type of 'stypy_return_type' (line 327)
        stypy_return_type_25140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25140)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'button3'
        return stypy_return_type_25140


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_25141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 46), 'int')
        int_25142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 51), 'int')
        int_25143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 63), 'int')
        defaults = [int_25141, int_25142, int_25143]
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 342, 4, False)
        # Assigning a type to the variable 'self' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingContourLabeler.__call__.__dict__.__setitem__('stypy_localization', localization)
        BlockingContourLabeler.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingContourLabeler.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingContourLabeler.__call__.__dict__.__setitem__('stypy_function_name', 'BlockingContourLabeler.__call__')
        BlockingContourLabeler.__call__.__dict__.__setitem__('stypy_param_names_list', ['inline', 'inline_spacing', 'n', 'timeout'])
        BlockingContourLabeler.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingContourLabeler.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingContourLabeler.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingContourLabeler.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingContourLabeler.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingContourLabeler.__call__.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingContourLabeler.__call__', ['inline', 'inline_spacing', 'n', 'timeout'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['inline', 'inline_spacing', 'n', 'timeout'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 343):
        # Getting the type of 'inline' (line 343)
        inline_25144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 22), 'inline')
        # Getting the type of 'self' (line 343)
        self_25145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'self')
        # Setting the type of the member 'inline' of a type (line 343)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 8), self_25145, 'inline', inline_25144)
        
        # Assigning a Name to a Attribute (line 344):
        # Getting the type of 'inline_spacing' (line 344)
        inline_spacing_25146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 30), 'inline_spacing')
        # Getting the type of 'self' (line 344)
        self_25147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'self')
        # Setting the type of the member 'inline_spacing' of a type (line 344)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 8), self_25147, 'inline_spacing', inline_spacing_25146)
        
        # Call to __call__(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'self' (line 346)
        self_25150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 36), 'self', False)
        # Processing the call keyword arguments (line 346)
        # Getting the type of 'n' (line 346)
        n_25151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 44), 'n', False)
        keyword_25152 = n_25151
        # Getting the type of 'timeout' (line 346)
        timeout_25153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 55), 'timeout', False)
        keyword_25154 = timeout_25153
        # Getting the type of 'False' (line 347)
        False_25155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 48), 'False', False)
        keyword_25156 = False_25155
        kwargs_25157 = {'show_clicks': keyword_25156, 'timeout': keyword_25154, 'n': keyword_25152}
        # Getting the type of 'BlockingMouseInput' (line 346)
        BlockingMouseInput_25148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'BlockingMouseInput', False)
        # Obtaining the member '__call__' of a type (line 346)
        call___25149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 8), BlockingMouseInput_25148, '__call__')
        # Calling __call__(args, kwargs) (line 346)
        call___call_result_25158 = invoke(stypy.reporting.localization.Localization(__file__, 346, 8), call___25149, *[self_25150], **kwargs_25157)
        
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 342)
        stypy_return_type_25159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25159)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_25159


# Assigning a type to the variable 'BlockingContourLabeler' (line 297)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 0), 'BlockingContourLabeler', BlockingContourLabeler)
# Declaration of the 'BlockingKeyMouseInput' class
# Getting the type of 'BlockingInput' (line 350)
BlockingInput_25160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 28), 'BlockingInput')

class BlockingKeyMouseInput(BlockingInput_25160, ):
    unicode_25161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, (-1)), 'unicode', u'\n    Class that creates a callable object to retrieve a single mouse or\n    keyboard click\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 355, 4, False)
        # Assigning a type to the variable 'self' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingKeyMouseInput.__init__', ['fig'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['fig'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'self' (line 356)
        self_25164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 31), 'self', False)
        # Processing the call keyword arguments (line 356)
        # Getting the type of 'fig' (line 356)
        fig_25165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 41), 'fig', False)
        keyword_25166 = fig_25165
        
        # Obtaining an instance of the builtin type 'tuple' (line 357)
        tuple_25167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 12), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 357)
        # Adding element type (line 357)
        unicode_25168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 12), 'unicode', u'button_press_event')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 12), tuple_25167, unicode_25168)
        # Adding element type (line 357)
        unicode_25169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 34), 'unicode', u'key_press_event')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 12), tuple_25167, unicode_25169)
        
        keyword_25170 = tuple_25167
        kwargs_25171 = {'fig': keyword_25166, 'eventslist': keyword_25170}
        # Getting the type of 'BlockingInput' (line 356)
        BlockingInput_25162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'BlockingInput', False)
        # Obtaining the member '__init__' of a type (line 356)
        init___25163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 8), BlockingInput_25162, '__init__')
        # Calling __init__(args, kwargs) (line 356)
        init___call_result_25172 = invoke(stypy.reporting.localization.Localization(__file__, 356, 8), init___25163, *[self_25164], **kwargs_25171)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def post_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'post_event'
        module_type_store = module_type_store.open_function_context('post_event', 359, 4, False)
        # Assigning a type to the variable 'self' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingKeyMouseInput.post_event.__dict__.__setitem__('stypy_localization', localization)
        BlockingKeyMouseInput.post_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingKeyMouseInput.post_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingKeyMouseInput.post_event.__dict__.__setitem__('stypy_function_name', 'BlockingKeyMouseInput.post_event')
        BlockingKeyMouseInput.post_event.__dict__.__setitem__('stypy_param_names_list', [])
        BlockingKeyMouseInput.post_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingKeyMouseInput.post_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingKeyMouseInput.post_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingKeyMouseInput.post_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingKeyMouseInput.post_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingKeyMouseInput.post_event.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingKeyMouseInput.post_event', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'post_event', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'post_event(...)' code ##################

        unicode_25173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, (-1)), 'unicode', u'\n        Determines if it is a key event\n        ')
        
        
        
        # Call to len(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'self' (line 363)
        self_25175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 15), 'self', False)
        # Obtaining the member 'events' of a type (line 363)
        events_25176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 15), self_25175, 'events')
        # Processing the call keyword arguments (line 363)
        kwargs_25177 = {}
        # Getting the type of 'len' (line 363)
        len_25174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 11), 'len', False)
        # Calling len(args, kwargs) (line 363)
        len_call_result_25178 = invoke(stypy.reporting.localization.Localization(__file__, 363, 11), len_25174, *[events_25176], **kwargs_25177)
        
        int_25179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 31), 'int')
        # Applying the binary operator '==' (line 363)
        result_eq_25180 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 11), '==', len_call_result_25178, int_25179)
        
        # Testing the type of an if condition (line 363)
        if_condition_25181 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 363, 8), result_eq_25180)
        # Assigning a type to the variable 'if_condition_25181' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'if_condition_25181', if_condition_25181)
        # SSA begins for if statement (line 363)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 364)
        # Processing the call arguments (line 364)
        unicode_25184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 26), 'unicode', u'No events yet')
        # Processing the call keyword arguments (line 364)
        kwargs_25185 = {}
        # Getting the type of 'warnings' (line 364)
        warnings_25182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 364)
        warn_25183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 12), warnings_25182, 'warn')
        # Calling warn(args, kwargs) (line 364)
        warn_call_result_25186 = invoke(stypy.reporting.localization.Localization(__file__, 364, 12), warn_25183, *[unicode_25184], **kwargs_25185)
        
        # SSA branch for the else part of an if statement (line 363)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Compare to a Attribute (line 366):
        
        
        # Obtaining the type of the subscript
        int_25187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 42), 'int')
        # Getting the type of 'self' (line 366)
        self_25188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 30), 'self')
        # Obtaining the member 'events' of a type (line 366)
        events_25189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 30), self_25188, 'events')
        # Obtaining the member '__getitem__' of a type (line 366)
        getitem___25190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 30), events_25189, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 366)
        subscript_call_result_25191 = invoke(stypy.reporting.localization.Localization(__file__, 366, 30), getitem___25190, int_25187)
        
        # Obtaining the member 'name' of a type (line 366)
        name_25192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 30), subscript_call_result_25191, 'name')
        unicode_25193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 54), 'unicode', u'key_press_event')
        # Applying the binary operator '==' (line 366)
        result_eq_25194 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 30), '==', name_25192, unicode_25193)
        
        # Getting the type of 'self' (line 366)
        self_25195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'self')
        # Setting the type of the member 'keyormouse' of a type (line 366)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 12), self_25195, 'keyormouse', result_eq_25194)
        # SSA join for if statement (line 363)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'post_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'post_event' in the type store
        # Getting the type of 'stypy_return_type' (line 359)
        stypy_return_type_25196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25196)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'post_event'
        return stypy_return_type_25196


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_25197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 31), 'int')
        defaults = [int_25197]
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 368, 4, False)
        # Assigning a type to the variable 'self' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BlockingKeyMouseInput.__call__.__dict__.__setitem__('stypy_localization', localization)
        BlockingKeyMouseInput.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BlockingKeyMouseInput.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        BlockingKeyMouseInput.__call__.__dict__.__setitem__('stypy_function_name', 'BlockingKeyMouseInput.__call__')
        BlockingKeyMouseInput.__call__.__dict__.__setitem__('stypy_param_names_list', ['timeout'])
        BlockingKeyMouseInput.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        BlockingKeyMouseInput.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BlockingKeyMouseInput.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        BlockingKeyMouseInput.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        BlockingKeyMouseInput.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BlockingKeyMouseInput.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BlockingKeyMouseInput.__call__', ['timeout'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['timeout'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        unicode_25198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, (-1)), 'unicode', u'\n        Blocking call to retrieve a single mouse or key click\n        Returns True if key click, False if mouse, or None if timeout\n        ')
        
        # Assigning a Name to a Attribute (line 373):
        # Getting the type of 'None' (line 373)
        None_25199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 26), 'None')
        # Getting the type of 'self' (line 373)
        self_25200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'self')
        # Setting the type of the member 'keyormouse' of a type (line 373)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 8), self_25200, 'keyormouse', None_25199)
        
        # Call to __call__(...): (line 374)
        # Processing the call arguments (line 374)
        # Getting the type of 'self' (line 374)
        self_25203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 31), 'self', False)
        # Processing the call keyword arguments (line 374)
        int_25204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 39), 'int')
        keyword_25205 = int_25204
        # Getting the type of 'timeout' (line 374)
        timeout_25206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 50), 'timeout', False)
        keyword_25207 = timeout_25206
        kwargs_25208 = {'timeout': keyword_25207, 'n': keyword_25205}
        # Getting the type of 'BlockingInput' (line 374)
        BlockingInput_25201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'BlockingInput', False)
        # Obtaining the member '__call__' of a type (line 374)
        call___25202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 8), BlockingInput_25201, '__call__')
        # Calling __call__(args, kwargs) (line 374)
        call___call_result_25209 = invoke(stypy.reporting.localization.Localization(__file__, 374, 8), call___25202, *[self_25203], **kwargs_25208)
        
        # Getting the type of 'self' (line 376)
        self_25210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 15), 'self')
        # Obtaining the member 'keyormouse' of a type (line 376)
        keyormouse_25211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 15), self_25210, 'keyormouse')
        # Assigning a type to the variable 'stypy_return_type' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'stypy_return_type', keyormouse_25211)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 368)
        stypy_return_type_25212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25212)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_25212


# Assigning a type to the variable 'BlockingKeyMouseInput' (line 350)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 0), 'BlockingKeyMouseInput', BlockingKeyMouseInput)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
