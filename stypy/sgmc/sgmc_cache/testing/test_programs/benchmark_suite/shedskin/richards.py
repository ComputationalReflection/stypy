
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #  Based on original version written in BCPL by Dr Martin Richards
2: #  in 1981 at Cambridge University Computer Laboratory, England
3: #  and a C++ version derived from a Smalltalk version written by
4: #  L Peter Deutsch.
5: #  Translation from C++, Mario Wolczko
6: #  Outer loop added by Alex Jacoby
7: 
8: # Task IDs
9: I_IDLE = 1
10: I_WORK = 2
11: I_HANDLERA = 3
12: I_HANDLERB = 4
13: I_DEVA = 5
14: I_DEVB = 6
15: 
16: # Packet types
17: K_DEV = 1000
18: K_WORK = 1001
19: 
20: # Packet
21: 
22: BUFSIZE = 4
23: 
24: BUFSIZE_RANGE = range(BUFSIZE)
25: 
26: 
27: class Packet(object):
28:     def __init__(self, l, i, k):
29:         self.link = l
30:         self.ident = i
31:         self.kind = k
32:         self.datum = 0
33:         self.data = [0] * BUFSIZE
34: 
35:     def append_to(self, lst):
36:         self.link = None
37:         if lst is None:
38:             return self
39:         else:
40:             p = lst
41:             next = p.link
42:             while next is not None:
43:                 p = next
44:                 next = p.link
45:             p.link = self
46:             return lst
47: 
48: 
49: # Task Records
50: 
51: class TaskRec(object):
52:     pass
53: 
54: 
55: class DeviceTaskRec(TaskRec):
56:     def __init__(self):
57:         self.pending = None
58: 
59: 
60: class IdleTaskRec(TaskRec):
61:     def __init__(self):
62:         self.control = 1
63:         self.count = 10000
64: 
65: 
66: class HandlerTaskRec(TaskRec):
67:     def __init__(self):
68:         self.work_in = None
69:         self.device_in = None
70: 
71:     def workInAdd(self, p):
72:         self.work_in = p.append_to(self.work_in)
73:         return self.work_in
74: 
75:     def deviceInAdd(self, p):
76:         self.device_in = p.append_to(self.device_in)
77:         return self.device_in
78: 
79: 
80: class WorkerTaskRec(TaskRec):
81:     def __init__(self):
82:         self.destination = I_HANDLERA
83:         self.count = 0
84: 
85: 
86: # Task
87: 
88: class TaskState(object):
89:     def __init__(self):
90:         self.packet_pending = True
91:         self.task_waiting = False
92:         self.task_holding = False
93: 
94:     def packetPending(self):
95:         self.packet_pending = True
96:         self.task_waiting = False
97:         self.task_holding = False
98:         return self
99: 
100:     def waiting(self):
101:         self.packet_pending = False
102:         self.task_waiting = True
103:         self.task_holding = False
104:         return self
105: 
106:     def running(self):
107:         self.packet_pending = False
108:         self.task_waiting = False
109:         self.task_holding = False
110:         return self
111: 
112:     def waitingWithPacket(self):
113:         self.packet_pending = True
114:         self.task_waiting = True
115:         self.task_holding = False
116:         return self
117: 
118:     def isPacketPending(self):
119:         return self.packet_pending
120: 
121:     def isTaskWaiting(self):
122:         return self.task_waiting
123: 
124:     def isTaskHolding(self):
125:         return self.task_holding
126: 
127:     def isTaskHoldingOrWaiting(self):
128:         return self.task_holding or (not self.packet_pending and self.task_waiting)
129: 
130:     def isWaitingWithPacket(self):
131:         return self.packet_pending and self.task_waiting and not self.task_holding
132: 
133: 
134: tracing = False
135: layout = 0
136: 
137: 
138: def trace(a):
139:     global layout
140:     layout -= 1
141:     if layout <= 0:
142:         ##        print
143:         layout = 50
144: 
145: 
146: ##    print a
147: ##    print a,
148: 
149: 
150: TASKTABSIZE = 10
151: 
152: 
153: class TaskWorkArea(object):
154:     def __init__(self):
155:         self.taskTab = [None] * TASKTABSIZE
156: 
157:         self.taskList = None
158: 
159:         self.holdCount = 0
160:         self.qpktCount = 0
161: 
162: 
163: taskWorkArea = TaskWorkArea()
164: 
165: 
166: class Task(TaskState):
167: 
168:     def __init__(self, i, p, w, initialState, r):
169:         self.link = taskWorkArea.taskList
170:         self.ident = i
171:         self.priority = p
172:         self.input = w
173: 
174:         self.packet_pending = initialState.isPacketPending()
175:         self.task_waiting = initialState.isTaskWaiting()
176:         self.task_holding = initialState.isTaskHolding()
177: 
178:         self.handle = r
179: 
180:         taskWorkArea.taskList = self
181:         taskWorkArea.taskTab[i] = self
182: 
183:     def fn(self, pkt, r):
184:         raise NotImplementedError
185: 
186:     def addPacket(self, p, old):
187:         if self.input is None:
188:             self.input = p
189:             self.packet_pending = True
190:             if self.priority > old.priority:
191:                 return self
192:         else:
193:             p.append_to(self.input)
194:         return old
195: 
196:     def runTask(self):
197:         if self.isWaitingWithPacket():
198:             msg = self.input
199:             self.input = msg.link
200:             if self.input is None:
201:                 self.running()
202:             else:
203:                 self.packetPending()
204:         else:
205:             msg = None
206: 
207:         self
208:         return self.fn(msg, self.handle)
209: 
210:     def waitTask(self):
211:         self.task_waiting = True
212:         return self
213: 
214:     def hold(self):
215:         taskWorkArea.holdCount += 1
216:         self.task_holding = True
217:         return self.link
218: 
219:     def release(self, i):
220:         t = self.findtcb(i)
221:         t.task_holding = False
222:         if t.priority > self.priority:
223:             return t
224:         else:
225:             return self
226: 
227:     def qpkt(self, pkt):
228:         t = self.findtcb(pkt.ident)
229:         taskWorkArea.qpktCount += 1
230:         pkt.link = None
231:         pkt.ident = self.ident
232:         return t.addPacket(pkt, self)
233: 
234:     def findtcb(self, id):
235:         t = taskWorkArea.taskTab[id]
236:         if t is None:
237:             raise Exception("Bad task id %d" % id)
238:         return t
239: 
240: 
241: # DeviceTask
242: 
243: 
244: class DeviceTask(Task):
245:     def __init__(self, i, p, w, s, r):
246:         Task.__init__(self, i, p, w, s, r)
247: 
248:     def fn(self, pkt, r):
249:         d = r
250:         assert isinstance(d, DeviceTaskRec)
251:         if pkt is None:
252:             pkt = d.pending
253:             if pkt is None:
254:                 return self.waitTask()
255:             else:
256:                 d.pending = None
257:                 return self.qpkt(pkt)
258:         else:
259:             d.pending = pkt
260:             if tracing: trace(str(pkt.datum))
261:             return self.hold()
262: 
263: 
264: class HandlerTask(Task):
265:     def __init__(self, i, p, w, s, r):
266:         Task.__init__(self, i, p, w, s, r)
267: 
268:     def fn(self, pkt, r):
269:         h = r
270:         assert isinstance(h, HandlerTaskRec)
271:         if pkt is not None:
272:             if pkt.kind == K_WORK:
273:                 h.workInAdd(pkt)
274:             else:
275:                 h.deviceInAdd(pkt)
276:         work = h.work_in
277:         if work is None:
278:             return self.waitTask()
279:         count = work.datum
280:         if count >= BUFSIZE:
281:             h.work_in = work.link
282:             return self.qpkt(work)
283: 
284:         dev = h.device_in
285:         if dev is None:
286:             return self.waitTask()
287: 
288:         h.device_in = dev.link
289:         dev.datum = work.data[count]
290:         work.datum = count + 1
291:         return self.qpkt(dev)
292: 
293: 
294: # IdleTask
295: 
296: 
297: class IdleTask(Task):
298:     def __init__(self, i, p, w, s, r):
299:         Task.__init__(self, i, 0, None, s, r)
300: 
301:     def fn(self, pkt, r):
302:         i = r
303:         assert isinstance(i, IdleTaskRec)
304:         i.count -= 1
305:         if i.count == 0:
306:             return self.hold()
307:         elif i.control & 1 == 0:
308:             i.control /= 2
309:             return self.release(I_DEVA)
310:         else:
311:             i.control = i.control / 2 ^ 0xd008
312:             return self.release(I_DEVB)
313: 
314: 
315: # WorkTask
316: 
317: 
318: A = ord('A')
319: 
320: 
321: class WorkTask(Task):
322:     def __init__(self, i, p, w, s, r):
323:         Task.__init__(self, i, p, w, s, r)
324: 
325:     def fn(self, pkt, r):
326:         w = r
327:         assert isinstance(w, WorkerTaskRec)
328:         if pkt is None:
329:             return self.waitTask()
330: 
331:         if w.destination == I_HANDLERA:
332:             dest = I_HANDLERB
333:         else:
334:             dest = I_HANDLERA
335: 
336:         w.destination = dest
337:         pkt.ident = dest
338:         pkt.datum = 0
339: 
340:         for i in BUFSIZE_RANGE:  # xrange(BUFSIZE)
341:             w.count += 1
342:             if w.count > 26:
343:                 w.count = 1
344:             pkt.data[i] = A + w.count - 1
345: 
346:         return self.qpkt(pkt)
347: 
348: 
349: import time
350: 
351: 
352: def schedule():
353:     t = taskWorkArea.taskList
354:     while t is not None:
355:         pkt = None
356: 
357:         if tracing:
358:             pass  # print "tcb =",t.ident
359: 
360:         # print '*', t.__class__
361: 
362:         if t.isTaskHoldingOrWaiting():
363:             t = t.link
364:         else:
365:             if tracing: trace(chr(ord("0") + t.ident))
366:             t = t.runTask()
367: 
368: 
369: class Richards(object):
370: 
371:     def run(self, iterations):
372:         for i in xrange(iterations):
373:             taskWorkArea.holdCount = 0
374:             taskWorkArea.qpktCount = 0
375: 
376:             IdleTask(I_IDLE, 1, 10000, TaskState().running(), IdleTaskRec())
377: 
378:             wkq = Packet(None, 0, K_WORK)
379:             wkq = Packet(wkq, 0, K_WORK)
380:             WorkTask(I_WORK, 1000, wkq, TaskState().waitingWithPacket(), WorkerTaskRec())
381: 
382:             wkq = Packet(None, I_DEVA, K_DEV)
383:             wkq = Packet(wkq, I_DEVA, K_DEV)
384:             wkq = Packet(wkq, I_DEVA, K_DEV)
385:             HandlerTask(I_HANDLERA, 2000, wkq, TaskState().waitingWithPacket(), HandlerTaskRec())
386: 
387:             wkq = Packet(None, I_DEVB, K_DEV)
388:             wkq = Packet(wkq, I_DEVB, K_DEV)
389:             wkq = Packet(wkq, I_DEVB, K_DEV)
390:             HandlerTask(I_HANDLERB, 3000, wkq, TaskState().waitingWithPacket(), HandlerTaskRec())
391: 
392:             wkq = None;
393:             DeviceTask(I_DEVA, 4000, wkq, TaskState().waiting(), DeviceTaskRec());
394:             DeviceTask(I_DEVB, 5000, wkq, TaskState().waiting(), DeviceTaskRec());
395: 
396:             schedule()
397: 
398:             if taskWorkArea.holdCount == 9297 and taskWorkArea.qpktCount == 23246:
399:                 pass
400:             else:
401:                 return False
402: 
403:         return True
404: 
405: 
406: def run():
407:     r = Richards()
408:     iterations = 10
409:     result = r.run(iterations)
410:     ##    print result
411: 
412:     return True
413: 
414: 
415: run()
416: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 9):
int_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 9), 'int')
# Assigning a type to the variable 'I_IDLE' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'I_IDLE', int_1)

# Assigning a Num to a Name (line 10):
int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 9), 'int')
# Assigning a type to the variable 'I_WORK' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'I_WORK', int_2)

# Assigning a Num to a Name (line 11):
int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 13), 'int')
# Assigning a type to the variable 'I_HANDLERA' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'I_HANDLERA', int_3)

# Assigning a Num to a Name (line 12):
int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 13), 'int')
# Assigning a type to the variable 'I_HANDLERB' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'I_HANDLERB', int_4)

# Assigning a Num to a Name (line 13):
int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 9), 'int')
# Assigning a type to the variable 'I_DEVA' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'I_DEVA', int_5)

# Assigning a Num to a Name (line 14):
int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 9), 'int')
# Assigning a type to the variable 'I_DEVB' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'I_DEVB', int_6)

# Assigning a Num to a Name (line 17):
int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'int')
# Assigning a type to the variable 'K_DEV' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'K_DEV', int_7)

# Assigning a Num to a Name (line 18):
int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 9), 'int')
# Assigning a type to the variable 'K_WORK' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'K_WORK', int_8)

# Assigning a Num to a Name (line 22):
int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 10), 'int')
# Assigning a type to the variable 'BUFSIZE' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'BUFSIZE', int_9)

# Assigning a Call to a Name (line 24):

# Call to range(...): (line 24)
# Processing the call arguments (line 24)
# Getting the type of 'BUFSIZE' (line 24)
BUFSIZE_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 22), 'BUFSIZE', False)
# Processing the call keyword arguments (line 24)
kwargs_12 = {}
# Getting the type of 'range' (line 24)
range_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'range', False)
# Calling range(args, kwargs) (line 24)
range_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 24, 16), range_10, *[BUFSIZE_11], **kwargs_12)

# Assigning a type to the variable 'BUFSIZE_RANGE' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'BUFSIZE_RANGE', range_call_result_13)
# Declaration of the 'Packet' class

class Packet(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 28, 4, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Packet.__init__', ['l', 'i', 'k'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['l', 'i', 'k'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 29):
        # Getting the type of 'l' (line 29)
        l_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 20), 'l')
        # Getting the type of 'self' (line 29)
        self_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self')
        # Setting the type of the member 'link' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_15, 'link', l_14)
        
        # Assigning a Name to a Attribute (line 30):
        # Getting the type of 'i' (line 30)
        i_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 21), 'i')
        # Getting the type of 'self' (line 30)
        self_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member 'ident' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_17, 'ident', i_16)
        
        # Assigning a Name to a Attribute (line 31):
        # Getting the type of 'k' (line 31)
        k_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'k')
        # Getting the type of 'self' (line 31)
        self_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member 'kind' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_19, 'kind', k_18)
        
        # Assigning a Num to a Attribute (line 32):
        int_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 21), 'int')
        # Getting the type of 'self' (line 32)
        self_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self')
        # Setting the type of the member 'datum' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), self_21, 'datum', int_20)
        
        # Assigning a BinOp to a Attribute (line 33):
        
        # Obtaining an instance of the builtin type 'list' (line 33)
        list_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 33)
        # Adding element type (line 33)
        int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_22, int_23)
        
        # Getting the type of 'BUFSIZE' (line 33)
        BUFSIZE_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 26), 'BUFSIZE')
        # Applying the binary operator '*' (line 33)
        result_mul_25 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 20), '*', list_22, BUFSIZE_24)
        
        # Getting the type of 'self' (line 33)
        self_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self')
        # Setting the type of the member 'data' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_26, 'data', result_mul_25)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def append_to(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'append_to'
        module_type_store = module_type_store.open_function_context('append_to', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Packet.append_to.__dict__.__setitem__('stypy_localization', localization)
        Packet.append_to.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Packet.append_to.__dict__.__setitem__('stypy_type_store', module_type_store)
        Packet.append_to.__dict__.__setitem__('stypy_function_name', 'Packet.append_to')
        Packet.append_to.__dict__.__setitem__('stypy_param_names_list', ['lst'])
        Packet.append_to.__dict__.__setitem__('stypy_varargs_param_name', None)
        Packet.append_to.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Packet.append_to.__dict__.__setitem__('stypy_call_defaults', defaults)
        Packet.append_to.__dict__.__setitem__('stypy_call_varargs', varargs)
        Packet.append_to.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Packet.append_to.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Packet.append_to', ['lst'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'append_to', localization, ['lst'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'append_to(...)' code ##################

        
        # Assigning a Name to a Attribute (line 36):
        # Getting the type of 'None' (line 36)
        None_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 20), 'None')
        # Getting the type of 'self' (line 36)
        self_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self')
        # Setting the type of the member 'link' of a type (line 36)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), self_28, 'link', None_27)
        
        # Type idiom detected: calculating its left and rigth part (line 37)
        # Getting the type of 'lst' (line 37)
        lst_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'lst')
        # Getting the type of 'None' (line 37)
        None_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 18), 'None')
        
        (may_be_31, more_types_in_union_32) = may_be_none(lst_29, None_30)

        if may_be_31:

            if more_types_in_union_32:
                # Runtime conditional SSA (line 37)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 38)
            self_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 19), 'self')
            # Assigning a type to the variable 'stypy_return_type' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'stypy_return_type', self_33)

            if more_types_in_union_32:
                # Runtime conditional SSA for else branch (line 37)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_31) or more_types_in_union_32):
            
            # Assigning a Name to a Name (line 40):
            # Getting the type of 'lst' (line 40)
            lst_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'lst')
            # Assigning a type to the variable 'p' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'p', lst_34)
            
            # Assigning a Attribute to a Name (line 41):
            # Getting the type of 'p' (line 41)
            p_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 19), 'p')
            # Obtaining the member 'link' of a type (line 41)
            link_36 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 19), p_35, 'link')
            # Assigning a type to the variable 'next' (line 41)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'next', link_36)
            
            
            # Getting the type of 'next' (line 42)
            next_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 18), 'next')
            # Getting the type of 'None' (line 42)
            None_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 30), 'None')
            # Applying the binary operator 'isnot' (line 42)
            result_is_not_39 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 18), 'isnot', next_37, None_38)
            
            # Assigning a type to the variable 'result_is_not_39' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'result_is_not_39', result_is_not_39)
            # Testing if the while is going to be iterated (line 42)
            # Testing the type of an if condition (line 42)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 12), result_is_not_39)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 42, 12), result_is_not_39):
                # SSA begins for while statement (line 42)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
                
                # Assigning a Name to a Name (line 43):
                # Getting the type of 'next' (line 43)
                next_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 20), 'next')
                # Assigning a type to the variable 'p' (line 43)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'p', next_40)
                
                # Assigning a Attribute to a Name (line 44):
                # Getting the type of 'p' (line 44)
                p_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), 'p')
                # Obtaining the member 'link' of a type (line 44)
                link_42 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 23), p_41, 'link')
                # Assigning a type to the variable 'next' (line 44)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'next', link_42)
                # SSA join for while statement (line 42)
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a Name to a Attribute (line 45):
            # Getting the type of 'self' (line 45)
            self_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 21), 'self')
            # Getting the type of 'p' (line 45)
            p_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'p')
            # Setting the type of the member 'link' of a type (line 45)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 12), p_44, 'link', self_43)
            # Getting the type of 'lst' (line 46)
            lst_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 19), 'lst')
            # Assigning a type to the variable 'stypy_return_type' (line 46)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'stypy_return_type', lst_45)

            if (may_be_31 and more_types_in_union_32):
                # SSA join for if statement (line 37)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'lst' (line 37)
        lst_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'lst')
        # Assigning a type to the variable 'lst' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'lst', remove_type_from_union(lst_46, types.NoneType))
        
        # ################# End of 'append_to(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'append_to' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_47)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'append_to'
        return stypy_return_type_47


# Assigning a type to the variable 'Packet' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'Packet', Packet)
# Declaration of the 'TaskRec' class

class TaskRec(object, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 51, 0, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TaskRec.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TaskRec' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'TaskRec', TaskRec)
# Declaration of the 'DeviceTaskRec' class
# Getting the type of 'TaskRec' (line 55)
TaskRec_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 20), 'TaskRec')

class DeviceTaskRec(TaskRec_48, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DeviceTaskRec.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 57):
        # Getting the type of 'None' (line 57)
        None_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'None')
        # Getting the type of 'self' (line 57)
        self_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self')
        # Setting the type of the member 'pending' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_50, 'pending', None_49)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'DeviceTaskRec' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'DeviceTaskRec', DeviceTaskRec)
# Declaration of the 'IdleTaskRec' class
# Getting the type of 'TaskRec' (line 60)
TaskRec_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 18), 'TaskRec')

class IdleTaskRec(TaskRec_51, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 61, 4, False)
        # Assigning a type to the variable 'self' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IdleTaskRec.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Num to a Attribute (line 62):
        int_52 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 23), 'int')
        # Getting the type of 'self' (line 62)
        self_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self')
        # Setting the type of the member 'control' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_53, 'control', int_52)
        
        # Assigning a Num to a Attribute (line 63):
        int_54 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 21), 'int')
        # Getting the type of 'self' (line 63)
        self_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self')
        # Setting the type of the member 'count' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_55, 'count', int_54)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'IdleTaskRec' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'IdleTaskRec', IdleTaskRec)
# Declaration of the 'HandlerTaskRec' class
# Getting the type of 'TaskRec' (line 66)
TaskRec_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 21), 'TaskRec')

class HandlerTaskRec(TaskRec_56, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerTaskRec.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 68):
        # Getting the type of 'None' (line 68)
        None_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 23), 'None')
        # Getting the type of 'self' (line 68)
        self_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self')
        # Setting the type of the member 'work_in' of a type (line 68)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_58, 'work_in', None_57)
        
        # Assigning a Name to a Attribute (line 69):
        # Getting the type of 'None' (line 69)
        None_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 25), 'None')
        # Getting the type of 'self' (line 69)
        self_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'self')
        # Setting the type of the member 'device_in' of a type (line 69)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), self_60, 'device_in', None_59)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def workInAdd(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'workInAdd'
        module_type_store = module_type_store.open_function_context('workInAdd', 71, 4, False)
        # Assigning a type to the variable 'self' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerTaskRec.workInAdd.__dict__.__setitem__('stypy_localization', localization)
        HandlerTaskRec.workInAdd.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerTaskRec.workInAdd.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerTaskRec.workInAdd.__dict__.__setitem__('stypy_function_name', 'HandlerTaskRec.workInAdd')
        HandlerTaskRec.workInAdd.__dict__.__setitem__('stypy_param_names_list', ['p'])
        HandlerTaskRec.workInAdd.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerTaskRec.workInAdd.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerTaskRec.workInAdd.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerTaskRec.workInAdd.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerTaskRec.workInAdd.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerTaskRec.workInAdd.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerTaskRec.workInAdd', ['p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'workInAdd', localization, ['p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'workInAdd(...)' code ##################

        
        # Assigning a Call to a Attribute (line 72):
        
        # Call to append_to(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'self' (line 72)
        self_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 35), 'self', False)
        # Obtaining the member 'work_in' of a type (line 72)
        work_in_64 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 35), self_63, 'work_in')
        # Processing the call keyword arguments (line 72)
        kwargs_65 = {}
        # Getting the type of 'p' (line 72)
        p_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 23), 'p', False)
        # Obtaining the member 'append_to' of a type (line 72)
        append_to_62 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 23), p_61, 'append_to')
        # Calling append_to(args, kwargs) (line 72)
        append_to_call_result_66 = invoke(stypy.reporting.localization.Localization(__file__, 72, 23), append_to_62, *[work_in_64], **kwargs_65)
        
        # Getting the type of 'self' (line 72)
        self_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'self')
        # Setting the type of the member 'work_in' of a type (line 72)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), self_67, 'work_in', append_to_call_result_66)
        # Getting the type of 'self' (line 73)
        self_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 15), 'self')
        # Obtaining the member 'work_in' of a type (line 73)
        work_in_69 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 15), self_68, 'work_in')
        # Assigning a type to the variable 'stypy_return_type' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'stypy_return_type', work_in_69)
        
        # ################# End of 'workInAdd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'workInAdd' in the type store
        # Getting the type of 'stypy_return_type' (line 71)
        stypy_return_type_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_70)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'workInAdd'
        return stypy_return_type_70


    @norecursion
    def deviceInAdd(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'deviceInAdd'
        module_type_store = module_type_store.open_function_context('deviceInAdd', 75, 4, False)
        # Assigning a type to the variable 'self' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerTaskRec.deviceInAdd.__dict__.__setitem__('stypy_localization', localization)
        HandlerTaskRec.deviceInAdd.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerTaskRec.deviceInAdd.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerTaskRec.deviceInAdd.__dict__.__setitem__('stypy_function_name', 'HandlerTaskRec.deviceInAdd')
        HandlerTaskRec.deviceInAdd.__dict__.__setitem__('stypy_param_names_list', ['p'])
        HandlerTaskRec.deviceInAdd.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerTaskRec.deviceInAdd.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerTaskRec.deviceInAdd.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerTaskRec.deviceInAdd.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerTaskRec.deviceInAdd.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerTaskRec.deviceInAdd.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerTaskRec.deviceInAdd', ['p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'deviceInAdd', localization, ['p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'deviceInAdd(...)' code ##################

        
        # Assigning a Call to a Attribute (line 76):
        
        # Call to append_to(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'self' (line 76)
        self_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 37), 'self', False)
        # Obtaining the member 'device_in' of a type (line 76)
        device_in_74 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 37), self_73, 'device_in')
        # Processing the call keyword arguments (line 76)
        kwargs_75 = {}
        # Getting the type of 'p' (line 76)
        p_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 25), 'p', False)
        # Obtaining the member 'append_to' of a type (line 76)
        append_to_72 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 25), p_71, 'append_to')
        # Calling append_to(args, kwargs) (line 76)
        append_to_call_result_76 = invoke(stypy.reporting.localization.Localization(__file__, 76, 25), append_to_72, *[device_in_74], **kwargs_75)
        
        # Getting the type of 'self' (line 76)
        self_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'self')
        # Setting the type of the member 'device_in' of a type (line 76)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), self_77, 'device_in', append_to_call_result_76)
        # Getting the type of 'self' (line 77)
        self_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 15), 'self')
        # Obtaining the member 'device_in' of a type (line 77)
        device_in_79 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 15), self_78, 'device_in')
        # Assigning a type to the variable 'stypy_return_type' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'stypy_return_type', device_in_79)
        
        # ################# End of 'deviceInAdd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'deviceInAdd' in the type store
        # Getting the type of 'stypy_return_type' (line 75)
        stypy_return_type_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_80)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'deviceInAdd'
        return stypy_return_type_80


# Assigning a type to the variable 'HandlerTaskRec' (line 66)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'HandlerTaskRec', HandlerTaskRec)
# Declaration of the 'WorkerTaskRec' class
# Getting the type of 'TaskRec' (line 80)
TaskRec_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'TaskRec')

class WorkerTaskRec(TaskRec_81, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 81, 4, False)
        # Assigning a type to the variable 'self' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'WorkerTaskRec.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 82):
        # Getting the type of 'I_HANDLERA' (line 82)
        I_HANDLERA_82 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 27), 'I_HANDLERA')
        # Getting the type of 'self' (line 82)
        self_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'self')
        # Setting the type of the member 'destination' of a type (line 82)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), self_83, 'destination', I_HANDLERA_82)
        
        # Assigning a Num to a Attribute (line 83):
        int_84 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 21), 'int')
        # Getting the type of 'self' (line 83)
        self_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'self')
        # Setting the type of the member 'count' of a type (line 83)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), self_85, 'count', int_84)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'WorkerTaskRec' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'WorkerTaskRec', WorkerTaskRec)
# Declaration of the 'TaskState' class

class TaskState(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 89, 4, False)
        # Assigning a type to the variable 'self' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TaskState.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 90):
        # Getting the type of 'True' (line 90)
        True_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 30), 'True')
        # Getting the type of 'self' (line 90)
        self_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self')
        # Setting the type of the member 'packet_pending' of a type (line 90)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_87, 'packet_pending', True_86)
        
        # Assigning a Name to a Attribute (line 91):
        # Getting the type of 'False' (line 91)
        False_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 28), 'False')
        # Getting the type of 'self' (line 91)
        self_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'self')
        # Setting the type of the member 'task_waiting' of a type (line 91)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), self_89, 'task_waiting', False_88)
        
        # Assigning a Name to a Attribute (line 92):
        # Getting the type of 'False' (line 92)
        False_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'False')
        # Getting the type of 'self' (line 92)
        self_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'self')
        # Setting the type of the member 'task_holding' of a type (line 92)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), self_91, 'task_holding', False_90)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def packetPending(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'packetPending'
        module_type_store = module_type_store.open_function_context('packetPending', 94, 4, False)
        # Assigning a type to the variable 'self' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TaskState.packetPending.__dict__.__setitem__('stypy_localization', localization)
        TaskState.packetPending.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TaskState.packetPending.__dict__.__setitem__('stypy_type_store', module_type_store)
        TaskState.packetPending.__dict__.__setitem__('stypy_function_name', 'TaskState.packetPending')
        TaskState.packetPending.__dict__.__setitem__('stypy_param_names_list', [])
        TaskState.packetPending.__dict__.__setitem__('stypy_varargs_param_name', None)
        TaskState.packetPending.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TaskState.packetPending.__dict__.__setitem__('stypy_call_defaults', defaults)
        TaskState.packetPending.__dict__.__setitem__('stypy_call_varargs', varargs)
        TaskState.packetPending.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TaskState.packetPending.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TaskState.packetPending', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'packetPending', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'packetPending(...)' code ##################

        
        # Assigning a Name to a Attribute (line 95):
        # Getting the type of 'True' (line 95)
        True_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 30), 'True')
        # Getting the type of 'self' (line 95)
        self_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'self')
        # Setting the type of the member 'packet_pending' of a type (line 95)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), self_93, 'packet_pending', True_92)
        
        # Assigning a Name to a Attribute (line 96):
        # Getting the type of 'False' (line 96)
        False_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 28), 'False')
        # Getting the type of 'self' (line 96)
        self_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'self')
        # Setting the type of the member 'task_waiting' of a type (line 96)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), self_95, 'task_waiting', False_94)
        
        # Assigning a Name to a Attribute (line 97):
        # Getting the type of 'False' (line 97)
        False_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 28), 'False')
        # Getting the type of 'self' (line 97)
        self_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'self')
        # Setting the type of the member 'task_holding' of a type (line 97)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), self_97, 'task_holding', False_96)
        # Getting the type of 'self' (line 98)
        self_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'stypy_return_type', self_98)
        
        # ################# End of 'packetPending(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'packetPending' in the type store
        # Getting the type of 'stypy_return_type' (line 94)
        stypy_return_type_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_99)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'packetPending'
        return stypy_return_type_99


    @norecursion
    def waiting(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'waiting'
        module_type_store = module_type_store.open_function_context('waiting', 100, 4, False)
        # Assigning a type to the variable 'self' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TaskState.waiting.__dict__.__setitem__('stypy_localization', localization)
        TaskState.waiting.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TaskState.waiting.__dict__.__setitem__('stypy_type_store', module_type_store)
        TaskState.waiting.__dict__.__setitem__('stypy_function_name', 'TaskState.waiting')
        TaskState.waiting.__dict__.__setitem__('stypy_param_names_list', [])
        TaskState.waiting.__dict__.__setitem__('stypy_varargs_param_name', None)
        TaskState.waiting.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TaskState.waiting.__dict__.__setitem__('stypy_call_defaults', defaults)
        TaskState.waiting.__dict__.__setitem__('stypy_call_varargs', varargs)
        TaskState.waiting.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TaskState.waiting.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TaskState.waiting', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'waiting', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'waiting(...)' code ##################

        
        # Assigning a Name to a Attribute (line 101):
        # Getting the type of 'False' (line 101)
        False_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'False')
        # Getting the type of 'self' (line 101)
        self_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'self')
        # Setting the type of the member 'packet_pending' of a type (line 101)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), self_101, 'packet_pending', False_100)
        
        # Assigning a Name to a Attribute (line 102):
        # Getting the type of 'True' (line 102)
        True_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 28), 'True')
        # Getting the type of 'self' (line 102)
        self_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'self')
        # Setting the type of the member 'task_waiting' of a type (line 102)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), self_103, 'task_waiting', True_102)
        
        # Assigning a Name to a Attribute (line 103):
        # Getting the type of 'False' (line 103)
        False_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 28), 'False')
        # Getting the type of 'self' (line 103)
        self_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'self')
        # Setting the type of the member 'task_holding' of a type (line 103)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), self_105, 'task_holding', False_104)
        # Getting the type of 'self' (line 104)
        self_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'stypy_return_type', self_106)
        
        # ################# End of 'waiting(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'waiting' in the type store
        # Getting the type of 'stypy_return_type' (line 100)
        stypy_return_type_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_107)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'waiting'
        return stypy_return_type_107


    @norecursion
    def running(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'running'
        module_type_store = module_type_store.open_function_context('running', 106, 4, False)
        # Assigning a type to the variable 'self' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TaskState.running.__dict__.__setitem__('stypy_localization', localization)
        TaskState.running.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TaskState.running.__dict__.__setitem__('stypy_type_store', module_type_store)
        TaskState.running.__dict__.__setitem__('stypy_function_name', 'TaskState.running')
        TaskState.running.__dict__.__setitem__('stypy_param_names_list', [])
        TaskState.running.__dict__.__setitem__('stypy_varargs_param_name', None)
        TaskState.running.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TaskState.running.__dict__.__setitem__('stypy_call_defaults', defaults)
        TaskState.running.__dict__.__setitem__('stypy_call_varargs', varargs)
        TaskState.running.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TaskState.running.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TaskState.running', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'running', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'running(...)' code ##################

        
        # Assigning a Name to a Attribute (line 107):
        # Getting the type of 'False' (line 107)
        False_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 30), 'False')
        # Getting the type of 'self' (line 107)
        self_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'self')
        # Setting the type of the member 'packet_pending' of a type (line 107)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 8), self_109, 'packet_pending', False_108)
        
        # Assigning a Name to a Attribute (line 108):
        # Getting the type of 'False' (line 108)
        False_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 28), 'False')
        # Getting the type of 'self' (line 108)
        self_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'self')
        # Setting the type of the member 'task_waiting' of a type (line 108)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), self_111, 'task_waiting', False_110)
        
        # Assigning a Name to a Attribute (line 109):
        # Getting the type of 'False' (line 109)
        False_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 28), 'False')
        # Getting the type of 'self' (line 109)
        self_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'self')
        # Setting the type of the member 'task_holding' of a type (line 109)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), self_113, 'task_holding', False_112)
        # Getting the type of 'self' (line 110)
        self_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'stypy_return_type', self_114)
        
        # ################# End of 'running(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'running' in the type store
        # Getting the type of 'stypy_return_type' (line 106)
        stypy_return_type_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_115)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'running'
        return stypy_return_type_115


    @norecursion
    def waitingWithPacket(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'waitingWithPacket'
        module_type_store = module_type_store.open_function_context('waitingWithPacket', 112, 4, False)
        # Assigning a type to the variable 'self' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TaskState.waitingWithPacket.__dict__.__setitem__('stypy_localization', localization)
        TaskState.waitingWithPacket.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TaskState.waitingWithPacket.__dict__.__setitem__('stypy_type_store', module_type_store)
        TaskState.waitingWithPacket.__dict__.__setitem__('stypy_function_name', 'TaskState.waitingWithPacket')
        TaskState.waitingWithPacket.__dict__.__setitem__('stypy_param_names_list', [])
        TaskState.waitingWithPacket.__dict__.__setitem__('stypy_varargs_param_name', None)
        TaskState.waitingWithPacket.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TaskState.waitingWithPacket.__dict__.__setitem__('stypy_call_defaults', defaults)
        TaskState.waitingWithPacket.__dict__.__setitem__('stypy_call_varargs', varargs)
        TaskState.waitingWithPacket.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TaskState.waitingWithPacket.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TaskState.waitingWithPacket', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'waitingWithPacket', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'waitingWithPacket(...)' code ##################

        
        # Assigning a Name to a Attribute (line 113):
        # Getting the type of 'True' (line 113)
        True_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 30), 'True')
        # Getting the type of 'self' (line 113)
        self_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'self')
        # Setting the type of the member 'packet_pending' of a type (line 113)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), self_117, 'packet_pending', True_116)
        
        # Assigning a Name to a Attribute (line 114):
        # Getting the type of 'True' (line 114)
        True_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 28), 'True')
        # Getting the type of 'self' (line 114)
        self_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'self')
        # Setting the type of the member 'task_waiting' of a type (line 114)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), self_119, 'task_waiting', True_118)
        
        # Assigning a Name to a Attribute (line 115):
        # Getting the type of 'False' (line 115)
        False_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 28), 'False')
        # Getting the type of 'self' (line 115)
        self_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'self')
        # Setting the type of the member 'task_holding' of a type (line 115)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 8), self_121, 'task_holding', False_120)
        # Getting the type of 'self' (line 116)
        self_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'stypy_return_type', self_122)
        
        # ################# End of 'waitingWithPacket(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'waitingWithPacket' in the type store
        # Getting the type of 'stypy_return_type' (line 112)
        stypy_return_type_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_123)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'waitingWithPacket'
        return stypy_return_type_123


    @norecursion
    def isPacketPending(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'isPacketPending'
        module_type_store = module_type_store.open_function_context('isPacketPending', 118, 4, False)
        # Assigning a type to the variable 'self' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TaskState.isPacketPending.__dict__.__setitem__('stypy_localization', localization)
        TaskState.isPacketPending.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TaskState.isPacketPending.__dict__.__setitem__('stypy_type_store', module_type_store)
        TaskState.isPacketPending.__dict__.__setitem__('stypy_function_name', 'TaskState.isPacketPending')
        TaskState.isPacketPending.__dict__.__setitem__('stypy_param_names_list', [])
        TaskState.isPacketPending.__dict__.__setitem__('stypy_varargs_param_name', None)
        TaskState.isPacketPending.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TaskState.isPacketPending.__dict__.__setitem__('stypy_call_defaults', defaults)
        TaskState.isPacketPending.__dict__.__setitem__('stypy_call_varargs', varargs)
        TaskState.isPacketPending.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TaskState.isPacketPending.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TaskState.isPacketPending', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'isPacketPending', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'isPacketPending(...)' code ##################

        # Getting the type of 'self' (line 119)
        self_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'self')
        # Obtaining the member 'packet_pending' of a type (line 119)
        packet_pending_125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 15), self_124, 'packet_pending')
        # Assigning a type to the variable 'stypy_return_type' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'stypy_return_type', packet_pending_125)
        
        # ################# End of 'isPacketPending(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'isPacketPending' in the type store
        # Getting the type of 'stypy_return_type' (line 118)
        stypy_return_type_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'isPacketPending'
        return stypy_return_type_126


    @norecursion
    def isTaskWaiting(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'isTaskWaiting'
        module_type_store = module_type_store.open_function_context('isTaskWaiting', 121, 4, False)
        # Assigning a type to the variable 'self' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TaskState.isTaskWaiting.__dict__.__setitem__('stypy_localization', localization)
        TaskState.isTaskWaiting.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TaskState.isTaskWaiting.__dict__.__setitem__('stypy_type_store', module_type_store)
        TaskState.isTaskWaiting.__dict__.__setitem__('stypy_function_name', 'TaskState.isTaskWaiting')
        TaskState.isTaskWaiting.__dict__.__setitem__('stypy_param_names_list', [])
        TaskState.isTaskWaiting.__dict__.__setitem__('stypy_varargs_param_name', None)
        TaskState.isTaskWaiting.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TaskState.isTaskWaiting.__dict__.__setitem__('stypy_call_defaults', defaults)
        TaskState.isTaskWaiting.__dict__.__setitem__('stypy_call_varargs', varargs)
        TaskState.isTaskWaiting.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TaskState.isTaskWaiting.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TaskState.isTaskWaiting', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'isTaskWaiting', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'isTaskWaiting(...)' code ##################

        # Getting the type of 'self' (line 122)
        self_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 15), 'self')
        # Obtaining the member 'task_waiting' of a type (line 122)
        task_waiting_128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 15), self_127, 'task_waiting')
        # Assigning a type to the variable 'stypy_return_type' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'stypy_return_type', task_waiting_128)
        
        # ################# End of 'isTaskWaiting(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'isTaskWaiting' in the type store
        # Getting the type of 'stypy_return_type' (line 121)
        stypy_return_type_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_129)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'isTaskWaiting'
        return stypy_return_type_129


    @norecursion
    def isTaskHolding(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'isTaskHolding'
        module_type_store = module_type_store.open_function_context('isTaskHolding', 124, 4, False)
        # Assigning a type to the variable 'self' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TaskState.isTaskHolding.__dict__.__setitem__('stypy_localization', localization)
        TaskState.isTaskHolding.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TaskState.isTaskHolding.__dict__.__setitem__('stypy_type_store', module_type_store)
        TaskState.isTaskHolding.__dict__.__setitem__('stypy_function_name', 'TaskState.isTaskHolding')
        TaskState.isTaskHolding.__dict__.__setitem__('stypy_param_names_list', [])
        TaskState.isTaskHolding.__dict__.__setitem__('stypy_varargs_param_name', None)
        TaskState.isTaskHolding.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TaskState.isTaskHolding.__dict__.__setitem__('stypy_call_defaults', defaults)
        TaskState.isTaskHolding.__dict__.__setitem__('stypy_call_varargs', varargs)
        TaskState.isTaskHolding.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TaskState.isTaskHolding.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TaskState.isTaskHolding', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'isTaskHolding', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'isTaskHolding(...)' code ##################

        # Getting the type of 'self' (line 125)
        self_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 15), 'self')
        # Obtaining the member 'task_holding' of a type (line 125)
        task_holding_131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 15), self_130, 'task_holding')
        # Assigning a type to the variable 'stypy_return_type' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'stypy_return_type', task_holding_131)
        
        # ################# End of 'isTaskHolding(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'isTaskHolding' in the type store
        # Getting the type of 'stypy_return_type' (line 124)
        stypy_return_type_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'isTaskHolding'
        return stypy_return_type_132


    @norecursion
    def isTaskHoldingOrWaiting(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'isTaskHoldingOrWaiting'
        module_type_store = module_type_store.open_function_context('isTaskHoldingOrWaiting', 127, 4, False)
        # Assigning a type to the variable 'self' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TaskState.isTaskHoldingOrWaiting.__dict__.__setitem__('stypy_localization', localization)
        TaskState.isTaskHoldingOrWaiting.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TaskState.isTaskHoldingOrWaiting.__dict__.__setitem__('stypy_type_store', module_type_store)
        TaskState.isTaskHoldingOrWaiting.__dict__.__setitem__('stypy_function_name', 'TaskState.isTaskHoldingOrWaiting')
        TaskState.isTaskHoldingOrWaiting.__dict__.__setitem__('stypy_param_names_list', [])
        TaskState.isTaskHoldingOrWaiting.__dict__.__setitem__('stypy_varargs_param_name', None)
        TaskState.isTaskHoldingOrWaiting.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TaskState.isTaskHoldingOrWaiting.__dict__.__setitem__('stypy_call_defaults', defaults)
        TaskState.isTaskHoldingOrWaiting.__dict__.__setitem__('stypy_call_varargs', varargs)
        TaskState.isTaskHoldingOrWaiting.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TaskState.isTaskHoldingOrWaiting.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TaskState.isTaskHoldingOrWaiting', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'isTaskHoldingOrWaiting', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'isTaskHoldingOrWaiting(...)' code ##################

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 128)
        self_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), 'self')
        # Obtaining the member 'task_holding' of a type (line 128)
        task_holding_134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 15), self_133, 'task_holding')
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 128)
        self_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 41), 'self')
        # Obtaining the member 'packet_pending' of a type (line 128)
        packet_pending_136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 41), self_135, 'packet_pending')
        # Applying the 'not' unary operator (line 128)
        result_not__137 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 37), 'not', packet_pending_136)
        
        # Getting the type of 'self' (line 128)
        self_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 65), 'self')
        # Obtaining the member 'task_waiting' of a type (line 128)
        task_waiting_139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 65), self_138, 'task_waiting')
        # Applying the binary operator 'and' (line 128)
        result_and_keyword_140 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 37), 'and', result_not__137, task_waiting_139)
        
        # Applying the binary operator 'or' (line 128)
        result_or_keyword_141 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 15), 'or', task_holding_134, result_and_keyword_140)
        
        # Assigning a type to the variable 'stypy_return_type' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'stypy_return_type', result_or_keyword_141)
        
        # ################# End of 'isTaskHoldingOrWaiting(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'isTaskHoldingOrWaiting' in the type store
        # Getting the type of 'stypy_return_type' (line 127)
        stypy_return_type_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_142)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'isTaskHoldingOrWaiting'
        return stypy_return_type_142


    @norecursion
    def isWaitingWithPacket(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'isWaitingWithPacket'
        module_type_store = module_type_store.open_function_context('isWaitingWithPacket', 130, 4, False)
        # Assigning a type to the variable 'self' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TaskState.isWaitingWithPacket.__dict__.__setitem__('stypy_localization', localization)
        TaskState.isWaitingWithPacket.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TaskState.isWaitingWithPacket.__dict__.__setitem__('stypy_type_store', module_type_store)
        TaskState.isWaitingWithPacket.__dict__.__setitem__('stypy_function_name', 'TaskState.isWaitingWithPacket')
        TaskState.isWaitingWithPacket.__dict__.__setitem__('stypy_param_names_list', [])
        TaskState.isWaitingWithPacket.__dict__.__setitem__('stypy_varargs_param_name', None)
        TaskState.isWaitingWithPacket.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TaskState.isWaitingWithPacket.__dict__.__setitem__('stypy_call_defaults', defaults)
        TaskState.isWaitingWithPacket.__dict__.__setitem__('stypy_call_varargs', varargs)
        TaskState.isWaitingWithPacket.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TaskState.isWaitingWithPacket.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TaskState.isWaitingWithPacket', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'isWaitingWithPacket', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'isWaitingWithPacket(...)' code ##################

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 131)
        self_143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'self')
        # Obtaining the member 'packet_pending' of a type (line 131)
        packet_pending_144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 15), self_143, 'packet_pending')
        # Getting the type of 'self' (line 131)
        self_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 39), 'self')
        # Obtaining the member 'task_waiting' of a type (line 131)
        task_waiting_146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 39), self_145, 'task_waiting')
        # Applying the binary operator 'and' (line 131)
        result_and_keyword_147 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 15), 'and', packet_pending_144, task_waiting_146)
        
        # Getting the type of 'self' (line 131)
        self_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 65), 'self')
        # Obtaining the member 'task_holding' of a type (line 131)
        task_holding_149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 65), self_148, 'task_holding')
        # Applying the 'not' unary operator (line 131)
        result_not__150 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 61), 'not', task_holding_149)
        
        # Applying the binary operator 'and' (line 131)
        result_and_keyword_151 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 15), 'and', result_and_keyword_147, result_not__150)
        
        # Assigning a type to the variable 'stypy_return_type' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'stypy_return_type', result_and_keyword_151)
        
        # ################# End of 'isWaitingWithPacket(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'isWaitingWithPacket' in the type store
        # Getting the type of 'stypy_return_type' (line 130)
        stypy_return_type_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_152)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'isWaitingWithPacket'
        return stypy_return_type_152


# Assigning a type to the variable 'TaskState' (line 88)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'TaskState', TaskState)

# Assigning a Name to a Name (line 134):
# Getting the type of 'False' (line 134)
False_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 10), 'False')
# Assigning a type to the variable 'tracing' (line 134)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 0), 'tracing', False_153)

# Assigning a Num to a Name (line 135):
int_154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 9), 'int')
# Assigning a type to the variable 'layout' (line 135)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'layout', int_154)

@norecursion
def trace(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'trace'
    module_type_store = module_type_store.open_function_context('trace', 138, 0, False)
    
    # Passed parameters checking function
    trace.stypy_localization = localization
    trace.stypy_type_of_self = None
    trace.stypy_type_store = module_type_store
    trace.stypy_function_name = 'trace'
    trace.stypy_param_names_list = ['a']
    trace.stypy_varargs_param_name = None
    trace.stypy_kwargs_param_name = None
    trace.stypy_call_defaults = defaults
    trace.stypy_call_varargs = varargs
    trace.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'trace', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'trace', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'trace(...)' code ##################

    # Marking variables as global (line 139)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 139, 4), 'layout')
    
    # Getting the type of 'layout' (line 140)
    layout_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'layout')
    int_156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 14), 'int')
    # Applying the binary operator '-=' (line 140)
    result_isub_157 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 4), '-=', layout_155, int_156)
    # Assigning a type to the variable 'layout' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'layout', result_isub_157)
    
    
    # Getting the type of 'layout' (line 141)
    layout_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 7), 'layout')
    int_159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 17), 'int')
    # Applying the binary operator '<=' (line 141)
    result_le_160 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 7), '<=', layout_158, int_159)
    
    # Testing if the type of an if condition is none (line 141)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 141, 4), result_le_160):
        pass
    else:
        
        # Testing the type of an if condition (line 141)
        if_condition_161 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 4), result_le_160)
        # Assigning a type to the variable 'if_condition_161' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'if_condition_161', if_condition_161)
        # SSA begins for if statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 143):
        int_162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 17), 'int')
        # Assigning a type to the variable 'layout' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'layout', int_162)
        # SSA join for if statement (line 141)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'trace(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'trace' in the type store
    # Getting the type of 'stypy_return_type' (line 138)
    stypy_return_type_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_163)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'trace'
    return stypy_return_type_163

# Assigning a type to the variable 'trace' (line 138)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'trace', trace)

# Assigning a Num to a Name (line 150):
int_164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 14), 'int')
# Assigning a type to the variable 'TASKTABSIZE' (line 150)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 0), 'TASKTABSIZE', int_164)
# Declaration of the 'TaskWorkArea' class

class TaskWorkArea(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 154, 4, False)
        # Assigning a type to the variable 'self' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TaskWorkArea.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a BinOp to a Attribute (line 155):
        
        # Obtaining an instance of the builtin type 'list' (line 155)
        list_165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 155)
        # Adding element type (line 155)
        # Getting the type of 'None' (line 155)
        None_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 24), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 23), list_165, None_166)
        
        # Getting the type of 'TASKTABSIZE' (line 155)
        TASKTABSIZE_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 32), 'TASKTABSIZE')
        # Applying the binary operator '*' (line 155)
        result_mul_168 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 23), '*', list_165, TASKTABSIZE_167)
        
        # Getting the type of 'self' (line 155)
        self_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'self')
        # Setting the type of the member 'taskTab' of a type (line 155)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), self_169, 'taskTab', result_mul_168)
        
        # Assigning a Name to a Attribute (line 157):
        # Getting the type of 'None' (line 157)
        None_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 24), 'None')
        # Getting the type of 'self' (line 157)
        self_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'self')
        # Setting the type of the member 'taskList' of a type (line 157)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), self_171, 'taskList', None_170)
        
        # Assigning a Num to a Attribute (line 159):
        int_172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 25), 'int')
        # Getting the type of 'self' (line 159)
        self_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'self')
        # Setting the type of the member 'holdCount' of a type (line 159)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), self_173, 'holdCount', int_172)
        
        # Assigning a Num to a Attribute (line 160):
        int_174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 25), 'int')
        # Getting the type of 'self' (line 160)
        self_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'self')
        # Setting the type of the member 'qpktCount' of a type (line 160)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), self_175, 'qpktCount', int_174)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'TaskWorkArea' (line 153)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 0), 'TaskWorkArea', TaskWorkArea)

# Assigning a Call to a Name (line 163):

# Call to TaskWorkArea(...): (line 163)
# Processing the call keyword arguments (line 163)
kwargs_177 = {}
# Getting the type of 'TaskWorkArea' (line 163)
TaskWorkArea_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), 'TaskWorkArea', False)
# Calling TaskWorkArea(args, kwargs) (line 163)
TaskWorkArea_call_result_178 = invoke(stypy.reporting.localization.Localization(__file__, 163, 15), TaskWorkArea_176, *[], **kwargs_177)

# Assigning a type to the variable 'taskWorkArea' (line 163)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), 'taskWorkArea', TaskWorkArea_call_result_178)
# Declaration of the 'Task' class
# Getting the type of 'TaskState' (line 166)
TaskState_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 11), 'TaskState')

class Task(TaskState_179, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 168, 4, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Task.__init__', ['i', 'p', 'w', 'initialState', 'r'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['i', 'p', 'w', 'initialState', 'r'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 169):
        # Getting the type of 'taskWorkArea' (line 169)
        taskWorkArea_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 20), 'taskWorkArea')
        # Obtaining the member 'taskList' of a type (line 169)
        taskList_181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 20), taskWorkArea_180, 'taskList')
        # Getting the type of 'self' (line 169)
        self_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'self')
        # Setting the type of the member 'link' of a type (line 169)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), self_182, 'link', taskList_181)
        
        # Assigning a Name to a Attribute (line 170):
        # Getting the type of 'i' (line 170)
        i_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 21), 'i')
        # Getting the type of 'self' (line 170)
        self_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'self')
        # Setting the type of the member 'ident' of a type (line 170)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), self_184, 'ident', i_183)
        
        # Assigning a Name to a Attribute (line 171):
        # Getting the type of 'p' (line 171)
        p_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 24), 'p')
        # Getting the type of 'self' (line 171)
        self_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'self')
        # Setting the type of the member 'priority' of a type (line 171)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), self_186, 'priority', p_185)
        
        # Assigning a Name to a Attribute (line 172):
        # Getting the type of 'w' (line 172)
        w_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 21), 'w')
        # Getting the type of 'self' (line 172)
        self_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'self')
        # Setting the type of the member 'input' of a type (line 172)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), self_188, 'input', w_187)
        
        # Assigning a Call to a Attribute (line 174):
        
        # Call to isPacketPending(...): (line 174)
        # Processing the call keyword arguments (line 174)
        kwargs_191 = {}
        # Getting the type of 'initialState' (line 174)
        initialState_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 30), 'initialState', False)
        # Obtaining the member 'isPacketPending' of a type (line 174)
        isPacketPending_190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 30), initialState_189, 'isPacketPending')
        # Calling isPacketPending(args, kwargs) (line 174)
        isPacketPending_call_result_192 = invoke(stypy.reporting.localization.Localization(__file__, 174, 30), isPacketPending_190, *[], **kwargs_191)
        
        # Getting the type of 'self' (line 174)
        self_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'self')
        # Setting the type of the member 'packet_pending' of a type (line 174)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), self_193, 'packet_pending', isPacketPending_call_result_192)
        
        # Assigning a Call to a Attribute (line 175):
        
        # Call to isTaskWaiting(...): (line 175)
        # Processing the call keyword arguments (line 175)
        kwargs_196 = {}
        # Getting the type of 'initialState' (line 175)
        initialState_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 28), 'initialState', False)
        # Obtaining the member 'isTaskWaiting' of a type (line 175)
        isTaskWaiting_195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 28), initialState_194, 'isTaskWaiting')
        # Calling isTaskWaiting(args, kwargs) (line 175)
        isTaskWaiting_call_result_197 = invoke(stypy.reporting.localization.Localization(__file__, 175, 28), isTaskWaiting_195, *[], **kwargs_196)
        
        # Getting the type of 'self' (line 175)
        self_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'self')
        # Setting the type of the member 'task_waiting' of a type (line 175)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), self_198, 'task_waiting', isTaskWaiting_call_result_197)
        
        # Assigning a Call to a Attribute (line 176):
        
        # Call to isTaskHolding(...): (line 176)
        # Processing the call keyword arguments (line 176)
        kwargs_201 = {}
        # Getting the type of 'initialState' (line 176)
        initialState_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 28), 'initialState', False)
        # Obtaining the member 'isTaskHolding' of a type (line 176)
        isTaskHolding_200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 28), initialState_199, 'isTaskHolding')
        # Calling isTaskHolding(args, kwargs) (line 176)
        isTaskHolding_call_result_202 = invoke(stypy.reporting.localization.Localization(__file__, 176, 28), isTaskHolding_200, *[], **kwargs_201)
        
        # Getting the type of 'self' (line 176)
        self_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'self')
        # Setting the type of the member 'task_holding' of a type (line 176)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), self_203, 'task_holding', isTaskHolding_call_result_202)
        
        # Assigning a Name to a Attribute (line 178):
        # Getting the type of 'r' (line 178)
        r_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 22), 'r')
        # Getting the type of 'self' (line 178)
        self_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'self')
        # Setting the type of the member 'handle' of a type (line 178)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), self_205, 'handle', r_204)
        
        # Assigning a Name to a Attribute (line 180):
        # Getting the type of 'self' (line 180)
        self_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 32), 'self')
        # Getting the type of 'taskWorkArea' (line 180)
        taskWorkArea_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'taskWorkArea')
        # Setting the type of the member 'taskList' of a type (line 180)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), taskWorkArea_207, 'taskList', self_206)
        
        # Assigning a Name to a Subscript (line 181):
        # Getting the type of 'self' (line 181)
        self_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 34), 'self')
        # Getting the type of 'taskWorkArea' (line 181)
        taskWorkArea_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'taskWorkArea')
        # Obtaining the member 'taskTab' of a type (line 181)
        taskTab_210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), taskWorkArea_209, 'taskTab')
        # Getting the type of 'i' (line 181)
        i_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 29), 'i')
        # Storing an element on a container (line 181)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 8), taskTab_210, (i_211, self_208))
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def fn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fn'
        module_type_store = module_type_store.open_function_context('fn', 183, 4, False)
        # Assigning a type to the variable 'self' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Task.fn.__dict__.__setitem__('stypy_localization', localization)
        Task.fn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Task.fn.__dict__.__setitem__('stypy_type_store', module_type_store)
        Task.fn.__dict__.__setitem__('stypy_function_name', 'Task.fn')
        Task.fn.__dict__.__setitem__('stypy_param_names_list', ['pkt', 'r'])
        Task.fn.__dict__.__setitem__('stypy_varargs_param_name', None)
        Task.fn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Task.fn.__dict__.__setitem__('stypy_call_defaults', defaults)
        Task.fn.__dict__.__setitem__('stypy_call_varargs', varargs)
        Task.fn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Task.fn.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Task.fn', ['pkt', 'r'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fn', localization, ['pkt', 'r'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fn(...)' code ##################

        # Getting the type of 'NotImplementedError' (line 184)
        NotImplementedError_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 14), 'NotImplementedError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 184, 8), NotImplementedError_212, 'raise parameter', BaseException)
        
        # ################# End of 'fn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fn' in the type store
        # Getting the type of 'stypy_return_type' (line 183)
        stypy_return_type_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_213)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fn'
        return stypy_return_type_213


    @norecursion
    def addPacket(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addPacket'
        module_type_store = module_type_store.open_function_context('addPacket', 186, 4, False)
        # Assigning a type to the variable 'self' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Task.addPacket.__dict__.__setitem__('stypy_localization', localization)
        Task.addPacket.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Task.addPacket.__dict__.__setitem__('stypy_type_store', module_type_store)
        Task.addPacket.__dict__.__setitem__('stypy_function_name', 'Task.addPacket')
        Task.addPacket.__dict__.__setitem__('stypy_param_names_list', ['p', 'old'])
        Task.addPacket.__dict__.__setitem__('stypy_varargs_param_name', None)
        Task.addPacket.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Task.addPacket.__dict__.__setitem__('stypy_call_defaults', defaults)
        Task.addPacket.__dict__.__setitem__('stypy_call_varargs', varargs)
        Task.addPacket.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Task.addPacket.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Task.addPacket', ['p', 'old'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'addPacket', localization, ['p', 'old'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'addPacket(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 187)
        # Getting the type of 'self' (line 187)
        self_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 11), 'self')
        # Obtaining the member 'input' of a type (line 187)
        input_215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 11), self_214, 'input')
        # Getting the type of 'None' (line 187)
        None_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 25), 'None')
        
        (may_be_217, more_types_in_union_218) = may_be_none(input_215, None_216)

        if may_be_217:

            if more_types_in_union_218:
                # Runtime conditional SSA (line 187)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 188):
            # Getting the type of 'p' (line 188)
            p_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 25), 'p')
            # Getting the type of 'self' (line 188)
            self_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'self')
            # Setting the type of the member 'input' of a type (line 188)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 12), self_220, 'input', p_219)
            
            # Assigning a Name to a Attribute (line 189):
            # Getting the type of 'True' (line 189)
            True_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 34), 'True')
            # Getting the type of 'self' (line 189)
            self_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'self')
            # Setting the type of the member 'packet_pending' of a type (line 189)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 12), self_222, 'packet_pending', True_221)
            
            # Getting the type of 'self' (line 190)
            self_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 15), 'self')
            # Obtaining the member 'priority' of a type (line 190)
            priority_224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 15), self_223, 'priority')
            # Getting the type of 'old' (line 190)
            old_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 31), 'old')
            # Obtaining the member 'priority' of a type (line 190)
            priority_226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 31), old_225, 'priority')
            # Applying the binary operator '>' (line 190)
            result_gt_227 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 15), '>', priority_224, priority_226)
            
            # Testing if the type of an if condition is none (line 190)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 190, 12), result_gt_227):
                pass
            else:
                
                # Testing the type of an if condition (line 190)
                if_condition_228 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 190, 12), result_gt_227)
                # Assigning a type to the variable 'if_condition_228' (line 190)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'if_condition_228', if_condition_228)
                # SSA begins for if statement (line 190)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'self' (line 191)
                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 23), 'self')
                # Assigning a type to the variable 'stypy_return_type' (line 191)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'stypy_return_type', self_229)
                # SSA join for if statement (line 190)
                module_type_store = module_type_store.join_ssa_context()
                


            if more_types_in_union_218:
                # Runtime conditional SSA for else branch (line 187)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_217) or more_types_in_union_218):
            
            # Call to append_to(...): (line 193)
            # Processing the call arguments (line 193)
            # Getting the type of 'self' (line 193)
            self_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 24), 'self', False)
            # Obtaining the member 'input' of a type (line 193)
            input_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 24), self_232, 'input')
            # Processing the call keyword arguments (line 193)
            kwargs_234 = {}
            # Getting the type of 'p' (line 193)
            p_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'p', False)
            # Obtaining the member 'append_to' of a type (line 193)
            append_to_231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), p_230, 'append_to')
            # Calling append_to(args, kwargs) (line 193)
            append_to_call_result_235 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), append_to_231, *[input_233], **kwargs_234)
            

            if (may_be_217 and more_types_in_union_218):
                # SSA join for if statement (line 187)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'old' (line 194)
        old_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 15), 'old')
        # Assigning a type to the variable 'stypy_return_type' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'stypy_return_type', old_236)
        
        # ################# End of 'addPacket(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addPacket' in the type store
        # Getting the type of 'stypy_return_type' (line 186)
        stypy_return_type_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addPacket'
        return stypy_return_type_237


    @norecursion
    def runTask(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'runTask'
        module_type_store = module_type_store.open_function_context('runTask', 196, 4, False)
        # Assigning a type to the variable 'self' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Task.runTask.__dict__.__setitem__('stypy_localization', localization)
        Task.runTask.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Task.runTask.__dict__.__setitem__('stypy_type_store', module_type_store)
        Task.runTask.__dict__.__setitem__('stypy_function_name', 'Task.runTask')
        Task.runTask.__dict__.__setitem__('stypy_param_names_list', [])
        Task.runTask.__dict__.__setitem__('stypy_varargs_param_name', None)
        Task.runTask.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Task.runTask.__dict__.__setitem__('stypy_call_defaults', defaults)
        Task.runTask.__dict__.__setitem__('stypy_call_varargs', varargs)
        Task.runTask.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Task.runTask.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Task.runTask', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'runTask', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'runTask(...)' code ##################

        
        # Call to isWaitingWithPacket(...): (line 197)
        # Processing the call keyword arguments (line 197)
        kwargs_240 = {}
        # Getting the type of 'self' (line 197)
        self_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 11), 'self', False)
        # Obtaining the member 'isWaitingWithPacket' of a type (line 197)
        isWaitingWithPacket_239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 11), self_238, 'isWaitingWithPacket')
        # Calling isWaitingWithPacket(args, kwargs) (line 197)
        isWaitingWithPacket_call_result_241 = invoke(stypy.reporting.localization.Localization(__file__, 197, 11), isWaitingWithPacket_239, *[], **kwargs_240)
        
        # Testing if the type of an if condition is none (line 197)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 197, 8), isWaitingWithPacket_call_result_241):
            
            # Assigning a Name to a Name (line 205):
            # Getting the type of 'None' (line 205)
            None_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 18), 'None')
            # Assigning a type to the variable 'msg' (line 205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'msg', None_261)
        else:
            
            # Testing the type of an if condition (line 197)
            if_condition_242 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 8), isWaitingWithPacket_call_result_241)
            # Assigning a type to the variable 'if_condition_242' (line 197)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'if_condition_242', if_condition_242)
            # SSA begins for if statement (line 197)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 198):
            # Getting the type of 'self' (line 198)
            self_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 18), 'self')
            # Obtaining the member 'input' of a type (line 198)
            input_244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 18), self_243, 'input')
            # Assigning a type to the variable 'msg' (line 198)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'msg', input_244)
            
            # Assigning a Attribute to a Attribute (line 199):
            # Getting the type of 'msg' (line 199)
            msg_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 25), 'msg')
            # Obtaining the member 'link' of a type (line 199)
            link_246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 25), msg_245, 'link')
            # Getting the type of 'self' (line 199)
            self_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'self')
            # Setting the type of the member 'input' of a type (line 199)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 12), self_247, 'input', link_246)
            
            # Type idiom detected: calculating its left and rigth part (line 200)
            # Getting the type of 'self' (line 200)
            self_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'self')
            # Obtaining the member 'input' of a type (line 200)
            input_249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 15), self_248, 'input')
            # Getting the type of 'None' (line 200)
            None_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 29), 'None')
            
            (may_be_251, more_types_in_union_252) = may_be_none(input_249, None_250)

            if may_be_251:

                if more_types_in_union_252:
                    # Runtime conditional SSA (line 200)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to running(...): (line 201)
                # Processing the call keyword arguments (line 201)
                kwargs_255 = {}
                # Getting the type of 'self' (line 201)
                self_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'self', False)
                # Obtaining the member 'running' of a type (line 201)
                running_254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 16), self_253, 'running')
                # Calling running(args, kwargs) (line 201)
                running_call_result_256 = invoke(stypy.reporting.localization.Localization(__file__, 201, 16), running_254, *[], **kwargs_255)
                

                if more_types_in_union_252:
                    # Runtime conditional SSA for else branch (line 200)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_251) or more_types_in_union_252):
                
                # Call to packetPending(...): (line 203)
                # Processing the call keyword arguments (line 203)
                kwargs_259 = {}
                # Getting the type of 'self' (line 203)
                self_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'self', False)
                # Obtaining the member 'packetPending' of a type (line 203)
                packetPending_258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 16), self_257, 'packetPending')
                # Calling packetPending(args, kwargs) (line 203)
                packetPending_call_result_260 = invoke(stypy.reporting.localization.Localization(__file__, 203, 16), packetPending_258, *[], **kwargs_259)
                

                if (may_be_251 and more_types_in_union_252):
                    # SSA join for if statement (line 200)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA branch for the else part of an if statement (line 197)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Name (line 205):
            # Getting the type of 'None' (line 205)
            None_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 18), 'None')
            # Assigning a type to the variable 'msg' (line 205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'msg', None_261)
            # SSA join for if statement (line 197)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'self' (line 207)
        self_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'self')
        
        # Call to fn(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'msg' (line 208)
        msg_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 23), 'msg', False)
        # Getting the type of 'self' (line 208)
        self_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 28), 'self', False)
        # Obtaining the member 'handle' of a type (line 208)
        handle_267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 28), self_266, 'handle')
        # Processing the call keyword arguments (line 208)
        kwargs_268 = {}
        # Getting the type of 'self' (line 208)
        self_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'self', False)
        # Obtaining the member 'fn' of a type (line 208)
        fn_264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 15), self_263, 'fn')
        # Calling fn(args, kwargs) (line 208)
        fn_call_result_269 = invoke(stypy.reporting.localization.Localization(__file__, 208, 15), fn_264, *[msg_265, handle_267], **kwargs_268)
        
        # Assigning a type to the variable 'stypy_return_type' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'stypy_return_type', fn_call_result_269)
        
        # ################# End of 'runTask(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'runTask' in the type store
        # Getting the type of 'stypy_return_type' (line 196)
        stypy_return_type_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_270)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'runTask'
        return stypy_return_type_270


    @norecursion
    def waitTask(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'waitTask'
        module_type_store = module_type_store.open_function_context('waitTask', 210, 4, False)
        # Assigning a type to the variable 'self' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Task.waitTask.__dict__.__setitem__('stypy_localization', localization)
        Task.waitTask.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Task.waitTask.__dict__.__setitem__('stypy_type_store', module_type_store)
        Task.waitTask.__dict__.__setitem__('stypy_function_name', 'Task.waitTask')
        Task.waitTask.__dict__.__setitem__('stypy_param_names_list', [])
        Task.waitTask.__dict__.__setitem__('stypy_varargs_param_name', None)
        Task.waitTask.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Task.waitTask.__dict__.__setitem__('stypy_call_defaults', defaults)
        Task.waitTask.__dict__.__setitem__('stypy_call_varargs', varargs)
        Task.waitTask.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Task.waitTask.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Task.waitTask', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'waitTask', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'waitTask(...)' code ##################

        
        # Assigning a Name to a Attribute (line 211):
        # Getting the type of 'True' (line 211)
        True_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 28), 'True')
        # Getting the type of 'self' (line 211)
        self_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'self')
        # Setting the type of the member 'task_waiting' of a type (line 211)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), self_272, 'task_waiting', True_271)
        # Getting the type of 'self' (line 212)
        self_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'stypy_return_type', self_273)
        
        # ################# End of 'waitTask(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'waitTask' in the type store
        # Getting the type of 'stypy_return_type' (line 210)
        stypy_return_type_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_274)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'waitTask'
        return stypy_return_type_274


    @norecursion
    def hold(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'hold'
        module_type_store = module_type_store.open_function_context('hold', 214, 4, False)
        # Assigning a type to the variable 'self' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Task.hold.__dict__.__setitem__('stypy_localization', localization)
        Task.hold.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Task.hold.__dict__.__setitem__('stypy_type_store', module_type_store)
        Task.hold.__dict__.__setitem__('stypy_function_name', 'Task.hold')
        Task.hold.__dict__.__setitem__('stypy_param_names_list', [])
        Task.hold.__dict__.__setitem__('stypy_varargs_param_name', None)
        Task.hold.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Task.hold.__dict__.__setitem__('stypy_call_defaults', defaults)
        Task.hold.__dict__.__setitem__('stypy_call_varargs', varargs)
        Task.hold.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Task.hold.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Task.hold', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'hold', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'hold(...)' code ##################

        
        # Getting the type of 'taskWorkArea' (line 215)
        taskWorkArea_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'taskWorkArea')
        # Obtaining the member 'holdCount' of a type (line 215)
        holdCount_276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), taskWorkArea_275, 'holdCount')
        int_277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 34), 'int')
        # Applying the binary operator '+=' (line 215)
        result_iadd_278 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 8), '+=', holdCount_276, int_277)
        # Getting the type of 'taskWorkArea' (line 215)
        taskWorkArea_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'taskWorkArea')
        # Setting the type of the member 'holdCount' of a type (line 215)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), taskWorkArea_279, 'holdCount', result_iadd_278)
        
        
        # Assigning a Name to a Attribute (line 216):
        # Getting the type of 'True' (line 216)
        True_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 28), 'True')
        # Getting the type of 'self' (line 216)
        self_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'self')
        # Setting the type of the member 'task_holding' of a type (line 216)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), self_281, 'task_holding', True_280)
        # Getting the type of 'self' (line 217)
        self_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'self')
        # Obtaining the member 'link' of a type (line 217)
        link_283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 15), self_282, 'link')
        # Assigning a type to the variable 'stypy_return_type' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'stypy_return_type', link_283)
        
        # ################# End of 'hold(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'hold' in the type store
        # Getting the type of 'stypy_return_type' (line 214)
        stypy_return_type_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_284)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'hold'
        return stypy_return_type_284


    @norecursion
    def release(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'release'
        module_type_store = module_type_store.open_function_context('release', 219, 4, False)
        # Assigning a type to the variable 'self' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Task.release.__dict__.__setitem__('stypy_localization', localization)
        Task.release.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Task.release.__dict__.__setitem__('stypy_type_store', module_type_store)
        Task.release.__dict__.__setitem__('stypy_function_name', 'Task.release')
        Task.release.__dict__.__setitem__('stypy_param_names_list', ['i'])
        Task.release.__dict__.__setitem__('stypy_varargs_param_name', None)
        Task.release.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Task.release.__dict__.__setitem__('stypy_call_defaults', defaults)
        Task.release.__dict__.__setitem__('stypy_call_varargs', varargs)
        Task.release.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Task.release.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Task.release', ['i'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'release', localization, ['i'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'release(...)' code ##################

        
        # Assigning a Call to a Name (line 220):
        
        # Call to findtcb(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'i' (line 220)
        i_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 25), 'i', False)
        # Processing the call keyword arguments (line 220)
        kwargs_288 = {}
        # Getting the type of 'self' (line 220)
        self_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'self', False)
        # Obtaining the member 'findtcb' of a type (line 220)
        findtcb_286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 12), self_285, 'findtcb')
        # Calling findtcb(args, kwargs) (line 220)
        findtcb_call_result_289 = invoke(stypy.reporting.localization.Localization(__file__, 220, 12), findtcb_286, *[i_287], **kwargs_288)
        
        # Assigning a type to the variable 't' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 't', findtcb_call_result_289)
        
        # Assigning a Name to a Attribute (line 221):
        # Getting the type of 'False' (line 221)
        False_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 25), 'False')
        # Getting the type of 't' (line 221)
        t_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 't')
        # Setting the type of the member 'task_holding' of a type (line 221)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), t_291, 'task_holding', False_290)
        
        # Getting the type of 't' (line 222)
        t_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 11), 't')
        # Obtaining the member 'priority' of a type (line 222)
        priority_293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 11), t_292, 'priority')
        # Getting the type of 'self' (line 222)
        self_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 24), 'self')
        # Obtaining the member 'priority' of a type (line 222)
        priority_295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 24), self_294, 'priority')
        # Applying the binary operator '>' (line 222)
        result_gt_296 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 11), '>', priority_293, priority_295)
        
        # Testing if the type of an if condition is none (line 222)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 222, 8), result_gt_296):
            # Getting the type of 'self' (line 225)
            self_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 19), 'self')
            # Assigning a type to the variable 'stypy_return_type' (line 225)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'stypy_return_type', self_299)
        else:
            
            # Testing the type of an if condition (line 222)
            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 222, 8), result_gt_296)
            # Assigning a type to the variable 'if_condition_297' (line 222)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'if_condition_297', if_condition_297)
            # SSA begins for if statement (line 222)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 't' (line 223)
            t_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 19), 't')
            # Assigning a type to the variable 'stypy_return_type' (line 223)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'stypy_return_type', t_298)
            # SSA branch for the else part of an if statement (line 222)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'self' (line 225)
            self_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 19), 'self')
            # Assigning a type to the variable 'stypy_return_type' (line 225)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'stypy_return_type', self_299)
            # SSA join for if statement (line 222)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'release(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'release' in the type store
        # Getting the type of 'stypy_return_type' (line 219)
        stypy_return_type_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_300)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'release'
        return stypy_return_type_300


    @norecursion
    def qpkt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'qpkt'
        module_type_store = module_type_store.open_function_context('qpkt', 227, 4, False)
        # Assigning a type to the variable 'self' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Task.qpkt.__dict__.__setitem__('stypy_localization', localization)
        Task.qpkt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Task.qpkt.__dict__.__setitem__('stypy_type_store', module_type_store)
        Task.qpkt.__dict__.__setitem__('stypy_function_name', 'Task.qpkt')
        Task.qpkt.__dict__.__setitem__('stypy_param_names_list', ['pkt'])
        Task.qpkt.__dict__.__setitem__('stypy_varargs_param_name', None)
        Task.qpkt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Task.qpkt.__dict__.__setitem__('stypy_call_defaults', defaults)
        Task.qpkt.__dict__.__setitem__('stypy_call_varargs', varargs)
        Task.qpkt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Task.qpkt.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Task.qpkt', ['pkt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'qpkt', localization, ['pkt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'qpkt(...)' code ##################

        
        # Assigning a Call to a Name (line 228):
        
        # Call to findtcb(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'pkt' (line 228)
        pkt_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 25), 'pkt', False)
        # Obtaining the member 'ident' of a type (line 228)
        ident_304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 25), pkt_303, 'ident')
        # Processing the call keyword arguments (line 228)
        kwargs_305 = {}
        # Getting the type of 'self' (line 228)
        self_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'self', False)
        # Obtaining the member 'findtcb' of a type (line 228)
        findtcb_302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 12), self_301, 'findtcb')
        # Calling findtcb(args, kwargs) (line 228)
        findtcb_call_result_306 = invoke(stypy.reporting.localization.Localization(__file__, 228, 12), findtcb_302, *[ident_304], **kwargs_305)
        
        # Assigning a type to the variable 't' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 't', findtcb_call_result_306)
        
        # Getting the type of 'taskWorkArea' (line 229)
        taskWorkArea_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'taskWorkArea')
        # Obtaining the member 'qpktCount' of a type (line 229)
        qpktCount_308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), taskWorkArea_307, 'qpktCount')
        int_309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 34), 'int')
        # Applying the binary operator '+=' (line 229)
        result_iadd_310 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 8), '+=', qpktCount_308, int_309)
        # Getting the type of 'taskWorkArea' (line 229)
        taskWorkArea_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'taskWorkArea')
        # Setting the type of the member 'qpktCount' of a type (line 229)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), taskWorkArea_311, 'qpktCount', result_iadd_310)
        
        
        # Assigning a Name to a Attribute (line 230):
        # Getting the type of 'None' (line 230)
        None_312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 19), 'None')
        # Getting the type of 'pkt' (line 230)
        pkt_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'pkt')
        # Setting the type of the member 'link' of a type (line 230)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), pkt_313, 'link', None_312)
        
        # Assigning a Attribute to a Attribute (line 231):
        # Getting the type of 'self' (line 231)
        self_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'self')
        # Obtaining the member 'ident' of a type (line 231)
        ident_315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 20), self_314, 'ident')
        # Getting the type of 'pkt' (line 231)
        pkt_316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'pkt')
        # Setting the type of the member 'ident' of a type (line 231)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 8), pkt_316, 'ident', ident_315)
        
        # Call to addPacket(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'pkt' (line 232)
        pkt_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 27), 'pkt', False)
        # Getting the type of 'self' (line 232)
        self_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 32), 'self', False)
        # Processing the call keyword arguments (line 232)
        kwargs_321 = {}
        # Getting the type of 't' (line 232)
        t_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 15), 't', False)
        # Obtaining the member 'addPacket' of a type (line 232)
        addPacket_318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 15), t_317, 'addPacket')
        # Calling addPacket(args, kwargs) (line 232)
        addPacket_call_result_322 = invoke(stypy.reporting.localization.Localization(__file__, 232, 15), addPacket_318, *[pkt_319, self_320], **kwargs_321)
        
        # Assigning a type to the variable 'stypy_return_type' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'stypy_return_type', addPacket_call_result_322)
        
        # ################# End of 'qpkt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'qpkt' in the type store
        # Getting the type of 'stypy_return_type' (line 227)
        stypy_return_type_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_323)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'qpkt'
        return stypy_return_type_323


    @norecursion
    def findtcb(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'findtcb'
        module_type_store = module_type_store.open_function_context('findtcb', 234, 4, False)
        # Assigning a type to the variable 'self' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Task.findtcb.__dict__.__setitem__('stypy_localization', localization)
        Task.findtcb.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Task.findtcb.__dict__.__setitem__('stypy_type_store', module_type_store)
        Task.findtcb.__dict__.__setitem__('stypy_function_name', 'Task.findtcb')
        Task.findtcb.__dict__.__setitem__('stypy_param_names_list', ['id'])
        Task.findtcb.__dict__.__setitem__('stypy_varargs_param_name', None)
        Task.findtcb.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Task.findtcb.__dict__.__setitem__('stypy_call_defaults', defaults)
        Task.findtcb.__dict__.__setitem__('stypy_call_varargs', varargs)
        Task.findtcb.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Task.findtcb.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Task.findtcb', ['id'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'findtcb', localization, ['id'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'findtcb(...)' code ##################

        
        # Assigning a Subscript to a Name (line 235):
        
        # Obtaining the type of the subscript
        # Getting the type of 'id' (line 235)
        id_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 33), 'id')
        # Getting the type of 'taskWorkArea' (line 235)
        taskWorkArea_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'taskWorkArea')
        # Obtaining the member 'taskTab' of a type (line 235)
        taskTab_326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), taskWorkArea_325, 'taskTab')
        # Obtaining the member '__getitem__' of a type (line 235)
        getitem___327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), taskTab_326, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 235)
        subscript_call_result_328 = invoke(stypy.reporting.localization.Localization(__file__, 235, 12), getitem___327, id_324)
        
        # Assigning a type to the variable 't' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 't', subscript_call_result_328)
        
        # Type idiom detected: calculating its left and rigth part (line 236)
        # Getting the type of 't' (line 236)
        t_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 11), 't')
        # Getting the type of 'None' (line 236)
        None_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 16), 'None')
        
        (may_be_331, more_types_in_union_332) = may_be_none(t_329, None_330)

        if may_be_331:

            if more_types_in_union_332:
                # Runtime conditional SSA (line 236)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to Exception(...): (line 237)
            # Processing the call arguments (line 237)
            str_334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 28), 'str', 'Bad task id %d')
            # Getting the type of 'id' (line 237)
            id_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 47), 'id', False)
            # Applying the binary operator '%' (line 237)
            result_mod_336 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 28), '%', str_334, id_335)
            
            # Processing the call keyword arguments (line 237)
            kwargs_337 = {}
            # Getting the type of 'Exception' (line 237)
            Exception_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 18), 'Exception', False)
            # Calling Exception(args, kwargs) (line 237)
            Exception_call_result_338 = invoke(stypy.reporting.localization.Localization(__file__, 237, 18), Exception_333, *[result_mod_336], **kwargs_337)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 237, 12), Exception_call_result_338, 'raise parameter', BaseException)

            if more_types_in_union_332:
                # SSA join for if statement (line 236)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 't' (line 236)
        t_339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 't')
        # Assigning a type to the variable 't' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 't', remove_type_from_union(t_339, types.NoneType))
        # Getting the type of 't' (line 238)
        t_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 15), 't')
        # Assigning a type to the variable 'stypy_return_type' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'stypy_return_type', t_340)
        
        # ################# End of 'findtcb(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'findtcb' in the type store
        # Getting the type of 'stypy_return_type' (line 234)
        stypy_return_type_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_341)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'findtcb'
        return stypy_return_type_341


# Assigning a type to the variable 'Task' (line 166)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'Task', Task)
# Declaration of the 'DeviceTask' class
# Getting the type of 'Task' (line 244)
Task_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 17), 'Task')

class DeviceTask(Task_342, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 245, 4, False)
        # Assigning a type to the variable 'self' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DeviceTask.__init__', ['i', 'p', 'w', 's', 'r'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['i', 'p', 'w', 's', 'r'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'self' (line 246)
        self_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 22), 'self', False)
        # Getting the type of 'i' (line 246)
        i_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 28), 'i', False)
        # Getting the type of 'p' (line 246)
        p_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 31), 'p', False)
        # Getting the type of 'w' (line 246)
        w_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 34), 'w', False)
        # Getting the type of 's' (line 246)
        s_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 37), 's', False)
        # Getting the type of 'r' (line 246)
        r_350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 40), 'r', False)
        # Processing the call keyword arguments (line 246)
        kwargs_351 = {}
        # Getting the type of 'Task' (line 246)
        Task_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'Task', False)
        # Obtaining the member '__init__' of a type (line 246)
        init___344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), Task_343, '__init__')
        # Calling __init__(args, kwargs) (line 246)
        init___call_result_352 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), init___344, *[self_345, i_346, p_347, w_348, s_349, r_350], **kwargs_351)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def fn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fn'
        module_type_store = module_type_store.open_function_context('fn', 248, 4, False)
        # Assigning a type to the variable 'self' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DeviceTask.fn.__dict__.__setitem__('stypy_localization', localization)
        DeviceTask.fn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DeviceTask.fn.__dict__.__setitem__('stypy_type_store', module_type_store)
        DeviceTask.fn.__dict__.__setitem__('stypy_function_name', 'DeviceTask.fn')
        DeviceTask.fn.__dict__.__setitem__('stypy_param_names_list', ['pkt', 'r'])
        DeviceTask.fn.__dict__.__setitem__('stypy_varargs_param_name', None)
        DeviceTask.fn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DeviceTask.fn.__dict__.__setitem__('stypy_call_defaults', defaults)
        DeviceTask.fn.__dict__.__setitem__('stypy_call_varargs', varargs)
        DeviceTask.fn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DeviceTask.fn.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DeviceTask.fn', ['pkt', 'r'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fn', localization, ['pkt', 'r'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fn(...)' code ##################

        
        # Assigning a Name to a Name (line 249):
        # Getting the type of 'r' (line 249)
        r_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'r')
        # Assigning a type to the variable 'd' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'd', r_353)
        # Evaluating assert statement condition
        
        # Call to isinstance(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'd' (line 250)
        d_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 26), 'd', False)
        # Getting the type of 'DeviceTaskRec' (line 250)
        DeviceTaskRec_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 29), 'DeviceTaskRec', False)
        # Processing the call keyword arguments (line 250)
        kwargs_357 = {}
        # Getting the type of 'isinstance' (line 250)
        isinstance_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 250)
        isinstance_call_result_358 = invoke(stypy.reporting.localization.Localization(__file__, 250, 15), isinstance_354, *[d_355, DeviceTaskRec_356], **kwargs_357)
        
        assert_359 = isinstance_call_result_358
        # Assigning a type to the variable 'assert_359' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'assert_359', isinstance_call_result_358)
        
        # Type idiom detected: calculating its left and rigth part (line 251)
        # Getting the type of 'pkt' (line 251)
        pkt_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 11), 'pkt')
        # Getting the type of 'None' (line 251)
        None_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 18), 'None')
        
        (may_be_362, more_types_in_union_363) = may_be_none(pkt_360, None_361)

        if may_be_362:

            if more_types_in_union_363:
                # Runtime conditional SSA (line 251)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 252):
            # Getting the type of 'd' (line 252)
            d_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 18), 'd')
            # Obtaining the member 'pending' of a type (line 252)
            pending_365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 18), d_364, 'pending')
            # Assigning a type to the variable 'pkt' (line 252)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'pkt', pending_365)
            
            # Type idiom detected: calculating its left and rigth part (line 253)
            # Getting the type of 'pkt' (line 253)
            pkt_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 15), 'pkt')
            # Getting the type of 'None' (line 253)
            None_367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 22), 'None')
            
            (may_be_368, more_types_in_union_369) = may_be_none(pkt_366, None_367)

            if may_be_368:

                if more_types_in_union_369:
                    # Runtime conditional SSA (line 253)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to waitTask(...): (line 254)
                # Processing the call keyword arguments (line 254)
                kwargs_372 = {}
                # Getting the type of 'self' (line 254)
                self_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 23), 'self', False)
                # Obtaining the member 'waitTask' of a type (line 254)
                waitTask_371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 23), self_370, 'waitTask')
                # Calling waitTask(args, kwargs) (line 254)
                waitTask_call_result_373 = invoke(stypy.reporting.localization.Localization(__file__, 254, 23), waitTask_371, *[], **kwargs_372)
                
                # Assigning a type to the variable 'stypy_return_type' (line 254)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'stypy_return_type', waitTask_call_result_373)

                if more_types_in_union_369:
                    # Runtime conditional SSA for else branch (line 253)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_368) or more_types_in_union_369):
                
                # Assigning a Name to a Attribute (line 256):
                # Getting the type of 'None' (line 256)
                None_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 28), 'None')
                # Getting the type of 'd' (line 256)
                d_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 16), 'd')
                # Setting the type of the member 'pending' of a type (line 256)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 16), d_375, 'pending', None_374)
                
                # Call to qpkt(...): (line 257)
                # Processing the call arguments (line 257)
                # Getting the type of 'pkt' (line 257)
                pkt_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 33), 'pkt', False)
                # Processing the call keyword arguments (line 257)
                kwargs_379 = {}
                # Getting the type of 'self' (line 257)
                self_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 23), 'self', False)
                # Obtaining the member 'qpkt' of a type (line 257)
                qpkt_377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 23), self_376, 'qpkt')
                # Calling qpkt(args, kwargs) (line 257)
                qpkt_call_result_380 = invoke(stypy.reporting.localization.Localization(__file__, 257, 23), qpkt_377, *[pkt_378], **kwargs_379)
                
                # Assigning a type to the variable 'stypy_return_type' (line 257)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'stypy_return_type', qpkt_call_result_380)

                if (may_be_368 and more_types_in_union_369):
                    # SSA join for if statement (line 253)
                    module_type_store = module_type_store.join_ssa_context()


            
            # Getting the type of 'pkt' (line 253)
            pkt_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'pkt')
            # Assigning a type to the variable 'pkt' (line 253)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'pkt', remove_type_from_union(pkt_381, types.NoneType))

            if more_types_in_union_363:
                # Runtime conditional SSA for else branch (line 251)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_362) or more_types_in_union_363):
            
            # Assigning a Name to a Attribute (line 259):
            # Getting the type of 'pkt' (line 259)
            pkt_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 24), 'pkt')
            # Getting the type of 'd' (line 259)
            d_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'd')
            # Setting the type of the member 'pending' of a type (line 259)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 12), d_383, 'pending', pkt_382)
            # Getting the type of 'tracing' (line 260)
            tracing_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 15), 'tracing')
            # Testing if the type of an if condition is none (line 260)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 260, 12), tracing_384):
                pass
            else:
                
                # Testing the type of an if condition (line 260)
                if_condition_385 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 12), tracing_384)
                # Assigning a type to the variable 'if_condition_385' (line 260)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'if_condition_385', if_condition_385)
                # SSA begins for if statement (line 260)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to trace(...): (line 260)
                # Processing the call arguments (line 260)
                
                # Call to str(...): (line 260)
                # Processing the call arguments (line 260)
                # Getting the type of 'pkt' (line 260)
                pkt_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 34), 'pkt', False)
                # Obtaining the member 'datum' of a type (line 260)
                datum_389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 34), pkt_388, 'datum')
                # Processing the call keyword arguments (line 260)
                kwargs_390 = {}
                # Getting the type of 'str' (line 260)
                str_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 30), 'str', False)
                # Calling str(args, kwargs) (line 260)
                str_call_result_391 = invoke(stypy.reporting.localization.Localization(__file__, 260, 30), str_387, *[datum_389], **kwargs_390)
                
                # Processing the call keyword arguments (line 260)
                kwargs_392 = {}
                # Getting the type of 'trace' (line 260)
                trace_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 24), 'trace', False)
                # Calling trace(args, kwargs) (line 260)
                trace_call_result_393 = invoke(stypy.reporting.localization.Localization(__file__, 260, 24), trace_386, *[str_call_result_391], **kwargs_392)
                
                # SSA join for if statement (line 260)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to hold(...): (line 261)
            # Processing the call keyword arguments (line 261)
            kwargs_396 = {}
            # Getting the type of 'self' (line 261)
            self_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 19), 'self', False)
            # Obtaining the member 'hold' of a type (line 261)
            hold_395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 19), self_394, 'hold')
            # Calling hold(args, kwargs) (line 261)
            hold_call_result_397 = invoke(stypy.reporting.localization.Localization(__file__, 261, 19), hold_395, *[], **kwargs_396)
            
            # Assigning a type to the variable 'stypy_return_type' (line 261)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'stypy_return_type', hold_call_result_397)

            if (may_be_362 and more_types_in_union_363):
                # SSA join for if statement (line 251)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'fn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fn' in the type store
        # Getting the type of 'stypy_return_type' (line 248)
        stypy_return_type_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_398)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fn'
        return stypy_return_type_398


# Assigning a type to the variable 'DeviceTask' (line 244)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 0), 'DeviceTask', DeviceTask)
# Declaration of the 'HandlerTask' class
# Getting the type of 'Task' (line 264)
Task_399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 18), 'Task')

class HandlerTask(Task_399, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 265, 4, False)
        # Assigning a type to the variable 'self' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerTask.__init__', ['i', 'p', 'w', 's', 'r'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['i', 'p', 'w', 's', 'r'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'self' (line 266)
        self_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 22), 'self', False)
        # Getting the type of 'i' (line 266)
        i_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 28), 'i', False)
        # Getting the type of 'p' (line 266)
        p_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 31), 'p', False)
        # Getting the type of 'w' (line 266)
        w_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 34), 'w', False)
        # Getting the type of 's' (line 266)
        s_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 37), 's', False)
        # Getting the type of 'r' (line 266)
        r_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 40), 'r', False)
        # Processing the call keyword arguments (line 266)
        kwargs_408 = {}
        # Getting the type of 'Task' (line 266)
        Task_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'Task', False)
        # Obtaining the member '__init__' of a type (line 266)
        init___401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 8), Task_400, '__init__')
        # Calling __init__(args, kwargs) (line 266)
        init___call_result_409 = invoke(stypy.reporting.localization.Localization(__file__, 266, 8), init___401, *[self_402, i_403, p_404, w_405, s_406, r_407], **kwargs_408)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def fn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fn'
        module_type_store = module_type_store.open_function_context('fn', 268, 4, False)
        # Assigning a type to the variable 'self' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerTask.fn.__dict__.__setitem__('stypy_localization', localization)
        HandlerTask.fn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerTask.fn.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerTask.fn.__dict__.__setitem__('stypy_function_name', 'HandlerTask.fn')
        HandlerTask.fn.__dict__.__setitem__('stypy_param_names_list', ['pkt', 'r'])
        HandlerTask.fn.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerTask.fn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerTask.fn.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerTask.fn.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerTask.fn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerTask.fn.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerTask.fn', ['pkt', 'r'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fn', localization, ['pkt', 'r'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fn(...)' code ##################

        
        # Assigning a Name to a Name (line 269):
        # Getting the type of 'r' (line 269)
        r_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'r')
        # Assigning a type to the variable 'h' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'h', r_410)
        # Evaluating assert statement condition
        
        # Call to isinstance(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'h' (line 270)
        h_412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 26), 'h', False)
        # Getting the type of 'HandlerTaskRec' (line 270)
        HandlerTaskRec_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 29), 'HandlerTaskRec', False)
        # Processing the call keyword arguments (line 270)
        kwargs_414 = {}
        # Getting the type of 'isinstance' (line 270)
        isinstance_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 270)
        isinstance_call_result_415 = invoke(stypy.reporting.localization.Localization(__file__, 270, 15), isinstance_411, *[h_412, HandlerTaskRec_413], **kwargs_414)
        
        assert_416 = isinstance_call_result_415
        # Assigning a type to the variable 'assert_416' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'assert_416', isinstance_call_result_415)
        
        # Type idiom detected: calculating its left and rigth part (line 271)
        # Getting the type of 'pkt' (line 271)
        pkt_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'pkt')
        # Getting the type of 'None' (line 271)
        None_418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 22), 'None')
        
        (may_be_419, more_types_in_union_420) = may_not_be_none(pkt_417, None_418)

        if may_be_419:

            if more_types_in_union_420:
                # Runtime conditional SSA (line 271)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 'pkt' (line 272)
            pkt_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 15), 'pkt')
            # Obtaining the member 'kind' of a type (line 272)
            kind_422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 15), pkt_421, 'kind')
            # Getting the type of 'K_WORK' (line 272)
            K_WORK_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 27), 'K_WORK')
            # Applying the binary operator '==' (line 272)
            result_eq_424 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 15), '==', kind_422, K_WORK_423)
            
            # Testing if the type of an if condition is none (line 272)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 272, 12), result_eq_424):
                
                # Call to deviceInAdd(...): (line 275)
                # Processing the call arguments (line 275)
                # Getting the type of 'pkt' (line 275)
                pkt_433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 30), 'pkt', False)
                # Processing the call keyword arguments (line 275)
                kwargs_434 = {}
                # Getting the type of 'h' (line 275)
                h_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'h', False)
                # Obtaining the member 'deviceInAdd' of a type (line 275)
                deviceInAdd_432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 16), h_431, 'deviceInAdd')
                # Calling deviceInAdd(args, kwargs) (line 275)
                deviceInAdd_call_result_435 = invoke(stypy.reporting.localization.Localization(__file__, 275, 16), deviceInAdd_432, *[pkt_433], **kwargs_434)
                
            else:
                
                # Testing the type of an if condition (line 272)
                if_condition_425 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 12), result_eq_424)
                # Assigning a type to the variable 'if_condition_425' (line 272)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'if_condition_425', if_condition_425)
                # SSA begins for if statement (line 272)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to workInAdd(...): (line 273)
                # Processing the call arguments (line 273)
                # Getting the type of 'pkt' (line 273)
                pkt_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 28), 'pkt', False)
                # Processing the call keyword arguments (line 273)
                kwargs_429 = {}
                # Getting the type of 'h' (line 273)
                h_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'h', False)
                # Obtaining the member 'workInAdd' of a type (line 273)
                workInAdd_427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 16), h_426, 'workInAdd')
                # Calling workInAdd(args, kwargs) (line 273)
                workInAdd_call_result_430 = invoke(stypy.reporting.localization.Localization(__file__, 273, 16), workInAdd_427, *[pkt_428], **kwargs_429)
                
                # SSA branch for the else part of an if statement (line 272)
                module_type_store.open_ssa_branch('else')
                
                # Call to deviceInAdd(...): (line 275)
                # Processing the call arguments (line 275)
                # Getting the type of 'pkt' (line 275)
                pkt_433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 30), 'pkt', False)
                # Processing the call keyword arguments (line 275)
                kwargs_434 = {}
                # Getting the type of 'h' (line 275)
                h_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'h', False)
                # Obtaining the member 'deviceInAdd' of a type (line 275)
                deviceInAdd_432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 16), h_431, 'deviceInAdd')
                # Calling deviceInAdd(args, kwargs) (line 275)
                deviceInAdd_call_result_435 = invoke(stypy.reporting.localization.Localization(__file__, 275, 16), deviceInAdd_432, *[pkt_433], **kwargs_434)
                
                # SSA join for if statement (line 272)
                module_type_store = module_type_store.join_ssa_context()
                


            if more_types_in_union_420:
                # SSA join for if statement (line 271)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Name (line 276):
        # Getting the type of 'h' (line 276)
        h_436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 15), 'h')
        # Obtaining the member 'work_in' of a type (line 276)
        work_in_437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 15), h_436, 'work_in')
        # Assigning a type to the variable 'work' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'work', work_in_437)
        
        # Type idiom detected: calculating its left and rigth part (line 277)
        # Getting the type of 'work' (line 277)
        work_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 11), 'work')
        # Getting the type of 'None' (line 277)
        None_439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 19), 'None')
        
        (may_be_440, more_types_in_union_441) = may_be_none(work_438, None_439)

        if may_be_440:

            if more_types_in_union_441:
                # Runtime conditional SSA (line 277)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to waitTask(...): (line 278)
            # Processing the call keyword arguments (line 278)
            kwargs_444 = {}
            # Getting the type of 'self' (line 278)
            self_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 19), 'self', False)
            # Obtaining the member 'waitTask' of a type (line 278)
            waitTask_443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 19), self_442, 'waitTask')
            # Calling waitTask(args, kwargs) (line 278)
            waitTask_call_result_445 = invoke(stypy.reporting.localization.Localization(__file__, 278, 19), waitTask_443, *[], **kwargs_444)
            
            # Assigning a type to the variable 'stypy_return_type' (line 278)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'stypy_return_type', waitTask_call_result_445)

            if more_types_in_union_441:
                # SSA join for if statement (line 277)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'work' (line 277)
        work_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'work')
        # Assigning a type to the variable 'work' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'work', remove_type_from_union(work_446, types.NoneType))
        
        # Assigning a Attribute to a Name (line 279):
        # Getting the type of 'work' (line 279)
        work_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 16), 'work')
        # Obtaining the member 'datum' of a type (line 279)
        datum_448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 16), work_447, 'datum')
        # Assigning a type to the variable 'count' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'count', datum_448)
        
        # Getting the type of 'count' (line 280)
        count_449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 11), 'count')
        # Getting the type of 'BUFSIZE' (line 280)
        BUFSIZE_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 20), 'BUFSIZE')
        # Applying the binary operator '>=' (line 280)
        result_ge_451 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 11), '>=', count_449, BUFSIZE_450)
        
        # Testing if the type of an if condition is none (line 280)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 280, 8), result_ge_451):
            pass
        else:
            
            # Testing the type of an if condition (line 280)
            if_condition_452 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 8), result_ge_451)
            # Assigning a type to the variable 'if_condition_452' (line 280)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'if_condition_452', if_condition_452)
            # SSA begins for if statement (line 280)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Attribute (line 281):
            # Getting the type of 'work' (line 281)
            work_453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 24), 'work')
            # Obtaining the member 'link' of a type (line 281)
            link_454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 24), work_453, 'link')
            # Getting the type of 'h' (line 281)
            h_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'h')
            # Setting the type of the member 'work_in' of a type (line 281)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 12), h_455, 'work_in', link_454)
            
            # Call to qpkt(...): (line 282)
            # Processing the call arguments (line 282)
            # Getting the type of 'work' (line 282)
            work_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 29), 'work', False)
            # Processing the call keyword arguments (line 282)
            kwargs_459 = {}
            # Getting the type of 'self' (line 282)
            self_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 19), 'self', False)
            # Obtaining the member 'qpkt' of a type (line 282)
            qpkt_457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 19), self_456, 'qpkt')
            # Calling qpkt(args, kwargs) (line 282)
            qpkt_call_result_460 = invoke(stypy.reporting.localization.Localization(__file__, 282, 19), qpkt_457, *[work_458], **kwargs_459)
            
            # Assigning a type to the variable 'stypy_return_type' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'stypy_return_type', qpkt_call_result_460)
            # SSA join for if statement (line 280)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Attribute to a Name (line 284):
        # Getting the type of 'h' (line 284)
        h_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 14), 'h')
        # Obtaining the member 'device_in' of a type (line 284)
        device_in_462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 14), h_461, 'device_in')
        # Assigning a type to the variable 'dev' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'dev', device_in_462)
        
        # Type idiom detected: calculating its left and rigth part (line 285)
        # Getting the type of 'dev' (line 285)
        dev_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 11), 'dev')
        # Getting the type of 'None' (line 285)
        None_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 18), 'None')
        
        (may_be_465, more_types_in_union_466) = may_be_none(dev_463, None_464)

        if may_be_465:

            if more_types_in_union_466:
                # Runtime conditional SSA (line 285)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to waitTask(...): (line 286)
            # Processing the call keyword arguments (line 286)
            kwargs_469 = {}
            # Getting the type of 'self' (line 286)
            self_467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 19), 'self', False)
            # Obtaining the member 'waitTask' of a type (line 286)
            waitTask_468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 19), self_467, 'waitTask')
            # Calling waitTask(args, kwargs) (line 286)
            waitTask_call_result_470 = invoke(stypy.reporting.localization.Localization(__file__, 286, 19), waitTask_468, *[], **kwargs_469)
            
            # Assigning a type to the variable 'stypy_return_type' (line 286)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'stypy_return_type', waitTask_call_result_470)

            if more_types_in_union_466:
                # SSA join for if statement (line 285)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'dev' (line 285)
        dev_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'dev')
        # Assigning a type to the variable 'dev' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'dev', remove_type_from_union(dev_471, types.NoneType))
        
        # Assigning a Attribute to a Attribute (line 288):
        # Getting the type of 'dev' (line 288)
        dev_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 22), 'dev')
        # Obtaining the member 'link' of a type (line 288)
        link_473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 22), dev_472, 'link')
        # Getting the type of 'h' (line 288)
        h_474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'h')
        # Setting the type of the member 'device_in' of a type (line 288)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 8), h_474, 'device_in', link_473)
        
        # Assigning a Subscript to a Attribute (line 289):
        
        # Obtaining the type of the subscript
        # Getting the type of 'count' (line 289)
        count_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 30), 'count')
        # Getting the type of 'work' (line 289)
        work_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 20), 'work')
        # Obtaining the member 'data' of a type (line 289)
        data_477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 20), work_476, 'data')
        # Obtaining the member '__getitem__' of a type (line 289)
        getitem___478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 20), data_477, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 289)
        subscript_call_result_479 = invoke(stypy.reporting.localization.Localization(__file__, 289, 20), getitem___478, count_475)
        
        # Getting the type of 'dev' (line 289)
        dev_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'dev')
        # Setting the type of the member 'datum' of a type (line 289)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 8), dev_480, 'datum', subscript_call_result_479)
        
        # Assigning a BinOp to a Attribute (line 290):
        # Getting the type of 'count' (line 290)
        count_481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 21), 'count')
        int_482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 29), 'int')
        # Applying the binary operator '+' (line 290)
        result_add_483 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 21), '+', count_481, int_482)
        
        # Getting the type of 'work' (line 290)
        work_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'work')
        # Setting the type of the member 'datum' of a type (line 290)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), work_484, 'datum', result_add_483)
        
        # Call to qpkt(...): (line 291)
        # Processing the call arguments (line 291)
        # Getting the type of 'dev' (line 291)
        dev_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 25), 'dev', False)
        # Processing the call keyword arguments (line 291)
        kwargs_488 = {}
        # Getting the type of 'self' (line 291)
        self_485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 15), 'self', False)
        # Obtaining the member 'qpkt' of a type (line 291)
        qpkt_486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 15), self_485, 'qpkt')
        # Calling qpkt(args, kwargs) (line 291)
        qpkt_call_result_489 = invoke(stypy.reporting.localization.Localization(__file__, 291, 15), qpkt_486, *[dev_487], **kwargs_488)
        
        # Assigning a type to the variable 'stypy_return_type' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'stypy_return_type', qpkt_call_result_489)
        
        # ################# End of 'fn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fn' in the type store
        # Getting the type of 'stypy_return_type' (line 268)
        stypy_return_type_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_490)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fn'
        return stypy_return_type_490


# Assigning a type to the variable 'HandlerTask' (line 264)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 0), 'HandlerTask', HandlerTask)
# Declaration of the 'IdleTask' class
# Getting the type of 'Task' (line 297)
Task_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 15), 'Task')

class IdleTask(Task_491, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 298, 4, False)
        # Assigning a type to the variable 'self' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IdleTask.__init__', ['i', 'p', 'w', 's', 'r'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['i', 'p', 'w', 's', 'r'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'self' (line 299)
        self_494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 22), 'self', False)
        # Getting the type of 'i' (line 299)
        i_495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 28), 'i', False)
        int_496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 31), 'int')
        # Getting the type of 'None' (line 299)
        None_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 34), 'None', False)
        # Getting the type of 's' (line 299)
        s_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 40), 's', False)
        # Getting the type of 'r' (line 299)
        r_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 43), 'r', False)
        # Processing the call keyword arguments (line 299)
        kwargs_500 = {}
        # Getting the type of 'Task' (line 299)
        Task_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'Task', False)
        # Obtaining the member '__init__' of a type (line 299)
        init___493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 8), Task_492, '__init__')
        # Calling __init__(args, kwargs) (line 299)
        init___call_result_501 = invoke(stypy.reporting.localization.Localization(__file__, 299, 8), init___493, *[self_494, i_495, int_496, None_497, s_498, r_499], **kwargs_500)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def fn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fn'
        module_type_store = module_type_store.open_function_context('fn', 301, 4, False)
        # Assigning a type to the variable 'self' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IdleTask.fn.__dict__.__setitem__('stypy_localization', localization)
        IdleTask.fn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IdleTask.fn.__dict__.__setitem__('stypy_type_store', module_type_store)
        IdleTask.fn.__dict__.__setitem__('stypy_function_name', 'IdleTask.fn')
        IdleTask.fn.__dict__.__setitem__('stypy_param_names_list', ['pkt', 'r'])
        IdleTask.fn.__dict__.__setitem__('stypy_varargs_param_name', None)
        IdleTask.fn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IdleTask.fn.__dict__.__setitem__('stypy_call_defaults', defaults)
        IdleTask.fn.__dict__.__setitem__('stypy_call_varargs', varargs)
        IdleTask.fn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IdleTask.fn.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IdleTask.fn', ['pkt', 'r'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fn', localization, ['pkt', 'r'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fn(...)' code ##################

        
        # Assigning a Name to a Name (line 302):
        # Getting the type of 'r' (line 302)
        r_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'r')
        # Assigning a type to the variable 'i' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'i', r_502)
        # Evaluating assert statement condition
        
        # Call to isinstance(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 'i' (line 303)
        i_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 26), 'i', False)
        # Getting the type of 'IdleTaskRec' (line 303)
        IdleTaskRec_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 29), 'IdleTaskRec', False)
        # Processing the call keyword arguments (line 303)
        kwargs_506 = {}
        # Getting the type of 'isinstance' (line 303)
        isinstance_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 303)
        isinstance_call_result_507 = invoke(stypy.reporting.localization.Localization(__file__, 303, 15), isinstance_503, *[i_504, IdleTaskRec_505], **kwargs_506)
        
        assert_508 = isinstance_call_result_507
        # Assigning a type to the variable 'assert_508' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'assert_508', isinstance_call_result_507)
        
        # Getting the type of 'i' (line 304)
        i_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'i')
        # Obtaining the member 'count' of a type (line 304)
        count_510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), i_509, 'count')
        int_511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 19), 'int')
        # Applying the binary operator '-=' (line 304)
        result_isub_512 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 8), '-=', count_510, int_511)
        # Getting the type of 'i' (line 304)
        i_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'i')
        # Setting the type of the member 'count' of a type (line 304)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), i_513, 'count', result_isub_512)
        
        
        # Getting the type of 'i' (line 305)
        i_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 11), 'i')
        # Obtaining the member 'count' of a type (line 305)
        count_515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 11), i_514, 'count')
        int_516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 22), 'int')
        # Applying the binary operator '==' (line 305)
        result_eq_517 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 11), '==', count_515, int_516)
        
        # Testing if the type of an if condition is none (line 305)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 305, 8), result_eq_517):
            
            # Getting the type of 'i' (line 307)
            i_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 13), 'i')
            # Obtaining the member 'control' of a type (line 307)
            control_524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 13), i_523, 'control')
            int_525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 25), 'int')
            # Applying the binary operator '&' (line 307)
            result_and__526 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 13), '&', control_524, int_525)
            
            int_527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 30), 'int')
            # Applying the binary operator '==' (line 307)
            result_eq_528 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 13), '==', result_and__526, int_527)
            
            # Testing if the type of an if condition is none (line 307)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 307, 13), result_eq_528):
                
                # Assigning a BinOp to a Attribute (line 311):
                # Getting the type of 'i' (line 311)
                i_540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 24), 'i')
                # Obtaining the member 'control' of a type (line 311)
                control_541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 24), i_540, 'control')
                int_542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 36), 'int')
                # Applying the binary operator 'div' (line 311)
                result_div_543 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 24), 'div', control_541, int_542)
                
                int_544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 40), 'int')
                # Applying the binary operator '^' (line 311)
                result_xor_545 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 24), '^', result_div_543, int_544)
                
                # Getting the type of 'i' (line 311)
                i_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'i')
                # Setting the type of the member 'control' of a type (line 311)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 12), i_546, 'control', result_xor_545)
                
                # Call to release(...): (line 312)
                # Processing the call arguments (line 312)
                # Getting the type of 'I_DEVB' (line 312)
                I_DEVB_549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 32), 'I_DEVB', False)
                # Processing the call keyword arguments (line 312)
                kwargs_550 = {}
                # Getting the type of 'self' (line 312)
                self_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 19), 'self', False)
                # Obtaining the member 'release' of a type (line 312)
                release_548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 19), self_547, 'release')
                # Calling release(args, kwargs) (line 312)
                release_call_result_551 = invoke(stypy.reporting.localization.Localization(__file__, 312, 19), release_548, *[I_DEVB_549], **kwargs_550)
                
                # Assigning a type to the variable 'stypy_return_type' (line 312)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'stypy_return_type', release_call_result_551)
            else:
                
                # Testing the type of an if condition (line 307)
                if_condition_529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 307, 13), result_eq_528)
                # Assigning a type to the variable 'if_condition_529' (line 307)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 13), 'if_condition_529', if_condition_529)
                # SSA begins for if statement (line 307)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'i' (line 308)
                i_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'i')
                # Obtaining the member 'control' of a type (line 308)
                control_531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 12), i_530, 'control')
                int_532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 25), 'int')
                # Applying the binary operator 'div=' (line 308)
                result_div_533 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 12), 'div=', control_531, int_532)
                # Getting the type of 'i' (line 308)
                i_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'i')
                # Setting the type of the member 'control' of a type (line 308)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 12), i_534, 'control', result_div_533)
                
                
                # Call to release(...): (line 309)
                # Processing the call arguments (line 309)
                # Getting the type of 'I_DEVA' (line 309)
                I_DEVA_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 32), 'I_DEVA', False)
                # Processing the call keyword arguments (line 309)
                kwargs_538 = {}
                # Getting the type of 'self' (line 309)
                self_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 19), 'self', False)
                # Obtaining the member 'release' of a type (line 309)
                release_536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 19), self_535, 'release')
                # Calling release(args, kwargs) (line 309)
                release_call_result_539 = invoke(stypy.reporting.localization.Localization(__file__, 309, 19), release_536, *[I_DEVA_537], **kwargs_538)
                
                # Assigning a type to the variable 'stypy_return_type' (line 309)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'stypy_return_type', release_call_result_539)
                # SSA branch for the else part of an if statement (line 307)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a BinOp to a Attribute (line 311):
                # Getting the type of 'i' (line 311)
                i_540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 24), 'i')
                # Obtaining the member 'control' of a type (line 311)
                control_541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 24), i_540, 'control')
                int_542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 36), 'int')
                # Applying the binary operator 'div' (line 311)
                result_div_543 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 24), 'div', control_541, int_542)
                
                int_544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 40), 'int')
                # Applying the binary operator '^' (line 311)
                result_xor_545 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 24), '^', result_div_543, int_544)
                
                # Getting the type of 'i' (line 311)
                i_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'i')
                # Setting the type of the member 'control' of a type (line 311)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 12), i_546, 'control', result_xor_545)
                
                # Call to release(...): (line 312)
                # Processing the call arguments (line 312)
                # Getting the type of 'I_DEVB' (line 312)
                I_DEVB_549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 32), 'I_DEVB', False)
                # Processing the call keyword arguments (line 312)
                kwargs_550 = {}
                # Getting the type of 'self' (line 312)
                self_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 19), 'self', False)
                # Obtaining the member 'release' of a type (line 312)
                release_548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 19), self_547, 'release')
                # Calling release(args, kwargs) (line 312)
                release_call_result_551 = invoke(stypy.reporting.localization.Localization(__file__, 312, 19), release_548, *[I_DEVB_549], **kwargs_550)
                
                # Assigning a type to the variable 'stypy_return_type' (line 312)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'stypy_return_type', release_call_result_551)
                # SSA join for if statement (line 307)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 305)
            if_condition_518 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 8), result_eq_517)
            # Assigning a type to the variable 'if_condition_518' (line 305)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'if_condition_518', if_condition_518)
            # SSA begins for if statement (line 305)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to hold(...): (line 306)
            # Processing the call keyword arguments (line 306)
            kwargs_521 = {}
            # Getting the type of 'self' (line 306)
            self_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 19), 'self', False)
            # Obtaining the member 'hold' of a type (line 306)
            hold_520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 19), self_519, 'hold')
            # Calling hold(args, kwargs) (line 306)
            hold_call_result_522 = invoke(stypy.reporting.localization.Localization(__file__, 306, 19), hold_520, *[], **kwargs_521)
            
            # Assigning a type to the variable 'stypy_return_type' (line 306)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'stypy_return_type', hold_call_result_522)
            # SSA branch for the else part of an if statement (line 305)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'i' (line 307)
            i_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 13), 'i')
            # Obtaining the member 'control' of a type (line 307)
            control_524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 13), i_523, 'control')
            int_525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 25), 'int')
            # Applying the binary operator '&' (line 307)
            result_and__526 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 13), '&', control_524, int_525)
            
            int_527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 30), 'int')
            # Applying the binary operator '==' (line 307)
            result_eq_528 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 13), '==', result_and__526, int_527)
            
            # Testing if the type of an if condition is none (line 307)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 307, 13), result_eq_528):
                
                # Assigning a BinOp to a Attribute (line 311):
                # Getting the type of 'i' (line 311)
                i_540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 24), 'i')
                # Obtaining the member 'control' of a type (line 311)
                control_541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 24), i_540, 'control')
                int_542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 36), 'int')
                # Applying the binary operator 'div' (line 311)
                result_div_543 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 24), 'div', control_541, int_542)
                
                int_544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 40), 'int')
                # Applying the binary operator '^' (line 311)
                result_xor_545 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 24), '^', result_div_543, int_544)
                
                # Getting the type of 'i' (line 311)
                i_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'i')
                # Setting the type of the member 'control' of a type (line 311)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 12), i_546, 'control', result_xor_545)
                
                # Call to release(...): (line 312)
                # Processing the call arguments (line 312)
                # Getting the type of 'I_DEVB' (line 312)
                I_DEVB_549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 32), 'I_DEVB', False)
                # Processing the call keyword arguments (line 312)
                kwargs_550 = {}
                # Getting the type of 'self' (line 312)
                self_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 19), 'self', False)
                # Obtaining the member 'release' of a type (line 312)
                release_548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 19), self_547, 'release')
                # Calling release(args, kwargs) (line 312)
                release_call_result_551 = invoke(stypy.reporting.localization.Localization(__file__, 312, 19), release_548, *[I_DEVB_549], **kwargs_550)
                
                # Assigning a type to the variable 'stypy_return_type' (line 312)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'stypy_return_type', release_call_result_551)
            else:
                
                # Testing the type of an if condition (line 307)
                if_condition_529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 307, 13), result_eq_528)
                # Assigning a type to the variable 'if_condition_529' (line 307)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 13), 'if_condition_529', if_condition_529)
                # SSA begins for if statement (line 307)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'i' (line 308)
                i_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'i')
                # Obtaining the member 'control' of a type (line 308)
                control_531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 12), i_530, 'control')
                int_532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 25), 'int')
                # Applying the binary operator 'div=' (line 308)
                result_div_533 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 12), 'div=', control_531, int_532)
                # Getting the type of 'i' (line 308)
                i_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'i')
                # Setting the type of the member 'control' of a type (line 308)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 12), i_534, 'control', result_div_533)
                
                
                # Call to release(...): (line 309)
                # Processing the call arguments (line 309)
                # Getting the type of 'I_DEVA' (line 309)
                I_DEVA_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 32), 'I_DEVA', False)
                # Processing the call keyword arguments (line 309)
                kwargs_538 = {}
                # Getting the type of 'self' (line 309)
                self_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 19), 'self', False)
                # Obtaining the member 'release' of a type (line 309)
                release_536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 19), self_535, 'release')
                # Calling release(args, kwargs) (line 309)
                release_call_result_539 = invoke(stypy.reporting.localization.Localization(__file__, 309, 19), release_536, *[I_DEVA_537], **kwargs_538)
                
                # Assigning a type to the variable 'stypy_return_type' (line 309)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'stypy_return_type', release_call_result_539)
                # SSA branch for the else part of an if statement (line 307)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a BinOp to a Attribute (line 311):
                # Getting the type of 'i' (line 311)
                i_540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 24), 'i')
                # Obtaining the member 'control' of a type (line 311)
                control_541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 24), i_540, 'control')
                int_542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 36), 'int')
                # Applying the binary operator 'div' (line 311)
                result_div_543 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 24), 'div', control_541, int_542)
                
                int_544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 40), 'int')
                # Applying the binary operator '^' (line 311)
                result_xor_545 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 24), '^', result_div_543, int_544)
                
                # Getting the type of 'i' (line 311)
                i_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'i')
                # Setting the type of the member 'control' of a type (line 311)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 12), i_546, 'control', result_xor_545)
                
                # Call to release(...): (line 312)
                # Processing the call arguments (line 312)
                # Getting the type of 'I_DEVB' (line 312)
                I_DEVB_549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 32), 'I_DEVB', False)
                # Processing the call keyword arguments (line 312)
                kwargs_550 = {}
                # Getting the type of 'self' (line 312)
                self_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 19), 'self', False)
                # Obtaining the member 'release' of a type (line 312)
                release_548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 19), self_547, 'release')
                # Calling release(args, kwargs) (line 312)
                release_call_result_551 = invoke(stypy.reporting.localization.Localization(__file__, 312, 19), release_548, *[I_DEVB_549], **kwargs_550)
                
                # Assigning a type to the variable 'stypy_return_type' (line 312)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'stypy_return_type', release_call_result_551)
                # SSA join for if statement (line 307)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 305)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'fn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fn' in the type store
        # Getting the type of 'stypy_return_type' (line 301)
        stypy_return_type_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_552)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fn'
        return stypy_return_type_552


# Assigning a type to the variable 'IdleTask' (line 297)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 0), 'IdleTask', IdleTask)

# Assigning a Call to a Name (line 318):

# Call to ord(...): (line 318)
# Processing the call arguments (line 318)
str_554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 8), 'str', 'A')
# Processing the call keyword arguments (line 318)
kwargs_555 = {}
# Getting the type of 'ord' (line 318)
ord_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'ord', False)
# Calling ord(args, kwargs) (line 318)
ord_call_result_556 = invoke(stypy.reporting.localization.Localization(__file__, 318, 4), ord_553, *[str_554], **kwargs_555)

# Assigning a type to the variable 'A' (line 318)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 0), 'A', ord_call_result_556)
# Declaration of the 'WorkTask' class
# Getting the type of 'Task' (line 321)
Task_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 15), 'Task')

class WorkTask(Task_557, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 322, 4, False)
        # Assigning a type to the variable 'self' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'WorkTask.__init__', ['i', 'p', 'w', 's', 'r'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['i', 'p', 'w', 's', 'r'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 323)
        # Processing the call arguments (line 323)
        # Getting the type of 'self' (line 323)
        self_560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 22), 'self', False)
        # Getting the type of 'i' (line 323)
        i_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 28), 'i', False)
        # Getting the type of 'p' (line 323)
        p_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 31), 'p', False)
        # Getting the type of 'w' (line 323)
        w_563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 34), 'w', False)
        # Getting the type of 's' (line 323)
        s_564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 37), 's', False)
        # Getting the type of 'r' (line 323)
        r_565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 40), 'r', False)
        # Processing the call keyword arguments (line 323)
        kwargs_566 = {}
        # Getting the type of 'Task' (line 323)
        Task_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'Task', False)
        # Obtaining the member '__init__' of a type (line 323)
        init___559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 8), Task_558, '__init__')
        # Calling __init__(args, kwargs) (line 323)
        init___call_result_567 = invoke(stypy.reporting.localization.Localization(__file__, 323, 8), init___559, *[self_560, i_561, p_562, w_563, s_564, r_565], **kwargs_566)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def fn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fn'
        module_type_store = module_type_store.open_function_context('fn', 325, 4, False)
        # Assigning a type to the variable 'self' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        WorkTask.fn.__dict__.__setitem__('stypy_localization', localization)
        WorkTask.fn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        WorkTask.fn.__dict__.__setitem__('stypy_type_store', module_type_store)
        WorkTask.fn.__dict__.__setitem__('stypy_function_name', 'WorkTask.fn')
        WorkTask.fn.__dict__.__setitem__('stypy_param_names_list', ['pkt', 'r'])
        WorkTask.fn.__dict__.__setitem__('stypy_varargs_param_name', None)
        WorkTask.fn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        WorkTask.fn.__dict__.__setitem__('stypy_call_defaults', defaults)
        WorkTask.fn.__dict__.__setitem__('stypy_call_varargs', varargs)
        WorkTask.fn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        WorkTask.fn.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'WorkTask.fn', ['pkt', 'r'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fn', localization, ['pkt', 'r'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fn(...)' code ##################

        
        # Assigning a Name to a Name (line 326):
        # Getting the type of 'r' (line 326)
        r_568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'r')
        # Assigning a type to the variable 'w' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'w', r_568)
        # Evaluating assert statement condition
        
        # Call to isinstance(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'w' (line 327)
        w_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 26), 'w', False)
        # Getting the type of 'WorkerTaskRec' (line 327)
        WorkerTaskRec_571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 29), 'WorkerTaskRec', False)
        # Processing the call keyword arguments (line 327)
        kwargs_572 = {}
        # Getting the type of 'isinstance' (line 327)
        isinstance_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 327)
        isinstance_call_result_573 = invoke(stypy.reporting.localization.Localization(__file__, 327, 15), isinstance_569, *[w_570, WorkerTaskRec_571], **kwargs_572)
        
        assert_574 = isinstance_call_result_573
        # Assigning a type to the variable 'assert_574' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'assert_574', isinstance_call_result_573)
        
        # Type idiom detected: calculating its left and rigth part (line 328)
        # Getting the type of 'pkt' (line 328)
        pkt_575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 11), 'pkt')
        # Getting the type of 'None' (line 328)
        None_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 18), 'None')
        
        (may_be_577, more_types_in_union_578) = may_be_none(pkt_575, None_576)

        if may_be_577:

            if more_types_in_union_578:
                # Runtime conditional SSA (line 328)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to waitTask(...): (line 329)
            # Processing the call keyword arguments (line 329)
            kwargs_581 = {}
            # Getting the type of 'self' (line 329)
            self_579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 19), 'self', False)
            # Obtaining the member 'waitTask' of a type (line 329)
            waitTask_580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 19), self_579, 'waitTask')
            # Calling waitTask(args, kwargs) (line 329)
            waitTask_call_result_582 = invoke(stypy.reporting.localization.Localization(__file__, 329, 19), waitTask_580, *[], **kwargs_581)
            
            # Assigning a type to the variable 'stypy_return_type' (line 329)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'stypy_return_type', waitTask_call_result_582)

            if more_types_in_union_578:
                # SSA join for if statement (line 328)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'pkt' (line 328)
        pkt_583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'pkt')
        # Assigning a type to the variable 'pkt' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'pkt', remove_type_from_union(pkt_583, types.NoneType))
        
        # Getting the type of 'w' (line 331)
        w_584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 11), 'w')
        # Obtaining the member 'destination' of a type (line 331)
        destination_585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 11), w_584, 'destination')
        # Getting the type of 'I_HANDLERA' (line 331)
        I_HANDLERA_586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 28), 'I_HANDLERA')
        # Applying the binary operator '==' (line 331)
        result_eq_587 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 11), '==', destination_585, I_HANDLERA_586)
        
        # Testing if the type of an if condition is none (line 331)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 331, 8), result_eq_587):
            
            # Assigning a Name to a Name (line 334):
            # Getting the type of 'I_HANDLERA' (line 334)
            I_HANDLERA_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 19), 'I_HANDLERA')
            # Assigning a type to the variable 'dest' (line 334)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'dest', I_HANDLERA_590)
        else:
            
            # Testing the type of an if condition (line 331)
            if_condition_588 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 331, 8), result_eq_587)
            # Assigning a type to the variable 'if_condition_588' (line 331)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'if_condition_588', if_condition_588)
            # SSA begins for if statement (line 331)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 332):
            # Getting the type of 'I_HANDLERB' (line 332)
            I_HANDLERB_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 19), 'I_HANDLERB')
            # Assigning a type to the variable 'dest' (line 332)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'dest', I_HANDLERB_589)
            # SSA branch for the else part of an if statement (line 331)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Name (line 334):
            # Getting the type of 'I_HANDLERA' (line 334)
            I_HANDLERA_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 19), 'I_HANDLERA')
            # Assigning a type to the variable 'dest' (line 334)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'dest', I_HANDLERA_590)
            # SSA join for if statement (line 331)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Name to a Attribute (line 336):
        # Getting the type of 'dest' (line 336)
        dest_591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 24), 'dest')
        # Getting the type of 'w' (line 336)
        w_592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'w')
        # Setting the type of the member 'destination' of a type (line 336)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 8), w_592, 'destination', dest_591)
        
        # Assigning a Name to a Attribute (line 337):
        # Getting the type of 'dest' (line 337)
        dest_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 20), 'dest')
        # Getting the type of 'pkt' (line 337)
        pkt_594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'pkt')
        # Setting the type of the member 'ident' of a type (line 337)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 8), pkt_594, 'ident', dest_593)
        
        # Assigning a Num to a Attribute (line 338):
        int_595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 20), 'int')
        # Getting the type of 'pkt' (line 338)
        pkt_596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'pkt')
        # Setting the type of the member 'datum' of a type (line 338)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 8), pkt_596, 'datum', int_595)
        
        # Getting the type of 'BUFSIZE_RANGE' (line 340)
        BUFSIZE_RANGE_597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 17), 'BUFSIZE_RANGE')
        # Assigning a type to the variable 'BUFSIZE_RANGE_597' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'BUFSIZE_RANGE_597', BUFSIZE_RANGE_597)
        # Testing if the for loop is going to be iterated (line 340)
        # Testing the type of a for loop iterable (line 340)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 340, 8), BUFSIZE_RANGE_597)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 340, 8), BUFSIZE_RANGE_597):
            # Getting the type of the for loop variable (line 340)
            for_loop_var_598 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 340, 8), BUFSIZE_RANGE_597)
            # Assigning a type to the variable 'i' (line 340)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'i', for_loop_var_598)
            # SSA begins for a for statement (line 340)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'w' (line 341)
            w_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'w')
            # Obtaining the member 'count' of a type (line 341)
            count_600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 12), w_599, 'count')
            int_601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 23), 'int')
            # Applying the binary operator '+=' (line 341)
            result_iadd_602 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 12), '+=', count_600, int_601)
            # Getting the type of 'w' (line 341)
            w_603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'w')
            # Setting the type of the member 'count' of a type (line 341)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 12), w_603, 'count', result_iadd_602)
            
            
            # Getting the type of 'w' (line 342)
            w_604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 15), 'w')
            # Obtaining the member 'count' of a type (line 342)
            count_605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 15), w_604, 'count')
            int_606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 25), 'int')
            # Applying the binary operator '>' (line 342)
            result_gt_607 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 15), '>', count_605, int_606)
            
            # Testing if the type of an if condition is none (line 342)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 342, 12), result_gt_607):
                pass
            else:
                
                # Testing the type of an if condition (line 342)
                if_condition_608 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 342, 12), result_gt_607)
                # Assigning a type to the variable 'if_condition_608' (line 342)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'if_condition_608', if_condition_608)
                # SSA begins for if statement (line 342)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Num to a Attribute (line 343):
                int_609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 26), 'int')
                # Getting the type of 'w' (line 343)
                w_610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 16), 'w')
                # Setting the type of the member 'count' of a type (line 343)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 16), w_610, 'count', int_609)
                # SSA join for if statement (line 342)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a BinOp to a Subscript (line 344):
            # Getting the type of 'A' (line 344)
            A_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 26), 'A')
            # Getting the type of 'w' (line 344)
            w_612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 30), 'w')
            # Obtaining the member 'count' of a type (line 344)
            count_613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 30), w_612, 'count')
            # Applying the binary operator '+' (line 344)
            result_add_614 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 26), '+', A_611, count_613)
            
            int_615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 40), 'int')
            # Applying the binary operator '-' (line 344)
            result_sub_616 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 38), '-', result_add_614, int_615)
            
            # Getting the type of 'pkt' (line 344)
            pkt_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'pkt')
            # Obtaining the member 'data' of a type (line 344)
            data_618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 12), pkt_617, 'data')
            # Getting the type of 'i' (line 344)
            i_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 21), 'i')
            # Storing an element on a container (line 344)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 12), data_618, (i_619, result_sub_616))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to qpkt(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'pkt' (line 346)
        pkt_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 25), 'pkt', False)
        # Processing the call keyword arguments (line 346)
        kwargs_623 = {}
        # Getting the type of 'self' (line 346)
        self_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 15), 'self', False)
        # Obtaining the member 'qpkt' of a type (line 346)
        qpkt_621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 15), self_620, 'qpkt')
        # Calling qpkt(args, kwargs) (line 346)
        qpkt_call_result_624 = invoke(stypy.reporting.localization.Localization(__file__, 346, 15), qpkt_621, *[pkt_622], **kwargs_623)
        
        # Assigning a type to the variable 'stypy_return_type' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'stypy_return_type', qpkt_call_result_624)
        
        # ################# End of 'fn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fn' in the type store
        # Getting the type of 'stypy_return_type' (line 325)
        stypy_return_type_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_625)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fn'
        return stypy_return_type_625


# Assigning a type to the variable 'WorkTask' (line 321)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 0), 'WorkTask', WorkTask)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 349, 0))

# 'import time' statement (line 349)
import time

import_module(stypy.reporting.localization.Localization(__file__, 349, 0), 'time', time, module_type_store)


@norecursion
def schedule(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'schedule'
    module_type_store = module_type_store.open_function_context('schedule', 352, 0, False)
    
    # Passed parameters checking function
    schedule.stypy_localization = localization
    schedule.stypy_type_of_self = None
    schedule.stypy_type_store = module_type_store
    schedule.stypy_function_name = 'schedule'
    schedule.stypy_param_names_list = []
    schedule.stypy_varargs_param_name = None
    schedule.stypy_kwargs_param_name = None
    schedule.stypy_call_defaults = defaults
    schedule.stypy_call_varargs = varargs
    schedule.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'schedule', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'schedule', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'schedule(...)' code ##################

    
    # Assigning a Attribute to a Name (line 353):
    # Getting the type of 'taskWorkArea' (line 353)
    taskWorkArea_626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'taskWorkArea')
    # Obtaining the member 'taskList' of a type (line 353)
    taskList_627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 8), taskWorkArea_626, 'taskList')
    # Assigning a type to the variable 't' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 't', taskList_627)
    
    
    # Getting the type of 't' (line 354)
    t_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 10), 't')
    # Getting the type of 'None' (line 354)
    None_629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 19), 'None')
    # Applying the binary operator 'isnot' (line 354)
    result_is_not_630 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 10), 'isnot', t_628, None_629)
    
    # Assigning a type to the variable 'result_is_not_630' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'result_is_not_630', result_is_not_630)
    # Testing if the while is going to be iterated (line 354)
    # Testing the type of an if condition (line 354)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 354, 4), result_is_not_630)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 354, 4), result_is_not_630):
        # SSA begins for while statement (line 354)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Name to a Name (line 355):
        # Getting the type of 'None' (line 355)
        None_631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 14), 'None')
        # Assigning a type to the variable 'pkt' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'pkt', None_631)
        # Getting the type of 'tracing' (line 357)
        tracing_632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 11), 'tracing')
        # Testing if the type of an if condition is none (line 357)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 357, 8), tracing_632):
            pass
        else:
            
            # Testing the type of an if condition (line 357)
            if_condition_633 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 357, 8), tracing_632)
            # Assigning a type to the variable 'if_condition_633' (line 357)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'if_condition_633', if_condition_633)
            # SSA begins for if statement (line 357)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA join for if statement (line 357)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to isTaskHoldingOrWaiting(...): (line 362)
        # Processing the call keyword arguments (line 362)
        kwargs_636 = {}
        # Getting the type of 't' (line 362)
        t_634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 11), 't', False)
        # Obtaining the member 'isTaskHoldingOrWaiting' of a type (line 362)
        isTaskHoldingOrWaiting_635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 11), t_634, 'isTaskHoldingOrWaiting')
        # Calling isTaskHoldingOrWaiting(args, kwargs) (line 362)
        isTaskHoldingOrWaiting_call_result_637 = invoke(stypy.reporting.localization.Localization(__file__, 362, 11), isTaskHoldingOrWaiting_635, *[], **kwargs_636)
        
        # Testing if the type of an if condition is none (line 362)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 362, 8), isTaskHoldingOrWaiting_call_result_637):
            # Getting the type of 'tracing' (line 365)
            tracing_641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 15), 'tracing')
            # Testing if the type of an if condition is none (line 365)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 365, 12), tracing_641):
                pass
            else:
                
                # Testing the type of an if condition (line 365)
                if_condition_642 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 365, 12), tracing_641)
                # Assigning a type to the variable 'if_condition_642' (line 365)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'if_condition_642', if_condition_642)
                # SSA begins for if statement (line 365)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to trace(...): (line 365)
                # Processing the call arguments (line 365)
                
                # Call to chr(...): (line 365)
                # Processing the call arguments (line 365)
                
                # Call to ord(...): (line 365)
                # Processing the call arguments (line 365)
                str_646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 38), 'str', '0')
                # Processing the call keyword arguments (line 365)
                kwargs_647 = {}
                # Getting the type of 'ord' (line 365)
                ord_645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 34), 'ord', False)
                # Calling ord(args, kwargs) (line 365)
                ord_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 365, 34), ord_645, *[str_646], **kwargs_647)
                
                # Getting the type of 't' (line 365)
                t_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 45), 't', False)
                # Obtaining the member 'ident' of a type (line 365)
                ident_650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 45), t_649, 'ident')
                # Applying the binary operator '+' (line 365)
                result_add_651 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 34), '+', ord_call_result_648, ident_650)
                
                # Processing the call keyword arguments (line 365)
                kwargs_652 = {}
                # Getting the type of 'chr' (line 365)
                chr_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 30), 'chr', False)
                # Calling chr(args, kwargs) (line 365)
                chr_call_result_653 = invoke(stypy.reporting.localization.Localization(__file__, 365, 30), chr_644, *[result_add_651], **kwargs_652)
                
                # Processing the call keyword arguments (line 365)
                kwargs_654 = {}
                # Getting the type of 'trace' (line 365)
                trace_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 24), 'trace', False)
                # Calling trace(args, kwargs) (line 365)
                trace_call_result_655 = invoke(stypy.reporting.localization.Localization(__file__, 365, 24), trace_643, *[chr_call_result_653], **kwargs_654)
                
                # SSA join for if statement (line 365)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Call to a Name (line 366):
            
            # Call to runTask(...): (line 366)
            # Processing the call keyword arguments (line 366)
            kwargs_658 = {}
            # Getting the type of 't' (line 366)
            t_656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 16), 't', False)
            # Obtaining the member 'runTask' of a type (line 366)
            runTask_657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 16), t_656, 'runTask')
            # Calling runTask(args, kwargs) (line 366)
            runTask_call_result_659 = invoke(stypy.reporting.localization.Localization(__file__, 366, 16), runTask_657, *[], **kwargs_658)
            
            # Assigning a type to the variable 't' (line 366)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 't', runTask_call_result_659)
        else:
            
            # Testing the type of an if condition (line 362)
            if_condition_638 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 362, 8), isTaskHoldingOrWaiting_call_result_637)
            # Assigning a type to the variable 'if_condition_638' (line 362)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'if_condition_638', if_condition_638)
            # SSA begins for if statement (line 362)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 363):
            # Getting the type of 't' (line 363)
            t_639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 16), 't')
            # Obtaining the member 'link' of a type (line 363)
            link_640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 16), t_639, 'link')
            # Assigning a type to the variable 't' (line 363)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 't', link_640)
            # SSA branch for the else part of an if statement (line 362)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'tracing' (line 365)
            tracing_641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 15), 'tracing')
            # Testing if the type of an if condition is none (line 365)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 365, 12), tracing_641):
                pass
            else:
                
                # Testing the type of an if condition (line 365)
                if_condition_642 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 365, 12), tracing_641)
                # Assigning a type to the variable 'if_condition_642' (line 365)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'if_condition_642', if_condition_642)
                # SSA begins for if statement (line 365)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to trace(...): (line 365)
                # Processing the call arguments (line 365)
                
                # Call to chr(...): (line 365)
                # Processing the call arguments (line 365)
                
                # Call to ord(...): (line 365)
                # Processing the call arguments (line 365)
                str_646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 38), 'str', '0')
                # Processing the call keyword arguments (line 365)
                kwargs_647 = {}
                # Getting the type of 'ord' (line 365)
                ord_645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 34), 'ord', False)
                # Calling ord(args, kwargs) (line 365)
                ord_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 365, 34), ord_645, *[str_646], **kwargs_647)
                
                # Getting the type of 't' (line 365)
                t_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 45), 't', False)
                # Obtaining the member 'ident' of a type (line 365)
                ident_650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 45), t_649, 'ident')
                # Applying the binary operator '+' (line 365)
                result_add_651 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 34), '+', ord_call_result_648, ident_650)
                
                # Processing the call keyword arguments (line 365)
                kwargs_652 = {}
                # Getting the type of 'chr' (line 365)
                chr_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 30), 'chr', False)
                # Calling chr(args, kwargs) (line 365)
                chr_call_result_653 = invoke(stypy.reporting.localization.Localization(__file__, 365, 30), chr_644, *[result_add_651], **kwargs_652)
                
                # Processing the call keyword arguments (line 365)
                kwargs_654 = {}
                # Getting the type of 'trace' (line 365)
                trace_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 24), 'trace', False)
                # Calling trace(args, kwargs) (line 365)
                trace_call_result_655 = invoke(stypy.reporting.localization.Localization(__file__, 365, 24), trace_643, *[chr_call_result_653], **kwargs_654)
                
                # SSA join for if statement (line 365)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Call to a Name (line 366):
            
            # Call to runTask(...): (line 366)
            # Processing the call keyword arguments (line 366)
            kwargs_658 = {}
            # Getting the type of 't' (line 366)
            t_656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 16), 't', False)
            # Obtaining the member 'runTask' of a type (line 366)
            runTask_657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 16), t_656, 'runTask')
            # Calling runTask(args, kwargs) (line 366)
            runTask_call_result_659 = invoke(stypy.reporting.localization.Localization(__file__, 366, 16), runTask_657, *[], **kwargs_658)
            
            # Assigning a type to the variable 't' (line 366)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 't', runTask_call_result_659)
            # SSA join for if statement (line 362)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for while statement (line 354)
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'schedule(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'schedule' in the type store
    # Getting the type of 'stypy_return_type' (line 352)
    stypy_return_type_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_660)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'schedule'
    return stypy_return_type_660

# Assigning a type to the variable 'schedule' (line 352)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 0), 'schedule', schedule)
# Declaration of the 'Richards' class

class Richards(object, ):

    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 371, 4, False)
        # Assigning a type to the variable 'self' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Richards.run.__dict__.__setitem__('stypy_localization', localization)
        Richards.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Richards.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        Richards.run.__dict__.__setitem__('stypy_function_name', 'Richards.run')
        Richards.run.__dict__.__setitem__('stypy_param_names_list', ['iterations'])
        Richards.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        Richards.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Richards.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        Richards.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        Richards.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Richards.run.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Richards.run', ['iterations'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run', localization, ['iterations'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run(...)' code ##################

        
        
        # Call to xrange(...): (line 372)
        # Processing the call arguments (line 372)
        # Getting the type of 'iterations' (line 372)
        iterations_662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 24), 'iterations', False)
        # Processing the call keyword arguments (line 372)
        kwargs_663 = {}
        # Getting the type of 'xrange' (line 372)
        xrange_661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 372)
        xrange_call_result_664 = invoke(stypy.reporting.localization.Localization(__file__, 372, 17), xrange_661, *[iterations_662], **kwargs_663)
        
        # Assigning a type to the variable 'xrange_call_result_664' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'xrange_call_result_664', xrange_call_result_664)
        # Testing if the for loop is going to be iterated (line 372)
        # Testing the type of a for loop iterable (line 372)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 372, 8), xrange_call_result_664)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 372, 8), xrange_call_result_664):
            # Getting the type of the for loop variable (line 372)
            for_loop_var_665 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 372, 8), xrange_call_result_664)
            # Assigning a type to the variable 'i' (line 372)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'i', for_loop_var_665)
            # SSA begins for a for statement (line 372)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Num to a Attribute (line 373):
            int_666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 37), 'int')
            # Getting the type of 'taskWorkArea' (line 373)
            taskWorkArea_667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'taskWorkArea')
            # Setting the type of the member 'holdCount' of a type (line 373)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 12), taskWorkArea_667, 'holdCount', int_666)
            
            # Assigning a Num to a Attribute (line 374):
            int_668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 37), 'int')
            # Getting the type of 'taskWorkArea' (line 374)
            taskWorkArea_669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'taskWorkArea')
            # Setting the type of the member 'qpktCount' of a type (line 374)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 12), taskWorkArea_669, 'qpktCount', int_668)
            
            # Call to IdleTask(...): (line 376)
            # Processing the call arguments (line 376)
            # Getting the type of 'I_IDLE' (line 376)
            I_IDLE_671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 21), 'I_IDLE', False)
            int_672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 29), 'int')
            int_673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 32), 'int')
            
            # Call to running(...): (line 376)
            # Processing the call keyword arguments (line 376)
            kwargs_678 = {}
            
            # Call to TaskState(...): (line 376)
            # Processing the call keyword arguments (line 376)
            kwargs_675 = {}
            # Getting the type of 'TaskState' (line 376)
            TaskState_674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 39), 'TaskState', False)
            # Calling TaskState(args, kwargs) (line 376)
            TaskState_call_result_676 = invoke(stypy.reporting.localization.Localization(__file__, 376, 39), TaskState_674, *[], **kwargs_675)
            
            # Obtaining the member 'running' of a type (line 376)
            running_677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 39), TaskState_call_result_676, 'running')
            # Calling running(args, kwargs) (line 376)
            running_call_result_679 = invoke(stypy.reporting.localization.Localization(__file__, 376, 39), running_677, *[], **kwargs_678)
            
            
            # Call to IdleTaskRec(...): (line 376)
            # Processing the call keyword arguments (line 376)
            kwargs_681 = {}
            # Getting the type of 'IdleTaskRec' (line 376)
            IdleTaskRec_680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 62), 'IdleTaskRec', False)
            # Calling IdleTaskRec(args, kwargs) (line 376)
            IdleTaskRec_call_result_682 = invoke(stypy.reporting.localization.Localization(__file__, 376, 62), IdleTaskRec_680, *[], **kwargs_681)
            
            # Processing the call keyword arguments (line 376)
            kwargs_683 = {}
            # Getting the type of 'IdleTask' (line 376)
            IdleTask_670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'IdleTask', False)
            # Calling IdleTask(args, kwargs) (line 376)
            IdleTask_call_result_684 = invoke(stypy.reporting.localization.Localization(__file__, 376, 12), IdleTask_670, *[I_IDLE_671, int_672, int_673, running_call_result_679, IdleTaskRec_call_result_682], **kwargs_683)
            
            
            # Assigning a Call to a Name (line 378):
            
            # Call to Packet(...): (line 378)
            # Processing the call arguments (line 378)
            # Getting the type of 'None' (line 378)
            None_686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 25), 'None', False)
            int_687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 31), 'int')
            # Getting the type of 'K_WORK' (line 378)
            K_WORK_688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 34), 'K_WORK', False)
            # Processing the call keyword arguments (line 378)
            kwargs_689 = {}
            # Getting the type of 'Packet' (line 378)
            Packet_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 18), 'Packet', False)
            # Calling Packet(args, kwargs) (line 378)
            Packet_call_result_690 = invoke(stypy.reporting.localization.Localization(__file__, 378, 18), Packet_685, *[None_686, int_687, K_WORK_688], **kwargs_689)
            
            # Assigning a type to the variable 'wkq' (line 378)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 12), 'wkq', Packet_call_result_690)
            
            # Assigning a Call to a Name (line 379):
            
            # Call to Packet(...): (line 379)
            # Processing the call arguments (line 379)
            # Getting the type of 'wkq' (line 379)
            wkq_692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 25), 'wkq', False)
            int_693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 30), 'int')
            # Getting the type of 'K_WORK' (line 379)
            K_WORK_694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 33), 'K_WORK', False)
            # Processing the call keyword arguments (line 379)
            kwargs_695 = {}
            # Getting the type of 'Packet' (line 379)
            Packet_691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 18), 'Packet', False)
            # Calling Packet(args, kwargs) (line 379)
            Packet_call_result_696 = invoke(stypy.reporting.localization.Localization(__file__, 379, 18), Packet_691, *[wkq_692, int_693, K_WORK_694], **kwargs_695)
            
            # Assigning a type to the variable 'wkq' (line 379)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 12), 'wkq', Packet_call_result_696)
            
            # Call to WorkTask(...): (line 380)
            # Processing the call arguments (line 380)
            # Getting the type of 'I_WORK' (line 380)
            I_WORK_698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 21), 'I_WORK', False)
            int_699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 29), 'int')
            # Getting the type of 'wkq' (line 380)
            wkq_700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 35), 'wkq', False)
            
            # Call to waitingWithPacket(...): (line 380)
            # Processing the call keyword arguments (line 380)
            kwargs_705 = {}
            
            # Call to TaskState(...): (line 380)
            # Processing the call keyword arguments (line 380)
            kwargs_702 = {}
            # Getting the type of 'TaskState' (line 380)
            TaskState_701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 40), 'TaskState', False)
            # Calling TaskState(args, kwargs) (line 380)
            TaskState_call_result_703 = invoke(stypy.reporting.localization.Localization(__file__, 380, 40), TaskState_701, *[], **kwargs_702)
            
            # Obtaining the member 'waitingWithPacket' of a type (line 380)
            waitingWithPacket_704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 40), TaskState_call_result_703, 'waitingWithPacket')
            # Calling waitingWithPacket(args, kwargs) (line 380)
            waitingWithPacket_call_result_706 = invoke(stypy.reporting.localization.Localization(__file__, 380, 40), waitingWithPacket_704, *[], **kwargs_705)
            
            
            # Call to WorkerTaskRec(...): (line 380)
            # Processing the call keyword arguments (line 380)
            kwargs_708 = {}
            # Getting the type of 'WorkerTaskRec' (line 380)
            WorkerTaskRec_707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 73), 'WorkerTaskRec', False)
            # Calling WorkerTaskRec(args, kwargs) (line 380)
            WorkerTaskRec_call_result_709 = invoke(stypy.reporting.localization.Localization(__file__, 380, 73), WorkerTaskRec_707, *[], **kwargs_708)
            
            # Processing the call keyword arguments (line 380)
            kwargs_710 = {}
            # Getting the type of 'WorkTask' (line 380)
            WorkTask_697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'WorkTask', False)
            # Calling WorkTask(args, kwargs) (line 380)
            WorkTask_call_result_711 = invoke(stypy.reporting.localization.Localization(__file__, 380, 12), WorkTask_697, *[I_WORK_698, int_699, wkq_700, waitingWithPacket_call_result_706, WorkerTaskRec_call_result_709], **kwargs_710)
            
            
            # Assigning a Call to a Name (line 382):
            
            # Call to Packet(...): (line 382)
            # Processing the call arguments (line 382)
            # Getting the type of 'None' (line 382)
            None_713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 25), 'None', False)
            # Getting the type of 'I_DEVA' (line 382)
            I_DEVA_714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 31), 'I_DEVA', False)
            # Getting the type of 'K_DEV' (line 382)
            K_DEV_715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 39), 'K_DEV', False)
            # Processing the call keyword arguments (line 382)
            kwargs_716 = {}
            # Getting the type of 'Packet' (line 382)
            Packet_712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 18), 'Packet', False)
            # Calling Packet(args, kwargs) (line 382)
            Packet_call_result_717 = invoke(stypy.reporting.localization.Localization(__file__, 382, 18), Packet_712, *[None_713, I_DEVA_714, K_DEV_715], **kwargs_716)
            
            # Assigning a type to the variable 'wkq' (line 382)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'wkq', Packet_call_result_717)
            
            # Assigning a Call to a Name (line 383):
            
            # Call to Packet(...): (line 383)
            # Processing the call arguments (line 383)
            # Getting the type of 'wkq' (line 383)
            wkq_719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 25), 'wkq', False)
            # Getting the type of 'I_DEVA' (line 383)
            I_DEVA_720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 30), 'I_DEVA', False)
            # Getting the type of 'K_DEV' (line 383)
            K_DEV_721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 38), 'K_DEV', False)
            # Processing the call keyword arguments (line 383)
            kwargs_722 = {}
            # Getting the type of 'Packet' (line 383)
            Packet_718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 18), 'Packet', False)
            # Calling Packet(args, kwargs) (line 383)
            Packet_call_result_723 = invoke(stypy.reporting.localization.Localization(__file__, 383, 18), Packet_718, *[wkq_719, I_DEVA_720, K_DEV_721], **kwargs_722)
            
            # Assigning a type to the variable 'wkq' (line 383)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'wkq', Packet_call_result_723)
            
            # Assigning a Call to a Name (line 384):
            
            # Call to Packet(...): (line 384)
            # Processing the call arguments (line 384)
            # Getting the type of 'wkq' (line 384)
            wkq_725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 25), 'wkq', False)
            # Getting the type of 'I_DEVA' (line 384)
            I_DEVA_726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 30), 'I_DEVA', False)
            # Getting the type of 'K_DEV' (line 384)
            K_DEV_727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 38), 'K_DEV', False)
            # Processing the call keyword arguments (line 384)
            kwargs_728 = {}
            # Getting the type of 'Packet' (line 384)
            Packet_724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 18), 'Packet', False)
            # Calling Packet(args, kwargs) (line 384)
            Packet_call_result_729 = invoke(stypy.reporting.localization.Localization(__file__, 384, 18), Packet_724, *[wkq_725, I_DEVA_726, K_DEV_727], **kwargs_728)
            
            # Assigning a type to the variable 'wkq' (line 384)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'wkq', Packet_call_result_729)
            
            # Call to HandlerTask(...): (line 385)
            # Processing the call arguments (line 385)
            # Getting the type of 'I_HANDLERA' (line 385)
            I_HANDLERA_731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 24), 'I_HANDLERA', False)
            int_732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 36), 'int')
            # Getting the type of 'wkq' (line 385)
            wkq_733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 42), 'wkq', False)
            
            # Call to waitingWithPacket(...): (line 385)
            # Processing the call keyword arguments (line 385)
            kwargs_738 = {}
            
            # Call to TaskState(...): (line 385)
            # Processing the call keyword arguments (line 385)
            kwargs_735 = {}
            # Getting the type of 'TaskState' (line 385)
            TaskState_734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 47), 'TaskState', False)
            # Calling TaskState(args, kwargs) (line 385)
            TaskState_call_result_736 = invoke(stypy.reporting.localization.Localization(__file__, 385, 47), TaskState_734, *[], **kwargs_735)
            
            # Obtaining the member 'waitingWithPacket' of a type (line 385)
            waitingWithPacket_737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 47), TaskState_call_result_736, 'waitingWithPacket')
            # Calling waitingWithPacket(args, kwargs) (line 385)
            waitingWithPacket_call_result_739 = invoke(stypy.reporting.localization.Localization(__file__, 385, 47), waitingWithPacket_737, *[], **kwargs_738)
            
            
            # Call to HandlerTaskRec(...): (line 385)
            # Processing the call keyword arguments (line 385)
            kwargs_741 = {}
            # Getting the type of 'HandlerTaskRec' (line 385)
            HandlerTaskRec_740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 80), 'HandlerTaskRec', False)
            # Calling HandlerTaskRec(args, kwargs) (line 385)
            HandlerTaskRec_call_result_742 = invoke(stypy.reporting.localization.Localization(__file__, 385, 80), HandlerTaskRec_740, *[], **kwargs_741)
            
            # Processing the call keyword arguments (line 385)
            kwargs_743 = {}
            # Getting the type of 'HandlerTask' (line 385)
            HandlerTask_730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'HandlerTask', False)
            # Calling HandlerTask(args, kwargs) (line 385)
            HandlerTask_call_result_744 = invoke(stypy.reporting.localization.Localization(__file__, 385, 12), HandlerTask_730, *[I_HANDLERA_731, int_732, wkq_733, waitingWithPacket_call_result_739, HandlerTaskRec_call_result_742], **kwargs_743)
            
            
            # Assigning a Call to a Name (line 387):
            
            # Call to Packet(...): (line 387)
            # Processing the call arguments (line 387)
            # Getting the type of 'None' (line 387)
            None_746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 25), 'None', False)
            # Getting the type of 'I_DEVB' (line 387)
            I_DEVB_747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 31), 'I_DEVB', False)
            # Getting the type of 'K_DEV' (line 387)
            K_DEV_748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 39), 'K_DEV', False)
            # Processing the call keyword arguments (line 387)
            kwargs_749 = {}
            # Getting the type of 'Packet' (line 387)
            Packet_745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 18), 'Packet', False)
            # Calling Packet(args, kwargs) (line 387)
            Packet_call_result_750 = invoke(stypy.reporting.localization.Localization(__file__, 387, 18), Packet_745, *[None_746, I_DEVB_747, K_DEV_748], **kwargs_749)
            
            # Assigning a type to the variable 'wkq' (line 387)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'wkq', Packet_call_result_750)
            
            # Assigning a Call to a Name (line 388):
            
            # Call to Packet(...): (line 388)
            # Processing the call arguments (line 388)
            # Getting the type of 'wkq' (line 388)
            wkq_752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 25), 'wkq', False)
            # Getting the type of 'I_DEVB' (line 388)
            I_DEVB_753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 30), 'I_DEVB', False)
            # Getting the type of 'K_DEV' (line 388)
            K_DEV_754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 38), 'K_DEV', False)
            # Processing the call keyword arguments (line 388)
            kwargs_755 = {}
            # Getting the type of 'Packet' (line 388)
            Packet_751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 18), 'Packet', False)
            # Calling Packet(args, kwargs) (line 388)
            Packet_call_result_756 = invoke(stypy.reporting.localization.Localization(__file__, 388, 18), Packet_751, *[wkq_752, I_DEVB_753, K_DEV_754], **kwargs_755)
            
            # Assigning a type to the variable 'wkq' (line 388)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'wkq', Packet_call_result_756)
            
            # Assigning a Call to a Name (line 389):
            
            # Call to Packet(...): (line 389)
            # Processing the call arguments (line 389)
            # Getting the type of 'wkq' (line 389)
            wkq_758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 25), 'wkq', False)
            # Getting the type of 'I_DEVB' (line 389)
            I_DEVB_759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 30), 'I_DEVB', False)
            # Getting the type of 'K_DEV' (line 389)
            K_DEV_760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 38), 'K_DEV', False)
            # Processing the call keyword arguments (line 389)
            kwargs_761 = {}
            # Getting the type of 'Packet' (line 389)
            Packet_757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 18), 'Packet', False)
            # Calling Packet(args, kwargs) (line 389)
            Packet_call_result_762 = invoke(stypy.reporting.localization.Localization(__file__, 389, 18), Packet_757, *[wkq_758, I_DEVB_759, K_DEV_760], **kwargs_761)
            
            # Assigning a type to the variable 'wkq' (line 389)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'wkq', Packet_call_result_762)
            
            # Call to HandlerTask(...): (line 390)
            # Processing the call arguments (line 390)
            # Getting the type of 'I_HANDLERB' (line 390)
            I_HANDLERB_764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 24), 'I_HANDLERB', False)
            int_765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 36), 'int')
            # Getting the type of 'wkq' (line 390)
            wkq_766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 42), 'wkq', False)
            
            # Call to waitingWithPacket(...): (line 390)
            # Processing the call keyword arguments (line 390)
            kwargs_771 = {}
            
            # Call to TaskState(...): (line 390)
            # Processing the call keyword arguments (line 390)
            kwargs_768 = {}
            # Getting the type of 'TaskState' (line 390)
            TaskState_767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 47), 'TaskState', False)
            # Calling TaskState(args, kwargs) (line 390)
            TaskState_call_result_769 = invoke(stypy.reporting.localization.Localization(__file__, 390, 47), TaskState_767, *[], **kwargs_768)
            
            # Obtaining the member 'waitingWithPacket' of a type (line 390)
            waitingWithPacket_770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 47), TaskState_call_result_769, 'waitingWithPacket')
            # Calling waitingWithPacket(args, kwargs) (line 390)
            waitingWithPacket_call_result_772 = invoke(stypy.reporting.localization.Localization(__file__, 390, 47), waitingWithPacket_770, *[], **kwargs_771)
            
            
            # Call to HandlerTaskRec(...): (line 390)
            # Processing the call keyword arguments (line 390)
            kwargs_774 = {}
            # Getting the type of 'HandlerTaskRec' (line 390)
            HandlerTaskRec_773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 80), 'HandlerTaskRec', False)
            # Calling HandlerTaskRec(args, kwargs) (line 390)
            HandlerTaskRec_call_result_775 = invoke(stypy.reporting.localization.Localization(__file__, 390, 80), HandlerTaskRec_773, *[], **kwargs_774)
            
            # Processing the call keyword arguments (line 390)
            kwargs_776 = {}
            # Getting the type of 'HandlerTask' (line 390)
            HandlerTask_763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'HandlerTask', False)
            # Calling HandlerTask(args, kwargs) (line 390)
            HandlerTask_call_result_777 = invoke(stypy.reporting.localization.Localization(__file__, 390, 12), HandlerTask_763, *[I_HANDLERB_764, int_765, wkq_766, waitingWithPacket_call_result_772, HandlerTaskRec_call_result_775], **kwargs_776)
            
            
            # Assigning a Name to a Name (line 392):
            # Getting the type of 'None' (line 392)
            None_778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 18), 'None')
            # Assigning a type to the variable 'wkq' (line 392)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'wkq', None_778)
            
            # Call to DeviceTask(...): (line 393)
            # Processing the call arguments (line 393)
            # Getting the type of 'I_DEVA' (line 393)
            I_DEVA_780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 23), 'I_DEVA', False)
            int_781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 31), 'int')
            # Getting the type of 'wkq' (line 393)
            wkq_782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 37), 'wkq', False)
            
            # Call to waiting(...): (line 393)
            # Processing the call keyword arguments (line 393)
            kwargs_787 = {}
            
            # Call to TaskState(...): (line 393)
            # Processing the call keyword arguments (line 393)
            kwargs_784 = {}
            # Getting the type of 'TaskState' (line 393)
            TaskState_783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 42), 'TaskState', False)
            # Calling TaskState(args, kwargs) (line 393)
            TaskState_call_result_785 = invoke(stypy.reporting.localization.Localization(__file__, 393, 42), TaskState_783, *[], **kwargs_784)
            
            # Obtaining the member 'waiting' of a type (line 393)
            waiting_786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 42), TaskState_call_result_785, 'waiting')
            # Calling waiting(args, kwargs) (line 393)
            waiting_call_result_788 = invoke(stypy.reporting.localization.Localization(__file__, 393, 42), waiting_786, *[], **kwargs_787)
            
            
            # Call to DeviceTaskRec(...): (line 393)
            # Processing the call keyword arguments (line 393)
            kwargs_790 = {}
            # Getting the type of 'DeviceTaskRec' (line 393)
            DeviceTaskRec_789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 65), 'DeviceTaskRec', False)
            # Calling DeviceTaskRec(args, kwargs) (line 393)
            DeviceTaskRec_call_result_791 = invoke(stypy.reporting.localization.Localization(__file__, 393, 65), DeviceTaskRec_789, *[], **kwargs_790)
            
            # Processing the call keyword arguments (line 393)
            kwargs_792 = {}
            # Getting the type of 'DeviceTask' (line 393)
            DeviceTask_779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'DeviceTask', False)
            # Calling DeviceTask(args, kwargs) (line 393)
            DeviceTask_call_result_793 = invoke(stypy.reporting.localization.Localization(__file__, 393, 12), DeviceTask_779, *[I_DEVA_780, int_781, wkq_782, waiting_call_result_788, DeviceTaskRec_call_result_791], **kwargs_792)
            
            
            # Call to DeviceTask(...): (line 394)
            # Processing the call arguments (line 394)
            # Getting the type of 'I_DEVB' (line 394)
            I_DEVB_795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 23), 'I_DEVB', False)
            int_796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 31), 'int')
            # Getting the type of 'wkq' (line 394)
            wkq_797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 37), 'wkq', False)
            
            # Call to waiting(...): (line 394)
            # Processing the call keyword arguments (line 394)
            kwargs_802 = {}
            
            # Call to TaskState(...): (line 394)
            # Processing the call keyword arguments (line 394)
            kwargs_799 = {}
            # Getting the type of 'TaskState' (line 394)
            TaskState_798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 42), 'TaskState', False)
            # Calling TaskState(args, kwargs) (line 394)
            TaskState_call_result_800 = invoke(stypy.reporting.localization.Localization(__file__, 394, 42), TaskState_798, *[], **kwargs_799)
            
            # Obtaining the member 'waiting' of a type (line 394)
            waiting_801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 42), TaskState_call_result_800, 'waiting')
            # Calling waiting(args, kwargs) (line 394)
            waiting_call_result_803 = invoke(stypy.reporting.localization.Localization(__file__, 394, 42), waiting_801, *[], **kwargs_802)
            
            
            # Call to DeviceTaskRec(...): (line 394)
            # Processing the call keyword arguments (line 394)
            kwargs_805 = {}
            # Getting the type of 'DeviceTaskRec' (line 394)
            DeviceTaskRec_804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 65), 'DeviceTaskRec', False)
            # Calling DeviceTaskRec(args, kwargs) (line 394)
            DeviceTaskRec_call_result_806 = invoke(stypy.reporting.localization.Localization(__file__, 394, 65), DeviceTaskRec_804, *[], **kwargs_805)
            
            # Processing the call keyword arguments (line 394)
            kwargs_807 = {}
            # Getting the type of 'DeviceTask' (line 394)
            DeviceTask_794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'DeviceTask', False)
            # Calling DeviceTask(args, kwargs) (line 394)
            DeviceTask_call_result_808 = invoke(stypy.reporting.localization.Localization(__file__, 394, 12), DeviceTask_794, *[I_DEVB_795, int_796, wkq_797, waiting_call_result_803, DeviceTaskRec_call_result_806], **kwargs_807)
            
            
            # Call to schedule(...): (line 396)
            # Processing the call keyword arguments (line 396)
            kwargs_810 = {}
            # Getting the type of 'schedule' (line 396)
            schedule_809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'schedule', False)
            # Calling schedule(args, kwargs) (line 396)
            schedule_call_result_811 = invoke(stypy.reporting.localization.Localization(__file__, 396, 12), schedule_809, *[], **kwargs_810)
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'taskWorkArea' (line 398)
            taskWorkArea_812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 15), 'taskWorkArea')
            # Obtaining the member 'holdCount' of a type (line 398)
            holdCount_813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 15), taskWorkArea_812, 'holdCount')
            int_814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 41), 'int')
            # Applying the binary operator '==' (line 398)
            result_eq_815 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 15), '==', holdCount_813, int_814)
            
            
            # Getting the type of 'taskWorkArea' (line 398)
            taskWorkArea_816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 50), 'taskWorkArea')
            # Obtaining the member 'qpktCount' of a type (line 398)
            qpktCount_817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 50), taskWorkArea_816, 'qpktCount')
            int_818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 76), 'int')
            # Applying the binary operator '==' (line 398)
            result_eq_819 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 50), '==', qpktCount_817, int_818)
            
            # Applying the binary operator 'and' (line 398)
            result_and_keyword_820 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 15), 'and', result_eq_815, result_eq_819)
            
            # Testing if the type of an if condition is none (line 398)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 398, 12), result_and_keyword_820):
                # Getting the type of 'False' (line 401)
                False_822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 401)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 16), 'stypy_return_type', False_822)
            else:
                
                # Testing the type of an if condition (line 398)
                if_condition_821 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 398, 12), result_and_keyword_820)
                # Assigning a type to the variable 'if_condition_821' (line 398)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'if_condition_821', if_condition_821)
                # SSA begins for if statement (line 398)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                pass
                # SSA branch for the else part of an if statement (line 398)
                module_type_store.open_ssa_branch('else')
                # Getting the type of 'False' (line 401)
                False_822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 401)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 16), 'stypy_return_type', False_822)
                # SSA join for if statement (line 398)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'True' (line 403)
        True_823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'stypy_return_type', True_823)
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 371)
        stypy_return_type_824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_824)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_824


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 369, 0, False)
        # Assigning a type to the variable 'self' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Richards.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Richards' (line 369)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 0), 'Richards', Richards)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 406, 0, False)
    
    # Passed parameters checking function
    run.stypy_localization = localization
    run.stypy_type_of_self = None
    run.stypy_type_store = module_type_store
    run.stypy_function_name = 'run'
    run.stypy_param_names_list = []
    run.stypy_varargs_param_name = None
    run.stypy_kwargs_param_name = None
    run.stypy_call_defaults = defaults
    run.stypy_call_varargs = varargs
    run.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'run', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'run', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'run(...)' code ##################

    
    # Assigning a Call to a Name (line 407):
    
    # Call to Richards(...): (line 407)
    # Processing the call keyword arguments (line 407)
    kwargs_826 = {}
    # Getting the type of 'Richards' (line 407)
    Richards_825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'Richards', False)
    # Calling Richards(args, kwargs) (line 407)
    Richards_call_result_827 = invoke(stypy.reporting.localization.Localization(__file__, 407, 8), Richards_825, *[], **kwargs_826)
    
    # Assigning a type to the variable 'r' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'r', Richards_call_result_827)
    
    # Assigning a Num to a Name (line 408):
    int_828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 17), 'int')
    # Assigning a type to the variable 'iterations' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'iterations', int_828)
    
    # Assigning a Call to a Name (line 409):
    
    # Call to run(...): (line 409)
    # Processing the call arguments (line 409)
    # Getting the type of 'iterations' (line 409)
    iterations_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 19), 'iterations', False)
    # Processing the call keyword arguments (line 409)
    kwargs_832 = {}
    # Getting the type of 'r' (line 409)
    r_829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 13), 'r', False)
    # Obtaining the member 'run' of a type (line 409)
    run_830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 13), r_829, 'run')
    # Calling run(args, kwargs) (line 409)
    run_call_result_833 = invoke(stypy.reporting.localization.Localization(__file__, 409, 13), run_830, *[iterations_831], **kwargs_832)
    
    # Assigning a type to the variable 'result' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'result', run_call_result_833)
    # Getting the type of 'True' (line 412)
    True_834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'stypy_return_type', True_834)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 406)
    stypy_return_type_835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_835)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_835

# Assigning a type to the variable 'run' (line 406)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 0), 'run', run)

# Call to run(...): (line 415)
# Processing the call keyword arguments (line 415)
kwargs_837 = {}
# Getting the type of 'run' (line 415)
run_836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 0), 'run', False)
# Calling run(args, kwargs) (line 415)
run_call_result_838 = invoke(stypy.reporting.localization.Localization(__file__, 415, 0), run_836, *[], **kwargs_837)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
