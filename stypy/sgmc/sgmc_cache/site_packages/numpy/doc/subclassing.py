
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: =============================
3: Subclassing ndarray in python
4: =============================
5: 
6: Credits
7: -------
8: 
9: This page is based with thanks on the wiki page on subclassing by Pierre
10: Gerard-Marchant - http://www.scipy.org/Subclasses.
11: 
12: Introduction
13: ------------
14: 
15: Subclassing ndarray is relatively simple, but it has some complications
16: compared to other Python objects.  On this page we explain the machinery
17: that allows you to subclass ndarray, and the implications for
18: implementing a subclass.
19: 
20: ndarrays and object creation
21: ============================
22: 
23: Subclassing ndarray is complicated by the fact that new instances of
24: ndarray classes can come about in three different ways.  These are:
25: 
26: #. Explicit constructor call - as in ``MySubClass(params)``.  This is
27:    the usual route to Python instance creation.
28: #. View casting - casting an existing ndarray as a given subclass
29: #. New from template - creating a new instance from a template
30:    instance. Examples include returning slices from a subclassed array,
31:    creating return types from ufuncs, and copying arrays.  See
32:    :ref:`new-from-template` for more details
33: 
34: The last two are characteristics of ndarrays - in order to support
35: things like array slicing.  The complications of subclassing ndarray are
36: due to the mechanisms numpy has to support these latter two routes of
37: instance creation.
38: 
39: .. _view-casting:
40: 
41: View casting
42: ------------
43: 
44: *View casting* is the standard ndarray mechanism by which you take an
45: ndarray of any subclass, and return a view of the array as another
46: (specified) subclass:
47: 
48: >>> import numpy as np
49: >>> # create a completely useless ndarray subclass
50: >>> class C(np.ndarray): pass
51: >>> # create a standard ndarray
52: >>> arr = np.zeros((3,))
53: >>> # take a view of it, as our useless subclass
54: >>> c_arr = arr.view(C)
55: >>> type(c_arr)
56: <class 'C'>
57: 
58: .. _new-from-template:
59: 
60: Creating new from template
61: --------------------------
62: 
63: New instances of an ndarray subclass can also come about by a very
64: similar mechanism to :ref:`view-casting`, when numpy finds it needs to
65: create a new instance from a template instance.  The most obvious place
66: this has to happen is when you are taking slices of subclassed arrays.
67: For example:
68: 
69: >>> v = c_arr[1:]
70: >>> type(v) # the view is of type 'C'
71: <class 'C'>
72: >>> v is c_arr # but it's a new instance
73: False
74: 
75: The slice is a *view* onto the original ``c_arr`` data.  So, when we
76: take a view from the ndarray, we return a new ndarray, of the same
77: class, that points to the data in the original.
78: 
79: There are other points in the use of ndarrays where we need such views,
80: such as copying arrays (``c_arr.copy()``), creating ufunc output arrays
81: (see also :ref:`array-wrap`), and reducing methods (like
82: ``c_arr.mean()``.
83: 
84: Relationship of view casting and new-from-template
85: --------------------------------------------------
86: 
87: These paths both use the same machinery.  We make the distinction here,
88: because they result in different input to your methods.  Specifically,
89: :ref:`view-casting` means you have created a new instance of your array
90: type from any potential subclass of ndarray.  :ref:`new-from-template`
91: means you have created a new instance of your class from a pre-existing
92: instance, allowing you - for example - to copy across attributes that
93: are particular to your subclass.
94: 
95: Implications for subclassing
96: ----------------------------
97: 
98: If we subclass ndarray, we need to deal not only with explicit
99: construction of our array type, but also :ref:`view-casting` or
100: :ref:`new-from-template`.  Numpy has the machinery to do this, and this
101: machinery that makes subclassing slightly non-standard.
102: 
103: There are two aspects to the machinery that ndarray uses to support
104: views and new-from-template in subclasses.
105: 
106: The first is the use of the ``ndarray.__new__`` method for the main work
107: of object initialization, rather then the more usual ``__init__``
108: method.  The second is the use of the ``__array_finalize__`` method to
109: allow subclasses to clean up after the creation of views and new
110: instances from templates.
111: 
112: A brief Python primer on ``__new__`` and ``__init__``
113: =====================================================
114: 
115: ``__new__`` is a standard Python method, and, if present, is called
116: before ``__init__`` when we create a class instance. See the `python
117: __new__ documentation
118: <http://docs.python.org/reference/datamodel.html#object.__new__>`_ for more detail.
119: 
120: For example, consider the following Python code:
121: 
122: .. testcode::
123: 
124:   class C(object):
125:       def __new__(cls, *args):
126:           print('Cls in __new__:', cls)
127:           print('Args in __new__:', args)
128:           return object.__new__(cls, *args)
129: 
130:       def __init__(self, *args):
131:           print('type(self) in __init__:', type(self))
132:           print('Args in __init__:', args)
133: 
134: meaning that we get:
135: 
136: >>> c = C('hello')
137: Cls in __new__: <class 'C'>
138: Args in __new__: ('hello',)
139: type(self) in __init__: <class 'C'>
140: Args in __init__: ('hello',)
141: 
142: When we call ``C('hello')``, the ``__new__`` method gets its own class
143: as first argument, and the passed argument, which is the string
144: ``'hello'``.  After python calls ``__new__``, it usually (see below)
145: calls our ``__init__`` method, with the output of ``__new__`` as the
146: first argument (now a class instance), and the passed arguments
147: following.
148: 
149: As you can see, the object can be initialized in the ``__new__``
150: method or the ``__init__`` method, or both, and in fact ndarray does
151: not have an ``__init__`` method, because all the initialization is
152: done in the ``__new__`` method.
153: 
154: Why use ``__new__`` rather than just the usual ``__init__``?  Because
155: in some cases, as for ndarray, we want to be able to return an object
156: of some other class.  Consider the following:
157: 
158: .. testcode::
159: 
160:   class D(C):
161:       def __new__(cls, *args):
162:           print('D cls is:', cls)
163:           print('D args in __new__:', args)
164:           return C.__new__(C, *args)
165: 
166:       def __init__(self, *args):
167:           # we never get here
168:           print('In D __init__')
169: 
170: meaning that:
171: 
172: >>> obj = D('hello')
173: D cls is: <class 'D'>
174: D args in __new__: ('hello',)
175: Cls in __new__: <class 'C'>
176: Args in __new__: ('hello',)
177: >>> type(obj)
178: <class 'C'>
179: 
180: The definition of ``C`` is the same as before, but for ``D``, the
181: ``__new__`` method returns an instance of class ``C`` rather than
182: ``D``.  Note that the ``__init__`` method of ``D`` does not get
183: called.  In general, when the ``__new__`` method returns an object of
184: class other than the class in which it is defined, the ``__init__``
185: method of that class is not called.
186: 
187: This is how subclasses of the ndarray class are able to return views
188: that preserve the class type.  When taking a view, the standard
189: ndarray machinery creates the new ndarray object with something
190: like::
191: 
192:   obj = ndarray.__new__(subtype, shape, ...
193: 
194: where ``subdtype`` is the subclass.  Thus the returned view is of the
195: same class as the subclass, rather than being of class ``ndarray``.
196: 
197: That solves the problem of returning views of the same type, but now
198: we have a new problem.  The machinery of ndarray can set the class
199: this way, in its standard methods for taking views, but the ndarray
200: ``__new__`` method knows nothing of what we have done in our own
201: ``__new__`` method in order to set attributes, and so on.  (Aside -
202: why not call ``obj = subdtype.__new__(...`` then?  Because we may not
203: have a ``__new__`` method with the same call signature).
204: 
205: The role of ``__array_finalize__``
206: ==================================
207: 
208: ``__array_finalize__`` is the mechanism that numpy provides to allow
209: subclasses to handle the various ways that new instances get created.
210: 
211: Remember that subclass instances can come about in these three ways:
212: 
213: #. explicit constructor call (``obj = MySubClass(params)``).  This will
214:    call the usual sequence of ``MySubClass.__new__`` then (if it exists)
215:    ``MySubClass.__init__``.
216: #. :ref:`view-casting`
217: #. :ref:`new-from-template`
218: 
219: Our ``MySubClass.__new__`` method only gets called in the case of the
220: explicit constructor call, so we can't rely on ``MySubClass.__new__`` or
221: ``MySubClass.__init__`` to deal with the view casting and
222: new-from-template.  It turns out that ``MySubClass.__array_finalize__``
223: *does* get called for all three methods of object creation, so this is
224: where our object creation housekeeping usually goes.
225: 
226: * For the explicit constructor call, our subclass will need to create a
227:   new ndarray instance of its own class.  In practice this means that
228:   we, the authors of the code, will need to make a call to
229:   ``ndarray.__new__(MySubClass,...)``, or do view casting of an existing
230:   array (see below)
231: * For view casting and new-from-template, the equivalent of
232:   ``ndarray.__new__(MySubClass,...`` is called, at the C level.
233: 
234: The arguments that ``__array_finalize__`` recieves differ for the three
235: methods of instance creation above.
236: 
237: The following code allows us to look at the call sequences and arguments:
238: 
239: .. testcode::
240: 
241:    import numpy as np
242: 
243:    class C(np.ndarray):
244:        def __new__(cls, *args, **kwargs):
245:            print('In __new__ with class %s' % cls)
246:            return np.ndarray.__new__(cls, *args, **kwargs)
247: 
248:        def __init__(self, *args, **kwargs):
249:            # in practice you probably will not need or want an __init__
250:            # method for your subclass
251:            print('In __init__ with class %s' % self.__class__)
252: 
253:        def __array_finalize__(self, obj):
254:            print('In array_finalize:')
255:            print('   self type is %s' % type(self))
256:            print('   obj type is %s' % type(obj))
257: 
258: 
259: Now:
260: 
261: >>> # Explicit constructor
262: >>> c = C((10,))
263: In __new__ with class <class 'C'>
264: In array_finalize:
265:    self type is <class 'C'>
266:    obj type is <type 'NoneType'>
267: In __init__ with class <class 'C'>
268: >>> # View casting
269: >>> a = np.arange(10)
270: >>> cast_a = a.view(C)
271: In array_finalize:
272:    self type is <class 'C'>
273:    obj type is <type 'numpy.ndarray'>
274: >>> # Slicing (example of new-from-template)
275: >>> cv = c[:1]
276: In array_finalize:
277:    self type is <class 'C'>
278:    obj type is <class 'C'>
279: 
280: The signature of ``__array_finalize__`` is::
281: 
282:     def __array_finalize__(self, obj):
283: 
284: ``ndarray.__new__`` passes ``__array_finalize__`` the new object, of our
285: own class (``self``) as well as the object from which the view has been
286: taken (``obj``).  As you can see from the output above, the ``self`` is
287: always a newly created instance of our subclass, and the type of ``obj``
288: differs for the three instance creation methods:
289: 
290: * When called from the explicit constructor, ``obj`` is ``None``
291: * When called from view casting, ``obj`` can be an instance of any
292:   subclass of ndarray, including our own.
293: * When called in new-from-template, ``obj`` is another instance of our
294:   own subclass, that we might use to update the new ``self`` instance.
295: 
296: Because ``__array_finalize__`` is the only method that always sees new
297: instances being created, it is the sensible place to fill in instance
298: defaults for new object attributes, among other tasks.
299: 
300: This may be clearer with an example.
301: 
302: Simple example - adding an extra attribute to ndarray
303: -----------------------------------------------------
304: 
305: .. testcode::
306: 
307:   import numpy as np
308: 
309:   class InfoArray(np.ndarray):
310: 
311:       def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
312:             strides=None, order=None, info=None):
313:           # Create the ndarray instance of our type, given the usual
314:           # ndarray input arguments.  This will call the standard
315:           # ndarray constructor, but return an object of our type.
316:           # It also triggers a call to InfoArray.__array_finalize__
317:           obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset, strides,
318:                            order)
319:           # set the new 'info' attribute to the value passed
320:           obj.info = info
321:           # Finally, we must return the newly created object:
322:           return obj
323: 
324:       def __array_finalize__(self, obj):
325:           # ``self`` is a new object resulting from
326:           # ndarray.__new__(InfoArray, ...), therefore it only has
327:           # attributes that the ndarray.__new__ constructor gave it -
328:           # i.e. those of a standard ndarray.
329:           #
330:           # We could have got to the ndarray.__new__ call in 3 ways:
331:           # From an explicit constructor - e.g. InfoArray():
332:           #    obj is None
333:           #    (we're in the middle of the InfoArray.__new__
334:           #    constructor, and self.info will be set when we return to
335:           #    InfoArray.__new__)
336:           if obj is None: return
337:           # From view casting - e.g arr.view(InfoArray):
338:           #    obj is arr
339:           #    (type(obj) can be InfoArray)
340:           # From new-from-template - e.g infoarr[:3]
341:           #    type(obj) is InfoArray
342:           #
343:           # Note that it is here, rather than in the __new__ method,
344:           # that we set the default value for 'info', because this
345:           # method sees all creation of default objects - with the
346:           # InfoArray.__new__ constructor, but also with
347:           # arr.view(InfoArray).
348:           self.info = getattr(obj, 'info', None)
349:           # We do not need to return anything
350: 
351: 
352: Using the object looks like this:
353: 
354:   >>> obj = InfoArray(shape=(3,)) # explicit constructor
355:   >>> type(obj)
356:   <class 'InfoArray'>
357:   >>> obj.info is None
358:   True
359:   >>> obj = InfoArray(shape=(3,), info='information')
360:   >>> obj.info
361:   'information'
362:   >>> v = obj[1:] # new-from-template - here - slicing
363:   >>> type(v)
364:   <class 'InfoArray'>
365:   >>> v.info
366:   'information'
367:   >>> arr = np.arange(10)
368:   >>> cast_arr = arr.view(InfoArray) # view casting
369:   >>> type(cast_arr)
370:   <class 'InfoArray'>
371:   >>> cast_arr.info is None
372:   True
373: 
374: This class isn't very useful, because it has the same constructor as the
375: bare ndarray object, including passing in buffers and shapes and so on.
376: We would probably prefer the constructor to be able to take an already
377: formed ndarray from the usual numpy calls to ``np.array`` and return an
378: object.
379: 
380: Slightly more realistic example - attribute added to existing array
381: -------------------------------------------------------------------
382: 
383: Here is a class that takes a standard ndarray that already exists, casts
384: as our type, and adds an extra attribute.
385: 
386: .. testcode::
387: 
388:   import numpy as np
389: 
390:   class RealisticInfoArray(np.ndarray):
391: 
392:       def __new__(cls, input_array, info=None):
393:           # Input array is an already formed ndarray instance
394:           # We first cast to be our class type
395:           obj = np.asarray(input_array).view(cls)
396:           # add the new attribute to the created instance
397:           obj.info = info
398:           # Finally, we must return the newly created object:
399:           return obj
400: 
401:       def __array_finalize__(self, obj):
402:           # see InfoArray.__array_finalize__ for comments
403:           if obj is None: return
404:           self.info = getattr(obj, 'info', None)
405: 
406: 
407: So:
408: 
409:   >>> arr = np.arange(5)
410:   >>> obj = RealisticInfoArray(arr, info='information')
411:   >>> type(obj)
412:   <class 'RealisticInfoArray'>
413:   >>> obj.info
414:   'information'
415:   >>> v = obj[1:]
416:   >>> type(v)
417:   <class 'RealisticInfoArray'>
418:   >>> v.info
419:   'information'
420: 
421: .. _array-wrap:
422: 
423: ``__array_wrap__`` for ufuncs
424: -------------------------------------------------------
425: 
426: ``__array_wrap__`` gets called at the end of numpy ufuncs and other numpy
427: functions, to allow a subclass to set the type of the return value
428: and update attributes and metadata. Let's show how this works with an example.
429: First we make the same subclass as above, but with a different name and
430: some print statements:
431: 
432: .. testcode::
433: 
434:   import numpy as np
435: 
436:   class MySubClass(np.ndarray):
437: 
438:       def __new__(cls, input_array, info=None):
439:           obj = np.asarray(input_array).view(cls)
440:           obj.info = info
441:           return obj
442: 
443:       def __array_finalize__(self, obj):
444:           print('In __array_finalize__:')
445:           print('   self is %s' % repr(self))
446:           print('   obj is %s' % repr(obj))
447:           if obj is None: return
448:           self.info = getattr(obj, 'info', None)
449: 
450:       def __array_wrap__(self, out_arr, context=None):
451:           print('In __array_wrap__:')
452:           print('   self is %s' % repr(self))
453:           print('   arr is %s' % repr(out_arr))
454:           # then just call the parent
455:           return np.ndarray.__array_wrap__(self, out_arr, context)
456: 
457: We run a ufunc on an instance of our new array:
458: 
459: >>> obj = MySubClass(np.arange(5), info='spam')
460: In __array_finalize__:
461:    self is MySubClass([0, 1, 2, 3, 4])
462:    obj is array([0, 1, 2, 3, 4])
463: >>> arr2 = np.arange(5)+1
464: >>> ret = np.add(arr2, obj)
465: In __array_wrap__:
466:    self is MySubClass([0, 1, 2, 3, 4])
467:    arr is array([1, 3, 5, 7, 9])
468: In __array_finalize__:
469:    self is MySubClass([1, 3, 5, 7, 9])
470:    obj is MySubClass([0, 1, 2, 3, 4])
471: >>> ret
472: MySubClass([1, 3, 5, 7, 9])
473: >>> ret.info
474: 'spam'
475: 
476: Note that the ufunc (``np.add``) has called the ``__array_wrap__`` method of the
477: input with the highest ``__array_priority__`` value, in this case
478: ``MySubClass.__array_wrap__``, with arguments ``self`` as ``obj``, and
479: ``out_arr`` as the (ndarray) result of the addition.  In turn, the
480: default ``__array_wrap__`` (``ndarray.__array_wrap__``) has cast the
481: result to class ``MySubClass``, and called ``__array_finalize__`` -
482: hence the copying of the ``info`` attribute.  This has all happened at the C level.
483: 
484: But, we could do anything we wanted:
485: 
486: .. testcode::
487: 
488:   class SillySubClass(np.ndarray):
489: 
490:       def __array_wrap__(self, arr, context=None):
491:           return 'I lost your data'
492: 
493: >>> arr1 = np.arange(5)
494: >>> obj = arr1.view(SillySubClass)
495: >>> arr2 = np.arange(5)
496: >>> ret = np.multiply(obj, arr2)
497: >>> ret
498: 'I lost your data'
499: 
500: So, by defining a specific ``__array_wrap__`` method for our subclass,
501: we can tweak the output from ufuncs. The ``__array_wrap__`` method
502: requires ``self``, then an argument - which is the result of the ufunc -
503: and an optional parameter *context*. This parameter is returned by some
504: ufuncs as a 3-element tuple: (name of the ufunc, argument of the ufunc,
505: domain of the ufunc). ``__array_wrap__`` should return an instance of
506: its containing class.  See the masked array subclass for an
507: implementation.
508: 
509: In addition to ``__array_wrap__``, which is called on the way out of the
510: ufunc, there is also an ``__array_prepare__`` method which is called on
511: the way into the ufunc, after the output arrays are created but before any
512: computation has been performed. The default implementation does nothing
513: but pass through the array. ``__array_prepare__`` should not attempt to
514: access the array data or resize the array, it is intended for setting the
515: output array type, updating attributes and metadata, and performing any
516: checks based on the input that may be desired before computation begins.
517: Like ``__array_wrap__``, ``__array_prepare__`` must return an ndarray or
518: subclass thereof or raise an error.
519: 
520: Extra gotchas - custom ``__del__`` methods and ndarray.base
521: -----------------------------------------------------------
522: 
523: One of the problems that ndarray solves is keeping track of memory
524: ownership of ndarrays and their views.  Consider the case where we have
525: created an ndarray, ``arr`` and have taken a slice with ``v = arr[1:]``.
526: The two objects are looking at the same memory.  Numpy keeps track of
527: where the data came from for a particular array or view, with the
528: ``base`` attribute:
529: 
530: >>> # A normal ndarray, that owns its own data
531: >>> arr = np.zeros((4,))
532: >>> # In this case, base is None
533: >>> arr.base is None
534: True
535: >>> # We take a view
536: >>> v1 = arr[1:]
537: >>> # base now points to the array that it derived from
538: >>> v1.base is arr
539: True
540: >>> # Take a view of a view
541: >>> v2 = v1[1:]
542: >>> # base points to the view it derived from
543: >>> v2.base is v1
544: True
545: 
546: In general, if the array owns its own memory, as for ``arr`` in this
547: case, then ``arr.base`` will be None - there are some exceptions to this
548: - see the numpy book for more details.
549: 
550: The ``base`` attribute is useful in being able to tell whether we have
551: a view or the original array.  This in turn can be useful if we need
552: to know whether or not to do some specific cleanup when the subclassed
553: array is deleted.  For example, we may only want to do the cleanup if
554: the original array is deleted, but not the views.  For an example of
555: how this can work, have a look at the ``memmap`` class in
556: ``numpy.core``.
557: 
558: 
559: '''
560: from __future__ import division, absolute_import, print_function
561: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_66610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, (-1)), 'str', "\n=============================\nSubclassing ndarray in python\n=============================\n\nCredits\n-------\n\nThis page is based with thanks on the wiki page on subclassing by Pierre\nGerard-Marchant - http://www.scipy.org/Subclasses.\n\nIntroduction\n------------\n\nSubclassing ndarray is relatively simple, but it has some complications\ncompared to other Python objects.  On this page we explain the machinery\nthat allows you to subclass ndarray, and the implications for\nimplementing a subclass.\n\nndarrays and object creation\n============================\n\nSubclassing ndarray is complicated by the fact that new instances of\nndarray classes can come about in three different ways.  These are:\n\n#. Explicit constructor call - as in ``MySubClass(params)``.  This is\n   the usual route to Python instance creation.\n#. View casting - casting an existing ndarray as a given subclass\n#. New from template - creating a new instance from a template\n   instance. Examples include returning slices from a subclassed array,\n   creating return types from ufuncs, and copying arrays.  See\n   :ref:`new-from-template` for more details\n\nThe last two are characteristics of ndarrays - in order to support\nthings like array slicing.  The complications of subclassing ndarray are\ndue to the mechanisms numpy has to support these latter two routes of\ninstance creation.\n\n.. _view-casting:\n\nView casting\n------------\n\n*View casting* is the standard ndarray mechanism by which you take an\nndarray of any subclass, and return a view of the array as another\n(specified) subclass:\n\n>>> import numpy as np\n>>> # create a completely useless ndarray subclass\n>>> class C(np.ndarray): pass\n>>> # create a standard ndarray\n>>> arr = np.zeros((3,))\n>>> # take a view of it, as our useless subclass\n>>> c_arr = arr.view(C)\n>>> type(c_arr)\n<class 'C'>\n\n.. _new-from-template:\n\nCreating new from template\n--------------------------\n\nNew instances of an ndarray subclass can also come about by a very\nsimilar mechanism to :ref:`view-casting`, when numpy finds it needs to\ncreate a new instance from a template instance.  The most obvious place\nthis has to happen is when you are taking slices of subclassed arrays.\nFor example:\n\n>>> v = c_arr[1:]\n>>> type(v) # the view is of type 'C'\n<class 'C'>\n>>> v is c_arr # but it's a new instance\nFalse\n\nThe slice is a *view* onto the original ``c_arr`` data.  So, when we\ntake a view from the ndarray, we return a new ndarray, of the same\nclass, that points to the data in the original.\n\nThere are other points in the use of ndarrays where we need such views,\nsuch as copying arrays (``c_arr.copy()``), creating ufunc output arrays\n(see also :ref:`array-wrap`), and reducing methods (like\n``c_arr.mean()``.\n\nRelationship of view casting and new-from-template\n--------------------------------------------------\n\nThese paths both use the same machinery.  We make the distinction here,\nbecause they result in different input to your methods.  Specifically,\n:ref:`view-casting` means you have created a new instance of your array\ntype from any potential subclass of ndarray.  :ref:`new-from-template`\nmeans you have created a new instance of your class from a pre-existing\ninstance, allowing you - for example - to copy across attributes that\nare particular to your subclass.\n\nImplications for subclassing\n----------------------------\n\nIf we subclass ndarray, we need to deal not only with explicit\nconstruction of our array type, but also :ref:`view-casting` or\n:ref:`new-from-template`.  Numpy has the machinery to do this, and this\nmachinery that makes subclassing slightly non-standard.\n\nThere are two aspects to the machinery that ndarray uses to support\nviews and new-from-template in subclasses.\n\nThe first is the use of the ``ndarray.__new__`` method for the main work\nof object initialization, rather then the more usual ``__init__``\nmethod.  The second is the use of the ``__array_finalize__`` method to\nallow subclasses to clean up after the creation of views and new\ninstances from templates.\n\nA brief Python primer on ``__new__`` and ``__init__``\n=====================================================\n\n``__new__`` is a standard Python method, and, if present, is called\nbefore ``__init__`` when we create a class instance. See the `python\n__new__ documentation\n<http://docs.python.org/reference/datamodel.html#object.__new__>`_ for more detail.\n\nFor example, consider the following Python code:\n\n.. testcode::\n\n  class C(object):\n      def __new__(cls, *args):\n          print('Cls in __new__:', cls)\n          print('Args in __new__:', args)\n          return object.__new__(cls, *args)\n\n      def __init__(self, *args):\n          print('type(self) in __init__:', type(self))\n          print('Args in __init__:', args)\n\nmeaning that we get:\n\n>>> c = C('hello')\nCls in __new__: <class 'C'>\nArgs in __new__: ('hello',)\ntype(self) in __init__: <class 'C'>\nArgs in __init__: ('hello',)\n\nWhen we call ``C('hello')``, the ``__new__`` method gets its own class\nas first argument, and the passed argument, which is the string\n``'hello'``.  After python calls ``__new__``, it usually (see below)\ncalls our ``__init__`` method, with the output of ``__new__`` as the\nfirst argument (now a class instance), and the passed arguments\nfollowing.\n\nAs you can see, the object can be initialized in the ``__new__``\nmethod or the ``__init__`` method, or both, and in fact ndarray does\nnot have an ``__init__`` method, because all the initialization is\ndone in the ``__new__`` method.\n\nWhy use ``__new__`` rather than just the usual ``__init__``?  Because\nin some cases, as for ndarray, we want to be able to return an object\nof some other class.  Consider the following:\n\n.. testcode::\n\n  class D(C):\n      def __new__(cls, *args):\n          print('D cls is:', cls)\n          print('D args in __new__:', args)\n          return C.__new__(C, *args)\n\n      def __init__(self, *args):\n          # we never get here\n          print('In D __init__')\n\nmeaning that:\n\n>>> obj = D('hello')\nD cls is: <class 'D'>\nD args in __new__: ('hello',)\nCls in __new__: <class 'C'>\nArgs in __new__: ('hello',)\n>>> type(obj)\n<class 'C'>\n\nThe definition of ``C`` is the same as before, but for ``D``, the\n``__new__`` method returns an instance of class ``C`` rather than\n``D``.  Note that the ``__init__`` method of ``D`` does not get\ncalled.  In general, when the ``__new__`` method returns an object of\nclass other than the class in which it is defined, the ``__init__``\nmethod of that class is not called.\n\nThis is how subclasses of the ndarray class are able to return views\nthat preserve the class type.  When taking a view, the standard\nndarray machinery creates the new ndarray object with something\nlike::\n\n  obj = ndarray.__new__(subtype, shape, ...\n\nwhere ``subdtype`` is the subclass.  Thus the returned view is of the\nsame class as the subclass, rather than being of class ``ndarray``.\n\nThat solves the problem of returning views of the same type, but now\nwe have a new problem.  The machinery of ndarray can set the class\nthis way, in its standard methods for taking views, but the ndarray\n``__new__`` method knows nothing of what we have done in our own\n``__new__`` method in order to set attributes, and so on.  (Aside -\nwhy not call ``obj = subdtype.__new__(...`` then?  Because we may not\nhave a ``__new__`` method with the same call signature).\n\nThe role of ``__array_finalize__``\n==================================\n\n``__array_finalize__`` is the mechanism that numpy provides to allow\nsubclasses to handle the various ways that new instances get created.\n\nRemember that subclass instances can come about in these three ways:\n\n#. explicit constructor call (``obj = MySubClass(params)``).  This will\n   call the usual sequence of ``MySubClass.__new__`` then (if it exists)\n   ``MySubClass.__init__``.\n#. :ref:`view-casting`\n#. :ref:`new-from-template`\n\nOur ``MySubClass.__new__`` method only gets called in the case of the\nexplicit constructor call, so we can't rely on ``MySubClass.__new__`` or\n``MySubClass.__init__`` to deal with the view casting and\nnew-from-template.  It turns out that ``MySubClass.__array_finalize__``\n*does* get called for all three methods of object creation, so this is\nwhere our object creation housekeeping usually goes.\n\n* For the explicit constructor call, our subclass will need to create a\n  new ndarray instance of its own class.  In practice this means that\n  we, the authors of the code, will need to make a call to\n  ``ndarray.__new__(MySubClass,...)``, or do view casting of an existing\n  array (see below)\n* For view casting and new-from-template, the equivalent of\n  ``ndarray.__new__(MySubClass,...`` is called, at the C level.\n\nThe arguments that ``__array_finalize__`` recieves differ for the three\nmethods of instance creation above.\n\nThe following code allows us to look at the call sequences and arguments:\n\n.. testcode::\n\n   import numpy as np\n\n   class C(np.ndarray):\n       def __new__(cls, *args, **kwargs):\n           print('In __new__ with class %s' % cls)\n           return np.ndarray.__new__(cls, *args, **kwargs)\n\n       def __init__(self, *args, **kwargs):\n           # in practice you probably will not need or want an __init__\n           # method for your subclass\n           print('In __init__ with class %s' % self.__class__)\n\n       def __array_finalize__(self, obj):\n           print('In array_finalize:')\n           print('   self type is %s' % type(self))\n           print('   obj type is %s' % type(obj))\n\n\nNow:\n\n>>> # Explicit constructor\n>>> c = C((10,))\nIn __new__ with class <class 'C'>\nIn array_finalize:\n   self type is <class 'C'>\n   obj type is <type 'NoneType'>\nIn __init__ with class <class 'C'>\n>>> # View casting\n>>> a = np.arange(10)\n>>> cast_a = a.view(C)\nIn array_finalize:\n   self type is <class 'C'>\n   obj type is <type 'numpy.ndarray'>\n>>> # Slicing (example of new-from-template)\n>>> cv = c[:1]\nIn array_finalize:\n   self type is <class 'C'>\n   obj type is <class 'C'>\n\nThe signature of ``__array_finalize__`` is::\n\n    def __array_finalize__(self, obj):\n\n``ndarray.__new__`` passes ``__array_finalize__`` the new object, of our\nown class (``self``) as well as the object from which the view has been\ntaken (``obj``).  As you can see from the output above, the ``self`` is\nalways a newly created instance of our subclass, and the type of ``obj``\ndiffers for the three instance creation methods:\n\n* When called from the explicit constructor, ``obj`` is ``None``\n* When called from view casting, ``obj`` can be an instance of any\n  subclass of ndarray, including our own.\n* When called in new-from-template, ``obj`` is another instance of our\n  own subclass, that we might use to update the new ``self`` instance.\n\nBecause ``__array_finalize__`` is the only method that always sees new\ninstances being created, it is the sensible place to fill in instance\ndefaults for new object attributes, among other tasks.\n\nThis may be clearer with an example.\n\nSimple example - adding an extra attribute to ndarray\n-----------------------------------------------------\n\n.. testcode::\n\n  import numpy as np\n\n  class InfoArray(np.ndarray):\n\n      def __new__(subtype, shape, dtype=float, buffer=None, offset=0,\n            strides=None, order=None, info=None):\n          # Create the ndarray instance of our type, given the usual\n          # ndarray input arguments.  This will call the standard\n          # ndarray constructor, but return an object of our type.\n          # It also triggers a call to InfoArray.__array_finalize__\n          obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset, strides,\n                           order)\n          # set the new 'info' attribute to the value passed\n          obj.info = info\n          # Finally, we must return the newly created object:\n          return obj\n\n      def __array_finalize__(self, obj):\n          # ``self`` is a new object resulting from\n          # ndarray.__new__(InfoArray, ...), therefore it only has\n          # attributes that the ndarray.__new__ constructor gave it -\n          # i.e. those of a standard ndarray.\n          #\n          # We could have got to the ndarray.__new__ call in 3 ways:\n          # From an explicit constructor - e.g. InfoArray():\n          #    obj is None\n          #    (we're in the middle of the InfoArray.__new__\n          #    constructor, and self.info will be set when we return to\n          #    InfoArray.__new__)\n          if obj is None: return\n          # From view casting - e.g arr.view(InfoArray):\n          #    obj is arr\n          #    (type(obj) can be InfoArray)\n          # From new-from-template - e.g infoarr[:3]\n          #    type(obj) is InfoArray\n          #\n          # Note that it is here, rather than in the __new__ method,\n          # that we set the default value for 'info', because this\n          # method sees all creation of default objects - with the\n          # InfoArray.__new__ constructor, but also with\n          # arr.view(InfoArray).\n          self.info = getattr(obj, 'info', None)\n          # We do not need to return anything\n\n\nUsing the object looks like this:\n\n  >>> obj = InfoArray(shape=(3,)) # explicit constructor\n  >>> type(obj)\n  <class 'InfoArray'>\n  >>> obj.info is None\n  True\n  >>> obj = InfoArray(shape=(3,), info='information')\n  >>> obj.info\n  'information'\n  >>> v = obj[1:] # new-from-template - here - slicing\n  >>> type(v)\n  <class 'InfoArray'>\n  >>> v.info\n  'information'\n  >>> arr = np.arange(10)\n  >>> cast_arr = arr.view(InfoArray) # view casting\n  >>> type(cast_arr)\n  <class 'InfoArray'>\n  >>> cast_arr.info is None\n  True\n\nThis class isn't very useful, because it has the same constructor as the\nbare ndarray object, including passing in buffers and shapes and so on.\nWe would probably prefer the constructor to be able to take an already\nformed ndarray from the usual numpy calls to ``np.array`` and return an\nobject.\n\nSlightly more realistic example - attribute added to existing array\n-------------------------------------------------------------------\n\nHere is a class that takes a standard ndarray that already exists, casts\nas our type, and adds an extra attribute.\n\n.. testcode::\n\n  import numpy as np\n\n  class RealisticInfoArray(np.ndarray):\n\n      def __new__(cls, input_array, info=None):\n          # Input array is an already formed ndarray instance\n          # We first cast to be our class type\n          obj = np.asarray(input_array).view(cls)\n          # add the new attribute to the created instance\n          obj.info = info\n          # Finally, we must return the newly created object:\n          return obj\n\n      def __array_finalize__(self, obj):\n          # see InfoArray.__array_finalize__ for comments\n          if obj is None: return\n          self.info = getattr(obj, 'info', None)\n\n\nSo:\n\n  >>> arr = np.arange(5)\n  >>> obj = RealisticInfoArray(arr, info='information')\n  >>> type(obj)\n  <class 'RealisticInfoArray'>\n  >>> obj.info\n  'information'\n  >>> v = obj[1:]\n  >>> type(v)\n  <class 'RealisticInfoArray'>\n  >>> v.info\n  'information'\n\n.. _array-wrap:\n\n``__array_wrap__`` for ufuncs\n-------------------------------------------------------\n\n``__array_wrap__`` gets called at the end of numpy ufuncs and other numpy\nfunctions, to allow a subclass to set the type of the return value\nand update attributes and metadata. Let's show how this works with an example.\nFirst we make the same subclass as above, but with a different name and\nsome print statements:\n\n.. testcode::\n\n  import numpy as np\n\n  class MySubClass(np.ndarray):\n\n      def __new__(cls, input_array, info=None):\n          obj = np.asarray(input_array).view(cls)\n          obj.info = info\n          return obj\n\n      def __array_finalize__(self, obj):\n          print('In __array_finalize__:')\n          print('   self is %s' % repr(self))\n          print('   obj is %s' % repr(obj))\n          if obj is None: return\n          self.info = getattr(obj, 'info', None)\n\n      def __array_wrap__(self, out_arr, context=None):\n          print('In __array_wrap__:')\n          print('   self is %s' % repr(self))\n          print('   arr is %s' % repr(out_arr))\n          # then just call the parent\n          return np.ndarray.__array_wrap__(self, out_arr, context)\n\nWe run a ufunc on an instance of our new array:\n\n>>> obj = MySubClass(np.arange(5), info='spam')\nIn __array_finalize__:\n   self is MySubClass([0, 1, 2, 3, 4])\n   obj is array([0, 1, 2, 3, 4])\n>>> arr2 = np.arange(5)+1\n>>> ret = np.add(arr2, obj)\nIn __array_wrap__:\n   self is MySubClass([0, 1, 2, 3, 4])\n   arr is array([1, 3, 5, 7, 9])\nIn __array_finalize__:\n   self is MySubClass([1, 3, 5, 7, 9])\n   obj is MySubClass([0, 1, 2, 3, 4])\n>>> ret\nMySubClass([1, 3, 5, 7, 9])\n>>> ret.info\n'spam'\n\nNote that the ufunc (``np.add``) has called the ``__array_wrap__`` method of the\ninput with the highest ``__array_priority__`` value, in this case\n``MySubClass.__array_wrap__``, with arguments ``self`` as ``obj``, and\n``out_arr`` as the (ndarray) result of the addition.  In turn, the\ndefault ``__array_wrap__`` (``ndarray.__array_wrap__``) has cast the\nresult to class ``MySubClass``, and called ``__array_finalize__`` -\nhence the copying of the ``info`` attribute.  This has all happened at the C level.\n\nBut, we could do anything we wanted:\n\n.. testcode::\n\n  class SillySubClass(np.ndarray):\n\n      def __array_wrap__(self, arr, context=None):\n          return 'I lost your data'\n\n>>> arr1 = np.arange(5)\n>>> obj = arr1.view(SillySubClass)\n>>> arr2 = np.arange(5)\n>>> ret = np.multiply(obj, arr2)\n>>> ret\n'I lost your data'\n\nSo, by defining a specific ``__array_wrap__`` method for our subclass,\nwe can tweak the output from ufuncs. The ``__array_wrap__`` method\nrequires ``self``, then an argument - which is the result of the ufunc -\nand an optional parameter *context*. This parameter is returned by some\nufuncs as a 3-element tuple: (name of the ufunc, argument of the ufunc,\ndomain of the ufunc). ``__array_wrap__`` should return an instance of\nits containing class.  See the masked array subclass for an\nimplementation.\n\nIn addition to ``__array_wrap__``, which is called on the way out of the\nufunc, there is also an ``__array_prepare__`` method which is called on\nthe way into the ufunc, after the output arrays are created but before any\ncomputation has been performed. The default implementation does nothing\nbut pass through the array. ``__array_prepare__`` should not attempt to\naccess the array data or resize the array, it is intended for setting the\noutput array type, updating attributes and metadata, and performing any\nchecks based on the input that may be desired before computation begins.\nLike ``__array_wrap__``, ``__array_prepare__`` must return an ndarray or\nsubclass thereof or raise an error.\n\nExtra gotchas - custom ``__del__`` methods and ndarray.base\n-----------------------------------------------------------\n\nOne of the problems that ndarray solves is keeping track of memory\nownership of ndarrays and their views.  Consider the case where we have\ncreated an ndarray, ``arr`` and have taken a slice with ``v = arr[1:]``.\nThe two objects are looking at the same memory.  Numpy keeps track of\nwhere the data came from for a particular array or view, with the\n``base`` attribute:\n\n>>> # A normal ndarray, that owns its own data\n>>> arr = np.zeros((4,))\n>>> # In this case, base is None\n>>> arr.base is None\nTrue\n>>> # We take a view\n>>> v1 = arr[1:]\n>>> # base now points to the array that it derived from\n>>> v1.base is arr\nTrue\n>>> # Take a view of a view\n>>> v2 = v1[1:]\n>>> # base points to the view it derived from\n>>> v2.base is v1\nTrue\n\nIn general, if the array owns its own memory, as for ``arr`` in this\ncase, then ``arr.base`` will be None - there are some exceptions to this\n- see the numpy book for more details.\n\nThe ``base`` attribute is useful in being able to tell whether we have\na view or the original array.  This in turn can be useful if we need\nto know whether or not to do some specific cleanup when the subclassed\narray is deleted.  For example, we may only want to do the cleanup if\nthe original array is deleted, but not the views.  For an example of\nhow this can work, have a look at the ``memmap`` class in\n``numpy.core``.\n\n\n")

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
