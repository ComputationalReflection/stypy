
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import types
2: import collections
3: 
4: from stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import ExtraTypeDefinitions
5: from type_group_copy import TypeGroup
6: from stypy_copy.errors_copy.type_error_copy import TypeError
7: from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy
8: from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType
9: from stypy_copy.errors_copy.type_warning_copy import TypeWarning
10: from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import type_inference_proxy_management_copy
11: 
12: '''
13: File to define all type groups available to form type rules
14: '''
15: 
16: 
17: class DependentType:
18:     '''
19:     A DependentType is a special base class that indicates that a type group has to be called to obtain the real
20:     type it represent. Call is done using the parameters that are trying to match the rule. For example, imagine that
21:     we call the + operator with an object that defines the __add__ method and another type to add to. With an object
22:     that defines an __add__ method we don't really know what will be the result of calling __add__ over this object
23:     with the second parameter, so the __add__ method has to be called (well, in fact, the type inference equivalent
24:     version of the __add__ method will be called) to obtain the real return type.
25: 
26:     Dependent types are a powerful mechanism to calculate the return type of operations that depend on calls to
27:     certain object members or even to detect incorrect definitions of members in objects (__int__ method defined in
28:     object that do not return int, for example).
29:     '''
30: 
31:     def __init__(self, report_errors=False):
32:         '''
33:         Build a Dependent type instance
34:         :param report_errors: Flag to indicate if errors found when calling this type will be reported or not (in that
35:         case other code will do it)
36:         '''
37:         self.report_errors = report_errors
38:         self.call_arity = 0
39: 
40:     def __call__(self, *call_args, **call_kwargs):
41:         '''
42:         Call the dependent type. Empty in this implementation, concrete calls must be defined in subclasses
43:         '''
44:         pass
45: 
46: 
47: '''
48: Type groups with special meaning. All of them define a __eq__ method that indicates if the passed type matches with
49: the type group, storing this passed type. They also define a __call__ method that actually perform the type checking
50: and calculate the return type. __eq__ and __call__ methods are called sequentially if __eq__ result is True, so the
51: storage of the passed type is safe to use in the __call__ as each time an __eq__ is called is replaced. This is the
52: way the type rule checking mechanism works: TypeGroups are not meant to be used in other parts of the stypy runtime,
53: and if they do, only the __eq__ method should be used to check if a type belongs to a group.
54: '''
55: 
56: 
57: class HasMember(TypeGroup, DependentType):
58:     '''
59:         Type of any object that has a member with the specified arity, and that can be called with the corresponding
60:         params.
61:     '''
62: 
63:     def __init__(self, member, expected_return_type, call_arity=0, report_errors=False):
64:         DependentType.__init__(self, report_errors)
65:         TypeGroup.__init__(self, [])
66:         self.member = member
67:         self.expected_return_type = expected_return_type
68:         self.member_obj = None
69:         self.call_arity = call_arity
70: 
71:     def format_arity(self):
72:         str_ = "("
73:         for i in range(self.call_arity):
74:             str_ += "parameter" + str(i) + ", "
75: 
76:         if self.call_arity > 0:
77:             str_ = str_[:-2]
78: 
79:         return str_ + ")"
80: 
81:     def __eq__(self, type_):
82:         self.member_obj = type_.get_type_of_member(None, self.member)
83:         if isinstance(self.member_obj, TypeError):
84:             if not self.report_errors:
85:                 TypeError.remove_error_msg(self.member_obj)
86:             return False
87: 
88:         return True
89: 
90:     def __call__(self, localization, *call_args, **call_kwargs):
91:         if callable(self.member_obj.get_python_type()):
92:             # Call the member
93:             equivalent_type = self.member_obj.invoke(localization, *call_args, **call_kwargs)
94: 
95:             # Call was impossible: Invokation error has to be removed because we provide a general one later
96:             if isinstance(equivalent_type, TypeError):
97:                 if not self.report_errors:
98:                     TypeError.remove_error_msg(equivalent_type)
99:                 self.member_obj = None
100:                 return False, equivalent_type
101: 
102:             # Call was possible, but the expected return type cannot be predetermined (we have to recheck it later)
103:             if isinstance(self.expected_return_type, UndefinedType):
104:                 self.member_obj = None
105:                 return True, equivalent_type
106: 
107:             # Call was possible, but the expected return type is Any)
108:             if self.expected_return_type is DynamicType:
109:                 self.member_obj = None
110:                 return True, equivalent_type
111: 
112:             # Call was possible, so we check if the predetermined return type is the same that the one that is returned
113:             if not issubclass(equivalent_type.get_python_type(), self.expected_return_type):
114:                 self.member_obj = None
115:                 return False, equivalent_type
116:             else:
117:                 return True, equivalent_type
118: 
119:         self.member_obj = None
120:         return True, None
121: 
122:     def __repr__(self):
123:         ret_str = "Instance defining "
124:         ret_str += str(self.member)
125:         ret_str += self.format_arity()
126:         return ret_str
127: 
128: 
129: class IterableDataStructureWithTypedElements(TypeGroup, DependentType):
130:     '''
131:     Represent all iterable data structures that contain a certain type or types
132:     '''
133: 
134:     def __init__(self, *content_types):
135:         DependentType.__init__(self, True)
136:         TypeGroup.__init__(self, [])
137:         self.content_types = content_types
138:         self.type_ = None
139:         self.call_arity = 0
140: 
141:     def __eq__(self, type_):
142:         self.type_ = type_
143:         return type_.get_python_type() in TypeGroups.IterableDataStructure
144: 
145:     def __call__(self, localization, *call_args, **call_kwargs):
146:         contained_elements = self.type_.get_elements_type()
147:         if isinstance(contained_elements, union_type_copy.UnionType):
148:             types_to_examine = contained_elements.types
149:         else:
150:             types_to_examine = [contained_elements]
151: 
152:         right_types = []
153:         wrong_types = []
154: 
155:         for type_ in types_to_examine:
156:             match_found = False
157:             for declared_contained_type in self.content_types:
158:                 if declared_contained_type == type_:
159:                     if isinstance(declared_contained_type, DependentType):
160:                         if declared_contained_type.call_arity == 0:
161:                             correct, return_type = declared_contained_type(localization)
162:                         else:
163:                             correct, return_type = declared_contained_type(localization, type_)
164:                         if correct:
165:                             match_found = True
166:                             if type_ not in right_types:
167:                                 right_types.append(type_)
168:                                 if type_ in wrong_types:
169:                                     wrong_types.remove(type_)
170:                         else:
171:                             if type_ not in wrong_types and type_ not in right_types:
172:                                 wrong_types.append(type_)
173:                     else:
174:                         match_found = True
175:                         right_types.append(type_)
176: 
177:             if not match_found:
178:                 if type_ not in wrong_types and type_ not in right_types:
179:                     wrong_types.append(type_)
180:                 match_found = False
181: 
182:         if self.report_errors:
183:             # All types are wrong
184:             if len(right_types) == 0:
185:                 if len(wrong_types) > 0:
186:                     TypeError(localization,
187:                               "None of the iterable contained types: {0} match the expected ones {1}".format(
188:                                   str(types_to_examine), str(self.content_types)
189:                               ))
190:             else:
191:                 if len(wrong_types) > 0:
192:                     TypeWarning(localization,
193:                                 "Some of the iterable contained types: {0} do not match the expected ones {1}".format(
194:                                     str(wrong_types), str(self.content_types)
195:                                 ))
196:         else:
197:             if len(right_types) == 0 and len(wrong_types) > 0:
198:                 TypeWarning(localization,
199:                             "Some of the iterable contained types: {0} do not match the expected ones {1}".format(
200:                                 str(wrong_types), str(self.content_types)
201:                             ))
202: 
203:         if len(right_types) > 0:
204:             return True, None
205:         else:
206:             return False, wrong_types
207: 
208:     def __repr__(self):
209:         ret_str = "Iterable["
210: 
211:         contents = ""
212:         for content in self.content_types:
213:             contents += str(content) + ", "
214:         contents = contents[:-2]
215: 
216:         ret_str += contents
217:         ret_str += "]"
218:         return ret_str
219: 
220: 
221: class DynamicType(TypeGroup):
222:     '''
223:     Any type (type cannot be statically calculated)
224:     '''
225: 
226:     def __init__(self, *members):
227:         TypeGroup.__init__(self, [])
228:         self.members = members
229: 
230:     def __eq__(self, type_):
231:         return True
232: 
233: 
234: class SupportsStructuralIntercession(TypeGroup):
235:     '''
236:     Any Python object that supports structural intercession
237:     '''
238: 
239:     def __init__(self, *members):
240:         TypeGroup.__init__(self, [])
241:         self.members = members
242: 
243:     def __eq__(self, type_):
244:         self.type_ = type_
245:         return type_inference_proxy_management_copy.supports_structural_reflection(type_)
246: 
247:     def __call__(self, localization, *call_args, **call_kwargs):
248:         temp = self.type_
249:         self.type_ = None
250: 
251:         return temp
252: 
253: 
254: class SubtypeOf(TypeGroup):
255:     '''
256:     A subtype of the type passed in the constructor
257:     '''
258: 
259:     def __init__(self, *types_):
260:         TypeGroup.__init__(self, [])
261:         self.types = types_
262: 
263:     def __eq__(self, type_):
264:         self.type_ = type_
265:         for pattern_type in self.types:
266:             if not issubclass(type_, pattern_type):
267:                 return False
268:         return True
269: 
270:     def __call__(self, localization, *call_args, **call_kwargs):
271:         temp = self.type_
272:         self.type_ = None
273: 
274:         return temp
275: 
276: 
277: class IsHashable(TypeGroup):
278:     '''
279:     Represent types that can properly implement the __hash__ members, so it can be placed as keys on a dict
280:     '''
281: 
282:     def __init__(self, *types_):
283:         TypeGroup.__init__(self, [])
284:         self.types = types_
285: 
286:     def __eq__(self, type_):
287:         self.type_ = type_
288:         if issubclass(type_, collections.Hashable):
289:             return True
290:         return False
291: 
292:     def __call__(self, localization, *call_args, **call_kwargs):
293:         temp = self.type_
294:         self.type_ = None
295: 
296:         return temp
297: 
298: 
299: class TypeOfParam(TypeGroup, DependentType):
300:     '''
301:     This type group is special in the sense that it don't really group any types, only returns the param number
302:     passed in the constructor when it is called with a list of parameters. This is really used to simplify several
303:     type rules in which the type returned by a member call is equal to the type of one of its parameters
304:     '''
305: 
306:     def __init__(self, *param_number):
307:         DependentType.__init__(self)
308:         TypeGroup.__init__(self, [])
309:         self.param_number = param_number[0]
310: 
311:     def __eq__(self, type_):
312:         return False
313: 
314:     def __repr__(self):
315:         ret_str = type(self).__name__ + "(" + self.param_number + ")"
316: 
317:         return ret_str
318: 
319:     def __call__(self, localization, *call_args, **call_kwargs):
320:         return call_args[0][self.param_number - 1]
321: 
322: 
323: class Callable(TypeGroup):
324:     '''
325:     Represent all callable objects (those that define the member __call__)
326:     '''
327: 
328:     def __init__(self):
329:         TypeGroup.__init__(self, [])
330: 
331:     def __eq__(self, type_):
332:         self.member_obj = type_.get_type_of_member(None, "__call__")
333:         if isinstance(self.member_obj, TypeError):
334:             return False
335: 
336:         return True
337: 
338:     def __call__(self, localization, *call_args, **call_kwargs):
339:         temp = self.member_obj
340:         self.member_obj = None
341: 
342:         return temp
343: 
344: 
345: class TypeObject(TypeGroup):
346:     '''
347:     Represent type and types.ClassType types
348:     '''
349:     type_objs = [type, types.ClassType]
350: 
351:     def __init__(self):
352:         TypeGroup.__init__(self, [])
353: 
354:     def __eq__(self, type_):
355:         self.member_obj = type(type_.get_python_type())
356:         if self.member_obj in TypeObject.type_objs:
357:             return not type_.is_type_instance()
358: 
359:         return False
360: 
361:     def __call__(self, localization, *call_args, **call_kwargs):
362:         temp = self.member_obj
363:         self.member_obj = None
364: 
365:         return temp
366: 
367: 
368: class InstanceOfType(TypeGroup):
369:     '''
370:     Represent type and types.ClassType types
371:     '''
372:     type_objs = [type, types.ClassType]
373: 
374:     def __init__(self):
375:         TypeGroup.__init__(self, [])
376: 
377:     def __eq__(self, type_):
378:         self.member_obj = type(type_.get_python_type())
379:         if self.member_obj in TypeObject.type_objs:
380:             return type_.is_type_instance()
381: 
382:         return False
383: 
384:     def __call__(self, localization, *call_args, **call_kwargs):
385:         temp = self.member_obj
386:         self.member_obj = None
387: 
388:         return temp
389: 
390: 
391: class VarArgType(TypeGroup):
392:     '''
393:     Special type group indicating that a callable has an unlimited amount of parameters
394:     '''
395: 
396:     def __init__(self, *types_):
397:         TypeGroup.__init__(self, [])
398:         self.types = types_
399: 
400:     def __eq__(self, type_):
401:         return True
402: 
403:     def __call__(self, localization, *call_args, **call_kwargs):
404:         temp = self.type_
405:         self.type_ = None
406: 
407:         return temp
408: 
409: 
410: class TypeGroups:
411:     '''
412:     Class to hold definitions of type groups that are composed by lists of known Python types
413:     '''
414: 
415:     def __init__(self):
416:         pass
417: 
418:     @staticmethod
419:     def get_rule_groups():
420:         '''
421:         Obtain all the types defined in this class
422:         '''
423: 
424:         def filter_func(element):
425:             return isinstance(element, list)
426: 
427:         return filter(lambda member: filter_func(getattr(TypeGroups, member)), TypeGroups.__dict__)
428: 
429:     # Reals
430:     RealNumber = [int, long, float, bool]
431: 
432:     # Any number
433:     Number = [int, long, float, bool, complex]
434: 
435:     # Integers
436:     Integer = [int, long, bool]
437: 
438:     # strings
439:     Str = [str, unicode, buffer]
440: 
441:     # Bynary strings
442:     ByteSequence = [buffer, bytearray, str, memoryview]
443: 
444:     # Data structures that can be iterable plus iterators
445:     IterableDataStructure = [
446:         list,
447:         dict,
448:         ExtraTypeDefinitions.tupleiterator,
449:         ExtraTypeDefinitions.dict_values,
450:         frozenset,
451:         ExtraTypeDefinitions.rangeiterator,
452:         types.GeneratorType,
453:         enumerate,
454:         bytearray,
455:         iter,
456:         reversed,
457:         ExtraTypeDefinitions.dictionary_keyiterator,
458:         ExtraTypeDefinitions.bytearray_iterator,
459:         ExtraTypeDefinitions.dictionary_valueiterator,
460:         ExtraTypeDefinitions.dictionary_itemiterator,
461:         ExtraTypeDefinitions.listiterator,
462:         ExtraTypeDefinitions.listreverseiterator,
463:         tuple,
464:         set,
465:         xrange]
466: 
467:     # Data structures that can be iterable plus iterators plus iterable objects that are not necessarily data structures
468:     IterableObject = [
469:         list,
470:         dict,
471:         ExtraTypeDefinitions.tupleiterator,
472:         ExtraTypeDefinitions.dict_values,
473:         frozenset,
474:         ExtraTypeDefinitions.rangeiterator,
475:         types.GeneratorType,
476:         enumerate,
477:         bytearray,
478:         iter,
479:         reversed,
480:         ExtraTypeDefinitions.dictionary_keyiterator,
481:         ExtraTypeDefinitions.bytearray_iterator,
482:         ExtraTypeDefinitions.dictionary_valueiterator,
483:         ExtraTypeDefinitions.dictionary_itemiterator,
484:         ExtraTypeDefinitions.listiterator,
485:         ExtraTypeDefinitions.listreverseiterator,
486:         tuple,
487:         set,
488:         xrange,
489:         memoryview,
490:         types.DictProxyType]
491: 
492: 
493: '''
494: Instances of type groups. These are the ones that are really used in the type rules, as are concrete usages
495: of the previously defined type groups.
496: 
497: NOTE: To interpret instances of type groups, you should take into account the following:
498: 
499: - UndefinedType as expected return type: We cannot statically determine the return
500: type of this method. So we obtain it calling the member, obtaining its type
501: and reevaluating the member ruleset again with this type substituting the dependent
502: one.
503: 
504: - DynamicType as expected return type: We also cannot statically determine the return
505: type of this method. But this time we directly return the return type of the invoked
506: member.
507: '''
508: 
509: # Type conversion methods
510: CastsToInt = HasMember("__int__", int, 0)
511: CastsToLong = HasMember("__long__", long, 0)
512: CastsToFloat = HasMember("__float__", float, 0)
513: CastsToComplex = HasMember("__complex__", complex, 0)
514: CastsToOct = HasMember("__oct__", str, 0)
515: CastsToHex = HasMember("__hex__", str, 0)
516: CastsToIndex = HasMember("__index__", int, 0)
517: CastsToTrunc = HasMember("__trunc__", UndefinedType, 0)
518: CastsToCoerce = HasMember("__coerce__", UndefinedType, 0)
519: 
520: # TODO: Explicits calls to __cmp__ are allowed to return any type. Implict ones not.
521: # TODO: Is this controlled?
522: # Comparison magic methods:
523: Overloads__cmp__ = HasMember("__cmp__", DynamicType, 1)
524: Overloads__eq__ = HasMember("__eq__", DynamicType, 1)
525: Overloads__ne__ = HasMember("__ne__", DynamicType, 1)
526: Overloads__lt__ = HasMember("__lt__", DynamicType, 1)
527: Overloads__gt__ = HasMember("__gt__", DynamicType, 1)
528: Overloads__le__ = HasMember("__le__", DynamicType, 1)
529: Overloads__ge__ = HasMember("__ge__", DynamicType, 1)
530: 
531: # Unary operators and functions:
532: Overloads__pos__ = HasMember("__pos__", UndefinedType, 0)
533: Overloads__neg__ = HasMember("__neg__", UndefinedType, 0)
534: Overloads__abs__ = HasMember("__abs__", UndefinedType, 0)
535: Overloads__invert__ = HasMember("__invert__", UndefinedType, 0)
536: # TODO: round, ceil and floot seems to rely in __float__ implementation
537: Overloads__round__ = HasMember("__round__", int, 1)
538: Overloads__floor__ = HasMember("__floor__", int, 0)
539: Overloads__ceil__ = HasMember("__ceil__", int, 0)
540: 
541: Overloads__trunc__ = HasMember("__trunc__", int, 0)
542: 
543: # Normal numeric operators:
544: Overloads__add__ = HasMember("__add__", DynamicType, 1)
545: Overloads__sub__ = HasMember("__sub__", DynamicType, 1)
546: Overloads__mul__ = HasMember("__mul__", DynamicType, 1)
547: Overloads__floordiv__ = HasMember("__floordiv__", DynamicType, 1)
548: Overloads__div__ = HasMember("__div__", DynamicType, 1)
549: Overloads__truediv__ = HasMember("__truediv__", DynamicType, 1)
550: Overloads__mod__ = HasMember("__mod__", DynamicType, 1)
551: Overloads__divmod__ = HasMember("__divmod__", DynamicType, 1)
552: Overloads__pow__ = HasMember("__pow__", DynamicType, 2)
553: Overloads__lshift__ = HasMember("__lshift__", DynamicType, 1)
554: Overloads__rshift__ = HasMember("__rshift__", DynamicType, 1)
555: Overloads__and__ = HasMember("__and__", DynamicType, 1)
556: Overloads__or__ = HasMember("__or__", DynamicType, 1)
557: Overloads__xor__ = HasMember("__xor__", DynamicType, 1)
558: 
559: # Normal reflected numeric operators:
560: Overloads__radd__ = HasMember("__radd__", DynamicType, 1)
561: Overloads__rsub__ = HasMember("__rsub__", DynamicType, 1)
562: Overloads__rmul__ = HasMember("__rmul__", DynamicType, 1)
563: Overloads__rfloordiv__ = HasMember("__rfloordiv__", DynamicType, 1)
564: Overloads__rdiv__ = HasMember("__rdiv__", DynamicType, 1)
565: Overloads__rtruediv__ = HasMember("__rtruediv__", DynamicType, 1)
566: Overloads__rmod__ = HasMember("__rmod__", DynamicType, 1)
567: Overloads__rdivmod__ = HasMember("__rdivmod__", DynamicType, 1)
568: Overloads__rpow__ = HasMember("__rpow__", DynamicType, 1)
569: Overloads__rlshift__ = HasMember("__rlshift__", DynamicType, 1)
570: Overloads__rrshift__ = HasMember("__rrshift__", DynamicType, 1)
571: Overloads__rand__ = HasMember("__rand__", DynamicType, 1)
572: Overloads__ror__ = HasMember("__ror__", DynamicType, 1)
573: Overloads__rxor__ = HasMember("__rxor__", DynamicType, 1)
574: 
575: 
576: # Augmented assignment numeric operators:
577: Overloads__iadd__ = HasMember("__iadd__", DynamicType, 1)
578: Overloads__isub__ = HasMember("__isub__", DynamicType, 1)
579: Overloads__imul__ = HasMember("__imul__", DynamicType, 1)
580: Overloads__ifloordiv__ = HasMember("__ifloordiv__", DynamicType, 1)
581: Overloads__idiv__ = HasMember("__idiv__", DynamicType, 1)
582: Overloads__itruediv__ = HasMember("__itruediv__", DynamicType, 1)
583: Overloads__imod__ = HasMember("__imod__", DynamicType, 1)
584: Overloads__idivmod__ = HasMember("__idivmod__", DynamicType, 1)
585: Overloads__ipow__ = HasMember("__ipow__", DynamicType, 1)
586: Overloads__ilshift__ = HasMember("__ilshift__", DynamicType, 1)
587: Overloads__irshift__ = HasMember("__irshift__", DynamicType, 1)
588: Overloads__iand__ = HasMember("__iand__", DynamicType, 1)
589: Overloads__ior__ = HasMember("__ior__", DynamicType, 1)
590: Overloads__ixor__ = HasMember("__ixor__", DynamicType, 1)
591: 
592: # Class representation methods
593: Has__str__ = HasMember("__str__", str, 0)
594: Has__repr__ = HasMember("__repr__", str, 0)
595: Has__unicode__ = HasMember("__unicode__", unicode, 0)
596: Has__format__ = HasMember("__format__", str, 1)
597: Has__hash__ = HasMember("__hash__", int, 0)
598: Has__nonzero__ = HasMember("__nonzero__", bool, 0)
599: Has__dir__ = HasMember("__dir__", DynamicType, 0)
600: Has__sizeof__ = HasMember("__sizeof__", int, 0)
601: Has__call__ = Callable()  # HasMember("__call__", DynamicType, 0)
602: Has__mro__ = HasMember("__mro__", DynamicType, 0)
603: Has__class__ = HasMember("__class__", DynamicType, 0)
604: 
605: # Collections
606: # TODO: Check if this really need specific return types or they can be any
607: Has__len__ = HasMember("__len__", int, 0)
608: Has__getitem__ = HasMember("__getitem__", DynamicType, 1)
609: Has__setitem__ = HasMember("__setitem__", types.NoneType, 2)
610: # TODO: Really an int?
611: Has__delitem__ = HasMember("__delitem__", int, 0)
612: Has__iter__ = HasMember("__iter__", DynamicType, 0)
613: Has__reversed__ = HasMember("__reversed__", int, 0)
614: Has__contains__ = HasMember("__contains__", int, 0)
615: Has__missing__ = HasMember("__missing__", int, 0)
616: Has__getslice__ = HasMember("__getslice__", DynamicType, 2)
617: Has__setslice__ = HasMember("__setslice__", types.NoneType, 3)
618: Has__delslice__ = HasMember("__delslice__", types.NoneType, 2)
619: Has__next = HasMember("next", DynamicType, 0)
620: 
621: # Context managers
622: Has__enter__ = HasMember("__enter__", int, 0)
623: Has__exit__ = HasMember("__exit__", int, 3)
624: 
625: # Descriptor managers
626: Has__get__ = HasMember("__get__", DynamicType, 1)
627: Has__set__ = HasMember("__set__", types.NoneType, 2)
628: Has__del__ = HasMember("__del__", types.NoneType, 1)
629: 
630: 
631: # Copying
632: Has__copy__ = HasMember("__copy__", DynamicType, 0)
633: Has__deepcopy__ = HasMember("__deepcopy__", DynamicType, 1)
634: 
635: # Pickling
636: Has__getinitargs__ = HasMember("__getinitargs__", DynamicType, 0)
637: Has__getnewargs__ = HasMember("__getnewargs__", DynamicType, 0)
638: Has__getstate__ = HasMember("__getstate__", DynamicType, 0)
639: Has__setstate__ = HasMember("__setstate__", types.NoneType, 1)
640: Has__reduce__ = HasMember("__reduce__", DynamicType, 0)
641: Has__reduce_ex__ = HasMember("__reduce_ex__", DynamicType, 0)
642: 
643: # DynamicType instance
644: 
645: AnyType = DynamicType()
646: StructuralIntercessionType = SupportsStructuralIntercession()
647: 
648: # Other conditions
649: Hashable = IsHashable()
650: Type = TypeObject()
651: TypeInstance = InstanceOfType()
652: VarArgs = VarArgType()
653: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import types' statement (line 1)
import types

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'types', types, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import collections' statement (line 2)
import collections

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'collections', collections, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import ExtraTypeDefinitions' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')
import_18339 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy')

if (type(import_18339) is not StypyTypeError):

    if (import_18339 != 'pyd_module'):
        __import__(import_18339)
        sys_modules_18340 = sys.modules[import_18339]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', sys_modules_18340.module_type_store, module_type_store, ['ExtraTypeDefinitions'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_18340, sys_modules_18340.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import ExtraTypeDefinitions

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', None, module_type_store, ['ExtraTypeDefinitions'], [ExtraTypeDefinitions])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', import_18339)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from type_group_copy import TypeGroup' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')
import_18341 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'type_group_copy')

if (type(import_18341) is not StypyTypeError):

    if (import_18341 != 'pyd_module'):
        __import__(import_18341)
        sys_modules_18342 = sys.modules[import_18341]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'type_group_copy', sys_modules_18342.module_type_store, module_type_store, ['TypeGroup'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_18342, sys_modules_18342.module_type_store, module_type_store)
    else:
        from type_group_copy import TypeGroup

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'type_group_copy', None, module_type_store, ['TypeGroup'], [TypeGroup])

else:
    # Assigning a type to the variable 'type_group_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'type_group_copy', import_18341)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')
import_18343 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.errors_copy.type_error_copy')

if (type(import_18343) is not StypyTypeError):

    if (import_18343 != 'pyd_module'):
        __import__(import_18343)
        sys_modules_18344 = sys.modules[import_18343]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.errors_copy.type_error_copy', sys_modules_18344.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_18344, sys_modules_18344.module_type_store, module_type_store)
    else:
        from stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'stypy_copy.errors_copy.type_error_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.errors_copy.type_error_copy', import_18343)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')
import_18345 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_18345) is not StypyTypeError):

    if (import_18345 != 'pyd_module'):
        __import__(import_18345)
        sys_modules_18346 = sys.modules[import_18345]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', sys_modules_18346.module_type_store, module_type_store, ['union_type_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_18346, sys_modules_18346.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['union_type_copy'], [union_type_copy])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', import_18345)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')
import_18347 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy')

if (type(import_18347) is not StypyTypeError):

    if (import_18347 != 'pyd_module'):
        __import__(import_18347)
        sys_modules_18348 = sys.modules[import_18347]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', sys_modules_18348.module_type_store, module_type_store, ['UndefinedType'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_18348, sys_modules_18348.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', None, module_type_store, ['UndefinedType'], [UndefinedType])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', import_18347)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from stypy_copy.errors_copy.type_warning_copy import TypeWarning' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')
import_18349 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_copy.errors_copy.type_warning_copy')

if (type(import_18349) is not StypyTypeError):

    if (import_18349 != 'pyd_module'):
        __import__(import_18349)
        sys_modules_18350 = sys.modules[import_18349]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_copy.errors_copy.type_warning_copy', sys_modules_18350.module_type_store, module_type_store, ['TypeWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_18350, sys_modules_18350.module_type_store, module_type_store)
    else:
        from stypy_copy.errors_copy.type_warning_copy import TypeWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_copy.errors_copy.type_warning_copy', None, module_type_store, ['TypeWarning'], [TypeWarning])

else:
    # Assigning a type to the variable 'stypy_copy.errors_copy.type_warning_copy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_copy.errors_copy.type_warning_copy', import_18349)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import type_inference_proxy_management_copy' statement (line 10)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')
import_18351 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_18351) is not StypyTypeError):

    if (import_18351 != 'pyd_module'):
        __import__(import_18351)
        sys_modules_18352 = sys.modules[import_18351]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', sys_modules_18352.module_type_store, module_type_store, ['type_inference_proxy_management_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_18352, sys_modules_18352.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import type_inference_proxy_management_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['type_inference_proxy_management_copy'], [type_inference_proxy_management_copy])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', import_18351)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')

str_18353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, (-1)), 'str', '\nFile to define all type groups available to form type rules\n')
# Declaration of the 'DependentType' class

class DependentType:
    str_18354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, (-1)), 'str', "\n    A DependentType is a special base class that indicates that a type group has to be called to obtain the real\n    type it represent. Call is done using the parameters that are trying to match the rule. For example, imagine that\n    we call the + operator with an object that defines the __add__ method and another type to add to. With an object\n    that defines an __add__ method we don't really know what will be the result of calling __add__ over this object\n    with the second parameter, so the __add__ method has to be called (well, in fact, the type inference equivalent\n    version of the __add__ method will be called) to obtain the real return type.\n\n    Dependent types are a powerful mechanism to calculate the return type of operations that depend on calls to\n    certain object members or even to detect incorrect definitions of members in objects (__int__ method defined in\n    object that do not return int, for example).\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 31)
        False_18355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 37), 'False')
        defaults = [False_18355]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 31, 4, False)
        # Assigning a type to the variable 'self' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DependentType.__init__', ['report_errors'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['report_errors'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_18356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, (-1)), 'str', '\n        Build a Dependent type instance\n        :param report_errors: Flag to indicate if errors found when calling this type will be reported or not (in that\n        case other code will do it)\n        ')
        
        # Assigning a Name to a Attribute (line 37):
        
        # Assigning a Name to a Attribute (line 37):
        # Getting the type of 'report_errors' (line 37)
        report_errors_18357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 29), 'report_errors')
        # Getting the type of 'self' (line 37)
        self_18358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'self')
        # Setting the type of the member 'report_errors' of a type (line 37)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), self_18358, 'report_errors', report_errors_18357)
        
        # Assigning a Num to a Attribute (line 38):
        
        # Assigning a Num to a Attribute (line 38):
        int_18359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 26), 'int')
        # Getting the type of 'self' (line 38)
        self_18360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self')
        # Setting the type of the member 'call_arity' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_18360, 'call_arity', int_18359)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 40, 4, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DependentType.__call__.__dict__.__setitem__('stypy_localization', localization)
        DependentType.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DependentType.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        DependentType.__call__.__dict__.__setitem__('stypy_function_name', 'DependentType.__call__')
        DependentType.__call__.__dict__.__setitem__('stypy_param_names_list', [])
        DependentType.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'call_args')
        DependentType.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'call_kwargs')
        DependentType.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        DependentType.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        DependentType.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DependentType.__call__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DependentType.__call__', [], 'call_args', 'call_kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        str_18361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'str', '\n        Call the dependent type. Empty in this implementation, concrete calls must be defined in subclasses\n        ')
        pass
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 40)
        stypy_return_type_18362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18362)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_18362


# Assigning a type to the variable 'DependentType' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'DependentType', DependentType)
str_18363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, (-1)), 'str', '\nType groups with special meaning. All of them define a __eq__ method that indicates if the passed type matches with\nthe type group, storing this passed type. They also define a __call__ method that actually perform the type checking\nand calculate the return type. __eq__ and __call__ methods are called sequentially if __eq__ result is True, so the\nstorage of the passed type is safe to use in the __call__ as each time an __eq__ is called is replaced. This is the\nway the type rule checking mechanism works: TypeGroups are not meant to be used in other parts of the stypy runtime,\nand if they do, only the __eq__ method should be used to check if a type belongs to a group.\n')
# Declaration of the 'HasMember' class
# Getting the type of 'TypeGroup' (line 57)
TypeGroup_18364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'TypeGroup')
# Getting the type of 'DependentType' (line 57)
DependentType_18365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 27), 'DependentType')

class HasMember(TypeGroup_18364, DependentType_18365, ):
    str_18366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, (-1)), 'str', '\n        Type of any object that has a member with the specified arity, and that can be called with the corresponding\n        params.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_18367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 64), 'int')
        # Getting the type of 'False' (line 63)
        False_18368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 81), 'False')
        defaults = [int_18367, False_18368]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 63, 4, False)
        # Assigning a type to the variable 'self' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HasMember.__init__', ['member', 'expected_return_type', 'call_arity', 'report_errors'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['member', 'expected_return_type', 'call_arity', 'report_errors'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'self' (line 64)
        self_18371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 31), 'self', False)
        # Getting the type of 'report_errors' (line 64)
        report_errors_18372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 37), 'report_errors', False)
        # Processing the call keyword arguments (line 64)
        kwargs_18373 = {}
        # Getting the type of 'DependentType' (line 64)
        DependentType_18369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'DependentType', False)
        # Obtaining the member '__init__' of a type (line 64)
        init___18370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), DependentType_18369, '__init__')
        # Calling __init__(args, kwargs) (line 64)
        init___call_result_18374 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), init___18370, *[self_18371, report_errors_18372], **kwargs_18373)
        
        
        # Call to __init__(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'self' (line 65)
        self_18377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_18378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        
        # Processing the call keyword arguments (line 65)
        kwargs_18379 = {}
        # Getting the type of 'TypeGroup' (line 65)
        TypeGroup_18375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 65)
        init___18376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), TypeGroup_18375, '__init__')
        # Calling __init__(args, kwargs) (line 65)
        init___call_result_18380 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), init___18376, *[self_18377, list_18378], **kwargs_18379)
        
        
        # Assigning a Name to a Attribute (line 66):
        
        # Assigning a Name to a Attribute (line 66):
        # Getting the type of 'member' (line 66)
        member_18381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 22), 'member')
        # Getting the type of 'self' (line 66)
        self_18382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self')
        # Setting the type of the member 'member' of a type (line 66)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_18382, 'member', member_18381)
        
        # Assigning a Name to a Attribute (line 67):
        
        # Assigning a Name to a Attribute (line 67):
        # Getting the type of 'expected_return_type' (line 67)
        expected_return_type_18383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 36), 'expected_return_type')
        # Getting the type of 'self' (line 67)
        self_18384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self')
        # Setting the type of the member 'expected_return_type' of a type (line 67)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_18384, 'expected_return_type', expected_return_type_18383)
        
        # Assigning a Name to a Attribute (line 68):
        
        # Assigning a Name to a Attribute (line 68):
        # Getting the type of 'None' (line 68)
        None_18385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 26), 'None')
        # Getting the type of 'self' (line 68)
        self_18386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self')
        # Setting the type of the member 'member_obj' of a type (line 68)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_18386, 'member_obj', None_18385)
        
        # Assigning a Name to a Attribute (line 69):
        
        # Assigning a Name to a Attribute (line 69):
        # Getting the type of 'call_arity' (line 69)
        call_arity_18387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 26), 'call_arity')
        # Getting the type of 'self' (line 69)
        self_18388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'self')
        # Setting the type of the member 'call_arity' of a type (line 69)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), self_18388, 'call_arity', call_arity_18387)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def format_arity(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'format_arity'
        module_type_store = module_type_store.open_function_context('format_arity', 71, 4, False)
        # Assigning a type to the variable 'self' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HasMember.format_arity.__dict__.__setitem__('stypy_localization', localization)
        HasMember.format_arity.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HasMember.format_arity.__dict__.__setitem__('stypy_type_store', module_type_store)
        HasMember.format_arity.__dict__.__setitem__('stypy_function_name', 'HasMember.format_arity')
        HasMember.format_arity.__dict__.__setitem__('stypy_param_names_list', [])
        HasMember.format_arity.__dict__.__setitem__('stypy_varargs_param_name', None)
        HasMember.format_arity.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HasMember.format_arity.__dict__.__setitem__('stypy_call_defaults', defaults)
        HasMember.format_arity.__dict__.__setitem__('stypy_call_varargs', varargs)
        HasMember.format_arity.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HasMember.format_arity.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HasMember.format_arity', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'format_arity', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'format_arity(...)' code ##################

        
        # Assigning a Str to a Name (line 72):
        
        # Assigning a Str to a Name (line 72):
        str_18389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 15), 'str', '(')
        # Assigning a type to the variable 'str_' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'str_', str_18389)
        
        
        # Call to range(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'self' (line 73)
        self_18391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'self', False)
        # Obtaining the member 'call_arity' of a type (line 73)
        call_arity_18392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), self_18391, 'call_arity')
        # Processing the call keyword arguments (line 73)
        kwargs_18393 = {}
        # Getting the type of 'range' (line 73)
        range_18390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 17), 'range', False)
        # Calling range(args, kwargs) (line 73)
        range_call_result_18394 = invoke(stypy.reporting.localization.Localization(__file__, 73, 17), range_18390, *[call_arity_18392], **kwargs_18393)
        
        # Assigning a type to the variable 'range_call_result_18394' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'range_call_result_18394', range_call_result_18394)
        # Testing if the for loop is going to be iterated (line 73)
        # Testing the type of a for loop iterable (line 73)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 73, 8), range_call_result_18394)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 73, 8), range_call_result_18394):
            # Getting the type of the for loop variable (line 73)
            for_loop_var_18395 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 73, 8), range_call_result_18394)
            # Assigning a type to the variable 'i' (line 73)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'i', for_loop_var_18395)
            # SSA begins for a for statement (line 73)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'str_' (line 74)
            str__18396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'str_')
            str_18397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 20), 'str', 'parameter')
            
            # Call to str(...): (line 74)
            # Processing the call arguments (line 74)
            # Getting the type of 'i' (line 74)
            i_18399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 38), 'i', False)
            # Processing the call keyword arguments (line 74)
            kwargs_18400 = {}
            # Getting the type of 'str' (line 74)
            str_18398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 34), 'str', False)
            # Calling str(args, kwargs) (line 74)
            str_call_result_18401 = invoke(stypy.reporting.localization.Localization(__file__, 74, 34), str_18398, *[i_18399], **kwargs_18400)
            
            # Applying the binary operator '+' (line 74)
            result_add_18402 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 20), '+', str_18397, str_call_result_18401)
            
            str_18403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 43), 'str', ', ')
            # Applying the binary operator '+' (line 74)
            result_add_18404 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 41), '+', result_add_18402, str_18403)
            
            # Applying the binary operator '+=' (line 74)
            result_iadd_18405 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 12), '+=', str__18396, result_add_18404)
            # Assigning a type to the variable 'str_' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'str_', result_iadd_18405)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 'self' (line 76)
        self_18406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'self')
        # Obtaining the member 'call_arity' of a type (line 76)
        call_arity_18407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 11), self_18406, 'call_arity')
        int_18408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 29), 'int')
        # Applying the binary operator '>' (line 76)
        result_gt_18409 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 11), '>', call_arity_18407, int_18408)
        
        # Testing if the type of an if condition is none (line 76)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 8), result_gt_18409):
            pass
        else:
            
            # Testing the type of an if condition (line 76)
            if_condition_18410 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 8), result_gt_18409)
            # Assigning a type to the variable 'if_condition_18410' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'if_condition_18410', if_condition_18410)
            # SSA begins for if statement (line 76)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 77):
            
            # Assigning a Subscript to a Name (line 77):
            
            # Obtaining the type of the subscript
            int_18411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 25), 'int')
            slice_18412 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 77, 19), None, int_18411, None)
            # Getting the type of 'str_' (line 77)
            str__18413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'str_')
            # Obtaining the member '__getitem__' of a type (line 77)
            getitem___18414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 19), str__18413, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 77)
            subscript_call_result_18415 = invoke(stypy.reporting.localization.Localization(__file__, 77, 19), getitem___18414, slice_18412)
            
            # Assigning a type to the variable 'str_' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'str_', subscript_call_result_18415)
            # SSA join for if statement (line 76)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'str_' (line 79)
        str__18416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'str_')
        str_18417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 22), 'str', ')')
        # Applying the binary operator '+' (line 79)
        result_add_18418 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 15), '+', str__18416, str_18417)
        
        # Assigning a type to the variable 'stypy_return_type' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'stypy_return_type', result_add_18418)
        
        # ################# End of 'format_arity(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'format_arity' in the type store
        # Getting the type of 'stypy_return_type' (line 71)
        stypy_return_type_18419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18419)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'format_arity'
        return stypy_return_type_18419


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 81, 4, False)
        # Assigning a type to the variable 'self' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HasMember.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        HasMember.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HasMember.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        HasMember.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'HasMember.stypy__eq__')
        HasMember.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['type_'])
        HasMember.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        HasMember.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HasMember.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        HasMember.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        HasMember.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HasMember.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HasMember.stypy__eq__', ['type_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['type_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 82):
        
        # Assigning a Call to a Attribute (line 82):
        
        # Call to get_type_of_member(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'None' (line 82)
        None_18422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 51), 'None', False)
        # Getting the type of 'self' (line 82)
        self_18423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 57), 'self', False)
        # Obtaining the member 'member' of a type (line 82)
        member_18424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 57), self_18423, 'member')
        # Processing the call keyword arguments (line 82)
        kwargs_18425 = {}
        # Getting the type of 'type_' (line 82)
        type__18420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'type_', False)
        # Obtaining the member 'get_type_of_member' of a type (line 82)
        get_type_of_member_18421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 26), type__18420, 'get_type_of_member')
        # Calling get_type_of_member(args, kwargs) (line 82)
        get_type_of_member_call_result_18426 = invoke(stypy.reporting.localization.Localization(__file__, 82, 26), get_type_of_member_18421, *[None_18422, member_18424], **kwargs_18425)
        
        # Getting the type of 'self' (line 82)
        self_18427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'self')
        # Setting the type of the member 'member_obj' of a type (line 82)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), self_18427, 'member_obj', get_type_of_member_call_result_18426)
        
        # Type idiom detected: calculating its left and rigth part (line 83)
        # Getting the type of 'TypeError' (line 83)
        TypeError_18428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 39), 'TypeError')
        # Getting the type of 'self' (line 83)
        self_18429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 22), 'self')
        # Obtaining the member 'member_obj' of a type (line 83)
        member_obj_18430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 22), self_18429, 'member_obj')
        
        (may_be_18431, more_types_in_union_18432) = may_be_subtype(TypeError_18428, member_obj_18430)

        if may_be_18431:

            if more_types_in_union_18432:
                # Runtime conditional SSA (line 83)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 83)
            self_18433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'self')
            # Obtaining the member 'member_obj' of a type (line 83)
            member_obj_18434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), self_18433, 'member_obj')
            # Setting the type of the member 'member_obj' of a type (line 83)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), self_18433, 'member_obj', remove_not_subtype_from_union(member_obj_18430, TypeError))
            
            # Getting the type of 'self' (line 84)
            self_18435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 19), 'self')
            # Obtaining the member 'report_errors' of a type (line 84)
            report_errors_18436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 19), self_18435, 'report_errors')
            # Applying the 'not' unary operator (line 84)
            result_not__18437 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 15), 'not', report_errors_18436)
            
            # Testing if the type of an if condition is none (line 84)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 84, 12), result_not__18437):
                pass
            else:
                
                # Testing the type of an if condition (line 84)
                if_condition_18438 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 12), result_not__18437)
                # Assigning a type to the variable 'if_condition_18438' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'if_condition_18438', if_condition_18438)
                # SSA begins for if statement (line 84)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to remove_error_msg(...): (line 85)
                # Processing the call arguments (line 85)
                # Getting the type of 'self' (line 85)
                self_18441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 43), 'self', False)
                # Obtaining the member 'member_obj' of a type (line 85)
                member_obj_18442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 43), self_18441, 'member_obj')
                # Processing the call keyword arguments (line 85)
                kwargs_18443 = {}
                # Getting the type of 'TypeError' (line 85)
                TypeError_18439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'TypeError', False)
                # Obtaining the member 'remove_error_msg' of a type (line 85)
                remove_error_msg_18440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 16), TypeError_18439, 'remove_error_msg')
                # Calling remove_error_msg(args, kwargs) (line 85)
                remove_error_msg_call_result_18444 = invoke(stypy.reporting.localization.Localization(__file__, 85, 16), remove_error_msg_18440, *[member_obj_18442], **kwargs_18443)
                
                # SSA join for if statement (line 84)
                module_type_store = module_type_store.join_ssa_context()
                

            # Getting the type of 'False' (line 86)
            False_18445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'stypy_return_type', False_18445)

            if more_types_in_union_18432:
                # SSA join for if statement (line 83)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'True' (line 88)
        True_18446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'stypy_return_type', True_18446)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_18447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18447)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_18447


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 90, 4, False)
        # Assigning a type to the variable 'self' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HasMember.__call__.__dict__.__setitem__('stypy_localization', localization)
        HasMember.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HasMember.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        HasMember.__call__.__dict__.__setitem__('stypy_function_name', 'HasMember.__call__')
        HasMember.__call__.__dict__.__setitem__('stypy_param_names_list', ['localization'])
        HasMember.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'call_args')
        HasMember.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'call_kwargs')
        HasMember.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        HasMember.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        HasMember.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HasMember.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HasMember.__call__', ['localization'], 'call_args', 'call_kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['localization'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Call to callable(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Call to get_python_type(...): (line 91)
        # Processing the call keyword arguments (line 91)
        kwargs_18452 = {}
        # Getting the type of 'self' (line 91)
        self_18449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'self', False)
        # Obtaining the member 'member_obj' of a type (line 91)
        member_obj_18450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 20), self_18449, 'member_obj')
        # Obtaining the member 'get_python_type' of a type (line 91)
        get_python_type_18451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 20), member_obj_18450, 'get_python_type')
        # Calling get_python_type(args, kwargs) (line 91)
        get_python_type_call_result_18453 = invoke(stypy.reporting.localization.Localization(__file__, 91, 20), get_python_type_18451, *[], **kwargs_18452)
        
        # Processing the call keyword arguments (line 91)
        kwargs_18454 = {}
        # Getting the type of 'callable' (line 91)
        callable_18448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 11), 'callable', False)
        # Calling callable(args, kwargs) (line 91)
        callable_call_result_18455 = invoke(stypy.reporting.localization.Localization(__file__, 91, 11), callable_18448, *[get_python_type_call_result_18453], **kwargs_18454)
        
        # Testing if the type of an if condition is none (line 91)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 91, 8), callable_call_result_18455):
            pass
        else:
            
            # Testing the type of an if condition (line 91)
            if_condition_18456 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 8), callable_call_result_18455)
            # Assigning a type to the variable 'if_condition_18456' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'if_condition_18456', if_condition_18456)
            # SSA begins for if statement (line 91)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 93):
            
            # Assigning a Call to a Name (line 93):
            
            # Call to invoke(...): (line 93)
            # Processing the call arguments (line 93)
            # Getting the type of 'localization' (line 93)
            localization_18460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 53), 'localization', False)
            # Getting the type of 'call_args' (line 93)
            call_args_18461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 68), 'call_args', False)
            # Processing the call keyword arguments (line 93)
            # Getting the type of 'call_kwargs' (line 93)
            call_kwargs_18462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 81), 'call_kwargs', False)
            kwargs_18463 = {'call_kwargs_18462': call_kwargs_18462}
            # Getting the type of 'self' (line 93)
            self_18457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 30), 'self', False)
            # Obtaining the member 'member_obj' of a type (line 93)
            member_obj_18458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 30), self_18457, 'member_obj')
            # Obtaining the member 'invoke' of a type (line 93)
            invoke_18459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 30), member_obj_18458, 'invoke')
            # Calling invoke(args, kwargs) (line 93)
            invoke_call_result_18464 = invoke(stypy.reporting.localization.Localization(__file__, 93, 30), invoke_18459, *[localization_18460, call_args_18461], **kwargs_18463)
            
            # Assigning a type to the variable 'equivalent_type' (line 93)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'equivalent_type', invoke_call_result_18464)
            
            # Type idiom detected: calculating its left and rigth part (line 96)
            # Getting the type of 'TypeError' (line 96)
            TypeError_18465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 43), 'TypeError')
            # Getting the type of 'equivalent_type' (line 96)
            equivalent_type_18466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'equivalent_type')
            
            (may_be_18467, more_types_in_union_18468) = may_be_subtype(TypeError_18465, equivalent_type_18466)

            if may_be_18467:

                if more_types_in_union_18468:
                    # Runtime conditional SSA (line 96)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'equivalent_type' (line 96)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'equivalent_type', remove_not_subtype_from_union(equivalent_type_18466, TypeError))
                
                # Getting the type of 'self' (line 97)
                self_18469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 23), 'self')
                # Obtaining the member 'report_errors' of a type (line 97)
                report_errors_18470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 23), self_18469, 'report_errors')
                # Applying the 'not' unary operator (line 97)
                result_not__18471 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 19), 'not', report_errors_18470)
                
                # Testing if the type of an if condition is none (line 97)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 97, 16), result_not__18471):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 97)
                    if_condition_18472 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 16), result_not__18471)
                    # Assigning a type to the variable 'if_condition_18472' (line 97)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'if_condition_18472', if_condition_18472)
                    # SSA begins for if statement (line 97)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to remove_error_msg(...): (line 98)
                    # Processing the call arguments (line 98)
                    # Getting the type of 'equivalent_type' (line 98)
                    equivalent_type_18475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 47), 'equivalent_type', False)
                    # Processing the call keyword arguments (line 98)
                    kwargs_18476 = {}
                    # Getting the type of 'TypeError' (line 98)
                    TypeError_18473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'TypeError', False)
                    # Obtaining the member 'remove_error_msg' of a type (line 98)
                    remove_error_msg_18474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 20), TypeError_18473, 'remove_error_msg')
                    # Calling remove_error_msg(args, kwargs) (line 98)
                    remove_error_msg_call_result_18477 = invoke(stypy.reporting.localization.Localization(__file__, 98, 20), remove_error_msg_18474, *[equivalent_type_18475], **kwargs_18476)
                    
                    # SSA join for if statement (line 97)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Name to a Attribute (line 99):
                
                # Assigning a Name to a Attribute (line 99):
                # Getting the type of 'None' (line 99)
                None_18478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 34), 'None')
                # Getting the type of 'self' (line 99)
                self_18479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'self')
                # Setting the type of the member 'member_obj' of a type (line 99)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), self_18479, 'member_obj', None_18478)
                
                # Obtaining an instance of the builtin type 'tuple' (line 100)
                tuple_18480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 100)
                # Adding element type (line 100)
                # Getting the type of 'False' (line 100)
                False_18481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 23), 'False')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 23), tuple_18480, False_18481)
                # Adding element type (line 100)
                # Getting the type of 'equivalent_type' (line 100)
                equivalent_type_18482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 30), 'equivalent_type')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 23), tuple_18480, equivalent_type_18482)
                
                # Assigning a type to the variable 'stypy_return_type' (line 100)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'stypy_return_type', tuple_18480)

                if more_types_in_union_18468:
                    # SSA join for if statement (line 96)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Call to isinstance(...): (line 103)
            # Processing the call arguments (line 103)
            # Getting the type of 'self' (line 103)
            self_18484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'self', False)
            # Obtaining the member 'expected_return_type' of a type (line 103)
            expected_return_type_18485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 26), self_18484, 'expected_return_type')
            # Getting the type of 'UndefinedType' (line 103)
            UndefinedType_18486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 53), 'UndefinedType', False)
            # Processing the call keyword arguments (line 103)
            kwargs_18487 = {}
            # Getting the type of 'isinstance' (line 103)
            isinstance_18483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 103)
            isinstance_call_result_18488 = invoke(stypy.reporting.localization.Localization(__file__, 103, 15), isinstance_18483, *[expected_return_type_18485, UndefinedType_18486], **kwargs_18487)
            
            # Testing if the type of an if condition is none (line 103)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 103, 12), isinstance_call_result_18488):
                pass
            else:
                
                # Testing the type of an if condition (line 103)
                if_condition_18489 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 12), isinstance_call_result_18488)
                # Assigning a type to the variable 'if_condition_18489' (line 103)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'if_condition_18489', if_condition_18489)
                # SSA begins for if statement (line 103)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Attribute (line 104):
                
                # Assigning a Name to a Attribute (line 104):
                # Getting the type of 'None' (line 104)
                None_18490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 34), 'None')
                # Getting the type of 'self' (line 104)
                self_18491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'self')
                # Setting the type of the member 'member_obj' of a type (line 104)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 16), self_18491, 'member_obj', None_18490)
                
                # Obtaining an instance of the builtin type 'tuple' (line 105)
                tuple_18492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 105)
                # Adding element type (line 105)
                # Getting the type of 'True' (line 105)
                True_18493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 23), 'True')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 23), tuple_18492, True_18493)
                # Adding element type (line 105)
                # Getting the type of 'equivalent_type' (line 105)
                equivalent_type_18494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 29), 'equivalent_type')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 23), tuple_18492, equivalent_type_18494)
                
                # Assigning a type to the variable 'stypy_return_type' (line 105)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'stypy_return_type', tuple_18492)
                # SSA join for if statement (line 103)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'self' (line 108)
            self_18495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'self')
            # Obtaining the member 'expected_return_type' of a type (line 108)
            expected_return_type_18496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 15), self_18495, 'expected_return_type')
            # Getting the type of 'DynamicType' (line 108)
            DynamicType_18497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 44), 'DynamicType')
            # Applying the binary operator 'is' (line 108)
            result_is__18498 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 15), 'is', expected_return_type_18496, DynamicType_18497)
            
            # Testing if the type of an if condition is none (line 108)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 12), result_is__18498):
                pass
            else:
                
                # Testing the type of an if condition (line 108)
                if_condition_18499 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 12), result_is__18498)
                # Assigning a type to the variable 'if_condition_18499' (line 108)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'if_condition_18499', if_condition_18499)
                # SSA begins for if statement (line 108)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Attribute (line 109):
                
                # Assigning a Name to a Attribute (line 109):
                # Getting the type of 'None' (line 109)
                None_18500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 34), 'None')
                # Getting the type of 'self' (line 109)
                self_18501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'self')
                # Setting the type of the member 'member_obj' of a type (line 109)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 16), self_18501, 'member_obj', None_18500)
                
                # Obtaining an instance of the builtin type 'tuple' (line 110)
                tuple_18502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 110)
                # Adding element type (line 110)
                # Getting the type of 'True' (line 110)
                True_18503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 23), 'True')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 23), tuple_18502, True_18503)
                # Adding element type (line 110)
                # Getting the type of 'equivalent_type' (line 110)
                equivalent_type_18504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 29), 'equivalent_type')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 23), tuple_18502, equivalent_type_18504)
                
                # Assigning a type to the variable 'stypy_return_type' (line 110)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'stypy_return_type', tuple_18502)
                # SSA join for if statement (line 108)
                module_type_store = module_type_store.join_ssa_context()
                

            
            
            # Call to issubclass(...): (line 113)
            # Processing the call arguments (line 113)
            
            # Call to get_python_type(...): (line 113)
            # Processing the call keyword arguments (line 113)
            kwargs_18508 = {}
            # Getting the type of 'equivalent_type' (line 113)
            equivalent_type_18506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 30), 'equivalent_type', False)
            # Obtaining the member 'get_python_type' of a type (line 113)
            get_python_type_18507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 30), equivalent_type_18506, 'get_python_type')
            # Calling get_python_type(args, kwargs) (line 113)
            get_python_type_call_result_18509 = invoke(stypy.reporting.localization.Localization(__file__, 113, 30), get_python_type_18507, *[], **kwargs_18508)
            
            # Getting the type of 'self' (line 113)
            self_18510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 65), 'self', False)
            # Obtaining the member 'expected_return_type' of a type (line 113)
            expected_return_type_18511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 65), self_18510, 'expected_return_type')
            # Processing the call keyword arguments (line 113)
            kwargs_18512 = {}
            # Getting the type of 'issubclass' (line 113)
            issubclass_18505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'issubclass', False)
            # Calling issubclass(args, kwargs) (line 113)
            issubclass_call_result_18513 = invoke(stypy.reporting.localization.Localization(__file__, 113, 19), issubclass_18505, *[get_python_type_call_result_18509, expected_return_type_18511], **kwargs_18512)
            
            # Applying the 'not' unary operator (line 113)
            result_not__18514 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 15), 'not', issubclass_call_result_18513)
            
            # Testing if the type of an if condition is none (line 113)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 113, 12), result_not__18514):
                
                # Obtaining an instance of the builtin type 'tuple' (line 117)
                tuple_18521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 117)
                # Adding element type (line 117)
                # Getting the type of 'True' (line 117)
                True_18522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 23), 'True')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 23), tuple_18521, True_18522)
                # Adding element type (line 117)
                # Getting the type of 'equivalent_type' (line 117)
                equivalent_type_18523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 29), 'equivalent_type')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 23), tuple_18521, equivalent_type_18523)
                
                # Assigning a type to the variable 'stypy_return_type' (line 117)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'stypy_return_type', tuple_18521)
            else:
                
                # Testing the type of an if condition (line 113)
                if_condition_18515 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 12), result_not__18514)
                # Assigning a type to the variable 'if_condition_18515' (line 113)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'if_condition_18515', if_condition_18515)
                # SSA begins for if statement (line 113)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Attribute (line 114):
                
                # Assigning a Name to a Attribute (line 114):
                # Getting the type of 'None' (line 114)
                None_18516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 34), 'None')
                # Getting the type of 'self' (line 114)
                self_18517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'self')
                # Setting the type of the member 'member_obj' of a type (line 114)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 16), self_18517, 'member_obj', None_18516)
                
                # Obtaining an instance of the builtin type 'tuple' (line 115)
                tuple_18518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 115)
                # Adding element type (line 115)
                # Getting the type of 'False' (line 115)
                False_18519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 23), 'False')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 23), tuple_18518, False_18519)
                # Adding element type (line 115)
                # Getting the type of 'equivalent_type' (line 115)
                equivalent_type_18520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 30), 'equivalent_type')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 23), tuple_18518, equivalent_type_18520)
                
                # Assigning a type to the variable 'stypy_return_type' (line 115)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'stypy_return_type', tuple_18518)
                # SSA branch for the else part of an if statement (line 113)
                module_type_store.open_ssa_branch('else')
                
                # Obtaining an instance of the builtin type 'tuple' (line 117)
                tuple_18521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 117)
                # Adding element type (line 117)
                # Getting the type of 'True' (line 117)
                True_18522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 23), 'True')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 23), tuple_18521, True_18522)
                # Adding element type (line 117)
                # Getting the type of 'equivalent_type' (line 117)
                equivalent_type_18523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 29), 'equivalent_type')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 23), tuple_18521, equivalent_type_18523)
                
                # Assigning a type to the variable 'stypy_return_type' (line 117)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'stypy_return_type', tuple_18521)
                # SSA join for if statement (line 113)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 91)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Name to a Attribute (line 119):
        
        # Assigning a Name to a Attribute (line 119):
        # Getting the type of 'None' (line 119)
        None_18524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 26), 'None')
        # Getting the type of 'self' (line 119)
        self_18525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'self')
        # Setting the type of the member 'member_obj' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), self_18525, 'member_obj', None_18524)
        
        # Obtaining an instance of the builtin type 'tuple' (line 120)
        tuple_18526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 120)
        # Adding element type (line 120)
        # Getting the type of 'True' (line 120)
        True_18527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 15), tuple_18526, True_18527)
        # Adding element type (line 120)
        # Getting the type of 'None' (line 120)
        None_18528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 21), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 15), tuple_18526, None_18528)
        
        # Assigning a type to the variable 'stypy_return_type' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'stypy_return_type', tuple_18526)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 90)
        stypy_return_type_18529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18529)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_18529


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 122, 4, False)
        # Assigning a type to the variable 'self' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HasMember.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        HasMember.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HasMember.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        HasMember.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'HasMember.stypy__repr__')
        HasMember.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        HasMember.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        HasMember.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HasMember.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        HasMember.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        HasMember.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HasMember.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HasMember.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Name (line 123):
        
        # Assigning a Str to a Name (line 123):
        str_18530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 18), 'str', 'Instance defining ')
        # Assigning a type to the variable 'ret_str' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'ret_str', str_18530)
        
        # Getting the type of 'ret_str' (line 124)
        ret_str_18531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'ret_str')
        
        # Call to str(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'self' (line 124)
        self_18533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 23), 'self', False)
        # Obtaining the member 'member' of a type (line 124)
        member_18534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 23), self_18533, 'member')
        # Processing the call keyword arguments (line 124)
        kwargs_18535 = {}
        # Getting the type of 'str' (line 124)
        str_18532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'str', False)
        # Calling str(args, kwargs) (line 124)
        str_call_result_18536 = invoke(stypy.reporting.localization.Localization(__file__, 124, 19), str_18532, *[member_18534], **kwargs_18535)
        
        # Applying the binary operator '+=' (line 124)
        result_iadd_18537 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 8), '+=', ret_str_18531, str_call_result_18536)
        # Assigning a type to the variable 'ret_str' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'ret_str', result_iadd_18537)
        
        
        # Getting the type of 'ret_str' (line 125)
        ret_str_18538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'ret_str')
        
        # Call to format_arity(...): (line 125)
        # Processing the call keyword arguments (line 125)
        kwargs_18541 = {}
        # Getting the type of 'self' (line 125)
        self_18539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 19), 'self', False)
        # Obtaining the member 'format_arity' of a type (line 125)
        format_arity_18540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 19), self_18539, 'format_arity')
        # Calling format_arity(args, kwargs) (line 125)
        format_arity_call_result_18542 = invoke(stypy.reporting.localization.Localization(__file__, 125, 19), format_arity_18540, *[], **kwargs_18541)
        
        # Applying the binary operator '+=' (line 125)
        result_iadd_18543 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 8), '+=', ret_str_18538, format_arity_call_result_18542)
        # Assigning a type to the variable 'ret_str' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'ret_str', result_iadd_18543)
        
        # Getting the type of 'ret_str' (line 126)
        ret_str_18544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 15), 'ret_str')
        # Assigning a type to the variable 'stypy_return_type' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'stypy_return_type', ret_str_18544)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 122)
        stypy_return_type_18545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18545)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_18545


# Assigning a type to the variable 'HasMember' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'HasMember', HasMember)
# Declaration of the 'IterableDataStructureWithTypedElements' class
# Getting the type of 'TypeGroup' (line 129)
TypeGroup_18546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 45), 'TypeGroup')
# Getting the type of 'DependentType' (line 129)
DependentType_18547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 56), 'DependentType')

class IterableDataStructureWithTypedElements(TypeGroup_18546, DependentType_18547, ):
    str_18548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, (-1)), 'str', '\n    Represent all iterable data structures that contain a certain type or types\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 134, 4, False)
        # Assigning a type to the variable 'self' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IterableDataStructureWithTypedElements.__init__', [], 'content_types', None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'self' (line 135)
        self_18551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 31), 'self', False)
        # Getting the type of 'True' (line 135)
        True_18552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 37), 'True', False)
        # Processing the call keyword arguments (line 135)
        kwargs_18553 = {}
        # Getting the type of 'DependentType' (line 135)
        DependentType_18549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'DependentType', False)
        # Obtaining the member '__init__' of a type (line 135)
        init___18550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), DependentType_18549, '__init__')
        # Calling __init__(args, kwargs) (line 135)
        init___call_result_18554 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), init___18550, *[self_18551, True_18552], **kwargs_18553)
        
        
        # Call to __init__(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'self' (line 136)
        self_18557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 136)
        list_18558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 136)
        
        # Processing the call keyword arguments (line 136)
        kwargs_18559 = {}
        # Getting the type of 'TypeGroup' (line 136)
        TypeGroup_18555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 136)
        init___18556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), TypeGroup_18555, '__init__')
        # Calling __init__(args, kwargs) (line 136)
        init___call_result_18560 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), init___18556, *[self_18557, list_18558], **kwargs_18559)
        
        
        # Assigning a Name to a Attribute (line 137):
        
        # Assigning a Name to a Attribute (line 137):
        # Getting the type of 'content_types' (line 137)
        content_types_18561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 29), 'content_types')
        # Getting the type of 'self' (line 137)
        self_18562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'self')
        # Setting the type of the member 'content_types' of a type (line 137)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), self_18562, 'content_types', content_types_18561)
        
        # Assigning a Name to a Attribute (line 138):
        
        # Assigning a Name to a Attribute (line 138):
        # Getting the type of 'None' (line 138)
        None_18563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 21), 'None')
        # Getting the type of 'self' (line 138)
        self_18564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'self')
        # Setting the type of the member 'type_' of a type (line 138)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), self_18564, 'type_', None_18563)
        
        # Assigning a Num to a Attribute (line 139):
        
        # Assigning a Num to a Attribute (line 139):
        int_18565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 26), 'int')
        # Getting the type of 'self' (line 139)
        self_18566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'self')
        # Setting the type of the member 'call_arity' of a type (line 139)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), self_18566, 'call_arity', int_18565)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 141, 4, False)
        # Assigning a type to the variable 'self' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IterableDataStructureWithTypedElements.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        IterableDataStructureWithTypedElements.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IterableDataStructureWithTypedElements.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        IterableDataStructureWithTypedElements.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'IterableDataStructureWithTypedElements.stypy__eq__')
        IterableDataStructureWithTypedElements.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['type_'])
        IterableDataStructureWithTypedElements.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        IterableDataStructureWithTypedElements.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IterableDataStructureWithTypedElements.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        IterableDataStructureWithTypedElements.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        IterableDataStructureWithTypedElements.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IterableDataStructureWithTypedElements.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IterableDataStructureWithTypedElements.stypy__eq__', ['type_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['type_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 142):
        
        # Assigning a Name to a Attribute (line 142):
        # Getting the type of 'type_' (line 142)
        type__18567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 21), 'type_')
        # Getting the type of 'self' (line 142)
        self_18568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'self')
        # Setting the type of the member 'type_' of a type (line 142)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), self_18568, 'type_', type__18567)
        
        
        # Call to get_python_type(...): (line 143)
        # Processing the call keyword arguments (line 143)
        kwargs_18571 = {}
        # Getting the type of 'type_' (line 143)
        type__18569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 15), 'type_', False)
        # Obtaining the member 'get_python_type' of a type (line 143)
        get_python_type_18570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 15), type__18569, 'get_python_type')
        # Calling get_python_type(args, kwargs) (line 143)
        get_python_type_call_result_18572 = invoke(stypy.reporting.localization.Localization(__file__, 143, 15), get_python_type_18570, *[], **kwargs_18571)
        
        # Getting the type of 'TypeGroups' (line 143)
        TypeGroups_18573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 42), 'TypeGroups')
        # Obtaining the member 'IterableDataStructure' of a type (line 143)
        IterableDataStructure_18574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 42), TypeGroups_18573, 'IterableDataStructure')
        # Applying the binary operator 'in' (line 143)
        result_contains_18575 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 15), 'in', get_python_type_call_result_18572, IterableDataStructure_18574)
        
        # Assigning a type to the variable 'stypy_return_type' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'stypy_return_type', result_contains_18575)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 141)
        stypy_return_type_18576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18576)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_18576


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 145, 4, False)
        # Assigning a type to the variable 'self' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IterableDataStructureWithTypedElements.__call__.__dict__.__setitem__('stypy_localization', localization)
        IterableDataStructureWithTypedElements.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IterableDataStructureWithTypedElements.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        IterableDataStructureWithTypedElements.__call__.__dict__.__setitem__('stypy_function_name', 'IterableDataStructureWithTypedElements.__call__')
        IterableDataStructureWithTypedElements.__call__.__dict__.__setitem__('stypy_param_names_list', ['localization'])
        IterableDataStructureWithTypedElements.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'call_args')
        IterableDataStructureWithTypedElements.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'call_kwargs')
        IterableDataStructureWithTypedElements.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        IterableDataStructureWithTypedElements.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        IterableDataStructureWithTypedElements.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IterableDataStructureWithTypedElements.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IterableDataStructureWithTypedElements.__call__', ['localization'], 'call_args', 'call_kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['localization'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Assigning a Call to a Name (line 146):
        
        # Assigning a Call to a Name (line 146):
        
        # Call to get_elements_type(...): (line 146)
        # Processing the call keyword arguments (line 146)
        kwargs_18580 = {}
        # Getting the type of 'self' (line 146)
        self_18577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 29), 'self', False)
        # Obtaining the member 'type_' of a type (line 146)
        type__18578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 29), self_18577, 'type_')
        # Obtaining the member 'get_elements_type' of a type (line 146)
        get_elements_type_18579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 29), type__18578, 'get_elements_type')
        # Calling get_elements_type(args, kwargs) (line 146)
        get_elements_type_call_result_18581 = invoke(stypy.reporting.localization.Localization(__file__, 146, 29), get_elements_type_18579, *[], **kwargs_18580)
        
        # Assigning a type to the variable 'contained_elements' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'contained_elements', get_elements_type_call_result_18581)
        
        # Call to isinstance(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'contained_elements' (line 147)
        contained_elements_18583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 22), 'contained_elements', False)
        # Getting the type of 'union_type_copy' (line 147)
        union_type_copy_18584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 42), 'union_type_copy', False)
        # Obtaining the member 'UnionType' of a type (line 147)
        UnionType_18585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 42), union_type_copy_18584, 'UnionType')
        # Processing the call keyword arguments (line 147)
        kwargs_18586 = {}
        # Getting the type of 'isinstance' (line 147)
        isinstance_18582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 147)
        isinstance_call_result_18587 = invoke(stypy.reporting.localization.Localization(__file__, 147, 11), isinstance_18582, *[contained_elements_18583, UnionType_18585], **kwargs_18586)
        
        # Testing if the type of an if condition is none (line 147)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 147, 8), isinstance_call_result_18587):
            
            # Assigning a List to a Name (line 150):
            
            # Assigning a List to a Name (line 150):
            
            # Obtaining an instance of the builtin type 'list' (line 150)
            list_18591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 31), 'list')
            # Adding type elements to the builtin type 'list' instance (line 150)
            # Adding element type (line 150)
            # Getting the type of 'contained_elements' (line 150)
            contained_elements_18592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 32), 'contained_elements')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 31), list_18591, contained_elements_18592)
            
            # Assigning a type to the variable 'types_to_examine' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'types_to_examine', list_18591)
        else:
            
            # Testing the type of an if condition (line 147)
            if_condition_18588 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 8), isinstance_call_result_18587)
            # Assigning a type to the variable 'if_condition_18588' (line 147)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'if_condition_18588', if_condition_18588)
            # SSA begins for if statement (line 147)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 148):
            
            # Assigning a Attribute to a Name (line 148):
            # Getting the type of 'contained_elements' (line 148)
            contained_elements_18589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 31), 'contained_elements')
            # Obtaining the member 'types' of a type (line 148)
            types_18590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 31), contained_elements_18589, 'types')
            # Assigning a type to the variable 'types_to_examine' (line 148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'types_to_examine', types_18590)
            # SSA branch for the else part of an if statement (line 147)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a List to a Name (line 150):
            
            # Assigning a List to a Name (line 150):
            
            # Obtaining an instance of the builtin type 'list' (line 150)
            list_18591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 31), 'list')
            # Adding type elements to the builtin type 'list' instance (line 150)
            # Adding element type (line 150)
            # Getting the type of 'contained_elements' (line 150)
            contained_elements_18592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 32), 'contained_elements')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 31), list_18591, contained_elements_18592)
            
            # Assigning a type to the variable 'types_to_examine' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'types_to_examine', list_18591)
            # SSA join for if statement (line 147)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a List to a Name (line 152):
        
        # Assigning a List to a Name (line 152):
        
        # Obtaining an instance of the builtin type 'list' (line 152)
        list_18593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 152)
        
        # Assigning a type to the variable 'right_types' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'right_types', list_18593)
        
        # Assigning a List to a Name (line 153):
        
        # Assigning a List to a Name (line 153):
        
        # Obtaining an instance of the builtin type 'list' (line 153)
        list_18594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 153)
        
        # Assigning a type to the variable 'wrong_types' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'wrong_types', list_18594)
        
        # Getting the type of 'types_to_examine' (line 155)
        types_to_examine_18595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 21), 'types_to_examine')
        # Assigning a type to the variable 'types_to_examine_18595' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'types_to_examine_18595', types_to_examine_18595)
        # Testing if the for loop is going to be iterated (line 155)
        # Testing the type of a for loop iterable (line 155)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 155, 8), types_to_examine_18595)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 155, 8), types_to_examine_18595):
            # Getting the type of the for loop variable (line 155)
            for_loop_var_18596 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 155, 8), types_to_examine_18595)
            # Assigning a type to the variable 'type_' (line 155)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'type_', for_loop_var_18596)
            # SSA begins for a for statement (line 155)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Name to a Name (line 156):
            
            # Assigning a Name to a Name (line 156):
            # Getting the type of 'False' (line 156)
            False_18597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 26), 'False')
            # Assigning a type to the variable 'match_found' (line 156)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'match_found', False_18597)
            
            # Getting the type of 'self' (line 157)
            self_18598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 43), 'self')
            # Obtaining the member 'content_types' of a type (line 157)
            content_types_18599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 43), self_18598, 'content_types')
            # Assigning a type to the variable 'content_types_18599' (line 157)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'content_types_18599', content_types_18599)
            # Testing if the for loop is going to be iterated (line 157)
            # Testing the type of a for loop iterable (line 157)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 157, 12), content_types_18599)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 157, 12), content_types_18599):
                # Getting the type of the for loop variable (line 157)
                for_loop_var_18600 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 157, 12), content_types_18599)
                # Assigning a type to the variable 'declared_contained_type' (line 157)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'declared_contained_type', for_loop_var_18600)
                # SSA begins for a for statement (line 157)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'declared_contained_type' (line 158)
                declared_contained_type_18601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 19), 'declared_contained_type')
                # Getting the type of 'type_' (line 158)
                type__18602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 46), 'type_')
                # Applying the binary operator '==' (line 158)
                result_eq_18603 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 19), '==', declared_contained_type_18601, type__18602)
                
                # Testing if the type of an if condition is none (line 158)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 158, 16), result_eq_18603):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 158)
                    if_condition_18604 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 16), result_eq_18603)
                    # Assigning a type to the variable 'if_condition_18604' (line 158)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'if_condition_18604', if_condition_18604)
                    # SSA begins for if statement (line 158)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to isinstance(...): (line 159)
                    # Processing the call arguments (line 159)
                    # Getting the type of 'declared_contained_type' (line 159)
                    declared_contained_type_18606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 34), 'declared_contained_type', False)
                    # Getting the type of 'DependentType' (line 159)
                    DependentType_18607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 59), 'DependentType', False)
                    # Processing the call keyword arguments (line 159)
                    kwargs_18608 = {}
                    # Getting the type of 'isinstance' (line 159)
                    isinstance_18605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 'isinstance', False)
                    # Calling isinstance(args, kwargs) (line 159)
                    isinstance_call_result_18609 = invoke(stypy.reporting.localization.Localization(__file__, 159, 23), isinstance_18605, *[declared_contained_type_18606, DependentType_18607], **kwargs_18608)
                    
                    # Testing if the type of an if condition is none (line 159)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 159, 20), isinstance_call_result_18609):
                        
                        # Assigning a Name to a Name (line 174):
                        
                        # Assigning a Name to a Name (line 174):
                        # Getting the type of 'True' (line 174)
                        True_18671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 38), 'True')
                        # Assigning a type to the variable 'match_found' (line 174)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 24), 'match_found', True_18671)
                        
                        # Call to append(...): (line 175)
                        # Processing the call arguments (line 175)
                        # Getting the type of 'type_' (line 175)
                        type__18674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 43), 'type_', False)
                        # Processing the call keyword arguments (line 175)
                        kwargs_18675 = {}
                        # Getting the type of 'right_types' (line 175)
                        right_types_18672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 24), 'right_types', False)
                        # Obtaining the member 'append' of a type (line 175)
                        append_18673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 24), right_types_18672, 'append')
                        # Calling append(args, kwargs) (line 175)
                        append_call_result_18676 = invoke(stypy.reporting.localization.Localization(__file__, 175, 24), append_18673, *[type__18674], **kwargs_18675)
                        
                    else:
                        
                        # Testing the type of an if condition (line 159)
                        if_condition_18610 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 20), isinstance_call_result_18609)
                        # Assigning a type to the variable 'if_condition_18610' (line 159)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 20), 'if_condition_18610', if_condition_18610)
                        # SSA begins for if statement (line 159)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Getting the type of 'declared_contained_type' (line 160)
                        declared_contained_type_18611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 27), 'declared_contained_type')
                        # Obtaining the member 'call_arity' of a type (line 160)
                        call_arity_18612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 27), declared_contained_type_18611, 'call_arity')
                        int_18613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 65), 'int')
                        # Applying the binary operator '==' (line 160)
                        result_eq_18614 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 27), '==', call_arity_18612, int_18613)
                        
                        # Testing if the type of an if condition is none (line 160)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 160, 24), result_eq_18614):
                            
                            # Assigning a Call to a Tuple (line 163):
                            
                            # Assigning a Call to a Name:
                            
                            # Call to declared_contained_type(...): (line 163)
                            # Processing the call arguments (line 163)
                            # Getting the type of 'localization' (line 163)
                            localization_18627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 75), 'localization', False)
                            # Getting the type of 'type_' (line 163)
                            type__18628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 89), 'type_', False)
                            # Processing the call keyword arguments (line 163)
                            kwargs_18629 = {}
                            # Getting the type of 'declared_contained_type' (line 163)
                            declared_contained_type_18626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 51), 'declared_contained_type', False)
                            # Calling declared_contained_type(args, kwargs) (line 163)
                            declared_contained_type_call_result_18630 = invoke(stypy.reporting.localization.Localization(__file__, 163, 51), declared_contained_type_18626, *[localization_18627, type__18628], **kwargs_18629)
                            
                            # Assigning a type to the variable 'call_assignment_18336' (line 163)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_18336', declared_contained_type_call_result_18630)
                            
                            # Assigning a Call to a Name (line 163):
                            
                            # Call to stypy_get_value_from_tuple(...):
                            # Processing the call arguments
                            # Getting the type of 'call_assignment_18336' (line 163)
                            call_assignment_18336_18631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_18336', False)
                            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                            stypy_get_value_from_tuple_call_result_18632 = stypy_get_value_from_tuple(call_assignment_18336_18631, 2, 0)
                            
                            # Assigning a type to the variable 'call_assignment_18337' (line 163)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_18337', stypy_get_value_from_tuple_call_result_18632)
                            
                            # Assigning a Name to a Name (line 163):
                            # Getting the type of 'call_assignment_18337' (line 163)
                            call_assignment_18337_18633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_18337')
                            # Assigning a type to the variable 'correct' (line 163)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'correct', call_assignment_18337_18633)
                            
                            # Assigning a Call to a Name (line 163):
                            
                            # Call to stypy_get_value_from_tuple(...):
                            # Processing the call arguments
                            # Getting the type of 'call_assignment_18336' (line 163)
                            call_assignment_18336_18634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_18336', False)
                            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                            stypy_get_value_from_tuple_call_result_18635 = stypy_get_value_from_tuple(call_assignment_18336_18634, 2, 1)
                            
                            # Assigning a type to the variable 'call_assignment_18338' (line 163)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_18338', stypy_get_value_from_tuple_call_result_18635)
                            
                            # Assigning a Name to a Name (line 163):
                            # Getting the type of 'call_assignment_18338' (line 163)
                            call_assignment_18338_18636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_18338')
                            # Assigning a type to the variable 'return_type' (line 163)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 37), 'return_type', call_assignment_18338_18636)
                        else:
                            
                            # Testing the type of an if condition (line 160)
                            if_condition_18615 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 24), result_eq_18614)
                            # Assigning a type to the variable 'if_condition_18615' (line 160)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'if_condition_18615', if_condition_18615)
                            # SSA begins for if statement (line 160)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Call to a Tuple (line 161):
                            
                            # Assigning a Call to a Name:
                            
                            # Call to declared_contained_type(...): (line 161)
                            # Processing the call arguments (line 161)
                            # Getting the type of 'localization' (line 161)
                            localization_18617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 75), 'localization', False)
                            # Processing the call keyword arguments (line 161)
                            kwargs_18618 = {}
                            # Getting the type of 'declared_contained_type' (line 161)
                            declared_contained_type_18616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 51), 'declared_contained_type', False)
                            # Calling declared_contained_type(args, kwargs) (line 161)
                            declared_contained_type_call_result_18619 = invoke(stypy.reporting.localization.Localization(__file__, 161, 51), declared_contained_type_18616, *[localization_18617], **kwargs_18618)
                            
                            # Assigning a type to the variable 'call_assignment_18333' (line 161)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'call_assignment_18333', declared_contained_type_call_result_18619)
                            
                            # Assigning a Call to a Name (line 161):
                            
                            # Call to stypy_get_value_from_tuple(...):
                            # Processing the call arguments
                            # Getting the type of 'call_assignment_18333' (line 161)
                            call_assignment_18333_18620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'call_assignment_18333', False)
                            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                            stypy_get_value_from_tuple_call_result_18621 = stypy_get_value_from_tuple(call_assignment_18333_18620, 2, 0)
                            
                            # Assigning a type to the variable 'call_assignment_18334' (line 161)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'call_assignment_18334', stypy_get_value_from_tuple_call_result_18621)
                            
                            # Assigning a Name to a Name (line 161):
                            # Getting the type of 'call_assignment_18334' (line 161)
                            call_assignment_18334_18622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'call_assignment_18334')
                            # Assigning a type to the variable 'correct' (line 161)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'correct', call_assignment_18334_18622)
                            
                            # Assigning a Call to a Name (line 161):
                            
                            # Call to stypy_get_value_from_tuple(...):
                            # Processing the call arguments
                            # Getting the type of 'call_assignment_18333' (line 161)
                            call_assignment_18333_18623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'call_assignment_18333', False)
                            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                            stypy_get_value_from_tuple_call_result_18624 = stypy_get_value_from_tuple(call_assignment_18333_18623, 2, 1)
                            
                            # Assigning a type to the variable 'call_assignment_18335' (line 161)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'call_assignment_18335', stypy_get_value_from_tuple_call_result_18624)
                            
                            # Assigning a Name to a Name (line 161):
                            # Getting the type of 'call_assignment_18335' (line 161)
                            call_assignment_18335_18625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'call_assignment_18335')
                            # Assigning a type to the variable 'return_type' (line 161)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 37), 'return_type', call_assignment_18335_18625)
                            # SSA branch for the else part of an if statement (line 160)
                            module_type_store.open_ssa_branch('else')
                            
                            # Assigning a Call to a Tuple (line 163):
                            
                            # Assigning a Call to a Name:
                            
                            # Call to declared_contained_type(...): (line 163)
                            # Processing the call arguments (line 163)
                            # Getting the type of 'localization' (line 163)
                            localization_18627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 75), 'localization', False)
                            # Getting the type of 'type_' (line 163)
                            type__18628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 89), 'type_', False)
                            # Processing the call keyword arguments (line 163)
                            kwargs_18629 = {}
                            # Getting the type of 'declared_contained_type' (line 163)
                            declared_contained_type_18626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 51), 'declared_contained_type', False)
                            # Calling declared_contained_type(args, kwargs) (line 163)
                            declared_contained_type_call_result_18630 = invoke(stypy.reporting.localization.Localization(__file__, 163, 51), declared_contained_type_18626, *[localization_18627, type__18628], **kwargs_18629)
                            
                            # Assigning a type to the variable 'call_assignment_18336' (line 163)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_18336', declared_contained_type_call_result_18630)
                            
                            # Assigning a Call to a Name (line 163):
                            
                            # Call to stypy_get_value_from_tuple(...):
                            # Processing the call arguments
                            # Getting the type of 'call_assignment_18336' (line 163)
                            call_assignment_18336_18631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_18336', False)
                            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                            stypy_get_value_from_tuple_call_result_18632 = stypy_get_value_from_tuple(call_assignment_18336_18631, 2, 0)
                            
                            # Assigning a type to the variable 'call_assignment_18337' (line 163)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_18337', stypy_get_value_from_tuple_call_result_18632)
                            
                            # Assigning a Name to a Name (line 163):
                            # Getting the type of 'call_assignment_18337' (line 163)
                            call_assignment_18337_18633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_18337')
                            # Assigning a type to the variable 'correct' (line 163)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'correct', call_assignment_18337_18633)
                            
                            # Assigning a Call to a Name (line 163):
                            
                            # Call to stypy_get_value_from_tuple(...):
                            # Processing the call arguments
                            # Getting the type of 'call_assignment_18336' (line 163)
                            call_assignment_18336_18634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_18336', False)
                            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                            stypy_get_value_from_tuple_call_result_18635 = stypy_get_value_from_tuple(call_assignment_18336_18634, 2, 1)
                            
                            # Assigning a type to the variable 'call_assignment_18338' (line 163)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_18338', stypy_get_value_from_tuple_call_result_18635)
                            
                            # Assigning a Name to a Name (line 163):
                            # Getting the type of 'call_assignment_18338' (line 163)
                            call_assignment_18338_18636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_18338')
                            # Assigning a type to the variable 'return_type' (line 163)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 37), 'return_type', call_assignment_18338_18636)
                            # SSA join for if statement (line 160)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # Getting the type of 'correct' (line 164)
                        correct_18637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 27), 'correct')
                        # Testing if the type of an if condition is none (line 164)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 164, 24), correct_18637):
                            
                            # Evaluating a boolean operation
                            
                            # Getting the type of 'type_' (line 171)
                            type__18658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 31), 'type_')
                            # Getting the type of 'wrong_types' (line 171)
                            wrong_types_18659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 44), 'wrong_types')
                            # Applying the binary operator 'notin' (line 171)
                            result_contains_18660 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 31), 'notin', type__18658, wrong_types_18659)
                            
                            
                            # Getting the type of 'type_' (line 171)
                            type__18661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 60), 'type_')
                            # Getting the type of 'right_types' (line 171)
                            right_types_18662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 73), 'right_types')
                            # Applying the binary operator 'notin' (line 171)
                            result_contains_18663 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 60), 'notin', type__18661, right_types_18662)
                            
                            # Applying the binary operator 'and' (line 171)
                            result_and_keyword_18664 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 31), 'and', result_contains_18660, result_contains_18663)
                            
                            # Testing if the type of an if condition is none (line 171)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 171, 28), result_and_keyword_18664):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 171)
                                if_condition_18665 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 28), result_and_keyword_18664)
                                # Assigning a type to the variable 'if_condition_18665' (line 171)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 28), 'if_condition_18665', if_condition_18665)
                                # SSA begins for if statement (line 171)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to append(...): (line 172)
                                # Processing the call arguments (line 172)
                                # Getting the type of 'type_' (line 172)
                                type__18668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 51), 'type_', False)
                                # Processing the call keyword arguments (line 172)
                                kwargs_18669 = {}
                                # Getting the type of 'wrong_types' (line 172)
                                wrong_types_18666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 32), 'wrong_types', False)
                                # Obtaining the member 'append' of a type (line 172)
                                append_18667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 32), wrong_types_18666, 'append')
                                # Calling append(args, kwargs) (line 172)
                                append_call_result_18670 = invoke(stypy.reporting.localization.Localization(__file__, 172, 32), append_18667, *[type__18668], **kwargs_18669)
                                
                                # SSA join for if statement (line 171)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 164)
                            if_condition_18638 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 24), correct_18637)
                            # Assigning a type to the variable 'if_condition_18638' (line 164)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 24), 'if_condition_18638', if_condition_18638)
                            # SSA begins for if statement (line 164)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Name to a Name (line 165):
                            
                            # Assigning a Name to a Name (line 165):
                            # Getting the type of 'True' (line 165)
                            True_18639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 42), 'True')
                            # Assigning a type to the variable 'match_found' (line 165)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 28), 'match_found', True_18639)
                            
                            # Getting the type of 'type_' (line 166)
                            type__18640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 31), 'type_')
                            # Getting the type of 'right_types' (line 166)
                            right_types_18641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 44), 'right_types')
                            # Applying the binary operator 'notin' (line 166)
                            result_contains_18642 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 31), 'notin', type__18640, right_types_18641)
                            
                            # Testing if the type of an if condition is none (line 166)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 166, 28), result_contains_18642):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 166)
                                if_condition_18643 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 166, 28), result_contains_18642)
                                # Assigning a type to the variable 'if_condition_18643' (line 166)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 28), 'if_condition_18643', if_condition_18643)
                                # SSA begins for if statement (line 166)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to append(...): (line 167)
                                # Processing the call arguments (line 167)
                                # Getting the type of 'type_' (line 167)
                                type__18646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 51), 'type_', False)
                                # Processing the call keyword arguments (line 167)
                                kwargs_18647 = {}
                                # Getting the type of 'right_types' (line 167)
                                right_types_18644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 32), 'right_types', False)
                                # Obtaining the member 'append' of a type (line 167)
                                append_18645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 32), right_types_18644, 'append')
                                # Calling append(args, kwargs) (line 167)
                                append_call_result_18648 = invoke(stypy.reporting.localization.Localization(__file__, 167, 32), append_18645, *[type__18646], **kwargs_18647)
                                
                                
                                # Getting the type of 'type_' (line 168)
                                type__18649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 35), 'type_')
                                # Getting the type of 'wrong_types' (line 168)
                                wrong_types_18650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 44), 'wrong_types')
                                # Applying the binary operator 'in' (line 168)
                                result_contains_18651 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 35), 'in', type__18649, wrong_types_18650)
                                
                                # Testing if the type of an if condition is none (line 168)

                                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 168, 32), result_contains_18651):
                                    pass
                                else:
                                    
                                    # Testing the type of an if condition (line 168)
                                    if_condition_18652 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 32), result_contains_18651)
                                    # Assigning a type to the variable 'if_condition_18652' (line 168)
                                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 32), 'if_condition_18652', if_condition_18652)
                                    # SSA begins for if statement (line 168)
                                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                    
                                    # Call to remove(...): (line 169)
                                    # Processing the call arguments (line 169)
                                    # Getting the type of 'type_' (line 169)
                                    type__18655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 55), 'type_', False)
                                    # Processing the call keyword arguments (line 169)
                                    kwargs_18656 = {}
                                    # Getting the type of 'wrong_types' (line 169)
                                    wrong_types_18653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 36), 'wrong_types', False)
                                    # Obtaining the member 'remove' of a type (line 169)
                                    remove_18654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 36), wrong_types_18653, 'remove')
                                    # Calling remove(args, kwargs) (line 169)
                                    remove_call_result_18657 = invoke(stypy.reporting.localization.Localization(__file__, 169, 36), remove_18654, *[type__18655], **kwargs_18656)
                                    
                                    # SSA join for if statement (line 168)
                                    module_type_store = module_type_store.join_ssa_context()
                                    

                                # SSA join for if statement (line 166)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA branch for the else part of an if statement (line 164)
                            module_type_store.open_ssa_branch('else')
                            
                            # Evaluating a boolean operation
                            
                            # Getting the type of 'type_' (line 171)
                            type__18658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 31), 'type_')
                            # Getting the type of 'wrong_types' (line 171)
                            wrong_types_18659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 44), 'wrong_types')
                            # Applying the binary operator 'notin' (line 171)
                            result_contains_18660 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 31), 'notin', type__18658, wrong_types_18659)
                            
                            
                            # Getting the type of 'type_' (line 171)
                            type__18661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 60), 'type_')
                            # Getting the type of 'right_types' (line 171)
                            right_types_18662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 73), 'right_types')
                            # Applying the binary operator 'notin' (line 171)
                            result_contains_18663 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 60), 'notin', type__18661, right_types_18662)
                            
                            # Applying the binary operator 'and' (line 171)
                            result_and_keyword_18664 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 31), 'and', result_contains_18660, result_contains_18663)
                            
                            # Testing if the type of an if condition is none (line 171)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 171, 28), result_and_keyword_18664):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 171)
                                if_condition_18665 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 28), result_and_keyword_18664)
                                # Assigning a type to the variable 'if_condition_18665' (line 171)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 28), 'if_condition_18665', if_condition_18665)
                                # SSA begins for if statement (line 171)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to append(...): (line 172)
                                # Processing the call arguments (line 172)
                                # Getting the type of 'type_' (line 172)
                                type__18668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 51), 'type_', False)
                                # Processing the call keyword arguments (line 172)
                                kwargs_18669 = {}
                                # Getting the type of 'wrong_types' (line 172)
                                wrong_types_18666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 32), 'wrong_types', False)
                                # Obtaining the member 'append' of a type (line 172)
                                append_18667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 32), wrong_types_18666, 'append')
                                # Calling append(args, kwargs) (line 172)
                                append_call_result_18670 = invoke(stypy.reporting.localization.Localization(__file__, 172, 32), append_18667, *[type__18668], **kwargs_18669)
                                
                                # SSA join for if statement (line 171)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 164)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA branch for the else part of an if statement (line 159)
                        module_type_store.open_ssa_branch('else')
                        
                        # Assigning a Name to a Name (line 174):
                        
                        # Assigning a Name to a Name (line 174):
                        # Getting the type of 'True' (line 174)
                        True_18671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 38), 'True')
                        # Assigning a type to the variable 'match_found' (line 174)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 24), 'match_found', True_18671)
                        
                        # Call to append(...): (line 175)
                        # Processing the call arguments (line 175)
                        # Getting the type of 'type_' (line 175)
                        type__18674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 43), 'type_', False)
                        # Processing the call keyword arguments (line 175)
                        kwargs_18675 = {}
                        # Getting the type of 'right_types' (line 175)
                        right_types_18672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 24), 'right_types', False)
                        # Obtaining the member 'append' of a type (line 175)
                        append_18673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 24), right_types_18672, 'append')
                        # Calling append(args, kwargs) (line 175)
                        append_call_result_18676 = invoke(stypy.reporting.localization.Localization(__file__, 175, 24), append_18673, *[type__18674], **kwargs_18675)
                        
                        # SSA join for if statement (line 159)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 158)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Getting the type of 'match_found' (line 177)
            match_found_18677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 19), 'match_found')
            # Applying the 'not' unary operator (line 177)
            result_not__18678 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 15), 'not', match_found_18677)
            
            # Testing if the type of an if condition is none (line 177)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 177, 12), result_not__18678):
                pass
            else:
                
                # Testing the type of an if condition (line 177)
                if_condition_18679 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 12), result_not__18678)
                # Assigning a type to the variable 'if_condition_18679' (line 177)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'if_condition_18679', if_condition_18679)
                # SSA begins for if statement (line 177)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Evaluating a boolean operation
                
                # Getting the type of 'type_' (line 178)
                type__18680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 19), 'type_')
                # Getting the type of 'wrong_types' (line 178)
                wrong_types_18681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 32), 'wrong_types')
                # Applying the binary operator 'notin' (line 178)
                result_contains_18682 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 19), 'notin', type__18680, wrong_types_18681)
                
                
                # Getting the type of 'type_' (line 178)
                type__18683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 48), 'type_')
                # Getting the type of 'right_types' (line 178)
                right_types_18684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 61), 'right_types')
                # Applying the binary operator 'notin' (line 178)
                result_contains_18685 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 48), 'notin', type__18683, right_types_18684)
                
                # Applying the binary operator 'and' (line 178)
                result_and_keyword_18686 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 19), 'and', result_contains_18682, result_contains_18685)
                
                # Testing if the type of an if condition is none (line 178)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 178, 16), result_and_keyword_18686):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 178)
                    if_condition_18687 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 16), result_and_keyword_18686)
                    # Assigning a type to the variable 'if_condition_18687' (line 178)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'if_condition_18687', if_condition_18687)
                    # SSA begins for if statement (line 178)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to append(...): (line 179)
                    # Processing the call arguments (line 179)
                    # Getting the type of 'type_' (line 179)
                    type__18690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 39), 'type_', False)
                    # Processing the call keyword arguments (line 179)
                    kwargs_18691 = {}
                    # Getting the type of 'wrong_types' (line 179)
                    wrong_types_18688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 20), 'wrong_types', False)
                    # Obtaining the member 'append' of a type (line 179)
                    append_18689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 20), wrong_types_18688, 'append')
                    # Calling append(args, kwargs) (line 179)
                    append_call_result_18692 = invoke(stypy.reporting.localization.Localization(__file__, 179, 20), append_18689, *[type__18690], **kwargs_18691)
                    
                    # SSA join for if statement (line 178)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Name to a Name (line 180):
                
                # Assigning a Name to a Name (line 180):
                # Getting the type of 'False' (line 180)
                False_18693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 30), 'False')
                # Assigning a type to the variable 'match_found' (line 180)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'match_found', False_18693)
                # SSA join for if statement (line 177)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'self' (line 182)
        self_18694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 11), 'self')
        # Obtaining the member 'report_errors' of a type (line 182)
        report_errors_18695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 11), self_18694, 'report_errors')
        # Testing if the type of an if condition is none (line 182)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 182, 8), report_errors_18695):
            
            # Evaluating a boolean operation
            
            
            # Call to len(...): (line 197)
            # Processing the call arguments (line 197)
            # Getting the type of 'right_types' (line 197)
            right_types_18753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'right_types', False)
            # Processing the call keyword arguments (line 197)
            kwargs_18754 = {}
            # Getting the type of 'len' (line 197)
            len_18752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 15), 'len', False)
            # Calling len(args, kwargs) (line 197)
            len_call_result_18755 = invoke(stypy.reporting.localization.Localization(__file__, 197, 15), len_18752, *[right_types_18753], **kwargs_18754)
            
            int_18756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 35), 'int')
            # Applying the binary operator '==' (line 197)
            result_eq_18757 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 15), '==', len_call_result_18755, int_18756)
            
            
            
            # Call to len(...): (line 197)
            # Processing the call arguments (line 197)
            # Getting the type of 'wrong_types' (line 197)
            wrong_types_18759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 45), 'wrong_types', False)
            # Processing the call keyword arguments (line 197)
            kwargs_18760 = {}
            # Getting the type of 'len' (line 197)
            len_18758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 41), 'len', False)
            # Calling len(args, kwargs) (line 197)
            len_call_result_18761 = invoke(stypy.reporting.localization.Localization(__file__, 197, 41), len_18758, *[wrong_types_18759], **kwargs_18760)
            
            int_18762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 60), 'int')
            # Applying the binary operator '>' (line 197)
            result_gt_18763 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 41), '>', len_call_result_18761, int_18762)
            
            # Applying the binary operator 'and' (line 197)
            result_and_keyword_18764 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 15), 'and', result_eq_18757, result_gt_18763)
            
            # Testing if the type of an if condition is none (line 197)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 197, 12), result_and_keyword_18764):
                pass
            else:
                
                # Testing the type of an if condition (line 197)
                if_condition_18765 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 12), result_and_keyword_18764)
                # Assigning a type to the variable 'if_condition_18765' (line 197)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'if_condition_18765', if_condition_18765)
                # SSA begins for if statement (line 197)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to TypeWarning(...): (line 198)
                # Processing the call arguments (line 198)
                # Getting the type of 'localization' (line 198)
                localization_18767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 28), 'localization', False)
                
                # Call to format(...): (line 199)
                # Processing the call arguments (line 199)
                
                # Call to str(...): (line 200)
                # Processing the call arguments (line 200)
                # Getting the type of 'wrong_types' (line 200)
                wrong_types_18771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 36), 'wrong_types', False)
                # Processing the call keyword arguments (line 200)
                kwargs_18772 = {}
                # Getting the type of 'str' (line 200)
                str_18770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 32), 'str', False)
                # Calling str(args, kwargs) (line 200)
                str_call_result_18773 = invoke(stypy.reporting.localization.Localization(__file__, 200, 32), str_18770, *[wrong_types_18771], **kwargs_18772)
                
                
                # Call to str(...): (line 200)
                # Processing the call arguments (line 200)
                # Getting the type of 'self' (line 200)
                self_18775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 54), 'self', False)
                # Obtaining the member 'content_types' of a type (line 200)
                content_types_18776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 54), self_18775, 'content_types')
                # Processing the call keyword arguments (line 200)
                kwargs_18777 = {}
                # Getting the type of 'str' (line 200)
                str_18774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 50), 'str', False)
                # Calling str(args, kwargs) (line 200)
                str_call_result_18778 = invoke(stypy.reporting.localization.Localization(__file__, 200, 50), str_18774, *[content_types_18776], **kwargs_18777)
                
                # Processing the call keyword arguments (line 199)
                kwargs_18779 = {}
                str_18768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 28), 'str', 'Some of the iterable contained types: {0} do not match the expected ones {1}')
                # Obtaining the member 'format' of a type (line 199)
                format_18769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 28), str_18768, 'format')
                # Calling format(args, kwargs) (line 199)
                format_call_result_18780 = invoke(stypy.reporting.localization.Localization(__file__, 199, 28), format_18769, *[str_call_result_18773, str_call_result_18778], **kwargs_18779)
                
                # Processing the call keyword arguments (line 198)
                kwargs_18781 = {}
                # Getting the type of 'TypeWarning' (line 198)
                TypeWarning_18766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'TypeWarning', False)
                # Calling TypeWarning(args, kwargs) (line 198)
                TypeWarning_call_result_18782 = invoke(stypy.reporting.localization.Localization(__file__, 198, 16), TypeWarning_18766, *[localization_18767, format_call_result_18780], **kwargs_18781)
                
                # SSA join for if statement (line 197)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 182)
            if_condition_18696 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 8), report_errors_18695)
            # Assigning a type to the variable 'if_condition_18696' (line 182)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'if_condition_18696', if_condition_18696)
            # SSA begins for if statement (line 182)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to len(...): (line 184)
            # Processing the call arguments (line 184)
            # Getting the type of 'right_types' (line 184)
            right_types_18698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 19), 'right_types', False)
            # Processing the call keyword arguments (line 184)
            kwargs_18699 = {}
            # Getting the type of 'len' (line 184)
            len_18697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 15), 'len', False)
            # Calling len(args, kwargs) (line 184)
            len_call_result_18700 = invoke(stypy.reporting.localization.Localization(__file__, 184, 15), len_18697, *[right_types_18698], **kwargs_18699)
            
            int_18701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 35), 'int')
            # Applying the binary operator '==' (line 184)
            result_eq_18702 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 15), '==', len_call_result_18700, int_18701)
            
            # Testing if the type of an if condition is none (line 184)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 184, 12), result_eq_18702):
                
                
                # Call to len(...): (line 191)
                # Processing the call arguments (line 191)
                # Getting the type of 'wrong_types' (line 191)
                wrong_types_18729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 23), 'wrong_types', False)
                # Processing the call keyword arguments (line 191)
                kwargs_18730 = {}
                # Getting the type of 'len' (line 191)
                len_18728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 19), 'len', False)
                # Calling len(args, kwargs) (line 191)
                len_call_result_18731 = invoke(stypy.reporting.localization.Localization(__file__, 191, 19), len_18728, *[wrong_types_18729], **kwargs_18730)
                
                int_18732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 38), 'int')
                # Applying the binary operator '>' (line 191)
                result_gt_18733 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 19), '>', len_call_result_18731, int_18732)
                
                # Testing if the type of an if condition is none (line 191)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 191, 16), result_gt_18733):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 191)
                    if_condition_18734 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 16), result_gt_18733)
                    # Assigning a type to the variable 'if_condition_18734' (line 191)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'if_condition_18734', if_condition_18734)
                    # SSA begins for if statement (line 191)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to TypeWarning(...): (line 192)
                    # Processing the call arguments (line 192)
                    # Getting the type of 'localization' (line 192)
                    localization_18736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 32), 'localization', False)
                    
                    # Call to format(...): (line 193)
                    # Processing the call arguments (line 193)
                    
                    # Call to str(...): (line 194)
                    # Processing the call arguments (line 194)
                    # Getting the type of 'wrong_types' (line 194)
                    wrong_types_18740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 40), 'wrong_types', False)
                    # Processing the call keyword arguments (line 194)
                    kwargs_18741 = {}
                    # Getting the type of 'str' (line 194)
                    str_18739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 36), 'str', False)
                    # Calling str(args, kwargs) (line 194)
                    str_call_result_18742 = invoke(stypy.reporting.localization.Localization(__file__, 194, 36), str_18739, *[wrong_types_18740], **kwargs_18741)
                    
                    
                    # Call to str(...): (line 194)
                    # Processing the call arguments (line 194)
                    # Getting the type of 'self' (line 194)
                    self_18744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 58), 'self', False)
                    # Obtaining the member 'content_types' of a type (line 194)
                    content_types_18745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 58), self_18744, 'content_types')
                    # Processing the call keyword arguments (line 194)
                    kwargs_18746 = {}
                    # Getting the type of 'str' (line 194)
                    str_18743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 54), 'str', False)
                    # Calling str(args, kwargs) (line 194)
                    str_call_result_18747 = invoke(stypy.reporting.localization.Localization(__file__, 194, 54), str_18743, *[content_types_18745], **kwargs_18746)
                    
                    # Processing the call keyword arguments (line 193)
                    kwargs_18748 = {}
                    str_18737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 32), 'str', 'Some of the iterable contained types: {0} do not match the expected ones {1}')
                    # Obtaining the member 'format' of a type (line 193)
                    format_18738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 32), str_18737, 'format')
                    # Calling format(args, kwargs) (line 193)
                    format_call_result_18749 = invoke(stypy.reporting.localization.Localization(__file__, 193, 32), format_18738, *[str_call_result_18742, str_call_result_18747], **kwargs_18748)
                    
                    # Processing the call keyword arguments (line 192)
                    kwargs_18750 = {}
                    # Getting the type of 'TypeWarning' (line 192)
                    TypeWarning_18735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 20), 'TypeWarning', False)
                    # Calling TypeWarning(args, kwargs) (line 192)
                    TypeWarning_call_result_18751 = invoke(stypy.reporting.localization.Localization(__file__, 192, 20), TypeWarning_18735, *[localization_18736, format_call_result_18749], **kwargs_18750)
                    
                    # SSA join for if statement (line 191)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 184)
                if_condition_18703 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 12), result_eq_18702)
                # Assigning a type to the variable 'if_condition_18703' (line 184)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'if_condition_18703', if_condition_18703)
                # SSA begins for if statement (line 184)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                
                # Call to len(...): (line 185)
                # Processing the call arguments (line 185)
                # Getting the type of 'wrong_types' (line 185)
                wrong_types_18705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 23), 'wrong_types', False)
                # Processing the call keyword arguments (line 185)
                kwargs_18706 = {}
                # Getting the type of 'len' (line 185)
                len_18704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 19), 'len', False)
                # Calling len(args, kwargs) (line 185)
                len_call_result_18707 = invoke(stypy.reporting.localization.Localization(__file__, 185, 19), len_18704, *[wrong_types_18705], **kwargs_18706)
                
                int_18708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 38), 'int')
                # Applying the binary operator '>' (line 185)
                result_gt_18709 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 19), '>', len_call_result_18707, int_18708)
                
                # Testing if the type of an if condition is none (line 185)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 185, 16), result_gt_18709):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 185)
                    if_condition_18710 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 16), result_gt_18709)
                    # Assigning a type to the variable 'if_condition_18710' (line 185)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 16), 'if_condition_18710', if_condition_18710)
                    # SSA begins for if statement (line 185)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to TypeError(...): (line 186)
                    # Processing the call arguments (line 186)
                    # Getting the type of 'localization' (line 186)
                    localization_18712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 30), 'localization', False)
                    
                    # Call to format(...): (line 187)
                    # Processing the call arguments (line 187)
                    
                    # Call to str(...): (line 188)
                    # Processing the call arguments (line 188)
                    # Getting the type of 'types_to_examine' (line 188)
                    types_to_examine_18716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 38), 'types_to_examine', False)
                    # Processing the call keyword arguments (line 188)
                    kwargs_18717 = {}
                    # Getting the type of 'str' (line 188)
                    str_18715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 34), 'str', False)
                    # Calling str(args, kwargs) (line 188)
                    str_call_result_18718 = invoke(stypy.reporting.localization.Localization(__file__, 188, 34), str_18715, *[types_to_examine_18716], **kwargs_18717)
                    
                    
                    # Call to str(...): (line 188)
                    # Processing the call arguments (line 188)
                    # Getting the type of 'self' (line 188)
                    self_18720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 61), 'self', False)
                    # Obtaining the member 'content_types' of a type (line 188)
                    content_types_18721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 61), self_18720, 'content_types')
                    # Processing the call keyword arguments (line 188)
                    kwargs_18722 = {}
                    # Getting the type of 'str' (line 188)
                    str_18719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 57), 'str', False)
                    # Calling str(args, kwargs) (line 188)
                    str_call_result_18723 = invoke(stypy.reporting.localization.Localization(__file__, 188, 57), str_18719, *[content_types_18721], **kwargs_18722)
                    
                    # Processing the call keyword arguments (line 187)
                    kwargs_18724 = {}
                    str_18713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 30), 'str', 'None of the iterable contained types: {0} match the expected ones {1}')
                    # Obtaining the member 'format' of a type (line 187)
                    format_18714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 30), str_18713, 'format')
                    # Calling format(args, kwargs) (line 187)
                    format_call_result_18725 = invoke(stypy.reporting.localization.Localization(__file__, 187, 30), format_18714, *[str_call_result_18718, str_call_result_18723], **kwargs_18724)
                    
                    # Processing the call keyword arguments (line 186)
                    kwargs_18726 = {}
                    # Getting the type of 'TypeError' (line 186)
                    TypeError_18711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 20), 'TypeError', False)
                    # Calling TypeError(args, kwargs) (line 186)
                    TypeError_call_result_18727 = invoke(stypy.reporting.localization.Localization(__file__, 186, 20), TypeError_18711, *[localization_18712, format_call_result_18725], **kwargs_18726)
                    
                    # SSA join for if statement (line 185)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA branch for the else part of an if statement (line 184)
                module_type_store.open_ssa_branch('else')
                
                
                # Call to len(...): (line 191)
                # Processing the call arguments (line 191)
                # Getting the type of 'wrong_types' (line 191)
                wrong_types_18729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 23), 'wrong_types', False)
                # Processing the call keyword arguments (line 191)
                kwargs_18730 = {}
                # Getting the type of 'len' (line 191)
                len_18728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 19), 'len', False)
                # Calling len(args, kwargs) (line 191)
                len_call_result_18731 = invoke(stypy.reporting.localization.Localization(__file__, 191, 19), len_18728, *[wrong_types_18729], **kwargs_18730)
                
                int_18732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 38), 'int')
                # Applying the binary operator '>' (line 191)
                result_gt_18733 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 19), '>', len_call_result_18731, int_18732)
                
                # Testing if the type of an if condition is none (line 191)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 191, 16), result_gt_18733):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 191)
                    if_condition_18734 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 16), result_gt_18733)
                    # Assigning a type to the variable 'if_condition_18734' (line 191)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'if_condition_18734', if_condition_18734)
                    # SSA begins for if statement (line 191)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to TypeWarning(...): (line 192)
                    # Processing the call arguments (line 192)
                    # Getting the type of 'localization' (line 192)
                    localization_18736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 32), 'localization', False)
                    
                    # Call to format(...): (line 193)
                    # Processing the call arguments (line 193)
                    
                    # Call to str(...): (line 194)
                    # Processing the call arguments (line 194)
                    # Getting the type of 'wrong_types' (line 194)
                    wrong_types_18740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 40), 'wrong_types', False)
                    # Processing the call keyword arguments (line 194)
                    kwargs_18741 = {}
                    # Getting the type of 'str' (line 194)
                    str_18739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 36), 'str', False)
                    # Calling str(args, kwargs) (line 194)
                    str_call_result_18742 = invoke(stypy.reporting.localization.Localization(__file__, 194, 36), str_18739, *[wrong_types_18740], **kwargs_18741)
                    
                    
                    # Call to str(...): (line 194)
                    # Processing the call arguments (line 194)
                    # Getting the type of 'self' (line 194)
                    self_18744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 58), 'self', False)
                    # Obtaining the member 'content_types' of a type (line 194)
                    content_types_18745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 58), self_18744, 'content_types')
                    # Processing the call keyword arguments (line 194)
                    kwargs_18746 = {}
                    # Getting the type of 'str' (line 194)
                    str_18743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 54), 'str', False)
                    # Calling str(args, kwargs) (line 194)
                    str_call_result_18747 = invoke(stypy.reporting.localization.Localization(__file__, 194, 54), str_18743, *[content_types_18745], **kwargs_18746)
                    
                    # Processing the call keyword arguments (line 193)
                    kwargs_18748 = {}
                    str_18737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 32), 'str', 'Some of the iterable contained types: {0} do not match the expected ones {1}')
                    # Obtaining the member 'format' of a type (line 193)
                    format_18738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 32), str_18737, 'format')
                    # Calling format(args, kwargs) (line 193)
                    format_call_result_18749 = invoke(stypy.reporting.localization.Localization(__file__, 193, 32), format_18738, *[str_call_result_18742, str_call_result_18747], **kwargs_18748)
                    
                    # Processing the call keyword arguments (line 192)
                    kwargs_18750 = {}
                    # Getting the type of 'TypeWarning' (line 192)
                    TypeWarning_18735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 20), 'TypeWarning', False)
                    # Calling TypeWarning(args, kwargs) (line 192)
                    TypeWarning_call_result_18751 = invoke(stypy.reporting.localization.Localization(__file__, 192, 20), TypeWarning_18735, *[localization_18736, format_call_result_18749], **kwargs_18750)
                    
                    # SSA join for if statement (line 191)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 184)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA branch for the else part of an if statement (line 182)
            module_type_store.open_ssa_branch('else')
            
            # Evaluating a boolean operation
            
            
            # Call to len(...): (line 197)
            # Processing the call arguments (line 197)
            # Getting the type of 'right_types' (line 197)
            right_types_18753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'right_types', False)
            # Processing the call keyword arguments (line 197)
            kwargs_18754 = {}
            # Getting the type of 'len' (line 197)
            len_18752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 15), 'len', False)
            # Calling len(args, kwargs) (line 197)
            len_call_result_18755 = invoke(stypy.reporting.localization.Localization(__file__, 197, 15), len_18752, *[right_types_18753], **kwargs_18754)
            
            int_18756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 35), 'int')
            # Applying the binary operator '==' (line 197)
            result_eq_18757 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 15), '==', len_call_result_18755, int_18756)
            
            
            
            # Call to len(...): (line 197)
            # Processing the call arguments (line 197)
            # Getting the type of 'wrong_types' (line 197)
            wrong_types_18759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 45), 'wrong_types', False)
            # Processing the call keyword arguments (line 197)
            kwargs_18760 = {}
            # Getting the type of 'len' (line 197)
            len_18758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 41), 'len', False)
            # Calling len(args, kwargs) (line 197)
            len_call_result_18761 = invoke(stypy.reporting.localization.Localization(__file__, 197, 41), len_18758, *[wrong_types_18759], **kwargs_18760)
            
            int_18762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 60), 'int')
            # Applying the binary operator '>' (line 197)
            result_gt_18763 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 41), '>', len_call_result_18761, int_18762)
            
            # Applying the binary operator 'and' (line 197)
            result_and_keyword_18764 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 15), 'and', result_eq_18757, result_gt_18763)
            
            # Testing if the type of an if condition is none (line 197)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 197, 12), result_and_keyword_18764):
                pass
            else:
                
                # Testing the type of an if condition (line 197)
                if_condition_18765 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 12), result_and_keyword_18764)
                # Assigning a type to the variable 'if_condition_18765' (line 197)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'if_condition_18765', if_condition_18765)
                # SSA begins for if statement (line 197)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to TypeWarning(...): (line 198)
                # Processing the call arguments (line 198)
                # Getting the type of 'localization' (line 198)
                localization_18767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 28), 'localization', False)
                
                # Call to format(...): (line 199)
                # Processing the call arguments (line 199)
                
                # Call to str(...): (line 200)
                # Processing the call arguments (line 200)
                # Getting the type of 'wrong_types' (line 200)
                wrong_types_18771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 36), 'wrong_types', False)
                # Processing the call keyword arguments (line 200)
                kwargs_18772 = {}
                # Getting the type of 'str' (line 200)
                str_18770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 32), 'str', False)
                # Calling str(args, kwargs) (line 200)
                str_call_result_18773 = invoke(stypy.reporting.localization.Localization(__file__, 200, 32), str_18770, *[wrong_types_18771], **kwargs_18772)
                
                
                # Call to str(...): (line 200)
                # Processing the call arguments (line 200)
                # Getting the type of 'self' (line 200)
                self_18775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 54), 'self', False)
                # Obtaining the member 'content_types' of a type (line 200)
                content_types_18776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 54), self_18775, 'content_types')
                # Processing the call keyword arguments (line 200)
                kwargs_18777 = {}
                # Getting the type of 'str' (line 200)
                str_18774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 50), 'str', False)
                # Calling str(args, kwargs) (line 200)
                str_call_result_18778 = invoke(stypy.reporting.localization.Localization(__file__, 200, 50), str_18774, *[content_types_18776], **kwargs_18777)
                
                # Processing the call keyword arguments (line 199)
                kwargs_18779 = {}
                str_18768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 28), 'str', 'Some of the iterable contained types: {0} do not match the expected ones {1}')
                # Obtaining the member 'format' of a type (line 199)
                format_18769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 28), str_18768, 'format')
                # Calling format(args, kwargs) (line 199)
                format_call_result_18780 = invoke(stypy.reporting.localization.Localization(__file__, 199, 28), format_18769, *[str_call_result_18773, str_call_result_18778], **kwargs_18779)
                
                # Processing the call keyword arguments (line 198)
                kwargs_18781 = {}
                # Getting the type of 'TypeWarning' (line 198)
                TypeWarning_18766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'TypeWarning', False)
                # Calling TypeWarning(args, kwargs) (line 198)
                TypeWarning_call_result_18782 = invoke(stypy.reporting.localization.Localization(__file__, 198, 16), TypeWarning_18766, *[localization_18767, format_call_result_18780], **kwargs_18781)
                
                # SSA join for if statement (line 197)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 182)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to len(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'right_types' (line 203)
        right_types_18784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), 'right_types', False)
        # Processing the call keyword arguments (line 203)
        kwargs_18785 = {}
        # Getting the type of 'len' (line 203)
        len_18783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 11), 'len', False)
        # Calling len(args, kwargs) (line 203)
        len_call_result_18786 = invoke(stypy.reporting.localization.Localization(__file__, 203, 11), len_18783, *[right_types_18784], **kwargs_18785)
        
        int_18787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 30), 'int')
        # Applying the binary operator '>' (line 203)
        result_gt_18788 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 11), '>', len_call_result_18786, int_18787)
        
        # Testing if the type of an if condition is none (line 203)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 203, 8), result_gt_18788):
            
            # Obtaining an instance of the builtin type 'tuple' (line 206)
            tuple_18793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 206)
            # Adding element type (line 206)
            # Getting the type of 'False' (line 206)
            False_18794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 19), 'False')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 19), tuple_18793, False_18794)
            # Adding element type (line 206)
            # Getting the type of 'wrong_types' (line 206)
            wrong_types_18795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 26), 'wrong_types')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 19), tuple_18793, wrong_types_18795)
            
            # Assigning a type to the variable 'stypy_return_type' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'stypy_return_type', tuple_18793)
        else:
            
            # Testing the type of an if condition (line 203)
            if_condition_18789 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 8), result_gt_18788)
            # Assigning a type to the variable 'if_condition_18789' (line 203)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'if_condition_18789', if_condition_18789)
            # SSA begins for if statement (line 203)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining an instance of the builtin type 'tuple' (line 204)
            tuple_18790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 204)
            # Adding element type (line 204)
            # Getting the type of 'True' (line 204)
            True_18791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 19), 'True')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 19), tuple_18790, True_18791)
            # Adding element type (line 204)
            # Getting the type of 'None' (line 204)
            None_18792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 25), 'None')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 19), tuple_18790, None_18792)
            
            # Assigning a type to the variable 'stypy_return_type' (line 204)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'stypy_return_type', tuple_18790)
            # SSA branch for the else part of an if statement (line 203)
            module_type_store.open_ssa_branch('else')
            
            # Obtaining an instance of the builtin type 'tuple' (line 206)
            tuple_18793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 206)
            # Adding element type (line 206)
            # Getting the type of 'False' (line 206)
            False_18794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 19), 'False')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 19), tuple_18793, False_18794)
            # Adding element type (line 206)
            # Getting the type of 'wrong_types' (line 206)
            wrong_types_18795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 26), 'wrong_types')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 19), tuple_18793, wrong_types_18795)
            
            # Assigning a type to the variable 'stypy_return_type' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'stypy_return_type', tuple_18793)
            # SSA join for if statement (line 203)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 145)
        stypy_return_type_18796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18796)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_18796


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 208, 4, False)
        # Assigning a type to the variable 'self' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IterableDataStructureWithTypedElements.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        IterableDataStructureWithTypedElements.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IterableDataStructureWithTypedElements.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        IterableDataStructureWithTypedElements.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'IterableDataStructureWithTypedElements.stypy__repr__')
        IterableDataStructureWithTypedElements.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        IterableDataStructureWithTypedElements.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        IterableDataStructureWithTypedElements.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IterableDataStructureWithTypedElements.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        IterableDataStructureWithTypedElements.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        IterableDataStructureWithTypedElements.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IterableDataStructureWithTypedElements.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IterableDataStructureWithTypedElements.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Name (line 209):
        
        # Assigning a Str to a Name (line 209):
        str_18797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 18), 'str', 'Iterable[')
        # Assigning a type to the variable 'ret_str' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'ret_str', str_18797)
        
        # Assigning a Str to a Name (line 211):
        
        # Assigning a Str to a Name (line 211):
        str_18798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 19), 'str', '')
        # Assigning a type to the variable 'contents' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'contents', str_18798)
        
        # Getting the type of 'self' (line 212)
        self_18799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 23), 'self')
        # Obtaining the member 'content_types' of a type (line 212)
        content_types_18800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 23), self_18799, 'content_types')
        # Assigning a type to the variable 'content_types_18800' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'content_types_18800', content_types_18800)
        # Testing if the for loop is going to be iterated (line 212)
        # Testing the type of a for loop iterable (line 212)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 212, 8), content_types_18800)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 212, 8), content_types_18800):
            # Getting the type of the for loop variable (line 212)
            for_loop_var_18801 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 212, 8), content_types_18800)
            # Assigning a type to the variable 'content' (line 212)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'content', for_loop_var_18801)
            # SSA begins for a for statement (line 212)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'contents' (line 213)
            contents_18802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'contents')
            
            # Call to str(...): (line 213)
            # Processing the call arguments (line 213)
            # Getting the type of 'content' (line 213)
            content_18804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 28), 'content', False)
            # Processing the call keyword arguments (line 213)
            kwargs_18805 = {}
            # Getting the type of 'str' (line 213)
            str_18803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 24), 'str', False)
            # Calling str(args, kwargs) (line 213)
            str_call_result_18806 = invoke(stypy.reporting.localization.Localization(__file__, 213, 24), str_18803, *[content_18804], **kwargs_18805)
            
            str_18807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 39), 'str', ', ')
            # Applying the binary operator '+' (line 213)
            result_add_18808 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 24), '+', str_call_result_18806, str_18807)
            
            # Applying the binary operator '+=' (line 213)
            result_iadd_18809 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 12), '+=', contents_18802, result_add_18808)
            # Assigning a type to the variable 'contents' (line 213)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'contents', result_iadd_18809)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Subscript to a Name (line 214):
        
        # Assigning a Subscript to a Name (line 214):
        
        # Obtaining the type of the subscript
        int_18810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 29), 'int')
        slice_18811 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 214, 19), None, int_18810, None)
        # Getting the type of 'contents' (line 214)
        contents_18812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 19), 'contents')
        # Obtaining the member '__getitem__' of a type (line 214)
        getitem___18813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 19), contents_18812, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 214)
        subscript_call_result_18814 = invoke(stypy.reporting.localization.Localization(__file__, 214, 19), getitem___18813, slice_18811)
        
        # Assigning a type to the variable 'contents' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'contents', subscript_call_result_18814)
        
        # Getting the type of 'ret_str' (line 216)
        ret_str_18815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'ret_str')
        # Getting the type of 'contents' (line 216)
        contents_18816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 19), 'contents')
        # Applying the binary operator '+=' (line 216)
        result_iadd_18817 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 8), '+=', ret_str_18815, contents_18816)
        # Assigning a type to the variable 'ret_str' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'ret_str', result_iadd_18817)
        
        
        # Getting the type of 'ret_str' (line 217)
        ret_str_18818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'ret_str')
        str_18819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 19), 'str', ']')
        # Applying the binary operator '+=' (line 217)
        result_iadd_18820 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 8), '+=', ret_str_18818, str_18819)
        # Assigning a type to the variable 'ret_str' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'ret_str', result_iadd_18820)
        
        # Getting the type of 'ret_str' (line 218)
        ret_str_18821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), 'ret_str')
        # Assigning a type to the variable 'stypy_return_type' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'stypy_return_type', ret_str_18821)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 208)
        stypy_return_type_18822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18822)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_18822


# Assigning a type to the variable 'IterableDataStructureWithTypedElements' (line 129)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 0), 'IterableDataStructureWithTypedElements', IterableDataStructureWithTypedElements)
# Declaration of the 'DynamicType' class
# Getting the type of 'TypeGroup' (line 221)
TypeGroup_18823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 18), 'TypeGroup')

class DynamicType(TypeGroup_18823, ):
    str_18824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, (-1)), 'str', '\n    Any type (type cannot be statically calculated)\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 226, 4, False)
        # Assigning a type to the variable 'self' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DynamicType.__init__', [], 'members', None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'self' (line 227)
        self_18827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 227)
        list_18828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 227)
        
        # Processing the call keyword arguments (line 227)
        kwargs_18829 = {}
        # Getting the type of 'TypeGroup' (line 227)
        TypeGroup_18825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 227)
        init___18826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), TypeGroup_18825, '__init__')
        # Calling __init__(args, kwargs) (line 227)
        init___call_result_18830 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), init___18826, *[self_18827, list_18828], **kwargs_18829)
        
        
        # Assigning a Name to a Attribute (line 228):
        
        # Assigning a Name to a Attribute (line 228):
        # Getting the type of 'members' (line 228)
        members_18831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 23), 'members')
        # Getting the type of 'self' (line 228)
        self_18832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'self')
        # Setting the type of the member 'members' of a type (line 228)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), self_18832, 'members', members_18831)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 230, 4, False)
        # Assigning a type to the variable 'self' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DynamicType.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        DynamicType.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DynamicType.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        DynamicType.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'DynamicType.stypy__eq__')
        DynamicType.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['type_'])
        DynamicType.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        DynamicType.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DynamicType.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        DynamicType.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        DynamicType.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DynamicType.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DynamicType.stypy__eq__', ['type_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['type_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        # Getting the type of 'True' (line 231)
        True_18833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'stypy_return_type', True_18833)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 230)
        stypy_return_type_18834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18834)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_18834


# Assigning a type to the variable 'DynamicType' (line 221)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 0), 'DynamicType', DynamicType)
# Declaration of the 'SupportsStructuralIntercession' class
# Getting the type of 'TypeGroup' (line 234)
TypeGroup_18835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 37), 'TypeGroup')

class SupportsStructuralIntercession(TypeGroup_18835, ):
    str_18836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, (-1)), 'str', '\n    Any Python object that supports structural intercession\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 239, 4, False)
        # Assigning a type to the variable 'self' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SupportsStructuralIntercession.__init__', [], 'members', None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of 'self' (line 240)
        self_18839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 240)
        list_18840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 240)
        
        # Processing the call keyword arguments (line 240)
        kwargs_18841 = {}
        # Getting the type of 'TypeGroup' (line 240)
        TypeGroup_18837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 240)
        init___18838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), TypeGroup_18837, '__init__')
        # Calling __init__(args, kwargs) (line 240)
        init___call_result_18842 = invoke(stypy.reporting.localization.Localization(__file__, 240, 8), init___18838, *[self_18839, list_18840], **kwargs_18841)
        
        
        # Assigning a Name to a Attribute (line 241):
        
        # Assigning a Name to a Attribute (line 241):
        # Getting the type of 'members' (line 241)
        members_18843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 23), 'members')
        # Getting the type of 'self' (line 241)
        self_18844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'self')
        # Setting the type of the member 'members' of a type (line 241)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 8), self_18844, 'members', members_18843)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 243, 4, False)
        # Assigning a type to the variable 'self' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SupportsStructuralIntercession.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        SupportsStructuralIntercession.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SupportsStructuralIntercession.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        SupportsStructuralIntercession.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'SupportsStructuralIntercession.stypy__eq__')
        SupportsStructuralIntercession.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['type_'])
        SupportsStructuralIntercession.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        SupportsStructuralIntercession.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SupportsStructuralIntercession.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        SupportsStructuralIntercession.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        SupportsStructuralIntercession.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SupportsStructuralIntercession.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SupportsStructuralIntercession.stypy__eq__', ['type_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['type_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 244):
        
        # Assigning a Name to a Attribute (line 244):
        # Getting the type of 'type_' (line 244)
        type__18845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 21), 'type_')
        # Getting the type of 'self' (line 244)
        self_18846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'self')
        # Setting the type of the member 'type_' of a type (line 244)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), self_18846, 'type_', type__18845)
        
        # Call to supports_structural_reflection(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'type_' (line 245)
        type__18849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 83), 'type_', False)
        # Processing the call keyword arguments (line 245)
        kwargs_18850 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 245)
        type_inference_proxy_management_copy_18847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 15), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'supports_structural_reflection' of a type (line 245)
        supports_structural_reflection_18848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 15), type_inference_proxy_management_copy_18847, 'supports_structural_reflection')
        # Calling supports_structural_reflection(args, kwargs) (line 245)
        supports_structural_reflection_call_result_18851 = invoke(stypy.reporting.localization.Localization(__file__, 245, 15), supports_structural_reflection_18848, *[type__18849], **kwargs_18850)
        
        # Assigning a type to the variable 'stypy_return_type' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'stypy_return_type', supports_structural_reflection_call_result_18851)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 243)
        stypy_return_type_18852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18852)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_18852


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 247, 4, False)
        # Assigning a type to the variable 'self' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SupportsStructuralIntercession.__call__.__dict__.__setitem__('stypy_localization', localization)
        SupportsStructuralIntercession.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SupportsStructuralIntercession.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        SupportsStructuralIntercession.__call__.__dict__.__setitem__('stypy_function_name', 'SupportsStructuralIntercession.__call__')
        SupportsStructuralIntercession.__call__.__dict__.__setitem__('stypy_param_names_list', ['localization'])
        SupportsStructuralIntercession.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'call_args')
        SupportsStructuralIntercession.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'call_kwargs')
        SupportsStructuralIntercession.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        SupportsStructuralIntercession.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        SupportsStructuralIntercession.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SupportsStructuralIntercession.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SupportsStructuralIntercession.__call__', ['localization'], 'call_args', 'call_kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['localization'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Assigning a Attribute to a Name (line 248):
        
        # Assigning a Attribute to a Name (line 248):
        # Getting the type of 'self' (line 248)
        self_18853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 15), 'self')
        # Obtaining the member 'type_' of a type (line 248)
        type__18854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 15), self_18853, 'type_')
        # Assigning a type to the variable 'temp' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'temp', type__18854)
        
        # Assigning a Name to a Attribute (line 249):
        
        # Assigning a Name to a Attribute (line 249):
        # Getting the type of 'None' (line 249)
        None_18855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 21), 'None')
        # Getting the type of 'self' (line 249)
        self_18856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'self')
        # Setting the type of the member 'type_' of a type (line 249)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), self_18856, 'type_', None_18855)
        # Getting the type of 'temp' (line 251)
        temp_18857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'stypy_return_type', temp_18857)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 247)
        stypy_return_type_18858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18858)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_18858


# Assigning a type to the variable 'SupportsStructuralIntercession' (line 234)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 0), 'SupportsStructuralIntercession', SupportsStructuralIntercession)
# Declaration of the 'SubtypeOf' class
# Getting the type of 'TypeGroup' (line 254)
TypeGroup_18859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'TypeGroup')

class SubtypeOf(TypeGroup_18859, ):
    str_18860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, (-1)), 'str', '\n    A subtype of the type passed in the constructor\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 259, 4, False)
        # Assigning a type to the variable 'self' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubtypeOf.__init__', [], 'types_', None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'self' (line 260)
        self_18863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 260)
        list_18864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 260)
        
        # Processing the call keyword arguments (line 260)
        kwargs_18865 = {}
        # Getting the type of 'TypeGroup' (line 260)
        TypeGroup_18861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 260)
        init___18862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 8), TypeGroup_18861, '__init__')
        # Calling __init__(args, kwargs) (line 260)
        init___call_result_18866 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), init___18862, *[self_18863, list_18864], **kwargs_18865)
        
        
        # Assigning a Name to a Attribute (line 261):
        
        # Assigning a Name to a Attribute (line 261):
        # Getting the type of 'types_' (line 261)
        types__18867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 21), 'types_')
        # Getting the type of 'self' (line 261)
        self_18868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'self')
        # Setting the type of the member 'types' of a type (line 261)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), self_18868, 'types', types__18867)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 263, 4, False)
        # Assigning a type to the variable 'self' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubtypeOf.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        SubtypeOf.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubtypeOf.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubtypeOf.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'SubtypeOf.stypy__eq__')
        SubtypeOf.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['type_'])
        SubtypeOf.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        SubtypeOf.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SubtypeOf.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubtypeOf.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubtypeOf.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubtypeOf.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubtypeOf.stypy__eq__', ['type_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['type_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 264):
        
        # Assigning a Name to a Attribute (line 264):
        # Getting the type of 'type_' (line 264)
        type__18869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 21), 'type_')
        # Getting the type of 'self' (line 264)
        self_18870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'self')
        # Setting the type of the member 'type_' of a type (line 264)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 8), self_18870, 'type_', type__18869)
        
        # Getting the type of 'self' (line 265)
        self_18871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 28), 'self')
        # Obtaining the member 'types' of a type (line 265)
        types_18872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 28), self_18871, 'types')
        # Assigning a type to the variable 'types_18872' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'types_18872', types_18872)
        # Testing if the for loop is going to be iterated (line 265)
        # Testing the type of a for loop iterable (line 265)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 265, 8), types_18872)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 265, 8), types_18872):
            # Getting the type of the for loop variable (line 265)
            for_loop_var_18873 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 265, 8), types_18872)
            # Assigning a type to the variable 'pattern_type' (line 265)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'pattern_type', for_loop_var_18873)
            # SSA begins for a for statement (line 265)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to issubclass(...): (line 266)
            # Processing the call arguments (line 266)
            # Getting the type of 'type_' (line 266)
            type__18875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 30), 'type_', False)
            # Getting the type of 'pattern_type' (line 266)
            pattern_type_18876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 37), 'pattern_type', False)
            # Processing the call keyword arguments (line 266)
            kwargs_18877 = {}
            # Getting the type of 'issubclass' (line 266)
            issubclass_18874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 19), 'issubclass', False)
            # Calling issubclass(args, kwargs) (line 266)
            issubclass_call_result_18878 = invoke(stypy.reporting.localization.Localization(__file__, 266, 19), issubclass_18874, *[type__18875, pattern_type_18876], **kwargs_18877)
            
            # Applying the 'not' unary operator (line 266)
            result_not__18879 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 15), 'not', issubclass_call_result_18878)
            
            # Testing if the type of an if condition is none (line 266)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 266, 12), result_not__18879):
                pass
            else:
                
                # Testing the type of an if condition (line 266)
                if_condition_18880 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 266, 12), result_not__18879)
                # Assigning a type to the variable 'if_condition_18880' (line 266)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'if_condition_18880', if_condition_18880)
                # SSA begins for if statement (line 266)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'False' (line 267)
                False_18881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 267)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 16), 'stypy_return_type', False_18881)
                # SSA join for if statement (line 266)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'True' (line 268)
        True_18882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'stypy_return_type', True_18882)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 263)
        stypy_return_type_18883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18883)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_18883


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 270, 4, False)
        # Assigning a type to the variable 'self' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubtypeOf.__call__.__dict__.__setitem__('stypy_localization', localization)
        SubtypeOf.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubtypeOf.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubtypeOf.__call__.__dict__.__setitem__('stypy_function_name', 'SubtypeOf.__call__')
        SubtypeOf.__call__.__dict__.__setitem__('stypy_param_names_list', ['localization'])
        SubtypeOf.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'call_args')
        SubtypeOf.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'call_kwargs')
        SubtypeOf.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubtypeOf.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubtypeOf.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubtypeOf.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubtypeOf.__call__', ['localization'], 'call_args', 'call_kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['localization'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Assigning a Attribute to a Name (line 271):
        
        # Assigning a Attribute to a Name (line 271):
        # Getting the type of 'self' (line 271)
        self_18884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 15), 'self')
        # Obtaining the member 'type_' of a type (line 271)
        type__18885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 15), self_18884, 'type_')
        # Assigning a type to the variable 'temp' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'temp', type__18885)
        
        # Assigning a Name to a Attribute (line 272):
        
        # Assigning a Name to a Attribute (line 272):
        # Getting the type of 'None' (line 272)
        None_18886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 21), 'None')
        # Getting the type of 'self' (line 272)
        self_18887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'self')
        # Setting the type of the member 'type_' of a type (line 272)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), self_18887, 'type_', None_18886)
        # Getting the type of 'temp' (line 274)
        temp_18888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'stypy_return_type', temp_18888)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 270)
        stypy_return_type_18889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18889)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_18889


# Assigning a type to the variable 'SubtypeOf' (line 254)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 0), 'SubtypeOf', SubtypeOf)
# Declaration of the 'IsHashable' class
# Getting the type of 'TypeGroup' (line 277)
TypeGroup_18890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 17), 'TypeGroup')

class IsHashable(TypeGroup_18890, ):
    str_18891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, (-1)), 'str', '\n    Represent types that can properly implement the __hash__ members, so it can be placed as keys on a dict\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 282, 4, False)
        # Assigning a type to the variable 'self' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IsHashable.__init__', [], 'types_', None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'self' (line 283)
        self_18894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 283)
        list_18895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 283)
        
        # Processing the call keyword arguments (line 283)
        kwargs_18896 = {}
        # Getting the type of 'TypeGroup' (line 283)
        TypeGroup_18892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 283)
        init___18893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), TypeGroup_18892, '__init__')
        # Calling __init__(args, kwargs) (line 283)
        init___call_result_18897 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), init___18893, *[self_18894, list_18895], **kwargs_18896)
        
        
        # Assigning a Name to a Attribute (line 284):
        
        # Assigning a Name to a Attribute (line 284):
        # Getting the type of 'types_' (line 284)
        types__18898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 21), 'types_')
        # Getting the type of 'self' (line 284)
        self_18899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'self')
        # Setting the type of the member 'types' of a type (line 284)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 8), self_18899, 'types', types__18898)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 286, 4, False)
        # Assigning a type to the variable 'self' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IsHashable.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        IsHashable.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IsHashable.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        IsHashable.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'IsHashable.stypy__eq__')
        IsHashable.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['type_'])
        IsHashable.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        IsHashable.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IsHashable.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        IsHashable.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        IsHashable.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IsHashable.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IsHashable.stypy__eq__', ['type_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['type_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 287):
        
        # Assigning a Name to a Attribute (line 287):
        # Getting the type of 'type_' (line 287)
        type__18900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 21), 'type_')
        # Getting the type of 'self' (line 287)
        self_18901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'self')
        # Setting the type of the member 'type_' of a type (line 287)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), self_18901, 'type_', type__18900)
        
        # Call to issubclass(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 'type_' (line 288)
        type__18903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 22), 'type_', False)
        # Getting the type of 'collections' (line 288)
        collections_18904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 29), 'collections', False)
        # Obtaining the member 'Hashable' of a type (line 288)
        Hashable_18905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 29), collections_18904, 'Hashable')
        # Processing the call keyword arguments (line 288)
        kwargs_18906 = {}
        # Getting the type of 'issubclass' (line 288)
        issubclass_18902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 11), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 288)
        issubclass_call_result_18907 = invoke(stypy.reporting.localization.Localization(__file__, 288, 11), issubclass_18902, *[type__18903, Hashable_18905], **kwargs_18906)
        
        # Testing if the type of an if condition is none (line 288)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 288, 8), issubclass_call_result_18907):
            pass
        else:
            
            # Testing the type of an if condition (line 288)
            if_condition_18908 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 8), issubclass_call_result_18907)
            # Assigning a type to the variable 'if_condition_18908' (line 288)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'if_condition_18908', if_condition_18908)
            # SSA begins for if statement (line 288)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'True' (line 289)
            True_18909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 289)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'stypy_return_type', True_18909)
            # SSA join for if statement (line 288)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'False' (line 290)
        False_18910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'stypy_return_type', False_18910)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 286)
        stypy_return_type_18911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18911)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_18911


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 292, 4, False)
        # Assigning a type to the variable 'self' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IsHashable.__call__.__dict__.__setitem__('stypy_localization', localization)
        IsHashable.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IsHashable.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        IsHashable.__call__.__dict__.__setitem__('stypy_function_name', 'IsHashable.__call__')
        IsHashable.__call__.__dict__.__setitem__('stypy_param_names_list', ['localization'])
        IsHashable.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'call_args')
        IsHashable.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'call_kwargs')
        IsHashable.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        IsHashable.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        IsHashable.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IsHashable.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IsHashable.__call__', ['localization'], 'call_args', 'call_kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['localization'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Assigning a Attribute to a Name (line 293):
        
        # Assigning a Attribute to a Name (line 293):
        # Getting the type of 'self' (line 293)
        self_18912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 15), 'self')
        # Obtaining the member 'type_' of a type (line 293)
        type__18913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 15), self_18912, 'type_')
        # Assigning a type to the variable 'temp' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'temp', type__18913)
        
        # Assigning a Name to a Attribute (line 294):
        
        # Assigning a Name to a Attribute (line 294):
        # Getting the type of 'None' (line 294)
        None_18914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 21), 'None')
        # Getting the type of 'self' (line 294)
        self_18915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'self')
        # Setting the type of the member 'type_' of a type (line 294)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), self_18915, 'type_', None_18914)
        # Getting the type of 'temp' (line 296)
        temp_18916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'stypy_return_type', temp_18916)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 292)
        stypy_return_type_18917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18917)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_18917


# Assigning a type to the variable 'IsHashable' (line 277)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 0), 'IsHashable', IsHashable)
# Declaration of the 'TypeOfParam' class
# Getting the type of 'TypeGroup' (line 299)
TypeGroup_18918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 18), 'TypeGroup')
# Getting the type of 'DependentType' (line 299)
DependentType_18919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 29), 'DependentType')

class TypeOfParam(TypeGroup_18918, DependentType_18919, ):
    str_18920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, (-1)), 'str', "\n    This type group is special in the sense that it don't really group any types, only returns the param number\n    passed in the constructor when it is called with a list of parameters. This is really used to simplify several\n    type rules in which the type returned by a member call is equal to the type of one of its parameters\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 306, 4, False)
        # Assigning a type to the variable 'self' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeOfParam.__init__', [], 'param_number', None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'self' (line 307)
        self_18923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 31), 'self', False)
        # Processing the call keyword arguments (line 307)
        kwargs_18924 = {}
        # Getting the type of 'DependentType' (line 307)
        DependentType_18921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'DependentType', False)
        # Obtaining the member '__init__' of a type (line 307)
        init___18922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 8), DependentType_18921, '__init__')
        # Calling __init__(args, kwargs) (line 307)
        init___call_result_18925 = invoke(stypy.reporting.localization.Localization(__file__, 307, 8), init___18922, *[self_18923], **kwargs_18924)
        
        
        # Call to __init__(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 'self' (line 308)
        self_18928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 308)
        list_18929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 308)
        
        # Processing the call keyword arguments (line 308)
        kwargs_18930 = {}
        # Getting the type of 'TypeGroup' (line 308)
        TypeGroup_18926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 308)
        init___18927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 8), TypeGroup_18926, '__init__')
        # Calling __init__(args, kwargs) (line 308)
        init___call_result_18931 = invoke(stypy.reporting.localization.Localization(__file__, 308, 8), init___18927, *[self_18928, list_18929], **kwargs_18930)
        
        
        # Assigning a Subscript to a Attribute (line 309):
        
        # Assigning a Subscript to a Attribute (line 309):
        
        # Obtaining the type of the subscript
        int_18932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 41), 'int')
        # Getting the type of 'param_number' (line 309)
        param_number_18933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 28), 'param_number')
        # Obtaining the member '__getitem__' of a type (line 309)
        getitem___18934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 28), param_number_18933, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 309)
        subscript_call_result_18935 = invoke(stypy.reporting.localization.Localization(__file__, 309, 28), getitem___18934, int_18932)
        
        # Getting the type of 'self' (line 309)
        self_18936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'self')
        # Setting the type of the member 'param_number' of a type (line 309)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 8), self_18936, 'param_number', subscript_call_result_18935)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 311, 4, False)
        # Assigning a type to the variable 'self' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeOfParam.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        TypeOfParam.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeOfParam.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeOfParam.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'TypeOfParam.stypy__eq__')
        TypeOfParam.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['type_'])
        TypeOfParam.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeOfParam.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeOfParam.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeOfParam.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeOfParam.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeOfParam.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeOfParam.stypy__eq__', ['type_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['type_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        # Getting the type of 'False' (line 312)
        False_18937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'stypy_return_type', False_18937)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 311)
        stypy_return_type_18938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18938)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_18938


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 314, 4, False)
        # Assigning a type to the variable 'self' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeOfParam.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        TypeOfParam.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeOfParam.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeOfParam.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'TypeOfParam.stypy__repr__')
        TypeOfParam.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        TypeOfParam.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeOfParam.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeOfParam.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeOfParam.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeOfParam.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeOfParam.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeOfParam.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a BinOp to a Name (line 315):
        
        # Assigning a BinOp to a Name (line 315):
        
        # Call to type(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of 'self' (line 315)
        self_18940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 23), 'self', False)
        # Processing the call keyword arguments (line 315)
        kwargs_18941 = {}
        # Getting the type of 'type' (line 315)
        type_18939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 18), 'type', False)
        # Calling type(args, kwargs) (line 315)
        type_call_result_18942 = invoke(stypy.reporting.localization.Localization(__file__, 315, 18), type_18939, *[self_18940], **kwargs_18941)
        
        # Obtaining the member '__name__' of a type (line 315)
        name___18943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 18), type_call_result_18942, '__name__')
        str_18944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 40), 'str', '(')
        # Applying the binary operator '+' (line 315)
        result_add_18945 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 18), '+', name___18943, str_18944)
        
        # Getting the type of 'self' (line 315)
        self_18946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 46), 'self')
        # Obtaining the member 'param_number' of a type (line 315)
        param_number_18947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 46), self_18946, 'param_number')
        # Applying the binary operator '+' (line 315)
        result_add_18948 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 44), '+', result_add_18945, param_number_18947)
        
        str_18949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 66), 'str', ')')
        # Applying the binary operator '+' (line 315)
        result_add_18950 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 64), '+', result_add_18948, str_18949)
        
        # Assigning a type to the variable 'ret_str' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'ret_str', result_add_18950)
        # Getting the type of 'ret_str' (line 317)
        ret_str_18951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 15), 'ret_str')
        # Assigning a type to the variable 'stypy_return_type' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'stypy_return_type', ret_str_18951)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 314)
        stypy_return_type_18952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18952)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_18952


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 319, 4, False)
        # Assigning a type to the variable 'self' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeOfParam.__call__.__dict__.__setitem__('stypy_localization', localization)
        TypeOfParam.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeOfParam.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeOfParam.__call__.__dict__.__setitem__('stypy_function_name', 'TypeOfParam.__call__')
        TypeOfParam.__call__.__dict__.__setitem__('stypy_param_names_list', ['localization'])
        TypeOfParam.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'call_args')
        TypeOfParam.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'call_kwargs')
        TypeOfParam.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeOfParam.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeOfParam.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeOfParam.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeOfParam.__call__', ['localization'], 'call_args', 'call_kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['localization'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 320)
        self_18953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 28), 'self')
        # Obtaining the member 'param_number' of a type (line 320)
        param_number_18954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 28), self_18953, 'param_number')
        int_18955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 48), 'int')
        # Applying the binary operator '-' (line 320)
        result_sub_18956 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 28), '-', param_number_18954, int_18955)
        
        
        # Obtaining the type of the subscript
        int_18957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 25), 'int')
        # Getting the type of 'call_args' (line 320)
        call_args_18958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 15), 'call_args')
        # Obtaining the member '__getitem__' of a type (line 320)
        getitem___18959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 15), call_args_18958, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 320)
        subscript_call_result_18960 = invoke(stypy.reporting.localization.Localization(__file__, 320, 15), getitem___18959, int_18957)
        
        # Obtaining the member '__getitem__' of a type (line 320)
        getitem___18961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 15), subscript_call_result_18960, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 320)
        subscript_call_result_18962 = invoke(stypy.reporting.localization.Localization(__file__, 320, 15), getitem___18961, result_sub_18956)
        
        # Assigning a type to the variable 'stypy_return_type' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'stypy_return_type', subscript_call_result_18962)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 319)
        stypy_return_type_18963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18963)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_18963


# Assigning a type to the variable 'TypeOfParam' (line 299)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 0), 'TypeOfParam', TypeOfParam)
# Declaration of the 'Callable' class
# Getting the type of 'TypeGroup' (line 323)
TypeGroup_18964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 15), 'TypeGroup')

class Callable(TypeGroup_18964, ):
    str_18965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, (-1)), 'str', '\n    Represent all callable objects (those that define the member __call__)\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 328, 4, False)
        # Assigning a type to the variable 'self' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Callable.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 329)
        # Processing the call arguments (line 329)
        # Getting the type of 'self' (line 329)
        self_18968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 329)
        list_18969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 329)
        
        # Processing the call keyword arguments (line 329)
        kwargs_18970 = {}
        # Getting the type of 'TypeGroup' (line 329)
        TypeGroup_18966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 329)
        init___18967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), TypeGroup_18966, '__init__')
        # Calling __init__(args, kwargs) (line 329)
        init___call_result_18971 = invoke(stypy.reporting.localization.Localization(__file__, 329, 8), init___18967, *[self_18968, list_18969], **kwargs_18970)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 331, 4, False)
        # Assigning a type to the variable 'self' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Callable.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        Callable.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Callable.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Callable.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'Callable.stypy__eq__')
        Callable.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['type_'])
        Callable.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Callable.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Callable.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Callable.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Callable.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Callable.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Callable.stypy__eq__', ['type_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['type_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 332):
        
        # Assigning a Call to a Attribute (line 332):
        
        # Call to get_type_of_member(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'None' (line 332)
        None_18974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 51), 'None', False)
        str_18975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 57), 'str', '__call__')
        # Processing the call keyword arguments (line 332)
        kwargs_18976 = {}
        # Getting the type of 'type_' (line 332)
        type__18972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 26), 'type_', False)
        # Obtaining the member 'get_type_of_member' of a type (line 332)
        get_type_of_member_18973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 26), type__18972, 'get_type_of_member')
        # Calling get_type_of_member(args, kwargs) (line 332)
        get_type_of_member_call_result_18977 = invoke(stypy.reporting.localization.Localization(__file__, 332, 26), get_type_of_member_18973, *[None_18974, str_18975], **kwargs_18976)
        
        # Getting the type of 'self' (line 332)
        self_18978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'self')
        # Setting the type of the member 'member_obj' of a type (line 332)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 8), self_18978, 'member_obj', get_type_of_member_call_result_18977)
        
        # Type idiom detected: calculating its left and rigth part (line 333)
        # Getting the type of 'TypeError' (line 333)
        TypeError_18979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 39), 'TypeError')
        # Getting the type of 'self' (line 333)
        self_18980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 22), 'self')
        # Obtaining the member 'member_obj' of a type (line 333)
        member_obj_18981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 22), self_18980, 'member_obj')
        
        (may_be_18982, more_types_in_union_18983) = may_be_subtype(TypeError_18979, member_obj_18981)

        if may_be_18982:

            if more_types_in_union_18983:
                # Runtime conditional SSA (line 333)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 333)
            self_18984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'self')
            # Obtaining the member 'member_obj' of a type (line 333)
            member_obj_18985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 8), self_18984, 'member_obj')
            # Setting the type of the member 'member_obj' of a type (line 333)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 8), self_18984, 'member_obj', remove_not_subtype_from_union(member_obj_18981, TypeError))
            # Getting the type of 'False' (line 334)
            False_18986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 334)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'stypy_return_type', False_18986)

            if more_types_in_union_18983:
                # SSA join for if statement (line 333)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'True' (line 336)
        True_18987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'stypy_return_type', True_18987)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 331)
        stypy_return_type_18988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18988)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_18988


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 338, 4, False)
        # Assigning a type to the variable 'self' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Callable.__call__.__dict__.__setitem__('stypy_localization', localization)
        Callable.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Callable.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Callable.__call__.__dict__.__setitem__('stypy_function_name', 'Callable.__call__')
        Callable.__call__.__dict__.__setitem__('stypy_param_names_list', ['localization'])
        Callable.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'call_args')
        Callable.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'call_kwargs')
        Callable.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Callable.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Callable.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Callable.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Callable.__call__', ['localization'], 'call_args', 'call_kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['localization'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Assigning a Attribute to a Name (line 339):
        
        # Assigning a Attribute to a Name (line 339):
        # Getting the type of 'self' (line 339)
        self_18989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 15), 'self')
        # Obtaining the member 'member_obj' of a type (line 339)
        member_obj_18990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 15), self_18989, 'member_obj')
        # Assigning a type to the variable 'temp' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'temp', member_obj_18990)
        
        # Assigning a Name to a Attribute (line 340):
        
        # Assigning a Name to a Attribute (line 340):
        # Getting the type of 'None' (line 340)
        None_18991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 26), 'None')
        # Getting the type of 'self' (line 340)
        self_18992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'self')
        # Setting the type of the member 'member_obj' of a type (line 340)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), self_18992, 'member_obj', None_18991)
        # Getting the type of 'temp' (line 342)
        temp_18993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'stypy_return_type', temp_18993)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 338)
        stypy_return_type_18994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18994)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_18994


# Assigning a type to the variable 'Callable' (line 323)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 0), 'Callable', Callable)
# Declaration of the 'TypeObject' class
# Getting the type of 'TypeGroup' (line 345)
TypeGroup_18995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 17), 'TypeGroup')

class TypeObject(TypeGroup_18995, ):
    str_18996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, (-1)), 'str', '\n    Represent type and types.ClassType types\n    ')
    
    # Assigning a List to a Name (line 349):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 351, 4, False)
        # Assigning a type to the variable 'self' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeObject.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 352)
        # Processing the call arguments (line 352)
        # Getting the type of 'self' (line 352)
        self_18999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 352)
        list_19000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 352)
        
        # Processing the call keyword arguments (line 352)
        kwargs_19001 = {}
        # Getting the type of 'TypeGroup' (line 352)
        TypeGroup_18997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 352)
        init___18998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 8), TypeGroup_18997, '__init__')
        # Calling __init__(args, kwargs) (line 352)
        init___call_result_19002 = invoke(stypy.reporting.localization.Localization(__file__, 352, 8), init___18998, *[self_18999, list_19000], **kwargs_19001)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 354, 4, False)
        # Assigning a type to the variable 'self' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeObject.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        TypeObject.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeObject.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeObject.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'TypeObject.stypy__eq__')
        TypeObject.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['type_'])
        TypeObject.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeObject.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeObject.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeObject.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeObject.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeObject.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeObject.stypy__eq__', ['type_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['type_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 355):
        
        # Assigning a Call to a Attribute (line 355):
        
        # Call to type(...): (line 355)
        # Processing the call arguments (line 355)
        
        # Call to get_python_type(...): (line 355)
        # Processing the call keyword arguments (line 355)
        kwargs_19006 = {}
        # Getting the type of 'type_' (line 355)
        type__19004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 31), 'type_', False)
        # Obtaining the member 'get_python_type' of a type (line 355)
        get_python_type_19005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 31), type__19004, 'get_python_type')
        # Calling get_python_type(args, kwargs) (line 355)
        get_python_type_call_result_19007 = invoke(stypy.reporting.localization.Localization(__file__, 355, 31), get_python_type_19005, *[], **kwargs_19006)
        
        # Processing the call keyword arguments (line 355)
        kwargs_19008 = {}
        # Getting the type of 'type' (line 355)
        type_19003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 26), 'type', False)
        # Calling type(args, kwargs) (line 355)
        type_call_result_19009 = invoke(stypy.reporting.localization.Localization(__file__, 355, 26), type_19003, *[get_python_type_call_result_19007], **kwargs_19008)
        
        # Getting the type of 'self' (line 355)
        self_19010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'self')
        # Setting the type of the member 'member_obj' of a type (line 355)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 8), self_19010, 'member_obj', type_call_result_19009)
        
        # Getting the type of 'self' (line 356)
        self_19011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 11), 'self')
        # Obtaining the member 'member_obj' of a type (line 356)
        member_obj_19012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 11), self_19011, 'member_obj')
        # Getting the type of 'TypeObject' (line 356)
        TypeObject_19013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 30), 'TypeObject')
        # Obtaining the member 'type_objs' of a type (line 356)
        type_objs_19014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 30), TypeObject_19013, 'type_objs')
        # Applying the binary operator 'in' (line 356)
        result_contains_19015 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 11), 'in', member_obj_19012, type_objs_19014)
        
        # Testing if the type of an if condition is none (line 356)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 356, 8), result_contains_19015):
            pass
        else:
            
            # Testing the type of an if condition (line 356)
            if_condition_19016 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 356, 8), result_contains_19015)
            # Assigning a type to the variable 'if_condition_19016' (line 356)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'if_condition_19016', if_condition_19016)
            # SSA begins for if statement (line 356)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to is_type_instance(...): (line 357)
            # Processing the call keyword arguments (line 357)
            kwargs_19019 = {}
            # Getting the type of 'type_' (line 357)
            type__19017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 23), 'type_', False)
            # Obtaining the member 'is_type_instance' of a type (line 357)
            is_type_instance_19018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 23), type__19017, 'is_type_instance')
            # Calling is_type_instance(args, kwargs) (line 357)
            is_type_instance_call_result_19020 = invoke(stypy.reporting.localization.Localization(__file__, 357, 23), is_type_instance_19018, *[], **kwargs_19019)
            
            # Applying the 'not' unary operator (line 357)
            result_not__19021 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 19), 'not', is_type_instance_call_result_19020)
            
            # Assigning a type to the variable 'stypy_return_type' (line 357)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'stypy_return_type', result_not__19021)
            # SSA join for if statement (line 356)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'False' (line 359)
        False_19022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'stypy_return_type', False_19022)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 354)
        stypy_return_type_19023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19023)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_19023


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 361, 4, False)
        # Assigning a type to the variable 'self' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeObject.__call__.__dict__.__setitem__('stypy_localization', localization)
        TypeObject.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeObject.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeObject.__call__.__dict__.__setitem__('stypy_function_name', 'TypeObject.__call__')
        TypeObject.__call__.__dict__.__setitem__('stypy_param_names_list', ['localization'])
        TypeObject.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'call_args')
        TypeObject.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'call_kwargs')
        TypeObject.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeObject.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeObject.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeObject.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeObject.__call__', ['localization'], 'call_args', 'call_kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['localization'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Assigning a Attribute to a Name (line 362):
        
        # Assigning a Attribute to a Name (line 362):
        # Getting the type of 'self' (line 362)
        self_19024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 15), 'self')
        # Obtaining the member 'member_obj' of a type (line 362)
        member_obj_19025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 15), self_19024, 'member_obj')
        # Assigning a type to the variable 'temp' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'temp', member_obj_19025)
        
        # Assigning a Name to a Attribute (line 363):
        
        # Assigning a Name to a Attribute (line 363):
        # Getting the type of 'None' (line 363)
        None_19026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 26), 'None')
        # Getting the type of 'self' (line 363)
        self_19027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'self')
        # Setting the type of the member 'member_obj' of a type (line 363)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 8), self_19027, 'member_obj', None_19026)
        # Getting the type of 'temp' (line 365)
        temp_19028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'stypy_return_type', temp_19028)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 361)
        stypy_return_type_19029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19029)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_19029


# Assigning a type to the variable 'TypeObject' (line 345)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 0), 'TypeObject', TypeObject)

# Assigning a List to a Name (line 349):

# Obtaining an instance of the builtin type 'list' (line 349)
list_19030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 349)
# Adding element type (line 349)
# Getting the type of 'type' (line 349)
type_19031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 17), 'type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 16), list_19030, type_19031)
# Adding element type (line 349)
# Getting the type of 'types' (line 349)
types_19032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 23), 'types')
# Obtaining the member 'ClassType' of a type (line 349)
ClassType_19033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 23), types_19032, 'ClassType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 16), list_19030, ClassType_19033)

# Getting the type of 'TypeObject'
TypeObject_19034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeObject')
# Setting the type of the member 'type_objs' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeObject_19034, 'type_objs', list_19030)
# Declaration of the 'InstanceOfType' class
# Getting the type of 'TypeGroup' (line 368)
TypeGroup_19035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 21), 'TypeGroup')

class InstanceOfType(TypeGroup_19035, ):
    str_19036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, (-1)), 'str', '\n    Represent type and types.ClassType types\n    ')
    
    # Assigning a List to a Name (line 372):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 374, 4, False)
        # Assigning a type to the variable 'self' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstanceOfType.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 375)
        # Processing the call arguments (line 375)
        # Getting the type of 'self' (line 375)
        self_19039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 375)
        list_19040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 375)
        
        # Processing the call keyword arguments (line 375)
        kwargs_19041 = {}
        # Getting the type of 'TypeGroup' (line 375)
        TypeGroup_19037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 375)
        init___19038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 8), TypeGroup_19037, '__init__')
        # Calling __init__(args, kwargs) (line 375)
        init___call_result_19042 = invoke(stypy.reporting.localization.Localization(__file__, 375, 8), init___19038, *[self_19039, list_19040], **kwargs_19041)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 377, 4, False)
        # Assigning a type to the variable 'self' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InstanceOfType.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        InstanceOfType.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InstanceOfType.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        InstanceOfType.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'InstanceOfType.stypy__eq__')
        InstanceOfType.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['type_'])
        InstanceOfType.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        InstanceOfType.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InstanceOfType.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        InstanceOfType.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        InstanceOfType.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InstanceOfType.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstanceOfType.stypy__eq__', ['type_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['type_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 378):
        
        # Assigning a Call to a Attribute (line 378):
        
        # Call to type(...): (line 378)
        # Processing the call arguments (line 378)
        
        # Call to get_python_type(...): (line 378)
        # Processing the call keyword arguments (line 378)
        kwargs_19046 = {}
        # Getting the type of 'type_' (line 378)
        type__19044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 31), 'type_', False)
        # Obtaining the member 'get_python_type' of a type (line 378)
        get_python_type_19045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 31), type__19044, 'get_python_type')
        # Calling get_python_type(args, kwargs) (line 378)
        get_python_type_call_result_19047 = invoke(stypy.reporting.localization.Localization(__file__, 378, 31), get_python_type_19045, *[], **kwargs_19046)
        
        # Processing the call keyword arguments (line 378)
        kwargs_19048 = {}
        # Getting the type of 'type' (line 378)
        type_19043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 26), 'type', False)
        # Calling type(args, kwargs) (line 378)
        type_call_result_19049 = invoke(stypy.reporting.localization.Localization(__file__, 378, 26), type_19043, *[get_python_type_call_result_19047], **kwargs_19048)
        
        # Getting the type of 'self' (line 378)
        self_19050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'self')
        # Setting the type of the member 'member_obj' of a type (line 378)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 8), self_19050, 'member_obj', type_call_result_19049)
        
        # Getting the type of 'self' (line 379)
        self_19051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 11), 'self')
        # Obtaining the member 'member_obj' of a type (line 379)
        member_obj_19052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 11), self_19051, 'member_obj')
        # Getting the type of 'TypeObject' (line 379)
        TypeObject_19053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 30), 'TypeObject')
        # Obtaining the member 'type_objs' of a type (line 379)
        type_objs_19054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 30), TypeObject_19053, 'type_objs')
        # Applying the binary operator 'in' (line 379)
        result_contains_19055 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 11), 'in', member_obj_19052, type_objs_19054)
        
        # Testing if the type of an if condition is none (line 379)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 379, 8), result_contains_19055):
            pass
        else:
            
            # Testing the type of an if condition (line 379)
            if_condition_19056 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 379, 8), result_contains_19055)
            # Assigning a type to the variable 'if_condition_19056' (line 379)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'if_condition_19056', if_condition_19056)
            # SSA begins for if statement (line 379)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to is_type_instance(...): (line 380)
            # Processing the call keyword arguments (line 380)
            kwargs_19059 = {}
            # Getting the type of 'type_' (line 380)
            type__19057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 19), 'type_', False)
            # Obtaining the member 'is_type_instance' of a type (line 380)
            is_type_instance_19058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 19), type__19057, 'is_type_instance')
            # Calling is_type_instance(args, kwargs) (line 380)
            is_type_instance_call_result_19060 = invoke(stypy.reporting.localization.Localization(__file__, 380, 19), is_type_instance_19058, *[], **kwargs_19059)
            
            # Assigning a type to the variable 'stypy_return_type' (line 380)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'stypy_return_type', is_type_instance_call_result_19060)
            # SSA join for if statement (line 379)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'False' (line 382)
        False_19061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'stypy_return_type', False_19061)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 377)
        stypy_return_type_19062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19062)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_19062


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 384, 4, False)
        # Assigning a type to the variable 'self' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InstanceOfType.__call__.__dict__.__setitem__('stypy_localization', localization)
        InstanceOfType.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InstanceOfType.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        InstanceOfType.__call__.__dict__.__setitem__('stypy_function_name', 'InstanceOfType.__call__')
        InstanceOfType.__call__.__dict__.__setitem__('stypy_param_names_list', ['localization'])
        InstanceOfType.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'call_args')
        InstanceOfType.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'call_kwargs')
        InstanceOfType.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        InstanceOfType.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        InstanceOfType.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InstanceOfType.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstanceOfType.__call__', ['localization'], 'call_args', 'call_kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['localization'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Assigning a Attribute to a Name (line 385):
        
        # Assigning a Attribute to a Name (line 385):
        # Getting the type of 'self' (line 385)
        self_19063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 15), 'self')
        # Obtaining the member 'member_obj' of a type (line 385)
        member_obj_19064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 15), self_19063, 'member_obj')
        # Assigning a type to the variable 'temp' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'temp', member_obj_19064)
        
        # Assigning a Name to a Attribute (line 386):
        
        # Assigning a Name to a Attribute (line 386):
        # Getting the type of 'None' (line 386)
        None_19065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 26), 'None')
        # Getting the type of 'self' (line 386)
        self_19066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'self')
        # Setting the type of the member 'member_obj' of a type (line 386)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 8), self_19066, 'member_obj', None_19065)
        # Getting the type of 'temp' (line 388)
        temp_19067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'stypy_return_type', temp_19067)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 384)
        stypy_return_type_19068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19068)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_19068


# Assigning a type to the variable 'InstanceOfType' (line 368)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 0), 'InstanceOfType', InstanceOfType)

# Assigning a List to a Name (line 372):

# Obtaining an instance of the builtin type 'list' (line 372)
list_19069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 372)
# Adding element type (line 372)
# Getting the type of 'type' (line 372)
type_19070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 17), 'type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 16), list_19069, type_19070)
# Adding element type (line 372)
# Getting the type of 'types' (line 372)
types_19071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 23), 'types')
# Obtaining the member 'ClassType' of a type (line 372)
ClassType_19072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 23), types_19071, 'ClassType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 16), list_19069, ClassType_19072)

# Getting the type of 'InstanceOfType'
InstanceOfType_19073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'InstanceOfType')
# Setting the type of the member 'type_objs' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), InstanceOfType_19073, 'type_objs', list_19069)
# Declaration of the 'VarArgType' class
# Getting the type of 'TypeGroup' (line 391)
TypeGroup_19074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 17), 'TypeGroup')

class VarArgType(TypeGroup_19074, ):
    str_19075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, (-1)), 'str', '\n    Special type group indicating that a callable has an unlimited amount of parameters\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 396, 4, False)
        # Assigning a type to the variable 'self' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarArgType.__init__', [], 'types_', None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 'self' (line 397)
        self_19078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 397)
        list_19079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 397)
        
        # Processing the call keyword arguments (line 397)
        kwargs_19080 = {}
        # Getting the type of 'TypeGroup' (line 397)
        TypeGroup_19076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 397)
        init___19077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 8), TypeGroup_19076, '__init__')
        # Calling __init__(args, kwargs) (line 397)
        init___call_result_19081 = invoke(stypy.reporting.localization.Localization(__file__, 397, 8), init___19077, *[self_19078, list_19079], **kwargs_19080)
        
        
        # Assigning a Name to a Attribute (line 398):
        
        # Assigning a Name to a Attribute (line 398):
        # Getting the type of 'types_' (line 398)
        types__19082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 21), 'types_')
        # Getting the type of 'self' (line 398)
        self_19083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'self')
        # Setting the type of the member 'types' of a type (line 398)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), self_19083, 'types', types__19082)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 400, 4, False)
        # Assigning a type to the variable 'self' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarArgType.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        VarArgType.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarArgType.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarArgType.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'VarArgType.stypy__eq__')
        VarArgType.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['type_'])
        VarArgType.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarArgType.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarArgType.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarArgType.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarArgType.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarArgType.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarArgType.stypy__eq__', ['type_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['type_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        # Getting the type of 'True' (line 401)
        True_19084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'stypy_return_type', True_19084)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 400)
        stypy_return_type_19085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19085)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_19085


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 403, 4, False)
        # Assigning a type to the variable 'self' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarArgType.__call__.__dict__.__setitem__('stypy_localization', localization)
        VarArgType.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarArgType.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarArgType.__call__.__dict__.__setitem__('stypy_function_name', 'VarArgType.__call__')
        VarArgType.__call__.__dict__.__setitem__('stypy_param_names_list', ['localization'])
        VarArgType.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'call_args')
        VarArgType.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'call_kwargs')
        VarArgType.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarArgType.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarArgType.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarArgType.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarArgType.__call__', ['localization'], 'call_args', 'call_kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['localization'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Assigning a Attribute to a Name (line 404):
        
        # Assigning a Attribute to a Name (line 404):
        # Getting the type of 'self' (line 404)
        self_19086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 15), 'self')
        # Obtaining the member 'type_' of a type (line 404)
        type__19087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 15), self_19086, 'type_')
        # Assigning a type to the variable 'temp' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'temp', type__19087)
        
        # Assigning a Name to a Attribute (line 405):
        
        # Assigning a Name to a Attribute (line 405):
        # Getting the type of 'None' (line 405)
        None_19088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 21), 'None')
        # Getting the type of 'self' (line 405)
        self_19089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'self')
        # Setting the type of the member 'type_' of a type (line 405)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 8), self_19089, 'type_', None_19088)
        # Getting the type of 'temp' (line 407)
        temp_19090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'stypy_return_type', temp_19090)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 403)
        stypy_return_type_19091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19091)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_19091


# Assigning a type to the variable 'VarArgType' (line 391)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 0), 'VarArgType', VarArgType)
# Declaration of the 'TypeGroups' class

class TypeGroups:
    str_19092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, (-1)), 'str', '\n    Class to hold definitions of type groups that are composed by lists of known Python types\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 415, 4, False)
        # Assigning a type to the variable 'self' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeGroups.__init__', [], None, None, defaults, varargs, kwargs)

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


    @staticmethod
    @norecursion
    def get_rule_groups(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_rule_groups'
        module_type_store = module_type_store.open_function_context('get_rule_groups', 418, 4, False)
        
        # Passed parameters checking function
        TypeGroups.get_rule_groups.__dict__.__setitem__('stypy_localization', localization)
        TypeGroups.get_rule_groups.__dict__.__setitem__('stypy_type_of_self', None)
        TypeGroups.get_rule_groups.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeGroups.get_rule_groups.__dict__.__setitem__('stypy_function_name', 'get_rule_groups')
        TypeGroups.get_rule_groups.__dict__.__setitem__('stypy_param_names_list', [])
        TypeGroups.get_rule_groups.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeGroups.get_rule_groups.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeGroups.get_rule_groups.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeGroups.get_rule_groups.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeGroups.get_rule_groups.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeGroups.get_rule_groups.__dict__.__setitem__('stypy_declared_arg_number', 0)
        arguments = process_argument_values(localization, None, module_type_store, 'get_rule_groups', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_rule_groups', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_rule_groups(...)' code ##################

        str_19093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, (-1)), 'str', '\n        Obtain all the types defined in this class\n        ')

        @norecursion
        def filter_func(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'filter_func'
            module_type_store = module_type_store.open_function_context('filter_func', 424, 8, False)
            
            # Passed parameters checking function
            filter_func.stypy_localization = localization
            filter_func.stypy_type_of_self = None
            filter_func.stypy_type_store = module_type_store
            filter_func.stypy_function_name = 'filter_func'
            filter_func.stypy_param_names_list = ['element']
            filter_func.stypy_varargs_param_name = None
            filter_func.stypy_kwargs_param_name = None
            filter_func.stypy_call_defaults = defaults
            filter_func.stypy_call_varargs = varargs
            filter_func.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'filter_func', ['element'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'filter_func', localization, ['element'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'filter_func(...)' code ##################

            
            # Call to isinstance(...): (line 425)
            # Processing the call arguments (line 425)
            # Getting the type of 'element' (line 425)
            element_19095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 30), 'element', False)
            # Getting the type of 'list' (line 425)
            list_19096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 39), 'list', False)
            # Processing the call keyword arguments (line 425)
            kwargs_19097 = {}
            # Getting the type of 'isinstance' (line 425)
            isinstance_19094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 425)
            isinstance_call_result_19098 = invoke(stypy.reporting.localization.Localization(__file__, 425, 19), isinstance_19094, *[element_19095, list_19096], **kwargs_19097)
            
            # Assigning a type to the variable 'stypy_return_type' (line 425)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'stypy_return_type', isinstance_call_result_19098)
            
            # ################# End of 'filter_func(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'filter_func' in the type store
            # Getting the type of 'stypy_return_type' (line 424)
            stypy_return_type_19099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_19099)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'filter_func'
            return stypy_return_type_19099

        # Assigning a type to the variable 'filter_func' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'filter_func', filter_func)
        
        # Call to filter(...): (line 427)
        # Processing the call arguments (line 427)

        @norecursion
        def _stypy_temp_lambda_23(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_23'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_23', 427, 22, True)
            # Passed parameters checking function
            _stypy_temp_lambda_23.stypy_localization = localization
            _stypy_temp_lambda_23.stypy_type_of_self = None
            _stypy_temp_lambda_23.stypy_type_store = module_type_store
            _stypy_temp_lambda_23.stypy_function_name = '_stypy_temp_lambda_23'
            _stypy_temp_lambda_23.stypy_param_names_list = ['member']
            _stypy_temp_lambda_23.stypy_varargs_param_name = None
            _stypy_temp_lambda_23.stypy_kwargs_param_name = None
            _stypy_temp_lambda_23.stypy_call_defaults = defaults
            _stypy_temp_lambda_23.stypy_call_varargs = varargs
            _stypy_temp_lambda_23.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_23', ['member'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_23', ['member'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to filter_func(...): (line 427)
            # Processing the call arguments (line 427)
            
            # Call to getattr(...): (line 427)
            # Processing the call arguments (line 427)
            # Getting the type of 'TypeGroups' (line 427)
            TypeGroups_19103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 57), 'TypeGroups', False)
            # Getting the type of 'member' (line 427)
            member_19104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 69), 'member', False)
            # Processing the call keyword arguments (line 427)
            kwargs_19105 = {}
            # Getting the type of 'getattr' (line 427)
            getattr_19102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 49), 'getattr', False)
            # Calling getattr(args, kwargs) (line 427)
            getattr_call_result_19106 = invoke(stypy.reporting.localization.Localization(__file__, 427, 49), getattr_19102, *[TypeGroups_19103, member_19104], **kwargs_19105)
            
            # Processing the call keyword arguments (line 427)
            kwargs_19107 = {}
            # Getting the type of 'filter_func' (line 427)
            filter_func_19101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 37), 'filter_func', False)
            # Calling filter_func(args, kwargs) (line 427)
            filter_func_call_result_19108 = invoke(stypy.reporting.localization.Localization(__file__, 427, 37), filter_func_19101, *[getattr_call_result_19106], **kwargs_19107)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 427)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 22), 'stypy_return_type', filter_func_call_result_19108)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_23' in the type store
            # Getting the type of 'stypy_return_type' (line 427)
            stypy_return_type_19109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 22), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_19109)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_23'
            return stypy_return_type_19109

        # Assigning a type to the variable '_stypy_temp_lambda_23' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 22), '_stypy_temp_lambda_23', _stypy_temp_lambda_23)
        # Getting the type of '_stypy_temp_lambda_23' (line 427)
        _stypy_temp_lambda_23_19110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 22), '_stypy_temp_lambda_23')
        # Getting the type of 'TypeGroups' (line 427)
        TypeGroups_19111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 79), 'TypeGroups', False)
        # Obtaining the member '__dict__' of a type (line 427)
        dict___19112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 79), TypeGroups_19111, '__dict__')
        # Processing the call keyword arguments (line 427)
        kwargs_19113 = {}
        # Getting the type of 'filter' (line 427)
        filter_19100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 15), 'filter', False)
        # Calling filter(args, kwargs) (line 427)
        filter_call_result_19114 = invoke(stypy.reporting.localization.Localization(__file__, 427, 15), filter_19100, *[_stypy_temp_lambda_23_19110, dict___19112], **kwargs_19113)
        
        # Assigning a type to the variable 'stypy_return_type' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'stypy_return_type', filter_call_result_19114)
        
        # ################# End of 'get_rule_groups(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_rule_groups' in the type store
        # Getting the type of 'stypy_return_type' (line 418)
        stypy_return_type_19115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19115)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_rule_groups'
        return stypy_return_type_19115

    
    # Assigning a List to a Name (line 430):
    
    # Assigning a List to a Name (line 433):
    
    # Assigning a List to a Name (line 436):
    
    # Assigning a List to a Name (line 439):
    
    # Assigning a List to a Name (line 442):
    
    # Assigning a List to a Name (line 445):
    
    # Assigning a List to a Name (line 468):

# Assigning a type to the variable 'TypeGroups' (line 410)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 0), 'TypeGroups', TypeGroups)

# Assigning a List to a Name (line 430):

# Obtaining an instance of the builtin type 'list' (line 430)
list_19116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 430)
# Adding element type (line 430)
# Getting the type of 'int' (line 430)
int_19117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 17), list_19116, int_19117)
# Adding element type (line 430)
# Getting the type of 'long' (line 430)
long_19118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 23), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 17), list_19116, long_19118)
# Adding element type (line 430)
# Getting the type of 'float' (line 430)
float_19119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 29), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 17), list_19116, float_19119)
# Adding element type (line 430)
# Getting the type of 'bool' (line 430)
bool_19120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 36), 'bool')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 17), list_19116, bool_19120)

# Getting the type of 'TypeGroups'
TypeGroups_19121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeGroups')
# Setting the type of the member 'RealNumber' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeGroups_19121, 'RealNumber', list_19116)

# Assigning a List to a Name (line 433):

# Obtaining an instance of the builtin type 'list' (line 433)
list_19122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 433)
# Adding element type (line 433)
# Getting the type of 'int' (line 433)
int_19123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 13), list_19122, int_19123)
# Adding element type (line 433)
# Getting the type of 'long' (line 433)
long_19124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 19), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 13), list_19122, long_19124)
# Adding element type (line 433)
# Getting the type of 'float' (line 433)
float_19125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 25), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 13), list_19122, float_19125)
# Adding element type (line 433)
# Getting the type of 'bool' (line 433)
bool_19126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 32), 'bool')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 13), list_19122, bool_19126)
# Adding element type (line 433)
# Getting the type of 'complex' (line 433)
complex_19127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 38), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 13), list_19122, complex_19127)

# Getting the type of 'TypeGroups'
TypeGroups_19128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeGroups')
# Setting the type of the member 'Number' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeGroups_19128, 'Number', list_19122)

# Assigning a List to a Name (line 436):

# Obtaining an instance of the builtin type 'list' (line 436)
list_19129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 436)
# Adding element type (line 436)
# Getting the type of 'int' (line 436)
int_19130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 14), list_19129, int_19130)
# Adding element type (line 436)
# Getting the type of 'long' (line 436)
long_19131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 20), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 14), list_19129, long_19131)
# Adding element type (line 436)
# Getting the type of 'bool' (line 436)
bool_19132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 26), 'bool')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 14), list_19129, bool_19132)

# Getting the type of 'TypeGroups'
TypeGroups_19133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeGroups')
# Setting the type of the member 'Integer' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeGroups_19133, 'Integer', list_19129)

# Assigning a List to a Name (line 439):

# Obtaining an instance of the builtin type 'list' (line 439)
list_19134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 439)
# Adding element type (line 439)
# Getting the type of 'str' (line 439)
str_19135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 11), 'str')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 10), list_19134, str_19135)
# Adding element type (line 439)
# Getting the type of 'unicode' (line 439)
unicode_19136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 16), 'unicode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 10), list_19134, unicode_19136)
# Adding element type (line 439)
# Getting the type of 'buffer' (line 439)
buffer_19137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 25), 'buffer')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 10), list_19134, buffer_19137)

# Getting the type of 'TypeGroups'
TypeGroups_19138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeGroups')
# Setting the type of the member 'Str' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeGroups_19138, 'Str', list_19134)

# Assigning a List to a Name (line 442):

# Obtaining an instance of the builtin type 'list' (line 442)
list_19139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 442)
# Adding element type (line 442)
# Getting the type of 'buffer' (line 442)
buffer_19140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 20), 'buffer')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 19), list_19139, buffer_19140)
# Adding element type (line 442)
# Getting the type of 'bytearray' (line 442)
bytearray_19141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 28), 'bytearray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 19), list_19139, bytearray_19141)
# Adding element type (line 442)
# Getting the type of 'str' (line 442)
str_19142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 39), 'str')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 19), list_19139, str_19142)
# Adding element type (line 442)
# Getting the type of 'memoryview' (line 442)
memoryview_19143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 44), 'memoryview')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 19), list_19139, memoryview_19143)

# Getting the type of 'TypeGroups'
TypeGroups_19144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeGroups')
# Setting the type of the member 'ByteSequence' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeGroups_19144, 'ByteSequence', list_19139)

# Assigning a List to a Name (line 445):

# Obtaining an instance of the builtin type 'list' (line 445)
list_19145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 445)
# Adding element type (line 445)
# Getting the type of 'list' (line 446)
list_19146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'list')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_19145, list_19146)
# Adding element type (line 445)
# Getting the type of 'dict' (line 447)
dict_19147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'dict')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_19145, dict_19147)
# Adding element type (line 445)
# Getting the type of 'ExtraTypeDefinitions' (line 448)
ExtraTypeDefinitions_19148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'tupleiterator' of a type (line 448)
tupleiterator_19149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 8), ExtraTypeDefinitions_19148, 'tupleiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_19145, tupleiterator_19149)
# Adding element type (line 445)
# Getting the type of 'ExtraTypeDefinitions' (line 449)
ExtraTypeDefinitions_19150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'dict_values' of a type (line 449)
dict_values_19151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 8), ExtraTypeDefinitions_19150, 'dict_values')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_19145, dict_values_19151)
# Adding element type (line 445)
# Getting the type of 'frozenset' (line 450)
frozenset_19152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'frozenset')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_19145, frozenset_19152)
# Adding element type (line 445)
# Getting the type of 'ExtraTypeDefinitions' (line 451)
ExtraTypeDefinitions_19153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'rangeiterator' of a type (line 451)
rangeiterator_19154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 8), ExtraTypeDefinitions_19153, 'rangeiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_19145, rangeiterator_19154)
# Adding element type (line 445)
# Getting the type of 'types' (line 452)
types_19155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'types')
# Obtaining the member 'GeneratorType' of a type (line 452)
GeneratorType_19156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 8), types_19155, 'GeneratorType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_19145, GeneratorType_19156)
# Adding element type (line 445)
# Getting the type of 'enumerate' (line 453)
enumerate_19157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'enumerate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_19145, enumerate_19157)
# Adding element type (line 445)
# Getting the type of 'bytearray' (line 454)
bytearray_19158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'bytearray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_19145, bytearray_19158)
# Adding element type (line 445)
# Getting the type of 'iter' (line 455)
iter_19159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'iter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_19145, iter_19159)
# Adding element type (line 445)
# Getting the type of 'reversed' (line 456)
reversed_19160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'reversed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_19145, reversed_19160)
# Adding element type (line 445)
# Getting the type of 'ExtraTypeDefinitions' (line 457)
ExtraTypeDefinitions_19161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'dictionary_keyiterator' of a type (line 457)
dictionary_keyiterator_19162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 8), ExtraTypeDefinitions_19161, 'dictionary_keyiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_19145, dictionary_keyiterator_19162)
# Adding element type (line 445)
# Getting the type of 'ExtraTypeDefinitions' (line 458)
ExtraTypeDefinitions_19163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'bytearray_iterator' of a type (line 458)
bytearray_iterator_19164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 8), ExtraTypeDefinitions_19163, 'bytearray_iterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_19145, bytearray_iterator_19164)
# Adding element type (line 445)
# Getting the type of 'ExtraTypeDefinitions' (line 459)
ExtraTypeDefinitions_19165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'dictionary_valueiterator' of a type (line 459)
dictionary_valueiterator_19166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 8), ExtraTypeDefinitions_19165, 'dictionary_valueiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_19145, dictionary_valueiterator_19166)
# Adding element type (line 445)
# Getting the type of 'ExtraTypeDefinitions' (line 460)
ExtraTypeDefinitions_19167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'dictionary_itemiterator' of a type (line 460)
dictionary_itemiterator_19168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 8), ExtraTypeDefinitions_19167, 'dictionary_itemiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_19145, dictionary_itemiterator_19168)
# Adding element type (line 445)
# Getting the type of 'ExtraTypeDefinitions' (line 461)
ExtraTypeDefinitions_19169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'listiterator' of a type (line 461)
listiterator_19170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 8), ExtraTypeDefinitions_19169, 'listiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_19145, listiterator_19170)
# Adding element type (line 445)
# Getting the type of 'ExtraTypeDefinitions' (line 462)
ExtraTypeDefinitions_19171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'listreverseiterator' of a type (line 462)
listreverseiterator_19172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 8), ExtraTypeDefinitions_19171, 'listreverseiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_19145, listreverseiterator_19172)
# Adding element type (line 445)
# Getting the type of 'tuple' (line 463)
tuple_19173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'tuple')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_19145, tuple_19173)
# Adding element type (line 445)
# Getting the type of 'set' (line 464)
set_19174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'set')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_19145, set_19174)
# Adding element type (line 445)
# Getting the type of 'xrange' (line 465)
xrange_19175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'xrange')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_19145, xrange_19175)

# Getting the type of 'TypeGroups'
TypeGroups_19176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeGroups')
# Setting the type of the member 'IterableDataStructure' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeGroups_19176, 'IterableDataStructure', list_19145)

# Assigning a List to a Name (line 468):

# Obtaining an instance of the builtin type 'list' (line 468)
list_19177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 468)
# Adding element type (line 468)
# Getting the type of 'list' (line 469)
list_19178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'list')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_19177, list_19178)
# Adding element type (line 468)
# Getting the type of 'dict' (line 470)
dict_19179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'dict')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_19177, dict_19179)
# Adding element type (line 468)
# Getting the type of 'ExtraTypeDefinitions' (line 471)
ExtraTypeDefinitions_19180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'tupleiterator' of a type (line 471)
tupleiterator_19181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 8), ExtraTypeDefinitions_19180, 'tupleiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_19177, tupleiterator_19181)
# Adding element type (line 468)
# Getting the type of 'ExtraTypeDefinitions' (line 472)
ExtraTypeDefinitions_19182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'dict_values' of a type (line 472)
dict_values_19183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 8), ExtraTypeDefinitions_19182, 'dict_values')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_19177, dict_values_19183)
# Adding element type (line 468)
# Getting the type of 'frozenset' (line 473)
frozenset_19184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'frozenset')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_19177, frozenset_19184)
# Adding element type (line 468)
# Getting the type of 'ExtraTypeDefinitions' (line 474)
ExtraTypeDefinitions_19185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'rangeiterator' of a type (line 474)
rangeiterator_19186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 8), ExtraTypeDefinitions_19185, 'rangeiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_19177, rangeiterator_19186)
# Adding element type (line 468)
# Getting the type of 'types' (line 475)
types_19187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'types')
# Obtaining the member 'GeneratorType' of a type (line 475)
GeneratorType_19188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 8), types_19187, 'GeneratorType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_19177, GeneratorType_19188)
# Adding element type (line 468)
# Getting the type of 'enumerate' (line 476)
enumerate_19189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'enumerate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_19177, enumerate_19189)
# Adding element type (line 468)
# Getting the type of 'bytearray' (line 477)
bytearray_19190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'bytearray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_19177, bytearray_19190)
# Adding element type (line 468)
# Getting the type of 'iter' (line 478)
iter_19191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'iter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_19177, iter_19191)
# Adding element type (line 468)
# Getting the type of 'reversed' (line 479)
reversed_19192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'reversed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_19177, reversed_19192)
# Adding element type (line 468)
# Getting the type of 'ExtraTypeDefinitions' (line 480)
ExtraTypeDefinitions_19193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'dictionary_keyiterator' of a type (line 480)
dictionary_keyiterator_19194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 8), ExtraTypeDefinitions_19193, 'dictionary_keyiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_19177, dictionary_keyiterator_19194)
# Adding element type (line 468)
# Getting the type of 'ExtraTypeDefinitions' (line 481)
ExtraTypeDefinitions_19195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'bytearray_iterator' of a type (line 481)
bytearray_iterator_19196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 8), ExtraTypeDefinitions_19195, 'bytearray_iterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_19177, bytearray_iterator_19196)
# Adding element type (line 468)
# Getting the type of 'ExtraTypeDefinitions' (line 482)
ExtraTypeDefinitions_19197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'dictionary_valueiterator' of a type (line 482)
dictionary_valueiterator_19198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 8), ExtraTypeDefinitions_19197, 'dictionary_valueiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_19177, dictionary_valueiterator_19198)
# Adding element type (line 468)
# Getting the type of 'ExtraTypeDefinitions' (line 483)
ExtraTypeDefinitions_19199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'dictionary_itemiterator' of a type (line 483)
dictionary_itemiterator_19200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 8), ExtraTypeDefinitions_19199, 'dictionary_itemiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_19177, dictionary_itemiterator_19200)
# Adding element type (line 468)
# Getting the type of 'ExtraTypeDefinitions' (line 484)
ExtraTypeDefinitions_19201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'listiterator' of a type (line 484)
listiterator_19202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 8), ExtraTypeDefinitions_19201, 'listiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_19177, listiterator_19202)
# Adding element type (line 468)
# Getting the type of 'ExtraTypeDefinitions' (line 485)
ExtraTypeDefinitions_19203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'listreverseiterator' of a type (line 485)
listreverseiterator_19204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 8), ExtraTypeDefinitions_19203, 'listreverseiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_19177, listreverseiterator_19204)
# Adding element type (line 468)
# Getting the type of 'tuple' (line 486)
tuple_19205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'tuple')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_19177, tuple_19205)
# Adding element type (line 468)
# Getting the type of 'set' (line 487)
set_19206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'set')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_19177, set_19206)
# Adding element type (line 468)
# Getting the type of 'xrange' (line 488)
xrange_19207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'xrange')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_19177, xrange_19207)
# Adding element type (line 468)
# Getting the type of 'memoryview' (line 489)
memoryview_19208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'memoryview')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_19177, memoryview_19208)
# Adding element type (line 468)
# Getting the type of 'types' (line 490)
types_19209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'types')
# Obtaining the member 'DictProxyType' of a type (line 490)
DictProxyType_19210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 8), types_19209, 'DictProxyType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_19177, DictProxyType_19210)

# Getting the type of 'TypeGroups'
TypeGroups_19211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeGroups')
# Setting the type of the member 'IterableObject' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeGroups_19211, 'IterableObject', list_19177)
str_19212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, (-1)), 'str', '\nInstances of type groups. These are the ones that are really used in the type rules, as are concrete usages\nof the previously defined type groups.\n\nNOTE: To interpret instances of type groups, you should take into account the following:\n\n- UndefinedType as expected return type: We cannot statically determine the return\ntype of this method. So we obtain it calling the member, obtaining its type\nand reevaluating the member ruleset again with this type substituting the dependent\none.\n\n- DynamicType as expected return type: We also cannot statically determine the return\ntype of this method. But this time we directly return the return type of the invoked\nmember.\n')

# Assigning a Call to a Name (line 510):

# Assigning a Call to a Name (line 510):

# Call to HasMember(...): (line 510)
# Processing the call arguments (line 510)
str_19214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 23), 'str', '__int__')
# Getting the type of 'int' (line 510)
int_19215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 34), 'int', False)
int_19216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 39), 'int')
# Processing the call keyword arguments (line 510)
kwargs_19217 = {}
# Getting the type of 'HasMember' (line 510)
HasMember_19213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 13), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 510)
HasMember_call_result_19218 = invoke(stypy.reporting.localization.Localization(__file__, 510, 13), HasMember_19213, *[str_19214, int_19215, int_19216], **kwargs_19217)

# Assigning a type to the variable 'CastsToInt' (line 510)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 0), 'CastsToInt', HasMember_call_result_19218)

# Assigning a Call to a Name (line 511):

# Assigning a Call to a Name (line 511):

# Call to HasMember(...): (line 511)
# Processing the call arguments (line 511)
str_19220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 24), 'str', '__long__')
# Getting the type of 'long' (line 511)
long_19221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 36), 'long', False)
int_19222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 42), 'int')
# Processing the call keyword arguments (line 511)
kwargs_19223 = {}
# Getting the type of 'HasMember' (line 511)
HasMember_19219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 14), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 511)
HasMember_call_result_19224 = invoke(stypy.reporting.localization.Localization(__file__, 511, 14), HasMember_19219, *[str_19220, long_19221, int_19222], **kwargs_19223)

# Assigning a type to the variable 'CastsToLong' (line 511)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 0), 'CastsToLong', HasMember_call_result_19224)

# Assigning a Call to a Name (line 512):

# Assigning a Call to a Name (line 512):

# Call to HasMember(...): (line 512)
# Processing the call arguments (line 512)
str_19226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 25), 'str', '__float__')
# Getting the type of 'float' (line 512)
float_19227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 38), 'float', False)
int_19228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 45), 'int')
# Processing the call keyword arguments (line 512)
kwargs_19229 = {}
# Getting the type of 'HasMember' (line 512)
HasMember_19225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 15), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 512)
HasMember_call_result_19230 = invoke(stypy.reporting.localization.Localization(__file__, 512, 15), HasMember_19225, *[str_19226, float_19227, int_19228], **kwargs_19229)

# Assigning a type to the variable 'CastsToFloat' (line 512)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 0), 'CastsToFloat', HasMember_call_result_19230)

# Assigning a Call to a Name (line 513):

# Assigning a Call to a Name (line 513):

# Call to HasMember(...): (line 513)
# Processing the call arguments (line 513)
str_19232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 27), 'str', '__complex__')
# Getting the type of 'complex' (line 513)
complex_19233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 42), 'complex', False)
int_19234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 51), 'int')
# Processing the call keyword arguments (line 513)
kwargs_19235 = {}
# Getting the type of 'HasMember' (line 513)
HasMember_19231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 17), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 513)
HasMember_call_result_19236 = invoke(stypy.reporting.localization.Localization(__file__, 513, 17), HasMember_19231, *[str_19232, complex_19233, int_19234], **kwargs_19235)

# Assigning a type to the variable 'CastsToComplex' (line 513)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 0), 'CastsToComplex', HasMember_call_result_19236)

# Assigning a Call to a Name (line 514):

# Assigning a Call to a Name (line 514):

# Call to HasMember(...): (line 514)
# Processing the call arguments (line 514)
str_19238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 23), 'str', '__oct__')
# Getting the type of 'str' (line 514)
str_19239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 34), 'str', False)
int_19240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 39), 'int')
# Processing the call keyword arguments (line 514)
kwargs_19241 = {}
# Getting the type of 'HasMember' (line 514)
HasMember_19237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 13), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 514)
HasMember_call_result_19242 = invoke(stypy.reporting.localization.Localization(__file__, 514, 13), HasMember_19237, *[str_19238, str_19239, int_19240], **kwargs_19241)

# Assigning a type to the variable 'CastsToOct' (line 514)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 0), 'CastsToOct', HasMember_call_result_19242)

# Assigning a Call to a Name (line 515):

# Assigning a Call to a Name (line 515):

# Call to HasMember(...): (line 515)
# Processing the call arguments (line 515)
str_19244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 23), 'str', '__hex__')
# Getting the type of 'str' (line 515)
str_19245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 34), 'str', False)
int_19246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 39), 'int')
# Processing the call keyword arguments (line 515)
kwargs_19247 = {}
# Getting the type of 'HasMember' (line 515)
HasMember_19243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 13), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 515)
HasMember_call_result_19248 = invoke(stypy.reporting.localization.Localization(__file__, 515, 13), HasMember_19243, *[str_19244, str_19245, int_19246], **kwargs_19247)

# Assigning a type to the variable 'CastsToHex' (line 515)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 0), 'CastsToHex', HasMember_call_result_19248)

# Assigning a Call to a Name (line 516):

# Assigning a Call to a Name (line 516):

# Call to HasMember(...): (line 516)
# Processing the call arguments (line 516)
str_19250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 25), 'str', '__index__')
# Getting the type of 'int' (line 516)
int_19251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 38), 'int', False)
int_19252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 43), 'int')
# Processing the call keyword arguments (line 516)
kwargs_19253 = {}
# Getting the type of 'HasMember' (line 516)
HasMember_19249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 15), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 516)
HasMember_call_result_19254 = invoke(stypy.reporting.localization.Localization(__file__, 516, 15), HasMember_19249, *[str_19250, int_19251, int_19252], **kwargs_19253)

# Assigning a type to the variable 'CastsToIndex' (line 516)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 0), 'CastsToIndex', HasMember_call_result_19254)

# Assigning a Call to a Name (line 517):

# Assigning a Call to a Name (line 517):

# Call to HasMember(...): (line 517)
# Processing the call arguments (line 517)
str_19256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 25), 'str', '__trunc__')
# Getting the type of 'UndefinedType' (line 517)
UndefinedType_19257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 38), 'UndefinedType', False)
int_19258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 53), 'int')
# Processing the call keyword arguments (line 517)
kwargs_19259 = {}
# Getting the type of 'HasMember' (line 517)
HasMember_19255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 15), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 517)
HasMember_call_result_19260 = invoke(stypy.reporting.localization.Localization(__file__, 517, 15), HasMember_19255, *[str_19256, UndefinedType_19257, int_19258], **kwargs_19259)

# Assigning a type to the variable 'CastsToTrunc' (line 517)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 0), 'CastsToTrunc', HasMember_call_result_19260)

# Assigning a Call to a Name (line 518):

# Assigning a Call to a Name (line 518):

# Call to HasMember(...): (line 518)
# Processing the call arguments (line 518)
str_19262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 26), 'str', '__coerce__')
# Getting the type of 'UndefinedType' (line 518)
UndefinedType_19263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 40), 'UndefinedType', False)
int_19264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 55), 'int')
# Processing the call keyword arguments (line 518)
kwargs_19265 = {}
# Getting the type of 'HasMember' (line 518)
HasMember_19261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 16), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 518)
HasMember_call_result_19266 = invoke(stypy.reporting.localization.Localization(__file__, 518, 16), HasMember_19261, *[str_19262, UndefinedType_19263, int_19264], **kwargs_19265)

# Assigning a type to the variable 'CastsToCoerce' (line 518)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 0), 'CastsToCoerce', HasMember_call_result_19266)

# Assigning a Call to a Name (line 523):

# Assigning a Call to a Name (line 523):

# Call to HasMember(...): (line 523)
# Processing the call arguments (line 523)
str_19268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 29), 'str', '__cmp__')
# Getting the type of 'DynamicType' (line 523)
DynamicType_19269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 40), 'DynamicType', False)
int_19270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 53), 'int')
# Processing the call keyword arguments (line 523)
kwargs_19271 = {}
# Getting the type of 'HasMember' (line 523)
HasMember_19267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 523)
HasMember_call_result_19272 = invoke(stypy.reporting.localization.Localization(__file__, 523, 19), HasMember_19267, *[str_19268, DynamicType_19269, int_19270], **kwargs_19271)

# Assigning a type to the variable 'Overloads__cmp__' (line 523)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 0), 'Overloads__cmp__', HasMember_call_result_19272)

# Assigning a Call to a Name (line 524):

# Assigning a Call to a Name (line 524):

# Call to HasMember(...): (line 524)
# Processing the call arguments (line 524)
str_19274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 28), 'str', '__eq__')
# Getting the type of 'DynamicType' (line 524)
DynamicType_19275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 38), 'DynamicType', False)
int_19276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 51), 'int')
# Processing the call keyword arguments (line 524)
kwargs_19277 = {}
# Getting the type of 'HasMember' (line 524)
HasMember_19273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 524)
HasMember_call_result_19278 = invoke(stypy.reporting.localization.Localization(__file__, 524, 18), HasMember_19273, *[str_19274, DynamicType_19275, int_19276], **kwargs_19277)

# Assigning a type to the variable 'Overloads__eq__' (line 524)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 0), 'Overloads__eq__', HasMember_call_result_19278)

# Assigning a Call to a Name (line 525):

# Assigning a Call to a Name (line 525):

# Call to HasMember(...): (line 525)
# Processing the call arguments (line 525)
str_19280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 28), 'str', '__ne__')
# Getting the type of 'DynamicType' (line 525)
DynamicType_19281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 38), 'DynamicType', False)
int_19282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 51), 'int')
# Processing the call keyword arguments (line 525)
kwargs_19283 = {}
# Getting the type of 'HasMember' (line 525)
HasMember_19279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 525)
HasMember_call_result_19284 = invoke(stypy.reporting.localization.Localization(__file__, 525, 18), HasMember_19279, *[str_19280, DynamicType_19281, int_19282], **kwargs_19283)

# Assigning a type to the variable 'Overloads__ne__' (line 525)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 0), 'Overloads__ne__', HasMember_call_result_19284)

# Assigning a Call to a Name (line 526):

# Assigning a Call to a Name (line 526):

# Call to HasMember(...): (line 526)
# Processing the call arguments (line 526)
str_19286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 28), 'str', '__lt__')
# Getting the type of 'DynamicType' (line 526)
DynamicType_19287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 38), 'DynamicType', False)
int_19288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 51), 'int')
# Processing the call keyword arguments (line 526)
kwargs_19289 = {}
# Getting the type of 'HasMember' (line 526)
HasMember_19285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 526)
HasMember_call_result_19290 = invoke(stypy.reporting.localization.Localization(__file__, 526, 18), HasMember_19285, *[str_19286, DynamicType_19287, int_19288], **kwargs_19289)

# Assigning a type to the variable 'Overloads__lt__' (line 526)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 0), 'Overloads__lt__', HasMember_call_result_19290)

# Assigning a Call to a Name (line 527):

# Assigning a Call to a Name (line 527):

# Call to HasMember(...): (line 527)
# Processing the call arguments (line 527)
str_19292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 28), 'str', '__gt__')
# Getting the type of 'DynamicType' (line 527)
DynamicType_19293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 38), 'DynamicType', False)
int_19294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 51), 'int')
# Processing the call keyword arguments (line 527)
kwargs_19295 = {}
# Getting the type of 'HasMember' (line 527)
HasMember_19291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 527)
HasMember_call_result_19296 = invoke(stypy.reporting.localization.Localization(__file__, 527, 18), HasMember_19291, *[str_19292, DynamicType_19293, int_19294], **kwargs_19295)

# Assigning a type to the variable 'Overloads__gt__' (line 527)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 0), 'Overloads__gt__', HasMember_call_result_19296)

# Assigning a Call to a Name (line 528):

# Assigning a Call to a Name (line 528):

# Call to HasMember(...): (line 528)
# Processing the call arguments (line 528)
str_19298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 28), 'str', '__le__')
# Getting the type of 'DynamicType' (line 528)
DynamicType_19299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 38), 'DynamicType', False)
int_19300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 51), 'int')
# Processing the call keyword arguments (line 528)
kwargs_19301 = {}
# Getting the type of 'HasMember' (line 528)
HasMember_19297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 528)
HasMember_call_result_19302 = invoke(stypy.reporting.localization.Localization(__file__, 528, 18), HasMember_19297, *[str_19298, DynamicType_19299, int_19300], **kwargs_19301)

# Assigning a type to the variable 'Overloads__le__' (line 528)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 0), 'Overloads__le__', HasMember_call_result_19302)

# Assigning a Call to a Name (line 529):

# Assigning a Call to a Name (line 529):

# Call to HasMember(...): (line 529)
# Processing the call arguments (line 529)
str_19304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 28), 'str', '__ge__')
# Getting the type of 'DynamicType' (line 529)
DynamicType_19305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 38), 'DynamicType', False)
int_19306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 51), 'int')
# Processing the call keyword arguments (line 529)
kwargs_19307 = {}
# Getting the type of 'HasMember' (line 529)
HasMember_19303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 529)
HasMember_call_result_19308 = invoke(stypy.reporting.localization.Localization(__file__, 529, 18), HasMember_19303, *[str_19304, DynamicType_19305, int_19306], **kwargs_19307)

# Assigning a type to the variable 'Overloads__ge__' (line 529)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 0), 'Overloads__ge__', HasMember_call_result_19308)

# Assigning a Call to a Name (line 532):

# Assigning a Call to a Name (line 532):

# Call to HasMember(...): (line 532)
# Processing the call arguments (line 532)
str_19310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 29), 'str', '__pos__')
# Getting the type of 'UndefinedType' (line 532)
UndefinedType_19311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 40), 'UndefinedType', False)
int_19312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 55), 'int')
# Processing the call keyword arguments (line 532)
kwargs_19313 = {}
# Getting the type of 'HasMember' (line 532)
HasMember_19309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 532)
HasMember_call_result_19314 = invoke(stypy.reporting.localization.Localization(__file__, 532, 19), HasMember_19309, *[str_19310, UndefinedType_19311, int_19312], **kwargs_19313)

# Assigning a type to the variable 'Overloads__pos__' (line 532)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 0), 'Overloads__pos__', HasMember_call_result_19314)

# Assigning a Call to a Name (line 533):

# Assigning a Call to a Name (line 533):

# Call to HasMember(...): (line 533)
# Processing the call arguments (line 533)
str_19316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 29), 'str', '__neg__')
# Getting the type of 'UndefinedType' (line 533)
UndefinedType_19317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 40), 'UndefinedType', False)
int_19318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 55), 'int')
# Processing the call keyword arguments (line 533)
kwargs_19319 = {}
# Getting the type of 'HasMember' (line 533)
HasMember_19315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 533)
HasMember_call_result_19320 = invoke(stypy.reporting.localization.Localization(__file__, 533, 19), HasMember_19315, *[str_19316, UndefinedType_19317, int_19318], **kwargs_19319)

# Assigning a type to the variable 'Overloads__neg__' (line 533)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 0), 'Overloads__neg__', HasMember_call_result_19320)

# Assigning a Call to a Name (line 534):

# Assigning a Call to a Name (line 534):

# Call to HasMember(...): (line 534)
# Processing the call arguments (line 534)
str_19322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 29), 'str', '__abs__')
# Getting the type of 'UndefinedType' (line 534)
UndefinedType_19323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 40), 'UndefinedType', False)
int_19324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 55), 'int')
# Processing the call keyword arguments (line 534)
kwargs_19325 = {}
# Getting the type of 'HasMember' (line 534)
HasMember_19321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 534)
HasMember_call_result_19326 = invoke(stypy.reporting.localization.Localization(__file__, 534, 19), HasMember_19321, *[str_19322, UndefinedType_19323, int_19324], **kwargs_19325)

# Assigning a type to the variable 'Overloads__abs__' (line 534)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 0), 'Overloads__abs__', HasMember_call_result_19326)

# Assigning a Call to a Name (line 535):

# Assigning a Call to a Name (line 535):

# Call to HasMember(...): (line 535)
# Processing the call arguments (line 535)
str_19328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 32), 'str', '__invert__')
# Getting the type of 'UndefinedType' (line 535)
UndefinedType_19329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 46), 'UndefinedType', False)
int_19330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 61), 'int')
# Processing the call keyword arguments (line 535)
kwargs_19331 = {}
# Getting the type of 'HasMember' (line 535)
HasMember_19327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 22), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 535)
HasMember_call_result_19332 = invoke(stypy.reporting.localization.Localization(__file__, 535, 22), HasMember_19327, *[str_19328, UndefinedType_19329, int_19330], **kwargs_19331)

# Assigning a type to the variable 'Overloads__invert__' (line 535)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 0), 'Overloads__invert__', HasMember_call_result_19332)

# Assigning a Call to a Name (line 537):

# Assigning a Call to a Name (line 537):

# Call to HasMember(...): (line 537)
# Processing the call arguments (line 537)
str_19334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 31), 'str', '__round__')
# Getting the type of 'int' (line 537)
int_19335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 44), 'int', False)
int_19336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 49), 'int')
# Processing the call keyword arguments (line 537)
kwargs_19337 = {}
# Getting the type of 'HasMember' (line 537)
HasMember_19333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 21), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 537)
HasMember_call_result_19338 = invoke(stypy.reporting.localization.Localization(__file__, 537, 21), HasMember_19333, *[str_19334, int_19335, int_19336], **kwargs_19337)

# Assigning a type to the variable 'Overloads__round__' (line 537)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 0), 'Overloads__round__', HasMember_call_result_19338)

# Assigning a Call to a Name (line 538):

# Assigning a Call to a Name (line 538):

# Call to HasMember(...): (line 538)
# Processing the call arguments (line 538)
str_19340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 31), 'str', '__floor__')
# Getting the type of 'int' (line 538)
int_19341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 44), 'int', False)
int_19342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 49), 'int')
# Processing the call keyword arguments (line 538)
kwargs_19343 = {}
# Getting the type of 'HasMember' (line 538)
HasMember_19339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 21), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 538)
HasMember_call_result_19344 = invoke(stypy.reporting.localization.Localization(__file__, 538, 21), HasMember_19339, *[str_19340, int_19341, int_19342], **kwargs_19343)

# Assigning a type to the variable 'Overloads__floor__' (line 538)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 0), 'Overloads__floor__', HasMember_call_result_19344)

# Assigning a Call to a Name (line 539):

# Assigning a Call to a Name (line 539):

# Call to HasMember(...): (line 539)
# Processing the call arguments (line 539)
str_19346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 30), 'str', '__ceil__')
# Getting the type of 'int' (line 539)
int_19347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 42), 'int', False)
int_19348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 47), 'int')
# Processing the call keyword arguments (line 539)
kwargs_19349 = {}
# Getting the type of 'HasMember' (line 539)
HasMember_19345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 539)
HasMember_call_result_19350 = invoke(stypy.reporting.localization.Localization(__file__, 539, 20), HasMember_19345, *[str_19346, int_19347, int_19348], **kwargs_19349)

# Assigning a type to the variable 'Overloads__ceil__' (line 539)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 0), 'Overloads__ceil__', HasMember_call_result_19350)

# Assigning a Call to a Name (line 541):

# Assigning a Call to a Name (line 541):

# Call to HasMember(...): (line 541)
# Processing the call arguments (line 541)
str_19352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 31), 'str', '__trunc__')
# Getting the type of 'int' (line 541)
int_19353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 44), 'int', False)
int_19354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 49), 'int')
# Processing the call keyword arguments (line 541)
kwargs_19355 = {}
# Getting the type of 'HasMember' (line 541)
HasMember_19351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 21), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 541)
HasMember_call_result_19356 = invoke(stypy.reporting.localization.Localization(__file__, 541, 21), HasMember_19351, *[str_19352, int_19353, int_19354], **kwargs_19355)

# Assigning a type to the variable 'Overloads__trunc__' (line 541)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 0), 'Overloads__trunc__', HasMember_call_result_19356)

# Assigning a Call to a Name (line 544):

# Assigning a Call to a Name (line 544):

# Call to HasMember(...): (line 544)
# Processing the call arguments (line 544)
str_19358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 29), 'str', '__add__')
# Getting the type of 'DynamicType' (line 544)
DynamicType_19359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 40), 'DynamicType', False)
int_19360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 53), 'int')
# Processing the call keyword arguments (line 544)
kwargs_19361 = {}
# Getting the type of 'HasMember' (line 544)
HasMember_19357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 544)
HasMember_call_result_19362 = invoke(stypy.reporting.localization.Localization(__file__, 544, 19), HasMember_19357, *[str_19358, DynamicType_19359, int_19360], **kwargs_19361)

# Assigning a type to the variable 'Overloads__add__' (line 544)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 0), 'Overloads__add__', HasMember_call_result_19362)

# Assigning a Call to a Name (line 545):

# Assigning a Call to a Name (line 545):

# Call to HasMember(...): (line 545)
# Processing the call arguments (line 545)
str_19364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 29), 'str', '__sub__')
# Getting the type of 'DynamicType' (line 545)
DynamicType_19365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 40), 'DynamicType', False)
int_19366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 53), 'int')
# Processing the call keyword arguments (line 545)
kwargs_19367 = {}
# Getting the type of 'HasMember' (line 545)
HasMember_19363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 545)
HasMember_call_result_19368 = invoke(stypy.reporting.localization.Localization(__file__, 545, 19), HasMember_19363, *[str_19364, DynamicType_19365, int_19366], **kwargs_19367)

# Assigning a type to the variable 'Overloads__sub__' (line 545)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 0), 'Overloads__sub__', HasMember_call_result_19368)

# Assigning a Call to a Name (line 546):

# Assigning a Call to a Name (line 546):

# Call to HasMember(...): (line 546)
# Processing the call arguments (line 546)
str_19370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 29), 'str', '__mul__')
# Getting the type of 'DynamicType' (line 546)
DynamicType_19371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 40), 'DynamicType', False)
int_19372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 53), 'int')
# Processing the call keyword arguments (line 546)
kwargs_19373 = {}
# Getting the type of 'HasMember' (line 546)
HasMember_19369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 546)
HasMember_call_result_19374 = invoke(stypy.reporting.localization.Localization(__file__, 546, 19), HasMember_19369, *[str_19370, DynamicType_19371, int_19372], **kwargs_19373)

# Assigning a type to the variable 'Overloads__mul__' (line 546)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 0), 'Overloads__mul__', HasMember_call_result_19374)

# Assigning a Call to a Name (line 547):

# Assigning a Call to a Name (line 547):

# Call to HasMember(...): (line 547)
# Processing the call arguments (line 547)
str_19376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 34), 'str', '__floordiv__')
# Getting the type of 'DynamicType' (line 547)
DynamicType_19377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 50), 'DynamicType', False)
int_19378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 63), 'int')
# Processing the call keyword arguments (line 547)
kwargs_19379 = {}
# Getting the type of 'HasMember' (line 547)
HasMember_19375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 24), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 547)
HasMember_call_result_19380 = invoke(stypy.reporting.localization.Localization(__file__, 547, 24), HasMember_19375, *[str_19376, DynamicType_19377, int_19378], **kwargs_19379)

# Assigning a type to the variable 'Overloads__floordiv__' (line 547)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 0), 'Overloads__floordiv__', HasMember_call_result_19380)

# Assigning a Call to a Name (line 548):

# Assigning a Call to a Name (line 548):

# Call to HasMember(...): (line 548)
# Processing the call arguments (line 548)
str_19382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 29), 'str', '__div__')
# Getting the type of 'DynamicType' (line 548)
DynamicType_19383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 40), 'DynamicType', False)
int_19384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 53), 'int')
# Processing the call keyword arguments (line 548)
kwargs_19385 = {}
# Getting the type of 'HasMember' (line 548)
HasMember_19381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 548)
HasMember_call_result_19386 = invoke(stypy.reporting.localization.Localization(__file__, 548, 19), HasMember_19381, *[str_19382, DynamicType_19383, int_19384], **kwargs_19385)

# Assigning a type to the variable 'Overloads__div__' (line 548)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 0), 'Overloads__div__', HasMember_call_result_19386)

# Assigning a Call to a Name (line 549):

# Assigning a Call to a Name (line 549):

# Call to HasMember(...): (line 549)
# Processing the call arguments (line 549)
str_19388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 33), 'str', '__truediv__')
# Getting the type of 'DynamicType' (line 549)
DynamicType_19389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 48), 'DynamicType', False)
int_19390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 61), 'int')
# Processing the call keyword arguments (line 549)
kwargs_19391 = {}
# Getting the type of 'HasMember' (line 549)
HasMember_19387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 23), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 549)
HasMember_call_result_19392 = invoke(stypy.reporting.localization.Localization(__file__, 549, 23), HasMember_19387, *[str_19388, DynamicType_19389, int_19390], **kwargs_19391)

# Assigning a type to the variable 'Overloads__truediv__' (line 549)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 0), 'Overloads__truediv__', HasMember_call_result_19392)

# Assigning a Call to a Name (line 550):

# Assigning a Call to a Name (line 550):

# Call to HasMember(...): (line 550)
# Processing the call arguments (line 550)
str_19394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 29), 'str', '__mod__')
# Getting the type of 'DynamicType' (line 550)
DynamicType_19395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 40), 'DynamicType', False)
int_19396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 53), 'int')
# Processing the call keyword arguments (line 550)
kwargs_19397 = {}
# Getting the type of 'HasMember' (line 550)
HasMember_19393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 550)
HasMember_call_result_19398 = invoke(stypy.reporting.localization.Localization(__file__, 550, 19), HasMember_19393, *[str_19394, DynamicType_19395, int_19396], **kwargs_19397)

# Assigning a type to the variable 'Overloads__mod__' (line 550)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 0), 'Overloads__mod__', HasMember_call_result_19398)

# Assigning a Call to a Name (line 551):

# Assigning a Call to a Name (line 551):

# Call to HasMember(...): (line 551)
# Processing the call arguments (line 551)
str_19400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 32), 'str', '__divmod__')
# Getting the type of 'DynamicType' (line 551)
DynamicType_19401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 46), 'DynamicType', False)
int_19402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 59), 'int')
# Processing the call keyword arguments (line 551)
kwargs_19403 = {}
# Getting the type of 'HasMember' (line 551)
HasMember_19399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 22), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 551)
HasMember_call_result_19404 = invoke(stypy.reporting.localization.Localization(__file__, 551, 22), HasMember_19399, *[str_19400, DynamicType_19401, int_19402], **kwargs_19403)

# Assigning a type to the variable 'Overloads__divmod__' (line 551)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 0), 'Overloads__divmod__', HasMember_call_result_19404)

# Assigning a Call to a Name (line 552):

# Assigning a Call to a Name (line 552):

# Call to HasMember(...): (line 552)
# Processing the call arguments (line 552)
str_19406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 29), 'str', '__pow__')
# Getting the type of 'DynamicType' (line 552)
DynamicType_19407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 40), 'DynamicType', False)
int_19408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 53), 'int')
# Processing the call keyword arguments (line 552)
kwargs_19409 = {}
# Getting the type of 'HasMember' (line 552)
HasMember_19405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 552)
HasMember_call_result_19410 = invoke(stypy.reporting.localization.Localization(__file__, 552, 19), HasMember_19405, *[str_19406, DynamicType_19407, int_19408], **kwargs_19409)

# Assigning a type to the variable 'Overloads__pow__' (line 552)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 0), 'Overloads__pow__', HasMember_call_result_19410)

# Assigning a Call to a Name (line 553):

# Assigning a Call to a Name (line 553):

# Call to HasMember(...): (line 553)
# Processing the call arguments (line 553)
str_19412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 32), 'str', '__lshift__')
# Getting the type of 'DynamicType' (line 553)
DynamicType_19413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 46), 'DynamicType', False)
int_19414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 59), 'int')
# Processing the call keyword arguments (line 553)
kwargs_19415 = {}
# Getting the type of 'HasMember' (line 553)
HasMember_19411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 22), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 553)
HasMember_call_result_19416 = invoke(stypy.reporting.localization.Localization(__file__, 553, 22), HasMember_19411, *[str_19412, DynamicType_19413, int_19414], **kwargs_19415)

# Assigning a type to the variable 'Overloads__lshift__' (line 553)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 0), 'Overloads__lshift__', HasMember_call_result_19416)

# Assigning a Call to a Name (line 554):

# Assigning a Call to a Name (line 554):

# Call to HasMember(...): (line 554)
# Processing the call arguments (line 554)
str_19418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 32), 'str', '__rshift__')
# Getting the type of 'DynamicType' (line 554)
DynamicType_19419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 46), 'DynamicType', False)
int_19420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 59), 'int')
# Processing the call keyword arguments (line 554)
kwargs_19421 = {}
# Getting the type of 'HasMember' (line 554)
HasMember_19417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 22), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 554)
HasMember_call_result_19422 = invoke(stypy.reporting.localization.Localization(__file__, 554, 22), HasMember_19417, *[str_19418, DynamicType_19419, int_19420], **kwargs_19421)

# Assigning a type to the variable 'Overloads__rshift__' (line 554)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 0), 'Overloads__rshift__', HasMember_call_result_19422)

# Assigning a Call to a Name (line 555):

# Assigning a Call to a Name (line 555):

# Call to HasMember(...): (line 555)
# Processing the call arguments (line 555)
str_19424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 29), 'str', '__and__')
# Getting the type of 'DynamicType' (line 555)
DynamicType_19425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 40), 'DynamicType', False)
int_19426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 53), 'int')
# Processing the call keyword arguments (line 555)
kwargs_19427 = {}
# Getting the type of 'HasMember' (line 555)
HasMember_19423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 555)
HasMember_call_result_19428 = invoke(stypy.reporting.localization.Localization(__file__, 555, 19), HasMember_19423, *[str_19424, DynamicType_19425, int_19426], **kwargs_19427)

# Assigning a type to the variable 'Overloads__and__' (line 555)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 0), 'Overloads__and__', HasMember_call_result_19428)

# Assigning a Call to a Name (line 556):

# Assigning a Call to a Name (line 556):

# Call to HasMember(...): (line 556)
# Processing the call arguments (line 556)
str_19430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 28), 'str', '__or__')
# Getting the type of 'DynamicType' (line 556)
DynamicType_19431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 38), 'DynamicType', False)
int_19432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 51), 'int')
# Processing the call keyword arguments (line 556)
kwargs_19433 = {}
# Getting the type of 'HasMember' (line 556)
HasMember_19429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 556)
HasMember_call_result_19434 = invoke(stypy.reporting.localization.Localization(__file__, 556, 18), HasMember_19429, *[str_19430, DynamicType_19431, int_19432], **kwargs_19433)

# Assigning a type to the variable 'Overloads__or__' (line 556)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 0), 'Overloads__or__', HasMember_call_result_19434)

# Assigning a Call to a Name (line 557):

# Assigning a Call to a Name (line 557):

# Call to HasMember(...): (line 557)
# Processing the call arguments (line 557)
str_19436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 29), 'str', '__xor__')
# Getting the type of 'DynamicType' (line 557)
DynamicType_19437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 40), 'DynamicType', False)
int_19438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 53), 'int')
# Processing the call keyword arguments (line 557)
kwargs_19439 = {}
# Getting the type of 'HasMember' (line 557)
HasMember_19435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 557)
HasMember_call_result_19440 = invoke(stypy.reporting.localization.Localization(__file__, 557, 19), HasMember_19435, *[str_19436, DynamicType_19437, int_19438], **kwargs_19439)

# Assigning a type to the variable 'Overloads__xor__' (line 557)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 0), 'Overloads__xor__', HasMember_call_result_19440)

# Assigning a Call to a Name (line 560):

# Assigning a Call to a Name (line 560):

# Call to HasMember(...): (line 560)
# Processing the call arguments (line 560)
str_19442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 30), 'str', '__radd__')
# Getting the type of 'DynamicType' (line 560)
DynamicType_19443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 42), 'DynamicType', False)
int_19444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 55), 'int')
# Processing the call keyword arguments (line 560)
kwargs_19445 = {}
# Getting the type of 'HasMember' (line 560)
HasMember_19441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 560)
HasMember_call_result_19446 = invoke(stypy.reporting.localization.Localization(__file__, 560, 20), HasMember_19441, *[str_19442, DynamicType_19443, int_19444], **kwargs_19445)

# Assigning a type to the variable 'Overloads__radd__' (line 560)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 0), 'Overloads__radd__', HasMember_call_result_19446)

# Assigning a Call to a Name (line 561):

# Assigning a Call to a Name (line 561):

# Call to HasMember(...): (line 561)
# Processing the call arguments (line 561)
str_19448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 30), 'str', '__rsub__')
# Getting the type of 'DynamicType' (line 561)
DynamicType_19449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 42), 'DynamicType', False)
int_19450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 55), 'int')
# Processing the call keyword arguments (line 561)
kwargs_19451 = {}
# Getting the type of 'HasMember' (line 561)
HasMember_19447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 561)
HasMember_call_result_19452 = invoke(stypy.reporting.localization.Localization(__file__, 561, 20), HasMember_19447, *[str_19448, DynamicType_19449, int_19450], **kwargs_19451)

# Assigning a type to the variable 'Overloads__rsub__' (line 561)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 0), 'Overloads__rsub__', HasMember_call_result_19452)

# Assigning a Call to a Name (line 562):

# Assigning a Call to a Name (line 562):

# Call to HasMember(...): (line 562)
# Processing the call arguments (line 562)
str_19454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 30), 'str', '__rmul__')
# Getting the type of 'DynamicType' (line 562)
DynamicType_19455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 42), 'DynamicType', False)
int_19456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 55), 'int')
# Processing the call keyword arguments (line 562)
kwargs_19457 = {}
# Getting the type of 'HasMember' (line 562)
HasMember_19453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 562)
HasMember_call_result_19458 = invoke(stypy.reporting.localization.Localization(__file__, 562, 20), HasMember_19453, *[str_19454, DynamicType_19455, int_19456], **kwargs_19457)

# Assigning a type to the variable 'Overloads__rmul__' (line 562)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 0), 'Overloads__rmul__', HasMember_call_result_19458)

# Assigning a Call to a Name (line 563):

# Assigning a Call to a Name (line 563):

# Call to HasMember(...): (line 563)
# Processing the call arguments (line 563)
str_19460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 35), 'str', '__rfloordiv__')
# Getting the type of 'DynamicType' (line 563)
DynamicType_19461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 52), 'DynamicType', False)
int_19462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 65), 'int')
# Processing the call keyword arguments (line 563)
kwargs_19463 = {}
# Getting the type of 'HasMember' (line 563)
HasMember_19459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 25), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 563)
HasMember_call_result_19464 = invoke(stypy.reporting.localization.Localization(__file__, 563, 25), HasMember_19459, *[str_19460, DynamicType_19461, int_19462], **kwargs_19463)

# Assigning a type to the variable 'Overloads__rfloordiv__' (line 563)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 0), 'Overloads__rfloordiv__', HasMember_call_result_19464)

# Assigning a Call to a Name (line 564):

# Assigning a Call to a Name (line 564):

# Call to HasMember(...): (line 564)
# Processing the call arguments (line 564)
str_19466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 30), 'str', '__rdiv__')
# Getting the type of 'DynamicType' (line 564)
DynamicType_19467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 42), 'DynamicType', False)
int_19468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 55), 'int')
# Processing the call keyword arguments (line 564)
kwargs_19469 = {}
# Getting the type of 'HasMember' (line 564)
HasMember_19465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 564)
HasMember_call_result_19470 = invoke(stypy.reporting.localization.Localization(__file__, 564, 20), HasMember_19465, *[str_19466, DynamicType_19467, int_19468], **kwargs_19469)

# Assigning a type to the variable 'Overloads__rdiv__' (line 564)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 0), 'Overloads__rdiv__', HasMember_call_result_19470)

# Assigning a Call to a Name (line 565):

# Assigning a Call to a Name (line 565):

# Call to HasMember(...): (line 565)
# Processing the call arguments (line 565)
str_19472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 34), 'str', '__rtruediv__')
# Getting the type of 'DynamicType' (line 565)
DynamicType_19473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 50), 'DynamicType', False)
int_19474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 63), 'int')
# Processing the call keyword arguments (line 565)
kwargs_19475 = {}
# Getting the type of 'HasMember' (line 565)
HasMember_19471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 24), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 565)
HasMember_call_result_19476 = invoke(stypy.reporting.localization.Localization(__file__, 565, 24), HasMember_19471, *[str_19472, DynamicType_19473, int_19474], **kwargs_19475)

# Assigning a type to the variable 'Overloads__rtruediv__' (line 565)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 0), 'Overloads__rtruediv__', HasMember_call_result_19476)

# Assigning a Call to a Name (line 566):

# Assigning a Call to a Name (line 566):

# Call to HasMember(...): (line 566)
# Processing the call arguments (line 566)
str_19478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 30), 'str', '__rmod__')
# Getting the type of 'DynamicType' (line 566)
DynamicType_19479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 42), 'DynamicType', False)
int_19480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 55), 'int')
# Processing the call keyword arguments (line 566)
kwargs_19481 = {}
# Getting the type of 'HasMember' (line 566)
HasMember_19477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 566)
HasMember_call_result_19482 = invoke(stypy.reporting.localization.Localization(__file__, 566, 20), HasMember_19477, *[str_19478, DynamicType_19479, int_19480], **kwargs_19481)

# Assigning a type to the variable 'Overloads__rmod__' (line 566)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 0), 'Overloads__rmod__', HasMember_call_result_19482)

# Assigning a Call to a Name (line 567):

# Assigning a Call to a Name (line 567):

# Call to HasMember(...): (line 567)
# Processing the call arguments (line 567)
str_19484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 33), 'str', '__rdivmod__')
# Getting the type of 'DynamicType' (line 567)
DynamicType_19485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 48), 'DynamicType', False)
int_19486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 61), 'int')
# Processing the call keyword arguments (line 567)
kwargs_19487 = {}
# Getting the type of 'HasMember' (line 567)
HasMember_19483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 23), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 567)
HasMember_call_result_19488 = invoke(stypy.reporting.localization.Localization(__file__, 567, 23), HasMember_19483, *[str_19484, DynamicType_19485, int_19486], **kwargs_19487)

# Assigning a type to the variable 'Overloads__rdivmod__' (line 567)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 0), 'Overloads__rdivmod__', HasMember_call_result_19488)

# Assigning a Call to a Name (line 568):

# Assigning a Call to a Name (line 568):

# Call to HasMember(...): (line 568)
# Processing the call arguments (line 568)
str_19490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 30), 'str', '__rpow__')
# Getting the type of 'DynamicType' (line 568)
DynamicType_19491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 42), 'DynamicType', False)
int_19492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 55), 'int')
# Processing the call keyword arguments (line 568)
kwargs_19493 = {}
# Getting the type of 'HasMember' (line 568)
HasMember_19489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 568)
HasMember_call_result_19494 = invoke(stypy.reporting.localization.Localization(__file__, 568, 20), HasMember_19489, *[str_19490, DynamicType_19491, int_19492], **kwargs_19493)

# Assigning a type to the variable 'Overloads__rpow__' (line 568)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 0), 'Overloads__rpow__', HasMember_call_result_19494)

# Assigning a Call to a Name (line 569):

# Assigning a Call to a Name (line 569):

# Call to HasMember(...): (line 569)
# Processing the call arguments (line 569)
str_19496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 33), 'str', '__rlshift__')
# Getting the type of 'DynamicType' (line 569)
DynamicType_19497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 48), 'DynamicType', False)
int_19498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 61), 'int')
# Processing the call keyword arguments (line 569)
kwargs_19499 = {}
# Getting the type of 'HasMember' (line 569)
HasMember_19495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 23), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 569)
HasMember_call_result_19500 = invoke(stypy.reporting.localization.Localization(__file__, 569, 23), HasMember_19495, *[str_19496, DynamicType_19497, int_19498], **kwargs_19499)

# Assigning a type to the variable 'Overloads__rlshift__' (line 569)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 0), 'Overloads__rlshift__', HasMember_call_result_19500)

# Assigning a Call to a Name (line 570):

# Assigning a Call to a Name (line 570):

# Call to HasMember(...): (line 570)
# Processing the call arguments (line 570)
str_19502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 33), 'str', '__rrshift__')
# Getting the type of 'DynamicType' (line 570)
DynamicType_19503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 48), 'DynamicType', False)
int_19504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 61), 'int')
# Processing the call keyword arguments (line 570)
kwargs_19505 = {}
# Getting the type of 'HasMember' (line 570)
HasMember_19501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 23), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 570)
HasMember_call_result_19506 = invoke(stypy.reporting.localization.Localization(__file__, 570, 23), HasMember_19501, *[str_19502, DynamicType_19503, int_19504], **kwargs_19505)

# Assigning a type to the variable 'Overloads__rrshift__' (line 570)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 0), 'Overloads__rrshift__', HasMember_call_result_19506)

# Assigning a Call to a Name (line 571):

# Assigning a Call to a Name (line 571):

# Call to HasMember(...): (line 571)
# Processing the call arguments (line 571)
str_19508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 30), 'str', '__rand__')
# Getting the type of 'DynamicType' (line 571)
DynamicType_19509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 42), 'DynamicType', False)
int_19510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 55), 'int')
# Processing the call keyword arguments (line 571)
kwargs_19511 = {}
# Getting the type of 'HasMember' (line 571)
HasMember_19507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 571)
HasMember_call_result_19512 = invoke(stypy.reporting.localization.Localization(__file__, 571, 20), HasMember_19507, *[str_19508, DynamicType_19509, int_19510], **kwargs_19511)

# Assigning a type to the variable 'Overloads__rand__' (line 571)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 0), 'Overloads__rand__', HasMember_call_result_19512)

# Assigning a Call to a Name (line 572):

# Assigning a Call to a Name (line 572):

# Call to HasMember(...): (line 572)
# Processing the call arguments (line 572)
str_19514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 29), 'str', '__ror__')
# Getting the type of 'DynamicType' (line 572)
DynamicType_19515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 40), 'DynamicType', False)
int_19516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 53), 'int')
# Processing the call keyword arguments (line 572)
kwargs_19517 = {}
# Getting the type of 'HasMember' (line 572)
HasMember_19513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 572)
HasMember_call_result_19518 = invoke(stypy.reporting.localization.Localization(__file__, 572, 19), HasMember_19513, *[str_19514, DynamicType_19515, int_19516], **kwargs_19517)

# Assigning a type to the variable 'Overloads__ror__' (line 572)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 0), 'Overloads__ror__', HasMember_call_result_19518)

# Assigning a Call to a Name (line 573):

# Assigning a Call to a Name (line 573):

# Call to HasMember(...): (line 573)
# Processing the call arguments (line 573)
str_19520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 30), 'str', '__rxor__')
# Getting the type of 'DynamicType' (line 573)
DynamicType_19521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 42), 'DynamicType', False)
int_19522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 55), 'int')
# Processing the call keyword arguments (line 573)
kwargs_19523 = {}
# Getting the type of 'HasMember' (line 573)
HasMember_19519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 573)
HasMember_call_result_19524 = invoke(stypy.reporting.localization.Localization(__file__, 573, 20), HasMember_19519, *[str_19520, DynamicType_19521, int_19522], **kwargs_19523)

# Assigning a type to the variable 'Overloads__rxor__' (line 573)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 0), 'Overloads__rxor__', HasMember_call_result_19524)

# Assigning a Call to a Name (line 577):

# Assigning a Call to a Name (line 577):

# Call to HasMember(...): (line 577)
# Processing the call arguments (line 577)
str_19526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 30), 'str', '__iadd__')
# Getting the type of 'DynamicType' (line 577)
DynamicType_19527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 42), 'DynamicType', False)
int_19528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 55), 'int')
# Processing the call keyword arguments (line 577)
kwargs_19529 = {}
# Getting the type of 'HasMember' (line 577)
HasMember_19525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 577)
HasMember_call_result_19530 = invoke(stypy.reporting.localization.Localization(__file__, 577, 20), HasMember_19525, *[str_19526, DynamicType_19527, int_19528], **kwargs_19529)

# Assigning a type to the variable 'Overloads__iadd__' (line 577)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 0), 'Overloads__iadd__', HasMember_call_result_19530)

# Assigning a Call to a Name (line 578):

# Assigning a Call to a Name (line 578):

# Call to HasMember(...): (line 578)
# Processing the call arguments (line 578)
str_19532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 30), 'str', '__isub__')
# Getting the type of 'DynamicType' (line 578)
DynamicType_19533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 42), 'DynamicType', False)
int_19534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 55), 'int')
# Processing the call keyword arguments (line 578)
kwargs_19535 = {}
# Getting the type of 'HasMember' (line 578)
HasMember_19531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 578)
HasMember_call_result_19536 = invoke(stypy.reporting.localization.Localization(__file__, 578, 20), HasMember_19531, *[str_19532, DynamicType_19533, int_19534], **kwargs_19535)

# Assigning a type to the variable 'Overloads__isub__' (line 578)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 0), 'Overloads__isub__', HasMember_call_result_19536)

# Assigning a Call to a Name (line 579):

# Assigning a Call to a Name (line 579):

# Call to HasMember(...): (line 579)
# Processing the call arguments (line 579)
str_19538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 30), 'str', '__imul__')
# Getting the type of 'DynamicType' (line 579)
DynamicType_19539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 42), 'DynamicType', False)
int_19540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 55), 'int')
# Processing the call keyword arguments (line 579)
kwargs_19541 = {}
# Getting the type of 'HasMember' (line 579)
HasMember_19537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 579)
HasMember_call_result_19542 = invoke(stypy.reporting.localization.Localization(__file__, 579, 20), HasMember_19537, *[str_19538, DynamicType_19539, int_19540], **kwargs_19541)

# Assigning a type to the variable 'Overloads__imul__' (line 579)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 0), 'Overloads__imul__', HasMember_call_result_19542)

# Assigning a Call to a Name (line 580):

# Assigning a Call to a Name (line 580):

# Call to HasMember(...): (line 580)
# Processing the call arguments (line 580)
str_19544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 35), 'str', '__ifloordiv__')
# Getting the type of 'DynamicType' (line 580)
DynamicType_19545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 52), 'DynamicType', False)
int_19546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 65), 'int')
# Processing the call keyword arguments (line 580)
kwargs_19547 = {}
# Getting the type of 'HasMember' (line 580)
HasMember_19543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 25), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 580)
HasMember_call_result_19548 = invoke(stypy.reporting.localization.Localization(__file__, 580, 25), HasMember_19543, *[str_19544, DynamicType_19545, int_19546], **kwargs_19547)

# Assigning a type to the variable 'Overloads__ifloordiv__' (line 580)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 0), 'Overloads__ifloordiv__', HasMember_call_result_19548)

# Assigning a Call to a Name (line 581):

# Assigning a Call to a Name (line 581):

# Call to HasMember(...): (line 581)
# Processing the call arguments (line 581)
str_19550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 30), 'str', '__idiv__')
# Getting the type of 'DynamicType' (line 581)
DynamicType_19551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 42), 'DynamicType', False)
int_19552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 55), 'int')
# Processing the call keyword arguments (line 581)
kwargs_19553 = {}
# Getting the type of 'HasMember' (line 581)
HasMember_19549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 581)
HasMember_call_result_19554 = invoke(stypy.reporting.localization.Localization(__file__, 581, 20), HasMember_19549, *[str_19550, DynamicType_19551, int_19552], **kwargs_19553)

# Assigning a type to the variable 'Overloads__idiv__' (line 581)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 0), 'Overloads__idiv__', HasMember_call_result_19554)

# Assigning a Call to a Name (line 582):

# Assigning a Call to a Name (line 582):

# Call to HasMember(...): (line 582)
# Processing the call arguments (line 582)
str_19556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 34), 'str', '__itruediv__')
# Getting the type of 'DynamicType' (line 582)
DynamicType_19557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 50), 'DynamicType', False)
int_19558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 63), 'int')
# Processing the call keyword arguments (line 582)
kwargs_19559 = {}
# Getting the type of 'HasMember' (line 582)
HasMember_19555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 24), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 582)
HasMember_call_result_19560 = invoke(stypy.reporting.localization.Localization(__file__, 582, 24), HasMember_19555, *[str_19556, DynamicType_19557, int_19558], **kwargs_19559)

# Assigning a type to the variable 'Overloads__itruediv__' (line 582)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 0), 'Overloads__itruediv__', HasMember_call_result_19560)

# Assigning a Call to a Name (line 583):

# Assigning a Call to a Name (line 583):

# Call to HasMember(...): (line 583)
# Processing the call arguments (line 583)
str_19562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 30), 'str', '__imod__')
# Getting the type of 'DynamicType' (line 583)
DynamicType_19563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 42), 'DynamicType', False)
int_19564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 55), 'int')
# Processing the call keyword arguments (line 583)
kwargs_19565 = {}
# Getting the type of 'HasMember' (line 583)
HasMember_19561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 583)
HasMember_call_result_19566 = invoke(stypy.reporting.localization.Localization(__file__, 583, 20), HasMember_19561, *[str_19562, DynamicType_19563, int_19564], **kwargs_19565)

# Assigning a type to the variable 'Overloads__imod__' (line 583)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 0), 'Overloads__imod__', HasMember_call_result_19566)

# Assigning a Call to a Name (line 584):

# Assigning a Call to a Name (line 584):

# Call to HasMember(...): (line 584)
# Processing the call arguments (line 584)
str_19568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 33), 'str', '__idivmod__')
# Getting the type of 'DynamicType' (line 584)
DynamicType_19569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 48), 'DynamicType', False)
int_19570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 61), 'int')
# Processing the call keyword arguments (line 584)
kwargs_19571 = {}
# Getting the type of 'HasMember' (line 584)
HasMember_19567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 23), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 584)
HasMember_call_result_19572 = invoke(stypy.reporting.localization.Localization(__file__, 584, 23), HasMember_19567, *[str_19568, DynamicType_19569, int_19570], **kwargs_19571)

# Assigning a type to the variable 'Overloads__idivmod__' (line 584)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 0), 'Overloads__idivmod__', HasMember_call_result_19572)

# Assigning a Call to a Name (line 585):

# Assigning a Call to a Name (line 585):

# Call to HasMember(...): (line 585)
# Processing the call arguments (line 585)
str_19574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 30), 'str', '__ipow__')
# Getting the type of 'DynamicType' (line 585)
DynamicType_19575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 42), 'DynamicType', False)
int_19576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 55), 'int')
# Processing the call keyword arguments (line 585)
kwargs_19577 = {}
# Getting the type of 'HasMember' (line 585)
HasMember_19573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 585)
HasMember_call_result_19578 = invoke(stypy.reporting.localization.Localization(__file__, 585, 20), HasMember_19573, *[str_19574, DynamicType_19575, int_19576], **kwargs_19577)

# Assigning a type to the variable 'Overloads__ipow__' (line 585)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 0), 'Overloads__ipow__', HasMember_call_result_19578)

# Assigning a Call to a Name (line 586):

# Assigning a Call to a Name (line 586):

# Call to HasMember(...): (line 586)
# Processing the call arguments (line 586)
str_19580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 33), 'str', '__ilshift__')
# Getting the type of 'DynamicType' (line 586)
DynamicType_19581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 48), 'DynamicType', False)
int_19582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 61), 'int')
# Processing the call keyword arguments (line 586)
kwargs_19583 = {}
# Getting the type of 'HasMember' (line 586)
HasMember_19579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 23), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 586)
HasMember_call_result_19584 = invoke(stypy.reporting.localization.Localization(__file__, 586, 23), HasMember_19579, *[str_19580, DynamicType_19581, int_19582], **kwargs_19583)

# Assigning a type to the variable 'Overloads__ilshift__' (line 586)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 0), 'Overloads__ilshift__', HasMember_call_result_19584)

# Assigning a Call to a Name (line 587):

# Assigning a Call to a Name (line 587):

# Call to HasMember(...): (line 587)
# Processing the call arguments (line 587)
str_19586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 33), 'str', '__irshift__')
# Getting the type of 'DynamicType' (line 587)
DynamicType_19587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 48), 'DynamicType', False)
int_19588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 61), 'int')
# Processing the call keyword arguments (line 587)
kwargs_19589 = {}
# Getting the type of 'HasMember' (line 587)
HasMember_19585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 23), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 587)
HasMember_call_result_19590 = invoke(stypy.reporting.localization.Localization(__file__, 587, 23), HasMember_19585, *[str_19586, DynamicType_19587, int_19588], **kwargs_19589)

# Assigning a type to the variable 'Overloads__irshift__' (line 587)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 0), 'Overloads__irshift__', HasMember_call_result_19590)

# Assigning a Call to a Name (line 588):

# Assigning a Call to a Name (line 588):

# Call to HasMember(...): (line 588)
# Processing the call arguments (line 588)
str_19592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 30), 'str', '__iand__')
# Getting the type of 'DynamicType' (line 588)
DynamicType_19593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 42), 'DynamicType', False)
int_19594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 55), 'int')
# Processing the call keyword arguments (line 588)
kwargs_19595 = {}
# Getting the type of 'HasMember' (line 588)
HasMember_19591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 588)
HasMember_call_result_19596 = invoke(stypy.reporting.localization.Localization(__file__, 588, 20), HasMember_19591, *[str_19592, DynamicType_19593, int_19594], **kwargs_19595)

# Assigning a type to the variable 'Overloads__iand__' (line 588)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 0), 'Overloads__iand__', HasMember_call_result_19596)

# Assigning a Call to a Name (line 589):

# Assigning a Call to a Name (line 589):

# Call to HasMember(...): (line 589)
# Processing the call arguments (line 589)
str_19598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 29), 'str', '__ior__')
# Getting the type of 'DynamicType' (line 589)
DynamicType_19599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 40), 'DynamicType', False)
int_19600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 53), 'int')
# Processing the call keyword arguments (line 589)
kwargs_19601 = {}
# Getting the type of 'HasMember' (line 589)
HasMember_19597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 589)
HasMember_call_result_19602 = invoke(stypy.reporting.localization.Localization(__file__, 589, 19), HasMember_19597, *[str_19598, DynamicType_19599, int_19600], **kwargs_19601)

# Assigning a type to the variable 'Overloads__ior__' (line 589)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 0), 'Overloads__ior__', HasMember_call_result_19602)

# Assigning a Call to a Name (line 590):

# Assigning a Call to a Name (line 590):

# Call to HasMember(...): (line 590)
# Processing the call arguments (line 590)
str_19604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 30), 'str', '__ixor__')
# Getting the type of 'DynamicType' (line 590)
DynamicType_19605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 42), 'DynamicType', False)
int_19606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 55), 'int')
# Processing the call keyword arguments (line 590)
kwargs_19607 = {}
# Getting the type of 'HasMember' (line 590)
HasMember_19603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 590)
HasMember_call_result_19608 = invoke(stypy.reporting.localization.Localization(__file__, 590, 20), HasMember_19603, *[str_19604, DynamicType_19605, int_19606], **kwargs_19607)

# Assigning a type to the variable 'Overloads__ixor__' (line 590)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 0), 'Overloads__ixor__', HasMember_call_result_19608)

# Assigning a Call to a Name (line 593):

# Assigning a Call to a Name (line 593):

# Call to HasMember(...): (line 593)
# Processing the call arguments (line 593)
str_19610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 23), 'str', '__str__')
# Getting the type of 'str' (line 593)
str_19611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 34), 'str', False)
int_19612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 39), 'int')
# Processing the call keyword arguments (line 593)
kwargs_19613 = {}
# Getting the type of 'HasMember' (line 593)
HasMember_19609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 13), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 593)
HasMember_call_result_19614 = invoke(stypy.reporting.localization.Localization(__file__, 593, 13), HasMember_19609, *[str_19610, str_19611, int_19612], **kwargs_19613)

# Assigning a type to the variable 'Has__str__' (line 593)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 0), 'Has__str__', HasMember_call_result_19614)

# Assigning a Call to a Name (line 594):

# Assigning a Call to a Name (line 594):

# Call to HasMember(...): (line 594)
# Processing the call arguments (line 594)
str_19616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 24), 'str', '__repr__')
# Getting the type of 'str' (line 594)
str_19617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 36), 'str', False)
int_19618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 41), 'int')
# Processing the call keyword arguments (line 594)
kwargs_19619 = {}
# Getting the type of 'HasMember' (line 594)
HasMember_19615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 14), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 594)
HasMember_call_result_19620 = invoke(stypy.reporting.localization.Localization(__file__, 594, 14), HasMember_19615, *[str_19616, str_19617, int_19618], **kwargs_19619)

# Assigning a type to the variable 'Has__repr__' (line 594)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 0), 'Has__repr__', HasMember_call_result_19620)

# Assigning a Call to a Name (line 595):

# Assigning a Call to a Name (line 595):

# Call to HasMember(...): (line 595)
# Processing the call arguments (line 595)
str_19622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 27), 'str', '__unicode__')
# Getting the type of 'unicode' (line 595)
unicode_19623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 42), 'unicode', False)
int_19624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 51), 'int')
# Processing the call keyword arguments (line 595)
kwargs_19625 = {}
# Getting the type of 'HasMember' (line 595)
HasMember_19621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 17), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 595)
HasMember_call_result_19626 = invoke(stypy.reporting.localization.Localization(__file__, 595, 17), HasMember_19621, *[str_19622, unicode_19623, int_19624], **kwargs_19625)

# Assigning a type to the variable 'Has__unicode__' (line 595)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 0), 'Has__unicode__', HasMember_call_result_19626)

# Assigning a Call to a Name (line 596):

# Assigning a Call to a Name (line 596):

# Call to HasMember(...): (line 596)
# Processing the call arguments (line 596)
str_19628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 26), 'str', '__format__')
# Getting the type of 'str' (line 596)
str_19629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 40), 'str', False)
int_19630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 45), 'int')
# Processing the call keyword arguments (line 596)
kwargs_19631 = {}
# Getting the type of 'HasMember' (line 596)
HasMember_19627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 16), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 596)
HasMember_call_result_19632 = invoke(stypy.reporting.localization.Localization(__file__, 596, 16), HasMember_19627, *[str_19628, str_19629, int_19630], **kwargs_19631)

# Assigning a type to the variable 'Has__format__' (line 596)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 0), 'Has__format__', HasMember_call_result_19632)

# Assigning a Call to a Name (line 597):

# Assigning a Call to a Name (line 597):

# Call to HasMember(...): (line 597)
# Processing the call arguments (line 597)
str_19634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 24), 'str', '__hash__')
# Getting the type of 'int' (line 597)
int_19635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 36), 'int', False)
int_19636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 41), 'int')
# Processing the call keyword arguments (line 597)
kwargs_19637 = {}
# Getting the type of 'HasMember' (line 597)
HasMember_19633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 14), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 597)
HasMember_call_result_19638 = invoke(stypy.reporting.localization.Localization(__file__, 597, 14), HasMember_19633, *[str_19634, int_19635, int_19636], **kwargs_19637)

# Assigning a type to the variable 'Has__hash__' (line 597)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 0), 'Has__hash__', HasMember_call_result_19638)

# Assigning a Call to a Name (line 598):

# Assigning a Call to a Name (line 598):

# Call to HasMember(...): (line 598)
# Processing the call arguments (line 598)
str_19640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 27), 'str', '__nonzero__')
# Getting the type of 'bool' (line 598)
bool_19641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 42), 'bool', False)
int_19642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 48), 'int')
# Processing the call keyword arguments (line 598)
kwargs_19643 = {}
# Getting the type of 'HasMember' (line 598)
HasMember_19639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 17), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 598)
HasMember_call_result_19644 = invoke(stypy.reporting.localization.Localization(__file__, 598, 17), HasMember_19639, *[str_19640, bool_19641, int_19642], **kwargs_19643)

# Assigning a type to the variable 'Has__nonzero__' (line 598)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 0), 'Has__nonzero__', HasMember_call_result_19644)

# Assigning a Call to a Name (line 599):

# Assigning a Call to a Name (line 599):

# Call to HasMember(...): (line 599)
# Processing the call arguments (line 599)
str_19646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 23), 'str', '__dir__')
# Getting the type of 'DynamicType' (line 599)
DynamicType_19647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 34), 'DynamicType', False)
int_19648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 47), 'int')
# Processing the call keyword arguments (line 599)
kwargs_19649 = {}
# Getting the type of 'HasMember' (line 599)
HasMember_19645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 13), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 599)
HasMember_call_result_19650 = invoke(stypy.reporting.localization.Localization(__file__, 599, 13), HasMember_19645, *[str_19646, DynamicType_19647, int_19648], **kwargs_19649)

# Assigning a type to the variable 'Has__dir__' (line 599)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 0), 'Has__dir__', HasMember_call_result_19650)

# Assigning a Call to a Name (line 600):

# Assigning a Call to a Name (line 600):

# Call to HasMember(...): (line 600)
# Processing the call arguments (line 600)
str_19652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 26), 'str', '__sizeof__')
# Getting the type of 'int' (line 600)
int_19653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 40), 'int', False)
int_19654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 45), 'int')
# Processing the call keyword arguments (line 600)
kwargs_19655 = {}
# Getting the type of 'HasMember' (line 600)
HasMember_19651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 16), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 600)
HasMember_call_result_19656 = invoke(stypy.reporting.localization.Localization(__file__, 600, 16), HasMember_19651, *[str_19652, int_19653, int_19654], **kwargs_19655)

# Assigning a type to the variable 'Has__sizeof__' (line 600)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 0), 'Has__sizeof__', HasMember_call_result_19656)

# Assigning a Call to a Name (line 601):

# Assigning a Call to a Name (line 601):

# Call to Callable(...): (line 601)
# Processing the call keyword arguments (line 601)
kwargs_19658 = {}
# Getting the type of 'Callable' (line 601)
Callable_19657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 14), 'Callable', False)
# Calling Callable(args, kwargs) (line 601)
Callable_call_result_19659 = invoke(stypy.reporting.localization.Localization(__file__, 601, 14), Callable_19657, *[], **kwargs_19658)

# Assigning a type to the variable 'Has__call__' (line 601)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 0), 'Has__call__', Callable_call_result_19659)

# Assigning a Call to a Name (line 602):

# Assigning a Call to a Name (line 602):

# Call to HasMember(...): (line 602)
# Processing the call arguments (line 602)
str_19661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 23), 'str', '__mro__')
# Getting the type of 'DynamicType' (line 602)
DynamicType_19662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 34), 'DynamicType', False)
int_19663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 47), 'int')
# Processing the call keyword arguments (line 602)
kwargs_19664 = {}
# Getting the type of 'HasMember' (line 602)
HasMember_19660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 13), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 602)
HasMember_call_result_19665 = invoke(stypy.reporting.localization.Localization(__file__, 602, 13), HasMember_19660, *[str_19661, DynamicType_19662, int_19663], **kwargs_19664)

# Assigning a type to the variable 'Has__mro__' (line 602)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 0), 'Has__mro__', HasMember_call_result_19665)

# Assigning a Call to a Name (line 603):

# Assigning a Call to a Name (line 603):

# Call to HasMember(...): (line 603)
# Processing the call arguments (line 603)
str_19667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 25), 'str', '__class__')
# Getting the type of 'DynamicType' (line 603)
DynamicType_19668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 38), 'DynamicType', False)
int_19669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 51), 'int')
# Processing the call keyword arguments (line 603)
kwargs_19670 = {}
# Getting the type of 'HasMember' (line 603)
HasMember_19666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 15), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 603)
HasMember_call_result_19671 = invoke(stypy.reporting.localization.Localization(__file__, 603, 15), HasMember_19666, *[str_19667, DynamicType_19668, int_19669], **kwargs_19670)

# Assigning a type to the variable 'Has__class__' (line 603)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 0), 'Has__class__', HasMember_call_result_19671)

# Assigning a Call to a Name (line 607):

# Assigning a Call to a Name (line 607):

# Call to HasMember(...): (line 607)
# Processing the call arguments (line 607)
str_19673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 23), 'str', '__len__')
# Getting the type of 'int' (line 607)
int_19674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 34), 'int', False)
int_19675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 39), 'int')
# Processing the call keyword arguments (line 607)
kwargs_19676 = {}
# Getting the type of 'HasMember' (line 607)
HasMember_19672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 13), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 607)
HasMember_call_result_19677 = invoke(stypy.reporting.localization.Localization(__file__, 607, 13), HasMember_19672, *[str_19673, int_19674, int_19675], **kwargs_19676)

# Assigning a type to the variable 'Has__len__' (line 607)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 0), 'Has__len__', HasMember_call_result_19677)

# Assigning a Call to a Name (line 608):

# Assigning a Call to a Name (line 608):

# Call to HasMember(...): (line 608)
# Processing the call arguments (line 608)
str_19679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 27), 'str', '__getitem__')
# Getting the type of 'DynamicType' (line 608)
DynamicType_19680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 42), 'DynamicType', False)
int_19681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 55), 'int')
# Processing the call keyword arguments (line 608)
kwargs_19682 = {}
# Getting the type of 'HasMember' (line 608)
HasMember_19678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 17), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 608)
HasMember_call_result_19683 = invoke(stypy.reporting.localization.Localization(__file__, 608, 17), HasMember_19678, *[str_19679, DynamicType_19680, int_19681], **kwargs_19682)

# Assigning a type to the variable 'Has__getitem__' (line 608)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 0), 'Has__getitem__', HasMember_call_result_19683)

# Assigning a Call to a Name (line 609):

# Assigning a Call to a Name (line 609):

# Call to HasMember(...): (line 609)
# Processing the call arguments (line 609)
str_19685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 27), 'str', '__setitem__')
# Getting the type of 'types' (line 609)
types_19686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 42), 'types', False)
# Obtaining the member 'NoneType' of a type (line 609)
NoneType_19687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 42), types_19686, 'NoneType')
int_19688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 58), 'int')
# Processing the call keyword arguments (line 609)
kwargs_19689 = {}
# Getting the type of 'HasMember' (line 609)
HasMember_19684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 17), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 609)
HasMember_call_result_19690 = invoke(stypy.reporting.localization.Localization(__file__, 609, 17), HasMember_19684, *[str_19685, NoneType_19687, int_19688], **kwargs_19689)

# Assigning a type to the variable 'Has__setitem__' (line 609)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 0), 'Has__setitem__', HasMember_call_result_19690)

# Assigning a Call to a Name (line 611):

# Assigning a Call to a Name (line 611):

# Call to HasMember(...): (line 611)
# Processing the call arguments (line 611)
str_19692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 27), 'str', '__delitem__')
# Getting the type of 'int' (line 611)
int_19693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 42), 'int', False)
int_19694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 47), 'int')
# Processing the call keyword arguments (line 611)
kwargs_19695 = {}
# Getting the type of 'HasMember' (line 611)
HasMember_19691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 17), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 611)
HasMember_call_result_19696 = invoke(stypy.reporting.localization.Localization(__file__, 611, 17), HasMember_19691, *[str_19692, int_19693, int_19694], **kwargs_19695)

# Assigning a type to the variable 'Has__delitem__' (line 611)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 0), 'Has__delitem__', HasMember_call_result_19696)

# Assigning a Call to a Name (line 612):

# Assigning a Call to a Name (line 612):

# Call to HasMember(...): (line 612)
# Processing the call arguments (line 612)
str_19698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 24), 'str', '__iter__')
# Getting the type of 'DynamicType' (line 612)
DynamicType_19699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 36), 'DynamicType', False)
int_19700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 49), 'int')
# Processing the call keyword arguments (line 612)
kwargs_19701 = {}
# Getting the type of 'HasMember' (line 612)
HasMember_19697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 14), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 612)
HasMember_call_result_19702 = invoke(stypy.reporting.localization.Localization(__file__, 612, 14), HasMember_19697, *[str_19698, DynamicType_19699, int_19700], **kwargs_19701)

# Assigning a type to the variable 'Has__iter__' (line 612)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 0), 'Has__iter__', HasMember_call_result_19702)

# Assigning a Call to a Name (line 613):

# Assigning a Call to a Name (line 613):

# Call to HasMember(...): (line 613)
# Processing the call arguments (line 613)
str_19704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 28), 'str', '__reversed__')
# Getting the type of 'int' (line 613)
int_19705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 44), 'int', False)
int_19706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 49), 'int')
# Processing the call keyword arguments (line 613)
kwargs_19707 = {}
# Getting the type of 'HasMember' (line 613)
HasMember_19703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 613)
HasMember_call_result_19708 = invoke(stypy.reporting.localization.Localization(__file__, 613, 18), HasMember_19703, *[str_19704, int_19705, int_19706], **kwargs_19707)

# Assigning a type to the variable 'Has__reversed__' (line 613)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 0), 'Has__reversed__', HasMember_call_result_19708)

# Assigning a Call to a Name (line 614):

# Assigning a Call to a Name (line 614):

# Call to HasMember(...): (line 614)
# Processing the call arguments (line 614)
str_19710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 28), 'str', '__contains__')
# Getting the type of 'int' (line 614)
int_19711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 44), 'int', False)
int_19712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 49), 'int')
# Processing the call keyword arguments (line 614)
kwargs_19713 = {}
# Getting the type of 'HasMember' (line 614)
HasMember_19709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 614)
HasMember_call_result_19714 = invoke(stypy.reporting.localization.Localization(__file__, 614, 18), HasMember_19709, *[str_19710, int_19711, int_19712], **kwargs_19713)

# Assigning a type to the variable 'Has__contains__' (line 614)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 0), 'Has__contains__', HasMember_call_result_19714)

# Assigning a Call to a Name (line 615):

# Assigning a Call to a Name (line 615):

# Call to HasMember(...): (line 615)
# Processing the call arguments (line 615)
str_19716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 27), 'str', '__missing__')
# Getting the type of 'int' (line 615)
int_19717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 42), 'int', False)
int_19718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 47), 'int')
# Processing the call keyword arguments (line 615)
kwargs_19719 = {}
# Getting the type of 'HasMember' (line 615)
HasMember_19715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 17), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 615)
HasMember_call_result_19720 = invoke(stypy.reporting.localization.Localization(__file__, 615, 17), HasMember_19715, *[str_19716, int_19717, int_19718], **kwargs_19719)

# Assigning a type to the variable 'Has__missing__' (line 615)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 0), 'Has__missing__', HasMember_call_result_19720)

# Assigning a Call to a Name (line 616):

# Assigning a Call to a Name (line 616):

# Call to HasMember(...): (line 616)
# Processing the call arguments (line 616)
str_19722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 28), 'str', '__getslice__')
# Getting the type of 'DynamicType' (line 616)
DynamicType_19723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 44), 'DynamicType', False)
int_19724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 57), 'int')
# Processing the call keyword arguments (line 616)
kwargs_19725 = {}
# Getting the type of 'HasMember' (line 616)
HasMember_19721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 616)
HasMember_call_result_19726 = invoke(stypy.reporting.localization.Localization(__file__, 616, 18), HasMember_19721, *[str_19722, DynamicType_19723, int_19724], **kwargs_19725)

# Assigning a type to the variable 'Has__getslice__' (line 616)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 0), 'Has__getslice__', HasMember_call_result_19726)

# Assigning a Call to a Name (line 617):

# Assigning a Call to a Name (line 617):

# Call to HasMember(...): (line 617)
# Processing the call arguments (line 617)
str_19728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 28), 'str', '__setslice__')
# Getting the type of 'types' (line 617)
types_19729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 44), 'types', False)
# Obtaining the member 'NoneType' of a type (line 617)
NoneType_19730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 44), types_19729, 'NoneType')
int_19731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 60), 'int')
# Processing the call keyword arguments (line 617)
kwargs_19732 = {}
# Getting the type of 'HasMember' (line 617)
HasMember_19727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 617)
HasMember_call_result_19733 = invoke(stypy.reporting.localization.Localization(__file__, 617, 18), HasMember_19727, *[str_19728, NoneType_19730, int_19731], **kwargs_19732)

# Assigning a type to the variable 'Has__setslice__' (line 617)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 0), 'Has__setslice__', HasMember_call_result_19733)

# Assigning a Call to a Name (line 618):

# Assigning a Call to a Name (line 618):

# Call to HasMember(...): (line 618)
# Processing the call arguments (line 618)
str_19735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 28), 'str', '__delslice__')
# Getting the type of 'types' (line 618)
types_19736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 44), 'types', False)
# Obtaining the member 'NoneType' of a type (line 618)
NoneType_19737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 44), types_19736, 'NoneType')
int_19738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 60), 'int')
# Processing the call keyword arguments (line 618)
kwargs_19739 = {}
# Getting the type of 'HasMember' (line 618)
HasMember_19734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 618)
HasMember_call_result_19740 = invoke(stypy.reporting.localization.Localization(__file__, 618, 18), HasMember_19734, *[str_19735, NoneType_19737, int_19738], **kwargs_19739)

# Assigning a type to the variable 'Has__delslice__' (line 618)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 0), 'Has__delslice__', HasMember_call_result_19740)

# Assigning a Call to a Name (line 619):

# Assigning a Call to a Name (line 619):

# Call to HasMember(...): (line 619)
# Processing the call arguments (line 619)
str_19742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 22), 'str', 'next')
# Getting the type of 'DynamicType' (line 619)
DynamicType_19743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 30), 'DynamicType', False)
int_19744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 43), 'int')
# Processing the call keyword arguments (line 619)
kwargs_19745 = {}
# Getting the type of 'HasMember' (line 619)
HasMember_19741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 12), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 619)
HasMember_call_result_19746 = invoke(stypy.reporting.localization.Localization(__file__, 619, 12), HasMember_19741, *[str_19742, DynamicType_19743, int_19744], **kwargs_19745)

# Assigning a type to the variable 'Has__next' (line 619)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 0), 'Has__next', HasMember_call_result_19746)

# Assigning a Call to a Name (line 622):

# Assigning a Call to a Name (line 622):

# Call to HasMember(...): (line 622)
# Processing the call arguments (line 622)
str_19748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 25), 'str', '__enter__')
# Getting the type of 'int' (line 622)
int_19749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 38), 'int', False)
int_19750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 43), 'int')
# Processing the call keyword arguments (line 622)
kwargs_19751 = {}
# Getting the type of 'HasMember' (line 622)
HasMember_19747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 15), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 622)
HasMember_call_result_19752 = invoke(stypy.reporting.localization.Localization(__file__, 622, 15), HasMember_19747, *[str_19748, int_19749, int_19750], **kwargs_19751)

# Assigning a type to the variable 'Has__enter__' (line 622)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 0), 'Has__enter__', HasMember_call_result_19752)

# Assigning a Call to a Name (line 623):

# Assigning a Call to a Name (line 623):

# Call to HasMember(...): (line 623)
# Processing the call arguments (line 623)
str_19754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 24), 'str', '__exit__')
# Getting the type of 'int' (line 623)
int_19755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 36), 'int', False)
int_19756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 41), 'int')
# Processing the call keyword arguments (line 623)
kwargs_19757 = {}
# Getting the type of 'HasMember' (line 623)
HasMember_19753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 14), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 623)
HasMember_call_result_19758 = invoke(stypy.reporting.localization.Localization(__file__, 623, 14), HasMember_19753, *[str_19754, int_19755, int_19756], **kwargs_19757)

# Assigning a type to the variable 'Has__exit__' (line 623)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 0), 'Has__exit__', HasMember_call_result_19758)

# Assigning a Call to a Name (line 626):

# Assigning a Call to a Name (line 626):

# Call to HasMember(...): (line 626)
# Processing the call arguments (line 626)
str_19760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 23), 'str', '__get__')
# Getting the type of 'DynamicType' (line 626)
DynamicType_19761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 34), 'DynamicType', False)
int_19762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 47), 'int')
# Processing the call keyword arguments (line 626)
kwargs_19763 = {}
# Getting the type of 'HasMember' (line 626)
HasMember_19759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 13), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 626)
HasMember_call_result_19764 = invoke(stypy.reporting.localization.Localization(__file__, 626, 13), HasMember_19759, *[str_19760, DynamicType_19761, int_19762], **kwargs_19763)

# Assigning a type to the variable 'Has__get__' (line 626)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 0), 'Has__get__', HasMember_call_result_19764)

# Assigning a Call to a Name (line 627):

# Assigning a Call to a Name (line 627):

# Call to HasMember(...): (line 627)
# Processing the call arguments (line 627)
str_19766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 23), 'str', '__set__')
# Getting the type of 'types' (line 627)
types_19767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 34), 'types', False)
# Obtaining the member 'NoneType' of a type (line 627)
NoneType_19768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 34), types_19767, 'NoneType')
int_19769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 50), 'int')
# Processing the call keyword arguments (line 627)
kwargs_19770 = {}
# Getting the type of 'HasMember' (line 627)
HasMember_19765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 13), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 627)
HasMember_call_result_19771 = invoke(stypy.reporting.localization.Localization(__file__, 627, 13), HasMember_19765, *[str_19766, NoneType_19768, int_19769], **kwargs_19770)

# Assigning a type to the variable 'Has__set__' (line 627)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 0), 'Has__set__', HasMember_call_result_19771)

# Assigning a Call to a Name (line 628):

# Assigning a Call to a Name (line 628):

# Call to HasMember(...): (line 628)
# Processing the call arguments (line 628)
str_19773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 23), 'str', '__del__')
# Getting the type of 'types' (line 628)
types_19774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 34), 'types', False)
# Obtaining the member 'NoneType' of a type (line 628)
NoneType_19775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 34), types_19774, 'NoneType')
int_19776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 50), 'int')
# Processing the call keyword arguments (line 628)
kwargs_19777 = {}
# Getting the type of 'HasMember' (line 628)
HasMember_19772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 13), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 628)
HasMember_call_result_19778 = invoke(stypy.reporting.localization.Localization(__file__, 628, 13), HasMember_19772, *[str_19773, NoneType_19775, int_19776], **kwargs_19777)

# Assigning a type to the variable 'Has__del__' (line 628)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 0), 'Has__del__', HasMember_call_result_19778)

# Assigning a Call to a Name (line 632):

# Assigning a Call to a Name (line 632):

# Call to HasMember(...): (line 632)
# Processing the call arguments (line 632)
str_19780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 24), 'str', '__copy__')
# Getting the type of 'DynamicType' (line 632)
DynamicType_19781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 36), 'DynamicType', False)
int_19782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 49), 'int')
# Processing the call keyword arguments (line 632)
kwargs_19783 = {}
# Getting the type of 'HasMember' (line 632)
HasMember_19779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 14), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 632)
HasMember_call_result_19784 = invoke(stypy.reporting.localization.Localization(__file__, 632, 14), HasMember_19779, *[str_19780, DynamicType_19781, int_19782], **kwargs_19783)

# Assigning a type to the variable 'Has__copy__' (line 632)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 0), 'Has__copy__', HasMember_call_result_19784)

# Assigning a Call to a Name (line 633):

# Assigning a Call to a Name (line 633):

# Call to HasMember(...): (line 633)
# Processing the call arguments (line 633)
str_19786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 28), 'str', '__deepcopy__')
# Getting the type of 'DynamicType' (line 633)
DynamicType_19787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 44), 'DynamicType', False)
int_19788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 57), 'int')
# Processing the call keyword arguments (line 633)
kwargs_19789 = {}
# Getting the type of 'HasMember' (line 633)
HasMember_19785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 633)
HasMember_call_result_19790 = invoke(stypy.reporting.localization.Localization(__file__, 633, 18), HasMember_19785, *[str_19786, DynamicType_19787, int_19788], **kwargs_19789)

# Assigning a type to the variable 'Has__deepcopy__' (line 633)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 0), 'Has__deepcopy__', HasMember_call_result_19790)

# Assigning a Call to a Name (line 636):

# Assigning a Call to a Name (line 636):

# Call to HasMember(...): (line 636)
# Processing the call arguments (line 636)
str_19792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 31), 'str', '__getinitargs__')
# Getting the type of 'DynamicType' (line 636)
DynamicType_19793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 50), 'DynamicType', False)
int_19794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 63), 'int')
# Processing the call keyword arguments (line 636)
kwargs_19795 = {}
# Getting the type of 'HasMember' (line 636)
HasMember_19791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 21), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 636)
HasMember_call_result_19796 = invoke(stypy.reporting.localization.Localization(__file__, 636, 21), HasMember_19791, *[str_19792, DynamicType_19793, int_19794], **kwargs_19795)

# Assigning a type to the variable 'Has__getinitargs__' (line 636)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 0), 'Has__getinitargs__', HasMember_call_result_19796)

# Assigning a Call to a Name (line 637):

# Assigning a Call to a Name (line 637):

# Call to HasMember(...): (line 637)
# Processing the call arguments (line 637)
str_19798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 30), 'str', '__getnewargs__')
# Getting the type of 'DynamicType' (line 637)
DynamicType_19799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 48), 'DynamicType', False)
int_19800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 61), 'int')
# Processing the call keyword arguments (line 637)
kwargs_19801 = {}
# Getting the type of 'HasMember' (line 637)
HasMember_19797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 637)
HasMember_call_result_19802 = invoke(stypy.reporting.localization.Localization(__file__, 637, 20), HasMember_19797, *[str_19798, DynamicType_19799, int_19800], **kwargs_19801)

# Assigning a type to the variable 'Has__getnewargs__' (line 637)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 0), 'Has__getnewargs__', HasMember_call_result_19802)

# Assigning a Call to a Name (line 638):

# Assigning a Call to a Name (line 638):

# Call to HasMember(...): (line 638)
# Processing the call arguments (line 638)
str_19804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 28), 'str', '__getstate__')
# Getting the type of 'DynamicType' (line 638)
DynamicType_19805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 44), 'DynamicType', False)
int_19806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 57), 'int')
# Processing the call keyword arguments (line 638)
kwargs_19807 = {}
# Getting the type of 'HasMember' (line 638)
HasMember_19803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 638)
HasMember_call_result_19808 = invoke(stypy.reporting.localization.Localization(__file__, 638, 18), HasMember_19803, *[str_19804, DynamicType_19805, int_19806], **kwargs_19807)

# Assigning a type to the variable 'Has__getstate__' (line 638)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 0), 'Has__getstate__', HasMember_call_result_19808)

# Assigning a Call to a Name (line 639):

# Assigning a Call to a Name (line 639):

# Call to HasMember(...): (line 639)
# Processing the call arguments (line 639)
str_19810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 28), 'str', '__setstate__')
# Getting the type of 'types' (line 639)
types_19811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 44), 'types', False)
# Obtaining the member 'NoneType' of a type (line 639)
NoneType_19812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 44), types_19811, 'NoneType')
int_19813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 60), 'int')
# Processing the call keyword arguments (line 639)
kwargs_19814 = {}
# Getting the type of 'HasMember' (line 639)
HasMember_19809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 639)
HasMember_call_result_19815 = invoke(stypy.reporting.localization.Localization(__file__, 639, 18), HasMember_19809, *[str_19810, NoneType_19812, int_19813], **kwargs_19814)

# Assigning a type to the variable 'Has__setstate__' (line 639)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 0), 'Has__setstate__', HasMember_call_result_19815)

# Assigning a Call to a Name (line 640):

# Assigning a Call to a Name (line 640):

# Call to HasMember(...): (line 640)
# Processing the call arguments (line 640)
str_19817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 26), 'str', '__reduce__')
# Getting the type of 'DynamicType' (line 640)
DynamicType_19818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 40), 'DynamicType', False)
int_19819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 53), 'int')
# Processing the call keyword arguments (line 640)
kwargs_19820 = {}
# Getting the type of 'HasMember' (line 640)
HasMember_19816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 16), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 640)
HasMember_call_result_19821 = invoke(stypy.reporting.localization.Localization(__file__, 640, 16), HasMember_19816, *[str_19817, DynamicType_19818, int_19819], **kwargs_19820)

# Assigning a type to the variable 'Has__reduce__' (line 640)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 0), 'Has__reduce__', HasMember_call_result_19821)

# Assigning a Call to a Name (line 641):

# Assigning a Call to a Name (line 641):

# Call to HasMember(...): (line 641)
# Processing the call arguments (line 641)
str_19823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 29), 'str', '__reduce_ex__')
# Getting the type of 'DynamicType' (line 641)
DynamicType_19824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 46), 'DynamicType', False)
int_19825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 59), 'int')
# Processing the call keyword arguments (line 641)
kwargs_19826 = {}
# Getting the type of 'HasMember' (line 641)
HasMember_19822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 641)
HasMember_call_result_19827 = invoke(stypy.reporting.localization.Localization(__file__, 641, 19), HasMember_19822, *[str_19823, DynamicType_19824, int_19825], **kwargs_19826)

# Assigning a type to the variable 'Has__reduce_ex__' (line 641)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 0), 'Has__reduce_ex__', HasMember_call_result_19827)

# Assigning a Call to a Name (line 645):

# Assigning a Call to a Name (line 645):

# Call to DynamicType(...): (line 645)
# Processing the call keyword arguments (line 645)
kwargs_19829 = {}
# Getting the type of 'DynamicType' (line 645)
DynamicType_19828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 10), 'DynamicType', False)
# Calling DynamicType(args, kwargs) (line 645)
DynamicType_call_result_19830 = invoke(stypy.reporting.localization.Localization(__file__, 645, 10), DynamicType_19828, *[], **kwargs_19829)

# Assigning a type to the variable 'AnyType' (line 645)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 0), 'AnyType', DynamicType_call_result_19830)

# Assigning a Call to a Name (line 646):

# Assigning a Call to a Name (line 646):

# Call to SupportsStructuralIntercession(...): (line 646)
# Processing the call keyword arguments (line 646)
kwargs_19832 = {}
# Getting the type of 'SupportsStructuralIntercession' (line 646)
SupportsStructuralIntercession_19831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 29), 'SupportsStructuralIntercession', False)
# Calling SupportsStructuralIntercession(args, kwargs) (line 646)
SupportsStructuralIntercession_call_result_19833 = invoke(stypy.reporting.localization.Localization(__file__, 646, 29), SupportsStructuralIntercession_19831, *[], **kwargs_19832)

# Assigning a type to the variable 'StructuralIntercessionType' (line 646)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 0), 'StructuralIntercessionType', SupportsStructuralIntercession_call_result_19833)

# Assigning a Call to a Name (line 649):

# Assigning a Call to a Name (line 649):

# Call to IsHashable(...): (line 649)
# Processing the call keyword arguments (line 649)
kwargs_19835 = {}
# Getting the type of 'IsHashable' (line 649)
IsHashable_19834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 11), 'IsHashable', False)
# Calling IsHashable(args, kwargs) (line 649)
IsHashable_call_result_19836 = invoke(stypy.reporting.localization.Localization(__file__, 649, 11), IsHashable_19834, *[], **kwargs_19835)

# Assigning a type to the variable 'Hashable' (line 649)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 0), 'Hashable', IsHashable_call_result_19836)

# Assigning a Call to a Name (line 650):

# Assigning a Call to a Name (line 650):

# Call to TypeObject(...): (line 650)
# Processing the call keyword arguments (line 650)
kwargs_19838 = {}
# Getting the type of 'TypeObject' (line 650)
TypeObject_19837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 7), 'TypeObject', False)
# Calling TypeObject(args, kwargs) (line 650)
TypeObject_call_result_19839 = invoke(stypy.reporting.localization.Localization(__file__, 650, 7), TypeObject_19837, *[], **kwargs_19838)

# Assigning a type to the variable 'Type' (line 650)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 0), 'Type', TypeObject_call_result_19839)

# Assigning a Call to a Name (line 651):

# Assigning a Call to a Name (line 651):

# Call to InstanceOfType(...): (line 651)
# Processing the call keyword arguments (line 651)
kwargs_19841 = {}
# Getting the type of 'InstanceOfType' (line 651)
InstanceOfType_19840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 15), 'InstanceOfType', False)
# Calling InstanceOfType(args, kwargs) (line 651)
InstanceOfType_call_result_19842 = invoke(stypy.reporting.localization.Localization(__file__, 651, 15), InstanceOfType_19840, *[], **kwargs_19841)

# Assigning a type to the variable 'TypeInstance' (line 651)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 0), 'TypeInstance', InstanceOfType_call_result_19842)

# Assigning a Call to a Name (line 652):

# Assigning a Call to a Name (line 652):

# Call to VarArgType(...): (line 652)
# Processing the call keyword arguments (line 652)
kwargs_19844 = {}
# Getting the type of 'VarArgType' (line 652)
VarArgType_19843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 10), 'VarArgType', False)
# Calling VarArgType(args, kwargs) (line 652)
VarArgType_call_result_19845 = invoke(stypy.reporting.localization.Localization(__file__, 652, 10), VarArgType_19843, *[], **kwargs_19844)

# Assigning a type to the variable 'VarArgs' (line 652)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 0), 'VarArgs', VarArgType_call_result_19845)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
