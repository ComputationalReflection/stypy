
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import types
2: import collections
3: 
4: from ....python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import ExtraTypeDefinitions
5: from type_group_copy import TypeGroup
6: from ....errors_copy.type_error_copy import TypeError
7: from ....python_lib_copy.python_types_copy.type_inference_copy import union_type_copy
8: from ....python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType
9: from ....errors_copy.type_warning_copy import TypeWarning
10: from ....python_lib_copy.python_types_copy.type_inference_copy import type_inference_proxy_management_copy
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

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import ExtraTypeDefinitions' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')
import_14404 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy')

if (type(import_14404) is not StypyTypeError):

    if (import_14404 != 'pyd_module'):
        __import__(import_14404)
        sys_modules_14405 = sys.modules[import_14404]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', sys_modules_14405.module_type_store, module_type_store, ['ExtraTypeDefinitions'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_14405, sys_modules_14405.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import ExtraTypeDefinitions

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', None, module_type_store, ['ExtraTypeDefinitions'], [ExtraTypeDefinitions])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', import_14404)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from type_group_copy import TypeGroup' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')
import_14406 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'type_group_copy')

if (type(import_14406) is not StypyTypeError):

    if (import_14406 != 'pyd_module'):
        __import__(import_14406)
        sys_modules_14407 = sys.modules[import_14406]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'type_group_copy', sys_modules_14407.module_type_store, module_type_store, ['TypeGroup'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_14407, sys_modules_14407.module_type_store, module_type_store)
    else:
        from type_group_copy import TypeGroup

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'type_group_copy', None, module_type_store, ['TypeGroup'], [TypeGroup])

else:
    # Assigning a type to the variable 'type_group_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'type_group_copy', import_14406)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')
import_14408 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy')

if (type(import_14408) is not StypyTypeError):

    if (import_14408 != 'pyd_module'):
        __import__(import_14408)
        sys_modules_14409 = sys.modules[import_14408]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', sys_modules_14409.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_14409, sys_modules_14409.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', import_14408)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')
import_14410 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_14410) is not StypyTypeError):

    if (import_14410 != 'pyd_module'):
        __import__(import_14410)
        sys_modules_14411 = sys.modules[import_14410]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', sys_modules_14411.module_type_store, module_type_store, ['union_type_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_14411, sys_modules_14411.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['union_type_copy'], [union_type_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', import_14410)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')
import_14412 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy')

if (type(import_14412) is not StypyTypeError):

    if (import_14412 != 'pyd_module'):
        __import__(import_14412)
        sys_modules_14413 = sys.modules[import_14412]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', sys_modules_14413.module_type_store, module_type_store, ['UndefinedType'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_14413, sys_modules_14413.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', None, module_type_store, ['UndefinedType'], [UndefinedType])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', import_14412)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy import TypeWarning' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')
import_14414 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy')

if (type(import_14414) is not StypyTypeError):

    if (import_14414 != 'pyd_module'):
        __import__(import_14414)
        sys_modules_14415 = sys.modules[import_14414]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy', sys_modules_14415.module_type_store, module_type_store, ['TypeWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_14415, sys_modules_14415.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy import TypeWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy', None, module_type_store, ['TypeWarning'], [TypeWarning])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy', import_14414)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import type_inference_proxy_management_copy' statement (line 10)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')
import_14416 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_14416) is not StypyTypeError):

    if (import_14416 != 'pyd_module'):
        __import__(import_14416)
        sys_modules_14417 = sys.modules[import_14416]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', sys_modules_14417.module_type_store, module_type_store, ['type_inference_proxy_management_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_14417, sys_modules_14417.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import type_inference_proxy_management_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['type_inference_proxy_management_copy'], [type_inference_proxy_management_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', import_14416)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')

str_14418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, (-1)), 'str', '\nFile to define all type groups available to form type rules\n')
# Declaration of the 'DependentType' class

class DependentType:
    str_14419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, (-1)), 'str', "\n    A DependentType is a special base class that indicates that a type group has to be called to obtain the real\n    type it represent. Call is done using the parameters that are trying to match the rule. For example, imagine that\n    we call the + operator with an object that defines the __add__ method and another type to add to. With an object\n    that defines an __add__ method we don't really know what will be the result of calling __add__ over this object\n    with the second parameter, so the __add__ method has to be called (well, in fact, the type inference equivalent\n    version of the __add__ method will be called) to obtain the real return type.\n\n    Dependent types are a powerful mechanism to calculate the return type of operations that depend on calls to\n    certain object members or even to detect incorrect definitions of members in objects (__int__ method defined in\n    object that do not return int, for example).\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 31)
        False_14420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 37), 'False')
        defaults = [False_14420]
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

        str_14421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, (-1)), 'str', '\n        Build a Dependent type instance\n        :param report_errors: Flag to indicate if errors found when calling this type will be reported or not (in that\n        case other code will do it)\n        ')
        
        # Assigning a Name to a Attribute (line 37):
        
        # Assigning a Name to a Attribute (line 37):
        # Getting the type of 'report_errors' (line 37)
        report_errors_14422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 29), 'report_errors')
        # Getting the type of 'self' (line 37)
        self_14423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'self')
        # Setting the type of the member 'report_errors' of a type (line 37)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), self_14423, 'report_errors', report_errors_14422)
        
        # Assigning a Num to a Attribute (line 38):
        
        # Assigning a Num to a Attribute (line 38):
        int_14424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 26), 'int')
        # Getting the type of 'self' (line 38)
        self_14425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self')
        # Setting the type of the member 'call_arity' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_14425, 'call_arity', int_14424)
        
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

        str_14426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'str', '\n        Call the dependent type. Empty in this implementation, concrete calls must be defined in subclasses\n        ')
        pass
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 40)
        stypy_return_type_14427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14427)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_14427


# Assigning a type to the variable 'DependentType' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'DependentType', DependentType)
str_14428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, (-1)), 'str', '\nType groups with special meaning. All of them define a __eq__ method that indicates if the passed type matches with\nthe type group, storing this passed type. They also define a __call__ method that actually perform the type checking\nand calculate the return type. __eq__ and __call__ methods are called sequentially if __eq__ result is True, so the\nstorage of the passed type is safe to use in the __call__ as each time an __eq__ is called is replaced. This is the\nway the type rule checking mechanism works: TypeGroups are not meant to be used in other parts of the stypy runtime,\nand if they do, only the __eq__ method should be used to check if a type belongs to a group.\n')
# Declaration of the 'HasMember' class
# Getting the type of 'TypeGroup' (line 57)
TypeGroup_14429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'TypeGroup')
# Getting the type of 'DependentType' (line 57)
DependentType_14430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 27), 'DependentType')

class HasMember(TypeGroup_14429, DependentType_14430, ):
    str_14431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, (-1)), 'str', '\n        Type of any object that has a member with the specified arity, and that can be called with the corresponding\n        params.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_14432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 64), 'int')
        # Getting the type of 'False' (line 63)
        False_14433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 81), 'False')
        defaults = [int_14432, False_14433]
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
        self_14436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 31), 'self', False)
        # Getting the type of 'report_errors' (line 64)
        report_errors_14437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 37), 'report_errors', False)
        # Processing the call keyword arguments (line 64)
        kwargs_14438 = {}
        # Getting the type of 'DependentType' (line 64)
        DependentType_14434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'DependentType', False)
        # Obtaining the member '__init__' of a type (line 64)
        init___14435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), DependentType_14434, '__init__')
        # Calling __init__(args, kwargs) (line 64)
        init___call_result_14439 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), init___14435, *[self_14436, report_errors_14437], **kwargs_14438)
        
        
        # Call to __init__(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'self' (line 65)
        self_14442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_14443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        
        # Processing the call keyword arguments (line 65)
        kwargs_14444 = {}
        # Getting the type of 'TypeGroup' (line 65)
        TypeGroup_14440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 65)
        init___14441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), TypeGroup_14440, '__init__')
        # Calling __init__(args, kwargs) (line 65)
        init___call_result_14445 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), init___14441, *[self_14442, list_14443], **kwargs_14444)
        
        
        # Assigning a Name to a Attribute (line 66):
        
        # Assigning a Name to a Attribute (line 66):
        # Getting the type of 'member' (line 66)
        member_14446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 22), 'member')
        # Getting the type of 'self' (line 66)
        self_14447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self')
        # Setting the type of the member 'member' of a type (line 66)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_14447, 'member', member_14446)
        
        # Assigning a Name to a Attribute (line 67):
        
        # Assigning a Name to a Attribute (line 67):
        # Getting the type of 'expected_return_type' (line 67)
        expected_return_type_14448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 36), 'expected_return_type')
        # Getting the type of 'self' (line 67)
        self_14449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self')
        # Setting the type of the member 'expected_return_type' of a type (line 67)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_14449, 'expected_return_type', expected_return_type_14448)
        
        # Assigning a Name to a Attribute (line 68):
        
        # Assigning a Name to a Attribute (line 68):
        # Getting the type of 'None' (line 68)
        None_14450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 26), 'None')
        # Getting the type of 'self' (line 68)
        self_14451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self')
        # Setting the type of the member 'member_obj' of a type (line 68)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_14451, 'member_obj', None_14450)
        
        # Assigning a Name to a Attribute (line 69):
        
        # Assigning a Name to a Attribute (line 69):
        # Getting the type of 'call_arity' (line 69)
        call_arity_14452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 26), 'call_arity')
        # Getting the type of 'self' (line 69)
        self_14453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'self')
        # Setting the type of the member 'call_arity' of a type (line 69)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), self_14453, 'call_arity', call_arity_14452)
        
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
        str_14454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 15), 'str', '(')
        # Assigning a type to the variable 'str_' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'str_', str_14454)
        
        
        # Call to range(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'self' (line 73)
        self_14456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'self', False)
        # Obtaining the member 'call_arity' of a type (line 73)
        call_arity_14457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), self_14456, 'call_arity')
        # Processing the call keyword arguments (line 73)
        kwargs_14458 = {}
        # Getting the type of 'range' (line 73)
        range_14455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 17), 'range', False)
        # Calling range(args, kwargs) (line 73)
        range_call_result_14459 = invoke(stypy.reporting.localization.Localization(__file__, 73, 17), range_14455, *[call_arity_14457], **kwargs_14458)
        
        # Assigning a type to the variable 'range_call_result_14459' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'range_call_result_14459', range_call_result_14459)
        # Testing if the for loop is going to be iterated (line 73)
        # Testing the type of a for loop iterable (line 73)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 73, 8), range_call_result_14459)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 73, 8), range_call_result_14459):
            # Getting the type of the for loop variable (line 73)
            for_loop_var_14460 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 73, 8), range_call_result_14459)
            # Assigning a type to the variable 'i' (line 73)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'i', for_loop_var_14460)
            # SSA begins for a for statement (line 73)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'str_' (line 74)
            str__14461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'str_')
            str_14462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 20), 'str', 'parameter')
            
            # Call to str(...): (line 74)
            # Processing the call arguments (line 74)
            # Getting the type of 'i' (line 74)
            i_14464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 38), 'i', False)
            # Processing the call keyword arguments (line 74)
            kwargs_14465 = {}
            # Getting the type of 'str' (line 74)
            str_14463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 34), 'str', False)
            # Calling str(args, kwargs) (line 74)
            str_call_result_14466 = invoke(stypy.reporting.localization.Localization(__file__, 74, 34), str_14463, *[i_14464], **kwargs_14465)
            
            # Applying the binary operator '+' (line 74)
            result_add_14467 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 20), '+', str_14462, str_call_result_14466)
            
            str_14468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 43), 'str', ', ')
            # Applying the binary operator '+' (line 74)
            result_add_14469 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 41), '+', result_add_14467, str_14468)
            
            # Applying the binary operator '+=' (line 74)
            result_iadd_14470 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 12), '+=', str__14461, result_add_14469)
            # Assigning a type to the variable 'str_' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'str_', result_iadd_14470)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 'self' (line 76)
        self_14471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'self')
        # Obtaining the member 'call_arity' of a type (line 76)
        call_arity_14472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 11), self_14471, 'call_arity')
        int_14473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 29), 'int')
        # Applying the binary operator '>' (line 76)
        result_gt_14474 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 11), '>', call_arity_14472, int_14473)
        
        # Testing if the type of an if condition is none (line 76)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 8), result_gt_14474):
            pass
        else:
            
            # Testing the type of an if condition (line 76)
            if_condition_14475 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 8), result_gt_14474)
            # Assigning a type to the variable 'if_condition_14475' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'if_condition_14475', if_condition_14475)
            # SSA begins for if statement (line 76)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 77):
            
            # Assigning a Subscript to a Name (line 77):
            
            # Obtaining the type of the subscript
            int_14476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 25), 'int')
            slice_14477 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 77, 19), None, int_14476, None)
            # Getting the type of 'str_' (line 77)
            str__14478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'str_')
            # Obtaining the member '__getitem__' of a type (line 77)
            getitem___14479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 19), str__14478, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 77)
            subscript_call_result_14480 = invoke(stypy.reporting.localization.Localization(__file__, 77, 19), getitem___14479, slice_14477)
            
            # Assigning a type to the variable 'str_' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'str_', subscript_call_result_14480)
            # SSA join for if statement (line 76)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'str_' (line 79)
        str__14481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'str_')
        str_14482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 22), 'str', ')')
        # Applying the binary operator '+' (line 79)
        result_add_14483 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 15), '+', str__14481, str_14482)
        
        # Assigning a type to the variable 'stypy_return_type' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'stypy_return_type', result_add_14483)
        
        # ################# End of 'format_arity(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'format_arity' in the type store
        # Getting the type of 'stypy_return_type' (line 71)
        stypy_return_type_14484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14484)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'format_arity'
        return stypy_return_type_14484


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
        None_14487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 51), 'None', False)
        # Getting the type of 'self' (line 82)
        self_14488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 57), 'self', False)
        # Obtaining the member 'member' of a type (line 82)
        member_14489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 57), self_14488, 'member')
        # Processing the call keyword arguments (line 82)
        kwargs_14490 = {}
        # Getting the type of 'type_' (line 82)
        type__14485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'type_', False)
        # Obtaining the member 'get_type_of_member' of a type (line 82)
        get_type_of_member_14486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 26), type__14485, 'get_type_of_member')
        # Calling get_type_of_member(args, kwargs) (line 82)
        get_type_of_member_call_result_14491 = invoke(stypy.reporting.localization.Localization(__file__, 82, 26), get_type_of_member_14486, *[None_14487, member_14489], **kwargs_14490)
        
        # Getting the type of 'self' (line 82)
        self_14492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'self')
        # Setting the type of the member 'member_obj' of a type (line 82)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), self_14492, 'member_obj', get_type_of_member_call_result_14491)
        
        # Type idiom detected: calculating its left and rigth part (line 83)
        # Getting the type of 'TypeError' (line 83)
        TypeError_14493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 39), 'TypeError')
        # Getting the type of 'self' (line 83)
        self_14494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 22), 'self')
        # Obtaining the member 'member_obj' of a type (line 83)
        member_obj_14495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 22), self_14494, 'member_obj')
        
        (may_be_14496, more_types_in_union_14497) = may_be_subtype(TypeError_14493, member_obj_14495)

        if may_be_14496:

            if more_types_in_union_14497:
                # Runtime conditional SSA (line 83)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 83)
            self_14498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'self')
            # Obtaining the member 'member_obj' of a type (line 83)
            member_obj_14499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), self_14498, 'member_obj')
            # Setting the type of the member 'member_obj' of a type (line 83)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), self_14498, 'member_obj', remove_not_subtype_from_union(member_obj_14495, TypeError))
            
            # Getting the type of 'self' (line 84)
            self_14500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 19), 'self')
            # Obtaining the member 'report_errors' of a type (line 84)
            report_errors_14501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 19), self_14500, 'report_errors')
            # Applying the 'not' unary operator (line 84)
            result_not__14502 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 15), 'not', report_errors_14501)
            
            # Testing if the type of an if condition is none (line 84)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 84, 12), result_not__14502):
                pass
            else:
                
                # Testing the type of an if condition (line 84)
                if_condition_14503 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 12), result_not__14502)
                # Assigning a type to the variable 'if_condition_14503' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'if_condition_14503', if_condition_14503)
                # SSA begins for if statement (line 84)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to remove_error_msg(...): (line 85)
                # Processing the call arguments (line 85)
                # Getting the type of 'self' (line 85)
                self_14506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 43), 'self', False)
                # Obtaining the member 'member_obj' of a type (line 85)
                member_obj_14507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 43), self_14506, 'member_obj')
                # Processing the call keyword arguments (line 85)
                kwargs_14508 = {}
                # Getting the type of 'TypeError' (line 85)
                TypeError_14504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'TypeError', False)
                # Obtaining the member 'remove_error_msg' of a type (line 85)
                remove_error_msg_14505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 16), TypeError_14504, 'remove_error_msg')
                # Calling remove_error_msg(args, kwargs) (line 85)
                remove_error_msg_call_result_14509 = invoke(stypy.reporting.localization.Localization(__file__, 85, 16), remove_error_msg_14505, *[member_obj_14507], **kwargs_14508)
                
                # SSA join for if statement (line 84)
                module_type_store = module_type_store.join_ssa_context()
                

            # Getting the type of 'False' (line 86)
            False_14510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'stypy_return_type', False_14510)

            if more_types_in_union_14497:
                # SSA join for if statement (line 83)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'True' (line 88)
        True_14511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'stypy_return_type', True_14511)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_14512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14512)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_14512


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
        kwargs_14517 = {}
        # Getting the type of 'self' (line 91)
        self_14514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'self', False)
        # Obtaining the member 'member_obj' of a type (line 91)
        member_obj_14515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 20), self_14514, 'member_obj')
        # Obtaining the member 'get_python_type' of a type (line 91)
        get_python_type_14516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 20), member_obj_14515, 'get_python_type')
        # Calling get_python_type(args, kwargs) (line 91)
        get_python_type_call_result_14518 = invoke(stypy.reporting.localization.Localization(__file__, 91, 20), get_python_type_14516, *[], **kwargs_14517)
        
        # Processing the call keyword arguments (line 91)
        kwargs_14519 = {}
        # Getting the type of 'callable' (line 91)
        callable_14513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 11), 'callable', False)
        # Calling callable(args, kwargs) (line 91)
        callable_call_result_14520 = invoke(stypy.reporting.localization.Localization(__file__, 91, 11), callable_14513, *[get_python_type_call_result_14518], **kwargs_14519)
        
        # Testing if the type of an if condition is none (line 91)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 91, 8), callable_call_result_14520):
            pass
        else:
            
            # Testing the type of an if condition (line 91)
            if_condition_14521 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 8), callable_call_result_14520)
            # Assigning a type to the variable 'if_condition_14521' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'if_condition_14521', if_condition_14521)
            # SSA begins for if statement (line 91)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 93):
            
            # Assigning a Call to a Name (line 93):
            
            # Call to invoke(...): (line 93)
            # Processing the call arguments (line 93)
            # Getting the type of 'localization' (line 93)
            localization_14525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 53), 'localization', False)
            # Getting the type of 'call_args' (line 93)
            call_args_14526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 68), 'call_args', False)
            # Processing the call keyword arguments (line 93)
            # Getting the type of 'call_kwargs' (line 93)
            call_kwargs_14527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 81), 'call_kwargs', False)
            kwargs_14528 = {'call_kwargs_14527': call_kwargs_14527}
            # Getting the type of 'self' (line 93)
            self_14522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 30), 'self', False)
            # Obtaining the member 'member_obj' of a type (line 93)
            member_obj_14523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 30), self_14522, 'member_obj')
            # Obtaining the member 'invoke' of a type (line 93)
            invoke_14524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 30), member_obj_14523, 'invoke')
            # Calling invoke(args, kwargs) (line 93)
            invoke_call_result_14529 = invoke(stypy.reporting.localization.Localization(__file__, 93, 30), invoke_14524, *[localization_14525, call_args_14526], **kwargs_14528)
            
            # Assigning a type to the variable 'equivalent_type' (line 93)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'equivalent_type', invoke_call_result_14529)
            
            # Type idiom detected: calculating its left and rigth part (line 96)
            # Getting the type of 'TypeError' (line 96)
            TypeError_14530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 43), 'TypeError')
            # Getting the type of 'equivalent_type' (line 96)
            equivalent_type_14531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'equivalent_type')
            
            (may_be_14532, more_types_in_union_14533) = may_be_subtype(TypeError_14530, equivalent_type_14531)

            if may_be_14532:

                if more_types_in_union_14533:
                    # Runtime conditional SSA (line 96)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'equivalent_type' (line 96)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'equivalent_type', remove_not_subtype_from_union(equivalent_type_14531, TypeError))
                
                # Getting the type of 'self' (line 97)
                self_14534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 23), 'self')
                # Obtaining the member 'report_errors' of a type (line 97)
                report_errors_14535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 23), self_14534, 'report_errors')
                # Applying the 'not' unary operator (line 97)
                result_not__14536 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 19), 'not', report_errors_14535)
                
                # Testing if the type of an if condition is none (line 97)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 97, 16), result_not__14536):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 97)
                    if_condition_14537 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 16), result_not__14536)
                    # Assigning a type to the variable 'if_condition_14537' (line 97)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'if_condition_14537', if_condition_14537)
                    # SSA begins for if statement (line 97)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to remove_error_msg(...): (line 98)
                    # Processing the call arguments (line 98)
                    # Getting the type of 'equivalent_type' (line 98)
                    equivalent_type_14540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 47), 'equivalent_type', False)
                    # Processing the call keyword arguments (line 98)
                    kwargs_14541 = {}
                    # Getting the type of 'TypeError' (line 98)
                    TypeError_14538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'TypeError', False)
                    # Obtaining the member 'remove_error_msg' of a type (line 98)
                    remove_error_msg_14539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 20), TypeError_14538, 'remove_error_msg')
                    # Calling remove_error_msg(args, kwargs) (line 98)
                    remove_error_msg_call_result_14542 = invoke(stypy.reporting.localization.Localization(__file__, 98, 20), remove_error_msg_14539, *[equivalent_type_14540], **kwargs_14541)
                    
                    # SSA join for if statement (line 97)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Name to a Attribute (line 99):
                
                # Assigning a Name to a Attribute (line 99):
                # Getting the type of 'None' (line 99)
                None_14543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 34), 'None')
                # Getting the type of 'self' (line 99)
                self_14544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'self')
                # Setting the type of the member 'member_obj' of a type (line 99)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), self_14544, 'member_obj', None_14543)
                
                # Obtaining an instance of the builtin type 'tuple' (line 100)
                tuple_14545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 100)
                # Adding element type (line 100)
                # Getting the type of 'False' (line 100)
                False_14546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 23), 'False')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 23), tuple_14545, False_14546)
                # Adding element type (line 100)
                # Getting the type of 'equivalent_type' (line 100)
                equivalent_type_14547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 30), 'equivalent_type')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 23), tuple_14545, equivalent_type_14547)
                
                # Assigning a type to the variable 'stypy_return_type' (line 100)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'stypy_return_type', tuple_14545)

                if more_types_in_union_14533:
                    # SSA join for if statement (line 96)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Call to isinstance(...): (line 103)
            # Processing the call arguments (line 103)
            # Getting the type of 'self' (line 103)
            self_14549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'self', False)
            # Obtaining the member 'expected_return_type' of a type (line 103)
            expected_return_type_14550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 26), self_14549, 'expected_return_type')
            # Getting the type of 'UndefinedType' (line 103)
            UndefinedType_14551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 53), 'UndefinedType', False)
            # Processing the call keyword arguments (line 103)
            kwargs_14552 = {}
            # Getting the type of 'isinstance' (line 103)
            isinstance_14548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 103)
            isinstance_call_result_14553 = invoke(stypy.reporting.localization.Localization(__file__, 103, 15), isinstance_14548, *[expected_return_type_14550, UndefinedType_14551], **kwargs_14552)
            
            # Testing if the type of an if condition is none (line 103)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 103, 12), isinstance_call_result_14553):
                pass
            else:
                
                # Testing the type of an if condition (line 103)
                if_condition_14554 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 12), isinstance_call_result_14553)
                # Assigning a type to the variable 'if_condition_14554' (line 103)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'if_condition_14554', if_condition_14554)
                # SSA begins for if statement (line 103)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Attribute (line 104):
                
                # Assigning a Name to a Attribute (line 104):
                # Getting the type of 'None' (line 104)
                None_14555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 34), 'None')
                # Getting the type of 'self' (line 104)
                self_14556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'self')
                # Setting the type of the member 'member_obj' of a type (line 104)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 16), self_14556, 'member_obj', None_14555)
                
                # Obtaining an instance of the builtin type 'tuple' (line 105)
                tuple_14557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 105)
                # Adding element type (line 105)
                # Getting the type of 'True' (line 105)
                True_14558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 23), 'True')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 23), tuple_14557, True_14558)
                # Adding element type (line 105)
                # Getting the type of 'equivalent_type' (line 105)
                equivalent_type_14559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 29), 'equivalent_type')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 23), tuple_14557, equivalent_type_14559)
                
                # Assigning a type to the variable 'stypy_return_type' (line 105)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'stypy_return_type', tuple_14557)
                # SSA join for if statement (line 103)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'self' (line 108)
            self_14560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'self')
            # Obtaining the member 'expected_return_type' of a type (line 108)
            expected_return_type_14561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 15), self_14560, 'expected_return_type')
            # Getting the type of 'DynamicType' (line 108)
            DynamicType_14562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 44), 'DynamicType')
            # Applying the binary operator 'is' (line 108)
            result_is__14563 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 15), 'is', expected_return_type_14561, DynamicType_14562)
            
            # Testing if the type of an if condition is none (line 108)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 12), result_is__14563):
                pass
            else:
                
                # Testing the type of an if condition (line 108)
                if_condition_14564 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 12), result_is__14563)
                # Assigning a type to the variable 'if_condition_14564' (line 108)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'if_condition_14564', if_condition_14564)
                # SSA begins for if statement (line 108)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Attribute (line 109):
                
                # Assigning a Name to a Attribute (line 109):
                # Getting the type of 'None' (line 109)
                None_14565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 34), 'None')
                # Getting the type of 'self' (line 109)
                self_14566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'self')
                # Setting the type of the member 'member_obj' of a type (line 109)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 16), self_14566, 'member_obj', None_14565)
                
                # Obtaining an instance of the builtin type 'tuple' (line 110)
                tuple_14567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 110)
                # Adding element type (line 110)
                # Getting the type of 'True' (line 110)
                True_14568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 23), 'True')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 23), tuple_14567, True_14568)
                # Adding element type (line 110)
                # Getting the type of 'equivalent_type' (line 110)
                equivalent_type_14569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 29), 'equivalent_type')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 23), tuple_14567, equivalent_type_14569)
                
                # Assigning a type to the variable 'stypy_return_type' (line 110)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'stypy_return_type', tuple_14567)
                # SSA join for if statement (line 108)
                module_type_store = module_type_store.join_ssa_context()
                

            
            
            # Call to issubclass(...): (line 113)
            # Processing the call arguments (line 113)
            
            # Call to get_python_type(...): (line 113)
            # Processing the call keyword arguments (line 113)
            kwargs_14573 = {}
            # Getting the type of 'equivalent_type' (line 113)
            equivalent_type_14571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 30), 'equivalent_type', False)
            # Obtaining the member 'get_python_type' of a type (line 113)
            get_python_type_14572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 30), equivalent_type_14571, 'get_python_type')
            # Calling get_python_type(args, kwargs) (line 113)
            get_python_type_call_result_14574 = invoke(stypy.reporting.localization.Localization(__file__, 113, 30), get_python_type_14572, *[], **kwargs_14573)
            
            # Getting the type of 'self' (line 113)
            self_14575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 65), 'self', False)
            # Obtaining the member 'expected_return_type' of a type (line 113)
            expected_return_type_14576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 65), self_14575, 'expected_return_type')
            # Processing the call keyword arguments (line 113)
            kwargs_14577 = {}
            # Getting the type of 'issubclass' (line 113)
            issubclass_14570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'issubclass', False)
            # Calling issubclass(args, kwargs) (line 113)
            issubclass_call_result_14578 = invoke(stypy.reporting.localization.Localization(__file__, 113, 19), issubclass_14570, *[get_python_type_call_result_14574, expected_return_type_14576], **kwargs_14577)
            
            # Applying the 'not' unary operator (line 113)
            result_not__14579 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 15), 'not', issubclass_call_result_14578)
            
            # Testing if the type of an if condition is none (line 113)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 113, 12), result_not__14579):
                
                # Obtaining an instance of the builtin type 'tuple' (line 117)
                tuple_14586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 117)
                # Adding element type (line 117)
                # Getting the type of 'True' (line 117)
                True_14587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 23), 'True')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 23), tuple_14586, True_14587)
                # Adding element type (line 117)
                # Getting the type of 'equivalent_type' (line 117)
                equivalent_type_14588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 29), 'equivalent_type')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 23), tuple_14586, equivalent_type_14588)
                
                # Assigning a type to the variable 'stypy_return_type' (line 117)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'stypy_return_type', tuple_14586)
            else:
                
                # Testing the type of an if condition (line 113)
                if_condition_14580 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 12), result_not__14579)
                # Assigning a type to the variable 'if_condition_14580' (line 113)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'if_condition_14580', if_condition_14580)
                # SSA begins for if statement (line 113)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Attribute (line 114):
                
                # Assigning a Name to a Attribute (line 114):
                # Getting the type of 'None' (line 114)
                None_14581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 34), 'None')
                # Getting the type of 'self' (line 114)
                self_14582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'self')
                # Setting the type of the member 'member_obj' of a type (line 114)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 16), self_14582, 'member_obj', None_14581)
                
                # Obtaining an instance of the builtin type 'tuple' (line 115)
                tuple_14583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 115)
                # Adding element type (line 115)
                # Getting the type of 'False' (line 115)
                False_14584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 23), 'False')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 23), tuple_14583, False_14584)
                # Adding element type (line 115)
                # Getting the type of 'equivalent_type' (line 115)
                equivalent_type_14585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 30), 'equivalent_type')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 23), tuple_14583, equivalent_type_14585)
                
                # Assigning a type to the variable 'stypy_return_type' (line 115)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'stypy_return_type', tuple_14583)
                # SSA branch for the else part of an if statement (line 113)
                module_type_store.open_ssa_branch('else')
                
                # Obtaining an instance of the builtin type 'tuple' (line 117)
                tuple_14586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 117)
                # Adding element type (line 117)
                # Getting the type of 'True' (line 117)
                True_14587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 23), 'True')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 23), tuple_14586, True_14587)
                # Adding element type (line 117)
                # Getting the type of 'equivalent_type' (line 117)
                equivalent_type_14588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 29), 'equivalent_type')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 23), tuple_14586, equivalent_type_14588)
                
                # Assigning a type to the variable 'stypy_return_type' (line 117)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'stypy_return_type', tuple_14586)
                # SSA join for if statement (line 113)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 91)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Name to a Attribute (line 119):
        
        # Assigning a Name to a Attribute (line 119):
        # Getting the type of 'None' (line 119)
        None_14589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 26), 'None')
        # Getting the type of 'self' (line 119)
        self_14590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'self')
        # Setting the type of the member 'member_obj' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), self_14590, 'member_obj', None_14589)
        
        # Obtaining an instance of the builtin type 'tuple' (line 120)
        tuple_14591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 120)
        # Adding element type (line 120)
        # Getting the type of 'True' (line 120)
        True_14592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 15), tuple_14591, True_14592)
        # Adding element type (line 120)
        # Getting the type of 'None' (line 120)
        None_14593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 21), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 15), tuple_14591, None_14593)
        
        # Assigning a type to the variable 'stypy_return_type' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'stypy_return_type', tuple_14591)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 90)
        stypy_return_type_14594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14594)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_14594


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
        str_14595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 18), 'str', 'Instance defining ')
        # Assigning a type to the variable 'ret_str' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'ret_str', str_14595)
        
        # Getting the type of 'ret_str' (line 124)
        ret_str_14596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'ret_str')
        
        # Call to str(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'self' (line 124)
        self_14598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 23), 'self', False)
        # Obtaining the member 'member' of a type (line 124)
        member_14599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 23), self_14598, 'member')
        # Processing the call keyword arguments (line 124)
        kwargs_14600 = {}
        # Getting the type of 'str' (line 124)
        str_14597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'str', False)
        # Calling str(args, kwargs) (line 124)
        str_call_result_14601 = invoke(stypy.reporting.localization.Localization(__file__, 124, 19), str_14597, *[member_14599], **kwargs_14600)
        
        # Applying the binary operator '+=' (line 124)
        result_iadd_14602 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 8), '+=', ret_str_14596, str_call_result_14601)
        # Assigning a type to the variable 'ret_str' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'ret_str', result_iadd_14602)
        
        
        # Getting the type of 'ret_str' (line 125)
        ret_str_14603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'ret_str')
        
        # Call to format_arity(...): (line 125)
        # Processing the call keyword arguments (line 125)
        kwargs_14606 = {}
        # Getting the type of 'self' (line 125)
        self_14604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 19), 'self', False)
        # Obtaining the member 'format_arity' of a type (line 125)
        format_arity_14605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 19), self_14604, 'format_arity')
        # Calling format_arity(args, kwargs) (line 125)
        format_arity_call_result_14607 = invoke(stypy.reporting.localization.Localization(__file__, 125, 19), format_arity_14605, *[], **kwargs_14606)
        
        # Applying the binary operator '+=' (line 125)
        result_iadd_14608 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 8), '+=', ret_str_14603, format_arity_call_result_14607)
        # Assigning a type to the variable 'ret_str' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'ret_str', result_iadd_14608)
        
        # Getting the type of 'ret_str' (line 126)
        ret_str_14609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 15), 'ret_str')
        # Assigning a type to the variable 'stypy_return_type' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'stypy_return_type', ret_str_14609)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 122)
        stypy_return_type_14610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14610)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_14610


# Assigning a type to the variable 'HasMember' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'HasMember', HasMember)
# Declaration of the 'IterableDataStructureWithTypedElements' class
# Getting the type of 'TypeGroup' (line 129)
TypeGroup_14611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 45), 'TypeGroup')
# Getting the type of 'DependentType' (line 129)
DependentType_14612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 56), 'DependentType')

class IterableDataStructureWithTypedElements(TypeGroup_14611, DependentType_14612, ):
    str_14613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, (-1)), 'str', '\n    Represent all iterable data structures that contain a certain type or types\n    ')

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
        self_14616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 31), 'self', False)
        # Getting the type of 'True' (line 135)
        True_14617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 37), 'True', False)
        # Processing the call keyword arguments (line 135)
        kwargs_14618 = {}
        # Getting the type of 'DependentType' (line 135)
        DependentType_14614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'DependentType', False)
        # Obtaining the member '__init__' of a type (line 135)
        init___14615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), DependentType_14614, '__init__')
        # Calling __init__(args, kwargs) (line 135)
        init___call_result_14619 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), init___14615, *[self_14616, True_14617], **kwargs_14618)
        
        
        # Call to __init__(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'self' (line 136)
        self_14622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 136)
        list_14623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 136)
        
        # Processing the call keyword arguments (line 136)
        kwargs_14624 = {}
        # Getting the type of 'TypeGroup' (line 136)
        TypeGroup_14620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 136)
        init___14621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), TypeGroup_14620, '__init__')
        # Calling __init__(args, kwargs) (line 136)
        init___call_result_14625 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), init___14621, *[self_14622, list_14623], **kwargs_14624)
        
        
        # Assigning a Name to a Attribute (line 137):
        
        # Assigning a Name to a Attribute (line 137):
        # Getting the type of 'content_types' (line 137)
        content_types_14626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 29), 'content_types')
        # Getting the type of 'self' (line 137)
        self_14627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'self')
        # Setting the type of the member 'content_types' of a type (line 137)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), self_14627, 'content_types', content_types_14626)
        
        # Assigning a Name to a Attribute (line 138):
        
        # Assigning a Name to a Attribute (line 138):
        # Getting the type of 'None' (line 138)
        None_14628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 21), 'None')
        # Getting the type of 'self' (line 138)
        self_14629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'self')
        # Setting the type of the member 'type_' of a type (line 138)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), self_14629, 'type_', None_14628)
        
        # Assigning a Num to a Attribute (line 139):
        
        # Assigning a Num to a Attribute (line 139):
        int_14630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 26), 'int')
        # Getting the type of 'self' (line 139)
        self_14631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'self')
        # Setting the type of the member 'call_arity' of a type (line 139)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), self_14631, 'call_arity', int_14630)
        
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
        type__14632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 21), 'type_')
        # Getting the type of 'self' (line 142)
        self_14633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'self')
        # Setting the type of the member 'type_' of a type (line 142)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), self_14633, 'type_', type__14632)
        
        
        # Call to get_python_type(...): (line 143)
        # Processing the call keyword arguments (line 143)
        kwargs_14636 = {}
        # Getting the type of 'type_' (line 143)
        type__14634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 15), 'type_', False)
        # Obtaining the member 'get_python_type' of a type (line 143)
        get_python_type_14635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 15), type__14634, 'get_python_type')
        # Calling get_python_type(args, kwargs) (line 143)
        get_python_type_call_result_14637 = invoke(stypy.reporting.localization.Localization(__file__, 143, 15), get_python_type_14635, *[], **kwargs_14636)
        
        # Getting the type of 'TypeGroups' (line 143)
        TypeGroups_14638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 42), 'TypeGroups')
        # Obtaining the member 'IterableDataStructure' of a type (line 143)
        IterableDataStructure_14639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 42), TypeGroups_14638, 'IterableDataStructure')
        # Applying the binary operator 'in' (line 143)
        result_contains_14640 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 15), 'in', get_python_type_call_result_14637, IterableDataStructure_14639)
        
        # Assigning a type to the variable 'stypy_return_type' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'stypy_return_type', result_contains_14640)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 141)
        stypy_return_type_14641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14641)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_14641


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
        kwargs_14645 = {}
        # Getting the type of 'self' (line 146)
        self_14642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 29), 'self', False)
        # Obtaining the member 'type_' of a type (line 146)
        type__14643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 29), self_14642, 'type_')
        # Obtaining the member 'get_elements_type' of a type (line 146)
        get_elements_type_14644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 29), type__14643, 'get_elements_type')
        # Calling get_elements_type(args, kwargs) (line 146)
        get_elements_type_call_result_14646 = invoke(stypy.reporting.localization.Localization(__file__, 146, 29), get_elements_type_14644, *[], **kwargs_14645)
        
        # Assigning a type to the variable 'contained_elements' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'contained_elements', get_elements_type_call_result_14646)
        
        # Call to isinstance(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'contained_elements' (line 147)
        contained_elements_14648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 22), 'contained_elements', False)
        # Getting the type of 'union_type_copy' (line 147)
        union_type_copy_14649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 42), 'union_type_copy', False)
        # Obtaining the member 'UnionType' of a type (line 147)
        UnionType_14650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 42), union_type_copy_14649, 'UnionType')
        # Processing the call keyword arguments (line 147)
        kwargs_14651 = {}
        # Getting the type of 'isinstance' (line 147)
        isinstance_14647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 147)
        isinstance_call_result_14652 = invoke(stypy.reporting.localization.Localization(__file__, 147, 11), isinstance_14647, *[contained_elements_14648, UnionType_14650], **kwargs_14651)
        
        # Testing if the type of an if condition is none (line 147)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 147, 8), isinstance_call_result_14652):
            
            # Assigning a List to a Name (line 150):
            
            # Assigning a List to a Name (line 150):
            
            # Obtaining an instance of the builtin type 'list' (line 150)
            list_14656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 31), 'list')
            # Adding type elements to the builtin type 'list' instance (line 150)
            # Adding element type (line 150)
            # Getting the type of 'contained_elements' (line 150)
            contained_elements_14657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 32), 'contained_elements')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 31), list_14656, contained_elements_14657)
            
            # Assigning a type to the variable 'types_to_examine' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'types_to_examine', list_14656)
        else:
            
            # Testing the type of an if condition (line 147)
            if_condition_14653 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 8), isinstance_call_result_14652)
            # Assigning a type to the variable 'if_condition_14653' (line 147)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'if_condition_14653', if_condition_14653)
            # SSA begins for if statement (line 147)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 148):
            
            # Assigning a Attribute to a Name (line 148):
            # Getting the type of 'contained_elements' (line 148)
            contained_elements_14654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 31), 'contained_elements')
            # Obtaining the member 'types' of a type (line 148)
            types_14655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 31), contained_elements_14654, 'types')
            # Assigning a type to the variable 'types_to_examine' (line 148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'types_to_examine', types_14655)
            # SSA branch for the else part of an if statement (line 147)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a List to a Name (line 150):
            
            # Assigning a List to a Name (line 150):
            
            # Obtaining an instance of the builtin type 'list' (line 150)
            list_14656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 31), 'list')
            # Adding type elements to the builtin type 'list' instance (line 150)
            # Adding element type (line 150)
            # Getting the type of 'contained_elements' (line 150)
            contained_elements_14657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 32), 'contained_elements')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 31), list_14656, contained_elements_14657)
            
            # Assigning a type to the variable 'types_to_examine' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'types_to_examine', list_14656)
            # SSA join for if statement (line 147)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a List to a Name (line 152):
        
        # Assigning a List to a Name (line 152):
        
        # Obtaining an instance of the builtin type 'list' (line 152)
        list_14658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 152)
        
        # Assigning a type to the variable 'right_types' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'right_types', list_14658)
        
        # Assigning a List to a Name (line 153):
        
        # Assigning a List to a Name (line 153):
        
        # Obtaining an instance of the builtin type 'list' (line 153)
        list_14659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 153)
        
        # Assigning a type to the variable 'wrong_types' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'wrong_types', list_14659)
        
        # Getting the type of 'types_to_examine' (line 155)
        types_to_examine_14660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 21), 'types_to_examine')
        # Assigning a type to the variable 'types_to_examine_14660' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'types_to_examine_14660', types_to_examine_14660)
        # Testing if the for loop is going to be iterated (line 155)
        # Testing the type of a for loop iterable (line 155)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 155, 8), types_to_examine_14660)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 155, 8), types_to_examine_14660):
            # Getting the type of the for loop variable (line 155)
            for_loop_var_14661 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 155, 8), types_to_examine_14660)
            # Assigning a type to the variable 'type_' (line 155)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'type_', for_loop_var_14661)
            # SSA begins for a for statement (line 155)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Name to a Name (line 156):
            
            # Assigning a Name to a Name (line 156):
            # Getting the type of 'False' (line 156)
            False_14662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 26), 'False')
            # Assigning a type to the variable 'match_found' (line 156)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'match_found', False_14662)
            
            # Getting the type of 'self' (line 157)
            self_14663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 43), 'self')
            # Obtaining the member 'content_types' of a type (line 157)
            content_types_14664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 43), self_14663, 'content_types')
            # Assigning a type to the variable 'content_types_14664' (line 157)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'content_types_14664', content_types_14664)
            # Testing if the for loop is going to be iterated (line 157)
            # Testing the type of a for loop iterable (line 157)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 157, 12), content_types_14664)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 157, 12), content_types_14664):
                # Getting the type of the for loop variable (line 157)
                for_loop_var_14665 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 157, 12), content_types_14664)
                # Assigning a type to the variable 'declared_contained_type' (line 157)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'declared_contained_type', for_loop_var_14665)
                # SSA begins for a for statement (line 157)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'declared_contained_type' (line 158)
                declared_contained_type_14666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 19), 'declared_contained_type')
                # Getting the type of 'type_' (line 158)
                type__14667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 46), 'type_')
                # Applying the binary operator '==' (line 158)
                result_eq_14668 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 19), '==', declared_contained_type_14666, type__14667)
                
                # Testing if the type of an if condition is none (line 158)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 158, 16), result_eq_14668):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 158)
                    if_condition_14669 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 16), result_eq_14668)
                    # Assigning a type to the variable 'if_condition_14669' (line 158)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'if_condition_14669', if_condition_14669)
                    # SSA begins for if statement (line 158)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to isinstance(...): (line 159)
                    # Processing the call arguments (line 159)
                    # Getting the type of 'declared_contained_type' (line 159)
                    declared_contained_type_14671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 34), 'declared_contained_type', False)
                    # Getting the type of 'DependentType' (line 159)
                    DependentType_14672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 59), 'DependentType', False)
                    # Processing the call keyword arguments (line 159)
                    kwargs_14673 = {}
                    # Getting the type of 'isinstance' (line 159)
                    isinstance_14670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 'isinstance', False)
                    # Calling isinstance(args, kwargs) (line 159)
                    isinstance_call_result_14674 = invoke(stypy.reporting.localization.Localization(__file__, 159, 23), isinstance_14670, *[declared_contained_type_14671, DependentType_14672], **kwargs_14673)
                    
                    # Testing if the type of an if condition is none (line 159)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 159, 20), isinstance_call_result_14674):
                        
                        # Assigning a Name to a Name (line 174):
                        
                        # Assigning a Name to a Name (line 174):
                        # Getting the type of 'True' (line 174)
                        True_14736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 38), 'True')
                        # Assigning a type to the variable 'match_found' (line 174)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 24), 'match_found', True_14736)
                        
                        # Call to append(...): (line 175)
                        # Processing the call arguments (line 175)
                        # Getting the type of 'type_' (line 175)
                        type__14739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 43), 'type_', False)
                        # Processing the call keyword arguments (line 175)
                        kwargs_14740 = {}
                        # Getting the type of 'right_types' (line 175)
                        right_types_14737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 24), 'right_types', False)
                        # Obtaining the member 'append' of a type (line 175)
                        append_14738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 24), right_types_14737, 'append')
                        # Calling append(args, kwargs) (line 175)
                        append_call_result_14741 = invoke(stypy.reporting.localization.Localization(__file__, 175, 24), append_14738, *[type__14739], **kwargs_14740)
                        
                    else:
                        
                        # Testing the type of an if condition (line 159)
                        if_condition_14675 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 20), isinstance_call_result_14674)
                        # Assigning a type to the variable 'if_condition_14675' (line 159)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 20), 'if_condition_14675', if_condition_14675)
                        # SSA begins for if statement (line 159)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Getting the type of 'declared_contained_type' (line 160)
                        declared_contained_type_14676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 27), 'declared_contained_type')
                        # Obtaining the member 'call_arity' of a type (line 160)
                        call_arity_14677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 27), declared_contained_type_14676, 'call_arity')
                        int_14678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 65), 'int')
                        # Applying the binary operator '==' (line 160)
                        result_eq_14679 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 27), '==', call_arity_14677, int_14678)
                        
                        # Testing if the type of an if condition is none (line 160)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 160, 24), result_eq_14679):
                            
                            # Assigning a Call to a Tuple (line 163):
                            
                            # Assigning a Call to a Name:
                            
                            # Call to declared_contained_type(...): (line 163)
                            # Processing the call arguments (line 163)
                            # Getting the type of 'localization' (line 163)
                            localization_14692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 75), 'localization', False)
                            # Getting the type of 'type_' (line 163)
                            type__14693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 89), 'type_', False)
                            # Processing the call keyword arguments (line 163)
                            kwargs_14694 = {}
                            # Getting the type of 'declared_contained_type' (line 163)
                            declared_contained_type_14691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 51), 'declared_contained_type', False)
                            # Calling declared_contained_type(args, kwargs) (line 163)
                            declared_contained_type_call_result_14695 = invoke(stypy.reporting.localization.Localization(__file__, 163, 51), declared_contained_type_14691, *[localization_14692, type__14693], **kwargs_14694)
                            
                            # Assigning a type to the variable 'call_assignment_14401' (line 163)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_14401', declared_contained_type_call_result_14695)
                            
                            # Assigning a Call to a Name (line 163):
                            
                            # Call to stypy_get_value_from_tuple(...):
                            # Processing the call arguments
                            # Getting the type of 'call_assignment_14401' (line 163)
                            call_assignment_14401_14696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_14401', False)
                            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                            stypy_get_value_from_tuple_call_result_14697 = stypy_get_value_from_tuple(call_assignment_14401_14696, 2, 0)
                            
                            # Assigning a type to the variable 'call_assignment_14402' (line 163)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_14402', stypy_get_value_from_tuple_call_result_14697)
                            
                            # Assigning a Name to a Name (line 163):
                            # Getting the type of 'call_assignment_14402' (line 163)
                            call_assignment_14402_14698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_14402')
                            # Assigning a type to the variable 'correct' (line 163)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'correct', call_assignment_14402_14698)
                            
                            # Assigning a Call to a Name (line 163):
                            
                            # Call to stypy_get_value_from_tuple(...):
                            # Processing the call arguments
                            # Getting the type of 'call_assignment_14401' (line 163)
                            call_assignment_14401_14699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_14401', False)
                            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                            stypy_get_value_from_tuple_call_result_14700 = stypy_get_value_from_tuple(call_assignment_14401_14699, 2, 1)
                            
                            # Assigning a type to the variable 'call_assignment_14403' (line 163)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_14403', stypy_get_value_from_tuple_call_result_14700)
                            
                            # Assigning a Name to a Name (line 163):
                            # Getting the type of 'call_assignment_14403' (line 163)
                            call_assignment_14403_14701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_14403')
                            # Assigning a type to the variable 'return_type' (line 163)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 37), 'return_type', call_assignment_14403_14701)
                        else:
                            
                            # Testing the type of an if condition (line 160)
                            if_condition_14680 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 24), result_eq_14679)
                            # Assigning a type to the variable 'if_condition_14680' (line 160)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'if_condition_14680', if_condition_14680)
                            # SSA begins for if statement (line 160)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Call to a Tuple (line 161):
                            
                            # Assigning a Call to a Name:
                            
                            # Call to declared_contained_type(...): (line 161)
                            # Processing the call arguments (line 161)
                            # Getting the type of 'localization' (line 161)
                            localization_14682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 75), 'localization', False)
                            # Processing the call keyword arguments (line 161)
                            kwargs_14683 = {}
                            # Getting the type of 'declared_contained_type' (line 161)
                            declared_contained_type_14681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 51), 'declared_contained_type', False)
                            # Calling declared_contained_type(args, kwargs) (line 161)
                            declared_contained_type_call_result_14684 = invoke(stypy.reporting.localization.Localization(__file__, 161, 51), declared_contained_type_14681, *[localization_14682], **kwargs_14683)
                            
                            # Assigning a type to the variable 'call_assignment_14398' (line 161)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'call_assignment_14398', declared_contained_type_call_result_14684)
                            
                            # Assigning a Call to a Name (line 161):
                            
                            # Call to stypy_get_value_from_tuple(...):
                            # Processing the call arguments
                            # Getting the type of 'call_assignment_14398' (line 161)
                            call_assignment_14398_14685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'call_assignment_14398', False)
                            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                            stypy_get_value_from_tuple_call_result_14686 = stypy_get_value_from_tuple(call_assignment_14398_14685, 2, 0)
                            
                            # Assigning a type to the variable 'call_assignment_14399' (line 161)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'call_assignment_14399', stypy_get_value_from_tuple_call_result_14686)
                            
                            # Assigning a Name to a Name (line 161):
                            # Getting the type of 'call_assignment_14399' (line 161)
                            call_assignment_14399_14687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'call_assignment_14399')
                            # Assigning a type to the variable 'correct' (line 161)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'correct', call_assignment_14399_14687)
                            
                            # Assigning a Call to a Name (line 161):
                            
                            # Call to stypy_get_value_from_tuple(...):
                            # Processing the call arguments
                            # Getting the type of 'call_assignment_14398' (line 161)
                            call_assignment_14398_14688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'call_assignment_14398', False)
                            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                            stypy_get_value_from_tuple_call_result_14689 = stypy_get_value_from_tuple(call_assignment_14398_14688, 2, 1)
                            
                            # Assigning a type to the variable 'call_assignment_14400' (line 161)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'call_assignment_14400', stypy_get_value_from_tuple_call_result_14689)
                            
                            # Assigning a Name to a Name (line 161):
                            # Getting the type of 'call_assignment_14400' (line 161)
                            call_assignment_14400_14690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'call_assignment_14400')
                            # Assigning a type to the variable 'return_type' (line 161)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 37), 'return_type', call_assignment_14400_14690)
                            # SSA branch for the else part of an if statement (line 160)
                            module_type_store.open_ssa_branch('else')
                            
                            # Assigning a Call to a Tuple (line 163):
                            
                            # Assigning a Call to a Name:
                            
                            # Call to declared_contained_type(...): (line 163)
                            # Processing the call arguments (line 163)
                            # Getting the type of 'localization' (line 163)
                            localization_14692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 75), 'localization', False)
                            # Getting the type of 'type_' (line 163)
                            type__14693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 89), 'type_', False)
                            # Processing the call keyword arguments (line 163)
                            kwargs_14694 = {}
                            # Getting the type of 'declared_contained_type' (line 163)
                            declared_contained_type_14691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 51), 'declared_contained_type', False)
                            # Calling declared_contained_type(args, kwargs) (line 163)
                            declared_contained_type_call_result_14695 = invoke(stypy.reporting.localization.Localization(__file__, 163, 51), declared_contained_type_14691, *[localization_14692, type__14693], **kwargs_14694)
                            
                            # Assigning a type to the variable 'call_assignment_14401' (line 163)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_14401', declared_contained_type_call_result_14695)
                            
                            # Assigning a Call to a Name (line 163):
                            
                            # Call to stypy_get_value_from_tuple(...):
                            # Processing the call arguments
                            # Getting the type of 'call_assignment_14401' (line 163)
                            call_assignment_14401_14696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_14401', False)
                            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                            stypy_get_value_from_tuple_call_result_14697 = stypy_get_value_from_tuple(call_assignment_14401_14696, 2, 0)
                            
                            # Assigning a type to the variable 'call_assignment_14402' (line 163)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_14402', stypy_get_value_from_tuple_call_result_14697)
                            
                            # Assigning a Name to a Name (line 163):
                            # Getting the type of 'call_assignment_14402' (line 163)
                            call_assignment_14402_14698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_14402')
                            # Assigning a type to the variable 'correct' (line 163)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'correct', call_assignment_14402_14698)
                            
                            # Assigning a Call to a Name (line 163):
                            
                            # Call to stypy_get_value_from_tuple(...):
                            # Processing the call arguments
                            # Getting the type of 'call_assignment_14401' (line 163)
                            call_assignment_14401_14699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_14401', False)
                            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                            stypy_get_value_from_tuple_call_result_14700 = stypy_get_value_from_tuple(call_assignment_14401_14699, 2, 1)
                            
                            # Assigning a type to the variable 'call_assignment_14403' (line 163)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_14403', stypy_get_value_from_tuple_call_result_14700)
                            
                            # Assigning a Name to a Name (line 163):
                            # Getting the type of 'call_assignment_14403' (line 163)
                            call_assignment_14403_14701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'call_assignment_14403')
                            # Assigning a type to the variable 'return_type' (line 163)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 37), 'return_type', call_assignment_14403_14701)
                            # SSA join for if statement (line 160)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # Getting the type of 'correct' (line 164)
                        correct_14702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 27), 'correct')
                        # Testing if the type of an if condition is none (line 164)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 164, 24), correct_14702):
                            
                            # Evaluating a boolean operation
                            
                            # Getting the type of 'type_' (line 171)
                            type__14723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 31), 'type_')
                            # Getting the type of 'wrong_types' (line 171)
                            wrong_types_14724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 44), 'wrong_types')
                            # Applying the binary operator 'notin' (line 171)
                            result_contains_14725 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 31), 'notin', type__14723, wrong_types_14724)
                            
                            
                            # Getting the type of 'type_' (line 171)
                            type__14726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 60), 'type_')
                            # Getting the type of 'right_types' (line 171)
                            right_types_14727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 73), 'right_types')
                            # Applying the binary operator 'notin' (line 171)
                            result_contains_14728 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 60), 'notin', type__14726, right_types_14727)
                            
                            # Applying the binary operator 'and' (line 171)
                            result_and_keyword_14729 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 31), 'and', result_contains_14725, result_contains_14728)
                            
                            # Testing if the type of an if condition is none (line 171)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 171, 28), result_and_keyword_14729):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 171)
                                if_condition_14730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 28), result_and_keyword_14729)
                                # Assigning a type to the variable 'if_condition_14730' (line 171)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 28), 'if_condition_14730', if_condition_14730)
                                # SSA begins for if statement (line 171)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to append(...): (line 172)
                                # Processing the call arguments (line 172)
                                # Getting the type of 'type_' (line 172)
                                type__14733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 51), 'type_', False)
                                # Processing the call keyword arguments (line 172)
                                kwargs_14734 = {}
                                # Getting the type of 'wrong_types' (line 172)
                                wrong_types_14731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 32), 'wrong_types', False)
                                # Obtaining the member 'append' of a type (line 172)
                                append_14732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 32), wrong_types_14731, 'append')
                                # Calling append(args, kwargs) (line 172)
                                append_call_result_14735 = invoke(stypy.reporting.localization.Localization(__file__, 172, 32), append_14732, *[type__14733], **kwargs_14734)
                                
                                # SSA join for if statement (line 171)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 164)
                            if_condition_14703 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 24), correct_14702)
                            # Assigning a type to the variable 'if_condition_14703' (line 164)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 24), 'if_condition_14703', if_condition_14703)
                            # SSA begins for if statement (line 164)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Name to a Name (line 165):
                            
                            # Assigning a Name to a Name (line 165):
                            # Getting the type of 'True' (line 165)
                            True_14704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 42), 'True')
                            # Assigning a type to the variable 'match_found' (line 165)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 28), 'match_found', True_14704)
                            
                            # Getting the type of 'type_' (line 166)
                            type__14705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 31), 'type_')
                            # Getting the type of 'right_types' (line 166)
                            right_types_14706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 44), 'right_types')
                            # Applying the binary operator 'notin' (line 166)
                            result_contains_14707 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 31), 'notin', type__14705, right_types_14706)
                            
                            # Testing if the type of an if condition is none (line 166)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 166, 28), result_contains_14707):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 166)
                                if_condition_14708 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 166, 28), result_contains_14707)
                                # Assigning a type to the variable 'if_condition_14708' (line 166)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 28), 'if_condition_14708', if_condition_14708)
                                # SSA begins for if statement (line 166)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to append(...): (line 167)
                                # Processing the call arguments (line 167)
                                # Getting the type of 'type_' (line 167)
                                type__14711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 51), 'type_', False)
                                # Processing the call keyword arguments (line 167)
                                kwargs_14712 = {}
                                # Getting the type of 'right_types' (line 167)
                                right_types_14709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 32), 'right_types', False)
                                # Obtaining the member 'append' of a type (line 167)
                                append_14710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 32), right_types_14709, 'append')
                                # Calling append(args, kwargs) (line 167)
                                append_call_result_14713 = invoke(stypy.reporting.localization.Localization(__file__, 167, 32), append_14710, *[type__14711], **kwargs_14712)
                                
                                
                                # Getting the type of 'type_' (line 168)
                                type__14714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 35), 'type_')
                                # Getting the type of 'wrong_types' (line 168)
                                wrong_types_14715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 44), 'wrong_types')
                                # Applying the binary operator 'in' (line 168)
                                result_contains_14716 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 35), 'in', type__14714, wrong_types_14715)
                                
                                # Testing if the type of an if condition is none (line 168)

                                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 168, 32), result_contains_14716):
                                    pass
                                else:
                                    
                                    # Testing the type of an if condition (line 168)
                                    if_condition_14717 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 32), result_contains_14716)
                                    # Assigning a type to the variable 'if_condition_14717' (line 168)
                                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 32), 'if_condition_14717', if_condition_14717)
                                    # SSA begins for if statement (line 168)
                                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                    
                                    # Call to remove(...): (line 169)
                                    # Processing the call arguments (line 169)
                                    # Getting the type of 'type_' (line 169)
                                    type__14720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 55), 'type_', False)
                                    # Processing the call keyword arguments (line 169)
                                    kwargs_14721 = {}
                                    # Getting the type of 'wrong_types' (line 169)
                                    wrong_types_14718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 36), 'wrong_types', False)
                                    # Obtaining the member 'remove' of a type (line 169)
                                    remove_14719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 36), wrong_types_14718, 'remove')
                                    # Calling remove(args, kwargs) (line 169)
                                    remove_call_result_14722 = invoke(stypy.reporting.localization.Localization(__file__, 169, 36), remove_14719, *[type__14720], **kwargs_14721)
                                    
                                    # SSA join for if statement (line 168)
                                    module_type_store = module_type_store.join_ssa_context()
                                    

                                # SSA join for if statement (line 166)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA branch for the else part of an if statement (line 164)
                            module_type_store.open_ssa_branch('else')
                            
                            # Evaluating a boolean operation
                            
                            # Getting the type of 'type_' (line 171)
                            type__14723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 31), 'type_')
                            # Getting the type of 'wrong_types' (line 171)
                            wrong_types_14724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 44), 'wrong_types')
                            # Applying the binary operator 'notin' (line 171)
                            result_contains_14725 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 31), 'notin', type__14723, wrong_types_14724)
                            
                            
                            # Getting the type of 'type_' (line 171)
                            type__14726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 60), 'type_')
                            # Getting the type of 'right_types' (line 171)
                            right_types_14727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 73), 'right_types')
                            # Applying the binary operator 'notin' (line 171)
                            result_contains_14728 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 60), 'notin', type__14726, right_types_14727)
                            
                            # Applying the binary operator 'and' (line 171)
                            result_and_keyword_14729 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 31), 'and', result_contains_14725, result_contains_14728)
                            
                            # Testing if the type of an if condition is none (line 171)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 171, 28), result_and_keyword_14729):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 171)
                                if_condition_14730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 28), result_and_keyword_14729)
                                # Assigning a type to the variable 'if_condition_14730' (line 171)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 28), 'if_condition_14730', if_condition_14730)
                                # SSA begins for if statement (line 171)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to append(...): (line 172)
                                # Processing the call arguments (line 172)
                                # Getting the type of 'type_' (line 172)
                                type__14733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 51), 'type_', False)
                                # Processing the call keyword arguments (line 172)
                                kwargs_14734 = {}
                                # Getting the type of 'wrong_types' (line 172)
                                wrong_types_14731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 32), 'wrong_types', False)
                                # Obtaining the member 'append' of a type (line 172)
                                append_14732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 32), wrong_types_14731, 'append')
                                # Calling append(args, kwargs) (line 172)
                                append_call_result_14735 = invoke(stypy.reporting.localization.Localization(__file__, 172, 32), append_14732, *[type__14733], **kwargs_14734)
                                
                                # SSA join for if statement (line 171)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 164)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA branch for the else part of an if statement (line 159)
                        module_type_store.open_ssa_branch('else')
                        
                        # Assigning a Name to a Name (line 174):
                        
                        # Assigning a Name to a Name (line 174):
                        # Getting the type of 'True' (line 174)
                        True_14736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 38), 'True')
                        # Assigning a type to the variable 'match_found' (line 174)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 24), 'match_found', True_14736)
                        
                        # Call to append(...): (line 175)
                        # Processing the call arguments (line 175)
                        # Getting the type of 'type_' (line 175)
                        type__14739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 43), 'type_', False)
                        # Processing the call keyword arguments (line 175)
                        kwargs_14740 = {}
                        # Getting the type of 'right_types' (line 175)
                        right_types_14737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 24), 'right_types', False)
                        # Obtaining the member 'append' of a type (line 175)
                        append_14738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 24), right_types_14737, 'append')
                        # Calling append(args, kwargs) (line 175)
                        append_call_result_14741 = invoke(stypy.reporting.localization.Localization(__file__, 175, 24), append_14738, *[type__14739], **kwargs_14740)
                        
                        # SSA join for if statement (line 159)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 158)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Getting the type of 'match_found' (line 177)
            match_found_14742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 19), 'match_found')
            # Applying the 'not' unary operator (line 177)
            result_not__14743 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 15), 'not', match_found_14742)
            
            # Testing if the type of an if condition is none (line 177)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 177, 12), result_not__14743):
                pass
            else:
                
                # Testing the type of an if condition (line 177)
                if_condition_14744 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 12), result_not__14743)
                # Assigning a type to the variable 'if_condition_14744' (line 177)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'if_condition_14744', if_condition_14744)
                # SSA begins for if statement (line 177)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Evaluating a boolean operation
                
                # Getting the type of 'type_' (line 178)
                type__14745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 19), 'type_')
                # Getting the type of 'wrong_types' (line 178)
                wrong_types_14746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 32), 'wrong_types')
                # Applying the binary operator 'notin' (line 178)
                result_contains_14747 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 19), 'notin', type__14745, wrong_types_14746)
                
                
                # Getting the type of 'type_' (line 178)
                type__14748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 48), 'type_')
                # Getting the type of 'right_types' (line 178)
                right_types_14749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 61), 'right_types')
                # Applying the binary operator 'notin' (line 178)
                result_contains_14750 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 48), 'notin', type__14748, right_types_14749)
                
                # Applying the binary operator 'and' (line 178)
                result_and_keyword_14751 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 19), 'and', result_contains_14747, result_contains_14750)
                
                # Testing if the type of an if condition is none (line 178)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 178, 16), result_and_keyword_14751):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 178)
                    if_condition_14752 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 16), result_and_keyword_14751)
                    # Assigning a type to the variable 'if_condition_14752' (line 178)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'if_condition_14752', if_condition_14752)
                    # SSA begins for if statement (line 178)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to append(...): (line 179)
                    # Processing the call arguments (line 179)
                    # Getting the type of 'type_' (line 179)
                    type__14755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 39), 'type_', False)
                    # Processing the call keyword arguments (line 179)
                    kwargs_14756 = {}
                    # Getting the type of 'wrong_types' (line 179)
                    wrong_types_14753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 20), 'wrong_types', False)
                    # Obtaining the member 'append' of a type (line 179)
                    append_14754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 20), wrong_types_14753, 'append')
                    # Calling append(args, kwargs) (line 179)
                    append_call_result_14757 = invoke(stypy.reporting.localization.Localization(__file__, 179, 20), append_14754, *[type__14755], **kwargs_14756)
                    
                    # SSA join for if statement (line 178)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Name to a Name (line 180):
                
                # Assigning a Name to a Name (line 180):
                # Getting the type of 'False' (line 180)
                False_14758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 30), 'False')
                # Assigning a type to the variable 'match_found' (line 180)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'match_found', False_14758)
                # SSA join for if statement (line 177)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'self' (line 182)
        self_14759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 11), 'self')
        # Obtaining the member 'report_errors' of a type (line 182)
        report_errors_14760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 11), self_14759, 'report_errors')
        # Testing if the type of an if condition is none (line 182)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 182, 8), report_errors_14760):
            
            # Evaluating a boolean operation
            
            
            # Call to len(...): (line 197)
            # Processing the call arguments (line 197)
            # Getting the type of 'right_types' (line 197)
            right_types_14818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'right_types', False)
            # Processing the call keyword arguments (line 197)
            kwargs_14819 = {}
            # Getting the type of 'len' (line 197)
            len_14817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 15), 'len', False)
            # Calling len(args, kwargs) (line 197)
            len_call_result_14820 = invoke(stypy.reporting.localization.Localization(__file__, 197, 15), len_14817, *[right_types_14818], **kwargs_14819)
            
            int_14821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 35), 'int')
            # Applying the binary operator '==' (line 197)
            result_eq_14822 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 15), '==', len_call_result_14820, int_14821)
            
            
            
            # Call to len(...): (line 197)
            # Processing the call arguments (line 197)
            # Getting the type of 'wrong_types' (line 197)
            wrong_types_14824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 45), 'wrong_types', False)
            # Processing the call keyword arguments (line 197)
            kwargs_14825 = {}
            # Getting the type of 'len' (line 197)
            len_14823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 41), 'len', False)
            # Calling len(args, kwargs) (line 197)
            len_call_result_14826 = invoke(stypy.reporting.localization.Localization(__file__, 197, 41), len_14823, *[wrong_types_14824], **kwargs_14825)
            
            int_14827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 60), 'int')
            # Applying the binary operator '>' (line 197)
            result_gt_14828 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 41), '>', len_call_result_14826, int_14827)
            
            # Applying the binary operator 'and' (line 197)
            result_and_keyword_14829 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 15), 'and', result_eq_14822, result_gt_14828)
            
            # Testing if the type of an if condition is none (line 197)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 197, 12), result_and_keyword_14829):
                pass
            else:
                
                # Testing the type of an if condition (line 197)
                if_condition_14830 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 12), result_and_keyword_14829)
                # Assigning a type to the variable 'if_condition_14830' (line 197)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'if_condition_14830', if_condition_14830)
                # SSA begins for if statement (line 197)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to TypeWarning(...): (line 198)
                # Processing the call arguments (line 198)
                # Getting the type of 'localization' (line 198)
                localization_14832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 28), 'localization', False)
                
                # Call to format(...): (line 199)
                # Processing the call arguments (line 199)
                
                # Call to str(...): (line 200)
                # Processing the call arguments (line 200)
                # Getting the type of 'wrong_types' (line 200)
                wrong_types_14836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 36), 'wrong_types', False)
                # Processing the call keyword arguments (line 200)
                kwargs_14837 = {}
                # Getting the type of 'str' (line 200)
                str_14835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 32), 'str', False)
                # Calling str(args, kwargs) (line 200)
                str_call_result_14838 = invoke(stypy.reporting.localization.Localization(__file__, 200, 32), str_14835, *[wrong_types_14836], **kwargs_14837)
                
                
                # Call to str(...): (line 200)
                # Processing the call arguments (line 200)
                # Getting the type of 'self' (line 200)
                self_14840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 54), 'self', False)
                # Obtaining the member 'content_types' of a type (line 200)
                content_types_14841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 54), self_14840, 'content_types')
                # Processing the call keyword arguments (line 200)
                kwargs_14842 = {}
                # Getting the type of 'str' (line 200)
                str_14839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 50), 'str', False)
                # Calling str(args, kwargs) (line 200)
                str_call_result_14843 = invoke(stypy.reporting.localization.Localization(__file__, 200, 50), str_14839, *[content_types_14841], **kwargs_14842)
                
                # Processing the call keyword arguments (line 199)
                kwargs_14844 = {}
                str_14833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 28), 'str', 'Some of the iterable contained types: {0} do not match the expected ones {1}')
                # Obtaining the member 'format' of a type (line 199)
                format_14834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 28), str_14833, 'format')
                # Calling format(args, kwargs) (line 199)
                format_call_result_14845 = invoke(stypy.reporting.localization.Localization(__file__, 199, 28), format_14834, *[str_call_result_14838, str_call_result_14843], **kwargs_14844)
                
                # Processing the call keyword arguments (line 198)
                kwargs_14846 = {}
                # Getting the type of 'TypeWarning' (line 198)
                TypeWarning_14831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'TypeWarning', False)
                # Calling TypeWarning(args, kwargs) (line 198)
                TypeWarning_call_result_14847 = invoke(stypy.reporting.localization.Localization(__file__, 198, 16), TypeWarning_14831, *[localization_14832, format_call_result_14845], **kwargs_14846)
                
                # SSA join for if statement (line 197)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 182)
            if_condition_14761 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 8), report_errors_14760)
            # Assigning a type to the variable 'if_condition_14761' (line 182)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'if_condition_14761', if_condition_14761)
            # SSA begins for if statement (line 182)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to len(...): (line 184)
            # Processing the call arguments (line 184)
            # Getting the type of 'right_types' (line 184)
            right_types_14763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 19), 'right_types', False)
            # Processing the call keyword arguments (line 184)
            kwargs_14764 = {}
            # Getting the type of 'len' (line 184)
            len_14762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 15), 'len', False)
            # Calling len(args, kwargs) (line 184)
            len_call_result_14765 = invoke(stypy.reporting.localization.Localization(__file__, 184, 15), len_14762, *[right_types_14763], **kwargs_14764)
            
            int_14766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 35), 'int')
            # Applying the binary operator '==' (line 184)
            result_eq_14767 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 15), '==', len_call_result_14765, int_14766)
            
            # Testing if the type of an if condition is none (line 184)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 184, 12), result_eq_14767):
                
                
                # Call to len(...): (line 191)
                # Processing the call arguments (line 191)
                # Getting the type of 'wrong_types' (line 191)
                wrong_types_14794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 23), 'wrong_types', False)
                # Processing the call keyword arguments (line 191)
                kwargs_14795 = {}
                # Getting the type of 'len' (line 191)
                len_14793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 19), 'len', False)
                # Calling len(args, kwargs) (line 191)
                len_call_result_14796 = invoke(stypy.reporting.localization.Localization(__file__, 191, 19), len_14793, *[wrong_types_14794], **kwargs_14795)
                
                int_14797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 38), 'int')
                # Applying the binary operator '>' (line 191)
                result_gt_14798 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 19), '>', len_call_result_14796, int_14797)
                
                # Testing if the type of an if condition is none (line 191)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 191, 16), result_gt_14798):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 191)
                    if_condition_14799 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 16), result_gt_14798)
                    # Assigning a type to the variable 'if_condition_14799' (line 191)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'if_condition_14799', if_condition_14799)
                    # SSA begins for if statement (line 191)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to TypeWarning(...): (line 192)
                    # Processing the call arguments (line 192)
                    # Getting the type of 'localization' (line 192)
                    localization_14801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 32), 'localization', False)
                    
                    # Call to format(...): (line 193)
                    # Processing the call arguments (line 193)
                    
                    # Call to str(...): (line 194)
                    # Processing the call arguments (line 194)
                    # Getting the type of 'wrong_types' (line 194)
                    wrong_types_14805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 40), 'wrong_types', False)
                    # Processing the call keyword arguments (line 194)
                    kwargs_14806 = {}
                    # Getting the type of 'str' (line 194)
                    str_14804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 36), 'str', False)
                    # Calling str(args, kwargs) (line 194)
                    str_call_result_14807 = invoke(stypy.reporting.localization.Localization(__file__, 194, 36), str_14804, *[wrong_types_14805], **kwargs_14806)
                    
                    
                    # Call to str(...): (line 194)
                    # Processing the call arguments (line 194)
                    # Getting the type of 'self' (line 194)
                    self_14809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 58), 'self', False)
                    # Obtaining the member 'content_types' of a type (line 194)
                    content_types_14810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 58), self_14809, 'content_types')
                    # Processing the call keyword arguments (line 194)
                    kwargs_14811 = {}
                    # Getting the type of 'str' (line 194)
                    str_14808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 54), 'str', False)
                    # Calling str(args, kwargs) (line 194)
                    str_call_result_14812 = invoke(stypy.reporting.localization.Localization(__file__, 194, 54), str_14808, *[content_types_14810], **kwargs_14811)
                    
                    # Processing the call keyword arguments (line 193)
                    kwargs_14813 = {}
                    str_14802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 32), 'str', 'Some of the iterable contained types: {0} do not match the expected ones {1}')
                    # Obtaining the member 'format' of a type (line 193)
                    format_14803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 32), str_14802, 'format')
                    # Calling format(args, kwargs) (line 193)
                    format_call_result_14814 = invoke(stypy.reporting.localization.Localization(__file__, 193, 32), format_14803, *[str_call_result_14807, str_call_result_14812], **kwargs_14813)
                    
                    # Processing the call keyword arguments (line 192)
                    kwargs_14815 = {}
                    # Getting the type of 'TypeWarning' (line 192)
                    TypeWarning_14800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 20), 'TypeWarning', False)
                    # Calling TypeWarning(args, kwargs) (line 192)
                    TypeWarning_call_result_14816 = invoke(stypy.reporting.localization.Localization(__file__, 192, 20), TypeWarning_14800, *[localization_14801, format_call_result_14814], **kwargs_14815)
                    
                    # SSA join for if statement (line 191)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 184)
                if_condition_14768 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 12), result_eq_14767)
                # Assigning a type to the variable 'if_condition_14768' (line 184)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'if_condition_14768', if_condition_14768)
                # SSA begins for if statement (line 184)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                
                # Call to len(...): (line 185)
                # Processing the call arguments (line 185)
                # Getting the type of 'wrong_types' (line 185)
                wrong_types_14770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 23), 'wrong_types', False)
                # Processing the call keyword arguments (line 185)
                kwargs_14771 = {}
                # Getting the type of 'len' (line 185)
                len_14769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 19), 'len', False)
                # Calling len(args, kwargs) (line 185)
                len_call_result_14772 = invoke(stypy.reporting.localization.Localization(__file__, 185, 19), len_14769, *[wrong_types_14770], **kwargs_14771)
                
                int_14773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 38), 'int')
                # Applying the binary operator '>' (line 185)
                result_gt_14774 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 19), '>', len_call_result_14772, int_14773)
                
                # Testing if the type of an if condition is none (line 185)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 185, 16), result_gt_14774):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 185)
                    if_condition_14775 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 16), result_gt_14774)
                    # Assigning a type to the variable 'if_condition_14775' (line 185)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 16), 'if_condition_14775', if_condition_14775)
                    # SSA begins for if statement (line 185)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to TypeError(...): (line 186)
                    # Processing the call arguments (line 186)
                    # Getting the type of 'localization' (line 186)
                    localization_14777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 30), 'localization', False)
                    
                    # Call to format(...): (line 187)
                    # Processing the call arguments (line 187)
                    
                    # Call to str(...): (line 188)
                    # Processing the call arguments (line 188)
                    # Getting the type of 'types_to_examine' (line 188)
                    types_to_examine_14781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 38), 'types_to_examine', False)
                    # Processing the call keyword arguments (line 188)
                    kwargs_14782 = {}
                    # Getting the type of 'str' (line 188)
                    str_14780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 34), 'str', False)
                    # Calling str(args, kwargs) (line 188)
                    str_call_result_14783 = invoke(stypy.reporting.localization.Localization(__file__, 188, 34), str_14780, *[types_to_examine_14781], **kwargs_14782)
                    
                    
                    # Call to str(...): (line 188)
                    # Processing the call arguments (line 188)
                    # Getting the type of 'self' (line 188)
                    self_14785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 61), 'self', False)
                    # Obtaining the member 'content_types' of a type (line 188)
                    content_types_14786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 61), self_14785, 'content_types')
                    # Processing the call keyword arguments (line 188)
                    kwargs_14787 = {}
                    # Getting the type of 'str' (line 188)
                    str_14784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 57), 'str', False)
                    # Calling str(args, kwargs) (line 188)
                    str_call_result_14788 = invoke(stypy.reporting.localization.Localization(__file__, 188, 57), str_14784, *[content_types_14786], **kwargs_14787)
                    
                    # Processing the call keyword arguments (line 187)
                    kwargs_14789 = {}
                    str_14778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 30), 'str', 'None of the iterable contained types: {0} match the expected ones {1}')
                    # Obtaining the member 'format' of a type (line 187)
                    format_14779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 30), str_14778, 'format')
                    # Calling format(args, kwargs) (line 187)
                    format_call_result_14790 = invoke(stypy.reporting.localization.Localization(__file__, 187, 30), format_14779, *[str_call_result_14783, str_call_result_14788], **kwargs_14789)
                    
                    # Processing the call keyword arguments (line 186)
                    kwargs_14791 = {}
                    # Getting the type of 'TypeError' (line 186)
                    TypeError_14776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 20), 'TypeError', False)
                    # Calling TypeError(args, kwargs) (line 186)
                    TypeError_call_result_14792 = invoke(stypy.reporting.localization.Localization(__file__, 186, 20), TypeError_14776, *[localization_14777, format_call_result_14790], **kwargs_14791)
                    
                    # SSA join for if statement (line 185)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA branch for the else part of an if statement (line 184)
                module_type_store.open_ssa_branch('else')
                
                
                # Call to len(...): (line 191)
                # Processing the call arguments (line 191)
                # Getting the type of 'wrong_types' (line 191)
                wrong_types_14794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 23), 'wrong_types', False)
                # Processing the call keyword arguments (line 191)
                kwargs_14795 = {}
                # Getting the type of 'len' (line 191)
                len_14793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 19), 'len', False)
                # Calling len(args, kwargs) (line 191)
                len_call_result_14796 = invoke(stypy.reporting.localization.Localization(__file__, 191, 19), len_14793, *[wrong_types_14794], **kwargs_14795)
                
                int_14797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 38), 'int')
                # Applying the binary operator '>' (line 191)
                result_gt_14798 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 19), '>', len_call_result_14796, int_14797)
                
                # Testing if the type of an if condition is none (line 191)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 191, 16), result_gt_14798):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 191)
                    if_condition_14799 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 16), result_gt_14798)
                    # Assigning a type to the variable 'if_condition_14799' (line 191)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'if_condition_14799', if_condition_14799)
                    # SSA begins for if statement (line 191)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to TypeWarning(...): (line 192)
                    # Processing the call arguments (line 192)
                    # Getting the type of 'localization' (line 192)
                    localization_14801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 32), 'localization', False)
                    
                    # Call to format(...): (line 193)
                    # Processing the call arguments (line 193)
                    
                    # Call to str(...): (line 194)
                    # Processing the call arguments (line 194)
                    # Getting the type of 'wrong_types' (line 194)
                    wrong_types_14805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 40), 'wrong_types', False)
                    # Processing the call keyword arguments (line 194)
                    kwargs_14806 = {}
                    # Getting the type of 'str' (line 194)
                    str_14804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 36), 'str', False)
                    # Calling str(args, kwargs) (line 194)
                    str_call_result_14807 = invoke(stypy.reporting.localization.Localization(__file__, 194, 36), str_14804, *[wrong_types_14805], **kwargs_14806)
                    
                    
                    # Call to str(...): (line 194)
                    # Processing the call arguments (line 194)
                    # Getting the type of 'self' (line 194)
                    self_14809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 58), 'self', False)
                    # Obtaining the member 'content_types' of a type (line 194)
                    content_types_14810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 58), self_14809, 'content_types')
                    # Processing the call keyword arguments (line 194)
                    kwargs_14811 = {}
                    # Getting the type of 'str' (line 194)
                    str_14808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 54), 'str', False)
                    # Calling str(args, kwargs) (line 194)
                    str_call_result_14812 = invoke(stypy.reporting.localization.Localization(__file__, 194, 54), str_14808, *[content_types_14810], **kwargs_14811)
                    
                    # Processing the call keyword arguments (line 193)
                    kwargs_14813 = {}
                    str_14802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 32), 'str', 'Some of the iterable contained types: {0} do not match the expected ones {1}')
                    # Obtaining the member 'format' of a type (line 193)
                    format_14803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 32), str_14802, 'format')
                    # Calling format(args, kwargs) (line 193)
                    format_call_result_14814 = invoke(stypy.reporting.localization.Localization(__file__, 193, 32), format_14803, *[str_call_result_14807, str_call_result_14812], **kwargs_14813)
                    
                    # Processing the call keyword arguments (line 192)
                    kwargs_14815 = {}
                    # Getting the type of 'TypeWarning' (line 192)
                    TypeWarning_14800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 20), 'TypeWarning', False)
                    # Calling TypeWarning(args, kwargs) (line 192)
                    TypeWarning_call_result_14816 = invoke(stypy.reporting.localization.Localization(__file__, 192, 20), TypeWarning_14800, *[localization_14801, format_call_result_14814], **kwargs_14815)
                    
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
            right_types_14818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'right_types', False)
            # Processing the call keyword arguments (line 197)
            kwargs_14819 = {}
            # Getting the type of 'len' (line 197)
            len_14817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 15), 'len', False)
            # Calling len(args, kwargs) (line 197)
            len_call_result_14820 = invoke(stypy.reporting.localization.Localization(__file__, 197, 15), len_14817, *[right_types_14818], **kwargs_14819)
            
            int_14821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 35), 'int')
            # Applying the binary operator '==' (line 197)
            result_eq_14822 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 15), '==', len_call_result_14820, int_14821)
            
            
            
            # Call to len(...): (line 197)
            # Processing the call arguments (line 197)
            # Getting the type of 'wrong_types' (line 197)
            wrong_types_14824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 45), 'wrong_types', False)
            # Processing the call keyword arguments (line 197)
            kwargs_14825 = {}
            # Getting the type of 'len' (line 197)
            len_14823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 41), 'len', False)
            # Calling len(args, kwargs) (line 197)
            len_call_result_14826 = invoke(stypy.reporting.localization.Localization(__file__, 197, 41), len_14823, *[wrong_types_14824], **kwargs_14825)
            
            int_14827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 60), 'int')
            # Applying the binary operator '>' (line 197)
            result_gt_14828 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 41), '>', len_call_result_14826, int_14827)
            
            # Applying the binary operator 'and' (line 197)
            result_and_keyword_14829 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 15), 'and', result_eq_14822, result_gt_14828)
            
            # Testing if the type of an if condition is none (line 197)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 197, 12), result_and_keyword_14829):
                pass
            else:
                
                # Testing the type of an if condition (line 197)
                if_condition_14830 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 12), result_and_keyword_14829)
                # Assigning a type to the variable 'if_condition_14830' (line 197)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'if_condition_14830', if_condition_14830)
                # SSA begins for if statement (line 197)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to TypeWarning(...): (line 198)
                # Processing the call arguments (line 198)
                # Getting the type of 'localization' (line 198)
                localization_14832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 28), 'localization', False)
                
                # Call to format(...): (line 199)
                # Processing the call arguments (line 199)
                
                # Call to str(...): (line 200)
                # Processing the call arguments (line 200)
                # Getting the type of 'wrong_types' (line 200)
                wrong_types_14836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 36), 'wrong_types', False)
                # Processing the call keyword arguments (line 200)
                kwargs_14837 = {}
                # Getting the type of 'str' (line 200)
                str_14835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 32), 'str', False)
                # Calling str(args, kwargs) (line 200)
                str_call_result_14838 = invoke(stypy.reporting.localization.Localization(__file__, 200, 32), str_14835, *[wrong_types_14836], **kwargs_14837)
                
                
                # Call to str(...): (line 200)
                # Processing the call arguments (line 200)
                # Getting the type of 'self' (line 200)
                self_14840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 54), 'self', False)
                # Obtaining the member 'content_types' of a type (line 200)
                content_types_14841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 54), self_14840, 'content_types')
                # Processing the call keyword arguments (line 200)
                kwargs_14842 = {}
                # Getting the type of 'str' (line 200)
                str_14839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 50), 'str', False)
                # Calling str(args, kwargs) (line 200)
                str_call_result_14843 = invoke(stypy.reporting.localization.Localization(__file__, 200, 50), str_14839, *[content_types_14841], **kwargs_14842)
                
                # Processing the call keyword arguments (line 199)
                kwargs_14844 = {}
                str_14833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 28), 'str', 'Some of the iterable contained types: {0} do not match the expected ones {1}')
                # Obtaining the member 'format' of a type (line 199)
                format_14834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 28), str_14833, 'format')
                # Calling format(args, kwargs) (line 199)
                format_call_result_14845 = invoke(stypy.reporting.localization.Localization(__file__, 199, 28), format_14834, *[str_call_result_14838, str_call_result_14843], **kwargs_14844)
                
                # Processing the call keyword arguments (line 198)
                kwargs_14846 = {}
                # Getting the type of 'TypeWarning' (line 198)
                TypeWarning_14831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'TypeWarning', False)
                # Calling TypeWarning(args, kwargs) (line 198)
                TypeWarning_call_result_14847 = invoke(stypy.reporting.localization.Localization(__file__, 198, 16), TypeWarning_14831, *[localization_14832, format_call_result_14845], **kwargs_14846)
                
                # SSA join for if statement (line 197)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 182)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to len(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'right_types' (line 203)
        right_types_14849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), 'right_types', False)
        # Processing the call keyword arguments (line 203)
        kwargs_14850 = {}
        # Getting the type of 'len' (line 203)
        len_14848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 11), 'len', False)
        # Calling len(args, kwargs) (line 203)
        len_call_result_14851 = invoke(stypy.reporting.localization.Localization(__file__, 203, 11), len_14848, *[right_types_14849], **kwargs_14850)
        
        int_14852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 30), 'int')
        # Applying the binary operator '>' (line 203)
        result_gt_14853 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 11), '>', len_call_result_14851, int_14852)
        
        # Testing if the type of an if condition is none (line 203)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 203, 8), result_gt_14853):
            
            # Obtaining an instance of the builtin type 'tuple' (line 206)
            tuple_14858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 206)
            # Adding element type (line 206)
            # Getting the type of 'False' (line 206)
            False_14859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 19), 'False')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 19), tuple_14858, False_14859)
            # Adding element type (line 206)
            # Getting the type of 'wrong_types' (line 206)
            wrong_types_14860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 26), 'wrong_types')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 19), tuple_14858, wrong_types_14860)
            
            # Assigning a type to the variable 'stypy_return_type' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'stypy_return_type', tuple_14858)
        else:
            
            # Testing the type of an if condition (line 203)
            if_condition_14854 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 8), result_gt_14853)
            # Assigning a type to the variable 'if_condition_14854' (line 203)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'if_condition_14854', if_condition_14854)
            # SSA begins for if statement (line 203)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining an instance of the builtin type 'tuple' (line 204)
            tuple_14855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 204)
            # Adding element type (line 204)
            # Getting the type of 'True' (line 204)
            True_14856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 19), 'True')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 19), tuple_14855, True_14856)
            # Adding element type (line 204)
            # Getting the type of 'None' (line 204)
            None_14857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 25), 'None')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 19), tuple_14855, None_14857)
            
            # Assigning a type to the variable 'stypy_return_type' (line 204)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'stypy_return_type', tuple_14855)
            # SSA branch for the else part of an if statement (line 203)
            module_type_store.open_ssa_branch('else')
            
            # Obtaining an instance of the builtin type 'tuple' (line 206)
            tuple_14858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 206)
            # Adding element type (line 206)
            # Getting the type of 'False' (line 206)
            False_14859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 19), 'False')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 19), tuple_14858, False_14859)
            # Adding element type (line 206)
            # Getting the type of 'wrong_types' (line 206)
            wrong_types_14860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 26), 'wrong_types')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 19), tuple_14858, wrong_types_14860)
            
            # Assigning a type to the variable 'stypy_return_type' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'stypy_return_type', tuple_14858)
            # SSA join for if statement (line 203)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 145)
        stypy_return_type_14861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14861)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_14861


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
        str_14862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 18), 'str', 'Iterable[')
        # Assigning a type to the variable 'ret_str' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'ret_str', str_14862)
        
        # Assigning a Str to a Name (line 211):
        
        # Assigning a Str to a Name (line 211):
        str_14863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 19), 'str', '')
        # Assigning a type to the variable 'contents' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'contents', str_14863)
        
        # Getting the type of 'self' (line 212)
        self_14864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 23), 'self')
        # Obtaining the member 'content_types' of a type (line 212)
        content_types_14865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 23), self_14864, 'content_types')
        # Assigning a type to the variable 'content_types_14865' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'content_types_14865', content_types_14865)
        # Testing if the for loop is going to be iterated (line 212)
        # Testing the type of a for loop iterable (line 212)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 212, 8), content_types_14865)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 212, 8), content_types_14865):
            # Getting the type of the for loop variable (line 212)
            for_loop_var_14866 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 212, 8), content_types_14865)
            # Assigning a type to the variable 'content' (line 212)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'content', for_loop_var_14866)
            # SSA begins for a for statement (line 212)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'contents' (line 213)
            contents_14867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'contents')
            
            # Call to str(...): (line 213)
            # Processing the call arguments (line 213)
            # Getting the type of 'content' (line 213)
            content_14869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 28), 'content', False)
            # Processing the call keyword arguments (line 213)
            kwargs_14870 = {}
            # Getting the type of 'str' (line 213)
            str_14868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 24), 'str', False)
            # Calling str(args, kwargs) (line 213)
            str_call_result_14871 = invoke(stypy.reporting.localization.Localization(__file__, 213, 24), str_14868, *[content_14869], **kwargs_14870)
            
            str_14872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 39), 'str', ', ')
            # Applying the binary operator '+' (line 213)
            result_add_14873 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 24), '+', str_call_result_14871, str_14872)
            
            # Applying the binary operator '+=' (line 213)
            result_iadd_14874 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 12), '+=', contents_14867, result_add_14873)
            # Assigning a type to the variable 'contents' (line 213)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'contents', result_iadd_14874)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Subscript to a Name (line 214):
        
        # Assigning a Subscript to a Name (line 214):
        
        # Obtaining the type of the subscript
        int_14875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 29), 'int')
        slice_14876 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 214, 19), None, int_14875, None)
        # Getting the type of 'contents' (line 214)
        contents_14877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 19), 'contents')
        # Obtaining the member '__getitem__' of a type (line 214)
        getitem___14878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 19), contents_14877, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 214)
        subscript_call_result_14879 = invoke(stypy.reporting.localization.Localization(__file__, 214, 19), getitem___14878, slice_14876)
        
        # Assigning a type to the variable 'contents' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'contents', subscript_call_result_14879)
        
        # Getting the type of 'ret_str' (line 216)
        ret_str_14880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'ret_str')
        # Getting the type of 'contents' (line 216)
        contents_14881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 19), 'contents')
        # Applying the binary operator '+=' (line 216)
        result_iadd_14882 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 8), '+=', ret_str_14880, contents_14881)
        # Assigning a type to the variable 'ret_str' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'ret_str', result_iadd_14882)
        
        
        # Getting the type of 'ret_str' (line 217)
        ret_str_14883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'ret_str')
        str_14884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 19), 'str', ']')
        # Applying the binary operator '+=' (line 217)
        result_iadd_14885 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 8), '+=', ret_str_14883, str_14884)
        # Assigning a type to the variable 'ret_str' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'ret_str', result_iadd_14885)
        
        # Getting the type of 'ret_str' (line 218)
        ret_str_14886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), 'ret_str')
        # Assigning a type to the variable 'stypy_return_type' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'stypy_return_type', ret_str_14886)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 208)
        stypy_return_type_14887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14887)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_14887


# Assigning a type to the variable 'IterableDataStructureWithTypedElements' (line 129)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 0), 'IterableDataStructureWithTypedElements', IterableDataStructureWithTypedElements)
# Declaration of the 'DynamicType' class
# Getting the type of 'TypeGroup' (line 221)
TypeGroup_14888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 18), 'TypeGroup')

class DynamicType(TypeGroup_14888, ):
    str_14889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, (-1)), 'str', '\n    Any type (type cannot be statically calculated)\n    ')

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
        self_14892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 227)
        list_14893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 227)
        
        # Processing the call keyword arguments (line 227)
        kwargs_14894 = {}
        # Getting the type of 'TypeGroup' (line 227)
        TypeGroup_14890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 227)
        init___14891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), TypeGroup_14890, '__init__')
        # Calling __init__(args, kwargs) (line 227)
        init___call_result_14895 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), init___14891, *[self_14892, list_14893], **kwargs_14894)
        
        
        # Assigning a Name to a Attribute (line 228):
        
        # Assigning a Name to a Attribute (line 228):
        # Getting the type of 'members' (line 228)
        members_14896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 23), 'members')
        # Getting the type of 'self' (line 228)
        self_14897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'self')
        # Setting the type of the member 'members' of a type (line 228)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), self_14897, 'members', members_14896)
        
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
        True_14898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'stypy_return_type', True_14898)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 230)
        stypy_return_type_14899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14899)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_14899


# Assigning a type to the variable 'DynamicType' (line 221)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 0), 'DynamicType', DynamicType)
# Declaration of the 'SupportsStructuralIntercession' class
# Getting the type of 'TypeGroup' (line 234)
TypeGroup_14900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 37), 'TypeGroup')

class SupportsStructuralIntercession(TypeGroup_14900, ):
    str_14901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, (-1)), 'str', '\n    Any Python object that supports structural intercession\n    ')

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
        self_14904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 240)
        list_14905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 240)
        
        # Processing the call keyword arguments (line 240)
        kwargs_14906 = {}
        # Getting the type of 'TypeGroup' (line 240)
        TypeGroup_14902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 240)
        init___14903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), TypeGroup_14902, '__init__')
        # Calling __init__(args, kwargs) (line 240)
        init___call_result_14907 = invoke(stypy.reporting.localization.Localization(__file__, 240, 8), init___14903, *[self_14904, list_14905], **kwargs_14906)
        
        
        # Assigning a Name to a Attribute (line 241):
        
        # Assigning a Name to a Attribute (line 241):
        # Getting the type of 'members' (line 241)
        members_14908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 23), 'members')
        # Getting the type of 'self' (line 241)
        self_14909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'self')
        # Setting the type of the member 'members' of a type (line 241)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 8), self_14909, 'members', members_14908)
        
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
        type__14910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 21), 'type_')
        # Getting the type of 'self' (line 244)
        self_14911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'self')
        # Setting the type of the member 'type_' of a type (line 244)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), self_14911, 'type_', type__14910)
        
        # Call to supports_structural_reflection(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'type_' (line 245)
        type__14914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 83), 'type_', False)
        # Processing the call keyword arguments (line 245)
        kwargs_14915 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 245)
        type_inference_proxy_management_copy_14912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 15), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'supports_structural_reflection' of a type (line 245)
        supports_structural_reflection_14913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 15), type_inference_proxy_management_copy_14912, 'supports_structural_reflection')
        # Calling supports_structural_reflection(args, kwargs) (line 245)
        supports_structural_reflection_call_result_14916 = invoke(stypy.reporting.localization.Localization(__file__, 245, 15), supports_structural_reflection_14913, *[type__14914], **kwargs_14915)
        
        # Assigning a type to the variable 'stypy_return_type' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'stypy_return_type', supports_structural_reflection_call_result_14916)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 243)
        stypy_return_type_14917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14917)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_14917


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
        self_14918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 15), 'self')
        # Obtaining the member 'type_' of a type (line 248)
        type__14919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 15), self_14918, 'type_')
        # Assigning a type to the variable 'temp' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'temp', type__14919)
        
        # Assigning a Name to a Attribute (line 249):
        
        # Assigning a Name to a Attribute (line 249):
        # Getting the type of 'None' (line 249)
        None_14920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 21), 'None')
        # Getting the type of 'self' (line 249)
        self_14921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'self')
        # Setting the type of the member 'type_' of a type (line 249)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), self_14921, 'type_', None_14920)
        # Getting the type of 'temp' (line 251)
        temp_14922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'stypy_return_type', temp_14922)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 247)
        stypy_return_type_14923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14923)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_14923


# Assigning a type to the variable 'SupportsStructuralIntercession' (line 234)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 0), 'SupportsStructuralIntercession', SupportsStructuralIntercession)
# Declaration of the 'SubtypeOf' class
# Getting the type of 'TypeGroup' (line 254)
TypeGroup_14924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'TypeGroup')

class SubtypeOf(TypeGroup_14924, ):
    str_14925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, (-1)), 'str', '\n    A subtype of the type passed in the constructor\n    ')

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
        self_14928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 260)
        list_14929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 260)
        
        # Processing the call keyword arguments (line 260)
        kwargs_14930 = {}
        # Getting the type of 'TypeGroup' (line 260)
        TypeGroup_14926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 260)
        init___14927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 8), TypeGroup_14926, '__init__')
        # Calling __init__(args, kwargs) (line 260)
        init___call_result_14931 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), init___14927, *[self_14928, list_14929], **kwargs_14930)
        
        
        # Assigning a Name to a Attribute (line 261):
        
        # Assigning a Name to a Attribute (line 261):
        # Getting the type of 'types_' (line 261)
        types__14932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 21), 'types_')
        # Getting the type of 'self' (line 261)
        self_14933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'self')
        # Setting the type of the member 'types' of a type (line 261)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), self_14933, 'types', types__14932)
        
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
        type__14934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 21), 'type_')
        # Getting the type of 'self' (line 264)
        self_14935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'self')
        # Setting the type of the member 'type_' of a type (line 264)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 8), self_14935, 'type_', type__14934)
        
        # Getting the type of 'self' (line 265)
        self_14936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 28), 'self')
        # Obtaining the member 'types' of a type (line 265)
        types_14937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 28), self_14936, 'types')
        # Assigning a type to the variable 'types_14937' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'types_14937', types_14937)
        # Testing if the for loop is going to be iterated (line 265)
        # Testing the type of a for loop iterable (line 265)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 265, 8), types_14937)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 265, 8), types_14937):
            # Getting the type of the for loop variable (line 265)
            for_loop_var_14938 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 265, 8), types_14937)
            # Assigning a type to the variable 'pattern_type' (line 265)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'pattern_type', for_loop_var_14938)
            # SSA begins for a for statement (line 265)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to issubclass(...): (line 266)
            # Processing the call arguments (line 266)
            # Getting the type of 'type_' (line 266)
            type__14940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 30), 'type_', False)
            # Getting the type of 'pattern_type' (line 266)
            pattern_type_14941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 37), 'pattern_type', False)
            # Processing the call keyword arguments (line 266)
            kwargs_14942 = {}
            # Getting the type of 'issubclass' (line 266)
            issubclass_14939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 19), 'issubclass', False)
            # Calling issubclass(args, kwargs) (line 266)
            issubclass_call_result_14943 = invoke(stypy.reporting.localization.Localization(__file__, 266, 19), issubclass_14939, *[type__14940, pattern_type_14941], **kwargs_14942)
            
            # Applying the 'not' unary operator (line 266)
            result_not__14944 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 15), 'not', issubclass_call_result_14943)
            
            # Testing if the type of an if condition is none (line 266)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 266, 12), result_not__14944):
                pass
            else:
                
                # Testing the type of an if condition (line 266)
                if_condition_14945 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 266, 12), result_not__14944)
                # Assigning a type to the variable 'if_condition_14945' (line 266)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'if_condition_14945', if_condition_14945)
                # SSA begins for if statement (line 266)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'False' (line 267)
                False_14946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 267)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 16), 'stypy_return_type', False_14946)
                # SSA join for if statement (line 266)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'True' (line 268)
        True_14947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'stypy_return_type', True_14947)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 263)
        stypy_return_type_14948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14948)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_14948


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
        self_14949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 15), 'self')
        # Obtaining the member 'type_' of a type (line 271)
        type__14950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 15), self_14949, 'type_')
        # Assigning a type to the variable 'temp' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'temp', type__14950)
        
        # Assigning a Name to a Attribute (line 272):
        
        # Assigning a Name to a Attribute (line 272):
        # Getting the type of 'None' (line 272)
        None_14951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 21), 'None')
        # Getting the type of 'self' (line 272)
        self_14952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'self')
        # Setting the type of the member 'type_' of a type (line 272)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), self_14952, 'type_', None_14951)
        # Getting the type of 'temp' (line 274)
        temp_14953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'stypy_return_type', temp_14953)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 270)
        stypy_return_type_14954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14954)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_14954


# Assigning a type to the variable 'SubtypeOf' (line 254)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 0), 'SubtypeOf', SubtypeOf)
# Declaration of the 'IsHashable' class
# Getting the type of 'TypeGroup' (line 277)
TypeGroup_14955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 17), 'TypeGroup')

class IsHashable(TypeGroup_14955, ):
    str_14956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, (-1)), 'str', '\n    Represent types that can properly implement the __hash__ members, so it can be placed as keys on a dict\n    ')

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
        self_14959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 283)
        list_14960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 283)
        
        # Processing the call keyword arguments (line 283)
        kwargs_14961 = {}
        # Getting the type of 'TypeGroup' (line 283)
        TypeGroup_14957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 283)
        init___14958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), TypeGroup_14957, '__init__')
        # Calling __init__(args, kwargs) (line 283)
        init___call_result_14962 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), init___14958, *[self_14959, list_14960], **kwargs_14961)
        
        
        # Assigning a Name to a Attribute (line 284):
        
        # Assigning a Name to a Attribute (line 284):
        # Getting the type of 'types_' (line 284)
        types__14963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 21), 'types_')
        # Getting the type of 'self' (line 284)
        self_14964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'self')
        # Setting the type of the member 'types' of a type (line 284)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 8), self_14964, 'types', types__14963)
        
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
        type__14965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 21), 'type_')
        # Getting the type of 'self' (line 287)
        self_14966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'self')
        # Setting the type of the member 'type_' of a type (line 287)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), self_14966, 'type_', type__14965)
        
        # Call to issubclass(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 'type_' (line 288)
        type__14968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 22), 'type_', False)
        # Getting the type of 'collections' (line 288)
        collections_14969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 29), 'collections', False)
        # Obtaining the member 'Hashable' of a type (line 288)
        Hashable_14970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 29), collections_14969, 'Hashable')
        # Processing the call keyword arguments (line 288)
        kwargs_14971 = {}
        # Getting the type of 'issubclass' (line 288)
        issubclass_14967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 11), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 288)
        issubclass_call_result_14972 = invoke(stypy.reporting.localization.Localization(__file__, 288, 11), issubclass_14967, *[type__14968, Hashable_14970], **kwargs_14971)
        
        # Testing if the type of an if condition is none (line 288)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 288, 8), issubclass_call_result_14972):
            pass
        else:
            
            # Testing the type of an if condition (line 288)
            if_condition_14973 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 8), issubclass_call_result_14972)
            # Assigning a type to the variable 'if_condition_14973' (line 288)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'if_condition_14973', if_condition_14973)
            # SSA begins for if statement (line 288)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'True' (line 289)
            True_14974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 289)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'stypy_return_type', True_14974)
            # SSA join for if statement (line 288)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'False' (line 290)
        False_14975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'stypy_return_type', False_14975)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 286)
        stypy_return_type_14976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14976)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_14976


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
        self_14977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 15), 'self')
        # Obtaining the member 'type_' of a type (line 293)
        type__14978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 15), self_14977, 'type_')
        # Assigning a type to the variable 'temp' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'temp', type__14978)
        
        # Assigning a Name to a Attribute (line 294):
        
        # Assigning a Name to a Attribute (line 294):
        # Getting the type of 'None' (line 294)
        None_14979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 21), 'None')
        # Getting the type of 'self' (line 294)
        self_14980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'self')
        # Setting the type of the member 'type_' of a type (line 294)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), self_14980, 'type_', None_14979)
        # Getting the type of 'temp' (line 296)
        temp_14981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'stypy_return_type', temp_14981)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 292)
        stypy_return_type_14982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14982)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_14982


# Assigning a type to the variable 'IsHashable' (line 277)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 0), 'IsHashable', IsHashable)
# Declaration of the 'TypeOfParam' class
# Getting the type of 'TypeGroup' (line 299)
TypeGroup_14983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 18), 'TypeGroup')
# Getting the type of 'DependentType' (line 299)
DependentType_14984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 29), 'DependentType')

class TypeOfParam(TypeGroup_14983, DependentType_14984, ):
    str_14985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, (-1)), 'str', "\n    This type group is special in the sense that it don't really group any types, only returns the param number\n    passed in the constructor when it is called with a list of parameters. This is really used to simplify several\n    type rules in which the type returned by a member call is equal to the type of one of its parameters\n    ")

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
        self_14988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 31), 'self', False)
        # Processing the call keyword arguments (line 307)
        kwargs_14989 = {}
        # Getting the type of 'DependentType' (line 307)
        DependentType_14986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'DependentType', False)
        # Obtaining the member '__init__' of a type (line 307)
        init___14987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 8), DependentType_14986, '__init__')
        # Calling __init__(args, kwargs) (line 307)
        init___call_result_14990 = invoke(stypy.reporting.localization.Localization(__file__, 307, 8), init___14987, *[self_14988], **kwargs_14989)
        
        
        # Call to __init__(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 'self' (line 308)
        self_14993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 308)
        list_14994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 308)
        
        # Processing the call keyword arguments (line 308)
        kwargs_14995 = {}
        # Getting the type of 'TypeGroup' (line 308)
        TypeGroup_14991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 308)
        init___14992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 8), TypeGroup_14991, '__init__')
        # Calling __init__(args, kwargs) (line 308)
        init___call_result_14996 = invoke(stypy.reporting.localization.Localization(__file__, 308, 8), init___14992, *[self_14993, list_14994], **kwargs_14995)
        
        
        # Assigning a Subscript to a Attribute (line 309):
        
        # Assigning a Subscript to a Attribute (line 309):
        
        # Obtaining the type of the subscript
        int_14997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 41), 'int')
        # Getting the type of 'param_number' (line 309)
        param_number_14998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 28), 'param_number')
        # Obtaining the member '__getitem__' of a type (line 309)
        getitem___14999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 28), param_number_14998, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 309)
        subscript_call_result_15000 = invoke(stypy.reporting.localization.Localization(__file__, 309, 28), getitem___14999, int_14997)
        
        # Getting the type of 'self' (line 309)
        self_15001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'self')
        # Setting the type of the member 'param_number' of a type (line 309)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 8), self_15001, 'param_number', subscript_call_result_15000)
        
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
        False_15002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'stypy_return_type', False_15002)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 311)
        stypy_return_type_15003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15003)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_15003


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
        self_15005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 23), 'self', False)
        # Processing the call keyword arguments (line 315)
        kwargs_15006 = {}
        # Getting the type of 'type' (line 315)
        type_15004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 18), 'type', False)
        # Calling type(args, kwargs) (line 315)
        type_call_result_15007 = invoke(stypy.reporting.localization.Localization(__file__, 315, 18), type_15004, *[self_15005], **kwargs_15006)
        
        # Obtaining the member '__name__' of a type (line 315)
        name___15008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 18), type_call_result_15007, '__name__')
        str_15009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 40), 'str', '(')
        # Applying the binary operator '+' (line 315)
        result_add_15010 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 18), '+', name___15008, str_15009)
        
        # Getting the type of 'self' (line 315)
        self_15011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 46), 'self')
        # Obtaining the member 'param_number' of a type (line 315)
        param_number_15012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 46), self_15011, 'param_number')
        # Applying the binary operator '+' (line 315)
        result_add_15013 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 44), '+', result_add_15010, param_number_15012)
        
        str_15014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 66), 'str', ')')
        # Applying the binary operator '+' (line 315)
        result_add_15015 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 64), '+', result_add_15013, str_15014)
        
        # Assigning a type to the variable 'ret_str' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'ret_str', result_add_15015)
        # Getting the type of 'ret_str' (line 317)
        ret_str_15016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 15), 'ret_str')
        # Assigning a type to the variable 'stypy_return_type' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'stypy_return_type', ret_str_15016)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 314)
        stypy_return_type_15017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15017)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_15017


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
        self_15018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 28), 'self')
        # Obtaining the member 'param_number' of a type (line 320)
        param_number_15019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 28), self_15018, 'param_number')
        int_15020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 48), 'int')
        # Applying the binary operator '-' (line 320)
        result_sub_15021 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 28), '-', param_number_15019, int_15020)
        
        
        # Obtaining the type of the subscript
        int_15022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 25), 'int')
        # Getting the type of 'call_args' (line 320)
        call_args_15023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 15), 'call_args')
        # Obtaining the member '__getitem__' of a type (line 320)
        getitem___15024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 15), call_args_15023, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 320)
        subscript_call_result_15025 = invoke(stypy.reporting.localization.Localization(__file__, 320, 15), getitem___15024, int_15022)
        
        # Obtaining the member '__getitem__' of a type (line 320)
        getitem___15026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 15), subscript_call_result_15025, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 320)
        subscript_call_result_15027 = invoke(stypy.reporting.localization.Localization(__file__, 320, 15), getitem___15026, result_sub_15021)
        
        # Assigning a type to the variable 'stypy_return_type' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'stypy_return_type', subscript_call_result_15027)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 319)
        stypy_return_type_15028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15028)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_15028


# Assigning a type to the variable 'TypeOfParam' (line 299)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 0), 'TypeOfParam', TypeOfParam)
# Declaration of the 'Callable' class
# Getting the type of 'TypeGroup' (line 323)
TypeGroup_15029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 15), 'TypeGroup')

class Callable(TypeGroup_15029, ):
    str_15030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, (-1)), 'str', '\n    Represent all callable objects (those that define the member __call__)\n    ')

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
        self_15033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 329)
        list_15034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 329)
        
        # Processing the call keyword arguments (line 329)
        kwargs_15035 = {}
        # Getting the type of 'TypeGroup' (line 329)
        TypeGroup_15031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 329)
        init___15032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), TypeGroup_15031, '__init__')
        # Calling __init__(args, kwargs) (line 329)
        init___call_result_15036 = invoke(stypy.reporting.localization.Localization(__file__, 329, 8), init___15032, *[self_15033, list_15034], **kwargs_15035)
        
        
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
        None_15039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 51), 'None', False)
        str_15040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 57), 'str', '__call__')
        # Processing the call keyword arguments (line 332)
        kwargs_15041 = {}
        # Getting the type of 'type_' (line 332)
        type__15037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 26), 'type_', False)
        # Obtaining the member 'get_type_of_member' of a type (line 332)
        get_type_of_member_15038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 26), type__15037, 'get_type_of_member')
        # Calling get_type_of_member(args, kwargs) (line 332)
        get_type_of_member_call_result_15042 = invoke(stypy.reporting.localization.Localization(__file__, 332, 26), get_type_of_member_15038, *[None_15039, str_15040], **kwargs_15041)
        
        # Getting the type of 'self' (line 332)
        self_15043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'self')
        # Setting the type of the member 'member_obj' of a type (line 332)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 8), self_15043, 'member_obj', get_type_of_member_call_result_15042)
        
        # Type idiom detected: calculating its left and rigth part (line 333)
        # Getting the type of 'TypeError' (line 333)
        TypeError_15044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 39), 'TypeError')
        # Getting the type of 'self' (line 333)
        self_15045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 22), 'self')
        # Obtaining the member 'member_obj' of a type (line 333)
        member_obj_15046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 22), self_15045, 'member_obj')
        
        (may_be_15047, more_types_in_union_15048) = may_be_subtype(TypeError_15044, member_obj_15046)

        if may_be_15047:

            if more_types_in_union_15048:
                # Runtime conditional SSA (line 333)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 333)
            self_15049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'self')
            # Obtaining the member 'member_obj' of a type (line 333)
            member_obj_15050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 8), self_15049, 'member_obj')
            # Setting the type of the member 'member_obj' of a type (line 333)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 8), self_15049, 'member_obj', remove_not_subtype_from_union(member_obj_15046, TypeError))
            # Getting the type of 'False' (line 334)
            False_15051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 334)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'stypy_return_type', False_15051)

            if more_types_in_union_15048:
                # SSA join for if statement (line 333)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'True' (line 336)
        True_15052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'stypy_return_type', True_15052)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 331)
        stypy_return_type_15053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15053)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_15053


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
        self_15054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 15), 'self')
        # Obtaining the member 'member_obj' of a type (line 339)
        member_obj_15055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 15), self_15054, 'member_obj')
        # Assigning a type to the variable 'temp' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'temp', member_obj_15055)
        
        # Assigning a Name to a Attribute (line 340):
        
        # Assigning a Name to a Attribute (line 340):
        # Getting the type of 'None' (line 340)
        None_15056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 26), 'None')
        # Getting the type of 'self' (line 340)
        self_15057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'self')
        # Setting the type of the member 'member_obj' of a type (line 340)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), self_15057, 'member_obj', None_15056)
        # Getting the type of 'temp' (line 342)
        temp_15058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'stypy_return_type', temp_15058)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 338)
        stypy_return_type_15059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15059)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_15059


# Assigning a type to the variable 'Callable' (line 323)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 0), 'Callable', Callable)
# Declaration of the 'TypeObject' class
# Getting the type of 'TypeGroup' (line 345)
TypeGroup_15060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 17), 'TypeGroup')

class TypeObject(TypeGroup_15060, ):
    str_15061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, (-1)), 'str', '\n    Represent type and types.ClassType types\n    ')
    
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
        self_15064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 352)
        list_15065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 352)
        
        # Processing the call keyword arguments (line 352)
        kwargs_15066 = {}
        # Getting the type of 'TypeGroup' (line 352)
        TypeGroup_15062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 352)
        init___15063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 8), TypeGroup_15062, '__init__')
        # Calling __init__(args, kwargs) (line 352)
        init___call_result_15067 = invoke(stypy.reporting.localization.Localization(__file__, 352, 8), init___15063, *[self_15064, list_15065], **kwargs_15066)
        
        
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
        kwargs_15071 = {}
        # Getting the type of 'type_' (line 355)
        type__15069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 31), 'type_', False)
        # Obtaining the member 'get_python_type' of a type (line 355)
        get_python_type_15070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 31), type__15069, 'get_python_type')
        # Calling get_python_type(args, kwargs) (line 355)
        get_python_type_call_result_15072 = invoke(stypy.reporting.localization.Localization(__file__, 355, 31), get_python_type_15070, *[], **kwargs_15071)
        
        # Processing the call keyword arguments (line 355)
        kwargs_15073 = {}
        # Getting the type of 'type' (line 355)
        type_15068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 26), 'type', False)
        # Calling type(args, kwargs) (line 355)
        type_call_result_15074 = invoke(stypy.reporting.localization.Localization(__file__, 355, 26), type_15068, *[get_python_type_call_result_15072], **kwargs_15073)
        
        # Getting the type of 'self' (line 355)
        self_15075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'self')
        # Setting the type of the member 'member_obj' of a type (line 355)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 8), self_15075, 'member_obj', type_call_result_15074)
        
        # Getting the type of 'self' (line 356)
        self_15076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 11), 'self')
        # Obtaining the member 'member_obj' of a type (line 356)
        member_obj_15077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 11), self_15076, 'member_obj')
        # Getting the type of 'TypeObject' (line 356)
        TypeObject_15078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 30), 'TypeObject')
        # Obtaining the member 'type_objs' of a type (line 356)
        type_objs_15079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 30), TypeObject_15078, 'type_objs')
        # Applying the binary operator 'in' (line 356)
        result_contains_15080 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 11), 'in', member_obj_15077, type_objs_15079)
        
        # Testing if the type of an if condition is none (line 356)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 356, 8), result_contains_15080):
            pass
        else:
            
            # Testing the type of an if condition (line 356)
            if_condition_15081 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 356, 8), result_contains_15080)
            # Assigning a type to the variable 'if_condition_15081' (line 356)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'if_condition_15081', if_condition_15081)
            # SSA begins for if statement (line 356)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to is_type_instance(...): (line 357)
            # Processing the call keyword arguments (line 357)
            kwargs_15084 = {}
            # Getting the type of 'type_' (line 357)
            type__15082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 23), 'type_', False)
            # Obtaining the member 'is_type_instance' of a type (line 357)
            is_type_instance_15083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 23), type__15082, 'is_type_instance')
            # Calling is_type_instance(args, kwargs) (line 357)
            is_type_instance_call_result_15085 = invoke(stypy.reporting.localization.Localization(__file__, 357, 23), is_type_instance_15083, *[], **kwargs_15084)
            
            # Applying the 'not' unary operator (line 357)
            result_not__15086 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 19), 'not', is_type_instance_call_result_15085)
            
            # Assigning a type to the variable 'stypy_return_type' (line 357)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'stypy_return_type', result_not__15086)
            # SSA join for if statement (line 356)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'False' (line 359)
        False_15087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'stypy_return_type', False_15087)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 354)
        stypy_return_type_15088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15088)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_15088


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
        self_15089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 15), 'self')
        # Obtaining the member 'member_obj' of a type (line 362)
        member_obj_15090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 15), self_15089, 'member_obj')
        # Assigning a type to the variable 'temp' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'temp', member_obj_15090)
        
        # Assigning a Name to a Attribute (line 363):
        
        # Assigning a Name to a Attribute (line 363):
        # Getting the type of 'None' (line 363)
        None_15091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 26), 'None')
        # Getting the type of 'self' (line 363)
        self_15092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'self')
        # Setting the type of the member 'member_obj' of a type (line 363)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 8), self_15092, 'member_obj', None_15091)
        # Getting the type of 'temp' (line 365)
        temp_15093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'stypy_return_type', temp_15093)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 361)
        stypy_return_type_15094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15094)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_15094


# Assigning a type to the variable 'TypeObject' (line 345)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 0), 'TypeObject', TypeObject)

# Assigning a List to a Name (line 349):

# Obtaining an instance of the builtin type 'list' (line 349)
list_15095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 349)
# Adding element type (line 349)
# Getting the type of 'type' (line 349)
type_15096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 17), 'type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 16), list_15095, type_15096)
# Adding element type (line 349)
# Getting the type of 'types' (line 349)
types_15097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 23), 'types')
# Obtaining the member 'ClassType' of a type (line 349)
ClassType_15098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 23), types_15097, 'ClassType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 16), list_15095, ClassType_15098)

# Getting the type of 'TypeObject'
TypeObject_15099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeObject')
# Setting the type of the member 'type_objs' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeObject_15099, 'type_objs', list_15095)
# Declaration of the 'InstanceOfType' class
# Getting the type of 'TypeGroup' (line 368)
TypeGroup_15100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 21), 'TypeGroup')

class InstanceOfType(TypeGroup_15100, ):
    str_15101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, (-1)), 'str', '\n    Represent type and types.ClassType types\n    ')
    
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
        self_15104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 375)
        list_15105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 375)
        
        # Processing the call keyword arguments (line 375)
        kwargs_15106 = {}
        # Getting the type of 'TypeGroup' (line 375)
        TypeGroup_15102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 375)
        init___15103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 8), TypeGroup_15102, '__init__')
        # Calling __init__(args, kwargs) (line 375)
        init___call_result_15107 = invoke(stypy.reporting.localization.Localization(__file__, 375, 8), init___15103, *[self_15104, list_15105], **kwargs_15106)
        
        
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
        kwargs_15111 = {}
        # Getting the type of 'type_' (line 378)
        type__15109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 31), 'type_', False)
        # Obtaining the member 'get_python_type' of a type (line 378)
        get_python_type_15110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 31), type__15109, 'get_python_type')
        # Calling get_python_type(args, kwargs) (line 378)
        get_python_type_call_result_15112 = invoke(stypy.reporting.localization.Localization(__file__, 378, 31), get_python_type_15110, *[], **kwargs_15111)
        
        # Processing the call keyword arguments (line 378)
        kwargs_15113 = {}
        # Getting the type of 'type' (line 378)
        type_15108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 26), 'type', False)
        # Calling type(args, kwargs) (line 378)
        type_call_result_15114 = invoke(stypy.reporting.localization.Localization(__file__, 378, 26), type_15108, *[get_python_type_call_result_15112], **kwargs_15113)
        
        # Getting the type of 'self' (line 378)
        self_15115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'self')
        # Setting the type of the member 'member_obj' of a type (line 378)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 8), self_15115, 'member_obj', type_call_result_15114)
        
        # Getting the type of 'self' (line 379)
        self_15116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 11), 'self')
        # Obtaining the member 'member_obj' of a type (line 379)
        member_obj_15117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 11), self_15116, 'member_obj')
        # Getting the type of 'TypeObject' (line 379)
        TypeObject_15118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 30), 'TypeObject')
        # Obtaining the member 'type_objs' of a type (line 379)
        type_objs_15119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 30), TypeObject_15118, 'type_objs')
        # Applying the binary operator 'in' (line 379)
        result_contains_15120 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 11), 'in', member_obj_15117, type_objs_15119)
        
        # Testing if the type of an if condition is none (line 379)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 379, 8), result_contains_15120):
            pass
        else:
            
            # Testing the type of an if condition (line 379)
            if_condition_15121 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 379, 8), result_contains_15120)
            # Assigning a type to the variable 'if_condition_15121' (line 379)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'if_condition_15121', if_condition_15121)
            # SSA begins for if statement (line 379)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to is_type_instance(...): (line 380)
            # Processing the call keyword arguments (line 380)
            kwargs_15124 = {}
            # Getting the type of 'type_' (line 380)
            type__15122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 19), 'type_', False)
            # Obtaining the member 'is_type_instance' of a type (line 380)
            is_type_instance_15123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 19), type__15122, 'is_type_instance')
            # Calling is_type_instance(args, kwargs) (line 380)
            is_type_instance_call_result_15125 = invoke(stypy.reporting.localization.Localization(__file__, 380, 19), is_type_instance_15123, *[], **kwargs_15124)
            
            # Assigning a type to the variable 'stypy_return_type' (line 380)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'stypy_return_type', is_type_instance_call_result_15125)
            # SSA join for if statement (line 379)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'False' (line 382)
        False_15126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'stypy_return_type', False_15126)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 377)
        stypy_return_type_15127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15127)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_15127


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
        self_15128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 15), 'self')
        # Obtaining the member 'member_obj' of a type (line 385)
        member_obj_15129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 15), self_15128, 'member_obj')
        # Assigning a type to the variable 'temp' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'temp', member_obj_15129)
        
        # Assigning a Name to a Attribute (line 386):
        
        # Assigning a Name to a Attribute (line 386):
        # Getting the type of 'None' (line 386)
        None_15130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 26), 'None')
        # Getting the type of 'self' (line 386)
        self_15131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'self')
        # Setting the type of the member 'member_obj' of a type (line 386)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 8), self_15131, 'member_obj', None_15130)
        # Getting the type of 'temp' (line 388)
        temp_15132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'stypy_return_type', temp_15132)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 384)
        stypy_return_type_15133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15133)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_15133


# Assigning a type to the variable 'InstanceOfType' (line 368)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 0), 'InstanceOfType', InstanceOfType)

# Assigning a List to a Name (line 372):

# Obtaining an instance of the builtin type 'list' (line 372)
list_15134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 372)
# Adding element type (line 372)
# Getting the type of 'type' (line 372)
type_15135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 17), 'type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 16), list_15134, type_15135)
# Adding element type (line 372)
# Getting the type of 'types' (line 372)
types_15136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 23), 'types')
# Obtaining the member 'ClassType' of a type (line 372)
ClassType_15137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 23), types_15136, 'ClassType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 16), list_15134, ClassType_15137)

# Getting the type of 'InstanceOfType'
InstanceOfType_15138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'InstanceOfType')
# Setting the type of the member 'type_objs' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), InstanceOfType_15138, 'type_objs', list_15134)
# Declaration of the 'VarArgType' class
# Getting the type of 'TypeGroup' (line 391)
TypeGroup_15139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 17), 'TypeGroup')

class VarArgType(TypeGroup_15139, ):
    str_15140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, (-1)), 'str', '\n    Special type group indicating that a callable has an unlimited amount of parameters\n    ')

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
        self_15143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 397)
        list_15144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 397)
        
        # Processing the call keyword arguments (line 397)
        kwargs_15145 = {}
        # Getting the type of 'TypeGroup' (line 397)
        TypeGroup_15141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 397)
        init___15142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 8), TypeGroup_15141, '__init__')
        # Calling __init__(args, kwargs) (line 397)
        init___call_result_15146 = invoke(stypy.reporting.localization.Localization(__file__, 397, 8), init___15142, *[self_15143, list_15144], **kwargs_15145)
        
        
        # Assigning a Name to a Attribute (line 398):
        
        # Assigning a Name to a Attribute (line 398):
        # Getting the type of 'types_' (line 398)
        types__15147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 21), 'types_')
        # Getting the type of 'self' (line 398)
        self_15148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'self')
        # Setting the type of the member 'types' of a type (line 398)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), self_15148, 'types', types__15147)
        
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
        True_15149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'stypy_return_type', True_15149)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 400)
        stypy_return_type_15150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15150)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_15150


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
        self_15151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 15), 'self')
        # Obtaining the member 'type_' of a type (line 404)
        type__15152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 15), self_15151, 'type_')
        # Assigning a type to the variable 'temp' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'temp', type__15152)
        
        # Assigning a Name to a Attribute (line 405):
        
        # Assigning a Name to a Attribute (line 405):
        # Getting the type of 'None' (line 405)
        None_15153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 21), 'None')
        # Getting the type of 'self' (line 405)
        self_15154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'self')
        # Setting the type of the member 'type_' of a type (line 405)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 8), self_15154, 'type_', None_15153)
        # Getting the type of 'temp' (line 407)
        temp_15155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'stypy_return_type', temp_15155)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 403)
        stypy_return_type_15156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15156)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_15156


# Assigning a type to the variable 'VarArgType' (line 391)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 0), 'VarArgType', VarArgType)
# Declaration of the 'TypeGroups' class

class TypeGroups:
    str_15157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, (-1)), 'str', '\n    Class to hold definitions of type groups that are composed by lists of known Python types\n    ')

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

        str_15158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, (-1)), 'str', '\n        Obtain all the types defined in this class\n        ')

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
            element_15160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 30), 'element', False)
            # Getting the type of 'list' (line 425)
            list_15161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 39), 'list', False)
            # Processing the call keyword arguments (line 425)
            kwargs_15162 = {}
            # Getting the type of 'isinstance' (line 425)
            isinstance_15159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 425)
            isinstance_call_result_15163 = invoke(stypy.reporting.localization.Localization(__file__, 425, 19), isinstance_15159, *[element_15160, list_15161], **kwargs_15162)
            
            # Assigning a type to the variable 'stypy_return_type' (line 425)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'stypy_return_type', isinstance_call_result_15163)
            
            # ################# End of 'filter_func(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'filter_func' in the type store
            # Getting the type of 'stypy_return_type' (line 424)
            stypy_return_type_15164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_15164)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'filter_func'
            return stypy_return_type_15164

        # Assigning a type to the variable 'filter_func' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'filter_func', filter_func)
        
        # Call to filter(...): (line 427)
        # Processing the call arguments (line 427)

        @norecursion
        def _stypy_temp_lambda_25(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_25'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_25', 427, 22, True)
            # Passed parameters checking function
            _stypy_temp_lambda_25.stypy_localization = localization
            _stypy_temp_lambda_25.stypy_type_of_self = None
            _stypy_temp_lambda_25.stypy_type_store = module_type_store
            _stypy_temp_lambda_25.stypy_function_name = '_stypy_temp_lambda_25'
            _stypy_temp_lambda_25.stypy_param_names_list = ['member']
            _stypy_temp_lambda_25.stypy_varargs_param_name = None
            _stypy_temp_lambda_25.stypy_kwargs_param_name = None
            _stypy_temp_lambda_25.stypy_call_defaults = defaults
            _stypy_temp_lambda_25.stypy_call_varargs = varargs
            _stypy_temp_lambda_25.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_25', ['member'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_25', ['member'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to filter_func(...): (line 427)
            # Processing the call arguments (line 427)
            
            # Call to getattr(...): (line 427)
            # Processing the call arguments (line 427)
            # Getting the type of 'TypeGroups' (line 427)
            TypeGroups_15168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 57), 'TypeGroups', False)
            # Getting the type of 'member' (line 427)
            member_15169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 69), 'member', False)
            # Processing the call keyword arguments (line 427)
            kwargs_15170 = {}
            # Getting the type of 'getattr' (line 427)
            getattr_15167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 49), 'getattr', False)
            # Calling getattr(args, kwargs) (line 427)
            getattr_call_result_15171 = invoke(stypy.reporting.localization.Localization(__file__, 427, 49), getattr_15167, *[TypeGroups_15168, member_15169], **kwargs_15170)
            
            # Processing the call keyword arguments (line 427)
            kwargs_15172 = {}
            # Getting the type of 'filter_func' (line 427)
            filter_func_15166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 37), 'filter_func', False)
            # Calling filter_func(args, kwargs) (line 427)
            filter_func_call_result_15173 = invoke(stypy.reporting.localization.Localization(__file__, 427, 37), filter_func_15166, *[getattr_call_result_15171], **kwargs_15172)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 427)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 22), 'stypy_return_type', filter_func_call_result_15173)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_25' in the type store
            # Getting the type of 'stypy_return_type' (line 427)
            stypy_return_type_15174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 22), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_15174)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_25'
            return stypy_return_type_15174

        # Assigning a type to the variable '_stypy_temp_lambda_25' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 22), '_stypy_temp_lambda_25', _stypy_temp_lambda_25)
        # Getting the type of '_stypy_temp_lambda_25' (line 427)
        _stypy_temp_lambda_25_15175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 22), '_stypy_temp_lambda_25')
        # Getting the type of 'TypeGroups' (line 427)
        TypeGroups_15176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 79), 'TypeGroups', False)
        # Obtaining the member '__dict__' of a type (line 427)
        dict___15177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 79), TypeGroups_15176, '__dict__')
        # Processing the call keyword arguments (line 427)
        kwargs_15178 = {}
        # Getting the type of 'filter' (line 427)
        filter_15165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 15), 'filter', False)
        # Calling filter(args, kwargs) (line 427)
        filter_call_result_15179 = invoke(stypy.reporting.localization.Localization(__file__, 427, 15), filter_15165, *[_stypy_temp_lambda_25_15175, dict___15177], **kwargs_15178)
        
        # Assigning a type to the variable 'stypy_return_type' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'stypy_return_type', filter_call_result_15179)
        
        # ################# End of 'get_rule_groups(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_rule_groups' in the type store
        # Getting the type of 'stypy_return_type' (line 418)
        stypy_return_type_15180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15180)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_rule_groups'
        return stypy_return_type_15180

    
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
list_15181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 430)
# Adding element type (line 430)
# Getting the type of 'int' (line 430)
int_15182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 17), list_15181, int_15182)
# Adding element type (line 430)
# Getting the type of 'long' (line 430)
long_15183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 23), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 17), list_15181, long_15183)
# Adding element type (line 430)
# Getting the type of 'float' (line 430)
float_15184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 29), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 17), list_15181, float_15184)
# Adding element type (line 430)
# Getting the type of 'bool' (line 430)
bool_15185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 36), 'bool')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 17), list_15181, bool_15185)

# Getting the type of 'TypeGroups'
TypeGroups_15186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeGroups')
# Setting the type of the member 'RealNumber' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeGroups_15186, 'RealNumber', list_15181)

# Assigning a List to a Name (line 433):

# Obtaining an instance of the builtin type 'list' (line 433)
list_15187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 433)
# Adding element type (line 433)
# Getting the type of 'int' (line 433)
int_15188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 13), list_15187, int_15188)
# Adding element type (line 433)
# Getting the type of 'long' (line 433)
long_15189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 19), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 13), list_15187, long_15189)
# Adding element type (line 433)
# Getting the type of 'float' (line 433)
float_15190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 25), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 13), list_15187, float_15190)
# Adding element type (line 433)
# Getting the type of 'bool' (line 433)
bool_15191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 32), 'bool')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 13), list_15187, bool_15191)
# Adding element type (line 433)
# Getting the type of 'complex' (line 433)
complex_15192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 38), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 13), list_15187, complex_15192)

# Getting the type of 'TypeGroups'
TypeGroups_15193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeGroups')
# Setting the type of the member 'Number' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeGroups_15193, 'Number', list_15187)

# Assigning a List to a Name (line 436):

# Obtaining an instance of the builtin type 'list' (line 436)
list_15194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 436)
# Adding element type (line 436)
# Getting the type of 'int' (line 436)
int_15195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 14), list_15194, int_15195)
# Adding element type (line 436)
# Getting the type of 'long' (line 436)
long_15196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 20), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 14), list_15194, long_15196)
# Adding element type (line 436)
# Getting the type of 'bool' (line 436)
bool_15197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 26), 'bool')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 14), list_15194, bool_15197)

# Getting the type of 'TypeGroups'
TypeGroups_15198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeGroups')
# Setting the type of the member 'Integer' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeGroups_15198, 'Integer', list_15194)

# Assigning a List to a Name (line 439):

# Obtaining an instance of the builtin type 'list' (line 439)
list_15199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 439)
# Adding element type (line 439)
# Getting the type of 'str' (line 439)
str_15200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 11), 'str')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 10), list_15199, str_15200)
# Adding element type (line 439)
# Getting the type of 'unicode' (line 439)
unicode_15201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 16), 'unicode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 10), list_15199, unicode_15201)
# Adding element type (line 439)
# Getting the type of 'buffer' (line 439)
buffer_15202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 25), 'buffer')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 10), list_15199, buffer_15202)

# Getting the type of 'TypeGroups'
TypeGroups_15203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeGroups')
# Setting the type of the member 'Str' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeGroups_15203, 'Str', list_15199)

# Assigning a List to a Name (line 442):

# Obtaining an instance of the builtin type 'list' (line 442)
list_15204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 442)
# Adding element type (line 442)
# Getting the type of 'buffer' (line 442)
buffer_15205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 20), 'buffer')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 19), list_15204, buffer_15205)
# Adding element type (line 442)
# Getting the type of 'bytearray' (line 442)
bytearray_15206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 28), 'bytearray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 19), list_15204, bytearray_15206)
# Adding element type (line 442)
# Getting the type of 'str' (line 442)
str_15207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 39), 'str')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 19), list_15204, str_15207)
# Adding element type (line 442)
# Getting the type of 'memoryview' (line 442)
memoryview_15208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 44), 'memoryview')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 19), list_15204, memoryview_15208)

# Getting the type of 'TypeGroups'
TypeGroups_15209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeGroups')
# Setting the type of the member 'ByteSequence' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeGroups_15209, 'ByteSequence', list_15204)

# Assigning a List to a Name (line 445):

# Obtaining an instance of the builtin type 'list' (line 445)
list_15210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 445)
# Adding element type (line 445)
# Getting the type of 'list' (line 446)
list_15211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'list')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_15210, list_15211)
# Adding element type (line 445)
# Getting the type of 'dict' (line 447)
dict_15212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'dict')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_15210, dict_15212)
# Adding element type (line 445)
# Getting the type of 'ExtraTypeDefinitions' (line 448)
ExtraTypeDefinitions_15213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'tupleiterator' of a type (line 448)
tupleiterator_15214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 8), ExtraTypeDefinitions_15213, 'tupleiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_15210, tupleiterator_15214)
# Adding element type (line 445)
# Getting the type of 'ExtraTypeDefinitions' (line 449)
ExtraTypeDefinitions_15215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'dict_values' of a type (line 449)
dict_values_15216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 8), ExtraTypeDefinitions_15215, 'dict_values')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_15210, dict_values_15216)
# Adding element type (line 445)
# Getting the type of 'frozenset' (line 450)
frozenset_15217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'frozenset')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_15210, frozenset_15217)
# Adding element type (line 445)
# Getting the type of 'ExtraTypeDefinitions' (line 451)
ExtraTypeDefinitions_15218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'rangeiterator' of a type (line 451)
rangeiterator_15219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 8), ExtraTypeDefinitions_15218, 'rangeiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_15210, rangeiterator_15219)
# Adding element type (line 445)
# Getting the type of 'types' (line 452)
types_15220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'types')
# Obtaining the member 'GeneratorType' of a type (line 452)
GeneratorType_15221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 8), types_15220, 'GeneratorType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_15210, GeneratorType_15221)
# Adding element type (line 445)
# Getting the type of 'enumerate' (line 453)
enumerate_15222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'enumerate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_15210, enumerate_15222)
# Adding element type (line 445)
# Getting the type of 'bytearray' (line 454)
bytearray_15223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'bytearray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_15210, bytearray_15223)
# Adding element type (line 445)
# Getting the type of 'iter' (line 455)
iter_15224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'iter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_15210, iter_15224)
# Adding element type (line 445)
# Getting the type of 'reversed' (line 456)
reversed_15225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'reversed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_15210, reversed_15225)
# Adding element type (line 445)
# Getting the type of 'ExtraTypeDefinitions' (line 457)
ExtraTypeDefinitions_15226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'dictionary_keyiterator' of a type (line 457)
dictionary_keyiterator_15227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 8), ExtraTypeDefinitions_15226, 'dictionary_keyiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_15210, dictionary_keyiterator_15227)
# Adding element type (line 445)
# Getting the type of 'ExtraTypeDefinitions' (line 458)
ExtraTypeDefinitions_15228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'bytearray_iterator' of a type (line 458)
bytearray_iterator_15229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 8), ExtraTypeDefinitions_15228, 'bytearray_iterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_15210, bytearray_iterator_15229)
# Adding element type (line 445)
# Getting the type of 'ExtraTypeDefinitions' (line 459)
ExtraTypeDefinitions_15230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'dictionary_valueiterator' of a type (line 459)
dictionary_valueiterator_15231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 8), ExtraTypeDefinitions_15230, 'dictionary_valueiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_15210, dictionary_valueiterator_15231)
# Adding element type (line 445)
# Getting the type of 'ExtraTypeDefinitions' (line 460)
ExtraTypeDefinitions_15232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'dictionary_itemiterator' of a type (line 460)
dictionary_itemiterator_15233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 8), ExtraTypeDefinitions_15232, 'dictionary_itemiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_15210, dictionary_itemiterator_15233)
# Adding element type (line 445)
# Getting the type of 'ExtraTypeDefinitions' (line 461)
ExtraTypeDefinitions_15234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'listiterator' of a type (line 461)
listiterator_15235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 8), ExtraTypeDefinitions_15234, 'listiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_15210, listiterator_15235)
# Adding element type (line 445)
# Getting the type of 'ExtraTypeDefinitions' (line 462)
ExtraTypeDefinitions_15236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'listreverseiterator' of a type (line 462)
listreverseiterator_15237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 8), ExtraTypeDefinitions_15236, 'listreverseiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_15210, listreverseiterator_15237)
# Adding element type (line 445)
# Getting the type of 'tuple' (line 463)
tuple_15238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'tuple')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_15210, tuple_15238)
# Adding element type (line 445)
# Getting the type of 'set' (line 464)
set_15239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'set')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_15210, set_15239)
# Adding element type (line 445)
# Getting the type of 'xrange' (line 465)
xrange_15240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'xrange')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), list_15210, xrange_15240)

# Getting the type of 'TypeGroups'
TypeGroups_15241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeGroups')
# Setting the type of the member 'IterableDataStructure' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeGroups_15241, 'IterableDataStructure', list_15210)

# Assigning a List to a Name (line 468):

# Obtaining an instance of the builtin type 'list' (line 468)
list_15242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 468)
# Adding element type (line 468)
# Getting the type of 'list' (line 469)
list_15243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'list')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_15242, list_15243)
# Adding element type (line 468)
# Getting the type of 'dict' (line 470)
dict_15244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'dict')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_15242, dict_15244)
# Adding element type (line 468)
# Getting the type of 'ExtraTypeDefinitions' (line 471)
ExtraTypeDefinitions_15245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'tupleiterator' of a type (line 471)
tupleiterator_15246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 8), ExtraTypeDefinitions_15245, 'tupleiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_15242, tupleiterator_15246)
# Adding element type (line 468)
# Getting the type of 'ExtraTypeDefinitions' (line 472)
ExtraTypeDefinitions_15247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'dict_values' of a type (line 472)
dict_values_15248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 8), ExtraTypeDefinitions_15247, 'dict_values')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_15242, dict_values_15248)
# Adding element type (line 468)
# Getting the type of 'frozenset' (line 473)
frozenset_15249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'frozenset')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_15242, frozenset_15249)
# Adding element type (line 468)
# Getting the type of 'ExtraTypeDefinitions' (line 474)
ExtraTypeDefinitions_15250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'rangeiterator' of a type (line 474)
rangeiterator_15251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 8), ExtraTypeDefinitions_15250, 'rangeiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_15242, rangeiterator_15251)
# Adding element type (line 468)
# Getting the type of 'types' (line 475)
types_15252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'types')
# Obtaining the member 'GeneratorType' of a type (line 475)
GeneratorType_15253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 8), types_15252, 'GeneratorType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_15242, GeneratorType_15253)
# Adding element type (line 468)
# Getting the type of 'enumerate' (line 476)
enumerate_15254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'enumerate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_15242, enumerate_15254)
# Adding element type (line 468)
# Getting the type of 'bytearray' (line 477)
bytearray_15255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'bytearray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_15242, bytearray_15255)
# Adding element type (line 468)
# Getting the type of 'iter' (line 478)
iter_15256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'iter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_15242, iter_15256)
# Adding element type (line 468)
# Getting the type of 'reversed' (line 479)
reversed_15257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'reversed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_15242, reversed_15257)
# Adding element type (line 468)
# Getting the type of 'ExtraTypeDefinitions' (line 480)
ExtraTypeDefinitions_15258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'dictionary_keyiterator' of a type (line 480)
dictionary_keyiterator_15259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 8), ExtraTypeDefinitions_15258, 'dictionary_keyiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_15242, dictionary_keyiterator_15259)
# Adding element type (line 468)
# Getting the type of 'ExtraTypeDefinitions' (line 481)
ExtraTypeDefinitions_15260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'bytearray_iterator' of a type (line 481)
bytearray_iterator_15261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 8), ExtraTypeDefinitions_15260, 'bytearray_iterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_15242, bytearray_iterator_15261)
# Adding element type (line 468)
# Getting the type of 'ExtraTypeDefinitions' (line 482)
ExtraTypeDefinitions_15262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'dictionary_valueiterator' of a type (line 482)
dictionary_valueiterator_15263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 8), ExtraTypeDefinitions_15262, 'dictionary_valueiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_15242, dictionary_valueiterator_15263)
# Adding element type (line 468)
# Getting the type of 'ExtraTypeDefinitions' (line 483)
ExtraTypeDefinitions_15264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'dictionary_itemiterator' of a type (line 483)
dictionary_itemiterator_15265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 8), ExtraTypeDefinitions_15264, 'dictionary_itemiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_15242, dictionary_itemiterator_15265)
# Adding element type (line 468)
# Getting the type of 'ExtraTypeDefinitions' (line 484)
ExtraTypeDefinitions_15266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'listiterator' of a type (line 484)
listiterator_15267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 8), ExtraTypeDefinitions_15266, 'listiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_15242, listiterator_15267)
# Adding element type (line 468)
# Getting the type of 'ExtraTypeDefinitions' (line 485)
ExtraTypeDefinitions_15268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'ExtraTypeDefinitions')
# Obtaining the member 'listreverseiterator' of a type (line 485)
listreverseiterator_15269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 8), ExtraTypeDefinitions_15268, 'listreverseiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_15242, listreverseiterator_15269)
# Adding element type (line 468)
# Getting the type of 'tuple' (line 486)
tuple_15270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'tuple')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_15242, tuple_15270)
# Adding element type (line 468)
# Getting the type of 'set' (line 487)
set_15271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'set')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_15242, set_15271)
# Adding element type (line 468)
# Getting the type of 'xrange' (line 488)
xrange_15272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'xrange')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_15242, xrange_15272)
# Adding element type (line 468)
# Getting the type of 'memoryview' (line 489)
memoryview_15273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'memoryview')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_15242, memoryview_15273)
# Adding element type (line 468)
# Getting the type of 'types' (line 490)
types_15274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'types')
# Obtaining the member 'DictProxyType' of a type (line 490)
DictProxyType_15275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 8), types_15274, 'DictProxyType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_15242, DictProxyType_15275)

# Getting the type of 'TypeGroups'
TypeGroups_15276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeGroups')
# Setting the type of the member 'IterableObject' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeGroups_15276, 'IterableObject', list_15242)
str_15277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, (-1)), 'str', '\nInstances of type groups. These are the ones that are really used in the type rules, as are concrete usages\nof the previously defined type groups.\n\nNOTE: To interpret instances of type groups, you should take into account the following:\n\n- UndefinedType as expected return type: We cannot statically determine the return\ntype of this method. So we obtain it calling the member, obtaining its type\nand reevaluating the member ruleset again with this type substituting the dependent\none.\n\n- DynamicType as expected return type: We also cannot statically determine the return\ntype of this method. But this time we directly return the return type of the invoked\nmember.\n')

# Assigning a Call to a Name (line 510):

# Assigning a Call to a Name (line 510):

# Call to HasMember(...): (line 510)
# Processing the call arguments (line 510)
str_15279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 23), 'str', '__int__')
# Getting the type of 'int' (line 510)
int_15280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 34), 'int', False)
int_15281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 39), 'int')
# Processing the call keyword arguments (line 510)
kwargs_15282 = {}
# Getting the type of 'HasMember' (line 510)
HasMember_15278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 13), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 510)
HasMember_call_result_15283 = invoke(stypy.reporting.localization.Localization(__file__, 510, 13), HasMember_15278, *[str_15279, int_15280, int_15281], **kwargs_15282)

# Assigning a type to the variable 'CastsToInt' (line 510)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 0), 'CastsToInt', HasMember_call_result_15283)

# Assigning a Call to a Name (line 511):

# Assigning a Call to a Name (line 511):

# Call to HasMember(...): (line 511)
# Processing the call arguments (line 511)
str_15285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 24), 'str', '__long__')
# Getting the type of 'long' (line 511)
long_15286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 36), 'long', False)
int_15287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 42), 'int')
# Processing the call keyword arguments (line 511)
kwargs_15288 = {}
# Getting the type of 'HasMember' (line 511)
HasMember_15284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 14), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 511)
HasMember_call_result_15289 = invoke(stypy.reporting.localization.Localization(__file__, 511, 14), HasMember_15284, *[str_15285, long_15286, int_15287], **kwargs_15288)

# Assigning a type to the variable 'CastsToLong' (line 511)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 0), 'CastsToLong', HasMember_call_result_15289)

# Assigning a Call to a Name (line 512):

# Assigning a Call to a Name (line 512):

# Call to HasMember(...): (line 512)
# Processing the call arguments (line 512)
str_15291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 25), 'str', '__float__')
# Getting the type of 'float' (line 512)
float_15292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 38), 'float', False)
int_15293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 45), 'int')
# Processing the call keyword arguments (line 512)
kwargs_15294 = {}
# Getting the type of 'HasMember' (line 512)
HasMember_15290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 15), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 512)
HasMember_call_result_15295 = invoke(stypy.reporting.localization.Localization(__file__, 512, 15), HasMember_15290, *[str_15291, float_15292, int_15293], **kwargs_15294)

# Assigning a type to the variable 'CastsToFloat' (line 512)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 0), 'CastsToFloat', HasMember_call_result_15295)

# Assigning a Call to a Name (line 513):

# Assigning a Call to a Name (line 513):

# Call to HasMember(...): (line 513)
# Processing the call arguments (line 513)
str_15297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 27), 'str', '__complex__')
# Getting the type of 'complex' (line 513)
complex_15298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 42), 'complex', False)
int_15299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 51), 'int')
# Processing the call keyword arguments (line 513)
kwargs_15300 = {}
# Getting the type of 'HasMember' (line 513)
HasMember_15296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 17), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 513)
HasMember_call_result_15301 = invoke(stypy.reporting.localization.Localization(__file__, 513, 17), HasMember_15296, *[str_15297, complex_15298, int_15299], **kwargs_15300)

# Assigning a type to the variable 'CastsToComplex' (line 513)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 0), 'CastsToComplex', HasMember_call_result_15301)

# Assigning a Call to a Name (line 514):

# Assigning a Call to a Name (line 514):

# Call to HasMember(...): (line 514)
# Processing the call arguments (line 514)
str_15303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 23), 'str', '__oct__')
# Getting the type of 'str' (line 514)
str_15304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 34), 'str', False)
int_15305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 39), 'int')
# Processing the call keyword arguments (line 514)
kwargs_15306 = {}
# Getting the type of 'HasMember' (line 514)
HasMember_15302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 13), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 514)
HasMember_call_result_15307 = invoke(stypy.reporting.localization.Localization(__file__, 514, 13), HasMember_15302, *[str_15303, str_15304, int_15305], **kwargs_15306)

# Assigning a type to the variable 'CastsToOct' (line 514)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 0), 'CastsToOct', HasMember_call_result_15307)

# Assigning a Call to a Name (line 515):

# Assigning a Call to a Name (line 515):

# Call to HasMember(...): (line 515)
# Processing the call arguments (line 515)
str_15309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 23), 'str', '__hex__')
# Getting the type of 'str' (line 515)
str_15310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 34), 'str', False)
int_15311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 39), 'int')
# Processing the call keyword arguments (line 515)
kwargs_15312 = {}
# Getting the type of 'HasMember' (line 515)
HasMember_15308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 13), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 515)
HasMember_call_result_15313 = invoke(stypy.reporting.localization.Localization(__file__, 515, 13), HasMember_15308, *[str_15309, str_15310, int_15311], **kwargs_15312)

# Assigning a type to the variable 'CastsToHex' (line 515)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 0), 'CastsToHex', HasMember_call_result_15313)

# Assigning a Call to a Name (line 516):

# Assigning a Call to a Name (line 516):

# Call to HasMember(...): (line 516)
# Processing the call arguments (line 516)
str_15315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 25), 'str', '__index__')
# Getting the type of 'int' (line 516)
int_15316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 38), 'int', False)
int_15317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 43), 'int')
# Processing the call keyword arguments (line 516)
kwargs_15318 = {}
# Getting the type of 'HasMember' (line 516)
HasMember_15314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 15), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 516)
HasMember_call_result_15319 = invoke(stypy.reporting.localization.Localization(__file__, 516, 15), HasMember_15314, *[str_15315, int_15316, int_15317], **kwargs_15318)

# Assigning a type to the variable 'CastsToIndex' (line 516)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 0), 'CastsToIndex', HasMember_call_result_15319)

# Assigning a Call to a Name (line 517):

# Assigning a Call to a Name (line 517):

# Call to HasMember(...): (line 517)
# Processing the call arguments (line 517)
str_15321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 25), 'str', '__trunc__')
# Getting the type of 'UndefinedType' (line 517)
UndefinedType_15322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 38), 'UndefinedType', False)
int_15323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 53), 'int')
# Processing the call keyword arguments (line 517)
kwargs_15324 = {}
# Getting the type of 'HasMember' (line 517)
HasMember_15320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 15), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 517)
HasMember_call_result_15325 = invoke(stypy.reporting.localization.Localization(__file__, 517, 15), HasMember_15320, *[str_15321, UndefinedType_15322, int_15323], **kwargs_15324)

# Assigning a type to the variable 'CastsToTrunc' (line 517)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 0), 'CastsToTrunc', HasMember_call_result_15325)

# Assigning a Call to a Name (line 518):

# Assigning a Call to a Name (line 518):

# Call to HasMember(...): (line 518)
# Processing the call arguments (line 518)
str_15327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 26), 'str', '__coerce__')
# Getting the type of 'UndefinedType' (line 518)
UndefinedType_15328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 40), 'UndefinedType', False)
int_15329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 55), 'int')
# Processing the call keyword arguments (line 518)
kwargs_15330 = {}
# Getting the type of 'HasMember' (line 518)
HasMember_15326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 16), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 518)
HasMember_call_result_15331 = invoke(stypy.reporting.localization.Localization(__file__, 518, 16), HasMember_15326, *[str_15327, UndefinedType_15328, int_15329], **kwargs_15330)

# Assigning a type to the variable 'CastsToCoerce' (line 518)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 0), 'CastsToCoerce', HasMember_call_result_15331)

# Assigning a Call to a Name (line 523):

# Assigning a Call to a Name (line 523):

# Call to HasMember(...): (line 523)
# Processing the call arguments (line 523)
str_15333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 29), 'str', '__cmp__')
# Getting the type of 'DynamicType' (line 523)
DynamicType_15334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 40), 'DynamicType', False)
int_15335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 53), 'int')
# Processing the call keyword arguments (line 523)
kwargs_15336 = {}
# Getting the type of 'HasMember' (line 523)
HasMember_15332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 523)
HasMember_call_result_15337 = invoke(stypy.reporting.localization.Localization(__file__, 523, 19), HasMember_15332, *[str_15333, DynamicType_15334, int_15335], **kwargs_15336)

# Assigning a type to the variable 'Overloads__cmp__' (line 523)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 0), 'Overloads__cmp__', HasMember_call_result_15337)

# Assigning a Call to a Name (line 524):

# Assigning a Call to a Name (line 524):

# Call to HasMember(...): (line 524)
# Processing the call arguments (line 524)
str_15339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 28), 'str', '__eq__')
# Getting the type of 'DynamicType' (line 524)
DynamicType_15340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 38), 'DynamicType', False)
int_15341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 51), 'int')
# Processing the call keyword arguments (line 524)
kwargs_15342 = {}
# Getting the type of 'HasMember' (line 524)
HasMember_15338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 524)
HasMember_call_result_15343 = invoke(stypy.reporting.localization.Localization(__file__, 524, 18), HasMember_15338, *[str_15339, DynamicType_15340, int_15341], **kwargs_15342)

# Assigning a type to the variable 'Overloads__eq__' (line 524)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 0), 'Overloads__eq__', HasMember_call_result_15343)

# Assigning a Call to a Name (line 525):

# Assigning a Call to a Name (line 525):

# Call to HasMember(...): (line 525)
# Processing the call arguments (line 525)
str_15345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 28), 'str', '__ne__')
# Getting the type of 'DynamicType' (line 525)
DynamicType_15346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 38), 'DynamicType', False)
int_15347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 51), 'int')
# Processing the call keyword arguments (line 525)
kwargs_15348 = {}
# Getting the type of 'HasMember' (line 525)
HasMember_15344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 525)
HasMember_call_result_15349 = invoke(stypy.reporting.localization.Localization(__file__, 525, 18), HasMember_15344, *[str_15345, DynamicType_15346, int_15347], **kwargs_15348)

# Assigning a type to the variable 'Overloads__ne__' (line 525)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 0), 'Overloads__ne__', HasMember_call_result_15349)

# Assigning a Call to a Name (line 526):

# Assigning a Call to a Name (line 526):

# Call to HasMember(...): (line 526)
# Processing the call arguments (line 526)
str_15351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 28), 'str', '__lt__')
# Getting the type of 'DynamicType' (line 526)
DynamicType_15352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 38), 'DynamicType', False)
int_15353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 51), 'int')
# Processing the call keyword arguments (line 526)
kwargs_15354 = {}
# Getting the type of 'HasMember' (line 526)
HasMember_15350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 526)
HasMember_call_result_15355 = invoke(stypy.reporting.localization.Localization(__file__, 526, 18), HasMember_15350, *[str_15351, DynamicType_15352, int_15353], **kwargs_15354)

# Assigning a type to the variable 'Overloads__lt__' (line 526)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 0), 'Overloads__lt__', HasMember_call_result_15355)

# Assigning a Call to a Name (line 527):

# Assigning a Call to a Name (line 527):

# Call to HasMember(...): (line 527)
# Processing the call arguments (line 527)
str_15357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 28), 'str', '__gt__')
# Getting the type of 'DynamicType' (line 527)
DynamicType_15358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 38), 'DynamicType', False)
int_15359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 51), 'int')
# Processing the call keyword arguments (line 527)
kwargs_15360 = {}
# Getting the type of 'HasMember' (line 527)
HasMember_15356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 527)
HasMember_call_result_15361 = invoke(stypy.reporting.localization.Localization(__file__, 527, 18), HasMember_15356, *[str_15357, DynamicType_15358, int_15359], **kwargs_15360)

# Assigning a type to the variable 'Overloads__gt__' (line 527)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 0), 'Overloads__gt__', HasMember_call_result_15361)

# Assigning a Call to a Name (line 528):

# Assigning a Call to a Name (line 528):

# Call to HasMember(...): (line 528)
# Processing the call arguments (line 528)
str_15363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 28), 'str', '__le__')
# Getting the type of 'DynamicType' (line 528)
DynamicType_15364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 38), 'DynamicType', False)
int_15365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 51), 'int')
# Processing the call keyword arguments (line 528)
kwargs_15366 = {}
# Getting the type of 'HasMember' (line 528)
HasMember_15362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 528)
HasMember_call_result_15367 = invoke(stypy.reporting.localization.Localization(__file__, 528, 18), HasMember_15362, *[str_15363, DynamicType_15364, int_15365], **kwargs_15366)

# Assigning a type to the variable 'Overloads__le__' (line 528)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 0), 'Overloads__le__', HasMember_call_result_15367)

# Assigning a Call to a Name (line 529):

# Assigning a Call to a Name (line 529):

# Call to HasMember(...): (line 529)
# Processing the call arguments (line 529)
str_15369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 28), 'str', '__ge__')
# Getting the type of 'DynamicType' (line 529)
DynamicType_15370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 38), 'DynamicType', False)
int_15371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 51), 'int')
# Processing the call keyword arguments (line 529)
kwargs_15372 = {}
# Getting the type of 'HasMember' (line 529)
HasMember_15368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 529)
HasMember_call_result_15373 = invoke(stypy.reporting.localization.Localization(__file__, 529, 18), HasMember_15368, *[str_15369, DynamicType_15370, int_15371], **kwargs_15372)

# Assigning a type to the variable 'Overloads__ge__' (line 529)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 0), 'Overloads__ge__', HasMember_call_result_15373)

# Assigning a Call to a Name (line 532):

# Assigning a Call to a Name (line 532):

# Call to HasMember(...): (line 532)
# Processing the call arguments (line 532)
str_15375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 29), 'str', '__pos__')
# Getting the type of 'UndefinedType' (line 532)
UndefinedType_15376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 40), 'UndefinedType', False)
int_15377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 55), 'int')
# Processing the call keyword arguments (line 532)
kwargs_15378 = {}
# Getting the type of 'HasMember' (line 532)
HasMember_15374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 532)
HasMember_call_result_15379 = invoke(stypy.reporting.localization.Localization(__file__, 532, 19), HasMember_15374, *[str_15375, UndefinedType_15376, int_15377], **kwargs_15378)

# Assigning a type to the variable 'Overloads__pos__' (line 532)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 0), 'Overloads__pos__', HasMember_call_result_15379)

# Assigning a Call to a Name (line 533):

# Assigning a Call to a Name (line 533):

# Call to HasMember(...): (line 533)
# Processing the call arguments (line 533)
str_15381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 29), 'str', '__neg__')
# Getting the type of 'UndefinedType' (line 533)
UndefinedType_15382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 40), 'UndefinedType', False)
int_15383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 55), 'int')
# Processing the call keyword arguments (line 533)
kwargs_15384 = {}
# Getting the type of 'HasMember' (line 533)
HasMember_15380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 533)
HasMember_call_result_15385 = invoke(stypy.reporting.localization.Localization(__file__, 533, 19), HasMember_15380, *[str_15381, UndefinedType_15382, int_15383], **kwargs_15384)

# Assigning a type to the variable 'Overloads__neg__' (line 533)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 0), 'Overloads__neg__', HasMember_call_result_15385)

# Assigning a Call to a Name (line 534):

# Assigning a Call to a Name (line 534):

# Call to HasMember(...): (line 534)
# Processing the call arguments (line 534)
str_15387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 29), 'str', '__abs__')
# Getting the type of 'UndefinedType' (line 534)
UndefinedType_15388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 40), 'UndefinedType', False)
int_15389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 55), 'int')
# Processing the call keyword arguments (line 534)
kwargs_15390 = {}
# Getting the type of 'HasMember' (line 534)
HasMember_15386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 534)
HasMember_call_result_15391 = invoke(stypy.reporting.localization.Localization(__file__, 534, 19), HasMember_15386, *[str_15387, UndefinedType_15388, int_15389], **kwargs_15390)

# Assigning a type to the variable 'Overloads__abs__' (line 534)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 0), 'Overloads__abs__', HasMember_call_result_15391)

# Assigning a Call to a Name (line 535):

# Assigning a Call to a Name (line 535):

# Call to HasMember(...): (line 535)
# Processing the call arguments (line 535)
str_15393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 32), 'str', '__invert__')
# Getting the type of 'UndefinedType' (line 535)
UndefinedType_15394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 46), 'UndefinedType', False)
int_15395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 61), 'int')
# Processing the call keyword arguments (line 535)
kwargs_15396 = {}
# Getting the type of 'HasMember' (line 535)
HasMember_15392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 22), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 535)
HasMember_call_result_15397 = invoke(stypy.reporting.localization.Localization(__file__, 535, 22), HasMember_15392, *[str_15393, UndefinedType_15394, int_15395], **kwargs_15396)

# Assigning a type to the variable 'Overloads__invert__' (line 535)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 0), 'Overloads__invert__', HasMember_call_result_15397)

# Assigning a Call to a Name (line 537):

# Assigning a Call to a Name (line 537):

# Call to HasMember(...): (line 537)
# Processing the call arguments (line 537)
str_15399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 31), 'str', '__round__')
# Getting the type of 'int' (line 537)
int_15400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 44), 'int', False)
int_15401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 49), 'int')
# Processing the call keyword arguments (line 537)
kwargs_15402 = {}
# Getting the type of 'HasMember' (line 537)
HasMember_15398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 21), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 537)
HasMember_call_result_15403 = invoke(stypy.reporting.localization.Localization(__file__, 537, 21), HasMember_15398, *[str_15399, int_15400, int_15401], **kwargs_15402)

# Assigning a type to the variable 'Overloads__round__' (line 537)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 0), 'Overloads__round__', HasMember_call_result_15403)

# Assigning a Call to a Name (line 538):

# Assigning a Call to a Name (line 538):

# Call to HasMember(...): (line 538)
# Processing the call arguments (line 538)
str_15405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 31), 'str', '__floor__')
# Getting the type of 'int' (line 538)
int_15406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 44), 'int', False)
int_15407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 49), 'int')
# Processing the call keyword arguments (line 538)
kwargs_15408 = {}
# Getting the type of 'HasMember' (line 538)
HasMember_15404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 21), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 538)
HasMember_call_result_15409 = invoke(stypy.reporting.localization.Localization(__file__, 538, 21), HasMember_15404, *[str_15405, int_15406, int_15407], **kwargs_15408)

# Assigning a type to the variable 'Overloads__floor__' (line 538)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 0), 'Overloads__floor__', HasMember_call_result_15409)

# Assigning a Call to a Name (line 539):

# Assigning a Call to a Name (line 539):

# Call to HasMember(...): (line 539)
# Processing the call arguments (line 539)
str_15411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 30), 'str', '__ceil__')
# Getting the type of 'int' (line 539)
int_15412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 42), 'int', False)
int_15413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 47), 'int')
# Processing the call keyword arguments (line 539)
kwargs_15414 = {}
# Getting the type of 'HasMember' (line 539)
HasMember_15410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 539)
HasMember_call_result_15415 = invoke(stypy.reporting.localization.Localization(__file__, 539, 20), HasMember_15410, *[str_15411, int_15412, int_15413], **kwargs_15414)

# Assigning a type to the variable 'Overloads__ceil__' (line 539)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 0), 'Overloads__ceil__', HasMember_call_result_15415)

# Assigning a Call to a Name (line 541):

# Assigning a Call to a Name (line 541):

# Call to HasMember(...): (line 541)
# Processing the call arguments (line 541)
str_15417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 31), 'str', '__trunc__')
# Getting the type of 'int' (line 541)
int_15418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 44), 'int', False)
int_15419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 49), 'int')
# Processing the call keyword arguments (line 541)
kwargs_15420 = {}
# Getting the type of 'HasMember' (line 541)
HasMember_15416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 21), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 541)
HasMember_call_result_15421 = invoke(stypy.reporting.localization.Localization(__file__, 541, 21), HasMember_15416, *[str_15417, int_15418, int_15419], **kwargs_15420)

# Assigning a type to the variable 'Overloads__trunc__' (line 541)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 0), 'Overloads__trunc__', HasMember_call_result_15421)

# Assigning a Call to a Name (line 544):

# Assigning a Call to a Name (line 544):

# Call to HasMember(...): (line 544)
# Processing the call arguments (line 544)
str_15423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 29), 'str', '__add__')
# Getting the type of 'DynamicType' (line 544)
DynamicType_15424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 40), 'DynamicType', False)
int_15425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 53), 'int')
# Processing the call keyword arguments (line 544)
kwargs_15426 = {}
# Getting the type of 'HasMember' (line 544)
HasMember_15422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 544)
HasMember_call_result_15427 = invoke(stypy.reporting.localization.Localization(__file__, 544, 19), HasMember_15422, *[str_15423, DynamicType_15424, int_15425], **kwargs_15426)

# Assigning a type to the variable 'Overloads__add__' (line 544)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 0), 'Overloads__add__', HasMember_call_result_15427)

# Assigning a Call to a Name (line 545):

# Assigning a Call to a Name (line 545):

# Call to HasMember(...): (line 545)
# Processing the call arguments (line 545)
str_15429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 29), 'str', '__sub__')
# Getting the type of 'DynamicType' (line 545)
DynamicType_15430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 40), 'DynamicType', False)
int_15431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 53), 'int')
# Processing the call keyword arguments (line 545)
kwargs_15432 = {}
# Getting the type of 'HasMember' (line 545)
HasMember_15428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 545)
HasMember_call_result_15433 = invoke(stypy.reporting.localization.Localization(__file__, 545, 19), HasMember_15428, *[str_15429, DynamicType_15430, int_15431], **kwargs_15432)

# Assigning a type to the variable 'Overloads__sub__' (line 545)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 0), 'Overloads__sub__', HasMember_call_result_15433)

# Assigning a Call to a Name (line 546):

# Assigning a Call to a Name (line 546):

# Call to HasMember(...): (line 546)
# Processing the call arguments (line 546)
str_15435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 29), 'str', '__mul__')
# Getting the type of 'DynamicType' (line 546)
DynamicType_15436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 40), 'DynamicType', False)
int_15437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 53), 'int')
# Processing the call keyword arguments (line 546)
kwargs_15438 = {}
# Getting the type of 'HasMember' (line 546)
HasMember_15434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 546)
HasMember_call_result_15439 = invoke(stypy.reporting.localization.Localization(__file__, 546, 19), HasMember_15434, *[str_15435, DynamicType_15436, int_15437], **kwargs_15438)

# Assigning a type to the variable 'Overloads__mul__' (line 546)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 0), 'Overloads__mul__', HasMember_call_result_15439)

# Assigning a Call to a Name (line 547):

# Assigning a Call to a Name (line 547):

# Call to HasMember(...): (line 547)
# Processing the call arguments (line 547)
str_15441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 34), 'str', '__floordiv__')
# Getting the type of 'DynamicType' (line 547)
DynamicType_15442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 50), 'DynamicType', False)
int_15443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 63), 'int')
# Processing the call keyword arguments (line 547)
kwargs_15444 = {}
# Getting the type of 'HasMember' (line 547)
HasMember_15440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 24), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 547)
HasMember_call_result_15445 = invoke(stypy.reporting.localization.Localization(__file__, 547, 24), HasMember_15440, *[str_15441, DynamicType_15442, int_15443], **kwargs_15444)

# Assigning a type to the variable 'Overloads__floordiv__' (line 547)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 0), 'Overloads__floordiv__', HasMember_call_result_15445)

# Assigning a Call to a Name (line 548):

# Assigning a Call to a Name (line 548):

# Call to HasMember(...): (line 548)
# Processing the call arguments (line 548)
str_15447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 29), 'str', '__div__')
# Getting the type of 'DynamicType' (line 548)
DynamicType_15448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 40), 'DynamicType', False)
int_15449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 53), 'int')
# Processing the call keyword arguments (line 548)
kwargs_15450 = {}
# Getting the type of 'HasMember' (line 548)
HasMember_15446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 548)
HasMember_call_result_15451 = invoke(stypy.reporting.localization.Localization(__file__, 548, 19), HasMember_15446, *[str_15447, DynamicType_15448, int_15449], **kwargs_15450)

# Assigning a type to the variable 'Overloads__div__' (line 548)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 0), 'Overloads__div__', HasMember_call_result_15451)

# Assigning a Call to a Name (line 549):

# Assigning a Call to a Name (line 549):

# Call to HasMember(...): (line 549)
# Processing the call arguments (line 549)
str_15453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 33), 'str', '__truediv__')
# Getting the type of 'DynamicType' (line 549)
DynamicType_15454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 48), 'DynamicType', False)
int_15455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 61), 'int')
# Processing the call keyword arguments (line 549)
kwargs_15456 = {}
# Getting the type of 'HasMember' (line 549)
HasMember_15452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 23), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 549)
HasMember_call_result_15457 = invoke(stypy.reporting.localization.Localization(__file__, 549, 23), HasMember_15452, *[str_15453, DynamicType_15454, int_15455], **kwargs_15456)

# Assigning a type to the variable 'Overloads__truediv__' (line 549)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 0), 'Overloads__truediv__', HasMember_call_result_15457)

# Assigning a Call to a Name (line 550):

# Assigning a Call to a Name (line 550):

# Call to HasMember(...): (line 550)
# Processing the call arguments (line 550)
str_15459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 29), 'str', '__mod__')
# Getting the type of 'DynamicType' (line 550)
DynamicType_15460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 40), 'DynamicType', False)
int_15461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 53), 'int')
# Processing the call keyword arguments (line 550)
kwargs_15462 = {}
# Getting the type of 'HasMember' (line 550)
HasMember_15458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 550)
HasMember_call_result_15463 = invoke(stypy.reporting.localization.Localization(__file__, 550, 19), HasMember_15458, *[str_15459, DynamicType_15460, int_15461], **kwargs_15462)

# Assigning a type to the variable 'Overloads__mod__' (line 550)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 0), 'Overloads__mod__', HasMember_call_result_15463)

# Assigning a Call to a Name (line 551):

# Assigning a Call to a Name (line 551):

# Call to HasMember(...): (line 551)
# Processing the call arguments (line 551)
str_15465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 32), 'str', '__divmod__')
# Getting the type of 'DynamicType' (line 551)
DynamicType_15466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 46), 'DynamicType', False)
int_15467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 59), 'int')
# Processing the call keyword arguments (line 551)
kwargs_15468 = {}
# Getting the type of 'HasMember' (line 551)
HasMember_15464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 22), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 551)
HasMember_call_result_15469 = invoke(stypy.reporting.localization.Localization(__file__, 551, 22), HasMember_15464, *[str_15465, DynamicType_15466, int_15467], **kwargs_15468)

# Assigning a type to the variable 'Overloads__divmod__' (line 551)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 0), 'Overloads__divmod__', HasMember_call_result_15469)

# Assigning a Call to a Name (line 552):

# Assigning a Call to a Name (line 552):

# Call to HasMember(...): (line 552)
# Processing the call arguments (line 552)
str_15471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 29), 'str', '__pow__')
# Getting the type of 'DynamicType' (line 552)
DynamicType_15472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 40), 'DynamicType', False)
int_15473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 53), 'int')
# Processing the call keyword arguments (line 552)
kwargs_15474 = {}
# Getting the type of 'HasMember' (line 552)
HasMember_15470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 552)
HasMember_call_result_15475 = invoke(stypy.reporting.localization.Localization(__file__, 552, 19), HasMember_15470, *[str_15471, DynamicType_15472, int_15473], **kwargs_15474)

# Assigning a type to the variable 'Overloads__pow__' (line 552)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 0), 'Overloads__pow__', HasMember_call_result_15475)

# Assigning a Call to a Name (line 553):

# Assigning a Call to a Name (line 553):

# Call to HasMember(...): (line 553)
# Processing the call arguments (line 553)
str_15477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 32), 'str', '__lshift__')
# Getting the type of 'DynamicType' (line 553)
DynamicType_15478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 46), 'DynamicType', False)
int_15479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 59), 'int')
# Processing the call keyword arguments (line 553)
kwargs_15480 = {}
# Getting the type of 'HasMember' (line 553)
HasMember_15476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 22), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 553)
HasMember_call_result_15481 = invoke(stypy.reporting.localization.Localization(__file__, 553, 22), HasMember_15476, *[str_15477, DynamicType_15478, int_15479], **kwargs_15480)

# Assigning a type to the variable 'Overloads__lshift__' (line 553)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 0), 'Overloads__lshift__', HasMember_call_result_15481)

# Assigning a Call to a Name (line 554):

# Assigning a Call to a Name (line 554):

# Call to HasMember(...): (line 554)
# Processing the call arguments (line 554)
str_15483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 32), 'str', '__rshift__')
# Getting the type of 'DynamicType' (line 554)
DynamicType_15484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 46), 'DynamicType', False)
int_15485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 59), 'int')
# Processing the call keyword arguments (line 554)
kwargs_15486 = {}
# Getting the type of 'HasMember' (line 554)
HasMember_15482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 22), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 554)
HasMember_call_result_15487 = invoke(stypy.reporting.localization.Localization(__file__, 554, 22), HasMember_15482, *[str_15483, DynamicType_15484, int_15485], **kwargs_15486)

# Assigning a type to the variable 'Overloads__rshift__' (line 554)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 0), 'Overloads__rshift__', HasMember_call_result_15487)

# Assigning a Call to a Name (line 555):

# Assigning a Call to a Name (line 555):

# Call to HasMember(...): (line 555)
# Processing the call arguments (line 555)
str_15489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 29), 'str', '__and__')
# Getting the type of 'DynamicType' (line 555)
DynamicType_15490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 40), 'DynamicType', False)
int_15491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 53), 'int')
# Processing the call keyword arguments (line 555)
kwargs_15492 = {}
# Getting the type of 'HasMember' (line 555)
HasMember_15488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 555)
HasMember_call_result_15493 = invoke(stypy.reporting.localization.Localization(__file__, 555, 19), HasMember_15488, *[str_15489, DynamicType_15490, int_15491], **kwargs_15492)

# Assigning a type to the variable 'Overloads__and__' (line 555)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 0), 'Overloads__and__', HasMember_call_result_15493)

# Assigning a Call to a Name (line 556):

# Assigning a Call to a Name (line 556):

# Call to HasMember(...): (line 556)
# Processing the call arguments (line 556)
str_15495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 28), 'str', '__or__')
# Getting the type of 'DynamicType' (line 556)
DynamicType_15496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 38), 'DynamicType', False)
int_15497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 51), 'int')
# Processing the call keyword arguments (line 556)
kwargs_15498 = {}
# Getting the type of 'HasMember' (line 556)
HasMember_15494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 556)
HasMember_call_result_15499 = invoke(stypy.reporting.localization.Localization(__file__, 556, 18), HasMember_15494, *[str_15495, DynamicType_15496, int_15497], **kwargs_15498)

# Assigning a type to the variable 'Overloads__or__' (line 556)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 0), 'Overloads__or__', HasMember_call_result_15499)

# Assigning a Call to a Name (line 557):

# Assigning a Call to a Name (line 557):

# Call to HasMember(...): (line 557)
# Processing the call arguments (line 557)
str_15501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 29), 'str', '__xor__')
# Getting the type of 'DynamicType' (line 557)
DynamicType_15502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 40), 'DynamicType', False)
int_15503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 53), 'int')
# Processing the call keyword arguments (line 557)
kwargs_15504 = {}
# Getting the type of 'HasMember' (line 557)
HasMember_15500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 557)
HasMember_call_result_15505 = invoke(stypy.reporting.localization.Localization(__file__, 557, 19), HasMember_15500, *[str_15501, DynamicType_15502, int_15503], **kwargs_15504)

# Assigning a type to the variable 'Overloads__xor__' (line 557)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 0), 'Overloads__xor__', HasMember_call_result_15505)

# Assigning a Call to a Name (line 560):

# Assigning a Call to a Name (line 560):

# Call to HasMember(...): (line 560)
# Processing the call arguments (line 560)
str_15507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 30), 'str', '__radd__')
# Getting the type of 'DynamicType' (line 560)
DynamicType_15508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 42), 'DynamicType', False)
int_15509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 55), 'int')
# Processing the call keyword arguments (line 560)
kwargs_15510 = {}
# Getting the type of 'HasMember' (line 560)
HasMember_15506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 560)
HasMember_call_result_15511 = invoke(stypy.reporting.localization.Localization(__file__, 560, 20), HasMember_15506, *[str_15507, DynamicType_15508, int_15509], **kwargs_15510)

# Assigning a type to the variable 'Overloads__radd__' (line 560)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 0), 'Overloads__radd__', HasMember_call_result_15511)

# Assigning a Call to a Name (line 561):

# Assigning a Call to a Name (line 561):

# Call to HasMember(...): (line 561)
# Processing the call arguments (line 561)
str_15513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 30), 'str', '__rsub__')
# Getting the type of 'DynamicType' (line 561)
DynamicType_15514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 42), 'DynamicType', False)
int_15515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 55), 'int')
# Processing the call keyword arguments (line 561)
kwargs_15516 = {}
# Getting the type of 'HasMember' (line 561)
HasMember_15512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 561)
HasMember_call_result_15517 = invoke(stypy.reporting.localization.Localization(__file__, 561, 20), HasMember_15512, *[str_15513, DynamicType_15514, int_15515], **kwargs_15516)

# Assigning a type to the variable 'Overloads__rsub__' (line 561)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 0), 'Overloads__rsub__', HasMember_call_result_15517)

# Assigning a Call to a Name (line 562):

# Assigning a Call to a Name (line 562):

# Call to HasMember(...): (line 562)
# Processing the call arguments (line 562)
str_15519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 30), 'str', '__rmul__')
# Getting the type of 'DynamicType' (line 562)
DynamicType_15520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 42), 'DynamicType', False)
int_15521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 55), 'int')
# Processing the call keyword arguments (line 562)
kwargs_15522 = {}
# Getting the type of 'HasMember' (line 562)
HasMember_15518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 562)
HasMember_call_result_15523 = invoke(stypy.reporting.localization.Localization(__file__, 562, 20), HasMember_15518, *[str_15519, DynamicType_15520, int_15521], **kwargs_15522)

# Assigning a type to the variable 'Overloads__rmul__' (line 562)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 0), 'Overloads__rmul__', HasMember_call_result_15523)

# Assigning a Call to a Name (line 563):

# Assigning a Call to a Name (line 563):

# Call to HasMember(...): (line 563)
# Processing the call arguments (line 563)
str_15525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 35), 'str', '__rfloordiv__')
# Getting the type of 'DynamicType' (line 563)
DynamicType_15526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 52), 'DynamicType', False)
int_15527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 65), 'int')
# Processing the call keyword arguments (line 563)
kwargs_15528 = {}
# Getting the type of 'HasMember' (line 563)
HasMember_15524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 25), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 563)
HasMember_call_result_15529 = invoke(stypy.reporting.localization.Localization(__file__, 563, 25), HasMember_15524, *[str_15525, DynamicType_15526, int_15527], **kwargs_15528)

# Assigning a type to the variable 'Overloads__rfloordiv__' (line 563)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 0), 'Overloads__rfloordiv__', HasMember_call_result_15529)

# Assigning a Call to a Name (line 564):

# Assigning a Call to a Name (line 564):

# Call to HasMember(...): (line 564)
# Processing the call arguments (line 564)
str_15531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 30), 'str', '__rdiv__')
# Getting the type of 'DynamicType' (line 564)
DynamicType_15532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 42), 'DynamicType', False)
int_15533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 55), 'int')
# Processing the call keyword arguments (line 564)
kwargs_15534 = {}
# Getting the type of 'HasMember' (line 564)
HasMember_15530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 564)
HasMember_call_result_15535 = invoke(stypy.reporting.localization.Localization(__file__, 564, 20), HasMember_15530, *[str_15531, DynamicType_15532, int_15533], **kwargs_15534)

# Assigning a type to the variable 'Overloads__rdiv__' (line 564)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 0), 'Overloads__rdiv__', HasMember_call_result_15535)

# Assigning a Call to a Name (line 565):

# Assigning a Call to a Name (line 565):

# Call to HasMember(...): (line 565)
# Processing the call arguments (line 565)
str_15537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 34), 'str', '__rtruediv__')
# Getting the type of 'DynamicType' (line 565)
DynamicType_15538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 50), 'DynamicType', False)
int_15539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 63), 'int')
# Processing the call keyword arguments (line 565)
kwargs_15540 = {}
# Getting the type of 'HasMember' (line 565)
HasMember_15536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 24), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 565)
HasMember_call_result_15541 = invoke(stypy.reporting.localization.Localization(__file__, 565, 24), HasMember_15536, *[str_15537, DynamicType_15538, int_15539], **kwargs_15540)

# Assigning a type to the variable 'Overloads__rtruediv__' (line 565)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 0), 'Overloads__rtruediv__', HasMember_call_result_15541)

# Assigning a Call to a Name (line 566):

# Assigning a Call to a Name (line 566):

# Call to HasMember(...): (line 566)
# Processing the call arguments (line 566)
str_15543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 30), 'str', '__rmod__')
# Getting the type of 'DynamicType' (line 566)
DynamicType_15544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 42), 'DynamicType', False)
int_15545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 55), 'int')
# Processing the call keyword arguments (line 566)
kwargs_15546 = {}
# Getting the type of 'HasMember' (line 566)
HasMember_15542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 566)
HasMember_call_result_15547 = invoke(stypy.reporting.localization.Localization(__file__, 566, 20), HasMember_15542, *[str_15543, DynamicType_15544, int_15545], **kwargs_15546)

# Assigning a type to the variable 'Overloads__rmod__' (line 566)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 0), 'Overloads__rmod__', HasMember_call_result_15547)

# Assigning a Call to a Name (line 567):

# Assigning a Call to a Name (line 567):

# Call to HasMember(...): (line 567)
# Processing the call arguments (line 567)
str_15549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 33), 'str', '__rdivmod__')
# Getting the type of 'DynamicType' (line 567)
DynamicType_15550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 48), 'DynamicType', False)
int_15551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 61), 'int')
# Processing the call keyword arguments (line 567)
kwargs_15552 = {}
# Getting the type of 'HasMember' (line 567)
HasMember_15548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 23), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 567)
HasMember_call_result_15553 = invoke(stypy.reporting.localization.Localization(__file__, 567, 23), HasMember_15548, *[str_15549, DynamicType_15550, int_15551], **kwargs_15552)

# Assigning a type to the variable 'Overloads__rdivmod__' (line 567)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 0), 'Overloads__rdivmod__', HasMember_call_result_15553)

# Assigning a Call to a Name (line 568):

# Assigning a Call to a Name (line 568):

# Call to HasMember(...): (line 568)
# Processing the call arguments (line 568)
str_15555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 30), 'str', '__rpow__')
# Getting the type of 'DynamicType' (line 568)
DynamicType_15556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 42), 'DynamicType', False)
int_15557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 55), 'int')
# Processing the call keyword arguments (line 568)
kwargs_15558 = {}
# Getting the type of 'HasMember' (line 568)
HasMember_15554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 568)
HasMember_call_result_15559 = invoke(stypy.reporting.localization.Localization(__file__, 568, 20), HasMember_15554, *[str_15555, DynamicType_15556, int_15557], **kwargs_15558)

# Assigning a type to the variable 'Overloads__rpow__' (line 568)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 0), 'Overloads__rpow__', HasMember_call_result_15559)

# Assigning a Call to a Name (line 569):

# Assigning a Call to a Name (line 569):

# Call to HasMember(...): (line 569)
# Processing the call arguments (line 569)
str_15561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 33), 'str', '__rlshift__')
# Getting the type of 'DynamicType' (line 569)
DynamicType_15562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 48), 'DynamicType', False)
int_15563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 61), 'int')
# Processing the call keyword arguments (line 569)
kwargs_15564 = {}
# Getting the type of 'HasMember' (line 569)
HasMember_15560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 23), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 569)
HasMember_call_result_15565 = invoke(stypy.reporting.localization.Localization(__file__, 569, 23), HasMember_15560, *[str_15561, DynamicType_15562, int_15563], **kwargs_15564)

# Assigning a type to the variable 'Overloads__rlshift__' (line 569)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 0), 'Overloads__rlshift__', HasMember_call_result_15565)

# Assigning a Call to a Name (line 570):

# Assigning a Call to a Name (line 570):

# Call to HasMember(...): (line 570)
# Processing the call arguments (line 570)
str_15567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 33), 'str', '__rrshift__')
# Getting the type of 'DynamicType' (line 570)
DynamicType_15568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 48), 'DynamicType', False)
int_15569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 61), 'int')
# Processing the call keyword arguments (line 570)
kwargs_15570 = {}
# Getting the type of 'HasMember' (line 570)
HasMember_15566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 23), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 570)
HasMember_call_result_15571 = invoke(stypy.reporting.localization.Localization(__file__, 570, 23), HasMember_15566, *[str_15567, DynamicType_15568, int_15569], **kwargs_15570)

# Assigning a type to the variable 'Overloads__rrshift__' (line 570)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 0), 'Overloads__rrshift__', HasMember_call_result_15571)

# Assigning a Call to a Name (line 571):

# Assigning a Call to a Name (line 571):

# Call to HasMember(...): (line 571)
# Processing the call arguments (line 571)
str_15573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 30), 'str', '__rand__')
# Getting the type of 'DynamicType' (line 571)
DynamicType_15574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 42), 'DynamicType', False)
int_15575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 55), 'int')
# Processing the call keyword arguments (line 571)
kwargs_15576 = {}
# Getting the type of 'HasMember' (line 571)
HasMember_15572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 571)
HasMember_call_result_15577 = invoke(stypy.reporting.localization.Localization(__file__, 571, 20), HasMember_15572, *[str_15573, DynamicType_15574, int_15575], **kwargs_15576)

# Assigning a type to the variable 'Overloads__rand__' (line 571)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 0), 'Overloads__rand__', HasMember_call_result_15577)

# Assigning a Call to a Name (line 572):

# Assigning a Call to a Name (line 572):

# Call to HasMember(...): (line 572)
# Processing the call arguments (line 572)
str_15579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 29), 'str', '__ror__')
# Getting the type of 'DynamicType' (line 572)
DynamicType_15580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 40), 'DynamicType', False)
int_15581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 53), 'int')
# Processing the call keyword arguments (line 572)
kwargs_15582 = {}
# Getting the type of 'HasMember' (line 572)
HasMember_15578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 572)
HasMember_call_result_15583 = invoke(stypy.reporting.localization.Localization(__file__, 572, 19), HasMember_15578, *[str_15579, DynamicType_15580, int_15581], **kwargs_15582)

# Assigning a type to the variable 'Overloads__ror__' (line 572)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 0), 'Overloads__ror__', HasMember_call_result_15583)

# Assigning a Call to a Name (line 573):

# Assigning a Call to a Name (line 573):

# Call to HasMember(...): (line 573)
# Processing the call arguments (line 573)
str_15585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 30), 'str', '__rxor__')
# Getting the type of 'DynamicType' (line 573)
DynamicType_15586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 42), 'DynamicType', False)
int_15587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 55), 'int')
# Processing the call keyword arguments (line 573)
kwargs_15588 = {}
# Getting the type of 'HasMember' (line 573)
HasMember_15584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 573)
HasMember_call_result_15589 = invoke(stypy.reporting.localization.Localization(__file__, 573, 20), HasMember_15584, *[str_15585, DynamicType_15586, int_15587], **kwargs_15588)

# Assigning a type to the variable 'Overloads__rxor__' (line 573)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 0), 'Overloads__rxor__', HasMember_call_result_15589)

# Assigning a Call to a Name (line 577):

# Assigning a Call to a Name (line 577):

# Call to HasMember(...): (line 577)
# Processing the call arguments (line 577)
str_15591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 30), 'str', '__iadd__')
# Getting the type of 'DynamicType' (line 577)
DynamicType_15592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 42), 'DynamicType', False)
int_15593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 55), 'int')
# Processing the call keyword arguments (line 577)
kwargs_15594 = {}
# Getting the type of 'HasMember' (line 577)
HasMember_15590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 577)
HasMember_call_result_15595 = invoke(stypy.reporting.localization.Localization(__file__, 577, 20), HasMember_15590, *[str_15591, DynamicType_15592, int_15593], **kwargs_15594)

# Assigning a type to the variable 'Overloads__iadd__' (line 577)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 0), 'Overloads__iadd__', HasMember_call_result_15595)

# Assigning a Call to a Name (line 578):

# Assigning a Call to a Name (line 578):

# Call to HasMember(...): (line 578)
# Processing the call arguments (line 578)
str_15597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 30), 'str', '__isub__')
# Getting the type of 'DynamicType' (line 578)
DynamicType_15598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 42), 'DynamicType', False)
int_15599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 55), 'int')
# Processing the call keyword arguments (line 578)
kwargs_15600 = {}
# Getting the type of 'HasMember' (line 578)
HasMember_15596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 578)
HasMember_call_result_15601 = invoke(stypy.reporting.localization.Localization(__file__, 578, 20), HasMember_15596, *[str_15597, DynamicType_15598, int_15599], **kwargs_15600)

# Assigning a type to the variable 'Overloads__isub__' (line 578)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 0), 'Overloads__isub__', HasMember_call_result_15601)

# Assigning a Call to a Name (line 579):

# Assigning a Call to a Name (line 579):

# Call to HasMember(...): (line 579)
# Processing the call arguments (line 579)
str_15603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 30), 'str', '__imul__')
# Getting the type of 'DynamicType' (line 579)
DynamicType_15604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 42), 'DynamicType', False)
int_15605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 55), 'int')
# Processing the call keyword arguments (line 579)
kwargs_15606 = {}
# Getting the type of 'HasMember' (line 579)
HasMember_15602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 579)
HasMember_call_result_15607 = invoke(stypy.reporting.localization.Localization(__file__, 579, 20), HasMember_15602, *[str_15603, DynamicType_15604, int_15605], **kwargs_15606)

# Assigning a type to the variable 'Overloads__imul__' (line 579)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 0), 'Overloads__imul__', HasMember_call_result_15607)

# Assigning a Call to a Name (line 580):

# Assigning a Call to a Name (line 580):

# Call to HasMember(...): (line 580)
# Processing the call arguments (line 580)
str_15609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 35), 'str', '__ifloordiv__')
# Getting the type of 'DynamicType' (line 580)
DynamicType_15610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 52), 'DynamicType', False)
int_15611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 65), 'int')
# Processing the call keyword arguments (line 580)
kwargs_15612 = {}
# Getting the type of 'HasMember' (line 580)
HasMember_15608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 25), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 580)
HasMember_call_result_15613 = invoke(stypy.reporting.localization.Localization(__file__, 580, 25), HasMember_15608, *[str_15609, DynamicType_15610, int_15611], **kwargs_15612)

# Assigning a type to the variable 'Overloads__ifloordiv__' (line 580)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 0), 'Overloads__ifloordiv__', HasMember_call_result_15613)

# Assigning a Call to a Name (line 581):

# Assigning a Call to a Name (line 581):

# Call to HasMember(...): (line 581)
# Processing the call arguments (line 581)
str_15615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 30), 'str', '__idiv__')
# Getting the type of 'DynamicType' (line 581)
DynamicType_15616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 42), 'DynamicType', False)
int_15617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 55), 'int')
# Processing the call keyword arguments (line 581)
kwargs_15618 = {}
# Getting the type of 'HasMember' (line 581)
HasMember_15614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 581)
HasMember_call_result_15619 = invoke(stypy.reporting.localization.Localization(__file__, 581, 20), HasMember_15614, *[str_15615, DynamicType_15616, int_15617], **kwargs_15618)

# Assigning a type to the variable 'Overloads__idiv__' (line 581)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 0), 'Overloads__idiv__', HasMember_call_result_15619)

# Assigning a Call to a Name (line 582):

# Assigning a Call to a Name (line 582):

# Call to HasMember(...): (line 582)
# Processing the call arguments (line 582)
str_15621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 34), 'str', '__itruediv__')
# Getting the type of 'DynamicType' (line 582)
DynamicType_15622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 50), 'DynamicType', False)
int_15623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 63), 'int')
# Processing the call keyword arguments (line 582)
kwargs_15624 = {}
# Getting the type of 'HasMember' (line 582)
HasMember_15620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 24), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 582)
HasMember_call_result_15625 = invoke(stypy.reporting.localization.Localization(__file__, 582, 24), HasMember_15620, *[str_15621, DynamicType_15622, int_15623], **kwargs_15624)

# Assigning a type to the variable 'Overloads__itruediv__' (line 582)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 0), 'Overloads__itruediv__', HasMember_call_result_15625)

# Assigning a Call to a Name (line 583):

# Assigning a Call to a Name (line 583):

# Call to HasMember(...): (line 583)
# Processing the call arguments (line 583)
str_15627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 30), 'str', '__imod__')
# Getting the type of 'DynamicType' (line 583)
DynamicType_15628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 42), 'DynamicType', False)
int_15629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 55), 'int')
# Processing the call keyword arguments (line 583)
kwargs_15630 = {}
# Getting the type of 'HasMember' (line 583)
HasMember_15626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 583)
HasMember_call_result_15631 = invoke(stypy.reporting.localization.Localization(__file__, 583, 20), HasMember_15626, *[str_15627, DynamicType_15628, int_15629], **kwargs_15630)

# Assigning a type to the variable 'Overloads__imod__' (line 583)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 0), 'Overloads__imod__', HasMember_call_result_15631)

# Assigning a Call to a Name (line 584):

# Assigning a Call to a Name (line 584):

# Call to HasMember(...): (line 584)
# Processing the call arguments (line 584)
str_15633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 33), 'str', '__idivmod__')
# Getting the type of 'DynamicType' (line 584)
DynamicType_15634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 48), 'DynamicType', False)
int_15635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 61), 'int')
# Processing the call keyword arguments (line 584)
kwargs_15636 = {}
# Getting the type of 'HasMember' (line 584)
HasMember_15632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 23), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 584)
HasMember_call_result_15637 = invoke(stypy.reporting.localization.Localization(__file__, 584, 23), HasMember_15632, *[str_15633, DynamicType_15634, int_15635], **kwargs_15636)

# Assigning a type to the variable 'Overloads__idivmod__' (line 584)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 0), 'Overloads__idivmod__', HasMember_call_result_15637)

# Assigning a Call to a Name (line 585):

# Assigning a Call to a Name (line 585):

# Call to HasMember(...): (line 585)
# Processing the call arguments (line 585)
str_15639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 30), 'str', '__ipow__')
# Getting the type of 'DynamicType' (line 585)
DynamicType_15640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 42), 'DynamicType', False)
int_15641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 55), 'int')
# Processing the call keyword arguments (line 585)
kwargs_15642 = {}
# Getting the type of 'HasMember' (line 585)
HasMember_15638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 585)
HasMember_call_result_15643 = invoke(stypy.reporting.localization.Localization(__file__, 585, 20), HasMember_15638, *[str_15639, DynamicType_15640, int_15641], **kwargs_15642)

# Assigning a type to the variable 'Overloads__ipow__' (line 585)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 0), 'Overloads__ipow__', HasMember_call_result_15643)

# Assigning a Call to a Name (line 586):

# Assigning a Call to a Name (line 586):

# Call to HasMember(...): (line 586)
# Processing the call arguments (line 586)
str_15645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 33), 'str', '__ilshift__')
# Getting the type of 'DynamicType' (line 586)
DynamicType_15646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 48), 'DynamicType', False)
int_15647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 61), 'int')
# Processing the call keyword arguments (line 586)
kwargs_15648 = {}
# Getting the type of 'HasMember' (line 586)
HasMember_15644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 23), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 586)
HasMember_call_result_15649 = invoke(stypy.reporting.localization.Localization(__file__, 586, 23), HasMember_15644, *[str_15645, DynamicType_15646, int_15647], **kwargs_15648)

# Assigning a type to the variable 'Overloads__ilshift__' (line 586)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 0), 'Overloads__ilshift__', HasMember_call_result_15649)

# Assigning a Call to a Name (line 587):

# Assigning a Call to a Name (line 587):

# Call to HasMember(...): (line 587)
# Processing the call arguments (line 587)
str_15651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 33), 'str', '__irshift__')
# Getting the type of 'DynamicType' (line 587)
DynamicType_15652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 48), 'DynamicType', False)
int_15653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 61), 'int')
# Processing the call keyword arguments (line 587)
kwargs_15654 = {}
# Getting the type of 'HasMember' (line 587)
HasMember_15650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 23), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 587)
HasMember_call_result_15655 = invoke(stypy.reporting.localization.Localization(__file__, 587, 23), HasMember_15650, *[str_15651, DynamicType_15652, int_15653], **kwargs_15654)

# Assigning a type to the variable 'Overloads__irshift__' (line 587)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 0), 'Overloads__irshift__', HasMember_call_result_15655)

# Assigning a Call to a Name (line 588):

# Assigning a Call to a Name (line 588):

# Call to HasMember(...): (line 588)
# Processing the call arguments (line 588)
str_15657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 30), 'str', '__iand__')
# Getting the type of 'DynamicType' (line 588)
DynamicType_15658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 42), 'DynamicType', False)
int_15659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 55), 'int')
# Processing the call keyword arguments (line 588)
kwargs_15660 = {}
# Getting the type of 'HasMember' (line 588)
HasMember_15656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 588)
HasMember_call_result_15661 = invoke(stypy.reporting.localization.Localization(__file__, 588, 20), HasMember_15656, *[str_15657, DynamicType_15658, int_15659], **kwargs_15660)

# Assigning a type to the variable 'Overloads__iand__' (line 588)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 0), 'Overloads__iand__', HasMember_call_result_15661)

# Assigning a Call to a Name (line 589):

# Assigning a Call to a Name (line 589):

# Call to HasMember(...): (line 589)
# Processing the call arguments (line 589)
str_15663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 29), 'str', '__ior__')
# Getting the type of 'DynamicType' (line 589)
DynamicType_15664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 40), 'DynamicType', False)
int_15665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 53), 'int')
# Processing the call keyword arguments (line 589)
kwargs_15666 = {}
# Getting the type of 'HasMember' (line 589)
HasMember_15662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 589)
HasMember_call_result_15667 = invoke(stypy.reporting.localization.Localization(__file__, 589, 19), HasMember_15662, *[str_15663, DynamicType_15664, int_15665], **kwargs_15666)

# Assigning a type to the variable 'Overloads__ior__' (line 589)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 0), 'Overloads__ior__', HasMember_call_result_15667)

# Assigning a Call to a Name (line 590):

# Assigning a Call to a Name (line 590):

# Call to HasMember(...): (line 590)
# Processing the call arguments (line 590)
str_15669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 30), 'str', '__ixor__')
# Getting the type of 'DynamicType' (line 590)
DynamicType_15670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 42), 'DynamicType', False)
int_15671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 55), 'int')
# Processing the call keyword arguments (line 590)
kwargs_15672 = {}
# Getting the type of 'HasMember' (line 590)
HasMember_15668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 590)
HasMember_call_result_15673 = invoke(stypy.reporting.localization.Localization(__file__, 590, 20), HasMember_15668, *[str_15669, DynamicType_15670, int_15671], **kwargs_15672)

# Assigning a type to the variable 'Overloads__ixor__' (line 590)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 0), 'Overloads__ixor__', HasMember_call_result_15673)

# Assigning a Call to a Name (line 593):

# Assigning a Call to a Name (line 593):

# Call to HasMember(...): (line 593)
# Processing the call arguments (line 593)
str_15675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 23), 'str', '__str__')
# Getting the type of 'str' (line 593)
str_15676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 34), 'str', False)
int_15677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 39), 'int')
# Processing the call keyword arguments (line 593)
kwargs_15678 = {}
# Getting the type of 'HasMember' (line 593)
HasMember_15674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 13), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 593)
HasMember_call_result_15679 = invoke(stypy.reporting.localization.Localization(__file__, 593, 13), HasMember_15674, *[str_15675, str_15676, int_15677], **kwargs_15678)

# Assigning a type to the variable 'Has__str__' (line 593)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 0), 'Has__str__', HasMember_call_result_15679)

# Assigning a Call to a Name (line 594):

# Assigning a Call to a Name (line 594):

# Call to HasMember(...): (line 594)
# Processing the call arguments (line 594)
str_15681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 24), 'str', '__repr__')
# Getting the type of 'str' (line 594)
str_15682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 36), 'str', False)
int_15683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 41), 'int')
# Processing the call keyword arguments (line 594)
kwargs_15684 = {}
# Getting the type of 'HasMember' (line 594)
HasMember_15680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 14), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 594)
HasMember_call_result_15685 = invoke(stypy.reporting.localization.Localization(__file__, 594, 14), HasMember_15680, *[str_15681, str_15682, int_15683], **kwargs_15684)

# Assigning a type to the variable 'Has__repr__' (line 594)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 0), 'Has__repr__', HasMember_call_result_15685)

# Assigning a Call to a Name (line 595):

# Assigning a Call to a Name (line 595):

# Call to HasMember(...): (line 595)
# Processing the call arguments (line 595)
str_15687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 27), 'str', '__unicode__')
# Getting the type of 'unicode' (line 595)
unicode_15688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 42), 'unicode', False)
int_15689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 51), 'int')
# Processing the call keyword arguments (line 595)
kwargs_15690 = {}
# Getting the type of 'HasMember' (line 595)
HasMember_15686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 17), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 595)
HasMember_call_result_15691 = invoke(stypy.reporting.localization.Localization(__file__, 595, 17), HasMember_15686, *[str_15687, unicode_15688, int_15689], **kwargs_15690)

# Assigning a type to the variable 'Has__unicode__' (line 595)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 0), 'Has__unicode__', HasMember_call_result_15691)

# Assigning a Call to a Name (line 596):

# Assigning a Call to a Name (line 596):

# Call to HasMember(...): (line 596)
# Processing the call arguments (line 596)
str_15693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 26), 'str', '__format__')
# Getting the type of 'str' (line 596)
str_15694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 40), 'str', False)
int_15695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 45), 'int')
# Processing the call keyword arguments (line 596)
kwargs_15696 = {}
# Getting the type of 'HasMember' (line 596)
HasMember_15692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 16), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 596)
HasMember_call_result_15697 = invoke(stypy.reporting.localization.Localization(__file__, 596, 16), HasMember_15692, *[str_15693, str_15694, int_15695], **kwargs_15696)

# Assigning a type to the variable 'Has__format__' (line 596)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 0), 'Has__format__', HasMember_call_result_15697)

# Assigning a Call to a Name (line 597):

# Assigning a Call to a Name (line 597):

# Call to HasMember(...): (line 597)
# Processing the call arguments (line 597)
str_15699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 24), 'str', '__hash__')
# Getting the type of 'int' (line 597)
int_15700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 36), 'int', False)
int_15701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 41), 'int')
# Processing the call keyword arguments (line 597)
kwargs_15702 = {}
# Getting the type of 'HasMember' (line 597)
HasMember_15698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 14), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 597)
HasMember_call_result_15703 = invoke(stypy.reporting.localization.Localization(__file__, 597, 14), HasMember_15698, *[str_15699, int_15700, int_15701], **kwargs_15702)

# Assigning a type to the variable 'Has__hash__' (line 597)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 0), 'Has__hash__', HasMember_call_result_15703)

# Assigning a Call to a Name (line 598):

# Assigning a Call to a Name (line 598):

# Call to HasMember(...): (line 598)
# Processing the call arguments (line 598)
str_15705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 27), 'str', '__nonzero__')
# Getting the type of 'bool' (line 598)
bool_15706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 42), 'bool', False)
int_15707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 48), 'int')
# Processing the call keyword arguments (line 598)
kwargs_15708 = {}
# Getting the type of 'HasMember' (line 598)
HasMember_15704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 17), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 598)
HasMember_call_result_15709 = invoke(stypy.reporting.localization.Localization(__file__, 598, 17), HasMember_15704, *[str_15705, bool_15706, int_15707], **kwargs_15708)

# Assigning a type to the variable 'Has__nonzero__' (line 598)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 0), 'Has__nonzero__', HasMember_call_result_15709)

# Assigning a Call to a Name (line 599):

# Assigning a Call to a Name (line 599):

# Call to HasMember(...): (line 599)
# Processing the call arguments (line 599)
str_15711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 23), 'str', '__dir__')
# Getting the type of 'DynamicType' (line 599)
DynamicType_15712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 34), 'DynamicType', False)
int_15713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 47), 'int')
# Processing the call keyword arguments (line 599)
kwargs_15714 = {}
# Getting the type of 'HasMember' (line 599)
HasMember_15710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 13), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 599)
HasMember_call_result_15715 = invoke(stypy.reporting.localization.Localization(__file__, 599, 13), HasMember_15710, *[str_15711, DynamicType_15712, int_15713], **kwargs_15714)

# Assigning a type to the variable 'Has__dir__' (line 599)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 0), 'Has__dir__', HasMember_call_result_15715)

# Assigning a Call to a Name (line 600):

# Assigning a Call to a Name (line 600):

# Call to HasMember(...): (line 600)
# Processing the call arguments (line 600)
str_15717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 26), 'str', '__sizeof__')
# Getting the type of 'int' (line 600)
int_15718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 40), 'int', False)
int_15719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 45), 'int')
# Processing the call keyword arguments (line 600)
kwargs_15720 = {}
# Getting the type of 'HasMember' (line 600)
HasMember_15716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 16), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 600)
HasMember_call_result_15721 = invoke(stypy.reporting.localization.Localization(__file__, 600, 16), HasMember_15716, *[str_15717, int_15718, int_15719], **kwargs_15720)

# Assigning a type to the variable 'Has__sizeof__' (line 600)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 0), 'Has__sizeof__', HasMember_call_result_15721)

# Assigning a Call to a Name (line 601):

# Assigning a Call to a Name (line 601):

# Call to Callable(...): (line 601)
# Processing the call keyword arguments (line 601)
kwargs_15723 = {}
# Getting the type of 'Callable' (line 601)
Callable_15722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 14), 'Callable', False)
# Calling Callable(args, kwargs) (line 601)
Callable_call_result_15724 = invoke(stypy.reporting.localization.Localization(__file__, 601, 14), Callable_15722, *[], **kwargs_15723)

# Assigning a type to the variable 'Has__call__' (line 601)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 0), 'Has__call__', Callable_call_result_15724)

# Assigning a Call to a Name (line 602):

# Assigning a Call to a Name (line 602):

# Call to HasMember(...): (line 602)
# Processing the call arguments (line 602)
str_15726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 23), 'str', '__mro__')
# Getting the type of 'DynamicType' (line 602)
DynamicType_15727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 34), 'DynamicType', False)
int_15728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 47), 'int')
# Processing the call keyword arguments (line 602)
kwargs_15729 = {}
# Getting the type of 'HasMember' (line 602)
HasMember_15725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 13), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 602)
HasMember_call_result_15730 = invoke(stypy.reporting.localization.Localization(__file__, 602, 13), HasMember_15725, *[str_15726, DynamicType_15727, int_15728], **kwargs_15729)

# Assigning a type to the variable 'Has__mro__' (line 602)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 0), 'Has__mro__', HasMember_call_result_15730)

# Assigning a Call to a Name (line 603):

# Assigning a Call to a Name (line 603):

# Call to HasMember(...): (line 603)
# Processing the call arguments (line 603)
str_15732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 25), 'str', '__class__')
# Getting the type of 'DynamicType' (line 603)
DynamicType_15733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 38), 'DynamicType', False)
int_15734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 51), 'int')
# Processing the call keyword arguments (line 603)
kwargs_15735 = {}
# Getting the type of 'HasMember' (line 603)
HasMember_15731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 15), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 603)
HasMember_call_result_15736 = invoke(stypy.reporting.localization.Localization(__file__, 603, 15), HasMember_15731, *[str_15732, DynamicType_15733, int_15734], **kwargs_15735)

# Assigning a type to the variable 'Has__class__' (line 603)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 0), 'Has__class__', HasMember_call_result_15736)

# Assigning a Call to a Name (line 607):

# Assigning a Call to a Name (line 607):

# Call to HasMember(...): (line 607)
# Processing the call arguments (line 607)
str_15738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 23), 'str', '__len__')
# Getting the type of 'int' (line 607)
int_15739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 34), 'int', False)
int_15740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 39), 'int')
# Processing the call keyword arguments (line 607)
kwargs_15741 = {}
# Getting the type of 'HasMember' (line 607)
HasMember_15737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 13), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 607)
HasMember_call_result_15742 = invoke(stypy.reporting.localization.Localization(__file__, 607, 13), HasMember_15737, *[str_15738, int_15739, int_15740], **kwargs_15741)

# Assigning a type to the variable 'Has__len__' (line 607)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 0), 'Has__len__', HasMember_call_result_15742)

# Assigning a Call to a Name (line 608):

# Assigning a Call to a Name (line 608):

# Call to HasMember(...): (line 608)
# Processing the call arguments (line 608)
str_15744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 27), 'str', '__getitem__')
# Getting the type of 'DynamicType' (line 608)
DynamicType_15745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 42), 'DynamicType', False)
int_15746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 55), 'int')
# Processing the call keyword arguments (line 608)
kwargs_15747 = {}
# Getting the type of 'HasMember' (line 608)
HasMember_15743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 17), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 608)
HasMember_call_result_15748 = invoke(stypy.reporting.localization.Localization(__file__, 608, 17), HasMember_15743, *[str_15744, DynamicType_15745, int_15746], **kwargs_15747)

# Assigning a type to the variable 'Has__getitem__' (line 608)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 0), 'Has__getitem__', HasMember_call_result_15748)

# Assigning a Call to a Name (line 609):

# Assigning a Call to a Name (line 609):

# Call to HasMember(...): (line 609)
# Processing the call arguments (line 609)
str_15750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 27), 'str', '__setitem__')
# Getting the type of 'types' (line 609)
types_15751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 42), 'types', False)
# Obtaining the member 'NoneType' of a type (line 609)
NoneType_15752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 42), types_15751, 'NoneType')
int_15753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 58), 'int')
# Processing the call keyword arguments (line 609)
kwargs_15754 = {}
# Getting the type of 'HasMember' (line 609)
HasMember_15749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 17), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 609)
HasMember_call_result_15755 = invoke(stypy.reporting.localization.Localization(__file__, 609, 17), HasMember_15749, *[str_15750, NoneType_15752, int_15753], **kwargs_15754)

# Assigning a type to the variable 'Has__setitem__' (line 609)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 0), 'Has__setitem__', HasMember_call_result_15755)

# Assigning a Call to a Name (line 611):

# Assigning a Call to a Name (line 611):

# Call to HasMember(...): (line 611)
# Processing the call arguments (line 611)
str_15757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 27), 'str', '__delitem__')
# Getting the type of 'int' (line 611)
int_15758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 42), 'int', False)
int_15759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 47), 'int')
# Processing the call keyword arguments (line 611)
kwargs_15760 = {}
# Getting the type of 'HasMember' (line 611)
HasMember_15756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 17), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 611)
HasMember_call_result_15761 = invoke(stypy.reporting.localization.Localization(__file__, 611, 17), HasMember_15756, *[str_15757, int_15758, int_15759], **kwargs_15760)

# Assigning a type to the variable 'Has__delitem__' (line 611)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 0), 'Has__delitem__', HasMember_call_result_15761)

# Assigning a Call to a Name (line 612):

# Assigning a Call to a Name (line 612):

# Call to HasMember(...): (line 612)
# Processing the call arguments (line 612)
str_15763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 24), 'str', '__iter__')
# Getting the type of 'DynamicType' (line 612)
DynamicType_15764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 36), 'DynamicType', False)
int_15765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 49), 'int')
# Processing the call keyword arguments (line 612)
kwargs_15766 = {}
# Getting the type of 'HasMember' (line 612)
HasMember_15762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 14), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 612)
HasMember_call_result_15767 = invoke(stypy.reporting.localization.Localization(__file__, 612, 14), HasMember_15762, *[str_15763, DynamicType_15764, int_15765], **kwargs_15766)

# Assigning a type to the variable 'Has__iter__' (line 612)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 0), 'Has__iter__', HasMember_call_result_15767)

# Assigning a Call to a Name (line 613):

# Assigning a Call to a Name (line 613):

# Call to HasMember(...): (line 613)
# Processing the call arguments (line 613)
str_15769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 28), 'str', '__reversed__')
# Getting the type of 'int' (line 613)
int_15770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 44), 'int', False)
int_15771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 49), 'int')
# Processing the call keyword arguments (line 613)
kwargs_15772 = {}
# Getting the type of 'HasMember' (line 613)
HasMember_15768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 613)
HasMember_call_result_15773 = invoke(stypy.reporting.localization.Localization(__file__, 613, 18), HasMember_15768, *[str_15769, int_15770, int_15771], **kwargs_15772)

# Assigning a type to the variable 'Has__reversed__' (line 613)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 0), 'Has__reversed__', HasMember_call_result_15773)

# Assigning a Call to a Name (line 614):

# Assigning a Call to a Name (line 614):

# Call to HasMember(...): (line 614)
# Processing the call arguments (line 614)
str_15775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 28), 'str', '__contains__')
# Getting the type of 'int' (line 614)
int_15776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 44), 'int', False)
int_15777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 49), 'int')
# Processing the call keyword arguments (line 614)
kwargs_15778 = {}
# Getting the type of 'HasMember' (line 614)
HasMember_15774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 614)
HasMember_call_result_15779 = invoke(stypy.reporting.localization.Localization(__file__, 614, 18), HasMember_15774, *[str_15775, int_15776, int_15777], **kwargs_15778)

# Assigning a type to the variable 'Has__contains__' (line 614)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 0), 'Has__contains__', HasMember_call_result_15779)

# Assigning a Call to a Name (line 615):

# Assigning a Call to a Name (line 615):

# Call to HasMember(...): (line 615)
# Processing the call arguments (line 615)
str_15781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 27), 'str', '__missing__')
# Getting the type of 'int' (line 615)
int_15782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 42), 'int', False)
int_15783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 47), 'int')
# Processing the call keyword arguments (line 615)
kwargs_15784 = {}
# Getting the type of 'HasMember' (line 615)
HasMember_15780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 17), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 615)
HasMember_call_result_15785 = invoke(stypy.reporting.localization.Localization(__file__, 615, 17), HasMember_15780, *[str_15781, int_15782, int_15783], **kwargs_15784)

# Assigning a type to the variable 'Has__missing__' (line 615)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 0), 'Has__missing__', HasMember_call_result_15785)

# Assigning a Call to a Name (line 616):

# Assigning a Call to a Name (line 616):

# Call to HasMember(...): (line 616)
# Processing the call arguments (line 616)
str_15787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 28), 'str', '__getslice__')
# Getting the type of 'DynamicType' (line 616)
DynamicType_15788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 44), 'DynamicType', False)
int_15789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 57), 'int')
# Processing the call keyword arguments (line 616)
kwargs_15790 = {}
# Getting the type of 'HasMember' (line 616)
HasMember_15786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 616)
HasMember_call_result_15791 = invoke(stypy.reporting.localization.Localization(__file__, 616, 18), HasMember_15786, *[str_15787, DynamicType_15788, int_15789], **kwargs_15790)

# Assigning a type to the variable 'Has__getslice__' (line 616)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 0), 'Has__getslice__', HasMember_call_result_15791)

# Assigning a Call to a Name (line 617):

# Assigning a Call to a Name (line 617):

# Call to HasMember(...): (line 617)
# Processing the call arguments (line 617)
str_15793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 28), 'str', '__setslice__')
# Getting the type of 'types' (line 617)
types_15794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 44), 'types', False)
# Obtaining the member 'NoneType' of a type (line 617)
NoneType_15795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 44), types_15794, 'NoneType')
int_15796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 60), 'int')
# Processing the call keyword arguments (line 617)
kwargs_15797 = {}
# Getting the type of 'HasMember' (line 617)
HasMember_15792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 617)
HasMember_call_result_15798 = invoke(stypy.reporting.localization.Localization(__file__, 617, 18), HasMember_15792, *[str_15793, NoneType_15795, int_15796], **kwargs_15797)

# Assigning a type to the variable 'Has__setslice__' (line 617)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 0), 'Has__setslice__', HasMember_call_result_15798)

# Assigning a Call to a Name (line 618):

# Assigning a Call to a Name (line 618):

# Call to HasMember(...): (line 618)
# Processing the call arguments (line 618)
str_15800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 28), 'str', '__delslice__')
# Getting the type of 'types' (line 618)
types_15801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 44), 'types', False)
# Obtaining the member 'NoneType' of a type (line 618)
NoneType_15802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 44), types_15801, 'NoneType')
int_15803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 60), 'int')
# Processing the call keyword arguments (line 618)
kwargs_15804 = {}
# Getting the type of 'HasMember' (line 618)
HasMember_15799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 618)
HasMember_call_result_15805 = invoke(stypy.reporting.localization.Localization(__file__, 618, 18), HasMember_15799, *[str_15800, NoneType_15802, int_15803], **kwargs_15804)

# Assigning a type to the variable 'Has__delslice__' (line 618)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 0), 'Has__delslice__', HasMember_call_result_15805)

# Assigning a Call to a Name (line 619):

# Assigning a Call to a Name (line 619):

# Call to HasMember(...): (line 619)
# Processing the call arguments (line 619)
str_15807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 22), 'str', 'next')
# Getting the type of 'DynamicType' (line 619)
DynamicType_15808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 30), 'DynamicType', False)
int_15809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 43), 'int')
# Processing the call keyword arguments (line 619)
kwargs_15810 = {}
# Getting the type of 'HasMember' (line 619)
HasMember_15806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 12), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 619)
HasMember_call_result_15811 = invoke(stypy.reporting.localization.Localization(__file__, 619, 12), HasMember_15806, *[str_15807, DynamicType_15808, int_15809], **kwargs_15810)

# Assigning a type to the variable 'Has__next' (line 619)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 0), 'Has__next', HasMember_call_result_15811)

# Assigning a Call to a Name (line 622):

# Assigning a Call to a Name (line 622):

# Call to HasMember(...): (line 622)
# Processing the call arguments (line 622)
str_15813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 25), 'str', '__enter__')
# Getting the type of 'int' (line 622)
int_15814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 38), 'int', False)
int_15815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 43), 'int')
# Processing the call keyword arguments (line 622)
kwargs_15816 = {}
# Getting the type of 'HasMember' (line 622)
HasMember_15812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 15), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 622)
HasMember_call_result_15817 = invoke(stypy.reporting.localization.Localization(__file__, 622, 15), HasMember_15812, *[str_15813, int_15814, int_15815], **kwargs_15816)

# Assigning a type to the variable 'Has__enter__' (line 622)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 0), 'Has__enter__', HasMember_call_result_15817)

# Assigning a Call to a Name (line 623):

# Assigning a Call to a Name (line 623):

# Call to HasMember(...): (line 623)
# Processing the call arguments (line 623)
str_15819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 24), 'str', '__exit__')
# Getting the type of 'int' (line 623)
int_15820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 36), 'int', False)
int_15821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 41), 'int')
# Processing the call keyword arguments (line 623)
kwargs_15822 = {}
# Getting the type of 'HasMember' (line 623)
HasMember_15818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 14), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 623)
HasMember_call_result_15823 = invoke(stypy.reporting.localization.Localization(__file__, 623, 14), HasMember_15818, *[str_15819, int_15820, int_15821], **kwargs_15822)

# Assigning a type to the variable 'Has__exit__' (line 623)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 0), 'Has__exit__', HasMember_call_result_15823)

# Assigning a Call to a Name (line 626):

# Assigning a Call to a Name (line 626):

# Call to HasMember(...): (line 626)
# Processing the call arguments (line 626)
str_15825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 23), 'str', '__get__')
# Getting the type of 'DynamicType' (line 626)
DynamicType_15826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 34), 'DynamicType', False)
int_15827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 47), 'int')
# Processing the call keyword arguments (line 626)
kwargs_15828 = {}
# Getting the type of 'HasMember' (line 626)
HasMember_15824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 13), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 626)
HasMember_call_result_15829 = invoke(stypy.reporting.localization.Localization(__file__, 626, 13), HasMember_15824, *[str_15825, DynamicType_15826, int_15827], **kwargs_15828)

# Assigning a type to the variable 'Has__get__' (line 626)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 0), 'Has__get__', HasMember_call_result_15829)

# Assigning a Call to a Name (line 627):

# Assigning a Call to a Name (line 627):

# Call to HasMember(...): (line 627)
# Processing the call arguments (line 627)
str_15831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 23), 'str', '__set__')
# Getting the type of 'types' (line 627)
types_15832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 34), 'types', False)
# Obtaining the member 'NoneType' of a type (line 627)
NoneType_15833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 34), types_15832, 'NoneType')
int_15834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 50), 'int')
# Processing the call keyword arguments (line 627)
kwargs_15835 = {}
# Getting the type of 'HasMember' (line 627)
HasMember_15830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 13), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 627)
HasMember_call_result_15836 = invoke(stypy.reporting.localization.Localization(__file__, 627, 13), HasMember_15830, *[str_15831, NoneType_15833, int_15834], **kwargs_15835)

# Assigning a type to the variable 'Has__set__' (line 627)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 0), 'Has__set__', HasMember_call_result_15836)

# Assigning a Call to a Name (line 628):

# Assigning a Call to a Name (line 628):

# Call to HasMember(...): (line 628)
# Processing the call arguments (line 628)
str_15838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 23), 'str', '__del__')
# Getting the type of 'types' (line 628)
types_15839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 34), 'types', False)
# Obtaining the member 'NoneType' of a type (line 628)
NoneType_15840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 34), types_15839, 'NoneType')
int_15841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 50), 'int')
# Processing the call keyword arguments (line 628)
kwargs_15842 = {}
# Getting the type of 'HasMember' (line 628)
HasMember_15837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 13), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 628)
HasMember_call_result_15843 = invoke(stypy.reporting.localization.Localization(__file__, 628, 13), HasMember_15837, *[str_15838, NoneType_15840, int_15841], **kwargs_15842)

# Assigning a type to the variable 'Has__del__' (line 628)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 0), 'Has__del__', HasMember_call_result_15843)

# Assigning a Call to a Name (line 632):

# Assigning a Call to a Name (line 632):

# Call to HasMember(...): (line 632)
# Processing the call arguments (line 632)
str_15845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 24), 'str', '__copy__')
# Getting the type of 'DynamicType' (line 632)
DynamicType_15846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 36), 'DynamicType', False)
int_15847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 49), 'int')
# Processing the call keyword arguments (line 632)
kwargs_15848 = {}
# Getting the type of 'HasMember' (line 632)
HasMember_15844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 14), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 632)
HasMember_call_result_15849 = invoke(stypy.reporting.localization.Localization(__file__, 632, 14), HasMember_15844, *[str_15845, DynamicType_15846, int_15847], **kwargs_15848)

# Assigning a type to the variable 'Has__copy__' (line 632)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 0), 'Has__copy__', HasMember_call_result_15849)

# Assigning a Call to a Name (line 633):

# Assigning a Call to a Name (line 633):

# Call to HasMember(...): (line 633)
# Processing the call arguments (line 633)
str_15851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 28), 'str', '__deepcopy__')
# Getting the type of 'DynamicType' (line 633)
DynamicType_15852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 44), 'DynamicType', False)
int_15853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 57), 'int')
# Processing the call keyword arguments (line 633)
kwargs_15854 = {}
# Getting the type of 'HasMember' (line 633)
HasMember_15850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 633)
HasMember_call_result_15855 = invoke(stypy.reporting.localization.Localization(__file__, 633, 18), HasMember_15850, *[str_15851, DynamicType_15852, int_15853], **kwargs_15854)

# Assigning a type to the variable 'Has__deepcopy__' (line 633)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 0), 'Has__deepcopy__', HasMember_call_result_15855)

# Assigning a Call to a Name (line 636):

# Assigning a Call to a Name (line 636):

# Call to HasMember(...): (line 636)
# Processing the call arguments (line 636)
str_15857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 31), 'str', '__getinitargs__')
# Getting the type of 'DynamicType' (line 636)
DynamicType_15858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 50), 'DynamicType', False)
int_15859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 63), 'int')
# Processing the call keyword arguments (line 636)
kwargs_15860 = {}
# Getting the type of 'HasMember' (line 636)
HasMember_15856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 21), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 636)
HasMember_call_result_15861 = invoke(stypy.reporting.localization.Localization(__file__, 636, 21), HasMember_15856, *[str_15857, DynamicType_15858, int_15859], **kwargs_15860)

# Assigning a type to the variable 'Has__getinitargs__' (line 636)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 0), 'Has__getinitargs__', HasMember_call_result_15861)

# Assigning a Call to a Name (line 637):

# Assigning a Call to a Name (line 637):

# Call to HasMember(...): (line 637)
# Processing the call arguments (line 637)
str_15863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 30), 'str', '__getnewargs__')
# Getting the type of 'DynamicType' (line 637)
DynamicType_15864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 48), 'DynamicType', False)
int_15865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 61), 'int')
# Processing the call keyword arguments (line 637)
kwargs_15866 = {}
# Getting the type of 'HasMember' (line 637)
HasMember_15862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 20), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 637)
HasMember_call_result_15867 = invoke(stypy.reporting.localization.Localization(__file__, 637, 20), HasMember_15862, *[str_15863, DynamicType_15864, int_15865], **kwargs_15866)

# Assigning a type to the variable 'Has__getnewargs__' (line 637)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 0), 'Has__getnewargs__', HasMember_call_result_15867)

# Assigning a Call to a Name (line 638):

# Assigning a Call to a Name (line 638):

# Call to HasMember(...): (line 638)
# Processing the call arguments (line 638)
str_15869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 28), 'str', '__getstate__')
# Getting the type of 'DynamicType' (line 638)
DynamicType_15870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 44), 'DynamicType', False)
int_15871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 57), 'int')
# Processing the call keyword arguments (line 638)
kwargs_15872 = {}
# Getting the type of 'HasMember' (line 638)
HasMember_15868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 638)
HasMember_call_result_15873 = invoke(stypy.reporting.localization.Localization(__file__, 638, 18), HasMember_15868, *[str_15869, DynamicType_15870, int_15871], **kwargs_15872)

# Assigning a type to the variable 'Has__getstate__' (line 638)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 0), 'Has__getstate__', HasMember_call_result_15873)

# Assigning a Call to a Name (line 639):

# Assigning a Call to a Name (line 639):

# Call to HasMember(...): (line 639)
# Processing the call arguments (line 639)
str_15875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 28), 'str', '__setstate__')
# Getting the type of 'types' (line 639)
types_15876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 44), 'types', False)
# Obtaining the member 'NoneType' of a type (line 639)
NoneType_15877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 44), types_15876, 'NoneType')
int_15878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 60), 'int')
# Processing the call keyword arguments (line 639)
kwargs_15879 = {}
# Getting the type of 'HasMember' (line 639)
HasMember_15874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 18), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 639)
HasMember_call_result_15880 = invoke(stypy.reporting.localization.Localization(__file__, 639, 18), HasMember_15874, *[str_15875, NoneType_15877, int_15878], **kwargs_15879)

# Assigning a type to the variable 'Has__setstate__' (line 639)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 0), 'Has__setstate__', HasMember_call_result_15880)

# Assigning a Call to a Name (line 640):

# Assigning a Call to a Name (line 640):

# Call to HasMember(...): (line 640)
# Processing the call arguments (line 640)
str_15882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 26), 'str', '__reduce__')
# Getting the type of 'DynamicType' (line 640)
DynamicType_15883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 40), 'DynamicType', False)
int_15884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 53), 'int')
# Processing the call keyword arguments (line 640)
kwargs_15885 = {}
# Getting the type of 'HasMember' (line 640)
HasMember_15881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 16), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 640)
HasMember_call_result_15886 = invoke(stypy.reporting.localization.Localization(__file__, 640, 16), HasMember_15881, *[str_15882, DynamicType_15883, int_15884], **kwargs_15885)

# Assigning a type to the variable 'Has__reduce__' (line 640)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 0), 'Has__reduce__', HasMember_call_result_15886)

# Assigning a Call to a Name (line 641):

# Assigning a Call to a Name (line 641):

# Call to HasMember(...): (line 641)
# Processing the call arguments (line 641)
str_15888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 29), 'str', '__reduce_ex__')
# Getting the type of 'DynamicType' (line 641)
DynamicType_15889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 46), 'DynamicType', False)
int_15890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 59), 'int')
# Processing the call keyword arguments (line 641)
kwargs_15891 = {}
# Getting the type of 'HasMember' (line 641)
HasMember_15887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 19), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 641)
HasMember_call_result_15892 = invoke(stypy.reporting.localization.Localization(__file__, 641, 19), HasMember_15887, *[str_15888, DynamicType_15889, int_15890], **kwargs_15891)

# Assigning a type to the variable 'Has__reduce_ex__' (line 641)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 0), 'Has__reduce_ex__', HasMember_call_result_15892)

# Assigning a Call to a Name (line 645):

# Assigning a Call to a Name (line 645):

# Call to DynamicType(...): (line 645)
# Processing the call keyword arguments (line 645)
kwargs_15894 = {}
# Getting the type of 'DynamicType' (line 645)
DynamicType_15893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 10), 'DynamicType', False)
# Calling DynamicType(args, kwargs) (line 645)
DynamicType_call_result_15895 = invoke(stypy.reporting.localization.Localization(__file__, 645, 10), DynamicType_15893, *[], **kwargs_15894)

# Assigning a type to the variable 'AnyType' (line 645)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 0), 'AnyType', DynamicType_call_result_15895)

# Assigning a Call to a Name (line 646):

# Assigning a Call to a Name (line 646):

# Call to SupportsStructuralIntercession(...): (line 646)
# Processing the call keyword arguments (line 646)
kwargs_15897 = {}
# Getting the type of 'SupportsStructuralIntercession' (line 646)
SupportsStructuralIntercession_15896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 29), 'SupportsStructuralIntercession', False)
# Calling SupportsStructuralIntercession(args, kwargs) (line 646)
SupportsStructuralIntercession_call_result_15898 = invoke(stypy.reporting.localization.Localization(__file__, 646, 29), SupportsStructuralIntercession_15896, *[], **kwargs_15897)

# Assigning a type to the variable 'StructuralIntercessionType' (line 646)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 0), 'StructuralIntercessionType', SupportsStructuralIntercession_call_result_15898)

# Assigning a Call to a Name (line 649):

# Assigning a Call to a Name (line 649):

# Call to IsHashable(...): (line 649)
# Processing the call keyword arguments (line 649)
kwargs_15900 = {}
# Getting the type of 'IsHashable' (line 649)
IsHashable_15899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 11), 'IsHashable', False)
# Calling IsHashable(args, kwargs) (line 649)
IsHashable_call_result_15901 = invoke(stypy.reporting.localization.Localization(__file__, 649, 11), IsHashable_15899, *[], **kwargs_15900)

# Assigning a type to the variable 'Hashable' (line 649)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 0), 'Hashable', IsHashable_call_result_15901)

# Assigning a Call to a Name (line 650):

# Assigning a Call to a Name (line 650):

# Call to TypeObject(...): (line 650)
# Processing the call keyword arguments (line 650)
kwargs_15903 = {}
# Getting the type of 'TypeObject' (line 650)
TypeObject_15902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 7), 'TypeObject', False)
# Calling TypeObject(args, kwargs) (line 650)
TypeObject_call_result_15904 = invoke(stypy.reporting.localization.Localization(__file__, 650, 7), TypeObject_15902, *[], **kwargs_15903)

# Assigning a type to the variable 'Type' (line 650)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 0), 'Type', TypeObject_call_result_15904)

# Assigning a Call to a Name (line 651):

# Assigning a Call to a Name (line 651):

# Call to InstanceOfType(...): (line 651)
# Processing the call keyword arguments (line 651)
kwargs_15906 = {}
# Getting the type of 'InstanceOfType' (line 651)
InstanceOfType_15905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 15), 'InstanceOfType', False)
# Calling InstanceOfType(args, kwargs) (line 651)
InstanceOfType_call_result_15907 = invoke(stypy.reporting.localization.Localization(__file__, 651, 15), InstanceOfType_15905, *[], **kwargs_15906)

# Assigning a type to the variable 'TypeInstance' (line 651)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 0), 'TypeInstance', InstanceOfType_call_result_15907)

# Assigning a Call to a Name (line 652):

# Assigning a Call to a Name (line 652):

# Call to VarArgType(...): (line 652)
# Processing the call keyword arguments (line 652)
kwargs_15909 = {}
# Getting the type of 'VarArgType' (line 652)
VarArgType_15908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 10), 'VarArgType', False)
# Calling VarArgType(args, kwargs) (line 652)
VarArgType_call_result_15910 = invoke(stypy.reporting.localization.Localization(__file__, 652, 10), VarArgType_15908, *[], **kwargs_15909)

# Assigning a type to the variable 'VarArgs' (line 652)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 0), 'VarArgs', VarArgType_call_result_15910)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
