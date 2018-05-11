
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # -----------
2: # Union types
3: # -----------
4: 
5: import copy
6: import inspect
7: import types
8: 
9: from ..... import stypy_copy
10: import undefined_type_copy
11: from ..non_python_type_copy import NonPythonType
12: from ...python_types_copy.type_copy import Type
13: from ....errors_copy.type_error_copy import TypeError
14: from ...python_types_copy import type_inference_copy
15: from ....reporting_copy.print_utils_copy import format_function_name
16: 
17: 
18: class UnionType(NonPythonType):
19:     '''
20:     UnionType is a collection of types that represent the fact that a certain Python element can have any of the listed
21:     types in a certain point of the execution of the program. UnionTypes are created by the application of the SSA
22:     algorithm when dealing with branches in the processed program source code.
23:     '''
24: 
25:     @staticmethod
26:     def _wrap_type(type_):
27:         '''
28:         Internal method to store Python types in a TypeInferenceProxy if they are not already a TypeInferenceProxy
29:         :param type_: Any Python object
30:         :return:
31:         '''
32:         if not isinstance(type_, Type):
33:             ret_type = type_inference_copy.type_inference_proxy.TypeInferenceProxy.instance(type_)
34: 
35:             if not ret_type.is_type_instance():
36:                 ret_type.set_type_instance(True)
37:             return ret_type
38:         else:
39:             # At least the Type instance has a value for this property, we set if to true
40:             if not type_.has_type_instance_value():
41:                 type_.set_type_instance(True)
42: 
43:         return type_
44: 
45:     # ############################### UNION TYPE CREATION ################################
46: 
47:     def __init__(self, type1=None, type2=None):
48:         '''
49:         Creates a new UnionType, optionally adding the passed parameters. If only a type is passed, this type
50:         is returned instead
51:         :param type1: Optional type to add. It can be another union type.
52:         :param type2: Optional type to add . It cannot be another union type
53:         :return:
54:         '''
55:         self.types = []
56: 
57:         # If the first type is a UnionType, add all its types to the newly created union type
58:         if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_union_type(type1):
59:             for type_ in type1.types:
60:                 self.types.append(type_)
61:             return
62: 
63:         # Append passed types, if it exist
64:         if type1 is not None:
65:             self.types.append(UnionType._wrap_type(type1))
66: 
67:         if type2 is not None:
68:             self.types.append(UnionType._wrap_type(type2))
69: 
70:     @staticmethod
71:     def create_union_type_from_types(*types):
72:         '''
73:         Utility method to create a union type from a list of types
74:         :param types: List of types
75:         :return: UnionType
76:         '''
77:         union_instance = UnionType()
78: 
79:         for type_ in types:
80:             UnionType.__add_unconditionally(union_instance, type_)
81: 
82:         if len(union_instance.types) == 1:
83:             return union_instance.types[0]
84:         return union_instance
85: 
86:     # ############################### ADD TYPES TO THE UNION ################################
87: 
88:     @staticmethod
89:     def __add_unconditionally(type1, type2):
90:         '''
91:         Helper method of create_union_type_from_types
92:         :param type1: Type to add
93:         :param type2: Type to add
94:         :return: UnionType
95:         '''
96:         if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_union_type(type1):
97:             return type1._add(UnionType._wrap_type(type2))
98:         if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_union_type(type2):
99:             return type2._add(UnionType._wrap_type(type1))
100: 
101:         if type1 == type2:
102:             return UnionType._wrap_type(type1)
103: 
104:         return UnionType(type1, type2)
105: 
106:     @staticmethod
107:     def add(type1, type2):
108:         '''
109:         Adds type1 and type2 to potentially form a UnionType, with the following rules:
110:         - If either type1 or type2 are None, the other type is returned and no UnionType is formed
111:         - If either type1 or type2 are UndefinedType, the other type is returned and no UnionType is formed
112:         - If either type1 or type2 are UnionTypes, they are mergued in a new UnionType that contains the types
113:         represented by both of them.
114:         - If both types are the same, the first is returned
115:         - Else, a new UnionType formed by the two passed types are returned.
116: 
117:         :param type1: Type to add
118:         :param type2: Type to add
119:         :return: A UnionType
120:         '''
121:         if type1 is None:
122:             return UnionType._wrap_type(type2)
123: 
124:         if type2 is None:
125:             return UnionType._wrap_type(type1)
126: 
127:         if isinstance(type1, TypeError) and isinstance(type2, TypeError):
128:             if UnionType._wrap_type(type1) == UnionType._wrap_type(type2):
129:                 return UnionType._wrap_type(type1) # Equal errors are not added
130:             else:
131:                 type1.error_msg += type2.error_msg
132:                 TypeError.remove_error_msg(type2)
133:                 return type1
134: 
135:         if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_undefined_type(type1):
136:             return UnionType._wrap_type(type2)
137:         if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_undefined_type(type2):
138:             return UnionType._wrap_type(type1)
139: 
140:         if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_union_type(type1):
141:             return type1._add(type2)
142:         if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_union_type(type2):
143:             return type2._add(type1)
144: 
145:         if UnionType._wrap_type(type1) == UnionType._wrap_type(type2):
146:             return UnionType._wrap_type(type1)
147: 
148:         return UnionType(type1, type2)
149: 
150:     def _add(self, other_type):
151:         '''
152:         Adds the passed type to the current UnionType object. If other_type is a UnionType, all its contained types
153:         are added to the current.
154:         :param other_type: Type to add
155:         :return: The self object
156:         '''
157:         if other_type is None:
158:             return self
159:         if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_union_type(other_type):
160:             for t in other_type.types:
161:                 self._add(t)
162:             return self
163: 
164:         other_type = UnionType._wrap_type(other_type)
165: 
166:         # Do the current UnionType contain the passed type, then we do not add it again
167:         for t in self.types:
168:             if t == other_type:
169:                 return self
170: 
171:         self.types.append(other_type)
172: 
173:         return self
174: 
175:     # ############################### PYTHON METHODS ################################
176: 
177:     def __repr__(self):
178:         '''
179:         Visual representation of the UnionType
180:         :return:
181:         '''
182:         return self.__str__()
183: 
184:     def __str__(self):
185:         '''
186:         Visual representation of the UnionType
187:         :return:
188:         '''
189:         the_str = ""
190:         for i in range(len(self.types)):
191:             the_str += str(self.types[i])
192:             if i < len(self.types) - 1:
193:                 the_str += " \/ "
194:         return the_str
195: 
196:     def __iter__(self):
197:         '''
198:         Iterator interface, to iterate through the contained types
199:         :return:
200:         '''
201:         for elem in self.types:
202:             yield elem
203: 
204:     def __contains__(self, item):
205:         '''
206:         The in operator, to determine if a type is inside a UnionType
207:         :param item: Type to test. If it is another UnionType and this passed UnionType types are all inside the
208:         current one, then the method returns true
209:         :return: bool
210:         '''
211:         if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_union_type(item):
212:             for elem in item:
213:                 if elem not in self.types:
214:                     return False
215:             return True
216:         else:
217:             if isinstance(item, undefined_type_copy.UndefinedType):
218:                 found = False
219:                 for elem in self.types:
220:                     if isinstance(elem, undefined_type_copy.UndefinedType):
221:                         found = True
222:                 return found
223:             else:
224:                 return item in self.types
225: 
226:     def __eq__(self, other):
227:         '''
228:         The == operator, to compare UnionTypes
229: 
230:         :param other: Another UnionType (used in type inference code) or a list of types (used in unit testing)
231:         :return: True if the passed UnionType or list contains exactly the same amount and type of types that the
232:         passed entities
233:         '''
234:         if isinstance(other, list):
235:             type_list = other
236:         else:
237:             if isinstance(other, UnionType):
238:                 type_list = other.types
239:             else:
240:                 return False
241: 
242:         if not len(self.types) == len(type_list):
243:             return False
244: 
245:         for type_ in self.types:
246:             if isinstance(type_, TypeError):
247:                 for type_2 in type_list:
248:                     if type(type_2) is TypeError:
249:                         continue
250:             if type_ not in type_list:
251:                 return False
252: 
253:         return True
254: 
255:     def __getitem__(self, item):
256:         '''
257:         The [] operator, to obtain individual types stored within the union type
258: 
259:         :param item: Indexer
260:         :return:
261:         '''
262:         return self.types[item]
263: 
264:     # ############################## MEMBER TYPE GET / SET ###############################
265: 
266:     def get_type_of_member(self, localization, member_name):
267:         '''
268:         For all the types stored in the union type, obtain the type of the member named member_name, returning a
269:         Union Type with the union of all the possible types that member_name has inside the UnionType. For example,
270:         if a UnionType has the types Class1 and Class2, both with the member "attr" so Class1.attr: int and
271:         Class2.attr: str, this method will return int \/ str.
272:         :param localization: Caller information
273:         :param member_name: Name of the member to get
274:         :return All the types that member_name could have, examining the UnionType stored types
275:         '''
276:         result = []
277: 
278:         # Add all the results of get_type_of_member for all stored typs in a list
279:         for type_ in self.types:
280:             temp = type_.get_type_of_member(localization, member_name)
281:             result.append(temp)
282: 
283:         # Count errors
284:         errors = filter(lambda t: isinstance(t, TypeError), result)
285:         # Count correct invocations
286:         types_to_return = filter(lambda t: not isinstance(t, TypeError), result)
287: 
288:         # If all types contained in the union do not have this member, the whole access is an error.
289:         if len(errors) == len(result):
290:             return TypeError(localization, "None of the possible types ('{1}') has the member '{0}'".format(
291:                 member_name, self.types))
292:         else:
293:             # If there is an error, it means that the obtained member could be undefined in one of the contained objects
294:             if len(errors) > 0:
295:                 types_to_return.append(undefined_type_copy.UndefinedType())
296: 
297:             # If not all the types return an error when accessing the members, the policy is different:
298:             # - Notified errors are turned to warnings in the general error log, as there are combinations of types
299:             # that are valid
300:             # - ErrorTypes are eliminated from the error collection.
301:             for error in errors:
302:                 error.turn_to_warning()
303: 
304:         # Calculate return type: If there is only one type, return it. If there are several types, return a UnionType
305:         # with all of them contained
306:         if len(types_to_return) == 1:
307:             return types_to_return[0]
308:         else:
309:             ret_union = None
310:             for type_ in types_to_return:
311:                 ret_union = UnionType.add(ret_union, type_)
312: 
313:             return ret_union
314: 
315:     @staticmethod
316:     def __parse_member_value(destination, member_value):
317:         '''
318:         When setting a member of a UnionType to a certain value, each one of the contained types are assigned this
319:         member with the specified value (type). However, certain values have to be carefully handled to provide valid
320:         values. For example, methods have to be handler in order to provide valid methods to add to each of the
321:         UnionType types. This helper method convert a method to a valid method belonging to the destination object.
322: 
323:         :param destination: New owner of the method
324:         :param member_value: Method
325:         :return THe passed member value, either transformed or not
326:         '''
327:         if inspect.ismethod(member_value):
328:             # Each component of the union type has to have its own method reference for model consistency
329:             met = types.MethodType(member_value.im_func, destination)
330:             return met
331: 
332:         return member_value
333: 
334:     def set_type_of_member(self, localization, member_name, member_value):
335:         '''
336:         For all the types stored in the union type, set the type of the member named member_name to the type
337:         specified in member_value. For example,
338:         if a UnionType has the types Class1 and Class2, both with the member "attr" so Class1.attr: int and
339:         Class2.attr: str, this method, if passsed a float as member_value will turn both classes "attr" to float.
340:         :param localization: Caller information
341:         :param member_name: Name of the member to set
342:         :param member_value New type of the member
343:         :return None or a TypeError if the member cannot be set. Warnings are generated if the member of some of the
344:         stored objects cannot be set
345:         '''
346: 
347:         errors = []
348: 
349:         for type_ in self.types:
350:             final_value = self.__parse_member_value(type_, member_value)
351:             temp = type_.set_type_of_member(localization, member_name, final_value)
352:             if temp is not None:
353:                 errors.append(temp)
354: 
355:         # If all types contained in the union do not have this member, the whole access is an error.
356:         if len(errors) == len(self.types):
357:             return TypeError(localization, "None of the possible types ('{1}') can set the member '{0}'".format(
358:                 member_name, self.types))
359:         else:
360:             # If not all the types return an error when accessing the members, the policy is different:
361:             # - Notified errors are turned to warnings in the general error log
362:             # - ErrorTypes are eliminated.
363:             for error in errors:
364:                 error.turn_to_warning()
365: 
366:         return None
367: 
368:     # ############################## MEMBER INVOKATION ###############################
369: 
370:     def invoke(self, localization, *args, **kwargs):
371:         '''
372:         For all the types stored in the union type, invoke them with the provided parameters.
373:         :param localization: Caller information
374:         :param args: Arguments of the call
375:         :param kwargs: Keyword arguments of the call
376:         :return All the types that the call could return, examining the UnionType stored types
377:         '''
378:         result = []
379: 
380:         for type_ in self.types:
381:             # Invoke all types
382:             temp = type_.invoke(localization, *args, **kwargs)
383:             result.append(temp)
384: 
385:         # Collect errors
386:         errors = filter(lambda t: isinstance(t, TypeError), result)
387: 
388:         # Collect returned types
389:         types_to_return = filter(lambda t: not isinstance(t, TypeError), result)
390: 
391:         # If all types contained in the union do not have this member, the whole access is an error.
392:         if len(errors) == len(result):
393:             for error in errors:
394:                 TypeError.remove_error_msg(error)
395:             params = tuple(list(args) + kwargs.values())
396:             return TypeError(localization, "Cannot invoke {0} with parameters {1}".format(
397:                 format_function_name(self.types[0].name), params))
398:         else:
399:             # If not all the types return an error when accessing the members, the policy is different:
400:             # - Notified errors are turned to warnings in the general error log
401:             # - ErrorTypes are eliminated from the error collection.
402:             for error in errors:
403:                 error.turn_to_warning()
404: 
405:         # Return type
406:         if len(types_to_return) == 1:
407:             return types_to_return[0]
408:         else:
409:             ret_union = None
410:             for type_ in types_to_return:
411:                 ret_union = UnionType.add(ret_union, type_)
412: 
413:             return ret_union
414: 
415:     # ############################## STRUCTURAL REFLECTION ###############################
416: 
417:     def delete_member(self, localization, member):
418:         '''
419:         For all the types stored in the union type, delete the member named member_name, returning None or a TypeError
420:         if no type stored in the UnionType supports member deletion.
421:         :param localization: Caller information
422:         :param member: Member to delete
423:         :return None or TypeError
424:         '''
425:         errors = []
426: 
427:         for type_ in self.types:
428:             temp = type_.delete_member(localization, member)
429:             if temp is not None:
430:                 errors.append(temp)
431: 
432:         # If all types contained in the union fail to delete this member, the whole operation is an error.
433:         if len(errors) == len(self.types):
434:             return TypeError(localization, "The member '{0}' cannot be deleted from none of the possible types ('{1}')".
435:                              format(member, self.types))
436:         else:
437:             # If not all the types return an error when accessing the members, the policy is different:
438:             # - Notified errors are turned to warnings in the general error log
439:             # - ErrorTypes are eliminated.
440:             for error in errors:
441:                 error.turn_to_warning()
442: 
443:         return None
444: 
445:     def supports_structural_reflection(self):
446:         '''
447:         Determines if at least one of the stored types supports structural reflection.
448:         '''
449:         supports = False
450: 
451:         for type_ in self.types:
452:             supports = supports or type_.supports_structural_reflection()
453: 
454:         return supports
455: 
456:     def change_type(self, localization, new_type):
457:         '''
458:         For all the types stored in the union type, change the base type to new_type, returning None or a TypeError
459:         if no type stored in the UnionType supports a type change.
460:         :param localization: Caller information
461:         :param new_type: Type to change to
462:         :return None or TypeError
463:         '''
464:         errors = []
465: 
466:         for type_ in self.types:
467:             temp = type_.change_type(localization, new_type)
468:             if temp is not None:
469:                 errors.append(temp)
470: 
471:         # If all types contained in the union do not support the operation, the whole operation is an error.
472:         if len(errors) == len(self.types):
473:             return TypeError(localization, "None of the possible types ('{1}') can be assigned a new type '{0}'".
474:                              format(new_type, self.types))
475:         else:
476:             # If not all the types return an error when changing types, the policy is different:
477:             # - Notified errors are turned to warnings in the general error log
478:             # - ErrorTypes are eliminated.
479:             for error in errors:
480:                 error.turn_to_warning()
481: 
482:         return None
483: 
484:     def change_base_types(self, localization, new_types):
485:         '''
486:         For all the types stored in the union type, change the base types to the ones contained in the list new_types,
487:         returning None or a TypeError if no type stored in the UnionType supports a supertype change.
488:         :param localization: Caller information
489:         :param new_types: Types to change its base type to
490:         :return None or TypeError
491:         '''
492:         errors = []
493: 
494:         for type_ in self.types:
495:             temp = type_.change_base_types(localization, new_types)
496:             if temp is not None:
497:                 errors.append(temp)
498: 
499:         # Is the whole operation an error?
500:         if len(errors) == len(self.types):
501:             return TypeError(localization, "None of the possible types ('{1}') can be assigned new base types '{0}'".
502:                              format(new_types, self.types))
503:         else:
504:             # If not all the types return an error when accessing the members, the policy is different:
505:             # - Notified errors are turned to warnings in the general error log
506:             # - ErrorTypes are eliminated.
507:             for error in errors:
508:                 error.turn_to_warning()
509: 
510:         return None
511: 
512:     def add_base_types(self, localization, new_types):
513:         '''
514:         For all the types stored in the union type, add to the base types the ones contained in the list new_types,
515:         returning None or a TypeError if no type stored in the UnionType supports a supertype change.
516:         :param localization: Caller information
517:         :param new_types: Types to change its base type to
518:         :return None or TypeError
519:         '''
520:         errors = []
521: 
522:         for type_ in self.types:
523:             temp = type_.change_base_types(localization, new_types)
524:             if temp is not None:
525:                 errors.append(temp)
526: 
527:         # Is the whole operation an error?
528:         if len(errors) == len(self.types):
529:             return TypeError(localization, "The base types of all the possible types ('{0}') cannot be modified".
530:                              format(self.types))
531:         else:
532:             # If not all the types return an error when accessing the members, the policy is different:
533:             # - Notified errors are turned to warnings in the general error log
534:             # - ErrorTypes are eliminated.
535:             for error in errors:
536:                 error.turn_to_warning()
537: 
538:         return None
539: 
540:     # ############################## TYPE CLONING ###############################
541: 
542:     def clone(self):
543:         '''
544:         Clone the whole UnionType and its contained types
545:         '''
546:         result_union = self.types[0].clone()
547:         for i in range(1, len(self.types)):
548:             if isinstance(self.types[i], Type):
549:                 result_union = UnionType.add(result_union, self.types[i].clone())
550:             else:
551:                 result_union = UnionType.add(result_union, copy.deepcopy(self.types[i]))
552: 
553:         return result_union
554: 
555:     def can_store_elements(self):
556:         temp = False
557:         for type_ in self.types:
558:             temp |= type_.can_store_elements()
559: 
560:         return temp
561: 
562:     def can_store_keypairs(self):
563:         temp = False
564:         for type_ in self.types:
565:             temp |= type_.can_store_keypairs()
566: 
567:         return temp
568: 
569:     def get_elements_type(self):
570:         errors = []
571: 
572:         temp = None
573:         for type_ in self.types:
574:             res = type_.get_elements_type()
575:             if isinstance(res, TypeError):
576:                 errors.append(temp)
577:             else:
578:                 temp = UnionType.add(temp, res)
579: 
580:         # If all types contained in the union do not have this member, the whole access is an error.
581:         if len(errors) == len(self.types):
582:             return TypeError(None, "None of the possible types ('{1}') can invoke the member '{0}'".format(
583:                 "get_elements_type", self.types))
584:         else:
585:             # If not all the types return an error when accessing the members, the policy is different:
586:             # - Notified errors are turned to warnings in the general error log
587:             # - ErrorTypes are eliminated.
588:             for error in errors:
589:                 error.turn_to_warning()
590: 
591:         return temp
592: 
593:     def set_elements_type(self, localization, elements_type, record_annotation=True):
594:         errors = []
595: 
596:         temp = None
597:         for type_ in self.types:
598:             res = type_.set_elements_type(localization, elements_type, record_annotation)
599:             if isinstance(res, TypeError):
600:                 errors.append(temp)
601: 
602:         # If all types contained in the union do not have this member, the whole access is an error.
603:         if len(errors) == len(self.types):
604:             return TypeError(localization, "None of the possible types ('{1}') can invoke the member '{0}'".format(
605:                 "set_elements_type", self.types))
606:         else:
607:             # If not all the types return an error when accessing the members, the policy is different:
608:             # - Notified errors are turned to warnings in the general error log
609:             # - ErrorTypes are eliminated.
610:             for error in errors:
611:                 error.turn_to_warning()
612: 
613:         return temp
614: 
615:     def add_type(self, localization, type_, record_annotation=True):
616:         errors = []
617: 
618:         temp = None
619:         for type_ in self.types:
620:             res = type_.add_type(localization, type_, record_annotation)
621:             if isinstance(res, TypeError):
622:                 errors.append(temp)
623: 
624:         # If all types contained in the union do not have this member, the whole access is an error.
625:         if len(errors) == len(self.types):
626:             return TypeError(localization, "None of the possible types ('{1}') can invoke the member '{0}'".format(
627:                 "add_type", self.types))
628:         else:
629:             # If not all the types return an error when accessing the members, the policy is different:
630:             # - Notified errors are turned to warnings in the general error log
631:             # - ErrorTypes are eliminated.
632:             for error in errors:
633:                 error.turn_to_warning()
634: 
635:         return temp
636: 
637:     def add_types_from_list(self, localization, type_list, record_annotation=True):
638:         errors = []
639: 
640:         temp = None
641:         for type_ in self.types:
642:             res = type_.add_types_from_list(localization, type_list, record_annotation)
643:             if isinstance(res, TypeError):
644:                 errors.append(temp)
645: 
646:         # If all types contained in the union do not have this member, the whole access is an error.
647:         if len(errors) == len(self.types):
648:             return TypeError(localization, "None of the possible types ('{1}') can invoke the member '{0}'".format(
649:                 "add_types_from_list", self.types))
650:         else:
651:             # If not all the types return an error when accessing the members, the policy is different:
652:             # - Notified errors are turned to warnings in the general error log
653:             # - ErrorTypes are eliminated.
654:             for error in errors:
655:                 error.turn_to_warning()
656: 
657:         return temp
658: 
659:     def get_values_from_key(self, localization, key):
660:         errors = []
661: 
662:         temp = None
663:         for type_ in self.types:
664:             res = type_.get_values_from_key(localization, key)
665:             if isinstance(res, TypeError):
666:                 errors.append(temp)
667:             else:
668:                 temp = UnionType.add(temp, res)
669: 
670:         # If all types contained in the union do not have this member, the whole access is an error.
671:         if len(errors) == len(self.types):
672:             return TypeError(localization, "None of the possible types ('{1}') can invoke the member '{0}'".format(
673:                 "get_values_from_key", self.types))
674:         else:
675:             # If not all the types return an error when accessing the members, the policy is different:
676:             # - Notified errors are turned to warnings in the general error log
677:             # - ErrorTypes are eliminated.
678:             for error in errors:
679:                 error.turn_to_warning()
680: 
681:         return temp
682: 
683:     def add_key_and_value_type(self, localization, type_tuple, record_annotation=True):
684:         errors = []
685: 
686:         for type_ in self.types:
687:             temp = type_.add_key_and_value_type(localization, type_tuple, record_annotation)
688:             if temp is not None:
689:                 errors.append(temp)
690: 
691:         # If all types contained in the union do not have this member, the whole access is an error.
692:         if len(errors) == len(self.types):
693:             return TypeError(localization, "None of the possible types ('{1}') can invoke the member '{0}'".format(
694:                 "add_key_and_value_type", self.types))
695:         else:
696:             # If not all the types return an error when accessing the members, the policy is different:
697:             # - Notified errors are turned to warnings in the general error log
698:             # - ErrorTypes are eliminated.
699:             for error in errors:
700:                 error.turn_to_warning()
701: 
702:         return None
703: 
704: 
705: class OrderedUnionType(UnionType):
706:     '''
707:     A special type of UnionType that maintain the order of its added types and admits repeated elements. This will be
708:     used in the future implementation of tuples.
709:     '''
710: 
711:     def __init__(self, type1=None, type2=None):
712:         UnionType.__init__(self, type1, type2)
713:         self.ordered_types = []
714: 
715:         if type1 is not None:
716:             self.ordered_types.append(type1)
717: 
718:         if type2 is not None:
719:             self.ordered_types.append(type2)
720: 
721:     @staticmethod
722:     def add(type1, type2):
723:         if type1 is None:
724:             return UnionType._wrap_type(type2)
725: 
726:         if type2 is None:
727:             return UnionType._wrap_type(type1)
728: 
729:         if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_undefined_type(type1):
730:             return UnionType._wrap_type(type2)
731:         if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_undefined_type(type2):
732:             return UnionType._wrap_type(type1)
733: 
734:         if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_union_type(type1):
735:             return type1._add(type2)
736:         if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_union_type(type2):
737:             return type2._add(type1)
738: 
739:         if UnionType._wrap_type(type1) == UnionType._wrap_type(type2):
740:             return UnionType._wrap_type(type1)
741: 
742:         return OrderedUnionType(type1, type2)
743: 
744:     def _add(self, other_type):
745:         ret = UnionType._add(self, other_type)
746:         self.ordered_types.append(other_type)
747:         return ret
748: 
749:     def get_ordered_types(self):
750:         '''
751:         Obtain the stored types in the same order they were added, including repetitions
752:         :return:
753:         '''
754:         return self.ordered_types
755: 
756:     def clone(self):
757:         '''
758:         Clone the whole OrderedUnionType and its contained types
759:         '''
760:         result_union = self.types[0].clone()
761:         for i in range(1, len(self.types)):
762:             if isinstance(self.types[i], Type):
763:                 result_union = OrderedUnionType.add(result_union, self.types[i].clone())
764:             else:
765:                 result_union = OrderedUnionType.add(result_union, copy.deepcopy(self.types[i]))
766: 
767:         return result_union
768: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import copy' statement (line 5)
import copy

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'copy', copy, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import inspect' statement (line 6)
import inspect

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'inspect', inspect, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import types' statement (line 7)
import types

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'types', types, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from testing.test_programs.stypy_code_copy import stypy_copy' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_12587 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy')

if (type(import_12587) is not StypyTypeError):

    if (import_12587 != 'pyd_module'):
        __import__(import_12587)
        sys_modules_12588 = sys.modules[import_12587]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy', sys_modules_12588.module_type_store, module_type_store, ['stypy_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_12588, sys_modules_12588.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy import stypy_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy', None, module_type_store, ['stypy_copy'], [stypy_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy', import_12587)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import undefined_type_copy' statement (line 10)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_12589 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'undefined_type_copy')

if (type(import_12589) is not StypyTypeError):

    if (import_12589 != 'pyd_module'):
        __import__(import_12589)
        sys_modules_12590 = sys.modules[import_12589]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'undefined_type_copy', sys_modules_12590.module_type_store, module_type_store)
    else:
        import undefined_type_copy

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'undefined_type_copy', undefined_type_copy, module_type_store)

else:
    # Assigning a type to the variable 'undefined_type_copy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'undefined_type_copy', import_12589)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy import NonPythonType' statement (line 11)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_12591 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy')

if (type(import_12591) is not StypyTypeError):

    if (import_12591 != 'pyd_module'):
        __import__(import_12591)
        sys_modules_12592 = sys.modules[import_12591]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy', sys_modules_12592.module_type_store, module_type_store, ['NonPythonType'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_12592, sys_modules_12592.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy import NonPythonType

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy', None, module_type_store, ['NonPythonType'], [NonPythonType])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy', import_12591)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy import Type' statement (line 12)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_12593 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy')

if (type(import_12593) is not StypyTypeError):

    if (import_12593 != 'pyd_module'):
        __import__(import_12593)
        sys_modules_12594 = sys.modules[import_12593]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy', sys_modules_12594.module_type_store, module_type_store, ['Type'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_12594, sys_modules_12594.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy import Type

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy', None, module_type_store, ['Type'], [Type])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy', import_12593)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 13)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_12595 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy')

if (type(import_12595) is not StypyTypeError):

    if (import_12595 != 'pyd_module'):
        __import__(import_12595)
        sys_modules_12596 = sys.modules[import_12595]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', sys_modules_12596.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_12596, sys_modules_12596.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', import_12595)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy import type_inference_copy' statement (line 14)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_12597 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy')

if (type(import_12597) is not StypyTypeError):

    if (import_12597 != 'pyd_module'):
        __import__(import_12597)
        sys_modules_12598 = sys.modules[import_12597]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy', sys_modules_12598.module_type_store, module_type_store, ['type_inference_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_12598, sys_modules_12598.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy import type_inference_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy', None, module_type_store, ['type_inference_copy'], [type_inference_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy', import_12597)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy.print_utils_copy import format_function_name' statement (line 15)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_12599 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy.print_utils_copy')

if (type(import_12599) is not StypyTypeError):

    if (import_12599 != 'pyd_module'):
        __import__(import_12599)
        sys_modules_12600 = sys.modules[import_12599]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy.print_utils_copy', sys_modules_12600.module_type_store, module_type_store, ['format_function_name'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_12600, sys_modules_12600.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy.print_utils_copy import format_function_name

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy.print_utils_copy', None, module_type_store, ['format_function_name'], [format_function_name])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy.print_utils_copy' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy.print_utils_copy', import_12599)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

# Declaration of the 'UnionType' class
# Getting the type of 'NonPythonType' (line 18)
NonPythonType_12601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 16), 'NonPythonType')

class UnionType(NonPythonType_12601, ):
    str_12602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, (-1)), 'str', '\n    UnionType is a collection of types that represent the fact that a certain Python element can have any of the listed\n    types in a certain point of the execution of the program. UnionTypes are created by the application of the SSA\n    algorithm when dealing with branches in the processed program source code.\n    ')

    @staticmethod
    @norecursion
    def _wrap_type(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_wrap_type'
        module_type_store = module_type_store.open_function_context('_wrap_type', 25, 4, False)
        
        # Passed parameters checking function
        UnionType._wrap_type.__dict__.__setitem__('stypy_localization', localization)
        UnionType._wrap_type.__dict__.__setitem__('stypy_type_of_self', None)
        UnionType._wrap_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType._wrap_type.__dict__.__setitem__('stypy_function_name', '_wrap_type')
        UnionType._wrap_type.__dict__.__setitem__('stypy_param_names_list', ['type_'])
        UnionType._wrap_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType._wrap_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType._wrap_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType._wrap_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType._wrap_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType._wrap_type.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, '_wrap_type', ['type_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_wrap_type', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_wrap_type(...)' code ##################

        str_12603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, (-1)), 'str', '\n        Internal method to store Python types in a TypeInferenceProxy if they are not already a TypeInferenceProxy\n        :param type_: Any Python object\n        :return:\n        ')
        
        
        # Call to isinstance(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'type_' (line 32)
        type__12605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 26), 'type_', False)
        # Getting the type of 'Type' (line 32)
        Type_12606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 33), 'Type', False)
        # Processing the call keyword arguments (line 32)
        kwargs_12607 = {}
        # Getting the type of 'isinstance' (line 32)
        isinstance_12604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 32)
        isinstance_call_result_12608 = invoke(stypy.reporting.localization.Localization(__file__, 32, 15), isinstance_12604, *[type__12605, Type_12606], **kwargs_12607)
        
        # Applying the 'not' unary operator (line 32)
        result_not__12609 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 11), 'not', isinstance_call_result_12608)
        
        # Testing if the type of an if condition is none (line 32)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 32, 8), result_not__12609):
            
            
            # Call to has_type_instance_value(...): (line 40)
            # Processing the call keyword arguments (line 40)
            kwargs_12632 = {}
            # Getting the type of 'type_' (line 40)
            type__12630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'type_', False)
            # Obtaining the member 'has_type_instance_value' of a type (line 40)
            has_type_instance_value_12631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 19), type__12630, 'has_type_instance_value')
            # Calling has_type_instance_value(args, kwargs) (line 40)
            has_type_instance_value_call_result_12633 = invoke(stypy.reporting.localization.Localization(__file__, 40, 19), has_type_instance_value_12631, *[], **kwargs_12632)
            
            # Applying the 'not' unary operator (line 40)
            result_not__12634 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 15), 'not', has_type_instance_value_call_result_12633)
            
            # Testing if the type of an if condition is none (line 40)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 40, 12), result_not__12634):
                pass
            else:
                
                # Testing the type of an if condition (line 40)
                if_condition_12635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 12), result_not__12634)
                # Assigning a type to the variable 'if_condition_12635' (line 40)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'if_condition_12635', if_condition_12635)
                # SSA begins for if statement (line 40)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to set_type_instance(...): (line 41)
                # Processing the call arguments (line 41)
                # Getting the type of 'True' (line 41)
                True_12638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 40), 'True', False)
                # Processing the call keyword arguments (line 41)
                kwargs_12639 = {}
                # Getting the type of 'type_' (line 41)
                type__12636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'type_', False)
                # Obtaining the member 'set_type_instance' of a type (line 41)
                set_type_instance_12637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 16), type__12636, 'set_type_instance')
                # Calling set_type_instance(args, kwargs) (line 41)
                set_type_instance_call_result_12640 = invoke(stypy.reporting.localization.Localization(__file__, 41, 16), set_type_instance_12637, *[True_12638], **kwargs_12639)
                
                # SSA join for if statement (line 40)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 32)
            if_condition_12610 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 8), result_not__12609)
            # Assigning a type to the variable 'if_condition_12610' (line 32)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'if_condition_12610', if_condition_12610)
            # SSA begins for if statement (line 32)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 33):
            
            # Call to instance(...): (line 33)
            # Processing the call arguments (line 33)
            # Getting the type of 'type_' (line 33)
            type__12615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 92), 'type_', False)
            # Processing the call keyword arguments (line 33)
            kwargs_12616 = {}
            # Getting the type of 'type_inference_copy' (line 33)
            type_inference_copy_12611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 23), 'type_inference_copy', False)
            # Obtaining the member 'type_inference_proxy' of a type (line 33)
            type_inference_proxy_12612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 23), type_inference_copy_12611, 'type_inference_proxy')
            # Obtaining the member 'TypeInferenceProxy' of a type (line 33)
            TypeInferenceProxy_12613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 23), type_inference_proxy_12612, 'TypeInferenceProxy')
            # Obtaining the member 'instance' of a type (line 33)
            instance_12614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 23), TypeInferenceProxy_12613, 'instance')
            # Calling instance(args, kwargs) (line 33)
            instance_call_result_12617 = invoke(stypy.reporting.localization.Localization(__file__, 33, 23), instance_12614, *[type__12615], **kwargs_12616)
            
            # Assigning a type to the variable 'ret_type' (line 33)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'ret_type', instance_call_result_12617)
            
            
            # Call to is_type_instance(...): (line 35)
            # Processing the call keyword arguments (line 35)
            kwargs_12620 = {}
            # Getting the type of 'ret_type' (line 35)
            ret_type_12618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 19), 'ret_type', False)
            # Obtaining the member 'is_type_instance' of a type (line 35)
            is_type_instance_12619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 19), ret_type_12618, 'is_type_instance')
            # Calling is_type_instance(args, kwargs) (line 35)
            is_type_instance_call_result_12621 = invoke(stypy.reporting.localization.Localization(__file__, 35, 19), is_type_instance_12619, *[], **kwargs_12620)
            
            # Applying the 'not' unary operator (line 35)
            result_not__12622 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 15), 'not', is_type_instance_call_result_12621)
            
            # Testing if the type of an if condition is none (line 35)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 35, 12), result_not__12622):
                pass
            else:
                
                # Testing the type of an if condition (line 35)
                if_condition_12623 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 12), result_not__12622)
                # Assigning a type to the variable 'if_condition_12623' (line 35)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'if_condition_12623', if_condition_12623)
                # SSA begins for if statement (line 35)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to set_type_instance(...): (line 36)
                # Processing the call arguments (line 36)
                # Getting the type of 'True' (line 36)
                True_12626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 43), 'True', False)
                # Processing the call keyword arguments (line 36)
                kwargs_12627 = {}
                # Getting the type of 'ret_type' (line 36)
                ret_type_12624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'ret_type', False)
                # Obtaining the member 'set_type_instance' of a type (line 36)
                set_type_instance_12625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 16), ret_type_12624, 'set_type_instance')
                # Calling set_type_instance(args, kwargs) (line 36)
                set_type_instance_call_result_12628 = invoke(stypy.reporting.localization.Localization(__file__, 36, 16), set_type_instance_12625, *[True_12626], **kwargs_12627)
                
                # SSA join for if statement (line 35)
                module_type_store = module_type_store.join_ssa_context()
                

            # Getting the type of 'ret_type' (line 37)
            ret_type_12629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 19), 'ret_type')
            # Assigning a type to the variable 'stypy_return_type' (line 37)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'stypy_return_type', ret_type_12629)
            # SSA branch for the else part of an if statement (line 32)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to has_type_instance_value(...): (line 40)
            # Processing the call keyword arguments (line 40)
            kwargs_12632 = {}
            # Getting the type of 'type_' (line 40)
            type__12630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'type_', False)
            # Obtaining the member 'has_type_instance_value' of a type (line 40)
            has_type_instance_value_12631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 19), type__12630, 'has_type_instance_value')
            # Calling has_type_instance_value(args, kwargs) (line 40)
            has_type_instance_value_call_result_12633 = invoke(stypy.reporting.localization.Localization(__file__, 40, 19), has_type_instance_value_12631, *[], **kwargs_12632)
            
            # Applying the 'not' unary operator (line 40)
            result_not__12634 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 15), 'not', has_type_instance_value_call_result_12633)
            
            # Testing if the type of an if condition is none (line 40)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 40, 12), result_not__12634):
                pass
            else:
                
                # Testing the type of an if condition (line 40)
                if_condition_12635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 12), result_not__12634)
                # Assigning a type to the variable 'if_condition_12635' (line 40)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'if_condition_12635', if_condition_12635)
                # SSA begins for if statement (line 40)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to set_type_instance(...): (line 41)
                # Processing the call arguments (line 41)
                # Getting the type of 'True' (line 41)
                True_12638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 40), 'True', False)
                # Processing the call keyword arguments (line 41)
                kwargs_12639 = {}
                # Getting the type of 'type_' (line 41)
                type__12636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'type_', False)
                # Obtaining the member 'set_type_instance' of a type (line 41)
                set_type_instance_12637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 16), type__12636, 'set_type_instance')
                # Calling set_type_instance(args, kwargs) (line 41)
                set_type_instance_call_result_12640 = invoke(stypy.reporting.localization.Localization(__file__, 41, 16), set_type_instance_12637, *[True_12638], **kwargs_12639)
                
                # SSA join for if statement (line 40)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 32)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'type_' (line 43)
        type__12641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 15), 'type_')
        # Assigning a type to the variable 'stypy_return_type' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'stypy_return_type', type__12641)
        
        # ################# End of '_wrap_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_wrap_type' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_12642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12642)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_wrap_type'
        return stypy_return_type_12642


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 47)
        None_12643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 29), 'None')
        # Getting the type of 'None' (line 47)
        None_12644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 41), 'None')
        defaults = [None_12643, None_12644]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 47, 4, False)
        # Assigning a type to the variable 'self' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.__init__', ['type1', 'type2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['type1', 'type2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_12645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, (-1)), 'str', '\n        Creates a new UnionType, optionally adding the passed parameters. If only a type is passed, this type\n        is returned instead\n        :param type1: Optional type to add. It can be another union type.\n        :param type2: Optional type to add . It cannot be another union type\n        :return:\n        ')
        
        # Assigning a List to a Attribute (line 55):
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_12646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        
        # Getting the type of 'self' (line 55)
        self_12647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'self')
        # Setting the type of the member 'types' of a type (line 55)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), self_12647, 'types', list_12646)
        
        # Call to is_union_type(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'type1' (line 58)
        type1_12654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 103), 'type1', False)
        # Processing the call keyword arguments (line 58)
        kwargs_12655 = {}
        # Getting the type of 'stypy_copy' (line 58)
        stypy_copy_12648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 58)
        python_lib_12649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 11), stypy_copy_12648, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 58)
        python_types_12650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 11), python_lib_12649, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 58)
        type_introspection_12651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 11), python_types_12650, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 58)
        runtime_type_inspection_12652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 11), type_introspection_12651, 'runtime_type_inspection')
        # Obtaining the member 'is_union_type' of a type (line 58)
        is_union_type_12653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 11), runtime_type_inspection_12652, 'is_union_type')
        # Calling is_union_type(args, kwargs) (line 58)
        is_union_type_call_result_12656 = invoke(stypy.reporting.localization.Localization(__file__, 58, 11), is_union_type_12653, *[type1_12654], **kwargs_12655)
        
        # Testing if the type of an if condition is none (line 58)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 58, 8), is_union_type_call_result_12656):
            pass
        else:
            
            # Testing the type of an if condition (line 58)
            if_condition_12657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 8), is_union_type_call_result_12656)
            # Assigning a type to the variable 'if_condition_12657' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'if_condition_12657', if_condition_12657)
            # SSA begins for if statement (line 58)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'type1' (line 59)
            type1_12658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 25), 'type1')
            # Obtaining the member 'types' of a type (line 59)
            types_12659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 25), type1_12658, 'types')
            # Assigning a type to the variable 'types_12659' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'types_12659', types_12659)
            # Testing if the for loop is going to be iterated (line 59)
            # Testing the type of a for loop iterable (line 59)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 59, 12), types_12659)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 59, 12), types_12659):
                # Getting the type of the for loop variable (line 59)
                for_loop_var_12660 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 59, 12), types_12659)
                # Assigning a type to the variable 'type_' (line 59)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'type_', for_loop_var_12660)
                # SSA begins for a for statement (line 59)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to append(...): (line 60)
                # Processing the call arguments (line 60)
                # Getting the type of 'type_' (line 60)
                type__12664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 34), 'type_', False)
                # Processing the call keyword arguments (line 60)
                kwargs_12665 = {}
                # Getting the type of 'self' (line 60)
                self_12661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'self', False)
                # Obtaining the member 'types' of a type (line 60)
                types_12662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 16), self_12661, 'types')
                # Obtaining the member 'append' of a type (line 60)
                append_12663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 16), types_12662, 'append')
                # Calling append(args, kwargs) (line 60)
                append_call_result_12666 = invoke(stypy.reporting.localization.Localization(__file__, 60, 16), append_12663, *[type__12664], **kwargs_12665)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Assigning a type to the variable 'stypy_return_type' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 58)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Type idiom detected: calculating its left and rigth part (line 64)
        # Getting the type of 'type1' (line 64)
        type1_12667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'type1')
        # Getting the type of 'None' (line 64)
        None_12668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 24), 'None')
        
        (may_be_12669, more_types_in_union_12670) = may_not_be_none(type1_12667, None_12668)

        if may_be_12669:

            if more_types_in_union_12670:
                # Runtime conditional SSA (line 64)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to append(...): (line 65)
            # Processing the call arguments (line 65)
            
            # Call to _wrap_type(...): (line 65)
            # Processing the call arguments (line 65)
            # Getting the type of 'type1' (line 65)
            type1_12676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 51), 'type1', False)
            # Processing the call keyword arguments (line 65)
            kwargs_12677 = {}
            # Getting the type of 'UnionType' (line 65)
            UnionType_12674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 65)
            _wrap_type_12675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 30), UnionType_12674, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 65)
            _wrap_type_call_result_12678 = invoke(stypy.reporting.localization.Localization(__file__, 65, 30), _wrap_type_12675, *[type1_12676], **kwargs_12677)
            
            # Processing the call keyword arguments (line 65)
            kwargs_12679 = {}
            # Getting the type of 'self' (line 65)
            self_12671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'self', False)
            # Obtaining the member 'types' of a type (line 65)
            types_12672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), self_12671, 'types')
            # Obtaining the member 'append' of a type (line 65)
            append_12673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), types_12672, 'append')
            # Calling append(args, kwargs) (line 65)
            append_call_result_12680 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), append_12673, *[_wrap_type_call_result_12678], **kwargs_12679)
            

            if more_types_in_union_12670:
                # SSA join for if statement (line 64)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 67)
        # Getting the type of 'type2' (line 67)
        type2_12681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'type2')
        # Getting the type of 'None' (line 67)
        None_12682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), 'None')
        
        (may_be_12683, more_types_in_union_12684) = may_not_be_none(type2_12681, None_12682)

        if may_be_12683:

            if more_types_in_union_12684:
                # Runtime conditional SSA (line 67)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to append(...): (line 68)
            # Processing the call arguments (line 68)
            
            # Call to _wrap_type(...): (line 68)
            # Processing the call arguments (line 68)
            # Getting the type of 'type2' (line 68)
            type2_12690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 51), 'type2', False)
            # Processing the call keyword arguments (line 68)
            kwargs_12691 = {}
            # Getting the type of 'UnionType' (line 68)
            UnionType_12688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 30), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 68)
            _wrap_type_12689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 30), UnionType_12688, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 68)
            _wrap_type_call_result_12692 = invoke(stypy.reporting.localization.Localization(__file__, 68, 30), _wrap_type_12689, *[type2_12690], **kwargs_12691)
            
            # Processing the call keyword arguments (line 68)
            kwargs_12693 = {}
            # Getting the type of 'self' (line 68)
            self_12685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'self', False)
            # Obtaining the member 'types' of a type (line 68)
            types_12686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), self_12685, 'types')
            # Obtaining the member 'append' of a type (line 68)
            append_12687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), types_12686, 'append')
            # Calling append(args, kwargs) (line 68)
            append_call_result_12694 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), append_12687, *[_wrap_type_call_result_12692], **kwargs_12693)
            

            if more_types_in_union_12684:
                # SSA join for if statement (line 67)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @staticmethod
    @norecursion
    def create_union_type_from_types(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_union_type_from_types'
        module_type_store = module_type_store.open_function_context('create_union_type_from_types', 70, 4, False)
        
        # Passed parameters checking function
        UnionType.create_union_type_from_types.__dict__.__setitem__('stypy_localization', localization)
        UnionType.create_union_type_from_types.__dict__.__setitem__('stypy_type_of_self', None)
        UnionType.create_union_type_from_types.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.create_union_type_from_types.__dict__.__setitem__('stypy_function_name', 'create_union_type_from_types')
        UnionType.create_union_type_from_types.__dict__.__setitem__('stypy_param_names_list', [])
        UnionType.create_union_type_from_types.__dict__.__setitem__('stypy_varargs_param_name', 'types')
        UnionType.create_union_type_from_types.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.create_union_type_from_types.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.create_union_type_from_types.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.create_union_type_from_types.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.create_union_type_from_types.__dict__.__setitem__('stypy_declared_arg_number', 0)
        arguments = process_argument_values(localization, None, module_type_store, 'create_union_type_from_types', [], 'types', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_union_type_from_types', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_union_type_from_types(...)' code ##################

        str_12695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, (-1)), 'str', '\n        Utility method to create a union type from a list of types\n        :param types: List of types\n        :return: UnionType\n        ')
        
        # Assigning a Call to a Name (line 77):
        
        # Call to UnionType(...): (line 77)
        # Processing the call keyword arguments (line 77)
        kwargs_12697 = {}
        # Getting the type of 'UnionType' (line 77)
        UnionType_12696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 25), 'UnionType', False)
        # Calling UnionType(args, kwargs) (line 77)
        UnionType_call_result_12698 = invoke(stypy.reporting.localization.Localization(__file__, 77, 25), UnionType_12696, *[], **kwargs_12697)
        
        # Assigning a type to the variable 'union_instance' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'union_instance', UnionType_call_result_12698)
        
        # Getting the type of 'types' (line 79)
        types_12699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 21), 'types')
        # Assigning a type to the variable 'types_12699' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'types_12699', types_12699)
        # Testing if the for loop is going to be iterated (line 79)
        # Testing the type of a for loop iterable (line 79)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 79, 8), types_12699)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 79, 8), types_12699):
            # Getting the type of the for loop variable (line 79)
            for_loop_var_12700 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 79, 8), types_12699)
            # Assigning a type to the variable 'type_' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'type_', for_loop_var_12700)
            # SSA begins for a for statement (line 79)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to __add_unconditionally(...): (line 80)
            # Processing the call arguments (line 80)
            # Getting the type of 'union_instance' (line 80)
            union_instance_12703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 44), 'union_instance', False)
            # Getting the type of 'type_' (line 80)
            type__12704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 60), 'type_', False)
            # Processing the call keyword arguments (line 80)
            kwargs_12705 = {}
            # Getting the type of 'UnionType' (line 80)
            UnionType_12701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'UnionType', False)
            # Obtaining the member '__add_unconditionally' of a type (line 80)
            add_unconditionally_12702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), UnionType_12701, '__add_unconditionally')
            # Calling __add_unconditionally(args, kwargs) (line 80)
            add_unconditionally_call_result_12706 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), add_unconditionally_12702, *[union_instance_12703, type__12704], **kwargs_12705)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'union_instance' (line 82)
        union_instance_12708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'union_instance', False)
        # Obtaining the member 'types' of a type (line 82)
        types_12709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 15), union_instance_12708, 'types')
        # Processing the call keyword arguments (line 82)
        kwargs_12710 = {}
        # Getting the type of 'len' (line 82)
        len_12707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'len', False)
        # Calling len(args, kwargs) (line 82)
        len_call_result_12711 = invoke(stypy.reporting.localization.Localization(__file__, 82, 11), len_12707, *[types_12709], **kwargs_12710)
        
        int_12712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 40), 'int')
        # Applying the binary operator '==' (line 82)
        result_eq_12713 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 11), '==', len_call_result_12711, int_12712)
        
        # Testing if the type of an if condition is none (line 82)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 82, 8), result_eq_12713):
            pass
        else:
            
            # Testing the type of an if condition (line 82)
            if_condition_12714 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 8), result_eq_12713)
            # Assigning a type to the variable 'if_condition_12714' (line 82)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'if_condition_12714', if_condition_12714)
            # SSA begins for if statement (line 82)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining the type of the subscript
            int_12715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 40), 'int')
            # Getting the type of 'union_instance' (line 83)
            union_instance_12716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 'union_instance')
            # Obtaining the member 'types' of a type (line 83)
            types_12717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 19), union_instance_12716, 'types')
            # Obtaining the member '__getitem__' of a type (line 83)
            getitem___12718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 19), types_12717, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 83)
            subscript_call_result_12719 = invoke(stypy.reporting.localization.Localization(__file__, 83, 19), getitem___12718, int_12715)
            
            # Assigning a type to the variable 'stypy_return_type' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'stypy_return_type', subscript_call_result_12719)
            # SSA join for if statement (line 82)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'union_instance' (line 84)
        union_instance_12720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 15), 'union_instance')
        # Assigning a type to the variable 'stypy_return_type' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'stypy_return_type', union_instance_12720)
        
        # ################# End of 'create_union_type_from_types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_union_type_from_types' in the type store
        # Getting the type of 'stypy_return_type' (line 70)
        stypy_return_type_12721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12721)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_union_type_from_types'
        return stypy_return_type_12721


    @staticmethod
    @norecursion
    def __add_unconditionally(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__add_unconditionally'
        module_type_store = module_type_store.open_function_context('__add_unconditionally', 88, 4, False)
        
        # Passed parameters checking function
        UnionType.__add_unconditionally.__dict__.__setitem__('stypy_localization', localization)
        UnionType.__add_unconditionally.__dict__.__setitem__('stypy_type_of_self', None)
        UnionType.__add_unconditionally.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.__add_unconditionally.__dict__.__setitem__('stypy_function_name', '__add_unconditionally')
        UnionType.__add_unconditionally.__dict__.__setitem__('stypy_param_names_list', ['type1', 'type2'])
        UnionType.__add_unconditionally.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.__add_unconditionally.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.__add_unconditionally.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.__add_unconditionally.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.__add_unconditionally.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.__add_unconditionally.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, '__add_unconditionally', ['type1', 'type2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__add_unconditionally', localization, ['type2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__add_unconditionally(...)' code ##################

        str_12722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, (-1)), 'str', '\n        Helper method of create_union_type_from_types\n        :param type1: Type to add\n        :param type2: Type to add\n        :return: UnionType\n        ')
        
        # Call to is_union_type(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'type1' (line 96)
        type1_12729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 103), 'type1', False)
        # Processing the call keyword arguments (line 96)
        kwargs_12730 = {}
        # Getting the type of 'stypy_copy' (line 96)
        stypy_copy_12723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 96)
        python_lib_12724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 11), stypy_copy_12723, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 96)
        python_types_12725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 11), python_lib_12724, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 96)
        type_introspection_12726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 11), python_types_12725, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 96)
        runtime_type_inspection_12727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 11), type_introspection_12726, 'runtime_type_inspection')
        # Obtaining the member 'is_union_type' of a type (line 96)
        is_union_type_12728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 11), runtime_type_inspection_12727, 'is_union_type')
        # Calling is_union_type(args, kwargs) (line 96)
        is_union_type_call_result_12731 = invoke(stypy.reporting.localization.Localization(__file__, 96, 11), is_union_type_12728, *[type1_12729], **kwargs_12730)
        
        # Testing if the type of an if condition is none (line 96)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 96, 8), is_union_type_call_result_12731):
            pass
        else:
            
            # Testing the type of an if condition (line 96)
            if_condition_12732 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 8), is_union_type_call_result_12731)
            # Assigning a type to the variable 'if_condition_12732' (line 96)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'if_condition_12732', if_condition_12732)
            # SSA begins for if statement (line 96)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _add(...): (line 97)
            # Processing the call arguments (line 97)
            
            # Call to _wrap_type(...): (line 97)
            # Processing the call arguments (line 97)
            # Getting the type of 'type2' (line 97)
            type2_12737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 51), 'type2', False)
            # Processing the call keyword arguments (line 97)
            kwargs_12738 = {}
            # Getting the type of 'UnionType' (line 97)
            UnionType_12735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 30), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 97)
            _wrap_type_12736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 30), UnionType_12735, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 97)
            _wrap_type_call_result_12739 = invoke(stypy.reporting.localization.Localization(__file__, 97, 30), _wrap_type_12736, *[type2_12737], **kwargs_12738)
            
            # Processing the call keyword arguments (line 97)
            kwargs_12740 = {}
            # Getting the type of 'type1' (line 97)
            type1_12733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 19), 'type1', False)
            # Obtaining the member '_add' of a type (line 97)
            _add_12734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 19), type1_12733, '_add')
            # Calling _add(args, kwargs) (line 97)
            _add_call_result_12741 = invoke(stypy.reporting.localization.Localization(__file__, 97, 19), _add_12734, *[_wrap_type_call_result_12739], **kwargs_12740)
            
            # Assigning a type to the variable 'stypy_return_type' (line 97)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'stypy_return_type', _add_call_result_12741)
            # SSA join for if statement (line 96)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to is_union_type(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'type2' (line 98)
        type2_12748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 103), 'type2', False)
        # Processing the call keyword arguments (line 98)
        kwargs_12749 = {}
        # Getting the type of 'stypy_copy' (line 98)
        stypy_copy_12742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 98)
        python_lib_12743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 11), stypy_copy_12742, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 98)
        python_types_12744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 11), python_lib_12743, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 98)
        type_introspection_12745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 11), python_types_12744, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 98)
        runtime_type_inspection_12746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 11), type_introspection_12745, 'runtime_type_inspection')
        # Obtaining the member 'is_union_type' of a type (line 98)
        is_union_type_12747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 11), runtime_type_inspection_12746, 'is_union_type')
        # Calling is_union_type(args, kwargs) (line 98)
        is_union_type_call_result_12750 = invoke(stypy.reporting.localization.Localization(__file__, 98, 11), is_union_type_12747, *[type2_12748], **kwargs_12749)
        
        # Testing if the type of an if condition is none (line 98)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 98, 8), is_union_type_call_result_12750):
            pass
        else:
            
            # Testing the type of an if condition (line 98)
            if_condition_12751 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 8), is_union_type_call_result_12750)
            # Assigning a type to the variable 'if_condition_12751' (line 98)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'if_condition_12751', if_condition_12751)
            # SSA begins for if statement (line 98)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _add(...): (line 99)
            # Processing the call arguments (line 99)
            
            # Call to _wrap_type(...): (line 99)
            # Processing the call arguments (line 99)
            # Getting the type of 'type1' (line 99)
            type1_12756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 51), 'type1', False)
            # Processing the call keyword arguments (line 99)
            kwargs_12757 = {}
            # Getting the type of 'UnionType' (line 99)
            UnionType_12754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 30), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 99)
            _wrap_type_12755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 30), UnionType_12754, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 99)
            _wrap_type_call_result_12758 = invoke(stypy.reporting.localization.Localization(__file__, 99, 30), _wrap_type_12755, *[type1_12756], **kwargs_12757)
            
            # Processing the call keyword arguments (line 99)
            kwargs_12759 = {}
            # Getting the type of 'type2' (line 99)
            type2_12752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 19), 'type2', False)
            # Obtaining the member '_add' of a type (line 99)
            _add_12753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 19), type2_12752, '_add')
            # Calling _add(args, kwargs) (line 99)
            _add_call_result_12760 = invoke(stypy.reporting.localization.Localization(__file__, 99, 19), _add_12753, *[_wrap_type_call_result_12758], **kwargs_12759)
            
            # Assigning a type to the variable 'stypy_return_type' (line 99)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'stypy_return_type', _add_call_result_12760)
            # SSA join for if statement (line 98)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'type1' (line 101)
        type1_12761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'type1')
        # Getting the type of 'type2' (line 101)
        type2_12762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 20), 'type2')
        # Applying the binary operator '==' (line 101)
        result_eq_12763 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 11), '==', type1_12761, type2_12762)
        
        # Testing if the type of an if condition is none (line 101)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 101, 8), result_eq_12763):
            pass
        else:
            
            # Testing the type of an if condition (line 101)
            if_condition_12764 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 8), result_eq_12763)
            # Assigning a type to the variable 'if_condition_12764' (line 101)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'if_condition_12764', if_condition_12764)
            # SSA begins for if statement (line 101)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _wrap_type(...): (line 102)
            # Processing the call arguments (line 102)
            # Getting the type of 'type1' (line 102)
            type1_12767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 40), 'type1', False)
            # Processing the call keyword arguments (line 102)
            kwargs_12768 = {}
            # Getting the type of 'UnionType' (line 102)
            UnionType_12765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 19), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 102)
            _wrap_type_12766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 19), UnionType_12765, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 102)
            _wrap_type_call_result_12769 = invoke(stypy.reporting.localization.Localization(__file__, 102, 19), _wrap_type_12766, *[type1_12767], **kwargs_12768)
            
            # Assigning a type to the variable 'stypy_return_type' (line 102)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'stypy_return_type', _wrap_type_call_result_12769)
            # SSA join for if statement (line 101)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to UnionType(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'type1' (line 104)
        type1_12771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 25), 'type1', False)
        # Getting the type of 'type2' (line 104)
        type2_12772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 32), 'type2', False)
        # Processing the call keyword arguments (line 104)
        kwargs_12773 = {}
        # Getting the type of 'UnionType' (line 104)
        UnionType_12770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'UnionType', False)
        # Calling UnionType(args, kwargs) (line 104)
        UnionType_call_result_12774 = invoke(stypy.reporting.localization.Localization(__file__, 104, 15), UnionType_12770, *[type1_12771, type2_12772], **kwargs_12773)
        
        # Assigning a type to the variable 'stypy_return_type' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'stypy_return_type', UnionType_call_result_12774)
        
        # ################# End of '__add_unconditionally(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__add_unconditionally' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_12775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12775)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__add_unconditionally'
        return stypy_return_type_12775


    @staticmethod
    @norecursion
    def add(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add'
        module_type_store = module_type_store.open_function_context('add', 106, 4, False)
        
        # Passed parameters checking function
        UnionType.add.__dict__.__setitem__('stypy_localization', localization)
        UnionType.add.__dict__.__setitem__('stypy_type_of_self', None)
        UnionType.add.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.add.__dict__.__setitem__('stypy_function_name', 'add')
        UnionType.add.__dict__.__setitem__('stypy_param_names_list', ['type1', 'type2'])
        UnionType.add.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.add.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.add.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.add.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.add.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.add.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, 'add', ['type1', 'type2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add', localization, ['type2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add(...)' code ##################

        str_12776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, (-1)), 'str', '\n        Adds type1 and type2 to potentially form a UnionType, with the following rules:\n        - If either type1 or type2 are None, the other type is returned and no UnionType is formed\n        - If either type1 or type2 are UndefinedType, the other type is returned and no UnionType is formed\n        - If either type1 or type2 are UnionTypes, they are mergued in a new UnionType that contains the types\n        represented by both of them.\n        - If both types are the same, the first is returned\n        - Else, a new UnionType formed by the two passed types are returned.\n\n        :param type1: Type to add\n        :param type2: Type to add\n        :return: A UnionType\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 121)
        # Getting the type of 'type1' (line 121)
        type1_12777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'type1')
        # Getting the type of 'None' (line 121)
        None_12778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 20), 'None')
        
        (may_be_12779, more_types_in_union_12780) = may_be_none(type1_12777, None_12778)

        if may_be_12779:

            if more_types_in_union_12780:
                # Runtime conditional SSA (line 121)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to _wrap_type(...): (line 122)
            # Processing the call arguments (line 122)
            # Getting the type of 'type2' (line 122)
            type2_12783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 40), 'type2', False)
            # Processing the call keyword arguments (line 122)
            kwargs_12784 = {}
            # Getting the type of 'UnionType' (line 122)
            UnionType_12781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 19), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 122)
            _wrap_type_12782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 19), UnionType_12781, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 122)
            _wrap_type_call_result_12785 = invoke(stypy.reporting.localization.Localization(__file__, 122, 19), _wrap_type_12782, *[type2_12783], **kwargs_12784)
            
            # Assigning a type to the variable 'stypy_return_type' (line 122)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'stypy_return_type', _wrap_type_call_result_12785)

            if more_types_in_union_12780:
                # SSA join for if statement (line 121)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'type1' (line 121)
        type1_12786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'type1')
        # Assigning a type to the variable 'type1' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'type1', remove_type_from_union(type1_12786, types.NoneType))
        
        # Type idiom detected: calculating its left and rigth part (line 124)
        # Getting the type of 'type2' (line 124)
        type2_12787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 11), 'type2')
        # Getting the type of 'None' (line 124)
        None_12788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'None')
        
        (may_be_12789, more_types_in_union_12790) = may_be_none(type2_12787, None_12788)

        if may_be_12789:

            if more_types_in_union_12790:
                # Runtime conditional SSA (line 124)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to _wrap_type(...): (line 125)
            # Processing the call arguments (line 125)
            # Getting the type of 'type1' (line 125)
            type1_12793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 40), 'type1', False)
            # Processing the call keyword arguments (line 125)
            kwargs_12794 = {}
            # Getting the type of 'UnionType' (line 125)
            UnionType_12791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 19), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 125)
            _wrap_type_12792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 19), UnionType_12791, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 125)
            _wrap_type_call_result_12795 = invoke(stypy.reporting.localization.Localization(__file__, 125, 19), _wrap_type_12792, *[type1_12793], **kwargs_12794)
            
            # Assigning a type to the variable 'stypy_return_type' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'stypy_return_type', _wrap_type_call_result_12795)

            if more_types_in_union_12790:
                # SSA join for if statement (line 124)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'type2' (line 124)
        type2_12796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'type2')
        # Assigning a type to the variable 'type2' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'type2', remove_type_from_union(type2_12796, types.NoneType))
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'type1' (line 127)
        type1_12798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 22), 'type1', False)
        # Getting the type of 'TypeError' (line 127)
        TypeError_12799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 29), 'TypeError', False)
        # Processing the call keyword arguments (line 127)
        kwargs_12800 = {}
        # Getting the type of 'isinstance' (line 127)
        isinstance_12797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 127)
        isinstance_call_result_12801 = invoke(stypy.reporting.localization.Localization(__file__, 127, 11), isinstance_12797, *[type1_12798, TypeError_12799], **kwargs_12800)
        
        
        # Call to isinstance(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'type2' (line 127)
        type2_12803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 55), 'type2', False)
        # Getting the type of 'TypeError' (line 127)
        TypeError_12804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 62), 'TypeError', False)
        # Processing the call keyword arguments (line 127)
        kwargs_12805 = {}
        # Getting the type of 'isinstance' (line 127)
        isinstance_12802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 44), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 127)
        isinstance_call_result_12806 = invoke(stypy.reporting.localization.Localization(__file__, 127, 44), isinstance_12802, *[type2_12803, TypeError_12804], **kwargs_12805)
        
        # Applying the binary operator 'and' (line 127)
        result_and_keyword_12807 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 11), 'and', isinstance_call_result_12801, isinstance_call_result_12806)
        
        # Testing if the type of an if condition is none (line 127)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 127, 8), result_and_keyword_12807):
            pass
        else:
            
            # Testing the type of an if condition (line 127)
            if_condition_12808 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 8), result_and_keyword_12807)
            # Assigning a type to the variable 'if_condition_12808' (line 127)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'if_condition_12808', if_condition_12808)
            # SSA begins for if statement (line 127)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to _wrap_type(...): (line 128)
            # Processing the call arguments (line 128)
            # Getting the type of 'type1' (line 128)
            type1_12811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 36), 'type1', False)
            # Processing the call keyword arguments (line 128)
            kwargs_12812 = {}
            # Getting the type of 'UnionType' (line 128)
            UnionType_12809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 128)
            _wrap_type_12810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 15), UnionType_12809, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 128)
            _wrap_type_call_result_12813 = invoke(stypy.reporting.localization.Localization(__file__, 128, 15), _wrap_type_12810, *[type1_12811], **kwargs_12812)
            
            
            # Call to _wrap_type(...): (line 128)
            # Processing the call arguments (line 128)
            # Getting the type of 'type2' (line 128)
            type2_12816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 67), 'type2', False)
            # Processing the call keyword arguments (line 128)
            kwargs_12817 = {}
            # Getting the type of 'UnionType' (line 128)
            UnionType_12814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 46), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 128)
            _wrap_type_12815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 46), UnionType_12814, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 128)
            _wrap_type_call_result_12818 = invoke(stypy.reporting.localization.Localization(__file__, 128, 46), _wrap_type_12815, *[type2_12816], **kwargs_12817)
            
            # Applying the binary operator '==' (line 128)
            result_eq_12819 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 15), '==', _wrap_type_call_result_12813, _wrap_type_call_result_12818)
            
            # Testing if the type of an if condition is none (line 128)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 128, 12), result_eq_12819):
                
                # Getting the type of 'type1' (line 131)
                type1_12826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'type1')
                # Obtaining the member 'error_msg' of a type (line 131)
                error_msg_12827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 16), type1_12826, 'error_msg')
                # Getting the type of 'type2' (line 131)
                type2_12828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 35), 'type2')
                # Obtaining the member 'error_msg' of a type (line 131)
                error_msg_12829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 35), type2_12828, 'error_msg')
                # Applying the binary operator '+=' (line 131)
                result_iadd_12830 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 16), '+=', error_msg_12827, error_msg_12829)
                # Getting the type of 'type1' (line 131)
                type1_12831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'type1')
                # Setting the type of the member 'error_msg' of a type (line 131)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 16), type1_12831, 'error_msg', result_iadd_12830)
                
                
                # Call to remove_error_msg(...): (line 132)
                # Processing the call arguments (line 132)
                # Getting the type of 'type2' (line 132)
                type2_12834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 43), 'type2', False)
                # Processing the call keyword arguments (line 132)
                kwargs_12835 = {}
                # Getting the type of 'TypeError' (line 132)
                TypeError_12832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'TypeError', False)
                # Obtaining the member 'remove_error_msg' of a type (line 132)
                remove_error_msg_12833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 16), TypeError_12832, 'remove_error_msg')
                # Calling remove_error_msg(args, kwargs) (line 132)
                remove_error_msg_call_result_12836 = invoke(stypy.reporting.localization.Localization(__file__, 132, 16), remove_error_msg_12833, *[type2_12834], **kwargs_12835)
                
                # Getting the type of 'type1' (line 133)
                type1_12837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'type1')
                # Assigning a type to the variable 'stypy_return_type' (line 133)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'stypy_return_type', type1_12837)
            else:
                
                # Testing the type of an if condition (line 128)
                if_condition_12820 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 128, 12), result_eq_12819)
                # Assigning a type to the variable 'if_condition_12820' (line 128)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'if_condition_12820', if_condition_12820)
                # SSA begins for if statement (line 128)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to _wrap_type(...): (line 129)
                # Processing the call arguments (line 129)
                # Getting the type of 'type1' (line 129)
                type1_12823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 44), 'type1', False)
                # Processing the call keyword arguments (line 129)
                kwargs_12824 = {}
                # Getting the type of 'UnionType' (line 129)
                UnionType_12821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), 'UnionType', False)
                # Obtaining the member '_wrap_type' of a type (line 129)
                _wrap_type_12822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 23), UnionType_12821, '_wrap_type')
                # Calling _wrap_type(args, kwargs) (line 129)
                _wrap_type_call_result_12825 = invoke(stypy.reporting.localization.Localization(__file__, 129, 23), _wrap_type_12822, *[type1_12823], **kwargs_12824)
                
                # Assigning a type to the variable 'stypy_return_type' (line 129)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'stypy_return_type', _wrap_type_call_result_12825)
                # SSA branch for the else part of an if statement (line 128)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'type1' (line 131)
                type1_12826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'type1')
                # Obtaining the member 'error_msg' of a type (line 131)
                error_msg_12827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 16), type1_12826, 'error_msg')
                # Getting the type of 'type2' (line 131)
                type2_12828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 35), 'type2')
                # Obtaining the member 'error_msg' of a type (line 131)
                error_msg_12829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 35), type2_12828, 'error_msg')
                # Applying the binary operator '+=' (line 131)
                result_iadd_12830 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 16), '+=', error_msg_12827, error_msg_12829)
                # Getting the type of 'type1' (line 131)
                type1_12831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'type1')
                # Setting the type of the member 'error_msg' of a type (line 131)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 16), type1_12831, 'error_msg', result_iadd_12830)
                
                
                # Call to remove_error_msg(...): (line 132)
                # Processing the call arguments (line 132)
                # Getting the type of 'type2' (line 132)
                type2_12834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 43), 'type2', False)
                # Processing the call keyword arguments (line 132)
                kwargs_12835 = {}
                # Getting the type of 'TypeError' (line 132)
                TypeError_12832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'TypeError', False)
                # Obtaining the member 'remove_error_msg' of a type (line 132)
                remove_error_msg_12833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 16), TypeError_12832, 'remove_error_msg')
                # Calling remove_error_msg(args, kwargs) (line 132)
                remove_error_msg_call_result_12836 = invoke(stypy.reporting.localization.Localization(__file__, 132, 16), remove_error_msg_12833, *[type2_12834], **kwargs_12835)
                
                # Getting the type of 'type1' (line 133)
                type1_12837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'type1')
                # Assigning a type to the variable 'stypy_return_type' (line 133)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'stypy_return_type', type1_12837)
                # SSA join for if statement (line 128)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 127)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to is_undefined_type(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'type1' (line 135)
        type1_12844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 107), 'type1', False)
        # Processing the call keyword arguments (line 135)
        kwargs_12845 = {}
        # Getting the type of 'stypy_copy' (line 135)
        stypy_copy_12838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 135)
        python_lib_12839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 11), stypy_copy_12838, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 135)
        python_types_12840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 11), python_lib_12839, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 135)
        type_introspection_12841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 11), python_types_12840, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 135)
        runtime_type_inspection_12842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 11), type_introspection_12841, 'runtime_type_inspection')
        # Obtaining the member 'is_undefined_type' of a type (line 135)
        is_undefined_type_12843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 11), runtime_type_inspection_12842, 'is_undefined_type')
        # Calling is_undefined_type(args, kwargs) (line 135)
        is_undefined_type_call_result_12846 = invoke(stypy.reporting.localization.Localization(__file__, 135, 11), is_undefined_type_12843, *[type1_12844], **kwargs_12845)
        
        # Testing if the type of an if condition is none (line 135)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 135, 8), is_undefined_type_call_result_12846):
            pass
        else:
            
            # Testing the type of an if condition (line 135)
            if_condition_12847 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 8), is_undefined_type_call_result_12846)
            # Assigning a type to the variable 'if_condition_12847' (line 135)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'if_condition_12847', if_condition_12847)
            # SSA begins for if statement (line 135)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _wrap_type(...): (line 136)
            # Processing the call arguments (line 136)
            # Getting the type of 'type2' (line 136)
            type2_12850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 40), 'type2', False)
            # Processing the call keyword arguments (line 136)
            kwargs_12851 = {}
            # Getting the type of 'UnionType' (line 136)
            UnionType_12848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 136)
            _wrap_type_12849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 19), UnionType_12848, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 136)
            _wrap_type_call_result_12852 = invoke(stypy.reporting.localization.Localization(__file__, 136, 19), _wrap_type_12849, *[type2_12850], **kwargs_12851)
            
            # Assigning a type to the variable 'stypy_return_type' (line 136)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'stypy_return_type', _wrap_type_call_result_12852)
            # SSA join for if statement (line 135)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to is_undefined_type(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'type2' (line 137)
        type2_12859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 107), 'type2', False)
        # Processing the call keyword arguments (line 137)
        kwargs_12860 = {}
        # Getting the type of 'stypy_copy' (line 137)
        stypy_copy_12853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 137)
        python_lib_12854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 11), stypy_copy_12853, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 137)
        python_types_12855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 11), python_lib_12854, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 137)
        type_introspection_12856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 11), python_types_12855, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 137)
        runtime_type_inspection_12857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 11), type_introspection_12856, 'runtime_type_inspection')
        # Obtaining the member 'is_undefined_type' of a type (line 137)
        is_undefined_type_12858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 11), runtime_type_inspection_12857, 'is_undefined_type')
        # Calling is_undefined_type(args, kwargs) (line 137)
        is_undefined_type_call_result_12861 = invoke(stypy.reporting.localization.Localization(__file__, 137, 11), is_undefined_type_12858, *[type2_12859], **kwargs_12860)
        
        # Testing if the type of an if condition is none (line 137)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 137, 8), is_undefined_type_call_result_12861):
            pass
        else:
            
            # Testing the type of an if condition (line 137)
            if_condition_12862 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 8), is_undefined_type_call_result_12861)
            # Assigning a type to the variable 'if_condition_12862' (line 137)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'if_condition_12862', if_condition_12862)
            # SSA begins for if statement (line 137)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _wrap_type(...): (line 138)
            # Processing the call arguments (line 138)
            # Getting the type of 'type1' (line 138)
            type1_12865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 40), 'type1', False)
            # Processing the call keyword arguments (line 138)
            kwargs_12866 = {}
            # Getting the type of 'UnionType' (line 138)
            UnionType_12863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 19), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 138)
            _wrap_type_12864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 19), UnionType_12863, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 138)
            _wrap_type_call_result_12867 = invoke(stypy.reporting.localization.Localization(__file__, 138, 19), _wrap_type_12864, *[type1_12865], **kwargs_12866)
            
            # Assigning a type to the variable 'stypy_return_type' (line 138)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'stypy_return_type', _wrap_type_call_result_12867)
            # SSA join for if statement (line 137)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to is_union_type(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'type1' (line 140)
        type1_12874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 103), 'type1', False)
        # Processing the call keyword arguments (line 140)
        kwargs_12875 = {}
        # Getting the type of 'stypy_copy' (line 140)
        stypy_copy_12868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 140)
        python_lib_12869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 11), stypy_copy_12868, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 140)
        python_types_12870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 11), python_lib_12869, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 140)
        type_introspection_12871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 11), python_types_12870, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 140)
        runtime_type_inspection_12872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 11), type_introspection_12871, 'runtime_type_inspection')
        # Obtaining the member 'is_union_type' of a type (line 140)
        is_union_type_12873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 11), runtime_type_inspection_12872, 'is_union_type')
        # Calling is_union_type(args, kwargs) (line 140)
        is_union_type_call_result_12876 = invoke(stypy.reporting.localization.Localization(__file__, 140, 11), is_union_type_12873, *[type1_12874], **kwargs_12875)
        
        # Testing if the type of an if condition is none (line 140)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 140, 8), is_union_type_call_result_12876):
            pass
        else:
            
            # Testing the type of an if condition (line 140)
            if_condition_12877 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 8), is_union_type_call_result_12876)
            # Assigning a type to the variable 'if_condition_12877' (line 140)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'if_condition_12877', if_condition_12877)
            # SSA begins for if statement (line 140)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _add(...): (line 141)
            # Processing the call arguments (line 141)
            # Getting the type of 'type2' (line 141)
            type2_12880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 30), 'type2', False)
            # Processing the call keyword arguments (line 141)
            kwargs_12881 = {}
            # Getting the type of 'type1' (line 141)
            type1_12878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 19), 'type1', False)
            # Obtaining the member '_add' of a type (line 141)
            _add_12879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 19), type1_12878, '_add')
            # Calling _add(args, kwargs) (line 141)
            _add_call_result_12882 = invoke(stypy.reporting.localization.Localization(__file__, 141, 19), _add_12879, *[type2_12880], **kwargs_12881)
            
            # Assigning a type to the variable 'stypy_return_type' (line 141)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'stypy_return_type', _add_call_result_12882)
            # SSA join for if statement (line 140)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to is_union_type(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'type2' (line 142)
        type2_12889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 103), 'type2', False)
        # Processing the call keyword arguments (line 142)
        kwargs_12890 = {}
        # Getting the type of 'stypy_copy' (line 142)
        stypy_copy_12883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 142)
        python_lib_12884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 11), stypy_copy_12883, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 142)
        python_types_12885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 11), python_lib_12884, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 142)
        type_introspection_12886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 11), python_types_12885, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 142)
        runtime_type_inspection_12887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 11), type_introspection_12886, 'runtime_type_inspection')
        # Obtaining the member 'is_union_type' of a type (line 142)
        is_union_type_12888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 11), runtime_type_inspection_12887, 'is_union_type')
        # Calling is_union_type(args, kwargs) (line 142)
        is_union_type_call_result_12891 = invoke(stypy.reporting.localization.Localization(__file__, 142, 11), is_union_type_12888, *[type2_12889], **kwargs_12890)
        
        # Testing if the type of an if condition is none (line 142)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 142, 8), is_union_type_call_result_12891):
            pass
        else:
            
            # Testing the type of an if condition (line 142)
            if_condition_12892 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 8), is_union_type_call_result_12891)
            # Assigning a type to the variable 'if_condition_12892' (line 142)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'if_condition_12892', if_condition_12892)
            # SSA begins for if statement (line 142)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _add(...): (line 143)
            # Processing the call arguments (line 143)
            # Getting the type of 'type1' (line 143)
            type1_12895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 30), 'type1', False)
            # Processing the call keyword arguments (line 143)
            kwargs_12896 = {}
            # Getting the type of 'type2' (line 143)
            type2_12893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 19), 'type2', False)
            # Obtaining the member '_add' of a type (line 143)
            _add_12894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 19), type2_12893, '_add')
            # Calling _add(args, kwargs) (line 143)
            _add_call_result_12897 = invoke(stypy.reporting.localization.Localization(__file__, 143, 19), _add_12894, *[type1_12895], **kwargs_12896)
            
            # Assigning a type to the variable 'stypy_return_type' (line 143)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'stypy_return_type', _add_call_result_12897)
            # SSA join for if statement (line 142)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to _wrap_type(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'type1' (line 145)
        type1_12900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 32), 'type1', False)
        # Processing the call keyword arguments (line 145)
        kwargs_12901 = {}
        # Getting the type of 'UnionType' (line 145)
        UnionType_12898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 11), 'UnionType', False)
        # Obtaining the member '_wrap_type' of a type (line 145)
        _wrap_type_12899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 11), UnionType_12898, '_wrap_type')
        # Calling _wrap_type(args, kwargs) (line 145)
        _wrap_type_call_result_12902 = invoke(stypy.reporting.localization.Localization(__file__, 145, 11), _wrap_type_12899, *[type1_12900], **kwargs_12901)
        
        
        # Call to _wrap_type(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'type2' (line 145)
        type2_12905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 63), 'type2', False)
        # Processing the call keyword arguments (line 145)
        kwargs_12906 = {}
        # Getting the type of 'UnionType' (line 145)
        UnionType_12903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 42), 'UnionType', False)
        # Obtaining the member '_wrap_type' of a type (line 145)
        _wrap_type_12904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 42), UnionType_12903, '_wrap_type')
        # Calling _wrap_type(args, kwargs) (line 145)
        _wrap_type_call_result_12907 = invoke(stypy.reporting.localization.Localization(__file__, 145, 42), _wrap_type_12904, *[type2_12905], **kwargs_12906)
        
        # Applying the binary operator '==' (line 145)
        result_eq_12908 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 11), '==', _wrap_type_call_result_12902, _wrap_type_call_result_12907)
        
        # Testing if the type of an if condition is none (line 145)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 145, 8), result_eq_12908):
            pass
        else:
            
            # Testing the type of an if condition (line 145)
            if_condition_12909 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 8), result_eq_12908)
            # Assigning a type to the variable 'if_condition_12909' (line 145)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'if_condition_12909', if_condition_12909)
            # SSA begins for if statement (line 145)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _wrap_type(...): (line 146)
            # Processing the call arguments (line 146)
            # Getting the type of 'type1' (line 146)
            type1_12912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 40), 'type1', False)
            # Processing the call keyword arguments (line 146)
            kwargs_12913 = {}
            # Getting the type of 'UnionType' (line 146)
            UnionType_12910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 19), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 146)
            _wrap_type_12911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 19), UnionType_12910, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 146)
            _wrap_type_call_result_12914 = invoke(stypy.reporting.localization.Localization(__file__, 146, 19), _wrap_type_12911, *[type1_12912], **kwargs_12913)
            
            # Assigning a type to the variable 'stypy_return_type' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'stypy_return_type', _wrap_type_call_result_12914)
            # SSA join for if statement (line 145)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to UnionType(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'type1' (line 148)
        type1_12916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 25), 'type1', False)
        # Getting the type of 'type2' (line 148)
        type2_12917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 32), 'type2', False)
        # Processing the call keyword arguments (line 148)
        kwargs_12918 = {}
        # Getting the type of 'UnionType' (line 148)
        UnionType_12915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 15), 'UnionType', False)
        # Calling UnionType(args, kwargs) (line 148)
        UnionType_call_result_12919 = invoke(stypy.reporting.localization.Localization(__file__, 148, 15), UnionType_12915, *[type1_12916, type2_12917], **kwargs_12918)
        
        # Assigning a type to the variable 'stypy_return_type' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'stypy_return_type', UnionType_call_result_12919)
        
        # ################# End of 'add(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add' in the type store
        # Getting the type of 'stypy_return_type' (line 106)
        stypy_return_type_12920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12920)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add'
        return stypy_return_type_12920


    @norecursion
    def _add(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_add'
        module_type_store = module_type_store.open_function_context('_add', 150, 4, False)
        # Assigning a type to the variable 'self' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType._add.__dict__.__setitem__('stypy_localization', localization)
        UnionType._add.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType._add.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType._add.__dict__.__setitem__('stypy_function_name', 'UnionType._add')
        UnionType._add.__dict__.__setitem__('stypy_param_names_list', ['other_type'])
        UnionType._add.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType._add.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType._add.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType._add.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType._add.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType._add.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType._add', ['other_type'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_add', localization, ['other_type'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_add(...)' code ##################

        str_12921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, (-1)), 'str', '\n        Adds the passed type to the current UnionType object. If other_type is a UnionType, all its contained types\n        are added to the current.\n        :param other_type: Type to add\n        :return: The self object\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 157)
        # Getting the type of 'other_type' (line 157)
        other_type_12922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 11), 'other_type')
        # Getting the type of 'None' (line 157)
        None_12923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 25), 'None')
        
        (may_be_12924, more_types_in_union_12925) = may_be_none(other_type_12922, None_12923)

        if may_be_12924:

            if more_types_in_union_12925:
                # Runtime conditional SSA (line 157)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 158)
            self_12926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 19), 'self')
            # Assigning a type to the variable 'stypy_return_type' (line 158)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'stypy_return_type', self_12926)

            if more_types_in_union_12925:
                # SSA join for if statement (line 157)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'other_type' (line 157)
        other_type_12927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'other_type')
        # Assigning a type to the variable 'other_type' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'other_type', remove_type_from_union(other_type_12927, types.NoneType))
        
        # Call to is_union_type(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'other_type' (line 159)
        other_type_12934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 103), 'other_type', False)
        # Processing the call keyword arguments (line 159)
        kwargs_12935 = {}
        # Getting the type of 'stypy_copy' (line 159)
        stypy_copy_12928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 159)
        python_lib_12929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 11), stypy_copy_12928, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 159)
        python_types_12930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 11), python_lib_12929, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 159)
        type_introspection_12931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 11), python_types_12930, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 159)
        runtime_type_inspection_12932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 11), type_introspection_12931, 'runtime_type_inspection')
        # Obtaining the member 'is_union_type' of a type (line 159)
        is_union_type_12933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 11), runtime_type_inspection_12932, 'is_union_type')
        # Calling is_union_type(args, kwargs) (line 159)
        is_union_type_call_result_12936 = invoke(stypy.reporting.localization.Localization(__file__, 159, 11), is_union_type_12933, *[other_type_12934], **kwargs_12935)
        
        # Testing if the type of an if condition is none (line 159)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 159, 8), is_union_type_call_result_12936):
            pass
        else:
            
            # Testing the type of an if condition (line 159)
            if_condition_12937 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 8), is_union_type_call_result_12936)
            # Assigning a type to the variable 'if_condition_12937' (line 159)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'if_condition_12937', if_condition_12937)
            # SSA begins for if statement (line 159)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'other_type' (line 160)
            other_type_12938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 21), 'other_type')
            # Obtaining the member 'types' of a type (line 160)
            types_12939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 21), other_type_12938, 'types')
            # Assigning a type to the variable 'types_12939' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'types_12939', types_12939)
            # Testing if the for loop is going to be iterated (line 160)
            # Testing the type of a for loop iterable (line 160)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 160, 12), types_12939)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 160, 12), types_12939):
                # Getting the type of the for loop variable (line 160)
                for_loop_var_12940 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 160, 12), types_12939)
                # Assigning a type to the variable 't' (line 160)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 't', for_loop_var_12940)
                # SSA begins for a for statement (line 160)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to _add(...): (line 161)
                # Processing the call arguments (line 161)
                # Getting the type of 't' (line 161)
                t_12943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 26), 't', False)
                # Processing the call keyword arguments (line 161)
                kwargs_12944 = {}
                # Getting the type of 'self' (line 161)
                self_12941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'self', False)
                # Obtaining the member '_add' of a type (line 161)
                _add_12942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 16), self_12941, '_add')
                # Calling _add(args, kwargs) (line 161)
                _add_call_result_12945 = invoke(stypy.reporting.localization.Localization(__file__, 161, 16), _add_12942, *[t_12943], **kwargs_12944)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Getting the type of 'self' (line 162)
            self_12946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 19), 'self')
            # Assigning a type to the variable 'stypy_return_type' (line 162)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'stypy_return_type', self_12946)
            # SSA join for if statement (line 159)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 164):
        
        # Call to _wrap_type(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'other_type' (line 164)
        other_type_12949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 42), 'other_type', False)
        # Processing the call keyword arguments (line 164)
        kwargs_12950 = {}
        # Getting the type of 'UnionType' (line 164)
        UnionType_12947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 21), 'UnionType', False)
        # Obtaining the member '_wrap_type' of a type (line 164)
        _wrap_type_12948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 21), UnionType_12947, '_wrap_type')
        # Calling _wrap_type(args, kwargs) (line 164)
        _wrap_type_call_result_12951 = invoke(stypy.reporting.localization.Localization(__file__, 164, 21), _wrap_type_12948, *[other_type_12949], **kwargs_12950)
        
        # Assigning a type to the variable 'other_type' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'other_type', _wrap_type_call_result_12951)
        
        # Getting the type of 'self' (line 167)
        self_12952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 17), 'self')
        # Obtaining the member 'types' of a type (line 167)
        types_12953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 17), self_12952, 'types')
        # Assigning a type to the variable 'types_12953' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'types_12953', types_12953)
        # Testing if the for loop is going to be iterated (line 167)
        # Testing the type of a for loop iterable (line 167)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 167, 8), types_12953)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 167, 8), types_12953):
            # Getting the type of the for loop variable (line 167)
            for_loop_var_12954 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 167, 8), types_12953)
            # Assigning a type to the variable 't' (line 167)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 't', for_loop_var_12954)
            # SSA begins for a for statement (line 167)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 't' (line 168)
            t_12955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 15), 't')
            # Getting the type of 'other_type' (line 168)
            other_type_12956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 20), 'other_type')
            # Applying the binary operator '==' (line 168)
            result_eq_12957 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 15), '==', t_12955, other_type_12956)
            
            # Testing if the type of an if condition is none (line 168)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 168, 12), result_eq_12957):
                pass
            else:
                
                # Testing the type of an if condition (line 168)
                if_condition_12958 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 12), result_eq_12957)
                # Assigning a type to the variable 'if_condition_12958' (line 168)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'if_condition_12958', if_condition_12958)
                # SSA begins for if statement (line 168)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'self' (line 169)
                self_12959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 23), 'self')
                # Assigning a type to the variable 'stypy_return_type' (line 169)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'stypy_return_type', self_12959)
                # SSA join for if statement (line 168)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to append(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'other_type' (line 171)
        other_type_12963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 26), 'other_type', False)
        # Processing the call keyword arguments (line 171)
        kwargs_12964 = {}
        # Getting the type of 'self' (line 171)
        self_12960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'self', False)
        # Obtaining the member 'types' of a type (line 171)
        types_12961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), self_12960, 'types')
        # Obtaining the member 'append' of a type (line 171)
        append_12962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), types_12961, 'append')
        # Calling append(args, kwargs) (line 171)
        append_call_result_12965 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), append_12962, *[other_type_12963], **kwargs_12964)
        
        # Getting the type of 'self' (line 173)
        self_12966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'stypy_return_type', self_12966)
        
        # ################# End of '_add(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_add' in the type store
        # Getting the type of 'stypy_return_type' (line 150)
        stypy_return_type_12967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12967)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_add'
        return stypy_return_type_12967


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 177, 4, False)
        # Assigning a type to the variable 'self' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        UnionType.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'UnionType.stypy__repr__')
        UnionType.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        UnionType.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        str_12968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, (-1)), 'str', '\n        Visual representation of the UnionType\n        :return:\n        ')
        
        # Call to __str__(...): (line 182)
        # Processing the call keyword arguments (line 182)
        kwargs_12971 = {}
        # Getting the type of 'self' (line 182)
        self_12969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'self', False)
        # Obtaining the member '__str__' of a type (line 182)
        str___12970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 15), self_12969, '__str__')
        # Calling __str__(args, kwargs) (line 182)
        str___call_result_12972 = invoke(stypy.reporting.localization.Localization(__file__, 182, 15), str___12970, *[], **kwargs_12971)
        
        # Assigning a type to the variable 'stypy_return_type' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'stypy_return_type', str___call_result_12972)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 177)
        stypy_return_type_12973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12973)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_12973


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 184, 4, False)
        # Assigning a type to the variable 'self' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        UnionType.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.stypy__str__.__dict__.__setitem__('stypy_function_name', 'UnionType.stypy__str__')
        UnionType.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        UnionType.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.stypy__str__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__str__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__str__(...)' code ##################

        str_12974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, (-1)), 'str', '\n        Visual representation of the UnionType\n        :return:\n        ')
        
        # Assigning a Str to a Name (line 189):
        str_12975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 18), 'str', '')
        # Assigning a type to the variable 'the_str' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'the_str', str_12975)
        
        
        # Call to range(...): (line 190)
        # Processing the call arguments (line 190)
        
        # Call to len(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'self' (line 190)
        self_12978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 27), 'self', False)
        # Obtaining the member 'types' of a type (line 190)
        types_12979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 27), self_12978, 'types')
        # Processing the call keyword arguments (line 190)
        kwargs_12980 = {}
        # Getting the type of 'len' (line 190)
        len_12977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 23), 'len', False)
        # Calling len(args, kwargs) (line 190)
        len_call_result_12981 = invoke(stypy.reporting.localization.Localization(__file__, 190, 23), len_12977, *[types_12979], **kwargs_12980)
        
        # Processing the call keyword arguments (line 190)
        kwargs_12982 = {}
        # Getting the type of 'range' (line 190)
        range_12976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 17), 'range', False)
        # Calling range(args, kwargs) (line 190)
        range_call_result_12983 = invoke(stypy.reporting.localization.Localization(__file__, 190, 17), range_12976, *[len_call_result_12981], **kwargs_12982)
        
        # Assigning a type to the variable 'range_call_result_12983' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'range_call_result_12983', range_call_result_12983)
        # Testing if the for loop is going to be iterated (line 190)
        # Testing the type of a for loop iterable (line 190)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 190, 8), range_call_result_12983)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 190, 8), range_call_result_12983):
            # Getting the type of the for loop variable (line 190)
            for_loop_var_12984 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 190, 8), range_call_result_12983)
            # Assigning a type to the variable 'i' (line 190)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'i', for_loop_var_12984)
            # SSA begins for a for statement (line 190)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'the_str' (line 191)
            the_str_12985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'the_str')
            
            # Call to str(...): (line 191)
            # Processing the call arguments (line 191)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 191)
            i_12987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 38), 'i', False)
            # Getting the type of 'self' (line 191)
            self_12988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 27), 'self', False)
            # Obtaining the member 'types' of a type (line 191)
            types_12989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 27), self_12988, 'types')
            # Obtaining the member '__getitem__' of a type (line 191)
            getitem___12990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 27), types_12989, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 191)
            subscript_call_result_12991 = invoke(stypy.reporting.localization.Localization(__file__, 191, 27), getitem___12990, i_12987)
            
            # Processing the call keyword arguments (line 191)
            kwargs_12992 = {}
            # Getting the type of 'str' (line 191)
            str_12986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 23), 'str', False)
            # Calling str(args, kwargs) (line 191)
            str_call_result_12993 = invoke(stypy.reporting.localization.Localization(__file__, 191, 23), str_12986, *[subscript_call_result_12991], **kwargs_12992)
            
            # Applying the binary operator '+=' (line 191)
            result_iadd_12994 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 12), '+=', the_str_12985, str_call_result_12993)
            # Assigning a type to the variable 'the_str' (line 191)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'the_str', result_iadd_12994)
            
            
            # Getting the type of 'i' (line 192)
            i_12995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 15), 'i')
            
            # Call to len(...): (line 192)
            # Processing the call arguments (line 192)
            # Getting the type of 'self' (line 192)
            self_12997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 23), 'self', False)
            # Obtaining the member 'types' of a type (line 192)
            types_12998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 23), self_12997, 'types')
            # Processing the call keyword arguments (line 192)
            kwargs_12999 = {}
            # Getting the type of 'len' (line 192)
            len_12996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 19), 'len', False)
            # Calling len(args, kwargs) (line 192)
            len_call_result_13000 = invoke(stypy.reporting.localization.Localization(__file__, 192, 19), len_12996, *[types_12998], **kwargs_12999)
            
            int_13001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 37), 'int')
            # Applying the binary operator '-' (line 192)
            result_sub_13002 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 19), '-', len_call_result_13000, int_13001)
            
            # Applying the binary operator '<' (line 192)
            result_lt_13003 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 15), '<', i_12995, result_sub_13002)
            
            # Testing if the type of an if condition is none (line 192)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 192, 12), result_lt_13003):
                pass
            else:
                
                # Testing the type of an if condition (line 192)
                if_condition_13004 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 12), result_lt_13003)
                # Assigning a type to the variable 'if_condition_13004' (line 192)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'if_condition_13004', if_condition_13004)
                # SSA begins for if statement (line 192)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'the_str' (line 193)
                the_str_13005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 16), 'the_str')
                str_13006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 27), 'str', ' \\/ ')
                # Applying the binary operator '+=' (line 193)
                result_iadd_13007 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 16), '+=', the_str_13005, str_13006)
                # Assigning a type to the variable 'the_str' (line 193)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 16), 'the_str', result_iadd_13007)
                
                # SSA join for if statement (line 192)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'the_str' (line 194)
        the_str_13008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 15), 'the_str')
        # Assigning a type to the variable 'stypy_return_type' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'stypy_return_type', the_str_13008)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 184)
        stypy_return_type_13009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13009)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_13009


    @norecursion
    def __iter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iter__'
        module_type_store = module_type_store.open_function_context('__iter__', 196, 4, False)
        # Assigning a type to the variable 'self' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.__iter__.__dict__.__setitem__('stypy_localization', localization)
        UnionType.__iter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.__iter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.__iter__.__dict__.__setitem__('stypy_function_name', 'UnionType.__iter__')
        UnionType.__iter__.__dict__.__setitem__('stypy_param_names_list', [])
        UnionType.__iter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.__iter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.__iter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.__iter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.__iter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.__iter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.__iter__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__iter__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__iter__(...)' code ##################

        str_13010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, (-1)), 'str', '\n        Iterator interface, to iterate through the contained types\n        :return:\n        ')
        
        # Getting the type of 'self' (line 201)
        self_13011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 20), 'self')
        # Obtaining the member 'types' of a type (line 201)
        types_13012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 20), self_13011, 'types')
        # Assigning a type to the variable 'types_13012' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'types_13012', types_13012)
        # Testing if the for loop is going to be iterated (line 201)
        # Testing the type of a for loop iterable (line 201)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 201, 8), types_13012)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 201, 8), types_13012):
            # Getting the type of the for loop variable (line 201)
            for_loop_var_13013 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 201, 8), types_13012)
            # Assigning a type to the variable 'elem' (line 201)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'elem', for_loop_var_13013)
            # SSA begins for a for statement (line 201)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Creating a generator
            # Getting the type of 'elem' (line 202)
            elem_13014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 18), 'elem')
            GeneratorType_13015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 12), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 12), GeneratorType_13015, elem_13014)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'stypy_return_type', GeneratorType_13015)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of '__iter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iter__' in the type store
        # Getting the type of 'stypy_return_type' (line 196)
        stypy_return_type_13016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13016)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iter__'
        return stypy_return_type_13016


    @norecursion
    def __contains__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__contains__'
        module_type_store = module_type_store.open_function_context('__contains__', 204, 4, False)
        # Assigning a type to the variable 'self' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.__contains__.__dict__.__setitem__('stypy_localization', localization)
        UnionType.__contains__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.__contains__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.__contains__.__dict__.__setitem__('stypy_function_name', 'UnionType.__contains__')
        UnionType.__contains__.__dict__.__setitem__('stypy_param_names_list', ['item'])
        UnionType.__contains__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.__contains__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.__contains__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.__contains__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.__contains__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.__contains__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.__contains__', ['item'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__contains__', localization, ['item'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__contains__(...)' code ##################

        str_13017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, (-1)), 'str', '\n        The in operator, to determine if a type is inside a UnionType\n        :param item: Type to test. If it is another UnionType and this passed UnionType types are all inside the\n        current one, then the method returns true\n        :return: bool\n        ')
        
        # Call to is_union_type(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'item' (line 211)
        item_13024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 103), 'item', False)
        # Processing the call keyword arguments (line 211)
        kwargs_13025 = {}
        # Getting the type of 'stypy_copy' (line 211)
        stypy_copy_13018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 211)
        python_lib_13019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 11), stypy_copy_13018, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 211)
        python_types_13020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 11), python_lib_13019, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 211)
        type_introspection_13021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 11), python_types_13020, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 211)
        runtime_type_inspection_13022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 11), type_introspection_13021, 'runtime_type_inspection')
        # Obtaining the member 'is_union_type' of a type (line 211)
        is_union_type_13023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 11), runtime_type_inspection_13022, 'is_union_type')
        # Calling is_union_type(args, kwargs) (line 211)
        is_union_type_call_result_13026 = invoke(stypy.reporting.localization.Localization(__file__, 211, 11), is_union_type_13023, *[item_13024], **kwargs_13025)
        
        # Testing if the type of an if condition is none (line 211)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 211, 8), is_union_type_call_result_13026):
            
            # Call to isinstance(...): (line 217)
            # Processing the call arguments (line 217)
            # Getting the type of 'item' (line 217)
            item_13038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 26), 'item', False)
            # Getting the type of 'undefined_type_copy' (line 217)
            undefined_type_copy_13039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 32), 'undefined_type_copy', False)
            # Obtaining the member 'UndefinedType' of a type (line 217)
            UndefinedType_13040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 32), undefined_type_copy_13039, 'UndefinedType')
            # Processing the call keyword arguments (line 217)
            kwargs_13041 = {}
            # Getting the type of 'isinstance' (line 217)
            isinstance_13037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 217)
            isinstance_call_result_13042 = invoke(stypy.reporting.localization.Localization(__file__, 217, 15), isinstance_13037, *[item_13038, UndefinedType_13040], **kwargs_13041)
            
            # Testing if the type of an if condition is none (line 217)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 217, 12), isinstance_call_result_13042):
                
                # Getting the type of 'item' (line 224)
                item_13057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 23), 'item')
                # Getting the type of 'self' (line 224)
                self_13058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 31), 'self')
                # Obtaining the member 'types' of a type (line 224)
                types_13059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 31), self_13058, 'types')
                # Applying the binary operator 'in' (line 224)
                result_contains_13060 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 23), 'in', item_13057, types_13059)
                
                # Assigning a type to the variable 'stypy_return_type' (line 224)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'stypy_return_type', result_contains_13060)
            else:
                
                # Testing the type of an if condition (line 217)
                if_condition_13043 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 12), isinstance_call_result_13042)
                # Assigning a type to the variable 'if_condition_13043' (line 217)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'if_condition_13043', if_condition_13043)
                # SSA begins for if statement (line 217)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 218):
                # Getting the type of 'False' (line 218)
                False_13044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 24), 'False')
                # Assigning a type to the variable 'found' (line 218)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'found', False_13044)
                
                # Getting the type of 'self' (line 219)
                self_13045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 28), 'self')
                # Obtaining the member 'types' of a type (line 219)
                types_13046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 28), self_13045, 'types')
                # Assigning a type to the variable 'types_13046' (line 219)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'types_13046', types_13046)
                # Testing if the for loop is going to be iterated (line 219)
                # Testing the type of a for loop iterable (line 219)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 219, 16), types_13046)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 219, 16), types_13046):
                    # Getting the type of the for loop variable (line 219)
                    for_loop_var_13047 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 219, 16), types_13046)
                    # Assigning a type to the variable 'elem' (line 219)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'elem', for_loop_var_13047)
                    # SSA begins for a for statement (line 219)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Call to isinstance(...): (line 220)
                    # Processing the call arguments (line 220)
                    # Getting the type of 'elem' (line 220)
                    elem_13049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 34), 'elem', False)
                    # Getting the type of 'undefined_type_copy' (line 220)
                    undefined_type_copy_13050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 40), 'undefined_type_copy', False)
                    # Obtaining the member 'UndefinedType' of a type (line 220)
                    UndefinedType_13051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 40), undefined_type_copy_13050, 'UndefinedType')
                    # Processing the call keyword arguments (line 220)
                    kwargs_13052 = {}
                    # Getting the type of 'isinstance' (line 220)
                    isinstance_13048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 23), 'isinstance', False)
                    # Calling isinstance(args, kwargs) (line 220)
                    isinstance_call_result_13053 = invoke(stypy.reporting.localization.Localization(__file__, 220, 23), isinstance_13048, *[elem_13049, UndefinedType_13051], **kwargs_13052)
                    
                    # Testing if the type of an if condition is none (line 220)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 220, 20), isinstance_call_result_13053):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 220)
                        if_condition_13054 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 20), isinstance_call_result_13053)
                        # Assigning a type to the variable 'if_condition_13054' (line 220)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 20), 'if_condition_13054', if_condition_13054)
                        # SSA begins for if statement (line 220)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Name to a Name (line 221):
                        # Getting the type of 'True' (line 221)
                        True_13055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 32), 'True')
                        # Assigning a type to the variable 'found' (line 221)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 24), 'found', True_13055)
                        # SSA join for if statement (line 220)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # Getting the type of 'found' (line 222)
                found_13056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 23), 'found')
                # Assigning a type to the variable 'stypy_return_type' (line 222)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'stypy_return_type', found_13056)
                # SSA branch for the else part of an if statement (line 217)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'item' (line 224)
                item_13057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 23), 'item')
                # Getting the type of 'self' (line 224)
                self_13058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 31), 'self')
                # Obtaining the member 'types' of a type (line 224)
                types_13059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 31), self_13058, 'types')
                # Applying the binary operator 'in' (line 224)
                result_contains_13060 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 23), 'in', item_13057, types_13059)
                
                # Assigning a type to the variable 'stypy_return_type' (line 224)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'stypy_return_type', result_contains_13060)
                # SSA join for if statement (line 217)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 211)
            if_condition_13027 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 8), is_union_type_call_result_13026)
            # Assigning a type to the variable 'if_condition_13027' (line 211)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'if_condition_13027', if_condition_13027)
            # SSA begins for if statement (line 211)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'item' (line 212)
            item_13028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 24), 'item')
            # Assigning a type to the variable 'item_13028' (line 212)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'item_13028', item_13028)
            # Testing if the for loop is going to be iterated (line 212)
            # Testing the type of a for loop iterable (line 212)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 212, 12), item_13028)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 212, 12), item_13028):
                # Getting the type of the for loop variable (line 212)
                for_loop_var_13029 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 212, 12), item_13028)
                # Assigning a type to the variable 'elem' (line 212)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'elem', for_loop_var_13029)
                # SSA begins for a for statement (line 212)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'elem' (line 213)
                elem_13030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), 'elem')
                # Getting the type of 'self' (line 213)
                self_13031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 31), 'self')
                # Obtaining the member 'types' of a type (line 213)
                types_13032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 31), self_13031, 'types')
                # Applying the binary operator 'notin' (line 213)
                result_contains_13033 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 19), 'notin', elem_13030, types_13032)
                
                # Testing if the type of an if condition is none (line 213)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 213, 16), result_contains_13033):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 213)
                    if_condition_13034 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 16), result_contains_13033)
                    # Assigning a type to the variable 'if_condition_13034' (line 213)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'if_condition_13034', if_condition_13034)
                    # SSA begins for if statement (line 213)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'False' (line 214)
                    False_13035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 27), 'False')
                    # Assigning a type to the variable 'stypy_return_type' (line 214)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 20), 'stypy_return_type', False_13035)
                    # SSA join for if statement (line 213)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Getting the type of 'True' (line 215)
            True_13036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 215)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'stypy_return_type', True_13036)
            # SSA branch for the else part of an if statement (line 211)
            module_type_store.open_ssa_branch('else')
            
            # Call to isinstance(...): (line 217)
            # Processing the call arguments (line 217)
            # Getting the type of 'item' (line 217)
            item_13038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 26), 'item', False)
            # Getting the type of 'undefined_type_copy' (line 217)
            undefined_type_copy_13039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 32), 'undefined_type_copy', False)
            # Obtaining the member 'UndefinedType' of a type (line 217)
            UndefinedType_13040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 32), undefined_type_copy_13039, 'UndefinedType')
            # Processing the call keyword arguments (line 217)
            kwargs_13041 = {}
            # Getting the type of 'isinstance' (line 217)
            isinstance_13037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 217)
            isinstance_call_result_13042 = invoke(stypy.reporting.localization.Localization(__file__, 217, 15), isinstance_13037, *[item_13038, UndefinedType_13040], **kwargs_13041)
            
            # Testing if the type of an if condition is none (line 217)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 217, 12), isinstance_call_result_13042):
                
                # Getting the type of 'item' (line 224)
                item_13057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 23), 'item')
                # Getting the type of 'self' (line 224)
                self_13058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 31), 'self')
                # Obtaining the member 'types' of a type (line 224)
                types_13059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 31), self_13058, 'types')
                # Applying the binary operator 'in' (line 224)
                result_contains_13060 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 23), 'in', item_13057, types_13059)
                
                # Assigning a type to the variable 'stypy_return_type' (line 224)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'stypy_return_type', result_contains_13060)
            else:
                
                # Testing the type of an if condition (line 217)
                if_condition_13043 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 12), isinstance_call_result_13042)
                # Assigning a type to the variable 'if_condition_13043' (line 217)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'if_condition_13043', if_condition_13043)
                # SSA begins for if statement (line 217)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 218):
                # Getting the type of 'False' (line 218)
                False_13044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 24), 'False')
                # Assigning a type to the variable 'found' (line 218)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'found', False_13044)
                
                # Getting the type of 'self' (line 219)
                self_13045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 28), 'self')
                # Obtaining the member 'types' of a type (line 219)
                types_13046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 28), self_13045, 'types')
                # Assigning a type to the variable 'types_13046' (line 219)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'types_13046', types_13046)
                # Testing if the for loop is going to be iterated (line 219)
                # Testing the type of a for loop iterable (line 219)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 219, 16), types_13046)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 219, 16), types_13046):
                    # Getting the type of the for loop variable (line 219)
                    for_loop_var_13047 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 219, 16), types_13046)
                    # Assigning a type to the variable 'elem' (line 219)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'elem', for_loop_var_13047)
                    # SSA begins for a for statement (line 219)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Call to isinstance(...): (line 220)
                    # Processing the call arguments (line 220)
                    # Getting the type of 'elem' (line 220)
                    elem_13049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 34), 'elem', False)
                    # Getting the type of 'undefined_type_copy' (line 220)
                    undefined_type_copy_13050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 40), 'undefined_type_copy', False)
                    # Obtaining the member 'UndefinedType' of a type (line 220)
                    UndefinedType_13051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 40), undefined_type_copy_13050, 'UndefinedType')
                    # Processing the call keyword arguments (line 220)
                    kwargs_13052 = {}
                    # Getting the type of 'isinstance' (line 220)
                    isinstance_13048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 23), 'isinstance', False)
                    # Calling isinstance(args, kwargs) (line 220)
                    isinstance_call_result_13053 = invoke(stypy.reporting.localization.Localization(__file__, 220, 23), isinstance_13048, *[elem_13049, UndefinedType_13051], **kwargs_13052)
                    
                    # Testing if the type of an if condition is none (line 220)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 220, 20), isinstance_call_result_13053):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 220)
                        if_condition_13054 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 20), isinstance_call_result_13053)
                        # Assigning a type to the variable 'if_condition_13054' (line 220)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 20), 'if_condition_13054', if_condition_13054)
                        # SSA begins for if statement (line 220)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Name to a Name (line 221):
                        # Getting the type of 'True' (line 221)
                        True_13055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 32), 'True')
                        # Assigning a type to the variable 'found' (line 221)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 24), 'found', True_13055)
                        # SSA join for if statement (line 220)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # Getting the type of 'found' (line 222)
                found_13056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 23), 'found')
                # Assigning a type to the variable 'stypy_return_type' (line 222)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'stypy_return_type', found_13056)
                # SSA branch for the else part of an if statement (line 217)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'item' (line 224)
                item_13057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 23), 'item')
                # Getting the type of 'self' (line 224)
                self_13058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 31), 'self')
                # Obtaining the member 'types' of a type (line 224)
                types_13059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 31), self_13058, 'types')
                # Applying the binary operator 'in' (line 224)
                result_contains_13060 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 23), 'in', item_13057, types_13059)
                
                # Assigning a type to the variable 'stypy_return_type' (line 224)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'stypy_return_type', result_contains_13060)
                # SSA join for if statement (line 217)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 211)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '__contains__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__contains__' in the type store
        # Getting the type of 'stypy_return_type' (line 204)
        stypy_return_type_13061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13061)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__contains__'
        return stypy_return_type_13061


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 226, 4, False)
        # Assigning a type to the variable 'self' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        UnionType.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'UnionType.stypy__eq__')
        UnionType.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        UnionType.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.stypy__eq__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        str_13062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, (-1)), 'str', '\n        The == operator, to compare UnionTypes\n\n        :param other: Another UnionType (used in type inference code) or a list of types (used in unit testing)\n        :return: True if the passed UnionType or list contains exactly the same amount and type of types that the\n        passed entities\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 234)
        # Getting the type of 'list' (line 234)
        list_13063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 29), 'list')
        # Getting the type of 'other' (line 234)
        other_13064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 22), 'other')
        
        (may_be_13065, more_types_in_union_13066) = may_be_subtype(list_13063, other_13064)

        if may_be_13065:

            if more_types_in_union_13066:
                # Runtime conditional SSA (line 234)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'other' (line 234)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'other', remove_not_subtype_from_union(other_13064, list))
            
            # Assigning a Name to a Name (line 235):
            # Getting the type of 'other' (line 235)
            other_13067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 24), 'other')
            # Assigning a type to the variable 'type_list' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'type_list', other_13067)

            if more_types_in_union_13066:
                # Runtime conditional SSA for else branch (line 234)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_13065) or more_types_in_union_13066):
            # Assigning a type to the variable 'other' (line 234)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'other', remove_subtype_from_union(other_13064, list))
            
            # Call to isinstance(...): (line 237)
            # Processing the call arguments (line 237)
            # Getting the type of 'other' (line 237)
            other_13069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 26), 'other', False)
            # Getting the type of 'UnionType' (line 237)
            UnionType_13070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 33), 'UnionType', False)
            # Processing the call keyword arguments (line 237)
            kwargs_13071 = {}
            # Getting the type of 'isinstance' (line 237)
            isinstance_13068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 237)
            isinstance_call_result_13072 = invoke(stypy.reporting.localization.Localization(__file__, 237, 15), isinstance_13068, *[other_13069, UnionType_13070], **kwargs_13071)
            
            # Testing if the type of an if condition is none (line 237)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 237, 12), isinstance_call_result_13072):
                # Getting the type of 'False' (line 240)
                False_13076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 240)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 16), 'stypy_return_type', False_13076)
            else:
                
                # Testing the type of an if condition (line 237)
                if_condition_13073 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 237, 12), isinstance_call_result_13072)
                # Assigning a type to the variable 'if_condition_13073' (line 237)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'if_condition_13073', if_condition_13073)
                # SSA begins for if statement (line 237)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Attribute to a Name (line 238):
                # Getting the type of 'other' (line 238)
                other_13074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 28), 'other')
                # Obtaining the member 'types' of a type (line 238)
                types_13075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 28), other_13074, 'types')
                # Assigning a type to the variable 'type_list' (line 238)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'type_list', types_13075)
                # SSA branch for the else part of an if statement (line 237)
                module_type_store.open_ssa_branch('else')
                # Getting the type of 'False' (line 240)
                False_13076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 240)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 16), 'stypy_return_type', False_13076)
                # SSA join for if statement (line 237)
                module_type_store = module_type_store.join_ssa_context()
                


            if (may_be_13065 and more_types_in_union_13066):
                # SSA join for if statement (line 234)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        
        # Call to len(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'self' (line 242)
        self_13078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 19), 'self', False)
        # Obtaining the member 'types' of a type (line 242)
        types_13079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 19), self_13078, 'types')
        # Processing the call keyword arguments (line 242)
        kwargs_13080 = {}
        # Getting the type of 'len' (line 242)
        len_13077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 15), 'len', False)
        # Calling len(args, kwargs) (line 242)
        len_call_result_13081 = invoke(stypy.reporting.localization.Localization(__file__, 242, 15), len_13077, *[types_13079], **kwargs_13080)
        
        
        # Call to len(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'type_list' (line 242)
        type_list_13083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 38), 'type_list', False)
        # Processing the call keyword arguments (line 242)
        kwargs_13084 = {}
        # Getting the type of 'len' (line 242)
        len_13082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 34), 'len', False)
        # Calling len(args, kwargs) (line 242)
        len_call_result_13085 = invoke(stypy.reporting.localization.Localization(__file__, 242, 34), len_13082, *[type_list_13083], **kwargs_13084)
        
        # Applying the binary operator '==' (line 242)
        result_eq_13086 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 15), '==', len_call_result_13081, len_call_result_13085)
        
        # Applying the 'not' unary operator (line 242)
        result_not__13087 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 11), 'not', result_eq_13086)
        
        # Testing if the type of an if condition is none (line 242)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 242, 8), result_not__13087):
            pass
        else:
            
            # Testing the type of an if condition (line 242)
            if_condition_13088 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 8), result_not__13087)
            # Assigning a type to the variable 'if_condition_13088' (line 242)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'if_condition_13088', if_condition_13088)
            # SSA begins for if statement (line 242)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'False' (line 243)
            False_13089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 243)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'stypy_return_type', False_13089)
            # SSA join for if statement (line 242)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 245)
        self_13090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 21), 'self')
        # Obtaining the member 'types' of a type (line 245)
        types_13091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 21), self_13090, 'types')
        # Assigning a type to the variable 'types_13091' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'types_13091', types_13091)
        # Testing if the for loop is going to be iterated (line 245)
        # Testing the type of a for loop iterable (line 245)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 245, 8), types_13091)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 245, 8), types_13091):
            # Getting the type of the for loop variable (line 245)
            for_loop_var_13092 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 245, 8), types_13091)
            # Assigning a type to the variable 'type_' (line 245)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'type_', for_loop_var_13092)
            # SSA begins for a for statement (line 245)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Type idiom detected: calculating its left and rigth part (line 246)
            # Getting the type of 'TypeError' (line 246)
            TypeError_13093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 33), 'TypeError')
            # Getting the type of 'type_' (line 246)
            type__13094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 26), 'type_')
            
            (may_be_13095, more_types_in_union_13096) = may_be_subtype(TypeError_13093, type__13094)

            if may_be_13095:

                if more_types_in_union_13096:
                    # Runtime conditional SSA (line 246)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'type_' (line 246)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'type_', remove_not_subtype_from_union(type__13094, TypeError))
                
                # Getting the type of 'type_list' (line 247)
                type_list_13097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 30), 'type_list')
                # Assigning a type to the variable 'type_list_13097' (line 247)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'type_list_13097', type_list_13097)
                # Testing if the for loop is going to be iterated (line 247)
                # Testing the type of a for loop iterable (line 247)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 247, 16), type_list_13097)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 247, 16), type_list_13097):
                    # Getting the type of the for loop variable (line 247)
                    for_loop_var_13098 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 247, 16), type_list_13097)
                    # Assigning a type to the variable 'type_2' (line 247)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'type_2', for_loop_var_13098)
                    # SSA begins for a for statement (line 247)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Type idiom detected: calculating its left and rigth part (line 248)
                    # Getting the type of 'type_2' (line 248)
                    type_2_13099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 28), 'type_2')
                    # Getting the type of 'TypeError' (line 248)
                    TypeError_13100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 39), 'TypeError')
                    
                    (may_be_13101, more_types_in_union_13102) = may_be_type(type_2_13099, TypeError_13100)

                    if may_be_13101:

                        if more_types_in_union_13102:
                            # Runtime conditional SSA (line 248)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                        else:
                            module_type_store = module_type_store

                        # Assigning a type to the variable 'type_2' (line 248)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 20), 'type_2', TypeError_13100())

                        if more_types_in_union_13102:
                            # SSA join for if statement (line 248)
                            module_type_store = module_type_store.join_ssa_context()


                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                

                if more_types_in_union_13096:
                    # SSA join for if statement (line 246)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Getting the type of 'type_' (line 250)
            type__13103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 15), 'type_')
            # Getting the type of 'type_list' (line 250)
            type_list_13104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 28), 'type_list')
            # Applying the binary operator 'notin' (line 250)
            result_contains_13105 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 15), 'notin', type__13103, type_list_13104)
            
            # Testing if the type of an if condition is none (line 250)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 250, 12), result_contains_13105):
                pass
            else:
                
                # Testing the type of an if condition (line 250)
                if_condition_13106 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 250, 12), result_contains_13105)
                # Assigning a type to the variable 'if_condition_13106' (line 250)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'if_condition_13106', if_condition_13106)
                # SSA begins for if statement (line 250)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'False' (line 251)
                False_13107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 251)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 16), 'stypy_return_type', False_13107)
                # SSA join for if statement (line 250)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'True' (line 253)
        True_13108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'stypy_return_type', True_13108)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 226)
        stypy_return_type_13109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13109)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_13109


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 255, 4, False)
        # Assigning a type to the variable 'self' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        UnionType.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.__getitem__.__dict__.__setitem__('stypy_function_name', 'UnionType.__getitem__')
        UnionType.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['item'])
        UnionType.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.__getitem__', ['item'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['item'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        str_13110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, (-1)), 'str', '\n        The [] operator, to obtain individual types stored within the union type\n\n        :param item: Indexer\n        :return:\n        ')
        
        # Obtaining the type of the subscript
        # Getting the type of 'item' (line 262)
        item_13111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 26), 'item')
        # Getting the type of 'self' (line 262)
        self_13112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 15), 'self')
        # Obtaining the member 'types' of a type (line 262)
        types_13113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 15), self_13112, 'types')
        # Obtaining the member '__getitem__' of a type (line 262)
        getitem___13114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 15), types_13113, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 262)
        subscript_call_result_13115 = invoke(stypy.reporting.localization.Localization(__file__, 262, 15), getitem___13114, item_13111)
        
        # Assigning a type to the variable 'stypy_return_type' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'stypy_return_type', subscript_call_result_13115)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 255)
        stypy_return_type_13116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13116)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_13116


    @norecursion
    def get_type_of_member(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_type_of_member'
        module_type_store = module_type_store.open_function_context('get_type_of_member', 266, 4, False)
        # Assigning a type to the variable 'self' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.get_type_of_member.__dict__.__setitem__('stypy_localization', localization)
        UnionType.get_type_of_member.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.get_type_of_member.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.get_type_of_member.__dict__.__setitem__('stypy_function_name', 'UnionType.get_type_of_member')
        UnionType.get_type_of_member.__dict__.__setitem__('stypy_param_names_list', ['localization', 'member_name'])
        UnionType.get_type_of_member.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.get_type_of_member.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.get_type_of_member.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.get_type_of_member.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.get_type_of_member.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.get_type_of_member.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.get_type_of_member', ['localization', 'member_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_type_of_member', localization, ['localization', 'member_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_type_of_member(...)' code ##################

        str_13117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, (-1)), 'str', '\n        For all the types stored in the union type, obtain the type of the member named member_name, returning a\n        Union Type with the union of all the possible types that member_name has inside the UnionType. For example,\n        if a UnionType has the types Class1 and Class2, both with the member "attr" so Class1.attr: int and\n        Class2.attr: str, this method will return int \\/ str.\n        :param localization: Caller information\n        :param member_name: Name of the member to get\n        :return All the types that member_name could have, examining the UnionType stored types\n        ')
        
        # Assigning a List to a Name (line 276):
        
        # Obtaining an instance of the builtin type 'list' (line 276)
        list_13118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 276)
        
        # Assigning a type to the variable 'result' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'result', list_13118)
        
        # Getting the type of 'self' (line 279)
        self_13119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 21), 'self')
        # Obtaining the member 'types' of a type (line 279)
        types_13120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 21), self_13119, 'types')
        # Assigning a type to the variable 'types_13120' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'types_13120', types_13120)
        # Testing if the for loop is going to be iterated (line 279)
        # Testing the type of a for loop iterable (line 279)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 279, 8), types_13120)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 279, 8), types_13120):
            # Getting the type of the for loop variable (line 279)
            for_loop_var_13121 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 279, 8), types_13120)
            # Assigning a type to the variable 'type_' (line 279)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'type_', for_loop_var_13121)
            # SSA begins for a for statement (line 279)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 280):
            
            # Call to get_type_of_member(...): (line 280)
            # Processing the call arguments (line 280)
            # Getting the type of 'localization' (line 280)
            localization_13124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 44), 'localization', False)
            # Getting the type of 'member_name' (line 280)
            member_name_13125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 58), 'member_name', False)
            # Processing the call keyword arguments (line 280)
            kwargs_13126 = {}
            # Getting the type of 'type_' (line 280)
            type__13122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 19), 'type_', False)
            # Obtaining the member 'get_type_of_member' of a type (line 280)
            get_type_of_member_13123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 19), type__13122, 'get_type_of_member')
            # Calling get_type_of_member(args, kwargs) (line 280)
            get_type_of_member_call_result_13127 = invoke(stypy.reporting.localization.Localization(__file__, 280, 19), get_type_of_member_13123, *[localization_13124, member_name_13125], **kwargs_13126)
            
            # Assigning a type to the variable 'temp' (line 280)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'temp', get_type_of_member_call_result_13127)
            
            # Call to append(...): (line 281)
            # Processing the call arguments (line 281)
            # Getting the type of 'temp' (line 281)
            temp_13130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 26), 'temp', False)
            # Processing the call keyword arguments (line 281)
            kwargs_13131 = {}
            # Getting the type of 'result' (line 281)
            result_13128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'result', False)
            # Obtaining the member 'append' of a type (line 281)
            append_13129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 12), result_13128, 'append')
            # Calling append(args, kwargs) (line 281)
            append_call_result_13132 = invoke(stypy.reporting.localization.Localization(__file__, 281, 12), append_13129, *[temp_13130], **kwargs_13131)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 284):
        
        # Call to filter(...): (line 284)
        # Processing the call arguments (line 284)

        @norecursion
        def _stypy_temp_lambda_21(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_21'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_21', 284, 24, True)
            # Passed parameters checking function
            _stypy_temp_lambda_21.stypy_localization = localization
            _stypy_temp_lambda_21.stypy_type_of_self = None
            _stypy_temp_lambda_21.stypy_type_store = module_type_store
            _stypy_temp_lambda_21.stypy_function_name = '_stypy_temp_lambda_21'
            _stypy_temp_lambda_21.stypy_param_names_list = ['t']
            _stypy_temp_lambda_21.stypy_varargs_param_name = None
            _stypy_temp_lambda_21.stypy_kwargs_param_name = None
            _stypy_temp_lambda_21.stypy_call_defaults = defaults
            _stypy_temp_lambda_21.stypy_call_varargs = varargs
            _stypy_temp_lambda_21.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_21', ['t'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_21', ['t'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to isinstance(...): (line 284)
            # Processing the call arguments (line 284)
            # Getting the type of 't' (line 284)
            t_13135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 45), 't', False)
            # Getting the type of 'TypeError' (line 284)
            TypeError_13136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 48), 'TypeError', False)
            # Processing the call keyword arguments (line 284)
            kwargs_13137 = {}
            # Getting the type of 'isinstance' (line 284)
            isinstance_13134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 34), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 284)
            isinstance_call_result_13138 = invoke(stypy.reporting.localization.Localization(__file__, 284, 34), isinstance_13134, *[t_13135, TypeError_13136], **kwargs_13137)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 284)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 24), 'stypy_return_type', isinstance_call_result_13138)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_21' in the type store
            # Getting the type of 'stypy_return_type' (line 284)
            stypy_return_type_13139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 24), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_13139)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_21'
            return stypy_return_type_13139

        # Assigning a type to the variable '_stypy_temp_lambda_21' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 24), '_stypy_temp_lambda_21', _stypy_temp_lambda_21)
        # Getting the type of '_stypy_temp_lambda_21' (line 284)
        _stypy_temp_lambda_21_13140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 24), '_stypy_temp_lambda_21')
        # Getting the type of 'result' (line 284)
        result_13141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 60), 'result', False)
        # Processing the call keyword arguments (line 284)
        kwargs_13142 = {}
        # Getting the type of 'filter' (line 284)
        filter_13133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 17), 'filter', False)
        # Calling filter(args, kwargs) (line 284)
        filter_call_result_13143 = invoke(stypy.reporting.localization.Localization(__file__, 284, 17), filter_13133, *[_stypy_temp_lambda_21_13140, result_13141], **kwargs_13142)
        
        # Assigning a type to the variable 'errors' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'errors', filter_call_result_13143)
        
        # Assigning a Call to a Name (line 286):
        
        # Call to filter(...): (line 286)
        # Processing the call arguments (line 286)

        @norecursion
        def _stypy_temp_lambda_22(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_22'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_22', 286, 33, True)
            # Passed parameters checking function
            _stypy_temp_lambda_22.stypy_localization = localization
            _stypy_temp_lambda_22.stypy_type_of_self = None
            _stypy_temp_lambda_22.stypy_type_store = module_type_store
            _stypy_temp_lambda_22.stypy_function_name = '_stypy_temp_lambda_22'
            _stypy_temp_lambda_22.stypy_param_names_list = ['t']
            _stypy_temp_lambda_22.stypy_varargs_param_name = None
            _stypy_temp_lambda_22.stypy_kwargs_param_name = None
            _stypy_temp_lambda_22.stypy_call_defaults = defaults
            _stypy_temp_lambda_22.stypy_call_varargs = varargs
            _stypy_temp_lambda_22.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_22', ['t'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_22', ['t'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            
            # Call to isinstance(...): (line 286)
            # Processing the call arguments (line 286)
            # Getting the type of 't' (line 286)
            t_13146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 58), 't', False)
            # Getting the type of 'TypeError' (line 286)
            TypeError_13147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 61), 'TypeError', False)
            # Processing the call keyword arguments (line 286)
            kwargs_13148 = {}
            # Getting the type of 'isinstance' (line 286)
            isinstance_13145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 47), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 286)
            isinstance_call_result_13149 = invoke(stypy.reporting.localization.Localization(__file__, 286, 47), isinstance_13145, *[t_13146, TypeError_13147], **kwargs_13148)
            
            # Applying the 'not' unary operator (line 286)
            result_not__13150 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 43), 'not', isinstance_call_result_13149)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 286)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 33), 'stypy_return_type', result_not__13150)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_22' in the type store
            # Getting the type of 'stypy_return_type' (line 286)
            stypy_return_type_13151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 33), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_13151)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_22'
            return stypy_return_type_13151

        # Assigning a type to the variable '_stypy_temp_lambda_22' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 33), '_stypy_temp_lambda_22', _stypy_temp_lambda_22)
        # Getting the type of '_stypy_temp_lambda_22' (line 286)
        _stypy_temp_lambda_22_13152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 33), '_stypy_temp_lambda_22')
        # Getting the type of 'result' (line 286)
        result_13153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 73), 'result', False)
        # Processing the call keyword arguments (line 286)
        kwargs_13154 = {}
        # Getting the type of 'filter' (line 286)
        filter_13144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 26), 'filter', False)
        # Calling filter(args, kwargs) (line 286)
        filter_call_result_13155 = invoke(stypy.reporting.localization.Localization(__file__, 286, 26), filter_13144, *[_stypy_temp_lambda_22_13152, result_13153], **kwargs_13154)
        
        # Assigning a type to the variable 'types_to_return' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'types_to_return', filter_call_result_13155)
        
        
        # Call to len(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'errors' (line 289)
        errors_13157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 15), 'errors', False)
        # Processing the call keyword arguments (line 289)
        kwargs_13158 = {}
        # Getting the type of 'len' (line 289)
        len_13156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 11), 'len', False)
        # Calling len(args, kwargs) (line 289)
        len_call_result_13159 = invoke(stypy.reporting.localization.Localization(__file__, 289, 11), len_13156, *[errors_13157], **kwargs_13158)
        
        
        # Call to len(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'result' (line 289)
        result_13161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 30), 'result', False)
        # Processing the call keyword arguments (line 289)
        kwargs_13162 = {}
        # Getting the type of 'len' (line 289)
        len_13160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 26), 'len', False)
        # Calling len(args, kwargs) (line 289)
        len_call_result_13163 = invoke(stypy.reporting.localization.Localization(__file__, 289, 26), len_13160, *[result_13161], **kwargs_13162)
        
        # Applying the binary operator '==' (line 289)
        result_eq_13164 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 11), '==', len_call_result_13159, len_call_result_13163)
        
        # Testing if the type of an if condition is none (line 289)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 289, 8), result_eq_13164):
            
            
            # Call to len(...): (line 294)
            # Processing the call arguments (line 294)
            # Getting the type of 'errors' (line 294)
            errors_13178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 19), 'errors', False)
            # Processing the call keyword arguments (line 294)
            kwargs_13179 = {}
            # Getting the type of 'len' (line 294)
            len_13177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 15), 'len', False)
            # Calling len(args, kwargs) (line 294)
            len_call_result_13180 = invoke(stypy.reporting.localization.Localization(__file__, 294, 15), len_13177, *[errors_13178], **kwargs_13179)
            
            int_13181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 29), 'int')
            # Applying the binary operator '>' (line 294)
            result_gt_13182 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 15), '>', len_call_result_13180, int_13181)
            
            # Testing if the type of an if condition is none (line 294)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 294, 12), result_gt_13182):
                pass
            else:
                
                # Testing the type of an if condition (line 294)
                if_condition_13183 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 12), result_gt_13182)
                # Assigning a type to the variable 'if_condition_13183' (line 294)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'if_condition_13183', if_condition_13183)
                # SSA begins for if statement (line 294)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 295)
                # Processing the call arguments (line 295)
                
                # Call to UndefinedType(...): (line 295)
                # Processing the call keyword arguments (line 295)
                kwargs_13188 = {}
                # Getting the type of 'undefined_type_copy' (line 295)
                undefined_type_copy_13186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 39), 'undefined_type_copy', False)
                # Obtaining the member 'UndefinedType' of a type (line 295)
                UndefinedType_13187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 39), undefined_type_copy_13186, 'UndefinedType')
                # Calling UndefinedType(args, kwargs) (line 295)
                UndefinedType_call_result_13189 = invoke(stypy.reporting.localization.Localization(__file__, 295, 39), UndefinedType_13187, *[], **kwargs_13188)
                
                # Processing the call keyword arguments (line 295)
                kwargs_13190 = {}
                # Getting the type of 'types_to_return' (line 295)
                types_to_return_13184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'types_to_return', False)
                # Obtaining the member 'append' of a type (line 295)
                append_13185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 16), types_to_return_13184, 'append')
                # Calling append(args, kwargs) (line 295)
                append_call_result_13191 = invoke(stypy.reporting.localization.Localization(__file__, 295, 16), append_13185, *[UndefinedType_call_result_13189], **kwargs_13190)
                
                # SSA join for if statement (line 294)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'errors' (line 301)
            errors_13192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 25), 'errors')
            # Assigning a type to the variable 'errors_13192' (line 301)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'errors_13192', errors_13192)
            # Testing if the for loop is going to be iterated (line 301)
            # Testing the type of a for loop iterable (line 301)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 301, 12), errors_13192)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 301, 12), errors_13192):
                # Getting the type of the for loop variable (line 301)
                for_loop_var_13193 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 301, 12), errors_13192)
                # Assigning a type to the variable 'error' (line 301)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'error', for_loop_var_13193)
                # SSA begins for a for statement (line 301)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 302)
                # Processing the call keyword arguments (line 302)
                kwargs_13196 = {}
                # Getting the type of 'error' (line 302)
                error_13194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 302)
                turn_to_warning_13195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 16), error_13194, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 302)
                turn_to_warning_call_result_13197 = invoke(stypy.reporting.localization.Localization(__file__, 302, 16), turn_to_warning_13195, *[], **kwargs_13196)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 289)
            if_condition_13165 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 289, 8), result_eq_13164)
            # Assigning a type to the variable 'if_condition_13165' (line 289)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'if_condition_13165', if_condition_13165)
            # SSA begins for if statement (line 289)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 290)
            # Processing the call arguments (line 290)
            # Getting the type of 'localization' (line 290)
            localization_13167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 29), 'localization', False)
            
            # Call to format(...): (line 290)
            # Processing the call arguments (line 290)
            # Getting the type of 'member_name' (line 291)
            member_name_13170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 16), 'member_name', False)
            # Getting the type of 'self' (line 291)
            self_13171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 29), 'self', False)
            # Obtaining the member 'types' of a type (line 291)
            types_13172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 29), self_13171, 'types')
            # Processing the call keyword arguments (line 290)
            kwargs_13173 = {}
            str_13168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 43), 'str', "None of the possible types ('{1}') has the member '{0}'")
            # Obtaining the member 'format' of a type (line 290)
            format_13169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 43), str_13168, 'format')
            # Calling format(args, kwargs) (line 290)
            format_call_result_13174 = invoke(stypy.reporting.localization.Localization(__file__, 290, 43), format_13169, *[member_name_13170, types_13172], **kwargs_13173)
            
            # Processing the call keyword arguments (line 290)
            kwargs_13175 = {}
            # Getting the type of 'TypeError' (line 290)
            TypeError_13166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 290)
            TypeError_call_result_13176 = invoke(stypy.reporting.localization.Localization(__file__, 290, 19), TypeError_13166, *[localization_13167, format_call_result_13174], **kwargs_13175)
            
            # Assigning a type to the variable 'stypy_return_type' (line 290)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'stypy_return_type', TypeError_call_result_13176)
            # SSA branch for the else part of an if statement (line 289)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to len(...): (line 294)
            # Processing the call arguments (line 294)
            # Getting the type of 'errors' (line 294)
            errors_13178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 19), 'errors', False)
            # Processing the call keyword arguments (line 294)
            kwargs_13179 = {}
            # Getting the type of 'len' (line 294)
            len_13177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 15), 'len', False)
            # Calling len(args, kwargs) (line 294)
            len_call_result_13180 = invoke(stypy.reporting.localization.Localization(__file__, 294, 15), len_13177, *[errors_13178], **kwargs_13179)
            
            int_13181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 29), 'int')
            # Applying the binary operator '>' (line 294)
            result_gt_13182 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 15), '>', len_call_result_13180, int_13181)
            
            # Testing if the type of an if condition is none (line 294)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 294, 12), result_gt_13182):
                pass
            else:
                
                # Testing the type of an if condition (line 294)
                if_condition_13183 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 12), result_gt_13182)
                # Assigning a type to the variable 'if_condition_13183' (line 294)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'if_condition_13183', if_condition_13183)
                # SSA begins for if statement (line 294)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 295)
                # Processing the call arguments (line 295)
                
                # Call to UndefinedType(...): (line 295)
                # Processing the call keyword arguments (line 295)
                kwargs_13188 = {}
                # Getting the type of 'undefined_type_copy' (line 295)
                undefined_type_copy_13186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 39), 'undefined_type_copy', False)
                # Obtaining the member 'UndefinedType' of a type (line 295)
                UndefinedType_13187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 39), undefined_type_copy_13186, 'UndefinedType')
                # Calling UndefinedType(args, kwargs) (line 295)
                UndefinedType_call_result_13189 = invoke(stypy.reporting.localization.Localization(__file__, 295, 39), UndefinedType_13187, *[], **kwargs_13188)
                
                # Processing the call keyword arguments (line 295)
                kwargs_13190 = {}
                # Getting the type of 'types_to_return' (line 295)
                types_to_return_13184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'types_to_return', False)
                # Obtaining the member 'append' of a type (line 295)
                append_13185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 16), types_to_return_13184, 'append')
                # Calling append(args, kwargs) (line 295)
                append_call_result_13191 = invoke(stypy.reporting.localization.Localization(__file__, 295, 16), append_13185, *[UndefinedType_call_result_13189], **kwargs_13190)
                
                # SSA join for if statement (line 294)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'errors' (line 301)
            errors_13192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 25), 'errors')
            # Assigning a type to the variable 'errors_13192' (line 301)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'errors_13192', errors_13192)
            # Testing if the for loop is going to be iterated (line 301)
            # Testing the type of a for loop iterable (line 301)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 301, 12), errors_13192)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 301, 12), errors_13192):
                # Getting the type of the for loop variable (line 301)
                for_loop_var_13193 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 301, 12), errors_13192)
                # Assigning a type to the variable 'error' (line 301)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'error', for_loop_var_13193)
                # SSA begins for a for statement (line 301)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 302)
                # Processing the call keyword arguments (line 302)
                kwargs_13196 = {}
                # Getting the type of 'error' (line 302)
                error_13194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 302)
                turn_to_warning_13195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 16), error_13194, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 302)
                turn_to_warning_call_result_13197 = invoke(stypy.reporting.localization.Localization(__file__, 302, 16), turn_to_warning_13195, *[], **kwargs_13196)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 289)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to len(...): (line 306)
        # Processing the call arguments (line 306)
        # Getting the type of 'types_to_return' (line 306)
        types_to_return_13199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 15), 'types_to_return', False)
        # Processing the call keyword arguments (line 306)
        kwargs_13200 = {}
        # Getting the type of 'len' (line 306)
        len_13198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 11), 'len', False)
        # Calling len(args, kwargs) (line 306)
        len_call_result_13201 = invoke(stypy.reporting.localization.Localization(__file__, 306, 11), len_13198, *[types_to_return_13199], **kwargs_13200)
        
        int_13202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 35), 'int')
        # Applying the binary operator '==' (line 306)
        result_eq_13203 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 11), '==', len_call_result_13201, int_13202)
        
        # Testing if the type of an if condition is none (line 306)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 306, 8), result_eq_13203):
            
            # Assigning a Name to a Name (line 309):
            # Getting the type of 'None' (line 309)
            None_13209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 24), 'None')
            # Assigning a type to the variable 'ret_union' (line 309)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'ret_union', None_13209)
            
            # Getting the type of 'types_to_return' (line 310)
            types_to_return_13210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 25), 'types_to_return')
            # Assigning a type to the variable 'types_to_return_13210' (line 310)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'types_to_return_13210', types_to_return_13210)
            # Testing if the for loop is going to be iterated (line 310)
            # Testing the type of a for loop iterable (line 310)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 310, 12), types_to_return_13210)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 310, 12), types_to_return_13210):
                # Getting the type of the for loop variable (line 310)
                for_loop_var_13211 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 310, 12), types_to_return_13210)
                # Assigning a type to the variable 'type_' (line 310)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'type_', for_loop_var_13211)
                # SSA begins for a for statement (line 310)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Call to a Name (line 311):
                
                # Call to add(...): (line 311)
                # Processing the call arguments (line 311)
                # Getting the type of 'ret_union' (line 311)
                ret_union_13214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 42), 'ret_union', False)
                # Getting the type of 'type_' (line 311)
                type__13215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 53), 'type_', False)
                # Processing the call keyword arguments (line 311)
                kwargs_13216 = {}
                # Getting the type of 'UnionType' (line 311)
                UnionType_13212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 28), 'UnionType', False)
                # Obtaining the member 'add' of a type (line 311)
                add_13213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 28), UnionType_13212, 'add')
                # Calling add(args, kwargs) (line 311)
                add_call_result_13217 = invoke(stypy.reporting.localization.Localization(__file__, 311, 28), add_13213, *[ret_union_13214, type__13215], **kwargs_13216)
                
                # Assigning a type to the variable 'ret_union' (line 311)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 16), 'ret_union', add_call_result_13217)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Getting the type of 'ret_union' (line 313)
            ret_union_13218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 19), 'ret_union')
            # Assigning a type to the variable 'stypy_return_type' (line 313)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'stypy_return_type', ret_union_13218)
        else:
            
            # Testing the type of an if condition (line 306)
            if_condition_13204 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 306, 8), result_eq_13203)
            # Assigning a type to the variable 'if_condition_13204' (line 306)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'if_condition_13204', if_condition_13204)
            # SSA begins for if statement (line 306)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining the type of the subscript
            int_13205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 35), 'int')
            # Getting the type of 'types_to_return' (line 307)
            types_to_return_13206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 19), 'types_to_return')
            # Obtaining the member '__getitem__' of a type (line 307)
            getitem___13207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 19), types_to_return_13206, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 307)
            subscript_call_result_13208 = invoke(stypy.reporting.localization.Localization(__file__, 307, 19), getitem___13207, int_13205)
            
            # Assigning a type to the variable 'stypy_return_type' (line 307)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'stypy_return_type', subscript_call_result_13208)
            # SSA branch for the else part of an if statement (line 306)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Name (line 309):
            # Getting the type of 'None' (line 309)
            None_13209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 24), 'None')
            # Assigning a type to the variable 'ret_union' (line 309)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'ret_union', None_13209)
            
            # Getting the type of 'types_to_return' (line 310)
            types_to_return_13210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 25), 'types_to_return')
            # Assigning a type to the variable 'types_to_return_13210' (line 310)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'types_to_return_13210', types_to_return_13210)
            # Testing if the for loop is going to be iterated (line 310)
            # Testing the type of a for loop iterable (line 310)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 310, 12), types_to_return_13210)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 310, 12), types_to_return_13210):
                # Getting the type of the for loop variable (line 310)
                for_loop_var_13211 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 310, 12), types_to_return_13210)
                # Assigning a type to the variable 'type_' (line 310)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'type_', for_loop_var_13211)
                # SSA begins for a for statement (line 310)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Call to a Name (line 311):
                
                # Call to add(...): (line 311)
                # Processing the call arguments (line 311)
                # Getting the type of 'ret_union' (line 311)
                ret_union_13214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 42), 'ret_union', False)
                # Getting the type of 'type_' (line 311)
                type__13215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 53), 'type_', False)
                # Processing the call keyword arguments (line 311)
                kwargs_13216 = {}
                # Getting the type of 'UnionType' (line 311)
                UnionType_13212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 28), 'UnionType', False)
                # Obtaining the member 'add' of a type (line 311)
                add_13213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 28), UnionType_13212, 'add')
                # Calling add(args, kwargs) (line 311)
                add_call_result_13217 = invoke(stypy.reporting.localization.Localization(__file__, 311, 28), add_13213, *[ret_union_13214, type__13215], **kwargs_13216)
                
                # Assigning a type to the variable 'ret_union' (line 311)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 16), 'ret_union', add_call_result_13217)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Getting the type of 'ret_union' (line 313)
            ret_union_13218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 19), 'ret_union')
            # Assigning a type to the variable 'stypy_return_type' (line 313)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'stypy_return_type', ret_union_13218)
            # SSA join for if statement (line 306)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'get_type_of_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_type_of_member' in the type store
        # Getting the type of 'stypy_return_type' (line 266)
        stypy_return_type_13219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13219)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_type_of_member'
        return stypy_return_type_13219


    @staticmethod
    @norecursion
    def __parse_member_value(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__parse_member_value'
        module_type_store = module_type_store.open_function_context('__parse_member_value', 315, 4, False)
        
        # Passed parameters checking function
        UnionType.__parse_member_value.__dict__.__setitem__('stypy_localization', localization)
        UnionType.__parse_member_value.__dict__.__setitem__('stypy_type_of_self', None)
        UnionType.__parse_member_value.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.__parse_member_value.__dict__.__setitem__('stypy_function_name', '__parse_member_value')
        UnionType.__parse_member_value.__dict__.__setitem__('stypy_param_names_list', ['destination', 'member_value'])
        UnionType.__parse_member_value.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.__parse_member_value.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.__parse_member_value.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.__parse_member_value.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.__parse_member_value.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.__parse_member_value.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, '__parse_member_value', ['destination', 'member_value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__parse_member_value', localization, ['member_value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__parse_member_value(...)' code ##################

        str_13220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, (-1)), 'str', '\n        When setting a member of a UnionType to a certain value, each one of the contained types are assigned this\n        member with the specified value (type). However, certain values have to be carefully handled to provide valid\n        values. For example, methods have to be handler in order to provide valid methods to add to each of the\n        UnionType types. This helper method convert a method to a valid method belonging to the destination object.\n\n        :param destination: New owner of the method\n        :param member_value: Method\n        :return THe passed member value, either transformed or not\n        ')
        
        # Call to ismethod(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'member_value' (line 327)
        member_value_13223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 28), 'member_value', False)
        # Processing the call keyword arguments (line 327)
        kwargs_13224 = {}
        # Getting the type of 'inspect' (line 327)
        inspect_13221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 11), 'inspect', False)
        # Obtaining the member 'ismethod' of a type (line 327)
        ismethod_13222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 11), inspect_13221, 'ismethod')
        # Calling ismethod(args, kwargs) (line 327)
        ismethod_call_result_13225 = invoke(stypy.reporting.localization.Localization(__file__, 327, 11), ismethod_13222, *[member_value_13223], **kwargs_13224)
        
        # Testing if the type of an if condition is none (line 327)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 327, 8), ismethod_call_result_13225):
            pass
        else:
            
            # Testing the type of an if condition (line 327)
            if_condition_13226 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 327, 8), ismethod_call_result_13225)
            # Assigning a type to the variable 'if_condition_13226' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'if_condition_13226', if_condition_13226)
            # SSA begins for if statement (line 327)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 329):
            
            # Call to MethodType(...): (line 329)
            # Processing the call arguments (line 329)
            # Getting the type of 'member_value' (line 329)
            member_value_13229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 35), 'member_value', False)
            # Obtaining the member 'im_func' of a type (line 329)
            im_func_13230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 35), member_value_13229, 'im_func')
            # Getting the type of 'destination' (line 329)
            destination_13231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 57), 'destination', False)
            # Processing the call keyword arguments (line 329)
            kwargs_13232 = {}
            # Getting the type of 'types' (line 329)
            types_13227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 18), 'types', False)
            # Obtaining the member 'MethodType' of a type (line 329)
            MethodType_13228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 18), types_13227, 'MethodType')
            # Calling MethodType(args, kwargs) (line 329)
            MethodType_call_result_13233 = invoke(stypy.reporting.localization.Localization(__file__, 329, 18), MethodType_13228, *[im_func_13230, destination_13231], **kwargs_13232)
            
            # Assigning a type to the variable 'met' (line 329)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'met', MethodType_call_result_13233)
            # Getting the type of 'met' (line 330)
            met_13234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 19), 'met')
            # Assigning a type to the variable 'stypy_return_type' (line 330)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'stypy_return_type', met_13234)
            # SSA join for if statement (line 327)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'member_value' (line 332)
        member_value_13235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 15), 'member_value')
        # Assigning a type to the variable 'stypy_return_type' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'stypy_return_type', member_value_13235)
        
        # ################# End of '__parse_member_value(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__parse_member_value' in the type store
        # Getting the type of 'stypy_return_type' (line 315)
        stypy_return_type_13236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13236)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__parse_member_value'
        return stypy_return_type_13236


    @norecursion
    def set_type_of_member(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_type_of_member'
        module_type_store = module_type_store.open_function_context('set_type_of_member', 334, 4, False)
        # Assigning a type to the variable 'self' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.set_type_of_member.__dict__.__setitem__('stypy_localization', localization)
        UnionType.set_type_of_member.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.set_type_of_member.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.set_type_of_member.__dict__.__setitem__('stypy_function_name', 'UnionType.set_type_of_member')
        UnionType.set_type_of_member.__dict__.__setitem__('stypy_param_names_list', ['localization', 'member_name', 'member_value'])
        UnionType.set_type_of_member.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.set_type_of_member.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.set_type_of_member.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.set_type_of_member.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.set_type_of_member.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.set_type_of_member.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.set_type_of_member', ['localization', 'member_name', 'member_value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_type_of_member', localization, ['localization', 'member_name', 'member_value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_type_of_member(...)' code ##################

        str_13237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, (-1)), 'str', '\n        For all the types stored in the union type, set the type of the member named member_name to the type\n        specified in member_value. For example,\n        if a UnionType has the types Class1 and Class2, both with the member "attr" so Class1.attr: int and\n        Class2.attr: str, this method, if passsed a float as member_value will turn both classes "attr" to float.\n        :param localization: Caller information\n        :param member_name: Name of the member to set\n        :param member_value New type of the member\n        :return None or a TypeError if the member cannot be set. Warnings are generated if the member of some of the\n        stored objects cannot be set\n        ')
        
        # Assigning a List to a Name (line 347):
        
        # Obtaining an instance of the builtin type 'list' (line 347)
        list_13238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 347)
        
        # Assigning a type to the variable 'errors' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'errors', list_13238)
        
        # Getting the type of 'self' (line 349)
        self_13239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 21), 'self')
        # Obtaining the member 'types' of a type (line 349)
        types_13240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 21), self_13239, 'types')
        # Assigning a type to the variable 'types_13240' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'types_13240', types_13240)
        # Testing if the for loop is going to be iterated (line 349)
        # Testing the type of a for loop iterable (line 349)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 349, 8), types_13240)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 349, 8), types_13240):
            # Getting the type of the for loop variable (line 349)
            for_loop_var_13241 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 349, 8), types_13240)
            # Assigning a type to the variable 'type_' (line 349)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'type_', for_loop_var_13241)
            # SSA begins for a for statement (line 349)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 350):
            
            # Call to __parse_member_value(...): (line 350)
            # Processing the call arguments (line 350)
            # Getting the type of 'type_' (line 350)
            type__13244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 52), 'type_', False)
            # Getting the type of 'member_value' (line 350)
            member_value_13245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 59), 'member_value', False)
            # Processing the call keyword arguments (line 350)
            kwargs_13246 = {}
            # Getting the type of 'self' (line 350)
            self_13242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 26), 'self', False)
            # Obtaining the member '__parse_member_value' of a type (line 350)
            parse_member_value_13243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 26), self_13242, '__parse_member_value')
            # Calling __parse_member_value(args, kwargs) (line 350)
            parse_member_value_call_result_13247 = invoke(stypy.reporting.localization.Localization(__file__, 350, 26), parse_member_value_13243, *[type__13244, member_value_13245], **kwargs_13246)
            
            # Assigning a type to the variable 'final_value' (line 350)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'final_value', parse_member_value_call_result_13247)
            
            # Assigning a Call to a Name (line 351):
            
            # Call to set_type_of_member(...): (line 351)
            # Processing the call arguments (line 351)
            # Getting the type of 'localization' (line 351)
            localization_13250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 44), 'localization', False)
            # Getting the type of 'member_name' (line 351)
            member_name_13251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 58), 'member_name', False)
            # Getting the type of 'final_value' (line 351)
            final_value_13252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 71), 'final_value', False)
            # Processing the call keyword arguments (line 351)
            kwargs_13253 = {}
            # Getting the type of 'type_' (line 351)
            type__13248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 19), 'type_', False)
            # Obtaining the member 'set_type_of_member' of a type (line 351)
            set_type_of_member_13249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 19), type__13248, 'set_type_of_member')
            # Calling set_type_of_member(args, kwargs) (line 351)
            set_type_of_member_call_result_13254 = invoke(stypy.reporting.localization.Localization(__file__, 351, 19), set_type_of_member_13249, *[localization_13250, member_name_13251, final_value_13252], **kwargs_13253)
            
            # Assigning a type to the variable 'temp' (line 351)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'temp', set_type_of_member_call_result_13254)
            
            # Type idiom detected: calculating its left and rigth part (line 352)
            # Getting the type of 'temp' (line 352)
            temp_13255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'temp')
            # Getting the type of 'None' (line 352)
            None_13256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 27), 'None')
            
            (may_be_13257, more_types_in_union_13258) = may_not_be_none(temp_13255, None_13256)

            if may_be_13257:

                if more_types_in_union_13258:
                    # Runtime conditional SSA (line 352)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to append(...): (line 353)
                # Processing the call arguments (line 353)
                # Getting the type of 'temp' (line 353)
                temp_13261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 30), 'temp', False)
                # Processing the call keyword arguments (line 353)
                kwargs_13262 = {}
                # Getting the type of 'errors' (line 353)
                errors_13259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 16), 'errors', False)
                # Obtaining the member 'append' of a type (line 353)
                append_13260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 16), errors_13259, 'append')
                # Calling append(args, kwargs) (line 353)
                append_call_result_13263 = invoke(stypy.reporting.localization.Localization(__file__, 353, 16), append_13260, *[temp_13261], **kwargs_13262)
                

                if more_types_in_union_13258:
                    # SSA join for if statement (line 352)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'errors' (line 356)
        errors_13265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 15), 'errors', False)
        # Processing the call keyword arguments (line 356)
        kwargs_13266 = {}
        # Getting the type of 'len' (line 356)
        len_13264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 11), 'len', False)
        # Calling len(args, kwargs) (line 356)
        len_call_result_13267 = invoke(stypy.reporting.localization.Localization(__file__, 356, 11), len_13264, *[errors_13265], **kwargs_13266)
        
        
        # Call to len(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'self' (line 356)
        self_13269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 356)
        types_13270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 30), self_13269, 'types')
        # Processing the call keyword arguments (line 356)
        kwargs_13271 = {}
        # Getting the type of 'len' (line 356)
        len_13268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 26), 'len', False)
        # Calling len(args, kwargs) (line 356)
        len_call_result_13272 = invoke(stypy.reporting.localization.Localization(__file__, 356, 26), len_13268, *[types_13270], **kwargs_13271)
        
        # Applying the binary operator '==' (line 356)
        result_eq_13273 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 11), '==', len_call_result_13267, len_call_result_13272)
        
        # Testing if the type of an if condition is none (line 356)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 356, 8), result_eq_13273):
            
            # Getting the type of 'errors' (line 363)
            errors_13286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 25), 'errors')
            # Assigning a type to the variable 'errors_13286' (line 363)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'errors_13286', errors_13286)
            # Testing if the for loop is going to be iterated (line 363)
            # Testing the type of a for loop iterable (line 363)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 363, 12), errors_13286)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 363, 12), errors_13286):
                # Getting the type of the for loop variable (line 363)
                for_loop_var_13287 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 363, 12), errors_13286)
                # Assigning a type to the variable 'error' (line 363)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'error', for_loop_var_13287)
                # SSA begins for a for statement (line 363)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 364)
                # Processing the call keyword arguments (line 364)
                kwargs_13290 = {}
                # Getting the type of 'error' (line 364)
                error_13288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 364)
                turn_to_warning_13289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 16), error_13288, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 364)
                turn_to_warning_call_result_13291 = invoke(stypy.reporting.localization.Localization(__file__, 364, 16), turn_to_warning_13289, *[], **kwargs_13290)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 356)
            if_condition_13274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 356, 8), result_eq_13273)
            # Assigning a type to the variable 'if_condition_13274' (line 356)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'if_condition_13274', if_condition_13274)
            # SSA begins for if statement (line 356)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 357)
            # Processing the call arguments (line 357)
            # Getting the type of 'localization' (line 357)
            localization_13276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 29), 'localization', False)
            
            # Call to format(...): (line 357)
            # Processing the call arguments (line 357)
            # Getting the type of 'member_name' (line 358)
            member_name_13279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 16), 'member_name', False)
            # Getting the type of 'self' (line 358)
            self_13280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 29), 'self', False)
            # Obtaining the member 'types' of a type (line 358)
            types_13281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 29), self_13280, 'types')
            # Processing the call keyword arguments (line 357)
            kwargs_13282 = {}
            str_13277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 43), 'str', "None of the possible types ('{1}') can set the member '{0}'")
            # Obtaining the member 'format' of a type (line 357)
            format_13278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 43), str_13277, 'format')
            # Calling format(args, kwargs) (line 357)
            format_call_result_13283 = invoke(stypy.reporting.localization.Localization(__file__, 357, 43), format_13278, *[member_name_13279, types_13281], **kwargs_13282)
            
            # Processing the call keyword arguments (line 357)
            kwargs_13284 = {}
            # Getting the type of 'TypeError' (line 357)
            TypeError_13275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 357)
            TypeError_call_result_13285 = invoke(stypy.reporting.localization.Localization(__file__, 357, 19), TypeError_13275, *[localization_13276, format_call_result_13283], **kwargs_13284)
            
            # Assigning a type to the variable 'stypy_return_type' (line 357)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'stypy_return_type', TypeError_call_result_13285)
            # SSA branch for the else part of an if statement (line 356)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 363)
            errors_13286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 25), 'errors')
            # Assigning a type to the variable 'errors_13286' (line 363)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'errors_13286', errors_13286)
            # Testing if the for loop is going to be iterated (line 363)
            # Testing the type of a for loop iterable (line 363)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 363, 12), errors_13286)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 363, 12), errors_13286):
                # Getting the type of the for loop variable (line 363)
                for_loop_var_13287 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 363, 12), errors_13286)
                # Assigning a type to the variable 'error' (line 363)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'error', for_loop_var_13287)
                # SSA begins for a for statement (line 363)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 364)
                # Processing the call keyword arguments (line 364)
                kwargs_13290 = {}
                # Getting the type of 'error' (line 364)
                error_13288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 364)
                turn_to_warning_13289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 16), error_13288, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 364)
                turn_to_warning_call_result_13291 = invoke(stypy.reporting.localization.Localization(__file__, 364, 16), turn_to_warning_13289, *[], **kwargs_13290)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 356)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'None' (line 366)
        None_13292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'stypy_return_type', None_13292)
        
        # ################# End of 'set_type_of_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_type_of_member' in the type store
        # Getting the type of 'stypy_return_type' (line 334)
        stypy_return_type_13293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13293)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_type_of_member'
        return stypy_return_type_13293


    @norecursion
    def invoke(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'invoke'
        module_type_store = module_type_store.open_function_context('invoke', 370, 4, False)
        # Assigning a type to the variable 'self' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.invoke.__dict__.__setitem__('stypy_localization', localization)
        UnionType.invoke.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.invoke.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.invoke.__dict__.__setitem__('stypy_function_name', 'UnionType.invoke')
        UnionType.invoke.__dict__.__setitem__('stypy_param_names_list', ['localization'])
        UnionType.invoke.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        UnionType.invoke.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        UnionType.invoke.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.invoke.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.invoke.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.invoke.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.invoke', ['localization'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'invoke', localization, ['localization'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'invoke(...)' code ##################

        str_13294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, (-1)), 'str', '\n        For all the types stored in the union type, invoke them with the provided parameters.\n        :param localization: Caller information\n        :param args: Arguments of the call\n        :param kwargs: Keyword arguments of the call\n        :return All the types that the call could return, examining the UnionType stored types\n        ')
        
        # Assigning a List to a Name (line 378):
        
        # Obtaining an instance of the builtin type 'list' (line 378)
        list_13295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 378)
        
        # Assigning a type to the variable 'result' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'result', list_13295)
        
        # Getting the type of 'self' (line 380)
        self_13296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 21), 'self')
        # Obtaining the member 'types' of a type (line 380)
        types_13297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 21), self_13296, 'types')
        # Assigning a type to the variable 'types_13297' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'types_13297', types_13297)
        # Testing if the for loop is going to be iterated (line 380)
        # Testing the type of a for loop iterable (line 380)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 380, 8), types_13297)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 380, 8), types_13297):
            # Getting the type of the for loop variable (line 380)
            for_loop_var_13298 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 380, 8), types_13297)
            # Assigning a type to the variable 'type_' (line 380)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'type_', for_loop_var_13298)
            # SSA begins for a for statement (line 380)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 382):
            
            # Call to invoke(...): (line 382)
            # Processing the call arguments (line 382)
            # Getting the type of 'localization' (line 382)
            localization_13301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 32), 'localization', False)
            # Getting the type of 'args' (line 382)
            args_13302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 47), 'args', False)
            # Processing the call keyword arguments (line 382)
            # Getting the type of 'kwargs' (line 382)
            kwargs_13303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 55), 'kwargs', False)
            kwargs_13304 = {'kwargs_13303': kwargs_13303}
            # Getting the type of 'type_' (line 382)
            type__13299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 19), 'type_', False)
            # Obtaining the member 'invoke' of a type (line 382)
            invoke_13300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 19), type__13299, 'invoke')
            # Calling invoke(args, kwargs) (line 382)
            invoke_call_result_13305 = invoke(stypy.reporting.localization.Localization(__file__, 382, 19), invoke_13300, *[localization_13301, args_13302], **kwargs_13304)
            
            # Assigning a type to the variable 'temp' (line 382)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'temp', invoke_call_result_13305)
            
            # Call to append(...): (line 383)
            # Processing the call arguments (line 383)
            # Getting the type of 'temp' (line 383)
            temp_13308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 26), 'temp', False)
            # Processing the call keyword arguments (line 383)
            kwargs_13309 = {}
            # Getting the type of 'result' (line 383)
            result_13306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'result', False)
            # Obtaining the member 'append' of a type (line 383)
            append_13307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 12), result_13306, 'append')
            # Calling append(args, kwargs) (line 383)
            append_call_result_13310 = invoke(stypy.reporting.localization.Localization(__file__, 383, 12), append_13307, *[temp_13308], **kwargs_13309)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 386):
        
        # Call to filter(...): (line 386)
        # Processing the call arguments (line 386)

        @norecursion
        def _stypy_temp_lambda_23(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_23'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_23', 386, 24, True)
            # Passed parameters checking function
            _stypy_temp_lambda_23.stypy_localization = localization
            _stypy_temp_lambda_23.stypy_type_of_self = None
            _stypy_temp_lambda_23.stypy_type_store = module_type_store
            _stypy_temp_lambda_23.stypy_function_name = '_stypy_temp_lambda_23'
            _stypy_temp_lambda_23.stypy_param_names_list = ['t']
            _stypy_temp_lambda_23.stypy_varargs_param_name = None
            _stypy_temp_lambda_23.stypy_kwargs_param_name = None
            _stypy_temp_lambda_23.stypy_call_defaults = defaults
            _stypy_temp_lambda_23.stypy_call_varargs = varargs
            _stypy_temp_lambda_23.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_23', ['t'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_23', ['t'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to isinstance(...): (line 386)
            # Processing the call arguments (line 386)
            # Getting the type of 't' (line 386)
            t_13313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 45), 't', False)
            # Getting the type of 'TypeError' (line 386)
            TypeError_13314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 48), 'TypeError', False)
            # Processing the call keyword arguments (line 386)
            kwargs_13315 = {}
            # Getting the type of 'isinstance' (line 386)
            isinstance_13312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 34), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 386)
            isinstance_call_result_13316 = invoke(stypy.reporting.localization.Localization(__file__, 386, 34), isinstance_13312, *[t_13313, TypeError_13314], **kwargs_13315)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 386)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 24), 'stypy_return_type', isinstance_call_result_13316)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_23' in the type store
            # Getting the type of 'stypy_return_type' (line 386)
            stypy_return_type_13317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 24), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_13317)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_23'
            return stypy_return_type_13317

        # Assigning a type to the variable '_stypy_temp_lambda_23' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 24), '_stypy_temp_lambda_23', _stypy_temp_lambda_23)
        # Getting the type of '_stypy_temp_lambda_23' (line 386)
        _stypy_temp_lambda_23_13318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 24), '_stypy_temp_lambda_23')
        # Getting the type of 'result' (line 386)
        result_13319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 60), 'result', False)
        # Processing the call keyword arguments (line 386)
        kwargs_13320 = {}
        # Getting the type of 'filter' (line 386)
        filter_13311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 17), 'filter', False)
        # Calling filter(args, kwargs) (line 386)
        filter_call_result_13321 = invoke(stypy.reporting.localization.Localization(__file__, 386, 17), filter_13311, *[_stypy_temp_lambda_23_13318, result_13319], **kwargs_13320)
        
        # Assigning a type to the variable 'errors' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'errors', filter_call_result_13321)
        
        # Assigning a Call to a Name (line 389):
        
        # Call to filter(...): (line 389)
        # Processing the call arguments (line 389)

        @norecursion
        def _stypy_temp_lambda_24(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_24'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_24', 389, 33, True)
            # Passed parameters checking function
            _stypy_temp_lambda_24.stypy_localization = localization
            _stypy_temp_lambda_24.stypy_type_of_self = None
            _stypy_temp_lambda_24.stypy_type_store = module_type_store
            _stypy_temp_lambda_24.stypy_function_name = '_stypy_temp_lambda_24'
            _stypy_temp_lambda_24.stypy_param_names_list = ['t']
            _stypy_temp_lambda_24.stypy_varargs_param_name = None
            _stypy_temp_lambda_24.stypy_kwargs_param_name = None
            _stypy_temp_lambda_24.stypy_call_defaults = defaults
            _stypy_temp_lambda_24.stypy_call_varargs = varargs
            _stypy_temp_lambda_24.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_24', ['t'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_24', ['t'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            
            # Call to isinstance(...): (line 389)
            # Processing the call arguments (line 389)
            # Getting the type of 't' (line 389)
            t_13324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 58), 't', False)
            # Getting the type of 'TypeError' (line 389)
            TypeError_13325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 61), 'TypeError', False)
            # Processing the call keyword arguments (line 389)
            kwargs_13326 = {}
            # Getting the type of 'isinstance' (line 389)
            isinstance_13323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 47), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 389)
            isinstance_call_result_13327 = invoke(stypy.reporting.localization.Localization(__file__, 389, 47), isinstance_13323, *[t_13324, TypeError_13325], **kwargs_13326)
            
            # Applying the 'not' unary operator (line 389)
            result_not__13328 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 43), 'not', isinstance_call_result_13327)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 389)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 33), 'stypy_return_type', result_not__13328)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_24' in the type store
            # Getting the type of 'stypy_return_type' (line 389)
            stypy_return_type_13329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 33), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_13329)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_24'
            return stypy_return_type_13329

        # Assigning a type to the variable '_stypy_temp_lambda_24' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 33), '_stypy_temp_lambda_24', _stypy_temp_lambda_24)
        # Getting the type of '_stypy_temp_lambda_24' (line 389)
        _stypy_temp_lambda_24_13330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 33), '_stypy_temp_lambda_24')
        # Getting the type of 'result' (line 389)
        result_13331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 73), 'result', False)
        # Processing the call keyword arguments (line 389)
        kwargs_13332 = {}
        # Getting the type of 'filter' (line 389)
        filter_13322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 26), 'filter', False)
        # Calling filter(args, kwargs) (line 389)
        filter_call_result_13333 = invoke(stypy.reporting.localization.Localization(__file__, 389, 26), filter_13322, *[_stypy_temp_lambda_24_13330, result_13331], **kwargs_13332)
        
        # Assigning a type to the variable 'types_to_return' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'types_to_return', filter_call_result_13333)
        
        
        # Call to len(...): (line 392)
        # Processing the call arguments (line 392)
        # Getting the type of 'errors' (line 392)
        errors_13335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 15), 'errors', False)
        # Processing the call keyword arguments (line 392)
        kwargs_13336 = {}
        # Getting the type of 'len' (line 392)
        len_13334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 11), 'len', False)
        # Calling len(args, kwargs) (line 392)
        len_call_result_13337 = invoke(stypy.reporting.localization.Localization(__file__, 392, 11), len_13334, *[errors_13335], **kwargs_13336)
        
        
        # Call to len(...): (line 392)
        # Processing the call arguments (line 392)
        # Getting the type of 'result' (line 392)
        result_13339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 30), 'result', False)
        # Processing the call keyword arguments (line 392)
        kwargs_13340 = {}
        # Getting the type of 'len' (line 392)
        len_13338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 26), 'len', False)
        # Calling len(args, kwargs) (line 392)
        len_call_result_13341 = invoke(stypy.reporting.localization.Localization(__file__, 392, 26), len_13338, *[result_13339], **kwargs_13340)
        
        # Applying the binary operator '==' (line 392)
        result_eq_13342 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 11), '==', len_call_result_13337, len_call_result_13341)
        
        # Testing if the type of an if condition is none (line 392)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 392, 8), result_eq_13342):
            
            # Getting the type of 'errors' (line 402)
            errors_13381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 25), 'errors')
            # Assigning a type to the variable 'errors_13381' (line 402)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'errors_13381', errors_13381)
            # Testing if the for loop is going to be iterated (line 402)
            # Testing the type of a for loop iterable (line 402)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 402, 12), errors_13381)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 402, 12), errors_13381):
                # Getting the type of the for loop variable (line 402)
                for_loop_var_13382 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 402, 12), errors_13381)
                # Assigning a type to the variable 'error' (line 402)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'error', for_loop_var_13382)
                # SSA begins for a for statement (line 402)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 403)
                # Processing the call keyword arguments (line 403)
                kwargs_13385 = {}
                # Getting the type of 'error' (line 403)
                error_13383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 403)
                turn_to_warning_13384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 16), error_13383, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 403)
                turn_to_warning_call_result_13386 = invoke(stypy.reporting.localization.Localization(__file__, 403, 16), turn_to_warning_13384, *[], **kwargs_13385)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 392)
            if_condition_13343 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 392, 8), result_eq_13342)
            # Assigning a type to the variable 'if_condition_13343' (line 392)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'if_condition_13343', if_condition_13343)
            # SSA begins for if statement (line 392)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'errors' (line 393)
            errors_13344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 25), 'errors')
            # Assigning a type to the variable 'errors_13344' (line 393)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'errors_13344', errors_13344)
            # Testing if the for loop is going to be iterated (line 393)
            # Testing the type of a for loop iterable (line 393)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 393, 12), errors_13344)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 393, 12), errors_13344):
                # Getting the type of the for loop variable (line 393)
                for_loop_var_13345 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 393, 12), errors_13344)
                # Assigning a type to the variable 'error' (line 393)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'error', for_loop_var_13345)
                # SSA begins for a for statement (line 393)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to remove_error_msg(...): (line 394)
                # Processing the call arguments (line 394)
                # Getting the type of 'error' (line 394)
                error_13348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 43), 'error', False)
                # Processing the call keyword arguments (line 394)
                kwargs_13349 = {}
                # Getting the type of 'TypeError' (line 394)
                TypeError_13346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 16), 'TypeError', False)
                # Obtaining the member 'remove_error_msg' of a type (line 394)
                remove_error_msg_13347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 16), TypeError_13346, 'remove_error_msg')
                # Calling remove_error_msg(args, kwargs) (line 394)
                remove_error_msg_call_result_13350 = invoke(stypy.reporting.localization.Localization(__file__, 394, 16), remove_error_msg_13347, *[error_13348], **kwargs_13349)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a Call to a Name (line 395):
            
            # Call to tuple(...): (line 395)
            # Processing the call arguments (line 395)
            
            # Call to list(...): (line 395)
            # Processing the call arguments (line 395)
            # Getting the type of 'args' (line 395)
            args_13353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 32), 'args', False)
            # Processing the call keyword arguments (line 395)
            kwargs_13354 = {}
            # Getting the type of 'list' (line 395)
            list_13352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 27), 'list', False)
            # Calling list(args, kwargs) (line 395)
            list_call_result_13355 = invoke(stypy.reporting.localization.Localization(__file__, 395, 27), list_13352, *[args_13353], **kwargs_13354)
            
            
            # Call to values(...): (line 395)
            # Processing the call keyword arguments (line 395)
            kwargs_13358 = {}
            # Getting the type of 'kwargs' (line 395)
            kwargs_13356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 40), 'kwargs', False)
            # Obtaining the member 'values' of a type (line 395)
            values_13357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 40), kwargs_13356, 'values')
            # Calling values(args, kwargs) (line 395)
            values_call_result_13359 = invoke(stypy.reporting.localization.Localization(__file__, 395, 40), values_13357, *[], **kwargs_13358)
            
            # Applying the binary operator '+' (line 395)
            result_add_13360 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 27), '+', list_call_result_13355, values_call_result_13359)
            
            # Processing the call keyword arguments (line 395)
            kwargs_13361 = {}
            # Getting the type of 'tuple' (line 395)
            tuple_13351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 21), 'tuple', False)
            # Calling tuple(args, kwargs) (line 395)
            tuple_call_result_13362 = invoke(stypy.reporting.localization.Localization(__file__, 395, 21), tuple_13351, *[result_add_13360], **kwargs_13361)
            
            # Assigning a type to the variable 'params' (line 395)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'params', tuple_call_result_13362)
            
            # Call to TypeError(...): (line 396)
            # Processing the call arguments (line 396)
            # Getting the type of 'localization' (line 396)
            localization_13364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 29), 'localization', False)
            
            # Call to format(...): (line 396)
            # Processing the call arguments (line 396)
            
            # Call to format_function_name(...): (line 397)
            # Processing the call arguments (line 397)
            
            # Obtaining the type of the subscript
            int_13368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 48), 'int')
            # Getting the type of 'self' (line 397)
            self_13369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 37), 'self', False)
            # Obtaining the member 'types' of a type (line 397)
            types_13370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 37), self_13369, 'types')
            # Obtaining the member '__getitem__' of a type (line 397)
            getitem___13371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 37), types_13370, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 397)
            subscript_call_result_13372 = invoke(stypy.reporting.localization.Localization(__file__, 397, 37), getitem___13371, int_13368)
            
            # Obtaining the member 'name' of a type (line 397)
            name_13373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 37), subscript_call_result_13372, 'name')
            # Processing the call keyword arguments (line 397)
            kwargs_13374 = {}
            # Getting the type of 'format_function_name' (line 397)
            format_function_name_13367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 16), 'format_function_name', False)
            # Calling format_function_name(args, kwargs) (line 397)
            format_function_name_call_result_13375 = invoke(stypy.reporting.localization.Localization(__file__, 397, 16), format_function_name_13367, *[name_13373], **kwargs_13374)
            
            # Getting the type of 'params' (line 397)
            params_13376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 58), 'params', False)
            # Processing the call keyword arguments (line 396)
            kwargs_13377 = {}
            str_13365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 43), 'str', 'Cannot invoke {0} with parameters {1}')
            # Obtaining the member 'format' of a type (line 396)
            format_13366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 43), str_13365, 'format')
            # Calling format(args, kwargs) (line 396)
            format_call_result_13378 = invoke(stypy.reporting.localization.Localization(__file__, 396, 43), format_13366, *[format_function_name_call_result_13375, params_13376], **kwargs_13377)
            
            # Processing the call keyword arguments (line 396)
            kwargs_13379 = {}
            # Getting the type of 'TypeError' (line 396)
            TypeError_13363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 396)
            TypeError_call_result_13380 = invoke(stypy.reporting.localization.Localization(__file__, 396, 19), TypeError_13363, *[localization_13364, format_call_result_13378], **kwargs_13379)
            
            # Assigning a type to the variable 'stypy_return_type' (line 396)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'stypy_return_type', TypeError_call_result_13380)
            # SSA branch for the else part of an if statement (line 392)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 402)
            errors_13381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 25), 'errors')
            # Assigning a type to the variable 'errors_13381' (line 402)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'errors_13381', errors_13381)
            # Testing if the for loop is going to be iterated (line 402)
            # Testing the type of a for loop iterable (line 402)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 402, 12), errors_13381)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 402, 12), errors_13381):
                # Getting the type of the for loop variable (line 402)
                for_loop_var_13382 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 402, 12), errors_13381)
                # Assigning a type to the variable 'error' (line 402)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'error', for_loop_var_13382)
                # SSA begins for a for statement (line 402)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 403)
                # Processing the call keyword arguments (line 403)
                kwargs_13385 = {}
                # Getting the type of 'error' (line 403)
                error_13383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 403)
                turn_to_warning_13384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 16), error_13383, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 403)
                turn_to_warning_call_result_13386 = invoke(stypy.reporting.localization.Localization(__file__, 403, 16), turn_to_warning_13384, *[], **kwargs_13385)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 392)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to len(...): (line 406)
        # Processing the call arguments (line 406)
        # Getting the type of 'types_to_return' (line 406)
        types_to_return_13388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 15), 'types_to_return', False)
        # Processing the call keyword arguments (line 406)
        kwargs_13389 = {}
        # Getting the type of 'len' (line 406)
        len_13387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 11), 'len', False)
        # Calling len(args, kwargs) (line 406)
        len_call_result_13390 = invoke(stypy.reporting.localization.Localization(__file__, 406, 11), len_13387, *[types_to_return_13388], **kwargs_13389)
        
        int_13391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 35), 'int')
        # Applying the binary operator '==' (line 406)
        result_eq_13392 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 11), '==', len_call_result_13390, int_13391)
        
        # Testing if the type of an if condition is none (line 406)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 406, 8), result_eq_13392):
            
            # Assigning a Name to a Name (line 409):
            # Getting the type of 'None' (line 409)
            None_13398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 24), 'None')
            # Assigning a type to the variable 'ret_union' (line 409)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'ret_union', None_13398)
            
            # Getting the type of 'types_to_return' (line 410)
            types_to_return_13399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 25), 'types_to_return')
            # Assigning a type to the variable 'types_to_return_13399' (line 410)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'types_to_return_13399', types_to_return_13399)
            # Testing if the for loop is going to be iterated (line 410)
            # Testing the type of a for loop iterable (line 410)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 410, 12), types_to_return_13399)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 410, 12), types_to_return_13399):
                # Getting the type of the for loop variable (line 410)
                for_loop_var_13400 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 410, 12), types_to_return_13399)
                # Assigning a type to the variable 'type_' (line 410)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'type_', for_loop_var_13400)
                # SSA begins for a for statement (line 410)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Call to a Name (line 411):
                
                # Call to add(...): (line 411)
                # Processing the call arguments (line 411)
                # Getting the type of 'ret_union' (line 411)
                ret_union_13403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 42), 'ret_union', False)
                # Getting the type of 'type_' (line 411)
                type__13404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 53), 'type_', False)
                # Processing the call keyword arguments (line 411)
                kwargs_13405 = {}
                # Getting the type of 'UnionType' (line 411)
                UnionType_13401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 28), 'UnionType', False)
                # Obtaining the member 'add' of a type (line 411)
                add_13402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 28), UnionType_13401, 'add')
                # Calling add(args, kwargs) (line 411)
                add_call_result_13406 = invoke(stypy.reporting.localization.Localization(__file__, 411, 28), add_13402, *[ret_union_13403, type__13404], **kwargs_13405)
                
                # Assigning a type to the variable 'ret_union' (line 411)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 16), 'ret_union', add_call_result_13406)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Getting the type of 'ret_union' (line 413)
            ret_union_13407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 19), 'ret_union')
            # Assigning a type to the variable 'stypy_return_type' (line 413)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'stypy_return_type', ret_union_13407)
        else:
            
            # Testing the type of an if condition (line 406)
            if_condition_13393 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 406, 8), result_eq_13392)
            # Assigning a type to the variable 'if_condition_13393' (line 406)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'if_condition_13393', if_condition_13393)
            # SSA begins for if statement (line 406)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining the type of the subscript
            int_13394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 35), 'int')
            # Getting the type of 'types_to_return' (line 407)
            types_to_return_13395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 19), 'types_to_return')
            # Obtaining the member '__getitem__' of a type (line 407)
            getitem___13396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 19), types_to_return_13395, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 407)
            subscript_call_result_13397 = invoke(stypy.reporting.localization.Localization(__file__, 407, 19), getitem___13396, int_13394)
            
            # Assigning a type to the variable 'stypy_return_type' (line 407)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'stypy_return_type', subscript_call_result_13397)
            # SSA branch for the else part of an if statement (line 406)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Name (line 409):
            # Getting the type of 'None' (line 409)
            None_13398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 24), 'None')
            # Assigning a type to the variable 'ret_union' (line 409)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'ret_union', None_13398)
            
            # Getting the type of 'types_to_return' (line 410)
            types_to_return_13399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 25), 'types_to_return')
            # Assigning a type to the variable 'types_to_return_13399' (line 410)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'types_to_return_13399', types_to_return_13399)
            # Testing if the for loop is going to be iterated (line 410)
            # Testing the type of a for loop iterable (line 410)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 410, 12), types_to_return_13399)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 410, 12), types_to_return_13399):
                # Getting the type of the for loop variable (line 410)
                for_loop_var_13400 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 410, 12), types_to_return_13399)
                # Assigning a type to the variable 'type_' (line 410)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'type_', for_loop_var_13400)
                # SSA begins for a for statement (line 410)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Call to a Name (line 411):
                
                # Call to add(...): (line 411)
                # Processing the call arguments (line 411)
                # Getting the type of 'ret_union' (line 411)
                ret_union_13403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 42), 'ret_union', False)
                # Getting the type of 'type_' (line 411)
                type__13404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 53), 'type_', False)
                # Processing the call keyword arguments (line 411)
                kwargs_13405 = {}
                # Getting the type of 'UnionType' (line 411)
                UnionType_13401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 28), 'UnionType', False)
                # Obtaining the member 'add' of a type (line 411)
                add_13402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 28), UnionType_13401, 'add')
                # Calling add(args, kwargs) (line 411)
                add_call_result_13406 = invoke(stypy.reporting.localization.Localization(__file__, 411, 28), add_13402, *[ret_union_13403, type__13404], **kwargs_13405)
                
                # Assigning a type to the variable 'ret_union' (line 411)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 16), 'ret_union', add_call_result_13406)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Getting the type of 'ret_union' (line 413)
            ret_union_13407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 19), 'ret_union')
            # Assigning a type to the variable 'stypy_return_type' (line 413)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'stypy_return_type', ret_union_13407)
            # SSA join for if statement (line 406)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'invoke(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'invoke' in the type store
        # Getting the type of 'stypy_return_type' (line 370)
        stypy_return_type_13408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13408)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'invoke'
        return stypy_return_type_13408


    @norecursion
    def delete_member(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'delete_member'
        module_type_store = module_type_store.open_function_context('delete_member', 417, 4, False)
        # Assigning a type to the variable 'self' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.delete_member.__dict__.__setitem__('stypy_localization', localization)
        UnionType.delete_member.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.delete_member.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.delete_member.__dict__.__setitem__('stypy_function_name', 'UnionType.delete_member')
        UnionType.delete_member.__dict__.__setitem__('stypy_param_names_list', ['localization', 'member'])
        UnionType.delete_member.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.delete_member.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.delete_member.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.delete_member.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.delete_member.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.delete_member.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.delete_member', ['localization', 'member'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'delete_member', localization, ['localization', 'member'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'delete_member(...)' code ##################

        str_13409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, (-1)), 'str', '\n        For all the types stored in the union type, delete the member named member_name, returning None or a TypeError\n        if no type stored in the UnionType supports member deletion.\n        :param localization: Caller information\n        :param member: Member to delete\n        :return None or TypeError\n        ')
        
        # Assigning a List to a Name (line 425):
        
        # Obtaining an instance of the builtin type 'list' (line 425)
        list_13410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 425)
        
        # Assigning a type to the variable 'errors' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'errors', list_13410)
        
        # Getting the type of 'self' (line 427)
        self_13411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 21), 'self')
        # Obtaining the member 'types' of a type (line 427)
        types_13412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 21), self_13411, 'types')
        # Assigning a type to the variable 'types_13412' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'types_13412', types_13412)
        # Testing if the for loop is going to be iterated (line 427)
        # Testing the type of a for loop iterable (line 427)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 427, 8), types_13412)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 427, 8), types_13412):
            # Getting the type of the for loop variable (line 427)
            for_loop_var_13413 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 427, 8), types_13412)
            # Assigning a type to the variable 'type_' (line 427)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'type_', for_loop_var_13413)
            # SSA begins for a for statement (line 427)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 428):
            
            # Call to delete_member(...): (line 428)
            # Processing the call arguments (line 428)
            # Getting the type of 'localization' (line 428)
            localization_13416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 39), 'localization', False)
            # Getting the type of 'member' (line 428)
            member_13417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 53), 'member', False)
            # Processing the call keyword arguments (line 428)
            kwargs_13418 = {}
            # Getting the type of 'type_' (line 428)
            type__13414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 19), 'type_', False)
            # Obtaining the member 'delete_member' of a type (line 428)
            delete_member_13415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 19), type__13414, 'delete_member')
            # Calling delete_member(args, kwargs) (line 428)
            delete_member_call_result_13419 = invoke(stypy.reporting.localization.Localization(__file__, 428, 19), delete_member_13415, *[localization_13416, member_13417], **kwargs_13418)
            
            # Assigning a type to the variable 'temp' (line 428)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 12), 'temp', delete_member_call_result_13419)
            
            # Type idiom detected: calculating its left and rigth part (line 429)
            # Getting the type of 'temp' (line 429)
            temp_13420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'temp')
            # Getting the type of 'None' (line 429)
            None_13421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 27), 'None')
            
            (may_be_13422, more_types_in_union_13423) = may_not_be_none(temp_13420, None_13421)

            if may_be_13422:

                if more_types_in_union_13423:
                    # Runtime conditional SSA (line 429)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to append(...): (line 430)
                # Processing the call arguments (line 430)
                # Getting the type of 'temp' (line 430)
                temp_13426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 30), 'temp', False)
                # Processing the call keyword arguments (line 430)
                kwargs_13427 = {}
                # Getting the type of 'errors' (line 430)
                errors_13424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 16), 'errors', False)
                # Obtaining the member 'append' of a type (line 430)
                append_13425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 16), errors_13424, 'append')
                # Calling append(args, kwargs) (line 430)
                append_call_result_13428 = invoke(stypy.reporting.localization.Localization(__file__, 430, 16), append_13425, *[temp_13426], **kwargs_13427)
                

                if more_types_in_union_13423:
                    # SSA join for if statement (line 429)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 'errors' (line 433)
        errors_13430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 15), 'errors', False)
        # Processing the call keyword arguments (line 433)
        kwargs_13431 = {}
        # Getting the type of 'len' (line 433)
        len_13429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 11), 'len', False)
        # Calling len(args, kwargs) (line 433)
        len_call_result_13432 = invoke(stypy.reporting.localization.Localization(__file__, 433, 11), len_13429, *[errors_13430], **kwargs_13431)
        
        
        # Call to len(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 'self' (line 433)
        self_13434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 433)
        types_13435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 30), self_13434, 'types')
        # Processing the call keyword arguments (line 433)
        kwargs_13436 = {}
        # Getting the type of 'len' (line 433)
        len_13433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 26), 'len', False)
        # Calling len(args, kwargs) (line 433)
        len_call_result_13437 = invoke(stypy.reporting.localization.Localization(__file__, 433, 26), len_13433, *[types_13435], **kwargs_13436)
        
        # Applying the binary operator '==' (line 433)
        result_eq_13438 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 11), '==', len_call_result_13432, len_call_result_13437)
        
        # Testing if the type of an if condition is none (line 433)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 433, 8), result_eq_13438):
            
            # Getting the type of 'errors' (line 440)
            errors_13451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 25), 'errors')
            # Assigning a type to the variable 'errors_13451' (line 440)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'errors_13451', errors_13451)
            # Testing if the for loop is going to be iterated (line 440)
            # Testing the type of a for loop iterable (line 440)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 440, 12), errors_13451)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 440, 12), errors_13451):
                # Getting the type of the for loop variable (line 440)
                for_loop_var_13452 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 440, 12), errors_13451)
                # Assigning a type to the variable 'error' (line 440)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'error', for_loop_var_13452)
                # SSA begins for a for statement (line 440)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 441)
                # Processing the call keyword arguments (line 441)
                kwargs_13455 = {}
                # Getting the type of 'error' (line 441)
                error_13453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 441)
                turn_to_warning_13454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 16), error_13453, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 441)
                turn_to_warning_call_result_13456 = invoke(stypy.reporting.localization.Localization(__file__, 441, 16), turn_to_warning_13454, *[], **kwargs_13455)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 433)
            if_condition_13439 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 433, 8), result_eq_13438)
            # Assigning a type to the variable 'if_condition_13439' (line 433)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'if_condition_13439', if_condition_13439)
            # SSA begins for if statement (line 433)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 434)
            # Processing the call arguments (line 434)
            # Getting the type of 'localization' (line 434)
            localization_13441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 29), 'localization', False)
            
            # Call to format(...): (line 434)
            # Processing the call arguments (line 434)
            # Getting the type of 'member' (line 435)
            member_13444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 36), 'member', False)
            # Getting the type of 'self' (line 435)
            self_13445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 44), 'self', False)
            # Obtaining the member 'types' of a type (line 435)
            types_13446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 44), self_13445, 'types')
            # Processing the call keyword arguments (line 434)
            kwargs_13447 = {}
            str_13442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 43), 'str', "The member '{0}' cannot be deleted from none of the possible types ('{1}')")
            # Obtaining the member 'format' of a type (line 434)
            format_13443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 43), str_13442, 'format')
            # Calling format(args, kwargs) (line 434)
            format_call_result_13448 = invoke(stypy.reporting.localization.Localization(__file__, 434, 43), format_13443, *[member_13444, types_13446], **kwargs_13447)
            
            # Processing the call keyword arguments (line 434)
            kwargs_13449 = {}
            # Getting the type of 'TypeError' (line 434)
            TypeError_13440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 434)
            TypeError_call_result_13450 = invoke(stypy.reporting.localization.Localization(__file__, 434, 19), TypeError_13440, *[localization_13441, format_call_result_13448], **kwargs_13449)
            
            # Assigning a type to the variable 'stypy_return_type' (line 434)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'stypy_return_type', TypeError_call_result_13450)
            # SSA branch for the else part of an if statement (line 433)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 440)
            errors_13451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 25), 'errors')
            # Assigning a type to the variable 'errors_13451' (line 440)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'errors_13451', errors_13451)
            # Testing if the for loop is going to be iterated (line 440)
            # Testing the type of a for loop iterable (line 440)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 440, 12), errors_13451)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 440, 12), errors_13451):
                # Getting the type of the for loop variable (line 440)
                for_loop_var_13452 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 440, 12), errors_13451)
                # Assigning a type to the variable 'error' (line 440)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'error', for_loop_var_13452)
                # SSA begins for a for statement (line 440)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 441)
                # Processing the call keyword arguments (line 441)
                kwargs_13455 = {}
                # Getting the type of 'error' (line 441)
                error_13453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 441)
                turn_to_warning_13454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 16), error_13453, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 441)
                turn_to_warning_call_result_13456 = invoke(stypy.reporting.localization.Localization(__file__, 441, 16), turn_to_warning_13454, *[], **kwargs_13455)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 433)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'None' (line 443)
        None_13457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'stypy_return_type', None_13457)
        
        # ################# End of 'delete_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'delete_member' in the type store
        # Getting the type of 'stypy_return_type' (line 417)
        stypy_return_type_13458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13458)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'delete_member'
        return stypy_return_type_13458


    @norecursion
    def supports_structural_reflection(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'supports_structural_reflection'
        module_type_store = module_type_store.open_function_context('supports_structural_reflection', 445, 4, False)
        # Assigning a type to the variable 'self' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.supports_structural_reflection.__dict__.__setitem__('stypy_localization', localization)
        UnionType.supports_structural_reflection.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.supports_structural_reflection.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.supports_structural_reflection.__dict__.__setitem__('stypy_function_name', 'UnionType.supports_structural_reflection')
        UnionType.supports_structural_reflection.__dict__.__setitem__('stypy_param_names_list', [])
        UnionType.supports_structural_reflection.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.supports_structural_reflection.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.supports_structural_reflection.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.supports_structural_reflection.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.supports_structural_reflection.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.supports_structural_reflection.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.supports_structural_reflection', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'supports_structural_reflection', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'supports_structural_reflection(...)' code ##################

        str_13459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, (-1)), 'str', '\n        Determines if at least one of the stored types supports structural reflection.\n        ')
        
        # Assigning a Name to a Name (line 449):
        # Getting the type of 'False' (line 449)
        False_13460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 19), 'False')
        # Assigning a type to the variable 'supports' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'supports', False_13460)
        
        # Getting the type of 'self' (line 451)
        self_13461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 21), 'self')
        # Obtaining the member 'types' of a type (line 451)
        types_13462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 21), self_13461, 'types')
        # Assigning a type to the variable 'types_13462' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'types_13462', types_13462)
        # Testing if the for loop is going to be iterated (line 451)
        # Testing the type of a for loop iterable (line 451)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 451, 8), types_13462)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 451, 8), types_13462):
            # Getting the type of the for loop variable (line 451)
            for_loop_var_13463 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 451, 8), types_13462)
            # Assigning a type to the variable 'type_' (line 451)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'type_', for_loop_var_13463)
            # SSA begins for a for statement (line 451)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a BoolOp to a Name (line 452):
            
            # Evaluating a boolean operation
            # Getting the type of 'supports' (line 452)
            supports_13464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 23), 'supports')
            
            # Call to supports_structural_reflection(...): (line 452)
            # Processing the call keyword arguments (line 452)
            kwargs_13467 = {}
            # Getting the type of 'type_' (line 452)
            type__13465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 35), 'type_', False)
            # Obtaining the member 'supports_structural_reflection' of a type (line 452)
            supports_structural_reflection_13466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 35), type__13465, 'supports_structural_reflection')
            # Calling supports_structural_reflection(args, kwargs) (line 452)
            supports_structural_reflection_call_result_13468 = invoke(stypy.reporting.localization.Localization(__file__, 452, 35), supports_structural_reflection_13466, *[], **kwargs_13467)
            
            # Applying the binary operator 'or' (line 452)
            result_or_keyword_13469 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 23), 'or', supports_13464, supports_structural_reflection_call_result_13468)
            
            # Assigning a type to the variable 'supports' (line 452)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 12), 'supports', result_or_keyword_13469)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'supports' (line 454)
        supports_13470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 15), 'supports')
        # Assigning a type to the variable 'stypy_return_type' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'stypy_return_type', supports_13470)
        
        # ################# End of 'supports_structural_reflection(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'supports_structural_reflection' in the type store
        # Getting the type of 'stypy_return_type' (line 445)
        stypy_return_type_13471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13471)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'supports_structural_reflection'
        return stypy_return_type_13471


    @norecursion
    def change_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'change_type'
        module_type_store = module_type_store.open_function_context('change_type', 456, 4, False)
        # Assigning a type to the variable 'self' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.change_type.__dict__.__setitem__('stypy_localization', localization)
        UnionType.change_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.change_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.change_type.__dict__.__setitem__('stypy_function_name', 'UnionType.change_type')
        UnionType.change_type.__dict__.__setitem__('stypy_param_names_list', ['localization', 'new_type'])
        UnionType.change_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.change_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.change_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.change_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.change_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.change_type.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.change_type', ['localization', 'new_type'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'change_type', localization, ['localization', 'new_type'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'change_type(...)' code ##################

        str_13472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, (-1)), 'str', '\n        For all the types stored in the union type, change the base type to new_type, returning None or a TypeError\n        if no type stored in the UnionType supports a type change.\n        :param localization: Caller information\n        :param new_type: Type to change to\n        :return None or TypeError\n        ')
        
        # Assigning a List to a Name (line 464):
        
        # Obtaining an instance of the builtin type 'list' (line 464)
        list_13473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 464)
        
        # Assigning a type to the variable 'errors' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'errors', list_13473)
        
        # Getting the type of 'self' (line 466)
        self_13474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 21), 'self')
        # Obtaining the member 'types' of a type (line 466)
        types_13475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 21), self_13474, 'types')
        # Assigning a type to the variable 'types_13475' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'types_13475', types_13475)
        # Testing if the for loop is going to be iterated (line 466)
        # Testing the type of a for loop iterable (line 466)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 466, 8), types_13475)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 466, 8), types_13475):
            # Getting the type of the for loop variable (line 466)
            for_loop_var_13476 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 466, 8), types_13475)
            # Assigning a type to the variable 'type_' (line 466)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'type_', for_loop_var_13476)
            # SSA begins for a for statement (line 466)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 467):
            
            # Call to change_type(...): (line 467)
            # Processing the call arguments (line 467)
            # Getting the type of 'localization' (line 467)
            localization_13479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 37), 'localization', False)
            # Getting the type of 'new_type' (line 467)
            new_type_13480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 51), 'new_type', False)
            # Processing the call keyword arguments (line 467)
            kwargs_13481 = {}
            # Getting the type of 'type_' (line 467)
            type__13477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 19), 'type_', False)
            # Obtaining the member 'change_type' of a type (line 467)
            change_type_13478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 19), type__13477, 'change_type')
            # Calling change_type(args, kwargs) (line 467)
            change_type_call_result_13482 = invoke(stypy.reporting.localization.Localization(__file__, 467, 19), change_type_13478, *[localization_13479, new_type_13480], **kwargs_13481)
            
            # Assigning a type to the variable 'temp' (line 467)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'temp', change_type_call_result_13482)
            
            # Type idiom detected: calculating its left and rigth part (line 468)
            # Getting the type of 'temp' (line 468)
            temp_13483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 12), 'temp')
            # Getting the type of 'None' (line 468)
            None_13484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 27), 'None')
            
            (may_be_13485, more_types_in_union_13486) = may_not_be_none(temp_13483, None_13484)

            if may_be_13485:

                if more_types_in_union_13486:
                    # Runtime conditional SSA (line 468)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to append(...): (line 469)
                # Processing the call arguments (line 469)
                # Getting the type of 'temp' (line 469)
                temp_13489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 30), 'temp', False)
                # Processing the call keyword arguments (line 469)
                kwargs_13490 = {}
                # Getting the type of 'errors' (line 469)
                errors_13487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 16), 'errors', False)
                # Obtaining the member 'append' of a type (line 469)
                append_13488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 16), errors_13487, 'append')
                # Calling append(args, kwargs) (line 469)
                append_call_result_13491 = invoke(stypy.reporting.localization.Localization(__file__, 469, 16), append_13488, *[temp_13489], **kwargs_13490)
                

                if more_types_in_union_13486:
                    # SSA join for if statement (line 468)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 472)
        # Processing the call arguments (line 472)
        # Getting the type of 'errors' (line 472)
        errors_13493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 15), 'errors', False)
        # Processing the call keyword arguments (line 472)
        kwargs_13494 = {}
        # Getting the type of 'len' (line 472)
        len_13492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 11), 'len', False)
        # Calling len(args, kwargs) (line 472)
        len_call_result_13495 = invoke(stypy.reporting.localization.Localization(__file__, 472, 11), len_13492, *[errors_13493], **kwargs_13494)
        
        
        # Call to len(...): (line 472)
        # Processing the call arguments (line 472)
        # Getting the type of 'self' (line 472)
        self_13497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 472)
        types_13498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 30), self_13497, 'types')
        # Processing the call keyword arguments (line 472)
        kwargs_13499 = {}
        # Getting the type of 'len' (line 472)
        len_13496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 26), 'len', False)
        # Calling len(args, kwargs) (line 472)
        len_call_result_13500 = invoke(stypy.reporting.localization.Localization(__file__, 472, 26), len_13496, *[types_13498], **kwargs_13499)
        
        # Applying the binary operator '==' (line 472)
        result_eq_13501 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 11), '==', len_call_result_13495, len_call_result_13500)
        
        # Testing if the type of an if condition is none (line 472)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 472, 8), result_eq_13501):
            
            # Getting the type of 'errors' (line 479)
            errors_13514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 25), 'errors')
            # Assigning a type to the variable 'errors_13514' (line 479)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'errors_13514', errors_13514)
            # Testing if the for loop is going to be iterated (line 479)
            # Testing the type of a for loop iterable (line 479)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 479, 12), errors_13514)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 479, 12), errors_13514):
                # Getting the type of the for loop variable (line 479)
                for_loop_var_13515 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 479, 12), errors_13514)
                # Assigning a type to the variable 'error' (line 479)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'error', for_loop_var_13515)
                # SSA begins for a for statement (line 479)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 480)
                # Processing the call keyword arguments (line 480)
                kwargs_13518 = {}
                # Getting the type of 'error' (line 480)
                error_13516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 480)
                turn_to_warning_13517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 16), error_13516, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 480)
                turn_to_warning_call_result_13519 = invoke(stypy.reporting.localization.Localization(__file__, 480, 16), turn_to_warning_13517, *[], **kwargs_13518)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 472)
            if_condition_13502 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 472, 8), result_eq_13501)
            # Assigning a type to the variable 'if_condition_13502' (line 472)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'if_condition_13502', if_condition_13502)
            # SSA begins for if statement (line 472)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 473)
            # Processing the call arguments (line 473)
            # Getting the type of 'localization' (line 473)
            localization_13504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 29), 'localization', False)
            
            # Call to format(...): (line 473)
            # Processing the call arguments (line 473)
            # Getting the type of 'new_type' (line 474)
            new_type_13507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 36), 'new_type', False)
            # Getting the type of 'self' (line 474)
            self_13508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 46), 'self', False)
            # Obtaining the member 'types' of a type (line 474)
            types_13509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 46), self_13508, 'types')
            # Processing the call keyword arguments (line 473)
            kwargs_13510 = {}
            str_13505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 43), 'str', "None of the possible types ('{1}') can be assigned a new type '{0}'")
            # Obtaining the member 'format' of a type (line 473)
            format_13506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 43), str_13505, 'format')
            # Calling format(args, kwargs) (line 473)
            format_call_result_13511 = invoke(stypy.reporting.localization.Localization(__file__, 473, 43), format_13506, *[new_type_13507, types_13509], **kwargs_13510)
            
            # Processing the call keyword arguments (line 473)
            kwargs_13512 = {}
            # Getting the type of 'TypeError' (line 473)
            TypeError_13503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 473)
            TypeError_call_result_13513 = invoke(stypy.reporting.localization.Localization(__file__, 473, 19), TypeError_13503, *[localization_13504, format_call_result_13511], **kwargs_13512)
            
            # Assigning a type to the variable 'stypy_return_type' (line 473)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'stypy_return_type', TypeError_call_result_13513)
            # SSA branch for the else part of an if statement (line 472)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 479)
            errors_13514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 25), 'errors')
            # Assigning a type to the variable 'errors_13514' (line 479)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'errors_13514', errors_13514)
            # Testing if the for loop is going to be iterated (line 479)
            # Testing the type of a for loop iterable (line 479)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 479, 12), errors_13514)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 479, 12), errors_13514):
                # Getting the type of the for loop variable (line 479)
                for_loop_var_13515 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 479, 12), errors_13514)
                # Assigning a type to the variable 'error' (line 479)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'error', for_loop_var_13515)
                # SSA begins for a for statement (line 479)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 480)
                # Processing the call keyword arguments (line 480)
                kwargs_13518 = {}
                # Getting the type of 'error' (line 480)
                error_13516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 480)
                turn_to_warning_13517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 16), error_13516, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 480)
                turn_to_warning_call_result_13519 = invoke(stypy.reporting.localization.Localization(__file__, 480, 16), turn_to_warning_13517, *[], **kwargs_13518)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 472)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'None' (line 482)
        None_13520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'stypy_return_type', None_13520)
        
        # ################# End of 'change_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'change_type' in the type store
        # Getting the type of 'stypy_return_type' (line 456)
        stypy_return_type_13521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13521)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'change_type'
        return stypy_return_type_13521


    @norecursion
    def change_base_types(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'change_base_types'
        module_type_store = module_type_store.open_function_context('change_base_types', 484, 4, False)
        # Assigning a type to the variable 'self' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.change_base_types.__dict__.__setitem__('stypy_localization', localization)
        UnionType.change_base_types.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.change_base_types.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.change_base_types.__dict__.__setitem__('stypy_function_name', 'UnionType.change_base_types')
        UnionType.change_base_types.__dict__.__setitem__('stypy_param_names_list', ['localization', 'new_types'])
        UnionType.change_base_types.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.change_base_types.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.change_base_types.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.change_base_types.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.change_base_types.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.change_base_types.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.change_base_types', ['localization', 'new_types'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'change_base_types', localization, ['localization', 'new_types'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'change_base_types(...)' code ##################

        str_13522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, (-1)), 'str', '\n        For all the types stored in the union type, change the base types to the ones contained in the list new_types,\n        returning None or a TypeError if no type stored in the UnionType supports a supertype change.\n        :param localization: Caller information\n        :param new_types: Types to change its base type to\n        :return None or TypeError\n        ')
        
        # Assigning a List to a Name (line 492):
        
        # Obtaining an instance of the builtin type 'list' (line 492)
        list_13523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 492)
        
        # Assigning a type to the variable 'errors' (line 492)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'errors', list_13523)
        
        # Getting the type of 'self' (line 494)
        self_13524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 21), 'self')
        # Obtaining the member 'types' of a type (line 494)
        types_13525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 21), self_13524, 'types')
        # Assigning a type to the variable 'types_13525' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'types_13525', types_13525)
        # Testing if the for loop is going to be iterated (line 494)
        # Testing the type of a for loop iterable (line 494)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 494, 8), types_13525)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 494, 8), types_13525):
            # Getting the type of the for loop variable (line 494)
            for_loop_var_13526 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 494, 8), types_13525)
            # Assigning a type to the variable 'type_' (line 494)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'type_', for_loop_var_13526)
            # SSA begins for a for statement (line 494)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 495):
            
            # Call to change_base_types(...): (line 495)
            # Processing the call arguments (line 495)
            # Getting the type of 'localization' (line 495)
            localization_13529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 43), 'localization', False)
            # Getting the type of 'new_types' (line 495)
            new_types_13530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 57), 'new_types', False)
            # Processing the call keyword arguments (line 495)
            kwargs_13531 = {}
            # Getting the type of 'type_' (line 495)
            type__13527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 19), 'type_', False)
            # Obtaining the member 'change_base_types' of a type (line 495)
            change_base_types_13528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 19), type__13527, 'change_base_types')
            # Calling change_base_types(args, kwargs) (line 495)
            change_base_types_call_result_13532 = invoke(stypy.reporting.localization.Localization(__file__, 495, 19), change_base_types_13528, *[localization_13529, new_types_13530], **kwargs_13531)
            
            # Assigning a type to the variable 'temp' (line 495)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'temp', change_base_types_call_result_13532)
            
            # Type idiom detected: calculating its left and rigth part (line 496)
            # Getting the type of 'temp' (line 496)
            temp_13533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'temp')
            # Getting the type of 'None' (line 496)
            None_13534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 27), 'None')
            
            (may_be_13535, more_types_in_union_13536) = may_not_be_none(temp_13533, None_13534)

            if may_be_13535:

                if more_types_in_union_13536:
                    # Runtime conditional SSA (line 496)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to append(...): (line 497)
                # Processing the call arguments (line 497)
                # Getting the type of 'temp' (line 497)
                temp_13539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 30), 'temp', False)
                # Processing the call keyword arguments (line 497)
                kwargs_13540 = {}
                # Getting the type of 'errors' (line 497)
                errors_13537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 16), 'errors', False)
                # Obtaining the member 'append' of a type (line 497)
                append_13538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 16), errors_13537, 'append')
                # Calling append(args, kwargs) (line 497)
                append_call_result_13541 = invoke(stypy.reporting.localization.Localization(__file__, 497, 16), append_13538, *[temp_13539], **kwargs_13540)
                

                if more_types_in_union_13536:
                    # SSA join for if statement (line 496)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 500)
        # Processing the call arguments (line 500)
        # Getting the type of 'errors' (line 500)
        errors_13543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 15), 'errors', False)
        # Processing the call keyword arguments (line 500)
        kwargs_13544 = {}
        # Getting the type of 'len' (line 500)
        len_13542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 11), 'len', False)
        # Calling len(args, kwargs) (line 500)
        len_call_result_13545 = invoke(stypy.reporting.localization.Localization(__file__, 500, 11), len_13542, *[errors_13543], **kwargs_13544)
        
        
        # Call to len(...): (line 500)
        # Processing the call arguments (line 500)
        # Getting the type of 'self' (line 500)
        self_13547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 500)
        types_13548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 30), self_13547, 'types')
        # Processing the call keyword arguments (line 500)
        kwargs_13549 = {}
        # Getting the type of 'len' (line 500)
        len_13546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 26), 'len', False)
        # Calling len(args, kwargs) (line 500)
        len_call_result_13550 = invoke(stypy.reporting.localization.Localization(__file__, 500, 26), len_13546, *[types_13548], **kwargs_13549)
        
        # Applying the binary operator '==' (line 500)
        result_eq_13551 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 11), '==', len_call_result_13545, len_call_result_13550)
        
        # Testing if the type of an if condition is none (line 500)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 500, 8), result_eq_13551):
            
            # Getting the type of 'errors' (line 507)
            errors_13564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 25), 'errors')
            # Assigning a type to the variable 'errors_13564' (line 507)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 12), 'errors_13564', errors_13564)
            # Testing if the for loop is going to be iterated (line 507)
            # Testing the type of a for loop iterable (line 507)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 507, 12), errors_13564)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 507, 12), errors_13564):
                # Getting the type of the for loop variable (line 507)
                for_loop_var_13565 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 507, 12), errors_13564)
                # Assigning a type to the variable 'error' (line 507)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 12), 'error', for_loop_var_13565)
                # SSA begins for a for statement (line 507)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 508)
                # Processing the call keyword arguments (line 508)
                kwargs_13568 = {}
                # Getting the type of 'error' (line 508)
                error_13566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 508)
                turn_to_warning_13567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 16), error_13566, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 508)
                turn_to_warning_call_result_13569 = invoke(stypy.reporting.localization.Localization(__file__, 508, 16), turn_to_warning_13567, *[], **kwargs_13568)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 500)
            if_condition_13552 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 500, 8), result_eq_13551)
            # Assigning a type to the variable 'if_condition_13552' (line 500)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'if_condition_13552', if_condition_13552)
            # SSA begins for if statement (line 500)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 501)
            # Processing the call arguments (line 501)
            # Getting the type of 'localization' (line 501)
            localization_13554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 29), 'localization', False)
            
            # Call to format(...): (line 501)
            # Processing the call arguments (line 501)
            # Getting the type of 'new_types' (line 502)
            new_types_13557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 36), 'new_types', False)
            # Getting the type of 'self' (line 502)
            self_13558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 47), 'self', False)
            # Obtaining the member 'types' of a type (line 502)
            types_13559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 47), self_13558, 'types')
            # Processing the call keyword arguments (line 501)
            kwargs_13560 = {}
            str_13555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 43), 'str', "None of the possible types ('{1}') can be assigned new base types '{0}'")
            # Obtaining the member 'format' of a type (line 501)
            format_13556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 43), str_13555, 'format')
            # Calling format(args, kwargs) (line 501)
            format_call_result_13561 = invoke(stypy.reporting.localization.Localization(__file__, 501, 43), format_13556, *[new_types_13557, types_13559], **kwargs_13560)
            
            # Processing the call keyword arguments (line 501)
            kwargs_13562 = {}
            # Getting the type of 'TypeError' (line 501)
            TypeError_13553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 501)
            TypeError_call_result_13563 = invoke(stypy.reporting.localization.Localization(__file__, 501, 19), TypeError_13553, *[localization_13554, format_call_result_13561], **kwargs_13562)
            
            # Assigning a type to the variable 'stypy_return_type' (line 501)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 12), 'stypy_return_type', TypeError_call_result_13563)
            # SSA branch for the else part of an if statement (line 500)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 507)
            errors_13564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 25), 'errors')
            # Assigning a type to the variable 'errors_13564' (line 507)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 12), 'errors_13564', errors_13564)
            # Testing if the for loop is going to be iterated (line 507)
            # Testing the type of a for loop iterable (line 507)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 507, 12), errors_13564)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 507, 12), errors_13564):
                # Getting the type of the for loop variable (line 507)
                for_loop_var_13565 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 507, 12), errors_13564)
                # Assigning a type to the variable 'error' (line 507)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 12), 'error', for_loop_var_13565)
                # SSA begins for a for statement (line 507)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 508)
                # Processing the call keyword arguments (line 508)
                kwargs_13568 = {}
                # Getting the type of 'error' (line 508)
                error_13566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 508)
                turn_to_warning_13567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 16), error_13566, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 508)
                turn_to_warning_call_result_13569 = invoke(stypy.reporting.localization.Localization(__file__, 508, 16), turn_to_warning_13567, *[], **kwargs_13568)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 500)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'None' (line 510)
        None_13570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'stypy_return_type', None_13570)
        
        # ################# End of 'change_base_types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'change_base_types' in the type store
        # Getting the type of 'stypy_return_type' (line 484)
        stypy_return_type_13571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13571)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'change_base_types'
        return stypy_return_type_13571


    @norecursion
    def add_base_types(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_base_types'
        module_type_store = module_type_store.open_function_context('add_base_types', 512, 4, False)
        # Assigning a type to the variable 'self' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.add_base_types.__dict__.__setitem__('stypy_localization', localization)
        UnionType.add_base_types.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.add_base_types.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.add_base_types.__dict__.__setitem__('stypy_function_name', 'UnionType.add_base_types')
        UnionType.add_base_types.__dict__.__setitem__('stypy_param_names_list', ['localization', 'new_types'])
        UnionType.add_base_types.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.add_base_types.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.add_base_types.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.add_base_types.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.add_base_types.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.add_base_types.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.add_base_types', ['localization', 'new_types'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_base_types', localization, ['localization', 'new_types'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_base_types(...)' code ##################

        str_13572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, (-1)), 'str', '\n        For all the types stored in the union type, add to the base types the ones contained in the list new_types,\n        returning None or a TypeError if no type stored in the UnionType supports a supertype change.\n        :param localization: Caller information\n        :param new_types: Types to change its base type to\n        :return None or TypeError\n        ')
        
        # Assigning a List to a Name (line 520):
        
        # Obtaining an instance of the builtin type 'list' (line 520)
        list_13573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 520)
        
        # Assigning a type to the variable 'errors' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'errors', list_13573)
        
        # Getting the type of 'self' (line 522)
        self_13574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 21), 'self')
        # Obtaining the member 'types' of a type (line 522)
        types_13575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 21), self_13574, 'types')
        # Assigning a type to the variable 'types_13575' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'types_13575', types_13575)
        # Testing if the for loop is going to be iterated (line 522)
        # Testing the type of a for loop iterable (line 522)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 522, 8), types_13575)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 522, 8), types_13575):
            # Getting the type of the for loop variable (line 522)
            for_loop_var_13576 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 522, 8), types_13575)
            # Assigning a type to the variable 'type_' (line 522)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'type_', for_loop_var_13576)
            # SSA begins for a for statement (line 522)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 523):
            
            # Call to change_base_types(...): (line 523)
            # Processing the call arguments (line 523)
            # Getting the type of 'localization' (line 523)
            localization_13579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 43), 'localization', False)
            # Getting the type of 'new_types' (line 523)
            new_types_13580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 57), 'new_types', False)
            # Processing the call keyword arguments (line 523)
            kwargs_13581 = {}
            # Getting the type of 'type_' (line 523)
            type__13577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 19), 'type_', False)
            # Obtaining the member 'change_base_types' of a type (line 523)
            change_base_types_13578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 19), type__13577, 'change_base_types')
            # Calling change_base_types(args, kwargs) (line 523)
            change_base_types_call_result_13582 = invoke(stypy.reporting.localization.Localization(__file__, 523, 19), change_base_types_13578, *[localization_13579, new_types_13580], **kwargs_13581)
            
            # Assigning a type to the variable 'temp' (line 523)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 12), 'temp', change_base_types_call_result_13582)
            
            # Type idiom detected: calculating its left and rigth part (line 524)
            # Getting the type of 'temp' (line 524)
            temp_13583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 12), 'temp')
            # Getting the type of 'None' (line 524)
            None_13584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 27), 'None')
            
            (may_be_13585, more_types_in_union_13586) = may_not_be_none(temp_13583, None_13584)

            if may_be_13585:

                if more_types_in_union_13586:
                    # Runtime conditional SSA (line 524)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to append(...): (line 525)
                # Processing the call arguments (line 525)
                # Getting the type of 'temp' (line 525)
                temp_13589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 30), 'temp', False)
                # Processing the call keyword arguments (line 525)
                kwargs_13590 = {}
                # Getting the type of 'errors' (line 525)
                errors_13587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 16), 'errors', False)
                # Obtaining the member 'append' of a type (line 525)
                append_13588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 16), errors_13587, 'append')
                # Calling append(args, kwargs) (line 525)
                append_call_result_13591 = invoke(stypy.reporting.localization.Localization(__file__, 525, 16), append_13588, *[temp_13589], **kwargs_13590)
                

                if more_types_in_union_13586:
                    # SSA join for if statement (line 524)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 528)
        # Processing the call arguments (line 528)
        # Getting the type of 'errors' (line 528)
        errors_13593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 15), 'errors', False)
        # Processing the call keyword arguments (line 528)
        kwargs_13594 = {}
        # Getting the type of 'len' (line 528)
        len_13592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 11), 'len', False)
        # Calling len(args, kwargs) (line 528)
        len_call_result_13595 = invoke(stypy.reporting.localization.Localization(__file__, 528, 11), len_13592, *[errors_13593], **kwargs_13594)
        
        
        # Call to len(...): (line 528)
        # Processing the call arguments (line 528)
        # Getting the type of 'self' (line 528)
        self_13597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 528)
        types_13598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 30), self_13597, 'types')
        # Processing the call keyword arguments (line 528)
        kwargs_13599 = {}
        # Getting the type of 'len' (line 528)
        len_13596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 26), 'len', False)
        # Calling len(args, kwargs) (line 528)
        len_call_result_13600 = invoke(stypy.reporting.localization.Localization(__file__, 528, 26), len_13596, *[types_13598], **kwargs_13599)
        
        # Applying the binary operator '==' (line 528)
        result_eq_13601 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 11), '==', len_call_result_13595, len_call_result_13600)
        
        # Testing if the type of an if condition is none (line 528)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 528, 8), result_eq_13601):
            
            # Getting the type of 'errors' (line 535)
            errors_13613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 25), 'errors')
            # Assigning a type to the variable 'errors_13613' (line 535)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'errors_13613', errors_13613)
            # Testing if the for loop is going to be iterated (line 535)
            # Testing the type of a for loop iterable (line 535)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 535, 12), errors_13613)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 535, 12), errors_13613):
                # Getting the type of the for loop variable (line 535)
                for_loop_var_13614 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 535, 12), errors_13613)
                # Assigning a type to the variable 'error' (line 535)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'error', for_loop_var_13614)
                # SSA begins for a for statement (line 535)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 536)
                # Processing the call keyword arguments (line 536)
                kwargs_13617 = {}
                # Getting the type of 'error' (line 536)
                error_13615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 536)
                turn_to_warning_13616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 16), error_13615, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 536)
                turn_to_warning_call_result_13618 = invoke(stypy.reporting.localization.Localization(__file__, 536, 16), turn_to_warning_13616, *[], **kwargs_13617)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 528)
            if_condition_13602 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 528, 8), result_eq_13601)
            # Assigning a type to the variable 'if_condition_13602' (line 528)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'if_condition_13602', if_condition_13602)
            # SSA begins for if statement (line 528)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 529)
            # Processing the call arguments (line 529)
            # Getting the type of 'localization' (line 529)
            localization_13604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 29), 'localization', False)
            
            # Call to format(...): (line 529)
            # Processing the call arguments (line 529)
            # Getting the type of 'self' (line 530)
            self_13607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 36), 'self', False)
            # Obtaining the member 'types' of a type (line 530)
            types_13608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 36), self_13607, 'types')
            # Processing the call keyword arguments (line 529)
            kwargs_13609 = {}
            str_13605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 43), 'str', "The base types of all the possible types ('{0}') cannot be modified")
            # Obtaining the member 'format' of a type (line 529)
            format_13606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 43), str_13605, 'format')
            # Calling format(args, kwargs) (line 529)
            format_call_result_13610 = invoke(stypy.reporting.localization.Localization(__file__, 529, 43), format_13606, *[types_13608], **kwargs_13609)
            
            # Processing the call keyword arguments (line 529)
            kwargs_13611 = {}
            # Getting the type of 'TypeError' (line 529)
            TypeError_13603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 529)
            TypeError_call_result_13612 = invoke(stypy.reporting.localization.Localization(__file__, 529, 19), TypeError_13603, *[localization_13604, format_call_result_13610], **kwargs_13611)
            
            # Assigning a type to the variable 'stypy_return_type' (line 529)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'stypy_return_type', TypeError_call_result_13612)
            # SSA branch for the else part of an if statement (line 528)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 535)
            errors_13613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 25), 'errors')
            # Assigning a type to the variable 'errors_13613' (line 535)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'errors_13613', errors_13613)
            # Testing if the for loop is going to be iterated (line 535)
            # Testing the type of a for loop iterable (line 535)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 535, 12), errors_13613)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 535, 12), errors_13613):
                # Getting the type of the for loop variable (line 535)
                for_loop_var_13614 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 535, 12), errors_13613)
                # Assigning a type to the variable 'error' (line 535)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'error', for_loop_var_13614)
                # SSA begins for a for statement (line 535)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 536)
                # Processing the call keyword arguments (line 536)
                kwargs_13617 = {}
                # Getting the type of 'error' (line 536)
                error_13615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 536)
                turn_to_warning_13616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 16), error_13615, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 536)
                turn_to_warning_call_result_13618 = invoke(stypy.reporting.localization.Localization(__file__, 536, 16), turn_to_warning_13616, *[], **kwargs_13617)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 528)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'None' (line 538)
        None_13619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 538)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'stypy_return_type', None_13619)
        
        # ################# End of 'add_base_types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_base_types' in the type store
        # Getting the type of 'stypy_return_type' (line 512)
        stypy_return_type_13620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13620)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_base_types'
        return stypy_return_type_13620


    @norecursion
    def clone(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'clone'
        module_type_store = module_type_store.open_function_context('clone', 542, 4, False)
        # Assigning a type to the variable 'self' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.clone.__dict__.__setitem__('stypy_localization', localization)
        UnionType.clone.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.clone.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.clone.__dict__.__setitem__('stypy_function_name', 'UnionType.clone')
        UnionType.clone.__dict__.__setitem__('stypy_param_names_list', [])
        UnionType.clone.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.clone.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.clone.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.clone.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.clone.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.clone.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.clone', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'clone', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'clone(...)' code ##################

        str_13621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, (-1)), 'str', '\n        Clone the whole UnionType and its contained types\n        ')
        
        # Assigning a Call to a Name (line 546):
        
        # Call to clone(...): (line 546)
        # Processing the call keyword arguments (line 546)
        kwargs_13628 = {}
        
        # Obtaining the type of the subscript
        int_13622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 34), 'int')
        # Getting the type of 'self' (line 546)
        self_13623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 23), 'self', False)
        # Obtaining the member 'types' of a type (line 546)
        types_13624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 23), self_13623, 'types')
        # Obtaining the member '__getitem__' of a type (line 546)
        getitem___13625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 23), types_13624, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 546)
        subscript_call_result_13626 = invoke(stypy.reporting.localization.Localization(__file__, 546, 23), getitem___13625, int_13622)
        
        # Obtaining the member 'clone' of a type (line 546)
        clone_13627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 23), subscript_call_result_13626, 'clone')
        # Calling clone(args, kwargs) (line 546)
        clone_call_result_13629 = invoke(stypy.reporting.localization.Localization(__file__, 546, 23), clone_13627, *[], **kwargs_13628)
        
        # Assigning a type to the variable 'result_union' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'result_union', clone_call_result_13629)
        
        
        # Call to range(...): (line 547)
        # Processing the call arguments (line 547)
        int_13631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 23), 'int')
        
        # Call to len(...): (line 547)
        # Processing the call arguments (line 547)
        # Getting the type of 'self' (line 547)
        self_13633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 547)
        types_13634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 30), self_13633, 'types')
        # Processing the call keyword arguments (line 547)
        kwargs_13635 = {}
        # Getting the type of 'len' (line 547)
        len_13632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 26), 'len', False)
        # Calling len(args, kwargs) (line 547)
        len_call_result_13636 = invoke(stypy.reporting.localization.Localization(__file__, 547, 26), len_13632, *[types_13634], **kwargs_13635)
        
        # Processing the call keyword arguments (line 547)
        kwargs_13637 = {}
        # Getting the type of 'range' (line 547)
        range_13630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 17), 'range', False)
        # Calling range(args, kwargs) (line 547)
        range_call_result_13638 = invoke(stypy.reporting.localization.Localization(__file__, 547, 17), range_13630, *[int_13631, len_call_result_13636], **kwargs_13637)
        
        # Assigning a type to the variable 'range_call_result_13638' (line 547)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'range_call_result_13638', range_call_result_13638)
        # Testing if the for loop is going to be iterated (line 547)
        # Testing the type of a for loop iterable (line 547)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 547, 8), range_call_result_13638)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 547, 8), range_call_result_13638):
            # Getting the type of the for loop variable (line 547)
            for_loop_var_13639 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 547, 8), range_call_result_13638)
            # Assigning a type to the variable 'i' (line 547)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'i', for_loop_var_13639)
            # SSA begins for a for statement (line 547)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to isinstance(...): (line 548)
            # Processing the call arguments (line 548)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 548)
            i_13641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 37), 'i', False)
            # Getting the type of 'self' (line 548)
            self_13642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 26), 'self', False)
            # Obtaining the member 'types' of a type (line 548)
            types_13643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 26), self_13642, 'types')
            # Obtaining the member '__getitem__' of a type (line 548)
            getitem___13644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 26), types_13643, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 548)
            subscript_call_result_13645 = invoke(stypy.reporting.localization.Localization(__file__, 548, 26), getitem___13644, i_13641)
            
            # Getting the type of 'Type' (line 548)
            Type_13646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 41), 'Type', False)
            # Processing the call keyword arguments (line 548)
            kwargs_13647 = {}
            # Getting the type of 'isinstance' (line 548)
            isinstance_13640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 548)
            isinstance_call_result_13648 = invoke(stypy.reporting.localization.Localization(__file__, 548, 15), isinstance_13640, *[subscript_call_result_13645, Type_13646], **kwargs_13647)
            
            # Testing if the type of an if condition is none (line 548)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 548, 12), isinstance_call_result_13648):
                
                # Assigning a Call to a Name (line 551):
                
                # Call to add(...): (line 551)
                # Processing the call arguments (line 551)
                # Getting the type of 'result_union' (line 551)
                result_union_13665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 45), 'result_union', False)
                
                # Call to deepcopy(...): (line 551)
                # Processing the call arguments (line 551)
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 551)
                i_13668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 84), 'i', False)
                # Getting the type of 'self' (line 551)
                self_13669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 73), 'self', False)
                # Obtaining the member 'types' of a type (line 551)
                types_13670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 73), self_13669, 'types')
                # Obtaining the member '__getitem__' of a type (line 551)
                getitem___13671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 73), types_13670, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 551)
                subscript_call_result_13672 = invoke(stypy.reporting.localization.Localization(__file__, 551, 73), getitem___13671, i_13668)
                
                # Processing the call keyword arguments (line 551)
                kwargs_13673 = {}
                # Getting the type of 'copy' (line 551)
                copy_13666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 59), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 551)
                deepcopy_13667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 59), copy_13666, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 551)
                deepcopy_call_result_13674 = invoke(stypy.reporting.localization.Localization(__file__, 551, 59), deepcopy_13667, *[subscript_call_result_13672], **kwargs_13673)
                
                # Processing the call keyword arguments (line 551)
                kwargs_13675 = {}
                # Getting the type of 'UnionType' (line 551)
                UnionType_13663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 31), 'UnionType', False)
                # Obtaining the member 'add' of a type (line 551)
                add_13664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 31), UnionType_13663, 'add')
                # Calling add(args, kwargs) (line 551)
                add_call_result_13676 = invoke(stypy.reporting.localization.Localization(__file__, 551, 31), add_13664, *[result_union_13665, deepcopy_call_result_13674], **kwargs_13675)
                
                # Assigning a type to the variable 'result_union' (line 551)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 16), 'result_union', add_call_result_13676)
            else:
                
                # Testing the type of an if condition (line 548)
                if_condition_13649 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 548, 12), isinstance_call_result_13648)
                # Assigning a type to the variable 'if_condition_13649' (line 548)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 12), 'if_condition_13649', if_condition_13649)
                # SSA begins for if statement (line 548)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 549):
                
                # Call to add(...): (line 549)
                # Processing the call arguments (line 549)
                # Getting the type of 'result_union' (line 549)
                result_union_13652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 45), 'result_union', False)
                
                # Call to clone(...): (line 549)
                # Processing the call keyword arguments (line 549)
                kwargs_13659 = {}
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 549)
                i_13653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 70), 'i', False)
                # Getting the type of 'self' (line 549)
                self_13654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 59), 'self', False)
                # Obtaining the member 'types' of a type (line 549)
                types_13655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 59), self_13654, 'types')
                # Obtaining the member '__getitem__' of a type (line 549)
                getitem___13656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 59), types_13655, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 549)
                subscript_call_result_13657 = invoke(stypy.reporting.localization.Localization(__file__, 549, 59), getitem___13656, i_13653)
                
                # Obtaining the member 'clone' of a type (line 549)
                clone_13658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 59), subscript_call_result_13657, 'clone')
                # Calling clone(args, kwargs) (line 549)
                clone_call_result_13660 = invoke(stypy.reporting.localization.Localization(__file__, 549, 59), clone_13658, *[], **kwargs_13659)
                
                # Processing the call keyword arguments (line 549)
                kwargs_13661 = {}
                # Getting the type of 'UnionType' (line 549)
                UnionType_13650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 31), 'UnionType', False)
                # Obtaining the member 'add' of a type (line 549)
                add_13651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 31), UnionType_13650, 'add')
                # Calling add(args, kwargs) (line 549)
                add_call_result_13662 = invoke(stypy.reporting.localization.Localization(__file__, 549, 31), add_13651, *[result_union_13652, clone_call_result_13660], **kwargs_13661)
                
                # Assigning a type to the variable 'result_union' (line 549)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 16), 'result_union', add_call_result_13662)
                # SSA branch for the else part of an if statement (line 548)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Name (line 551):
                
                # Call to add(...): (line 551)
                # Processing the call arguments (line 551)
                # Getting the type of 'result_union' (line 551)
                result_union_13665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 45), 'result_union', False)
                
                # Call to deepcopy(...): (line 551)
                # Processing the call arguments (line 551)
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 551)
                i_13668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 84), 'i', False)
                # Getting the type of 'self' (line 551)
                self_13669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 73), 'self', False)
                # Obtaining the member 'types' of a type (line 551)
                types_13670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 73), self_13669, 'types')
                # Obtaining the member '__getitem__' of a type (line 551)
                getitem___13671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 73), types_13670, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 551)
                subscript_call_result_13672 = invoke(stypy.reporting.localization.Localization(__file__, 551, 73), getitem___13671, i_13668)
                
                # Processing the call keyword arguments (line 551)
                kwargs_13673 = {}
                # Getting the type of 'copy' (line 551)
                copy_13666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 59), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 551)
                deepcopy_13667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 59), copy_13666, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 551)
                deepcopy_call_result_13674 = invoke(stypy.reporting.localization.Localization(__file__, 551, 59), deepcopy_13667, *[subscript_call_result_13672], **kwargs_13673)
                
                # Processing the call keyword arguments (line 551)
                kwargs_13675 = {}
                # Getting the type of 'UnionType' (line 551)
                UnionType_13663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 31), 'UnionType', False)
                # Obtaining the member 'add' of a type (line 551)
                add_13664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 31), UnionType_13663, 'add')
                # Calling add(args, kwargs) (line 551)
                add_call_result_13676 = invoke(stypy.reporting.localization.Localization(__file__, 551, 31), add_13664, *[result_union_13665, deepcopy_call_result_13674], **kwargs_13675)
                
                # Assigning a type to the variable 'result_union' (line 551)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 16), 'result_union', add_call_result_13676)
                # SSA join for if statement (line 548)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'result_union' (line 553)
        result_union_13677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 15), 'result_union')
        # Assigning a type to the variable 'stypy_return_type' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'stypy_return_type', result_union_13677)
        
        # ################# End of 'clone(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clone' in the type store
        # Getting the type of 'stypy_return_type' (line 542)
        stypy_return_type_13678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13678)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clone'
        return stypy_return_type_13678


    @norecursion
    def can_store_elements(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'can_store_elements'
        module_type_store = module_type_store.open_function_context('can_store_elements', 555, 4, False)
        # Assigning a type to the variable 'self' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.can_store_elements.__dict__.__setitem__('stypy_localization', localization)
        UnionType.can_store_elements.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.can_store_elements.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.can_store_elements.__dict__.__setitem__('stypy_function_name', 'UnionType.can_store_elements')
        UnionType.can_store_elements.__dict__.__setitem__('stypy_param_names_list', [])
        UnionType.can_store_elements.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.can_store_elements.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.can_store_elements.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.can_store_elements.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.can_store_elements.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.can_store_elements.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.can_store_elements', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'can_store_elements', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'can_store_elements(...)' code ##################

        
        # Assigning a Name to a Name (line 556):
        # Getting the type of 'False' (line 556)
        False_13679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 15), 'False')
        # Assigning a type to the variable 'temp' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'temp', False_13679)
        
        # Getting the type of 'self' (line 557)
        self_13680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 21), 'self')
        # Obtaining the member 'types' of a type (line 557)
        types_13681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 21), self_13680, 'types')
        # Assigning a type to the variable 'types_13681' (line 557)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'types_13681', types_13681)
        # Testing if the for loop is going to be iterated (line 557)
        # Testing the type of a for loop iterable (line 557)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 557, 8), types_13681)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 557, 8), types_13681):
            # Getting the type of the for loop variable (line 557)
            for_loop_var_13682 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 557, 8), types_13681)
            # Assigning a type to the variable 'type_' (line 557)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'type_', for_loop_var_13682)
            # SSA begins for a for statement (line 557)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'temp' (line 558)
            temp_13683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 12), 'temp')
            
            # Call to can_store_elements(...): (line 558)
            # Processing the call keyword arguments (line 558)
            kwargs_13686 = {}
            # Getting the type of 'type_' (line 558)
            type__13684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 20), 'type_', False)
            # Obtaining the member 'can_store_elements' of a type (line 558)
            can_store_elements_13685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 20), type__13684, 'can_store_elements')
            # Calling can_store_elements(args, kwargs) (line 558)
            can_store_elements_call_result_13687 = invoke(stypy.reporting.localization.Localization(__file__, 558, 20), can_store_elements_13685, *[], **kwargs_13686)
            
            # Applying the binary operator '|=' (line 558)
            result_ior_13688 = python_operator(stypy.reporting.localization.Localization(__file__, 558, 12), '|=', temp_13683, can_store_elements_call_result_13687)
            # Assigning a type to the variable 'temp' (line 558)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 12), 'temp', result_ior_13688)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'temp' (line 560)
        temp_13689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'stypy_return_type', temp_13689)
        
        # ################# End of 'can_store_elements(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'can_store_elements' in the type store
        # Getting the type of 'stypy_return_type' (line 555)
        stypy_return_type_13690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13690)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'can_store_elements'
        return stypy_return_type_13690


    @norecursion
    def can_store_keypairs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'can_store_keypairs'
        module_type_store = module_type_store.open_function_context('can_store_keypairs', 562, 4, False)
        # Assigning a type to the variable 'self' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.can_store_keypairs.__dict__.__setitem__('stypy_localization', localization)
        UnionType.can_store_keypairs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.can_store_keypairs.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.can_store_keypairs.__dict__.__setitem__('stypy_function_name', 'UnionType.can_store_keypairs')
        UnionType.can_store_keypairs.__dict__.__setitem__('stypy_param_names_list', [])
        UnionType.can_store_keypairs.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.can_store_keypairs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.can_store_keypairs.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.can_store_keypairs.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.can_store_keypairs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.can_store_keypairs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.can_store_keypairs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'can_store_keypairs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'can_store_keypairs(...)' code ##################

        
        # Assigning a Name to a Name (line 563):
        # Getting the type of 'False' (line 563)
        False_13691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 15), 'False')
        # Assigning a type to the variable 'temp' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'temp', False_13691)
        
        # Getting the type of 'self' (line 564)
        self_13692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 21), 'self')
        # Obtaining the member 'types' of a type (line 564)
        types_13693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 21), self_13692, 'types')
        # Assigning a type to the variable 'types_13693' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'types_13693', types_13693)
        # Testing if the for loop is going to be iterated (line 564)
        # Testing the type of a for loop iterable (line 564)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 564, 8), types_13693)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 564, 8), types_13693):
            # Getting the type of the for loop variable (line 564)
            for_loop_var_13694 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 564, 8), types_13693)
            # Assigning a type to the variable 'type_' (line 564)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'type_', for_loop_var_13694)
            # SSA begins for a for statement (line 564)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'temp' (line 565)
            temp_13695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 12), 'temp')
            
            # Call to can_store_keypairs(...): (line 565)
            # Processing the call keyword arguments (line 565)
            kwargs_13698 = {}
            # Getting the type of 'type_' (line 565)
            type__13696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 20), 'type_', False)
            # Obtaining the member 'can_store_keypairs' of a type (line 565)
            can_store_keypairs_13697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 20), type__13696, 'can_store_keypairs')
            # Calling can_store_keypairs(args, kwargs) (line 565)
            can_store_keypairs_call_result_13699 = invoke(stypy.reporting.localization.Localization(__file__, 565, 20), can_store_keypairs_13697, *[], **kwargs_13698)
            
            # Applying the binary operator '|=' (line 565)
            result_ior_13700 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 12), '|=', temp_13695, can_store_keypairs_call_result_13699)
            # Assigning a type to the variable 'temp' (line 565)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 12), 'temp', result_ior_13700)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'temp' (line 567)
        temp_13701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 567)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'stypy_return_type', temp_13701)
        
        # ################# End of 'can_store_keypairs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'can_store_keypairs' in the type store
        # Getting the type of 'stypy_return_type' (line 562)
        stypy_return_type_13702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13702)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'can_store_keypairs'
        return stypy_return_type_13702


    @norecursion
    def get_elements_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_elements_type'
        module_type_store = module_type_store.open_function_context('get_elements_type', 569, 4, False)
        # Assigning a type to the variable 'self' (line 570)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.get_elements_type.__dict__.__setitem__('stypy_localization', localization)
        UnionType.get_elements_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.get_elements_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.get_elements_type.__dict__.__setitem__('stypy_function_name', 'UnionType.get_elements_type')
        UnionType.get_elements_type.__dict__.__setitem__('stypy_param_names_list', [])
        UnionType.get_elements_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.get_elements_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.get_elements_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.get_elements_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.get_elements_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.get_elements_type.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.get_elements_type', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_elements_type', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_elements_type(...)' code ##################

        
        # Assigning a List to a Name (line 570):
        
        # Obtaining an instance of the builtin type 'list' (line 570)
        list_13703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 570)
        
        # Assigning a type to the variable 'errors' (line 570)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'errors', list_13703)
        
        # Assigning a Name to a Name (line 572):
        # Getting the type of 'None' (line 572)
        None_13704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 15), 'None')
        # Assigning a type to the variable 'temp' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'temp', None_13704)
        
        # Getting the type of 'self' (line 573)
        self_13705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 21), 'self')
        # Obtaining the member 'types' of a type (line 573)
        types_13706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 21), self_13705, 'types')
        # Assigning a type to the variable 'types_13706' (line 573)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'types_13706', types_13706)
        # Testing if the for loop is going to be iterated (line 573)
        # Testing the type of a for loop iterable (line 573)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 573, 8), types_13706)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 573, 8), types_13706):
            # Getting the type of the for loop variable (line 573)
            for_loop_var_13707 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 573, 8), types_13706)
            # Assigning a type to the variable 'type_' (line 573)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'type_', for_loop_var_13707)
            # SSA begins for a for statement (line 573)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 574):
            
            # Call to get_elements_type(...): (line 574)
            # Processing the call keyword arguments (line 574)
            kwargs_13710 = {}
            # Getting the type of 'type_' (line 574)
            type__13708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 18), 'type_', False)
            # Obtaining the member 'get_elements_type' of a type (line 574)
            get_elements_type_13709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 18), type__13708, 'get_elements_type')
            # Calling get_elements_type(args, kwargs) (line 574)
            get_elements_type_call_result_13711 = invoke(stypy.reporting.localization.Localization(__file__, 574, 18), get_elements_type_13709, *[], **kwargs_13710)
            
            # Assigning a type to the variable 'res' (line 574)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 12), 'res', get_elements_type_call_result_13711)
            
            # Type idiom detected: calculating its left and rigth part (line 575)
            # Getting the type of 'TypeError' (line 575)
            TypeError_13712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 31), 'TypeError')
            # Getting the type of 'res' (line 575)
            res_13713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 26), 'res')
            
            (may_be_13714, more_types_in_union_13715) = may_be_subtype(TypeError_13712, res_13713)

            if may_be_13714:

                if more_types_in_union_13715:
                    # Runtime conditional SSA (line 575)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'res' (line 575)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'res', remove_not_subtype_from_union(res_13713, TypeError))
                
                # Call to append(...): (line 576)
                # Processing the call arguments (line 576)
                # Getting the type of 'temp' (line 576)
                temp_13718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 30), 'temp', False)
                # Processing the call keyword arguments (line 576)
                kwargs_13719 = {}
                # Getting the type of 'errors' (line 576)
                errors_13716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 16), 'errors', False)
                # Obtaining the member 'append' of a type (line 576)
                append_13717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 16), errors_13716, 'append')
                # Calling append(args, kwargs) (line 576)
                append_call_result_13720 = invoke(stypy.reporting.localization.Localization(__file__, 576, 16), append_13717, *[temp_13718], **kwargs_13719)
                

                if more_types_in_union_13715:
                    # Runtime conditional SSA for else branch (line 575)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_13714) or more_types_in_union_13715):
                # Assigning a type to the variable 'res' (line 575)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'res', remove_subtype_from_union(res_13713, TypeError))
                
                # Assigning a Call to a Name (line 578):
                
                # Call to add(...): (line 578)
                # Processing the call arguments (line 578)
                # Getting the type of 'temp' (line 578)
                temp_13723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 37), 'temp', False)
                # Getting the type of 'res' (line 578)
                res_13724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 43), 'res', False)
                # Processing the call keyword arguments (line 578)
                kwargs_13725 = {}
                # Getting the type of 'UnionType' (line 578)
                UnionType_13721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 23), 'UnionType', False)
                # Obtaining the member 'add' of a type (line 578)
                add_13722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 23), UnionType_13721, 'add')
                # Calling add(args, kwargs) (line 578)
                add_call_result_13726 = invoke(stypy.reporting.localization.Localization(__file__, 578, 23), add_13722, *[temp_13723, res_13724], **kwargs_13725)
                
                # Assigning a type to the variable 'temp' (line 578)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 16), 'temp', add_call_result_13726)

                if (may_be_13714 and more_types_in_union_13715):
                    # SSA join for if statement (line 575)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 581)
        # Processing the call arguments (line 581)
        # Getting the type of 'errors' (line 581)
        errors_13728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 15), 'errors', False)
        # Processing the call keyword arguments (line 581)
        kwargs_13729 = {}
        # Getting the type of 'len' (line 581)
        len_13727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 11), 'len', False)
        # Calling len(args, kwargs) (line 581)
        len_call_result_13730 = invoke(stypy.reporting.localization.Localization(__file__, 581, 11), len_13727, *[errors_13728], **kwargs_13729)
        
        
        # Call to len(...): (line 581)
        # Processing the call arguments (line 581)
        # Getting the type of 'self' (line 581)
        self_13732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 581)
        types_13733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 30), self_13732, 'types')
        # Processing the call keyword arguments (line 581)
        kwargs_13734 = {}
        # Getting the type of 'len' (line 581)
        len_13731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 26), 'len', False)
        # Calling len(args, kwargs) (line 581)
        len_call_result_13735 = invoke(stypy.reporting.localization.Localization(__file__, 581, 26), len_13731, *[types_13733], **kwargs_13734)
        
        # Applying the binary operator '==' (line 581)
        result_eq_13736 = python_operator(stypy.reporting.localization.Localization(__file__, 581, 11), '==', len_call_result_13730, len_call_result_13735)
        
        # Testing if the type of an if condition is none (line 581)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 581, 8), result_eq_13736):
            
            # Getting the type of 'errors' (line 588)
            errors_13749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 25), 'errors')
            # Assigning a type to the variable 'errors_13749' (line 588)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 12), 'errors_13749', errors_13749)
            # Testing if the for loop is going to be iterated (line 588)
            # Testing the type of a for loop iterable (line 588)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 588, 12), errors_13749)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 588, 12), errors_13749):
                # Getting the type of the for loop variable (line 588)
                for_loop_var_13750 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 588, 12), errors_13749)
                # Assigning a type to the variable 'error' (line 588)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 12), 'error', for_loop_var_13750)
                # SSA begins for a for statement (line 588)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 589)
                # Processing the call keyword arguments (line 589)
                kwargs_13753 = {}
                # Getting the type of 'error' (line 589)
                error_13751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 589)
                turn_to_warning_13752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 16), error_13751, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 589)
                turn_to_warning_call_result_13754 = invoke(stypy.reporting.localization.Localization(__file__, 589, 16), turn_to_warning_13752, *[], **kwargs_13753)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 581)
            if_condition_13737 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 581, 8), result_eq_13736)
            # Assigning a type to the variable 'if_condition_13737' (line 581)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'if_condition_13737', if_condition_13737)
            # SSA begins for if statement (line 581)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 582)
            # Processing the call arguments (line 582)
            # Getting the type of 'None' (line 582)
            None_13739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 29), 'None', False)
            
            # Call to format(...): (line 582)
            # Processing the call arguments (line 582)
            str_13742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 16), 'str', 'get_elements_type')
            # Getting the type of 'self' (line 583)
            self_13743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 37), 'self', False)
            # Obtaining the member 'types' of a type (line 583)
            types_13744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 37), self_13743, 'types')
            # Processing the call keyword arguments (line 582)
            kwargs_13745 = {}
            str_13740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 35), 'str', "None of the possible types ('{1}') can invoke the member '{0}'")
            # Obtaining the member 'format' of a type (line 582)
            format_13741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 35), str_13740, 'format')
            # Calling format(args, kwargs) (line 582)
            format_call_result_13746 = invoke(stypy.reporting.localization.Localization(__file__, 582, 35), format_13741, *[str_13742, types_13744], **kwargs_13745)
            
            # Processing the call keyword arguments (line 582)
            kwargs_13747 = {}
            # Getting the type of 'TypeError' (line 582)
            TypeError_13738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 582)
            TypeError_call_result_13748 = invoke(stypy.reporting.localization.Localization(__file__, 582, 19), TypeError_13738, *[None_13739, format_call_result_13746], **kwargs_13747)
            
            # Assigning a type to the variable 'stypy_return_type' (line 582)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 12), 'stypy_return_type', TypeError_call_result_13748)
            # SSA branch for the else part of an if statement (line 581)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 588)
            errors_13749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 25), 'errors')
            # Assigning a type to the variable 'errors_13749' (line 588)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 12), 'errors_13749', errors_13749)
            # Testing if the for loop is going to be iterated (line 588)
            # Testing the type of a for loop iterable (line 588)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 588, 12), errors_13749)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 588, 12), errors_13749):
                # Getting the type of the for loop variable (line 588)
                for_loop_var_13750 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 588, 12), errors_13749)
                # Assigning a type to the variable 'error' (line 588)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 12), 'error', for_loop_var_13750)
                # SSA begins for a for statement (line 588)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 589)
                # Processing the call keyword arguments (line 589)
                kwargs_13753 = {}
                # Getting the type of 'error' (line 589)
                error_13751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 589)
                turn_to_warning_13752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 16), error_13751, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 589)
                turn_to_warning_call_result_13754 = invoke(stypy.reporting.localization.Localization(__file__, 589, 16), turn_to_warning_13752, *[], **kwargs_13753)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 581)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'temp' (line 591)
        temp_13755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 591)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'stypy_return_type', temp_13755)
        
        # ################# End of 'get_elements_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_elements_type' in the type store
        # Getting the type of 'stypy_return_type' (line 569)
        stypy_return_type_13756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13756)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_elements_type'
        return stypy_return_type_13756


    @norecursion
    def set_elements_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 593)
        True_13757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 79), 'True')
        defaults = [True_13757]
        # Create a new context for function 'set_elements_type'
        module_type_store = module_type_store.open_function_context('set_elements_type', 593, 4, False)
        # Assigning a type to the variable 'self' (line 594)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.set_elements_type.__dict__.__setitem__('stypy_localization', localization)
        UnionType.set_elements_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.set_elements_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.set_elements_type.__dict__.__setitem__('stypy_function_name', 'UnionType.set_elements_type')
        UnionType.set_elements_type.__dict__.__setitem__('stypy_param_names_list', ['localization', 'elements_type', 'record_annotation'])
        UnionType.set_elements_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.set_elements_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.set_elements_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.set_elements_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.set_elements_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.set_elements_type.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.set_elements_type', ['localization', 'elements_type', 'record_annotation'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_elements_type', localization, ['localization', 'elements_type', 'record_annotation'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_elements_type(...)' code ##################

        
        # Assigning a List to a Name (line 594):
        
        # Obtaining an instance of the builtin type 'list' (line 594)
        list_13758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 594)
        
        # Assigning a type to the variable 'errors' (line 594)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'errors', list_13758)
        
        # Assigning a Name to a Name (line 596):
        # Getting the type of 'None' (line 596)
        None_13759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 15), 'None')
        # Assigning a type to the variable 'temp' (line 596)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'temp', None_13759)
        
        # Getting the type of 'self' (line 597)
        self_13760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 21), 'self')
        # Obtaining the member 'types' of a type (line 597)
        types_13761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 21), self_13760, 'types')
        # Assigning a type to the variable 'types_13761' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'types_13761', types_13761)
        # Testing if the for loop is going to be iterated (line 597)
        # Testing the type of a for loop iterable (line 597)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 597, 8), types_13761)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 597, 8), types_13761):
            # Getting the type of the for loop variable (line 597)
            for_loop_var_13762 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 597, 8), types_13761)
            # Assigning a type to the variable 'type_' (line 597)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'type_', for_loop_var_13762)
            # SSA begins for a for statement (line 597)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 598):
            
            # Call to set_elements_type(...): (line 598)
            # Processing the call arguments (line 598)
            # Getting the type of 'localization' (line 598)
            localization_13765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 42), 'localization', False)
            # Getting the type of 'elements_type' (line 598)
            elements_type_13766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 56), 'elements_type', False)
            # Getting the type of 'record_annotation' (line 598)
            record_annotation_13767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 71), 'record_annotation', False)
            # Processing the call keyword arguments (line 598)
            kwargs_13768 = {}
            # Getting the type of 'type_' (line 598)
            type__13763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 18), 'type_', False)
            # Obtaining the member 'set_elements_type' of a type (line 598)
            set_elements_type_13764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 18), type__13763, 'set_elements_type')
            # Calling set_elements_type(args, kwargs) (line 598)
            set_elements_type_call_result_13769 = invoke(stypy.reporting.localization.Localization(__file__, 598, 18), set_elements_type_13764, *[localization_13765, elements_type_13766, record_annotation_13767], **kwargs_13768)
            
            # Assigning a type to the variable 'res' (line 598)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 12), 'res', set_elements_type_call_result_13769)
            
            # Type idiom detected: calculating its left and rigth part (line 599)
            # Getting the type of 'TypeError' (line 599)
            TypeError_13770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 31), 'TypeError')
            # Getting the type of 'res' (line 599)
            res_13771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 26), 'res')
            
            (may_be_13772, more_types_in_union_13773) = may_be_subtype(TypeError_13770, res_13771)

            if may_be_13772:

                if more_types_in_union_13773:
                    # Runtime conditional SSA (line 599)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'res' (line 599)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 12), 'res', remove_not_subtype_from_union(res_13771, TypeError))
                
                # Call to append(...): (line 600)
                # Processing the call arguments (line 600)
                # Getting the type of 'temp' (line 600)
                temp_13776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 30), 'temp', False)
                # Processing the call keyword arguments (line 600)
                kwargs_13777 = {}
                # Getting the type of 'errors' (line 600)
                errors_13774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 16), 'errors', False)
                # Obtaining the member 'append' of a type (line 600)
                append_13775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 16), errors_13774, 'append')
                # Calling append(args, kwargs) (line 600)
                append_call_result_13778 = invoke(stypy.reporting.localization.Localization(__file__, 600, 16), append_13775, *[temp_13776], **kwargs_13777)
                

                if more_types_in_union_13773:
                    # SSA join for if statement (line 599)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 603)
        # Processing the call arguments (line 603)
        # Getting the type of 'errors' (line 603)
        errors_13780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 15), 'errors', False)
        # Processing the call keyword arguments (line 603)
        kwargs_13781 = {}
        # Getting the type of 'len' (line 603)
        len_13779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 11), 'len', False)
        # Calling len(args, kwargs) (line 603)
        len_call_result_13782 = invoke(stypy.reporting.localization.Localization(__file__, 603, 11), len_13779, *[errors_13780], **kwargs_13781)
        
        
        # Call to len(...): (line 603)
        # Processing the call arguments (line 603)
        # Getting the type of 'self' (line 603)
        self_13784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 603)
        types_13785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 30), self_13784, 'types')
        # Processing the call keyword arguments (line 603)
        kwargs_13786 = {}
        # Getting the type of 'len' (line 603)
        len_13783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 26), 'len', False)
        # Calling len(args, kwargs) (line 603)
        len_call_result_13787 = invoke(stypy.reporting.localization.Localization(__file__, 603, 26), len_13783, *[types_13785], **kwargs_13786)
        
        # Applying the binary operator '==' (line 603)
        result_eq_13788 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 11), '==', len_call_result_13782, len_call_result_13787)
        
        # Testing if the type of an if condition is none (line 603)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 603, 8), result_eq_13788):
            
            # Getting the type of 'errors' (line 610)
            errors_13801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 25), 'errors')
            # Assigning a type to the variable 'errors_13801' (line 610)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'errors_13801', errors_13801)
            # Testing if the for loop is going to be iterated (line 610)
            # Testing the type of a for loop iterable (line 610)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 610, 12), errors_13801)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 610, 12), errors_13801):
                # Getting the type of the for loop variable (line 610)
                for_loop_var_13802 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 610, 12), errors_13801)
                # Assigning a type to the variable 'error' (line 610)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'error', for_loop_var_13802)
                # SSA begins for a for statement (line 610)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 611)
                # Processing the call keyword arguments (line 611)
                kwargs_13805 = {}
                # Getting the type of 'error' (line 611)
                error_13803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 611)
                turn_to_warning_13804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 16), error_13803, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 611)
                turn_to_warning_call_result_13806 = invoke(stypy.reporting.localization.Localization(__file__, 611, 16), turn_to_warning_13804, *[], **kwargs_13805)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 603)
            if_condition_13789 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 603, 8), result_eq_13788)
            # Assigning a type to the variable 'if_condition_13789' (line 603)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'if_condition_13789', if_condition_13789)
            # SSA begins for if statement (line 603)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 604)
            # Processing the call arguments (line 604)
            # Getting the type of 'localization' (line 604)
            localization_13791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 29), 'localization', False)
            
            # Call to format(...): (line 604)
            # Processing the call arguments (line 604)
            str_13794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 16), 'str', 'set_elements_type')
            # Getting the type of 'self' (line 605)
            self_13795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 37), 'self', False)
            # Obtaining the member 'types' of a type (line 605)
            types_13796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 37), self_13795, 'types')
            # Processing the call keyword arguments (line 604)
            kwargs_13797 = {}
            str_13792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 43), 'str', "None of the possible types ('{1}') can invoke the member '{0}'")
            # Obtaining the member 'format' of a type (line 604)
            format_13793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 43), str_13792, 'format')
            # Calling format(args, kwargs) (line 604)
            format_call_result_13798 = invoke(stypy.reporting.localization.Localization(__file__, 604, 43), format_13793, *[str_13794, types_13796], **kwargs_13797)
            
            # Processing the call keyword arguments (line 604)
            kwargs_13799 = {}
            # Getting the type of 'TypeError' (line 604)
            TypeError_13790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 604)
            TypeError_call_result_13800 = invoke(stypy.reporting.localization.Localization(__file__, 604, 19), TypeError_13790, *[localization_13791, format_call_result_13798], **kwargs_13799)
            
            # Assigning a type to the variable 'stypy_return_type' (line 604)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 12), 'stypy_return_type', TypeError_call_result_13800)
            # SSA branch for the else part of an if statement (line 603)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 610)
            errors_13801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 25), 'errors')
            # Assigning a type to the variable 'errors_13801' (line 610)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'errors_13801', errors_13801)
            # Testing if the for loop is going to be iterated (line 610)
            # Testing the type of a for loop iterable (line 610)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 610, 12), errors_13801)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 610, 12), errors_13801):
                # Getting the type of the for loop variable (line 610)
                for_loop_var_13802 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 610, 12), errors_13801)
                # Assigning a type to the variable 'error' (line 610)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'error', for_loop_var_13802)
                # SSA begins for a for statement (line 610)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 611)
                # Processing the call keyword arguments (line 611)
                kwargs_13805 = {}
                # Getting the type of 'error' (line 611)
                error_13803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 611)
                turn_to_warning_13804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 16), error_13803, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 611)
                turn_to_warning_call_result_13806 = invoke(stypy.reporting.localization.Localization(__file__, 611, 16), turn_to_warning_13804, *[], **kwargs_13805)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 603)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'temp' (line 613)
        temp_13807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 8), 'stypy_return_type', temp_13807)
        
        # ################# End of 'set_elements_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_elements_type' in the type store
        # Getting the type of 'stypy_return_type' (line 593)
        stypy_return_type_13808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13808)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_elements_type'
        return stypy_return_type_13808


    @norecursion
    def add_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 615)
        True_13809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 62), 'True')
        defaults = [True_13809]
        # Create a new context for function 'add_type'
        module_type_store = module_type_store.open_function_context('add_type', 615, 4, False)
        # Assigning a type to the variable 'self' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.add_type.__dict__.__setitem__('stypy_localization', localization)
        UnionType.add_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.add_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.add_type.__dict__.__setitem__('stypy_function_name', 'UnionType.add_type')
        UnionType.add_type.__dict__.__setitem__('stypy_param_names_list', ['localization', 'type_', 'record_annotation'])
        UnionType.add_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.add_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.add_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.add_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.add_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.add_type.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.add_type', ['localization', 'type_', 'record_annotation'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_type', localization, ['localization', 'type_', 'record_annotation'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_type(...)' code ##################

        
        # Assigning a List to a Name (line 616):
        
        # Obtaining an instance of the builtin type 'list' (line 616)
        list_13810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 616)
        
        # Assigning a type to the variable 'errors' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 8), 'errors', list_13810)
        
        # Assigning a Name to a Name (line 618):
        # Getting the type of 'None' (line 618)
        None_13811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 15), 'None')
        # Assigning a type to the variable 'temp' (line 618)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'temp', None_13811)
        
        # Getting the type of 'self' (line 619)
        self_13812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 21), 'self')
        # Obtaining the member 'types' of a type (line 619)
        types_13813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 21), self_13812, 'types')
        # Assigning a type to the variable 'types_13813' (line 619)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), 'types_13813', types_13813)
        # Testing if the for loop is going to be iterated (line 619)
        # Testing the type of a for loop iterable (line 619)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 619, 8), types_13813)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 619, 8), types_13813):
            # Getting the type of the for loop variable (line 619)
            for_loop_var_13814 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 619, 8), types_13813)
            # Assigning a type to the variable 'type_' (line 619)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), 'type_', for_loop_var_13814)
            # SSA begins for a for statement (line 619)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 620):
            
            # Call to add_type(...): (line 620)
            # Processing the call arguments (line 620)
            # Getting the type of 'localization' (line 620)
            localization_13817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 33), 'localization', False)
            # Getting the type of 'type_' (line 620)
            type__13818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 47), 'type_', False)
            # Getting the type of 'record_annotation' (line 620)
            record_annotation_13819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 54), 'record_annotation', False)
            # Processing the call keyword arguments (line 620)
            kwargs_13820 = {}
            # Getting the type of 'type_' (line 620)
            type__13815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 18), 'type_', False)
            # Obtaining the member 'add_type' of a type (line 620)
            add_type_13816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 18), type__13815, 'add_type')
            # Calling add_type(args, kwargs) (line 620)
            add_type_call_result_13821 = invoke(stypy.reporting.localization.Localization(__file__, 620, 18), add_type_13816, *[localization_13817, type__13818, record_annotation_13819], **kwargs_13820)
            
            # Assigning a type to the variable 'res' (line 620)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 12), 'res', add_type_call_result_13821)
            
            # Type idiom detected: calculating its left and rigth part (line 621)
            # Getting the type of 'TypeError' (line 621)
            TypeError_13822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 31), 'TypeError')
            # Getting the type of 'res' (line 621)
            res_13823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 26), 'res')
            
            (may_be_13824, more_types_in_union_13825) = may_be_subtype(TypeError_13822, res_13823)

            if may_be_13824:

                if more_types_in_union_13825:
                    # Runtime conditional SSA (line 621)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'res' (line 621)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 12), 'res', remove_not_subtype_from_union(res_13823, TypeError))
                
                # Call to append(...): (line 622)
                # Processing the call arguments (line 622)
                # Getting the type of 'temp' (line 622)
                temp_13828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 30), 'temp', False)
                # Processing the call keyword arguments (line 622)
                kwargs_13829 = {}
                # Getting the type of 'errors' (line 622)
                errors_13826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 16), 'errors', False)
                # Obtaining the member 'append' of a type (line 622)
                append_13827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 16), errors_13826, 'append')
                # Calling append(args, kwargs) (line 622)
                append_call_result_13830 = invoke(stypy.reporting.localization.Localization(__file__, 622, 16), append_13827, *[temp_13828], **kwargs_13829)
                

                if more_types_in_union_13825:
                    # SSA join for if statement (line 621)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 625)
        # Processing the call arguments (line 625)
        # Getting the type of 'errors' (line 625)
        errors_13832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 15), 'errors', False)
        # Processing the call keyword arguments (line 625)
        kwargs_13833 = {}
        # Getting the type of 'len' (line 625)
        len_13831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 11), 'len', False)
        # Calling len(args, kwargs) (line 625)
        len_call_result_13834 = invoke(stypy.reporting.localization.Localization(__file__, 625, 11), len_13831, *[errors_13832], **kwargs_13833)
        
        
        # Call to len(...): (line 625)
        # Processing the call arguments (line 625)
        # Getting the type of 'self' (line 625)
        self_13836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 625)
        types_13837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 30), self_13836, 'types')
        # Processing the call keyword arguments (line 625)
        kwargs_13838 = {}
        # Getting the type of 'len' (line 625)
        len_13835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 26), 'len', False)
        # Calling len(args, kwargs) (line 625)
        len_call_result_13839 = invoke(stypy.reporting.localization.Localization(__file__, 625, 26), len_13835, *[types_13837], **kwargs_13838)
        
        # Applying the binary operator '==' (line 625)
        result_eq_13840 = python_operator(stypy.reporting.localization.Localization(__file__, 625, 11), '==', len_call_result_13834, len_call_result_13839)
        
        # Testing if the type of an if condition is none (line 625)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 625, 8), result_eq_13840):
            
            # Getting the type of 'errors' (line 632)
            errors_13853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 25), 'errors')
            # Assigning a type to the variable 'errors_13853' (line 632)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 12), 'errors_13853', errors_13853)
            # Testing if the for loop is going to be iterated (line 632)
            # Testing the type of a for loop iterable (line 632)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 632, 12), errors_13853)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 632, 12), errors_13853):
                # Getting the type of the for loop variable (line 632)
                for_loop_var_13854 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 632, 12), errors_13853)
                # Assigning a type to the variable 'error' (line 632)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 12), 'error', for_loop_var_13854)
                # SSA begins for a for statement (line 632)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 633)
                # Processing the call keyword arguments (line 633)
                kwargs_13857 = {}
                # Getting the type of 'error' (line 633)
                error_13855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 633)
                turn_to_warning_13856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 16), error_13855, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 633)
                turn_to_warning_call_result_13858 = invoke(stypy.reporting.localization.Localization(__file__, 633, 16), turn_to_warning_13856, *[], **kwargs_13857)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 625)
            if_condition_13841 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 625, 8), result_eq_13840)
            # Assigning a type to the variable 'if_condition_13841' (line 625)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 8), 'if_condition_13841', if_condition_13841)
            # SSA begins for if statement (line 625)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 626)
            # Processing the call arguments (line 626)
            # Getting the type of 'localization' (line 626)
            localization_13843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 29), 'localization', False)
            
            # Call to format(...): (line 626)
            # Processing the call arguments (line 626)
            str_13846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 16), 'str', 'add_type')
            # Getting the type of 'self' (line 627)
            self_13847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 28), 'self', False)
            # Obtaining the member 'types' of a type (line 627)
            types_13848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 28), self_13847, 'types')
            # Processing the call keyword arguments (line 626)
            kwargs_13849 = {}
            str_13844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 43), 'str', "None of the possible types ('{1}') can invoke the member '{0}'")
            # Obtaining the member 'format' of a type (line 626)
            format_13845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 43), str_13844, 'format')
            # Calling format(args, kwargs) (line 626)
            format_call_result_13850 = invoke(stypy.reporting.localization.Localization(__file__, 626, 43), format_13845, *[str_13846, types_13848], **kwargs_13849)
            
            # Processing the call keyword arguments (line 626)
            kwargs_13851 = {}
            # Getting the type of 'TypeError' (line 626)
            TypeError_13842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 626)
            TypeError_call_result_13852 = invoke(stypy.reporting.localization.Localization(__file__, 626, 19), TypeError_13842, *[localization_13843, format_call_result_13850], **kwargs_13851)
            
            # Assigning a type to the variable 'stypy_return_type' (line 626)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 12), 'stypy_return_type', TypeError_call_result_13852)
            # SSA branch for the else part of an if statement (line 625)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 632)
            errors_13853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 25), 'errors')
            # Assigning a type to the variable 'errors_13853' (line 632)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 12), 'errors_13853', errors_13853)
            # Testing if the for loop is going to be iterated (line 632)
            # Testing the type of a for loop iterable (line 632)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 632, 12), errors_13853)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 632, 12), errors_13853):
                # Getting the type of the for loop variable (line 632)
                for_loop_var_13854 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 632, 12), errors_13853)
                # Assigning a type to the variable 'error' (line 632)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 12), 'error', for_loop_var_13854)
                # SSA begins for a for statement (line 632)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 633)
                # Processing the call keyword arguments (line 633)
                kwargs_13857 = {}
                # Getting the type of 'error' (line 633)
                error_13855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 633)
                turn_to_warning_13856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 16), error_13855, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 633)
                turn_to_warning_call_result_13858 = invoke(stypy.reporting.localization.Localization(__file__, 633, 16), turn_to_warning_13856, *[], **kwargs_13857)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 625)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'temp' (line 635)
        temp_13859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 635)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 8), 'stypy_return_type', temp_13859)
        
        # ################# End of 'add_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_type' in the type store
        # Getting the type of 'stypy_return_type' (line 615)
        stypy_return_type_13860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13860)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_type'
        return stypy_return_type_13860


    @norecursion
    def add_types_from_list(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 637)
        True_13861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 77), 'True')
        defaults = [True_13861]
        # Create a new context for function 'add_types_from_list'
        module_type_store = module_type_store.open_function_context('add_types_from_list', 637, 4, False)
        # Assigning a type to the variable 'self' (line 638)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.add_types_from_list.__dict__.__setitem__('stypy_localization', localization)
        UnionType.add_types_from_list.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.add_types_from_list.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.add_types_from_list.__dict__.__setitem__('stypy_function_name', 'UnionType.add_types_from_list')
        UnionType.add_types_from_list.__dict__.__setitem__('stypy_param_names_list', ['localization', 'type_list', 'record_annotation'])
        UnionType.add_types_from_list.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.add_types_from_list.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.add_types_from_list.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.add_types_from_list.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.add_types_from_list.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.add_types_from_list.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.add_types_from_list', ['localization', 'type_list', 'record_annotation'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_types_from_list', localization, ['localization', 'type_list', 'record_annotation'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_types_from_list(...)' code ##################

        
        # Assigning a List to a Name (line 638):
        
        # Obtaining an instance of the builtin type 'list' (line 638)
        list_13862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 638)
        
        # Assigning a type to the variable 'errors' (line 638)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 8), 'errors', list_13862)
        
        # Assigning a Name to a Name (line 640):
        # Getting the type of 'None' (line 640)
        None_13863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 15), 'None')
        # Assigning a type to the variable 'temp' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'temp', None_13863)
        
        # Getting the type of 'self' (line 641)
        self_13864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 21), 'self')
        # Obtaining the member 'types' of a type (line 641)
        types_13865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 21), self_13864, 'types')
        # Assigning a type to the variable 'types_13865' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'types_13865', types_13865)
        # Testing if the for loop is going to be iterated (line 641)
        # Testing the type of a for loop iterable (line 641)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 641, 8), types_13865)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 641, 8), types_13865):
            # Getting the type of the for loop variable (line 641)
            for_loop_var_13866 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 641, 8), types_13865)
            # Assigning a type to the variable 'type_' (line 641)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'type_', for_loop_var_13866)
            # SSA begins for a for statement (line 641)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 642):
            
            # Call to add_types_from_list(...): (line 642)
            # Processing the call arguments (line 642)
            # Getting the type of 'localization' (line 642)
            localization_13869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 44), 'localization', False)
            # Getting the type of 'type_list' (line 642)
            type_list_13870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 58), 'type_list', False)
            # Getting the type of 'record_annotation' (line 642)
            record_annotation_13871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 69), 'record_annotation', False)
            # Processing the call keyword arguments (line 642)
            kwargs_13872 = {}
            # Getting the type of 'type_' (line 642)
            type__13867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 18), 'type_', False)
            # Obtaining the member 'add_types_from_list' of a type (line 642)
            add_types_from_list_13868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 18), type__13867, 'add_types_from_list')
            # Calling add_types_from_list(args, kwargs) (line 642)
            add_types_from_list_call_result_13873 = invoke(stypy.reporting.localization.Localization(__file__, 642, 18), add_types_from_list_13868, *[localization_13869, type_list_13870, record_annotation_13871], **kwargs_13872)
            
            # Assigning a type to the variable 'res' (line 642)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 12), 'res', add_types_from_list_call_result_13873)
            
            # Type idiom detected: calculating its left and rigth part (line 643)
            # Getting the type of 'TypeError' (line 643)
            TypeError_13874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 31), 'TypeError')
            # Getting the type of 'res' (line 643)
            res_13875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 26), 'res')
            
            (may_be_13876, more_types_in_union_13877) = may_be_subtype(TypeError_13874, res_13875)

            if may_be_13876:

                if more_types_in_union_13877:
                    # Runtime conditional SSA (line 643)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'res' (line 643)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 12), 'res', remove_not_subtype_from_union(res_13875, TypeError))
                
                # Call to append(...): (line 644)
                # Processing the call arguments (line 644)
                # Getting the type of 'temp' (line 644)
                temp_13880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 30), 'temp', False)
                # Processing the call keyword arguments (line 644)
                kwargs_13881 = {}
                # Getting the type of 'errors' (line 644)
                errors_13878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 16), 'errors', False)
                # Obtaining the member 'append' of a type (line 644)
                append_13879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 16), errors_13878, 'append')
                # Calling append(args, kwargs) (line 644)
                append_call_result_13882 = invoke(stypy.reporting.localization.Localization(__file__, 644, 16), append_13879, *[temp_13880], **kwargs_13881)
                

                if more_types_in_union_13877:
                    # SSA join for if statement (line 643)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 647)
        # Processing the call arguments (line 647)
        # Getting the type of 'errors' (line 647)
        errors_13884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 15), 'errors', False)
        # Processing the call keyword arguments (line 647)
        kwargs_13885 = {}
        # Getting the type of 'len' (line 647)
        len_13883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 11), 'len', False)
        # Calling len(args, kwargs) (line 647)
        len_call_result_13886 = invoke(stypy.reporting.localization.Localization(__file__, 647, 11), len_13883, *[errors_13884], **kwargs_13885)
        
        
        # Call to len(...): (line 647)
        # Processing the call arguments (line 647)
        # Getting the type of 'self' (line 647)
        self_13888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 647)
        types_13889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 30), self_13888, 'types')
        # Processing the call keyword arguments (line 647)
        kwargs_13890 = {}
        # Getting the type of 'len' (line 647)
        len_13887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 26), 'len', False)
        # Calling len(args, kwargs) (line 647)
        len_call_result_13891 = invoke(stypy.reporting.localization.Localization(__file__, 647, 26), len_13887, *[types_13889], **kwargs_13890)
        
        # Applying the binary operator '==' (line 647)
        result_eq_13892 = python_operator(stypy.reporting.localization.Localization(__file__, 647, 11), '==', len_call_result_13886, len_call_result_13891)
        
        # Testing if the type of an if condition is none (line 647)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 647, 8), result_eq_13892):
            
            # Getting the type of 'errors' (line 654)
            errors_13905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 25), 'errors')
            # Assigning a type to the variable 'errors_13905' (line 654)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 12), 'errors_13905', errors_13905)
            # Testing if the for loop is going to be iterated (line 654)
            # Testing the type of a for loop iterable (line 654)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 654, 12), errors_13905)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 654, 12), errors_13905):
                # Getting the type of the for loop variable (line 654)
                for_loop_var_13906 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 654, 12), errors_13905)
                # Assigning a type to the variable 'error' (line 654)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 12), 'error', for_loop_var_13906)
                # SSA begins for a for statement (line 654)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 655)
                # Processing the call keyword arguments (line 655)
                kwargs_13909 = {}
                # Getting the type of 'error' (line 655)
                error_13907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 655)
                turn_to_warning_13908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 16), error_13907, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 655)
                turn_to_warning_call_result_13910 = invoke(stypy.reporting.localization.Localization(__file__, 655, 16), turn_to_warning_13908, *[], **kwargs_13909)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 647)
            if_condition_13893 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 647, 8), result_eq_13892)
            # Assigning a type to the variable 'if_condition_13893' (line 647)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'if_condition_13893', if_condition_13893)
            # SSA begins for if statement (line 647)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 648)
            # Processing the call arguments (line 648)
            # Getting the type of 'localization' (line 648)
            localization_13895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 29), 'localization', False)
            
            # Call to format(...): (line 648)
            # Processing the call arguments (line 648)
            str_13898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 16), 'str', 'add_types_from_list')
            # Getting the type of 'self' (line 649)
            self_13899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 39), 'self', False)
            # Obtaining the member 'types' of a type (line 649)
            types_13900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 39), self_13899, 'types')
            # Processing the call keyword arguments (line 648)
            kwargs_13901 = {}
            str_13896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 43), 'str', "None of the possible types ('{1}') can invoke the member '{0}'")
            # Obtaining the member 'format' of a type (line 648)
            format_13897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 43), str_13896, 'format')
            # Calling format(args, kwargs) (line 648)
            format_call_result_13902 = invoke(stypy.reporting.localization.Localization(__file__, 648, 43), format_13897, *[str_13898, types_13900], **kwargs_13901)
            
            # Processing the call keyword arguments (line 648)
            kwargs_13903 = {}
            # Getting the type of 'TypeError' (line 648)
            TypeError_13894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 648)
            TypeError_call_result_13904 = invoke(stypy.reporting.localization.Localization(__file__, 648, 19), TypeError_13894, *[localization_13895, format_call_result_13902], **kwargs_13903)
            
            # Assigning a type to the variable 'stypy_return_type' (line 648)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 12), 'stypy_return_type', TypeError_call_result_13904)
            # SSA branch for the else part of an if statement (line 647)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 654)
            errors_13905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 25), 'errors')
            # Assigning a type to the variable 'errors_13905' (line 654)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 12), 'errors_13905', errors_13905)
            # Testing if the for loop is going to be iterated (line 654)
            # Testing the type of a for loop iterable (line 654)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 654, 12), errors_13905)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 654, 12), errors_13905):
                # Getting the type of the for loop variable (line 654)
                for_loop_var_13906 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 654, 12), errors_13905)
                # Assigning a type to the variable 'error' (line 654)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 12), 'error', for_loop_var_13906)
                # SSA begins for a for statement (line 654)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 655)
                # Processing the call keyword arguments (line 655)
                kwargs_13909 = {}
                # Getting the type of 'error' (line 655)
                error_13907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 655)
                turn_to_warning_13908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 16), error_13907, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 655)
                turn_to_warning_call_result_13910 = invoke(stypy.reporting.localization.Localization(__file__, 655, 16), turn_to_warning_13908, *[], **kwargs_13909)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 647)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'temp' (line 657)
        temp_13911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 657)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 8), 'stypy_return_type', temp_13911)
        
        # ################# End of 'add_types_from_list(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_types_from_list' in the type store
        # Getting the type of 'stypy_return_type' (line 637)
        stypy_return_type_13912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13912)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_types_from_list'
        return stypy_return_type_13912


    @norecursion
    def get_values_from_key(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_values_from_key'
        module_type_store = module_type_store.open_function_context('get_values_from_key', 659, 4, False)
        # Assigning a type to the variable 'self' (line 660)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.get_values_from_key.__dict__.__setitem__('stypy_localization', localization)
        UnionType.get_values_from_key.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.get_values_from_key.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.get_values_from_key.__dict__.__setitem__('stypy_function_name', 'UnionType.get_values_from_key')
        UnionType.get_values_from_key.__dict__.__setitem__('stypy_param_names_list', ['localization', 'key'])
        UnionType.get_values_from_key.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.get_values_from_key.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.get_values_from_key.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.get_values_from_key.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.get_values_from_key.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.get_values_from_key.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.get_values_from_key', ['localization', 'key'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_values_from_key', localization, ['localization', 'key'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_values_from_key(...)' code ##################

        
        # Assigning a List to a Name (line 660):
        
        # Obtaining an instance of the builtin type 'list' (line 660)
        list_13913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 660)
        
        # Assigning a type to the variable 'errors' (line 660)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 8), 'errors', list_13913)
        
        # Assigning a Name to a Name (line 662):
        # Getting the type of 'None' (line 662)
        None_13914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 15), 'None')
        # Assigning a type to the variable 'temp' (line 662)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 8), 'temp', None_13914)
        
        # Getting the type of 'self' (line 663)
        self_13915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 21), 'self')
        # Obtaining the member 'types' of a type (line 663)
        types_13916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 21), self_13915, 'types')
        # Assigning a type to the variable 'types_13916' (line 663)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'types_13916', types_13916)
        # Testing if the for loop is going to be iterated (line 663)
        # Testing the type of a for loop iterable (line 663)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 663, 8), types_13916)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 663, 8), types_13916):
            # Getting the type of the for loop variable (line 663)
            for_loop_var_13917 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 663, 8), types_13916)
            # Assigning a type to the variable 'type_' (line 663)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'type_', for_loop_var_13917)
            # SSA begins for a for statement (line 663)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 664):
            
            # Call to get_values_from_key(...): (line 664)
            # Processing the call arguments (line 664)
            # Getting the type of 'localization' (line 664)
            localization_13920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 44), 'localization', False)
            # Getting the type of 'key' (line 664)
            key_13921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 58), 'key', False)
            # Processing the call keyword arguments (line 664)
            kwargs_13922 = {}
            # Getting the type of 'type_' (line 664)
            type__13918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 18), 'type_', False)
            # Obtaining the member 'get_values_from_key' of a type (line 664)
            get_values_from_key_13919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 18), type__13918, 'get_values_from_key')
            # Calling get_values_from_key(args, kwargs) (line 664)
            get_values_from_key_call_result_13923 = invoke(stypy.reporting.localization.Localization(__file__, 664, 18), get_values_from_key_13919, *[localization_13920, key_13921], **kwargs_13922)
            
            # Assigning a type to the variable 'res' (line 664)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 12), 'res', get_values_from_key_call_result_13923)
            
            # Type idiom detected: calculating its left and rigth part (line 665)
            # Getting the type of 'TypeError' (line 665)
            TypeError_13924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 31), 'TypeError')
            # Getting the type of 'res' (line 665)
            res_13925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 26), 'res')
            
            (may_be_13926, more_types_in_union_13927) = may_be_subtype(TypeError_13924, res_13925)

            if may_be_13926:

                if more_types_in_union_13927:
                    # Runtime conditional SSA (line 665)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'res' (line 665)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 12), 'res', remove_not_subtype_from_union(res_13925, TypeError))
                
                # Call to append(...): (line 666)
                # Processing the call arguments (line 666)
                # Getting the type of 'temp' (line 666)
                temp_13930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 30), 'temp', False)
                # Processing the call keyword arguments (line 666)
                kwargs_13931 = {}
                # Getting the type of 'errors' (line 666)
                errors_13928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 16), 'errors', False)
                # Obtaining the member 'append' of a type (line 666)
                append_13929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 16), errors_13928, 'append')
                # Calling append(args, kwargs) (line 666)
                append_call_result_13932 = invoke(stypy.reporting.localization.Localization(__file__, 666, 16), append_13929, *[temp_13930], **kwargs_13931)
                

                if more_types_in_union_13927:
                    # Runtime conditional SSA for else branch (line 665)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_13926) or more_types_in_union_13927):
                # Assigning a type to the variable 'res' (line 665)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 12), 'res', remove_subtype_from_union(res_13925, TypeError))
                
                # Assigning a Call to a Name (line 668):
                
                # Call to add(...): (line 668)
                # Processing the call arguments (line 668)
                # Getting the type of 'temp' (line 668)
                temp_13935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 37), 'temp', False)
                # Getting the type of 'res' (line 668)
                res_13936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 43), 'res', False)
                # Processing the call keyword arguments (line 668)
                kwargs_13937 = {}
                # Getting the type of 'UnionType' (line 668)
                UnionType_13933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 23), 'UnionType', False)
                # Obtaining the member 'add' of a type (line 668)
                add_13934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 23), UnionType_13933, 'add')
                # Calling add(args, kwargs) (line 668)
                add_call_result_13938 = invoke(stypy.reporting.localization.Localization(__file__, 668, 23), add_13934, *[temp_13935, res_13936], **kwargs_13937)
                
                # Assigning a type to the variable 'temp' (line 668)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 16), 'temp', add_call_result_13938)

                if (may_be_13926 and more_types_in_union_13927):
                    # SSA join for if statement (line 665)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 671)
        # Processing the call arguments (line 671)
        # Getting the type of 'errors' (line 671)
        errors_13940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 15), 'errors', False)
        # Processing the call keyword arguments (line 671)
        kwargs_13941 = {}
        # Getting the type of 'len' (line 671)
        len_13939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 11), 'len', False)
        # Calling len(args, kwargs) (line 671)
        len_call_result_13942 = invoke(stypy.reporting.localization.Localization(__file__, 671, 11), len_13939, *[errors_13940], **kwargs_13941)
        
        
        # Call to len(...): (line 671)
        # Processing the call arguments (line 671)
        # Getting the type of 'self' (line 671)
        self_13944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 671)
        types_13945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 30), self_13944, 'types')
        # Processing the call keyword arguments (line 671)
        kwargs_13946 = {}
        # Getting the type of 'len' (line 671)
        len_13943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 26), 'len', False)
        # Calling len(args, kwargs) (line 671)
        len_call_result_13947 = invoke(stypy.reporting.localization.Localization(__file__, 671, 26), len_13943, *[types_13945], **kwargs_13946)
        
        # Applying the binary operator '==' (line 671)
        result_eq_13948 = python_operator(stypy.reporting.localization.Localization(__file__, 671, 11), '==', len_call_result_13942, len_call_result_13947)
        
        # Testing if the type of an if condition is none (line 671)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 671, 8), result_eq_13948):
            
            # Getting the type of 'errors' (line 678)
            errors_13961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 25), 'errors')
            # Assigning a type to the variable 'errors_13961' (line 678)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 12), 'errors_13961', errors_13961)
            # Testing if the for loop is going to be iterated (line 678)
            # Testing the type of a for loop iterable (line 678)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 678, 12), errors_13961)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 678, 12), errors_13961):
                # Getting the type of the for loop variable (line 678)
                for_loop_var_13962 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 678, 12), errors_13961)
                # Assigning a type to the variable 'error' (line 678)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 12), 'error', for_loop_var_13962)
                # SSA begins for a for statement (line 678)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 679)
                # Processing the call keyword arguments (line 679)
                kwargs_13965 = {}
                # Getting the type of 'error' (line 679)
                error_13963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 679)
                turn_to_warning_13964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 16), error_13963, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 679)
                turn_to_warning_call_result_13966 = invoke(stypy.reporting.localization.Localization(__file__, 679, 16), turn_to_warning_13964, *[], **kwargs_13965)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 671)
            if_condition_13949 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 671, 8), result_eq_13948)
            # Assigning a type to the variable 'if_condition_13949' (line 671)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 8), 'if_condition_13949', if_condition_13949)
            # SSA begins for if statement (line 671)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 672)
            # Processing the call arguments (line 672)
            # Getting the type of 'localization' (line 672)
            localization_13951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 29), 'localization', False)
            
            # Call to format(...): (line 672)
            # Processing the call arguments (line 672)
            str_13954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 16), 'str', 'get_values_from_key')
            # Getting the type of 'self' (line 673)
            self_13955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 39), 'self', False)
            # Obtaining the member 'types' of a type (line 673)
            types_13956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 39), self_13955, 'types')
            # Processing the call keyword arguments (line 672)
            kwargs_13957 = {}
            str_13952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 43), 'str', "None of the possible types ('{1}') can invoke the member '{0}'")
            # Obtaining the member 'format' of a type (line 672)
            format_13953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 43), str_13952, 'format')
            # Calling format(args, kwargs) (line 672)
            format_call_result_13958 = invoke(stypy.reporting.localization.Localization(__file__, 672, 43), format_13953, *[str_13954, types_13956], **kwargs_13957)
            
            # Processing the call keyword arguments (line 672)
            kwargs_13959 = {}
            # Getting the type of 'TypeError' (line 672)
            TypeError_13950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 672)
            TypeError_call_result_13960 = invoke(stypy.reporting.localization.Localization(__file__, 672, 19), TypeError_13950, *[localization_13951, format_call_result_13958], **kwargs_13959)
            
            # Assigning a type to the variable 'stypy_return_type' (line 672)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 12), 'stypy_return_type', TypeError_call_result_13960)
            # SSA branch for the else part of an if statement (line 671)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 678)
            errors_13961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 25), 'errors')
            # Assigning a type to the variable 'errors_13961' (line 678)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 12), 'errors_13961', errors_13961)
            # Testing if the for loop is going to be iterated (line 678)
            # Testing the type of a for loop iterable (line 678)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 678, 12), errors_13961)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 678, 12), errors_13961):
                # Getting the type of the for loop variable (line 678)
                for_loop_var_13962 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 678, 12), errors_13961)
                # Assigning a type to the variable 'error' (line 678)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 12), 'error', for_loop_var_13962)
                # SSA begins for a for statement (line 678)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 679)
                # Processing the call keyword arguments (line 679)
                kwargs_13965 = {}
                # Getting the type of 'error' (line 679)
                error_13963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 679)
                turn_to_warning_13964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 16), error_13963, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 679)
                turn_to_warning_call_result_13966 = invoke(stypy.reporting.localization.Localization(__file__, 679, 16), turn_to_warning_13964, *[], **kwargs_13965)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 671)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'temp' (line 681)
        temp_13967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 681)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 8), 'stypy_return_type', temp_13967)
        
        # ################# End of 'get_values_from_key(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_values_from_key' in the type store
        # Getting the type of 'stypy_return_type' (line 659)
        stypy_return_type_13968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13968)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_values_from_key'
        return stypy_return_type_13968


    @norecursion
    def add_key_and_value_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 683)
        True_13969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 81), 'True')
        defaults = [True_13969]
        # Create a new context for function 'add_key_and_value_type'
        module_type_store = module_type_store.open_function_context('add_key_and_value_type', 683, 4, False)
        # Assigning a type to the variable 'self' (line 684)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnionType.add_key_and_value_type.__dict__.__setitem__('stypy_localization', localization)
        UnionType.add_key_and_value_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnionType.add_key_and_value_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnionType.add_key_and_value_type.__dict__.__setitem__('stypy_function_name', 'UnionType.add_key_and_value_type')
        UnionType.add_key_and_value_type.__dict__.__setitem__('stypy_param_names_list', ['localization', 'type_tuple', 'record_annotation'])
        UnionType.add_key_and_value_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnionType.add_key_and_value_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnionType.add_key_and_value_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnionType.add_key_and_value_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnionType.add_key_and_value_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnionType.add_key_and_value_type.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnionType.add_key_and_value_type', ['localization', 'type_tuple', 'record_annotation'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_key_and_value_type', localization, ['localization', 'type_tuple', 'record_annotation'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_key_and_value_type(...)' code ##################

        
        # Assigning a List to a Name (line 684):
        
        # Obtaining an instance of the builtin type 'list' (line 684)
        list_13970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 684)
        
        # Assigning a type to the variable 'errors' (line 684)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 8), 'errors', list_13970)
        
        # Getting the type of 'self' (line 686)
        self_13971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 21), 'self')
        # Obtaining the member 'types' of a type (line 686)
        types_13972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 21), self_13971, 'types')
        # Assigning a type to the variable 'types_13972' (line 686)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 8), 'types_13972', types_13972)
        # Testing if the for loop is going to be iterated (line 686)
        # Testing the type of a for loop iterable (line 686)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 686, 8), types_13972)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 686, 8), types_13972):
            # Getting the type of the for loop variable (line 686)
            for_loop_var_13973 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 686, 8), types_13972)
            # Assigning a type to the variable 'type_' (line 686)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 8), 'type_', for_loop_var_13973)
            # SSA begins for a for statement (line 686)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 687):
            
            # Call to add_key_and_value_type(...): (line 687)
            # Processing the call arguments (line 687)
            # Getting the type of 'localization' (line 687)
            localization_13976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 48), 'localization', False)
            # Getting the type of 'type_tuple' (line 687)
            type_tuple_13977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 62), 'type_tuple', False)
            # Getting the type of 'record_annotation' (line 687)
            record_annotation_13978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 74), 'record_annotation', False)
            # Processing the call keyword arguments (line 687)
            kwargs_13979 = {}
            # Getting the type of 'type_' (line 687)
            type__13974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 19), 'type_', False)
            # Obtaining the member 'add_key_and_value_type' of a type (line 687)
            add_key_and_value_type_13975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 19), type__13974, 'add_key_and_value_type')
            # Calling add_key_and_value_type(args, kwargs) (line 687)
            add_key_and_value_type_call_result_13980 = invoke(stypy.reporting.localization.Localization(__file__, 687, 19), add_key_and_value_type_13975, *[localization_13976, type_tuple_13977, record_annotation_13978], **kwargs_13979)
            
            # Assigning a type to the variable 'temp' (line 687)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 12), 'temp', add_key_and_value_type_call_result_13980)
            
            # Type idiom detected: calculating its left and rigth part (line 688)
            # Getting the type of 'temp' (line 688)
            temp_13981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 12), 'temp')
            # Getting the type of 'None' (line 688)
            None_13982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 27), 'None')
            
            (may_be_13983, more_types_in_union_13984) = may_not_be_none(temp_13981, None_13982)

            if may_be_13983:

                if more_types_in_union_13984:
                    # Runtime conditional SSA (line 688)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to append(...): (line 689)
                # Processing the call arguments (line 689)
                # Getting the type of 'temp' (line 689)
                temp_13987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 30), 'temp', False)
                # Processing the call keyword arguments (line 689)
                kwargs_13988 = {}
                # Getting the type of 'errors' (line 689)
                errors_13985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 16), 'errors', False)
                # Obtaining the member 'append' of a type (line 689)
                append_13986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 16), errors_13985, 'append')
                # Calling append(args, kwargs) (line 689)
                append_call_result_13989 = invoke(stypy.reporting.localization.Localization(__file__, 689, 16), append_13986, *[temp_13987], **kwargs_13988)
                

                if more_types_in_union_13984:
                    # SSA join for if statement (line 688)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 692)
        # Processing the call arguments (line 692)
        # Getting the type of 'errors' (line 692)
        errors_13991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 15), 'errors', False)
        # Processing the call keyword arguments (line 692)
        kwargs_13992 = {}
        # Getting the type of 'len' (line 692)
        len_13990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 11), 'len', False)
        # Calling len(args, kwargs) (line 692)
        len_call_result_13993 = invoke(stypy.reporting.localization.Localization(__file__, 692, 11), len_13990, *[errors_13991], **kwargs_13992)
        
        
        # Call to len(...): (line 692)
        # Processing the call arguments (line 692)
        # Getting the type of 'self' (line 692)
        self_13995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 692)
        types_13996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 30), self_13995, 'types')
        # Processing the call keyword arguments (line 692)
        kwargs_13997 = {}
        # Getting the type of 'len' (line 692)
        len_13994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 26), 'len', False)
        # Calling len(args, kwargs) (line 692)
        len_call_result_13998 = invoke(stypy.reporting.localization.Localization(__file__, 692, 26), len_13994, *[types_13996], **kwargs_13997)
        
        # Applying the binary operator '==' (line 692)
        result_eq_13999 = python_operator(stypy.reporting.localization.Localization(__file__, 692, 11), '==', len_call_result_13993, len_call_result_13998)
        
        # Testing if the type of an if condition is none (line 692)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 692, 8), result_eq_13999):
            
            # Getting the type of 'errors' (line 699)
            errors_14012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 25), 'errors')
            # Assigning a type to the variable 'errors_14012' (line 699)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 12), 'errors_14012', errors_14012)
            # Testing if the for loop is going to be iterated (line 699)
            # Testing the type of a for loop iterable (line 699)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 699, 12), errors_14012)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 699, 12), errors_14012):
                # Getting the type of the for loop variable (line 699)
                for_loop_var_14013 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 699, 12), errors_14012)
                # Assigning a type to the variable 'error' (line 699)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 12), 'error', for_loop_var_14013)
                # SSA begins for a for statement (line 699)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 700)
                # Processing the call keyword arguments (line 700)
                kwargs_14016 = {}
                # Getting the type of 'error' (line 700)
                error_14014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 700)
                turn_to_warning_14015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 16), error_14014, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 700)
                turn_to_warning_call_result_14017 = invoke(stypy.reporting.localization.Localization(__file__, 700, 16), turn_to_warning_14015, *[], **kwargs_14016)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 692)
            if_condition_14000 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 692, 8), result_eq_13999)
            # Assigning a type to the variable 'if_condition_14000' (line 692)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 8), 'if_condition_14000', if_condition_14000)
            # SSA begins for if statement (line 692)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 693)
            # Processing the call arguments (line 693)
            # Getting the type of 'localization' (line 693)
            localization_14002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 29), 'localization', False)
            
            # Call to format(...): (line 693)
            # Processing the call arguments (line 693)
            str_14005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 16), 'str', 'add_key_and_value_type')
            # Getting the type of 'self' (line 694)
            self_14006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 42), 'self', False)
            # Obtaining the member 'types' of a type (line 694)
            types_14007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 42), self_14006, 'types')
            # Processing the call keyword arguments (line 693)
            kwargs_14008 = {}
            str_14003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 43), 'str', "None of the possible types ('{1}') can invoke the member '{0}'")
            # Obtaining the member 'format' of a type (line 693)
            format_14004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 43), str_14003, 'format')
            # Calling format(args, kwargs) (line 693)
            format_call_result_14009 = invoke(stypy.reporting.localization.Localization(__file__, 693, 43), format_14004, *[str_14005, types_14007], **kwargs_14008)
            
            # Processing the call keyword arguments (line 693)
            kwargs_14010 = {}
            # Getting the type of 'TypeError' (line 693)
            TypeError_14001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 693)
            TypeError_call_result_14011 = invoke(stypy.reporting.localization.Localization(__file__, 693, 19), TypeError_14001, *[localization_14002, format_call_result_14009], **kwargs_14010)
            
            # Assigning a type to the variable 'stypy_return_type' (line 693)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 12), 'stypy_return_type', TypeError_call_result_14011)
            # SSA branch for the else part of an if statement (line 692)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 699)
            errors_14012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 25), 'errors')
            # Assigning a type to the variable 'errors_14012' (line 699)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 12), 'errors_14012', errors_14012)
            # Testing if the for loop is going to be iterated (line 699)
            # Testing the type of a for loop iterable (line 699)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 699, 12), errors_14012)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 699, 12), errors_14012):
                # Getting the type of the for loop variable (line 699)
                for_loop_var_14013 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 699, 12), errors_14012)
                # Assigning a type to the variable 'error' (line 699)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 12), 'error', for_loop_var_14013)
                # SSA begins for a for statement (line 699)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 700)
                # Processing the call keyword arguments (line 700)
                kwargs_14016 = {}
                # Getting the type of 'error' (line 700)
                error_14014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 700)
                turn_to_warning_14015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 16), error_14014, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 700)
                turn_to_warning_call_result_14017 = invoke(stypy.reporting.localization.Localization(__file__, 700, 16), turn_to_warning_14015, *[], **kwargs_14016)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 692)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'None' (line 702)
        None_14018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'stypy_return_type', None_14018)
        
        # ################# End of 'add_key_and_value_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_key_and_value_type' in the type store
        # Getting the type of 'stypy_return_type' (line 683)
        stypy_return_type_14019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14019)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_key_and_value_type'
        return stypy_return_type_14019


# Assigning a type to the variable 'UnionType' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'UnionType', UnionType)
# Declaration of the 'OrderedUnionType' class
# Getting the type of 'UnionType' (line 705)
UnionType_14020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 23), 'UnionType')

class OrderedUnionType(UnionType_14020, ):
    str_14021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, (-1)), 'str', '\n    A special type of UnionType that maintain the order of its added types and admits repeated elements. This will be\n    used in the future implementation of tuples.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 711)
        None_14022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 29), 'None')
        # Getting the type of 'None' (line 711)
        None_14023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 41), 'None')
        defaults = [None_14022, None_14023]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 711, 4, False)
        # Assigning a type to the variable 'self' (line 712)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'OrderedUnionType.__init__', ['type1', 'type2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['type1', 'type2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 712)
        # Processing the call arguments (line 712)
        # Getting the type of 'self' (line 712)
        self_14026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 27), 'self', False)
        # Getting the type of 'type1' (line 712)
        type1_14027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 33), 'type1', False)
        # Getting the type of 'type2' (line 712)
        type2_14028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 40), 'type2', False)
        # Processing the call keyword arguments (line 712)
        kwargs_14029 = {}
        # Getting the type of 'UnionType' (line 712)
        UnionType_14024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'UnionType', False)
        # Obtaining the member '__init__' of a type (line 712)
        init___14025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 8), UnionType_14024, '__init__')
        # Calling __init__(args, kwargs) (line 712)
        init___call_result_14030 = invoke(stypy.reporting.localization.Localization(__file__, 712, 8), init___14025, *[self_14026, type1_14027, type2_14028], **kwargs_14029)
        
        
        # Assigning a List to a Attribute (line 713):
        
        # Obtaining an instance of the builtin type 'list' (line 713)
        list_14031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 713)
        
        # Getting the type of 'self' (line 713)
        self_14032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 8), 'self')
        # Setting the type of the member 'ordered_types' of a type (line 713)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 8), self_14032, 'ordered_types', list_14031)
        
        # Type idiom detected: calculating its left and rigth part (line 715)
        # Getting the type of 'type1' (line 715)
        type1_14033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 8), 'type1')
        # Getting the type of 'None' (line 715)
        None_14034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 24), 'None')
        
        (may_be_14035, more_types_in_union_14036) = may_not_be_none(type1_14033, None_14034)

        if may_be_14035:

            if more_types_in_union_14036:
                # Runtime conditional SSA (line 715)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to append(...): (line 716)
            # Processing the call arguments (line 716)
            # Getting the type of 'type1' (line 716)
            type1_14040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 38), 'type1', False)
            # Processing the call keyword arguments (line 716)
            kwargs_14041 = {}
            # Getting the type of 'self' (line 716)
            self_14037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 12), 'self', False)
            # Obtaining the member 'ordered_types' of a type (line 716)
            ordered_types_14038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 12), self_14037, 'ordered_types')
            # Obtaining the member 'append' of a type (line 716)
            append_14039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 12), ordered_types_14038, 'append')
            # Calling append(args, kwargs) (line 716)
            append_call_result_14042 = invoke(stypy.reporting.localization.Localization(__file__, 716, 12), append_14039, *[type1_14040], **kwargs_14041)
            

            if more_types_in_union_14036:
                # SSA join for if statement (line 715)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 718)
        # Getting the type of 'type2' (line 718)
        type2_14043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'type2')
        # Getting the type of 'None' (line 718)
        None_14044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 24), 'None')
        
        (may_be_14045, more_types_in_union_14046) = may_not_be_none(type2_14043, None_14044)

        if may_be_14045:

            if more_types_in_union_14046:
                # Runtime conditional SSA (line 718)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to append(...): (line 719)
            # Processing the call arguments (line 719)
            # Getting the type of 'type2' (line 719)
            type2_14050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 38), 'type2', False)
            # Processing the call keyword arguments (line 719)
            kwargs_14051 = {}
            # Getting the type of 'self' (line 719)
            self_14047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 12), 'self', False)
            # Obtaining the member 'ordered_types' of a type (line 719)
            ordered_types_14048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 12), self_14047, 'ordered_types')
            # Obtaining the member 'append' of a type (line 719)
            append_14049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 12), ordered_types_14048, 'append')
            # Calling append(args, kwargs) (line 719)
            append_call_result_14052 = invoke(stypy.reporting.localization.Localization(__file__, 719, 12), append_14049, *[type2_14050], **kwargs_14051)
            

            if more_types_in_union_14046:
                # SSA join for if statement (line 718)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @staticmethod
    @norecursion
    def add(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add'
        module_type_store = module_type_store.open_function_context('add', 721, 4, False)
        
        # Passed parameters checking function
        OrderedUnionType.add.__dict__.__setitem__('stypy_localization', localization)
        OrderedUnionType.add.__dict__.__setitem__('stypy_type_of_self', None)
        OrderedUnionType.add.__dict__.__setitem__('stypy_type_store', module_type_store)
        OrderedUnionType.add.__dict__.__setitem__('stypy_function_name', 'add')
        OrderedUnionType.add.__dict__.__setitem__('stypy_param_names_list', ['type1', 'type2'])
        OrderedUnionType.add.__dict__.__setitem__('stypy_varargs_param_name', None)
        OrderedUnionType.add.__dict__.__setitem__('stypy_kwargs_param_name', None)
        OrderedUnionType.add.__dict__.__setitem__('stypy_call_defaults', defaults)
        OrderedUnionType.add.__dict__.__setitem__('stypy_call_varargs', varargs)
        OrderedUnionType.add.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        OrderedUnionType.add.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, 'add', ['type1', 'type2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add', localization, ['type2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 723)
        # Getting the type of 'type1' (line 723)
        type1_14053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 11), 'type1')
        # Getting the type of 'None' (line 723)
        None_14054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 20), 'None')
        
        (may_be_14055, more_types_in_union_14056) = may_be_none(type1_14053, None_14054)

        if may_be_14055:

            if more_types_in_union_14056:
                # Runtime conditional SSA (line 723)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to _wrap_type(...): (line 724)
            # Processing the call arguments (line 724)
            # Getting the type of 'type2' (line 724)
            type2_14059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 40), 'type2', False)
            # Processing the call keyword arguments (line 724)
            kwargs_14060 = {}
            # Getting the type of 'UnionType' (line 724)
            UnionType_14057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 19), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 724)
            _wrap_type_14058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 19), UnionType_14057, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 724)
            _wrap_type_call_result_14061 = invoke(stypy.reporting.localization.Localization(__file__, 724, 19), _wrap_type_14058, *[type2_14059], **kwargs_14060)
            
            # Assigning a type to the variable 'stypy_return_type' (line 724)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 12), 'stypy_return_type', _wrap_type_call_result_14061)

            if more_types_in_union_14056:
                # SSA join for if statement (line 723)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'type1' (line 723)
        type1_14062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'type1')
        # Assigning a type to the variable 'type1' (line 723)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'type1', remove_type_from_union(type1_14062, types.NoneType))
        
        # Type idiom detected: calculating its left and rigth part (line 726)
        # Getting the type of 'type2' (line 726)
        type2_14063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 11), 'type2')
        # Getting the type of 'None' (line 726)
        None_14064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 20), 'None')
        
        (may_be_14065, more_types_in_union_14066) = may_be_none(type2_14063, None_14064)

        if may_be_14065:

            if more_types_in_union_14066:
                # Runtime conditional SSA (line 726)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to _wrap_type(...): (line 727)
            # Processing the call arguments (line 727)
            # Getting the type of 'type1' (line 727)
            type1_14069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 40), 'type1', False)
            # Processing the call keyword arguments (line 727)
            kwargs_14070 = {}
            # Getting the type of 'UnionType' (line 727)
            UnionType_14067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 19), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 727)
            _wrap_type_14068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 19), UnionType_14067, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 727)
            _wrap_type_call_result_14071 = invoke(stypy.reporting.localization.Localization(__file__, 727, 19), _wrap_type_14068, *[type1_14069], **kwargs_14070)
            
            # Assigning a type to the variable 'stypy_return_type' (line 727)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 12), 'stypy_return_type', _wrap_type_call_result_14071)

            if more_types_in_union_14066:
                # SSA join for if statement (line 726)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'type2' (line 726)
        type2_14072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 8), 'type2')
        # Assigning a type to the variable 'type2' (line 726)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 8), 'type2', remove_type_from_union(type2_14072, types.NoneType))
        
        # Call to is_undefined_type(...): (line 729)
        # Processing the call arguments (line 729)
        # Getting the type of 'type1' (line 729)
        type1_14079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 107), 'type1', False)
        # Processing the call keyword arguments (line 729)
        kwargs_14080 = {}
        # Getting the type of 'stypy_copy' (line 729)
        stypy_copy_14073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 729)
        python_lib_14074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 11), stypy_copy_14073, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 729)
        python_types_14075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 11), python_lib_14074, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 729)
        type_introspection_14076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 11), python_types_14075, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 729)
        runtime_type_inspection_14077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 11), type_introspection_14076, 'runtime_type_inspection')
        # Obtaining the member 'is_undefined_type' of a type (line 729)
        is_undefined_type_14078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 11), runtime_type_inspection_14077, 'is_undefined_type')
        # Calling is_undefined_type(args, kwargs) (line 729)
        is_undefined_type_call_result_14081 = invoke(stypy.reporting.localization.Localization(__file__, 729, 11), is_undefined_type_14078, *[type1_14079], **kwargs_14080)
        
        # Testing if the type of an if condition is none (line 729)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 729, 8), is_undefined_type_call_result_14081):
            pass
        else:
            
            # Testing the type of an if condition (line 729)
            if_condition_14082 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 729, 8), is_undefined_type_call_result_14081)
            # Assigning a type to the variable 'if_condition_14082' (line 729)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 8), 'if_condition_14082', if_condition_14082)
            # SSA begins for if statement (line 729)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _wrap_type(...): (line 730)
            # Processing the call arguments (line 730)
            # Getting the type of 'type2' (line 730)
            type2_14085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 40), 'type2', False)
            # Processing the call keyword arguments (line 730)
            kwargs_14086 = {}
            # Getting the type of 'UnionType' (line 730)
            UnionType_14083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 19), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 730)
            _wrap_type_14084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 19), UnionType_14083, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 730)
            _wrap_type_call_result_14087 = invoke(stypy.reporting.localization.Localization(__file__, 730, 19), _wrap_type_14084, *[type2_14085], **kwargs_14086)
            
            # Assigning a type to the variable 'stypy_return_type' (line 730)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 12), 'stypy_return_type', _wrap_type_call_result_14087)
            # SSA join for if statement (line 729)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to is_undefined_type(...): (line 731)
        # Processing the call arguments (line 731)
        # Getting the type of 'type2' (line 731)
        type2_14094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 107), 'type2', False)
        # Processing the call keyword arguments (line 731)
        kwargs_14095 = {}
        # Getting the type of 'stypy_copy' (line 731)
        stypy_copy_14088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 731)
        python_lib_14089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 11), stypy_copy_14088, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 731)
        python_types_14090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 11), python_lib_14089, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 731)
        type_introspection_14091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 11), python_types_14090, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 731)
        runtime_type_inspection_14092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 11), type_introspection_14091, 'runtime_type_inspection')
        # Obtaining the member 'is_undefined_type' of a type (line 731)
        is_undefined_type_14093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 11), runtime_type_inspection_14092, 'is_undefined_type')
        # Calling is_undefined_type(args, kwargs) (line 731)
        is_undefined_type_call_result_14096 = invoke(stypy.reporting.localization.Localization(__file__, 731, 11), is_undefined_type_14093, *[type2_14094], **kwargs_14095)
        
        # Testing if the type of an if condition is none (line 731)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 731, 8), is_undefined_type_call_result_14096):
            pass
        else:
            
            # Testing the type of an if condition (line 731)
            if_condition_14097 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 731, 8), is_undefined_type_call_result_14096)
            # Assigning a type to the variable 'if_condition_14097' (line 731)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 8), 'if_condition_14097', if_condition_14097)
            # SSA begins for if statement (line 731)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _wrap_type(...): (line 732)
            # Processing the call arguments (line 732)
            # Getting the type of 'type1' (line 732)
            type1_14100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 40), 'type1', False)
            # Processing the call keyword arguments (line 732)
            kwargs_14101 = {}
            # Getting the type of 'UnionType' (line 732)
            UnionType_14098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 19), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 732)
            _wrap_type_14099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 732, 19), UnionType_14098, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 732)
            _wrap_type_call_result_14102 = invoke(stypy.reporting.localization.Localization(__file__, 732, 19), _wrap_type_14099, *[type1_14100], **kwargs_14101)
            
            # Assigning a type to the variable 'stypy_return_type' (line 732)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 12), 'stypy_return_type', _wrap_type_call_result_14102)
            # SSA join for if statement (line 731)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to is_union_type(...): (line 734)
        # Processing the call arguments (line 734)
        # Getting the type of 'type1' (line 734)
        type1_14109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 103), 'type1', False)
        # Processing the call keyword arguments (line 734)
        kwargs_14110 = {}
        # Getting the type of 'stypy_copy' (line 734)
        stypy_copy_14103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 734)
        python_lib_14104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 11), stypy_copy_14103, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 734)
        python_types_14105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 11), python_lib_14104, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 734)
        type_introspection_14106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 11), python_types_14105, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 734)
        runtime_type_inspection_14107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 11), type_introspection_14106, 'runtime_type_inspection')
        # Obtaining the member 'is_union_type' of a type (line 734)
        is_union_type_14108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 11), runtime_type_inspection_14107, 'is_union_type')
        # Calling is_union_type(args, kwargs) (line 734)
        is_union_type_call_result_14111 = invoke(stypy.reporting.localization.Localization(__file__, 734, 11), is_union_type_14108, *[type1_14109], **kwargs_14110)
        
        # Testing if the type of an if condition is none (line 734)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 734, 8), is_union_type_call_result_14111):
            pass
        else:
            
            # Testing the type of an if condition (line 734)
            if_condition_14112 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 734, 8), is_union_type_call_result_14111)
            # Assigning a type to the variable 'if_condition_14112' (line 734)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 8), 'if_condition_14112', if_condition_14112)
            # SSA begins for if statement (line 734)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _add(...): (line 735)
            # Processing the call arguments (line 735)
            # Getting the type of 'type2' (line 735)
            type2_14115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 30), 'type2', False)
            # Processing the call keyword arguments (line 735)
            kwargs_14116 = {}
            # Getting the type of 'type1' (line 735)
            type1_14113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 19), 'type1', False)
            # Obtaining the member '_add' of a type (line 735)
            _add_14114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 19), type1_14113, '_add')
            # Calling _add(args, kwargs) (line 735)
            _add_call_result_14117 = invoke(stypy.reporting.localization.Localization(__file__, 735, 19), _add_14114, *[type2_14115], **kwargs_14116)
            
            # Assigning a type to the variable 'stypy_return_type' (line 735)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 12), 'stypy_return_type', _add_call_result_14117)
            # SSA join for if statement (line 734)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to is_union_type(...): (line 736)
        # Processing the call arguments (line 736)
        # Getting the type of 'type2' (line 736)
        type2_14124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 103), 'type2', False)
        # Processing the call keyword arguments (line 736)
        kwargs_14125 = {}
        # Getting the type of 'stypy_copy' (line 736)
        stypy_copy_14118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 736)
        python_lib_14119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 11), stypy_copy_14118, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 736)
        python_types_14120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 11), python_lib_14119, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 736)
        type_introspection_14121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 11), python_types_14120, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 736)
        runtime_type_inspection_14122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 11), type_introspection_14121, 'runtime_type_inspection')
        # Obtaining the member 'is_union_type' of a type (line 736)
        is_union_type_14123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 11), runtime_type_inspection_14122, 'is_union_type')
        # Calling is_union_type(args, kwargs) (line 736)
        is_union_type_call_result_14126 = invoke(stypy.reporting.localization.Localization(__file__, 736, 11), is_union_type_14123, *[type2_14124], **kwargs_14125)
        
        # Testing if the type of an if condition is none (line 736)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 736, 8), is_union_type_call_result_14126):
            pass
        else:
            
            # Testing the type of an if condition (line 736)
            if_condition_14127 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 736, 8), is_union_type_call_result_14126)
            # Assigning a type to the variable 'if_condition_14127' (line 736)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 8), 'if_condition_14127', if_condition_14127)
            # SSA begins for if statement (line 736)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _add(...): (line 737)
            # Processing the call arguments (line 737)
            # Getting the type of 'type1' (line 737)
            type1_14130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 30), 'type1', False)
            # Processing the call keyword arguments (line 737)
            kwargs_14131 = {}
            # Getting the type of 'type2' (line 737)
            type2_14128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 19), 'type2', False)
            # Obtaining the member '_add' of a type (line 737)
            _add_14129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 19), type2_14128, '_add')
            # Calling _add(args, kwargs) (line 737)
            _add_call_result_14132 = invoke(stypy.reporting.localization.Localization(__file__, 737, 19), _add_14129, *[type1_14130], **kwargs_14131)
            
            # Assigning a type to the variable 'stypy_return_type' (line 737)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 12), 'stypy_return_type', _add_call_result_14132)
            # SSA join for if statement (line 736)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to _wrap_type(...): (line 739)
        # Processing the call arguments (line 739)
        # Getting the type of 'type1' (line 739)
        type1_14135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 32), 'type1', False)
        # Processing the call keyword arguments (line 739)
        kwargs_14136 = {}
        # Getting the type of 'UnionType' (line 739)
        UnionType_14133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 11), 'UnionType', False)
        # Obtaining the member '_wrap_type' of a type (line 739)
        _wrap_type_14134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 11), UnionType_14133, '_wrap_type')
        # Calling _wrap_type(args, kwargs) (line 739)
        _wrap_type_call_result_14137 = invoke(stypy.reporting.localization.Localization(__file__, 739, 11), _wrap_type_14134, *[type1_14135], **kwargs_14136)
        
        
        # Call to _wrap_type(...): (line 739)
        # Processing the call arguments (line 739)
        # Getting the type of 'type2' (line 739)
        type2_14140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 63), 'type2', False)
        # Processing the call keyword arguments (line 739)
        kwargs_14141 = {}
        # Getting the type of 'UnionType' (line 739)
        UnionType_14138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 42), 'UnionType', False)
        # Obtaining the member '_wrap_type' of a type (line 739)
        _wrap_type_14139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 42), UnionType_14138, '_wrap_type')
        # Calling _wrap_type(args, kwargs) (line 739)
        _wrap_type_call_result_14142 = invoke(stypy.reporting.localization.Localization(__file__, 739, 42), _wrap_type_14139, *[type2_14140], **kwargs_14141)
        
        # Applying the binary operator '==' (line 739)
        result_eq_14143 = python_operator(stypy.reporting.localization.Localization(__file__, 739, 11), '==', _wrap_type_call_result_14137, _wrap_type_call_result_14142)
        
        # Testing if the type of an if condition is none (line 739)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 739, 8), result_eq_14143):
            pass
        else:
            
            # Testing the type of an if condition (line 739)
            if_condition_14144 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 739, 8), result_eq_14143)
            # Assigning a type to the variable 'if_condition_14144' (line 739)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 8), 'if_condition_14144', if_condition_14144)
            # SSA begins for if statement (line 739)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _wrap_type(...): (line 740)
            # Processing the call arguments (line 740)
            # Getting the type of 'type1' (line 740)
            type1_14147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 40), 'type1', False)
            # Processing the call keyword arguments (line 740)
            kwargs_14148 = {}
            # Getting the type of 'UnionType' (line 740)
            UnionType_14145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 19), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 740)
            _wrap_type_14146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 740, 19), UnionType_14145, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 740)
            _wrap_type_call_result_14149 = invoke(stypy.reporting.localization.Localization(__file__, 740, 19), _wrap_type_14146, *[type1_14147], **kwargs_14148)
            
            # Assigning a type to the variable 'stypy_return_type' (line 740)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 12), 'stypy_return_type', _wrap_type_call_result_14149)
            # SSA join for if statement (line 739)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to OrderedUnionType(...): (line 742)
        # Processing the call arguments (line 742)
        # Getting the type of 'type1' (line 742)
        type1_14151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 32), 'type1', False)
        # Getting the type of 'type2' (line 742)
        type2_14152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 39), 'type2', False)
        # Processing the call keyword arguments (line 742)
        kwargs_14153 = {}
        # Getting the type of 'OrderedUnionType' (line 742)
        OrderedUnionType_14150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 15), 'OrderedUnionType', False)
        # Calling OrderedUnionType(args, kwargs) (line 742)
        OrderedUnionType_call_result_14154 = invoke(stypy.reporting.localization.Localization(__file__, 742, 15), OrderedUnionType_14150, *[type1_14151, type2_14152], **kwargs_14153)
        
        # Assigning a type to the variable 'stypy_return_type' (line 742)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'stypy_return_type', OrderedUnionType_call_result_14154)
        
        # ################# End of 'add(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add' in the type store
        # Getting the type of 'stypy_return_type' (line 721)
        stypy_return_type_14155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14155)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add'
        return stypy_return_type_14155


    @norecursion
    def _add(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_add'
        module_type_store = module_type_store.open_function_context('_add', 744, 4, False)
        # Assigning a type to the variable 'self' (line 745)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        OrderedUnionType._add.__dict__.__setitem__('stypy_localization', localization)
        OrderedUnionType._add.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        OrderedUnionType._add.__dict__.__setitem__('stypy_type_store', module_type_store)
        OrderedUnionType._add.__dict__.__setitem__('stypy_function_name', 'OrderedUnionType._add')
        OrderedUnionType._add.__dict__.__setitem__('stypy_param_names_list', ['other_type'])
        OrderedUnionType._add.__dict__.__setitem__('stypy_varargs_param_name', None)
        OrderedUnionType._add.__dict__.__setitem__('stypy_kwargs_param_name', None)
        OrderedUnionType._add.__dict__.__setitem__('stypy_call_defaults', defaults)
        OrderedUnionType._add.__dict__.__setitem__('stypy_call_varargs', varargs)
        OrderedUnionType._add.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        OrderedUnionType._add.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'OrderedUnionType._add', ['other_type'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_add', localization, ['other_type'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_add(...)' code ##################

        
        # Assigning a Call to a Name (line 745):
        
        # Call to _add(...): (line 745)
        # Processing the call arguments (line 745)
        # Getting the type of 'self' (line 745)
        self_14158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 29), 'self', False)
        # Getting the type of 'other_type' (line 745)
        other_type_14159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 35), 'other_type', False)
        # Processing the call keyword arguments (line 745)
        kwargs_14160 = {}
        # Getting the type of 'UnionType' (line 745)
        UnionType_14156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 14), 'UnionType', False)
        # Obtaining the member '_add' of a type (line 745)
        _add_14157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 14), UnionType_14156, '_add')
        # Calling _add(args, kwargs) (line 745)
        _add_call_result_14161 = invoke(stypy.reporting.localization.Localization(__file__, 745, 14), _add_14157, *[self_14158, other_type_14159], **kwargs_14160)
        
        # Assigning a type to the variable 'ret' (line 745)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'ret', _add_call_result_14161)
        
        # Call to append(...): (line 746)
        # Processing the call arguments (line 746)
        # Getting the type of 'other_type' (line 746)
        other_type_14165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 34), 'other_type', False)
        # Processing the call keyword arguments (line 746)
        kwargs_14166 = {}
        # Getting the type of 'self' (line 746)
        self_14162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 8), 'self', False)
        # Obtaining the member 'ordered_types' of a type (line 746)
        ordered_types_14163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 8), self_14162, 'ordered_types')
        # Obtaining the member 'append' of a type (line 746)
        append_14164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 8), ordered_types_14163, 'append')
        # Calling append(args, kwargs) (line 746)
        append_call_result_14167 = invoke(stypy.reporting.localization.Localization(__file__, 746, 8), append_14164, *[other_type_14165], **kwargs_14166)
        
        # Getting the type of 'ret' (line 747)
        ret_14168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 15), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 747)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 8), 'stypy_return_type', ret_14168)
        
        # ################# End of '_add(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_add' in the type store
        # Getting the type of 'stypy_return_type' (line 744)
        stypy_return_type_14169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14169)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_add'
        return stypy_return_type_14169


    @norecursion
    def get_ordered_types(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_ordered_types'
        module_type_store = module_type_store.open_function_context('get_ordered_types', 749, 4, False)
        # Assigning a type to the variable 'self' (line 750)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        OrderedUnionType.get_ordered_types.__dict__.__setitem__('stypy_localization', localization)
        OrderedUnionType.get_ordered_types.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        OrderedUnionType.get_ordered_types.__dict__.__setitem__('stypy_type_store', module_type_store)
        OrderedUnionType.get_ordered_types.__dict__.__setitem__('stypy_function_name', 'OrderedUnionType.get_ordered_types')
        OrderedUnionType.get_ordered_types.__dict__.__setitem__('stypy_param_names_list', [])
        OrderedUnionType.get_ordered_types.__dict__.__setitem__('stypy_varargs_param_name', None)
        OrderedUnionType.get_ordered_types.__dict__.__setitem__('stypy_kwargs_param_name', None)
        OrderedUnionType.get_ordered_types.__dict__.__setitem__('stypy_call_defaults', defaults)
        OrderedUnionType.get_ordered_types.__dict__.__setitem__('stypy_call_varargs', varargs)
        OrderedUnionType.get_ordered_types.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        OrderedUnionType.get_ordered_types.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'OrderedUnionType.get_ordered_types', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_ordered_types', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_ordered_types(...)' code ##################

        str_14170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, (-1)), 'str', '\n        Obtain the stored types in the same order they were added, including repetitions\n        :return:\n        ')
        # Getting the type of 'self' (line 754)
        self_14171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 15), 'self')
        # Obtaining the member 'ordered_types' of a type (line 754)
        ordered_types_14172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 754, 15), self_14171, 'ordered_types')
        # Assigning a type to the variable 'stypy_return_type' (line 754)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 8), 'stypy_return_type', ordered_types_14172)
        
        # ################# End of 'get_ordered_types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_ordered_types' in the type store
        # Getting the type of 'stypy_return_type' (line 749)
        stypy_return_type_14173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14173)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_ordered_types'
        return stypy_return_type_14173


    @norecursion
    def clone(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'clone'
        module_type_store = module_type_store.open_function_context('clone', 756, 4, False)
        # Assigning a type to the variable 'self' (line 757)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 757, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        OrderedUnionType.clone.__dict__.__setitem__('stypy_localization', localization)
        OrderedUnionType.clone.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        OrderedUnionType.clone.__dict__.__setitem__('stypy_type_store', module_type_store)
        OrderedUnionType.clone.__dict__.__setitem__('stypy_function_name', 'OrderedUnionType.clone')
        OrderedUnionType.clone.__dict__.__setitem__('stypy_param_names_list', [])
        OrderedUnionType.clone.__dict__.__setitem__('stypy_varargs_param_name', None)
        OrderedUnionType.clone.__dict__.__setitem__('stypy_kwargs_param_name', None)
        OrderedUnionType.clone.__dict__.__setitem__('stypy_call_defaults', defaults)
        OrderedUnionType.clone.__dict__.__setitem__('stypy_call_varargs', varargs)
        OrderedUnionType.clone.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        OrderedUnionType.clone.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'OrderedUnionType.clone', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'clone', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'clone(...)' code ##################

        str_14174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, (-1)), 'str', '\n        Clone the whole OrderedUnionType and its contained types\n        ')
        
        # Assigning a Call to a Name (line 760):
        
        # Call to clone(...): (line 760)
        # Processing the call keyword arguments (line 760)
        kwargs_14181 = {}
        
        # Obtaining the type of the subscript
        int_14175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 34), 'int')
        # Getting the type of 'self' (line 760)
        self_14176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 23), 'self', False)
        # Obtaining the member 'types' of a type (line 760)
        types_14177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 23), self_14176, 'types')
        # Obtaining the member '__getitem__' of a type (line 760)
        getitem___14178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 23), types_14177, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 760)
        subscript_call_result_14179 = invoke(stypy.reporting.localization.Localization(__file__, 760, 23), getitem___14178, int_14175)
        
        # Obtaining the member 'clone' of a type (line 760)
        clone_14180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 23), subscript_call_result_14179, 'clone')
        # Calling clone(args, kwargs) (line 760)
        clone_call_result_14182 = invoke(stypy.reporting.localization.Localization(__file__, 760, 23), clone_14180, *[], **kwargs_14181)
        
        # Assigning a type to the variable 'result_union' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'result_union', clone_call_result_14182)
        
        
        # Call to range(...): (line 761)
        # Processing the call arguments (line 761)
        int_14184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 23), 'int')
        
        # Call to len(...): (line 761)
        # Processing the call arguments (line 761)
        # Getting the type of 'self' (line 761)
        self_14186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 761)
        types_14187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 30), self_14186, 'types')
        # Processing the call keyword arguments (line 761)
        kwargs_14188 = {}
        # Getting the type of 'len' (line 761)
        len_14185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 26), 'len', False)
        # Calling len(args, kwargs) (line 761)
        len_call_result_14189 = invoke(stypy.reporting.localization.Localization(__file__, 761, 26), len_14185, *[types_14187], **kwargs_14188)
        
        # Processing the call keyword arguments (line 761)
        kwargs_14190 = {}
        # Getting the type of 'range' (line 761)
        range_14183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 17), 'range', False)
        # Calling range(args, kwargs) (line 761)
        range_call_result_14191 = invoke(stypy.reporting.localization.Localization(__file__, 761, 17), range_14183, *[int_14184, len_call_result_14189], **kwargs_14190)
        
        # Assigning a type to the variable 'range_call_result_14191' (line 761)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 8), 'range_call_result_14191', range_call_result_14191)
        # Testing if the for loop is going to be iterated (line 761)
        # Testing the type of a for loop iterable (line 761)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 761, 8), range_call_result_14191)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 761, 8), range_call_result_14191):
            # Getting the type of the for loop variable (line 761)
            for_loop_var_14192 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 761, 8), range_call_result_14191)
            # Assigning a type to the variable 'i' (line 761)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 8), 'i', for_loop_var_14192)
            # SSA begins for a for statement (line 761)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to isinstance(...): (line 762)
            # Processing the call arguments (line 762)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 762)
            i_14194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 37), 'i', False)
            # Getting the type of 'self' (line 762)
            self_14195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 26), 'self', False)
            # Obtaining the member 'types' of a type (line 762)
            types_14196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 26), self_14195, 'types')
            # Obtaining the member '__getitem__' of a type (line 762)
            getitem___14197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 26), types_14196, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 762)
            subscript_call_result_14198 = invoke(stypy.reporting.localization.Localization(__file__, 762, 26), getitem___14197, i_14194)
            
            # Getting the type of 'Type' (line 762)
            Type_14199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 41), 'Type', False)
            # Processing the call keyword arguments (line 762)
            kwargs_14200 = {}
            # Getting the type of 'isinstance' (line 762)
            isinstance_14193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 762)
            isinstance_call_result_14201 = invoke(stypy.reporting.localization.Localization(__file__, 762, 15), isinstance_14193, *[subscript_call_result_14198, Type_14199], **kwargs_14200)
            
            # Testing if the type of an if condition is none (line 762)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 762, 12), isinstance_call_result_14201):
                
                # Assigning a Call to a Name (line 765):
                
                # Call to add(...): (line 765)
                # Processing the call arguments (line 765)
                # Getting the type of 'result_union' (line 765)
                result_union_14218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 52), 'result_union', False)
                
                # Call to deepcopy(...): (line 765)
                # Processing the call arguments (line 765)
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 765)
                i_14221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 91), 'i', False)
                # Getting the type of 'self' (line 765)
                self_14222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 80), 'self', False)
                # Obtaining the member 'types' of a type (line 765)
                types_14223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 80), self_14222, 'types')
                # Obtaining the member '__getitem__' of a type (line 765)
                getitem___14224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 80), types_14223, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 765)
                subscript_call_result_14225 = invoke(stypy.reporting.localization.Localization(__file__, 765, 80), getitem___14224, i_14221)
                
                # Processing the call keyword arguments (line 765)
                kwargs_14226 = {}
                # Getting the type of 'copy' (line 765)
                copy_14219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 66), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 765)
                deepcopy_14220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 66), copy_14219, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 765)
                deepcopy_call_result_14227 = invoke(stypy.reporting.localization.Localization(__file__, 765, 66), deepcopy_14220, *[subscript_call_result_14225], **kwargs_14226)
                
                # Processing the call keyword arguments (line 765)
                kwargs_14228 = {}
                # Getting the type of 'OrderedUnionType' (line 765)
                OrderedUnionType_14216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 31), 'OrderedUnionType', False)
                # Obtaining the member 'add' of a type (line 765)
                add_14217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 31), OrderedUnionType_14216, 'add')
                # Calling add(args, kwargs) (line 765)
                add_call_result_14229 = invoke(stypy.reporting.localization.Localization(__file__, 765, 31), add_14217, *[result_union_14218, deepcopy_call_result_14227], **kwargs_14228)
                
                # Assigning a type to the variable 'result_union' (line 765)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 16), 'result_union', add_call_result_14229)
            else:
                
                # Testing the type of an if condition (line 762)
                if_condition_14202 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 762, 12), isinstance_call_result_14201)
                # Assigning a type to the variable 'if_condition_14202' (line 762)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 12), 'if_condition_14202', if_condition_14202)
                # SSA begins for if statement (line 762)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 763):
                
                # Call to add(...): (line 763)
                # Processing the call arguments (line 763)
                # Getting the type of 'result_union' (line 763)
                result_union_14205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 52), 'result_union', False)
                
                # Call to clone(...): (line 763)
                # Processing the call keyword arguments (line 763)
                kwargs_14212 = {}
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 763)
                i_14206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 77), 'i', False)
                # Getting the type of 'self' (line 763)
                self_14207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 66), 'self', False)
                # Obtaining the member 'types' of a type (line 763)
                types_14208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 66), self_14207, 'types')
                # Obtaining the member '__getitem__' of a type (line 763)
                getitem___14209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 66), types_14208, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 763)
                subscript_call_result_14210 = invoke(stypy.reporting.localization.Localization(__file__, 763, 66), getitem___14209, i_14206)
                
                # Obtaining the member 'clone' of a type (line 763)
                clone_14211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 66), subscript_call_result_14210, 'clone')
                # Calling clone(args, kwargs) (line 763)
                clone_call_result_14213 = invoke(stypy.reporting.localization.Localization(__file__, 763, 66), clone_14211, *[], **kwargs_14212)
                
                # Processing the call keyword arguments (line 763)
                kwargs_14214 = {}
                # Getting the type of 'OrderedUnionType' (line 763)
                OrderedUnionType_14203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 31), 'OrderedUnionType', False)
                # Obtaining the member 'add' of a type (line 763)
                add_14204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 31), OrderedUnionType_14203, 'add')
                # Calling add(args, kwargs) (line 763)
                add_call_result_14215 = invoke(stypy.reporting.localization.Localization(__file__, 763, 31), add_14204, *[result_union_14205, clone_call_result_14213], **kwargs_14214)
                
                # Assigning a type to the variable 'result_union' (line 763)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 16), 'result_union', add_call_result_14215)
                # SSA branch for the else part of an if statement (line 762)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Name (line 765):
                
                # Call to add(...): (line 765)
                # Processing the call arguments (line 765)
                # Getting the type of 'result_union' (line 765)
                result_union_14218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 52), 'result_union', False)
                
                # Call to deepcopy(...): (line 765)
                # Processing the call arguments (line 765)
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 765)
                i_14221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 91), 'i', False)
                # Getting the type of 'self' (line 765)
                self_14222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 80), 'self', False)
                # Obtaining the member 'types' of a type (line 765)
                types_14223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 80), self_14222, 'types')
                # Obtaining the member '__getitem__' of a type (line 765)
                getitem___14224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 80), types_14223, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 765)
                subscript_call_result_14225 = invoke(stypy.reporting.localization.Localization(__file__, 765, 80), getitem___14224, i_14221)
                
                # Processing the call keyword arguments (line 765)
                kwargs_14226 = {}
                # Getting the type of 'copy' (line 765)
                copy_14219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 66), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 765)
                deepcopy_14220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 66), copy_14219, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 765)
                deepcopy_call_result_14227 = invoke(stypy.reporting.localization.Localization(__file__, 765, 66), deepcopy_14220, *[subscript_call_result_14225], **kwargs_14226)
                
                # Processing the call keyword arguments (line 765)
                kwargs_14228 = {}
                # Getting the type of 'OrderedUnionType' (line 765)
                OrderedUnionType_14216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 31), 'OrderedUnionType', False)
                # Obtaining the member 'add' of a type (line 765)
                add_14217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 31), OrderedUnionType_14216, 'add')
                # Calling add(args, kwargs) (line 765)
                add_call_result_14229 = invoke(stypy.reporting.localization.Localization(__file__, 765, 31), add_14217, *[result_union_14218, deepcopy_call_result_14227], **kwargs_14228)
                
                # Assigning a type to the variable 'result_union' (line 765)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 16), 'result_union', add_call_result_14229)
                # SSA join for if statement (line 762)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'result_union' (line 767)
        result_union_14230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 15), 'result_union')
        # Assigning a type to the variable 'stypy_return_type' (line 767)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 8), 'stypy_return_type', result_union_14230)
        
        # ################# End of 'clone(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clone' in the type store
        # Getting the type of 'stypy_return_type' (line 756)
        stypy_return_type_14231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14231)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clone'
        return stypy_return_type_14231


# Assigning a type to the variable 'OrderedUnionType' (line 705)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 0), 'OrderedUnionType', OrderedUnionType)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
