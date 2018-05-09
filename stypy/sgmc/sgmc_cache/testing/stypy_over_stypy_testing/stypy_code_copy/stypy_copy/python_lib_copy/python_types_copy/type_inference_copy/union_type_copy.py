
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
9: import stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy.runtime_type_inspection_copy
10: import undefined_type_copy
11: from stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy import NonPythonType
12: from stypy_copy.python_lib_copy.python_types_copy.type_copy import Type
13: from stypy_copy.errors_copy.type_error_copy import TypeError
14: from stypy_copy.python_lib_copy.python_types_copy import type_inference_copy
15: from stypy_copy.reporting_copy.print_utils_copy import format_function_name
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

# 'import stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy.runtime_type_inspection_copy' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_481 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy.runtime_type_inspection_copy')

if (type(import_481) is not StypyTypeError):

    if (import_481 != 'pyd_module'):
        __import__(import_481)
        sys_modules_482 = sys.modules[import_481]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy.runtime_type_inspection_copy', sys_modules_482.module_type_store, module_type_store)
    else:
        import stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy.runtime_type_inspection_copy

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy.runtime_type_inspection_copy', stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy.runtime_type_inspection_copy, module_type_store)

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy.runtime_type_inspection_copy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy.runtime_type_inspection_copy', import_481)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import undefined_type_copy' statement (line 10)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_483 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'undefined_type_copy')

if (type(import_483) is not StypyTypeError):

    if (import_483 != 'pyd_module'):
        __import__(import_483)
        sys_modules_484 = sys.modules[import_483]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'undefined_type_copy', sys_modules_484.module_type_store, module_type_store)
    else:
        import undefined_type_copy

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'undefined_type_copy', undefined_type_copy, module_type_store)

else:
    # Assigning a type to the variable 'undefined_type_copy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'undefined_type_copy', import_483)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy import NonPythonType' statement (line 11)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_485 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy')

if (type(import_485) is not StypyTypeError):

    if (import_485 != 'pyd_module'):
        __import__(import_485)
        sys_modules_486 = sys.modules[import_485]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy', sys_modules_486.module_type_store, module_type_store, ['NonPythonType'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_486, sys_modules_486.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy import NonPythonType

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy', None, module_type_store, ['NonPythonType'], [NonPythonType])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy', import_485)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.type_copy import Type' statement (line 12)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_487 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_copy')

if (type(import_487) is not StypyTypeError):

    if (import_487 != 'pyd_module'):
        __import__(import_487)
        sys_modules_488 = sys.modules[import_487]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_copy', sys_modules_488.module_type_store, module_type_store, ['Type'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_488, sys_modules_488.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.type_copy import Type

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_copy', None, module_type_store, ['Type'], [Type])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.type_copy' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_copy', import_487)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 13)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_489 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_copy.errors_copy.type_error_copy')

if (type(import_489) is not StypyTypeError):

    if (import_489 != 'pyd_module'):
        __import__(import_489)
        sys_modules_490 = sys.modules[import_489]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_copy.errors_copy.type_error_copy', sys_modules_490.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_490, sys_modules_490.module_type_store, module_type_store)
    else:
        from stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'stypy_copy.errors_copy.type_error_copy' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_copy.errors_copy.type_error_copy', import_489)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy import type_inference_copy' statement (line 14)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_491 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_copy.python_lib_copy.python_types_copy')

if (type(import_491) is not StypyTypeError):

    if (import_491 != 'pyd_module'):
        __import__(import_491)
        sys_modules_492 = sys.modules[import_491]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_copy.python_lib_copy.python_types_copy', sys_modules_492.module_type_store, module_type_store, ['type_inference_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_492, sys_modules_492.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy import type_inference_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_copy.python_lib_copy.python_types_copy', None, module_type_store, ['type_inference_copy'], [type_inference_copy])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_copy.python_lib_copy.python_types_copy', import_491)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from stypy_copy.reporting_copy.print_utils_copy import format_function_name' statement (line 15)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_493 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_copy.reporting_copy.print_utils_copy')

if (type(import_493) is not StypyTypeError):

    if (import_493 != 'pyd_module'):
        __import__(import_493)
        sys_modules_494 = sys.modules[import_493]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_copy.reporting_copy.print_utils_copy', sys_modules_494.module_type_store, module_type_store, ['format_function_name'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_494, sys_modules_494.module_type_store, module_type_store)
    else:
        from stypy_copy.reporting_copy.print_utils_copy import format_function_name

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_copy.reporting_copy.print_utils_copy', None, module_type_store, ['format_function_name'], [format_function_name])

else:
    # Assigning a type to the variable 'stypy_copy.reporting_copy.print_utils_copy' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_copy.reporting_copy.print_utils_copy', import_493)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

# Declaration of the 'UnionType' class
# Getting the type of 'NonPythonType' (line 18)
NonPythonType_495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 16), 'NonPythonType')

class UnionType(NonPythonType_495, ):
    str_496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, (-1)), 'str', '\n    UnionType is a collection of types that represent the fact that a certain Python element can have any of the listed\n    types in a certain point of the execution of the program. UnionTypes are created by the application of the SSA\n    algorithm when dealing with branches in the processed program source code.\n    ')

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

        str_497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, (-1)), 'str', '\n        Internal method to store Python types in a TypeInferenceProxy if they are not already a TypeInferenceProxy\n        :param type_: Any Python object\n        :return:\n        ')
        
        
        # Call to isinstance(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'type_' (line 32)
        type__499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 26), 'type_', False)
        # Getting the type of 'Type' (line 32)
        Type_500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 33), 'Type', False)
        # Processing the call keyword arguments (line 32)
        kwargs_501 = {}
        # Getting the type of 'isinstance' (line 32)
        isinstance_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 32)
        isinstance_call_result_502 = invoke(stypy.reporting.localization.Localization(__file__, 32, 15), isinstance_498, *[type__499, Type_500], **kwargs_501)
        
        # Applying the 'not' unary operator (line 32)
        result_not__503 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 11), 'not', isinstance_call_result_502)
        
        # Testing if the type of an if condition is none (line 32)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 32, 8), result_not__503):
            
            
            # Call to has_type_instance_value(...): (line 40)
            # Processing the call keyword arguments (line 40)
            kwargs_526 = {}
            # Getting the type of 'type_' (line 40)
            type__524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'type_', False)
            # Obtaining the member 'has_type_instance_value' of a type (line 40)
            has_type_instance_value_525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 19), type__524, 'has_type_instance_value')
            # Calling has_type_instance_value(args, kwargs) (line 40)
            has_type_instance_value_call_result_527 = invoke(stypy.reporting.localization.Localization(__file__, 40, 19), has_type_instance_value_525, *[], **kwargs_526)
            
            # Applying the 'not' unary operator (line 40)
            result_not__528 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 15), 'not', has_type_instance_value_call_result_527)
            
            # Testing if the type of an if condition is none (line 40)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 40, 12), result_not__528):
                pass
            else:
                
                # Testing the type of an if condition (line 40)
                if_condition_529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 12), result_not__528)
                # Assigning a type to the variable 'if_condition_529' (line 40)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'if_condition_529', if_condition_529)
                # SSA begins for if statement (line 40)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to set_type_instance(...): (line 41)
                # Processing the call arguments (line 41)
                # Getting the type of 'True' (line 41)
                True_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 40), 'True', False)
                # Processing the call keyword arguments (line 41)
                kwargs_533 = {}
                # Getting the type of 'type_' (line 41)
                type__530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'type_', False)
                # Obtaining the member 'set_type_instance' of a type (line 41)
                set_type_instance_531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 16), type__530, 'set_type_instance')
                # Calling set_type_instance(args, kwargs) (line 41)
                set_type_instance_call_result_534 = invoke(stypy.reporting.localization.Localization(__file__, 41, 16), set_type_instance_531, *[True_532], **kwargs_533)
                
                # SSA join for if statement (line 40)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 32)
            if_condition_504 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 8), result_not__503)
            # Assigning a type to the variable 'if_condition_504' (line 32)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'if_condition_504', if_condition_504)
            # SSA begins for if statement (line 32)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 33):
            
            # Call to instance(...): (line 33)
            # Processing the call arguments (line 33)
            # Getting the type of 'type_' (line 33)
            type__509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 92), 'type_', False)
            # Processing the call keyword arguments (line 33)
            kwargs_510 = {}
            # Getting the type of 'type_inference_copy' (line 33)
            type_inference_copy_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 23), 'type_inference_copy', False)
            # Obtaining the member 'type_inference_proxy' of a type (line 33)
            type_inference_proxy_506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 23), type_inference_copy_505, 'type_inference_proxy')
            # Obtaining the member 'TypeInferenceProxy' of a type (line 33)
            TypeInferenceProxy_507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 23), type_inference_proxy_506, 'TypeInferenceProxy')
            # Obtaining the member 'instance' of a type (line 33)
            instance_508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 23), TypeInferenceProxy_507, 'instance')
            # Calling instance(args, kwargs) (line 33)
            instance_call_result_511 = invoke(stypy.reporting.localization.Localization(__file__, 33, 23), instance_508, *[type__509], **kwargs_510)
            
            # Assigning a type to the variable 'ret_type' (line 33)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'ret_type', instance_call_result_511)
            
            
            # Call to is_type_instance(...): (line 35)
            # Processing the call keyword arguments (line 35)
            kwargs_514 = {}
            # Getting the type of 'ret_type' (line 35)
            ret_type_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 19), 'ret_type', False)
            # Obtaining the member 'is_type_instance' of a type (line 35)
            is_type_instance_513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 19), ret_type_512, 'is_type_instance')
            # Calling is_type_instance(args, kwargs) (line 35)
            is_type_instance_call_result_515 = invoke(stypy.reporting.localization.Localization(__file__, 35, 19), is_type_instance_513, *[], **kwargs_514)
            
            # Applying the 'not' unary operator (line 35)
            result_not__516 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 15), 'not', is_type_instance_call_result_515)
            
            # Testing if the type of an if condition is none (line 35)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 35, 12), result_not__516):
                pass
            else:
                
                # Testing the type of an if condition (line 35)
                if_condition_517 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 12), result_not__516)
                # Assigning a type to the variable 'if_condition_517' (line 35)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'if_condition_517', if_condition_517)
                # SSA begins for if statement (line 35)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to set_type_instance(...): (line 36)
                # Processing the call arguments (line 36)
                # Getting the type of 'True' (line 36)
                True_520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 43), 'True', False)
                # Processing the call keyword arguments (line 36)
                kwargs_521 = {}
                # Getting the type of 'ret_type' (line 36)
                ret_type_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'ret_type', False)
                # Obtaining the member 'set_type_instance' of a type (line 36)
                set_type_instance_519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 16), ret_type_518, 'set_type_instance')
                # Calling set_type_instance(args, kwargs) (line 36)
                set_type_instance_call_result_522 = invoke(stypy.reporting.localization.Localization(__file__, 36, 16), set_type_instance_519, *[True_520], **kwargs_521)
                
                # SSA join for if statement (line 35)
                module_type_store = module_type_store.join_ssa_context()
                

            # Getting the type of 'ret_type' (line 37)
            ret_type_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 19), 'ret_type')
            # Assigning a type to the variable 'stypy_return_type' (line 37)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'stypy_return_type', ret_type_523)
            # SSA branch for the else part of an if statement (line 32)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to has_type_instance_value(...): (line 40)
            # Processing the call keyword arguments (line 40)
            kwargs_526 = {}
            # Getting the type of 'type_' (line 40)
            type__524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'type_', False)
            # Obtaining the member 'has_type_instance_value' of a type (line 40)
            has_type_instance_value_525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 19), type__524, 'has_type_instance_value')
            # Calling has_type_instance_value(args, kwargs) (line 40)
            has_type_instance_value_call_result_527 = invoke(stypy.reporting.localization.Localization(__file__, 40, 19), has_type_instance_value_525, *[], **kwargs_526)
            
            # Applying the 'not' unary operator (line 40)
            result_not__528 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 15), 'not', has_type_instance_value_call_result_527)
            
            # Testing if the type of an if condition is none (line 40)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 40, 12), result_not__528):
                pass
            else:
                
                # Testing the type of an if condition (line 40)
                if_condition_529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 12), result_not__528)
                # Assigning a type to the variable 'if_condition_529' (line 40)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'if_condition_529', if_condition_529)
                # SSA begins for if statement (line 40)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to set_type_instance(...): (line 41)
                # Processing the call arguments (line 41)
                # Getting the type of 'True' (line 41)
                True_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 40), 'True', False)
                # Processing the call keyword arguments (line 41)
                kwargs_533 = {}
                # Getting the type of 'type_' (line 41)
                type__530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'type_', False)
                # Obtaining the member 'set_type_instance' of a type (line 41)
                set_type_instance_531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 16), type__530, 'set_type_instance')
                # Calling set_type_instance(args, kwargs) (line 41)
                set_type_instance_call_result_534 = invoke(stypy.reporting.localization.Localization(__file__, 41, 16), set_type_instance_531, *[True_532], **kwargs_533)
                
                # SSA join for if statement (line 40)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 32)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'type_' (line 43)
        type__535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 15), 'type_')
        # Assigning a type to the variable 'stypy_return_type' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'stypy_return_type', type__535)
        
        # ################# End of '_wrap_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_wrap_type' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_536)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_wrap_type'
        return stypy_return_type_536


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 47)
        None_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 29), 'None')
        # Getting the type of 'None' (line 47)
        None_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 41), 'None')
        defaults = [None_537, None_538]
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

        str_539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, (-1)), 'str', '\n        Creates a new UnionType, optionally adding the passed parameters. If only a type is passed, this type\n        is returned instead\n        :param type1: Optional type to add. It can be another union type.\n        :param type2: Optional type to add . It cannot be another union type\n        :return:\n        ')
        
        # Assigning a List to a Attribute (line 55):
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        
        # Getting the type of 'self' (line 55)
        self_541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'self')
        # Setting the type of the member 'types' of a type (line 55)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), self_541, 'types', list_540)
        
        # Call to is_union_type(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'type1' (line 58)
        type1_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 103), 'type1', False)
        # Processing the call keyword arguments (line 58)
        kwargs_549 = {}
        # Getting the type of 'stypy_copy' (line 58)
        stypy_copy_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 58)
        python_lib_543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 11), stypy_copy_542, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 58)
        python_types_544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 11), python_lib_543, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 58)
        type_introspection_545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 11), python_types_544, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 58)
        runtime_type_inspection_546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 11), type_introspection_545, 'runtime_type_inspection')
        # Obtaining the member 'is_union_type' of a type (line 58)
        is_union_type_547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 11), runtime_type_inspection_546, 'is_union_type')
        # Calling is_union_type(args, kwargs) (line 58)
        is_union_type_call_result_550 = invoke(stypy.reporting.localization.Localization(__file__, 58, 11), is_union_type_547, *[type1_548], **kwargs_549)
        
        # Testing if the type of an if condition is none (line 58)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 58, 8), is_union_type_call_result_550):
            pass
        else:
            
            # Testing the type of an if condition (line 58)
            if_condition_551 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 8), is_union_type_call_result_550)
            # Assigning a type to the variable 'if_condition_551' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'if_condition_551', if_condition_551)
            # SSA begins for if statement (line 58)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'type1' (line 59)
            type1_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 25), 'type1')
            # Obtaining the member 'types' of a type (line 59)
            types_553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 25), type1_552, 'types')
            # Assigning a type to the variable 'types_553' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'types_553', types_553)
            # Testing if the for loop is going to be iterated (line 59)
            # Testing the type of a for loop iterable (line 59)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 59, 12), types_553)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 59, 12), types_553):
                # Getting the type of the for loop variable (line 59)
                for_loop_var_554 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 59, 12), types_553)
                # Assigning a type to the variable 'type_' (line 59)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'type_', for_loop_var_554)
                # SSA begins for a for statement (line 59)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to append(...): (line 60)
                # Processing the call arguments (line 60)
                # Getting the type of 'type_' (line 60)
                type__558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 34), 'type_', False)
                # Processing the call keyword arguments (line 60)
                kwargs_559 = {}
                # Getting the type of 'self' (line 60)
                self_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'self', False)
                # Obtaining the member 'types' of a type (line 60)
                types_556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 16), self_555, 'types')
                # Obtaining the member 'append' of a type (line 60)
                append_557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 16), types_556, 'append')
                # Calling append(args, kwargs) (line 60)
                append_call_result_560 = invoke(stypy.reporting.localization.Localization(__file__, 60, 16), append_557, *[type__558], **kwargs_559)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Assigning a type to the variable 'stypy_return_type' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 58)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Type idiom detected: calculating its left and rigth part (line 64)
        # Getting the type of 'type1' (line 64)
        type1_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'type1')
        # Getting the type of 'None' (line 64)
        None_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 24), 'None')
        
        (may_be_563, more_types_in_union_564) = may_not_be_none(type1_561, None_562)

        if may_be_563:

            if more_types_in_union_564:
                # Runtime conditional SSA (line 64)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to append(...): (line 65)
            # Processing the call arguments (line 65)
            
            # Call to _wrap_type(...): (line 65)
            # Processing the call arguments (line 65)
            # Getting the type of 'type1' (line 65)
            type1_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 51), 'type1', False)
            # Processing the call keyword arguments (line 65)
            kwargs_571 = {}
            # Getting the type of 'UnionType' (line 65)
            UnionType_568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 65)
            _wrap_type_569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 30), UnionType_568, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 65)
            _wrap_type_call_result_572 = invoke(stypy.reporting.localization.Localization(__file__, 65, 30), _wrap_type_569, *[type1_570], **kwargs_571)
            
            # Processing the call keyword arguments (line 65)
            kwargs_573 = {}
            # Getting the type of 'self' (line 65)
            self_565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'self', False)
            # Obtaining the member 'types' of a type (line 65)
            types_566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), self_565, 'types')
            # Obtaining the member 'append' of a type (line 65)
            append_567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), types_566, 'append')
            # Calling append(args, kwargs) (line 65)
            append_call_result_574 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), append_567, *[_wrap_type_call_result_572], **kwargs_573)
            

            if more_types_in_union_564:
                # SSA join for if statement (line 64)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 67)
        # Getting the type of 'type2' (line 67)
        type2_575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'type2')
        # Getting the type of 'None' (line 67)
        None_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), 'None')
        
        (may_be_577, more_types_in_union_578) = may_not_be_none(type2_575, None_576)

        if may_be_577:

            if more_types_in_union_578:
                # Runtime conditional SSA (line 67)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to append(...): (line 68)
            # Processing the call arguments (line 68)
            
            # Call to _wrap_type(...): (line 68)
            # Processing the call arguments (line 68)
            # Getting the type of 'type2' (line 68)
            type2_584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 51), 'type2', False)
            # Processing the call keyword arguments (line 68)
            kwargs_585 = {}
            # Getting the type of 'UnionType' (line 68)
            UnionType_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 30), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 68)
            _wrap_type_583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 30), UnionType_582, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 68)
            _wrap_type_call_result_586 = invoke(stypy.reporting.localization.Localization(__file__, 68, 30), _wrap_type_583, *[type2_584], **kwargs_585)
            
            # Processing the call keyword arguments (line 68)
            kwargs_587 = {}
            # Getting the type of 'self' (line 68)
            self_579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'self', False)
            # Obtaining the member 'types' of a type (line 68)
            types_580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), self_579, 'types')
            # Obtaining the member 'append' of a type (line 68)
            append_581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), types_580, 'append')
            # Calling append(args, kwargs) (line 68)
            append_call_result_588 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), append_581, *[_wrap_type_call_result_586], **kwargs_587)
            

            if more_types_in_union_578:
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

        str_589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, (-1)), 'str', '\n        Utility method to create a union type from a list of types\n        :param types: List of types\n        :return: UnionType\n        ')
        
        # Assigning a Call to a Name (line 77):
        
        # Call to UnionType(...): (line 77)
        # Processing the call keyword arguments (line 77)
        kwargs_591 = {}
        # Getting the type of 'UnionType' (line 77)
        UnionType_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 25), 'UnionType', False)
        # Calling UnionType(args, kwargs) (line 77)
        UnionType_call_result_592 = invoke(stypy.reporting.localization.Localization(__file__, 77, 25), UnionType_590, *[], **kwargs_591)
        
        # Assigning a type to the variable 'union_instance' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'union_instance', UnionType_call_result_592)
        
        # Getting the type of 'types' (line 79)
        types_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 21), 'types')
        # Assigning a type to the variable 'types_593' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'types_593', types_593)
        # Testing if the for loop is going to be iterated (line 79)
        # Testing the type of a for loop iterable (line 79)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 79, 8), types_593)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 79, 8), types_593):
            # Getting the type of the for loop variable (line 79)
            for_loop_var_594 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 79, 8), types_593)
            # Assigning a type to the variable 'type_' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'type_', for_loop_var_594)
            # SSA begins for a for statement (line 79)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to __add_unconditionally(...): (line 80)
            # Processing the call arguments (line 80)
            # Getting the type of 'union_instance' (line 80)
            union_instance_597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 44), 'union_instance', False)
            # Getting the type of 'type_' (line 80)
            type__598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 60), 'type_', False)
            # Processing the call keyword arguments (line 80)
            kwargs_599 = {}
            # Getting the type of 'UnionType' (line 80)
            UnionType_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'UnionType', False)
            # Obtaining the member '__add_unconditionally' of a type (line 80)
            add_unconditionally_596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), UnionType_595, '__add_unconditionally')
            # Calling __add_unconditionally(args, kwargs) (line 80)
            add_unconditionally_call_result_600 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), add_unconditionally_596, *[union_instance_597, type__598], **kwargs_599)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'union_instance' (line 82)
        union_instance_602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'union_instance', False)
        # Obtaining the member 'types' of a type (line 82)
        types_603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 15), union_instance_602, 'types')
        # Processing the call keyword arguments (line 82)
        kwargs_604 = {}
        # Getting the type of 'len' (line 82)
        len_601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'len', False)
        # Calling len(args, kwargs) (line 82)
        len_call_result_605 = invoke(stypy.reporting.localization.Localization(__file__, 82, 11), len_601, *[types_603], **kwargs_604)
        
        int_606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 40), 'int')
        # Applying the binary operator '==' (line 82)
        result_eq_607 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 11), '==', len_call_result_605, int_606)
        
        # Testing if the type of an if condition is none (line 82)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 82, 8), result_eq_607):
            pass
        else:
            
            # Testing the type of an if condition (line 82)
            if_condition_608 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 8), result_eq_607)
            # Assigning a type to the variable 'if_condition_608' (line 82)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'if_condition_608', if_condition_608)
            # SSA begins for if statement (line 82)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining the type of the subscript
            int_609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 40), 'int')
            # Getting the type of 'union_instance' (line 83)
            union_instance_610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 'union_instance')
            # Obtaining the member 'types' of a type (line 83)
            types_611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 19), union_instance_610, 'types')
            # Obtaining the member '__getitem__' of a type (line 83)
            getitem___612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 19), types_611, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 83)
            subscript_call_result_613 = invoke(stypy.reporting.localization.Localization(__file__, 83, 19), getitem___612, int_609)
            
            # Assigning a type to the variable 'stypy_return_type' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'stypy_return_type', subscript_call_result_613)
            # SSA join for if statement (line 82)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'union_instance' (line 84)
        union_instance_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 15), 'union_instance')
        # Assigning a type to the variable 'stypy_return_type' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'stypy_return_type', union_instance_614)
        
        # ################# End of 'create_union_type_from_types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_union_type_from_types' in the type store
        # Getting the type of 'stypy_return_type' (line 70)
        stypy_return_type_615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_615)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_union_type_from_types'
        return stypy_return_type_615


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

        str_616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, (-1)), 'str', '\n        Helper method of create_union_type_from_types\n        :param type1: Type to add\n        :param type2: Type to add\n        :return: UnionType\n        ')
        
        # Call to is_union_type(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'type1' (line 96)
        type1_623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 103), 'type1', False)
        # Processing the call keyword arguments (line 96)
        kwargs_624 = {}
        # Getting the type of 'stypy_copy' (line 96)
        stypy_copy_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 96)
        python_lib_618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 11), stypy_copy_617, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 96)
        python_types_619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 11), python_lib_618, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 96)
        type_introspection_620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 11), python_types_619, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 96)
        runtime_type_inspection_621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 11), type_introspection_620, 'runtime_type_inspection')
        # Obtaining the member 'is_union_type' of a type (line 96)
        is_union_type_622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 11), runtime_type_inspection_621, 'is_union_type')
        # Calling is_union_type(args, kwargs) (line 96)
        is_union_type_call_result_625 = invoke(stypy.reporting.localization.Localization(__file__, 96, 11), is_union_type_622, *[type1_623], **kwargs_624)
        
        # Testing if the type of an if condition is none (line 96)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 96, 8), is_union_type_call_result_625):
            pass
        else:
            
            # Testing the type of an if condition (line 96)
            if_condition_626 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 8), is_union_type_call_result_625)
            # Assigning a type to the variable 'if_condition_626' (line 96)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'if_condition_626', if_condition_626)
            # SSA begins for if statement (line 96)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _add(...): (line 97)
            # Processing the call arguments (line 97)
            
            # Call to _wrap_type(...): (line 97)
            # Processing the call arguments (line 97)
            # Getting the type of 'type2' (line 97)
            type2_631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 51), 'type2', False)
            # Processing the call keyword arguments (line 97)
            kwargs_632 = {}
            # Getting the type of 'UnionType' (line 97)
            UnionType_629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 30), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 97)
            _wrap_type_630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 30), UnionType_629, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 97)
            _wrap_type_call_result_633 = invoke(stypy.reporting.localization.Localization(__file__, 97, 30), _wrap_type_630, *[type2_631], **kwargs_632)
            
            # Processing the call keyword arguments (line 97)
            kwargs_634 = {}
            # Getting the type of 'type1' (line 97)
            type1_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 19), 'type1', False)
            # Obtaining the member '_add' of a type (line 97)
            _add_628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 19), type1_627, '_add')
            # Calling _add(args, kwargs) (line 97)
            _add_call_result_635 = invoke(stypy.reporting.localization.Localization(__file__, 97, 19), _add_628, *[_wrap_type_call_result_633], **kwargs_634)
            
            # Assigning a type to the variable 'stypy_return_type' (line 97)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'stypy_return_type', _add_call_result_635)
            # SSA join for if statement (line 96)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to is_union_type(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'type2' (line 98)
        type2_642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 103), 'type2', False)
        # Processing the call keyword arguments (line 98)
        kwargs_643 = {}
        # Getting the type of 'stypy_copy' (line 98)
        stypy_copy_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 98)
        python_lib_637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 11), stypy_copy_636, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 98)
        python_types_638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 11), python_lib_637, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 98)
        type_introspection_639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 11), python_types_638, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 98)
        runtime_type_inspection_640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 11), type_introspection_639, 'runtime_type_inspection')
        # Obtaining the member 'is_union_type' of a type (line 98)
        is_union_type_641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 11), runtime_type_inspection_640, 'is_union_type')
        # Calling is_union_type(args, kwargs) (line 98)
        is_union_type_call_result_644 = invoke(stypy.reporting.localization.Localization(__file__, 98, 11), is_union_type_641, *[type2_642], **kwargs_643)
        
        # Testing if the type of an if condition is none (line 98)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 98, 8), is_union_type_call_result_644):
            pass
        else:
            
            # Testing the type of an if condition (line 98)
            if_condition_645 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 8), is_union_type_call_result_644)
            # Assigning a type to the variable 'if_condition_645' (line 98)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'if_condition_645', if_condition_645)
            # SSA begins for if statement (line 98)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _add(...): (line 99)
            # Processing the call arguments (line 99)
            
            # Call to _wrap_type(...): (line 99)
            # Processing the call arguments (line 99)
            # Getting the type of 'type1' (line 99)
            type1_650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 51), 'type1', False)
            # Processing the call keyword arguments (line 99)
            kwargs_651 = {}
            # Getting the type of 'UnionType' (line 99)
            UnionType_648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 30), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 99)
            _wrap_type_649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 30), UnionType_648, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 99)
            _wrap_type_call_result_652 = invoke(stypy.reporting.localization.Localization(__file__, 99, 30), _wrap_type_649, *[type1_650], **kwargs_651)
            
            # Processing the call keyword arguments (line 99)
            kwargs_653 = {}
            # Getting the type of 'type2' (line 99)
            type2_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 19), 'type2', False)
            # Obtaining the member '_add' of a type (line 99)
            _add_647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 19), type2_646, '_add')
            # Calling _add(args, kwargs) (line 99)
            _add_call_result_654 = invoke(stypy.reporting.localization.Localization(__file__, 99, 19), _add_647, *[_wrap_type_call_result_652], **kwargs_653)
            
            # Assigning a type to the variable 'stypy_return_type' (line 99)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'stypy_return_type', _add_call_result_654)
            # SSA join for if statement (line 98)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'type1' (line 101)
        type1_655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'type1')
        # Getting the type of 'type2' (line 101)
        type2_656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 20), 'type2')
        # Applying the binary operator '==' (line 101)
        result_eq_657 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 11), '==', type1_655, type2_656)
        
        # Testing if the type of an if condition is none (line 101)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 101, 8), result_eq_657):
            pass
        else:
            
            # Testing the type of an if condition (line 101)
            if_condition_658 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 8), result_eq_657)
            # Assigning a type to the variable 'if_condition_658' (line 101)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'if_condition_658', if_condition_658)
            # SSA begins for if statement (line 101)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _wrap_type(...): (line 102)
            # Processing the call arguments (line 102)
            # Getting the type of 'type1' (line 102)
            type1_661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 40), 'type1', False)
            # Processing the call keyword arguments (line 102)
            kwargs_662 = {}
            # Getting the type of 'UnionType' (line 102)
            UnionType_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 19), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 102)
            _wrap_type_660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 19), UnionType_659, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 102)
            _wrap_type_call_result_663 = invoke(stypy.reporting.localization.Localization(__file__, 102, 19), _wrap_type_660, *[type1_661], **kwargs_662)
            
            # Assigning a type to the variable 'stypy_return_type' (line 102)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'stypy_return_type', _wrap_type_call_result_663)
            # SSA join for if statement (line 101)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to UnionType(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'type1' (line 104)
        type1_665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 25), 'type1', False)
        # Getting the type of 'type2' (line 104)
        type2_666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 32), 'type2', False)
        # Processing the call keyword arguments (line 104)
        kwargs_667 = {}
        # Getting the type of 'UnionType' (line 104)
        UnionType_664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'UnionType', False)
        # Calling UnionType(args, kwargs) (line 104)
        UnionType_call_result_668 = invoke(stypy.reporting.localization.Localization(__file__, 104, 15), UnionType_664, *[type1_665, type2_666], **kwargs_667)
        
        # Assigning a type to the variable 'stypy_return_type' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'stypy_return_type', UnionType_call_result_668)
        
        # ################# End of '__add_unconditionally(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__add_unconditionally' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_669)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__add_unconditionally'
        return stypy_return_type_669


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

        str_670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, (-1)), 'str', '\n        Adds type1 and type2 to potentially form a UnionType, with the following rules:\n        - If either type1 or type2 are None, the other type is returned and no UnionType is formed\n        - If either type1 or type2 are UndefinedType, the other type is returned and no UnionType is formed\n        - If either type1 or type2 are UnionTypes, they are mergued in a new UnionType that contains the types\n        represented by both of them.\n        - If both types are the same, the first is returned\n        - Else, a new UnionType formed by the two passed types are returned.\n\n        :param type1: Type to add\n        :param type2: Type to add\n        :return: A UnionType\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 121)
        # Getting the type of 'type1' (line 121)
        type1_671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'type1')
        # Getting the type of 'None' (line 121)
        None_672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 20), 'None')
        
        (may_be_673, more_types_in_union_674) = may_be_none(type1_671, None_672)

        if may_be_673:

            if more_types_in_union_674:
                # Runtime conditional SSA (line 121)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to _wrap_type(...): (line 122)
            # Processing the call arguments (line 122)
            # Getting the type of 'type2' (line 122)
            type2_677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 40), 'type2', False)
            # Processing the call keyword arguments (line 122)
            kwargs_678 = {}
            # Getting the type of 'UnionType' (line 122)
            UnionType_675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 19), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 122)
            _wrap_type_676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 19), UnionType_675, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 122)
            _wrap_type_call_result_679 = invoke(stypy.reporting.localization.Localization(__file__, 122, 19), _wrap_type_676, *[type2_677], **kwargs_678)
            
            # Assigning a type to the variable 'stypy_return_type' (line 122)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'stypy_return_type', _wrap_type_call_result_679)

            if more_types_in_union_674:
                # SSA join for if statement (line 121)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'type1' (line 121)
        type1_680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'type1')
        # Assigning a type to the variable 'type1' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'type1', remove_type_from_union(type1_680, types.NoneType))
        
        # Type idiom detected: calculating its left and rigth part (line 124)
        # Getting the type of 'type2' (line 124)
        type2_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 11), 'type2')
        # Getting the type of 'None' (line 124)
        None_682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'None')
        
        (may_be_683, more_types_in_union_684) = may_be_none(type2_681, None_682)

        if may_be_683:

            if more_types_in_union_684:
                # Runtime conditional SSA (line 124)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to _wrap_type(...): (line 125)
            # Processing the call arguments (line 125)
            # Getting the type of 'type1' (line 125)
            type1_687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 40), 'type1', False)
            # Processing the call keyword arguments (line 125)
            kwargs_688 = {}
            # Getting the type of 'UnionType' (line 125)
            UnionType_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 19), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 125)
            _wrap_type_686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 19), UnionType_685, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 125)
            _wrap_type_call_result_689 = invoke(stypy.reporting.localization.Localization(__file__, 125, 19), _wrap_type_686, *[type1_687], **kwargs_688)
            
            # Assigning a type to the variable 'stypy_return_type' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'stypy_return_type', _wrap_type_call_result_689)

            if more_types_in_union_684:
                # SSA join for if statement (line 124)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'type2' (line 124)
        type2_690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'type2')
        # Assigning a type to the variable 'type2' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'type2', remove_type_from_union(type2_690, types.NoneType))
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'type1' (line 127)
        type1_692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 22), 'type1', False)
        # Getting the type of 'TypeError' (line 127)
        TypeError_693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 29), 'TypeError', False)
        # Processing the call keyword arguments (line 127)
        kwargs_694 = {}
        # Getting the type of 'isinstance' (line 127)
        isinstance_691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 127)
        isinstance_call_result_695 = invoke(stypy.reporting.localization.Localization(__file__, 127, 11), isinstance_691, *[type1_692, TypeError_693], **kwargs_694)
        
        
        # Call to isinstance(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'type2' (line 127)
        type2_697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 55), 'type2', False)
        # Getting the type of 'TypeError' (line 127)
        TypeError_698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 62), 'TypeError', False)
        # Processing the call keyword arguments (line 127)
        kwargs_699 = {}
        # Getting the type of 'isinstance' (line 127)
        isinstance_696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 44), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 127)
        isinstance_call_result_700 = invoke(stypy.reporting.localization.Localization(__file__, 127, 44), isinstance_696, *[type2_697, TypeError_698], **kwargs_699)
        
        # Applying the binary operator 'and' (line 127)
        result_and_keyword_701 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 11), 'and', isinstance_call_result_695, isinstance_call_result_700)
        
        # Testing if the type of an if condition is none (line 127)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 127, 8), result_and_keyword_701):
            pass
        else:
            
            # Testing the type of an if condition (line 127)
            if_condition_702 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 8), result_and_keyword_701)
            # Assigning a type to the variable 'if_condition_702' (line 127)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'if_condition_702', if_condition_702)
            # SSA begins for if statement (line 127)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to _wrap_type(...): (line 128)
            # Processing the call arguments (line 128)
            # Getting the type of 'type1' (line 128)
            type1_705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 36), 'type1', False)
            # Processing the call keyword arguments (line 128)
            kwargs_706 = {}
            # Getting the type of 'UnionType' (line 128)
            UnionType_703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 128)
            _wrap_type_704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 15), UnionType_703, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 128)
            _wrap_type_call_result_707 = invoke(stypy.reporting.localization.Localization(__file__, 128, 15), _wrap_type_704, *[type1_705], **kwargs_706)
            
            
            # Call to _wrap_type(...): (line 128)
            # Processing the call arguments (line 128)
            # Getting the type of 'type2' (line 128)
            type2_710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 67), 'type2', False)
            # Processing the call keyword arguments (line 128)
            kwargs_711 = {}
            # Getting the type of 'UnionType' (line 128)
            UnionType_708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 46), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 128)
            _wrap_type_709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 46), UnionType_708, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 128)
            _wrap_type_call_result_712 = invoke(stypy.reporting.localization.Localization(__file__, 128, 46), _wrap_type_709, *[type2_710], **kwargs_711)
            
            # Applying the binary operator '==' (line 128)
            result_eq_713 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 15), '==', _wrap_type_call_result_707, _wrap_type_call_result_712)
            
            # Testing if the type of an if condition is none (line 128)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 128, 12), result_eq_713):
                
                # Getting the type of 'type1' (line 131)
                type1_720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'type1')
                # Obtaining the member 'error_msg' of a type (line 131)
                error_msg_721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 16), type1_720, 'error_msg')
                # Getting the type of 'type2' (line 131)
                type2_722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 35), 'type2')
                # Obtaining the member 'error_msg' of a type (line 131)
                error_msg_723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 35), type2_722, 'error_msg')
                # Applying the binary operator '+=' (line 131)
                result_iadd_724 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 16), '+=', error_msg_721, error_msg_723)
                # Getting the type of 'type1' (line 131)
                type1_725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'type1')
                # Setting the type of the member 'error_msg' of a type (line 131)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 16), type1_725, 'error_msg', result_iadd_724)
                
                
                # Call to remove_error_msg(...): (line 132)
                # Processing the call arguments (line 132)
                # Getting the type of 'type2' (line 132)
                type2_728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 43), 'type2', False)
                # Processing the call keyword arguments (line 132)
                kwargs_729 = {}
                # Getting the type of 'TypeError' (line 132)
                TypeError_726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'TypeError', False)
                # Obtaining the member 'remove_error_msg' of a type (line 132)
                remove_error_msg_727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 16), TypeError_726, 'remove_error_msg')
                # Calling remove_error_msg(args, kwargs) (line 132)
                remove_error_msg_call_result_730 = invoke(stypy.reporting.localization.Localization(__file__, 132, 16), remove_error_msg_727, *[type2_728], **kwargs_729)
                
                # Getting the type of 'type1' (line 133)
                type1_731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'type1')
                # Assigning a type to the variable 'stypy_return_type' (line 133)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'stypy_return_type', type1_731)
            else:
                
                # Testing the type of an if condition (line 128)
                if_condition_714 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 128, 12), result_eq_713)
                # Assigning a type to the variable 'if_condition_714' (line 128)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'if_condition_714', if_condition_714)
                # SSA begins for if statement (line 128)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to _wrap_type(...): (line 129)
                # Processing the call arguments (line 129)
                # Getting the type of 'type1' (line 129)
                type1_717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 44), 'type1', False)
                # Processing the call keyword arguments (line 129)
                kwargs_718 = {}
                # Getting the type of 'UnionType' (line 129)
                UnionType_715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), 'UnionType', False)
                # Obtaining the member '_wrap_type' of a type (line 129)
                _wrap_type_716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 23), UnionType_715, '_wrap_type')
                # Calling _wrap_type(args, kwargs) (line 129)
                _wrap_type_call_result_719 = invoke(stypy.reporting.localization.Localization(__file__, 129, 23), _wrap_type_716, *[type1_717], **kwargs_718)
                
                # Assigning a type to the variable 'stypy_return_type' (line 129)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'stypy_return_type', _wrap_type_call_result_719)
                # SSA branch for the else part of an if statement (line 128)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'type1' (line 131)
                type1_720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'type1')
                # Obtaining the member 'error_msg' of a type (line 131)
                error_msg_721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 16), type1_720, 'error_msg')
                # Getting the type of 'type2' (line 131)
                type2_722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 35), 'type2')
                # Obtaining the member 'error_msg' of a type (line 131)
                error_msg_723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 35), type2_722, 'error_msg')
                # Applying the binary operator '+=' (line 131)
                result_iadd_724 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 16), '+=', error_msg_721, error_msg_723)
                # Getting the type of 'type1' (line 131)
                type1_725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'type1')
                # Setting the type of the member 'error_msg' of a type (line 131)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 16), type1_725, 'error_msg', result_iadd_724)
                
                
                # Call to remove_error_msg(...): (line 132)
                # Processing the call arguments (line 132)
                # Getting the type of 'type2' (line 132)
                type2_728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 43), 'type2', False)
                # Processing the call keyword arguments (line 132)
                kwargs_729 = {}
                # Getting the type of 'TypeError' (line 132)
                TypeError_726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'TypeError', False)
                # Obtaining the member 'remove_error_msg' of a type (line 132)
                remove_error_msg_727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 16), TypeError_726, 'remove_error_msg')
                # Calling remove_error_msg(args, kwargs) (line 132)
                remove_error_msg_call_result_730 = invoke(stypy.reporting.localization.Localization(__file__, 132, 16), remove_error_msg_727, *[type2_728], **kwargs_729)
                
                # Getting the type of 'type1' (line 133)
                type1_731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'type1')
                # Assigning a type to the variable 'stypy_return_type' (line 133)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'stypy_return_type', type1_731)
                # SSA join for if statement (line 128)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 127)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to is_undefined_type(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'type1' (line 135)
        type1_738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 107), 'type1', False)
        # Processing the call keyword arguments (line 135)
        kwargs_739 = {}
        # Getting the type of 'stypy_copy' (line 135)
        stypy_copy_732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 135)
        python_lib_733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 11), stypy_copy_732, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 135)
        python_types_734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 11), python_lib_733, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 135)
        type_introspection_735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 11), python_types_734, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 135)
        runtime_type_inspection_736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 11), type_introspection_735, 'runtime_type_inspection')
        # Obtaining the member 'is_undefined_type' of a type (line 135)
        is_undefined_type_737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 11), runtime_type_inspection_736, 'is_undefined_type')
        # Calling is_undefined_type(args, kwargs) (line 135)
        is_undefined_type_call_result_740 = invoke(stypy.reporting.localization.Localization(__file__, 135, 11), is_undefined_type_737, *[type1_738], **kwargs_739)
        
        # Testing if the type of an if condition is none (line 135)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 135, 8), is_undefined_type_call_result_740):
            pass
        else:
            
            # Testing the type of an if condition (line 135)
            if_condition_741 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 8), is_undefined_type_call_result_740)
            # Assigning a type to the variable 'if_condition_741' (line 135)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'if_condition_741', if_condition_741)
            # SSA begins for if statement (line 135)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _wrap_type(...): (line 136)
            # Processing the call arguments (line 136)
            # Getting the type of 'type2' (line 136)
            type2_744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 40), 'type2', False)
            # Processing the call keyword arguments (line 136)
            kwargs_745 = {}
            # Getting the type of 'UnionType' (line 136)
            UnionType_742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 136)
            _wrap_type_743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 19), UnionType_742, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 136)
            _wrap_type_call_result_746 = invoke(stypy.reporting.localization.Localization(__file__, 136, 19), _wrap_type_743, *[type2_744], **kwargs_745)
            
            # Assigning a type to the variable 'stypy_return_type' (line 136)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'stypy_return_type', _wrap_type_call_result_746)
            # SSA join for if statement (line 135)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to is_undefined_type(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'type2' (line 137)
        type2_753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 107), 'type2', False)
        # Processing the call keyword arguments (line 137)
        kwargs_754 = {}
        # Getting the type of 'stypy_copy' (line 137)
        stypy_copy_747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 137)
        python_lib_748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 11), stypy_copy_747, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 137)
        python_types_749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 11), python_lib_748, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 137)
        type_introspection_750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 11), python_types_749, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 137)
        runtime_type_inspection_751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 11), type_introspection_750, 'runtime_type_inspection')
        # Obtaining the member 'is_undefined_type' of a type (line 137)
        is_undefined_type_752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 11), runtime_type_inspection_751, 'is_undefined_type')
        # Calling is_undefined_type(args, kwargs) (line 137)
        is_undefined_type_call_result_755 = invoke(stypy.reporting.localization.Localization(__file__, 137, 11), is_undefined_type_752, *[type2_753], **kwargs_754)
        
        # Testing if the type of an if condition is none (line 137)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 137, 8), is_undefined_type_call_result_755):
            pass
        else:
            
            # Testing the type of an if condition (line 137)
            if_condition_756 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 8), is_undefined_type_call_result_755)
            # Assigning a type to the variable 'if_condition_756' (line 137)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'if_condition_756', if_condition_756)
            # SSA begins for if statement (line 137)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _wrap_type(...): (line 138)
            # Processing the call arguments (line 138)
            # Getting the type of 'type1' (line 138)
            type1_759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 40), 'type1', False)
            # Processing the call keyword arguments (line 138)
            kwargs_760 = {}
            # Getting the type of 'UnionType' (line 138)
            UnionType_757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 19), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 138)
            _wrap_type_758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 19), UnionType_757, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 138)
            _wrap_type_call_result_761 = invoke(stypy.reporting.localization.Localization(__file__, 138, 19), _wrap_type_758, *[type1_759], **kwargs_760)
            
            # Assigning a type to the variable 'stypy_return_type' (line 138)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'stypy_return_type', _wrap_type_call_result_761)
            # SSA join for if statement (line 137)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to is_union_type(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'type1' (line 140)
        type1_768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 103), 'type1', False)
        # Processing the call keyword arguments (line 140)
        kwargs_769 = {}
        # Getting the type of 'stypy_copy' (line 140)
        stypy_copy_762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 140)
        python_lib_763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 11), stypy_copy_762, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 140)
        python_types_764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 11), python_lib_763, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 140)
        type_introspection_765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 11), python_types_764, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 140)
        runtime_type_inspection_766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 11), type_introspection_765, 'runtime_type_inspection')
        # Obtaining the member 'is_union_type' of a type (line 140)
        is_union_type_767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 11), runtime_type_inspection_766, 'is_union_type')
        # Calling is_union_type(args, kwargs) (line 140)
        is_union_type_call_result_770 = invoke(stypy.reporting.localization.Localization(__file__, 140, 11), is_union_type_767, *[type1_768], **kwargs_769)
        
        # Testing if the type of an if condition is none (line 140)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 140, 8), is_union_type_call_result_770):
            pass
        else:
            
            # Testing the type of an if condition (line 140)
            if_condition_771 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 8), is_union_type_call_result_770)
            # Assigning a type to the variable 'if_condition_771' (line 140)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'if_condition_771', if_condition_771)
            # SSA begins for if statement (line 140)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _add(...): (line 141)
            # Processing the call arguments (line 141)
            # Getting the type of 'type2' (line 141)
            type2_774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 30), 'type2', False)
            # Processing the call keyword arguments (line 141)
            kwargs_775 = {}
            # Getting the type of 'type1' (line 141)
            type1_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 19), 'type1', False)
            # Obtaining the member '_add' of a type (line 141)
            _add_773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 19), type1_772, '_add')
            # Calling _add(args, kwargs) (line 141)
            _add_call_result_776 = invoke(stypy.reporting.localization.Localization(__file__, 141, 19), _add_773, *[type2_774], **kwargs_775)
            
            # Assigning a type to the variable 'stypy_return_type' (line 141)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'stypy_return_type', _add_call_result_776)
            # SSA join for if statement (line 140)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to is_union_type(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'type2' (line 142)
        type2_783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 103), 'type2', False)
        # Processing the call keyword arguments (line 142)
        kwargs_784 = {}
        # Getting the type of 'stypy_copy' (line 142)
        stypy_copy_777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 142)
        python_lib_778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 11), stypy_copy_777, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 142)
        python_types_779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 11), python_lib_778, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 142)
        type_introspection_780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 11), python_types_779, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 142)
        runtime_type_inspection_781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 11), type_introspection_780, 'runtime_type_inspection')
        # Obtaining the member 'is_union_type' of a type (line 142)
        is_union_type_782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 11), runtime_type_inspection_781, 'is_union_type')
        # Calling is_union_type(args, kwargs) (line 142)
        is_union_type_call_result_785 = invoke(stypy.reporting.localization.Localization(__file__, 142, 11), is_union_type_782, *[type2_783], **kwargs_784)
        
        # Testing if the type of an if condition is none (line 142)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 142, 8), is_union_type_call_result_785):
            pass
        else:
            
            # Testing the type of an if condition (line 142)
            if_condition_786 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 8), is_union_type_call_result_785)
            # Assigning a type to the variable 'if_condition_786' (line 142)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'if_condition_786', if_condition_786)
            # SSA begins for if statement (line 142)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _add(...): (line 143)
            # Processing the call arguments (line 143)
            # Getting the type of 'type1' (line 143)
            type1_789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 30), 'type1', False)
            # Processing the call keyword arguments (line 143)
            kwargs_790 = {}
            # Getting the type of 'type2' (line 143)
            type2_787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 19), 'type2', False)
            # Obtaining the member '_add' of a type (line 143)
            _add_788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 19), type2_787, '_add')
            # Calling _add(args, kwargs) (line 143)
            _add_call_result_791 = invoke(stypy.reporting.localization.Localization(__file__, 143, 19), _add_788, *[type1_789], **kwargs_790)
            
            # Assigning a type to the variable 'stypy_return_type' (line 143)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'stypy_return_type', _add_call_result_791)
            # SSA join for if statement (line 142)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to _wrap_type(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'type1' (line 145)
        type1_794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 32), 'type1', False)
        # Processing the call keyword arguments (line 145)
        kwargs_795 = {}
        # Getting the type of 'UnionType' (line 145)
        UnionType_792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 11), 'UnionType', False)
        # Obtaining the member '_wrap_type' of a type (line 145)
        _wrap_type_793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 11), UnionType_792, '_wrap_type')
        # Calling _wrap_type(args, kwargs) (line 145)
        _wrap_type_call_result_796 = invoke(stypy.reporting.localization.Localization(__file__, 145, 11), _wrap_type_793, *[type1_794], **kwargs_795)
        
        
        # Call to _wrap_type(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'type2' (line 145)
        type2_799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 63), 'type2', False)
        # Processing the call keyword arguments (line 145)
        kwargs_800 = {}
        # Getting the type of 'UnionType' (line 145)
        UnionType_797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 42), 'UnionType', False)
        # Obtaining the member '_wrap_type' of a type (line 145)
        _wrap_type_798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 42), UnionType_797, '_wrap_type')
        # Calling _wrap_type(args, kwargs) (line 145)
        _wrap_type_call_result_801 = invoke(stypy.reporting.localization.Localization(__file__, 145, 42), _wrap_type_798, *[type2_799], **kwargs_800)
        
        # Applying the binary operator '==' (line 145)
        result_eq_802 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 11), '==', _wrap_type_call_result_796, _wrap_type_call_result_801)
        
        # Testing if the type of an if condition is none (line 145)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 145, 8), result_eq_802):
            pass
        else:
            
            # Testing the type of an if condition (line 145)
            if_condition_803 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 8), result_eq_802)
            # Assigning a type to the variable 'if_condition_803' (line 145)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'if_condition_803', if_condition_803)
            # SSA begins for if statement (line 145)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _wrap_type(...): (line 146)
            # Processing the call arguments (line 146)
            # Getting the type of 'type1' (line 146)
            type1_806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 40), 'type1', False)
            # Processing the call keyword arguments (line 146)
            kwargs_807 = {}
            # Getting the type of 'UnionType' (line 146)
            UnionType_804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 19), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 146)
            _wrap_type_805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 19), UnionType_804, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 146)
            _wrap_type_call_result_808 = invoke(stypy.reporting.localization.Localization(__file__, 146, 19), _wrap_type_805, *[type1_806], **kwargs_807)
            
            # Assigning a type to the variable 'stypy_return_type' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'stypy_return_type', _wrap_type_call_result_808)
            # SSA join for if statement (line 145)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to UnionType(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'type1' (line 148)
        type1_810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 25), 'type1', False)
        # Getting the type of 'type2' (line 148)
        type2_811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 32), 'type2', False)
        # Processing the call keyword arguments (line 148)
        kwargs_812 = {}
        # Getting the type of 'UnionType' (line 148)
        UnionType_809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 15), 'UnionType', False)
        # Calling UnionType(args, kwargs) (line 148)
        UnionType_call_result_813 = invoke(stypy.reporting.localization.Localization(__file__, 148, 15), UnionType_809, *[type1_810, type2_811], **kwargs_812)
        
        # Assigning a type to the variable 'stypy_return_type' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'stypy_return_type', UnionType_call_result_813)
        
        # ################# End of 'add(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add' in the type store
        # Getting the type of 'stypy_return_type' (line 106)
        stypy_return_type_814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_814)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add'
        return stypy_return_type_814


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

        str_815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, (-1)), 'str', '\n        Adds the passed type to the current UnionType object. If other_type is a UnionType, all its contained types\n        are added to the current.\n        :param other_type: Type to add\n        :return: The self object\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 157)
        # Getting the type of 'other_type' (line 157)
        other_type_816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 11), 'other_type')
        # Getting the type of 'None' (line 157)
        None_817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 25), 'None')
        
        (may_be_818, more_types_in_union_819) = may_be_none(other_type_816, None_817)

        if may_be_818:

            if more_types_in_union_819:
                # Runtime conditional SSA (line 157)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 158)
            self_820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 19), 'self')
            # Assigning a type to the variable 'stypy_return_type' (line 158)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'stypy_return_type', self_820)

            if more_types_in_union_819:
                # SSA join for if statement (line 157)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'other_type' (line 157)
        other_type_821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'other_type')
        # Assigning a type to the variable 'other_type' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'other_type', remove_type_from_union(other_type_821, types.NoneType))
        
        # Call to is_union_type(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'other_type' (line 159)
        other_type_828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 103), 'other_type', False)
        # Processing the call keyword arguments (line 159)
        kwargs_829 = {}
        # Getting the type of 'stypy_copy' (line 159)
        stypy_copy_822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 159)
        python_lib_823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 11), stypy_copy_822, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 159)
        python_types_824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 11), python_lib_823, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 159)
        type_introspection_825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 11), python_types_824, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 159)
        runtime_type_inspection_826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 11), type_introspection_825, 'runtime_type_inspection')
        # Obtaining the member 'is_union_type' of a type (line 159)
        is_union_type_827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 11), runtime_type_inspection_826, 'is_union_type')
        # Calling is_union_type(args, kwargs) (line 159)
        is_union_type_call_result_830 = invoke(stypy.reporting.localization.Localization(__file__, 159, 11), is_union_type_827, *[other_type_828], **kwargs_829)
        
        # Testing if the type of an if condition is none (line 159)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 159, 8), is_union_type_call_result_830):
            pass
        else:
            
            # Testing the type of an if condition (line 159)
            if_condition_831 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 8), is_union_type_call_result_830)
            # Assigning a type to the variable 'if_condition_831' (line 159)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'if_condition_831', if_condition_831)
            # SSA begins for if statement (line 159)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'other_type' (line 160)
            other_type_832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 21), 'other_type')
            # Obtaining the member 'types' of a type (line 160)
            types_833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 21), other_type_832, 'types')
            # Assigning a type to the variable 'types_833' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'types_833', types_833)
            # Testing if the for loop is going to be iterated (line 160)
            # Testing the type of a for loop iterable (line 160)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 160, 12), types_833)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 160, 12), types_833):
                # Getting the type of the for loop variable (line 160)
                for_loop_var_834 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 160, 12), types_833)
                # Assigning a type to the variable 't' (line 160)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 't', for_loop_var_834)
                # SSA begins for a for statement (line 160)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to _add(...): (line 161)
                # Processing the call arguments (line 161)
                # Getting the type of 't' (line 161)
                t_837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 26), 't', False)
                # Processing the call keyword arguments (line 161)
                kwargs_838 = {}
                # Getting the type of 'self' (line 161)
                self_835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'self', False)
                # Obtaining the member '_add' of a type (line 161)
                _add_836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 16), self_835, '_add')
                # Calling _add(args, kwargs) (line 161)
                _add_call_result_839 = invoke(stypy.reporting.localization.Localization(__file__, 161, 16), _add_836, *[t_837], **kwargs_838)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Getting the type of 'self' (line 162)
            self_840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 19), 'self')
            # Assigning a type to the variable 'stypy_return_type' (line 162)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'stypy_return_type', self_840)
            # SSA join for if statement (line 159)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 164):
        
        # Call to _wrap_type(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'other_type' (line 164)
        other_type_843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 42), 'other_type', False)
        # Processing the call keyword arguments (line 164)
        kwargs_844 = {}
        # Getting the type of 'UnionType' (line 164)
        UnionType_841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 21), 'UnionType', False)
        # Obtaining the member '_wrap_type' of a type (line 164)
        _wrap_type_842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 21), UnionType_841, '_wrap_type')
        # Calling _wrap_type(args, kwargs) (line 164)
        _wrap_type_call_result_845 = invoke(stypy.reporting.localization.Localization(__file__, 164, 21), _wrap_type_842, *[other_type_843], **kwargs_844)
        
        # Assigning a type to the variable 'other_type' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'other_type', _wrap_type_call_result_845)
        
        # Getting the type of 'self' (line 167)
        self_846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 17), 'self')
        # Obtaining the member 'types' of a type (line 167)
        types_847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 17), self_846, 'types')
        # Assigning a type to the variable 'types_847' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'types_847', types_847)
        # Testing if the for loop is going to be iterated (line 167)
        # Testing the type of a for loop iterable (line 167)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 167, 8), types_847)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 167, 8), types_847):
            # Getting the type of the for loop variable (line 167)
            for_loop_var_848 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 167, 8), types_847)
            # Assigning a type to the variable 't' (line 167)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 't', for_loop_var_848)
            # SSA begins for a for statement (line 167)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 't' (line 168)
            t_849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 15), 't')
            # Getting the type of 'other_type' (line 168)
            other_type_850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 20), 'other_type')
            # Applying the binary operator '==' (line 168)
            result_eq_851 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 15), '==', t_849, other_type_850)
            
            # Testing if the type of an if condition is none (line 168)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 168, 12), result_eq_851):
                pass
            else:
                
                # Testing the type of an if condition (line 168)
                if_condition_852 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 12), result_eq_851)
                # Assigning a type to the variable 'if_condition_852' (line 168)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'if_condition_852', if_condition_852)
                # SSA begins for if statement (line 168)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'self' (line 169)
                self_853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 23), 'self')
                # Assigning a type to the variable 'stypy_return_type' (line 169)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'stypy_return_type', self_853)
                # SSA join for if statement (line 168)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to append(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'other_type' (line 171)
        other_type_857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 26), 'other_type', False)
        # Processing the call keyword arguments (line 171)
        kwargs_858 = {}
        # Getting the type of 'self' (line 171)
        self_854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'self', False)
        # Obtaining the member 'types' of a type (line 171)
        types_855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), self_854, 'types')
        # Obtaining the member 'append' of a type (line 171)
        append_856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), types_855, 'append')
        # Calling append(args, kwargs) (line 171)
        append_call_result_859 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), append_856, *[other_type_857], **kwargs_858)
        
        # Getting the type of 'self' (line 173)
        self_860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'stypy_return_type', self_860)
        
        # ################# End of '_add(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_add' in the type store
        # Getting the type of 'stypy_return_type' (line 150)
        stypy_return_type_861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_861)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_add'
        return stypy_return_type_861


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

        str_862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, (-1)), 'str', '\n        Visual representation of the UnionType\n        :return:\n        ')
        
        # Call to __str__(...): (line 182)
        # Processing the call keyword arguments (line 182)
        kwargs_865 = {}
        # Getting the type of 'self' (line 182)
        self_863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'self', False)
        # Obtaining the member '__str__' of a type (line 182)
        str___864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 15), self_863, '__str__')
        # Calling __str__(args, kwargs) (line 182)
        str___call_result_866 = invoke(stypy.reporting.localization.Localization(__file__, 182, 15), str___864, *[], **kwargs_865)
        
        # Assigning a type to the variable 'stypy_return_type' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'stypy_return_type', str___call_result_866)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 177)
        stypy_return_type_867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_867)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_867


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

        str_868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, (-1)), 'str', '\n        Visual representation of the UnionType\n        :return:\n        ')
        
        # Assigning a Str to a Name (line 189):
        str_869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 18), 'str', '')
        # Assigning a type to the variable 'the_str' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'the_str', str_869)
        
        
        # Call to range(...): (line 190)
        # Processing the call arguments (line 190)
        
        # Call to len(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'self' (line 190)
        self_872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 27), 'self', False)
        # Obtaining the member 'types' of a type (line 190)
        types_873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 27), self_872, 'types')
        # Processing the call keyword arguments (line 190)
        kwargs_874 = {}
        # Getting the type of 'len' (line 190)
        len_871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 23), 'len', False)
        # Calling len(args, kwargs) (line 190)
        len_call_result_875 = invoke(stypy.reporting.localization.Localization(__file__, 190, 23), len_871, *[types_873], **kwargs_874)
        
        # Processing the call keyword arguments (line 190)
        kwargs_876 = {}
        # Getting the type of 'range' (line 190)
        range_870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 17), 'range', False)
        # Calling range(args, kwargs) (line 190)
        range_call_result_877 = invoke(stypy.reporting.localization.Localization(__file__, 190, 17), range_870, *[len_call_result_875], **kwargs_876)
        
        # Assigning a type to the variable 'range_call_result_877' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'range_call_result_877', range_call_result_877)
        # Testing if the for loop is going to be iterated (line 190)
        # Testing the type of a for loop iterable (line 190)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 190, 8), range_call_result_877)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 190, 8), range_call_result_877):
            # Getting the type of the for loop variable (line 190)
            for_loop_var_878 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 190, 8), range_call_result_877)
            # Assigning a type to the variable 'i' (line 190)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'i', for_loop_var_878)
            # SSA begins for a for statement (line 190)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'the_str' (line 191)
            the_str_879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'the_str')
            
            # Call to str(...): (line 191)
            # Processing the call arguments (line 191)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 191)
            i_881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 38), 'i', False)
            # Getting the type of 'self' (line 191)
            self_882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 27), 'self', False)
            # Obtaining the member 'types' of a type (line 191)
            types_883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 27), self_882, 'types')
            # Obtaining the member '__getitem__' of a type (line 191)
            getitem___884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 27), types_883, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 191)
            subscript_call_result_885 = invoke(stypy.reporting.localization.Localization(__file__, 191, 27), getitem___884, i_881)
            
            # Processing the call keyword arguments (line 191)
            kwargs_886 = {}
            # Getting the type of 'str' (line 191)
            str_880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 23), 'str', False)
            # Calling str(args, kwargs) (line 191)
            str_call_result_887 = invoke(stypy.reporting.localization.Localization(__file__, 191, 23), str_880, *[subscript_call_result_885], **kwargs_886)
            
            # Applying the binary operator '+=' (line 191)
            result_iadd_888 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 12), '+=', the_str_879, str_call_result_887)
            # Assigning a type to the variable 'the_str' (line 191)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'the_str', result_iadd_888)
            
            
            # Getting the type of 'i' (line 192)
            i_889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 15), 'i')
            
            # Call to len(...): (line 192)
            # Processing the call arguments (line 192)
            # Getting the type of 'self' (line 192)
            self_891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 23), 'self', False)
            # Obtaining the member 'types' of a type (line 192)
            types_892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 23), self_891, 'types')
            # Processing the call keyword arguments (line 192)
            kwargs_893 = {}
            # Getting the type of 'len' (line 192)
            len_890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 19), 'len', False)
            # Calling len(args, kwargs) (line 192)
            len_call_result_894 = invoke(stypy.reporting.localization.Localization(__file__, 192, 19), len_890, *[types_892], **kwargs_893)
            
            int_895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 37), 'int')
            # Applying the binary operator '-' (line 192)
            result_sub_896 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 19), '-', len_call_result_894, int_895)
            
            # Applying the binary operator '<' (line 192)
            result_lt_897 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 15), '<', i_889, result_sub_896)
            
            # Testing if the type of an if condition is none (line 192)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 192, 12), result_lt_897):
                pass
            else:
                
                # Testing the type of an if condition (line 192)
                if_condition_898 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 12), result_lt_897)
                # Assigning a type to the variable 'if_condition_898' (line 192)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'if_condition_898', if_condition_898)
                # SSA begins for if statement (line 192)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'the_str' (line 193)
                the_str_899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 16), 'the_str')
                str_900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 27), 'str', ' \\/ ')
                # Applying the binary operator '+=' (line 193)
                result_iadd_901 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 16), '+=', the_str_899, str_900)
                # Assigning a type to the variable 'the_str' (line 193)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 16), 'the_str', result_iadd_901)
                
                # SSA join for if statement (line 192)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'the_str' (line 194)
        the_str_902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 15), 'the_str')
        # Assigning a type to the variable 'stypy_return_type' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'stypy_return_type', the_str_902)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 184)
        stypy_return_type_903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_903)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_903


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

        str_904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, (-1)), 'str', '\n        Iterator interface, to iterate through the contained types\n        :return:\n        ')
        
        # Getting the type of 'self' (line 201)
        self_905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 20), 'self')
        # Obtaining the member 'types' of a type (line 201)
        types_906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 20), self_905, 'types')
        # Assigning a type to the variable 'types_906' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'types_906', types_906)
        # Testing if the for loop is going to be iterated (line 201)
        # Testing the type of a for loop iterable (line 201)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 201, 8), types_906)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 201, 8), types_906):
            # Getting the type of the for loop variable (line 201)
            for_loop_var_907 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 201, 8), types_906)
            # Assigning a type to the variable 'elem' (line 201)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'elem', for_loop_var_907)
            # SSA begins for a for statement (line 201)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Creating a generator
            # Getting the type of 'elem' (line 202)
            elem_908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 18), 'elem')
            GeneratorType_909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 12), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 12), GeneratorType_909, elem_908)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'stypy_return_type', GeneratorType_909)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of '__iter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iter__' in the type store
        # Getting the type of 'stypy_return_type' (line 196)
        stypy_return_type_910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_910)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iter__'
        return stypy_return_type_910


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

        str_911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, (-1)), 'str', '\n        The in operator, to determine if a type is inside a UnionType\n        :param item: Type to test. If it is another UnionType and this passed UnionType types are all inside the\n        current one, then the method returns true\n        :return: bool\n        ')
        
        # Call to is_union_type(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'item' (line 211)
        item_918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 103), 'item', False)
        # Processing the call keyword arguments (line 211)
        kwargs_919 = {}
        # Getting the type of 'stypy_copy' (line 211)
        stypy_copy_912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 211)
        python_lib_913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 11), stypy_copy_912, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 211)
        python_types_914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 11), python_lib_913, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 211)
        type_introspection_915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 11), python_types_914, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 211)
        runtime_type_inspection_916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 11), type_introspection_915, 'runtime_type_inspection')
        # Obtaining the member 'is_union_type' of a type (line 211)
        is_union_type_917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 11), runtime_type_inspection_916, 'is_union_type')
        # Calling is_union_type(args, kwargs) (line 211)
        is_union_type_call_result_920 = invoke(stypy.reporting.localization.Localization(__file__, 211, 11), is_union_type_917, *[item_918], **kwargs_919)
        
        # Testing if the type of an if condition is none (line 211)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 211, 8), is_union_type_call_result_920):
            
            # Call to isinstance(...): (line 217)
            # Processing the call arguments (line 217)
            # Getting the type of 'item' (line 217)
            item_932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 26), 'item', False)
            # Getting the type of 'undefined_type_copy' (line 217)
            undefined_type_copy_933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 32), 'undefined_type_copy', False)
            # Obtaining the member 'UndefinedType' of a type (line 217)
            UndefinedType_934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 32), undefined_type_copy_933, 'UndefinedType')
            # Processing the call keyword arguments (line 217)
            kwargs_935 = {}
            # Getting the type of 'isinstance' (line 217)
            isinstance_931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 217)
            isinstance_call_result_936 = invoke(stypy.reporting.localization.Localization(__file__, 217, 15), isinstance_931, *[item_932, UndefinedType_934], **kwargs_935)
            
            # Testing if the type of an if condition is none (line 217)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 217, 12), isinstance_call_result_936):
                
                # Getting the type of 'item' (line 224)
                item_951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 23), 'item')
                # Getting the type of 'self' (line 224)
                self_952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 31), 'self')
                # Obtaining the member 'types' of a type (line 224)
                types_953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 31), self_952, 'types')
                # Applying the binary operator 'in' (line 224)
                result_contains_954 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 23), 'in', item_951, types_953)
                
                # Assigning a type to the variable 'stypy_return_type' (line 224)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'stypy_return_type', result_contains_954)
            else:
                
                # Testing the type of an if condition (line 217)
                if_condition_937 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 12), isinstance_call_result_936)
                # Assigning a type to the variable 'if_condition_937' (line 217)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'if_condition_937', if_condition_937)
                # SSA begins for if statement (line 217)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 218):
                # Getting the type of 'False' (line 218)
                False_938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 24), 'False')
                # Assigning a type to the variable 'found' (line 218)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'found', False_938)
                
                # Getting the type of 'self' (line 219)
                self_939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 28), 'self')
                # Obtaining the member 'types' of a type (line 219)
                types_940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 28), self_939, 'types')
                # Assigning a type to the variable 'types_940' (line 219)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'types_940', types_940)
                # Testing if the for loop is going to be iterated (line 219)
                # Testing the type of a for loop iterable (line 219)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 219, 16), types_940)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 219, 16), types_940):
                    # Getting the type of the for loop variable (line 219)
                    for_loop_var_941 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 219, 16), types_940)
                    # Assigning a type to the variable 'elem' (line 219)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'elem', for_loop_var_941)
                    # SSA begins for a for statement (line 219)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Call to isinstance(...): (line 220)
                    # Processing the call arguments (line 220)
                    # Getting the type of 'elem' (line 220)
                    elem_943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 34), 'elem', False)
                    # Getting the type of 'undefined_type_copy' (line 220)
                    undefined_type_copy_944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 40), 'undefined_type_copy', False)
                    # Obtaining the member 'UndefinedType' of a type (line 220)
                    UndefinedType_945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 40), undefined_type_copy_944, 'UndefinedType')
                    # Processing the call keyword arguments (line 220)
                    kwargs_946 = {}
                    # Getting the type of 'isinstance' (line 220)
                    isinstance_942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 23), 'isinstance', False)
                    # Calling isinstance(args, kwargs) (line 220)
                    isinstance_call_result_947 = invoke(stypy.reporting.localization.Localization(__file__, 220, 23), isinstance_942, *[elem_943, UndefinedType_945], **kwargs_946)
                    
                    # Testing if the type of an if condition is none (line 220)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 220, 20), isinstance_call_result_947):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 220)
                        if_condition_948 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 20), isinstance_call_result_947)
                        # Assigning a type to the variable 'if_condition_948' (line 220)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 20), 'if_condition_948', if_condition_948)
                        # SSA begins for if statement (line 220)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Name to a Name (line 221):
                        # Getting the type of 'True' (line 221)
                        True_949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 32), 'True')
                        # Assigning a type to the variable 'found' (line 221)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 24), 'found', True_949)
                        # SSA join for if statement (line 220)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # Getting the type of 'found' (line 222)
                found_950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 23), 'found')
                # Assigning a type to the variable 'stypy_return_type' (line 222)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'stypy_return_type', found_950)
                # SSA branch for the else part of an if statement (line 217)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'item' (line 224)
                item_951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 23), 'item')
                # Getting the type of 'self' (line 224)
                self_952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 31), 'self')
                # Obtaining the member 'types' of a type (line 224)
                types_953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 31), self_952, 'types')
                # Applying the binary operator 'in' (line 224)
                result_contains_954 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 23), 'in', item_951, types_953)
                
                # Assigning a type to the variable 'stypy_return_type' (line 224)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'stypy_return_type', result_contains_954)
                # SSA join for if statement (line 217)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 211)
            if_condition_921 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 8), is_union_type_call_result_920)
            # Assigning a type to the variable 'if_condition_921' (line 211)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'if_condition_921', if_condition_921)
            # SSA begins for if statement (line 211)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'item' (line 212)
            item_922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 24), 'item')
            # Assigning a type to the variable 'item_922' (line 212)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'item_922', item_922)
            # Testing if the for loop is going to be iterated (line 212)
            # Testing the type of a for loop iterable (line 212)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 212, 12), item_922)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 212, 12), item_922):
                # Getting the type of the for loop variable (line 212)
                for_loop_var_923 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 212, 12), item_922)
                # Assigning a type to the variable 'elem' (line 212)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'elem', for_loop_var_923)
                # SSA begins for a for statement (line 212)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'elem' (line 213)
                elem_924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), 'elem')
                # Getting the type of 'self' (line 213)
                self_925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 31), 'self')
                # Obtaining the member 'types' of a type (line 213)
                types_926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 31), self_925, 'types')
                # Applying the binary operator 'notin' (line 213)
                result_contains_927 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 19), 'notin', elem_924, types_926)
                
                # Testing if the type of an if condition is none (line 213)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 213, 16), result_contains_927):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 213)
                    if_condition_928 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 16), result_contains_927)
                    # Assigning a type to the variable 'if_condition_928' (line 213)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'if_condition_928', if_condition_928)
                    # SSA begins for if statement (line 213)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'False' (line 214)
                    False_929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 27), 'False')
                    # Assigning a type to the variable 'stypy_return_type' (line 214)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 20), 'stypy_return_type', False_929)
                    # SSA join for if statement (line 213)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Getting the type of 'True' (line 215)
            True_930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 215)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'stypy_return_type', True_930)
            # SSA branch for the else part of an if statement (line 211)
            module_type_store.open_ssa_branch('else')
            
            # Call to isinstance(...): (line 217)
            # Processing the call arguments (line 217)
            # Getting the type of 'item' (line 217)
            item_932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 26), 'item', False)
            # Getting the type of 'undefined_type_copy' (line 217)
            undefined_type_copy_933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 32), 'undefined_type_copy', False)
            # Obtaining the member 'UndefinedType' of a type (line 217)
            UndefinedType_934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 32), undefined_type_copy_933, 'UndefinedType')
            # Processing the call keyword arguments (line 217)
            kwargs_935 = {}
            # Getting the type of 'isinstance' (line 217)
            isinstance_931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 217)
            isinstance_call_result_936 = invoke(stypy.reporting.localization.Localization(__file__, 217, 15), isinstance_931, *[item_932, UndefinedType_934], **kwargs_935)
            
            # Testing if the type of an if condition is none (line 217)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 217, 12), isinstance_call_result_936):
                
                # Getting the type of 'item' (line 224)
                item_951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 23), 'item')
                # Getting the type of 'self' (line 224)
                self_952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 31), 'self')
                # Obtaining the member 'types' of a type (line 224)
                types_953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 31), self_952, 'types')
                # Applying the binary operator 'in' (line 224)
                result_contains_954 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 23), 'in', item_951, types_953)
                
                # Assigning a type to the variable 'stypy_return_type' (line 224)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'stypy_return_type', result_contains_954)
            else:
                
                # Testing the type of an if condition (line 217)
                if_condition_937 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 12), isinstance_call_result_936)
                # Assigning a type to the variable 'if_condition_937' (line 217)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'if_condition_937', if_condition_937)
                # SSA begins for if statement (line 217)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 218):
                # Getting the type of 'False' (line 218)
                False_938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 24), 'False')
                # Assigning a type to the variable 'found' (line 218)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'found', False_938)
                
                # Getting the type of 'self' (line 219)
                self_939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 28), 'self')
                # Obtaining the member 'types' of a type (line 219)
                types_940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 28), self_939, 'types')
                # Assigning a type to the variable 'types_940' (line 219)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'types_940', types_940)
                # Testing if the for loop is going to be iterated (line 219)
                # Testing the type of a for loop iterable (line 219)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 219, 16), types_940)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 219, 16), types_940):
                    # Getting the type of the for loop variable (line 219)
                    for_loop_var_941 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 219, 16), types_940)
                    # Assigning a type to the variable 'elem' (line 219)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'elem', for_loop_var_941)
                    # SSA begins for a for statement (line 219)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Call to isinstance(...): (line 220)
                    # Processing the call arguments (line 220)
                    # Getting the type of 'elem' (line 220)
                    elem_943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 34), 'elem', False)
                    # Getting the type of 'undefined_type_copy' (line 220)
                    undefined_type_copy_944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 40), 'undefined_type_copy', False)
                    # Obtaining the member 'UndefinedType' of a type (line 220)
                    UndefinedType_945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 40), undefined_type_copy_944, 'UndefinedType')
                    # Processing the call keyword arguments (line 220)
                    kwargs_946 = {}
                    # Getting the type of 'isinstance' (line 220)
                    isinstance_942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 23), 'isinstance', False)
                    # Calling isinstance(args, kwargs) (line 220)
                    isinstance_call_result_947 = invoke(stypy.reporting.localization.Localization(__file__, 220, 23), isinstance_942, *[elem_943, UndefinedType_945], **kwargs_946)
                    
                    # Testing if the type of an if condition is none (line 220)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 220, 20), isinstance_call_result_947):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 220)
                        if_condition_948 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 20), isinstance_call_result_947)
                        # Assigning a type to the variable 'if_condition_948' (line 220)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 20), 'if_condition_948', if_condition_948)
                        # SSA begins for if statement (line 220)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Name to a Name (line 221):
                        # Getting the type of 'True' (line 221)
                        True_949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 32), 'True')
                        # Assigning a type to the variable 'found' (line 221)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 24), 'found', True_949)
                        # SSA join for if statement (line 220)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # Getting the type of 'found' (line 222)
                found_950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 23), 'found')
                # Assigning a type to the variable 'stypy_return_type' (line 222)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'stypy_return_type', found_950)
                # SSA branch for the else part of an if statement (line 217)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'item' (line 224)
                item_951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 23), 'item')
                # Getting the type of 'self' (line 224)
                self_952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 31), 'self')
                # Obtaining the member 'types' of a type (line 224)
                types_953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 31), self_952, 'types')
                # Applying the binary operator 'in' (line 224)
                result_contains_954 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 23), 'in', item_951, types_953)
                
                # Assigning a type to the variable 'stypy_return_type' (line 224)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'stypy_return_type', result_contains_954)
                # SSA join for if statement (line 217)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 211)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '__contains__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__contains__' in the type store
        # Getting the type of 'stypy_return_type' (line 204)
        stypy_return_type_955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_955)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__contains__'
        return stypy_return_type_955


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

        str_956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, (-1)), 'str', '\n        The == operator, to compare UnionTypes\n\n        :param other: Another UnionType (used in type inference code) or a list of types (used in unit testing)\n        :return: True if the passed UnionType or list contains exactly the same amount and type of types that the\n        passed entities\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 234)
        # Getting the type of 'list' (line 234)
        list_957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 29), 'list')
        # Getting the type of 'other' (line 234)
        other_958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 22), 'other')
        
        (may_be_959, more_types_in_union_960) = may_be_subtype(list_957, other_958)

        if may_be_959:

            if more_types_in_union_960:
                # Runtime conditional SSA (line 234)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'other' (line 234)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'other', remove_not_subtype_from_union(other_958, list))
            
            # Assigning a Name to a Name (line 235):
            # Getting the type of 'other' (line 235)
            other_961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 24), 'other')
            # Assigning a type to the variable 'type_list' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'type_list', other_961)

            if more_types_in_union_960:
                # Runtime conditional SSA for else branch (line 234)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_959) or more_types_in_union_960):
            # Assigning a type to the variable 'other' (line 234)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'other', remove_subtype_from_union(other_958, list))
            
            # Call to isinstance(...): (line 237)
            # Processing the call arguments (line 237)
            # Getting the type of 'other' (line 237)
            other_963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 26), 'other', False)
            # Getting the type of 'UnionType' (line 237)
            UnionType_964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 33), 'UnionType', False)
            # Processing the call keyword arguments (line 237)
            kwargs_965 = {}
            # Getting the type of 'isinstance' (line 237)
            isinstance_962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 237)
            isinstance_call_result_966 = invoke(stypy.reporting.localization.Localization(__file__, 237, 15), isinstance_962, *[other_963, UnionType_964], **kwargs_965)
            
            # Testing if the type of an if condition is none (line 237)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 237, 12), isinstance_call_result_966):
                # Getting the type of 'False' (line 240)
                False_970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 240)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 16), 'stypy_return_type', False_970)
            else:
                
                # Testing the type of an if condition (line 237)
                if_condition_967 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 237, 12), isinstance_call_result_966)
                # Assigning a type to the variable 'if_condition_967' (line 237)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'if_condition_967', if_condition_967)
                # SSA begins for if statement (line 237)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Attribute to a Name (line 238):
                # Getting the type of 'other' (line 238)
                other_968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 28), 'other')
                # Obtaining the member 'types' of a type (line 238)
                types_969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 28), other_968, 'types')
                # Assigning a type to the variable 'type_list' (line 238)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'type_list', types_969)
                # SSA branch for the else part of an if statement (line 237)
                module_type_store.open_ssa_branch('else')
                # Getting the type of 'False' (line 240)
                False_970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 240)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 16), 'stypy_return_type', False_970)
                # SSA join for if statement (line 237)
                module_type_store = module_type_store.join_ssa_context()
                


            if (may_be_959 and more_types_in_union_960):
                # SSA join for if statement (line 234)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        
        # Call to len(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'self' (line 242)
        self_972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 19), 'self', False)
        # Obtaining the member 'types' of a type (line 242)
        types_973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 19), self_972, 'types')
        # Processing the call keyword arguments (line 242)
        kwargs_974 = {}
        # Getting the type of 'len' (line 242)
        len_971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 15), 'len', False)
        # Calling len(args, kwargs) (line 242)
        len_call_result_975 = invoke(stypy.reporting.localization.Localization(__file__, 242, 15), len_971, *[types_973], **kwargs_974)
        
        
        # Call to len(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'type_list' (line 242)
        type_list_977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 38), 'type_list', False)
        # Processing the call keyword arguments (line 242)
        kwargs_978 = {}
        # Getting the type of 'len' (line 242)
        len_976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 34), 'len', False)
        # Calling len(args, kwargs) (line 242)
        len_call_result_979 = invoke(stypy.reporting.localization.Localization(__file__, 242, 34), len_976, *[type_list_977], **kwargs_978)
        
        # Applying the binary operator '==' (line 242)
        result_eq_980 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 15), '==', len_call_result_975, len_call_result_979)
        
        # Applying the 'not' unary operator (line 242)
        result_not__981 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 11), 'not', result_eq_980)
        
        # Testing if the type of an if condition is none (line 242)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 242, 8), result_not__981):
            pass
        else:
            
            # Testing the type of an if condition (line 242)
            if_condition_982 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 8), result_not__981)
            # Assigning a type to the variable 'if_condition_982' (line 242)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'if_condition_982', if_condition_982)
            # SSA begins for if statement (line 242)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'False' (line 243)
            False_983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 243)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'stypy_return_type', False_983)
            # SSA join for if statement (line 242)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 245)
        self_984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 21), 'self')
        # Obtaining the member 'types' of a type (line 245)
        types_985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 21), self_984, 'types')
        # Assigning a type to the variable 'types_985' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'types_985', types_985)
        # Testing if the for loop is going to be iterated (line 245)
        # Testing the type of a for loop iterable (line 245)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 245, 8), types_985)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 245, 8), types_985):
            # Getting the type of the for loop variable (line 245)
            for_loop_var_986 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 245, 8), types_985)
            # Assigning a type to the variable 'type_' (line 245)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'type_', for_loop_var_986)
            # SSA begins for a for statement (line 245)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Type idiom detected: calculating its left and rigth part (line 246)
            # Getting the type of 'TypeError' (line 246)
            TypeError_987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 33), 'TypeError')
            # Getting the type of 'type_' (line 246)
            type__988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 26), 'type_')
            
            (may_be_989, more_types_in_union_990) = may_be_subtype(TypeError_987, type__988)

            if may_be_989:

                if more_types_in_union_990:
                    # Runtime conditional SSA (line 246)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'type_' (line 246)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'type_', remove_not_subtype_from_union(type__988, TypeError))
                
                # Getting the type of 'type_list' (line 247)
                type_list_991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 30), 'type_list')
                # Assigning a type to the variable 'type_list_991' (line 247)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'type_list_991', type_list_991)
                # Testing if the for loop is going to be iterated (line 247)
                # Testing the type of a for loop iterable (line 247)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 247, 16), type_list_991)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 247, 16), type_list_991):
                    # Getting the type of the for loop variable (line 247)
                    for_loop_var_992 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 247, 16), type_list_991)
                    # Assigning a type to the variable 'type_2' (line 247)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'type_2', for_loop_var_992)
                    # SSA begins for a for statement (line 247)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Type idiom detected: calculating its left and rigth part (line 248)
                    # Getting the type of 'type_2' (line 248)
                    type_2_993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 28), 'type_2')
                    # Getting the type of 'TypeError' (line 248)
                    TypeError_994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 39), 'TypeError')
                    
                    (may_be_995, more_types_in_union_996) = may_be_type(type_2_993, TypeError_994)

                    if may_be_995:

                        if more_types_in_union_996:
                            # Runtime conditional SSA (line 248)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                        else:
                            module_type_store = module_type_store

                        # Assigning a type to the variable 'type_2' (line 248)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 20), 'type_2', TypeError_994())

                        if more_types_in_union_996:
                            # SSA join for if statement (line 248)
                            module_type_store = module_type_store.join_ssa_context()


                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                

                if more_types_in_union_990:
                    # SSA join for if statement (line 246)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Getting the type of 'type_' (line 250)
            type__997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 15), 'type_')
            # Getting the type of 'type_list' (line 250)
            type_list_998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 28), 'type_list')
            # Applying the binary operator 'notin' (line 250)
            result_contains_999 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 15), 'notin', type__997, type_list_998)
            
            # Testing if the type of an if condition is none (line 250)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 250, 12), result_contains_999):
                pass
            else:
                
                # Testing the type of an if condition (line 250)
                if_condition_1000 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 250, 12), result_contains_999)
                # Assigning a type to the variable 'if_condition_1000' (line 250)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'if_condition_1000', if_condition_1000)
                # SSA begins for if statement (line 250)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'False' (line 251)
                False_1001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 251)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 16), 'stypy_return_type', False_1001)
                # SSA join for if statement (line 250)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'True' (line 253)
        True_1002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'stypy_return_type', True_1002)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 226)
        stypy_return_type_1003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1003)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_1003


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

        str_1004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, (-1)), 'str', '\n        The [] operator, to obtain individual types stored within the union type\n\n        :param item: Indexer\n        :return:\n        ')
        
        # Obtaining the type of the subscript
        # Getting the type of 'item' (line 262)
        item_1005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 26), 'item')
        # Getting the type of 'self' (line 262)
        self_1006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 15), 'self')
        # Obtaining the member 'types' of a type (line 262)
        types_1007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 15), self_1006, 'types')
        # Obtaining the member '__getitem__' of a type (line 262)
        getitem___1008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 15), types_1007, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 262)
        subscript_call_result_1009 = invoke(stypy.reporting.localization.Localization(__file__, 262, 15), getitem___1008, item_1005)
        
        # Assigning a type to the variable 'stypy_return_type' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'stypy_return_type', subscript_call_result_1009)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 255)
        stypy_return_type_1010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1010)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_1010


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

        str_1011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, (-1)), 'str', '\n        For all the types stored in the union type, obtain the type of the member named member_name, returning a\n        Union Type with the union of all the possible types that member_name has inside the UnionType. For example,\n        if a UnionType has the types Class1 and Class2, both with the member "attr" so Class1.attr: int and\n        Class2.attr: str, this method will return int \\/ str.\n        :param localization: Caller information\n        :param member_name: Name of the member to get\n        :return All the types that member_name could have, examining the UnionType stored types\n        ')
        
        # Assigning a List to a Name (line 276):
        
        # Obtaining an instance of the builtin type 'list' (line 276)
        list_1012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 276)
        
        # Assigning a type to the variable 'result' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'result', list_1012)
        
        # Getting the type of 'self' (line 279)
        self_1013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 21), 'self')
        # Obtaining the member 'types' of a type (line 279)
        types_1014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 21), self_1013, 'types')
        # Assigning a type to the variable 'types_1014' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'types_1014', types_1014)
        # Testing if the for loop is going to be iterated (line 279)
        # Testing the type of a for loop iterable (line 279)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 279, 8), types_1014)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 279, 8), types_1014):
            # Getting the type of the for loop variable (line 279)
            for_loop_var_1015 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 279, 8), types_1014)
            # Assigning a type to the variable 'type_' (line 279)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'type_', for_loop_var_1015)
            # SSA begins for a for statement (line 279)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 280):
            
            # Call to get_type_of_member(...): (line 280)
            # Processing the call arguments (line 280)
            # Getting the type of 'localization' (line 280)
            localization_1018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 44), 'localization', False)
            # Getting the type of 'member_name' (line 280)
            member_name_1019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 58), 'member_name', False)
            # Processing the call keyword arguments (line 280)
            kwargs_1020 = {}
            # Getting the type of 'type_' (line 280)
            type__1016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 19), 'type_', False)
            # Obtaining the member 'get_type_of_member' of a type (line 280)
            get_type_of_member_1017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 19), type__1016, 'get_type_of_member')
            # Calling get_type_of_member(args, kwargs) (line 280)
            get_type_of_member_call_result_1021 = invoke(stypy.reporting.localization.Localization(__file__, 280, 19), get_type_of_member_1017, *[localization_1018, member_name_1019], **kwargs_1020)
            
            # Assigning a type to the variable 'temp' (line 280)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'temp', get_type_of_member_call_result_1021)
            
            # Call to append(...): (line 281)
            # Processing the call arguments (line 281)
            # Getting the type of 'temp' (line 281)
            temp_1024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 26), 'temp', False)
            # Processing the call keyword arguments (line 281)
            kwargs_1025 = {}
            # Getting the type of 'result' (line 281)
            result_1022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'result', False)
            # Obtaining the member 'append' of a type (line 281)
            append_1023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 12), result_1022, 'append')
            # Calling append(args, kwargs) (line 281)
            append_call_result_1026 = invoke(stypy.reporting.localization.Localization(__file__, 281, 12), append_1023, *[temp_1024], **kwargs_1025)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 284):
        
        # Call to filter(...): (line 284)
        # Processing the call arguments (line 284)

        @norecursion
        def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_1'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 284, 24, True)
            # Passed parameters checking function
            _stypy_temp_lambda_1.stypy_localization = localization
            _stypy_temp_lambda_1.stypy_type_of_self = None
            _stypy_temp_lambda_1.stypy_type_store = module_type_store
            _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
            _stypy_temp_lambda_1.stypy_param_names_list = ['t']
            _stypy_temp_lambda_1.stypy_varargs_param_name = None
            _stypy_temp_lambda_1.stypy_kwargs_param_name = None
            _stypy_temp_lambda_1.stypy_call_defaults = defaults
            _stypy_temp_lambda_1.stypy_call_varargs = varargs
            _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['t'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_1', ['t'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to isinstance(...): (line 284)
            # Processing the call arguments (line 284)
            # Getting the type of 't' (line 284)
            t_1029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 45), 't', False)
            # Getting the type of 'TypeError' (line 284)
            TypeError_1030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 48), 'TypeError', False)
            # Processing the call keyword arguments (line 284)
            kwargs_1031 = {}
            # Getting the type of 'isinstance' (line 284)
            isinstance_1028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 34), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 284)
            isinstance_call_result_1032 = invoke(stypy.reporting.localization.Localization(__file__, 284, 34), isinstance_1028, *[t_1029, TypeError_1030], **kwargs_1031)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 284)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 24), 'stypy_return_type', isinstance_call_result_1032)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_1' in the type store
            # Getting the type of 'stypy_return_type' (line 284)
            stypy_return_type_1033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 24), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_1033)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_1'
            return stypy_return_type_1033

        # Assigning a type to the variable '_stypy_temp_lambda_1' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 24), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
        # Getting the type of '_stypy_temp_lambda_1' (line 284)
        _stypy_temp_lambda_1_1034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 24), '_stypy_temp_lambda_1')
        # Getting the type of 'result' (line 284)
        result_1035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 60), 'result', False)
        # Processing the call keyword arguments (line 284)
        kwargs_1036 = {}
        # Getting the type of 'filter' (line 284)
        filter_1027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 17), 'filter', False)
        # Calling filter(args, kwargs) (line 284)
        filter_call_result_1037 = invoke(stypy.reporting.localization.Localization(__file__, 284, 17), filter_1027, *[_stypy_temp_lambda_1_1034, result_1035], **kwargs_1036)
        
        # Assigning a type to the variable 'errors' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'errors', filter_call_result_1037)
        
        # Assigning a Call to a Name (line 286):
        
        # Call to filter(...): (line 286)
        # Processing the call arguments (line 286)

        @norecursion
        def _stypy_temp_lambda_2(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_2'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_2', 286, 33, True)
            # Passed parameters checking function
            _stypy_temp_lambda_2.stypy_localization = localization
            _stypy_temp_lambda_2.stypy_type_of_self = None
            _stypy_temp_lambda_2.stypy_type_store = module_type_store
            _stypy_temp_lambda_2.stypy_function_name = '_stypy_temp_lambda_2'
            _stypy_temp_lambda_2.stypy_param_names_list = ['t']
            _stypy_temp_lambda_2.stypy_varargs_param_name = None
            _stypy_temp_lambda_2.stypy_kwargs_param_name = None
            _stypy_temp_lambda_2.stypy_call_defaults = defaults
            _stypy_temp_lambda_2.stypy_call_varargs = varargs
            _stypy_temp_lambda_2.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_2', ['t'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_2', ['t'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            
            # Call to isinstance(...): (line 286)
            # Processing the call arguments (line 286)
            # Getting the type of 't' (line 286)
            t_1040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 58), 't', False)
            # Getting the type of 'TypeError' (line 286)
            TypeError_1041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 61), 'TypeError', False)
            # Processing the call keyword arguments (line 286)
            kwargs_1042 = {}
            # Getting the type of 'isinstance' (line 286)
            isinstance_1039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 47), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 286)
            isinstance_call_result_1043 = invoke(stypy.reporting.localization.Localization(__file__, 286, 47), isinstance_1039, *[t_1040, TypeError_1041], **kwargs_1042)
            
            # Applying the 'not' unary operator (line 286)
            result_not__1044 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 43), 'not', isinstance_call_result_1043)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 286)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 33), 'stypy_return_type', result_not__1044)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_2' in the type store
            # Getting the type of 'stypy_return_type' (line 286)
            stypy_return_type_1045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 33), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_1045)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_2'
            return stypy_return_type_1045

        # Assigning a type to the variable '_stypy_temp_lambda_2' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 33), '_stypy_temp_lambda_2', _stypy_temp_lambda_2)
        # Getting the type of '_stypy_temp_lambda_2' (line 286)
        _stypy_temp_lambda_2_1046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 33), '_stypy_temp_lambda_2')
        # Getting the type of 'result' (line 286)
        result_1047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 73), 'result', False)
        # Processing the call keyword arguments (line 286)
        kwargs_1048 = {}
        # Getting the type of 'filter' (line 286)
        filter_1038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 26), 'filter', False)
        # Calling filter(args, kwargs) (line 286)
        filter_call_result_1049 = invoke(stypy.reporting.localization.Localization(__file__, 286, 26), filter_1038, *[_stypy_temp_lambda_2_1046, result_1047], **kwargs_1048)
        
        # Assigning a type to the variable 'types_to_return' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'types_to_return', filter_call_result_1049)
        
        
        # Call to len(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'errors' (line 289)
        errors_1051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 15), 'errors', False)
        # Processing the call keyword arguments (line 289)
        kwargs_1052 = {}
        # Getting the type of 'len' (line 289)
        len_1050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 11), 'len', False)
        # Calling len(args, kwargs) (line 289)
        len_call_result_1053 = invoke(stypy.reporting.localization.Localization(__file__, 289, 11), len_1050, *[errors_1051], **kwargs_1052)
        
        
        # Call to len(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'result' (line 289)
        result_1055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 30), 'result', False)
        # Processing the call keyword arguments (line 289)
        kwargs_1056 = {}
        # Getting the type of 'len' (line 289)
        len_1054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 26), 'len', False)
        # Calling len(args, kwargs) (line 289)
        len_call_result_1057 = invoke(stypy.reporting.localization.Localization(__file__, 289, 26), len_1054, *[result_1055], **kwargs_1056)
        
        # Applying the binary operator '==' (line 289)
        result_eq_1058 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 11), '==', len_call_result_1053, len_call_result_1057)
        
        # Testing if the type of an if condition is none (line 289)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 289, 8), result_eq_1058):
            
            
            # Call to len(...): (line 294)
            # Processing the call arguments (line 294)
            # Getting the type of 'errors' (line 294)
            errors_1072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 19), 'errors', False)
            # Processing the call keyword arguments (line 294)
            kwargs_1073 = {}
            # Getting the type of 'len' (line 294)
            len_1071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 15), 'len', False)
            # Calling len(args, kwargs) (line 294)
            len_call_result_1074 = invoke(stypy.reporting.localization.Localization(__file__, 294, 15), len_1071, *[errors_1072], **kwargs_1073)
            
            int_1075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 29), 'int')
            # Applying the binary operator '>' (line 294)
            result_gt_1076 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 15), '>', len_call_result_1074, int_1075)
            
            # Testing if the type of an if condition is none (line 294)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 294, 12), result_gt_1076):
                pass
            else:
                
                # Testing the type of an if condition (line 294)
                if_condition_1077 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 12), result_gt_1076)
                # Assigning a type to the variable 'if_condition_1077' (line 294)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'if_condition_1077', if_condition_1077)
                # SSA begins for if statement (line 294)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 295)
                # Processing the call arguments (line 295)
                
                # Call to UndefinedType(...): (line 295)
                # Processing the call keyword arguments (line 295)
                kwargs_1082 = {}
                # Getting the type of 'undefined_type_copy' (line 295)
                undefined_type_copy_1080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 39), 'undefined_type_copy', False)
                # Obtaining the member 'UndefinedType' of a type (line 295)
                UndefinedType_1081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 39), undefined_type_copy_1080, 'UndefinedType')
                # Calling UndefinedType(args, kwargs) (line 295)
                UndefinedType_call_result_1083 = invoke(stypy.reporting.localization.Localization(__file__, 295, 39), UndefinedType_1081, *[], **kwargs_1082)
                
                # Processing the call keyword arguments (line 295)
                kwargs_1084 = {}
                # Getting the type of 'types_to_return' (line 295)
                types_to_return_1078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'types_to_return', False)
                # Obtaining the member 'append' of a type (line 295)
                append_1079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 16), types_to_return_1078, 'append')
                # Calling append(args, kwargs) (line 295)
                append_call_result_1085 = invoke(stypy.reporting.localization.Localization(__file__, 295, 16), append_1079, *[UndefinedType_call_result_1083], **kwargs_1084)
                
                # SSA join for if statement (line 294)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'errors' (line 301)
            errors_1086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 25), 'errors')
            # Assigning a type to the variable 'errors_1086' (line 301)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'errors_1086', errors_1086)
            # Testing if the for loop is going to be iterated (line 301)
            # Testing the type of a for loop iterable (line 301)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 301, 12), errors_1086)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 301, 12), errors_1086):
                # Getting the type of the for loop variable (line 301)
                for_loop_var_1087 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 301, 12), errors_1086)
                # Assigning a type to the variable 'error' (line 301)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'error', for_loop_var_1087)
                # SSA begins for a for statement (line 301)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 302)
                # Processing the call keyword arguments (line 302)
                kwargs_1090 = {}
                # Getting the type of 'error' (line 302)
                error_1088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 302)
                turn_to_warning_1089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 16), error_1088, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 302)
                turn_to_warning_call_result_1091 = invoke(stypy.reporting.localization.Localization(__file__, 302, 16), turn_to_warning_1089, *[], **kwargs_1090)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 289)
            if_condition_1059 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 289, 8), result_eq_1058)
            # Assigning a type to the variable 'if_condition_1059' (line 289)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'if_condition_1059', if_condition_1059)
            # SSA begins for if statement (line 289)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 290)
            # Processing the call arguments (line 290)
            # Getting the type of 'localization' (line 290)
            localization_1061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 29), 'localization', False)
            
            # Call to format(...): (line 290)
            # Processing the call arguments (line 290)
            # Getting the type of 'member_name' (line 291)
            member_name_1064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 16), 'member_name', False)
            # Getting the type of 'self' (line 291)
            self_1065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 29), 'self', False)
            # Obtaining the member 'types' of a type (line 291)
            types_1066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 29), self_1065, 'types')
            # Processing the call keyword arguments (line 290)
            kwargs_1067 = {}
            str_1062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 43), 'str', "None of the possible types ('{1}') has the member '{0}'")
            # Obtaining the member 'format' of a type (line 290)
            format_1063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 43), str_1062, 'format')
            # Calling format(args, kwargs) (line 290)
            format_call_result_1068 = invoke(stypy.reporting.localization.Localization(__file__, 290, 43), format_1063, *[member_name_1064, types_1066], **kwargs_1067)
            
            # Processing the call keyword arguments (line 290)
            kwargs_1069 = {}
            # Getting the type of 'TypeError' (line 290)
            TypeError_1060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 290)
            TypeError_call_result_1070 = invoke(stypy.reporting.localization.Localization(__file__, 290, 19), TypeError_1060, *[localization_1061, format_call_result_1068], **kwargs_1069)
            
            # Assigning a type to the variable 'stypy_return_type' (line 290)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'stypy_return_type', TypeError_call_result_1070)
            # SSA branch for the else part of an if statement (line 289)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to len(...): (line 294)
            # Processing the call arguments (line 294)
            # Getting the type of 'errors' (line 294)
            errors_1072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 19), 'errors', False)
            # Processing the call keyword arguments (line 294)
            kwargs_1073 = {}
            # Getting the type of 'len' (line 294)
            len_1071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 15), 'len', False)
            # Calling len(args, kwargs) (line 294)
            len_call_result_1074 = invoke(stypy.reporting.localization.Localization(__file__, 294, 15), len_1071, *[errors_1072], **kwargs_1073)
            
            int_1075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 29), 'int')
            # Applying the binary operator '>' (line 294)
            result_gt_1076 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 15), '>', len_call_result_1074, int_1075)
            
            # Testing if the type of an if condition is none (line 294)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 294, 12), result_gt_1076):
                pass
            else:
                
                # Testing the type of an if condition (line 294)
                if_condition_1077 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 12), result_gt_1076)
                # Assigning a type to the variable 'if_condition_1077' (line 294)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'if_condition_1077', if_condition_1077)
                # SSA begins for if statement (line 294)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 295)
                # Processing the call arguments (line 295)
                
                # Call to UndefinedType(...): (line 295)
                # Processing the call keyword arguments (line 295)
                kwargs_1082 = {}
                # Getting the type of 'undefined_type_copy' (line 295)
                undefined_type_copy_1080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 39), 'undefined_type_copy', False)
                # Obtaining the member 'UndefinedType' of a type (line 295)
                UndefinedType_1081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 39), undefined_type_copy_1080, 'UndefinedType')
                # Calling UndefinedType(args, kwargs) (line 295)
                UndefinedType_call_result_1083 = invoke(stypy.reporting.localization.Localization(__file__, 295, 39), UndefinedType_1081, *[], **kwargs_1082)
                
                # Processing the call keyword arguments (line 295)
                kwargs_1084 = {}
                # Getting the type of 'types_to_return' (line 295)
                types_to_return_1078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'types_to_return', False)
                # Obtaining the member 'append' of a type (line 295)
                append_1079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 16), types_to_return_1078, 'append')
                # Calling append(args, kwargs) (line 295)
                append_call_result_1085 = invoke(stypy.reporting.localization.Localization(__file__, 295, 16), append_1079, *[UndefinedType_call_result_1083], **kwargs_1084)
                
                # SSA join for if statement (line 294)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'errors' (line 301)
            errors_1086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 25), 'errors')
            # Assigning a type to the variable 'errors_1086' (line 301)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'errors_1086', errors_1086)
            # Testing if the for loop is going to be iterated (line 301)
            # Testing the type of a for loop iterable (line 301)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 301, 12), errors_1086)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 301, 12), errors_1086):
                # Getting the type of the for loop variable (line 301)
                for_loop_var_1087 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 301, 12), errors_1086)
                # Assigning a type to the variable 'error' (line 301)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'error', for_loop_var_1087)
                # SSA begins for a for statement (line 301)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 302)
                # Processing the call keyword arguments (line 302)
                kwargs_1090 = {}
                # Getting the type of 'error' (line 302)
                error_1088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 302)
                turn_to_warning_1089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 16), error_1088, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 302)
                turn_to_warning_call_result_1091 = invoke(stypy.reporting.localization.Localization(__file__, 302, 16), turn_to_warning_1089, *[], **kwargs_1090)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 289)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to len(...): (line 306)
        # Processing the call arguments (line 306)
        # Getting the type of 'types_to_return' (line 306)
        types_to_return_1093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 15), 'types_to_return', False)
        # Processing the call keyword arguments (line 306)
        kwargs_1094 = {}
        # Getting the type of 'len' (line 306)
        len_1092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 11), 'len', False)
        # Calling len(args, kwargs) (line 306)
        len_call_result_1095 = invoke(stypy.reporting.localization.Localization(__file__, 306, 11), len_1092, *[types_to_return_1093], **kwargs_1094)
        
        int_1096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 35), 'int')
        # Applying the binary operator '==' (line 306)
        result_eq_1097 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 11), '==', len_call_result_1095, int_1096)
        
        # Testing if the type of an if condition is none (line 306)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 306, 8), result_eq_1097):
            
            # Assigning a Name to a Name (line 309):
            # Getting the type of 'None' (line 309)
            None_1103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 24), 'None')
            # Assigning a type to the variable 'ret_union' (line 309)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'ret_union', None_1103)
            
            # Getting the type of 'types_to_return' (line 310)
            types_to_return_1104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 25), 'types_to_return')
            # Assigning a type to the variable 'types_to_return_1104' (line 310)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'types_to_return_1104', types_to_return_1104)
            # Testing if the for loop is going to be iterated (line 310)
            # Testing the type of a for loop iterable (line 310)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 310, 12), types_to_return_1104)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 310, 12), types_to_return_1104):
                # Getting the type of the for loop variable (line 310)
                for_loop_var_1105 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 310, 12), types_to_return_1104)
                # Assigning a type to the variable 'type_' (line 310)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'type_', for_loop_var_1105)
                # SSA begins for a for statement (line 310)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Call to a Name (line 311):
                
                # Call to add(...): (line 311)
                # Processing the call arguments (line 311)
                # Getting the type of 'ret_union' (line 311)
                ret_union_1108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 42), 'ret_union', False)
                # Getting the type of 'type_' (line 311)
                type__1109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 53), 'type_', False)
                # Processing the call keyword arguments (line 311)
                kwargs_1110 = {}
                # Getting the type of 'UnionType' (line 311)
                UnionType_1106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 28), 'UnionType', False)
                # Obtaining the member 'add' of a type (line 311)
                add_1107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 28), UnionType_1106, 'add')
                # Calling add(args, kwargs) (line 311)
                add_call_result_1111 = invoke(stypy.reporting.localization.Localization(__file__, 311, 28), add_1107, *[ret_union_1108, type__1109], **kwargs_1110)
                
                # Assigning a type to the variable 'ret_union' (line 311)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 16), 'ret_union', add_call_result_1111)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Getting the type of 'ret_union' (line 313)
            ret_union_1112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 19), 'ret_union')
            # Assigning a type to the variable 'stypy_return_type' (line 313)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'stypy_return_type', ret_union_1112)
        else:
            
            # Testing the type of an if condition (line 306)
            if_condition_1098 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 306, 8), result_eq_1097)
            # Assigning a type to the variable 'if_condition_1098' (line 306)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'if_condition_1098', if_condition_1098)
            # SSA begins for if statement (line 306)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining the type of the subscript
            int_1099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 35), 'int')
            # Getting the type of 'types_to_return' (line 307)
            types_to_return_1100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 19), 'types_to_return')
            # Obtaining the member '__getitem__' of a type (line 307)
            getitem___1101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 19), types_to_return_1100, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 307)
            subscript_call_result_1102 = invoke(stypy.reporting.localization.Localization(__file__, 307, 19), getitem___1101, int_1099)
            
            # Assigning a type to the variable 'stypy_return_type' (line 307)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'stypy_return_type', subscript_call_result_1102)
            # SSA branch for the else part of an if statement (line 306)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Name (line 309):
            # Getting the type of 'None' (line 309)
            None_1103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 24), 'None')
            # Assigning a type to the variable 'ret_union' (line 309)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'ret_union', None_1103)
            
            # Getting the type of 'types_to_return' (line 310)
            types_to_return_1104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 25), 'types_to_return')
            # Assigning a type to the variable 'types_to_return_1104' (line 310)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'types_to_return_1104', types_to_return_1104)
            # Testing if the for loop is going to be iterated (line 310)
            # Testing the type of a for loop iterable (line 310)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 310, 12), types_to_return_1104)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 310, 12), types_to_return_1104):
                # Getting the type of the for loop variable (line 310)
                for_loop_var_1105 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 310, 12), types_to_return_1104)
                # Assigning a type to the variable 'type_' (line 310)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'type_', for_loop_var_1105)
                # SSA begins for a for statement (line 310)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Call to a Name (line 311):
                
                # Call to add(...): (line 311)
                # Processing the call arguments (line 311)
                # Getting the type of 'ret_union' (line 311)
                ret_union_1108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 42), 'ret_union', False)
                # Getting the type of 'type_' (line 311)
                type__1109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 53), 'type_', False)
                # Processing the call keyword arguments (line 311)
                kwargs_1110 = {}
                # Getting the type of 'UnionType' (line 311)
                UnionType_1106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 28), 'UnionType', False)
                # Obtaining the member 'add' of a type (line 311)
                add_1107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 28), UnionType_1106, 'add')
                # Calling add(args, kwargs) (line 311)
                add_call_result_1111 = invoke(stypy.reporting.localization.Localization(__file__, 311, 28), add_1107, *[ret_union_1108, type__1109], **kwargs_1110)
                
                # Assigning a type to the variable 'ret_union' (line 311)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 16), 'ret_union', add_call_result_1111)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Getting the type of 'ret_union' (line 313)
            ret_union_1112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 19), 'ret_union')
            # Assigning a type to the variable 'stypy_return_type' (line 313)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'stypy_return_type', ret_union_1112)
            # SSA join for if statement (line 306)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'get_type_of_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_type_of_member' in the type store
        # Getting the type of 'stypy_return_type' (line 266)
        stypy_return_type_1113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1113)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_type_of_member'
        return stypy_return_type_1113


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

        str_1114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, (-1)), 'str', '\n        When setting a member of a UnionType to a certain value, each one of the contained types are assigned this\n        member with the specified value (type). However, certain values have to be carefully handled to provide valid\n        values. For example, methods have to be handler in order to provide valid methods to add to each of the\n        UnionType types. This helper method convert a method to a valid method belonging to the destination object.\n\n        :param destination: New owner of the method\n        :param member_value: Method\n        :return THe passed member value, either transformed or not\n        ')
        
        # Call to ismethod(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'member_value' (line 327)
        member_value_1117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 28), 'member_value', False)
        # Processing the call keyword arguments (line 327)
        kwargs_1118 = {}
        # Getting the type of 'inspect' (line 327)
        inspect_1115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 11), 'inspect', False)
        # Obtaining the member 'ismethod' of a type (line 327)
        ismethod_1116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 11), inspect_1115, 'ismethod')
        # Calling ismethod(args, kwargs) (line 327)
        ismethod_call_result_1119 = invoke(stypy.reporting.localization.Localization(__file__, 327, 11), ismethod_1116, *[member_value_1117], **kwargs_1118)
        
        # Testing if the type of an if condition is none (line 327)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 327, 8), ismethod_call_result_1119):
            pass
        else:
            
            # Testing the type of an if condition (line 327)
            if_condition_1120 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 327, 8), ismethod_call_result_1119)
            # Assigning a type to the variable 'if_condition_1120' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'if_condition_1120', if_condition_1120)
            # SSA begins for if statement (line 327)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 329):
            
            # Call to MethodType(...): (line 329)
            # Processing the call arguments (line 329)
            # Getting the type of 'member_value' (line 329)
            member_value_1123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 35), 'member_value', False)
            # Obtaining the member 'im_func' of a type (line 329)
            im_func_1124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 35), member_value_1123, 'im_func')
            # Getting the type of 'destination' (line 329)
            destination_1125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 57), 'destination', False)
            # Processing the call keyword arguments (line 329)
            kwargs_1126 = {}
            # Getting the type of 'types' (line 329)
            types_1121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 18), 'types', False)
            # Obtaining the member 'MethodType' of a type (line 329)
            MethodType_1122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 18), types_1121, 'MethodType')
            # Calling MethodType(args, kwargs) (line 329)
            MethodType_call_result_1127 = invoke(stypy.reporting.localization.Localization(__file__, 329, 18), MethodType_1122, *[im_func_1124, destination_1125], **kwargs_1126)
            
            # Assigning a type to the variable 'met' (line 329)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'met', MethodType_call_result_1127)
            # Getting the type of 'met' (line 330)
            met_1128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 19), 'met')
            # Assigning a type to the variable 'stypy_return_type' (line 330)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'stypy_return_type', met_1128)
            # SSA join for if statement (line 327)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'member_value' (line 332)
        member_value_1129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 15), 'member_value')
        # Assigning a type to the variable 'stypy_return_type' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'stypy_return_type', member_value_1129)
        
        # ################# End of '__parse_member_value(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__parse_member_value' in the type store
        # Getting the type of 'stypy_return_type' (line 315)
        stypy_return_type_1130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1130)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__parse_member_value'
        return stypy_return_type_1130


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

        str_1131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, (-1)), 'str', '\n        For all the types stored in the union type, set the type of the member named member_name to the type\n        specified in member_value. For example,\n        if a UnionType has the types Class1 and Class2, both with the member "attr" so Class1.attr: int and\n        Class2.attr: str, this method, if passsed a float as member_value will turn both classes "attr" to float.\n        :param localization: Caller information\n        :param member_name: Name of the member to set\n        :param member_value New type of the member\n        :return None or a TypeError if the member cannot be set. Warnings are generated if the member of some of the\n        stored objects cannot be set\n        ')
        
        # Assigning a List to a Name (line 347):
        
        # Obtaining an instance of the builtin type 'list' (line 347)
        list_1132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 347)
        
        # Assigning a type to the variable 'errors' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'errors', list_1132)
        
        # Getting the type of 'self' (line 349)
        self_1133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 21), 'self')
        # Obtaining the member 'types' of a type (line 349)
        types_1134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 21), self_1133, 'types')
        # Assigning a type to the variable 'types_1134' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'types_1134', types_1134)
        # Testing if the for loop is going to be iterated (line 349)
        # Testing the type of a for loop iterable (line 349)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 349, 8), types_1134)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 349, 8), types_1134):
            # Getting the type of the for loop variable (line 349)
            for_loop_var_1135 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 349, 8), types_1134)
            # Assigning a type to the variable 'type_' (line 349)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'type_', for_loop_var_1135)
            # SSA begins for a for statement (line 349)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 350):
            
            # Call to __parse_member_value(...): (line 350)
            # Processing the call arguments (line 350)
            # Getting the type of 'type_' (line 350)
            type__1138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 52), 'type_', False)
            # Getting the type of 'member_value' (line 350)
            member_value_1139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 59), 'member_value', False)
            # Processing the call keyword arguments (line 350)
            kwargs_1140 = {}
            # Getting the type of 'self' (line 350)
            self_1136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 26), 'self', False)
            # Obtaining the member '__parse_member_value' of a type (line 350)
            parse_member_value_1137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 26), self_1136, '__parse_member_value')
            # Calling __parse_member_value(args, kwargs) (line 350)
            parse_member_value_call_result_1141 = invoke(stypy.reporting.localization.Localization(__file__, 350, 26), parse_member_value_1137, *[type__1138, member_value_1139], **kwargs_1140)
            
            # Assigning a type to the variable 'final_value' (line 350)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'final_value', parse_member_value_call_result_1141)
            
            # Assigning a Call to a Name (line 351):
            
            # Call to set_type_of_member(...): (line 351)
            # Processing the call arguments (line 351)
            # Getting the type of 'localization' (line 351)
            localization_1144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 44), 'localization', False)
            # Getting the type of 'member_name' (line 351)
            member_name_1145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 58), 'member_name', False)
            # Getting the type of 'final_value' (line 351)
            final_value_1146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 71), 'final_value', False)
            # Processing the call keyword arguments (line 351)
            kwargs_1147 = {}
            # Getting the type of 'type_' (line 351)
            type__1142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 19), 'type_', False)
            # Obtaining the member 'set_type_of_member' of a type (line 351)
            set_type_of_member_1143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 19), type__1142, 'set_type_of_member')
            # Calling set_type_of_member(args, kwargs) (line 351)
            set_type_of_member_call_result_1148 = invoke(stypy.reporting.localization.Localization(__file__, 351, 19), set_type_of_member_1143, *[localization_1144, member_name_1145, final_value_1146], **kwargs_1147)
            
            # Assigning a type to the variable 'temp' (line 351)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'temp', set_type_of_member_call_result_1148)
            
            # Type idiom detected: calculating its left and rigth part (line 352)
            # Getting the type of 'temp' (line 352)
            temp_1149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'temp')
            # Getting the type of 'None' (line 352)
            None_1150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 27), 'None')
            
            (may_be_1151, more_types_in_union_1152) = may_not_be_none(temp_1149, None_1150)

            if may_be_1151:

                if more_types_in_union_1152:
                    # Runtime conditional SSA (line 352)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to append(...): (line 353)
                # Processing the call arguments (line 353)
                # Getting the type of 'temp' (line 353)
                temp_1155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 30), 'temp', False)
                # Processing the call keyword arguments (line 353)
                kwargs_1156 = {}
                # Getting the type of 'errors' (line 353)
                errors_1153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 16), 'errors', False)
                # Obtaining the member 'append' of a type (line 353)
                append_1154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 16), errors_1153, 'append')
                # Calling append(args, kwargs) (line 353)
                append_call_result_1157 = invoke(stypy.reporting.localization.Localization(__file__, 353, 16), append_1154, *[temp_1155], **kwargs_1156)
                

                if more_types_in_union_1152:
                    # SSA join for if statement (line 352)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'errors' (line 356)
        errors_1159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 15), 'errors', False)
        # Processing the call keyword arguments (line 356)
        kwargs_1160 = {}
        # Getting the type of 'len' (line 356)
        len_1158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 11), 'len', False)
        # Calling len(args, kwargs) (line 356)
        len_call_result_1161 = invoke(stypy.reporting.localization.Localization(__file__, 356, 11), len_1158, *[errors_1159], **kwargs_1160)
        
        
        # Call to len(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'self' (line 356)
        self_1163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 356)
        types_1164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 30), self_1163, 'types')
        # Processing the call keyword arguments (line 356)
        kwargs_1165 = {}
        # Getting the type of 'len' (line 356)
        len_1162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 26), 'len', False)
        # Calling len(args, kwargs) (line 356)
        len_call_result_1166 = invoke(stypy.reporting.localization.Localization(__file__, 356, 26), len_1162, *[types_1164], **kwargs_1165)
        
        # Applying the binary operator '==' (line 356)
        result_eq_1167 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 11), '==', len_call_result_1161, len_call_result_1166)
        
        # Testing if the type of an if condition is none (line 356)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 356, 8), result_eq_1167):
            
            # Getting the type of 'errors' (line 363)
            errors_1180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 25), 'errors')
            # Assigning a type to the variable 'errors_1180' (line 363)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'errors_1180', errors_1180)
            # Testing if the for loop is going to be iterated (line 363)
            # Testing the type of a for loop iterable (line 363)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 363, 12), errors_1180)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 363, 12), errors_1180):
                # Getting the type of the for loop variable (line 363)
                for_loop_var_1181 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 363, 12), errors_1180)
                # Assigning a type to the variable 'error' (line 363)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'error', for_loop_var_1181)
                # SSA begins for a for statement (line 363)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 364)
                # Processing the call keyword arguments (line 364)
                kwargs_1184 = {}
                # Getting the type of 'error' (line 364)
                error_1182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 364)
                turn_to_warning_1183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 16), error_1182, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 364)
                turn_to_warning_call_result_1185 = invoke(stypy.reporting.localization.Localization(__file__, 364, 16), turn_to_warning_1183, *[], **kwargs_1184)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 356)
            if_condition_1168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 356, 8), result_eq_1167)
            # Assigning a type to the variable 'if_condition_1168' (line 356)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'if_condition_1168', if_condition_1168)
            # SSA begins for if statement (line 356)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 357)
            # Processing the call arguments (line 357)
            # Getting the type of 'localization' (line 357)
            localization_1170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 29), 'localization', False)
            
            # Call to format(...): (line 357)
            # Processing the call arguments (line 357)
            # Getting the type of 'member_name' (line 358)
            member_name_1173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 16), 'member_name', False)
            # Getting the type of 'self' (line 358)
            self_1174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 29), 'self', False)
            # Obtaining the member 'types' of a type (line 358)
            types_1175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 29), self_1174, 'types')
            # Processing the call keyword arguments (line 357)
            kwargs_1176 = {}
            str_1171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 43), 'str', "None of the possible types ('{1}') can set the member '{0}'")
            # Obtaining the member 'format' of a type (line 357)
            format_1172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 43), str_1171, 'format')
            # Calling format(args, kwargs) (line 357)
            format_call_result_1177 = invoke(stypy.reporting.localization.Localization(__file__, 357, 43), format_1172, *[member_name_1173, types_1175], **kwargs_1176)
            
            # Processing the call keyword arguments (line 357)
            kwargs_1178 = {}
            # Getting the type of 'TypeError' (line 357)
            TypeError_1169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 357)
            TypeError_call_result_1179 = invoke(stypy.reporting.localization.Localization(__file__, 357, 19), TypeError_1169, *[localization_1170, format_call_result_1177], **kwargs_1178)
            
            # Assigning a type to the variable 'stypy_return_type' (line 357)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'stypy_return_type', TypeError_call_result_1179)
            # SSA branch for the else part of an if statement (line 356)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 363)
            errors_1180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 25), 'errors')
            # Assigning a type to the variable 'errors_1180' (line 363)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'errors_1180', errors_1180)
            # Testing if the for loop is going to be iterated (line 363)
            # Testing the type of a for loop iterable (line 363)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 363, 12), errors_1180)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 363, 12), errors_1180):
                # Getting the type of the for loop variable (line 363)
                for_loop_var_1181 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 363, 12), errors_1180)
                # Assigning a type to the variable 'error' (line 363)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'error', for_loop_var_1181)
                # SSA begins for a for statement (line 363)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 364)
                # Processing the call keyword arguments (line 364)
                kwargs_1184 = {}
                # Getting the type of 'error' (line 364)
                error_1182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 364)
                turn_to_warning_1183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 16), error_1182, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 364)
                turn_to_warning_call_result_1185 = invoke(stypy.reporting.localization.Localization(__file__, 364, 16), turn_to_warning_1183, *[], **kwargs_1184)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 356)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'None' (line 366)
        None_1186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'stypy_return_type', None_1186)
        
        # ################# End of 'set_type_of_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_type_of_member' in the type store
        # Getting the type of 'stypy_return_type' (line 334)
        stypy_return_type_1187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1187)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_type_of_member'
        return stypy_return_type_1187


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

        str_1188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, (-1)), 'str', '\n        For all the types stored in the union type, invoke them with the provided parameters.\n        :param localization: Caller information\n        :param args: Arguments of the call\n        :param kwargs: Keyword arguments of the call\n        :return All the types that the call could return, examining the UnionType stored types\n        ')
        
        # Assigning a List to a Name (line 378):
        
        # Obtaining an instance of the builtin type 'list' (line 378)
        list_1189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 378)
        
        # Assigning a type to the variable 'result' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'result', list_1189)
        
        # Getting the type of 'self' (line 380)
        self_1190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 21), 'self')
        # Obtaining the member 'types' of a type (line 380)
        types_1191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 21), self_1190, 'types')
        # Assigning a type to the variable 'types_1191' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'types_1191', types_1191)
        # Testing if the for loop is going to be iterated (line 380)
        # Testing the type of a for loop iterable (line 380)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 380, 8), types_1191)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 380, 8), types_1191):
            # Getting the type of the for loop variable (line 380)
            for_loop_var_1192 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 380, 8), types_1191)
            # Assigning a type to the variable 'type_' (line 380)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'type_', for_loop_var_1192)
            # SSA begins for a for statement (line 380)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 382):
            
            # Call to invoke(...): (line 382)
            # Processing the call arguments (line 382)
            # Getting the type of 'localization' (line 382)
            localization_1195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 32), 'localization', False)
            # Getting the type of 'args' (line 382)
            args_1196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 47), 'args', False)
            # Processing the call keyword arguments (line 382)
            # Getting the type of 'kwargs' (line 382)
            kwargs_1197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 55), 'kwargs', False)
            kwargs_1198 = {'kwargs_1197': kwargs_1197}
            # Getting the type of 'type_' (line 382)
            type__1193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 19), 'type_', False)
            # Obtaining the member 'invoke' of a type (line 382)
            invoke_1194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 19), type__1193, 'invoke')
            # Calling invoke(args, kwargs) (line 382)
            invoke_call_result_1199 = invoke(stypy.reporting.localization.Localization(__file__, 382, 19), invoke_1194, *[localization_1195, args_1196], **kwargs_1198)
            
            # Assigning a type to the variable 'temp' (line 382)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'temp', invoke_call_result_1199)
            
            # Call to append(...): (line 383)
            # Processing the call arguments (line 383)
            # Getting the type of 'temp' (line 383)
            temp_1202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 26), 'temp', False)
            # Processing the call keyword arguments (line 383)
            kwargs_1203 = {}
            # Getting the type of 'result' (line 383)
            result_1200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'result', False)
            # Obtaining the member 'append' of a type (line 383)
            append_1201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 12), result_1200, 'append')
            # Calling append(args, kwargs) (line 383)
            append_call_result_1204 = invoke(stypy.reporting.localization.Localization(__file__, 383, 12), append_1201, *[temp_1202], **kwargs_1203)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 386):
        
        # Call to filter(...): (line 386)
        # Processing the call arguments (line 386)

        @norecursion
        def _stypy_temp_lambda_3(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_3'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_3', 386, 24, True)
            # Passed parameters checking function
            _stypy_temp_lambda_3.stypy_localization = localization
            _stypy_temp_lambda_3.stypy_type_of_self = None
            _stypy_temp_lambda_3.stypy_type_store = module_type_store
            _stypy_temp_lambda_3.stypy_function_name = '_stypy_temp_lambda_3'
            _stypy_temp_lambda_3.stypy_param_names_list = ['t']
            _stypy_temp_lambda_3.stypy_varargs_param_name = None
            _stypy_temp_lambda_3.stypy_kwargs_param_name = None
            _stypy_temp_lambda_3.stypy_call_defaults = defaults
            _stypy_temp_lambda_3.stypy_call_varargs = varargs
            _stypy_temp_lambda_3.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_3', ['t'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_3', ['t'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to isinstance(...): (line 386)
            # Processing the call arguments (line 386)
            # Getting the type of 't' (line 386)
            t_1207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 45), 't', False)
            # Getting the type of 'TypeError' (line 386)
            TypeError_1208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 48), 'TypeError', False)
            # Processing the call keyword arguments (line 386)
            kwargs_1209 = {}
            # Getting the type of 'isinstance' (line 386)
            isinstance_1206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 34), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 386)
            isinstance_call_result_1210 = invoke(stypy.reporting.localization.Localization(__file__, 386, 34), isinstance_1206, *[t_1207, TypeError_1208], **kwargs_1209)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 386)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 24), 'stypy_return_type', isinstance_call_result_1210)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_3' in the type store
            # Getting the type of 'stypy_return_type' (line 386)
            stypy_return_type_1211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 24), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_1211)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_3'
            return stypy_return_type_1211

        # Assigning a type to the variable '_stypy_temp_lambda_3' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 24), '_stypy_temp_lambda_3', _stypy_temp_lambda_3)
        # Getting the type of '_stypy_temp_lambda_3' (line 386)
        _stypy_temp_lambda_3_1212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 24), '_stypy_temp_lambda_3')
        # Getting the type of 'result' (line 386)
        result_1213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 60), 'result', False)
        # Processing the call keyword arguments (line 386)
        kwargs_1214 = {}
        # Getting the type of 'filter' (line 386)
        filter_1205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 17), 'filter', False)
        # Calling filter(args, kwargs) (line 386)
        filter_call_result_1215 = invoke(stypy.reporting.localization.Localization(__file__, 386, 17), filter_1205, *[_stypy_temp_lambda_3_1212, result_1213], **kwargs_1214)
        
        # Assigning a type to the variable 'errors' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'errors', filter_call_result_1215)
        
        # Assigning a Call to a Name (line 389):
        
        # Call to filter(...): (line 389)
        # Processing the call arguments (line 389)

        @norecursion
        def _stypy_temp_lambda_4(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_4'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_4', 389, 33, True)
            # Passed parameters checking function
            _stypy_temp_lambda_4.stypy_localization = localization
            _stypy_temp_lambda_4.stypy_type_of_self = None
            _stypy_temp_lambda_4.stypy_type_store = module_type_store
            _stypy_temp_lambda_4.stypy_function_name = '_stypy_temp_lambda_4'
            _stypy_temp_lambda_4.stypy_param_names_list = ['t']
            _stypy_temp_lambda_4.stypy_varargs_param_name = None
            _stypy_temp_lambda_4.stypy_kwargs_param_name = None
            _stypy_temp_lambda_4.stypy_call_defaults = defaults
            _stypy_temp_lambda_4.stypy_call_varargs = varargs
            _stypy_temp_lambda_4.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_4', ['t'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_4', ['t'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            
            # Call to isinstance(...): (line 389)
            # Processing the call arguments (line 389)
            # Getting the type of 't' (line 389)
            t_1218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 58), 't', False)
            # Getting the type of 'TypeError' (line 389)
            TypeError_1219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 61), 'TypeError', False)
            # Processing the call keyword arguments (line 389)
            kwargs_1220 = {}
            # Getting the type of 'isinstance' (line 389)
            isinstance_1217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 47), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 389)
            isinstance_call_result_1221 = invoke(stypy.reporting.localization.Localization(__file__, 389, 47), isinstance_1217, *[t_1218, TypeError_1219], **kwargs_1220)
            
            # Applying the 'not' unary operator (line 389)
            result_not__1222 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 43), 'not', isinstance_call_result_1221)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 389)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 33), 'stypy_return_type', result_not__1222)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_4' in the type store
            # Getting the type of 'stypy_return_type' (line 389)
            stypy_return_type_1223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 33), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_1223)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_4'
            return stypy_return_type_1223

        # Assigning a type to the variable '_stypy_temp_lambda_4' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 33), '_stypy_temp_lambda_4', _stypy_temp_lambda_4)
        # Getting the type of '_stypy_temp_lambda_4' (line 389)
        _stypy_temp_lambda_4_1224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 33), '_stypy_temp_lambda_4')
        # Getting the type of 'result' (line 389)
        result_1225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 73), 'result', False)
        # Processing the call keyword arguments (line 389)
        kwargs_1226 = {}
        # Getting the type of 'filter' (line 389)
        filter_1216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 26), 'filter', False)
        # Calling filter(args, kwargs) (line 389)
        filter_call_result_1227 = invoke(stypy.reporting.localization.Localization(__file__, 389, 26), filter_1216, *[_stypy_temp_lambda_4_1224, result_1225], **kwargs_1226)
        
        # Assigning a type to the variable 'types_to_return' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'types_to_return', filter_call_result_1227)
        
        
        # Call to len(...): (line 392)
        # Processing the call arguments (line 392)
        # Getting the type of 'errors' (line 392)
        errors_1229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 15), 'errors', False)
        # Processing the call keyword arguments (line 392)
        kwargs_1230 = {}
        # Getting the type of 'len' (line 392)
        len_1228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 11), 'len', False)
        # Calling len(args, kwargs) (line 392)
        len_call_result_1231 = invoke(stypy.reporting.localization.Localization(__file__, 392, 11), len_1228, *[errors_1229], **kwargs_1230)
        
        
        # Call to len(...): (line 392)
        # Processing the call arguments (line 392)
        # Getting the type of 'result' (line 392)
        result_1233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 30), 'result', False)
        # Processing the call keyword arguments (line 392)
        kwargs_1234 = {}
        # Getting the type of 'len' (line 392)
        len_1232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 26), 'len', False)
        # Calling len(args, kwargs) (line 392)
        len_call_result_1235 = invoke(stypy.reporting.localization.Localization(__file__, 392, 26), len_1232, *[result_1233], **kwargs_1234)
        
        # Applying the binary operator '==' (line 392)
        result_eq_1236 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 11), '==', len_call_result_1231, len_call_result_1235)
        
        # Testing if the type of an if condition is none (line 392)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 392, 8), result_eq_1236):
            
            # Getting the type of 'errors' (line 402)
            errors_1275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 25), 'errors')
            # Assigning a type to the variable 'errors_1275' (line 402)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'errors_1275', errors_1275)
            # Testing if the for loop is going to be iterated (line 402)
            # Testing the type of a for loop iterable (line 402)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 402, 12), errors_1275)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 402, 12), errors_1275):
                # Getting the type of the for loop variable (line 402)
                for_loop_var_1276 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 402, 12), errors_1275)
                # Assigning a type to the variable 'error' (line 402)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'error', for_loop_var_1276)
                # SSA begins for a for statement (line 402)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 403)
                # Processing the call keyword arguments (line 403)
                kwargs_1279 = {}
                # Getting the type of 'error' (line 403)
                error_1277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 403)
                turn_to_warning_1278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 16), error_1277, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 403)
                turn_to_warning_call_result_1280 = invoke(stypy.reporting.localization.Localization(__file__, 403, 16), turn_to_warning_1278, *[], **kwargs_1279)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 392)
            if_condition_1237 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 392, 8), result_eq_1236)
            # Assigning a type to the variable 'if_condition_1237' (line 392)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'if_condition_1237', if_condition_1237)
            # SSA begins for if statement (line 392)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'errors' (line 393)
            errors_1238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 25), 'errors')
            # Assigning a type to the variable 'errors_1238' (line 393)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'errors_1238', errors_1238)
            # Testing if the for loop is going to be iterated (line 393)
            # Testing the type of a for loop iterable (line 393)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 393, 12), errors_1238)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 393, 12), errors_1238):
                # Getting the type of the for loop variable (line 393)
                for_loop_var_1239 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 393, 12), errors_1238)
                # Assigning a type to the variable 'error' (line 393)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'error', for_loop_var_1239)
                # SSA begins for a for statement (line 393)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to remove_error_msg(...): (line 394)
                # Processing the call arguments (line 394)
                # Getting the type of 'error' (line 394)
                error_1242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 43), 'error', False)
                # Processing the call keyword arguments (line 394)
                kwargs_1243 = {}
                # Getting the type of 'TypeError' (line 394)
                TypeError_1240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 16), 'TypeError', False)
                # Obtaining the member 'remove_error_msg' of a type (line 394)
                remove_error_msg_1241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 16), TypeError_1240, 'remove_error_msg')
                # Calling remove_error_msg(args, kwargs) (line 394)
                remove_error_msg_call_result_1244 = invoke(stypy.reporting.localization.Localization(__file__, 394, 16), remove_error_msg_1241, *[error_1242], **kwargs_1243)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a Call to a Name (line 395):
            
            # Call to tuple(...): (line 395)
            # Processing the call arguments (line 395)
            
            # Call to list(...): (line 395)
            # Processing the call arguments (line 395)
            # Getting the type of 'args' (line 395)
            args_1247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 32), 'args', False)
            # Processing the call keyword arguments (line 395)
            kwargs_1248 = {}
            # Getting the type of 'list' (line 395)
            list_1246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 27), 'list', False)
            # Calling list(args, kwargs) (line 395)
            list_call_result_1249 = invoke(stypy.reporting.localization.Localization(__file__, 395, 27), list_1246, *[args_1247], **kwargs_1248)
            
            
            # Call to values(...): (line 395)
            # Processing the call keyword arguments (line 395)
            kwargs_1252 = {}
            # Getting the type of 'kwargs' (line 395)
            kwargs_1250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 40), 'kwargs', False)
            # Obtaining the member 'values' of a type (line 395)
            values_1251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 40), kwargs_1250, 'values')
            # Calling values(args, kwargs) (line 395)
            values_call_result_1253 = invoke(stypy.reporting.localization.Localization(__file__, 395, 40), values_1251, *[], **kwargs_1252)
            
            # Applying the binary operator '+' (line 395)
            result_add_1254 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 27), '+', list_call_result_1249, values_call_result_1253)
            
            # Processing the call keyword arguments (line 395)
            kwargs_1255 = {}
            # Getting the type of 'tuple' (line 395)
            tuple_1245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 21), 'tuple', False)
            # Calling tuple(args, kwargs) (line 395)
            tuple_call_result_1256 = invoke(stypy.reporting.localization.Localization(__file__, 395, 21), tuple_1245, *[result_add_1254], **kwargs_1255)
            
            # Assigning a type to the variable 'params' (line 395)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'params', tuple_call_result_1256)
            
            # Call to TypeError(...): (line 396)
            # Processing the call arguments (line 396)
            # Getting the type of 'localization' (line 396)
            localization_1258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 29), 'localization', False)
            
            # Call to format(...): (line 396)
            # Processing the call arguments (line 396)
            
            # Call to format_function_name(...): (line 397)
            # Processing the call arguments (line 397)
            
            # Obtaining the type of the subscript
            int_1262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 48), 'int')
            # Getting the type of 'self' (line 397)
            self_1263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 37), 'self', False)
            # Obtaining the member 'types' of a type (line 397)
            types_1264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 37), self_1263, 'types')
            # Obtaining the member '__getitem__' of a type (line 397)
            getitem___1265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 37), types_1264, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 397)
            subscript_call_result_1266 = invoke(stypy.reporting.localization.Localization(__file__, 397, 37), getitem___1265, int_1262)
            
            # Obtaining the member 'name' of a type (line 397)
            name_1267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 37), subscript_call_result_1266, 'name')
            # Processing the call keyword arguments (line 397)
            kwargs_1268 = {}
            # Getting the type of 'format_function_name' (line 397)
            format_function_name_1261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 16), 'format_function_name', False)
            # Calling format_function_name(args, kwargs) (line 397)
            format_function_name_call_result_1269 = invoke(stypy.reporting.localization.Localization(__file__, 397, 16), format_function_name_1261, *[name_1267], **kwargs_1268)
            
            # Getting the type of 'params' (line 397)
            params_1270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 58), 'params', False)
            # Processing the call keyword arguments (line 396)
            kwargs_1271 = {}
            str_1259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 43), 'str', 'Cannot invoke {0} with parameters {1}')
            # Obtaining the member 'format' of a type (line 396)
            format_1260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 43), str_1259, 'format')
            # Calling format(args, kwargs) (line 396)
            format_call_result_1272 = invoke(stypy.reporting.localization.Localization(__file__, 396, 43), format_1260, *[format_function_name_call_result_1269, params_1270], **kwargs_1271)
            
            # Processing the call keyword arguments (line 396)
            kwargs_1273 = {}
            # Getting the type of 'TypeError' (line 396)
            TypeError_1257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 396)
            TypeError_call_result_1274 = invoke(stypy.reporting.localization.Localization(__file__, 396, 19), TypeError_1257, *[localization_1258, format_call_result_1272], **kwargs_1273)
            
            # Assigning a type to the variable 'stypy_return_type' (line 396)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'stypy_return_type', TypeError_call_result_1274)
            # SSA branch for the else part of an if statement (line 392)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 402)
            errors_1275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 25), 'errors')
            # Assigning a type to the variable 'errors_1275' (line 402)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'errors_1275', errors_1275)
            # Testing if the for loop is going to be iterated (line 402)
            # Testing the type of a for loop iterable (line 402)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 402, 12), errors_1275)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 402, 12), errors_1275):
                # Getting the type of the for loop variable (line 402)
                for_loop_var_1276 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 402, 12), errors_1275)
                # Assigning a type to the variable 'error' (line 402)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'error', for_loop_var_1276)
                # SSA begins for a for statement (line 402)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 403)
                # Processing the call keyword arguments (line 403)
                kwargs_1279 = {}
                # Getting the type of 'error' (line 403)
                error_1277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 403)
                turn_to_warning_1278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 16), error_1277, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 403)
                turn_to_warning_call_result_1280 = invoke(stypy.reporting.localization.Localization(__file__, 403, 16), turn_to_warning_1278, *[], **kwargs_1279)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 392)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to len(...): (line 406)
        # Processing the call arguments (line 406)
        # Getting the type of 'types_to_return' (line 406)
        types_to_return_1282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 15), 'types_to_return', False)
        # Processing the call keyword arguments (line 406)
        kwargs_1283 = {}
        # Getting the type of 'len' (line 406)
        len_1281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 11), 'len', False)
        # Calling len(args, kwargs) (line 406)
        len_call_result_1284 = invoke(stypy.reporting.localization.Localization(__file__, 406, 11), len_1281, *[types_to_return_1282], **kwargs_1283)
        
        int_1285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 35), 'int')
        # Applying the binary operator '==' (line 406)
        result_eq_1286 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 11), '==', len_call_result_1284, int_1285)
        
        # Testing if the type of an if condition is none (line 406)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 406, 8), result_eq_1286):
            
            # Assigning a Name to a Name (line 409):
            # Getting the type of 'None' (line 409)
            None_1292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 24), 'None')
            # Assigning a type to the variable 'ret_union' (line 409)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'ret_union', None_1292)
            
            # Getting the type of 'types_to_return' (line 410)
            types_to_return_1293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 25), 'types_to_return')
            # Assigning a type to the variable 'types_to_return_1293' (line 410)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'types_to_return_1293', types_to_return_1293)
            # Testing if the for loop is going to be iterated (line 410)
            # Testing the type of a for loop iterable (line 410)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 410, 12), types_to_return_1293)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 410, 12), types_to_return_1293):
                # Getting the type of the for loop variable (line 410)
                for_loop_var_1294 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 410, 12), types_to_return_1293)
                # Assigning a type to the variable 'type_' (line 410)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'type_', for_loop_var_1294)
                # SSA begins for a for statement (line 410)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Call to a Name (line 411):
                
                # Call to add(...): (line 411)
                # Processing the call arguments (line 411)
                # Getting the type of 'ret_union' (line 411)
                ret_union_1297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 42), 'ret_union', False)
                # Getting the type of 'type_' (line 411)
                type__1298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 53), 'type_', False)
                # Processing the call keyword arguments (line 411)
                kwargs_1299 = {}
                # Getting the type of 'UnionType' (line 411)
                UnionType_1295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 28), 'UnionType', False)
                # Obtaining the member 'add' of a type (line 411)
                add_1296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 28), UnionType_1295, 'add')
                # Calling add(args, kwargs) (line 411)
                add_call_result_1300 = invoke(stypy.reporting.localization.Localization(__file__, 411, 28), add_1296, *[ret_union_1297, type__1298], **kwargs_1299)
                
                # Assigning a type to the variable 'ret_union' (line 411)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 16), 'ret_union', add_call_result_1300)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Getting the type of 'ret_union' (line 413)
            ret_union_1301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 19), 'ret_union')
            # Assigning a type to the variable 'stypy_return_type' (line 413)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'stypy_return_type', ret_union_1301)
        else:
            
            # Testing the type of an if condition (line 406)
            if_condition_1287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 406, 8), result_eq_1286)
            # Assigning a type to the variable 'if_condition_1287' (line 406)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'if_condition_1287', if_condition_1287)
            # SSA begins for if statement (line 406)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining the type of the subscript
            int_1288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 35), 'int')
            # Getting the type of 'types_to_return' (line 407)
            types_to_return_1289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 19), 'types_to_return')
            # Obtaining the member '__getitem__' of a type (line 407)
            getitem___1290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 19), types_to_return_1289, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 407)
            subscript_call_result_1291 = invoke(stypy.reporting.localization.Localization(__file__, 407, 19), getitem___1290, int_1288)
            
            # Assigning a type to the variable 'stypy_return_type' (line 407)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'stypy_return_type', subscript_call_result_1291)
            # SSA branch for the else part of an if statement (line 406)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Name (line 409):
            # Getting the type of 'None' (line 409)
            None_1292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 24), 'None')
            # Assigning a type to the variable 'ret_union' (line 409)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'ret_union', None_1292)
            
            # Getting the type of 'types_to_return' (line 410)
            types_to_return_1293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 25), 'types_to_return')
            # Assigning a type to the variable 'types_to_return_1293' (line 410)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'types_to_return_1293', types_to_return_1293)
            # Testing if the for loop is going to be iterated (line 410)
            # Testing the type of a for loop iterable (line 410)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 410, 12), types_to_return_1293)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 410, 12), types_to_return_1293):
                # Getting the type of the for loop variable (line 410)
                for_loop_var_1294 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 410, 12), types_to_return_1293)
                # Assigning a type to the variable 'type_' (line 410)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'type_', for_loop_var_1294)
                # SSA begins for a for statement (line 410)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Call to a Name (line 411):
                
                # Call to add(...): (line 411)
                # Processing the call arguments (line 411)
                # Getting the type of 'ret_union' (line 411)
                ret_union_1297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 42), 'ret_union', False)
                # Getting the type of 'type_' (line 411)
                type__1298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 53), 'type_', False)
                # Processing the call keyword arguments (line 411)
                kwargs_1299 = {}
                # Getting the type of 'UnionType' (line 411)
                UnionType_1295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 28), 'UnionType', False)
                # Obtaining the member 'add' of a type (line 411)
                add_1296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 28), UnionType_1295, 'add')
                # Calling add(args, kwargs) (line 411)
                add_call_result_1300 = invoke(stypy.reporting.localization.Localization(__file__, 411, 28), add_1296, *[ret_union_1297, type__1298], **kwargs_1299)
                
                # Assigning a type to the variable 'ret_union' (line 411)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 16), 'ret_union', add_call_result_1300)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Getting the type of 'ret_union' (line 413)
            ret_union_1301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 19), 'ret_union')
            # Assigning a type to the variable 'stypy_return_type' (line 413)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'stypy_return_type', ret_union_1301)
            # SSA join for if statement (line 406)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'invoke(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'invoke' in the type store
        # Getting the type of 'stypy_return_type' (line 370)
        stypy_return_type_1302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1302)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'invoke'
        return stypy_return_type_1302


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

        str_1303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, (-1)), 'str', '\n        For all the types stored in the union type, delete the member named member_name, returning None or a TypeError\n        if no type stored in the UnionType supports member deletion.\n        :param localization: Caller information\n        :param member: Member to delete\n        :return None or TypeError\n        ')
        
        # Assigning a List to a Name (line 425):
        
        # Obtaining an instance of the builtin type 'list' (line 425)
        list_1304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 425)
        
        # Assigning a type to the variable 'errors' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'errors', list_1304)
        
        # Getting the type of 'self' (line 427)
        self_1305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 21), 'self')
        # Obtaining the member 'types' of a type (line 427)
        types_1306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 21), self_1305, 'types')
        # Assigning a type to the variable 'types_1306' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'types_1306', types_1306)
        # Testing if the for loop is going to be iterated (line 427)
        # Testing the type of a for loop iterable (line 427)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 427, 8), types_1306)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 427, 8), types_1306):
            # Getting the type of the for loop variable (line 427)
            for_loop_var_1307 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 427, 8), types_1306)
            # Assigning a type to the variable 'type_' (line 427)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'type_', for_loop_var_1307)
            # SSA begins for a for statement (line 427)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 428):
            
            # Call to delete_member(...): (line 428)
            # Processing the call arguments (line 428)
            # Getting the type of 'localization' (line 428)
            localization_1310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 39), 'localization', False)
            # Getting the type of 'member' (line 428)
            member_1311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 53), 'member', False)
            # Processing the call keyword arguments (line 428)
            kwargs_1312 = {}
            # Getting the type of 'type_' (line 428)
            type__1308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 19), 'type_', False)
            # Obtaining the member 'delete_member' of a type (line 428)
            delete_member_1309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 19), type__1308, 'delete_member')
            # Calling delete_member(args, kwargs) (line 428)
            delete_member_call_result_1313 = invoke(stypy.reporting.localization.Localization(__file__, 428, 19), delete_member_1309, *[localization_1310, member_1311], **kwargs_1312)
            
            # Assigning a type to the variable 'temp' (line 428)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 12), 'temp', delete_member_call_result_1313)
            
            # Type idiom detected: calculating its left and rigth part (line 429)
            # Getting the type of 'temp' (line 429)
            temp_1314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'temp')
            # Getting the type of 'None' (line 429)
            None_1315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 27), 'None')
            
            (may_be_1316, more_types_in_union_1317) = may_not_be_none(temp_1314, None_1315)

            if may_be_1316:

                if more_types_in_union_1317:
                    # Runtime conditional SSA (line 429)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to append(...): (line 430)
                # Processing the call arguments (line 430)
                # Getting the type of 'temp' (line 430)
                temp_1320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 30), 'temp', False)
                # Processing the call keyword arguments (line 430)
                kwargs_1321 = {}
                # Getting the type of 'errors' (line 430)
                errors_1318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 16), 'errors', False)
                # Obtaining the member 'append' of a type (line 430)
                append_1319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 16), errors_1318, 'append')
                # Calling append(args, kwargs) (line 430)
                append_call_result_1322 = invoke(stypy.reporting.localization.Localization(__file__, 430, 16), append_1319, *[temp_1320], **kwargs_1321)
                

                if more_types_in_union_1317:
                    # SSA join for if statement (line 429)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 'errors' (line 433)
        errors_1324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 15), 'errors', False)
        # Processing the call keyword arguments (line 433)
        kwargs_1325 = {}
        # Getting the type of 'len' (line 433)
        len_1323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 11), 'len', False)
        # Calling len(args, kwargs) (line 433)
        len_call_result_1326 = invoke(stypy.reporting.localization.Localization(__file__, 433, 11), len_1323, *[errors_1324], **kwargs_1325)
        
        
        # Call to len(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 'self' (line 433)
        self_1328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 433)
        types_1329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 30), self_1328, 'types')
        # Processing the call keyword arguments (line 433)
        kwargs_1330 = {}
        # Getting the type of 'len' (line 433)
        len_1327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 26), 'len', False)
        # Calling len(args, kwargs) (line 433)
        len_call_result_1331 = invoke(stypy.reporting.localization.Localization(__file__, 433, 26), len_1327, *[types_1329], **kwargs_1330)
        
        # Applying the binary operator '==' (line 433)
        result_eq_1332 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 11), '==', len_call_result_1326, len_call_result_1331)
        
        # Testing if the type of an if condition is none (line 433)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 433, 8), result_eq_1332):
            
            # Getting the type of 'errors' (line 440)
            errors_1345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 25), 'errors')
            # Assigning a type to the variable 'errors_1345' (line 440)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'errors_1345', errors_1345)
            # Testing if the for loop is going to be iterated (line 440)
            # Testing the type of a for loop iterable (line 440)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 440, 12), errors_1345)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 440, 12), errors_1345):
                # Getting the type of the for loop variable (line 440)
                for_loop_var_1346 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 440, 12), errors_1345)
                # Assigning a type to the variable 'error' (line 440)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'error', for_loop_var_1346)
                # SSA begins for a for statement (line 440)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 441)
                # Processing the call keyword arguments (line 441)
                kwargs_1349 = {}
                # Getting the type of 'error' (line 441)
                error_1347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 441)
                turn_to_warning_1348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 16), error_1347, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 441)
                turn_to_warning_call_result_1350 = invoke(stypy.reporting.localization.Localization(__file__, 441, 16), turn_to_warning_1348, *[], **kwargs_1349)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 433)
            if_condition_1333 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 433, 8), result_eq_1332)
            # Assigning a type to the variable 'if_condition_1333' (line 433)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'if_condition_1333', if_condition_1333)
            # SSA begins for if statement (line 433)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 434)
            # Processing the call arguments (line 434)
            # Getting the type of 'localization' (line 434)
            localization_1335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 29), 'localization', False)
            
            # Call to format(...): (line 434)
            # Processing the call arguments (line 434)
            # Getting the type of 'member' (line 435)
            member_1338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 36), 'member', False)
            # Getting the type of 'self' (line 435)
            self_1339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 44), 'self', False)
            # Obtaining the member 'types' of a type (line 435)
            types_1340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 44), self_1339, 'types')
            # Processing the call keyword arguments (line 434)
            kwargs_1341 = {}
            str_1336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 43), 'str', "The member '{0}' cannot be deleted from none of the possible types ('{1}')")
            # Obtaining the member 'format' of a type (line 434)
            format_1337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 43), str_1336, 'format')
            # Calling format(args, kwargs) (line 434)
            format_call_result_1342 = invoke(stypy.reporting.localization.Localization(__file__, 434, 43), format_1337, *[member_1338, types_1340], **kwargs_1341)
            
            # Processing the call keyword arguments (line 434)
            kwargs_1343 = {}
            # Getting the type of 'TypeError' (line 434)
            TypeError_1334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 434)
            TypeError_call_result_1344 = invoke(stypy.reporting.localization.Localization(__file__, 434, 19), TypeError_1334, *[localization_1335, format_call_result_1342], **kwargs_1343)
            
            # Assigning a type to the variable 'stypy_return_type' (line 434)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'stypy_return_type', TypeError_call_result_1344)
            # SSA branch for the else part of an if statement (line 433)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 440)
            errors_1345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 25), 'errors')
            # Assigning a type to the variable 'errors_1345' (line 440)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'errors_1345', errors_1345)
            # Testing if the for loop is going to be iterated (line 440)
            # Testing the type of a for loop iterable (line 440)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 440, 12), errors_1345)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 440, 12), errors_1345):
                # Getting the type of the for loop variable (line 440)
                for_loop_var_1346 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 440, 12), errors_1345)
                # Assigning a type to the variable 'error' (line 440)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'error', for_loop_var_1346)
                # SSA begins for a for statement (line 440)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 441)
                # Processing the call keyword arguments (line 441)
                kwargs_1349 = {}
                # Getting the type of 'error' (line 441)
                error_1347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 441)
                turn_to_warning_1348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 16), error_1347, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 441)
                turn_to_warning_call_result_1350 = invoke(stypy.reporting.localization.Localization(__file__, 441, 16), turn_to_warning_1348, *[], **kwargs_1349)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 433)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'None' (line 443)
        None_1351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'stypy_return_type', None_1351)
        
        # ################# End of 'delete_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'delete_member' in the type store
        # Getting the type of 'stypy_return_type' (line 417)
        stypy_return_type_1352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1352)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'delete_member'
        return stypy_return_type_1352


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

        str_1353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, (-1)), 'str', '\n        Determines if at least one of the stored types supports structural reflection.\n        ')
        
        # Assigning a Name to a Name (line 449):
        # Getting the type of 'False' (line 449)
        False_1354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 19), 'False')
        # Assigning a type to the variable 'supports' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'supports', False_1354)
        
        # Getting the type of 'self' (line 451)
        self_1355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 21), 'self')
        # Obtaining the member 'types' of a type (line 451)
        types_1356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 21), self_1355, 'types')
        # Assigning a type to the variable 'types_1356' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'types_1356', types_1356)
        # Testing if the for loop is going to be iterated (line 451)
        # Testing the type of a for loop iterable (line 451)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 451, 8), types_1356)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 451, 8), types_1356):
            # Getting the type of the for loop variable (line 451)
            for_loop_var_1357 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 451, 8), types_1356)
            # Assigning a type to the variable 'type_' (line 451)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'type_', for_loop_var_1357)
            # SSA begins for a for statement (line 451)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a BoolOp to a Name (line 452):
            
            # Evaluating a boolean operation
            # Getting the type of 'supports' (line 452)
            supports_1358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 23), 'supports')
            
            # Call to supports_structural_reflection(...): (line 452)
            # Processing the call keyword arguments (line 452)
            kwargs_1361 = {}
            # Getting the type of 'type_' (line 452)
            type__1359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 35), 'type_', False)
            # Obtaining the member 'supports_structural_reflection' of a type (line 452)
            supports_structural_reflection_1360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 35), type__1359, 'supports_structural_reflection')
            # Calling supports_structural_reflection(args, kwargs) (line 452)
            supports_structural_reflection_call_result_1362 = invoke(stypy.reporting.localization.Localization(__file__, 452, 35), supports_structural_reflection_1360, *[], **kwargs_1361)
            
            # Applying the binary operator 'or' (line 452)
            result_or_keyword_1363 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 23), 'or', supports_1358, supports_structural_reflection_call_result_1362)
            
            # Assigning a type to the variable 'supports' (line 452)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 12), 'supports', result_or_keyword_1363)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'supports' (line 454)
        supports_1364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 15), 'supports')
        # Assigning a type to the variable 'stypy_return_type' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'stypy_return_type', supports_1364)
        
        # ################# End of 'supports_structural_reflection(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'supports_structural_reflection' in the type store
        # Getting the type of 'stypy_return_type' (line 445)
        stypy_return_type_1365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1365)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'supports_structural_reflection'
        return stypy_return_type_1365


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

        str_1366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, (-1)), 'str', '\n        For all the types stored in the union type, change the base type to new_type, returning None or a TypeError\n        if no type stored in the UnionType supports a type change.\n        :param localization: Caller information\n        :param new_type: Type to change to\n        :return None or TypeError\n        ')
        
        # Assigning a List to a Name (line 464):
        
        # Obtaining an instance of the builtin type 'list' (line 464)
        list_1367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 464)
        
        # Assigning a type to the variable 'errors' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'errors', list_1367)
        
        # Getting the type of 'self' (line 466)
        self_1368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 21), 'self')
        # Obtaining the member 'types' of a type (line 466)
        types_1369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 21), self_1368, 'types')
        # Assigning a type to the variable 'types_1369' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'types_1369', types_1369)
        # Testing if the for loop is going to be iterated (line 466)
        # Testing the type of a for loop iterable (line 466)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 466, 8), types_1369)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 466, 8), types_1369):
            # Getting the type of the for loop variable (line 466)
            for_loop_var_1370 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 466, 8), types_1369)
            # Assigning a type to the variable 'type_' (line 466)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'type_', for_loop_var_1370)
            # SSA begins for a for statement (line 466)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 467):
            
            # Call to change_type(...): (line 467)
            # Processing the call arguments (line 467)
            # Getting the type of 'localization' (line 467)
            localization_1373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 37), 'localization', False)
            # Getting the type of 'new_type' (line 467)
            new_type_1374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 51), 'new_type', False)
            # Processing the call keyword arguments (line 467)
            kwargs_1375 = {}
            # Getting the type of 'type_' (line 467)
            type__1371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 19), 'type_', False)
            # Obtaining the member 'change_type' of a type (line 467)
            change_type_1372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 19), type__1371, 'change_type')
            # Calling change_type(args, kwargs) (line 467)
            change_type_call_result_1376 = invoke(stypy.reporting.localization.Localization(__file__, 467, 19), change_type_1372, *[localization_1373, new_type_1374], **kwargs_1375)
            
            # Assigning a type to the variable 'temp' (line 467)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'temp', change_type_call_result_1376)
            
            # Type idiom detected: calculating its left and rigth part (line 468)
            # Getting the type of 'temp' (line 468)
            temp_1377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 12), 'temp')
            # Getting the type of 'None' (line 468)
            None_1378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 27), 'None')
            
            (may_be_1379, more_types_in_union_1380) = may_not_be_none(temp_1377, None_1378)

            if may_be_1379:

                if more_types_in_union_1380:
                    # Runtime conditional SSA (line 468)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to append(...): (line 469)
                # Processing the call arguments (line 469)
                # Getting the type of 'temp' (line 469)
                temp_1383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 30), 'temp', False)
                # Processing the call keyword arguments (line 469)
                kwargs_1384 = {}
                # Getting the type of 'errors' (line 469)
                errors_1381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 16), 'errors', False)
                # Obtaining the member 'append' of a type (line 469)
                append_1382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 16), errors_1381, 'append')
                # Calling append(args, kwargs) (line 469)
                append_call_result_1385 = invoke(stypy.reporting.localization.Localization(__file__, 469, 16), append_1382, *[temp_1383], **kwargs_1384)
                

                if more_types_in_union_1380:
                    # SSA join for if statement (line 468)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 472)
        # Processing the call arguments (line 472)
        # Getting the type of 'errors' (line 472)
        errors_1387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 15), 'errors', False)
        # Processing the call keyword arguments (line 472)
        kwargs_1388 = {}
        # Getting the type of 'len' (line 472)
        len_1386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 11), 'len', False)
        # Calling len(args, kwargs) (line 472)
        len_call_result_1389 = invoke(stypy.reporting.localization.Localization(__file__, 472, 11), len_1386, *[errors_1387], **kwargs_1388)
        
        
        # Call to len(...): (line 472)
        # Processing the call arguments (line 472)
        # Getting the type of 'self' (line 472)
        self_1391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 472)
        types_1392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 30), self_1391, 'types')
        # Processing the call keyword arguments (line 472)
        kwargs_1393 = {}
        # Getting the type of 'len' (line 472)
        len_1390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 26), 'len', False)
        # Calling len(args, kwargs) (line 472)
        len_call_result_1394 = invoke(stypy.reporting.localization.Localization(__file__, 472, 26), len_1390, *[types_1392], **kwargs_1393)
        
        # Applying the binary operator '==' (line 472)
        result_eq_1395 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 11), '==', len_call_result_1389, len_call_result_1394)
        
        # Testing if the type of an if condition is none (line 472)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 472, 8), result_eq_1395):
            
            # Getting the type of 'errors' (line 479)
            errors_1408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 25), 'errors')
            # Assigning a type to the variable 'errors_1408' (line 479)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'errors_1408', errors_1408)
            # Testing if the for loop is going to be iterated (line 479)
            # Testing the type of a for loop iterable (line 479)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 479, 12), errors_1408)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 479, 12), errors_1408):
                # Getting the type of the for loop variable (line 479)
                for_loop_var_1409 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 479, 12), errors_1408)
                # Assigning a type to the variable 'error' (line 479)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'error', for_loop_var_1409)
                # SSA begins for a for statement (line 479)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 480)
                # Processing the call keyword arguments (line 480)
                kwargs_1412 = {}
                # Getting the type of 'error' (line 480)
                error_1410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 480)
                turn_to_warning_1411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 16), error_1410, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 480)
                turn_to_warning_call_result_1413 = invoke(stypy.reporting.localization.Localization(__file__, 480, 16), turn_to_warning_1411, *[], **kwargs_1412)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 472)
            if_condition_1396 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 472, 8), result_eq_1395)
            # Assigning a type to the variable 'if_condition_1396' (line 472)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'if_condition_1396', if_condition_1396)
            # SSA begins for if statement (line 472)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 473)
            # Processing the call arguments (line 473)
            # Getting the type of 'localization' (line 473)
            localization_1398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 29), 'localization', False)
            
            # Call to format(...): (line 473)
            # Processing the call arguments (line 473)
            # Getting the type of 'new_type' (line 474)
            new_type_1401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 36), 'new_type', False)
            # Getting the type of 'self' (line 474)
            self_1402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 46), 'self', False)
            # Obtaining the member 'types' of a type (line 474)
            types_1403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 46), self_1402, 'types')
            # Processing the call keyword arguments (line 473)
            kwargs_1404 = {}
            str_1399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 43), 'str', "None of the possible types ('{1}') can be assigned a new type '{0}'")
            # Obtaining the member 'format' of a type (line 473)
            format_1400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 43), str_1399, 'format')
            # Calling format(args, kwargs) (line 473)
            format_call_result_1405 = invoke(stypy.reporting.localization.Localization(__file__, 473, 43), format_1400, *[new_type_1401, types_1403], **kwargs_1404)
            
            # Processing the call keyword arguments (line 473)
            kwargs_1406 = {}
            # Getting the type of 'TypeError' (line 473)
            TypeError_1397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 473)
            TypeError_call_result_1407 = invoke(stypy.reporting.localization.Localization(__file__, 473, 19), TypeError_1397, *[localization_1398, format_call_result_1405], **kwargs_1406)
            
            # Assigning a type to the variable 'stypy_return_type' (line 473)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'stypy_return_type', TypeError_call_result_1407)
            # SSA branch for the else part of an if statement (line 472)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 479)
            errors_1408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 25), 'errors')
            # Assigning a type to the variable 'errors_1408' (line 479)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'errors_1408', errors_1408)
            # Testing if the for loop is going to be iterated (line 479)
            # Testing the type of a for loop iterable (line 479)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 479, 12), errors_1408)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 479, 12), errors_1408):
                # Getting the type of the for loop variable (line 479)
                for_loop_var_1409 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 479, 12), errors_1408)
                # Assigning a type to the variable 'error' (line 479)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'error', for_loop_var_1409)
                # SSA begins for a for statement (line 479)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 480)
                # Processing the call keyword arguments (line 480)
                kwargs_1412 = {}
                # Getting the type of 'error' (line 480)
                error_1410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 480)
                turn_to_warning_1411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 16), error_1410, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 480)
                turn_to_warning_call_result_1413 = invoke(stypy.reporting.localization.Localization(__file__, 480, 16), turn_to_warning_1411, *[], **kwargs_1412)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 472)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'None' (line 482)
        None_1414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'stypy_return_type', None_1414)
        
        # ################# End of 'change_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'change_type' in the type store
        # Getting the type of 'stypy_return_type' (line 456)
        stypy_return_type_1415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1415)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'change_type'
        return stypy_return_type_1415


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

        str_1416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, (-1)), 'str', '\n        For all the types stored in the union type, change the base types to the ones contained in the list new_types,\n        returning None or a TypeError if no type stored in the UnionType supports a supertype change.\n        :param localization: Caller information\n        :param new_types: Types to change its base type to\n        :return None or TypeError\n        ')
        
        # Assigning a List to a Name (line 492):
        
        # Obtaining an instance of the builtin type 'list' (line 492)
        list_1417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 492)
        
        # Assigning a type to the variable 'errors' (line 492)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'errors', list_1417)
        
        # Getting the type of 'self' (line 494)
        self_1418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 21), 'self')
        # Obtaining the member 'types' of a type (line 494)
        types_1419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 21), self_1418, 'types')
        # Assigning a type to the variable 'types_1419' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'types_1419', types_1419)
        # Testing if the for loop is going to be iterated (line 494)
        # Testing the type of a for loop iterable (line 494)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 494, 8), types_1419)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 494, 8), types_1419):
            # Getting the type of the for loop variable (line 494)
            for_loop_var_1420 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 494, 8), types_1419)
            # Assigning a type to the variable 'type_' (line 494)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'type_', for_loop_var_1420)
            # SSA begins for a for statement (line 494)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 495):
            
            # Call to change_base_types(...): (line 495)
            # Processing the call arguments (line 495)
            # Getting the type of 'localization' (line 495)
            localization_1423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 43), 'localization', False)
            # Getting the type of 'new_types' (line 495)
            new_types_1424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 57), 'new_types', False)
            # Processing the call keyword arguments (line 495)
            kwargs_1425 = {}
            # Getting the type of 'type_' (line 495)
            type__1421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 19), 'type_', False)
            # Obtaining the member 'change_base_types' of a type (line 495)
            change_base_types_1422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 19), type__1421, 'change_base_types')
            # Calling change_base_types(args, kwargs) (line 495)
            change_base_types_call_result_1426 = invoke(stypy.reporting.localization.Localization(__file__, 495, 19), change_base_types_1422, *[localization_1423, new_types_1424], **kwargs_1425)
            
            # Assigning a type to the variable 'temp' (line 495)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'temp', change_base_types_call_result_1426)
            
            # Type idiom detected: calculating its left and rigth part (line 496)
            # Getting the type of 'temp' (line 496)
            temp_1427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'temp')
            # Getting the type of 'None' (line 496)
            None_1428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 27), 'None')
            
            (may_be_1429, more_types_in_union_1430) = may_not_be_none(temp_1427, None_1428)

            if may_be_1429:

                if more_types_in_union_1430:
                    # Runtime conditional SSA (line 496)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to append(...): (line 497)
                # Processing the call arguments (line 497)
                # Getting the type of 'temp' (line 497)
                temp_1433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 30), 'temp', False)
                # Processing the call keyword arguments (line 497)
                kwargs_1434 = {}
                # Getting the type of 'errors' (line 497)
                errors_1431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 16), 'errors', False)
                # Obtaining the member 'append' of a type (line 497)
                append_1432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 16), errors_1431, 'append')
                # Calling append(args, kwargs) (line 497)
                append_call_result_1435 = invoke(stypy.reporting.localization.Localization(__file__, 497, 16), append_1432, *[temp_1433], **kwargs_1434)
                

                if more_types_in_union_1430:
                    # SSA join for if statement (line 496)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 500)
        # Processing the call arguments (line 500)
        # Getting the type of 'errors' (line 500)
        errors_1437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 15), 'errors', False)
        # Processing the call keyword arguments (line 500)
        kwargs_1438 = {}
        # Getting the type of 'len' (line 500)
        len_1436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 11), 'len', False)
        # Calling len(args, kwargs) (line 500)
        len_call_result_1439 = invoke(stypy.reporting.localization.Localization(__file__, 500, 11), len_1436, *[errors_1437], **kwargs_1438)
        
        
        # Call to len(...): (line 500)
        # Processing the call arguments (line 500)
        # Getting the type of 'self' (line 500)
        self_1441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 500)
        types_1442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 30), self_1441, 'types')
        # Processing the call keyword arguments (line 500)
        kwargs_1443 = {}
        # Getting the type of 'len' (line 500)
        len_1440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 26), 'len', False)
        # Calling len(args, kwargs) (line 500)
        len_call_result_1444 = invoke(stypy.reporting.localization.Localization(__file__, 500, 26), len_1440, *[types_1442], **kwargs_1443)
        
        # Applying the binary operator '==' (line 500)
        result_eq_1445 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 11), '==', len_call_result_1439, len_call_result_1444)
        
        # Testing if the type of an if condition is none (line 500)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 500, 8), result_eq_1445):
            
            # Getting the type of 'errors' (line 507)
            errors_1458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 25), 'errors')
            # Assigning a type to the variable 'errors_1458' (line 507)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 12), 'errors_1458', errors_1458)
            # Testing if the for loop is going to be iterated (line 507)
            # Testing the type of a for loop iterable (line 507)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 507, 12), errors_1458)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 507, 12), errors_1458):
                # Getting the type of the for loop variable (line 507)
                for_loop_var_1459 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 507, 12), errors_1458)
                # Assigning a type to the variable 'error' (line 507)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 12), 'error', for_loop_var_1459)
                # SSA begins for a for statement (line 507)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 508)
                # Processing the call keyword arguments (line 508)
                kwargs_1462 = {}
                # Getting the type of 'error' (line 508)
                error_1460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 508)
                turn_to_warning_1461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 16), error_1460, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 508)
                turn_to_warning_call_result_1463 = invoke(stypy.reporting.localization.Localization(__file__, 508, 16), turn_to_warning_1461, *[], **kwargs_1462)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 500)
            if_condition_1446 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 500, 8), result_eq_1445)
            # Assigning a type to the variable 'if_condition_1446' (line 500)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'if_condition_1446', if_condition_1446)
            # SSA begins for if statement (line 500)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 501)
            # Processing the call arguments (line 501)
            # Getting the type of 'localization' (line 501)
            localization_1448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 29), 'localization', False)
            
            # Call to format(...): (line 501)
            # Processing the call arguments (line 501)
            # Getting the type of 'new_types' (line 502)
            new_types_1451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 36), 'new_types', False)
            # Getting the type of 'self' (line 502)
            self_1452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 47), 'self', False)
            # Obtaining the member 'types' of a type (line 502)
            types_1453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 47), self_1452, 'types')
            # Processing the call keyword arguments (line 501)
            kwargs_1454 = {}
            str_1449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 43), 'str', "None of the possible types ('{1}') can be assigned new base types '{0}'")
            # Obtaining the member 'format' of a type (line 501)
            format_1450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 43), str_1449, 'format')
            # Calling format(args, kwargs) (line 501)
            format_call_result_1455 = invoke(stypy.reporting.localization.Localization(__file__, 501, 43), format_1450, *[new_types_1451, types_1453], **kwargs_1454)
            
            # Processing the call keyword arguments (line 501)
            kwargs_1456 = {}
            # Getting the type of 'TypeError' (line 501)
            TypeError_1447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 501)
            TypeError_call_result_1457 = invoke(stypy.reporting.localization.Localization(__file__, 501, 19), TypeError_1447, *[localization_1448, format_call_result_1455], **kwargs_1456)
            
            # Assigning a type to the variable 'stypy_return_type' (line 501)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 12), 'stypy_return_type', TypeError_call_result_1457)
            # SSA branch for the else part of an if statement (line 500)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 507)
            errors_1458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 25), 'errors')
            # Assigning a type to the variable 'errors_1458' (line 507)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 12), 'errors_1458', errors_1458)
            # Testing if the for loop is going to be iterated (line 507)
            # Testing the type of a for loop iterable (line 507)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 507, 12), errors_1458)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 507, 12), errors_1458):
                # Getting the type of the for loop variable (line 507)
                for_loop_var_1459 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 507, 12), errors_1458)
                # Assigning a type to the variable 'error' (line 507)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 12), 'error', for_loop_var_1459)
                # SSA begins for a for statement (line 507)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 508)
                # Processing the call keyword arguments (line 508)
                kwargs_1462 = {}
                # Getting the type of 'error' (line 508)
                error_1460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 508)
                turn_to_warning_1461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 16), error_1460, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 508)
                turn_to_warning_call_result_1463 = invoke(stypy.reporting.localization.Localization(__file__, 508, 16), turn_to_warning_1461, *[], **kwargs_1462)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 500)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'None' (line 510)
        None_1464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'stypy_return_type', None_1464)
        
        # ################# End of 'change_base_types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'change_base_types' in the type store
        # Getting the type of 'stypy_return_type' (line 484)
        stypy_return_type_1465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1465)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'change_base_types'
        return stypy_return_type_1465


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

        str_1466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, (-1)), 'str', '\n        For all the types stored in the union type, add to the base types the ones contained in the list new_types,\n        returning None or a TypeError if no type stored in the UnionType supports a supertype change.\n        :param localization: Caller information\n        :param new_types: Types to change its base type to\n        :return None or TypeError\n        ')
        
        # Assigning a List to a Name (line 520):
        
        # Obtaining an instance of the builtin type 'list' (line 520)
        list_1467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 520)
        
        # Assigning a type to the variable 'errors' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'errors', list_1467)
        
        # Getting the type of 'self' (line 522)
        self_1468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 21), 'self')
        # Obtaining the member 'types' of a type (line 522)
        types_1469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 21), self_1468, 'types')
        # Assigning a type to the variable 'types_1469' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'types_1469', types_1469)
        # Testing if the for loop is going to be iterated (line 522)
        # Testing the type of a for loop iterable (line 522)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 522, 8), types_1469)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 522, 8), types_1469):
            # Getting the type of the for loop variable (line 522)
            for_loop_var_1470 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 522, 8), types_1469)
            # Assigning a type to the variable 'type_' (line 522)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'type_', for_loop_var_1470)
            # SSA begins for a for statement (line 522)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 523):
            
            # Call to change_base_types(...): (line 523)
            # Processing the call arguments (line 523)
            # Getting the type of 'localization' (line 523)
            localization_1473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 43), 'localization', False)
            # Getting the type of 'new_types' (line 523)
            new_types_1474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 57), 'new_types', False)
            # Processing the call keyword arguments (line 523)
            kwargs_1475 = {}
            # Getting the type of 'type_' (line 523)
            type__1471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 19), 'type_', False)
            # Obtaining the member 'change_base_types' of a type (line 523)
            change_base_types_1472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 19), type__1471, 'change_base_types')
            # Calling change_base_types(args, kwargs) (line 523)
            change_base_types_call_result_1476 = invoke(stypy.reporting.localization.Localization(__file__, 523, 19), change_base_types_1472, *[localization_1473, new_types_1474], **kwargs_1475)
            
            # Assigning a type to the variable 'temp' (line 523)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 12), 'temp', change_base_types_call_result_1476)
            
            # Type idiom detected: calculating its left and rigth part (line 524)
            # Getting the type of 'temp' (line 524)
            temp_1477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 12), 'temp')
            # Getting the type of 'None' (line 524)
            None_1478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 27), 'None')
            
            (may_be_1479, more_types_in_union_1480) = may_not_be_none(temp_1477, None_1478)

            if may_be_1479:

                if more_types_in_union_1480:
                    # Runtime conditional SSA (line 524)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to append(...): (line 525)
                # Processing the call arguments (line 525)
                # Getting the type of 'temp' (line 525)
                temp_1483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 30), 'temp', False)
                # Processing the call keyword arguments (line 525)
                kwargs_1484 = {}
                # Getting the type of 'errors' (line 525)
                errors_1481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 16), 'errors', False)
                # Obtaining the member 'append' of a type (line 525)
                append_1482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 16), errors_1481, 'append')
                # Calling append(args, kwargs) (line 525)
                append_call_result_1485 = invoke(stypy.reporting.localization.Localization(__file__, 525, 16), append_1482, *[temp_1483], **kwargs_1484)
                

                if more_types_in_union_1480:
                    # SSA join for if statement (line 524)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 528)
        # Processing the call arguments (line 528)
        # Getting the type of 'errors' (line 528)
        errors_1487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 15), 'errors', False)
        # Processing the call keyword arguments (line 528)
        kwargs_1488 = {}
        # Getting the type of 'len' (line 528)
        len_1486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 11), 'len', False)
        # Calling len(args, kwargs) (line 528)
        len_call_result_1489 = invoke(stypy.reporting.localization.Localization(__file__, 528, 11), len_1486, *[errors_1487], **kwargs_1488)
        
        
        # Call to len(...): (line 528)
        # Processing the call arguments (line 528)
        # Getting the type of 'self' (line 528)
        self_1491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 528)
        types_1492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 30), self_1491, 'types')
        # Processing the call keyword arguments (line 528)
        kwargs_1493 = {}
        # Getting the type of 'len' (line 528)
        len_1490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 26), 'len', False)
        # Calling len(args, kwargs) (line 528)
        len_call_result_1494 = invoke(stypy.reporting.localization.Localization(__file__, 528, 26), len_1490, *[types_1492], **kwargs_1493)
        
        # Applying the binary operator '==' (line 528)
        result_eq_1495 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 11), '==', len_call_result_1489, len_call_result_1494)
        
        # Testing if the type of an if condition is none (line 528)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 528, 8), result_eq_1495):
            
            # Getting the type of 'errors' (line 535)
            errors_1507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 25), 'errors')
            # Assigning a type to the variable 'errors_1507' (line 535)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'errors_1507', errors_1507)
            # Testing if the for loop is going to be iterated (line 535)
            # Testing the type of a for loop iterable (line 535)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 535, 12), errors_1507)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 535, 12), errors_1507):
                # Getting the type of the for loop variable (line 535)
                for_loop_var_1508 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 535, 12), errors_1507)
                # Assigning a type to the variable 'error' (line 535)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'error', for_loop_var_1508)
                # SSA begins for a for statement (line 535)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 536)
                # Processing the call keyword arguments (line 536)
                kwargs_1511 = {}
                # Getting the type of 'error' (line 536)
                error_1509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 536)
                turn_to_warning_1510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 16), error_1509, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 536)
                turn_to_warning_call_result_1512 = invoke(stypy.reporting.localization.Localization(__file__, 536, 16), turn_to_warning_1510, *[], **kwargs_1511)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 528)
            if_condition_1496 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 528, 8), result_eq_1495)
            # Assigning a type to the variable 'if_condition_1496' (line 528)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'if_condition_1496', if_condition_1496)
            # SSA begins for if statement (line 528)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 529)
            # Processing the call arguments (line 529)
            # Getting the type of 'localization' (line 529)
            localization_1498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 29), 'localization', False)
            
            # Call to format(...): (line 529)
            # Processing the call arguments (line 529)
            # Getting the type of 'self' (line 530)
            self_1501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 36), 'self', False)
            # Obtaining the member 'types' of a type (line 530)
            types_1502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 36), self_1501, 'types')
            # Processing the call keyword arguments (line 529)
            kwargs_1503 = {}
            str_1499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 43), 'str', "The base types of all the possible types ('{0}') cannot be modified")
            # Obtaining the member 'format' of a type (line 529)
            format_1500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 43), str_1499, 'format')
            # Calling format(args, kwargs) (line 529)
            format_call_result_1504 = invoke(stypy.reporting.localization.Localization(__file__, 529, 43), format_1500, *[types_1502], **kwargs_1503)
            
            # Processing the call keyword arguments (line 529)
            kwargs_1505 = {}
            # Getting the type of 'TypeError' (line 529)
            TypeError_1497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 529)
            TypeError_call_result_1506 = invoke(stypy.reporting.localization.Localization(__file__, 529, 19), TypeError_1497, *[localization_1498, format_call_result_1504], **kwargs_1505)
            
            # Assigning a type to the variable 'stypy_return_type' (line 529)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'stypy_return_type', TypeError_call_result_1506)
            # SSA branch for the else part of an if statement (line 528)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 535)
            errors_1507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 25), 'errors')
            # Assigning a type to the variable 'errors_1507' (line 535)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'errors_1507', errors_1507)
            # Testing if the for loop is going to be iterated (line 535)
            # Testing the type of a for loop iterable (line 535)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 535, 12), errors_1507)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 535, 12), errors_1507):
                # Getting the type of the for loop variable (line 535)
                for_loop_var_1508 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 535, 12), errors_1507)
                # Assigning a type to the variable 'error' (line 535)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'error', for_loop_var_1508)
                # SSA begins for a for statement (line 535)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 536)
                # Processing the call keyword arguments (line 536)
                kwargs_1511 = {}
                # Getting the type of 'error' (line 536)
                error_1509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 536)
                turn_to_warning_1510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 16), error_1509, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 536)
                turn_to_warning_call_result_1512 = invoke(stypy.reporting.localization.Localization(__file__, 536, 16), turn_to_warning_1510, *[], **kwargs_1511)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 528)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'None' (line 538)
        None_1513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 538)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'stypy_return_type', None_1513)
        
        # ################# End of 'add_base_types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_base_types' in the type store
        # Getting the type of 'stypy_return_type' (line 512)
        stypy_return_type_1514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1514)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_base_types'
        return stypy_return_type_1514


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

        str_1515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, (-1)), 'str', '\n        Clone the whole UnionType and its contained types\n        ')
        
        # Assigning a Call to a Name (line 546):
        
        # Call to clone(...): (line 546)
        # Processing the call keyword arguments (line 546)
        kwargs_1522 = {}
        
        # Obtaining the type of the subscript
        int_1516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 34), 'int')
        # Getting the type of 'self' (line 546)
        self_1517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 23), 'self', False)
        # Obtaining the member 'types' of a type (line 546)
        types_1518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 23), self_1517, 'types')
        # Obtaining the member '__getitem__' of a type (line 546)
        getitem___1519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 23), types_1518, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 546)
        subscript_call_result_1520 = invoke(stypy.reporting.localization.Localization(__file__, 546, 23), getitem___1519, int_1516)
        
        # Obtaining the member 'clone' of a type (line 546)
        clone_1521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 23), subscript_call_result_1520, 'clone')
        # Calling clone(args, kwargs) (line 546)
        clone_call_result_1523 = invoke(stypy.reporting.localization.Localization(__file__, 546, 23), clone_1521, *[], **kwargs_1522)
        
        # Assigning a type to the variable 'result_union' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'result_union', clone_call_result_1523)
        
        
        # Call to range(...): (line 547)
        # Processing the call arguments (line 547)
        int_1525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 23), 'int')
        
        # Call to len(...): (line 547)
        # Processing the call arguments (line 547)
        # Getting the type of 'self' (line 547)
        self_1527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 547)
        types_1528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 30), self_1527, 'types')
        # Processing the call keyword arguments (line 547)
        kwargs_1529 = {}
        # Getting the type of 'len' (line 547)
        len_1526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 26), 'len', False)
        # Calling len(args, kwargs) (line 547)
        len_call_result_1530 = invoke(stypy.reporting.localization.Localization(__file__, 547, 26), len_1526, *[types_1528], **kwargs_1529)
        
        # Processing the call keyword arguments (line 547)
        kwargs_1531 = {}
        # Getting the type of 'range' (line 547)
        range_1524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 17), 'range', False)
        # Calling range(args, kwargs) (line 547)
        range_call_result_1532 = invoke(stypy.reporting.localization.Localization(__file__, 547, 17), range_1524, *[int_1525, len_call_result_1530], **kwargs_1531)
        
        # Assigning a type to the variable 'range_call_result_1532' (line 547)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'range_call_result_1532', range_call_result_1532)
        # Testing if the for loop is going to be iterated (line 547)
        # Testing the type of a for loop iterable (line 547)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 547, 8), range_call_result_1532)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 547, 8), range_call_result_1532):
            # Getting the type of the for loop variable (line 547)
            for_loop_var_1533 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 547, 8), range_call_result_1532)
            # Assigning a type to the variable 'i' (line 547)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'i', for_loop_var_1533)
            # SSA begins for a for statement (line 547)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to isinstance(...): (line 548)
            # Processing the call arguments (line 548)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 548)
            i_1535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 37), 'i', False)
            # Getting the type of 'self' (line 548)
            self_1536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 26), 'self', False)
            # Obtaining the member 'types' of a type (line 548)
            types_1537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 26), self_1536, 'types')
            # Obtaining the member '__getitem__' of a type (line 548)
            getitem___1538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 26), types_1537, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 548)
            subscript_call_result_1539 = invoke(stypy.reporting.localization.Localization(__file__, 548, 26), getitem___1538, i_1535)
            
            # Getting the type of 'Type' (line 548)
            Type_1540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 41), 'Type', False)
            # Processing the call keyword arguments (line 548)
            kwargs_1541 = {}
            # Getting the type of 'isinstance' (line 548)
            isinstance_1534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 548)
            isinstance_call_result_1542 = invoke(stypy.reporting.localization.Localization(__file__, 548, 15), isinstance_1534, *[subscript_call_result_1539, Type_1540], **kwargs_1541)
            
            # Testing if the type of an if condition is none (line 548)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 548, 12), isinstance_call_result_1542):
                
                # Assigning a Call to a Name (line 551):
                
                # Call to add(...): (line 551)
                # Processing the call arguments (line 551)
                # Getting the type of 'result_union' (line 551)
                result_union_1559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 45), 'result_union', False)
                
                # Call to deepcopy(...): (line 551)
                # Processing the call arguments (line 551)
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 551)
                i_1562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 84), 'i', False)
                # Getting the type of 'self' (line 551)
                self_1563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 73), 'self', False)
                # Obtaining the member 'types' of a type (line 551)
                types_1564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 73), self_1563, 'types')
                # Obtaining the member '__getitem__' of a type (line 551)
                getitem___1565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 73), types_1564, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 551)
                subscript_call_result_1566 = invoke(stypy.reporting.localization.Localization(__file__, 551, 73), getitem___1565, i_1562)
                
                # Processing the call keyword arguments (line 551)
                kwargs_1567 = {}
                # Getting the type of 'copy' (line 551)
                copy_1560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 59), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 551)
                deepcopy_1561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 59), copy_1560, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 551)
                deepcopy_call_result_1568 = invoke(stypy.reporting.localization.Localization(__file__, 551, 59), deepcopy_1561, *[subscript_call_result_1566], **kwargs_1567)
                
                # Processing the call keyword arguments (line 551)
                kwargs_1569 = {}
                # Getting the type of 'UnionType' (line 551)
                UnionType_1557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 31), 'UnionType', False)
                # Obtaining the member 'add' of a type (line 551)
                add_1558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 31), UnionType_1557, 'add')
                # Calling add(args, kwargs) (line 551)
                add_call_result_1570 = invoke(stypy.reporting.localization.Localization(__file__, 551, 31), add_1558, *[result_union_1559, deepcopy_call_result_1568], **kwargs_1569)
                
                # Assigning a type to the variable 'result_union' (line 551)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 16), 'result_union', add_call_result_1570)
            else:
                
                # Testing the type of an if condition (line 548)
                if_condition_1543 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 548, 12), isinstance_call_result_1542)
                # Assigning a type to the variable 'if_condition_1543' (line 548)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 12), 'if_condition_1543', if_condition_1543)
                # SSA begins for if statement (line 548)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 549):
                
                # Call to add(...): (line 549)
                # Processing the call arguments (line 549)
                # Getting the type of 'result_union' (line 549)
                result_union_1546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 45), 'result_union', False)
                
                # Call to clone(...): (line 549)
                # Processing the call keyword arguments (line 549)
                kwargs_1553 = {}
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 549)
                i_1547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 70), 'i', False)
                # Getting the type of 'self' (line 549)
                self_1548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 59), 'self', False)
                # Obtaining the member 'types' of a type (line 549)
                types_1549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 59), self_1548, 'types')
                # Obtaining the member '__getitem__' of a type (line 549)
                getitem___1550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 59), types_1549, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 549)
                subscript_call_result_1551 = invoke(stypy.reporting.localization.Localization(__file__, 549, 59), getitem___1550, i_1547)
                
                # Obtaining the member 'clone' of a type (line 549)
                clone_1552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 59), subscript_call_result_1551, 'clone')
                # Calling clone(args, kwargs) (line 549)
                clone_call_result_1554 = invoke(stypy.reporting.localization.Localization(__file__, 549, 59), clone_1552, *[], **kwargs_1553)
                
                # Processing the call keyword arguments (line 549)
                kwargs_1555 = {}
                # Getting the type of 'UnionType' (line 549)
                UnionType_1544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 31), 'UnionType', False)
                # Obtaining the member 'add' of a type (line 549)
                add_1545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 31), UnionType_1544, 'add')
                # Calling add(args, kwargs) (line 549)
                add_call_result_1556 = invoke(stypy.reporting.localization.Localization(__file__, 549, 31), add_1545, *[result_union_1546, clone_call_result_1554], **kwargs_1555)
                
                # Assigning a type to the variable 'result_union' (line 549)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 16), 'result_union', add_call_result_1556)
                # SSA branch for the else part of an if statement (line 548)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Name (line 551):
                
                # Call to add(...): (line 551)
                # Processing the call arguments (line 551)
                # Getting the type of 'result_union' (line 551)
                result_union_1559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 45), 'result_union', False)
                
                # Call to deepcopy(...): (line 551)
                # Processing the call arguments (line 551)
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 551)
                i_1562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 84), 'i', False)
                # Getting the type of 'self' (line 551)
                self_1563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 73), 'self', False)
                # Obtaining the member 'types' of a type (line 551)
                types_1564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 73), self_1563, 'types')
                # Obtaining the member '__getitem__' of a type (line 551)
                getitem___1565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 73), types_1564, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 551)
                subscript_call_result_1566 = invoke(stypy.reporting.localization.Localization(__file__, 551, 73), getitem___1565, i_1562)
                
                # Processing the call keyword arguments (line 551)
                kwargs_1567 = {}
                # Getting the type of 'copy' (line 551)
                copy_1560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 59), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 551)
                deepcopy_1561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 59), copy_1560, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 551)
                deepcopy_call_result_1568 = invoke(stypy.reporting.localization.Localization(__file__, 551, 59), deepcopy_1561, *[subscript_call_result_1566], **kwargs_1567)
                
                # Processing the call keyword arguments (line 551)
                kwargs_1569 = {}
                # Getting the type of 'UnionType' (line 551)
                UnionType_1557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 31), 'UnionType', False)
                # Obtaining the member 'add' of a type (line 551)
                add_1558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 31), UnionType_1557, 'add')
                # Calling add(args, kwargs) (line 551)
                add_call_result_1570 = invoke(stypy.reporting.localization.Localization(__file__, 551, 31), add_1558, *[result_union_1559, deepcopy_call_result_1568], **kwargs_1569)
                
                # Assigning a type to the variable 'result_union' (line 551)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 16), 'result_union', add_call_result_1570)
                # SSA join for if statement (line 548)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'result_union' (line 553)
        result_union_1571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 15), 'result_union')
        # Assigning a type to the variable 'stypy_return_type' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'stypy_return_type', result_union_1571)
        
        # ################# End of 'clone(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clone' in the type store
        # Getting the type of 'stypy_return_type' (line 542)
        stypy_return_type_1572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1572)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clone'
        return stypy_return_type_1572


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
        False_1573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 15), 'False')
        # Assigning a type to the variable 'temp' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'temp', False_1573)
        
        # Getting the type of 'self' (line 557)
        self_1574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 21), 'self')
        # Obtaining the member 'types' of a type (line 557)
        types_1575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 21), self_1574, 'types')
        # Assigning a type to the variable 'types_1575' (line 557)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'types_1575', types_1575)
        # Testing if the for loop is going to be iterated (line 557)
        # Testing the type of a for loop iterable (line 557)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 557, 8), types_1575)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 557, 8), types_1575):
            # Getting the type of the for loop variable (line 557)
            for_loop_var_1576 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 557, 8), types_1575)
            # Assigning a type to the variable 'type_' (line 557)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'type_', for_loop_var_1576)
            # SSA begins for a for statement (line 557)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'temp' (line 558)
            temp_1577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 12), 'temp')
            
            # Call to can_store_elements(...): (line 558)
            # Processing the call keyword arguments (line 558)
            kwargs_1580 = {}
            # Getting the type of 'type_' (line 558)
            type__1578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 20), 'type_', False)
            # Obtaining the member 'can_store_elements' of a type (line 558)
            can_store_elements_1579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 20), type__1578, 'can_store_elements')
            # Calling can_store_elements(args, kwargs) (line 558)
            can_store_elements_call_result_1581 = invoke(stypy.reporting.localization.Localization(__file__, 558, 20), can_store_elements_1579, *[], **kwargs_1580)
            
            # Applying the binary operator '|=' (line 558)
            result_ior_1582 = python_operator(stypy.reporting.localization.Localization(__file__, 558, 12), '|=', temp_1577, can_store_elements_call_result_1581)
            # Assigning a type to the variable 'temp' (line 558)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 12), 'temp', result_ior_1582)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'temp' (line 560)
        temp_1583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'stypy_return_type', temp_1583)
        
        # ################# End of 'can_store_elements(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'can_store_elements' in the type store
        # Getting the type of 'stypy_return_type' (line 555)
        stypy_return_type_1584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1584)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'can_store_elements'
        return stypy_return_type_1584


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
        False_1585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 15), 'False')
        # Assigning a type to the variable 'temp' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'temp', False_1585)
        
        # Getting the type of 'self' (line 564)
        self_1586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 21), 'self')
        # Obtaining the member 'types' of a type (line 564)
        types_1587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 21), self_1586, 'types')
        # Assigning a type to the variable 'types_1587' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'types_1587', types_1587)
        # Testing if the for loop is going to be iterated (line 564)
        # Testing the type of a for loop iterable (line 564)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 564, 8), types_1587)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 564, 8), types_1587):
            # Getting the type of the for loop variable (line 564)
            for_loop_var_1588 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 564, 8), types_1587)
            # Assigning a type to the variable 'type_' (line 564)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'type_', for_loop_var_1588)
            # SSA begins for a for statement (line 564)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'temp' (line 565)
            temp_1589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 12), 'temp')
            
            # Call to can_store_keypairs(...): (line 565)
            # Processing the call keyword arguments (line 565)
            kwargs_1592 = {}
            # Getting the type of 'type_' (line 565)
            type__1590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 20), 'type_', False)
            # Obtaining the member 'can_store_keypairs' of a type (line 565)
            can_store_keypairs_1591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 20), type__1590, 'can_store_keypairs')
            # Calling can_store_keypairs(args, kwargs) (line 565)
            can_store_keypairs_call_result_1593 = invoke(stypy.reporting.localization.Localization(__file__, 565, 20), can_store_keypairs_1591, *[], **kwargs_1592)
            
            # Applying the binary operator '|=' (line 565)
            result_ior_1594 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 12), '|=', temp_1589, can_store_keypairs_call_result_1593)
            # Assigning a type to the variable 'temp' (line 565)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 12), 'temp', result_ior_1594)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'temp' (line 567)
        temp_1595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 567)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'stypy_return_type', temp_1595)
        
        # ################# End of 'can_store_keypairs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'can_store_keypairs' in the type store
        # Getting the type of 'stypy_return_type' (line 562)
        stypy_return_type_1596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1596)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'can_store_keypairs'
        return stypy_return_type_1596


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
        list_1597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 570)
        
        # Assigning a type to the variable 'errors' (line 570)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'errors', list_1597)
        
        # Assigning a Name to a Name (line 572):
        # Getting the type of 'None' (line 572)
        None_1598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 15), 'None')
        # Assigning a type to the variable 'temp' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'temp', None_1598)
        
        # Getting the type of 'self' (line 573)
        self_1599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 21), 'self')
        # Obtaining the member 'types' of a type (line 573)
        types_1600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 21), self_1599, 'types')
        # Assigning a type to the variable 'types_1600' (line 573)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'types_1600', types_1600)
        # Testing if the for loop is going to be iterated (line 573)
        # Testing the type of a for loop iterable (line 573)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 573, 8), types_1600)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 573, 8), types_1600):
            # Getting the type of the for loop variable (line 573)
            for_loop_var_1601 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 573, 8), types_1600)
            # Assigning a type to the variable 'type_' (line 573)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'type_', for_loop_var_1601)
            # SSA begins for a for statement (line 573)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 574):
            
            # Call to get_elements_type(...): (line 574)
            # Processing the call keyword arguments (line 574)
            kwargs_1604 = {}
            # Getting the type of 'type_' (line 574)
            type__1602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 18), 'type_', False)
            # Obtaining the member 'get_elements_type' of a type (line 574)
            get_elements_type_1603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 18), type__1602, 'get_elements_type')
            # Calling get_elements_type(args, kwargs) (line 574)
            get_elements_type_call_result_1605 = invoke(stypy.reporting.localization.Localization(__file__, 574, 18), get_elements_type_1603, *[], **kwargs_1604)
            
            # Assigning a type to the variable 'res' (line 574)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 12), 'res', get_elements_type_call_result_1605)
            
            # Type idiom detected: calculating its left and rigth part (line 575)
            # Getting the type of 'TypeError' (line 575)
            TypeError_1606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 31), 'TypeError')
            # Getting the type of 'res' (line 575)
            res_1607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 26), 'res')
            
            (may_be_1608, more_types_in_union_1609) = may_be_subtype(TypeError_1606, res_1607)

            if may_be_1608:

                if more_types_in_union_1609:
                    # Runtime conditional SSA (line 575)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'res' (line 575)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'res', remove_not_subtype_from_union(res_1607, TypeError))
                
                # Call to append(...): (line 576)
                # Processing the call arguments (line 576)
                # Getting the type of 'temp' (line 576)
                temp_1612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 30), 'temp', False)
                # Processing the call keyword arguments (line 576)
                kwargs_1613 = {}
                # Getting the type of 'errors' (line 576)
                errors_1610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 16), 'errors', False)
                # Obtaining the member 'append' of a type (line 576)
                append_1611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 16), errors_1610, 'append')
                # Calling append(args, kwargs) (line 576)
                append_call_result_1614 = invoke(stypy.reporting.localization.Localization(__file__, 576, 16), append_1611, *[temp_1612], **kwargs_1613)
                

                if more_types_in_union_1609:
                    # Runtime conditional SSA for else branch (line 575)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_1608) or more_types_in_union_1609):
                # Assigning a type to the variable 'res' (line 575)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'res', remove_subtype_from_union(res_1607, TypeError))
                
                # Assigning a Call to a Name (line 578):
                
                # Call to add(...): (line 578)
                # Processing the call arguments (line 578)
                # Getting the type of 'temp' (line 578)
                temp_1617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 37), 'temp', False)
                # Getting the type of 'res' (line 578)
                res_1618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 43), 'res', False)
                # Processing the call keyword arguments (line 578)
                kwargs_1619 = {}
                # Getting the type of 'UnionType' (line 578)
                UnionType_1615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 23), 'UnionType', False)
                # Obtaining the member 'add' of a type (line 578)
                add_1616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 23), UnionType_1615, 'add')
                # Calling add(args, kwargs) (line 578)
                add_call_result_1620 = invoke(stypy.reporting.localization.Localization(__file__, 578, 23), add_1616, *[temp_1617, res_1618], **kwargs_1619)
                
                # Assigning a type to the variable 'temp' (line 578)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 16), 'temp', add_call_result_1620)

                if (may_be_1608 and more_types_in_union_1609):
                    # SSA join for if statement (line 575)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 581)
        # Processing the call arguments (line 581)
        # Getting the type of 'errors' (line 581)
        errors_1622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 15), 'errors', False)
        # Processing the call keyword arguments (line 581)
        kwargs_1623 = {}
        # Getting the type of 'len' (line 581)
        len_1621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 11), 'len', False)
        # Calling len(args, kwargs) (line 581)
        len_call_result_1624 = invoke(stypy.reporting.localization.Localization(__file__, 581, 11), len_1621, *[errors_1622], **kwargs_1623)
        
        
        # Call to len(...): (line 581)
        # Processing the call arguments (line 581)
        # Getting the type of 'self' (line 581)
        self_1626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 581)
        types_1627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 30), self_1626, 'types')
        # Processing the call keyword arguments (line 581)
        kwargs_1628 = {}
        # Getting the type of 'len' (line 581)
        len_1625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 26), 'len', False)
        # Calling len(args, kwargs) (line 581)
        len_call_result_1629 = invoke(stypy.reporting.localization.Localization(__file__, 581, 26), len_1625, *[types_1627], **kwargs_1628)
        
        # Applying the binary operator '==' (line 581)
        result_eq_1630 = python_operator(stypy.reporting.localization.Localization(__file__, 581, 11), '==', len_call_result_1624, len_call_result_1629)
        
        # Testing if the type of an if condition is none (line 581)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 581, 8), result_eq_1630):
            
            # Getting the type of 'errors' (line 588)
            errors_1643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 25), 'errors')
            # Assigning a type to the variable 'errors_1643' (line 588)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 12), 'errors_1643', errors_1643)
            # Testing if the for loop is going to be iterated (line 588)
            # Testing the type of a for loop iterable (line 588)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 588, 12), errors_1643)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 588, 12), errors_1643):
                # Getting the type of the for loop variable (line 588)
                for_loop_var_1644 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 588, 12), errors_1643)
                # Assigning a type to the variable 'error' (line 588)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 12), 'error', for_loop_var_1644)
                # SSA begins for a for statement (line 588)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 589)
                # Processing the call keyword arguments (line 589)
                kwargs_1647 = {}
                # Getting the type of 'error' (line 589)
                error_1645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 589)
                turn_to_warning_1646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 16), error_1645, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 589)
                turn_to_warning_call_result_1648 = invoke(stypy.reporting.localization.Localization(__file__, 589, 16), turn_to_warning_1646, *[], **kwargs_1647)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 581)
            if_condition_1631 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 581, 8), result_eq_1630)
            # Assigning a type to the variable 'if_condition_1631' (line 581)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'if_condition_1631', if_condition_1631)
            # SSA begins for if statement (line 581)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 582)
            # Processing the call arguments (line 582)
            # Getting the type of 'None' (line 582)
            None_1633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 29), 'None', False)
            
            # Call to format(...): (line 582)
            # Processing the call arguments (line 582)
            str_1636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 16), 'str', 'get_elements_type')
            # Getting the type of 'self' (line 583)
            self_1637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 37), 'self', False)
            # Obtaining the member 'types' of a type (line 583)
            types_1638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 37), self_1637, 'types')
            # Processing the call keyword arguments (line 582)
            kwargs_1639 = {}
            str_1634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 35), 'str', "None of the possible types ('{1}') can invoke the member '{0}'")
            # Obtaining the member 'format' of a type (line 582)
            format_1635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 35), str_1634, 'format')
            # Calling format(args, kwargs) (line 582)
            format_call_result_1640 = invoke(stypy.reporting.localization.Localization(__file__, 582, 35), format_1635, *[str_1636, types_1638], **kwargs_1639)
            
            # Processing the call keyword arguments (line 582)
            kwargs_1641 = {}
            # Getting the type of 'TypeError' (line 582)
            TypeError_1632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 582)
            TypeError_call_result_1642 = invoke(stypy.reporting.localization.Localization(__file__, 582, 19), TypeError_1632, *[None_1633, format_call_result_1640], **kwargs_1641)
            
            # Assigning a type to the variable 'stypy_return_type' (line 582)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 12), 'stypy_return_type', TypeError_call_result_1642)
            # SSA branch for the else part of an if statement (line 581)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 588)
            errors_1643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 25), 'errors')
            # Assigning a type to the variable 'errors_1643' (line 588)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 12), 'errors_1643', errors_1643)
            # Testing if the for loop is going to be iterated (line 588)
            # Testing the type of a for loop iterable (line 588)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 588, 12), errors_1643)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 588, 12), errors_1643):
                # Getting the type of the for loop variable (line 588)
                for_loop_var_1644 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 588, 12), errors_1643)
                # Assigning a type to the variable 'error' (line 588)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 12), 'error', for_loop_var_1644)
                # SSA begins for a for statement (line 588)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 589)
                # Processing the call keyword arguments (line 589)
                kwargs_1647 = {}
                # Getting the type of 'error' (line 589)
                error_1645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 589)
                turn_to_warning_1646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 16), error_1645, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 589)
                turn_to_warning_call_result_1648 = invoke(stypy.reporting.localization.Localization(__file__, 589, 16), turn_to_warning_1646, *[], **kwargs_1647)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 581)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'temp' (line 591)
        temp_1649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 591)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'stypy_return_type', temp_1649)
        
        # ################# End of 'get_elements_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_elements_type' in the type store
        # Getting the type of 'stypy_return_type' (line 569)
        stypy_return_type_1650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1650)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_elements_type'
        return stypy_return_type_1650


    @norecursion
    def set_elements_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 593)
        True_1651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 79), 'True')
        defaults = [True_1651]
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
        list_1652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 594)
        
        # Assigning a type to the variable 'errors' (line 594)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'errors', list_1652)
        
        # Assigning a Name to a Name (line 596):
        # Getting the type of 'None' (line 596)
        None_1653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 15), 'None')
        # Assigning a type to the variable 'temp' (line 596)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'temp', None_1653)
        
        # Getting the type of 'self' (line 597)
        self_1654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 21), 'self')
        # Obtaining the member 'types' of a type (line 597)
        types_1655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 21), self_1654, 'types')
        # Assigning a type to the variable 'types_1655' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'types_1655', types_1655)
        # Testing if the for loop is going to be iterated (line 597)
        # Testing the type of a for loop iterable (line 597)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 597, 8), types_1655)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 597, 8), types_1655):
            # Getting the type of the for loop variable (line 597)
            for_loop_var_1656 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 597, 8), types_1655)
            # Assigning a type to the variable 'type_' (line 597)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'type_', for_loop_var_1656)
            # SSA begins for a for statement (line 597)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 598):
            
            # Call to set_elements_type(...): (line 598)
            # Processing the call arguments (line 598)
            # Getting the type of 'localization' (line 598)
            localization_1659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 42), 'localization', False)
            # Getting the type of 'elements_type' (line 598)
            elements_type_1660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 56), 'elements_type', False)
            # Getting the type of 'record_annotation' (line 598)
            record_annotation_1661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 71), 'record_annotation', False)
            # Processing the call keyword arguments (line 598)
            kwargs_1662 = {}
            # Getting the type of 'type_' (line 598)
            type__1657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 18), 'type_', False)
            # Obtaining the member 'set_elements_type' of a type (line 598)
            set_elements_type_1658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 18), type__1657, 'set_elements_type')
            # Calling set_elements_type(args, kwargs) (line 598)
            set_elements_type_call_result_1663 = invoke(stypy.reporting.localization.Localization(__file__, 598, 18), set_elements_type_1658, *[localization_1659, elements_type_1660, record_annotation_1661], **kwargs_1662)
            
            # Assigning a type to the variable 'res' (line 598)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 12), 'res', set_elements_type_call_result_1663)
            
            # Type idiom detected: calculating its left and rigth part (line 599)
            # Getting the type of 'TypeError' (line 599)
            TypeError_1664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 31), 'TypeError')
            # Getting the type of 'res' (line 599)
            res_1665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 26), 'res')
            
            (may_be_1666, more_types_in_union_1667) = may_be_subtype(TypeError_1664, res_1665)

            if may_be_1666:

                if more_types_in_union_1667:
                    # Runtime conditional SSA (line 599)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'res' (line 599)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 12), 'res', remove_not_subtype_from_union(res_1665, TypeError))
                
                # Call to append(...): (line 600)
                # Processing the call arguments (line 600)
                # Getting the type of 'temp' (line 600)
                temp_1670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 30), 'temp', False)
                # Processing the call keyword arguments (line 600)
                kwargs_1671 = {}
                # Getting the type of 'errors' (line 600)
                errors_1668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 16), 'errors', False)
                # Obtaining the member 'append' of a type (line 600)
                append_1669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 16), errors_1668, 'append')
                # Calling append(args, kwargs) (line 600)
                append_call_result_1672 = invoke(stypy.reporting.localization.Localization(__file__, 600, 16), append_1669, *[temp_1670], **kwargs_1671)
                

                if more_types_in_union_1667:
                    # SSA join for if statement (line 599)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 603)
        # Processing the call arguments (line 603)
        # Getting the type of 'errors' (line 603)
        errors_1674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 15), 'errors', False)
        # Processing the call keyword arguments (line 603)
        kwargs_1675 = {}
        # Getting the type of 'len' (line 603)
        len_1673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 11), 'len', False)
        # Calling len(args, kwargs) (line 603)
        len_call_result_1676 = invoke(stypy.reporting.localization.Localization(__file__, 603, 11), len_1673, *[errors_1674], **kwargs_1675)
        
        
        # Call to len(...): (line 603)
        # Processing the call arguments (line 603)
        # Getting the type of 'self' (line 603)
        self_1678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 603)
        types_1679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 30), self_1678, 'types')
        # Processing the call keyword arguments (line 603)
        kwargs_1680 = {}
        # Getting the type of 'len' (line 603)
        len_1677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 26), 'len', False)
        # Calling len(args, kwargs) (line 603)
        len_call_result_1681 = invoke(stypy.reporting.localization.Localization(__file__, 603, 26), len_1677, *[types_1679], **kwargs_1680)
        
        # Applying the binary operator '==' (line 603)
        result_eq_1682 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 11), '==', len_call_result_1676, len_call_result_1681)
        
        # Testing if the type of an if condition is none (line 603)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 603, 8), result_eq_1682):
            
            # Getting the type of 'errors' (line 610)
            errors_1695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 25), 'errors')
            # Assigning a type to the variable 'errors_1695' (line 610)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'errors_1695', errors_1695)
            # Testing if the for loop is going to be iterated (line 610)
            # Testing the type of a for loop iterable (line 610)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 610, 12), errors_1695)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 610, 12), errors_1695):
                # Getting the type of the for loop variable (line 610)
                for_loop_var_1696 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 610, 12), errors_1695)
                # Assigning a type to the variable 'error' (line 610)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'error', for_loop_var_1696)
                # SSA begins for a for statement (line 610)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 611)
                # Processing the call keyword arguments (line 611)
                kwargs_1699 = {}
                # Getting the type of 'error' (line 611)
                error_1697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 611)
                turn_to_warning_1698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 16), error_1697, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 611)
                turn_to_warning_call_result_1700 = invoke(stypy.reporting.localization.Localization(__file__, 611, 16), turn_to_warning_1698, *[], **kwargs_1699)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 603)
            if_condition_1683 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 603, 8), result_eq_1682)
            # Assigning a type to the variable 'if_condition_1683' (line 603)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'if_condition_1683', if_condition_1683)
            # SSA begins for if statement (line 603)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 604)
            # Processing the call arguments (line 604)
            # Getting the type of 'localization' (line 604)
            localization_1685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 29), 'localization', False)
            
            # Call to format(...): (line 604)
            # Processing the call arguments (line 604)
            str_1688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 16), 'str', 'set_elements_type')
            # Getting the type of 'self' (line 605)
            self_1689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 37), 'self', False)
            # Obtaining the member 'types' of a type (line 605)
            types_1690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 37), self_1689, 'types')
            # Processing the call keyword arguments (line 604)
            kwargs_1691 = {}
            str_1686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 43), 'str', "None of the possible types ('{1}') can invoke the member '{0}'")
            # Obtaining the member 'format' of a type (line 604)
            format_1687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 43), str_1686, 'format')
            # Calling format(args, kwargs) (line 604)
            format_call_result_1692 = invoke(stypy.reporting.localization.Localization(__file__, 604, 43), format_1687, *[str_1688, types_1690], **kwargs_1691)
            
            # Processing the call keyword arguments (line 604)
            kwargs_1693 = {}
            # Getting the type of 'TypeError' (line 604)
            TypeError_1684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 604)
            TypeError_call_result_1694 = invoke(stypy.reporting.localization.Localization(__file__, 604, 19), TypeError_1684, *[localization_1685, format_call_result_1692], **kwargs_1693)
            
            # Assigning a type to the variable 'stypy_return_type' (line 604)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 12), 'stypy_return_type', TypeError_call_result_1694)
            # SSA branch for the else part of an if statement (line 603)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 610)
            errors_1695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 25), 'errors')
            # Assigning a type to the variable 'errors_1695' (line 610)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'errors_1695', errors_1695)
            # Testing if the for loop is going to be iterated (line 610)
            # Testing the type of a for loop iterable (line 610)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 610, 12), errors_1695)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 610, 12), errors_1695):
                # Getting the type of the for loop variable (line 610)
                for_loop_var_1696 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 610, 12), errors_1695)
                # Assigning a type to the variable 'error' (line 610)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'error', for_loop_var_1696)
                # SSA begins for a for statement (line 610)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 611)
                # Processing the call keyword arguments (line 611)
                kwargs_1699 = {}
                # Getting the type of 'error' (line 611)
                error_1697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 611)
                turn_to_warning_1698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 16), error_1697, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 611)
                turn_to_warning_call_result_1700 = invoke(stypy.reporting.localization.Localization(__file__, 611, 16), turn_to_warning_1698, *[], **kwargs_1699)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 603)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'temp' (line 613)
        temp_1701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 8), 'stypy_return_type', temp_1701)
        
        # ################# End of 'set_elements_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_elements_type' in the type store
        # Getting the type of 'stypy_return_type' (line 593)
        stypy_return_type_1702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1702)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_elements_type'
        return stypy_return_type_1702


    @norecursion
    def add_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 615)
        True_1703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 62), 'True')
        defaults = [True_1703]
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
        list_1704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 616)
        
        # Assigning a type to the variable 'errors' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 8), 'errors', list_1704)
        
        # Assigning a Name to a Name (line 618):
        # Getting the type of 'None' (line 618)
        None_1705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 15), 'None')
        # Assigning a type to the variable 'temp' (line 618)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'temp', None_1705)
        
        # Getting the type of 'self' (line 619)
        self_1706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 21), 'self')
        # Obtaining the member 'types' of a type (line 619)
        types_1707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 21), self_1706, 'types')
        # Assigning a type to the variable 'types_1707' (line 619)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), 'types_1707', types_1707)
        # Testing if the for loop is going to be iterated (line 619)
        # Testing the type of a for loop iterable (line 619)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 619, 8), types_1707)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 619, 8), types_1707):
            # Getting the type of the for loop variable (line 619)
            for_loop_var_1708 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 619, 8), types_1707)
            # Assigning a type to the variable 'type_' (line 619)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), 'type_', for_loop_var_1708)
            # SSA begins for a for statement (line 619)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 620):
            
            # Call to add_type(...): (line 620)
            # Processing the call arguments (line 620)
            # Getting the type of 'localization' (line 620)
            localization_1711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 33), 'localization', False)
            # Getting the type of 'type_' (line 620)
            type__1712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 47), 'type_', False)
            # Getting the type of 'record_annotation' (line 620)
            record_annotation_1713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 54), 'record_annotation', False)
            # Processing the call keyword arguments (line 620)
            kwargs_1714 = {}
            # Getting the type of 'type_' (line 620)
            type__1709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 18), 'type_', False)
            # Obtaining the member 'add_type' of a type (line 620)
            add_type_1710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 18), type__1709, 'add_type')
            # Calling add_type(args, kwargs) (line 620)
            add_type_call_result_1715 = invoke(stypy.reporting.localization.Localization(__file__, 620, 18), add_type_1710, *[localization_1711, type__1712, record_annotation_1713], **kwargs_1714)
            
            # Assigning a type to the variable 'res' (line 620)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 12), 'res', add_type_call_result_1715)
            
            # Type idiom detected: calculating its left and rigth part (line 621)
            # Getting the type of 'TypeError' (line 621)
            TypeError_1716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 31), 'TypeError')
            # Getting the type of 'res' (line 621)
            res_1717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 26), 'res')
            
            (may_be_1718, more_types_in_union_1719) = may_be_subtype(TypeError_1716, res_1717)

            if may_be_1718:

                if more_types_in_union_1719:
                    # Runtime conditional SSA (line 621)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'res' (line 621)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 12), 'res', remove_not_subtype_from_union(res_1717, TypeError))
                
                # Call to append(...): (line 622)
                # Processing the call arguments (line 622)
                # Getting the type of 'temp' (line 622)
                temp_1722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 30), 'temp', False)
                # Processing the call keyword arguments (line 622)
                kwargs_1723 = {}
                # Getting the type of 'errors' (line 622)
                errors_1720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 16), 'errors', False)
                # Obtaining the member 'append' of a type (line 622)
                append_1721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 16), errors_1720, 'append')
                # Calling append(args, kwargs) (line 622)
                append_call_result_1724 = invoke(stypy.reporting.localization.Localization(__file__, 622, 16), append_1721, *[temp_1722], **kwargs_1723)
                

                if more_types_in_union_1719:
                    # SSA join for if statement (line 621)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 625)
        # Processing the call arguments (line 625)
        # Getting the type of 'errors' (line 625)
        errors_1726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 15), 'errors', False)
        # Processing the call keyword arguments (line 625)
        kwargs_1727 = {}
        # Getting the type of 'len' (line 625)
        len_1725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 11), 'len', False)
        # Calling len(args, kwargs) (line 625)
        len_call_result_1728 = invoke(stypy.reporting.localization.Localization(__file__, 625, 11), len_1725, *[errors_1726], **kwargs_1727)
        
        
        # Call to len(...): (line 625)
        # Processing the call arguments (line 625)
        # Getting the type of 'self' (line 625)
        self_1730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 625)
        types_1731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 30), self_1730, 'types')
        # Processing the call keyword arguments (line 625)
        kwargs_1732 = {}
        # Getting the type of 'len' (line 625)
        len_1729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 26), 'len', False)
        # Calling len(args, kwargs) (line 625)
        len_call_result_1733 = invoke(stypy.reporting.localization.Localization(__file__, 625, 26), len_1729, *[types_1731], **kwargs_1732)
        
        # Applying the binary operator '==' (line 625)
        result_eq_1734 = python_operator(stypy.reporting.localization.Localization(__file__, 625, 11), '==', len_call_result_1728, len_call_result_1733)
        
        # Testing if the type of an if condition is none (line 625)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 625, 8), result_eq_1734):
            
            # Getting the type of 'errors' (line 632)
            errors_1747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 25), 'errors')
            # Assigning a type to the variable 'errors_1747' (line 632)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 12), 'errors_1747', errors_1747)
            # Testing if the for loop is going to be iterated (line 632)
            # Testing the type of a for loop iterable (line 632)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 632, 12), errors_1747)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 632, 12), errors_1747):
                # Getting the type of the for loop variable (line 632)
                for_loop_var_1748 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 632, 12), errors_1747)
                # Assigning a type to the variable 'error' (line 632)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 12), 'error', for_loop_var_1748)
                # SSA begins for a for statement (line 632)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 633)
                # Processing the call keyword arguments (line 633)
                kwargs_1751 = {}
                # Getting the type of 'error' (line 633)
                error_1749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 633)
                turn_to_warning_1750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 16), error_1749, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 633)
                turn_to_warning_call_result_1752 = invoke(stypy.reporting.localization.Localization(__file__, 633, 16), turn_to_warning_1750, *[], **kwargs_1751)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 625)
            if_condition_1735 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 625, 8), result_eq_1734)
            # Assigning a type to the variable 'if_condition_1735' (line 625)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 8), 'if_condition_1735', if_condition_1735)
            # SSA begins for if statement (line 625)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 626)
            # Processing the call arguments (line 626)
            # Getting the type of 'localization' (line 626)
            localization_1737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 29), 'localization', False)
            
            # Call to format(...): (line 626)
            # Processing the call arguments (line 626)
            str_1740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 16), 'str', 'add_type')
            # Getting the type of 'self' (line 627)
            self_1741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 28), 'self', False)
            # Obtaining the member 'types' of a type (line 627)
            types_1742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 28), self_1741, 'types')
            # Processing the call keyword arguments (line 626)
            kwargs_1743 = {}
            str_1738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 43), 'str', "None of the possible types ('{1}') can invoke the member '{0}'")
            # Obtaining the member 'format' of a type (line 626)
            format_1739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 43), str_1738, 'format')
            # Calling format(args, kwargs) (line 626)
            format_call_result_1744 = invoke(stypy.reporting.localization.Localization(__file__, 626, 43), format_1739, *[str_1740, types_1742], **kwargs_1743)
            
            # Processing the call keyword arguments (line 626)
            kwargs_1745 = {}
            # Getting the type of 'TypeError' (line 626)
            TypeError_1736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 626)
            TypeError_call_result_1746 = invoke(stypy.reporting.localization.Localization(__file__, 626, 19), TypeError_1736, *[localization_1737, format_call_result_1744], **kwargs_1745)
            
            # Assigning a type to the variable 'stypy_return_type' (line 626)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 12), 'stypy_return_type', TypeError_call_result_1746)
            # SSA branch for the else part of an if statement (line 625)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 632)
            errors_1747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 25), 'errors')
            # Assigning a type to the variable 'errors_1747' (line 632)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 12), 'errors_1747', errors_1747)
            # Testing if the for loop is going to be iterated (line 632)
            # Testing the type of a for loop iterable (line 632)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 632, 12), errors_1747)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 632, 12), errors_1747):
                # Getting the type of the for loop variable (line 632)
                for_loop_var_1748 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 632, 12), errors_1747)
                # Assigning a type to the variable 'error' (line 632)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 12), 'error', for_loop_var_1748)
                # SSA begins for a for statement (line 632)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 633)
                # Processing the call keyword arguments (line 633)
                kwargs_1751 = {}
                # Getting the type of 'error' (line 633)
                error_1749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 633)
                turn_to_warning_1750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 16), error_1749, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 633)
                turn_to_warning_call_result_1752 = invoke(stypy.reporting.localization.Localization(__file__, 633, 16), turn_to_warning_1750, *[], **kwargs_1751)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 625)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'temp' (line 635)
        temp_1753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 635)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 8), 'stypy_return_type', temp_1753)
        
        # ################# End of 'add_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_type' in the type store
        # Getting the type of 'stypy_return_type' (line 615)
        stypy_return_type_1754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1754)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_type'
        return stypy_return_type_1754


    @norecursion
    def add_types_from_list(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 637)
        True_1755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 77), 'True')
        defaults = [True_1755]
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
        list_1756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 638)
        
        # Assigning a type to the variable 'errors' (line 638)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 8), 'errors', list_1756)
        
        # Assigning a Name to a Name (line 640):
        # Getting the type of 'None' (line 640)
        None_1757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 15), 'None')
        # Assigning a type to the variable 'temp' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'temp', None_1757)
        
        # Getting the type of 'self' (line 641)
        self_1758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 21), 'self')
        # Obtaining the member 'types' of a type (line 641)
        types_1759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 21), self_1758, 'types')
        # Assigning a type to the variable 'types_1759' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'types_1759', types_1759)
        # Testing if the for loop is going to be iterated (line 641)
        # Testing the type of a for loop iterable (line 641)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 641, 8), types_1759)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 641, 8), types_1759):
            # Getting the type of the for loop variable (line 641)
            for_loop_var_1760 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 641, 8), types_1759)
            # Assigning a type to the variable 'type_' (line 641)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'type_', for_loop_var_1760)
            # SSA begins for a for statement (line 641)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 642):
            
            # Call to add_types_from_list(...): (line 642)
            # Processing the call arguments (line 642)
            # Getting the type of 'localization' (line 642)
            localization_1763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 44), 'localization', False)
            # Getting the type of 'type_list' (line 642)
            type_list_1764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 58), 'type_list', False)
            # Getting the type of 'record_annotation' (line 642)
            record_annotation_1765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 69), 'record_annotation', False)
            # Processing the call keyword arguments (line 642)
            kwargs_1766 = {}
            # Getting the type of 'type_' (line 642)
            type__1761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 18), 'type_', False)
            # Obtaining the member 'add_types_from_list' of a type (line 642)
            add_types_from_list_1762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 18), type__1761, 'add_types_from_list')
            # Calling add_types_from_list(args, kwargs) (line 642)
            add_types_from_list_call_result_1767 = invoke(stypy.reporting.localization.Localization(__file__, 642, 18), add_types_from_list_1762, *[localization_1763, type_list_1764, record_annotation_1765], **kwargs_1766)
            
            # Assigning a type to the variable 'res' (line 642)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 12), 'res', add_types_from_list_call_result_1767)
            
            # Type idiom detected: calculating its left and rigth part (line 643)
            # Getting the type of 'TypeError' (line 643)
            TypeError_1768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 31), 'TypeError')
            # Getting the type of 'res' (line 643)
            res_1769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 26), 'res')
            
            (may_be_1770, more_types_in_union_1771) = may_be_subtype(TypeError_1768, res_1769)

            if may_be_1770:

                if more_types_in_union_1771:
                    # Runtime conditional SSA (line 643)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'res' (line 643)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 12), 'res', remove_not_subtype_from_union(res_1769, TypeError))
                
                # Call to append(...): (line 644)
                # Processing the call arguments (line 644)
                # Getting the type of 'temp' (line 644)
                temp_1774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 30), 'temp', False)
                # Processing the call keyword arguments (line 644)
                kwargs_1775 = {}
                # Getting the type of 'errors' (line 644)
                errors_1772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 16), 'errors', False)
                # Obtaining the member 'append' of a type (line 644)
                append_1773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 16), errors_1772, 'append')
                # Calling append(args, kwargs) (line 644)
                append_call_result_1776 = invoke(stypy.reporting.localization.Localization(__file__, 644, 16), append_1773, *[temp_1774], **kwargs_1775)
                

                if more_types_in_union_1771:
                    # SSA join for if statement (line 643)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 647)
        # Processing the call arguments (line 647)
        # Getting the type of 'errors' (line 647)
        errors_1778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 15), 'errors', False)
        # Processing the call keyword arguments (line 647)
        kwargs_1779 = {}
        # Getting the type of 'len' (line 647)
        len_1777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 11), 'len', False)
        # Calling len(args, kwargs) (line 647)
        len_call_result_1780 = invoke(stypy.reporting.localization.Localization(__file__, 647, 11), len_1777, *[errors_1778], **kwargs_1779)
        
        
        # Call to len(...): (line 647)
        # Processing the call arguments (line 647)
        # Getting the type of 'self' (line 647)
        self_1782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 647)
        types_1783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 30), self_1782, 'types')
        # Processing the call keyword arguments (line 647)
        kwargs_1784 = {}
        # Getting the type of 'len' (line 647)
        len_1781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 26), 'len', False)
        # Calling len(args, kwargs) (line 647)
        len_call_result_1785 = invoke(stypy.reporting.localization.Localization(__file__, 647, 26), len_1781, *[types_1783], **kwargs_1784)
        
        # Applying the binary operator '==' (line 647)
        result_eq_1786 = python_operator(stypy.reporting.localization.Localization(__file__, 647, 11), '==', len_call_result_1780, len_call_result_1785)
        
        # Testing if the type of an if condition is none (line 647)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 647, 8), result_eq_1786):
            
            # Getting the type of 'errors' (line 654)
            errors_1799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 25), 'errors')
            # Assigning a type to the variable 'errors_1799' (line 654)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 12), 'errors_1799', errors_1799)
            # Testing if the for loop is going to be iterated (line 654)
            # Testing the type of a for loop iterable (line 654)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 654, 12), errors_1799)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 654, 12), errors_1799):
                # Getting the type of the for loop variable (line 654)
                for_loop_var_1800 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 654, 12), errors_1799)
                # Assigning a type to the variable 'error' (line 654)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 12), 'error', for_loop_var_1800)
                # SSA begins for a for statement (line 654)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 655)
                # Processing the call keyword arguments (line 655)
                kwargs_1803 = {}
                # Getting the type of 'error' (line 655)
                error_1801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 655)
                turn_to_warning_1802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 16), error_1801, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 655)
                turn_to_warning_call_result_1804 = invoke(stypy.reporting.localization.Localization(__file__, 655, 16), turn_to_warning_1802, *[], **kwargs_1803)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 647)
            if_condition_1787 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 647, 8), result_eq_1786)
            # Assigning a type to the variable 'if_condition_1787' (line 647)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'if_condition_1787', if_condition_1787)
            # SSA begins for if statement (line 647)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 648)
            # Processing the call arguments (line 648)
            # Getting the type of 'localization' (line 648)
            localization_1789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 29), 'localization', False)
            
            # Call to format(...): (line 648)
            # Processing the call arguments (line 648)
            str_1792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 16), 'str', 'add_types_from_list')
            # Getting the type of 'self' (line 649)
            self_1793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 39), 'self', False)
            # Obtaining the member 'types' of a type (line 649)
            types_1794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 39), self_1793, 'types')
            # Processing the call keyword arguments (line 648)
            kwargs_1795 = {}
            str_1790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 43), 'str', "None of the possible types ('{1}') can invoke the member '{0}'")
            # Obtaining the member 'format' of a type (line 648)
            format_1791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 43), str_1790, 'format')
            # Calling format(args, kwargs) (line 648)
            format_call_result_1796 = invoke(stypy.reporting.localization.Localization(__file__, 648, 43), format_1791, *[str_1792, types_1794], **kwargs_1795)
            
            # Processing the call keyword arguments (line 648)
            kwargs_1797 = {}
            # Getting the type of 'TypeError' (line 648)
            TypeError_1788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 648)
            TypeError_call_result_1798 = invoke(stypy.reporting.localization.Localization(__file__, 648, 19), TypeError_1788, *[localization_1789, format_call_result_1796], **kwargs_1797)
            
            # Assigning a type to the variable 'stypy_return_type' (line 648)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 12), 'stypy_return_type', TypeError_call_result_1798)
            # SSA branch for the else part of an if statement (line 647)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 654)
            errors_1799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 25), 'errors')
            # Assigning a type to the variable 'errors_1799' (line 654)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 12), 'errors_1799', errors_1799)
            # Testing if the for loop is going to be iterated (line 654)
            # Testing the type of a for loop iterable (line 654)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 654, 12), errors_1799)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 654, 12), errors_1799):
                # Getting the type of the for loop variable (line 654)
                for_loop_var_1800 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 654, 12), errors_1799)
                # Assigning a type to the variable 'error' (line 654)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 12), 'error', for_loop_var_1800)
                # SSA begins for a for statement (line 654)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 655)
                # Processing the call keyword arguments (line 655)
                kwargs_1803 = {}
                # Getting the type of 'error' (line 655)
                error_1801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 655)
                turn_to_warning_1802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 16), error_1801, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 655)
                turn_to_warning_call_result_1804 = invoke(stypy.reporting.localization.Localization(__file__, 655, 16), turn_to_warning_1802, *[], **kwargs_1803)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 647)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'temp' (line 657)
        temp_1805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 657)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 8), 'stypy_return_type', temp_1805)
        
        # ################# End of 'add_types_from_list(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_types_from_list' in the type store
        # Getting the type of 'stypy_return_type' (line 637)
        stypy_return_type_1806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1806)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_types_from_list'
        return stypy_return_type_1806


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
        list_1807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 660)
        
        # Assigning a type to the variable 'errors' (line 660)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 8), 'errors', list_1807)
        
        # Assigning a Name to a Name (line 662):
        # Getting the type of 'None' (line 662)
        None_1808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 15), 'None')
        # Assigning a type to the variable 'temp' (line 662)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 8), 'temp', None_1808)
        
        # Getting the type of 'self' (line 663)
        self_1809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 21), 'self')
        # Obtaining the member 'types' of a type (line 663)
        types_1810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 21), self_1809, 'types')
        # Assigning a type to the variable 'types_1810' (line 663)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'types_1810', types_1810)
        # Testing if the for loop is going to be iterated (line 663)
        # Testing the type of a for loop iterable (line 663)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 663, 8), types_1810)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 663, 8), types_1810):
            # Getting the type of the for loop variable (line 663)
            for_loop_var_1811 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 663, 8), types_1810)
            # Assigning a type to the variable 'type_' (line 663)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'type_', for_loop_var_1811)
            # SSA begins for a for statement (line 663)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 664):
            
            # Call to get_values_from_key(...): (line 664)
            # Processing the call arguments (line 664)
            # Getting the type of 'localization' (line 664)
            localization_1814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 44), 'localization', False)
            # Getting the type of 'key' (line 664)
            key_1815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 58), 'key', False)
            # Processing the call keyword arguments (line 664)
            kwargs_1816 = {}
            # Getting the type of 'type_' (line 664)
            type__1812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 18), 'type_', False)
            # Obtaining the member 'get_values_from_key' of a type (line 664)
            get_values_from_key_1813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 18), type__1812, 'get_values_from_key')
            # Calling get_values_from_key(args, kwargs) (line 664)
            get_values_from_key_call_result_1817 = invoke(stypy.reporting.localization.Localization(__file__, 664, 18), get_values_from_key_1813, *[localization_1814, key_1815], **kwargs_1816)
            
            # Assigning a type to the variable 'res' (line 664)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 12), 'res', get_values_from_key_call_result_1817)
            
            # Type idiom detected: calculating its left and rigth part (line 665)
            # Getting the type of 'TypeError' (line 665)
            TypeError_1818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 31), 'TypeError')
            # Getting the type of 'res' (line 665)
            res_1819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 26), 'res')
            
            (may_be_1820, more_types_in_union_1821) = may_be_subtype(TypeError_1818, res_1819)

            if may_be_1820:

                if more_types_in_union_1821:
                    # Runtime conditional SSA (line 665)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'res' (line 665)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 12), 'res', remove_not_subtype_from_union(res_1819, TypeError))
                
                # Call to append(...): (line 666)
                # Processing the call arguments (line 666)
                # Getting the type of 'temp' (line 666)
                temp_1824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 30), 'temp', False)
                # Processing the call keyword arguments (line 666)
                kwargs_1825 = {}
                # Getting the type of 'errors' (line 666)
                errors_1822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 16), 'errors', False)
                # Obtaining the member 'append' of a type (line 666)
                append_1823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 16), errors_1822, 'append')
                # Calling append(args, kwargs) (line 666)
                append_call_result_1826 = invoke(stypy.reporting.localization.Localization(__file__, 666, 16), append_1823, *[temp_1824], **kwargs_1825)
                

                if more_types_in_union_1821:
                    # Runtime conditional SSA for else branch (line 665)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_1820) or more_types_in_union_1821):
                # Assigning a type to the variable 'res' (line 665)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 12), 'res', remove_subtype_from_union(res_1819, TypeError))
                
                # Assigning a Call to a Name (line 668):
                
                # Call to add(...): (line 668)
                # Processing the call arguments (line 668)
                # Getting the type of 'temp' (line 668)
                temp_1829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 37), 'temp', False)
                # Getting the type of 'res' (line 668)
                res_1830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 43), 'res', False)
                # Processing the call keyword arguments (line 668)
                kwargs_1831 = {}
                # Getting the type of 'UnionType' (line 668)
                UnionType_1827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 23), 'UnionType', False)
                # Obtaining the member 'add' of a type (line 668)
                add_1828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 23), UnionType_1827, 'add')
                # Calling add(args, kwargs) (line 668)
                add_call_result_1832 = invoke(stypy.reporting.localization.Localization(__file__, 668, 23), add_1828, *[temp_1829, res_1830], **kwargs_1831)
                
                # Assigning a type to the variable 'temp' (line 668)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 16), 'temp', add_call_result_1832)

                if (may_be_1820 and more_types_in_union_1821):
                    # SSA join for if statement (line 665)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 671)
        # Processing the call arguments (line 671)
        # Getting the type of 'errors' (line 671)
        errors_1834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 15), 'errors', False)
        # Processing the call keyword arguments (line 671)
        kwargs_1835 = {}
        # Getting the type of 'len' (line 671)
        len_1833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 11), 'len', False)
        # Calling len(args, kwargs) (line 671)
        len_call_result_1836 = invoke(stypy.reporting.localization.Localization(__file__, 671, 11), len_1833, *[errors_1834], **kwargs_1835)
        
        
        # Call to len(...): (line 671)
        # Processing the call arguments (line 671)
        # Getting the type of 'self' (line 671)
        self_1838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 671)
        types_1839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 30), self_1838, 'types')
        # Processing the call keyword arguments (line 671)
        kwargs_1840 = {}
        # Getting the type of 'len' (line 671)
        len_1837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 26), 'len', False)
        # Calling len(args, kwargs) (line 671)
        len_call_result_1841 = invoke(stypy.reporting.localization.Localization(__file__, 671, 26), len_1837, *[types_1839], **kwargs_1840)
        
        # Applying the binary operator '==' (line 671)
        result_eq_1842 = python_operator(stypy.reporting.localization.Localization(__file__, 671, 11), '==', len_call_result_1836, len_call_result_1841)
        
        # Testing if the type of an if condition is none (line 671)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 671, 8), result_eq_1842):
            
            # Getting the type of 'errors' (line 678)
            errors_1855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 25), 'errors')
            # Assigning a type to the variable 'errors_1855' (line 678)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 12), 'errors_1855', errors_1855)
            # Testing if the for loop is going to be iterated (line 678)
            # Testing the type of a for loop iterable (line 678)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 678, 12), errors_1855)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 678, 12), errors_1855):
                # Getting the type of the for loop variable (line 678)
                for_loop_var_1856 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 678, 12), errors_1855)
                # Assigning a type to the variable 'error' (line 678)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 12), 'error', for_loop_var_1856)
                # SSA begins for a for statement (line 678)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 679)
                # Processing the call keyword arguments (line 679)
                kwargs_1859 = {}
                # Getting the type of 'error' (line 679)
                error_1857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 679)
                turn_to_warning_1858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 16), error_1857, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 679)
                turn_to_warning_call_result_1860 = invoke(stypy.reporting.localization.Localization(__file__, 679, 16), turn_to_warning_1858, *[], **kwargs_1859)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 671)
            if_condition_1843 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 671, 8), result_eq_1842)
            # Assigning a type to the variable 'if_condition_1843' (line 671)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 8), 'if_condition_1843', if_condition_1843)
            # SSA begins for if statement (line 671)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 672)
            # Processing the call arguments (line 672)
            # Getting the type of 'localization' (line 672)
            localization_1845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 29), 'localization', False)
            
            # Call to format(...): (line 672)
            # Processing the call arguments (line 672)
            str_1848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 16), 'str', 'get_values_from_key')
            # Getting the type of 'self' (line 673)
            self_1849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 39), 'self', False)
            # Obtaining the member 'types' of a type (line 673)
            types_1850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 39), self_1849, 'types')
            # Processing the call keyword arguments (line 672)
            kwargs_1851 = {}
            str_1846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 43), 'str', "None of the possible types ('{1}') can invoke the member '{0}'")
            # Obtaining the member 'format' of a type (line 672)
            format_1847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 43), str_1846, 'format')
            # Calling format(args, kwargs) (line 672)
            format_call_result_1852 = invoke(stypy.reporting.localization.Localization(__file__, 672, 43), format_1847, *[str_1848, types_1850], **kwargs_1851)
            
            # Processing the call keyword arguments (line 672)
            kwargs_1853 = {}
            # Getting the type of 'TypeError' (line 672)
            TypeError_1844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 672)
            TypeError_call_result_1854 = invoke(stypy.reporting.localization.Localization(__file__, 672, 19), TypeError_1844, *[localization_1845, format_call_result_1852], **kwargs_1853)
            
            # Assigning a type to the variable 'stypy_return_type' (line 672)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 12), 'stypy_return_type', TypeError_call_result_1854)
            # SSA branch for the else part of an if statement (line 671)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 678)
            errors_1855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 25), 'errors')
            # Assigning a type to the variable 'errors_1855' (line 678)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 12), 'errors_1855', errors_1855)
            # Testing if the for loop is going to be iterated (line 678)
            # Testing the type of a for loop iterable (line 678)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 678, 12), errors_1855)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 678, 12), errors_1855):
                # Getting the type of the for loop variable (line 678)
                for_loop_var_1856 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 678, 12), errors_1855)
                # Assigning a type to the variable 'error' (line 678)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 12), 'error', for_loop_var_1856)
                # SSA begins for a for statement (line 678)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 679)
                # Processing the call keyword arguments (line 679)
                kwargs_1859 = {}
                # Getting the type of 'error' (line 679)
                error_1857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 679)
                turn_to_warning_1858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 16), error_1857, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 679)
                turn_to_warning_call_result_1860 = invoke(stypy.reporting.localization.Localization(__file__, 679, 16), turn_to_warning_1858, *[], **kwargs_1859)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 671)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'temp' (line 681)
        temp_1861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 15), 'temp')
        # Assigning a type to the variable 'stypy_return_type' (line 681)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 8), 'stypy_return_type', temp_1861)
        
        # ################# End of 'get_values_from_key(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_values_from_key' in the type store
        # Getting the type of 'stypy_return_type' (line 659)
        stypy_return_type_1862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1862)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_values_from_key'
        return stypy_return_type_1862


    @norecursion
    def add_key_and_value_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 683)
        True_1863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 81), 'True')
        defaults = [True_1863]
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
        list_1864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 684)
        
        # Assigning a type to the variable 'errors' (line 684)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 8), 'errors', list_1864)
        
        # Getting the type of 'self' (line 686)
        self_1865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 21), 'self')
        # Obtaining the member 'types' of a type (line 686)
        types_1866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 21), self_1865, 'types')
        # Assigning a type to the variable 'types_1866' (line 686)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 8), 'types_1866', types_1866)
        # Testing if the for loop is going to be iterated (line 686)
        # Testing the type of a for loop iterable (line 686)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 686, 8), types_1866)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 686, 8), types_1866):
            # Getting the type of the for loop variable (line 686)
            for_loop_var_1867 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 686, 8), types_1866)
            # Assigning a type to the variable 'type_' (line 686)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 8), 'type_', for_loop_var_1867)
            # SSA begins for a for statement (line 686)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 687):
            
            # Call to add_key_and_value_type(...): (line 687)
            # Processing the call arguments (line 687)
            # Getting the type of 'localization' (line 687)
            localization_1870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 48), 'localization', False)
            # Getting the type of 'type_tuple' (line 687)
            type_tuple_1871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 62), 'type_tuple', False)
            # Getting the type of 'record_annotation' (line 687)
            record_annotation_1872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 74), 'record_annotation', False)
            # Processing the call keyword arguments (line 687)
            kwargs_1873 = {}
            # Getting the type of 'type_' (line 687)
            type__1868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 19), 'type_', False)
            # Obtaining the member 'add_key_and_value_type' of a type (line 687)
            add_key_and_value_type_1869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 19), type__1868, 'add_key_and_value_type')
            # Calling add_key_and_value_type(args, kwargs) (line 687)
            add_key_and_value_type_call_result_1874 = invoke(stypy.reporting.localization.Localization(__file__, 687, 19), add_key_and_value_type_1869, *[localization_1870, type_tuple_1871, record_annotation_1872], **kwargs_1873)
            
            # Assigning a type to the variable 'temp' (line 687)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 12), 'temp', add_key_and_value_type_call_result_1874)
            
            # Type idiom detected: calculating its left and rigth part (line 688)
            # Getting the type of 'temp' (line 688)
            temp_1875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 12), 'temp')
            # Getting the type of 'None' (line 688)
            None_1876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 27), 'None')
            
            (may_be_1877, more_types_in_union_1878) = may_not_be_none(temp_1875, None_1876)

            if may_be_1877:

                if more_types_in_union_1878:
                    # Runtime conditional SSA (line 688)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to append(...): (line 689)
                # Processing the call arguments (line 689)
                # Getting the type of 'temp' (line 689)
                temp_1881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 30), 'temp', False)
                # Processing the call keyword arguments (line 689)
                kwargs_1882 = {}
                # Getting the type of 'errors' (line 689)
                errors_1879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 16), 'errors', False)
                # Obtaining the member 'append' of a type (line 689)
                append_1880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 16), errors_1879, 'append')
                # Calling append(args, kwargs) (line 689)
                append_call_result_1883 = invoke(stypy.reporting.localization.Localization(__file__, 689, 16), append_1880, *[temp_1881], **kwargs_1882)
                

                if more_types_in_union_1878:
                    # SSA join for if statement (line 688)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 692)
        # Processing the call arguments (line 692)
        # Getting the type of 'errors' (line 692)
        errors_1885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 15), 'errors', False)
        # Processing the call keyword arguments (line 692)
        kwargs_1886 = {}
        # Getting the type of 'len' (line 692)
        len_1884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 11), 'len', False)
        # Calling len(args, kwargs) (line 692)
        len_call_result_1887 = invoke(stypy.reporting.localization.Localization(__file__, 692, 11), len_1884, *[errors_1885], **kwargs_1886)
        
        
        # Call to len(...): (line 692)
        # Processing the call arguments (line 692)
        # Getting the type of 'self' (line 692)
        self_1889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 692)
        types_1890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 30), self_1889, 'types')
        # Processing the call keyword arguments (line 692)
        kwargs_1891 = {}
        # Getting the type of 'len' (line 692)
        len_1888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 26), 'len', False)
        # Calling len(args, kwargs) (line 692)
        len_call_result_1892 = invoke(stypy.reporting.localization.Localization(__file__, 692, 26), len_1888, *[types_1890], **kwargs_1891)
        
        # Applying the binary operator '==' (line 692)
        result_eq_1893 = python_operator(stypy.reporting.localization.Localization(__file__, 692, 11), '==', len_call_result_1887, len_call_result_1892)
        
        # Testing if the type of an if condition is none (line 692)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 692, 8), result_eq_1893):
            
            # Getting the type of 'errors' (line 699)
            errors_1906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 25), 'errors')
            # Assigning a type to the variable 'errors_1906' (line 699)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 12), 'errors_1906', errors_1906)
            # Testing if the for loop is going to be iterated (line 699)
            # Testing the type of a for loop iterable (line 699)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 699, 12), errors_1906)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 699, 12), errors_1906):
                # Getting the type of the for loop variable (line 699)
                for_loop_var_1907 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 699, 12), errors_1906)
                # Assigning a type to the variable 'error' (line 699)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 12), 'error', for_loop_var_1907)
                # SSA begins for a for statement (line 699)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 700)
                # Processing the call keyword arguments (line 700)
                kwargs_1910 = {}
                # Getting the type of 'error' (line 700)
                error_1908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 700)
                turn_to_warning_1909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 16), error_1908, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 700)
                turn_to_warning_call_result_1911 = invoke(stypy.reporting.localization.Localization(__file__, 700, 16), turn_to_warning_1909, *[], **kwargs_1910)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 692)
            if_condition_1894 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 692, 8), result_eq_1893)
            # Assigning a type to the variable 'if_condition_1894' (line 692)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 8), 'if_condition_1894', if_condition_1894)
            # SSA begins for if statement (line 692)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 693)
            # Processing the call arguments (line 693)
            # Getting the type of 'localization' (line 693)
            localization_1896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 29), 'localization', False)
            
            # Call to format(...): (line 693)
            # Processing the call arguments (line 693)
            str_1899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 16), 'str', 'add_key_and_value_type')
            # Getting the type of 'self' (line 694)
            self_1900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 42), 'self', False)
            # Obtaining the member 'types' of a type (line 694)
            types_1901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 42), self_1900, 'types')
            # Processing the call keyword arguments (line 693)
            kwargs_1902 = {}
            str_1897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 43), 'str', "None of the possible types ('{1}') can invoke the member '{0}'")
            # Obtaining the member 'format' of a type (line 693)
            format_1898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 43), str_1897, 'format')
            # Calling format(args, kwargs) (line 693)
            format_call_result_1903 = invoke(stypy.reporting.localization.Localization(__file__, 693, 43), format_1898, *[str_1899, types_1901], **kwargs_1902)
            
            # Processing the call keyword arguments (line 693)
            kwargs_1904 = {}
            # Getting the type of 'TypeError' (line 693)
            TypeError_1895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 693)
            TypeError_call_result_1905 = invoke(stypy.reporting.localization.Localization(__file__, 693, 19), TypeError_1895, *[localization_1896, format_call_result_1903], **kwargs_1904)
            
            # Assigning a type to the variable 'stypy_return_type' (line 693)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 12), 'stypy_return_type', TypeError_call_result_1905)
            # SSA branch for the else part of an if statement (line 692)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'errors' (line 699)
            errors_1906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 25), 'errors')
            # Assigning a type to the variable 'errors_1906' (line 699)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 12), 'errors_1906', errors_1906)
            # Testing if the for loop is going to be iterated (line 699)
            # Testing the type of a for loop iterable (line 699)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 699, 12), errors_1906)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 699, 12), errors_1906):
                # Getting the type of the for loop variable (line 699)
                for_loop_var_1907 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 699, 12), errors_1906)
                # Assigning a type to the variable 'error' (line 699)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 12), 'error', for_loop_var_1907)
                # SSA begins for a for statement (line 699)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to turn_to_warning(...): (line 700)
                # Processing the call keyword arguments (line 700)
                kwargs_1910 = {}
                # Getting the type of 'error' (line 700)
                error_1908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 16), 'error', False)
                # Obtaining the member 'turn_to_warning' of a type (line 700)
                turn_to_warning_1909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 16), error_1908, 'turn_to_warning')
                # Calling turn_to_warning(args, kwargs) (line 700)
                turn_to_warning_call_result_1911 = invoke(stypy.reporting.localization.Localization(__file__, 700, 16), turn_to_warning_1909, *[], **kwargs_1910)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 692)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'None' (line 702)
        None_1912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'stypy_return_type', None_1912)
        
        # ################# End of 'add_key_and_value_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_key_and_value_type' in the type store
        # Getting the type of 'stypy_return_type' (line 683)
        stypy_return_type_1913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1913)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_key_and_value_type'
        return stypy_return_type_1913


# Assigning a type to the variable 'UnionType' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'UnionType', UnionType)
# Declaration of the 'OrderedUnionType' class
# Getting the type of 'UnionType' (line 705)
UnionType_1914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 23), 'UnionType')

class OrderedUnionType(UnionType_1914, ):
    str_1915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, (-1)), 'str', '\n    A special type of UnionType that maintain the order of its added types and admits repeated elements. This will be\n    used in the future implementation of tuples.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 711)
        None_1916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 29), 'None')
        # Getting the type of 'None' (line 711)
        None_1917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 41), 'None')
        defaults = [None_1916, None_1917]
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
        self_1920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 27), 'self', False)
        # Getting the type of 'type1' (line 712)
        type1_1921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 33), 'type1', False)
        # Getting the type of 'type2' (line 712)
        type2_1922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 40), 'type2', False)
        # Processing the call keyword arguments (line 712)
        kwargs_1923 = {}
        # Getting the type of 'UnionType' (line 712)
        UnionType_1918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'UnionType', False)
        # Obtaining the member '__init__' of a type (line 712)
        init___1919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 8), UnionType_1918, '__init__')
        # Calling __init__(args, kwargs) (line 712)
        init___call_result_1924 = invoke(stypy.reporting.localization.Localization(__file__, 712, 8), init___1919, *[self_1920, type1_1921, type2_1922], **kwargs_1923)
        
        
        # Assigning a List to a Attribute (line 713):
        
        # Obtaining an instance of the builtin type 'list' (line 713)
        list_1925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 713)
        
        # Getting the type of 'self' (line 713)
        self_1926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 8), 'self')
        # Setting the type of the member 'ordered_types' of a type (line 713)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 8), self_1926, 'ordered_types', list_1925)
        
        # Type idiom detected: calculating its left and rigth part (line 715)
        # Getting the type of 'type1' (line 715)
        type1_1927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 8), 'type1')
        # Getting the type of 'None' (line 715)
        None_1928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 24), 'None')
        
        (may_be_1929, more_types_in_union_1930) = may_not_be_none(type1_1927, None_1928)

        if may_be_1929:

            if more_types_in_union_1930:
                # Runtime conditional SSA (line 715)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to append(...): (line 716)
            # Processing the call arguments (line 716)
            # Getting the type of 'type1' (line 716)
            type1_1934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 38), 'type1', False)
            # Processing the call keyword arguments (line 716)
            kwargs_1935 = {}
            # Getting the type of 'self' (line 716)
            self_1931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 12), 'self', False)
            # Obtaining the member 'ordered_types' of a type (line 716)
            ordered_types_1932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 12), self_1931, 'ordered_types')
            # Obtaining the member 'append' of a type (line 716)
            append_1933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 12), ordered_types_1932, 'append')
            # Calling append(args, kwargs) (line 716)
            append_call_result_1936 = invoke(stypy.reporting.localization.Localization(__file__, 716, 12), append_1933, *[type1_1934], **kwargs_1935)
            

            if more_types_in_union_1930:
                # SSA join for if statement (line 715)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 718)
        # Getting the type of 'type2' (line 718)
        type2_1937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'type2')
        # Getting the type of 'None' (line 718)
        None_1938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 24), 'None')
        
        (may_be_1939, more_types_in_union_1940) = may_not_be_none(type2_1937, None_1938)

        if may_be_1939:

            if more_types_in_union_1940:
                # Runtime conditional SSA (line 718)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to append(...): (line 719)
            # Processing the call arguments (line 719)
            # Getting the type of 'type2' (line 719)
            type2_1944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 38), 'type2', False)
            # Processing the call keyword arguments (line 719)
            kwargs_1945 = {}
            # Getting the type of 'self' (line 719)
            self_1941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 12), 'self', False)
            # Obtaining the member 'ordered_types' of a type (line 719)
            ordered_types_1942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 12), self_1941, 'ordered_types')
            # Obtaining the member 'append' of a type (line 719)
            append_1943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 12), ordered_types_1942, 'append')
            # Calling append(args, kwargs) (line 719)
            append_call_result_1946 = invoke(stypy.reporting.localization.Localization(__file__, 719, 12), append_1943, *[type2_1944], **kwargs_1945)
            

            if more_types_in_union_1940:
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
        type1_1947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 11), 'type1')
        # Getting the type of 'None' (line 723)
        None_1948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 20), 'None')
        
        (may_be_1949, more_types_in_union_1950) = may_be_none(type1_1947, None_1948)

        if may_be_1949:

            if more_types_in_union_1950:
                # Runtime conditional SSA (line 723)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to _wrap_type(...): (line 724)
            # Processing the call arguments (line 724)
            # Getting the type of 'type2' (line 724)
            type2_1953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 40), 'type2', False)
            # Processing the call keyword arguments (line 724)
            kwargs_1954 = {}
            # Getting the type of 'UnionType' (line 724)
            UnionType_1951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 19), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 724)
            _wrap_type_1952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 19), UnionType_1951, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 724)
            _wrap_type_call_result_1955 = invoke(stypy.reporting.localization.Localization(__file__, 724, 19), _wrap_type_1952, *[type2_1953], **kwargs_1954)
            
            # Assigning a type to the variable 'stypy_return_type' (line 724)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 12), 'stypy_return_type', _wrap_type_call_result_1955)

            if more_types_in_union_1950:
                # SSA join for if statement (line 723)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'type1' (line 723)
        type1_1956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'type1')
        # Assigning a type to the variable 'type1' (line 723)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'type1', remove_type_from_union(type1_1956, types.NoneType))
        
        # Type idiom detected: calculating its left and rigth part (line 726)
        # Getting the type of 'type2' (line 726)
        type2_1957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 11), 'type2')
        # Getting the type of 'None' (line 726)
        None_1958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 20), 'None')
        
        (may_be_1959, more_types_in_union_1960) = may_be_none(type2_1957, None_1958)

        if may_be_1959:

            if more_types_in_union_1960:
                # Runtime conditional SSA (line 726)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to _wrap_type(...): (line 727)
            # Processing the call arguments (line 727)
            # Getting the type of 'type1' (line 727)
            type1_1963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 40), 'type1', False)
            # Processing the call keyword arguments (line 727)
            kwargs_1964 = {}
            # Getting the type of 'UnionType' (line 727)
            UnionType_1961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 19), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 727)
            _wrap_type_1962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 19), UnionType_1961, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 727)
            _wrap_type_call_result_1965 = invoke(stypy.reporting.localization.Localization(__file__, 727, 19), _wrap_type_1962, *[type1_1963], **kwargs_1964)
            
            # Assigning a type to the variable 'stypy_return_type' (line 727)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 12), 'stypy_return_type', _wrap_type_call_result_1965)

            if more_types_in_union_1960:
                # SSA join for if statement (line 726)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'type2' (line 726)
        type2_1966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 8), 'type2')
        # Assigning a type to the variable 'type2' (line 726)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 8), 'type2', remove_type_from_union(type2_1966, types.NoneType))
        
        # Call to is_undefined_type(...): (line 729)
        # Processing the call arguments (line 729)
        # Getting the type of 'type1' (line 729)
        type1_1973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 107), 'type1', False)
        # Processing the call keyword arguments (line 729)
        kwargs_1974 = {}
        # Getting the type of 'stypy_copy' (line 729)
        stypy_copy_1967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 729)
        python_lib_1968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 11), stypy_copy_1967, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 729)
        python_types_1969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 11), python_lib_1968, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 729)
        type_introspection_1970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 11), python_types_1969, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 729)
        runtime_type_inspection_1971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 11), type_introspection_1970, 'runtime_type_inspection')
        # Obtaining the member 'is_undefined_type' of a type (line 729)
        is_undefined_type_1972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 11), runtime_type_inspection_1971, 'is_undefined_type')
        # Calling is_undefined_type(args, kwargs) (line 729)
        is_undefined_type_call_result_1975 = invoke(stypy.reporting.localization.Localization(__file__, 729, 11), is_undefined_type_1972, *[type1_1973], **kwargs_1974)
        
        # Testing if the type of an if condition is none (line 729)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 729, 8), is_undefined_type_call_result_1975):
            pass
        else:
            
            # Testing the type of an if condition (line 729)
            if_condition_1976 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 729, 8), is_undefined_type_call_result_1975)
            # Assigning a type to the variable 'if_condition_1976' (line 729)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 8), 'if_condition_1976', if_condition_1976)
            # SSA begins for if statement (line 729)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _wrap_type(...): (line 730)
            # Processing the call arguments (line 730)
            # Getting the type of 'type2' (line 730)
            type2_1979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 40), 'type2', False)
            # Processing the call keyword arguments (line 730)
            kwargs_1980 = {}
            # Getting the type of 'UnionType' (line 730)
            UnionType_1977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 19), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 730)
            _wrap_type_1978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 19), UnionType_1977, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 730)
            _wrap_type_call_result_1981 = invoke(stypy.reporting.localization.Localization(__file__, 730, 19), _wrap_type_1978, *[type2_1979], **kwargs_1980)
            
            # Assigning a type to the variable 'stypy_return_type' (line 730)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 12), 'stypy_return_type', _wrap_type_call_result_1981)
            # SSA join for if statement (line 729)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to is_undefined_type(...): (line 731)
        # Processing the call arguments (line 731)
        # Getting the type of 'type2' (line 731)
        type2_1988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 107), 'type2', False)
        # Processing the call keyword arguments (line 731)
        kwargs_1989 = {}
        # Getting the type of 'stypy_copy' (line 731)
        stypy_copy_1982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 731)
        python_lib_1983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 11), stypy_copy_1982, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 731)
        python_types_1984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 11), python_lib_1983, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 731)
        type_introspection_1985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 11), python_types_1984, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 731)
        runtime_type_inspection_1986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 11), type_introspection_1985, 'runtime_type_inspection')
        # Obtaining the member 'is_undefined_type' of a type (line 731)
        is_undefined_type_1987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 11), runtime_type_inspection_1986, 'is_undefined_type')
        # Calling is_undefined_type(args, kwargs) (line 731)
        is_undefined_type_call_result_1990 = invoke(stypy.reporting.localization.Localization(__file__, 731, 11), is_undefined_type_1987, *[type2_1988], **kwargs_1989)
        
        # Testing if the type of an if condition is none (line 731)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 731, 8), is_undefined_type_call_result_1990):
            pass
        else:
            
            # Testing the type of an if condition (line 731)
            if_condition_1991 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 731, 8), is_undefined_type_call_result_1990)
            # Assigning a type to the variable 'if_condition_1991' (line 731)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 8), 'if_condition_1991', if_condition_1991)
            # SSA begins for if statement (line 731)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _wrap_type(...): (line 732)
            # Processing the call arguments (line 732)
            # Getting the type of 'type1' (line 732)
            type1_1994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 40), 'type1', False)
            # Processing the call keyword arguments (line 732)
            kwargs_1995 = {}
            # Getting the type of 'UnionType' (line 732)
            UnionType_1992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 19), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 732)
            _wrap_type_1993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 732, 19), UnionType_1992, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 732)
            _wrap_type_call_result_1996 = invoke(stypy.reporting.localization.Localization(__file__, 732, 19), _wrap_type_1993, *[type1_1994], **kwargs_1995)
            
            # Assigning a type to the variable 'stypy_return_type' (line 732)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 12), 'stypy_return_type', _wrap_type_call_result_1996)
            # SSA join for if statement (line 731)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to is_union_type(...): (line 734)
        # Processing the call arguments (line 734)
        # Getting the type of 'type1' (line 734)
        type1_2003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 103), 'type1', False)
        # Processing the call keyword arguments (line 734)
        kwargs_2004 = {}
        # Getting the type of 'stypy_copy' (line 734)
        stypy_copy_1997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 734)
        python_lib_1998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 11), stypy_copy_1997, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 734)
        python_types_1999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 11), python_lib_1998, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 734)
        type_introspection_2000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 11), python_types_1999, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 734)
        runtime_type_inspection_2001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 11), type_introspection_2000, 'runtime_type_inspection')
        # Obtaining the member 'is_union_type' of a type (line 734)
        is_union_type_2002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 11), runtime_type_inspection_2001, 'is_union_type')
        # Calling is_union_type(args, kwargs) (line 734)
        is_union_type_call_result_2005 = invoke(stypy.reporting.localization.Localization(__file__, 734, 11), is_union_type_2002, *[type1_2003], **kwargs_2004)
        
        # Testing if the type of an if condition is none (line 734)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 734, 8), is_union_type_call_result_2005):
            pass
        else:
            
            # Testing the type of an if condition (line 734)
            if_condition_2006 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 734, 8), is_union_type_call_result_2005)
            # Assigning a type to the variable 'if_condition_2006' (line 734)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 8), 'if_condition_2006', if_condition_2006)
            # SSA begins for if statement (line 734)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _add(...): (line 735)
            # Processing the call arguments (line 735)
            # Getting the type of 'type2' (line 735)
            type2_2009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 30), 'type2', False)
            # Processing the call keyword arguments (line 735)
            kwargs_2010 = {}
            # Getting the type of 'type1' (line 735)
            type1_2007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 19), 'type1', False)
            # Obtaining the member '_add' of a type (line 735)
            _add_2008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 19), type1_2007, '_add')
            # Calling _add(args, kwargs) (line 735)
            _add_call_result_2011 = invoke(stypy.reporting.localization.Localization(__file__, 735, 19), _add_2008, *[type2_2009], **kwargs_2010)
            
            # Assigning a type to the variable 'stypy_return_type' (line 735)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 12), 'stypy_return_type', _add_call_result_2011)
            # SSA join for if statement (line 734)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to is_union_type(...): (line 736)
        # Processing the call arguments (line 736)
        # Getting the type of 'type2' (line 736)
        type2_2018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 103), 'type2', False)
        # Processing the call keyword arguments (line 736)
        kwargs_2019 = {}
        # Getting the type of 'stypy_copy' (line 736)
        stypy_copy_2012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 11), 'stypy_copy', False)
        # Obtaining the member 'python_lib' of a type (line 736)
        python_lib_2013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 11), stypy_copy_2012, 'python_lib')
        # Obtaining the member 'python_types' of a type (line 736)
        python_types_2014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 11), python_lib_2013, 'python_types')
        # Obtaining the member 'type_introspection' of a type (line 736)
        type_introspection_2015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 11), python_types_2014, 'type_introspection')
        # Obtaining the member 'runtime_type_inspection' of a type (line 736)
        runtime_type_inspection_2016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 11), type_introspection_2015, 'runtime_type_inspection')
        # Obtaining the member 'is_union_type' of a type (line 736)
        is_union_type_2017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 11), runtime_type_inspection_2016, 'is_union_type')
        # Calling is_union_type(args, kwargs) (line 736)
        is_union_type_call_result_2020 = invoke(stypy.reporting.localization.Localization(__file__, 736, 11), is_union_type_2017, *[type2_2018], **kwargs_2019)
        
        # Testing if the type of an if condition is none (line 736)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 736, 8), is_union_type_call_result_2020):
            pass
        else:
            
            # Testing the type of an if condition (line 736)
            if_condition_2021 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 736, 8), is_union_type_call_result_2020)
            # Assigning a type to the variable 'if_condition_2021' (line 736)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 8), 'if_condition_2021', if_condition_2021)
            # SSA begins for if statement (line 736)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _add(...): (line 737)
            # Processing the call arguments (line 737)
            # Getting the type of 'type1' (line 737)
            type1_2024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 30), 'type1', False)
            # Processing the call keyword arguments (line 737)
            kwargs_2025 = {}
            # Getting the type of 'type2' (line 737)
            type2_2022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 19), 'type2', False)
            # Obtaining the member '_add' of a type (line 737)
            _add_2023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 19), type2_2022, '_add')
            # Calling _add(args, kwargs) (line 737)
            _add_call_result_2026 = invoke(stypy.reporting.localization.Localization(__file__, 737, 19), _add_2023, *[type1_2024], **kwargs_2025)
            
            # Assigning a type to the variable 'stypy_return_type' (line 737)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 12), 'stypy_return_type', _add_call_result_2026)
            # SSA join for if statement (line 736)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to _wrap_type(...): (line 739)
        # Processing the call arguments (line 739)
        # Getting the type of 'type1' (line 739)
        type1_2029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 32), 'type1', False)
        # Processing the call keyword arguments (line 739)
        kwargs_2030 = {}
        # Getting the type of 'UnionType' (line 739)
        UnionType_2027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 11), 'UnionType', False)
        # Obtaining the member '_wrap_type' of a type (line 739)
        _wrap_type_2028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 11), UnionType_2027, '_wrap_type')
        # Calling _wrap_type(args, kwargs) (line 739)
        _wrap_type_call_result_2031 = invoke(stypy.reporting.localization.Localization(__file__, 739, 11), _wrap_type_2028, *[type1_2029], **kwargs_2030)
        
        
        # Call to _wrap_type(...): (line 739)
        # Processing the call arguments (line 739)
        # Getting the type of 'type2' (line 739)
        type2_2034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 63), 'type2', False)
        # Processing the call keyword arguments (line 739)
        kwargs_2035 = {}
        # Getting the type of 'UnionType' (line 739)
        UnionType_2032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 42), 'UnionType', False)
        # Obtaining the member '_wrap_type' of a type (line 739)
        _wrap_type_2033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 42), UnionType_2032, '_wrap_type')
        # Calling _wrap_type(args, kwargs) (line 739)
        _wrap_type_call_result_2036 = invoke(stypy.reporting.localization.Localization(__file__, 739, 42), _wrap_type_2033, *[type2_2034], **kwargs_2035)
        
        # Applying the binary operator '==' (line 739)
        result_eq_2037 = python_operator(stypy.reporting.localization.Localization(__file__, 739, 11), '==', _wrap_type_call_result_2031, _wrap_type_call_result_2036)
        
        # Testing if the type of an if condition is none (line 739)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 739, 8), result_eq_2037):
            pass
        else:
            
            # Testing the type of an if condition (line 739)
            if_condition_2038 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 739, 8), result_eq_2037)
            # Assigning a type to the variable 'if_condition_2038' (line 739)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 8), 'if_condition_2038', if_condition_2038)
            # SSA begins for if statement (line 739)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _wrap_type(...): (line 740)
            # Processing the call arguments (line 740)
            # Getting the type of 'type1' (line 740)
            type1_2041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 40), 'type1', False)
            # Processing the call keyword arguments (line 740)
            kwargs_2042 = {}
            # Getting the type of 'UnionType' (line 740)
            UnionType_2039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 19), 'UnionType', False)
            # Obtaining the member '_wrap_type' of a type (line 740)
            _wrap_type_2040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 740, 19), UnionType_2039, '_wrap_type')
            # Calling _wrap_type(args, kwargs) (line 740)
            _wrap_type_call_result_2043 = invoke(stypy.reporting.localization.Localization(__file__, 740, 19), _wrap_type_2040, *[type1_2041], **kwargs_2042)
            
            # Assigning a type to the variable 'stypy_return_type' (line 740)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 12), 'stypy_return_type', _wrap_type_call_result_2043)
            # SSA join for if statement (line 739)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to OrderedUnionType(...): (line 742)
        # Processing the call arguments (line 742)
        # Getting the type of 'type1' (line 742)
        type1_2045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 32), 'type1', False)
        # Getting the type of 'type2' (line 742)
        type2_2046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 39), 'type2', False)
        # Processing the call keyword arguments (line 742)
        kwargs_2047 = {}
        # Getting the type of 'OrderedUnionType' (line 742)
        OrderedUnionType_2044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 15), 'OrderedUnionType', False)
        # Calling OrderedUnionType(args, kwargs) (line 742)
        OrderedUnionType_call_result_2048 = invoke(stypy.reporting.localization.Localization(__file__, 742, 15), OrderedUnionType_2044, *[type1_2045, type2_2046], **kwargs_2047)
        
        # Assigning a type to the variable 'stypy_return_type' (line 742)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'stypy_return_type', OrderedUnionType_call_result_2048)
        
        # ################# End of 'add(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add' in the type store
        # Getting the type of 'stypy_return_type' (line 721)
        stypy_return_type_2049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2049)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add'
        return stypy_return_type_2049


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
        self_2052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 29), 'self', False)
        # Getting the type of 'other_type' (line 745)
        other_type_2053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 35), 'other_type', False)
        # Processing the call keyword arguments (line 745)
        kwargs_2054 = {}
        # Getting the type of 'UnionType' (line 745)
        UnionType_2050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 14), 'UnionType', False)
        # Obtaining the member '_add' of a type (line 745)
        _add_2051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 14), UnionType_2050, '_add')
        # Calling _add(args, kwargs) (line 745)
        _add_call_result_2055 = invoke(stypy.reporting.localization.Localization(__file__, 745, 14), _add_2051, *[self_2052, other_type_2053], **kwargs_2054)
        
        # Assigning a type to the variable 'ret' (line 745)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'ret', _add_call_result_2055)
        
        # Call to append(...): (line 746)
        # Processing the call arguments (line 746)
        # Getting the type of 'other_type' (line 746)
        other_type_2059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 34), 'other_type', False)
        # Processing the call keyword arguments (line 746)
        kwargs_2060 = {}
        # Getting the type of 'self' (line 746)
        self_2056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 8), 'self', False)
        # Obtaining the member 'ordered_types' of a type (line 746)
        ordered_types_2057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 8), self_2056, 'ordered_types')
        # Obtaining the member 'append' of a type (line 746)
        append_2058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 8), ordered_types_2057, 'append')
        # Calling append(args, kwargs) (line 746)
        append_call_result_2061 = invoke(stypy.reporting.localization.Localization(__file__, 746, 8), append_2058, *[other_type_2059], **kwargs_2060)
        
        # Getting the type of 'ret' (line 747)
        ret_2062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 15), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 747)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 8), 'stypy_return_type', ret_2062)
        
        # ################# End of '_add(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_add' in the type store
        # Getting the type of 'stypy_return_type' (line 744)
        stypy_return_type_2063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2063)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_add'
        return stypy_return_type_2063


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

        str_2064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, (-1)), 'str', '\n        Obtain the stored types in the same order they were added, including repetitions\n        :return:\n        ')
        # Getting the type of 'self' (line 754)
        self_2065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 15), 'self')
        # Obtaining the member 'ordered_types' of a type (line 754)
        ordered_types_2066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 754, 15), self_2065, 'ordered_types')
        # Assigning a type to the variable 'stypy_return_type' (line 754)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 8), 'stypy_return_type', ordered_types_2066)
        
        # ################# End of 'get_ordered_types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_ordered_types' in the type store
        # Getting the type of 'stypy_return_type' (line 749)
        stypy_return_type_2067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2067)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_ordered_types'
        return stypy_return_type_2067


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

        str_2068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, (-1)), 'str', '\n        Clone the whole OrderedUnionType and its contained types\n        ')
        
        # Assigning a Call to a Name (line 760):
        
        # Call to clone(...): (line 760)
        # Processing the call keyword arguments (line 760)
        kwargs_2075 = {}
        
        # Obtaining the type of the subscript
        int_2069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 34), 'int')
        # Getting the type of 'self' (line 760)
        self_2070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 23), 'self', False)
        # Obtaining the member 'types' of a type (line 760)
        types_2071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 23), self_2070, 'types')
        # Obtaining the member '__getitem__' of a type (line 760)
        getitem___2072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 23), types_2071, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 760)
        subscript_call_result_2073 = invoke(stypy.reporting.localization.Localization(__file__, 760, 23), getitem___2072, int_2069)
        
        # Obtaining the member 'clone' of a type (line 760)
        clone_2074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 23), subscript_call_result_2073, 'clone')
        # Calling clone(args, kwargs) (line 760)
        clone_call_result_2076 = invoke(stypy.reporting.localization.Localization(__file__, 760, 23), clone_2074, *[], **kwargs_2075)
        
        # Assigning a type to the variable 'result_union' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'result_union', clone_call_result_2076)
        
        
        # Call to range(...): (line 761)
        # Processing the call arguments (line 761)
        int_2078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 23), 'int')
        
        # Call to len(...): (line 761)
        # Processing the call arguments (line 761)
        # Getting the type of 'self' (line 761)
        self_2080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 30), 'self', False)
        # Obtaining the member 'types' of a type (line 761)
        types_2081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 30), self_2080, 'types')
        # Processing the call keyword arguments (line 761)
        kwargs_2082 = {}
        # Getting the type of 'len' (line 761)
        len_2079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 26), 'len', False)
        # Calling len(args, kwargs) (line 761)
        len_call_result_2083 = invoke(stypy.reporting.localization.Localization(__file__, 761, 26), len_2079, *[types_2081], **kwargs_2082)
        
        # Processing the call keyword arguments (line 761)
        kwargs_2084 = {}
        # Getting the type of 'range' (line 761)
        range_2077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 17), 'range', False)
        # Calling range(args, kwargs) (line 761)
        range_call_result_2085 = invoke(stypy.reporting.localization.Localization(__file__, 761, 17), range_2077, *[int_2078, len_call_result_2083], **kwargs_2084)
        
        # Assigning a type to the variable 'range_call_result_2085' (line 761)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 8), 'range_call_result_2085', range_call_result_2085)
        # Testing if the for loop is going to be iterated (line 761)
        # Testing the type of a for loop iterable (line 761)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 761, 8), range_call_result_2085)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 761, 8), range_call_result_2085):
            # Getting the type of the for loop variable (line 761)
            for_loop_var_2086 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 761, 8), range_call_result_2085)
            # Assigning a type to the variable 'i' (line 761)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 8), 'i', for_loop_var_2086)
            # SSA begins for a for statement (line 761)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to isinstance(...): (line 762)
            # Processing the call arguments (line 762)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 762)
            i_2088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 37), 'i', False)
            # Getting the type of 'self' (line 762)
            self_2089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 26), 'self', False)
            # Obtaining the member 'types' of a type (line 762)
            types_2090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 26), self_2089, 'types')
            # Obtaining the member '__getitem__' of a type (line 762)
            getitem___2091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 26), types_2090, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 762)
            subscript_call_result_2092 = invoke(stypy.reporting.localization.Localization(__file__, 762, 26), getitem___2091, i_2088)
            
            # Getting the type of 'Type' (line 762)
            Type_2093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 41), 'Type', False)
            # Processing the call keyword arguments (line 762)
            kwargs_2094 = {}
            # Getting the type of 'isinstance' (line 762)
            isinstance_2087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 762)
            isinstance_call_result_2095 = invoke(stypy.reporting.localization.Localization(__file__, 762, 15), isinstance_2087, *[subscript_call_result_2092, Type_2093], **kwargs_2094)
            
            # Testing if the type of an if condition is none (line 762)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 762, 12), isinstance_call_result_2095):
                
                # Assigning a Call to a Name (line 765):
                
                # Call to add(...): (line 765)
                # Processing the call arguments (line 765)
                # Getting the type of 'result_union' (line 765)
                result_union_2112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 52), 'result_union', False)
                
                # Call to deepcopy(...): (line 765)
                # Processing the call arguments (line 765)
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 765)
                i_2115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 91), 'i', False)
                # Getting the type of 'self' (line 765)
                self_2116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 80), 'self', False)
                # Obtaining the member 'types' of a type (line 765)
                types_2117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 80), self_2116, 'types')
                # Obtaining the member '__getitem__' of a type (line 765)
                getitem___2118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 80), types_2117, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 765)
                subscript_call_result_2119 = invoke(stypy.reporting.localization.Localization(__file__, 765, 80), getitem___2118, i_2115)
                
                # Processing the call keyword arguments (line 765)
                kwargs_2120 = {}
                # Getting the type of 'copy' (line 765)
                copy_2113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 66), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 765)
                deepcopy_2114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 66), copy_2113, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 765)
                deepcopy_call_result_2121 = invoke(stypy.reporting.localization.Localization(__file__, 765, 66), deepcopy_2114, *[subscript_call_result_2119], **kwargs_2120)
                
                # Processing the call keyword arguments (line 765)
                kwargs_2122 = {}
                # Getting the type of 'OrderedUnionType' (line 765)
                OrderedUnionType_2110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 31), 'OrderedUnionType', False)
                # Obtaining the member 'add' of a type (line 765)
                add_2111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 31), OrderedUnionType_2110, 'add')
                # Calling add(args, kwargs) (line 765)
                add_call_result_2123 = invoke(stypy.reporting.localization.Localization(__file__, 765, 31), add_2111, *[result_union_2112, deepcopy_call_result_2121], **kwargs_2122)
                
                # Assigning a type to the variable 'result_union' (line 765)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 16), 'result_union', add_call_result_2123)
            else:
                
                # Testing the type of an if condition (line 762)
                if_condition_2096 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 762, 12), isinstance_call_result_2095)
                # Assigning a type to the variable 'if_condition_2096' (line 762)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 12), 'if_condition_2096', if_condition_2096)
                # SSA begins for if statement (line 762)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 763):
                
                # Call to add(...): (line 763)
                # Processing the call arguments (line 763)
                # Getting the type of 'result_union' (line 763)
                result_union_2099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 52), 'result_union', False)
                
                # Call to clone(...): (line 763)
                # Processing the call keyword arguments (line 763)
                kwargs_2106 = {}
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 763)
                i_2100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 77), 'i', False)
                # Getting the type of 'self' (line 763)
                self_2101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 66), 'self', False)
                # Obtaining the member 'types' of a type (line 763)
                types_2102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 66), self_2101, 'types')
                # Obtaining the member '__getitem__' of a type (line 763)
                getitem___2103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 66), types_2102, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 763)
                subscript_call_result_2104 = invoke(stypy.reporting.localization.Localization(__file__, 763, 66), getitem___2103, i_2100)
                
                # Obtaining the member 'clone' of a type (line 763)
                clone_2105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 66), subscript_call_result_2104, 'clone')
                # Calling clone(args, kwargs) (line 763)
                clone_call_result_2107 = invoke(stypy.reporting.localization.Localization(__file__, 763, 66), clone_2105, *[], **kwargs_2106)
                
                # Processing the call keyword arguments (line 763)
                kwargs_2108 = {}
                # Getting the type of 'OrderedUnionType' (line 763)
                OrderedUnionType_2097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 31), 'OrderedUnionType', False)
                # Obtaining the member 'add' of a type (line 763)
                add_2098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 31), OrderedUnionType_2097, 'add')
                # Calling add(args, kwargs) (line 763)
                add_call_result_2109 = invoke(stypy.reporting.localization.Localization(__file__, 763, 31), add_2098, *[result_union_2099, clone_call_result_2107], **kwargs_2108)
                
                # Assigning a type to the variable 'result_union' (line 763)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 16), 'result_union', add_call_result_2109)
                # SSA branch for the else part of an if statement (line 762)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Name (line 765):
                
                # Call to add(...): (line 765)
                # Processing the call arguments (line 765)
                # Getting the type of 'result_union' (line 765)
                result_union_2112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 52), 'result_union', False)
                
                # Call to deepcopy(...): (line 765)
                # Processing the call arguments (line 765)
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 765)
                i_2115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 91), 'i', False)
                # Getting the type of 'self' (line 765)
                self_2116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 80), 'self', False)
                # Obtaining the member 'types' of a type (line 765)
                types_2117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 80), self_2116, 'types')
                # Obtaining the member '__getitem__' of a type (line 765)
                getitem___2118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 80), types_2117, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 765)
                subscript_call_result_2119 = invoke(stypy.reporting.localization.Localization(__file__, 765, 80), getitem___2118, i_2115)
                
                # Processing the call keyword arguments (line 765)
                kwargs_2120 = {}
                # Getting the type of 'copy' (line 765)
                copy_2113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 66), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 765)
                deepcopy_2114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 66), copy_2113, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 765)
                deepcopy_call_result_2121 = invoke(stypy.reporting.localization.Localization(__file__, 765, 66), deepcopy_2114, *[subscript_call_result_2119], **kwargs_2120)
                
                # Processing the call keyword arguments (line 765)
                kwargs_2122 = {}
                # Getting the type of 'OrderedUnionType' (line 765)
                OrderedUnionType_2110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 31), 'OrderedUnionType', False)
                # Obtaining the member 'add' of a type (line 765)
                add_2111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 31), OrderedUnionType_2110, 'add')
                # Calling add(args, kwargs) (line 765)
                add_call_result_2123 = invoke(stypy.reporting.localization.Localization(__file__, 765, 31), add_2111, *[result_union_2112, deepcopy_call_result_2121], **kwargs_2122)
                
                # Assigning a type to the variable 'result_union' (line 765)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 16), 'result_union', add_call_result_2123)
                # SSA join for if statement (line 762)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'result_union' (line 767)
        result_union_2124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 15), 'result_union')
        # Assigning a type to the variable 'stypy_return_type' (line 767)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 8), 'stypy_return_type', result_union_2124)
        
        # ################# End of 'clone(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clone' in the type store
        # Getting the type of 'stypy_return_type' (line 756)
        stypy_return_type_2125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2125)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clone'
        return stypy_return_type_2125


# Assigning a type to the variable 'OrderedUnionType' (line 705)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 0), 'OrderedUnionType', OrderedUnionType)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
