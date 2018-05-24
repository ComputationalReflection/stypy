
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import inspect
2: import types
3: 
4: import type_inference_proxy_management_copy
5: from ....errors_copy.type_error_copy import TypeError
6: from ....errors_copy.type_warning_copy import TypeWarning
7: from ....python_lib_copy.member_call_copy import call_handlers_copy
8: from ....python_lib_copy.python_types_copy.type_copy import Type
9: from ....python_lib_copy.python_types_copy.type_introspection_copy import type_equivalence_copy
10: from ....python_lib_copy.python_types_copy.type_inference_copy import union_type_copy
11: from ....python_lib_copy.python_types_copy.type_inference_copy import undefined_type_copy
12: from ....python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import simple_python_types
13: from ....type_store_copy.type_annotation_record_copy import TypeAnnotationRecord
14: from .....stypy_copy import type_store_copy
15: from .....stypy_copy import stypy_parameters_copy
16: 
17: 
18: class TypeInferenceProxy(Type):
19:     '''
20:     The type inference proxy is the main class of stypy. Its main purpose is to represent any kind of Python type,
21:      holding a reference to it. It is also responsible of a lot of possible operations that can be done with the
22:      contained type, including:
23: 
24:      - Returning/setting the type of any member of the Python entity it holds.
25:      - Invoke any of its associated callable members, returning the invokation result
26:      - Support structural reflection operations, it the enclosed object is able to support them
27:      - Obtain relationships with other entities (modules that contain a represented function, class that contains
28:      a represented method,...)
29:      - Manipulate stored types (if the represented entity is able to store other types)
30:      - Clone itself to support the SSA algorithm
31:      - Respond to builtin operations such as dir and __dict__ calls
32:      - Hold values for the represented object type
33: 
34:      All Python entities (functions, variables, methods, classes, modules,...) might be enclosed in a type inference
35:      proxy. For those method that are not applicable to the enclosed Python entity, the class will return a TypeError.
36:     '''
37: 
38:     # Memoization of TypeInferenceProxy instances in those cases in which these instances can be reused. Proxies of
39:     # any Python entity that is not an instance and do not support structural reflection may be reused.
40:     type_proxy_cache = dict()
41: 
42:     # Type annotation is an special feature not related to the functionality of this class, but related to this class
43:     # and the FunctionContext class. Type annotation record an annotation on a special table that holds variable names,
44:     # types of these variables, and source lines. These data indicate that the variable name has been changed its type
45:     # to the annotated type in the passed line. This information is used to generate a type annotated source program,
46:     # which is an optional feature of stypy. This flags control if types are annotated or not when changing the
47:     # type of a member or other related type-changing operations.
48:     annotate_types = True
49: 
50:     # ################################ PRIVATE METHODS #################################
51: 
52:     @staticmethod
53:     def __get_parent_proxy(parent):
54:         '''
55:         Gets the python entity that can be considered the "parent" of the passed entity
56:         :param parent: Any Python entity
57:         :return: The parent of this entity, if any
58:         '''
59:         if hasattr(parent, '__module__'):
60:             return TypeInferenceProxy.instance(inspect.getmodule(parent))
61:         if hasattr(parent, '__class__'):
62:             return TypeInferenceProxy.instance(parent.__class__)
63: 
64:         return None
65: 
66:     def __assign_parent_proxy(self, parent):
67:         '''
68:         Changes the parent object of the represented object to the one specified. This is used to trace the nesting
69:         of proxies that hold types that are placed inside other proxies represented entities. This property is NOT
70:         related with dynamic inheritance.
71:         :param parent: The new parent object or None. If the passed parent is None, the class tries to autocalculate it.
72:         If there is no possible parent, it is assigned to None
73:         '''
74:         if parent is not None:
75:             self.parent_proxy = parent
76:         else:
77:             if not inspect.ismodule(self.python_entity):
78:                 self.parent_proxy = TypeInferenceProxy.__get_parent_proxy(self.python_entity)
79:             else:
80:                 self.parent_proxy = None  # Root Python entity
81: 
82:     def __change_class_base_types_checks(self, localization, new_type):
83:         '''
84:         Performs all the possible checks to see if a base type change is possible for the currently hold python entity.
85:         This includes:
86:         - Making sure that the currently hold object represent a class. No base type change is possible if the hold
87:         entity is not a class. For checking the availability of an instance type change, see the
88:         "__change_instance_type_checks" private method.
89:         - Making sure that the hold class is not a new-style class. New style Python classes cannot change its base
90:         type directly, as its __mro__ (Method Resolution Order) property is readonly. For this purpose a metaclass
91:         has to be created, like in this example:
92: 
93:         class change_mro_meta(type):
94:             def __new__(cls, cls_name, cls_bases, cls_dict):
95:                     out_cls = super(change_mro_meta, cls).__new__(cls, cls_name, cls_bases, cls_dict)
96:                     out_cls.change_mro = False
97:                     out_cls.hack_mro   = classmethod(cls.hack_mro)
98:                     out_cls.fix_mro    = classmethod(cls.fix_mro)
99:                     out_cls.recalc_mro = classmethod(cls.recalc_mro)
100:                     return out_cls
101: 
102:             @staticmethod
103:             def hack_mro(cls):
104:                 cls.change_mro = True
105:                 cls.recalc_mro()
106: 
107:             @staticmethod
108:             def fix_mro(cls):
109:                 cls.change_mro = False
110:                 cls.recalc_mro()
111: 
112:             @staticmethod
113:             def recalc_mro(cls):
114:                 # Changing a class' base causes __mro__ recalculation
115:                 cls.__bases__  = cls.__bases__ + tuple()
116: 
117:             def mro(cls):
118:                 default_mro = super(change_mro_meta, cls).mro()
119:                 if hasattr(cls, "change_mro") and cls.change_mro:
120:                     return default_mro[1:2] + default_mro
121:                 else:
122:                     return default_mro
123: 
124:         - Making sure that new base class do not belong to a different class style as the current one: base type of
125:         old-style classes can only be changed to another old-style class.
126: 
127:         :param localization: Call localization data
128:         :param new_type: New base type to change to
129:         :return: A Type error specifying the problem encountered with the base type change or None if no error is found
130:         '''
131:         if not type_inference_proxy_management_copy.is_class(self.python_entity):
132:             return TypeError(localization, "Cannot change the base type of a non-class Python entity")
133: 
134:         if type_inference_proxy_management_copy.is_new_style_class(self.python_entity):
135:             return TypeError(localization,
136:                              "Cannot change the class hierarchy of a new-style class: "
137:                              "The __mro__ (Method Resolution Order) property is readonly")
138: 
139:         if self.instance is not None:
140:             return TypeError(localization, "Cannot change the class hierarchy of a class using an instance")
141: 
142:         old_style_existing = type_inference_proxy_management_copy.is_old_style_class(self.python_entity)
143:         if not isinstance(new_type, TypeError):
144:             old_style_new = type_inference_proxy_management_copy.is_old_style_class(new_type.python_entity)
145:         else:
146:             return TypeError(localization, "Cannot change the class hierarchy to a type error")
147: 
148:         # Did the existing and new class belong to the same class definition type?
149:         if not old_style_existing == old_style_new:
150:             return TypeError(localization, "Cannot change the class hierarchy from an old-style Python parent class "
151:                                            "to a new-style Python parent class")
152: 
153:         return None
154: 
155:     def __change_instance_type_checks(self, localization, new_type):
156:         '''
157:         Performs all the checks that ensure that changing the type of an instance is possible. This includes:
158:         - Making sure that we are changing the type of an user-defined class instance. Type change for Python
159:         library classes instances is not possible.
160:         - Making sure that the old instance type and the new instance type are of the same class style, as mixing
161:         old-style and new-style types is not possible in Python.
162: 
163:         :param localization: Call localization data
164:         :param new_type: New instance type.
165:         :return:
166:         '''
167:         # Is a class?
168:         if not type_inference_proxy_management_copy.is_class(self.python_entity):
169:             return TypeError(localization, "Cannot change the type of a Python entity that it is not a class")
170: 
171:         # Is the class user-defined?
172:         if not type_inference_proxy_management_copy.is_user_defined_class(self.instance.__class__):
173:             return TypeError(localization, "Cannot change the type of an instance of a non user-defined class")
174: 
175:         # Is this object representing a class instance? (so we can change its type)
176:         if self.instance is None:
177:             return TypeError(localization, "Cannot change the type of a class object; Type change is only possible"
178:                                            "with class instances")
179: 
180:         old_style_existing = type_inference_proxy_management_copy.is_old_style_class(self.instance.__class__)
181:         old_style_new = type_inference_proxy_management_copy.is_old_style_class(new_type.python_entity)
182: 
183:         # Did the existing and new class belong to the same class definition type?
184:         if not old_style_existing == old_style_new:
185:             return TypeError(localization, "Cannot change the type of an instances from an old-style Python class to a "
186:                                            "new-style Python class or viceversa")
187: 
188:         return None
189: 
190:         # ################################ PYTHON METHODS #################################
191: 
192:     def __init__(self, python_entity, name=None, parent=None, instance=None, value=undefined_type_copy.UndefinedType):
193:         '''
194:         Creates a new Type inference proxy for the passed python entity (function, module, class, ...). This constructor
195:         should NOT be called directly. Use the instance(...) method instead to take advantage of the implemented
196:         type memoization of this class.
197:         :param python_entity: Represented python entity.
198:         :param name: Name of the represented Python entity. If nothing is provided, the Python entity __name__ property
199:         value is used instead. Instances have an special name indicating that this entity holds an instance of a class.
200:         :param parent: Parent proxy object. If nothing is provided, the parent proxy is autocalculated, if possible.
201:         :param instance: Instance of the represented class. If this proxy holds a class, it is possible that instead
202:         of representing the class is representing a particular class instance. This is important to properly model
203:         instance intercession, as altering the structure of single class instances is possible.
204:         '''
205:         if name is None:
206:             if hasattr(python_entity, "__name__"):
207:                 self.name = python_entity.__name__
208:             else:
209:                 if hasattr(python_entity, "__class__"):
210:                     self.name = python_entity.__class__.__name__
211:                 else:
212:                     if hasattr(python_entity, "__module__"):
213:                         self.name = python_entity.__module__
214: 
215:             if instance is not None:
216:                 self.name = "<" + self.name + " instance>"
217:         else:
218:             self.name = name
219: 
220:         self.python_entity = python_entity
221:         self.__assign_parent_proxy(parent)
222:         self.instance = instance
223:         if instance is not None:
224:             self.set_type_instance(True)
225: 
226:         # Attribute values that have not been name (structure of the object is not known)
227:         self.additional_members = list()
228: 
229:         # If this is a type, store the original variable whose type is
230:         self.type_of = None
231: 
232:         # Store if the structure of the object is fully known or it has been manipulated without knowing precise
233:         # attribute values
234:         self.known_structure = True
235: 
236:         if value is not undefined_type_copy.UndefinedType:
237:             self.value = value
238:             self.set_type_instance(True)
239: 
240:             # self.annotation_record = TypeAnnotationRecord.get_instance_for_file(__file__)
241: 
242:             # Instances of "immutable" entities are stored in a cache to save memory. Those include:
243:             # Python entities that do not support structural reflection, therefore its structure
244:             # will always be the same. This means that the entity has a dictproxy object as its __dict__ property
245:             # instead of a plain dict. If the proxy has a non-None instance, it also means that individual instances of
246:             # this class object are also mutable, and therefore individual instance types are held to allow this. In
247:             # this case the type proxy cache is also NOT used.
248:             # TODO: Remove? Temporally disabled because instance identity problems
249:             # if python_entity in inmutable_python_types:
250:             #     try:
251:             #         TypeInferenceProxy.type_proxy_cache[python_entity] = self
252:             #     except:
253:             #         pass
254: 
255:     # TODO: Remove?
256:     # def __get_member_type_repr(self):
257:     #     repr_str = ""
258:     #     members = self.dir()
259:     #     for member in members:
260:     #         entity = self.get_type_of_member(None, member).get_python_entity()
261:     #         if hasattr(entity, '__name__'):
262:     #             type_str = entity.__name__
263:     #         else:
264:     #             type_str = str(entity)
265:     #
266:     #             repr_str += member + ": " + type_str + "; "
267:     #
268:     #     if len(repr_str) > 2:
269:     #         repr_str = repr_str[:-2]
270:     #
271:     #     return repr_str
272: 
273:     def __repr__(self):
274:         '''
275:         String representation of this proxy and its contents. Python builtin types have a very concise representation.
276:         The method have been stripped down of much of its information gathering code to favor a more concise and clear
277:         representation of entities.
278:         :return: str
279:         '''
280: 
281:         if isinstance(self.python_entity, types.InstanceType) or isinstance(self.python_entity, types.ClassType):
282:             return self.name
283: 
284:         # Simple Python type only prints its name
285:         if self.python_entity in simple_python_types:
286:             return self.get_python_type().__name__
287: 
288:         parent_str = ""
289:         # TODO: Remove?
290:         # if self.parent_proxy is None:
291:         #     parent_str = ""
292:         # else:
293:         #     if not (self.parent_proxy.python_entity is None):
294:         #         if self.parent_proxy.python_entity.__name__ == no_recursion.__name__:
295:         #             parent_str = ""
296:         #         else:
297:         #             parent_str = " from <{0}>".format(self.parent_proxy)
298:         #     else:
299:         #         parent_str = " from <{0}>".format(self.parent_proxy)
300: 
301:         str_mark = ""
302:         # TODO: Remove?
303:         # if self.supports_structural_reflection():
304:         #     str_mark = "*"
305:         # else:
306:         #     str_mark = ""
307: 
308:         # Representation of instances
309:         # if not self.instance is None:
310:         #     instance_type = "Instance of the " + self.instance.__class__.__name__ + " class"
311:         #     return "{0}{1} {2}".format(instance_type, str_mark, self.__get_member_type_repr()) + parent_str
312:         #     #return "{0}{1} {2}".format(instance_type, str_mark, self.dir()) + parent_str
313:         # else:
314:         #     instance_type = ""
315: 
316:         # Instances of classes
317:         if self.instance is not None:
318:             instance_type = self.instance.__class__.__name__ + " instance"
319:             return instance_type
320:         else:
321:             instance_type = ""
322: 
323:         # Representation of lists, tuples, dicts, (types that contain other types)...
324:         if hasattr(self, self.contained_elements_property_name):
325:             contained_str = "[" + str(getattr(self, self.contained_elements_property_name)) + "]"
326:             return "{0}".format(self.get_python_type().__name__) \
327:                    + contained_str
328:         else:
329:             if self.can_store_elements():
330:                 contained_str = "[]"
331:                 return "{0}".format(self.get_python_type().__name__) \
332:                        + contained_str
333:             else:
334:                 if self.can_store_keypairs():
335:                     contained_str = "{}"
336:                     return "{0}".format(self.get_python_type().__name__) \
337:                            + contained_str
338: 
339:         own_name = ""
340:         # TODO: Remove?
341:         # if inspect.isfunction(self.python_entity):
342:         #     own_name = ""
343:         # else:
344:         #     own_name = self.name
345: 
346:         return "{0}{3}{1}{2}".format(self.get_python_type().__name__, own_name, instance_type, str_mark) + parent_str
347: 
348:         # TODO: Remove?
349:         # else:
350:         #     return "{0}{4} {1}{3} from <{2}>".format(self.get_python_type().__name__, self.name, self.parent_proxy,
351:         #                                                instance_type, str_mark)
352: 
353:     @staticmethod
354:     def __equal_property_value(property_name, obj1, obj2):
355:         '''
356:         Determines if a property of two objects have the same value.
357:         :param property_name: Name of the property to test
358:         :param obj1: First object
359:         :param obj2: Second object
360:         :return: bool (True if same value or both object do not have the property
361:         '''
362:         if hasattr(obj1, property_name) and hasattr(obj2, property_name):
363:             if not getattr(obj1, property_name) == getattr(obj2, property_name):
364:                 return False
365: 
366:         return True
367: 
368:     @staticmethod
369:     def contains_an_undefined_type(value):
370:         '''
371:         Determines if the passed argument is an UndefinedType or contains an UndefinedType
372:         :param value: Any Type
373:         :return: Tuple (bool, int) (contains an undefined type, the value holds n more types)
374:         '''
375:         if isinstance(value, union_type_copy.UnionType):
376:             for type_ in value.types:
377:                 if isinstance(type_, undefined_type_copy.UndefinedType):
378:                     return True, len(value.types) - 1
379:         else:
380:             if isinstance(value, undefined_type_copy.UndefinedType):
381:                 return True, 0
382: 
383:         return False, 0
384: 
385:     def __eq__(self, other):
386:         '''
387:         Type proxy equality. The equality algorithm is represented as follows:
388:         - Both objects have to be type inference proxies.
389:         - Both objects have to hold the same type of python entity
390:         - Both objects held entity name has to be the same (same class, same function, same module, ...), if the
391:         proxy is not holding an instance
392:         - If the hold entity do not support structural reflection, comparison will be done using the is operator
393:         (reference comparison)
394:         - If not, comparison by structure is performed (same amount of members, same types for these members)
395: 
396:         :param other: The other object to compare with
397:         :return: bool
398:         '''
399:         if self is other:
400:             return True  # Same reference
401: 
402:         if not type(other) is TypeInferenceProxy:
403:             return False
404: 
405:         # Both do not represent the same Python entity
406:         if not type(self.python_entity) == type(other.python_entity):
407:             return False
408: 
409:         # Different values for the "instance" property for both proxies (None vs filled)
410:         if (self.instance is None) ^ (other.instance is None):
411:             return False
412: 
413:         # One object is a type name and the other is a type instance (int is not the same as '3')
414:         self_instantiated = self.is_type_instance()
415:         other_instantiated = other.is_type_instance()
416:         if self_instantiated != other_instantiated:
417:             return False
418: 
419:         self_entity = self.python_entity
420:         other_entity = other.python_entity
421: 
422:         # Compare several properties key to determine object equality
423:         for prop_name in Type.special_properties_for_equality:
424:             if not self.__equal_property_value(prop_name, self_entity, other_entity):
425:                 return False
426: 
427:         # Contains the same elements?
428:         if not self.__equal_property_value(TypeInferenceProxy.contained_elements_property_name, self_entity,
429:                                            other_entity):
430:             return False
431: 
432:         # Class or instance?
433:         if self.instance is None:
434:             # Class
435: 
436:             # Both support structural reflection: structure comparison
437:             if self.supports_structural_reflection() and other.supports_structural_reflection():
438:                 # Compare class structure
439:                 return type_equivalence_copy.structural_equivalence(self_entity, other_entity, True)
440:             else:
441:                 # No structural reflection: reference comparison
442:                 return type(self_entity) is type(other_entity)
443:         else:
444:             # Instance: Compare the class first and the instance later.
445: 
446:             # Both support structural reflection: structure comparison
447:             if self.supports_structural_reflection() and other.supports_structural_reflection():
448:                 # Compare class structure
449:                 equivalent = type_equivalence_copy.structural_equivalence(self_entity, other_entity, True)
450: 
451:                 if not equivalent:
452:                     return False
453: 
454:                 # Compare instance structure
455:                 self_entity = self.instance
456:                 other_entity = other.instance
457:                 return type_equivalence_copy.structural_equivalence(self_entity, other_entity, False)
458:             else:
459:                 # No structural reflection: reference comparison
460:                 equivalent = type(self.python_entity) is type(other.python_entity)
461:                 if not equivalent:
462:                     return False
463: 
464:                 # No structural reflection: reference comparison
465:                 self_entity = self.instance
466:                 other_entity = other.instance
467:                 return self_entity is other_entity
468: 
469:     # ###################### INSTANCE CREATION ###############
470: 
471:     @staticmethod
472:     def instance(python_entity, name=None, parent=None, instance=None, value=undefined_type_copy.UndefinedType):
473:         '''
474:         Creates a new Type inference proxy for the passed python entity (function, module, class, ...). This is the
475:         preferred way to create proxy instances, as this method implement a memoization optimization.
476: 
477:         :param python_entity: Represented python entity.
478:         :param name: Name of the represented Python entity. If nothing is provided, the Python entity __name__ property
479:         value is used instead. Instances have an special name indicating that this entity holds an instance of a class.
480:         :param parent: Parent proxy object. If nothing is provided, the parent proxy is autocalculated, if possible.
481:         :param instance: Instance of the represented class. If this proxy holds a class, it is possible that instead
482:         of representing the class is representing a particular class instance. This is important to properly model
483:         instance intercession, as altering the structure of single instances is possible.
484:         '''
485: 
486:         # TODO: Remove? Disabled because identity problems
487:         # try:
488:         #     if python_entity in TypeInferenceProxy.type_proxy_cache:
489:         #         return TypeInferenceProxy.type_proxy_cache[python_entity]
490:         # except:
491:         #     pass
492: 
493:         if isinstance(python_entity, Type):
494:             return python_entity
495: 
496:         return TypeInferenceProxy(python_entity, name, parent, instance, value)
497: 
498:     # ################### STORED PYTHON ENTITY (CLASS, METHOD...) AND PYTHON TYPE/INSTANCE OF THE ENTITY ###############
499: 
500:     def get_python_entity(self):
501:         '''
502:         Returns the Python entity (function, method, class, object, module...) represented by this Type.
503:         :return: A Python entity
504:         '''
505:         return self.python_entity
506: 
507:     def get_python_type(self):
508:         '''
509:         Get the python type of the hold entity. This is equivalent to call the type(hold_python_entity). If a user-
510:         defined class instance is hold, a types.InstanceType is returned (as Python does)
511:         :return: A python type
512:         '''
513:         if not inspect.isclass(self.python_entity):
514:             return type(self.python_entity)
515: 
516:         if type_inference_proxy_management_copy.is_user_defined_class(self.python_entity) and self.instance is not None:
517:             return types.InstanceType
518: 
519:         return self.python_entity
520: 
521:     def get_instance(self):
522:         '''
523:         Gets the stored class instance (if any). Class instances are only stored for instance intercession purposes, as
524:         we need an entity to store these kind of changes.
525:         :return:
526:         '''
527:         return self.instance
528: 
529:     def has_value(self):
530:         '''
531:         Determines if this proxy holds a value to the type it represents
532:         :return:
533:         '''
534:         return hasattr(self, "value")
535: 
536:     def get_value(self):
537:         '''
538:         Gets the value held by this proxy
539:         :return: Value of the proxt
540:         '''
541:         return self.value
542: 
543:     def set_value(self, value):
544:         '''
545:         Sets the value held by this proxy. No type check is performed
546:         :return: Value of the proxt
547:         '''
548:         self.value = value
549: 
550:     # ############################## MEMBER TYPE GET / SET ###############################
551: 
552:     def __get_module_file(self):
553:         while True:
554:             current = self.parent_proxy.python_entity
555:             if current is None:
556:                 return ""
557:             if isinstance(current, types.ModuleType):
558:                 return current.__file__
559: 
560:     def get_type_of_member(self, localization, member_name):
561:         '''
562:         Returns the type of the passed member name or a TypeError if the stored entity has no member with the mentioned
563:         name.
564:         :param localization: Call localization data
565:         :param member_name: Member name
566:         :return: A type proxy with the member type or a TypeError
567:         '''
568:         try:
569:             if self.instance is None:
570:                 return TypeInferenceProxy.instance(getattr(self.python_entity, member_name),
571:                                                    self.name + "." + member_name,
572:                                                    parent=self)
573:             else:
574:                 # Copy-on-write attribute values for instances
575:                 if hasattr(self.instance, member_name):
576:                     return TypeInferenceProxy.instance(getattr(self.instance, member_name),
577:                                                        self.name + "." + member_name,
578:                                                        parent=self)
579:                 else:
580:                     # module_path = self.parent_proxy.python_entity.__file__.replace("__type_inference", "")
581:                     # module_path = module_path.replace("/type_inference", "")
582:                     # module_path = module_path.replace('\\', '/')
583: 
584:                     module_path = stypy_parameters_copy.get_original_program_from_type_inference_file(
585:                         self.__get_module_file())
586:                     ts = type_store_copy.typestore.TypeStore.get_type_store_of_module(module_path)
587:                     typ = ts.get_type_of(localization, self.python_entity.__name__)
588:                     return typ.get_type_of_member(localization, member_name)
589:                     # return TypeInferenceProxy.instance(getattr(self.python_entity, member_name),
590:                     #                                    self.name + "." + member_name,
591:                     #                                    parent=self)
592:         except AttributeError:
593:             return TypeError(localization,
594:                              "{0} has no member '{1}'".format(self.get_python_type().__name__, member_name))
595: 
596:     def set_type_of_member(self, localization, member_name, member_type):
597:         '''
598:         Set the type of a member of the represented object. If the member do not exist, it is created with the passed
599:         name and types (except iif the represented object do not support reflection, in that case a TypeError is
600:         returned)
601:         :param localization: Caller information
602:         :param member_name: Name of the member
603:         :param member_type: Type of the member
604:         :return: None or a TypeError
605:         '''
606:         try:
607:             contains_undefined, more_types_in_value = TypeInferenceProxy.contains_an_undefined_type(member_type)
608:             if contains_undefined:
609:                 if more_types_in_value == 0:
610:                     TypeError(localization, "Assigning to {0}.{1} the value of a previously undefined variable".
611:                               format(self.parent_proxy.name, member_name))
612:                 else:
613:                     TypeWarning.instance(localization,
614:                                          "Potentialy assigning to {0}.{1} the value of a previously undefined variable".
615:                                          format(self.parent_proxy.name, member_name))
616: 
617:             if self.instance is not None:
618:                 # value = self.__parse_member(self.instance, member_name, member_value)
619:                 setattr(self.instance, member_name, member_type)
620:                 if self.annotate_types:
621:                     self.__annotate_type(localization.line, localization.column, member_name,
622:                                          member_type)
623:                 return None
624: 
625:             if type_inference_proxy_management_copy.supports_structural_reflection(self.python_entity) or hasattr(
626:                     self.python_entity, member_name):
627:                 # value = self.__parse_member(self.python_entity, member_name, member_value)
628:                 setattr(self.python_entity, member_name, member_type)
629:                 if self.annotate_types:
630:                     self.__annotate_type(localization.line, localization.column, member_name,
631:                                          member_type)
632:                 return None
633:         except Exception as exc:
634:             return TypeError(localization,
635:                              "Cannot modify the structure of '{0}': {1}".format(self.__repr__(), str(exc)))
636: 
637:         return TypeError(localization,
638:                          "Cannot modify the structure of a python library type or instance")
639: 
640:     # ############################## MEMBER INVOKATION ###############################
641: 
642:     def invoke(self, localization, *args, **kwargs):
643:         '''
644:         Invoke a callable member of the hold python entity with the specified arguments and keyword arguments.
645:         NOTE: Calling a class constructor returns a type proxy of an instance of this class. But an instance object
646:         is only stored if the instances of this class support structural reflection.
647: 
648:         :param localization: Call localization data
649:         :param args: Arguments of the call
650:         :param kwargs: Keyword arguments of the call
651:         :return:
652:         '''
653: 
654:         # Is it callable?
655:         if not callable(self.python_entity):
656:             return TypeError(localization, "Cannot invoke on a non callable type")
657:         else:
658:             # If it is callable, call it using a call handler
659:             result_ = call_handlers_copy.perform_call(self, self.python_entity, localization, *args, **kwargs)
660: 
661:             if TypeAnnotationRecord.is_type_changing_method(self.name) and self.annotate_types:
662:                 self.__annotate_type(localization.line, localization.column, self.parent_proxy.name,
663:                                      self.parent_proxy.get_python_type())
664: 
665:             # If the result is an error, return it
666:             if isinstance(result_, TypeError):
667:                 return result_
668: 
669:             if isinstance(result_, Type):
670:                 result_.set_type_instance(True)
671:                 return result_
672: 
673:             # If calling a class then we are building an instance of this class. The instance is returned as a
674:             # consequence of the call, we built the rest of the instance if applicable
675:             if inspect.isclass(self.python_entity):
676:                 # Instances are stored only to handle object-based structural reflection
677:                 if type_inference_proxy_management_copy.supports_structural_reflection(result_):
678:                     instance = result_
679: 
680:                     # Calculate the class of the obtained instance
681:                     result_ = type(result_)
682:                 else:
683:                     instance = None
684:             else:
685:                 instance = None
686: 
687:             # If the returned object is not a Python type proxy but a Python type, build it.
688:             if not isinstance(result_, Type):
689:                 ret = TypeInferenceProxy.instance(result_, instance=instance)
690:                 ret.set_type_instance(True)
691: 
692:                 return ret
693:             else:
694:                 result_.set_type_instance(True)
695:                 return result_
696: 
697:     # ############################## STORED ELEMENTS TYPES (IF ANY) ###############################
698: 
699:     def __check_undefined_stored_value(self, localization, value):
700:         '''
701:         For represented containers, this method checks if we are trying to store Undefined variables inside them
702:         :param localization: Caller information
703:         :param value: Value we are trying to store
704:         :return:
705:         '''
706:         contains_undefined, more_types_in_value = TypeInferenceProxy.contains_an_undefined_type(value)
707:         if contains_undefined:
708:             if more_types_in_value == 0:
709:                 TypeError(localization, "Storing in '{0}' the value of a previously undefined variable".
710:                           format(self.name))
711:             else:
712:                 TypeWarning.instance(localization,
713:                                      "Potentially storing in '{0}' the value of a previously undefined variable".
714:                                      format(self.name))
715:         return contains_undefined, more_types_in_value
716: 
717:     def can_store_elements(self):
718:         '''
719:         Determines if this proxy represents a Python type able to store elements (lists, tuples, ...)
720:         :return: bool
721:         '''
722:         is_iterator = ("dictionary-" in self.name and "iterator" in self.name) or ("iterator" in self.name and
723:                                                                                    "dict" not in self.name)
724: 
725:         data_structures = [list, set, tuple, types.GeneratorType, bytearray, slice, range, xrange, enumerate, reversed,
726:                            frozenset]
727:         return (self.python_entity in data_structures) or is_iterator
728: 
729:     def can_store_keypairs(self):
730:         '''
731:         Determines if this proxy represents a Python type able to store keypairs (dict, dict iterators)
732:         :return: bool
733:         '''
734:         is_iterator = "iterator" in self.name and "dict" in self.name
735: 
736:         return self.python_entity is dict or is_iterator
737: 
738:     def is_empty(self):
739:         '''
740:         Determines if a proxy able to store elements can be considered empty (no elements were inserted through its
741:         lifespan
742:         :return: None or TypeError
743:         '''
744:         if not self.can_store_elements() and not self.can_store_keypairs():
745:             return TypeError(None,
746:                              "STYPY CRITICAL ERROR: Attempt to determine if a container is empty over a python type ({0}) "
747:                              "that is not able to do it")
748:         return hasattr(self, self.contained_elements_property_name)
749: 
750:     def get_elements_type(self):
751:         '''
752:         Obtains the elements stored by this type, returning an error if this is called over a proxy that represent
753:         a non element holding Python type
754:         :return: None or TypeError
755:         '''
756:         if not self.can_store_elements() and not self.can_store_keypairs():
757:             return TypeError(None,
758:                              "STYPY CRITICAL ERROR: Attempt to return stored elements over a python type ({0}) "
759:                              "that is not able to do it")
760:         if hasattr(self, self.contained_elements_property_name):
761:             return getattr(self, self.contained_elements_property_name)
762:         else:
763:             return undefined_type_copy.UndefinedType()
764: 
765:     def set_elements_type(self, localization, elements_type, record_annotation=True):
766:         '''
767:         Sets the elements stored by this type, returning an error if this is called over a proxy that represent
768:         a non element holding Python type. It also checks if we are trying to store an undefined variable.
769:         :param localization: Caller information
770:         :param elements_type: New stored elements type
771:         :param record_annotation: Whether to annotate the type change or not
772:         :return: The stored elements type
773:         '''
774:         if not self.can_store_elements() and not self.can_store_keypairs():
775:             return TypeError(localization,
776:                              "STYPY CRITICAL ERROR: Attempt to set stored elements types over a python type ({0}) "
777:                              "that is not able to do it".format(self.get_python_type()))
778: 
779:         contains_undefined, more_types_in_value = TypeInferenceProxy.contains_an_undefined_type(elements_type)
780:         if contains_undefined:
781:             if more_types_in_value == 0:
782:                 TypeError(localization, "Storing in '{0}' the value of a previously undefined variable".
783:                           format(self.name))
784:             else:
785:                 TypeWarning.instance(localization,
786:                                      "Potentially storing in '{0}' the value of a previously undefined variable".
787:                                      format(self.name))
788: 
789:         setattr(self, self.contained_elements_property_name, elements_type)
790:         if record_annotation and self.annotate_types:
791:             self.__annotate_type(localization.line, localization.column, "<container elements type>",
792:                                  getattr(self, self.contained_elements_property_name))
793: 
794:     def add_type(self, localization, type_, record_annotation=True):
795:         '''
796:         Adds type_ to the elements stored by this type, returning an error if this is called over a proxy that represent
797:         a non element holding Python type. It also checks if we are trying to store an undefined variable.
798:         :param localization: Caller information
799:         :param type_: Type to store
800:         :param record_annotation: Whether to annotate the type change or not
801:         :return: None or TypeError
802:         '''
803:         if not self.can_store_elements():
804:             return TypeError(localization,
805:                              "STYPY CRITICAL ERROR: Attempt to store elements over a python type ({0}) that is not"
806:                              " able to do it".format(self.get_python_type()))
807: 
808:         existing_type = None
809:         if hasattr(self, self.contained_elements_property_name):
810:             existing_type = getattr(self, self.contained_elements_property_name)
811: 
812:         value_to_store = union_type_copy.UnionType.add(existing_type, type_)
813:         self.__check_undefined_stored_value(localization, value_to_store)
814: 
815:         setattr(self, self.contained_elements_property_name, value_to_store)
816: 
817:         if record_annotation and self.annotate_types:
818:             self.__annotate_type(localization.line, localization.column, "<container elements type>",
819:                                  getattr(self, self.contained_elements_property_name))
820: 
821:     def add_types_from_list(self, localization, type_list, record_annotation=True):
822:         '''
823:         Adds the types on type_list to the elements stored by this type, returning an error if this is called over a
824:         proxy that represent a non element holding Python type. It also checks if we are trying to store an undefined
825:         variable.
826:         :param localization: Caller information
827:         :param type_list: List of types to add
828:         :param record_annotation: Whether to annotate the type change or not
829:         :return: None or TypeError
830:         '''
831:         if not self.can_store_elements():
832:             return TypeError(localization,
833:                              "STYPY CRITICAL ERROR: Attempt to store elements over a python type ({0}) that is not"
834:                              " able to do it".format(self.get_python_type()))
835: 
836:         if hasattr(self, self.contained_elements_property_name):
837:             existing_type = getattr(self, self.contained_elements_property_name)
838:             type_list = [existing_type] + type_list
839: 
840:         setattr(self, self.contained_elements_property_name,
841:                 union_type_copy.UnionType.create_union_type_from_types(*type_list))
842:         if record_annotation and self.annotate_types:
843:             self.__annotate_type(localization.line, localization.column, "<container elements type>",
844:                                  getattr(self, self.contained_elements_property_name))
845: 
846:     def __exist_key(self, key):
847:         '''
848:         Helper method to see if the stored keypairs contains a key equal to the passed one.
849:         :param key:
850:         :return:
851:         '''
852:         existing_type_map = getattr(self, self.contained_elements_property_name)
853:         keys = existing_type_map.keys()
854:         for element in keys:
855:             if key == element:
856:                 return True
857:         return False
858: 
859:     def add_key_and_value_type(self, localization, type_tuple, record_annotation=True):
860:         '''
861:         Adds type_tuple to the elements stored by this type, returning an error if this is called over a proxy that
862:         represent a non keypair holding Python type. It also checks if we are trying to store an undefined variable.
863:         :param localization: Caller information
864:         :param type_tuple: Tuple of types to store (key type, value type)
865:         :param record_annotation: Whether to annotate the type change or not
866:         :return: None or TypeError
867:         '''
868:         key = type_tuple[0]
869:         value = type_tuple[1]
870: 
871:         if not self.can_store_keypairs():
872:             if not self.can_store_elements():
873:                 return TypeError(localization,
874:                                  "STYPY CRITICAL ERROR: Attempt to store keypairs over a python type ({0}) that is not"
875:                                  "a dict".format(self.get_python_type()))
876:             else:
877:                 if key.get_python_type() is not int:
878:                     return TypeError(localization,
879:                                      "STYPY CRITICAL ERROR: Attempt to store keypairs on a python collection")
880:                 else:
881:                     self.add_type(localization, value, record_annotation)
882:                     return
883: 
884:         if not hasattr(self, self.contained_elements_property_name):
885:             setattr(self, self.contained_elements_property_name, dict())
886: 
887:         existing_type_map = getattr(self, self.contained_elements_property_name)
888: 
889:         self.__check_undefined_stored_value(localization, value)
890: 
891:         # if key in existing_type_map.keys():
892:         if self.__exist_key(key):
893:             # We cannot directly use the dictionary because type inference proxies are not hashable, but are comparable
894:             stored_key_index = existing_type_map.keys().index(key)
895:             stored_key = existing_type_map.keys()[stored_key_index]
896:             existing_type = existing_type_map[stored_key]
897:             existing_type_map[stored_key] = union_type_copy.UnionType.add(existing_type, value)
898:         else:
899:             existing_type_map[key] = value
900: 
901:         if record_annotation and self.annotate_types:
902:             self.__annotate_type(localization.line, localization.column, "<dictionary elements type>",
903:                                  getattr(self, self.contained_elements_property_name))
904: 
905:     def get_values_from_key(self, localization, key):
906:         '''
907:         Get the poosible values associated to a key type on a keypair storing proxy
908: 
909:         :param localization: Caller information
910:         :param key: Key type
911:         :return: Value type list
912:         '''
913:         existing_type_map = getattr(self, self.contained_elements_property_name)
914: 
915:         try:
916:             # We cannot directly use the dictionary because type inference proxies are not hashable, but are comparable
917:             stored_key_index = existing_type_map.keys().index(key)
918:             stored_key = existing_type_map.keys()[stored_key_index]
919:             value = existing_type_map[stored_key]
920:             return value
921:         except:
922:             return TypeError(localization, "No value is associated to key type '{0}'".format(key))
923: 
924:     # ############################## STRUCTURAL REFLECTION ###############################
925: 
926:     def supports_structural_reflection(self):
927:         '''
928:         Determines whether the stored python entity supports intercession. This means that this proxy stores an
929:         instance (which are created precisely for this purpose) or the stored entity has a dict as the type of
930:         its __dict__ property (and not a dictproxy instance, that is read-only).
931: 
932:         :return: bool
933:         '''
934:         return self.instance is not None or type_inference_proxy_management_copy.supports_structural_reflection(
935:             self.python_entity)
936: 
937:     def delete_member(self, localization, member_name):
938:         '''
939:         Set the type of the member whose name is passed to the specified value. There are cases in which deepcopies of
940:         the stored python entities are not supported when cloning the type proxy (cloning is needed for SSA), but
941:         structural reflection is supported. Therefore, the additional_members attribute have to be created to still
942:         support structural reflection while maintaining the ability to create fully independent clones of the stored
943:         python entity.
944: 
945:         :param localization: Call localization data
946:         :param member_name: Member name
947:         :return:
948:         '''
949:         try:
950:             if self.instance is not None:
951:                 # value = self.__parse_member(self.instance, member_name, member_value)
952:                 delattr(self.instance, member_name)
953:                 return None
954: 
955:             if type_inference_proxy_management_copy.supports_structural_reflection(self.python_entity):
956:                 # value = self.__parse_member(self.python_entity, member_name, member_value)
957:                 delattr(self.python_entity, member_name)
958:                 return None
959:         except Exception as exc:
960:             return TypeError(localization,
961:                              "'{2}' member deletion is impossible: Cannot modify the structure of '{0}': {1}".
962:                              format(self.__repr__(), str(exc), member_name))
963: 
964:         return TypeError(localization,
965:                          "'{0}' member deletion is impossible: Cannot modify the structure of a python library "
966:                          "type or instance".format(member_name))
967: 
968:     def change_type(self, localization, new_type):
969:         '''
970:         Changes the type of the stored entity, provided it is an instance (so it supports structural reflection).
971:         Type change is only available in Python for instances of user-defined classes.
972: 
973:         You can only assign to the __class__ attribute of an instance of a user-defined class
974:         (i.e. defined using the class keyword), and the new value must also be a user-defined class.
975:         Whether the classes are new-style or old-style does not matter. (You can't mix them, though.
976:         You can't turn an old-style class instance into a new-style class instance.)
977: 
978:         :param localization: Call localization data
979:         :param new_type: New type of the instance.
980:         :return: A TypeError or None
981:         '''
982:         result = self.__change_instance_type_checks(localization, new_type)
983: 
984:         if isinstance(result, TypeError):
985:             return result
986: 
987:         # If all these tests are passed, change the class:
988:         if type_inference_proxy_management_copy.is_user_defined_class(new_type.python_entity):
989:             self.python_entity = types.InstanceType
990:         else:
991:             self.python_entity = new_type.python_entity
992: 
993:         setattr(self.instance, '__class__', new_type.python_entity)
994:         return None
995: 
996:     def change_base_types(self, localization, new_types):
997:         '''
998:         Changes, if possible, the base types of the hold Python class. For determining if the change is possible, a
999:         series of checks (defined before) are made.
1000: 
1001:         For new-style classes, changing of the mro is not possible, you need to define a metaclass that does the trick
1002: 
1003:         Old-style classes admits changing its __bases__ attribute (its a tuple), so we can add or substitute
1004: 
1005:         :param localization: Call localization data
1006:         :param new_types: New base types (in the form of a tuple)
1007:         :return: A TypeError or None
1008:         '''
1009:         if not type(new_types) is tuple:
1010:             return TypeError(localization, "New subtypes have to be specified using a tuple")
1011: 
1012:         for base_type in new_types:
1013:             check = self.__change_class_base_types_checks(localization, base_type)
1014:             if isinstance(check, TypeError):
1015:                 return check
1016: 
1017:         base_classes = map(lambda tproxy: tproxy.python_entity, new_types)
1018: 
1019:         self.python_entity.__bases__ = tuple(base_classes)
1020:         return None
1021: 
1022:     def add_base_types(self, localization, new_types):
1023:         '''
1024:         Adds, if possible, the base types of the hold Python class existing base types.
1025:         For determining if the change is possible, a series of checks (defined before) are made.
1026: 
1027:         :param localization: Call localization data
1028:         :param new_types: New base types (in the form of a tuple)
1029:         :return: A TypeError or None
1030:         '''
1031:         if not type(new_types) is tuple:
1032:             return TypeError(localization, "New subtypes have to be specified using a tuple")
1033: 
1034:         for base_type in new_types:
1035:             check = self.__change_class_base_types_checks(localization, base_type)
1036:             if isinstance(check, TypeError):
1037:                 return check
1038: 
1039:         base_classes = map(lambda tproxy: tproxy.python_entity, new_types)
1040:         self.python_entity.__bases__ += tuple(base_classes)
1041:         return None
1042: 
1043:     # ############################## TYPE CLONING ###############################
1044: 
1045:     def clone(self):
1046:         '''
1047:         Clones the type proxy, making an independent copy of the stored python entity. Physical cloning is not
1048:         performed if the hold python entity do not support intercession, as its structure is immutable.
1049: 
1050:         :return: A clone of this proxy
1051:         '''
1052:         if not self.supports_structural_reflection() and not self.can_store_elements() and \
1053:                 not self.can_store_keypairs():
1054:             return self
1055:         else:
1056:             return type_inference_proxy_management_copy.create_duplicate(self)
1057: 
1058:     # ############################## TYPE INSPECTION ###############################
1059: 
1060:     def dir(self):
1061:         '''
1062:         Calls the dir Python builtin over the stored Python object and returns the result
1063:         :return: list of strings
1064:         '''
1065:         return dir(self.python_entity)
1066: 
1067:     # def dict(self, localization):
1068:     #     '''
1069:     #     Equivalent to call __dict__ over the stored Python instance
1070:     #     :param localization:
1071:     #     :return:
1072:     #     '''
1073:     #     members = self.dir()
1074:     #     ret_dict = TypeInferenceProxy.instance(dict)
1075:     #     ret_dict.set_type_instance(True)
1076:     #     for member in members:
1077:     #         str_instance = TypeInferenceProxy.instance(str, value=member)
1078:     #
1079:     #         value = self.get_type_of_member(localization, member)
1080:     #         ret_dict.add_key_and_value_type(localization, (str_instance, value), False)
1081:     #
1082:     #     return ret_dict
1083: 
1084:     def is_user_defined_class(self):
1085:         '''
1086:         Determines whether this proxy holds an user-defined class or not
1087:         :return:
1088:         '''
1089:         return type_inference_proxy_management_copy.is_user_defined_class(self.python_entity)
1090: 
1091:     # ############################## TYPE ANNOTATION ###############################
1092: 
1093:     def __annotate_type(self, line, column, name, type_):
1094:         '''
1095:         Annotate a type into the proxy type annotation record
1096:         :param line: Source code line when the type change is performed
1097:         :param column: Source code column when the type change is performed
1098:         :param name: Name of the variable whose type is changed
1099:         :param type_: New type
1100:         :return: None
1101:         '''
1102:         if hasattr(self, "annotation_record"):
1103:             self.annotation_record.annotate_type(line, column, name, type_)
1104: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import inspect' statement (line 1)
import inspect

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'inspect', inspect, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import types' statement (line 2)
import types

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'types', types, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import type_inference_proxy_management_copy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_10 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'type_inference_proxy_management_copy')

if (type(import_10) is not StypyTypeError):

    if (import_10 != 'pyd_module'):
        __import__(import_10)
        sys_modules_11 = sys.modules[import_10]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'type_inference_proxy_management_copy', sys_modules_11.module_type_store, module_type_store)
    else:
        import type_inference_proxy_management_copy

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'type_inference_proxy_management_copy', type_inference_proxy_management_copy, module_type_store)

else:
    # Assigning a type to the variable 'type_inference_proxy_management_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'type_inference_proxy_management_copy', import_10)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_12 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy')

if (type(import_12) is not StypyTypeError):

    if (import_12 != 'pyd_module'):
        __import__(import_12)
        sys_modules_13 = sys.modules[import_12]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', sys_modules_13.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_13, sys_modules_13.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', import_12)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy import TypeWarning' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_14 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy')

if (type(import_14) is not StypyTypeError):

    if (import_14 != 'pyd_module'):
        __import__(import_14)
        sys_modules_15 = sys.modules[import_14]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy', sys_modules_15.module_type_store, module_type_store, ['TypeWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_15, sys_modules_15.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy import TypeWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy', None, module_type_store, ['TypeWarning'], [TypeWarning])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy', import_14)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy import call_handlers_copy' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_16 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy')

if (type(import_16) is not StypyTypeError):

    if (import_16 != 'pyd_module'):
        __import__(import_16)
        sys_modules_17 = sys.modules[import_16]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy', sys_modules_17.module_type_store, module_type_store, ['call_handlers_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_17, sys_modules_17.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy import call_handlers_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy', None, module_type_store, ['call_handlers_copy'], [call_handlers_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy', import_16)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy import Type' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_18 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy')

if (type(import_18) is not StypyTypeError):

    if (import_18 != 'pyd_module'):
        __import__(import_18)
        sys_modules_19 = sys.modules[import_18]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy', sys_modules_19.module_type_store, module_type_store, ['Type'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_19, sys_modules_19.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy import Type

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy', None, module_type_store, ['Type'], [Type])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy', import_18)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy import type_equivalence_copy' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_20 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy')

if (type(import_20) is not StypyTypeError):

    if (import_20 != 'pyd_module'):
        __import__(import_20)
        sys_modules_21 = sys.modules[import_20]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy', sys_modules_21.module_type_store, module_type_store, ['type_equivalence_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_21, sys_modules_21.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy import type_equivalence_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy', None, module_type_store, ['type_equivalence_copy'], [type_equivalence_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy', import_20)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy' statement (line 10)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_22 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_22) is not StypyTypeError):

    if (import_22 != 'pyd_module'):
        __import__(import_22)
        sys_modules_23 = sys.modules[import_22]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', sys_modules_23.module_type_store, module_type_store, ['union_type_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_23, sys_modules_23.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['union_type_copy'], [union_type_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', import_22)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import undefined_type_copy' statement (line 11)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_24 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_24) is not StypyTypeError):

    if (import_24 != 'pyd_module'):
        __import__(import_24)
        sys_modules_25 = sys.modules[import_24]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', sys_modules_25.module_type_store, module_type_store, ['undefined_type_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_25, sys_modules_25.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import undefined_type_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['undefined_type_copy'], [undefined_type_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', import_24)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import simple_python_types' statement (line 12)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_26 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy')

if (type(import_26) is not StypyTypeError):

    if (import_26 != 'pyd_module'):
        __import__(import_26)
        sys_modules_27 = sys.modules[import_26]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', sys_modules_27.module_type_store, module_type_store, ['simple_python_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_27, sys_modules_27.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import simple_python_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', None, module_type_store, ['simple_python_types'], [simple_python_types])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', import_26)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy.type_annotation_record_copy import TypeAnnotationRecord' statement (line 13)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_28 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy.type_annotation_record_copy')

if (type(import_28) is not StypyTypeError):

    if (import_28 != 'pyd_module'):
        __import__(import_28)
        sys_modules_29 = sys.modules[import_28]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy.type_annotation_record_copy', sys_modules_29.module_type_store, module_type_store, ['TypeAnnotationRecord'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_29, sys_modules_29.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy.type_annotation_record_copy import TypeAnnotationRecord

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy.type_annotation_record_copy', None, module_type_store, ['TypeAnnotationRecord'], [TypeAnnotationRecord])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy.type_annotation_record_copy' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy.type_annotation_record_copy', import_28)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy import type_store_copy' statement (line 14)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_30 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'testing.test_programs.stypy_code_copy.stypy_copy')

if (type(import_30) is not StypyTypeError):

    if (import_30 != 'pyd_module'):
        __import__(import_30)
        sys_modules_31 = sys.modules[import_30]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', sys_modules_31.module_type_store, module_type_store, ['type_store_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_31, sys_modules_31.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy import type_store_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', None, module_type_store, ['type_store_copy'], [type_store_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', import_30)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy import stypy_parameters_copy' statement (line 15)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_32 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'testing.test_programs.stypy_code_copy.stypy_copy')

if (type(import_32) is not StypyTypeError):

    if (import_32 != 'pyd_module'):
        __import__(import_32)
        sys_modules_33 = sys.modules[import_32]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', sys_modules_33.module_type_store, module_type_store, ['stypy_parameters_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_33, sys_modules_33.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy import stypy_parameters_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', None, module_type_store, ['stypy_parameters_copy'], [stypy_parameters_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', import_32)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

# Declaration of the 'TypeInferenceProxy' class
# Getting the type of 'Type' (line 18)
Type_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 25), 'Type')

class TypeInferenceProxy(Type_34, ):
    str_35 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, (-1)), 'str', '\n    The type inference proxy is the main class of stypy. Its main purpose is to represent any kind of Python type,\n     holding a reference to it. It is also responsible of a lot of possible operations that can be done with the\n     contained type, including:\n\n     - Returning/setting the type of any member of the Python entity it holds.\n     - Invoke any of its associated callable members, returning the invokation result\n     - Support structural reflection operations, it the enclosed object is able to support them\n     - Obtain relationships with other entities (modules that contain a represented function, class that contains\n     a represented method,...)\n     - Manipulate stored types (if the represented entity is able to store other types)\n     - Clone itself to support the SSA algorithm\n     - Respond to builtin operations such as dir and __dict__ calls\n     - Hold values for the represented object type\n\n     All Python entities (functions, variables, methods, classes, modules,...) might be enclosed in a type inference\n     proxy. For those method that are not applicable to the enclosed Python entity, the class will return a TypeError.\n    ')
    
    # Assigning a Call to a Name (line 40):
    
    # Assigning a Name to a Name (line 48):

    @staticmethod
    @norecursion
    def __get_parent_proxy(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__get_parent_proxy'
        module_type_store = module_type_store.open_function_context('__get_parent_proxy', 52, 4, False)
        
        # Passed parameters checking function
        TypeInferenceProxy.__get_parent_proxy.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.__get_parent_proxy.__dict__.__setitem__('stypy_type_of_self', None)
        TypeInferenceProxy.__get_parent_proxy.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.__get_parent_proxy.__dict__.__setitem__('stypy_function_name', '__get_parent_proxy')
        TypeInferenceProxy.__get_parent_proxy.__dict__.__setitem__('stypy_param_names_list', ['parent'])
        TypeInferenceProxy.__get_parent_proxy.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.__get_parent_proxy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.__get_parent_proxy.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.__get_parent_proxy.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.__get_parent_proxy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.__get_parent_proxy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, '__get_parent_proxy', ['parent'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__get_parent_proxy', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__get_parent_proxy(...)' code ##################

        str_36 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, (-1)), 'str', '\n        Gets the python entity that can be considered the "parent" of the passed entity\n        :param parent: Any Python entity\n        :return: The parent of this entity, if any\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 59)
        str_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 27), 'str', '__module__')
        # Getting the type of 'parent' (line 59)
        parent_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 19), 'parent')
        
        (may_be_39, more_types_in_union_40) = may_provide_member(str_37, parent_38)

        if may_be_39:

            if more_types_in_union_40:
                # Runtime conditional SSA (line 59)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'parent' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'parent', remove_not_member_provider_from_union(parent_38, '__module__'))
            
            # Call to instance(...): (line 60)
            # Processing the call arguments (line 60)
            
            # Call to getmodule(...): (line 60)
            # Processing the call arguments (line 60)
            # Getting the type of 'parent' (line 60)
            parent_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 65), 'parent', False)
            # Processing the call keyword arguments (line 60)
            kwargs_46 = {}
            # Getting the type of 'inspect' (line 60)
            inspect_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 47), 'inspect', False)
            # Obtaining the member 'getmodule' of a type (line 60)
            getmodule_44 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 47), inspect_43, 'getmodule')
            # Calling getmodule(args, kwargs) (line 60)
            getmodule_call_result_47 = invoke(stypy.reporting.localization.Localization(__file__, 60, 47), getmodule_44, *[parent_45], **kwargs_46)
            
            # Processing the call keyword arguments (line 60)
            kwargs_48 = {}
            # Getting the type of 'TypeInferenceProxy' (line 60)
            TypeInferenceProxy_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 19), 'TypeInferenceProxy', False)
            # Obtaining the member 'instance' of a type (line 60)
            instance_42 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 19), TypeInferenceProxy_41, 'instance')
            # Calling instance(args, kwargs) (line 60)
            instance_call_result_49 = invoke(stypy.reporting.localization.Localization(__file__, 60, 19), instance_42, *[getmodule_call_result_47], **kwargs_48)
            
            # Assigning a type to the variable 'stypy_return_type' (line 60)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'stypy_return_type', instance_call_result_49)

            if more_types_in_union_40:
                # SSA join for if statement (line 59)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 61)
        str_50 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 27), 'str', '__class__')
        # Getting the type of 'parent' (line 61)
        parent_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 19), 'parent')
        
        (may_be_52, more_types_in_union_53) = may_provide_member(str_50, parent_51)

        if may_be_52:

            if more_types_in_union_53:
                # Runtime conditional SSA (line 61)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'parent' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'parent', remove_not_member_provider_from_union(parent_51, '__class__'))
            
            # Call to instance(...): (line 62)
            # Processing the call arguments (line 62)
            # Getting the type of 'parent' (line 62)
            parent_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 47), 'parent', False)
            # Obtaining the member '__class__' of a type (line 62)
            class___57 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 47), parent_56, '__class__')
            # Processing the call keyword arguments (line 62)
            kwargs_58 = {}
            # Getting the type of 'TypeInferenceProxy' (line 62)
            TypeInferenceProxy_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 19), 'TypeInferenceProxy', False)
            # Obtaining the member 'instance' of a type (line 62)
            instance_55 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 19), TypeInferenceProxy_54, 'instance')
            # Calling instance(args, kwargs) (line 62)
            instance_call_result_59 = invoke(stypy.reporting.localization.Localization(__file__, 62, 19), instance_55, *[class___57], **kwargs_58)
            
            # Assigning a type to the variable 'stypy_return_type' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'stypy_return_type', instance_call_result_59)

            if more_types_in_union_53:
                # SSA join for if statement (line 61)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'None' (line 64)
        None_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'stypy_return_type', None_60)
        
        # ################# End of '__get_parent_proxy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__get_parent_proxy' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__get_parent_proxy'
        return stypy_return_type_61


    @norecursion
    def __assign_parent_proxy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__assign_parent_proxy'
        module_type_store = module_type_store.open_function_context('__assign_parent_proxy', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.__assign_parent_proxy.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.__assign_parent_proxy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.__assign_parent_proxy.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.__assign_parent_proxy.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.__assign_parent_proxy')
        TypeInferenceProxy.__assign_parent_proxy.__dict__.__setitem__('stypy_param_names_list', ['parent'])
        TypeInferenceProxy.__assign_parent_proxy.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.__assign_parent_proxy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.__assign_parent_proxy.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.__assign_parent_proxy.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.__assign_parent_proxy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.__assign_parent_proxy.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.__assign_parent_proxy', ['parent'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__assign_parent_proxy', localization, ['parent'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__assign_parent_proxy(...)' code ##################

        str_62 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, (-1)), 'str', '\n        Changes the parent object of the represented object to the one specified. This is used to trace the nesting\n        of proxies that hold types that are placed inside other proxies represented entities. This property is NOT\n        related with dynamic inheritance.\n        :param parent: The new parent object or None. If the passed parent is None, the class tries to autocalculate it.\n        If there is no possible parent, it is assigned to None\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 74)
        # Getting the type of 'parent' (line 74)
        parent_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'parent')
        # Getting the type of 'None' (line 74)
        None_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), 'None')
        
        (may_be_65, more_types_in_union_66) = may_not_be_none(parent_63, None_64)

        if may_be_65:

            if more_types_in_union_66:
                # Runtime conditional SSA (line 74)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 75):
            
            # Assigning a Name to a Attribute (line 75):
            # Getting the type of 'parent' (line 75)
            parent_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 32), 'parent')
            # Getting the type of 'self' (line 75)
            self_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'self')
            # Setting the type of the member 'parent_proxy' of a type (line 75)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), self_68, 'parent_proxy', parent_67)

            if more_types_in_union_66:
                # Runtime conditional SSA for else branch (line 74)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_65) or more_types_in_union_66):
            
            
            # Call to ismodule(...): (line 77)
            # Processing the call arguments (line 77)
            # Getting the type of 'self' (line 77)
            self_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 36), 'self', False)
            # Obtaining the member 'python_entity' of a type (line 77)
            python_entity_72 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 36), self_71, 'python_entity')
            # Processing the call keyword arguments (line 77)
            kwargs_73 = {}
            # Getting the type of 'inspect' (line 77)
            inspect_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'inspect', False)
            # Obtaining the member 'ismodule' of a type (line 77)
            ismodule_70 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 19), inspect_69, 'ismodule')
            # Calling ismodule(args, kwargs) (line 77)
            ismodule_call_result_74 = invoke(stypy.reporting.localization.Localization(__file__, 77, 19), ismodule_70, *[python_entity_72], **kwargs_73)
            
            # Applying the 'not' unary operator (line 77)
            result_not__75 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 15), 'not', ismodule_call_result_74)
            
            # Testing if the type of an if condition is none (line 77)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 77, 12), result_not__75):
                
                # Assigning a Name to a Attribute (line 80):
                
                # Assigning a Name to a Attribute (line 80):
                # Getting the type of 'None' (line 80)
                None_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 36), 'None')
                # Getting the type of 'self' (line 80)
                self_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'self')
                # Setting the type of the member 'parent_proxy' of a type (line 80)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 16), self_85, 'parent_proxy', None_84)
            else:
                
                # Testing the type of an if condition (line 77)
                if_condition_76 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 12), result_not__75)
                # Assigning a type to the variable 'if_condition_76' (line 77)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'if_condition_76', if_condition_76)
                # SSA begins for if statement (line 77)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Attribute (line 78):
                
                # Assigning a Call to a Attribute (line 78):
                
                # Call to __get_parent_proxy(...): (line 78)
                # Processing the call arguments (line 78)
                # Getting the type of 'self' (line 78)
                self_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 74), 'self', False)
                # Obtaining the member 'python_entity' of a type (line 78)
                python_entity_80 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 74), self_79, 'python_entity')
                # Processing the call keyword arguments (line 78)
                kwargs_81 = {}
                # Getting the type of 'TypeInferenceProxy' (line 78)
                TypeInferenceProxy_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 36), 'TypeInferenceProxy', False)
                # Obtaining the member '__get_parent_proxy' of a type (line 78)
                get_parent_proxy_78 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 36), TypeInferenceProxy_77, '__get_parent_proxy')
                # Calling __get_parent_proxy(args, kwargs) (line 78)
                get_parent_proxy_call_result_82 = invoke(stypy.reporting.localization.Localization(__file__, 78, 36), get_parent_proxy_78, *[python_entity_80], **kwargs_81)
                
                # Getting the type of 'self' (line 78)
                self_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'self')
                # Setting the type of the member 'parent_proxy' of a type (line 78)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 16), self_83, 'parent_proxy', get_parent_proxy_call_result_82)
                # SSA branch for the else part of an if statement (line 77)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Attribute (line 80):
                
                # Assigning a Name to a Attribute (line 80):
                # Getting the type of 'None' (line 80)
                None_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 36), 'None')
                # Getting the type of 'self' (line 80)
                self_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'self')
                # Setting the type of the member 'parent_proxy' of a type (line 80)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 16), self_85, 'parent_proxy', None_84)
                # SSA join for if statement (line 77)
                module_type_store = module_type_store.join_ssa_context()
                


            if (may_be_65 and more_types_in_union_66):
                # SSA join for if statement (line 74)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__assign_parent_proxy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__assign_parent_proxy' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_86)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__assign_parent_proxy'
        return stypy_return_type_86


    @norecursion
    def __change_class_base_types_checks(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__change_class_base_types_checks'
        module_type_store = module_type_store.open_function_context('__change_class_base_types_checks', 82, 4, False)
        # Assigning a type to the variable 'self' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.__change_class_base_types_checks.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.__change_class_base_types_checks.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.__change_class_base_types_checks.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.__change_class_base_types_checks.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.__change_class_base_types_checks')
        TypeInferenceProxy.__change_class_base_types_checks.__dict__.__setitem__('stypy_param_names_list', ['localization', 'new_type'])
        TypeInferenceProxy.__change_class_base_types_checks.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.__change_class_base_types_checks.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.__change_class_base_types_checks.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.__change_class_base_types_checks.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.__change_class_base_types_checks.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.__change_class_base_types_checks.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.__change_class_base_types_checks', ['localization', 'new_type'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__change_class_base_types_checks', localization, ['localization', 'new_type'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__change_class_base_types_checks(...)' code ##################

        str_87 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, (-1)), 'str', '\n        Performs all the possible checks to see if a base type change is possible for the currently hold python entity.\n        This includes:\n        - Making sure that the currently hold object represent a class. No base type change is possible if the hold\n        entity is not a class. For checking the availability of an instance type change, see the\n        "__change_instance_type_checks" private method.\n        - Making sure that the hold class is not a new-style class. New style Python classes cannot change its base\n        type directly, as its __mro__ (Method Resolution Order) property is readonly. For this purpose a metaclass\n        has to be created, like in this example:\n\n        class change_mro_meta(type):\n            def __new__(cls, cls_name, cls_bases, cls_dict):\n                    out_cls = super(change_mro_meta, cls).__new__(cls, cls_name, cls_bases, cls_dict)\n                    out_cls.change_mro = False\n                    out_cls.hack_mro   = classmethod(cls.hack_mro)\n                    out_cls.fix_mro    = classmethod(cls.fix_mro)\n                    out_cls.recalc_mro = classmethod(cls.recalc_mro)\n                    return out_cls\n\n            @staticmethod\n            def hack_mro(cls):\n                cls.change_mro = True\n                cls.recalc_mro()\n\n            @staticmethod\n            def fix_mro(cls):\n                cls.change_mro = False\n                cls.recalc_mro()\n\n            @staticmethod\n            def recalc_mro(cls):\n                # Changing a class\' base causes __mro__ recalculation\n                cls.__bases__  = cls.__bases__ + tuple()\n\n            def mro(cls):\n                default_mro = super(change_mro_meta, cls).mro()\n                if hasattr(cls, "change_mro") and cls.change_mro:\n                    return default_mro[1:2] + default_mro\n                else:\n                    return default_mro\n\n        - Making sure that new base class do not belong to a different class style as the current one: base type of\n        old-style classes can only be changed to another old-style class.\n\n        :param localization: Call localization data\n        :param new_type: New base type to change to\n        :return: A Type error specifying the problem encountered with the base type change or None if no error is found\n        ')
        
        
        # Call to is_class(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'self' (line 131)
        self_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 61), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 131)
        python_entity_91 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 61), self_90, 'python_entity')
        # Processing the call keyword arguments (line 131)
        kwargs_92 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 131)
        type_inference_proxy_management_copy_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'is_class' of a type (line 131)
        is_class_89 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 15), type_inference_proxy_management_copy_88, 'is_class')
        # Calling is_class(args, kwargs) (line 131)
        is_class_call_result_93 = invoke(stypy.reporting.localization.Localization(__file__, 131, 15), is_class_89, *[python_entity_91], **kwargs_92)
        
        # Applying the 'not' unary operator (line 131)
        result_not__94 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 11), 'not', is_class_call_result_93)
        
        # Testing if the type of an if condition is none (line 131)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 131, 8), result_not__94):
            pass
        else:
            
            # Testing the type of an if condition (line 131)
            if_condition_95 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 131, 8), result_not__94)
            # Assigning a type to the variable 'if_condition_95' (line 131)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'if_condition_95', if_condition_95)
            # SSA begins for if statement (line 131)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 132)
            # Processing the call arguments (line 132)
            # Getting the type of 'localization' (line 132)
            localization_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 29), 'localization', False)
            str_98 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 43), 'str', 'Cannot change the base type of a non-class Python entity')
            # Processing the call keyword arguments (line 132)
            kwargs_99 = {}
            # Getting the type of 'TypeError' (line 132)
            TypeError_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 132)
            TypeError_call_result_100 = invoke(stypy.reporting.localization.Localization(__file__, 132, 19), TypeError_96, *[localization_97, str_98], **kwargs_99)
            
            # Assigning a type to the variable 'stypy_return_type' (line 132)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'stypy_return_type', TypeError_call_result_100)
            # SSA join for if statement (line 131)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to is_new_style_class(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'self' (line 134)
        self_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 67), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 134)
        python_entity_104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 67), self_103, 'python_entity')
        # Processing the call keyword arguments (line 134)
        kwargs_105 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 134)
        type_inference_proxy_management_copy_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 11), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'is_new_style_class' of a type (line 134)
        is_new_style_class_102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 11), type_inference_proxy_management_copy_101, 'is_new_style_class')
        # Calling is_new_style_class(args, kwargs) (line 134)
        is_new_style_class_call_result_106 = invoke(stypy.reporting.localization.Localization(__file__, 134, 11), is_new_style_class_102, *[python_entity_104], **kwargs_105)
        
        # Testing if the type of an if condition is none (line 134)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 134, 8), is_new_style_class_call_result_106):
            pass
        else:
            
            # Testing the type of an if condition (line 134)
            if_condition_107 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 8), is_new_style_class_call_result_106)
            # Assigning a type to the variable 'if_condition_107' (line 134)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'if_condition_107', if_condition_107)
            # SSA begins for if statement (line 134)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 135)
            # Processing the call arguments (line 135)
            # Getting the type of 'localization' (line 135)
            localization_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 29), 'localization', False)
            str_110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 29), 'str', 'Cannot change the class hierarchy of a new-style class: The __mro__ (Method Resolution Order) property is readonly')
            # Processing the call keyword arguments (line 135)
            kwargs_111 = {}
            # Getting the type of 'TypeError' (line 135)
            TypeError_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 135)
            TypeError_call_result_112 = invoke(stypy.reporting.localization.Localization(__file__, 135, 19), TypeError_108, *[localization_109, str_110], **kwargs_111)
            
            # Assigning a type to the variable 'stypy_return_type' (line 135)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'stypy_return_type', TypeError_call_result_112)
            # SSA join for if statement (line 134)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 139)
        self_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 11), 'self')
        # Obtaining the member 'instance' of a type (line 139)
        instance_114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 11), self_113, 'instance')
        # Getting the type of 'None' (line 139)
        None_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 32), 'None')
        # Applying the binary operator 'isnot' (line 139)
        result_is_not_116 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 11), 'isnot', instance_114, None_115)
        
        # Testing if the type of an if condition is none (line 139)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 139, 8), result_is_not_116):
            pass
        else:
            
            # Testing the type of an if condition (line 139)
            if_condition_117 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 8), result_is_not_116)
            # Assigning a type to the variable 'if_condition_117' (line 139)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'if_condition_117', if_condition_117)
            # SSA begins for if statement (line 139)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 140)
            # Processing the call arguments (line 140)
            # Getting the type of 'localization' (line 140)
            localization_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 29), 'localization', False)
            str_120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 43), 'str', 'Cannot change the class hierarchy of a class using an instance')
            # Processing the call keyword arguments (line 140)
            kwargs_121 = {}
            # Getting the type of 'TypeError' (line 140)
            TypeError_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 140)
            TypeError_call_result_122 = invoke(stypy.reporting.localization.Localization(__file__, 140, 19), TypeError_118, *[localization_119, str_120], **kwargs_121)
            
            # Assigning a type to the variable 'stypy_return_type' (line 140)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'stypy_return_type', TypeError_call_result_122)
            # SSA join for if statement (line 139)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 142):
        
        # Assigning a Call to a Name (line 142):
        
        # Call to is_old_style_class(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'self' (line 142)
        self_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 85), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 142)
        python_entity_126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 85), self_125, 'python_entity')
        # Processing the call keyword arguments (line 142)
        kwargs_127 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 142)
        type_inference_proxy_management_copy_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 29), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'is_old_style_class' of a type (line 142)
        is_old_style_class_124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 29), type_inference_proxy_management_copy_123, 'is_old_style_class')
        # Calling is_old_style_class(args, kwargs) (line 142)
        is_old_style_class_call_result_128 = invoke(stypy.reporting.localization.Localization(__file__, 142, 29), is_old_style_class_124, *[python_entity_126], **kwargs_127)
        
        # Assigning a type to the variable 'old_style_existing' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'old_style_existing', is_old_style_class_call_result_128)
        
        # Type idiom detected: calculating its left and rigth part (line 143)
        # Getting the type of 'TypeError' (line 143)
        TypeError_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 36), 'TypeError')
        # Getting the type of 'new_type' (line 143)
        new_type_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 26), 'new_type')
        
        (may_be_131, more_types_in_union_132) = may_not_be_subtype(TypeError_129, new_type_130)

        if may_be_131:

            if more_types_in_union_132:
                # Runtime conditional SSA (line 143)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'new_type' (line 143)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'new_type', remove_subtype_from_union(new_type_130, TypeError))
            
            # Assigning a Call to a Name (line 144):
            
            # Assigning a Call to a Name (line 144):
            
            # Call to is_old_style_class(...): (line 144)
            # Processing the call arguments (line 144)
            # Getting the type of 'new_type' (line 144)
            new_type_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 84), 'new_type', False)
            # Obtaining the member 'python_entity' of a type (line 144)
            python_entity_136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 84), new_type_135, 'python_entity')
            # Processing the call keyword arguments (line 144)
            kwargs_137 = {}
            # Getting the type of 'type_inference_proxy_management_copy' (line 144)
            type_inference_proxy_management_copy_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 28), 'type_inference_proxy_management_copy', False)
            # Obtaining the member 'is_old_style_class' of a type (line 144)
            is_old_style_class_134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 28), type_inference_proxy_management_copy_133, 'is_old_style_class')
            # Calling is_old_style_class(args, kwargs) (line 144)
            is_old_style_class_call_result_138 = invoke(stypy.reporting.localization.Localization(__file__, 144, 28), is_old_style_class_134, *[python_entity_136], **kwargs_137)
            
            # Assigning a type to the variable 'old_style_new' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'old_style_new', is_old_style_class_call_result_138)

            if more_types_in_union_132:
                # Runtime conditional SSA for else branch (line 143)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_131) or more_types_in_union_132):
            # Assigning a type to the variable 'new_type' (line 143)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'new_type', remove_not_subtype_from_union(new_type_130, TypeError))
            
            # Call to TypeError(...): (line 146)
            # Processing the call arguments (line 146)
            # Getting the type of 'localization' (line 146)
            localization_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 29), 'localization', False)
            str_141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 43), 'str', 'Cannot change the class hierarchy to a type error')
            # Processing the call keyword arguments (line 146)
            kwargs_142 = {}
            # Getting the type of 'TypeError' (line 146)
            TypeError_139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 146)
            TypeError_call_result_143 = invoke(stypy.reporting.localization.Localization(__file__, 146, 19), TypeError_139, *[localization_140, str_141], **kwargs_142)
            
            # Assigning a type to the variable 'stypy_return_type' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'stypy_return_type', TypeError_call_result_143)

            if (may_be_131 and more_types_in_union_132):
                # SSA join for if statement (line 143)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'old_style_existing' (line 149)
        old_style_existing_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'old_style_existing')
        # Getting the type of 'old_style_new' (line 149)
        old_style_new_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 37), 'old_style_new')
        # Applying the binary operator '==' (line 149)
        result_eq_146 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 15), '==', old_style_existing_144, old_style_new_145)
        
        # Applying the 'not' unary operator (line 149)
        result_not__147 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 11), 'not', result_eq_146)
        
        # Testing if the type of an if condition is none (line 149)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 149, 8), result_not__147):
            pass
        else:
            
            # Testing the type of an if condition (line 149)
            if_condition_148 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 8), result_not__147)
            # Assigning a type to the variable 'if_condition_148' (line 149)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'if_condition_148', if_condition_148)
            # SSA begins for if statement (line 149)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 150)
            # Processing the call arguments (line 150)
            # Getting the type of 'localization' (line 150)
            localization_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 29), 'localization', False)
            str_151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 43), 'str', 'Cannot change the class hierarchy from an old-style Python parent class to a new-style Python parent class')
            # Processing the call keyword arguments (line 150)
            kwargs_152 = {}
            # Getting the type of 'TypeError' (line 150)
            TypeError_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 150)
            TypeError_call_result_153 = invoke(stypy.reporting.localization.Localization(__file__, 150, 19), TypeError_149, *[localization_150, str_151], **kwargs_152)
            
            # Assigning a type to the variable 'stypy_return_type' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'stypy_return_type', TypeError_call_result_153)
            # SSA join for if statement (line 149)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'None' (line 153)
        None_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'stypy_return_type', None_154)
        
        # ################# End of '__change_class_base_types_checks(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__change_class_base_types_checks' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_155)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__change_class_base_types_checks'
        return stypy_return_type_155


    @norecursion
    def __change_instance_type_checks(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__change_instance_type_checks'
        module_type_store = module_type_store.open_function_context('__change_instance_type_checks', 155, 4, False)
        # Assigning a type to the variable 'self' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.__change_instance_type_checks.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.__change_instance_type_checks.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.__change_instance_type_checks.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.__change_instance_type_checks.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.__change_instance_type_checks')
        TypeInferenceProxy.__change_instance_type_checks.__dict__.__setitem__('stypy_param_names_list', ['localization', 'new_type'])
        TypeInferenceProxy.__change_instance_type_checks.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.__change_instance_type_checks.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.__change_instance_type_checks.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.__change_instance_type_checks.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.__change_instance_type_checks.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.__change_instance_type_checks.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.__change_instance_type_checks', ['localization', 'new_type'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__change_instance_type_checks', localization, ['localization', 'new_type'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__change_instance_type_checks(...)' code ##################

        str_156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, (-1)), 'str', '\n        Performs all the checks that ensure that changing the type of an instance is possible. This includes:\n        - Making sure that we are changing the type of an user-defined class instance. Type change for Python\n        library classes instances is not possible.\n        - Making sure that the old instance type and the new instance type are of the same class style, as mixing\n        old-style and new-style types is not possible in Python.\n\n        :param localization: Call localization data\n        :param new_type: New instance type.\n        :return:\n        ')
        
        
        # Call to is_class(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'self' (line 168)
        self_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 61), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 168)
        python_entity_160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 61), self_159, 'python_entity')
        # Processing the call keyword arguments (line 168)
        kwargs_161 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 168)
        type_inference_proxy_management_copy_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 15), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'is_class' of a type (line 168)
        is_class_158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 15), type_inference_proxy_management_copy_157, 'is_class')
        # Calling is_class(args, kwargs) (line 168)
        is_class_call_result_162 = invoke(stypy.reporting.localization.Localization(__file__, 168, 15), is_class_158, *[python_entity_160], **kwargs_161)
        
        # Applying the 'not' unary operator (line 168)
        result_not__163 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 11), 'not', is_class_call_result_162)
        
        # Testing if the type of an if condition is none (line 168)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 168, 8), result_not__163):
            pass
        else:
            
            # Testing the type of an if condition (line 168)
            if_condition_164 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 8), result_not__163)
            # Assigning a type to the variable 'if_condition_164' (line 168)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'if_condition_164', if_condition_164)
            # SSA begins for if statement (line 168)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 169)
            # Processing the call arguments (line 169)
            # Getting the type of 'localization' (line 169)
            localization_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 29), 'localization', False)
            str_167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 43), 'str', 'Cannot change the type of a Python entity that it is not a class')
            # Processing the call keyword arguments (line 169)
            kwargs_168 = {}
            # Getting the type of 'TypeError' (line 169)
            TypeError_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 169)
            TypeError_call_result_169 = invoke(stypy.reporting.localization.Localization(__file__, 169, 19), TypeError_165, *[localization_166, str_167], **kwargs_168)
            
            # Assigning a type to the variable 'stypy_return_type' (line 169)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'stypy_return_type', TypeError_call_result_169)
            # SSA join for if statement (line 168)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to is_user_defined_class(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'self' (line 172)
        self_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 74), 'self', False)
        # Obtaining the member 'instance' of a type (line 172)
        instance_173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 74), self_172, 'instance')
        # Obtaining the member '__class__' of a type (line 172)
        class___174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 74), instance_173, '__class__')
        # Processing the call keyword arguments (line 172)
        kwargs_175 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 172)
        type_inference_proxy_management_copy_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'is_user_defined_class' of a type (line 172)
        is_user_defined_class_171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 15), type_inference_proxy_management_copy_170, 'is_user_defined_class')
        # Calling is_user_defined_class(args, kwargs) (line 172)
        is_user_defined_class_call_result_176 = invoke(stypy.reporting.localization.Localization(__file__, 172, 15), is_user_defined_class_171, *[class___174], **kwargs_175)
        
        # Applying the 'not' unary operator (line 172)
        result_not__177 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 11), 'not', is_user_defined_class_call_result_176)
        
        # Testing if the type of an if condition is none (line 172)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 172, 8), result_not__177):
            pass
        else:
            
            # Testing the type of an if condition (line 172)
            if_condition_178 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 8), result_not__177)
            # Assigning a type to the variable 'if_condition_178' (line 172)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'if_condition_178', if_condition_178)
            # SSA begins for if statement (line 172)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 173)
            # Processing the call arguments (line 173)
            # Getting the type of 'localization' (line 173)
            localization_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 29), 'localization', False)
            str_181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 43), 'str', 'Cannot change the type of an instance of a non user-defined class')
            # Processing the call keyword arguments (line 173)
            kwargs_182 = {}
            # Getting the type of 'TypeError' (line 173)
            TypeError_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 173)
            TypeError_call_result_183 = invoke(stypy.reporting.localization.Localization(__file__, 173, 19), TypeError_179, *[localization_180, str_181], **kwargs_182)
            
            # Assigning a type to the variable 'stypy_return_type' (line 173)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'stypy_return_type', TypeError_call_result_183)
            # SSA join for if statement (line 172)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Type idiom detected: calculating its left and rigth part (line 176)
        # Getting the type of 'self' (line 176)
        self_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 11), 'self')
        # Obtaining the member 'instance' of a type (line 176)
        instance_185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 11), self_184, 'instance')
        # Getting the type of 'None' (line 176)
        None_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 28), 'None')
        
        (may_be_187, more_types_in_union_188) = may_be_none(instance_185, None_186)

        if may_be_187:

            if more_types_in_union_188:
                # Runtime conditional SSA (line 176)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to TypeError(...): (line 177)
            # Processing the call arguments (line 177)
            # Getting the type of 'localization' (line 177)
            localization_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 29), 'localization', False)
            str_191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 43), 'str', 'Cannot change the type of a class object; Type change is only possiblewith class instances')
            # Processing the call keyword arguments (line 177)
            kwargs_192 = {}
            # Getting the type of 'TypeError' (line 177)
            TypeError_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 177)
            TypeError_call_result_193 = invoke(stypy.reporting.localization.Localization(__file__, 177, 19), TypeError_189, *[localization_190, str_191], **kwargs_192)
            
            # Assigning a type to the variable 'stypy_return_type' (line 177)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'stypy_return_type', TypeError_call_result_193)

            if more_types_in_union_188:
                # SSA join for if statement (line 176)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 180):
        
        # Assigning a Call to a Name (line 180):
        
        # Call to is_old_style_class(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'self' (line 180)
        self_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 85), 'self', False)
        # Obtaining the member 'instance' of a type (line 180)
        instance_197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 85), self_196, 'instance')
        # Obtaining the member '__class__' of a type (line 180)
        class___198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 85), instance_197, '__class__')
        # Processing the call keyword arguments (line 180)
        kwargs_199 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 180)
        type_inference_proxy_management_copy_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 29), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'is_old_style_class' of a type (line 180)
        is_old_style_class_195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 29), type_inference_proxy_management_copy_194, 'is_old_style_class')
        # Calling is_old_style_class(args, kwargs) (line 180)
        is_old_style_class_call_result_200 = invoke(stypy.reporting.localization.Localization(__file__, 180, 29), is_old_style_class_195, *[class___198], **kwargs_199)
        
        # Assigning a type to the variable 'old_style_existing' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'old_style_existing', is_old_style_class_call_result_200)
        
        # Assigning a Call to a Name (line 181):
        
        # Assigning a Call to a Name (line 181):
        
        # Call to is_old_style_class(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'new_type' (line 181)
        new_type_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 80), 'new_type', False)
        # Obtaining the member 'python_entity' of a type (line 181)
        python_entity_204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 80), new_type_203, 'python_entity')
        # Processing the call keyword arguments (line 181)
        kwargs_205 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 181)
        type_inference_proxy_management_copy_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 24), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'is_old_style_class' of a type (line 181)
        is_old_style_class_202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 24), type_inference_proxy_management_copy_201, 'is_old_style_class')
        # Calling is_old_style_class(args, kwargs) (line 181)
        is_old_style_class_call_result_206 = invoke(stypy.reporting.localization.Localization(__file__, 181, 24), is_old_style_class_202, *[python_entity_204], **kwargs_205)
        
        # Assigning a type to the variable 'old_style_new' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'old_style_new', is_old_style_class_call_result_206)
        
        
        # Getting the type of 'old_style_existing' (line 184)
        old_style_existing_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 15), 'old_style_existing')
        # Getting the type of 'old_style_new' (line 184)
        old_style_new_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 37), 'old_style_new')
        # Applying the binary operator '==' (line 184)
        result_eq_209 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 15), '==', old_style_existing_207, old_style_new_208)
        
        # Applying the 'not' unary operator (line 184)
        result_not__210 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 11), 'not', result_eq_209)
        
        # Testing if the type of an if condition is none (line 184)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 184, 8), result_not__210):
            pass
        else:
            
            # Testing the type of an if condition (line 184)
            if_condition_211 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 8), result_not__210)
            # Assigning a type to the variable 'if_condition_211' (line 184)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'if_condition_211', if_condition_211)
            # SSA begins for if statement (line 184)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 185)
            # Processing the call arguments (line 185)
            # Getting the type of 'localization' (line 185)
            localization_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 29), 'localization', False)
            str_214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 43), 'str', 'Cannot change the type of an instances from an old-style Python class to a new-style Python class or viceversa')
            # Processing the call keyword arguments (line 185)
            kwargs_215 = {}
            # Getting the type of 'TypeError' (line 185)
            TypeError_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 185)
            TypeError_call_result_216 = invoke(stypy.reporting.localization.Localization(__file__, 185, 19), TypeError_212, *[localization_213, str_214], **kwargs_215)
            
            # Assigning a type to the variable 'stypy_return_type' (line 185)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'stypy_return_type', TypeError_call_result_216)
            # SSA join for if statement (line 184)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'None' (line 188)
        None_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'stypy_return_type', None_217)
        
        # ################# End of '__change_instance_type_checks(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__change_instance_type_checks' in the type store
        # Getting the type of 'stypy_return_type' (line 155)
        stypy_return_type_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_218)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__change_instance_type_checks'
        return stypy_return_type_218


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 192)
        None_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 43), 'None')
        # Getting the type of 'None' (line 192)
        None_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 56), 'None')
        # Getting the type of 'None' (line 192)
        None_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 71), 'None')
        # Getting the type of 'undefined_type_copy' (line 192)
        undefined_type_copy_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 83), 'undefined_type_copy')
        # Obtaining the member 'UndefinedType' of a type (line 192)
        UndefinedType_223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 83), undefined_type_copy_222, 'UndefinedType')
        defaults = [None_219, None_220, None_221, UndefinedType_223]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 192, 4, False)
        # Assigning a type to the variable 'self' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.__init__', ['python_entity', 'name', 'parent', 'instance', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['python_entity', 'name', 'parent', 'instance', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, (-1)), 'str', '\n        Creates a new Type inference proxy for the passed python entity (function, module, class, ...). This constructor\n        should NOT be called directly. Use the instance(...) method instead to take advantage of the implemented\n        type memoization of this class.\n        :param python_entity: Represented python entity.\n        :param name: Name of the represented Python entity. If nothing is provided, the Python entity __name__ property\n        value is used instead. Instances have an special name indicating that this entity holds an instance of a class.\n        :param parent: Parent proxy object. If nothing is provided, the parent proxy is autocalculated, if possible.\n        :param instance: Instance of the represented class. If this proxy holds a class, it is possible that instead\n        of representing the class is representing a particular class instance. This is important to properly model\n        instance intercession, as altering the structure of single class instances is possible.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 205)
        # Getting the type of 'name' (line 205)
        name_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 11), 'name')
        # Getting the type of 'None' (line 205)
        None_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 19), 'None')
        
        (may_be_227, more_types_in_union_228) = may_be_none(name_225, None_226)

        if may_be_227:

            if more_types_in_union_228:
                # Runtime conditional SSA (line 205)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 206)
            str_229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 38), 'str', '__name__')
            # Getting the type of 'python_entity' (line 206)
            python_entity_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 23), 'python_entity')
            
            (may_be_231, more_types_in_union_232) = may_provide_member(str_229, python_entity_230)

            if may_be_231:

                if more_types_in_union_232:
                    # Runtime conditional SSA (line 206)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'python_entity' (line 206)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'python_entity', remove_not_member_provider_from_union(python_entity_230, '__name__'))
                
                # Assigning a Attribute to a Attribute (line 207):
                
                # Assigning a Attribute to a Attribute (line 207):
                # Getting the type of 'python_entity' (line 207)
                python_entity_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 28), 'python_entity')
                # Obtaining the member '__name__' of a type (line 207)
                name___234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 28), python_entity_233, '__name__')
                # Getting the type of 'self' (line 207)
                self_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'self')
                # Setting the type of the member 'name' of a type (line 207)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 16), self_235, 'name', name___234)

                if more_types_in_union_232:
                    # Runtime conditional SSA for else branch (line 206)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_231) or more_types_in_union_232):
                # Assigning a type to the variable 'python_entity' (line 206)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'python_entity', remove_member_provider_from_union(python_entity_230, '__name__'))
                
                # Type idiom detected: calculating its left and rigth part (line 209)
                str_236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 42), 'str', '__class__')
                # Getting the type of 'python_entity' (line 209)
                python_entity_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 27), 'python_entity')
                
                (may_be_238, more_types_in_union_239) = may_provide_member(str_236, python_entity_237)

                if may_be_238:

                    if more_types_in_union_239:
                        # Runtime conditional SSA (line 209)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Assigning a type to the variable 'python_entity' (line 209)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 16), 'python_entity', remove_not_member_provider_from_union(python_entity_237, '__class__'))
                    
                    # Assigning a Attribute to a Attribute (line 210):
                    
                    # Assigning a Attribute to a Attribute (line 210):
                    # Getting the type of 'python_entity' (line 210)
                    python_entity_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 32), 'python_entity')
                    # Obtaining the member '__class__' of a type (line 210)
                    class___241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 32), python_entity_240, '__class__')
                    # Obtaining the member '__name__' of a type (line 210)
                    name___242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 32), class___241, '__name__')
                    # Getting the type of 'self' (line 210)
                    self_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 20), 'self')
                    # Setting the type of the member 'name' of a type (line 210)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 20), self_243, 'name', name___242)

                    if more_types_in_union_239:
                        # Runtime conditional SSA for else branch (line 209)
                        module_type_store.open_ssa_branch('idiom else')



                if ((not may_be_238) or more_types_in_union_239):
                    # Assigning a type to the variable 'python_entity' (line 209)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 16), 'python_entity', remove_member_provider_from_union(python_entity_237, '__class__'))
                    
                    # Type idiom detected: calculating its left and rigth part (line 212)
                    str_244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 46), 'str', '__module__')
                    # Getting the type of 'python_entity' (line 212)
                    python_entity_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 31), 'python_entity')
                    
                    (may_be_246, more_types_in_union_247) = may_provide_member(str_244, python_entity_245)

                    if may_be_246:

                        if more_types_in_union_247:
                            # Runtime conditional SSA (line 212)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                        else:
                            module_type_store = module_type_store

                        # Assigning a type to the variable 'python_entity' (line 212)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), 'python_entity', remove_not_member_provider_from_union(python_entity_245, '__module__'))
                        
                        # Assigning a Attribute to a Attribute (line 213):
                        
                        # Assigning a Attribute to a Attribute (line 213):
                        # Getting the type of 'python_entity' (line 213)
                        python_entity_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 36), 'python_entity')
                        # Obtaining the member '__module__' of a type (line 213)
                        module___249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 36), python_entity_248, '__module__')
                        # Getting the type of 'self' (line 213)
                        self_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 24), 'self')
                        # Setting the type of the member 'name' of a type (line 213)
                        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 24), self_250, 'name', module___249)

                        if more_types_in_union_247:
                            # SSA join for if statement (line 212)
                            module_type_store = module_type_store.join_ssa_context()


                    

                    if (may_be_238 and more_types_in_union_239):
                        # SSA join for if statement (line 209)
                        module_type_store = module_type_store.join_ssa_context()


                

                if (may_be_231 and more_types_in_union_232):
                    # SSA join for if statement (line 206)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Type idiom detected: calculating its left and rigth part (line 215)
            # Getting the type of 'instance' (line 215)
            instance_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'instance')
            # Getting the type of 'None' (line 215)
            None_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 31), 'None')
            
            (may_be_253, more_types_in_union_254) = may_not_be_none(instance_251, None_252)

            if may_be_253:

                if more_types_in_union_254:
                    # Runtime conditional SSA (line 215)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a BinOp to a Attribute (line 216):
                
                # Assigning a BinOp to a Attribute (line 216):
                str_255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 28), 'str', '<')
                # Getting the type of 'self' (line 216)
                self_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 34), 'self')
                # Obtaining the member 'name' of a type (line 216)
                name_257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 34), self_256, 'name')
                # Applying the binary operator '+' (line 216)
                result_add_258 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 28), '+', str_255, name_257)
                
                str_259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 46), 'str', ' instance>')
                # Applying the binary operator '+' (line 216)
                result_add_260 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 44), '+', result_add_258, str_259)
                
                # Getting the type of 'self' (line 216)
                self_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'self')
                # Setting the type of the member 'name' of a type (line 216)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 16), self_261, 'name', result_add_260)

                if more_types_in_union_254:
                    # SSA join for if statement (line 215)
                    module_type_store = module_type_store.join_ssa_context()


            

            if more_types_in_union_228:
                # Runtime conditional SSA for else branch (line 205)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_227) or more_types_in_union_228):
            
            # Assigning a Name to a Attribute (line 218):
            
            # Assigning a Name to a Attribute (line 218):
            # Getting the type of 'name' (line 218)
            name_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 24), 'name')
            # Getting the type of 'self' (line 218)
            self_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'self')
            # Setting the type of the member 'name' of a type (line 218)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 12), self_263, 'name', name_262)

            if (may_be_227 and more_types_in_union_228):
                # SSA join for if statement (line 205)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 220):
        
        # Assigning a Name to a Attribute (line 220):
        # Getting the type of 'python_entity' (line 220)
        python_entity_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 29), 'python_entity')
        # Getting the type of 'self' (line 220)
        self_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'self')
        # Setting the type of the member 'python_entity' of a type (line 220)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 8), self_265, 'python_entity', python_entity_264)
        
        # Call to __assign_parent_proxy(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'parent' (line 221)
        parent_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 35), 'parent', False)
        # Processing the call keyword arguments (line 221)
        kwargs_269 = {}
        # Getting the type of 'self' (line 221)
        self_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'self', False)
        # Obtaining the member '__assign_parent_proxy' of a type (line 221)
        assign_parent_proxy_267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), self_266, '__assign_parent_proxy')
        # Calling __assign_parent_proxy(args, kwargs) (line 221)
        assign_parent_proxy_call_result_270 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), assign_parent_proxy_267, *[parent_268], **kwargs_269)
        
        
        # Assigning a Name to a Attribute (line 222):
        
        # Assigning a Name to a Attribute (line 222):
        # Getting the type of 'instance' (line 222)
        instance_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 24), 'instance')
        # Getting the type of 'self' (line 222)
        self_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'self')
        # Setting the type of the member 'instance' of a type (line 222)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), self_272, 'instance', instance_271)
        
        # Type idiom detected: calculating its left and rigth part (line 223)
        # Getting the type of 'instance' (line 223)
        instance_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'instance')
        # Getting the type of 'None' (line 223)
        None_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 27), 'None')
        
        (may_be_275, more_types_in_union_276) = may_not_be_none(instance_273, None_274)

        if may_be_275:

            if more_types_in_union_276:
                # Runtime conditional SSA (line 223)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to set_type_instance(...): (line 224)
            # Processing the call arguments (line 224)
            # Getting the type of 'True' (line 224)
            True_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 35), 'True', False)
            # Processing the call keyword arguments (line 224)
            kwargs_280 = {}
            # Getting the type of 'self' (line 224)
            self_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'self', False)
            # Obtaining the member 'set_type_instance' of a type (line 224)
            set_type_instance_278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 12), self_277, 'set_type_instance')
            # Calling set_type_instance(args, kwargs) (line 224)
            set_type_instance_call_result_281 = invoke(stypy.reporting.localization.Localization(__file__, 224, 12), set_type_instance_278, *[True_279], **kwargs_280)
            

            if more_types_in_union_276:
                # SSA join for if statement (line 223)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Attribute (line 227):
        
        # Assigning a Call to a Attribute (line 227):
        
        # Call to list(...): (line 227)
        # Processing the call keyword arguments (line 227)
        kwargs_283 = {}
        # Getting the type of 'list' (line 227)
        list_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 34), 'list', False)
        # Calling list(args, kwargs) (line 227)
        list_call_result_284 = invoke(stypy.reporting.localization.Localization(__file__, 227, 34), list_282, *[], **kwargs_283)
        
        # Getting the type of 'self' (line 227)
        self_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'self')
        # Setting the type of the member 'additional_members' of a type (line 227)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), self_285, 'additional_members', list_call_result_284)
        
        # Assigning a Name to a Attribute (line 230):
        
        # Assigning a Name to a Attribute (line 230):
        # Getting the type of 'None' (line 230)
        None_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 23), 'None')
        # Getting the type of 'self' (line 230)
        self_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'self')
        # Setting the type of the member 'type_of' of a type (line 230)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), self_287, 'type_of', None_286)
        
        # Assigning a Name to a Attribute (line 234):
        
        # Assigning a Name to a Attribute (line 234):
        # Getting the type of 'True' (line 234)
        True_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 31), 'True')
        # Getting the type of 'self' (line 234)
        self_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'self')
        # Setting the type of the member 'known_structure' of a type (line 234)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), self_289, 'known_structure', True_288)
        
        # Getting the type of 'value' (line 236)
        value_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 11), 'value')
        # Getting the type of 'undefined_type_copy' (line 236)
        undefined_type_copy_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 24), 'undefined_type_copy')
        # Obtaining the member 'UndefinedType' of a type (line 236)
        UndefinedType_292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 24), undefined_type_copy_291, 'UndefinedType')
        # Applying the binary operator 'isnot' (line 236)
        result_is_not_293 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 11), 'isnot', value_290, UndefinedType_292)
        
        # Testing if the type of an if condition is none (line 236)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 236, 8), result_is_not_293):
            pass
        else:
            
            # Testing the type of an if condition (line 236)
            if_condition_294 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 8), result_is_not_293)
            # Assigning a type to the variable 'if_condition_294' (line 236)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'if_condition_294', if_condition_294)
            # SSA begins for if statement (line 236)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 237):
            
            # Assigning a Name to a Attribute (line 237):
            # Getting the type of 'value' (line 237)
            value_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 25), 'value')
            # Getting the type of 'self' (line 237)
            self_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'self')
            # Setting the type of the member 'value' of a type (line 237)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), self_296, 'value', value_295)
            
            # Call to set_type_instance(...): (line 238)
            # Processing the call arguments (line 238)
            # Getting the type of 'True' (line 238)
            True_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 35), 'True', False)
            # Processing the call keyword arguments (line 238)
            kwargs_300 = {}
            # Getting the type of 'self' (line 238)
            self_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'self', False)
            # Obtaining the member 'set_type_instance' of a type (line 238)
            set_type_instance_298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 12), self_297, 'set_type_instance')
            # Calling set_type_instance(args, kwargs) (line 238)
            set_type_instance_call_result_301 = invoke(stypy.reporting.localization.Localization(__file__, 238, 12), set_type_instance_298, *[True_299], **kwargs_300)
            
            # SSA join for if statement (line 236)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 273, 4, False)
        # Assigning a type to the variable 'self' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.stypy__repr__')
        TypeInferenceProxy.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        TypeInferenceProxy.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        str_302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, (-1)), 'str', '\n        String representation of this proxy and its contents. Python builtin types have a very concise representation.\n        The method have been stripped down of much of its information gathering code to favor a more concise and clear\n        representation of entities.\n        :return: str\n        ')
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'self' (line 281)
        self_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 22), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 281)
        python_entity_305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 22), self_304, 'python_entity')
        # Getting the type of 'types' (line 281)
        types_306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 42), 'types', False)
        # Obtaining the member 'InstanceType' of a type (line 281)
        InstanceType_307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 42), types_306, 'InstanceType')
        # Processing the call keyword arguments (line 281)
        kwargs_308 = {}
        # Getting the type of 'isinstance' (line 281)
        isinstance_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 281)
        isinstance_call_result_309 = invoke(stypy.reporting.localization.Localization(__file__, 281, 11), isinstance_303, *[python_entity_305, InstanceType_307], **kwargs_308)
        
        
        # Call to isinstance(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'self' (line 281)
        self_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 76), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 281)
        python_entity_312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 76), self_311, 'python_entity')
        # Getting the type of 'types' (line 281)
        types_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 96), 'types', False)
        # Obtaining the member 'ClassType' of a type (line 281)
        ClassType_314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 96), types_313, 'ClassType')
        # Processing the call keyword arguments (line 281)
        kwargs_315 = {}
        # Getting the type of 'isinstance' (line 281)
        isinstance_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 65), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 281)
        isinstance_call_result_316 = invoke(stypy.reporting.localization.Localization(__file__, 281, 65), isinstance_310, *[python_entity_312, ClassType_314], **kwargs_315)
        
        # Applying the binary operator 'or' (line 281)
        result_or_keyword_317 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 11), 'or', isinstance_call_result_309, isinstance_call_result_316)
        
        # Testing if the type of an if condition is none (line 281)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 281, 8), result_or_keyword_317):
            pass
        else:
            
            # Testing the type of an if condition (line 281)
            if_condition_318 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 281, 8), result_or_keyword_317)
            # Assigning a type to the variable 'if_condition_318' (line 281)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'if_condition_318', if_condition_318)
            # SSA begins for if statement (line 281)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'self' (line 282)
            self_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 19), 'self')
            # Obtaining the member 'name' of a type (line 282)
            name_320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 19), self_319, 'name')
            # Assigning a type to the variable 'stypy_return_type' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'stypy_return_type', name_320)
            # SSA join for if statement (line 281)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 285)
        self_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 11), 'self')
        # Obtaining the member 'python_entity' of a type (line 285)
        python_entity_322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 11), self_321, 'python_entity')
        # Getting the type of 'simple_python_types' (line 285)
        simple_python_types_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 33), 'simple_python_types')
        # Applying the binary operator 'in' (line 285)
        result_contains_324 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 11), 'in', python_entity_322, simple_python_types_323)
        
        # Testing if the type of an if condition is none (line 285)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 285, 8), result_contains_324):
            pass
        else:
            
            # Testing the type of an if condition (line 285)
            if_condition_325 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 285, 8), result_contains_324)
            # Assigning a type to the variable 'if_condition_325' (line 285)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'if_condition_325', if_condition_325)
            # SSA begins for if statement (line 285)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to get_python_type(...): (line 286)
            # Processing the call keyword arguments (line 286)
            kwargs_328 = {}
            # Getting the type of 'self' (line 286)
            self_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 19), 'self', False)
            # Obtaining the member 'get_python_type' of a type (line 286)
            get_python_type_327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 19), self_326, 'get_python_type')
            # Calling get_python_type(args, kwargs) (line 286)
            get_python_type_call_result_329 = invoke(stypy.reporting.localization.Localization(__file__, 286, 19), get_python_type_327, *[], **kwargs_328)
            
            # Obtaining the member '__name__' of a type (line 286)
            name___330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 19), get_python_type_call_result_329, '__name__')
            # Assigning a type to the variable 'stypy_return_type' (line 286)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'stypy_return_type', name___330)
            # SSA join for if statement (line 285)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Str to a Name (line 288):
        
        # Assigning a Str to a Name (line 288):
        str_331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 21), 'str', '')
        # Assigning a type to the variable 'parent_str' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'parent_str', str_331)
        
        # Assigning a Str to a Name (line 301):
        
        # Assigning a Str to a Name (line 301):
        str_332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 19), 'str', '')
        # Assigning a type to the variable 'str_mark' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'str_mark', str_332)
        
        # Getting the type of 'self' (line 317)
        self_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 11), 'self')
        # Obtaining the member 'instance' of a type (line 317)
        instance_334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 11), self_333, 'instance')
        # Getting the type of 'None' (line 317)
        None_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 32), 'None')
        # Applying the binary operator 'isnot' (line 317)
        result_is_not_336 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 11), 'isnot', instance_334, None_335)
        
        # Testing if the type of an if condition is none (line 317)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 317, 8), result_is_not_336):
            
            # Assigning a Str to a Name (line 321):
            
            # Assigning a Str to a Name (line 321):
            str_345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 28), 'str', '')
            # Assigning a type to the variable 'instance_type' (line 321)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'instance_type', str_345)
        else:
            
            # Testing the type of an if condition (line 317)
            if_condition_337 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 317, 8), result_is_not_336)
            # Assigning a type to the variable 'if_condition_337' (line 317)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'if_condition_337', if_condition_337)
            # SSA begins for if statement (line 317)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 318):
            
            # Assigning a BinOp to a Name (line 318):
            # Getting the type of 'self' (line 318)
            self_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 28), 'self')
            # Obtaining the member 'instance' of a type (line 318)
            instance_339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 28), self_338, 'instance')
            # Obtaining the member '__class__' of a type (line 318)
            class___340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 28), instance_339, '__class__')
            # Obtaining the member '__name__' of a type (line 318)
            name___341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 28), class___340, '__name__')
            str_342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 63), 'str', ' instance')
            # Applying the binary operator '+' (line 318)
            result_add_343 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 28), '+', name___341, str_342)
            
            # Assigning a type to the variable 'instance_type' (line 318)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'instance_type', result_add_343)
            # Getting the type of 'instance_type' (line 319)
            instance_type_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 19), 'instance_type')
            # Assigning a type to the variable 'stypy_return_type' (line 319)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'stypy_return_type', instance_type_344)
            # SSA branch for the else part of an if statement (line 317)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Str to a Name (line 321):
            
            # Assigning a Str to a Name (line 321):
            str_345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 28), 'str', '')
            # Assigning a type to the variable 'instance_type' (line 321)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'instance_type', str_345)
            # SSA join for if statement (line 317)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to hasattr(...): (line 324)
        # Processing the call arguments (line 324)
        # Getting the type of 'self' (line 324)
        self_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 19), 'self', False)
        # Getting the type of 'self' (line 324)
        self_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 25), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 324)
        contained_elements_property_name_349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 25), self_348, 'contained_elements_property_name')
        # Processing the call keyword arguments (line 324)
        kwargs_350 = {}
        # Getting the type of 'hasattr' (line 324)
        hasattr_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 11), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 324)
        hasattr_call_result_351 = invoke(stypy.reporting.localization.Localization(__file__, 324, 11), hasattr_346, *[self_347, contained_elements_property_name_349], **kwargs_350)
        
        # Testing if the type of an if condition is none (line 324)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 324, 8), hasattr_call_result_351):
            
            # Call to can_store_elements(...): (line 329)
            # Processing the call keyword arguments (line 329)
            kwargs_379 = {}
            # Getting the type of 'self' (line 329)
            self_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 15), 'self', False)
            # Obtaining the member 'can_store_elements' of a type (line 329)
            can_store_elements_378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 15), self_377, 'can_store_elements')
            # Calling can_store_elements(args, kwargs) (line 329)
            can_store_elements_call_result_380 = invoke(stypy.reporting.localization.Localization(__file__, 329, 15), can_store_elements_378, *[], **kwargs_379)
            
            # Testing if the type of an if condition is none (line 329)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 329, 12), can_store_elements_call_result_380):
                
                # Call to can_store_keypairs(...): (line 334)
                # Processing the call keyword arguments (line 334)
                kwargs_396 = {}
                # Getting the type of 'self' (line 334)
                self_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 19), 'self', False)
                # Obtaining the member 'can_store_keypairs' of a type (line 334)
                can_store_keypairs_395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 19), self_394, 'can_store_keypairs')
                # Calling can_store_keypairs(args, kwargs) (line 334)
                can_store_keypairs_call_result_397 = invoke(stypy.reporting.localization.Localization(__file__, 334, 19), can_store_keypairs_395, *[], **kwargs_396)
                
                # Testing if the type of an if condition is none (line 334)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 334, 16), can_store_keypairs_call_result_397):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 334)
                    if_condition_398 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 16), can_store_keypairs_call_result_397)
                    # Assigning a type to the variable 'if_condition_398' (line 334)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'if_condition_398', if_condition_398)
                    # SSA begins for if statement (line 334)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Str to a Name (line 335):
                    
                    # Assigning a Str to a Name (line 335):
                    str_399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 36), 'str', '{}')
                    # Assigning a type to the variable 'contained_str' (line 335)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 20), 'contained_str', str_399)
                    
                    # Call to format(...): (line 336)
                    # Processing the call arguments (line 336)
                    
                    # Call to get_python_type(...): (line 336)
                    # Processing the call keyword arguments (line 336)
                    kwargs_404 = {}
                    # Getting the type of 'self' (line 336)
                    self_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 40), 'self', False)
                    # Obtaining the member 'get_python_type' of a type (line 336)
                    get_python_type_403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 40), self_402, 'get_python_type')
                    # Calling get_python_type(args, kwargs) (line 336)
                    get_python_type_call_result_405 = invoke(stypy.reporting.localization.Localization(__file__, 336, 40), get_python_type_403, *[], **kwargs_404)
                    
                    # Obtaining the member '__name__' of a type (line 336)
                    name___406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 40), get_python_type_call_result_405, '__name__')
                    # Processing the call keyword arguments (line 336)
                    kwargs_407 = {}
                    str_400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 27), 'str', '{0}')
                    # Obtaining the member 'format' of a type (line 336)
                    format_401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 27), str_400, 'format')
                    # Calling format(args, kwargs) (line 336)
                    format_call_result_408 = invoke(stypy.reporting.localization.Localization(__file__, 336, 27), format_401, *[name___406], **kwargs_407)
                    
                    # Getting the type of 'contained_str' (line 337)
                    contained_str_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 29), 'contained_str')
                    # Applying the binary operator '+' (line 336)
                    result_add_410 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 27), '+', format_call_result_408, contained_str_409)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 336)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 20), 'stypy_return_type', result_add_410)
                    # SSA join for if statement (line 334)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 329)
                if_condition_381 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 329, 12), can_store_elements_call_result_380)
                # Assigning a type to the variable 'if_condition_381' (line 329)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'if_condition_381', if_condition_381)
                # SSA begins for if statement (line 329)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Str to a Name (line 330):
                
                # Assigning a Str to a Name (line 330):
                str_382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 32), 'str', '[]')
                # Assigning a type to the variable 'contained_str' (line 330)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'contained_str', str_382)
                
                # Call to format(...): (line 331)
                # Processing the call arguments (line 331)
                
                # Call to get_python_type(...): (line 331)
                # Processing the call keyword arguments (line 331)
                kwargs_387 = {}
                # Getting the type of 'self' (line 331)
                self_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 36), 'self', False)
                # Obtaining the member 'get_python_type' of a type (line 331)
                get_python_type_386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 36), self_385, 'get_python_type')
                # Calling get_python_type(args, kwargs) (line 331)
                get_python_type_call_result_388 = invoke(stypy.reporting.localization.Localization(__file__, 331, 36), get_python_type_386, *[], **kwargs_387)
                
                # Obtaining the member '__name__' of a type (line 331)
                name___389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 36), get_python_type_call_result_388, '__name__')
                # Processing the call keyword arguments (line 331)
                kwargs_390 = {}
                str_383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 23), 'str', '{0}')
                # Obtaining the member 'format' of a type (line 331)
                format_384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 23), str_383, 'format')
                # Calling format(args, kwargs) (line 331)
                format_call_result_391 = invoke(stypy.reporting.localization.Localization(__file__, 331, 23), format_384, *[name___389], **kwargs_390)
                
                # Getting the type of 'contained_str' (line 332)
                contained_str_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 25), 'contained_str')
                # Applying the binary operator '+' (line 331)
                result_add_393 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 23), '+', format_call_result_391, contained_str_392)
                
                # Assigning a type to the variable 'stypy_return_type' (line 331)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'stypy_return_type', result_add_393)
                # SSA branch for the else part of an if statement (line 329)
                module_type_store.open_ssa_branch('else')
                
                # Call to can_store_keypairs(...): (line 334)
                # Processing the call keyword arguments (line 334)
                kwargs_396 = {}
                # Getting the type of 'self' (line 334)
                self_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 19), 'self', False)
                # Obtaining the member 'can_store_keypairs' of a type (line 334)
                can_store_keypairs_395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 19), self_394, 'can_store_keypairs')
                # Calling can_store_keypairs(args, kwargs) (line 334)
                can_store_keypairs_call_result_397 = invoke(stypy.reporting.localization.Localization(__file__, 334, 19), can_store_keypairs_395, *[], **kwargs_396)
                
                # Testing if the type of an if condition is none (line 334)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 334, 16), can_store_keypairs_call_result_397):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 334)
                    if_condition_398 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 16), can_store_keypairs_call_result_397)
                    # Assigning a type to the variable 'if_condition_398' (line 334)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'if_condition_398', if_condition_398)
                    # SSA begins for if statement (line 334)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Str to a Name (line 335):
                    
                    # Assigning a Str to a Name (line 335):
                    str_399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 36), 'str', '{}')
                    # Assigning a type to the variable 'contained_str' (line 335)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 20), 'contained_str', str_399)
                    
                    # Call to format(...): (line 336)
                    # Processing the call arguments (line 336)
                    
                    # Call to get_python_type(...): (line 336)
                    # Processing the call keyword arguments (line 336)
                    kwargs_404 = {}
                    # Getting the type of 'self' (line 336)
                    self_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 40), 'self', False)
                    # Obtaining the member 'get_python_type' of a type (line 336)
                    get_python_type_403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 40), self_402, 'get_python_type')
                    # Calling get_python_type(args, kwargs) (line 336)
                    get_python_type_call_result_405 = invoke(stypy.reporting.localization.Localization(__file__, 336, 40), get_python_type_403, *[], **kwargs_404)
                    
                    # Obtaining the member '__name__' of a type (line 336)
                    name___406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 40), get_python_type_call_result_405, '__name__')
                    # Processing the call keyword arguments (line 336)
                    kwargs_407 = {}
                    str_400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 27), 'str', '{0}')
                    # Obtaining the member 'format' of a type (line 336)
                    format_401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 27), str_400, 'format')
                    # Calling format(args, kwargs) (line 336)
                    format_call_result_408 = invoke(stypy.reporting.localization.Localization(__file__, 336, 27), format_401, *[name___406], **kwargs_407)
                    
                    # Getting the type of 'contained_str' (line 337)
                    contained_str_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 29), 'contained_str')
                    # Applying the binary operator '+' (line 336)
                    result_add_410 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 27), '+', format_call_result_408, contained_str_409)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 336)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 20), 'stypy_return_type', result_add_410)
                    # SSA join for if statement (line 334)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 329)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 324)
            if_condition_352 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 324, 8), hasattr_call_result_351)
            # Assigning a type to the variable 'if_condition_352' (line 324)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'if_condition_352', if_condition_352)
            # SSA begins for if statement (line 324)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 325):
            
            # Assigning a BinOp to a Name (line 325):
            str_353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 28), 'str', '[')
            
            # Call to str(...): (line 325)
            # Processing the call arguments (line 325)
            
            # Call to getattr(...): (line 325)
            # Processing the call arguments (line 325)
            # Getting the type of 'self' (line 325)
            self_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 46), 'self', False)
            # Getting the type of 'self' (line 325)
            self_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 52), 'self', False)
            # Obtaining the member 'contained_elements_property_name' of a type (line 325)
            contained_elements_property_name_358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 52), self_357, 'contained_elements_property_name')
            # Processing the call keyword arguments (line 325)
            kwargs_359 = {}
            # Getting the type of 'getattr' (line 325)
            getattr_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 38), 'getattr', False)
            # Calling getattr(args, kwargs) (line 325)
            getattr_call_result_360 = invoke(stypy.reporting.localization.Localization(__file__, 325, 38), getattr_355, *[self_356, contained_elements_property_name_358], **kwargs_359)
            
            # Processing the call keyword arguments (line 325)
            kwargs_361 = {}
            # Getting the type of 'str' (line 325)
            str_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 34), 'str', False)
            # Calling str(args, kwargs) (line 325)
            str_call_result_362 = invoke(stypy.reporting.localization.Localization(__file__, 325, 34), str_354, *[getattr_call_result_360], **kwargs_361)
            
            # Applying the binary operator '+' (line 325)
            result_add_363 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 28), '+', str_353, str_call_result_362)
            
            str_364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 94), 'str', ']')
            # Applying the binary operator '+' (line 325)
            result_add_365 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 92), '+', result_add_363, str_364)
            
            # Assigning a type to the variable 'contained_str' (line 325)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'contained_str', result_add_365)
            
            # Call to format(...): (line 326)
            # Processing the call arguments (line 326)
            
            # Call to get_python_type(...): (line 326)
            # Processing the call keyword arguments (line 326)
            kwargs_370 = {}
            # Getting the type of 'self' (line 326)
            self_368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 32), 'self', False)
            # Obtaining the member 'get_python_type' of a type (line 326)
            get_python_type_369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 32), self_368, 'get_python_type')
            # Calling get_python_type(args, kwargs) (line 326)
            get_python_type_call_result_371 = invoke(stypy.reporting.localization.Localization(__file__, 326, 32), get_python_type_369, *[], **kwargs_370)
            
            # Obtaining the member '__name__' of a type (line 326)
            name___372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 32), get_python_type_call_result_371, '__name__')
            # Processing the call keyword arguments (line 326)
            kwargs_373 = {}
            str_366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 19), 'str', '{0}')
            # Obtaining the member 'format' of a type (line 326)
            format_367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 19), str_366, 'format')
            # Calling format(args, kwargs) (line 326)
            format_call_result_374 = invoke(stypy.reporting.localization.Localization(__file__, 326, 19), format_367, *[name___372], **kwargs_373)
            
            # Getting the type of 'contained_str' (line 327)
            contained_str_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 21), 'contained_str')
            # Applying the binary operator '+' (line 326)
            result_add_376 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 19), '+', format_call_result_374, contained_str_375)
            
            # Assigning a type to the variable 'stypy_return_type' (line 326)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'stypy_return_type', result_add_376)
            # SSA branch for the else part of an if statement (line 324)
            module_type_store.open_ssa_branch('else')
            
            # Call to can_store_elements(...): (line 329)
            # Processing the call keyword arguments (line 329)
            kwargs_379 = {}
            # Getting the type of 'self' (line 329)
            self_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 15), 'self', False)
            # Obtaining the member 'can_store_elements' of a type (line 329)
            can_store_elements_378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 15), self_377, 'can_store_elements')
            # Calling can_store_elements(args, kwargs) (line 329)
            can_store_elements_call_result_380 = invoke(stypy.reporting.localization.Localization(__file__, 329, 15), can_store_elements_378, *[], **kwargs_379)
            
            # Testing if the type of an if condition is none (line 329)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 329, 12), can_store_elements_call_result_380):
                
                # Call to can_store_keypairs(...): (line 334)
                # Processing the call keyword arguments (line 334)
                kwargs_396 = {}
                # Getting the type of 'self' (line 334)
                self_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 19), 'self', False)
                # Obtaining the member 'can_store_keypairs' of a type (line 334)
                can_store_keypairs_395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 19), self_394, 'can_store_keypairs')
                # Calling can_store_keypairs(args, kwargs) (line 334)
                can_store_keypairs_call_result_397 = invoke(stypy.reporting.localization.Localization(__file__, 334, 19), can_store_keypairs_395, *[], **kwargs_396)
                
                # Testing if the type of an if condition is none (line 334)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 334, 16), can_store_keypairs_call_result_397):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 334)
                    if_condition_398 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 16), can_store_keypairs_call_result_397)
                    # Assigning a type to the variable 'if_condition_398' (line 334)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'if_condition_398', if_condition_398)
                    # SSA begins for if statement (line 334)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Str to a Name (line 335):
                    
                    # Assigning a Str to a Name (line 335):
                    str_399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 36), 'str', '{}')
                    # Assigning a type to the variable 'contained_str' (line 335)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 20), 'contained_str', str_399)
                    
                    # Call to format(...): (line 336)
                    # Processing the call arguments (line 336)
                    
                    # Call to get_python_type(...): (line 336)
                    # Processing the call keyword arguments (line 336)
                    kwargs_404 = {}
                    # Getting the type of 'self' (line 336)
                    self_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 40), 'self', False)
                    # Obtaining the member 'get_python_type' of a type (line 336)
                    get_python_type_403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 40), self_402, 'get_python_type')
                    # Calling get_python_type(args, kwargs) (line 336)
                    get_python_type_call_result_405 = invoke(stypy.reporting.localization.Localization(__file__, 336, 40), get_python_type_403, *[], **kwargs_404)
                    
                    # Obtaining the member '__name__' of a type (line 336)
                    name___406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 40), get_python_type_call_result_405, '__name__')
                    # Processing the call keyword arguments (line 336)
                    kwargs_407 = {}
                    str_400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 27), 'str', '{0}')
                    # Obtaining the member 'format' of a type (line 336)
                    format_401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 27), str_400, 'format')
                    # Calling format(args, kwargs) (line 336)
                    format_call_result_408 = invoke(stypy.reporting.localization.Localization(__file__, 336, 27), format_401, *[name___406], **kwargs_407)
                    
                    # Getting the type of 'contained_str' (line 337)
                    contained_str_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 29), 'contained_str')
                    # Applying the binary operator '+' (line 336)
                    result_add_410 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 27), '+', format_call_result_408, contained_str_409)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 336)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 20), 'stypy_return_type', result_add_410)
                    # SSA join for if statement (line 334)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 329)
                if_condition_381 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 329, 12), can_store_elements_call_result_380)
                # Assigning a type to the variable 'if_condition_381' (line 329)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'if_condition_381', if_condition_381)
                # SSA begins for if statement (line 329)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Str to a Name (line 330):
                
                # Assigning a Str to a Name (line 330):
                str_382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 32), 'str', '[]')
                # Assigning a type to the variable 'contained_str' (line 330)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'contained_str', str_382)
                
                # Call to format(...): (line 331)
                # Processing the call arguments (line 331)
                
                # Call to get_python_type(...): (line 331)
                # Processing the call keyword arguments (line 331)
                kwargs_387 = {}
                # Getting the type of 'self' (line 331)
                self_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 36), 'self', False)
                # Obtaining the member 'get_python_type' of a type (line 331)
                get_python_type_386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 36), self_385, 'get_python_type')
                # Calling get_python_type(args, kwargs) (line 331)
                get_python_type_call_result_388 = invoke(stypy.reporting.localization.Localization(__file__, 331, 36), get_python_type_386, *[], **kwargs_387)
                
                # Obtaining the member '__name__' of a type (line 331)
                name___389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 36), get_python_type_call_result_388, '__name__')
                # Processing the call keyword arguments (line 331)
                kwargs_390 = {}
                str_383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 23), 'str', '{0}')
                # Obtaining the member 'format' of a type (line 331)
                format_384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 23), str_383, 'format')
                # Calling format(args, kwargs) (line 331)
                format_call_result_391 = invoke(stypy.reporting.localization.Localization(__file__, 331, 23), format_384, *[name___389], **kwargs_390)
                
                # Getting the type of 'contained_str' (line 332)
                contained_str_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 25), 'contained_str')
                # Applying the binary operator '+' (line 331)
                result_add_393 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 23), '+', format_call_result_391, contained_str_392)
                
                # Assigning a type to the variable 'stypy_return_type' (line 331)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'stypy_return_type', result_add_393)
                # SSA branch for the else part of an if statement (line 329)
                module_type_store.open_ssa_branch('else')
                
                # Call to can_store_keypairs(...): (line 334)
                # Processing the call keyword arguments (line 334)
                kwargs_396 = {}
                # Getting the type of 'self' (line 334)
                self_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 19), 'self', False)
                # Obtaining the member 'can_store_keypairs' of a type (line 334)
                can_store_keypairs_395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 19), self_394, 'can_store_keypairs')
                # Calling can_store_keypairs(args, kwargs) (line 334)
                can_store_keypairs_call_result_397 = invoke(stypy.reporting.localization.Localization(__file__, 334, 19), can_store_keypairs_395, *[], **kwargs_396)
                
                # Testing if the type of an if condition is none (line 334)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 334, 16), can_store_keypairs_call_result_397):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 334)
                    if_condition_398 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 16), can_store_keypairs_call_result_397)
                    # Assigning a type to the variable 'if_condition_398' (line 334)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'if_condition_398', if_condition_398)
                    # SSA begins for if statement (line 334)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Str to a Name (line 335):
                    
                    # Assigning a Str to a Name (line 335):
                    str_399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 36), 'str', '{}')
                    # Assigning a type to the variable 'contained_str' (line 335)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 20), 'contained_str', str_399)
                    
                    # Call to format(...): (line 336)
                    # Processing the call arguments (line 336)
                    
                    # Call to get_python_type(...): (line 336)
                    # Processing the call keyword arguments (line 336)
                    kwargs_404 = {}
                    # Getting the type of 'self' (line 336)
                    self_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 40), 'self', False)
                    # Obtaining the member 'get_python_type' of a type (line 336)
                    get_python_type_403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 40), self_402, 'get_python_type')
                    # Calling get_python_type(args, kwargs) (line 336)
                    get_python_type_call_result_405 = invoke(stypy.reporting.localization.Localization(__file__, 336, 40), get_python_type_403, *[], **kwargs_404)
                    
                    # Obtaining the member '__name__' of a type (line 336)
                    name___406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 40), get_python_type_call_result_405, '__name__')
                    # Processing the call keyword arguments (line 336)
                    kwargs_407 = {}
                    str_400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 27), 'str', '{0}')
                    # Obtaining the member 'format' of a type (line 336)
                    format_401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 27), str_400, 'format')
                    # Calling format(args, kwargs) (line 336)
                    format_call_result_408 = invoke(stypy.reporting.localization.Localization(__file__, 336, 27), format_401, *[name___406], **kwargs_407)
                    
                    # Getting the type of 'contained_str' (line 337)
                    contained_str_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 29), 'contained_str')
                    # Applying the binary operator '+' (line 336)
                    result_add_410 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 27), '+', format_call_result_408, contained_str_409)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 336)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 20), 'stypy_return_type', result_add_410)
                    # SSA join for if statement (line 334)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 329)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 324)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Str to a Name (line 339):
        
        # Assigning a Str to a Name (line 339):
        str_411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 19), 'str', '')
        # Assigning a type to the variable 'own_name' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'own_name', str_411)
        
        # Call to format(...): (line 346)
        # Processing the call arguments (line 346)
        
        # Call to get_python_type(...): (line 346)
        # Processing the call keyword arguments (line 346)
        kwargs_416 = {}
        # Getting the type of 'self' (line 346)
        self_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 37), 'self', False)
        # Obtaining the member 'get_python_type' of a type (line 346)
        get_python_type_415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 37), self_414, 'get_python_type')
        # Calling get_python_type(args, kwargs) (line 346)
        get_python_type_call_result_417 = invoke(stypy.reporting.localization.Localization(__file__, 346, 37), get_python_type_415, *[], **kwargs_416)
        
        # Obtaining the member '__name__' of a type (line 346)
        name___418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 37), get_python_type_call_result_417, '__name__')
        # Getting the type of 'own_name' (line 346)
        own_name_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 70), 'own_name', False)
        # Getting the type of 'instance_type' (line 346)
        instance_type_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 80), 'instance_type', False)
        # Getting the type of 'str_mark' (line 346)
        str_mark_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 95), 'str_mark', False)
        # Processing the call keyword arguments (line 346)
        kwargs_422 = {}
        str_412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 15), 'str', '{0}{3}{1}{2}')
        # Obtaining the member 'format' of a type (line 346)
        format_413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 15), str_412, 'format')
        # Calling format(args, kwargs) (line 346)
        format_call_result_423 = invoke(stypy.reporting.localization.Localization(__file__, 346, 15), format_413, *[name___418, own_name_419, instance_type_420, str_mark_421], **kwargs_422)
        
        # Getting the type of 'parent_str' (line 346)
        parent_str_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 107), 'parent_str')
        # Applying the binary operator '+' (line 346)
        result_add_425 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 15), '+', format_call_result_423, parent_str_424)
        
        # Assigning a type to the variable 'stypy_return_type' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'stypy_return_type', result_add_425)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 273)
        stypy_return_type_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_426)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_426


    @staticmethod
    @norecursion
    def __equal_property_value(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__equal_property_value'
        module_type_store = module_type_store.open_function_context('__equal_property_value', 353, 4, False)
        
        # Passed parameters checking function
        TypeInferenceProxy.__equal_property_value.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.__equal_property_value.__dict__.__setitem__('stypy_type_of_self', None)
        TypeInferenceProxy.__equal_property_value.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.__equal_property_value.__dict__.__setitem__('stypy_function_name', '__equal_property_value')
        TypeInferenceProxy.__equal_property_value.__dict__.__setitem__('stypy_param_names_list', ['property_name', 'obj1', 'obj2'])
        TypeInferenceProxy.__equal_property_value.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.__equal_property_value.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.__equal_property_value.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.__equal_property_value.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.__equal_property_value.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.__equal_property_value.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, None, module_type_store, '__equal_property_value', ['property_name', 'obj1', 'obj2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__equal_property_value', localization, ['obj1', 'obj2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__equal_property_value(...)' code ##################

        str_427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, (-1)), 'str', '\n        Determines if a property of two objects have the same value.\n        :param property_name: Name of the property to test\n        :param obj1: First object\n        :param obj2: Second object\n        :return: bool (True if same value or both object do not have the property\n        ')
        
        # Evaluating a boolean operation
        
        # Call to hasattr(...): (line 362)
        # Processing the call arguments (line 362)
        # Getting the type of 'obj1' (line 362)
        obj1_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 19), 'obj1', False)
        # Getting the type of 'property_name' (line 362)
        property_name_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 25), 'property_name', False)
        # Processing the call keyword arguments (line 362)
        kwargs_431 = {}
        # Getting the type of 'hasattr' (line 362)
        hasattr_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 11), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 362)
        hasattr_call_result_432 = invoke(stypy.reporting.localization.Localization(__file__, 362, 11), hasattr_428, *[obj1_429, property_name_430], **kwargs_431)
        
        
        # Call to hasattr(...): (line 362)
        # Processing the call arguments (line 362)
        # Getting the type of 'obj2' (line 362)
        obj2_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 52), 'obj2', False)
        # Getting the type of 'property_name' (line 362)
        property_name_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 58), 'property_name', False)
        # Processing the call keyword arguments (line 362)
        kwargs_436 = {}
        # Getting the type of 'hasattr' (line 362)
        hasattr_433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 44), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 362)
        hasattr_call_result_437 = invoke(stypy.reporting.localization.Localization(__file__, 362, 44), hasattr_433, *[obj2_434, property_name_435], **kwargs_436)
        
        # Applying the binary operator 'and' (line 362)
        result_and_keyword_438 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 11), 'and', hasattr_call_result_432, hasattr_call_result_437)
        
        # Testing if the type of an if condition is none (line 362)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 362, 8), result_and_keyword_438):
            pass
        else:
            
            # Testing the type of an if condition (line 362)
            if_condition_439 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 362, 8), result_and_keyword_438)
            # Assigning a type to the variable 'if_condition_439' (line 362)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'if_condition_439', if_condition_439)
            # SSA begins for if statement (line 362)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            
            # Call to getattr(...): (line 363)
            # Processing the call arguments (line 363)
            # Getting the type of 'obj1' (line 363)
            obj1_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 27), 'obj1', False)
            # Getting the type of 'property_name' (line 363)
            property_name_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 33), 'property_name', False)
            # Processing the call keyword arguments (line 363)
            kwargs_443 = {}
            # Getting the type of 'getattr' (line 363)
            getattr_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 19), 'getattr', False)
            # Calling getattr(args, kwargs) (line 363)
            getattr_call_result_444 = invoke(stypy.reporting.localization.Localization(__file__, 363, 19), getattr_440, *[obj1_441, property_name_442], **kwargs_443)
            
            
            # Call to getattr(...): (line 363)
            # Processing the call arguments (line 363)
            # Getting the type of 'obj2' (line 363)
            obj2_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 59), 'obj2', False)
            # Getting the type of 'property_name' (line 363)
            property_name_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 65), 'property_name', False)
            # Processing the call keyword arguments (line 363)
            kwargs_448 = {}
            # Getting the type of 'getattr' (line 363)
            getattr_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 51), 'getattr', False)
            # Calling getattr(args, kwargs) (line 363)
            getattr_call_result_449 = invoke(stypy.reporting.localization.Localization(__file__, 363, 51), getattr_445, *[obj2_446, property_name_447], **kwargs_448)
            
            # Applying the binary operator '==' (line 363)
            result_eq_450 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 19), '==', getattr_call_result_444, getattr_call_result_449)
            
            # Applying the 'not' unary operator (line 363)
            result_not__451 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 15), 'not', result_eq_450)
            
            # Testing if the type of an if condition is none (line 363)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 363, 12), result_not__451):
                pass
            else:
                
                # Testing the type of an if condition (line 363)
                if_condition_452 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 363, 12), result_not__451)
                # Assigning a type to the variable 'if_condition_452' (line 363)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'if_condition_452', if_condition_452)
                # SSA begins for if statement (line 363)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'False' (line 364)
                False_453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 364)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 16), 'stypy_return_type', False_453)
                # SSA join for if statement (line 363)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 362)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'True' (line 366)
        True_454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'stypy_return_type', True_454)
        
        # ################# End of '__equal_property_value(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__equal_property_value' in the type store
        # Getting the type of 'stypy_return_type' (line 353)
        stypy_return_type_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_455)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__equal_property_value'
        return stypy_return_type_455


    @staticmethod
    @norecursion
    def contains_an_undefined_type(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'contains_an_undefined_type'
        module_type_store = module_type_store.open_function_context('contains_an_undefined_type', 368, 4, False)
        
        # Passed parameters checking function
        TypeInferenceProxy.contains_an_undefined_type.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.contains_an_undefined_type.__dict__.__setitem__('stypy_type_of_self', None)
        TypeInferenceProxy.contains_an_undefined_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.contains_an_undefined_type.__dict__.__setitem__('stypy_function_name', 'contains_an_undefined_type')
        TypeInferenceProxy.contains_an_undefined_type.__dict__.__setitem__('stypy_param_names_list', ['value'])
        TypeInferenceProxy.contains_an_undefined_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.contains_an_undefined_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.contains_an_undefined_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.contains_an_undefined_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.contains_an_undefined_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.contains_an_undefined_type.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, 'contains_an_undefined_type', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'contains_an_undefined_type', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'contains_an_undefined_type(...)' code ##################

        str_456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, (-1)), 'str', '\n        Determines if the passed argument is an UndefinedType or contains an UndefinedType\n        :param value: Any Type\n        :return: Tuple (bool, int) (contains an undefined type, the value holds n more types)\n        ')
        
        # Call to isinstance(...): (line 375)
        # Processing the call arguments (line 375)
        # Getting the type of 'value' (line 375)
        value_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 22), 'value', False)
        # Getting the type of 'union_type_copy' (line 375)
        union_type_copy_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 29), 'union_type_copy', False)
        # Obtaining the member 'UnionType' of a type (line 375)
        UnionType_460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 29), union_type_copy_459, 'UnionType')
        # Processing the call keyword arguments (line 375)
        kwargs_461 = {}
        # Getting the type of 'isinstance' (line 375)
        isinstance_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 375)
        isinstance_call_result_462 = invoke(stypy.reporting.localization.Localization(__file__, 375, 11), isinstance_457, *[value_458, UnionType_460], **kwargs_461)
        
        # Testing if the type of an if condition is none (line 375)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 375, 8), isinstance_call_result_462):
            
            # Call to isinstance(...): (line 380)
            # Processing the call arguments (line 380)
            # Getting the type of 'value' (line 380)
            value_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 26), 'value', False)
            # Getting the type of 'undefined_type_copy' (line 380)
            undefined_type_copy_485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 33), 'undefined_type_copy', False)
            # Obtaining the member 'UndefinedType' of a type (line 380)
            UndefinedType_486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 33), undefined_type_copy_485, 'UndefinedType')
            # Processing the call keyword arguments (line 380)
            kwargs_487 = {}
            # Getting the type of 'isinstance' (line 380)
            isinstance_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 380)
            isinstance_call_result_488 = invoke(stypy.reporting.localization.Localization(__file__, 380, 15), isinstance_483, *[value_484, UndefinedType_486], **kwargs_487)
            
            # Testing if the type of an if condition is none (line 380)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 380, 12), isinstance_call_result_488):
                pass
            else:
                
                # Testing the type of an if condition (line 380)
                if_condition_489 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 380, 12), isinstance_call_result_488)
                # Assigning a type to the variable 'if_condition_489' (line 380)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'if_condition_489', if_condition_489)
                # SSA begins for if statement (line 380)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Obtaining an instance of the builtin type 'tuple' (line 381)
                tuple_490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 381)
                # Adding element type (line 381)
                # Getting the type of 'True' (line 381)
                True_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 23), 'True')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), tuple_490, True_491)
                # Adding element type (line 381)
                int_492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 29), 'int')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), tuple_490, int_492)
                
                # Assigning a type to the variable 'stypy_return_type' (line 381)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 16), 'stypy_return_type', tuple_490)
                # SSA join for if statement (line 380)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 375)
            if_condition_463 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 375, 8), isinstance_call_result_462)
            # Assigning a type to the variable 'if_condition_463' (line 375)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'if_condition_463', if_condition_463)
            # SSA begins for if statement (line 375)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'value' (line 376)
            value_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 25), 'value')
            # Obtaining the member 'types' of a type (line 376)
            types_465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 25), value_464, 'types')
            # Assigning a type to the variable 'types_465' (line 376)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'types_465', types_465)
            # Testing if the for loop is going to be iterated (line 376)
            # Testing the type of a for loop iterable (line 376)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 376, 12), types_465)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 376, 12), types_465):
                # Getting the type of the for loop variable (line 376)
                for_loop_var_466 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 376, 12), types_465)
                # Assigning a type to the variable 'type_' (line 376)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'type_', for_loop_var_466)
                # SSA begins for a for statement (line 376)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to isinstance(...): (line 377)
                # Processing the call arguments (line 377)
                # Getting the type of 'type_' (line 377)
                type__468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 30), 'type_', False)
                # Getting the type of 'undefined_type_copy' (line 377)
                undefined_type_copy_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 37), 'undefined_type_copy', False)
                # Obtaining the member 'UndefinedType' of a type (line 377)
                UndefinedType_470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 37), undefined_type_copy_469, 'UndefinedType')
                # Processing the call keyword arguments (line 377)
                kwargs_471 = {}
                # Getting the type of 'isinstance' (line 377)
                isinstance_467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 19), 'isinstance', False)
                # Calling isinstance(args, kwargs) (line 377)
                isinstance_call_result_472 = invoke(stypy.reporting.localization.Localization(__file__, 377, 19), isinstance_467, *[type__468, UndefinedType_470], **kwargs_471)
                
                # Testing if the type of an if condition is none (line 377)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 377, 16), isinstance_call_result_472):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 377)
                    if_condition_473 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 377, 16), isinstance_call_result_472)
                    # Assigning a type to the variable 'if_condition_473' (line 377)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 16), 'if_condition_473', if_condition_473)
                    # SSA begins for if statement (line 377)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 378)
                    tuple_474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 27), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 378)
                    # Adding element type (line 378)
                    # Getting the type of 'True' (line 378)
                    True_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 27), 'True')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 27), tuple_474, True_475)
                    # Adding element type (line 378)
                    
                    # Call to len(...): (line 378)
                    # Processing the call arguments (line 378)
                    # Getting the type of 'value' (line 378)
                    value_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 37), 'value', False)
                    # Obtaining the member 'types' of a type (line 378)
                    types_478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 37), value_477, 'types')
                    # Processing the call keyword arguments (line 378)
                    kwargs_479 = {}
                    # Getting the type of 'len' (line 378)
                    len_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 33), 'len', False)
                    # Calling len(args, kwargs) (line 378)
                    len_call_result_480 = invoke(stypy.reporting.localization.Localization(__file__, 378, 33), len_476, *[types_478], **kwargs_479)
                    
                    int_481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 52), 'int')
                    # Applying the binary operator '-' (line 378)
                    result_sub_482 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 33), '-', len_call_result_480, int_481)
                    
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 27), tuple_474, result_sub_482)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 378)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 20), 'stypy_return_type', tuple_474)
                    # SSA join for if statement (line 377)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA branch for the else part of an if statement (line 375)
            module_type_store.open_ssa_branch('else')
            
            # Call to isinstance(...): (line 380)
            # Processing the call arguments (line 380)
            # Getting the type of 'value' (line 380)
            value_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 26), 'value', False)
            # Getting the type of 'undefined_type_copy' (line 380)
            undefined_type_copy_485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 33), 'undefined_type_copy', False)
            # Obtaining the member 'UndefinedType' of a type (line 380)
            UndefinedType_486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 33), undefined_type_copy_485, 'UndefinedType')
            # Processing the call keyword arguments (line 380)
            kwargs_487 = {}
            # Getting the type of 'isinstance' (line 380)
            isinstance_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 380)
            isinstance_call_result_488 = invoke(stypy.reporting.localization.Localization(__file__, 380, 15), isinstance_483, *[value_484, UndefinedType_486], **kwargs_487)
            
            # Testing if the type of an if condition is none (line 380)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 380, 12), isinstance_call_result_488):
                pass
            else:
                
                # Testing the type of an if condition (line 380)
                if_condition_489 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 380, 12), isinstance_call_result_488)
                # Assigning a type to the variable 'if_condition_489' (line 380)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'if_condition_489', if_condition_489)
                # SSA begins for if statement (line 380)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Obtaining an instance of the builtin type 'tuple' (line 381)
                tuple_490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 381)
                # Adding element type (line 381)
                # Getting the type of 'True' (line 381)
                True_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 23), 'True')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), tuple_490, True_491)
                # Adding element type (line 381)
                int_492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 29), 'int')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), tuple_490, int_492)
                
                # Assigning a type to the variable 'stypy_return_type' (line 381)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 16), 'stypy_return_type', tuple_490)
                # SSA join for if statement (line 380)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 375)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Obtaining an instance of the builtin type 'tuple' (line 383)
        tuple_493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 383)
        # Adding element type (line 383)
        # Getting the type of 'False' (line 383)
        False_494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 15), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 15), tuple_493, False_494)
        # Adding element type (line 383)
        int_495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 15), tuple_493, int_495)
        
        # Assigning a type to the variable 'stypy_return_type' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'stypy_return_type', tuple_493)
        
        # ################# End of 'contains_an_undefined_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'contains_an_undefined_type' in the type store
        # Getting the type of 'stypy_return_type' (line 368)
        stypy_return_type_496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_496)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'contains_an_undefined_type'
        return stypy_return_type_496


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 385, 4, False)
        # Assigning a type to the variable 'self' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.stypy__eq__')
        TypeInferenceProxy.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        TypeInferenceProxy.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.stypy__eq__', ['other'], None, None, defaults, varargs, kwargs)

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

        str_497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, (-1)), 'str', '\n        Type proxy equality. The equality algorithm is represented as follows:\n        - Both objects have to be type inference proxies.\n        - Both objects have to hold the same type of python entity\n        - Both objects held entity name has to be the same (same class, same function, same module, ...), if the\n        proxy is not holding an instance\n        - If the hold entity do not support structural reflection, comparison will be done using the is operator\n        (reference comparison)\n        - If not, comparison by structure is performed (same amount of members, same types for these members)\n\n        :param other: The other object to compare with\n        :return: bool\n        ')
        
        # Getting the type of 'self' (line 399)
        self_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 11), 'self')
        # Getting the type of 'other' (line 399)
        other_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 19), 'other')
        # Applying the binary operator 'is' (line 399)
        result_is__500 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 11), 'is', self_498, other_499)
        
        # Testing if the type of an if condition is none (line 399)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 399, 8), result_is__500):
            pass
        else:
            
            # Testing the type of an if condition (line 399)
            if_condition_501 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 399, 8), result_is__500)
            # Assigning a type to the variable 'if_condition_501' (line 399)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'if_condition_501', if_condition_501)
            # SSA begins for if statement (line 399)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'True' (line 400)
            True_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 400)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'stypy_return_type', True_502)
            # SSA join for if statement (line 399)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        
        # Call to type(...): (line 402)
        # Processing the call arguments (line 402)
        # Getting the type of 'other' (line 402)
        other_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 20), 'other', False)
        # Processing the call keyword arguments (line 402)
        kwargs_505 = {}
        # Getting the type of 'type' (line 402)
        type_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 15), 'type', False)
        # Calling type(args, kwargs) (line 402)
        type_call_result_506 = invoke(stypy.reporting.localization.Localization(__file__, 402, 15), type_503, *[other_504], **kwargs_505)
        
        # Getting the type of 'TypeInferenceProxy' (line 402)
        TypeInferenceProxy_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 30), 'TypeInferenceProxy')
        # Applying the binary operator 'is' (line 402)
        result_is__508 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 15), 'is', type_call_result_506, TypeInferenceProxy_507)
        
        # Applying the 'not' unary operator (line 402)
        result_not__509 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 11), 'not', result_is__508)
        
        # Testing if the type of an if condition is none (line 402)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 402, 8), result_not__509):
            pass
        else:
            
            # Testing the type of an if condition (line 402)
            if_condition_510 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 402, 8), result_not__509)
            # Assigning a type to the variable 'if_condition_510' (line 402)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'if_condition_510', if_condition_510)
            # SSA begins for if statement (line 402)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'False' (line 403)
            False_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 403)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'stypy_return_type', False_511)
            # SSA join for if statement (line 402)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Type idiom detected: calculating its left and rigth part (line 406)
        # Getting the type of 'self' (line 406)
        self_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 20), 'self')
        # Obtaining the member 'python_entity' of a type (line 406)
        python_entity_513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 20), self_512, 'python_entity')
        
        # Call to type(...): (line 406)
        # Processing the call arguments (line 406)
        # Getting the type of 'other' (line 406)
        other_515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 48), 'other', False)
        # Obtaining the member 'python_entity' of a type (line 406)
        python_entity_516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 48), other_515, 'python_entity')
        # Processing the call keyword arguments (line 406)
        kwargs_517 = {}
        # Getting the type of 'type' (line 406)
        type_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 43), 'type', False)
        # Calling type(args, kwargs) (line 406)
        type_call_result_518 = invoke(stypy.reporting.localization.Localization(__file__, 406, 43), type_514, *[python_entity_516], **kwargs_517)
        
        
        (may_be_519, more_types_in_union_520) = may_not_be_type(python_entity_513, type_call_result_518)

        if may_be_519:

            if more_types_in_union_520:
                # Runtime conditional SSA (line 406)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 406)
            self_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'self')
            # Obtaining the member 'python_entity' of a type (line 406)
            python_entity_522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 8), self_521, 'python_entity')
            # Setting the type of the member 'python_entity' of a type (line 406)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 8), self_521, 'python_entity', remove_type_from_union(python_entity_522, type_call_result_518))
            # Getting the type of 'False' (line 407)
            False_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 407)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'stypy_return_type', False_523)

            if more_types_in_union_520:
                # SSA join for if statement (line 406)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'self' (line 410)
        self_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'self')
        # Obtaining the member 'instance' of a type (line 410)
        instance_525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 12), self_524, 'instance')
        # Getting the type of 'None' (line 410)
        None_526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 29), 'None')
        # Applying the binary operator 'is' (line 410)
        result_is__527 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 12), 'is', instance_525, None_526)
        
        
        # Getting the type of 'other' (line 410)
        other_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 38), 'other')
        # Obtaining the member 'instance' of a type (line 410)
        instance_529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 38), other_528, 'instance')
        # Getting the type of 'None' (line 410)
        None_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 56), 'None')
        # Applying the binary operator 'is' (line 410)
        result_is__531 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 38), 'is', instance_529, None_530)
        
        # Applying the binary operator '^' (line 410)
        result_xor_532 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 11), '^', result_is__527, result_is__531)
        
        # Testing if the type of an if condition is none (line 410)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 410, 8), result_xor_532):
            pass
        else:
            
            # Testing the type of an if condition (line 410)
            if_condition_533 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 410, 8), result_xor_532)
            # Assigning a type to the variable 'if_condition_533' (line 410)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'if_condition_533', if_condition_533)
            # SSA begins for if statement (line 410)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'False' (line 411)
            False_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 411)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 12), 'stypy_return_type', False_534)
            # SSA join for if statement (line 410)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 414):
        
        # Assigning a Call to a Name (line 414):
        
        # Call to is_type_instance(...): (line 414)
        # Processing the call keyword arguments (line 414)
        kwargs_537 = {}
        # Getting the type of 'self' (line 414)
        self_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 28), 'self', False)
        # Obtaining the member 'is_type_instance' of a type (line 414)
        is_type_instance_536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 28), self_535, 'is_type_instance')
        # Calling is_type_instance(args, kwargs) (line 414)
        is_type_instance_call_result_538 = invoke(stypy.reporting.localization.Localization(__file__, 414, 28), is_type_instance_536, *[], **kwargs_537)
        
        # Assigning a type to the variable 'self_instantiated' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'self_instantiated', is_type_instance_call_result_538)
        
        # Assigning a Call to a Name (line 415):
        
        # Assigning a Call to a Name (line 415):
        
        # Call to is_type_instance(...): (line 415)
        # Processing the call keyword arguments (line 415)
        kwargs_541 = {}
        # Getting the type of 'other' (line 415)
        other_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 29), 'other', False)
        # Obtaining the member 'is_type_instance' of a type (line 415)
        is_type_instance_540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 29), other_539, 'is_type_instance')
        # Calling is_type_instance(args, kwargs) (line 415)
        is_type_instance_call_result_542 = invoke(stypy.reporting.localization.Localization(__file__, 415, 29), is_type_instance_540, *[], **kwargs_541)
        
        # Assigning a type to the variable 'other_instantiated' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'other_instantiated', is_type_instance_call_result_542)
        
        # Getting the type of 'self_instantiated' (line 416)
        self_instantiated_543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 11), 'self_instantiated')
        # Getting the type of 'other_instantiated' (line 416)
        other_instantiated_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 32), 'other_instantiated')
        # Applying the binary operator '!=' (line 416)
        result_ne_545 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 11), '!=', self_instantiated_543, other_instantiated_544)
        
        # Testing if the type of an if condition is none (line 416)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 416, 8), result_ne_545):
            pass
        else:
            
            # Testing the type of an if condition (line 416)
            if_condition_546 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 416, 8), result_ne_545)
            # Assigning a type to the variable 'if_condition_546' (line 416)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'if_condition_546', if_condition_546)
            # SSA begins for if statement (line 416)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'False' (line 417)
            False_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 417)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'stypy_return_type', False_547)
            # SSA join for if statement (line 416)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Attribute to a Name (line 419):
        
        # Assigning a Attribute to a Name (line 419):
        # Getting the type of 'self' (line 419)
        self_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 22), 'self')
        # Obtaining the member 'python_entity' of a type (line 419)
        python_entity_549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 22), self_548, 'python_entity')
        # Assigning a type to the variable 'self_entity' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'self_entity', python_entity_549)
        
        # Assigning a Attribute to a Name (line 420):
        
        # Assigning a Attribute to a Name (line 420):
        # Getting the type of 'other' (line 420)
        other_550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 23), 'other')
        # Obtaining the member 'python_entity' of a type (line 420)
        python_entity_551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 23), other_550, 'python_entity')
        # Assigning a type to the variable 'other_entity' (line 420)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'other_entity', python_entity_551)
        
        # Getting the type of 'Type' (line 423)
        Type_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 25), 'Type')
        # Obtaining the member 'special_properties_for_equality' of a type (line 423)
        special_properties_for_equality_553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 25), Type_552, 'special_properties_for_equality')
        # Assigning a type to the variable 'special_properties_for_equality_553' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'special_properties_for_equality_553', special_properties_for_equality_553)
        # Testing if the for loop is going to be iterated (line 423)
        # Testing the type of a for loop iterable (line 423)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 423, 8), special_properties_for_equality_553)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 423, 8), special_properties_for_equality_553):
            # Getting the type of the for loop variable (line 423)
            for_loop_var_554 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 423, 8), special_properties_for_equality_553)
            # Assigning a type to the variable 'prop_name' (line 423)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'prop_name', for_loop_var_554)
            # SSA begins for a for statement (line 423)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to __equal_property_value(...): (line 424)
            # Processing the call arguments (line 424)
            # Getting the type of 'prop_name' (line 424)
            prop_name_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 47), 'prop_name', False)
            # Getting the type of 'self_entity' (line 424)
            self_entity_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 58), 'self_entity', False)
            # Getting the type of 'other_entity' (line 424)
            other_entity_559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 71), 'other_entity', False)
            # Processing the call keyword arguments (line 424)
            kwargs_560 = {}
            # Getting the type of 'self' (line 424)
            self_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 19), 'self', False)
            # Obtaining the member '__equal_property_value' of a type (line 424)
            equal_property_value_556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 19), self_555, '__equal_property_value')
            # Calling __equal_property_value(args, kwargs) (line 424)
            equal_property_value_call_result_561 = invoke(stypy.reporting.localization.Localization(__file__, 424, 19), equal_property_value_556, *[prop_name_557, self_entity_558, other_entity_559], **kwargs_560)
            
            # Applying the 'not' unary operator (line 424)
            result_not__562 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 15), 'not', equal_property_value_call_result_561)
            
            # Testing if the type of an if condition is none (line 424)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 424, 12), result_not__562):
                pass
            else:
                
                # Testing the type of an if condition (line 424)
                if_condition_563 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 424, 12), result_not__562)
                # Assigning a type to the variable 'if_condition_563' (line 424)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'if_condition_563', if_condition_563)
                # SSA begins for if statement (line 424)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'False' (line 425)
                False_564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 425)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 16), 'stypy_return_type', False_564)
                # SSA join for if statement (line 424)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to __equal_property_value(...): (line 428)
        # Processing the call arguments (line 428)
        # Getting the type of 'TypeInferenceProxy' (line 428)
        TypeInferenceProxy_567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 43), 'TypeInferenceProxy', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 428)
        contained_elements_property_name_568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 43), TypeInferenceProxy_567, 'contained_elements_property_name')
        # Getting the type of 'self_entity' (line 428)
        self_entity_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 96), 'self_entity', False)
        # Getting the type of 'other_entity' (line 429)
        other_entity_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 43), 'other_entity', False)
        # Processing the call keyword arguments (line 428)
        kwargs_571 = {}
        # Getting the type of 'self' (line 428)
        self_565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 15), 'self', False)
        # Obtaining the member '__equal_property_value' of a type (line 428)
        equal_property_value_566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 15), self_565, '__equal_property_value')
        # Calling __equal_property_value(args, kwargs) (line 428)
        equal_property_value_call_result_572 = invoke(stypy.reporting.localization.Localization(__file__, 428, 15), equal_property_value_566, *[contained_elements_property_name_568, self_entity_569, other_entity_570], **kwargs_571)
        
        # Applying the 'not' unary operator (line 428)
        result_not__573 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 11), 'not', equal_property_value_call_result_572)
        
        # Testing if the type of an if condition is none (line 428)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 428, 8), result_not__573):
            pass
        else:
            
            # Testing the type of an if condition (line 428)
            if_condition_574 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 428, 8), result_not__573)
            # Assigning a type to the variable 'if_condition_574' (line 428)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'if_condition_574', if_condition_574)
            # SSA begins for if statement (line 428)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'False' (line 430)
            False_575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 430)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'stypy_return_type', False_575)
            # SSA join for if statement (line 428)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Type idiom detected: calculating its left and rigth part (line 433)
        # Getting the type of 'self' (line 433)
        self_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 11), 'self')
        # Obtaining the member 'instance' of a type (line 433)
        instance_577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 11), self_576, 'instance')
        # Getting the type of 'None' (line 433)
        None_578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 28), 'None')
        
        (may_be_579, more_types_in_union_580) = may_be_none(instance_577, None_578)

        if may_be_579:

            if more_types_in_union_580:
                # Runtime conditional SSA (line 433)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Evaluating a boolean operation
            
            # Call to supports_structural_reflection(...): (line 437)
            # Processing the call keyword arguments (line 437)
            kwargs_583 = {}
            # Getting the type of 'self' (line 437)
            self_581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 15), 'self', False)
            # Obtaining the member 'supports_structural_reflection' of a type (line 437)
            supports_structural_reflection_582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 15), self_581, 'supports_structural_reflection')
            # Calling supports_structural_reflection(args, kwargs) (line 437)
            supports_structural_reflection_call_result_584 = invoke(stypy.reporting.localization.Localization(__file__, 437, 15), supports_structural_reflection_582, *[], **kwargs_583)
            
            
            # Call to supports_structural_reflection(...): (line 437)
            # Processing the call keyword arguments (line 437)
            kwargs_587 = {}
            # Getting the type of 'other' (line 437)
            other_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 57), 'other', False)
            # Obtaining the member 'supports_structural_reflection' of a type (line 437)
            supports_structural_reflection_586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 57), other_585, 'supports_structural_reflection')
            # Calling supports_structural_reflection(args, kwargs) (line 437)
            supports_structural_reflection_call_result_588 = invoke(stypy.reporting.localization.Localization(__file__, 437, 57), supports_structural_reflection_586, *[], **kwargs_587)
            
            # Applying the binary operator 'and' (line 437)
            result_and_keyword_589 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 15), 'and', supports_structural_reflection_call_result_584, supports_structural_reflection_call_result_588)
            
            # Testing if the type of an if condition is none (line 437)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 437, 12), result_and_keyword_589):
                
                
                # Call to type(...): (line 442)
                # Processing the call arguments (line 442)
                # Getting the type of 'self_entity' (line 442)
                self_entity_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 28), 'self_entity', False)
                # Processing the call keyword arguments (line 442)
                kwargs_600 = {}
                # Getting the type of 'type' (line 442)
                type_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 23), 'type', False)
                # Calling type(args, kwargs) (line 442)
                type_call_result_601 = invoke(stypy.reporting.localization.Localization(__file__, 442, 23), type_598, *[self_entity_599], **kwargs_600)
                
                
                # Call to type(...): (line 442)
                # Processing the call arguments (line 442)
                # Getting the type of 'other_entity' (line 442)
                other_entity_603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 49), 'other_entity', False)
                # Processing the call keyword arguments (line 442)
                kwargs_604 = {}
                # Getting the type of 'type' (line 442)
                type_602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 44), 'type', False)
                # Calling type(args, kwargs) (line 442)
                type_call_result_605 = invoke(stypy.reporting.localization.Localization(__file__, 442, 44), type_602, *[other_entity_603], **kwargs_604)
                
                # Applying the binary operator 'is' (line 442)
                result_is__606 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 23), 'is', type_call_result_601, type_call_result_605)
                
                # Assigning a type to the variable 'stypy_return_type' (line 442)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 16), 'stypy_return_type', result_is__606)
            else:
                
                # Testing the type of an if condition (line 437)
                if_condition_590 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 437, 12), result_and_keyword_589)
                # Assigning a type to the variable 'if_condition_590' (line 437)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'if_condition_590', if_condition_590)
                # SSA begins for if statement (line 437)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to structural_equivalence(...): (line 439)
                # Processing the call arguments (line 439)
                # Getting the type of 'self_entity' (line 439)
                self_entity_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 68), 'self_entity', False)
                # Getting the type of 'other_entity' (line 439)
                other_entity_594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 81), 'other_entity', False)
                # Getting the type of 'True' (line 439)
                True_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 95), 'True', False)
                # Processing the call keyword arguments (line 439)
                kwargs_596 = {}
                # Getting the type of 'type_equivalence_copy' (line 439)
                type_equivalence_copy_591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 23), 'type_equivalence_copy', False)
                # Obtaining the member 'structural_equivalence' of a type (line 439)
                structural_equivalence_592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 23), type_equivalence_copy_591, 'structural_equivalence')
                # Calling structural_equivalence(args, kwargs) (line 439)
                structural_equivalence_call_result_597 = invoke(stypy.reporting.localization.Localization(__file__, 439, 23), structural_equivalence_592, *[self_entity_593, other_entity_594, True_595], **kwargs_596)
                
                # Assigning a type to the variable 'stypy_return_type' (line 439)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 16), 'stypy_return_type', structural_equivalence_call_result_597)
                # SSA branch for the else part of an if statement (line 437)
                module_type_store.open_ssa_branch('else')
                
                
                # Call to type(...): (line 442)
                # Processing the call arguments (line 442)
                # Getting the type of 'self_entity' (line 442)
                self_entity_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 28), 'self_entity', False)
                # Processing the call keyword arguments (line 442)
                kwargs_600 = {}
                # Getting the type of 'type' (line 442)
                type_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 23), 'type', False)
                # Calling type(args, kwargs) (line 442)
                type_call_result_601 = invoke(stypy.reporting.localization.Localization(__file__, 442, 23), type_598, *[self_entity_599], **kwargs_600)
                
                
                # Call to type(...): (line 442)
                # Processing the call arguments (line 442)
                # Getting the type of 'other_entity' (line 442)
                other_entity_603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 49), 'other_entity', False)
                # Processing the call keyword arguments (line 442)
                kwargs_604 = {}
                # Getting the type of 'type' (line 442)
                type_602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 44), 'type', False)
                # Calling type(args, kwargs) (line 442)
                type_call_result_605 = invoke(stypy.reporting.localization.Localization(__file__, 442, 44), type_602, *[other_entity_603], **kwargs_604)
                
                # Applying the binary operator 'is' (line 442)
                result_is__606 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 23), 'is', type_call_result_601, type_call_result_605)
                
                # Assigning a type to the variable 'stypy_return_type' (line 442)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 16), 'stypy_return_type', result_is__606)
                # SSA join for if statement (line 437)
                module_type_store = module_type_store.join_ssa_context()
                


            if more_types_in_union_580:
                # Runtime conditional SSA for else branch (line 433)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_579) or more_types_in_union_580):
            
            # Evaluating a boolean operation
            
            # Call to supports_structural_reflection(...): (line 447)
            # Processing the call keyword arguments (line 447)
            kwargs_609 = {}
            # Getting the type of 'self' (line 447)
            self_607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 15), 'self', False)
            # Obtaining the member 'supports_structural_reflection' of a type (line 447)
            supports_structural_reflection_608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 15), self_607, 'supports_structural_reflection')
            # Calling supports_structural_reflection(args, kwargs) (line 447)
            supports_structural_reflection_call_result_610 = invoke(stypy.reporting.localization.Localization(__file__, 447, 15), supports_structural_reflection_608, *[], **kwargs_609)
            
            
            # Call to supports_structural_reflection(...): (line 447)
            # Processing the call keyword arguments (line 447)
            kwargs_613 = {}
            # Getting the type of 'other' (line 447)
            other_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 57), 'other', False)
            # Obtaining the member 'supports_structural_reflection' of a type (line 447)
            supports_structural_reflection_612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 57), other_611, 'supports_structural_reflection')
            # Calling supports_structural_reflection(args, kwargs) (line 447)
            supports_structural_reflection_call_result_614 = invoke(stypy.reporting.localization.Localization(__file__, 447, 57), supports_structural_reflection_612, *[], **kwargs_613)
            
            # Applying the binary operator 'and' (line 447)
            result_and_keyword_615 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 15), 'and', supports_structural_reflection_call_result_610, supports_structural_reflection_call_result_614)
            
            # Testing if the type of an if condition is none (line 447)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 447, 12), result_and_keyword_615):
                
                # Assigning a Compare to a Name (line 460):
                
                # Assigning a Compare to a Name (line 460):
                
                
                # Call to type(...): (line 460)
                # Processing the call arguments (line 460)
                # Getting the type of 'self' (line 460)
                self_640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 34), 'self', False)
                # Obtaining the member 'python_entity' of a type (line 460)
                python_entity_641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 34), self_640, 'python_entity')
                # Processing the call keyword arguments (line 460)
                kwargs_642 = {}
                # Getting the type of 'type' (line 460)
                type_639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 29), 'type', False)
                # Calling type(args, kwargs) (line 460)
                type_call_result_643 = invoke(stypy.reporting.localization.Localization(__file__, 460, 29), type_639, *[python_entity_641], **kwargs_642)
                
                
                # Call to type(...): (line 460)
                # Processing the call arguments (line 460)
                # Getting the type of 'other' (line 460)
                other_645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 62), 'other', False)
                # Obtaining the member 'python_entity' of a type (line 460)
                python_entity_646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 62), other_645, 'python_entity')
                # Processing the call keyword arguments (line 460)
                kwargs_647 = {}
                # Getting the type of 'type' (line 460)
                type_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 57), 'type', False)
                # Calling type(args, kwargs) (line 460)
                type_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 460, 57), type_644, *[python_entity_646], **kwargs_647)
                
                # Applying the binary operator 'is' (line 460)
                result_is__649 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 29), 'is', type_call_result_643, type_call_result_648)
                
                # Assigning a type to the variable 'equivalent' (line 460)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 16), 'equivalent', result_is__649)
                
                # Getting the type of 'equivalent' (line 461)
                equivalent_650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 23), 'equivalent')
                # Applying the 'not' unary operator (line 461)
                result_not__651 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 19), 'not', equivalent_650)
                
                # Testing if the type of an if condition is none (line 461)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 461, 16), result_not__651):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 461)
                    if_condition_652 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 461, 16), result_not__651)
                    # Assigning a type to the variable 'if_condition_652' (line 461)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 16), 'if_condition_652', if_condition_652)
                    # SSA begins for if statement (line 461)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'False' (line 462)
                    False_653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 27), 'False')
                    # Assigning a type to the variable 'stypy_return_type' (line 462)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 20), 'stypy_return_type', False_653)
                    # SSA join for if statement (line 461)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Attribute to a Name (line 465):
                
                # Assigning a Attribute to a Name (line 465):
                # Getting the type of 'self' (line 465)
                self_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 30), 'self')
                # Obtaining the member 'instance' of a type (line 465)
                instance_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 30), self_654, 'instance')
                # Assigning a type to the variable 'self_entity' (line 465)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 16), 'self_entity', instance_655)
                
                # Assigning a Attribute to a Name (line 466):
                
                # Assigning a Attribute to a Name (line 466):
                # Getting the type of 'other' (line 466)
                other_656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 31), 'other')
                # Obtaining the member 'instance' of a type (line 466)
                instance_657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 31), other_656, 'instance')
                # Assigning a type to the variable 'other_entity' (line 466)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 16), 'other_entity', instance_657)
                
                # Getting the type of 'self_entity' (line 467)
                self_entity_658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 23), 'self_entity')
                # Getting the type of 'other_entity' (line 467)
                other_entity_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 38), 'other_entity')
                # Applying the binary operator 'is' (line 467)
                result_is__660 = python_operator(stypy.reporting.localization.Localization(__file__, 467, 23), 'is', self_entity_658, other_entity_659)
                
                # Assigning a type to the variable 'stypy_return_type' (line 467)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 16), 'stypy_return_type', result_is__660)
            else:
                
                # Testing the type of an if condition (line 447)
                if_condition_616 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 447, 12), result_and_keyword_615)
                # Assigning a type to the variable 'if_condition_616' (line 447)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'if_condition_616', if_condition_616)
                # SSA begins for if statement (line 447)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 449):
                
                # Assigning a Call to a Name (line 449):
                
                # Call to structural_equivalence(...): (line 449)
                # Processing the call arguments (line 449)
                # Getting the type of 'self_entity' (line 449)
                self_entity_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 74), 'self_entity', False)
                # Getting the type of 'other_entity' (line 449)
                other_entity_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 87), 'other_entity', False)
                # Getting the type of 'True' (line 449)
                True_621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 101), 'True', False)
                # Processing the call keyword arguments (line 449)
                kwargs_622 = {}
                # Getting the type of 'type_equivalence_copy' (line 449)
                type_equivalence_copy_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 29), 'type_equivalence_copy', False)
                # Obtaining the member 'structural_equivalence' of a type (line 449)
                structural_equivalence_618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 29), type_equivalence_copy_617, 'structural_equivalence')
                # Calling structural_equivalence(args, kwargs) (line 449)
                structural_equivalence_call_result_623 = invoke(stypy.reporting.localization.Localization(__file__, 449, 29), structural_equivalence_618, *[self_entity_619, other_entity_620, True_621], **kwargs_622)
                
                # Assigning a type to the variable 'equivalent' (line 449)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 16), 'equivalent', structural_equivalence_call_result_623)
                
                # Getting the type of 'equivalent' (line 451)
                equivalent_624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 23), 'equivalent')
                # Applying the 'not' unary operator (line 451)
                result_not__625 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 19), 'not', equivalent_624)
                
                # Testing if the type of an if condition is none (line 451)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 451, 16), result_not__625):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 451)
                    if_condition_626 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 451, 16), result_not__625)
                    # Assigning a type to the variable 'if_condition_626' (line 451)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 16), 'if_condition_626', if_condition_626)
                    # SSA begins for if statement (line 451)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'False' (line 452)
                    False_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 27), 'False')
                    # Assigning a type to the variable 'stypy_return_type' (line 452)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 20), 'stypy_return_type', False_627)
                    # SSA join for if statement (line 451)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Attribute to a Name (line 455):
                
                # Assigning a Attribute to a Name (line 455):
                # Getting the type of 'self' (line 455)
                self_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 30), 'self')
                # Obtaining the member 'instance' of a type (line 455)
                instance_629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 30), self_628, 'instance')
                # Assigning a type to the variable 'self_entity' (line 455)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 16), 'self_entity', instance_629)
                
                # Assigning a Attribute to a Name (line 456):
                
                # Assigning a Attribute to a Name (line 456):
                # Getting the type of 'other' (line 456)
                other_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 31), 'other')
                # Obtaining the member 'instance' of a type (line 456)
                instance_631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 31), other_630, 'instance')
                # Assigning a type to the variable 'other_entity' (line 456)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'other_entity', instance_631)
                
                # Call to structural_equivalence(...): (line 457)
                # Processing the call arguments (line 457)
                # Getting the type of 'self_entity' (line 457)
                self_entity_634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 68), 'self_entity', False)
                # Getting the type of 'other_entity' (line 457)
                other_entity_635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 81), 'other_entity', False)
                # Getting the type of 'False' (line 457)
                False_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 95), 'False', False)
                # Processing the call keyword arguments (line 457)
                kwargs_637 = {}
                # Getting the type of 'type_equivalence_copy' (line 457)
                type_equivalence_copy_632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 23), 'type_equivalence_copy', False)
                # Obtaining the member 'structural_equivalence' of a type (line 457)
                structural_equivalence_633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 23), type_equivalence_copy_632, 'structural_equivalence')
                # Calling structural_equivalence(args, kwargs) (line 457)
                structural_equivalence_call_result_638 = invoke(stypy.reporting.localization.Localization(__file__, 457, 23), structural_equivalence_633, *[self_entity_634, other_entity_635, False_636], **kwargs_637)
                
                # Assigning a type to the variable 'stypy_return_type' (line 457)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 16), 'stypy_return_type', structural_equivalence_call_result_638)
                # SSA branch for the else part of an if statement (line 447)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Compare to a Name (line 460):
                
                # Assigning a Compare to a Name (line 460):
                
                
                # Call to type(...): (line 460)
                # Processing the call arguments (line 460)
                # Getting the type of 'self' (line 460)
                self_640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 34), 'self', False)
                # Obtaining the member 'python_entity' of a type (line 460)
                python_entity_641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 34), self_640, 'python_entity')
                # Processing the call keyword arguments (line 460)
                kwargs_642 = {}
                # Getting the type of 'type' (line 460)
                type_639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 29), 'type', False)
                # Calling type(args, kwargs) (line 460)
                type_call_result_643 = invoke(stypy.reporting.localization.Localization(__file__, 460, 29), type_639, *[python_entity_641], **kwargs_642)
                
                
                # Call to type(...): (line 460)
                # Processing the call arguments (line 460)
                # Getting the type of 'other' (line 460)
                other_645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 62), 'other', False)
                # Obtaining the member 'python_entity' of a type (line 460)
                python_entity_646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 62), other_645, 'python_entity')
                # Processing the call keyword arguments (line 460)
                kwargs_647 = {}
                # Getting the type of 'type' (line 460)
                type_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 57), 'type', False)
                # Calling type(args, kwargs) (line 460)
                type_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 460, 57), type_644, *[python_entity_646], **kwargs_647)
                
                # Applying the binary operator 'is' (line 460)
                result_is__649 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 29), 'is', type_call_result_643, type_call_result_648)
                
                # Assigning a type to the variable 'equivalent' (line 460)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 16), 'equivalent', result_is__649)
                
                # Getting the type of 'equivalent' (line 461)
                equivalent_650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 23), 'equivalent')
                # Applying the 'not' unary operator (line 461)
                result_not__651 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 19), 'not', equivalent_650)
                
                # Testing if the type of an if condition is none (line 461)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 461, 16), result_not__651):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 461)
                    if_condition_652 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 461, 16), result_not__651)
                    # Assigning a type to the variable 'if_condition_652' (line 461)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 16), 'if_condition_652', if_condition_652)
                    # SSA begins for if statement (line 461)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'False' (line 462)
                    False_653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 27), 'False')
                    # Assigning a type to the variable 'stypy_return_type' (line 462)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 20), 'stypy_return_type', False_653)
                    # SSA join for if statement (line 461)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Attribute to a Name (line 465):
                
                # Assigning a Attribute to a Name (line 465):
                # Getting the type of 'self' (line 465)
                self_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 30), 'self')
                # Obtaining the member 'instance' of a type (line 465)
                instance_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 30), self_654, 'instance')
                # Assigning a type to the variable 'self_entity' (line 465)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 16), 'self_entity', instance_655)
                
                # Assigning a Attribute to a Name (line 466):
                
                # Assigning a Attribute to a Name (line 466):
                # Getting the type of 'other' (line 466)
                other_656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 31), 'other')
                # Obtaining the member 'instance' of a type (line 466)
                instance_657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 31), other_656, 'instance')
                # Assigning a type to the variable 'other_entity' (line 466)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 16), 'other_entity', instance_657)
                
                # Getting the type of 'self_entity' (line 467)
                self_entity_658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 23), 'self_entity')
                # Getting the type of 'other_entity' (line 467)
                other_entity_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 38), 'other_entity')
                # Applying the binary operator 'is' (line 467)
                result_is__660 = python_operator(stypy.reporting.localization.Localization(__file__, 467, 23), 'is', self_entity_658, other_entity_659)
                
                # Assigning a type to the variable 'stypy_return_type' (line 467)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 16), 'stypy_return_type', result_is__660)
                # SSA join for if statement (line 447)
                module_type_store = module_type_store.join_ssa_context()
                


            if (may_be_579 and more_types_in_union_580):
                # SSA join for if statement (line 433)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 385)
        stypy_return_type_661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_661)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_661


    @staticmethod
    @norecursion
    def instance(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 472)
        None_662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 37), 'None')
        # Getting the type of 'None' (line 472)
        None_663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 50), 'None')
        # Getting the type of 'None' (line 472)
        None_664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 65), 'None')
        # Getting the type of 'undefined_type_copy' (line 472)
        undefined_type_copy_665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 77), 'undefined_type_copy')
        # Obtaining the member 'UndefinedType' of a type (line 472)
        UndefinedType_666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 77), undefined_type_copy_665, 'UndefinedType')
        defaults = [None_662, None_663, None_664, UndefinedType_666]
        # Create a new context for function 'instance'
        module_type_store = module_type_store.open_function_context('instance', 471, 4, False)
        
        # Passed parameters checking function
        TypeInferenceProxy.instance.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.instance.__dict__.__setitem__('stypy_type_of_self', None)
        TypeInferenceProxy.instance.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.instance.__dict__.__setitem__('stypy_function_name', 'instance')
        TypeInferenceProxy.instance.__dict__.__setitem__('stypy_param_names_list', ['python_entity', 'name', 'parent', 'instance', 'value'])
        TypeInferenceProxy.instance.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.instance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.instance.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.instance.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.instance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.instance.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, None, module_type_store, 'instance', ['python_entity', 'name', 'parent', 'instance', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'instance', localization, ['name', 'parent', 'instance', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'instance(...)' code ##################

        str_667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, (-1)), 'str', '\n        Creates a new Type inference proxy for the passed python entity (function, module, class, ...). This is the\n        preferred way to create proxy instances, as this method implement a memoization optimization.\n\n        :param python_entity: Represented python entity.\n        :param name: Name of the represented Python entity. If nothing is provided, the Python entity __name__ property\n        value is used instead. Instances have an special name indicating that this entity holds an instance of a class.\n        :param parent: Parent proxy object. If nothing is provided, the parent proxy is autocalculated, if possible.\n        :param instance: Instance of the represented class. If this proxy holds a class, it is possible that instead\n        of representing the class is representing a particular class instance. This is important to properly model\n        instance intercession, as altering the structure of single instances is possible.\n        ')
        
        # Call to isinstance(...): (line 493)
        # Processing the call arguments (line 493)
        # Getting the type of 'python_entity' (line 493)
        python_entity_669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 22), 'python_entity', False)
        # Getting the type of 'Type' (line 493)
        Type_670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 37), 'Type', False)
        # Processing the call keyword arguments (line 493)
        kwargs_671 = {}
        # Getting the type of 'isinstance' (line 493)
        isinstance_668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 493)
        isinstance_call_result_672 = invoke(stypy.reporting.localization.Localization(__file__, 493, 11), isinstance_668, *[python_entity_669, Type_670], **kwargs_671)
        
        # Testing if the type of an if condition is none (line 493)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 493, 8), isinstance_call_result_672):
            pass
        else:
            
            # Testing the type of an if condition (line 493)
            if_condition_673 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 493, 8), isinstance_call_result_672)
            # Assigning a type to the variable 'if_condition_673' (line 493)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'if_condition_673', if_condition_673)
            # SSA begins for if statement (line 493)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'python_entity' (line 494)
            python_entity_674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 19), 'python_entity')
            # Assigning a type to the variable 'stypy_return_type' (line 494)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'stypy_return_type', python_entity_674)
            # SSA join for if statement (line 493)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to TypeInferenceProxy(...): (line 496)
        # Processing the call arguments (line 496)
        # Getting the type of 'python_entity' (line 496)
        python_entity_676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 34), 'python_entity', False)
        # Getting the type of 'name' (line 496)
        name_677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 49), 'name', False)
        # Getting the type of 'parent' (line 496)
        parent_678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 55), 'parent', False)
        # Getting the type of 'instance' (line 496)
        instance_679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 63), 'instance', False)
        # Getting the type of 'value' (line 496)
        value_680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 73), 'value', False)
        # Processing the call keyword arguments (line 496)
        kwargs_681 = {}
        # Getting the type of 'TypeInferenceProxy' (line 496)
        TypeInferenceProxy_675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 15), 'TypeInferenceProxy', False)
        # Calling TypeInferenceProxy(args, kwargs) (line 496)
        TypeInferenceProxy_call_result_682 = invoke(stypy.reporting.localization.Localization(__file__, 496, 15), TypeInferenceProxy_675, *[python_entity_676, name_677, parent_678, instance_679, value_680], **kwargs_681)
        
        # Assigning a type to the variable 'stypy_return_type' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'stypy_return_type', TypeInferenceProxy_call_result_682)
        
        # ################# End of 'instance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'instance' in the type store
        # Getting the type of 'stypy_return_type' (line 471)
        stypy_return_type_683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_683)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'instance'
        return stypy_return_type_683


    @norecursion
    def get_python_entity(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_python_entity'
        module_type_store = module_type_store.open_function_context('get_python_entity', 500, 4, False)
        # Assigning a type to the variable 'self' (line 501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.get_python_entity.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.get_python_entity.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.get_python_entity.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.get_python_entity.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.get_python_entity')
        TypeInferenceProxy.get_python_entity.__dict__.__setitem__('stypy_param_names_list', [])
        TypeInferenceProxy.get_python_entity.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.get_python_entity.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.get_python_entity.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.get_python_entity.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.get_python_entity.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.get_python_entity.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.get_python_entity', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_python_entity', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_python_entity(...)' code ##################

        str_684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, (-1)), 'str', '\n        Returns the Python entity (function, method, class, object, module...) represented by this Type.\n        :return: A Python entity\n        ')
        # Getting the type of 'self' (line 505)
        self_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 15), 'self')
        # Obtaining the member 'python_entity' of a type (line 505)
        python_entity_686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 15), self_685, 'python_entity')
        # Assigning a type to the variable 'stypy_return_type' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'stypy_return_type', python_entity_686)
        
        # ################# End of 'get_python_entity(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_python_entity' in the type store
        # Getting the type of 'stypy_return_type' (line 500)
        stypy_return_type_687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_687)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_python_entity'
        return stypy_return_type_687


    @norecursion
    def get_python_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_python_type'
        module_type_store = module_type_store.open_function_context('get_python_type', 507, 4, False)
        # Assigning a type to the variable 'self' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.get_python_type.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.get_python_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.get_python_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.get_python_type.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.get_python_type')
        TypeInferenceProxy.get_python_type.__dict__.__setitem__('stypy_param_names_list', [])
        TypeInferenceProxy.get_python_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.get_python_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.get_python_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.get_python_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.get_python_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.get_python_type.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.get_python_type', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_python_type', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_python_type(...)' code ##################

        str_688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, (-1)), 'str', '\n        Get the python type of the hold entity. This is equivalent to call the type(hold_python_entity). If a user-\n        defined class instance is hold, a types.InstanceType is returned (as Python does)\n        :return: A python type\n        ')
        
        
        # Call to isclass(...): (line 513)
        # Processing the call arguments (line 513)
        # Getting the type of 'self' (line 513)
        self_691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 31), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 513)
        python_entity_692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 31), self_691, 'python_entity')
        # Processing the call keyword arguments (line 513)
        kwargs_693 = {}
        # Getting the type of 'inspect' (line 513)
        inspect_689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 15), 'inspect', False)
        # Obtaining the member 'isclass' of a type (line 513)
        isclass_690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 15), inspect_689, 'isclass')
        # Calling isclass(args, kwargs) (line 513)
        isclass_call_result_694 = invoke(stypy.reporting.localization.Localization(__file__, 513, 15), isclass_690, *[python_entity_692], **kwargs_693)
        
        # Applying the 'not' unary operator (line 513)
        result_not__695 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 11), 'not', isclass_call_result_694)
        
        # Testing if the type of an if condition is none (line 513)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 513, 8), result_not__695):
            pass
        else:
            
            # Testing the type of an if condition (line 513)
            if_condition_696 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 513, 8), result_not__695)
            # Assigning a type to the variable 'if_condition_696' (line 513)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'if_condition_696', if_condition_696)
            # SSA begins for if statement (line 513)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to type(...): (line 514)
            # Processing the call arguments (line 514)
            # Getting the type of 'self' (line 514)
            self_698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 24), 'self', False)
            # Obtaining the member 'python_entity' of a type (line 514)
            python_entity_699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 24), self_698, 'python_entity')
            # Processing the call keyword arguments (line 514)
            kwargs_700 = {}
            # Getting the type of 'type' (line 514)
            type_697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 19), 'type', False)
            # Calling type(args, kwargs) (line 514)
            type_call_result_701 = invoke(stypy.reporting.localization.Localization(__file__, 514, 19), type_697, *[python_entity_699], **kwargs_700)
            
            # Assigning a type to the variable 'stypy_return_type' (line 514)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 12), 'stypy_return_type', type_call_result_701)
            # SSA join for if statement (line 513)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Evaluating a boolean operation
        
        # Call to is_user_defined_class(...): (line 516)
        # Processing the call arguments (line 516)
        # Getting the type of 'self' (line 516)
        self_704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 70), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 516)
        python_entity_705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 70), self_704, 'python_entity')
        # Processing the call keyword arguments (line 516)
        kwargs_706 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 516)
        type_inference_proxy_management_copy_702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 11), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'is_user_defined_class' of a type (line 516)
        is_user_defined_class_703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 11), type_inference_proxy_management_copy_702, 'is_user_defined_class')
        # Calling is_user_defined_class(args, kwargs) (line 516)
        is_user_defined_class_call_result_707 = invoke(stypy.reporting.localization.Localization(__file__, 516, 11), is_user_defined_class_703, *[python_entity_705], **kwargs_706)
        
        
        # Getting the type of 'self' (line 516)
        self_708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 94), 'self')
        # Obtaining the member 'instance' of a type (line 516)
        instance_709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 94), self_708, 'instance')
        # Getting the type of 'None' (line 516)
        None_710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 115), 'None')
        # Applying the binary operator 'isnot' (line 516)
        result_is_not_711 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 94), 'isnot', instance_709, None_710)
        
        # Applying the binary operator 'and' (line 516)
        result_and_keyword_712 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 11), 'and', is_user_defined_class_call_result_707, result_is_not_711)
        
        # Testing if the type of an if condition is none (line 516)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 516, 8), result_and_keyword_712):
            pass
        else:
            
            # Testing the type of an if condition (line 516)
            if_condition_713 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 516, 8), result_and_keyword_712)
            # Assigning a type to the variable 'if_condition_713' (line 516)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'if_condition_713', if_condition_713)
            # SSA begins for if statement (line 516)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'types' (line 517)
            types_714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 19), 'types')
            # Obtaining the member 'InstanceType' of a type (line 517)
            InstanceType_715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 19), types_714, 'InstanceType')
            # Assigning a type to the variable 'stypy_return_type' (line 517)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 12), 'stypy_return_type', InstanceType_715)
            # SSA join for if statement (line 516)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'self' (line 519)
        self_716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 15), 'self')
        # Obtaining the member 'python_entity' of a type (line 519)
        python_entity_717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 15), self_716, 'python_entity')
        # Assigning a type to the variable 'stypy_return_type' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'stypy_return_type', python_entity_717)
        
        # ################# End of 'get_python_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_python_type' in the type store
        # Getting the type of 'stypy_return_type' (line 507)
        stypy_return_type_718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_718)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_python_type'
        return stypy_return_type_718


    @norecursion
    def get_instance(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_instance'
        module_type_store = module_type_store.open_function_context('get_instance', 521, 4, False)
        # Assigning a type to the variable 'self' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.get_instance.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.get_instance.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.get_instance.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.get_instance.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.get_instance')
        TypeInferenceProxy.get_instance.__dict__.__setitem__('stypy_param_names_list', [])
        TypeInferenceProxy.get_instance.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.get_instance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.get_instance.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.get_instance.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.get_instance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.get_instance.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.get_instance', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_instance', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_instance(...)' code ##################

        str_719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, (-1)), 'str', '\n        Gets the stored class instance (if any). Class instances are only stored for instance intercession purposes, as\n        we need an entity to store these kind of changes.\n        :return:\n        ')
        # Getting the type of 'self' (line 527)
        self_720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 15), 'self')
        # Obtaining the member 'instance' of a type (line 527)
        instance_721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 15), self_720, 'instance')
        # Assigning a type to the variable 'stypy_return_type' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'stypy_return_type', instance_721)
        
        # ################# End of 'get_instance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_instance' in the type store
        # Getting the type of 'stypy_return_type' (line 521)
        stypy_return_type_722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_722)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_instance'
        return stypy_return_type_722


    @norecursion
    def has_value(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_value'
        module_type_store = module_type_store.open_function_context('has_value', 529, 4, False)
        # Assigning a type to the variable 'self' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.has_value.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.has_value.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.has_value.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.has_value.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.has_value')
        TypeInferenceProxy.has_value.__dict__.__setitem__('stypy_param_names_list', [])
        TypeInferenceProxy.has_value.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.has_value.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.has_value.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.has_value.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.has_value.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.has_value.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.has_value', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_value', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_value(...)' code ##################

        str_723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, (-1)), 'str', '\n        Determines if this proxy holds a value to the type it represents\n        :return:\n        ')
        
        # Call to hasattr(...): (line 534)
        # Processing the call arguments (line 534)
        # Getting the type of 'self' (line 534)
        self_725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 23), 'self', False)
        str_726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 29), 'str', 'value')
        # Processing the call keyword arguments (line 534)
        kwargs_727 = {}
        # Getting the type of 'hasattr' (line 534)
        hasattr_724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 15), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 534)
        hasattr_call_result_728 = invoke(stypy.reporting.localization.Localization(__file__, 534, 15), hasattr_724, *[self_725, str_726], **kwargs_727)
        
        # Assigning a type to the variable 'stypy_return_type' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 8), 'stypy_return_type', hasattr_call_result_728)
        
        # ################# End of 'has_value(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_value' in the type store
        # Getting the type of 'stypy_return_type' (line 529)
        stypy_return_type_729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_729)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_value'
        return stypy_return_type_729


    @norecursion
    def get_value(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_value'
        module_type_store = module_type_store.open_function_context('get_value', 536, 4, False)
        # Assigning a type to the variable 'self' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.get_value.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.get_value.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.get_value.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.get_value.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.get_value')
        TypeInferenceProxy.get_value.__dict__.__setitem__('stypy_param_names_list', [])
        TypeInferenceProxy.get_value.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.get_value.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.get_value.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.get_value.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.get_value.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.get_value.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.get_value', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_value', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_value(...)' code ##################

        str_730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, (-1)), 'str', '\n        Gets the value held by this proxy\n        :return: Value of the proxt\n        ')
        # Getting the type of 'self' (line 541)
        self_731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 15), 'self')
        # Obtaining the member 'value' of a type (line 541)
        value_732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 15), self_731, 'value')
        # Assigning a type to the variable 'stypy_return_type' (line 541)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'stypy_return_type', value_732)
        
        # ################# End of 'get_value(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_value' in the type store
        # Getting the type of 'stypy_return_type' (line 536)
        stypy_return_type_733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_733)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_value'
        return stypy_return_type_733


    @norecursion
    def set_value(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_value'
        module_type_store = module_type_store.open_function_context('set_value', 543, 4, False)
        # Assigning a type to the variable 'self' (line 544)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.set_value.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.set_value.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.set_value.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.set_value.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.set_value')
        TypeInferenceProxy.set_value.__dict__.__setitem__('stypy_param_names_list', ['value'])
        TypeInferenceProxy.set_value.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.set_value.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.set_value.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.set_value.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.set_value.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.set_value.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.set_value', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_value', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_value(...)' code ##################

        str_734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, (-1)), 'str', '\n        Sets the value held by this proxy. No type check is performed\n        :return: Value of the proxt\n        ')
        
        # Assigning a Name to a Attribute (line 548):
        
        # Assigning a Name to a Attribute (line 548):
        # Getting the type of 'value' (line 548)
        value_735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 21), 'value')
        # Getting the type of 'self' (line 548)
        self_736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'self')
        # Setting the type of the member 'value' of a type (line 548)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 8), self_736, 'value', value_735)
        
        # ################# End of 'set_value(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_value' in the type store
        # Getting the type of 'stypy_return_type' (line 543)
        stypy_return_type_737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_737)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_value'
        return stypy_return_type_737


    @norecursion
    def __get_module_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__get_module_file'
        module_type_store = module_type_store.open_function_context('__get_module_file', 552, 4, False)
        # Assigning a type to the variable 'self' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.__get_module_file.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.__get_module_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.__get_module_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.__get_module_file.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.__get_module_file')
        TypeInferenceProxy.__get_module_file.__dict__.__setitem__('stypy_param_names_list', [])
        TypeInferenceProxy.__get_module_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.__get_module_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.__get_module_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.__get_module_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.__get_module_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.__get_module_file.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.__get_module_file', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__get_module_file', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__get_module_file(...)' code ##################

        
        # Getting the type of 'True' (line 553)
        True_738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 14), 'True')
        # Assigning a type to the variable 'True_738' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'True_738', True_738)
        # Testing if the while is going to be iterated (line 553)
        # Testing the type of an if condition (line 553)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 553, 8), True_738)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 553, 8), True_738):
            
            # Assigning a Attribute to a Name (line 554):
            
            # Assigning a Attribute to a Name (line 554):
            # Getting the type of 'self' (line 554)
            self_739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 22), 'self')
            # Obtaining the member 'parent_proxy' of a type (line 554)
            parent_proxy_740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 22), self_739, 'parent_proxy')
            # Obtaining the member 'python_entity' of a type (line 554)
            python_entity_741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 22), parent_proxy_740, 'python_entity')
            # Assigning a type to the variable 'current' (line 554)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 12), 'current', python_entity_741)
            
            # Type idiom detected: calculating its left and rigth part (line 555)
            # Getting the type of 'current' (line 555)
            current_742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 15), 'current')
            # Getting the type of 'None' (line 555)
            None_743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 26), 'None')
            
            (may_be_744, more_types_in_union_745) = may_be_none(current_742, None_743)

            if may_be_744:

                if more_types_in_union_745:
                    # Runtime conditional SSA (line 555)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                str_746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 23), 'str', '')
                # Assigning a type to the variable 'stypy_return_type' (line 556)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 16), 'stypy_return_type', str_746)

                if more_types_in_union_745:
                    # SSA join for if statement (line 555)
                    module_type_store = module_type_store.join_ssa_context()


            
            # Getting the type of 'current' (line 555)
            current_747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 12), 'current')
            # Assigning a type to the variable 'current' (line 555)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 12), 'current', remove_type_from_union(current_747, types.NoneType))
            
            # Call to isinstance(...): (line 557)
            # Processing the call arguments (line 557)
            # Getting the type of 'current' (line 557)
            current_749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 26), 'current', False)
            # Getting the type of 'types' (line 557)
            types_750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 35), 'types', False)
            # Obtaining the member 'ModuleType' of a type (line 557)
            ModuleType_751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 35), types_750, 'ModuleType')
            # Processing the call keyword arguments (line 557)
            kwargs_752 = {}
            # Getting the type of 'isinstance' (line 557)
            isinstance_748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 557)
            isinstance_call_result_753 = invoke(stypy.reporting.localization.Localization(__file__, 557, 15), isinstance_748, *[current_749, ModuleType_751], **kwargs_752)
            
            # Testing if the type of an if condition is none (line 557)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 557, 12), isinstance_call_result_753):
                pass
            else:
                
                # Testing the type of an if condition (line 557)
                if_condition_754 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 557, 12), isinstance_call_result_753)
                # Assigning a type to the variable 'if_condition_754' (line 557)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 12), 'if_condition_754', if_condition_754)
                # SSA begins for if statement (line 557)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'current' (line 558)
                current_755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 23), 'current')
                # Obtaining the member '__file__' of a type (line 558)
                file___756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 23), current_755, '__file__')
                # Assigning a type to the variable 'stypy_return_type' (line 558)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 16), 'stypy_return_type', file___756)
                # SSA join for if statement (line 557)
                module_type_store = module_type_store.join_ssa_context()
                


        
        
        # ################# End of '__get_module_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__get_module_file' in the type store
        # Getting the type of 'stypy_return_type' (line 552)
        stypy_return_type_757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_757)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__get_module_file'
        return stypy_return_type_757


    @norecursion
    def get_type_of_member(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_type_of_member'
        module_type_store = module_type_store.open_function_context('get_type_of_member', 560, 4, False)
        # Assigning a type to the variable 'self' (line 561)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.get_type_of_member.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.get_type_of_member.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.get_type_of_member.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.get_type_of_member.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.get_type_of_member')
        TypeInferenceProxy.get_type_of_member.__dict__.__setitem__('stypy_param_names_list', ['localization', 'member_name'])
        TypeInferenceProxy.get_type_of_member.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.get_type_of_member.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.get_type_of_member.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.get_type_of_member.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.get_type_of_member.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.get_type_of_member.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.get_type_of_member', ['localization', 'member_name'], None, None, defaults, varargs, kwargs)

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

        str_758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, (-1)), 'str', '\n        Returns the type of the passed member name or a TypeError if the stored entity has no member with the mentioned\n        name.\n        :param localization: Call localization data\n        :param member_name: Member name\n        :return: A type proxy with the member type or a TypeError\n        ')
        
        
        # SSA begins for try-except statement (line 568)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Type idiom detected: calculating its left and rigth part (line 569)
        # Getting the type of 'self' (line 569)
        self_759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 15), 'self')
        # Obtaining the member 'instance' of a type (line 569)
        instance_760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 15), self_759, 'instance')
        # Getting the type of 'None' (line 569)
        None_761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 32), 'None')
        
        (may_be_762, more_types_in_union_763) = may_be_none(instance_760, None_761)

        if may_be_762:

            if more_types_in_union_763:
                # Runtime conditional SSA (line 569)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to instance(...): (line 570)
            # Processing the call arguments (line 570)
            
            # Call to getattr(...): (line 570)
            # Processing the call arguments (line 570)
            # Getting the type of 'self' (line 570)
            self_767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 59), 'self', False)
            # Obtaining the member 'python_entity' of a type (line 570)
            python_entity_768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 59), self_767, 'python_entity')
            # Getting the type of 'member_name' (line 570)
            member_name_769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 79), 'member_name', False)
            # Processing the call keyword arguments (line 570)
            kwargs_770 = {}
            # Getting the type of 'getattr' (line 570)
            getattr_766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 51), 'getattr', False)
            # Calling getattr(args, kwargs) (line 570)
            getattr_call_result_771 = invoke(stypy.reporting.localization.Localization(__file__, 570, 51), getattr_766, *[python_entity_768, member_name_769], **kwargs_770)
            
            # Getting the type of 'self' (line 571)
            self_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 51), 'self', False)
            # Obtaining the member 'name' of a type (line 571)
            name_773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 51), self_772, 'name')
            str_774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 63), 'str', '.')
            # Applying the binary operator '+' (line 571)
            result_add_775 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 51), '+', name_773, str_774)
            
            # Getting the type of 'member_name' (line 571)
            member_name_776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 69), 'member_name', False)
            # Applying the binary operator '+' (line 571)
            result_add_777 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 67), '+', result_add_775, member_name_776)
            
            # Processing the call keyword arguments (line 570)
            # Getting the type of 'self' (line 572)
            self_778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 58), 'self', False)
            keyword_779 = self_778
            kwargs_780 = {'parent': keyword_779}
            # Getting the type of 'TypeInferenceProxy' (line 570)
            TypeInferenceProxy_764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 23), 'TypeInferenceProxy', False)
            # Obtaining the member 'instance' of a type (line 570)
            instance_765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 23), TypeInferenceProxy_764, 'instance')
            # Calling instance(args, kwargs) (line 570)
            instance_call_result_781 = invoke(stypy.reporting.localization.Localization(__file__, 570, 23), instance_765, *[getattr_call_result_771, result_add_777], **kwargs_780)
            
            # Assigning a type to the variable 'stypy_return_type' (line 570)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 16), 'stypy_return_type', instance_call_result_781)

            if more_types_in_union_763:
                # Runtime conditional SSA for else branch (line 569)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_762) or more_types_in_union_763):
            
            # Call to hasattr(...): (line 575)
            # Processing the call arguments (line 575)
            # Getting the type of 'self' (line 575)
            self_783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 27), 'self', False)
            # Obtaining the member 'instance' of a type (line 575)
            instance_784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 27), self_783, 'instance')
            # Getting the type of 'member_name' (line 575)
            member_name_785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 42), 'member_name', False)
            # Processing the call keyword arguments (line 575)
            kwargs_786 = {}
            # Getting the type of 'hasattr' (line 575)
            hasattr_782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 19), 'hasattr', False)
            # Calling hasattr(args, kwargs) (line 575)
            hasattr_call_result_787 = invoke(stypy.reporting.localization.Localization(__file__, 575, 19), hasattr_782, *[instance_784, member_name_785], **kwargs_786)
            
            # Testing if the type of an if condition is none (line 575)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 575, 16), hasattr_call_result_787):
                
                # Assigning a Call to a Name (line 584):
                
                # Assigning a Call to a Name (line 584):
                
                # Call to get_original_program_from_type_inference_file(...): (line 584)
                # Processing the call arguments (line 584)
                
                # Call to __get_module_file(...): (line 585)
                # Processing the call keyword arguments (line 585)
                kwargs_811 = {}
                # Getting the type of 'self' (line 585)
                self_809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 24), 'self', False)
                # Obtaining the member '__get_module_file' of a type (line 585)
                get_module_file_810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 24), self_809, '__get_module_file')
                # Calling __get_module_file(args, kwargs) (line 585)
                get_module_file_call_result_812 = invoke(stypy.reporting.localization.Localization(__file__, 585, 24), get_module_file_810, *[], **kwargs_811)
                
                # Processing the call keyword arguments (line 584)
                kwargs_813 = {}
                # Getting the type of 'stypy_parameters_copy' (line 584)
                stypy_parameters_copy_807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 34), 'stypy_parameters_copy', False)
                # Obtaining the member 'get_original_program_from_type_inference_file' of a type (line 584)
                get_original_program_from_type_inference_file_808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 34), stypy_parameters_copy_807, 'get_original_program_from_type_inference_file')
                # Calling get_original_program_from_type_inference_file(args, kwargs) (line 584)
                get_original_program_from_type_inference_file_call_result_814 = invoke(stypy.reporting.localization.Localization(__file__, 584, 34), get_original_program_from_type_inference_file_808, *[get_module_file_call_result_812], **kwargs_813)
                
                # Assigning a type to the variable 'module_path' (line 584)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 20), 'module_path', get_original_program_from_type_inference_file_call_result_814)
                
                # Assigning a Call to a Name (line 586):
                
                # Assigning a Call to a Name (line 586):
                
                # Call to get_type_store_of_module(...): (line 586)
                # Processing the call arguments (line 586)
                # Getting the type of 'module_path' (line 586)
                module_path_819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 86), 'module_path', False)
                # Processing the call keyword arguments (line 586)
                kwargs_820 = {}
                # Getting the type of 'type_store_copy' (line 586)
                type_store_copy_815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 25), 'type_store_copy', False)
                # Obtaining the member 'typestore' of a type (line 586)
                typestore_816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 25), type_store_copy_815, 'typestore')
                # Obtaining the member 'TypeStore' of a type (line 586)
                TypeStore_817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 25), typestore_816, 'TypeStore')
                # Obtaining the member 'get_type_store_of_module' of a type (line 586)
                get_type_store_of_module_818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 25), TypeStore_817, 'get_type_store_of_module')
                # Calling get_type_store_of_module(args, kwargs) (line 586)
                get_type_store_of_module_call_result_821 = invoke(stypy.reporting.localization.Localization(__file__, 586, 25), get_type_store_of_module_818, *[module_path_819], **kwargs_820)
                
                # Assigning a type to the variable 'ts' (line 586)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 20), 'ts', get_type_store_of_module_call_result_821)
                
                # Assigning a Call to a Name (line 587):
                
                # Assigning a Call to a Name (line 587):
                
                # Call to get_type_of(...): (line 587)
                # Processing the call arguments (line 587)
                # Getting the type of 'localization' (line 587)
                localization_824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 41), 'localization', False)
                # Getting the type of 'self' (line 587)
                self_825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 55), 'self', False)
                # Obtaining the member 'python_entity' of a type (line 587)
                python_entity_826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 55), self_825, 'python_entity')
                # Obtaining the member '__name__' of a type (line 587)
                name___827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 55), python_entity_826, '__name__')
                # Processing the call keyword arguments (line 587)
                kwargs_828 = {}
                # Getting the type of 'ts' (line 587)
                ts_822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 26), 'ts', False)
                # Obtaining the member 'get_type_of' of a type (line 587)
                get_type_of_823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 26), ts_822, 'get_type_of')
                # Calling get_type_of(args, kwargs) (line 587)
                get_type_of_call_result_829 = invoke(stypy.reporting.localization.Localization(__file__, 587, 26), get_type_of_823, *[localization_824, name___827], **kwargs_828)
                
                # Assigning a type to the variable 'typ' (line 587)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 20), 'typ', get_type_of_call_result_829)
                
                # Call to get_type_of_member(...): (line 588)
                # Processing the call arguments (line 588)
                # Getting the type of 'localization' (line 588)
                localization_832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 50), 'localization', False)
                # Getting the type of 'member_name' (line 588)
                member_name_833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 64), 'member_name', False)
                # Processing the call keyword arguments (line 588)
                kwargs_834 = {}
                # Getting the type of 'typ' (line 588)
                typ_830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 27), 'typ', False)
                # Obtaining the member 'get_type_of_member' of a type (line 588)
                get_type_of_member_831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 27), typ_830, 'get_type_of_member')
                # Calling get_type_of_member(args, kwargs) (line 588)
                get_type_of_member_call_result_835 = invoke(stypy.reporting.localization.Localization(__file__, 588, 27), get_type_of_member_831, *[localization_832, member_name_833], **kwargs_834)
                
                # Assigning a type to the variable 'stypy_return_type' (line 588)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 20), 'stypy_return_type', get_type_of_member_call_result_835)
            else:
                
                # Testing the type of an if condition (line 575)
                if_condition_788 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 575, 16), hasattr_call_result_787)
                # Assigning a type to the variable 'if_condition_788' (line 575)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 16), 'if_condition_788', if_condition_788)
                # SSA begins for if statement (line 575)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to instance(...): (line 576)
                # Processing the call arguments (line 576)
                
                # Call to getattr(...): (line 576)
                # Processing the call arguments (line 576)
                # Getting the type of 'self' (line 576)
                self_792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 63), 'self', False)
                # Obtaining the member 'instance' of a type (line 576)
                instance_793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 63), self_792, 'instance')
                # Getting the type of 'member_name' (line 576)
                member_name_794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 78), 'member_name', False)
                # Processing the call keyword arguments (line 576)
                kwargs_795 = {}
                # Getting the type of 'getattr' (line 576)
                getattr_791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 55), 'getattr', False)
                # Calling getattr(args, kwargs) (line 576)
                getattr_call_result_796 = invoke(stypy.reporting.localization.Localization(__file__, 576, 55), getattr_791, *[instance_793, member_name_794], **kwargs_795)
                
                # Getting the type of 'self' (line 577)
                self_797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 55), 'self', False)
                # Obtaining the member 'name' of a type (line 577)
                name_798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 55), self_797, 'name')
                str_799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 67), 'str', '.')
                # Applying the binary operator '+' (line 577)
                result_add_800 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 55), '+', name_798, str_799)
                
                # Getting the type of 'member_name' (line 577)
                member_name_801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 73), 'member_name', False)
                # Applying the binary operator '+' (line 577)
                result_add_802 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 71), '+', result_add_800, member_name_801)
                
                # Processing the call keyword arguments (line 576)
                # Getting the type of 'self' (line 578)
                self_803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 62), 'self', False)
                keyword_804 = self_803
                kwargs_805 = {'parent': keyword_804}
                # Getting the type of 'TypeInferenceProxy' (line 576)
                TypeInferenceProxy_789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 27), 'TypeInferenceProxy', False)
                # Obtaining the member 'instance' of a type (line 576)
                instance_790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 27), TypeInferenceProxy_789, 'instance')
                # Calling instance(args, kwargs) (line 576)
                instance_call_result_806 = invoke(stypy.reporting.localization.Localization(__file__, 576, 27), instance_790, *[getattr_call_result_796, result_add_802], **kwargs_805)
                
                # Assigning a type to the variable 'stypy_return_type' (line 576)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 20), 'stypy_return_type', instance_call_result_806)
                # SSA branch for the else part of an if statement (line 575)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Name (line 584):
                
                # Assigning a Call to a Name (line 584):
                
                # Call to get_original_program_from_type_inference_file(...): (line 584)
                # Processing the call arguments (line 584)
                
                # Call to __get_module_file(...): (line 585)
                # Processing the call keyword arguments (line 585)
                kwargs_811 = {}
                # Getting the type of 'self' (line 585)
                self_809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 24), 'self', False)
                # Obtaining the member '__get_module_file' of a type (line 585)
                get_module_file_810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 24), self_809, '__get_module_file')
                # Calling __get_module_file(args, kwargs) (line 585)
                get_module_file_call_result_812 = invoke(stypy.reporting.localization.Localization(__file__, 585, 24), get_module_file_810, *[], **kwargs_811)
                
                # Processing the call keyword arguments (line 584)
                kwargs_813 = {}
                # Getting the type of 'stypy_parameters_copy' (line 584)
                stypy_parameters_copy_807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 34), 'stypy_parameters_copy', False)
                # Obtaining the member 'get_original_program_from_type_inference_file' of a type (line 584)
                get_original_program_from_type_inference_file_808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 34), stypy_parameters_copy_807, 'get_original_program_from_type_inference_file')
                # Calling get_original_program_from_type_inference_file(args, kwargs) (line 584)
                get_original_program_from_type_inference_file_call_result_814 = invoke(stypy.reporting.localization.Localization(__file__, 584, 34), get_original_program_from_type_inference_file_808, *[get_module_file_call_result_812], **kwargs_813)
                
                # Assigning a type to the variable 'module_path' (line 584)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 20), 'module_path', get_original_program_from_type_inference_file_call_result_814)
                
                # Assigning a Call to a Name (line 586):
                
                # Assigning a Call to a Name (line 586):
                
                # Call to get_type_store_of_module(...): (line 586)
                # Processing the call arguments (line 586)
                # Getting the type of 'module_path' (line 586)
                module_path_819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 86), 'module_path', False)
                # Processing the call keyword arguments (line 586)
                kwargs_820 = {}
                # Getting the type of 'type_store_copy' (line 586)
                type_store_copy_815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 25), 'type_store_copy', False)
                # Obtaining the member 'typestore' of a type (line 586)
                typestore_816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 25), type_store_copy_815, 'typestore')
                # Obtaining the member 'TypeStore' of a type (line 586)
                TypeStore_817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 25), typestore_816, 'TypeStore')
                # Obtaining the member 'get_type_store_of_module' of a type (line 586)
                get_type_store_of_module_818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 25), TypeStore_817, 'get_type_store_of_module')
                # Calling get_type_store_of_module(args, kwargs) (line 586)
                get_type_store_of_module_call_result_821 = invoke(stypy.reporting.localization.Localization(__file__, 586, 25), get_type_store_of_module_818, *[module_path_819], **kwargs_820)
                
                # Assigning a type to the variable 'ts' (line 586)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 20), 'ts', get_type_store_of_module_call_result_821)
                
                # Assigning a Call to a Name (line 587):
                
                # Assigning a Call to a Name (line 587):
                
                # Call to get_type_of(...): (line 587)
                # Processing the call arguments (line 587)
                # Getting the type of 'localization' (line 587)
                localization_824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 41), 'localization', False)
                # Getting the type of 'self' (line 587)
                self_825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 55), 'self', False)
                # Obtaining the member 'python_entity' of a type (line 587)
                python_entity_826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 55), self_825, 'python_entity')
                # Obtaining the member '__name__' of a type (line 587)
                name___827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 55), python_entity_826, '__name__')
                # Processing the call keyword arguments (line 587)
                kwargs_828 = {}
                # Getting the type of 'ts' (line 587)
                ts_822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 26), 'ts', False)
                # Obtaining the member 'get_type_of' of a type (line 587)
                get_type_of_823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 26), ts_822, 'get_type_of')
                # Calling get_type_of(args, kwargs) (line 587)
                get_type_of_call_result_829 = invoke(stypy.reporting.localization.Localization(__file__, 587, 26), get_type_of_823, *[localization_824, name___827], **kwargs_828)
                
                # Assigning a type to the variable 'typ' (line 587)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 20), 'typ', get_type_of_call_result_829)
                
                # Call to get_type_of_member(...): (line 588)
                # Processing the call arguments (line 588)
                # Getting the type of 'localization' (line 588)
                localization_832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 50), 'localization', False)
                # Getting the type of 'member_name' (line 588)
                member_name_833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 64), 'member_name', False)
                # Processing the call keyword arguments (line 588)
                kwargs_834 = {}
                # Getting the type of 'typ' (line 588)
                typ_830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 27), 'typ', False)
                # Obtaining the member 'get_type_of_member' of a type (line 588)
                get_type_of_member_831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 27), typ_830, 'get_type_of_member')
                # Calling get_type_of_member(args, kwargs) (line 588)
                get_type_of_member_call_result_835 = invoke(stypy.reporting.localization.Localization(__file__, 588, 27), get_type_of_member_831, *[localization_832, member_name_833], **kwargs_834)
                
                # Assigning a type to the variable 'stypy_return_type' (line 588)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 20), 'stypy_return_type', get_type_of_member_call_result_835)
                # SSA join for if statement (line 575)
                module_type_store = module_type_store.join_ssa_context()
                


            if (may_be_762 and more_types_in_union_763):
                # SSA join for if statement (line 569)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA branch for the except part of a try statement (line 568)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 568)
        module_type_store.open_ssa_branch('except')
        
        # Call to TypeError(...): (line 593)
        # Processing the call arguments (line 593)
        # Getting the type of 'localization' (line 593)
        localization_837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 29), 'localization', False)
        
        # Call to format(...): (line 594)
        # Processing the call arguments (line 594)
        
        # Call to get_python_type(...): (line 594)
        # Processing the call keyword arguments (line 594)
        kwargs_842 = {}
        # Getting the type of 'self' (line 594)
        self_840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 62), 'self', False)
        # Obtaining the member 'get_python_type' of a type (line 594)
        get_python_type_841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 62), self_840, 'get_python_type')
        # Calling get_python_type(args, kwargs) (line 594)
        get_python_type_call_result_843 = invoke(stypy.reporting.localization.Localization(__file__, 594, 62), get_python_type_841, *[], **kwargs_842)
        
        # Obtaining the member '__name__' of a type (line 594)
        name___844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 62), get_python_type_call_result_843, '__name__')
        # Getting the type of 'member_name' (line 594)
        member_name_845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 95), 'member_name', False)
        # Processing the call keyword arguments (line 594)
        kwargs_846 = {}
        str_838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 29), 'str', "{0} has no member '{1}'")
        # Obtaining the member 'format' of a type (line 594)
        format_839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 29), str_838, 'format')
        # Calling format(args, kwargs) (line 594)
        format_call_result_847 = invoke(stypy.reporting.localization.Localization(__file__, 594, 29), format_839, *[name___844, member_name_845], **kwargs_846)
        
        # Processing the call keyword arguments (line 593)
        kwargs_848 = {}
        # Getting the type of 'TypeError' (line 593)
        TypeError_836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 19), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 593)
        TypeError_call_result_849 = invoke(stypy.reporting.localization.Localization(__file__, 593, 19), TypeError_836, *[localization_837, format_call_result_847], **kwargs_848)
        
        # Assigning a type to the variable 'stypy_return_type' (line 593)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 12), 'stypy_return_type', TypeError_call_result_849)
        # SSA join for try-except statement (line 568)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_type_of_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_type_of_member' in the type store
        # Getting the type of 'stypy_return_type' (line 560)
        stypy_return_type_850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_850)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_type_of_member'
        return stypy_return_type_850


    @norecursion
    def set_type_of_member(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_type_of_member'
        module_type_store = module_type_store.open_function_context('set_type_of_member', 596, 4, False)
        # Assigning a type to the variable 'self' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.set_type_of_member.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.set_type_of_member.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.set_type_of_member.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.set_type_of_member.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.set_type_of_member')
        TypeInferenceProxy.set_type_of_member.__dict__.__setitem__('stypy_param_names_list', ['localization', 'member_name', 'member_type'])
        TypeInferenceProxy.set_type_of_member.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.set_type_of_member.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.set_type_of_member.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.set_type_of_member.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.set_type_of_member.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.set_type_of_member.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.set_type_of_member', ['localization', 'member_name', 'member_type'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_type_of_member', localization, ['localization', 'member_name', 'member_type'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_type_of_member(...)' code ##################

        str_851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, (-1)), 'str', '\n        Set the type of a member of the represented object. If the member do not exist, it is created with the passed\n        name and types (except iif the represented object do not support reflection, in that case a TypeError is\n        returned)\n        :param localization: Caller information\n        :param member_name: Name of the member\n        :param member_type: Type of the member\n        :return: None or a TypeError\n        ')
        
        
        # SSA begins for try-except statement (line 606)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Tuple (line 607):
        
        # Assigning a Call to a Name:
        
        # Call to contains_an_undefined_type(...): (line 607)
        # Processing the call arguments (line 607)
        # Getting the type of 'member_type' (line 607)
        member_type_854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 100), 'member_type', False)
        # Processing the call keyword arguments (line 607)
        kwargs_855 = {}
        # Getting the type of 'TypeInferenceProxy' (line 607)
        TypeInferenceProxy_852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 54), 'TypeInferenceProxy', False)
        # Obtaining the member 'contains_an_undefined_type' of a type (line 607)
        contains_an_undefined_type_853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 54), TypeInferenceProxy_852, 'contains_an_undefined_type')
        # Calling contains_an_undefined_type(args, kwargs) (line 607)
        contains_an_undefined_type_call_result_856 = invoke(stypy.reporting.localization.Localization(__file__, 607, 54), contains_an_undefined_type_853, *[member_type_854], **kwargs_855)
        
        # Assigning a type to the variable 'call_assignment_1' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'call_assignment_1', contains_an_undefined_type_call_result_856)
        
        # Assigning a Call to a Name (line 607):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_1' (line 607)
        call_assignment_1_857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'call_assignment_1', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_858 = stypy_get_value_from_tuple(call_assignment_1_857, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_2' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'call_assignment_2', stypy_get_value_from_tuple_call_result_858)
        
        # Assigning a Name to a Name (line 607):
        # Getting the type of 'call_assignment_2' (line 607)
        call_assignment_2_859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'call_assignment_2')
        # Assigning a type to the variable 'contains_undefined' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'contains_undefined', call_assignment_2_859)
        
        # Assigning a Call to a Name (line 607):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_1' (line 607)
        call_assignment_1_860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'call_assignment_1', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_861 = stypy_get_value_from_tuple(call_assignment_1_860, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_3' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'call_assignment_3', stypy_get_value_from_tuple_call_result_861)
        
        # Assigning a Name to a Name (line 607):
        # Getting the type of 'call_assignment_3' (line 607)
        call_assignment_3_862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'call_assignment_3')
        # Assigning a type to the variable 'more_types_in_value' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 32), 'more_types_in_value', call_assignment_3_862)
        # Getting the type of 'contains_undefined' (line 608)
        contains_undefined_863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 15), 'contains_undefined')
        # Testing if the type of an if condition is none (line 608)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 608, 12), contains_undefined_863):
            pass
        else:
            
            # Testing the type of an if condition (line 608)
            if_condition_864 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 608, 12), contains_undefined_863)
            # Assigning a type to the variable 'if_condition_864' (line 608)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 12), 'if_condition_864', if_condition_864)
            # SSA begins for if statement (line 608)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'more_types_in_value' (line 609)
            more_types_in_value_865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 19), 'more_types_in_value')
            int_866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 42), 'int')
            # Applying the binary operator '==' (line 609)
            result_eq_867 = python_operator(stypy.reporting.localization.Localization(__file__, 609, 19), '==', more_types_in_value_865, int_866)
            
            # Testing if the type of an if condition is none (line 609)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 609, 16), result_eq_867):
                
                # Call to instance(...): (line 613)
                # Processing the call arguments (line 613)
                # Getting the type of 'localization' (line 613)
                localization_883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 41), 'localization', False)
                
                # Call to format(...): (line 614)
                # Processing the call arguments (line 614)
                # Getting the type of 'self' (line 615)
                self_886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 48), 'self', False)
                # Obtaining the member 'parent_proxy' of a type (line 615)
                parent_proxy_887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 48), self_886, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 615)
                name_888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 48), parent_proxy_887, 'name')
                # Getting the type of 'member_name' (line 615)
                member_name_889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 72), 'member_name', False)
                # Processing the call keyword arguments (line 614)
                kwargs_890 = {}
                str_884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 41), 'str', 'Potentialy assigning to {0}.{1} the value of a previously undefined variable')
                # Obtaining the member 'format' of a type (line 614)
                format_885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 41), str_884, 'format')
                # Calling format(args, kwargs) (line 614)
                format_call_result_891 = invoke(stypy.reporting.localization.Localization(__file__, 614, 41), format_885, *[name_888, member_name_889], **kwargs_890)
                
                # Processing the call keyword arguments (line 613)
                kwargs_892 = {}
                # Getting the type of 'TypeWarning' (line 613)
                TypeWarning_881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 20), 'TypeWarning', False)
                # Obtaining the member 'instance' of a type (line 613)
                instance_882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 20), TypeWarning_881, 'instance')
                # Calling instance(args, kwargs) (line 613)
                instance_call_result_893 = invoke(stypy.reporting.localization.Localization(__file__, 613, 20), instance_882, *[localization_883, format_call_result_891], **kwargs_892)
                
            else:
                
                # Testing the type of an if condition (line 609)
                if_condition_868 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 609, 16), result_eq_867)
                # Assigning a type to the variable 'if_condition_868' (line 609)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 16), 'if_condition_868', if_condition_868)
                # SSA begins for if statement (line 609)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to TypeError(...): (line 610)
                # Processing the call arguments (line 610)
                # Getting the type of 'localization' (line 610)
                localization_870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 30), 'localization', False)
                
                # Call to format(...): (line 610)
                # Processing the call arguments (line 610)
                # Getting the type of 'self' (line 611)
                self_873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 37), 'self', False)
                # Obtaining the member 'parent_proxy' of a type (line 611)
                parent_proxy_874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 37), self_873, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 611)
                name_875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 37), parent_proxy_874, 'name')
                # Getting the type of 'member_name' (line 611)
                member_name_876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 61), 'member_name', False)
                # Processing the call keyword arguments (line 610)
                kwargs_877 = {}
                str_871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 44), 'str', 'Assigning to {0}.{1} the value of a previously undefined variable')
                # Obtaining the member 'format' of a type (line 610)
                format_872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 44), str_871, 'format')
                # Calling format(args, kwargs) (line 610)
                format_call_result_878 = invoke(stypy.reporting.localization.Localization(__file__, 610, 44), format_872, *[name_875, member_name_876], **kwargs_877)
                
                # Processing the call keyword arguments (line 610)
                kwargs_879 = {}
                # Getting the type of 'TypeError' (line 610)
                TypeError_869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 20), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 610)
                TypeError_call_result_880 = invoke(stypy.reporting.localization.Localization(__file__, 610, 20), TypeError_869, *[localization_870, format_call_result_878], **kwargs_879)
                
                # SSA branch for the else part of an if statement (line 609)
                module_type_store.open_ssa_branch('else')
                
                # Call to instance(...): (line 613)
                # Processing the call arguments (line 613)
                # Getting the type of 'localization' (line 613)
                localization_883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 41), 'localization', False)
                
                # Call to format(...): (line 614)
                # Processing the call arguments (line 614)
                # Getting the type of 'self' (line 615)
                self_886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 48), 'self', False)
                # Obtaining the member 'parent_proxy' of a type (line 615)
                parent_proxy_887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 48), self_886, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 615)
                name_888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 48), parent_proxy_887, 'name')
                # Getting the type of 'member_name' (line 615)
                member_name_889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 72), 'member_name', False)
                # Processing the call keyword arguments (line 614)
                kwargs_890 = {}
                str_884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 41), 'str', 'Potentialy assigning to {0}.{1} the value of a previously undefined variable')
                # Obtaining the member 'format' of a type (line 614)
                format_885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 41), str_884, 'format')
                # Calling format(args, kwargs) (line 614)
                format_call_result_891 = invoke(stypy.reporting.localization.Localization(__file__, 614, 41), format_885, *[name_888, member_name_889], **kwargs_890)
                
                # Processing the call keyword arguments (line 613)
                kwargs_892 = {}
                # Getting the type of 'TypeWarning' (line 613)
                TypeWarning_881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 20), 'TypeWarning', False)
                # Obtaining the member 'instance' of a type (line 613)
                instance_882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 20), TypeWarning_881, 'instance')
                # Calling instance(args, kwargs) (line 613)
                instance_call_result_893 = invoke(stypy.reporting.localization.Localization(__file__, 613, 20), instance_882, *[localization_883, format_call_result_891], **kwargs_892)
                
                # SSA join for if statement (line 609)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 608)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 617)
        self_894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 15), 'self')
        # Obtaining the member 'instance' of a type (line 617)
        instance_895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 15), self_894, 'instance')
        # Getting the type of 'None' (line 617)
        None_896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 36), 'None')
        # Applying the binary operator 'isnot' (line 617)
        result_is_not_897 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 15), 'isnot', instance_895, None_896)
        
        # Testing if the type of an if condition is none (line 617)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 617, 12), result_is_not_897):
            pass
        else:
            
            # Testing the type of an if condition (line 617)
            if_condition_898 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 617, 12), result_is_not_897)
            # Assigning a type to the variable 'if_condition_898' (line 617)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 12), 'if_condition_898', if_condition_898)
            # SSA begins for if statement (line 617)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to setattr(...): (line 619)
            # Processing the call arguments (line 619)
            # Getting the type of 'self' (line 619)
            self_900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 24), 'self', False)
            # Obtaining the member 'instance' of a type (line 619)
            instance_901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 24), self_900, 'instance')
            # Getting the type of 'member_name' (line 619)
            member_name_902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 39), 'member_name', False)
            # Getting the type of 'member_type' (line 619)
            member_type_903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 52), 'member_type', False)
            # Processing the call keyword arguments (line 619)
            kwargs_904 = {}
            # Getting the type of 'setattr' (line 619)
            setattr_899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 16), 'setattr', False)
            # Calling setattr(args, kwargs) (line 619)
            setattr_call_result_905 = invoke(stypy.reporting.localization.Localization(__file__, 619, 16), setattr_899, *[instance_901, member_name_902, member_type_903], **kwargs_904)
            
            # Getting the type of 'self' (line 620)
            self_906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 19), 'self')
            # Obtaining the member 'annotate_types' of a type (line 620)
            annotate_types_907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 19), self_906, 'annotate_types')
            # Testing if the type of an if condition is none (line 620)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 620, 16), annotate_types_907):
                pass
            else:
                
                # Testing the type of an if condition (line 620)
                if_condition_908 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 620, 16), annotate_types_907)
                # Assigning a type to the variable 'if_condition_908' (line 620)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 16), 'if_condition_908', if_condition_908)
                # SSA begins for if statement (line 620)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to __annotate_type(...): (line 621)
                # Processing the call arguments (line 621)
                # Getting the type of 'localization' (line 621)
                localization_911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 41), 'localization', False)
                # Obtaining the member 'line' of a type (line 621)
                line_912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 41), localization_911, 'line')
                # Getting the type of 'localization' (line 621)
                localization_913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 60), 'localization', False)
                # Obtaining the member 'column' of a type (line 621)
                column_914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 60), localization_913, 'column')
                # Getting the type of 'member_name' (line 621)
                member_name_915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 81), 'member_name', False)
                # Getting the type of 'member_type' (line 622)
                member_type_916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 41), 'member_type', False)
                # Processing the call keyword arguments (line 621)
                kwargs_917 = {}
                # Getting the type of 'self' (line 621)
                self_909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 20), 'self', False)
                # Obtaining the member '__annotate_type' of a type (line 621)
                annotate_type_910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 20), self_909, '__annotate_type')
                # Calling __annotate_type(args, kwargs) (line 621)
                annotate_type_call_result_918 = invoke(stypy.reporting.localization.Localization(__file__, 621, 20), annotate_type_910, *[line_912, column_914, member_name_915, member_type_916], **kwargs_917)
                
                # SSA join for if statement (line 620)
                module_type_store = module_type_store.join_ssa_context()
                

            # Getting the type of 'None' (line 623)
            None_919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 23), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 623)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 16), 'stypy_return_type', None_919)
            # SSA join for if statement (line 617)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Evaluating a boolean operation
        
        # Call to supports_structural_reflection(...): (line 625)
        # Processing the call arguments (line 625)
        # Getting the type of 'self' (line 625)
        self_922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 83), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 625)
        python_entity_923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 83), self_922, 'python_entity')
        # Processing the call keyword arguments (line 625)
        kwargs_924 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 625)
        type_inference_proxy_management_copy_920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 15), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'supports_structural_reflection' of a type (line 625)
        supports_structural_reflection_921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 15), type_inference_proxy_management_copy_920, 'supports_structural_reflection')
        # Calling supports_structural_reflection(args, kwargs) (line 625)
        supports_structural_reflection_call_result_925 = invoke(stypy.reporting.localization.Localization(__file__, 625, 15), supports_structural_reflection_921, *[python_entity_923], **kwargs_924)
        
        
        # Call to hasattr(...): (line 625)
        # Processing the call arguments (line 625)
        # Getting the type of 'self' (line 626)
        self_927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 20), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 626)
        python_entity_928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 20), self_927, 'python_entity')
        # Getting the type of 'member_name' (line 626)
        member_name_929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 40), 'member_name', False)
        # Processing the call keyword arguments (line 625)
        kwargs_930 = {}
        # Getting the type of 'hasattr' (line 625)
        hasattr_926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 106), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 625)
        hasattr_call_result_931 = invoke(stypy.reporting.localization.Localization(__file__, 625, 106), hasattr_926, *[python_entity_928, member_name_929], **kwargs_930)
        
        # Applying the binary operator 'or' (line 625)
        result_or_keyword_932 = python_operator(stypy.reporting.localization.Localization(__file__, 625, 15), 'or', supports_structural_reflection_call_result_925, hasattr_call_result_931)
        
        # Testing if the type of an if condition is none (line 625)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 625, 12), result_or_keyword_932):
            pass
        else:
            
            # Testing the type of an if condition (line 625)
            if_condition_933 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 625, 12), result_or_keyword_932)
            # Assigning a type to the variable 'if_condition_933' (line 625)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 12), 'if_condition_933', if_condition_933)
            # SSA begins for if statement (line 625)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to setattr(...): (line 628)
            # Processing the call arguments (line 628)
            # Getting the type of 'self' (line 628)
            self_935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 24), 'self', False)
            # Obtaining the member 'python_entity' of a type (line 628)
            python_entity_936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 24), self_935, 'python_entity')
            # Getting the type of 'member_name' (line 628)
            member_name_937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 44), 'member_name', False)
            # Getting the type of 'member_type' (line 628)
            member_type_938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 57), 'member_type', False)
            # Processing the call keyword arguments (line 628)
            kwargs_939 = {}
            # Getting the type of 'setattr' (line 628)
            setattr_934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 16), 'setattr', False)
            # Calling setattr(args, kwargs) (line 628)
            setattr_call_result_940 = invoke(stypy.reporting.localization.Localization(__file__, 628, 16), setattr_934, *[python_entity_936, member_name_937, member_type_938], **kwargs_939)
            
            # Getting the type of 'self' (line 629)
            self_941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 19), 'self')
            # Obtaining the member 'annotate_types' of a type (line 629)
            annotate_types_942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 19), self_941, 'annotate_types')
            # Testing if the type of an if condition is none (line 629)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 629, 16), annotate_types_942):
                pass
            else:
                
                # Testing the type of an if condition (line 629)
                if_condition_943 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 629, 16), annotate_types_942)
                # Assigning a type to the variable 'if_condition_943' (line 629)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 16), 'if_condition_943', if_condition_943)
                # SSA begins for if statement (line 629)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to __annotate_type(...): (line 630)
                # Processing the call arguments (line 630)
                # Getting the type of 'localization' (line 630)
                localization_946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 41), 'localization', False)
                # Obtaining the member 'line' of a type (line 630)
                line_947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 41), localization_946, 'line')
                # Getting the type of 'localization' (line 630)
                localization_948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 60), 'localization', False)
                # Obtaining the member 'column' of a type (line 630)
                column_949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 60), localization_948, 'column')
                # Getting the type of 'member_name' (line 630)
                member_name_950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 81), 'member_name', False)
                # Getting the type of 'member_type' (line 631)
                member_type_951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 41), 'member_type', False)
                # Processing the call keyword arguments (line 630)
                kwargs_952 = {}
                # Getting the type of 'self' (line 630)
                self_944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 20), 'self', False)
                # Obtaining the member '__annotate_type' of a type (line 630)
                annotate_type_945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 20), self_944, '__annotate_type')
                # Calling __annotate_type(args, kwargs) (line 630)
                annotate_type_call_result_953 = invoke(stypy.reporting.localization.Localization(__file__, 630, 20), annotate_type_945, *[line_947, column_949, member_name_950, member_type_951], **kwargs_952)
                
                # SSA join for if statement (line 629)
                module_type_store = module_type_store.join_ssa_context()
                

            # Getting the type of 'None' (line 632)
            None_954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 23), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 632)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 16), 'stypy_return_type', None_954)
            # SSA join for if statement (line 625)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA branch for the except part of a try statement (line 606)
        # SSA branch for the except 'Exception' branch of a try statement (line 606)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'Exception' (line 633)
        Exception_955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 15), 'Exception')
        # Assigning a type to the variable 'exc' (line 633)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 8), 'exc', Exception_955)
        
        # Call to TypeError(...): (line 634)
        # Processing the call arguments (line 634)
        # Getting the type of 'localization' (line 634)
        localization_957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 29), 'localization', False)
        
        # Call to format(...): (line 635)
        # Processing the call arguments (line 635)
        
        # Call to __repr__(...): (line 635)
        # Processing the call keyword arguments (line 635)
        kwargs_962 = {}
        # Getting the type of 'self' (line 635)
        self_960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 80), 'self', False)
        # Obtaining the member '__repr__' of a type (line 635)
        repr___961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 80), self_960, '__repr__')
        # Calling __repr__(args, kwargs) (line 635)
        repr___call_result_963 = invoke(stypy.reporting.localization.Localization(__file__, 635, 80), repr___961, *[], **kwargs_962)
        
        
        # Call to str(...): (line 635)
        # Processing the call arguments (line 635)
        # Getting the type of 'exc' (line 635)
        exc_965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 101), 'exc', False)
        # Processing the call keyword arguments (line 635)
        kwargs_966 = {}
        # Getting the type of 'str' (line 635)
        str_964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 97), 'str', False)
        # Calling str(args, kwargs) (line 635)
        str_call_result_967 = invoke(stypy.reporting.localization.Localization(__file__, 635, 97), str_964, *[exc_965], **kwargs_966)
        
        # Processing the call keyword arguments (line 635)
        kwargs_968 = {}
        str_958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 29), 'str', "Cannot modify the structure of '{0}': {1}")
        # Obtaining the member 'format' of a type (line 635)
        format_959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 29), str_958, 'format')
        # Calling format(args, kwargs) (line 635)
        format_call_result_969 = invoke(stypy.reporting.localization.Localization(__file__, 635, 29), format_959, *[repr___call_result_963, str_call_result_967], **kwargs_968)
        
        # Processing the call keyword arguments (line 634)
        kwargs_970 = {}
        # Getting the type of 'TypeError' (line 634)
        TypeError_956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 19), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 634)
        TypeError_call_result_971 = invoke(stypy.reporting.localization.Localization(__file__, 634, 19), TypeError_956, *[localization_957, format_call_result_969], **kwargs_970)
        
        # Assigning a type to the variable 'stypy_return_type' (line 634)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 12), 'stypy_return_type', TypeError_call_result_971)
        # SSA join for try-except statement (line 606)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to TypeError(...): (line 637)
        # Processing the call arguments (line 637)
        # Getting the type of 'localization' (line 637)
        localization_973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 25), 'localization', False)
        str_974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 25), 'str', 'Cannot modify the structure of a python library type or instance')
        # Processing the call keyword arguments (line 637)
        kwargs_975 = {}
        # Getting the type of 'TypeError' (line 637)
        TypeError_972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 15), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 637)
        TypeError_call_result_976 = invoke(stypy.reporting.localization.Localization(__file__, 637, 15), TypeError_972, *[localization_973, str_974], **kwargs_975)
        
        # Assigning a type to the variable 'stypy_return_type' (line 637)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 8), 'stypy_return_type', TypeError_call_result_976)
        
        # ################# End of 'set_type_of_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_type_of_member' in the type store
        # Getting the type of 'stypy_return_type' (line 596)
        stypy_return_type_977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_977)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_type_of_member'
        return stypy_return_type_977


    @norecursion
    def invoke(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'invoke'
        module_type_store = module_type_store.open_function_context('invoke', 642, 4, False)
        # Assigning a type to the variable 'self' (line 643)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.invoke.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.invoke.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.invoke.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.invoke.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.invoke')
        TypeInferenceProxy.invoke.__dict__.__setitem__('stypy_param_names_list', ['localization'])
        TypeInferenceProxy.invoke.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        TypeInferenceProxy.invoke.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        TypeInferenceProxy.invoke.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.invoke.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.invoke.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.invoke.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.invoke', ['localization'], 'args', 'kwargs', defaults, varargs, kwargs)

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

        str_978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, (-1)), 'str', '\n        Invoke a callable member of the hold python entity with the specified arguments and keyword arguments.\n        NOTE: Calling a class constructor returns a type proxy of an instance of this class. But an instance object\n        is only stored if the instances of this class support structural reflection.\n\n        :param localization: Call localization data\n        :param args: Arguments of the call\n        :param kwargs: Keyword arguments of the call\n        :return:\n        ')
        
        
        # Call to callable(...): (line 655)
        # Processing the call arguments (line 655)
        # Getting the type of 'self' (line 655)
        self_980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 24), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 655)
        python_entity_981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 24), self_980, 'python_entity')
        # Processing the call keyword arguments (line 655)
        kwargs_982 = {}
        # Getting the type of 'callable' (line 655)
        callable_979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 15), 'callable', False)
        # Calling callable(args, kwargs) (line 655)
        callable_call_result_983 = invoke(stypy.reporting.localization.Localization(__file__, 655, 15), callable_979, *[python_entity_981], **kwargs_982)
        
        # Applying the 'not' unary operator (line 655)
        result_not__984 = python_operator(stypy.reporting.localization.Localization(__file__, 655, 11), 'not', callable_call_result_983)
        
        # Testing if the type of an if condition is none (line 655)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 655, 8), result_not__984):
            
            # Assigning a Call to a Name (line 659):
            
            # Assigning a Call to a Name (line 659):
            
            # Call to perform_call(...): (line 659)
            # Processing the call arguments (line 659)
            # Getting the type of 'self' (line 659)
            self_993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 54), 'self', False)
            # Getting the type of 'self' (line 659)
            self_994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 60), 'self', False)
            # Obtaining the member 'python_entity' of a type (line 659)
            python_entity_995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 60), self_994, 'python_entity')
            # Getting the type of 'localization' (line 659)
            localization_996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 80), 'localization', False)
            # Getting the type of 'args' (line 659)
            args_997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 95), 'args', False)
            # Processing the call keyword arguments (line 659)
            # Getting the type of 'kwargs' (line 659)
            kwargs_998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 103), 'kwargs', False)
            kwargs_999 = {'kwargs_998': kwargs_998}
            # Getting the type of 'call_handlers_copy' (line 659)
            call_handlers_copy_991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 22), 'call_handlers_copy', False)
            # Obtaining the member 'perform_call' of a type (line 659)
            perform_call_992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 22), call_handlers_copy_991, 'perform_call')
            # Calling perform_call(args, kwargs) (line 659)
            perform_call_call_result_1000 = invoke(stypy.reporting.localization.Localization(__file__, 659, 22), perform_call_992, *[self_993, python_entity_995, localization_996, args_997], **kwargs_999)
            
            # Assigning a type to the variable 'result_' (line 659)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 12), 'result_', perform_call_call_result_1000)
            
            # Evaluating a boolean operation
            
            # Call to is_type_changing_method(...): (line 661)
            # Processing the call arguments (line 661)
            # Getting the type of 'self' (line 661)
            self_1003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 60), 'self', False)
            # Obtaining the member 'name' of a type (line 661)
            name_1004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 60), self_1003, 'name')
            # Processing the call keyword arguments (line 661)
            kwargs_1005 = {}
            # Getting the type of 'TypeAnnotationRecord' (line 661)
            TypeAnnotationRecord_1001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 15), 'TypeAnnotationRecord', False)
            # Obtaining the member 'is_type_changing_method' of a type (line 661)
            is_type_changing_method_1002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 15), TypeAnnotationRecord_1001, 'is_type_changing_method')
            # Calling is_type_changing_method(args, kwargs) (line 661)
            is_type_changing_method_call_result_1006 = invoke(stypy.reporting.localization.Localization(__file__, 661, 15), is_type_changing_method_1002, *[name_1004], **kwargs_1005)
            
            # Getting the type of 'self' (line 661)
            self_1007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 75), 'self')
            # Obtaining the member 'annotate_types' of a type (line 661)
            annotate_types_1008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 75), self_1007, 'annotate_types')
            # Applying the binary operator 'and' (line 661)
            result_and_keyword_1009 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 15), 'and', is_type_changing_method_call_result_1006, annotate_types_1008)
            
            # Testing if the type of an if condition is none (line 661)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 661, 12), result_and_keyword_1009):
                pass
            else:
                
                # Testing the type of an if condition (line 661)
                if_condition_1010 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 661, 12), result_and_keyword_1009)
                # Assigning a type to the variable 'if_condition_1010' (line 661)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 12), 'if_condition_1010', if_condition_1010)
                # SSA begins for if statement (line 661)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to __annotate_type(...): (line 662)
                # Processing the call arguments (line 662)
                # Getting the type of 'localization' (line 662)
                localization_1013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 37), 'localization', False)
                # Obtaining the member 'line' of a type (line 662)
                line_1014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 37), localization_1013, 'line')
                # Getting the type of 'localization' (line 662)
                localization_1015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 56), 'localization', False)
                # Obtaining the member 'column' of a type (line 662)
                column_1016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 56), localization_1015, 'column')
                # Getting the type of 'self' (line 662)
                self_1017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 77), 'self', False)
                # Obtaining the member 'parent_proxy' of a type (line 662)
                parent_proxy_1018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 77), self_1017, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 662)
                name_1019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 77), parent_proxy_1018, 'name')
                
                # Call to get_python_type(...): (line 663)
                # Processing the call keyword arguments (line 663)
                kwargs_1023 = {}
                # Getting the type of 'self' (line 663)
                self_1020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 37), 'self', False)
                # Obtaining the member 'parent_proxy' of a type (line 663)
                parent_proxy_1021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 37), self_1020, 'parent_proxy')
                # Obtaining the member 'get_python_type' of a type (line 663)
                get_python_type_1022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 37), parent_proxy_1021, 'get_python_type')
                # Calling get_python_type(args, kwargs) (line 663)
                get_python_type_call_result_1024 = invoke(stypy.reporting.localization.Localization(__file__, 663, 37), get_python_type_1022, *[], **kwargs_1023)
                
                # Processing the call keyword arguments (line 662)
                kwargs_1025 = {}
                # Getting the type of 'self' (line 662)
                self_1011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 16), 'self', False)
                # Obtaining the member '__annotate_type' of a type (line 662)
                annotate_type_1012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 16), self_1011, '__annotate_type')
                # Calling __annotate_type(args, kwargs) (line 662)
                annotate_type_call_result_1026 = invoke(stypy.reporting.localization.Localization(__file__, 662, 16), annotate_type_1012, *[line_1014, column_1016, name_1019, get_python_type_call_result_1024], **kwargs_1025)
                
                # SSA join for if statement (line 661)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Type idiom detected: calculating its left and rigth part (line 666)
            # Getting the type of 'TypeError' (line 666)
            TypeError_1027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 35), 'TypeError')
            # Getting the type of 'result_' (line 666)
            result__1028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 26), 'result_')
            
            (may_be_1029, more_types_in_union_1030) = may_be_subtype(TypeError_1027, result__1028)

            if may_be_1029:

                if more_types_in_union_1030:
                    # Runtime conditional SSA (line 666)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'result_' (line 666)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 12), 'result_', remove_not_subtype_from_union(result__1028, TypeError))
                # Getting the type of 'result_' (line 667)
                result__1031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 23), 'result_')
                # Assigning a type to the variable 'stypy_return_type' (line 667)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 16), 'stypy_return_type', result__1031)

                if more_types_in_union_1030:
                    # SSA join for if statement (line 666)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Call to isinstance(...): (line 669)
            # Processing the call arguments (line 669)
            # Getting the type of 'result_' (line 669)
            result__1033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 26), 'result_', False)
            # Getting the type of 'Type' (line 669)
            Type_1034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 35), 'Type', False)
            # Processing the call keyword arguments (line 669)
            kwargs_1035 = {}
            # Getting the type of 'isinstance' (line 669)
            isinstance_1032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 669)
            isinstance_call_result_1036 = invoke(stypy.reporting.localization.Localization(__file__, 669, 15), isinstance_1032, *[result__1033, Type_1034], **kwargs_1035)
            
            # Testing if the type of an if condition is none (line 669)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 669, 12), isinstance_call_result_1036):
                pass
            else:
                
                # Testing the type of an if condition (line 669)
                if_condition_1037 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 669, 12), isinstance_call_result_1036)
                # Assigning a type to the variable 'if_condition_1037' (line 669)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 12), 'if_condition_1037', if_condition_1037)
                # SSA begins for if statement (line 669)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to set_type_instance(...): (line 670)
                # Processing the call arguments (line 670)
                # Getting the type of 'True' (line 670)
                True_1040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 42), 'True', False)
                # Processing the call keyword arguments (line 670)
                kwargs_1041 = {}
                # Getting the type of 'result_' (line 670)
                result__1038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 16), 'result_', False)
                # Obtaining the member 'set_type_instance' of a type (line 670)
                set_type_instance_1039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 16), result__1038, 'set_type_instance')
                # Calling set_type_instance(args, kwargs) (line 670)
                set_type_instance_call_result_1042 = invoke(stypy.reporting.localization.Localization(__file__, 670, 16), set_type_instance_1039, *[True_1040], **kwargs_1041)
                
                # Getting the type of 'result_' (line 671)
                result__1043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 23), 'result_')
                # Assigning a type to the variable 'stypy_return_type' (line 671)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 16), 'stypy_return_type', result__1043)
                # SSA join for if statement (line 669)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to isclass(...): (line 675)
            # Processing the call arguments (line 675)
            # Getting the type of 'self' (line 675)
            self_1046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 31), 'self', False)
            # Obtaining the member 'python_entity' of a type (line 675)
            python_entity_1047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 31), self_1046, 'python_entity')
            # Processing the call keyword arguments (line 675)
            kwargs_1048 = {}
            # Getting the type of 'inspect' (line 675)
            inspect_1044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 15), 'inspect', False)
            # Obtaining the member 'isclass' of a type (line 675)
            isclass_1045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 15), inspect_1044, 'isclass')
            # Calling isclass(args, kwargs) (line 675)
            isclass_call_result_1049 = invoke(stypy.reporting.localization.Localization(__file__, 675, 15), isclass_1045, *[python_entity_1047], **kwargs_1048)
            
            # Testing if the type of an if condition is none (line 675)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 675, 12), isclass_call_result_1049):
                
                # Assigning a Name to a Name (line 685):
                
                # Assigning a Name to a Name (line 685):
                # Getting the type of 'None' (line 685)
                None_1063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 27), 'None')
                # Assigning a type to the variable 'instance' (line 685)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 16), 'instance', None_1063)
            else:
                
                # Testing the type of an if condition (line 675)
                if_condition_1050 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 675, 12), isclass_call_result_1049)
                # Assigning a type to the variable 'if_condition_1050' (line 675)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 12), 'if_condition_1050', if_condition_1050)
                # SSA begins for if statement (line 675)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to supports_structural_reflection(...): (line 677)
                # Processing the call arguments (line 677)
                # Getting the type of 'result_' (line 677)
                result__1053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 87), 'result_', False)
                # Processing the call keyword arguments (line 677)
                kwargs_1054 = {}
                # Getting the type of 'type_inference_proxy_management_copy' (line 677)
                type_inference_proxy_management_copy_1051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 19), 'type_inference_proxy_management_copy', False)
                # Obtaining the member 'supports_structural_reflection' of a type (line 677)
                supports_structural_reflection_1052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 19), type_inference_proxy_management_copy_1051, 'supports_structural_reflection')
                # Calling supports_structural_reflection(args, kwargs) (line 677)
                supports_structural_reflection_call_result_1055 = invoke(stypy.reporting.localization.Localization(__file__, 677, 19), supports_structural_reflection_1052, *[result__1053], **kwargs_1054)
                
                # Testing if the type of an if condition is none (line 677)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 677, 16), supports_structural_reflection_call_result_1055):
                    
                    # Assigning a Name to a Name (line 683):
                    
                    # Assigning a Name to a Name (line 683):
                    # Getting the type of 'None' (line 683)
                    None_1062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 31), 'None')
                    # Assigning a type to the variable 'instance' (line 683)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 20), 'instance', None_1062)
                else:
                    
                    # Testing the type of an if condition (line 677)
                    if_condition_1056 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 677, 16), supports_structural_reflection_call_result_1055)
                    # Assigning a type to the variable 'if_condition_1056' (line 677)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 16), 'if_condition_1056', if_condition_1056)
                    # SSA begins for if statement (line 677)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Name (line 678):
                    
                    # Assigning a Name to a Name (line 678):
                    # Getting the type of 'result_' (line 678)
                    result__1057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 31), 'result_')
                    # Assigning a type to the variable 'instance' (line 678)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 20), 'instance', result__1057)
                    
                    # Assigning a Call to a Name (line 681):
                    
                    # Assigning a Call to a Name (line 681):
                    
                    # Call to type(...): (line 681)
                    # Processing the call arguments (line 681)
                    # Getting the type of 'result_' (line 681)
                    result__1059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 35), 'result_', False)
                    # Processing the call keyword arguments (line 681)
                    kwargs_1060 = {}
                    # Getting the type of 'type' (line 681)
                    type_1058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 30), 'type', False)
                    # Calling type(args, kwargs) (line 681)
                    type_call_result_1061 = invoke(stypy.reporting.localization.Localization(__file__, 681, 30), type_1058, *[result__1059], **kwargs_1060)
                    
                    # Assigning a type to the variable 'result_' (line 681)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 20), 'result_', type_call_result_1061)
                    # SSA branch for the else part of an if statement (line 677)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Name to a Name (line 683):
                    
                    # Assigning a Name to a Name (line 683):
                    # Getting the type of 'None' (line 683)
                    None_1062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 31), 'None')
                    # Assigning a type to the variable 'instance' (line 683)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 20), 'instance', None_1062)
                    # SSA join for if statement (line 677)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA branch for the else part of an if statement (line 675)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Name (line 685):
                
                # Assigning a Name to a Name (line 685):
                # Getting the type of 'None' (line 685)
                None_1063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 27), 'None')
                # Assigning a type to the variable 'instance' (line 685)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 16), 'instance', None_1063)
                # SSA join for if statement (line 675)
                module_type_store = module_type_store.join_ssa_context()
                

            
            
            # Call to isinstance(...): (line 688)
            # Processing the call arguments (line 688)
            # Getting the type of 'result_' (line 688)
            result__1065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 30), 'result_', False)
            # Getting the type of 'Type' (line 688)
            Type_1066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 39), 'Type', False)
            # Processing the call keyword arguments (line 688)
            kwargs_1067 = {}
            # Getting the type of 'isinstance' (line 688)
            isinstance_1064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 688)
            isinstance_call_result_1068 = invoke(stypy.reporting.localization.Localization(__file__, 688, 19), isinstance_1064, *[result__1065, Type_1066], **kwargs_1067)
            
            # Applying the 'not' unary operator (line 688)
            result_not__1069 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 15), 'not', isinstance_call_result_1068)
            
            # Testing if the type of an if condition is none (line 688)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 688, 12), result_not__1069):
                
                # Call to set_type_instance(...): (line 694)
                # Processing the call arguments (line 694)
                # Getting the type of 'True' (line 694)
                True_1086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 42), 'True', False)
                # Processing the call keyword arguments (line 694)
                kwargs_1087 = {}
                # Getting the type of 'result_' (line 694)
                result__1084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 16), 'result_', False)
                # Obtaining the member 'set_type_instance' of a type (line 694)
                set_type_instance_1085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 16), result__1084, 'set_type_instance')
                # Calling set_type_instance(args, kwargs) (line 694)
                set_type_instance_call_result_1088 = invoke(stypy.reporting.localization.Localization(__file__, 694, 16), set_type_instance_1085, *[True_1086], **kwargs_1087)
                
                # Getting the type of 'result_' (line 695)
                result__1089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 23), 'result_')
                # Assigning a type to the variable 'stypy_return_type' (line 695)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 16), 'stypy_return_type', result__1089)
            else:
                
                # Testing the type of an if condition (line 688)
                if_condition_1070 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 688, 12), result_not__1069)
                # Assigning a type to the variable 'if_condition_1070' (line 688)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 12), 'if_condition_1070', if_condition_1070)
                # SSA begins for if statement (line 688)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 689):
                
                # Assigning a Call to a Name (line 689):
                
                # Call to instance(...): (line 689)
                # Processing the call arguments (line 689)
                # Getting the type of 'result_' (line 689)
                result__1073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 50), 'result_', False)
                # Processing the call keyword arguments (line 689)
                # Getting the type of 'instance' (line 689)
                instance_1074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 68), 'instance', False)
                keyword_1075 = instance_1074
                kwargs_1076 = {'instance': keyword_1075}
                # Getting the type of 'TypeInferenceProxy' (line 689)
                TypeInferenceProxy_1071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 22), 'TypeInferenceProxy', False)
                # Obtaining the member 'instance' of a type (line 689)
                instance_1072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 22), TypeInferenceProxy_1071, 'instance')
                # Calling instance(args, kwargs) (line 689)
                instance_call_result_1077 = invoke(stypy.reporting.localization.Localization(__file__, 689, 22), instance_1072, *[result__1073], **kwargs_1076)
                
                # Assigning a type to the variable 'ret' (line 689)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 16), 'ret', instance_call_result_1077)
                
                # Call to set_type_instance(...): (line 690)
                # Processing the call arguments (line 690)
                # Getting the type of 'True' (line 690)
                True_1080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 38), 'True', False)
                # Processing the call keyword arguments (line 690)
                kwargs_1081 = {}
                # Getting the type of 'ret' (line 690)
                ret_1078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 16), 'ret', False)
                # Obtaining the member 'set_type_instance' of a type (line 690)
                set_type_instance_1079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 16), ret_1078, 'set_type_instance')
                # Calling set_type_instance(args, kwargs) (line 690)
                set_type_instance_call_result_1082 = invoke(stypy.reporting.localization.Localization(__file__, 690, 16), set_type_instance_1079, *[True_1080], **kwargs_1081)
                
                # Getting the type of 'ret' (line 692)
                ret_1083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 23), 'ret')
                # Assigning a type to the variable 'stypy_return_type' (line 692)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 16), 'stypy_return_type', ret_1083)
                # SSA branch for the else part of an if statement (line 688)
                module_type_store.open_ssa_branch('else')
                
                # Call to set_type_instance(...): (line 694)
                # Processing the call arguments (line 694)
                # Getting the type of 'True' (line 694)
                True_1086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 42), 'True', False)
                # Processing the call keyword arguments (line 694)
                kwargs_1087 = {}
                # Getting the type of 'result_' (line 694)
                result__1084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 16), 'result_', False)
                # Obtaining the member 'set_type_instance' of a type (line 694)
                set_type_instance_1085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 16), result__1084, 'set_type_instance')
                # Calling set_type_instance(args, kwargs) (line 694)
                set_type_instance_call_result_1088 = invoke(stypy.reporting.localization.Localization(__file__, 694, 16), set_type_instance_1085, *[True_1086], **kwargs_1087)
                
                # Getting the type of 'result_' (line 695)
                result__1089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 23), 'result_')
                # Assigning a type to the variable 'stypy_return_type' (line 695)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 16), 'stypy_return_type', result__1089)
                # SSA join for if statement (line 688)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 655)
            if_condition_985 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 655, 8), result_not__984)
            # Assigning a type to the variable 'if_condition_985' (line 655)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 8), 'if_condition_985', if_condition_985)
            # SSA begins for if statement (line 655)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 656)
            # Processing the call arguments (line 656)
            # Getting the type of 'localization' (line 656)
            localization_987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 29), 'localization', False)
            str_988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 43), 'str', 'Cannot invoke on a non callable type')
            # Processing the call keyword arguments (line 656)
            kwargs_989 = {}
            # Getting the type of 'TypeError' (line 656)
            TypeError_986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 656)
            TypeError_call_result_990 = invoke(stypy.reporting.localization.Localization(__file__, 656, 19), TypeError_986, *[localization_987, str_988], **kwargs_989)
            
            # Assigning a type to the variable 'stypy_return_type' (line 656)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 12), 'stypy_return_type', TypeError_call_result_990)
            # SSA branch for the else part of an if statement (line 655)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 659):
            
            # Assigning a Call to a Name (line 659):
            
            # Call to perform_call(...): (line 659)
            # Processing the call arguments (line 659)
            # Getting the type of 'self' (line 659)
            self_993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 54), 'self', False)
            # Getting the type of 'self' (line 659)
            self_994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 60), 'self', False)
            # Obtaining the member 'python_entity' of a type (line 659)
            python_entity_995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 60), self_994, 'python_entity')
            # Getting the type of 'localization' (line 659)
            localization_996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 80), 'localization', False)
            # Getting the type of 'args' (line 659)
            args_997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 95), 'args', False)
            # Processing the call keyword arguments (line 659)
            # Getting the type of 'kwargs' (line 659)
            kwargs_998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 103), 'kwargs', False)
            kwargs_999 = {'kwargs_998': kwargs_998}
            # Getting the type of 'call_handlers_copy' (line 659)
            call_handlers_copy_991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 22), 'call_handlers_copy', False)
            # Obtaining the member 'perform_call' of a type (line 659)
            perform_call_992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 22), call_handlers_copy_991, 'perform_call')
            # Calling perform_call(args, kwargs) (line 659)
            perform_call_call_result_1000 = invoke(stypy.reporting.localization.Localization(__file__, 659, 22), perform_call_992, *[self_993, python_entity_995, localization_996, args_997], **kwargs_999)
            
            # Assigning a type to the variable 'result_' (line 659)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 12), 'result_', perform_call_call_result_1000)
            
            # Evaluating a boolean operation
            
            # Call to is_type_changing_method(...): (line 661)
            # Processing the call arguments (line 661)
            # Getting the type of 'self' (line 661)
            self_1003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 60), 'self', False)
            # Obtaining the member 'name' of a type (line 661)
            name_1004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 60), self_1003, 'name')
            # Processing the call keyword arguments (line 661)
            kwargs_1005 = {}
            # Getting the type of 'TypeAnnotationRecord' (line 661)
            TypeAnnotationRecord_1001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 15), 'TypeAnnotationRecord', False)
            # Obtaining the member 'is_type_changing_method' of a type (line 661)
            is_type_changing_method_1002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 15), TypeAnnotationRecord_1001, 'is_type_changing_method')
            # Calling is_type_changing_method(args, kwargs) (line 661)
            is_type_changing_method_call_result_1006 = invoke(stypy.reporting.localization.Localization(__file__, 661, 15), is_type_changing_method_1002, *[name_1004], **kwargs_1005)
            
            # Getting the type of 'self' (line 661)
            self_1007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 75), 'self')
            # Obtaining the member 'annotate_types' of a type (line 661)
            annotate_types_1008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 75), self_1007, 'annotate_types')
            # Applying the binary operator 'and' (line 661)
            result_and_keyword_1009 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 15), 'and', is_type_changing_method_call_result_1006, annotate_types_1008)
            
            # Testing if the type of an if condition is none (line 661)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 661, 12), result_and_keyword_1009):
                pass
            else:
                
                # Testing the type of an if condition (line 661)
                if_condition_1010 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 661, 12), result_and_keyword_1009)
                # Assigning a type to the variable 'if_condition_1010' (line 661)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 12), 'if_condition_1010', if_condition_1010)
                # SSA begins for if statement (line 661)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to __annotate_type(...): (line 662)
                # Processing the call arguments (line 662)
                # Getting the type of 'localization' (line 662)
                localization_1013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 37), 'localization', False)
                # Obtaining the member 'line' of a type (line 662)
                line_1014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 37), localization_1013, 'line')
                # Getting the type of 'localization' (line 662)
                localization_1015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 56), 'localization', False)
                # Obtaining the member 'column' of a type (line 662)
                column_1016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 56), localization_1015, 'column')
                # Getting the type of 'self' (line 662)
                self_1017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 77), 'self', False)
                # Obtaining the member 'parent_proxy' of a type (line 662)
                parent_proxy_1018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 77), self_1017, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 662)
                name_1019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 77), parent_proxy_1018, 'name')
                
                # Call to get_python_type(...): (line 663)
                # Processing the call keyword arguments (line 663)
                kwargs_1023 = {}
                # Getting the type of 'self' (line 663)
                self_1020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 37), 'self', False)
                # Obtaining the member 'parent_proxy' of a type (line 663)
                parent_proxy_1021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 37), self_1020, 'parent_proxy')
                # Obtaining the member 'get_python_type' of a type (line 663)
                get_python_type_1022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 37), parent_proxy_1021, 'get_python_type')
                # Calling get_python_type(args, kwargs) (line 663)
                get_python_type_call_result_1024 = invoke(stypy.reporting.localization.Localization(__file__, 663, 37), get_python_type_1022, *[], **kwargs_1023)
                
                # Processing the call keyword arguments (line 662)
                kwargs_1025 = {}
                # Getting the type of 'self' (line 662)
                self_1011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 16), 'self', False)
                # Obtaining the member '__annotate_type' of a type (line 662)
                annotate_type_1012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 16), self_1011, '__annotate_type')
                # Calling __annotate_type(args, kwargs) (line 662)
                annotate_type_call_result_1026 = invoke(stypy.reporting.localization.Localization(__file__, 662, 16), annotate_type_1012, *[line_1014, column_1016, name_1019, get_python_type_call_result_1024], **kwargs_1025)
                
                # SSA join for if statement (line 661)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Type idiom detected: calculating its left and rigth part (line 666)
            # Getting the type of 'TypeError' (line 666)
            TypeError_1027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 35), 'TypeError')
            # Getting the type of 'result_' (line 666)
            result__1028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 26), 'result_')
            
            (may_be_1029, more_types_in_union_1030) = may_be_subtype(TypeError_1027, result__1028)

            if may_be_1029:

                if more_types_in_union_1030:
                    # Runtime conditional SSA (line 666)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'result_' (line 666)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 12), 'result_', remove_not_subtype_from_union(result__1028, TypeError))
                # Getting the type of 'result_' (line 667)
                result__1031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 23), 'result_')
                # Assigning a type to the variable 'stypy_return_type' (line 667)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 16), 'stypy_return_type', result__1031)

                if more_types_in_union_1030:
                    # SSA join for if statement (line 666)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Call to isinstance(...): (line 669)
            # Processing the call arguments (line 669)
            # Getting the type of 'result_' (line 669)
            result__1033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 26), 'result_', False)
            # Getting the type of 'Type' (line 669)
            Type_1034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 35), 'Type', False)
            # Processing the call keyword arguments (line 669)
            kwargs_1035 = {}
            # Getting the type of 'isinstance' (line 669)
            isinstance_1032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 669)
            isinstance_call_result_1036 = invoke(stypy.reporting.localization.Localization(__file__, 669, 15), isinstance_1032, *[result__1033, Type_1034], **kwargs_1035)
            
            # Testing if the type of an if condition is none (line 669)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 669, 12), isinstance_call_result_1036):
                pass
            else:
                
                # Testing the type of an if condition (line 669)
                if_condition_1037 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 669, 12), isinstance_call_result_1036)
                # Assigning a type to the variable 'if_condition_1037' (line 669)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 12), 'if_condition_1037', if_condition_1037)
                # SSA begins for if statement (line 669)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to set_type_instance(...): (line 670)
                # Processing the call arguments (line 670)
                # Getting the type of 'True' (line 670)
                True_1040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 42), 'True', False)
                # Processing the call keyword arguments (line 670)
                kwargs_1041 = {}
                # Getting the type of 'result_' (line 670)
                result__1038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 16), 'result_', False)
                # Obtaining the member 'set_type_instance' of a type (line 670)
                set_type_instance_1039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 16), result__1038, 'set_type_instance')
                # Calling set_type_instance(args, kwargs) (line 670)
                set_type_instance_call_result_1042 = invoke(stypy.reporting.localization.Localization(__file__, 670, 16), set_type_instance_1039, *[True_1040], **kwargs_1041)
                
                # Getting the type of 'result_' (line 671)
                result__1043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 23), 'result_')
                # Assigning a type to the variable 'stypy_return_type' (line 671)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 16), 'stypy_return_type', result__1043)
                # SSA join for if statement (line 669)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to isclass(...): (line 675)
            # Processing the call arguments (line 675)
            # Getting the type of 'self' (line 675)
            self_1046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 31), 'self', False)
            # Obtaining the member 'python_entity' of a type (line 675)
            python_entity_1047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 31), self_1046, 'python_entity')
            # Processing the call keyword arguments (line 675)
            kwargs_1048 = {}
            # Getting the type of 'inspect' (line 675)
            inspect_1044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 15), 'inspect', False)
            # Obtaining the member 'isclass' of a type (line 675)
            isclass_1045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 15), inspect_1044, 'isclass')
            # Calling isclass(args, kwargs) (line 675)
            isclass_call_result_1049 = invoke(stypy.reporting.localization.Localization(__file__, 675, 15), isclass_1045, *[python_entity_1047], **kwargs_1048)
            
            # Testing if the type of an if condition is none (line 675)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 675, 12), isclass_call_result_1049):
                
                # Assigning a Name to a Name (line 685):
                
                # Assigning a Name to a Name (line 685):
                # Getting the type of 'None' (line 685)
                None_1063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 27), 'None')
                # Assigning a type to the variable 'instance' (line 685)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 16), 'instance', None_1063)
            else:
                
                # Testing the type of an if condition (line 675)
                if_condition_1050 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 675, 12), isclass_call_result_1049)
                # Assigning a type to the variable 'if_condition_1050' (line 675)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 12), 'if_condition_1050', if_condition_1050)
                # SSA begins for if statement (line 675)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to supports_structural_reflection(...): (line 677)
                # Processing the call arguments (line 677)
                # Getting the type of 'result_' (line 677)
                result__1053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 87), 'result_', False)
                # Processing the call keyword arguments (line 677)
                kwargs_1054 = {}
                # Getting the type of 'type_inference_proxy_management_copy' (line 677)
                type_inference_proxy_management_copy_1051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 19), 'type_inference_proxy_management_copy', False)
                # Obtaining the member 'supports_structural_reflection' of a type (line 677)
                supports_structural_reflection_1052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 19), type_inference_proxy_management_copy_1051, 'supports_structural_reflection')
                # Calling supports_structural_reflection(args, kwargs) (line 677)
                supports_structural_reflection_call_result_1055 = invoke(stypy.reporting.localization.Localization(__file__, 677, 19), supports_structural_reflection_1052, *[result__1053], **kwargs_1054)
                
                # Testing if the type of an if condition is none (line 677)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 677, 16), supports_structural_reflection_call_result_1055):
                    
                    # Assigning a Name to a Name (line 683):
                    
                    # Assigning a Name to a Name (line 683):
                    # Getting the type of 'None' (line 683)
                    None_1062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 31), 'None')
                    # Assigning a type to the variable 'instance' (line 683)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 20), 'instance', None_1062)
                else:
                    
                    # Testing the type of an if condition (line 677)
                    if_condition_1056 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 677, 16), supports_structural_reflection_call_result_1055)
                    # Assigning a type to the variable 'if_condition_1056' (line 677)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 16), 'if_condition_1056', if_condition_1056)
                    # SSA begins for if statement (line 677)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Name (line 678):
                    
                    # Assigning a Name to a Name (line 678):
                    # Getting the type of 'result_' (line 678)
                    result__1057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 31), 'result_')
                    # Assigning a type to the variable 'instance' (line 678)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 20), 'instance', result__1057)
                    
                    # Assigning a Call to a Name (line 681):
                    
                    # Assigning a Call to a Name (line 681):
                    
                    # Call to type(...): (line 681)
                    # Processing the call arguments (line 681)
                    # Getting the type of 'result_' (line 681)
                    result__1059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 35), 'result_', False)
                    # Processing the call keyword arguments (line 681)
                    kwargs_1060 = {}
                    # Getting the type of 'type' (line 681)
                    type_1058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 30), 'type', False)
                    # Calling type(args, kwargs) (line 681)
                    type_call_result_1061 = invoke(stypy.reporting.localization.Localization(__file__, 681, 30), type_1058, *[result__1059], **kwargs_1060)
                    
                    # Assigning a type to the variable 'result_' (line 681)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 20), 'result_', type_call_result_1061)
                    # SSA branch for the else part of an if statement (line 677)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Name to a Name (line 683):
                    
                    # Assigning a Name to a Name (line 683):
                    # Getting the type of 'None' (line 683)
                    None_1062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 31), 'None')
                    # Assigning a type to the variable 'instance' (line 683)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 20), 'instance', None_1062)
                    # SSA join for if statement (line 677)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA branch for the else part of an if statement (line 675)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Name (line 685):
                
                # Assigning a Name to a Name (line 685):
                # Getting the type of 'None' (line 685)
                None_1063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 27), 'None')
                # Assigning a type to the variable 'instance' (line 685)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 16), 'instance', None_1063)
                # SSA join for if statement (line 675)
                module_type_store = module_type_store.join_ssa_context()
                

            
            
            # Call to isinstance(...): (line 688)
            # Processing the call arguments (line 688)
            # Getting the type of 'result_' (line 688)
            result__1065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 30), 'result_', False)
            # Getting the type of 'Type' (line 688)
            Type_1066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 39), 'Type', False)
            # Processing the call keyword arguments (line 688)
            kwargs_1067 = {}
            # Getting the type of 'isinstance' (line 688)
            isinstance_1064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 688)
            isinstance_call_result_1068 = invoke(stypy.reporting.localization.Localization(__file__, 688, 19), isinstance_1064, *[result__1065, Type_1066], **kwargs_1067)
            
            # Applying the 'not' unary operator (line 688)
            result_not__1069 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 15), 'not', isinstance_call_result_1068)
            
            # Testing if the type of an if condition is none (line 688)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 688, 12), result_not__1069):
                
                # Call to set_type_instance(...): (line 694)
                # Processing the call arguments (line 694)
                # Getting the type of 'True' (line 694)
                True_1086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 42), 'True', False)
                # Processing the call keyword arguments (line 694)
                kwargs_1087 = {}
                # Getting the type of 'result_' (line 694)
                result__1084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 16), 'result_', False)
                # Obtaining the member 'set_type_instance' of a type (line 694)
                set_type_instance_1085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 16), result__1084, 'set_type_instance')
                # Calling set_type_instance(args, kwargs) (line 694)
                set_type_instance_call_result_1088 = invoke(stypy.reporting.localization.Localization(__file__, 694, 16), set_type_instance_1085, *[True_1086], **kwargs_1087)
                
                # Getting the type of 'result_' (line 695)
                result__1089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 23), 'result_')
                # Assigning a type to the variable 'stypy_return_type' (line 695)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 16), 'stypy_return_type', result__1089)
            else:
                
                # Testing the type of an if condition (line 688)
                if_condition_1070 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 688, 12), result_not__1069)
                # Assigning a type to the variable 'if_condition_1070' (line 688)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 12), 'if_condition_1070', if_condition_1070)
                # SSA begins for if statement (line 688)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 689):
                
                # Assigning a Call to a Name (line 689):
                
                # Call to instance(...): (line 689)
                # Processing the call arguments (line 689)
                # Getting the type of 'result_' (line 689)
                result__1073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 50), 'result_', False)
                # Processing the call keyword arguments (line 689)
                # Getting the type of 'instance' (line 689)
                instance_1074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 68), 'instance', False)
                keyword_1075 = instance_1074
                kwargs_1076 = {'instance': keyword_1075}
                # Getting the type of 'TypeInferenceProxy' (line 689)
                TypeInferenceProxy_1071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 22), 'TypeInferenceProxy', False)
                # Obtaining the member 'instance' of a type (line 689)
                instance_1072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 22), TypeInferenceProxy_1071, 'instance')
                # Calling instance(args, kwargs) (line 689)
                instance_call_result_1077 = invoke(stypy.reporting.localization.Localization(__file__, 689, 22), instance_1072, *[result__1073], **kwargs_1076)
                
                # Assigning a type to the variable 'ret' (line 689)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 16), 'ret', instance_call_result_1077)
                
                # Call to set_type_instance(...): (line 690)
                # Processing the call arguments (line 690)
                # Getting the type of 'True' (line 690)
                True_1080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 38), 'True', False)
                # Processing the call keyword arguments (line 690)
                kwargs_1081 = {}
                # Getting the type of 'ret' (line 690)
                ret_1078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 16), 'ret', False)
                # Obtaining the member 'set_type_instance' of a type (line 690)
                set_type_instance_1079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 16), ret_1078, 'set_type_instance')
                # Calling set_type_instance(args, kwargs) (line 690)
                set_type_instance_call_result_1082 = invoke(stypy.reporting.localization.Localization(__file__, 690, 16), set_type_instance_1079, *[True_1080], **kwargs_1081)
                
                # Getting the type of 'ret' (line 692)
                ret_1083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 23), 'ret')
                # Assigning a type to the variable 'stypy_return_type' (line 692)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 16), 'stypy_return_type', ret_1083)
                # SSA branch for the else part of an if statement (line 688)
                module_type_store.open_ssa_branch('else')
                
                # Call to set_type_instance(...): (line 694)
                # Processing the call arguments (line 694)
                # Getting the type of 'True' (line 694)
                True_1086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 42), 'True', False)
                # Processing the call keyword arguments (line 694)
                kwargs_1087 = {}
                # Getting the type of 'result_' (line 694)
                result__1084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 16), 'result_', False)
                # Obtaining the member 'set_type_instance' of a type (line 694)
                set_type_instance_1085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 16), result__1084, 'set_type_instance')
                # Calling set_type_instance(args, kwargs) (line 694)
                set_type_instance_call_result_1088 = invoke(stypy.reporting.localization.Localization(__file__, 694, 16), set_type_instance_1085, *[True_1086], **kwargs_1087)
                
                # Getting the type of 'result_' (line 695)
                result__1089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 23), 'result_')
                # Assigning a type to the variable 'stypy_return_type' (line 695)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 16), 'stypy_return_type', result__1089)
                # SSA join for if statement (line 688)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 655)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'invoke(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'invoke' in the type store
        # Getting the type of 'stypy_return_type' (line 642)
        stypy_return_type_1090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1090)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'invoke'
        return stypy_return_type_1090


    @norecursion
    def __check_undefined_stored_value(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__check_undefined_stored_value'
        module_type_store = module_type_store.open_function_context('__check_undefined_stored_value', 699, 4, False)
        # Assigning a type to the variable 'self' (line 700)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.__check_undefined_stored_value.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.__check_undefined_stored_value.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.__check_undefined_stored_value.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.__check_undefined_stored_value.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.__check_undefined_stored_value')
        TypeInferenceProxy.__check_undefined_stored_value.__dict__.__setitem__('stypy_param_names_list', ['localization', 'value'])
        TypeInferenceProxy.__check_undefined_stored_value.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.__check_undefined_stored_value.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.__check_undefined_stored_value.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.__check_undefined_stored_value.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.__check_undefined_stored_value.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.__check_undefined_stored_value.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.__check_undefined_stored_value', ['localization', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__check_undefined_stored_value', localization, ['localization', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__check_undefined_stored_value(...)' code ##################

        str_1091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, (-1)), 'str', '\n        For represented containers, this method checks if we are trying to store Undefined variables inside them\n        :param localization: Caller information\n        :param value: Value we are trying to store\n        :return:\n        ')
        
        # Assigning a Call to a Tuple (line 706):
        
        # Assigning a Call to a Name:
        
        # Call to contains_an_undefined_type(...): (line 706)
        # Processing the call arguments (line 706)
        # Getting the type of 'value' (line 706)
        value_1094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 96), 'value', False)
        # Processing the call keyword arguments (line 706)
        kwargs_1095 = {}
        # Getting the type of 'TypeInferenceProxy' (line 706)
        TypeInferenceProxy_1092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 50), 'TypeInferenceProxy', False)
        # Obtaining the member 'contains_an_undefined_type' of a type (line 706)
        contains_an_undefined_type_1093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 50), TypeInferenceProxy_1092, 'contains_an_undefined_type')
        # Calling contains_an_undefined_type(args, kwargs) (line 706)
        contains_an_undefined_type_call_result_1096 = invoke(stypy.reporting.localization.Localization(__file__, 706, 50), contains_an_undefined_type_1093, *[value_1094], **kwargs_1095)
        
        # Assigning a type to the variable 'call_assignment_4' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'call_assignment_4', contains_an_undefined_type_call_result_1096)
        
        # Assigning a Call to a Name (line 706):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4' (line 706)
        call_assignment_4_1097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'call_assignment_4', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_1098 = stypy_get_value_from_tuple(call_assignment_4_1097, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_5' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'call_assignment_5', stypy_get_value_from_tuple_call_result_1098)
        
        # Assigning a Name to a Name (line 706):
        # Getting the type of 'call_assignment_5' (line 706)
        call_assignment_5_1099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'call_assignment_5')
        # Assigning a type to the variable 'contains_undefined' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'contains_undefined', call_assignment_5_1099)
        
        # Assigning a Call to a Name (line 706):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4' (line 706)
        call_assignment_4_1100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'call_assignment_4', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_1101 = stypy_get_value_from_tuple(call_assignment_4_1100, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_6' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'call_assignment_6', stypy_get_value_from_tuple_call_result_1101)
        
        # Assigning a Name to a Name (line 706):
        # Getting the type of 'call_assignment_6' (line 706)
        call_assignment_6_1102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'call_assignment_6')
        # Assigning a type to the variable 'more_types_in_value' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 28), 'more_types_in_value', call_assignment_6_1102)
        # Getting the type of 'contains_undefined' (line 707)
        contains_undefined_1103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 11), 'contains_undefined')
        # Testing if the type of an if condition is none (line 707)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 707, 8), contains_undefined_1103):
            pass
        else:
            
            # Testing the type of an if condition (line 707)
            if_condition_1104 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 707, 8), contains_undefined_1103)
            # Assigning a type to the variable 'if_condition_1104' (line 707)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 8), 'if_condition_1104', if_condition_1104)
            # SSA begins for if statement (line 707)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'more_types_in_value' (line 708)
            more_types_in_value_1105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 15), 'more_types_in_value')
            int_1106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 38), 'int')
            # Applying the binary operator '==' (line 708)
            result_eq_1107 = python_operator(stypy.reporting.localization.Localization(__file__, 708, 15), '==', more_types_in_value_1105, int_1106)
            
            # Testing if the type of an if condition is none (line 708)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 708, 12), result_eq_1107):
                
                # Call to instance(...): (line 712)
                # Processing the call arguments (line 712)
                # Getting the type of 'localization' (line 712)
                localization_1121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 37), 'localization', False)
                
                # Call to format(...): (line 713)
                # Processing the call arguments (line 713)
                # Getting the type of 'self' (line 714)
                self_1124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 44), 'self', False)
                # Obtaining the member 'name' of a type (line 714)
                name_1125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 44), self_1124, 'name')
                # Processing the call keyword arguments (line 713)
                kwargs_1126 = {}
                str_1122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 37), 'str', "Potentially storing in '{0}' the value of a previously undefined variable")
                # Obtaining the member 'format' of a type (line 713)
                format_1123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 37), str_1122, 'format')
                # Calling format(args, kwargs) (line 713)
                format_call_result_1127 = invoke(stypy.reporting.localization.Localization(__file__, 713, 37), format_1123, *[name_1125], **kwargs_1126)
                
                # Processing the call keyword arguments (line 712)
                kwargs_1128 = {}
                # Getting the type of 'TypeWarning' (line 712)
                TypeWarning_1119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 16), 'TypeWarning', False)
                # Obtaining the member 'instance' of a type (line 712)
                instance_1120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 16), TypeWarning_1119, 'instance')
                # Calling instance(args, kwargs) (line 712)
                instance_call_result_1129 = invoke(stypy.reporting.localization.Localization(__file__, 712, 16), instance_1120, *[localization_1121, format_call_result_1127], **kwargs_1128)
                
            else:
                
                # Testing the type of an if condition (line 708)
                if_condition_1108 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 708, 12), result_eq_1107)
                # Assigning a type to the variable 'if_condition_1108' (line 708)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 12), 'if_condition_1108', if_condition_1108)
                # SSA begins for if statement (line 708)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to TypeError(...): (line 709)
                # Processing the call arguments (line 709)
                # Getting the type of 'localization' (line 709)
                localization_1110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 26), 'localization', False)
                
                # Call to format(...): (line 709)
                # Processing the call arguments (line 709)
                # Getting the type of 'self' (line 710)
                self_1113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 33), 'self', False)
                # Obtaining the member 'name' of a type (line 710)
                name_1114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 33), self_1113, 'name')
                # Processing the call keyword arguments (line 709)
                kwargs_1115 = {}
                str_1111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 40), 'str', "Storing in '{0}' the value of a previously undefined variable")
                # Obtaining the member 'format' of a type (line 709)
                format_1112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 40), str_1111, 'format')
                # Calling format(args, kwargs) (line 709)
                format_call_result_1116 = invoke(stypy.reporting.localization.Localization(__file__, 709, 40), format_1112, *[name_1114], **kwargs_1115)
                
                # Processing the call keyword arguments (line 709)
                kwargs_1117 = {}
                # Getting the type of 'TypeError' (line 709)
                TypeError_1109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 16), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 709)
                TypeError_call_result_1118 = invoke(stypy.reporting.localization.Localization(__file__, 709, 16), TypeError_1109, *[localization_1110, format_call_result_1116], **kwargs_1117)
                
                # SSA branch for the else part of an if statement (line 708)
                module_type_store.open_ssa_branch('else')
                
                # Call to instance(...): (line 712)
                # Processing the call arguments (line 712)
                # Getting the type of 'localization' (line 712)
                localization_1121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 37), 'localization', False)
                
                # Call to format(...): (line 713)
                # Processing the call arguments (line 713)
                # Getting the type of 'self' (line 714)
                self_1124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 44), 'self', False)
                # Obtaining the member 'name' of a type (line 714)
                name_1125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 44), self_1124, 'name')
                # Processing the call keyword arguments (line 713)
                kwargs_1126 = {}
                str_1122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 37), 'str', "Potentially storing in '{0}' the value of a previously undefined variable")
                # Obtaining the member 'format' of a type (line 713)
                format_1123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 37), str_1122, 'format')
                # Calling format(args, kwargs) (line 713)
                format_call_result_1127 = invoke(stypy.reporting.localization.Localization(__file__, 713, 37), format_1123, *[name_1125], **kwargs_1126)
                
                # Processing the call keyword arguments (line 712)
                kwargs_1128 = {}
                # Getting the type of 'TypeWarning' (line 712)
                TypeWarning_1119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 16), 'TypeWarning', False)
                # Obtaining the member 'instance' of a type (line 712)
                instance_1120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 16), TypeWarning_1119, 'instance')
                # Calling instance(args, kwargs) (line 712)
                instance_call_result_1129 = invoke(stypy.reporting.localization.Localization(__file__, 712, 16), instance_1120, *[localization_1121, format_call_result_1127], **kwargs_1128)
                
                # SSA join for if statement (line 708)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 707)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Obtaining an instance of the builtin type 'tuple' (line 715)
        tuple_1130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 715)
        # Adding element type (line 715)
        # Getting the type of 'contains_undefined' (line 715)
        contains_undefined_1131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 15), 'contains_undefined')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 715, 15), tuple_1130, contains_undefined_1131)
        # Adding element type (line 715)
        # Getting the type of 'more_types_in_value' (line 715)
        more_types_in_value_1132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 35), 'more_types_in_value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 715, 15), tuple_1130, more_types_in_value_1132)
        
        # Assigning a type to the variable 'stypy_return_type' (line 715)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 8), 'stypy_return_type', tuple_1130)
        
        # ################# End of '__check_undefined_stored_value(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__check_undefined_stored_value' in the type store
        # Getting the type of 'stypy_return_type' (line 699)
        stypy_return_type_1133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1133)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__check_undefined_stored_value'
        return stypy_return_type_1133


    @norecursion
    def can_store_elements(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'can_store_elements'
        module_type_store = module_type_store.open_function_context('can_store_elements', 717, 4, False)
        # Assigning a type to the variable 'self' (line 718)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.can_store_elements.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.can_store_elements.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.can_store_elements.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.can_store_elements.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.can_store_elements')
        TypeInferenceProxy.can_store_elements.__dict__.__setitem__('stypy_param_names_list', [])
        TypeInferenceProxy.can_store_elements.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.can_store_elements.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.can_store_elements.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.can_store_elements.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.can_store_elements.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.can_store_elements.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.can_store_elements', [], None, None, defaults, varargs, kwargs)

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

        str_1134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, (-1)), 'str', '\n        Determines if this proxy represents a Python type able to store elements (lists, tuples, ...)\n        :return: bool\n        ')
        
        # Assigning a BoolOp to a Name (line 722):
        
        # Assigning a BoolOp to a Name (line 722):
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        
        str_1135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 23), 'str', 'dictionary-')
        # Getting the type of 'self' (line 722)
        self_1136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 40), 'self')
        # Obtaining the member 'name' of a type (line 722)
        name_1137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 40), self_1136, 'name')
        # Applying the binary operator 'in' (line 722)
        result_contains_1138 = python_operator(stypy.reporting.localization.Localization(__file__, 722, 23), 'in', str_1135, name_1137)
        
        
        str_1139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 54), 'str', 'iterator')
        # Getting the type of 'self' (line 722)
        self_1140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 68), 'self')
        # Obtaining the member 'name' of a type (line 722)
        name_1141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 68), self_1140, 'name')
        # Applying the binary operator 'in' (line 722)
        result_contains_1142 = python_operator(stypy.reporting.localization.Localization(__file__, 722, 54), 'in', str_1139, name_1141)
        
        # Applying the binary operator 'and' (line 722)
        result_and_keyword_1143 = python_operator(stypy.reporting.localization.Localization(__file__, 722, 23), 'and', result_contains_1138, result_contains_1142)
        
        
        # Evaluating a boolean operation
        
        str_1144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 83), 'str', 'iterator')
        # Getting the type of 'self' (line 722)
        self_1145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 97), 'self')
        # Obtaining the member 'name' of a type (line 722)
        name_1146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 97), self_1145, 'name')
        # Applying the binary operator 'in' (line 722)
        result_contains_1147 = python_operator(stypy.reporting.localization.Localization(__file__, 722, 83), 'in', str_1144, name_1146)
        
        
        str_1148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 83), 'str', 'dict')
        # Getting the type of 'self' (line 723)
        self_1149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 97), 'self')
        # Obtaining the member 'name' of a type (line 723)
        name_1150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 97), self_1149, 'name')
        # Applying the binary operator 'notin' (line 723)
        result_contains_1151 = python_operator(stypy.reporting.localization.Localization(__file__, 723, 83), 'notin', str_1148, name_1150)
        
        # Applying the binary operator 'and' (line 722)
        result_and_keyword_1152 = python_operator(stypy.reporting.localization.Localization(__file__, 722, 83), 'and', result_contains_1147, result_contains_1151)
        
        # Applying the binary operator 'or' (line 722)
        result_or_keyword_1153 = python_operator(stypy.reporting.localization.Localization(__file__, 722, 22), 'or', result_and_keyword_1143, result_and_keyword_1152)
        
        # Assigning a type to the variable 'is_iterator' (line 722)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 8), 'is_iterator', result_or_keyword_1153)
        
        # Assigning a List to a Name (line 725):
        
        # Assigning a List to a Name (line 725):
        
        # Obtaining an instance of the builtin type 'list' (line 725)
        list_1154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 725)
        # Adding element type (line 725)
        # Getting the type of 'list' (line 725)
        list_1155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 27), 'list')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 26), list_1154, list_1155)
        # Adding element type (line 725)
        # Getting the type of 'set' (line 725)
        set_1156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 33), 'set')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 26), list_1154, set_1156)
        # Adding element type (line 725)
        # Getting the type of 'tuple' (line 725)
        tuple_1157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 38), 'tuple')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 26), list_1154, tuple_1157)
        # Adding element type (line 725)
        # Getting the type of 'types' (line 725)
        types_1158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 45), 'types')
        # Obtaining the member 'GeneratorType' of a type (line 725)
        GeneratorType_1159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 45), types_1158, 'GeneratorType')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 26), list_1154, GeneratorType_1159)
        # Adding element type (line 725)
        # Getting the type of 'bytearray' (line 725)
        bytearray_1160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 66), 'bytearray')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 26), list_1154, bytearray_1160)
        # Adding element type (line 725)
        # Getting the type of 'slice' (line 725)
        slice_1161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 77), 'slice')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 26), list_1154, slice_1161)
        # Adding element type (line 725)
        # Getting the type of 'range' (line 725)
        range_1162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 84), 'range')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 26), list_1154, range_1162)
        # Adding element type (line 725)
        # Getting the type of 'xrange' (line 725)
        xrange_1163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 91), 'xrange')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 26), list_1154, xrange_1163)
        # Adding element type (line 725)
        # Getting the type of 'enumerate' (line 725)
        enumerate_1164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 99), 'enumerate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 26), list_1154, enumerate_1164)
        # Adding element type (line 725)
        # Getting the type of 'reversed' (line 725)
        reversed_1165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 110), 'reversed')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 26), list_1154, reversed_1165)
        # Adding element type (line 725)
        # Getting the type of 'frozenset' (line 726)
        frozenset_1166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 27), 'frozenset')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 26), list_1154, frozenset_1166)
        
        # Assigning a type to the variable 'data_structures' (line 725)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 725, 8), 'data_structures', list_1154)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 727)
        self_1167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 16), 'self')
        # Obtaining the member 'python_entity' of a type (line 727)
        python_entity_1168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 16), self_1167, 'python_entity')
        # Getting the type of 'data_structures' (line 727)
        data_structures_1169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 38), 'data_structures')
        # Applying the binary operator 'in' (line 727)
        result_contains_1170 = python_operator(stypy.reporting.localization.Localization(__file__, 727, 16), 'in', python_entity_1168, data_structures_1169)
        
        # Getting the type of 'is_iterator' (line 727)
        is_iterator_1171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 58), 'is_iterator')
        # Applying the binary operator 'or' (line 727)
        result_or_keyword_1172 = python_operator(stypy.reporting.localization.Localization(__file__, 727, 15), 'or', result_contains_1170, is_iterator_1171)
        
        # Assigning a type to the variable 'stypy_return_type' (line 727)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 8), 'stypy_return_type', result_or_keyword_1172)
        
        # ################# End of 'can_store_elements(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'can_store_elements' in the type store
        # Getting the type of 'stypy_return_type' (line 717)
        stypy_return_type_1173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1173)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'can_store_elements'
        return stypy_return_type_1173


    @norecursion
    def can_store_keypairs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'can_store_keypairs'
        module_type_store = module_type_store.open_function_context('can_store_keypairs', 729, 4, False)
        # Assigning a type to the variable 'self' (line 730)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.can_store_keypairs.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.can_store_keypairs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.can_store_keypairs.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.can_store_keypairs.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.can_store_keypairs')
        TypeInferenceProxy.can_store_keypairs.__dict__.__setitem__('stypy_param_names_list', [])
        TypeInferenceProxy.can_store_keypairs.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.can_store_keypairs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.can_store_keypairs.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.can_store_keypairs.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.can_store_keypairs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.can_store_keypairs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.can_store_keypairs', [], None, None, defaults, varargs, kwargs)

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

        str_1174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, (-1)), 'str', '\n        Determines if this proxy represents a Python type able to store keypairs (dict, dict iterators)\n        :return: bool\n        ')
        
        # Assigning a BoolOp to a Name (line 734):
        
        # Assigning a BoolOp to a Name (line 734):
        
        # Evaluating a boolean operation
        
        str_1175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 22), 'str', 'iterator')
        # Getting the type of 'self' (line 734)
        self_1176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 36), 'self')
        # Obtaining the member 'name' of a type (line 734)
        name_1177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 36), self_1176, 'name')
        # Applying the binary operator 'in' (line 734)
        result_contains_1178 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 22), 'in', str_1175, name_1177)
        
        
        str_1179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 50), 'str', 'dict')
        # Getting the type of 'self' (line 734)
        self_1180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 60), 'self')
        # Obtaining the member 'name' of a type (line 734)
        name_1181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 60), self_1180, 'name')
        # Applying the binary operator 'in' (line 734)
        result_contains_1182 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 50), 'in', str_1179, name_1181)
        
        # Applying the binary operator 'and' (line 734)
        result_and_keyword_1183 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 22), 'and', result_contains_1178, result_contains_1182)
        
        # Assigning a type to the variable 'is_iterator' (line 734)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 8), 'is_iterator', result_and_keyword_1183)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 736)
        self_1184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 15), 'self')
        # Obtaining the member 'python_entity' of a type (line 736)
        python_entity_1185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 15), self_1184, 'python_entity')
        # Getting the type of 'dict' (line 736)
        dict_1186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 37), 'dict')
        # Applying the binary operator 'is' (line 736)
        result_is__1187 = python_operator(stypy.reporting.localization.Localization(__file__, 736, 15), 'is', python_entity_1185, dict_1186)
        
        # Getting the type of 'is_iterator' (line 736)
        is_iterator_1188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 45), 'is_iterator')
        # Applying the binary operator 'or' (line 736)
        result_or_keyword_1189 = python_operator(stypy.reporting.localization.Localization(__file__, 736, 15), 'or', result_is__1187, is_iterator_1188)
        
        # Assigning a type to the variable 'stypy_return_type' (line 736)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 8), 'stypy_return_type', result_or_keyword_1189)
        
        # ################# End of 'can_store_keypairs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'can_store_keypairs' in the type store
        # Getting the type of 'stypy_return_type' (line 729)
        stypy_return_type_1190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1190)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'can_store_keypairs'
        return stypy_return_type_1190


    @norecursion
    def is_empty(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_empty'
        module_type_store = module_type_store.open_function_context('is_empty', 738, 4, False)
        # Assigning a type to the variable 'self' (line 739)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.is_empty.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.is_empty.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.is_empty.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.is_empty.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.is_empty')
        TypeInferenceProxy.is_empty.__dict__.__setitem__('stypy_param_names_list', [])
        TypeInferenceProxy.is_empty.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.is_empty.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.is_empty.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.is_empty.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.is_empty.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.is_empty.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.is_empty', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_empty', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_empty(...)' code ##################

        str_1191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, (-1)), 'str', '\n        Determines if a proxy able to store elements can be considered empty (no elements were inserted through its\n        lifespan\n        :return: None or TypeError\n        ')
        
        # Evaluating a boolean operation
        
        
        # Call to can_store_elements(...): (line 744)
        # Processing the call keyword arguments (line 744)
        kwargs_1194 = {}
        # Getting the type of 'self' (line 744)
        self_1192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 15), 'self', False)
        # Obtaining the member 'can_store_elements' of a type (line 744)
        can_store_elements_1193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 15), self_1192, 'can_store_elements')
        # Calling can_store_elements(args, kwargs) (line 744)
        can_store_elements_call_result_1195 = invoke(stypy.reporting.localization.Localization(__file__, 744, 15), can_store_elements_1193, *[], **kwargs_1194)
        
        # Applying the 'not' unary operator (line 744)
        result_not__1196 = python_operator(stypy.reporting.localization.Localization(__file__, 744, 11), 'not', can_store_elements_call_result_1195)
        
        
        
        # Call to can_store_keypairs(...): (line 744)
        # Processing the call keyword arguments (line 744)
        kwargs_1199 = {}
        # Getting the type of 'self' (line 744)
        self_1197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 49), 'self', False)
        # Obtaining the member 'can_store_keypairs' of a type (line 744)
        can_store_keypairs_1198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 49), self_1197, 'can_store_keypairs')
        # Calling can_store_keypairs(args, kwargs) (line 744)
        can_store_keypairs_call_result_1200 = invoke(stypy.reporting.localization.Localization(__file__, 744, 49), can_store_keypairs_1198, *[], **kwargs_1199)
        
        # Applying the 'not' unary operator (line 744)
        result_not__1201 = python_operator(stypy.reporting.localization.Localization(__file__, 744, 45), 'not', can_store_keypairs_call_result_1200)
        
        # Applying the binary operator 'and' (line 744)
        result_and_keyword_1202 = python_operator(stypy.reporting.localization.Localization(__file__, 744, 11), 'and', result_not__1196, result_not__1201)
        
        # Testing if the type of an if condition is none (line 744)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 744, 8), result_and_keyword_1202):
            pass
        else:
            
            # Testing the type of an if condition (line 744)
            if_condition_1203 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 744, 8), result_and_keyword_1202)
            # Assigning a type to the variable 'if_condition_1203' (line 744)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 8), 'if_condition_1203', if_condition_1203)
            # SSA begins for if statement (line 744)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 745)
            # Processing the call arguments (line 745)
            # Getting the type of 'None' (line 745)
            None_1205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 29), 'None', False)
            str_1206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 29), 'str', 'STYPY CRITICAL ERROR: Attempt to determine if a container is empty over a python type ({0}) that is not able to do it')
            # Processing the call keyword arguments (line 745)
            kwargs_1207 = {}
            # Getting the type of 'TypeError' (line 745)
            TypeError_1204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 745)
            TypeError_call_result_1208 = invoke(stypy.reporting.localization.Localization(__file__, 745, 19), TypeError_1204, *[None_1205, str_1206], **kwargs_1207)
            
            # Assigning a type to the variable 'stypy_return_type' (line 745)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 12), 'stypy_return_type', TypeError_call_result_1208)
            # SSA join for if statement (line 744)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to hasattr(...): (line 748)
        # Processing the call arguments (line 748)
        # Getting the type of 'self' (line 748)
        self_1210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 23), 'self', False)
        # Getting the type of 'self' (line 748)
        self_1211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 29), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 748)
        contained_elements_property_name_1212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 29), self_1211, 'contained_elements_property_name')
        # Processing the call keyword arguments (line 748)
        kwargs_1213 = {}
        # Getting the type of 'hasattr' (line 748)
        hasattr_1209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 15), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 748)
        hasattr_call_result_1214 = invoke(stypy.reporting.localization.Localization(__file__, 748, 15), hasattr_1209, *[self_1210, contained_elements_property_name_1212], **kwargs_1213)
        
        # Assigning a type to the variable 'stypy_return_type' (line 748)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 748, 8), 'stypy_return_type', hasattr_call_result_1214)
        
        # ################# End of 'is_empty(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_empty' in the type store
        # Getting the type of 'stypy_return_type' (line 738)
        stypy_return_type_1215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1215)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_empty'
        return stypy_return_type_1215


    @norecursion
    def get_elements_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_elements_type'
        module_type_store = module_type_store.open_function_context('get_elements_type', 750, 4, False)
        # Assigning a type to the variable 'self' (line 751)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.get_elements_type.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.get_elements_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.get_elements_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.get_elements_type.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.get_elements_type')
        TypeInferenceProxy.get_elements_type.__dict__.__setitem__('stypy_param_names_list', [])
        TypeInferenceProxy.get_elements_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.get_elements_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.get_elements_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.get_elements_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.get_elements_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.get_elements_type.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.get_elements_type', [], None, None, defaults, varargs, kwargs)

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

        str_1216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, (-1)), 'str', '\n        Obtains the elements stored by this type, returning an error if this is called over a proxy that represent\n        a non element holding Python type\n        :return: None or TypeError\n        ')
        
        # Evaluating a boolean operation
        
        
        # Call to can_store_elements(...): (line 756)
        # Processing the call keyword arguments (line 756)
        kwargs_1219 = {}
        # Getting the type of 'self' (line 756)
        self_1217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 15), 'self', False)
        # Obtaining the member 'can_store_elements' of a type (line 756)
        can_store_elements_1218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 15), self_1217, 'can_store_elements')
        # Calling can_store_elements(args, kwargs) (line 756)
        can_store_elements_call_result_1220 = invoke(stypy.reporting.localization.Localization(__file__, 756, 15), can_store_elements_1218, *[], **kwargs_1219)
        
        # Applying the 'not' unary operator (line 756)
        result_not__1221 = python_operator(stypy.reporting.localization.Localization(__file__, 756, 11), 'not', can_store_elements_call_result_1220)
        
        
        
        # Call to can_store_keypairs(...): (line 756)
        # Processing the call keyword arguments (line 756)
        kwargs_1224 = {}
        # Getting the type of 'self' (line 756)
        self_1222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 49), 'self', False)
        # Obtaining the member 'can_store_keypairs' of a type (line 756)
        can_store_keypairs_1223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 49), self_1222, 'can_store_keypairs')
        # Calling can_store_keypairs(args, kwargs) (line 756)
        can_store_keypairs_call_result_1225 = invoke(stypy.reporting.localization.Localization(__file__, 756, 49), can_store_keypairs_1223, *[], **kwargs_1224)
        
        # Applying the 'not' unary operator (line 756)
        result_not__1226 = python_operator(stypy.reporting.localization.Localization(__file__, 756, 45), 'not', can_store_keypairs_call_result_1225)
        
        # Applying the binary operator 'and' (line 756)
        result_and_keyword_1227 = python_operator(stypy.reporting.localization.Localization(__file__, 756, 11), 'and', result_not__1221, result_not__1226)
        
        # Testing if the type of an if condition is none (line 756)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 756, 8), result_and_keyword_1227):
            pass
        else:
            
            # Testing the type of an if condition (line 756)
            if_condition_1228 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 756, 8), result_and_keyword_1227)
            # Assigning a type to the variable 'if_condition_1228' (line 756)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 8), 'if_condition_1228', if_condition_1228)
            # SSA begins for if statement (line 756)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 757)
            # Processing the call arguments (line 757)
            # Getting the type of 'None' (line 757)
            None_1230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 29), 'None', False)
            str_1231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 29), 'str', 'STYPY CRITICAL ERROR: Attempt to return stored elements over a python type ({0}) that is not able to do it')
            # Processing the call keyword arguments (line 757)
            kwargs_1232 = {}
            # Getting the type of 'TypeError' (line 757)
            TypeError_1229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 757)
            TypeError_call_result_1233 = invoke(stypy.reporting.localization.Localization(__file__, 757, 19), TypeError_1229, *[None_1230, str_1231], **kwargs_1232)
            
            # Assigning a type to the variable 'stypy_return_type' (line 757)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 757, 12), 'stypy_return_type', TypeError_call_result_1233)
            # SSA join for if statement (line 756)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to hasattr(...): (line 760)
        # Processing the call arguments (line 760)
        # Getting the type of 'self' (line 760)
        self_1235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 19), 'self', False)
        # Getting the type of 'self' (line 760)
        self_1236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 25), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 760)
        contained_elements_property_name_1237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 25), self_1236, 'contained_elements_property_name')
        # Processing the call keyword arguments (line 760)
        kwargs_1238 = {}
        # Getting the type of 'hasattr' (line 760)
        hasattr_1234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 11), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 760)
        hasattr_call_result_1239 = invoke(stypy.reporting.localization.Localization(__file__, 760, 11), hasattr_1234, *[self_1235, contained_elements_property_name_1237], **kwargs_1238)
        
        # Testing if the type of an if condition is none (line 760)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 760, 8), hasattr_call_result_1239):
            
            # Call to UndefinedType(...): (line 763)
            # Processing the call keyword arguments (line 763)
            kwargs_1249 = {}
            # Getting the type of 'undefined_type_copy' (line 763)
            undefined_type_copy_1247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 19), 'undefined_type_copy', False)
            # Obtaining the member 'UndefinedType' of a type (line 763)
            UndefinedType_1248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 19), undefined_type_copy_1247, 'UndefinedType')
            # Calling UndefinedType(args, kwargs) (line 763)
            UndefinedType_call_result_1250 = invoke(stypy.reporting.localization.Localization(__file__, 763, 19), UndefinedType_1248, *[], **kwargs_1249)
            
            # Assigning a type to the variable 'stypy_return_type' (line 763)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 12), 'stypy_return_type', UndefinedType_call_result_1250)
        else:
            
            # Testing the type of an if condition (line 760)
            if_condition_1240 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 760, 8), hasattr_call_result_1239)
            # Assigning a type to the variable 'if_condition_1240' (line 760)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'if_condition_1240', if_condition_1240)
            # SSA begins for if statement (line 760)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to getattr(...): (line 761)
            # Processing the call arguments (line 761)
            # Getting the type of 'self' (line 761)
            self_1242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 27), 'self', False)
            # Getting the type of 'self' (line 761)
            self_1243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 33), 'self', False)
            # Obtaining the member 'contained_elements_property_name' of a type (line 761)
            contained_elements_property_name_1244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 33), self_1243, 'contained_elements_property_name')
            # Processing the call keyword arguments (line 761)
            kwargs_1245 = {}
            # Getting the type of 'getattr' (line 761)
            getattr_1241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 19), 'getattr', False)
            # Calling getattr(args, kwargs) (line 761)
            getattr_call_result_1246 = invoke(stypy.reporting.localization.Localization(__file__, 761, 19), getattr_1241, *[self_1242, contained_elements_property_name_1244], **kwargs_1245)
            
            # Assigning a type to the variable 'stypy_return_type' (line 761)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'stypy_return_type', getattr_call_result_1246)
            # SSA branch for the else part of an if statement (line 760)
            module_type_store.open_ssa_branch('else')
            
            # Call to UndefinedType(...): (line 763)
            # Processing the call keyword arguments (line 763)
            kwargs_1249 = {}
            # Getting the type of 'undefined_type_copy' (line 763)
            undefined_type_copy_1247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 19), 'undefined_type_copy', False)
            # Obtaining the member 'UndefinedType' of a type (line 763)
            UndefinedType_1248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 19), undefined_type_copy_1247, 'UndefinedType')
            # Calling UndefinedType(args, kwargs) (line 763)
            UndefinedType_call_result_1250 = invoke(stypy.reporting.localization.Localization(__file__, 763, 19), UndefinedType_1248, *[], **kwargs_1249)
            
            # Assigning a type to the variable 'stypy_return_type' (line 763)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 12), 'stypy_return_type', UndefinedType_call_result_1250)
            # SSA join for if statement (line 760)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'get_elements_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_elements_type' in the type store
        # Getting the type of 'stypy_return_type' (line 750)
        stypy_return_type_1251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1251)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_elements_type'
        return stypy_return_type_1251


    @norecursion
    def set_elements_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 765)
        True_1252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 79), 'True')
        defaults = [True_1252]
        # Create a new context for function 'set_elements_type'
        module_type_store = module_type_store.open_function_context('set_elements_type', 765, 4, False)
        # Assigning a type to the variable 'self' (line 766)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.set_elements_type.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.set_elements_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.set_elements_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.set_elements_type.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.set_elements_type')
        TypeInferenceProxy.set_elements_type.__dict__.__setitem__('stypy_param_names_list', ['localization', 'elements_type', 'record_annotation'])
        TypeInferenceProxy.set_elements_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.set_elements_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.set_elements_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.set_elements_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.set_elements_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.set_elements_type.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.set_elements_type', ['localization', 'elements_type', 'record_annotation'], None, None, defaults, varargs, kwargs)

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

        str_1253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, (-1)), 'str', '\n        Sets the elements stored by this type, returning an error if this is called over a proxy that represent\n        a non element holding Python type. It also checks if we are trying to store an undefined variable.\n        :param localization: Caller information\n        :param elements_type: New stored elements type\n        :param record_annotation: Whether to annotate the type change or not\n        :return: The stored elements type\n        ')
        
        # Evaluating a boolean operation
        
        
        # Call to can_store_elements(...): (line 774)
        # Processing the call keyword arguments (line 774)
        kwargs_1256 = {}
        # Getting the type of 'self' (line 774)
        self_1254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 15), 'self', False)
        # Obtaining the member 'can_store_elements' of a type (line 774)
        can_store_elements_1255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 15), self_1254, 'can_store_elements')
        # Calling can_store_elements(args, kwargs) (line 774)
        can_store_elements_call_result_1257 = invoke(stypy.reporting.localization.Localization(__file__, 774, 15), can_store_elements_1255, *[], **kwargs_1256)
        
        # Applying the 'not' unary operator (line 774)
        result_not__1258 = python_operator(stypy.reporting.localization.Localization(__file__, 774, 11), 'not', can_store_elements_call_result_1257)
        
        
        
        # Call to can_store_keypairs(...): (line 774)
        # Processing the call keyword arguments (line 774)
        kwargs_1261 = {}
        # Getting the type of 'self' (line 774)
        self_1259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 49), 'self', False)
        # Obtaining the member 'can_store_keypairs' of a type (line 774)
        can_store_keypairs_1260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 49), self_1259, 'can_store_keypairs')
        # Calling can_store_keypairs(args, kwargs) (line 774)
        can_store_keypairs_call_result_1262 = invoke(stypy.reporting.localization.Localization(__file__, 774, 49), can_store_keypairs_1260, *[], **kwargs_1261)
        
        # Applying the 'not' unary operator (line 774)
        result_not__1263 = python_operator(stypy.reporting.localization.Localization(__file__, 774, 45), 'not', can_store_keypairs_call_result_1262)
        
        # Applying the binary operator 'and' (line 774)
        result_and_keyword_1264 = python_operator(stypy.reporting.localization.Localization(__file__, 774, 11), 'and', result_not__1258, result_not__1263)
        
        # Testing if the type of an if condition is none (line 774)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 774, 8), result_and_keyword_1264):
            pass
        else:
            
            # Testing the type of an if condition (line 774)
            if_condition_1265 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 774, 8), result_and_keyword_1264)
            # Assigning a type to the variable 'if_condition_1265' (line 774)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 8), 'if_condition_1265', if_condition_1265)
            # SSA begins for if statement (line 774)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 775)
            # Processing the call arguments (line 775)
            # Getting the type of 'localization' (line 775)
            localization_1267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 29), 'localization', False)
            
            # Call to format(...): (line 776)
            # Processing the call arguments (line 776)
            
            # Call to get_python_type(...): (line 777)
            # Processing the call keyword arguments (line 777)
            kwargs_1272 = {}
            # Getting the type of 'self' (line 777)
            self_1270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 64), 'self', False)
            # Obtaining the member 'get_python_type' of a type (line 777)
            get_python_type_1271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 64), self_1270, 'get_python_type')
            # Calling get_python_type(args, kwargs) (line 777)
            get_python_type_call_result_1273 = invoke(stypy.reporting.localization.Localization(__file__, 777, 64), get_python_type_1271, *[], **kwargs_1272)
            
            # Processing the call keyword arguments (line 776)
            kwargs_1274 = {}
            str_1268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 29), 'str', 'STYPY CRITICAL ERROR: Attempt to set stored elements types over a python type ({0}) that is not able to do it')
            # Obtaining the member 'format' of a type (line 776)
            format_1269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 29), str_1268, 'format')
            # Calling format(args, kwargs) (line 776)
            format_call_result_1275 = invoke(stypy.reporting.localization.Localization(__file__, 776, 29), format_1269, *[get_python_type_call_result_1273], **kwargs_1274)
            
            # Processing the call keyword arguments (line 775)
            kwargs_1276 = {}
            # Getting the type of 'TypeError' (line 775)
            TypeError_1266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 775)
            TypeError_call_result_1277 = invoke(stypy.reporting.localization.Localization(__file__, 775, 19), TypeError_1266, *[localization_1267, format_call_result_1275], **kwargs_1276)
            
            # Assigning a type to the variable 'stypy_return_type' (line 775)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 12), 'stypy_return_type', TypeError_call_result_1277)
            # SSA join for if statement (line 774)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Tuple (line 779):
        
        # Assigning a Call to a Name:
        
        # Call to contains_an_undefined_type(...): (line 779)
        # Processing the call arguments (line 779)
        # Getting the type of 'elements_type' (line 779)
        elements_type_1280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 96), 'elements_type', False)
        # Processing the call keyword arguments (line 779)
        kwargs_1281 = {}
        # Getting the type of 'TypeInferenceProxy' (line 779)
        TypeInferenceProxy_1278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 50), 'TypeInferenceProxy', False)
        # Obtaining the member 'contains_an_undefined_type' of a type (line 779)
        contains_an_undefined_type_1279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 50), TypeInferenceProxy_1278, 'contains_an_undefined_type')
        # Calling contains_an_undefined_type(args, kwargs) (line 779)
        contains_an_undefined_type_call_result_1282 = invoke(stypy.reporting.localization.Localization(__file__, 779, 50), contains_an_undefined_type_1279, *[elements_type_1280], **kwargs_1281)
        
        # Assigning a type to the variable 'call_assignment_7' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'call_assignment_7', contains_an_undefined_type_call_result_1282)
        
        # Assigning a Call to a Name (line 779):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_7' (line 779)
        call_assignment_7_1283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'call_assignment_7', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_1284 = stypy_get_value_from_tuple(call_assignment_7_1283, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_8' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'call_assignment_8', stypy_get_value_from_tuple_call_result_1284)
        
        # Assigning a Name to a Name (line 779):
        # Getting the type of 'call_assignment_8' (line 779)
        call_assignment_8_1285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'call_assignment_8')
        # Assigning a type to the variable 'contains_undefined' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'contains_undefined', call_assignment_8_1285)
        
        # Assigning a Call to a Name (line 779):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_7' (line 779)
        call_assignment_7_1286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'call_assignment_7', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_1287 = stypy_get_value_from_tuple(call_assignment_7_1286, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_9' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'call_assignment_9', stypy_get_value_from_tuple_call_result_1287)
        
        # Assigning a Name to a Name (line 779):
        # Getting the type of 'call_assignment_9' (line 779)
        call_assignment_9_1288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'call_assignment_9')
        # Assigning a type to the variable 'more_types_in_value' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 28), 'more_types_in_value', call_assignment_9_1288)
        # Getting the type of 'contains_undefined' (line 780)
        contains_undefined_1289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 11), 'contains_undefined')
        # Testing if the type of an if condition is none (line 780)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 780, 8), contains_undefined_1289):
            pass
        else:
            
            # Testing the type of an if condition (line 780)
            if_condition_1290 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 780, 8), contains_undefined_1289)
            # Assigning a type to the variable 'if_condition_1290' (line 780)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 8), 'if_condition_1290', if_condition_1290)
            # SSA begins for if statement (line 780)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'more_types_in_value' (line 781)
            more_types_in_value_1291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 15), 'more_types_in_value')
            int_1292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 38), 'int')
            # Applying the binary operator '==' (line 781)
            result_eq_1293 = python_operator(stypy.reporting.localization.Localization(__file__, 781, 15), '==', more_types_in_value_1291, int_1292)
            
            # Testing if the type of an if condition is none (line 781)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 781, 12), result_eq_1293):
                
                # Call to instance(...): (line 785)
                # Processing the call arguments (line 785)
                # Getting the type of 'localization' (line 785)
                localization_1307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 37), 'localization', False)
                
                # Call to format(...): (line 786)
                # Processing the call arguments (line 786)
                # Getting the type of 'self' (line 787)
                self_1310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 44), 'self', False)
                # Obtaining the member 'name' of a type (line 787)
                name_1311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 44), self_1310, 'name')
                # Processing the call keyword arguments (line 786)
                kwargs_1312 = {}
                str_1308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 37), 'str', "Potentially storing in '{0}' the value of a previously undefined variable")
                # Obtaining the member 'format' of a type (line 786)
                format_1309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 37), str_1308, 'format')
                # Calling format(args, kwargs) (line 786)
                format_call_result_1313 = invoke(stypy.reporting.localization.Localization(__file__, 786, 37), format_1309, *[name_1311], **kwargs_1312)
                
                # Processing the call keyword arguments (line 785)
                kwargs_1314 = {}
                # Getting the type of 'TypeWarning' (line 785)
                TypeWarning_1305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 16), 'TypeWarning', False)
                # Obtaining the member 'instance' of a type (line 785)
                instance_1306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 16), TypeWarning_1305, 'instance')
                # Calling instance(args, kwargs) (line 785)
                instance_call_result_1315 = invoke(stypy.reporting.localization.Localization(__file__, 785, 16), instance_1306, *[localization_1307, format_call_result_1313], **kwargs_1314)
                
            else:
                
                # Testing the type of an if condition (line 781)
                if_condition_1294 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 781, 12), result_eq_1293)
                # Assigning a type to the variable 'if_condition_1294' (line 781)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 781, 12), 'if_condition_1294', if_condition_1294)
                # SSA begins for if statement (line 781)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to TypeError(...): (line 782)
                # Processing the call arguments (line 782)
                # Getting the type of 'localization' (line 782)
                localization_1296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 26), 'localization', False)
                
                # Call to format(...): (line 782)
                # Processing the call arguments (line 782)
                # Getting the type of 'self' (line 783)
                self_1299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 33), 'self', False)
                # Obtaining the member 'name' of a type (line 783)
                name_1300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 33), self_1299, 'name')
                # Processing the call keyword arguments (line 782)
                kwargs_1301 = {}
                str_1297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, 40), 'str', "Storing in '{0}' the value of a previously undefined variable")
                # Obtaining the member 'format' of a type (line 782)
                format_1298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 40), str_1297, 'format')
                # Calling format(args, kwargs) (line 782)
                format_call_result_1302 = invoke(stypy.reporting.localization.Localization(__file__, 782, 40), format_1298, *[name_1300], **kwargs_1301)
                
                # Processing the call keyword arguments (line 782)
                kwargs_1303 = {}
                # Getting the type of 'TypeError' (line 782)
                TypeError_1295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 16), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 782)
                TypeError_call_result_1304 = invoke(stypy.reporting.localization.Localization(__file__, 782, 16), TypeError_1295, *[localization_1296, format_call_result_1302], **kwargs_1303)
                
                # SSA branch for the else part of an if statement (line 781)
                module_type_store.open_ssa_branch('else')
                
                # Call to instance(...): (line 785)
                # Processing the call arguments (line 785)
                # Getting the type of 'localization' (line 785)
                localization_1307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 37), 'localization', False)
                
                # Call to format(...): (line 786)
                # Processing the call arguments (line 786)
                # Getting the type of 'self' (line 787)
                self_1310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 44), 'self', False)
                # Obtaining the member 'name' of a type (line 787)
                name_1311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 44), self_1310, 'name')
                # Processing the call keyword arguments (line 786)
                kwargs_1312 = {}
                str_1308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 37), 'str', "Potentially storing in '{0}' the value of a previously undefined variable")
                # Obtaining the member 'format' of a type (line 786)
                format_1309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 37), str_1308, 'format')
                # Calling format(args, kwargs) (line 786)
                format_call_result_1313 = invoke(stypy.reporting.localization.Localization(__file__, 786, 37), format_1309, *[name_1311], **kwargs_1312)
                
                # Processing the call keyword arguments (line 785)
                kwargs_1314 = {}
                # Getting the type of 'TypeWarning' (line 785)
                TypeWarning_1305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 16), 'TypeWarning', False)
                # Obtaining the member 'instance' of a type (line 785)
                instance_1306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 16), TypeWarning_1305, 'instance')
                # Calling instance(args, kwargs) (line 785)
                instance_call_result_1315 = invoke(stypy.reporting.localization.Localization(__file__, 785, 16), instance_1306, *[localization_1307, format_call_result_1313], **kwargs_1314)
                
                # SSA join for if statement (line 781)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 780)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to setattr(...): (line 789)
        # Processing the call arguments (line 789)
        # Getting the type of 'self' (line 789)
        self_1317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 16), 'self', False)
        # Getting the type of 'self' (line 789)
        self_1318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 22), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 789)
        contained_elements_property_name_1319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 789, 22), self_1318, 'contained_elements_property_name')
        # Getting the type of 'elements_type' (line 789)
        elements_type_1320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 61), 'elements_type', False)
        # Processing the call keyword arguments (line 789)
        kwargs_1321 = {}
        # Getting the type of 'setattr' (line 789)
        setattr_1316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 8), 'setattr', False)
        # Calling setattr(args, kwargs) (line 789)
        setattr_call_result_1322 = invoke(stypy.reporting.localization.Localization(__file__, 789, 8), setattr_1316, *[self_1317, contained_elements_property_name_1319, elements_type_1320], **kwargs_1321)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'record_annotation' (line 790)
        record_annotation_1323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 11), 'record_annotation')
        # Getting the type of 'self' (line 790)
        self_1324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 33), 'self')
        # Obtaining the member 'annotate_types' of a type (line 790)
        annotate_types_1325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 790, 33), self_1324, 'annotate_types')
        # Applying the binary operator 'and' (line 790)
        result_and_keyword_1326 = python_operator(stypy.reporting.localization.Localization(__file__, 790, 11), 'and', record_annotation_1323, annotate_types_1325)
        
        # Testing if the type of an if condition is none (line 790)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 790, 8), result_and_keyword_1326):
            pass
        else:
            
            # Testing the type of an if condition (line 790)
            if_condition_1327 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 790, 8), result_and_keyword_1326)
            # Assigning a type to the variable 'if_condition_1327' (line 790)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 790, 8), 'if_condition_1327', if_condition_1327)
            # SSA begins for if statement (line 790)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to __annotate_type(...): (line 791)
            # Processing the call arguments (line 791)
            # Getting the type of 'localization' (line 791)
            localization_1330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 33), 'localization', False)
            # Obtaining the member 'line' of a type (line 791)
            line_1331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 33), localization_1330, 'line')
            # Getting the type of 'localization' (line 791)
            localization_1332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 52), 'localization', False)
            # Obtaining the member 'column' of a type (line 791)
            column_1333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 52), localization_1332, 'column')
            str_1334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, 73), 'str', '<container elements type>')
            
            # Call to getattr(...): (line 792)
            # Processing the call arguments (line 792)
            # Getting the type of 'self' (line 792)
            self_1336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 41), 'self', False)
            # Getting the type of 'self' (line 792)
            self_1337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 47), 'self', False)
            # Obtaining the member 'contained_elements_property_name' of a type (line 792)
            contained_elements_property_name_1338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 47), self_1337, 'contained_elements_property_name')
            # Processing the call keyword arguments (line 792)
            kwargs_1339 = {}
            # Getting the type of 'getattr' (line 792)
            getattr_1335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 33), 'getattr', False)
            # Calling getattr(args, kwargs) (line 792)
            getattr_call_result_1340 = invoke(stypy.reporting.localization.Localization(__file__, 792, 33), getattr_1335, *[self_1336, contained_elements_property_name_1338], **kwargs_1339)
            
            # Processing the call keyword arguments (line 791)
            kwargs_1341 = {}
            # Getting the type of 'self' (line 791)
            self_1328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 12), 'self', False)
            # Obtaining the member '__annotate_type' of a type (line 791)
            annotate_type_1329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 12), self_1328, '__annotate_type')
            # Calling __annotate_type(args, kwargs) (line 791)
            annotate_type_call_result_1342 = invoke(stypy.reporting.localization.Localization(__file__, 791, 12), annotate_type_1329, *[line_1331, column_1333, str_1334, getattr_call_result_1340], **kwargs_1341)
            
            # SSA join for if statement (line 790)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'set_elements_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_elements_type' in the type store
        # Getting the type of 'stypy_return_type' (line 765)
        stypy_return_type_1343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1343)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_elements_type'
        return stypy_return_type_1343


    @norecursion
    def add_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 794)
        True_1344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 62), 'True')
        defaults = [True_1344]
        # Create a new context for function 'add_type'
        module_type_store = module_type_store.open_function_context('add_type', 794, 4, False)
        # Assigning a type to the variable 'self' (line 795)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 795, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.add_type.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.add_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.add_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.add_type.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.add_type')
        TypeInferenceProxy.add_type.__dict__.__setitem__('stypy_param_names_list', ['localization', 'type_', 'record_annotation'])
        TypeInferenceProxy.add_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.add_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.add_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.add_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.add_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.add_type.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.add_type', ['localization', 'type_', 'record_annotation'], None, None, defaults, varargs, kwargs)

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

        str_1345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 802, (-1)), 'str', '\n        Adds type_ to the elements stored by this type, returning an error if this is called over a proxy that represent\n        a non element holding Python type. It also checks if we are trying to store an undefined variable.\n        :param localization: Caller information\n        :param type_: Type to store\n        :param record_annotation: Whether to annotate the type change or not\n        :return: None or TypeError\n        ')
        
        
        # Call to can_store_elements(...): (line 803)
        # Processing the call keyword arguments (line 803)
        kwargs_1348 = {}
        # Getting the type of 'self' (line 803)
        self_1346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 15), 'self', False)
        # Obtaining the member 'can_store_elements' of a type (line 803)
        can_store_elements_1347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 803, 15), self_1346, 'can_store_elements')
        # Calling can_store_elements(args, kwargs) (line 803)
        can_store_elements_call_result_1349 = invoke(stypy.reporting.localization.Localization(__file__, 803, 15), can_store_elements_1347, *[], **kwargs_1348)
        
        # Applying the 'not' unary operator (line 803)
        result_not__1350 = python_operator(stypy.reporting.localization.Localization(__file__, 803, 11), 'not', can_store_elements_call_result_1349)
        
        # Testing if the type of an if condition is none (line 803)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 803, 8), result_not__1350):
            pass
        else:
            
            # Testing the type of an if condition (line 803)
            if_condition_1351 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 803, 8), result_not__1350)
            # Assigning a type to the variable 'if_condition_1351' (line 803)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 803, 8), 'if_condition_1351', if_condition_1351)
            # SSA begins for if statement (line 803)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 804)
            # Processing the call arguments (line 804)
            # Getting the type of 'localization' (line 804)
            localization_1353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 29), 'localization', False)
            
            # Call to format(...): (line 805)
            # Processing the call arguments (line 805)
            
            # Call to get_python_type(...): (line 806)
            # Processing the call keyword arguments (line 806)
            kwargs_1358 = {}
            # Getting the type of 'self' (line 806)
            self_1356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 53), 'self', False)
            # Obtaining the member 'get_python_type' of a type (line 806)
            get_python_type_1357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 53), self_1356, 'get_python_type')
            # Calling get_python_type(args, kwargs) (line 806)
            get_python_type_call_result_1359 = invoke(stypy.reporting.localization.Localization(__file__, 806, 53), get_python_type_1357, *[], **kwargs_1358)
            
            # Processing the call keyword arguments (line 805)
            kwargs_1360 = {}
            str_1354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 29), 'str', 'STYPY CRITICAL ERROR: Attempt to store elements over a python type ({0}) that is not able to do it')
            # Obtaining the member 'format' of a type (line 805)
            format_1355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 29), str_1354, 'format')
            # Calling format(args, kwargs) (line 805)
            format_call_result_1361 = invoke(stypy.reporting.localization.Localization(__file__, 805, 29), format_1355, *[get_python_type_call_result_1359], **kwargs_1360)
            
            # Processing the call keyword arguments (line 804)
            kwargs_1362 = {}
            # Getting the type of 'TypeError' (line 804)
            TypeError_1352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 804)
            TypeError_call_result_1363 = invoke(stypy.reporting.localization.Localization(__file__, 804, 19), TypeError_1352, *[localization_1353, format_call_result_1361], **kwargs_1362)
            
            # Assigning a type to the variable 'stypy_return_type' (line 804)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 804, 12), 'stypy_return_type', TypeError_call_result_1363)
            # SSA join for if statement (line 803)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Name to a Name (line 808):
        
        # Assigning a Name to a Name (line 808):
        # Getting the type of 'None' (line 808)
        None_1364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 24), 'None')
        # Assigning a type to the variable 'existing_type' (line 808)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 8), 'existing_type', None_1364)
        
        # Call to hasattr(...): (line 809)
        # Processing the call arguments (line 809)
        # Getting the type of 'self' (line 809)
        self_1366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 19), 'self', False)
        # Getting the type of 'self' (line 809)
        self_1367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 25), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 809)
        contained_elements_property_name_1368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 809, 25), self_1367, 'contained_elements_property_name')
        # Processing the call keyword arguments (line 809)
        kwargs_1369 = {}
        # Getting the type of 'hasattr' (line 809)
        hasattr_1365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 11), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 809)
        hasattr_call_result_1370 = invoke(stypy.reporting.localization.Localization(__file__, 809, 11), hasattr_1365, *[self_1366, contained_elements_property_name_1368], **kwargs_1369)
        
        # Testing if the type of an if condition is none (line 809)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 809, 8), hasattr_call_result_1370):
            pass
        else:
            
            # Testing the type of an if condition (line 809)
            if_condition_1371 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 809, 8), hasattr_call_result_1370)
            # Assigning a type to the variable 'if_condition_1371' (line 809)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 8), 'if_condition_1371', if_condition_1371)
            # SSA begins for if statement (line 809)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 810):
            
            # Assigning a Call to a Name (line 810):
            
            # Call to getattr(...): (line 810)
            # Processing the call arguments (line 810)
            # Getting the type of 'self' (line 810)
            self_1373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 36), 'self', False)
            # Getting the type of 'self' (line 810)
            self_1374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 42), 'self', False)
            # Obtaining the member 'contained_elements_property_name' of a type (line 810)
            contained_elements_property_name_1375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 42), self_1374, 'contained_elements_property_name')
            # Processing the call keyword arguments (line 810)
            kwargs_1376 = {}
            # Getting the type of 'getattr' (line 810)
            getattr_1372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 28), 'getattr', False)
            # Calling getattr(args, kwargs) (line 810)
            getattr_call_result_1377 = invoke(stypy.reporting.localization.Localization(__file__, 810, 28), getattr_1372, *[self_1373, contained_elements_property_name_1375], **kwargs_1376)
            
            # Assigning a type to the variable 'existing_type' (line 810)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 12), 'existing_type', getattr_call_result_1377)
            # SSA join for if statement (line 809)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 812):
        
        # Assigning a Call to a Name (line 812):
        
        # Call to add(...): (line 812)
        # Processing the call arguments (line 812)
        # Getting the type of 'existing_type' (line 812)
        existing_type_1381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 55), 'existing_type', False)
        # Getting the type of 'type_' (line 812)
        type__1382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 70), 'type_', False)
        # Processing the call keyword arguments (line 812)
        kwargs_1383 = {}
        # Getting the type of 'union_type_copy' (line 812)
        union_type_copy_1378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 25), 'union_type_copy', False)
        # Obtaining the member 'UnionType' of a type (line 812)
        UnionType_1379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 25), union_type_copy_1378, 'UnionType')
        # Obtaining the member 'add' of a type (line 812)
        add_1380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 25), UnionType_1379, 'add')
        # Calling add(args, kwargs) (line 812)
        add_call_result_1384 = invoke(stypy.reporting.localization.Localization(__file__, 812, 25), add_1380, *[existing_type_1381, type__1382], **kwargs_1383)
        
        # Assigning a type to the variable 'value_to_store' (line 812)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 8), 'value_to_store', add_call_result_1384)
        
        # Call to __check_undefined_stored_value(...): (line 813)
        # Processing the call arguments (line 813)
        # Getting the type of 'localization' (line 813)
        localization_1387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 44), 'localization', False)
        # Getting the type of 'value_to_store' (line 813)
        value_to_store_1388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 58), 'value_to_store', False)
        # Processing the call keyword arguments (line 813)
        kwargs_1389 = {}
        # Getting the type of 'self' (line 813)
        self_1385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 8), 'self', False)
        # Obtaining the member '__check_undefined_stored_value' of a type (line 813)
        check_undefined_stored_value_1386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 813, 8), self_1385, '__check_undefined_stored_value')
        # Calling __check_undefined_stored_value(args, kwargs) (line 813)
        check_undefined_stored_value_call_result_1390 = invoke(stypy.reporting.localization.Localization(__file__, 813, 8), check_undefined_stored_value_1386, *[localization_1387, value_to_store_1388], **kwargs_1389)
        
        
        # Call to setattr(...): (line 815)
        # Processing the call arguments (line 815)
        # Getting the type of 'self' (line 815)
        self_1392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 16), 'self', False)
        # Getting the type of 'self' (line 815)
        self_1393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 22), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 815)
        contained_elements_property_name_1394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 815, 22), self_1393, 'contained_elements_property_name')
        # Getting the type of 'value_to_store' (line 815)
        value_to_store_1395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 61), 'value_to_store', False)
        # Processing the call keyword arguments (line 815)
        kwargs_1396 = {}
        # Getting the type of 'setattr' (line 815)
        setattr_1391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 8), 'setattr', False)
        # Calling setattr(args, kwargs) (line 815)
        setattr_call_result_1397 = invoke(stypy.reporting.localization.Localization(__file__, 815, 8), setattr_1391, *[self_1392, contained_elements_property_name_1394, value_to_store_1395], **kwargs_1396)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'record_annotation' (line 817)
        record_annotation_1398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 11), 'record_annotation')
        # Getting the type of 'self' (line 817)
        self_1399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 33), 'self')
        # Obtaining the member 'annotate_types' of a type (line 817)
        annotate_types_1400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 33), self_1399, 'annotate_types')
        # Applying the binary operator 'and' (line 817)
        result_and_keyword_1401 = python_operator(stypy.reporting.localization.Localization(__file__, 817, 11), 'and', record_annotation_1398, annotate_types_1400)
        
        # Testing if the type of an if condition is none (line 817)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 817, 8), result_and_keyword_1401):
            pass
        else:
            
            # Testing the type of an if condition (line 817)
            if_condition_1402 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 817, 8), result_and_keyword_1401)
            # Assigning a type to the variable 'if_condition_1402' (line 817)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 8), 'if_condition_1402', if_condition_1402)
            # SSA begins for if statement (line 817)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to __annotate_type(...): (line 818)
            # Processing the call arguments (line 818)
            # Getting the type of 'localization' (line 818)
            localization_1405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 33), 'localization', False)
            # Obtaining the member 'line' of a type (line 818)
            line_1406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 818, 33), localization_1405, 'line')
            # Getting the type of 'localization' (line 818)
            localization_1407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 52), 'localization', False)
            # Obtaining the member 'column' of a type (line 818)
            column_1408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 818, 52), localization_1407, 'column')
            str_1409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 73), 'str', '<container elements type>')
            
            # Call to getattr(...): (line 819)
            # Processing the call arguments (line 819)
            # Getting the type of 'self' (line 819)
            self_1411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 41), 'self', False)
            # Getting the type of 'self' (line 819)
            self_1412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 47), 'self', False)
            # Obtaining the member 'contained_elements_property_name' of a type (line 819)
            contained_elements_property_name_1413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 819, 47), self_1412, 'contained_elements_property_name')
            # Processing the call keyword arguments (line 819)
            kwargs_1414 = {}
            # Getting the type of 'getattr' (line 819)
            getattr_1410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 33), 'getattr', False)
            # Calling getattr(args, kwargs) (line 819)
            getattr_call_result_1415 = invoke(stypy.reporting.localization.Localization(__file__, 819, 33), getattr_1410, *[self_1411, contained_elements_property_name_1413], **kwargs_1414)
            
            # Processing the call keyword arguments (line 818)
            kwargs_1416 = {}
            # Getting the type of 'self' (line 818)
            self_1403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 12), 'self', False)
            # Obtaining the member '__annotate_type' of a type (line 818)
            annotate_type_1404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 818, 12), self_1403, '__annotate_type')
            # Calling __annotate_type(args, kwargs) (line 818)
            annotate_type_call_result_1417 = invoke(stypy.reporting.localization.Localization(__file__, 818, 12), annotate_type_1404, *[line_1406, column_1408, str_1409, getattr_call_result_1415], **kwargs_1416)
            
            # SSA join for if statement (line 817)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'add_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_type' in the type store
        # Getting the type of 'stypy_return_type' (line 794)
        stypy_return_type_1418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1418)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_type'
        return stypy_return_type_1418


    @norecursion
    def add_types_from_list(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 821)
        True_1419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 77), 'True')
        defaults = [True_1419]
        # Create a new context for function 'add_types_from_list'
        module_type_store = module_type_store.open_function_context('add_types_from_list', 821, 4, False)
        # Assigning a type to the variable 'self' (line 822)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 822, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.add_types_from_list.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.add_types_from_list.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.add_types_from_list.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.add_types_from_list.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.add_types_from_list')
        TypeInferenceProxy.add_types_from_list.__dict__.__setitem__('stypy_param_names_list', ['localization', 'type_list', 'record_annotation'])
        TypeInferenceProxy.add_types_from_list.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.add_types_from_list.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.add_types_from_list.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.add_types_from_list.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.add_types_from_list.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.add_types_from_list.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.add_types_from_list', ['localization', 'type_list', 'record_annotation'], None, None, defaults, varargs, kwargs)

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

        str_1420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, (-1)), 'str', '\n        Adds the types on type_list to the elements stored by this type, returning an error if this is called over a\n        proxy that represent a non element holding Python type. It also checks if we are trying to store an undefined\n        variable.\n        :param localization: Caller information\n        :param type_list: List of types to add\n        :param record_annotation: Whether to annotate the type change or not\n        :return: None or TypeError\n        ')
        
        
        # Call to can_store_elements(...): (line 831)
        # Processing the call keyword arguments (line 831)
        kwargs_1423 = {}
        # Getting the type of 'self' (line 831)
        self_1421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 15), 'self', False)
        # Obtaining the member 'can_store_elements' of a type (line 831)
        can_store_elements_1422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 831, 15), self_1421, 'can_store_elements')
        # Calling can_store_elements(args, kwargs) (line 831)
        can_store_elements_call_result_1424 = invoke(stypy.reporting.localization.Localization(__file__, 831, 15), can_store_elements_1422, *[], **kwargs_1423)
        
        # Applying the 'not' unary operator (line 831)
        result_not__1425 = python_operator(stypy.reporting.localization.Localization(__file__, 831, 11), 'not', can_store_elements_call_result_1424)
        
        # Testing if the type of an if condition is none (line 831)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 831, 8), result_not__1425):
            pass
        else:
            
            # Testing the type of an if condition (line 831)
            if_condition_1426 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 831, 8), result_not__1425)
            # Assigning a type to the variable 'if_condition_1426' (line 831)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'if_condition_1426', if_condition_1426)
            # SSA begins for if statement (line 831)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 832)
            # Processing the call arguments (line 832)
            # Getting the type of 'localization' (line 832)
            localization_1428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 29), 'localization', False)
            
            # Call to format(...): (line 833)
            # Processing the call arguments (line 833)
            
            # Call to get_python_type(...): (line 834)
            # Processing the call keyword arguments (line 834)
            kwargs_1433 = {}
            # Getting the type of 'self' (line 834)
            self_1431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 53), 'self', False)
            # Obtaining the member 'get_python_type' of a type (line 834)
            get_python_type_1432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 53), self_1431, 'get_python_type')
            # Calling get_python_type(args, kwargs) (line 834)
            get_python_type_call_result_1434 = invoke(stypy.reporting.localization.Localization(__file__, 834, 53), get_python_type_1432, *[], **kwargs_1433)
            
            # Processing the call keyword arguments (line 833)
            kwargs_1435 = {}
            str_1429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 833, 29), 'str', 'STYPY CRITICAL ERROR: Attempt to store elements over a python type ({0}) that is not able to do it')
            # Obtaining the member 'format' of a type (line 833)
            format_1430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 833, 29), str_1429, 'format')
            # Calling format(args, kwargs) (line 833)
            format_call_result_1436 = invoke(stypy.reporting.localization.Localization(__file__, 833, 29), format_1430, *[get_python_type_call_result_1434], **kwargs_1435)
            
            # Processing the call keyword arguments (line 832)
            kwargs_1437 = {}
            # Getting the type of 'TypeError' (line 832)
            TypeError_1427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 832)
            TypeError_call_result_1438 = invoke(stypy.reporting.localization.Localization(__file__, 832, 19), TypeError_1427, *[localization_1428, format_call_result_1436], **kwargs_1437)
            
            # Assigning a type to the variable 'stypy_return_type' (line 832)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 832, 12), 'stypy_return_type', TypeError_call_result_1438)
            # SSA join for if statement (line 831)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to hasattr(...): (line 836)
        # Processing the call arguments (line 836)
        # Getting the type of 'self' (line 836)
        self_1440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 19), 'self', False)
        # Getting the type of 'self' (line 836)
        self_1441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 25), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 836)
        contained_elements_property_name_1442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 836, 25), self_1441, 'contained_elements_property_name')
        # Processing the call keyword arguments (line 836)
        kwargs_1443 = {}
        # Getting the type of 'hasattr' (line 836)
        hasattr_1439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 11), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 836)
        hasattr_call_result_1444 = invoke(stypy.reporting.localization.Localization(__file__, 836, 11), hasattr_1439, *[self_1440, contained_elements_property_name_1442], **kwargs_1443)
        
        # Testing if the type of an if condition is none (line 836)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 836, 8), hasattr_call_result_1444):
            pass
        else:
            
            # Testing the type of an if condition (line 836)
            if_condition_1445 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 836, 8), hasattr_call_result_1444)
            # Assigning a type to the variable 'if_condition_1445' (line 836)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 836, 8), 'if_condition_1445', if_condition_1445)
            # SSA begins for if statement (line 836)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 837):
            
            # Assigning a Call to a Name (line 837):
            
            # Call to getattr(...): (line 837)
            # Processing the call arguments (line 837)
            # Getting the type of 'self' (line 837)
            self_1447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 36), 'self', False)
            # Getting the type of 'self' (line 837)
            self_1448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 42), 'self', False)
            # Obtaining the member 'contained_elements_property_name' of a type (line 837)
            contained_elements_property_name_1449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 42), self_1448, 'contained_elements_property_name')
            # Processing the call keyword arguments (line 837)
            kwargs_1450 = {}
            # Getting the type of 'getattr' (line 837)
            getattr_1446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 28), 'getattr', False)
            # Calling getattr(args, kwargs) (line 837)
            getattr_call_result_1451 = invoke(stypy.reporting.localization.Localization(__file__, 837, 28), getattr_1446, *[self_1447, contained_elements_property_name_1449], **kwargs_1450)
            
            # Assigning a type to the variable 'existing_type' (line 837)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 12), 'existing_type', getattr_call_result_1451)
            
            # Assigning a BinOp to a Name (line 838):
            
            # Assigning a BinOp to a Name (line 838):
            
            # Obtaining an instance of the builtin type 'list' (line 838)
            list_1452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 24), 'list')
            # Adding type elements to the builtin type 'list' instance (line 838)
            # Adding element type (line 838)
            # Getting the type of 'existing_type' (line 838)
            existing_type_1453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 25), 'existing_type')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 838, 24), list_1452, existing_type_1453)
            
            # Getting the type of 'type_list' (line 838)
            type_list_1454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 42), 'type_list')
            # Applying the binary operator '+' (line 838)
            result_add_1455 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 24), '+', list_1452, type_list_1454)
            
            # Assigning a type to the variable 'type_list' (line 838)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 12), 'type_list', result_add_1455)
            # SSA join for if statement (line 836)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to setattr(...): (line 840)
        # Processing the call arguments (line 840)
        # Getting the type of 'self' (line 840)
        self_1457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 16), 'self', False)
        # Getting the type of 'self' (line 840)
        self_1458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 22), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 840)
        contained_elements_property_name_1459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 22), self_1458, 'contained_elements_property_name')
        
        # Call to create_union_type_from_types(...): (line 841)
        # Getting the type of 'type_list' (line 841)
        type_list_1463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 72), 'type_list', False)
        # Processing the call keyword arguments (line 841)
        kwargs_1464 = {}
        # Getting the type of 'union_type_copy' (line 841)
        union_type_copy_1460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 16), 'union_type_copy', False)
        # Obtaining the member 'UnionType' of a type (line 841)
        UnionType_1461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 841, 16), union_type_copy_1460, 'UnionType')
        # Obtaining the member 'create_union_type_from_types' of a type (line 841)
        create_union_type_from_types_1462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 841, 16), UnionType_1461, 'create_union_type_from_types')
        # Calling create_union_type_from_types(args, kwargs) (line 841)
        create_union_type_from_types_call_result_1465 = invoke(stypy.reporting.localization.Localization(__file__, 841, 16), create_union_type_from_types_1462, *[type_list_1463], **kwargs_1464)
        
        # Processing the call keyword arguments (line 840)
        kwargs_1466 = {}
        # Getting the type of 'setattr' (line 840)
        setattr_1456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 8), 'setattr', False)
        # Calling setattr(args, kwargs) (line 840)
        setattr_call_result_1467 = invoke(stypy.reporting.localization.Localization(__file__, 840, 8), setattr_1456, *[self_1457, contained_elements_property_name_1459, create_union_type_from_types_call_result_1465], **kwargs_1466)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'record_annotation' (line 842)
        record_annotation_1468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 11), 'record_annotation')
        # Getting the type of 'self' (line 842)
        self_1469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 33), 'self')
        # Obtaining the member 'annotate_types' of a type (line 842)
        annotate_types_1470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 842, 33), self_1469, 'annotate_types')
        # Applying the binary operator 'and' (line 842)
        result_and_keyword_1471 = python_operator(stypy.reporting.localization.Localization(__file__, 842, 11), 'and', record_annotation_1468, annotate_types_1470)
        
        # Testing if the type of an if condition is none (line 842)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 842, 8), result_and_keyword_1471):
            pass
        else:
            
            # Testing the type of an if condition (line 842)
            if_condition_1472 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 842, 8), result_and_keyword_1471)
            # Assigning a type to the variable 'if_condition_1472' (line 842)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 842, 8), 'if_condition_1472', if_condition_1472)
            # SSA begins for if statement (line 842)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to __annotate_type(...): (line 843)
            # Processing the call arguments (line 843)
            # Getting the type of 'localization' (line 843)
            localization_1475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 33), 'localization', False)
            # Obtaining the member 'line' of a type (line 843)
            line_1476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 33), localization_1475, 'line')
            # Getting the type of 'localization' (line 843)
            localization_1477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 52), 'localization', False)
            # Obtaining the member 'column' of a type (line 843)
            column_1478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 52), localization_1477, 'column')
            str_1479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 843, 73), 'str', '<container elements type>')
            
            # Call to getattr(...): (line 844)
            # Processing the call arguments (line 844)
            # Getting the type of 'self' (line 844)
            self_1481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 41), 'self', False)
            # Getting the type of 'self' (line 844)
            self_1482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 47), 'self', False)
            # Obtaining the member 'contained_elements_property_name' of a type (line 844)
            contained_elements_property_name_1483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 47), self_1482, 'contained_elements_property_name')
            # Processing the call keyword arguments (line 844)
            kwargs_1484 = {}
            # Getting the type of 'getattr' (line 844)
            getattr_1480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 33), 'getattr', False)
            # Calling getattr(args, kwargs) (line 844)
            getattr_call_result_1485 = invoke(stypy.reporting.localization.Localization(__file__, 844, 33), getattr_1480, *[self_1481, contained_elements_property_name_1483], **kwargs_1484)
            
            # Processing the call keyword arguments (line 843)
            kwargs_1486 = {}
            # Getting the type of 'self' (line 843)
            self_1473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 12), 'self', False)
            # Obtaining the member '__annotate_type' of a type (line 843)
            annotate_type_1474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 12), self_1473, '__annotate_type')
            # Calling __annotate_type(args, kwargs) (line 843)
            annotate_type_call_result_1487 = invoke(stypy.reporting.localization.Localization(__file__, 843, 12), annotate_type_1474, *[line_1476, column_1478, str_1479, getattr_call_result_1485], **kwargs_1486)
            
            # SSA join for if statement (line 842)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'add_types_from_list(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_types_from_list' in the type store
        # Getting the type of 'stypy_return_type' (line 821)
        stypy_return_type_1488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1488)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_types_from_list'
        return stypy_return_type_1488


    @norecursion
    def __exist_key(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__exist_key'
        module_type_store = module_type_store.open_function_context('__exist_key', 846, 4, False)
        # Assigning a type to the variable 'self' (line 847)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 847, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.__exist_key.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.__exist_key.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.__exist_key.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.__exist_key.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.__exist_key')
        TypeInferenceProxy.__exist_key.__dict__.__setitem__('stypy_param_names_list', ['key'])
        TypeInferenceProxy.__exist_key.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.__exist_key.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.__exist_key.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.__exist_key.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.__exist_key.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.__exist_key.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.__exist_key', ['key'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__exist_key', localization, ['key'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__exist_key(...)' code ##################

        str_1489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 851, (-1)), 'str', '\n        Helper method to see if the stored keypairs contains a key equal to the passed one.\n        :param key:\n        :return:\n        ')
        
        # Assigning a Call to a Name (line 852):
        
        # Assigning a Call to a Name (line 852):
        
        # Call to getattr(...): (line 852)
        # Processing the call arguments (line 852)
        # Getting the type of 'self' (line 852)
        self_1491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 36), 'self', False)
        # Getting the type of 'self' (line 852)
        self_1492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 42), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 852)
        contained_elements_property_name_1493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 852, 42), self_1492, 'contained_elements_property_name')
        # Processing the call keyword arguments (line 852)
        kwargs_1494 = {}
        # Getting the type of 'getattr' (line 852)
        getattr_1490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 28), 'getattr', False)
        # Calling getattr(args, kwargs) (line 852)
        getattr_call_result_1495 = invoke(stypy.reporting.localization.Localization(__file__, 852, 28), getattr_1490, *[self_1491, contained_elements_property_name_1493], **kwargs_1494)
        
        # Assigning a type to the variable 'existing_type_map' (line 852)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 852, 8), 'existing_type_map', getattr_call_result_1495)
        
        # Assigning a Call to a Name (line 853):
        
        # Assigning a Call to a Name (line 853):
        
        # Call to keys(...): (line 853)
        # Processing the call keyword arguments (line 853)
        kwargs_1498 = {}
        # Getting the type of 'existing_type_map' (line 853)
        existing_type_map_1496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 15), 'existing_type_map', False)
        # Obtaining the member 'keys' of a type (line 853)
        keys_1497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 853, 15), existing_type_map_1496, 'keys')
        # Calling keys(args, kwargs) (line 853)
        keys_call_result_1499 = invoke(stypy.reporting.localization.Localization(__file__, 853, 15), keys_1497, *[], **kwargs_1498)
        
        # Assigning a type to the variable 'keys' (line 853)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 853, 8), 'keys', keys_call_result_1499)
        
        # Getting the type of 'keys' (line 854)
        keys_1500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 23), 'keys')
        # Assigning a type to the variable 'keys_1500' (line 854)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 854, 8), 'keys_1500', keys_1500)
        # Testing if the for loop is going to be iterated (line 854)
        # Testing the type of a for loop iterable (line 854)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 854, 8), keys_1500)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 854, 8), keys_1500):
            # Getting the type of the for loop variable (line 854)
            for_loop_var_1501 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 854, 8), keys_1500)
            # Assigning a type to the variable 'element' (line 854)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 854, 8), 'element', for_loop_var_1501)
            # SSA begins for a for statement (line 854)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'key' (line 855)
            key_1502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 15), 'key')
            # Getting the type of 'element' (line 855)
            element_1503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 22), 'element')
            # Applying the binary operator '==' (line 855)
            result_eq_1504 = python_operator(stypy.reporting.localization.Localization(__file__, 855, 15), '==', key_1502, element_1503)
            
            # Testing if the type of an if condition is none (line 855)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 855, 12), result_eq_1504):
                pass
            else:
                
                # Testing the type of an if condition (line 855)
                if_condition_1505 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 855, 12), result_eq_1504)
                # Assigning a type to the variable 'if_condition_1505' (line 855)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 855, 12), 'if_condition_1505', if_condition_1505)
                # SSA begins for if statement (line 855)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'True' (line 856)
                True_1506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 23), 'True')
                # Assigning a type to the variable 'stypy_return_type' (line 856)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 856, 16), 'stypy_return_type', True_1506)
                # SSA join for if statement (line 855)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'False' (line 857)
        False_1507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 857)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 857, 8), 'stypy_return_type', False_1507)
        
        # ################# End of '__exist_key(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__exist_key' in the type store
        # Getting the type of 'stypy_return_type' (line 846)
        stypy_return_type_1508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1508)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__exist_key'
        return stypy_return_type_1508


    @norecursion
    def add_key_and_value_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 859)
        True_1509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 81), 'True')
        defaults = [True_1509]
        # Create a new context for function 'add_key_and_value_type'
        module_type_store = module_type_store.open_function_context('add_key_and_value_type', 859, 4, False)
        # Assigning a type to the variable 'self' (line 860)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 860, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.add_key_and_value_type.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.add_key_and_value_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.add_key_and_value_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.add_key_and_value_type.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.add_key_and_value_type')
        TypeInferenceProxy.add_key_and_value_type.__dict__.__setitem__('stypy_param_names_list', ['localization', 'type_tuple', 'record_annotation'])
        TypeInferenceProxy.add_key_and_value_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.add_key_and_value_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.add_key_and_value_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.add_key_and_value_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.add_key_and_value_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.add_key_and_value_type.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.add_key_and_value_type', ['localization', 'type_tuple', 'record_annotation'], None, None, defaults, varargs, kwargs)

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

        str_1510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 867, (-1)), 'str', '\n        Adds type_tuple to the elements stored by this type, returning an error if this is called over a proxy that\n        represent a non keypair holding Python type. It also checks if we are trying to store an undefined variable.\n        :param localization: Caller information\n        :param type_tuple: Tuple of types to store (key type, value type)\n        :param record_annotation: Whether to annotate the type change or not\n        :return: None or TypeError\n        ')
        
        # Assigning a Subscript to a Name (line 868):
        
        # Assigning a Subscript to a Name (line 868):
        
        # Obtaining the type of the subscript
        int_1511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 25), 'int')
        # Getting the type of 'type_tuple' (line 868)
        type_tuple_1512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 14), 'type_tuple')
        # Obtaining the member '__getitem__' of a type (line 868)
        getitem___1513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 868, 14), type_tuple_1512, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 868)
        subscript_call_result_1514 = invoke(stypy.reporting.localization.Localization(__file__, 868, 14), getitem___1513, int_1511)
        
        # Assigning a type to the variable 'key' (line 868)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 8), 'key', subscript_call_result_1514)
        
        # Assigning a Subscript to a Name (line 869):
        
        # Assigning a Subscript to a Name (line 869):
        
        # Obtaining the type of the subscript
        int_1515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 869, 27), 'int')
        # Getting the type of 'type_tuple' (line 869)
        type_tuple_1516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 16), 'type_tuple')
        # Obtaining the member '__getitem__' of a type (line 869)
        getitem___1517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 869, 16), type_tuple_1516, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 869)
        subscript_call_result_1518 = invoke(stypy.reporting.localization.Localization(__file__, 869, 16), getitem___1517, int_1515)
        
        # Assigning a type to the variable 'value' (line 869)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 869, 8), 'value', subscript_call_result_1518)
        
        
        # Call to can_store_keypairs(...): (line 871)
        # Processing the call keyword arguments (line 871)
        kwargs_1521 = {}
        # Getting the type of 'self' (line 871)
        self_1519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 15), 'self', False)
        # Obtaining the member 'can_store_keypairs' of a type (line 871)
        can_store_keypairs_1520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 15), self_1519, 'can_store_keypairs')
        # Calling can_store_keypairs(args, kwargs) (line 871)
        can_store_keypairs_call_result_1522 = invoke(stypy.reporting.localization.Localization(__file__, 871, 15), can_store_keypairs_1520, *[], **kwargs_1521)
        
        # Applying the 'not' unary operator (line 871)
        result_not__1523 = python_operator(stypy.reporting.localization.Localization(__file__, 871, 11), 'not', can_store_keypairs_call_result_1522)
        
        # Testing if the type of an if condition is none (line 871)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 871, 8), result_not__1523):
            pass
        else:
            
            # Testing the type of an if condition (line 871)
            if_condition_1524 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 871, 8), result_not__1523)
            # Assigning a type to the variable 'if_condition_1524' (line 871)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 871, 8), 'if_condition_1524', if_condition_1524)
            # SSA begins for if statement (line 871)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to can_store_elements(...): (line 872)
            # Processing the call keyword arguments (line 872)
            kwargs_1527 = {}
            # Getting the type of 'self' (line 872)
            self_1525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 19), 'self', False)
            # Obtaining the member 'can_store_elements' of a type (line 872)
            can_store_elements_1526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 872, 19), self_1525, 'can_store_elements')
            # Calling can_store_elements(args, kwargs) (line 872)
            can_store_elements_call_result_1528 = invoke(stypy.reporting.localization.Localization(__file__, 872, 19), can_store_elements_1526, *[], **kwargs_1527)
            
            # Applying the 'not' unary operator (line 872)
            result_not__1529 = python_operator(stypy.reporting.localization.Localization(__file__, 872, 15), 'not', can_store_elements_call_result_1528)
            
            # Testing if the type of an if condition is none (line 872)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 872, 12), result_not__1529):
                
                
                # Call to get_python_type(...): (line 877)
                # Processing the call keyword arguments (line 877)
                kwargs_1545 = {}
                # Getting the type of 'key' (line 877)
                key_1543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 19), 'key', False)
                # Obtaining the member 'get_python_type' of a type (line 877)
                get_python_type_1544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 19), key_1543, 'get_python_type')
                # Calling get_python_type(args, kwargs) (line 877)
                get_python_type_call_result_1546 = invoke(stypy.reporting.localization.Localization(__file__, 877, 19), get_python_type_1544, *[], **kwargs_1545)
                
                # Getting the type of 'int' (line 877)
                int_1547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 48), 'int')
                # Applying the binary operator 'isnot' (line 877)
                result_is_not_1548 = python_operator(stypy.reporting.localization.Localization(__file__, 877, 19), 'isnot', get_python_type_call_result_1546, int_1547)
                
                # Testing if the type of an if condition is none (line 877)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 877, 16), result_is_not_1548):
                    
                    # Call to add_type(...): (line 881)
                    # Processing the call arguments (line 881)
                    # Getting the type of 'localization' (line 881)
                    localization_1557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 34), 'localization', False)
                    # Getting the type of 'value' (line 881)
                    value_1558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 48), 'value', False)
                    # Getting the type of 'record_annotation' (line 881)
                    record_annotation_1559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 55), 'record_annotation', False)
                    # Processing the call keyword arguments (line 881)
                    kwargs_1560 = {}
                    # Getting the type of 'self' (line 881)
                    self_1555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 20), 'self', False)
                    # Obtaining the member 'add_type' of a type (line 881)
                    add_type_1556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 881, 20), self_1555, 'add_type')
                    # Calling add_type(args, kwargs) (line 881)
                    add_type_call_result_1561 = invoke(stypy.reporting.localization.Localization(__file__, 881, 20), add_type_1556, *[localization_1557, value_1558, record_annotation_1559], **kwargs_1560)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 882)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 20), 'stypy_return_type', types.NoneType)
                else:
                    
                    # Testing the type of an if condition (line 877)
                    if_condition_1549 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 877, 16), result_is_not_1548)
                    # Assigning a type to the variable 'if_condition_1549' (line 877)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 877, 16), 'if_condition_1549', if_condition_1549)
                    # SSA begins for if statement (line 877)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to TypeError(...): (line 878)
                    # Processing the call arguments (line 878)
                    # Getting the type of 'localization' (line 878)
                    localization_1551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 37), 'localization', False)
                    str_1552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 37), 'str', 'STYPY CRITICAL ERROR: Attempt to store keypairs on a python collection')
                    # Processing the call keyword arguments (line 878)
                    kwargs_1553 = {}
                    # Getting the type of 'TypeError' (line 878)
                    TypeError_1550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 27), 'TypeError', False)
                    # Calling TypeError(args, kwargs) (line 878)
                    TypeError_call_result_1554 = invoke(stypy.reporting.localization.Localization(__file__, 878, 27), TypeError_1550, *[localization_1551, str_1552], **kwargs_1553)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 878)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 20), 'stypy_return_type', TypeError_call_result_1554)
                    # SSA branch for the else part of an if statement (line 877)
                    module_type_store.open_ssa_branch('else')
                    
                    # Call to add_type(...): (line 881)
                    # Processing the call arguments (line 881)
                    # Getting the type of 'localization' (line 881)
                    localization_1557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 34), 'localization', False)
                    # Getting the type of 'value' (line 881)
                    value_1558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 48), 'value', False)
                    # Getting the type of 'record_annotation' (line 881)
                    record_annotation_1559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 55), 'record_annotation', False)
                    # Processing the call keyword arguments (line 881)
                    kwargs_1560 = {}
                    # Getting the type of 'self' (line 881)
                    self_1555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 20), 'self', False)
                    # Obtaining the member 'add_type' of a type (line 881)
                    add_type_1556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 881, 20), self_1555, 'add_type')
                    # Calling add_type(args, kwargs) (line 881)
                    add_type_call_result_1561 = invoke(stypy.reporting.localization.Localization(__file__, 881, 20), add_type_1556, *[localization_1557, value_1558, record_annotation_1559], **kwargs_1560)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 882)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 20), 'stypy_return_type', types.NoneType)
                    # SSA join for if statement (line 877)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 872)
                if_condition_1530 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 872, 12), result_not__1529)
                # Assigning a type to the variable 'if_condition_1530' (line 872)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 872, 12), 'if_condition_1530', if_condition_1530)
                # SSA begins for if statement (line 872)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to TypeError(...): (line 873)
                # Processing the call arguments (line 873)
                # Getting the type of 'localization' (line 873)
                localization_1532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 33), 'localization', False)
                
                # Call to format(...): (line 874)
                # Processing the call arguments (line 874)
                
                # Call to get_python_type(...): (line 875)
                # Processing the call keyword arguments (line 875)
                kwargs_1537 = {}
                # Getting the type of 'self' (line 875)
                self_1535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 49), 'self', False)
                # Obtaining the member 'get_python_type' of a type (line 875)
                get_python_type_1536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 875, 49), self_1535, 'get_python_type')
                # Calling get_python_type(args, kwargs) (line 875)
                get_python_type_call_result_1538 = invoke(stypy.reporting.localization.Localization(__file__, 875, 49), get_python_type_1536, *[], **kwargs_1537)
                
                # Processing the call keyword arguments (line 874)
                kwargs_1539 = {}
                str_1533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, 33), 'str', 'STYPY CRITICAL ERROR: Attempt to store keypairs over a python type ({0}) that is nota dict')
                # Obtaining the member 'format' of a type (line 874)
                format_1534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 874, 33), str_1533, 'format')
                # Calling format(args, kwargs) (line 874)
                format_call_result_1540 = invoke(stypy.reporting.localization.Localization(__file__, 874, 33), format_1534, *[get_python_type_call_result_1538], **kwargs_1539)
                
                # Processing the call keyword arguments (line 873)
                kwargs_1541 = {}
                # Getting the type of 'TypeError' (line 873)
                TypeError_1531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 23), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 873)
                TypeError_call_result_1542 = invoke(stypy.reporting.localization.Localization(__file__, 873, 23), TypeError_1531, *[localization_1532, format_call_result_1540], **kwargs_1541)
                
                # Assigning a type to the variable 'stypy_return_type' (line 873)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 873, 16), 'stypy_return_type', TypeError_call_result_1542)
                # SSA branch for the else part of an if statement (line 872)
                module_type_store.open_ssa_branch('else')
                
                
                # Call to get_python_type(...): (line 877)
                # Processing the call keyword arguments (line 877)
                kwargs_1545 = {}
                # Getting the type of 'key' (line 877)
                key_1543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 19), 'key', False)
                # Obtaining the member 'get_python_type' of a type (line 877)
                get_python_type_1544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 19), key_1543, 'get_python_type')
                # Calling get_python_type(args, kwargs) (line 877)
                get_python_type_call_result_1546 = invoke(stypy.reporting.localization.Localization(__file__, 877, 19), get_python_type_1544, *[], **kwargs_1545)
                
                # Getting the type of 'int' (line 877)
                int_1547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 48), 'int')
                # Applying the binary operator 'isnot' (line 877)
                result_is_not_1548 = python_operator(stypy.reporting.localization.Localization(__file__, 877, 19), 'isnot', get_python_type_call_result_1546, int_1547)
                
                # Testing if the type of an if condition is none (line 877)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 877, 16), result_is_not_1548):
                    
                    # Call to add_type(...): (line 881)
                    # Processing the call arguments (line 881)
                    # Getting the type of 'localization' (line 881)
                    localization_1557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 34), 'localization', False)
                    # Getting the type of 'value' (line 881)
                    value_1558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 48), 'value', False)
                    # Getting the type of 'record_annotation' (line 881)
                    record_annotation_1559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 55), 'record_annotation', False)
                    # Processing the call keyword arguments (line 881)
                    kwargs_1560 = {}
                    # Getting the type of 'self' (line 881)
                    self_1555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 20), 'self', False)
                    # Obtaining the member 'add_type' of a type (line 881)
                    add_type_1556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 881, 20), self_1555, 'add_type')
                    # Calling add_type(args, kwargs) (line 881)
                    add_type_call_result_1561 = invoke(stypy.reporting.localization.Localization(__file__, 881, 20), add_type_1556, *[localization_1557, value_1558, record_annotation_1559], **kwargs_1560)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 882)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 20), 'stypy_return_type', types.NoneType)
                else:
                    
                    # Testing the type of an if condition (line 877)
                    if_condition_1549 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 877, 16), result_is_not_1548)
                    # Assigning a type to the variable 'if_condition_1549' (line 877)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 877, 16), 'if_condition_1549', if_condition_1549)
                    # SSA begins for if statement (line 877)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to TypeError(...): (line 878)
                    # Processing the call arguments (line 878)
                    # Getting the type of 'localization' (line 878)
                    localization_1551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 37), 'localization', False)
                    str_1552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 37), 'str', 'STYPY CRITICAL ERROR: Attempt to store keypairs on a python collection')
                    # Processing the call keyword arguments (line 878)
                    kwargs_1553 = {}
                    # Getting the type of 'TypeError' (line 878)
                    TypeError_1550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 27), 'TypeError', False)
                    # Calling TypeError(args, kwargs) (line 878)
                    TypeError_call_result_1554 = invoke(stypy.reporting.localization.Localization(__file__, 878, 27), TypeError_1550, *[localization_1551, str_1552], **kwargs_1553)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 878)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 20), 'stypy_return_type', TypeError_call_result_1554)
                    # SSA branch for the else part of an if statement (line 877)
                    module_type_store.open_ssa_branch('else')
                    
                    # Call to add_type(...): (line 881)
                    # Processing the call arguments (line 881)
                    # Getting the type of 'localization' (line 881)
                    localization_1557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 34), 'localization', False)
                    # Getting the type of 'value' (line 881)
                    value_1558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 48), 'value', False)
                    # Getting the type of 'record_annotation' (line 881)
                    record_annotation_1559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 55), 'record_annotation', False)
                    # Processing the call keyword arguments (line 881)
                    kwargs_1560 = {}
                    # Getting the type of 'self' (line 881)
                    self_1555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 20), 'self', False)
                    # Obtaining the member 'add_type' of a type (line 881)
                    add_type_1556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 881, 20), self_1555, 'add_type')
                    # Calling add_type(args, kwargs) (line 881)
                    add_type_call_result_1561 = invoke(stypy.reporting.localization.Localization(__file__, 881, 20), add_type_1556, *[localization_1557, value_1558, record_annotation_1559], **kwargs_1560)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 882)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 20), 'stypy_return_type', types.NoneType)
                    # SSA join for if statement (line 877)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 872)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 871)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to hasattr(...): (line 884)
        # Processing the call arguments (line 884)
        # Getting the type of 'self' (line 884)
        self_1563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 23), 'self', False)
        # Getting the type of 'self' (line 884)
        self_1564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 29), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 884)
        contained_elements_property_name_1565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 884, 29), self_1564, 'contained_elements_property_name')
        # Processing the call keyword arguments (line 884)
        kwargs_1566 = {}
        # Getting the type of 'hasattr' (line 884)
        hasattr_1562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 15), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 884)
        hasattr_call_result_1567 = invoke(stypy.reporting.localization.Localization(__file__, 884, 15), hasattr_1562, *[self_1563, contained_elements_property_name_1565], **kwargs_1566)
        
        # Applying the 'not' unary operator (line 884)
        result_not__1568 = python_operator(stypy.reporting.localization.Localization(__file__, 884, 11), 'not', hasattr_call_result_1567)
        
        # Testing if the type of an if condition is none (line 884)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 884, 8), result_not__1568):
            pass
        else:
            
            # Testing the type of an if condition (line 884)
            if_condition_1569 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 884, 8), result_not__1568)
            # Assigning a type to the variable 'if_condition_1569' (line 884)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 884, 8), 'if_condition_1569', if_condition_1569)
            # SSA begins for if statement (line 884)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to setattr(...): (line 885)
            # Processing the call arguments (line 885)
            # Getting the type of 'self' (line 885)
            self_1571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 20), 'self', False)
            # Getting the type of 'self' (line 885)
            self_1572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 26), 'self', False)
            # Obtaining the member 'contained_elements_property_name' of a type (line 885)
            contained_elements_property_name_1573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 885, 26), self_1572, 'contained_elements_property_name')
            
            # Call to dict(...): (line 885)
            # Processing the call keyword arguments (line 885)
            kwargs_1575 = {}
            # Getting the type of 'dict' (line 885)
            dict_1574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 65), 'dict', False)
            # Calling dict(args, kwargs) (line 885)
            dict_call_result_1576 = invoke(stypy.reporting.localization.Localization(__file__, 885, 65), dict_1574, *[], **kwargs_1575)
            
            # Processing the call keyword arguments (line 885)
            kwargs_1577 = {}
            # Getting the type of 'setattr' (line 885)
            setattr_1570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 12), 'setattr', False)
            # Calling setattr(args, kwargs) (line 885)
            setattr_call_result_1578 = invoke(stypy.reporting.localization.Localization(__file__, 885, 12), setattr_1570, *[self_1571, contained_elements_property_name_1573, dict_call_result_1576], **kwargs_1577)
            
            # SSA join for if statement (line 884)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 887):
        
        # Assigning a Call to a Name (line 887):
        
        # Call to getattr(...): (line 887)
        # Processing the call arguments (line 887)
        # Getting the type of 'self' (line 887)
        self_1580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 36), 'self', False)
        # Getting the type of 'self' (line 887)
        self_1581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 42), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 887)
        contained_elements_property_name_1582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 42), self_1581, 'contained_elements_property_name')
        # Processing the call keyword arguments (line 887)
        kwargs_1583 = {}
        # Getting the type of 'getattr' (line 887)
        getattr_1579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 28), 'getattr', False)
        # Calling getattr(args, kwargs) (line 887)
        getattr_call_result_1584 = invoke(stypy.reporting.localization.Localization(__file__, 887, 28), getattr_1579, *[self_1580, contained_elements_property_name_1582], **kwargs_1583)
        
        # Assigning a type to the variable 'existing_type_map' (line 887)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 887, 8), 'existing_type_map', getattr_call_result_1584)
        
        # Call to __check_undefined_stored_value(...): (line 889)
        # Processing the call arguments (line 889)
        # Getting the type of 'localization' (line 889)
        localization_1587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 44), 'localization', False)
        # Getting the type of 'value' (line 889)
        value_1588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 58), 'value', False)
        # Processing the call keyword arguments (line 889)
        kwargs_1589 = {}
        # Getting the type of 'self' (line 889)
        self_1585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 8), 'self', False)
        # Obtaining the member '__check_undefined_stored_value' of a type (line 889)
        check_undefined_stored_value_1586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 889, 8), self_1585, '__check_undefined_stored_value')
        # Calling __check_undefined_stored_value(args, kwargs) (line 889)
        check_undefined_stored_value_call_result_1590 = invoke(stypy.reporting.localization.Localization(__file__, 889, 8), check_undefined_stored_value_1586, *[localization_1587, value_1588], **kwargs_1589)
        
        
        # Call to __exist_key(...): (line 892)
        # Processing the call arguments (line 892)
        # Getting the type of 'key' (line 892)
        key_1593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 28), 'key', False)
        # Processing the call keyword arguments (line 892)
        kwargs_1594 = {}
        # Getting the type of 'self' (line 892)
        self_1591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 11), 'self', False)
        # Obtaining the member '__exist_key' of a type (line 892)
        exist_key_1592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 892, 11), self_1591, '__exist_key')
        # Calling __exist_key(args, kwargs) (line 892)
        exist_key_call_result_1595 = invoke(stypy.reporting.localization.Localization(__file__, 892, 11), exist_key_1592, *[key_1593], **kwargs_1594)
        
        # Testing if the type of an if condition is none (line 892)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 892, 8), exist_key_call_result_1595):
            
            # Assigning a Name to a Subscript (line 899):
            
            # Assigning a Name to a Subscript (line 899):
            # Getting the type of 'value' (line 899)
            value_1625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 37), 'value')
            # Getting the type of 'existing_type_map' (line 899)
            existing_type_map_1626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 12), 'existing_type_map')
            # Getting the type of 'key' (line 899)
            key_1627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 30), 'key')
            # Storing an element on a container (line 899)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 899, 12), existing_type_map_1626, (key_1627, value_1625))
        else:
            
            # Testing the type of an if condition (line 892)
            if_condition_1596 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 892, 8), exist_key_call_result_1595)
            # Assigning a type to the variable 'if_condition_1596' (line 892)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 8), 'if_condition_1596', if_condition_1596)
            # SSA begins for if statement (line 892)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 894):
            
            # Assigning a Call to a Name (line 894):
            
            # Call to index(...): (line 894)
            # Processing the call arguments (line 894)
            # Getting the type of 'key' (line 894)
            key_1602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 62), 'key', False)
            # Processing the call keyword arguments (line 894)
            kwargs_1603 = {}
            
            # Call to keys(...): (line 894)
            # Processing the call keyword arguments (line 894)
            kwargs_1599 = {}
            # Getting the type of 'existing_type_map' (line 894)
            existing_type_map_1597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 31), 'existing_type_map', False)
            # Obtaining the member 'keys' of a type (line 894)
            keys_1598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 894, 31), existing_type_map_1597, 'keys')
            # Calling keys(args, kwargs) (line 894)
            keys_call_result_1600 = invoke(stypy.reporting.localization.Localization(__file__, 894, 31), keys_1598, *[], **kwargs_1599)
            
            # Obtaining the member 'index' of a type (line 894)
            index_1601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 894, 31), keys_call_result_1600, 'index')
            # Calling index(args, kwargs) (line 894)
            index_call_result_1604 = invoke(stypy.reporting.localization.Localization(__file__, 894, 31), index_1601, *[key_1602], **kwargs_1603)
            
            # Assigning a type to the variable 'stored_key_index' (line 894)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 894, 12), 'stored_key_index', index_call_result_1604)
            
            # Assigning a Subscript to a Name (line 895):
            
            # Assigning a Subscript to a Name (line 895):
            
            # Obtaining the type of the subscript
            # Getting the type of 'stored_key_index' (line 895)
            stored_key_index_1605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 50), 'stored_key_index')
            
            # Call to keys(...): (line 895)
            # Processing the call keyword arguments (line 895)
            kwargs_1608 = {}
            # Getting the type of 'existing_type_map' (line 895)
            existing_type_map_1606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 25), 'existing_type_map', False)
            # Obtaining the member 'keys' of a type (line 895)
            keys_1607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 895, 25), existing_type_map_1606, 'keys')
            # Calling keys(args, kwargs) (line 895)
            keys_call_result_1609 = invoke(stypy.reporting.localization.Localization(__file__, 895, 25), keys_1607, *[], **kwargs_1608)
            
            # Obtaining the member '__getitem__' of a type (line 895)
            getitem___1610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 895, 25), keys_call_result_1609, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 895)
            subscript_call_result_1611 = invoke(stypy.reporting.localization.Localization(__file__, 895, 25), getitem___1610, stored_key_index_1605)
            
            # Assigning a type to the variable 'stored_key' (line 895)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 895, 12), 'stored_key', subscript_call_result_1611)
            
            # Assigning a Subscript to a Name (line 896):
            
            # Assigning a Subscript to a Name (line 896):
            
            # Obtaining the type of the subscript
            # Getting the type of 'stored_key' (line 896)
            stored_key_1612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 46), 'stored_key')
            # Getting the type of 'existing_type_map' (line 896)
            existing_type_map_1613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 28), 'existing_type_map')
            # Obtaining the member '__getitem__' of a type (line 896)
            getitem___1614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 896, 28), existing_type_map_1613, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 896)
            subscript_call_result_1615 = invoke(stypy.reporting.localization.Localization(__file__, 896, 28), getitem___1614, stored_key_1612)
            
            # Assigning a type to the variable 'existing_type' (line 896)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 896, 12), 'existing_type', subscript_call_result_1615)
            
            # Assigning a Call to a Subscript (line 897):
            
            # Assigning a Call to a Subscript (line 897):
            
            # Call to add(...): (line 897)
            # Processing the call arguments (line 897)
            # Getting the type of 'existing_type' (line 897)
            existing_type_1619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 74), 'existing_type', False)
            # Getting the type of 'value' (line 897)
            value_1620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 89), 'value', False)
            # Processing the call keyword arguments (line 897)
            kwargs_1621 = {}
            # Getting the type of 'union_type_copy' (line 897)
            union_type_copy_1616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 44), 'union_type_copy', False)
            # Obtaining the member 'UnionType' of a type (line 897)
            UnionType_1617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 44), union_type_copy_1616, 'UnionType')
            # Obtaining the member 'add' of a type (line 897)
            add_1618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 44), UnionType_1617, 'add')
            # Calling add(args, kwargs) (line 897)
            add_call_result_1622 = invoke(stypy.reporting.localization.Localization(__file__, 897, 44), add_1618, *[existing_type_1619, value_1620], **kwargs_1621)
            
            # Getting the type of 'existing_type_map' (line 897)
            existing_type_map_1623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 12), 'existing_type_map')
            # Getting the type of 'stored_key' (line 897)
            stored_key_1624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 30), 'stored_key')
            # Storing an element on a container (line 897)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 12), existing_type_map_1623, (stored_key_1624, add_call_result_1622))
            # SSA branch for the else part of an if statement (line 892)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Subscript (line 899):
            
            # Assigning a Name to a Subscript (line 899):
            # Getting the type of 'value' (line 899)
            value_1625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 37), 'value')
            # Getting the type of 'existing_type_map' (line 899)
            existing_type_map_1626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 12), 'existing_type_map')
            # Getting the type of 'key' (line 899)
            key_1627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 30), 'key')
            # Storing an element on a container (line 899)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 899, 12), existing_type_map_1626, (key_1627, value_1625))
            # SSA join for if statement (line 892)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Evaluating a boolean operation
        # Getting the type of 'record_annotation' (line 901)
        record_annotation_1628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 11), 'record_annotation')
        # Getting the type of 'self' (line 901)
        self_1629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 33), 'self')
        # Obtaining the member 'annotate_types' of a type (line 901)
        annotate_types_1630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 901, 33), self_1629, 'annotate_types')
        # Applying the binary operator 'and' (line 901)
        result_and_keyword_1631 = python_operator(stypy.reporting.localization.Localization(__file__, 901, 11), 'and', record_annotation_1628, annotate_types_1630)
        
        # Testing if the type of an if condition is none (line 901)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 901, 8), result_and_keyword_1631):
            pass
        else:
            
            # Testing the type of an if condition (line 901)
            if_condition_1632 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 901, 8), result_and_keyword_1631)
            # Assigning a type to the variable 'if_condition_1632' (line 901)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 901, 8), 'if_condition_1632', if_condition_1632)
            # SSA begins for if statement (line 901)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to __annotate_type(...): (line 902)
            # Processing the call arguments (line 902)
            # Getting the type of 'localization' (line 902)
            localization_1635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 33), 'localization', False)
            # Obtaining the member 'line' of a type (line 902)
            line_1636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 902, 33), localization_1635, 'line')
            # Getting the type of 'localization' (line 902)
            localization_1637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 52), 'localization', False)
            # Obtaining the member 'column' of a type (line 902)
            column_1638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 902, 52), localization_1637, 'column')
            str_1639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 902, 73), 'str', '<dictionary elements type>')
            
            # Call to getattr(...): (line 903)
            # Processing the call arguments (line 903)
            # Getting the type of 'self' (line 903)
            self_1641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 41), 'self', False)
            # Getting the type of 'self' (line 903)
            self_1642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 47), 'self', False)
            # Obtaining the member 'contained_elements_property_name' of a type (line 903)
            contained_elements_property_name_1643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 903, 47), self_1642, 'contained_elements_property_name')
            # Processing the call keyword arguments (line 903)
            kwargs_1644 = {}
            # Getting the type of 'getattr' (line 903)
            getattr_1640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 33), 'getattr', False)
            # Calling getattr(args, kwargs) (line 903)
            getattr_call_result_1645 = invoke(stypy.reporting.localization.Localization(__file__, 903, 33), getattr_1640, *[self_1641, contained_elements_property_name_1643], **kwargs_1644)
            
            # Processing the call keyword arguments (line 902)
            kwargs_1646 = {}
            # Getting the type of 'self' (line 902)
            self_1633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 12), 'self', False)
            # Obtaining the member '__annotate_type' of a type (line 902)
            annotate_type_1634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 902, 12), self_1633, '__annotate_type')
            # Calling __annotate_type(args, kwargs) (line 902)
            annotate_type_call_result_1647 = invoke(stypy.reporting.localization.Localization(__file__, 902, 12), annotate_type_1634, *[line_1636, column_1638, str_1639, getattr_call_result_1645], **kwargs_1646)
            
            # SSA join for if statement (line 901)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'add_key_and_value_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_key_and_value_type' in the type store
        # Getting the type of 'stypy_return_type' (line 859)
        stypy_return_type_1648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1648)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_key_and_value_type'
        return stypy_return_type_1648


    @norecursion
    def get_values_from_key(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_values_from_key'
        module_type_store = module_type_store.open_function_context('get_values_from_key', 905, 4, False)
        # Assigning a type to the variable 'self' (line 906)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 906, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.get_values_from_key.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.get_values_from_key.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.get_values_from_key.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.get_values_from_key.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.get_values_from_key')
        TypeInferenceProxy.get_values_from_key.__dict__.__setitem__('stypy_param_names_list', ['localization', 'key'])
        TypeInferenceProxy.get_values_from_key.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.get_values_from_key.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.get_values_from_key.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.get_values_from_key.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.get_values_from_key.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.get_values_from_key.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.get_values_from_key', ['localization', 'key'], None, None, defaults, varargs, kwargs)

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

        str_1649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 912, (-1)), 'str', '\n        Get the poosible values associated to a key type on a keypair storing proxy\n\n        :param localization: Caller information\n        :param key: Key type\n        :return: Value type list\n        ')
        
        # Assigning a Call to a Name (line 913):
        
        # Assigning a Call to a Name (line 913):
        
        # Call to getattr(...): (line 913)
        # Processing the call arguments (line 913)
        # Getting the type of 'self' (line 913)
        self_1651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 36), 'self', False)
        # Getting the type of 'self' (line 913)
        self_1652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 42), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 913)
        contained_elements_property_name_1653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 913, 42), self_1652, 'contained_elements_property_name')
        # Processing the call keyword arguments (line 913)
        kwargs_1654 = {}
        # Getting the type of 'getattr' (line 913)
        getattr_1650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 28), 'getattr', False)
        # Calling getattr(args, kwargs) (line 913)
        getattr_call_result_1655 = invoke(stypy.reporting.localization.Localization(__file__, 913, 28), getattr_1650, *[self_1651, contained_elements_property_name_1653], **kwargs_1654)
        
        # Assigning a type to the variable 'existing_type_map' (line 913)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 913, 8), 'existing_type_map', getattr_call_result_1655)
        
        
        # SSA begins for try-except statement (line 915)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 917):
        
        # Assigning a Call to a Name (line 917):
        
        # Call to index(...): (line 917)
        # Processing the call arguments (line 917)
        # Getting the type of 'key' (line 917)
        key_1661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 62), 'key', False)
        # Processing the call keyword arguments (line 917)
        kwargs_1662 = {}
        
        # Call to keys(...): (line 917)
        # Processing the call keyword arguments (line 917)
        kwargs_1658 = {}
        # Getting the type of 'existing_type_map' (line 917)
        existing_type_map_1656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 31), 'existing_type_map', False)
        # Obtaining the member 'keys' of a type (line 917)
        keys_1657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 917, 31), existing_type_map_1656, 'keys')
        # Calling keys(args, kwargs) (line 917)
        keys_call_result_1659 = invoke(stypy.reporting.localization.Localization(__file__, 917, 31), keys_1657, *[], **kwargs_1658)
        
        # Obtaining the member 'index' of a type (line 917)
        index_1660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 917, 31), keys_call_result_1659, 'index')
        # Calling index(args, kwargs) (line 917)
        index_call_result_1663 = invoke(stypy.reporting.localization.Localization(__file__, 917, 31), index_1660, *[key_1661], **kwargs_1662)
        
        # Assigning a type to the variable 'stored_key_index' (line 917)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 917, 12), 'stored_key_index', index_call_result_1663)
        
        # Assigning a Subscript to a Name (line 918):
        
        # Assigning a Subscript to a Name (line 918):
        
        # Obtaining the type of the subscript
        # Getting the type of 'stored_key_index' (line 918)
        stored_key_index_1664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 50), 'stored_key_index')
        
        # Call to keys(...): (line 918)
        # Processing the call keyword arguments (line 918)
        kwargs_1667 = {}
        # Getting the type of 'existing_type_map' (line 918)
        existing_type_map_1665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 25), 'existing_type_map', False)
        # Obtaining the member 'keys' of a type (line 918)
        keys_1666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 918, 25), existing_type_map_1665, 'keys')
        # Calling keys(args, kwargs) (line 918)
        keys_call_result_1668 = invoke(stypy.reporting.localization.Localization(__file__, 918, 25), keys_1666, *[], **kwargs_1667)
        
        # Obtaining the member '__getitem__' of a type (line 918)
        getitem___1669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 918, 25), keys_call_result_1668, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 918)
        subscript_call_result_1670 = invoke(stypy.reporting.localization.Localization(__file__, 918, 25), getitem___1669, stored_key_index_1664)
        
        # Assigning a type to the variable 'stored_key' (line 918)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 918, 12), 'stored_key', subscript_call_result_1670)
        
        # Assigning a Subscript to a Name (line 919):
        
        # Assigning a Subscript to a Name (line 919):
        
        # Obtaining the type of the subscript
        # Getting the type of 'stored_key' (line 919)
        stored_key_1671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 38), 'stored_key')
        # Getting the type of 'existing_type_map' (line 919)
        existing_type_map_1672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 20), 'existing_type_map')
        # Obtaining the member '__getitem__' of a type (line 919)
        getitem___1673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 919, 20), existing_type_map_1672, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 919)
        subscript_call_result_1674 = invoke(stypy.reporting.localization.Localization(__file__, 919, 20), getitem___1673, stored_key_1671)
        
        # Assigning a type to the variable 'value' (line 919)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 919, 12), 'value', subscript_call_result_1674)
        # Getting the type of 'value' (line 920)
        value_1675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 19), 'value')
        # Assigning a type to the variable 'stypy_return_type' (line 920)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 920, 12), 'stypy_return_type', value_1675)
        # SSA branch for the except part of a try statement (line 915)
        # SSA branch for the except '<any exception>' branch of a try statement (line 915)
        module_type_store.open_ssa_branch('except')
        
        # Call to TypeError(...): (line 922)
        # Processing the call arguments (line 922)
        # Getting the type of 'localization' (line 922)
        localization_1677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 29), 'localization', False)
        
        # Call to format(...): (line 922)
        # Processing the call arguments (line 922)
        # Getting the type of 'key' (line 922)
        key_1680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 93), 'key', False)
        # Processing the call keyword arguments (line 922)
        kwargs_1681 = {}
        str_1678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 43), 'str', "No value is associated to key type '{0}'")
        # Obtaining the member 'format' of a type (line 922)
        format_1679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 43), str_1678, 'format')
        # Calling format(args, kwargs) (line 922)
        format_call_result_1682 = invoke(stypy.reporting.localization.Localization(__file__, 922, 43), format_1679, *[key_1680], **kwargs_1681)
        
        # Processing the call keyword arguments (line 922)
        kwargs_1683 = {}
        # Getting the type of 'TypeError' (line 922)
        TypeError_1676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 19), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 922)
        TypeError_call_result_1684 = invoke(stypy.reporting.localization.Localization(__file__, 922, 19), TypeError_1676, *[localization_1677, format_call_result_1682], **kwargs_1683)
        
        # Assigning a type to the variable 'stypy_return_type' (line 922)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 12), 'stypy_return_type', TypeError_call_result_1684)
        # SSA join for try-except statement (line 915)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_values_from_key(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_values_from_key' in the type store
        # Getting the type of 'stypy_return_type' (line 905)
        stypy_return_type_1685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1685)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_values_from_key'
        return stypy_return_type_1685


    @norecursion
    def supports_structural_reflection(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'supports_structural_reflection'
        module_type_store = module_type_store.open_function_context('supports_structural_reflection', 926, 4, False)
        # Assigning a type to the variable 'self' (line 927)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 927, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.supports_structural_reflection.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.supports_structural_reflection.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.supports_structural_reflection.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.supports_structural_reflection.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.supports_structural_reflection')
        TypeInferenceProxy.supports_structural_reflection.__dict__.__setitem__('stypy_param_names_list', [])
        TypeInferenceProxy.supports_structural_reflection.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.supports_structural_reflection.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.supports_structural_reflection.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.supports_structural_reflection.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.supports_structural_reflection.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.supports_structural_reflection.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.supports_structural_reflection', [], None, None, defaults, varargs, kwargs)

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

        str_1686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 933, (-1)), 'str', '\n        Determines whether the stored python entity supports intercession. This means that this proxy stores an\n        instance (which are created precisely for this purpose) or the stored entity has a dict as the type of\n        its __dict__ property (and not a dictproxy instance, that is read-only).\n\n        :return: bool\n        ')
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 934)
        self_1687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 15), 'self')
        # Obtaining the member 'instance' of a type (line 934)
        instance_1688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 934, 15), self_1687, 'instance')
        # Getting the type of 'None' (line 934)
        None_1689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 36), 'None')
        # Applying the binary operator 'isnot' (line 934)
        result_is_not_1690 = python_operator(stypy.reporting.localization.Localization(__file__, 934, 15), 'isnot', instance_1688, None_1689)
        
        
        # Call to supports_structural_reflection(...): (line 934)
        # Processing the call arguments (line 934)
        # Getting the type of 'self' (line 935)
        self_1693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 12), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 935)
        python_entity_1694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 935, 12), self_1693, 'python_entity')
        # Processing the call keyword arguments (line 934)
        kwargs_1695 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 934)
        type_inference_proxy_management_copy_1691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 44), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'supports_structural_reflection' of a type (line 934)
        supports_structural_reflection_1692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 934, 44), type_inference_proxy_management_copy_1691, 'supports_structural_reflection')
        # Calling supports_structural_reflection(args, kwargs) (line 934)
        supports_structural_reflection_call_result_1696 = invoke(stypy.reporting.localization.Localization(__file__, 934, 44), supports_structural_reflection_1692, *[python_entity_1694], **kwargs_1695)
        
        # Applying the binary operator 'or' (line 934)
        result_or_keyword_1697 = python_operator(stypy.reporting.localization.Localization(__file__, 934, 15), 'or', result_is_not_1690, supports_structural_reflection_call_result_1696)
        
        # Assigning a type to the variable 'stypy_return_type' (line 934)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 934, 8), 'stypy_return_type', result_or_keyword_1697)
        
        # ################# End of 'supports_structural_reflection(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'supports_structural_reflection' in the type store
        # Getting the type of 'stypy_return_type' (line 926)
        stypy_return_type_1698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1698)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'supports_structural_reflection'
        return stypy_return_type_1698


    @norecursion
    def delete_member(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'delete_member'
        module_type_store = module_type_store.open_function_context('delete_member', 937, 4, False)
        # Assigning a type to the variable 'self' (line 938)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 938, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.delete_member.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.delete_member.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.delete_member.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.delete_member.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.delete_member')
        TypeInferenceProxy.delete_member.__dict__.__setitem__('stypy_param_names_list', ['localization', 'member_name'])
        TypeInferenceProxy.delete_member.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.delete_member.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.delete_member.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.delete_member.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.delete_member.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.delete_member.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.delete_member', ['localization', 'member_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'delete_member', localization, ['localization', 'member_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'delete_member(...)' code ##################

        str_1699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 948, (-1)), 'str', '\n        Set the type of the member whose name is passed to the specified value. There are cases in which deepcopies of\n        the stored python entities are not supported when cloning the type proxy (cloning is needed for SSA), but\n        structural reflection is supported. Therefore, the additional_members attribute have to be created to still\n        support structural reflection while maintaining the ability to create fully independent clones of the stored\n        python entity.\n\n        :param localization: Call localization data\n        :param member_name: Member name\n        :return:\n        ')
        
        
        # SSA begins for try-except statement (line 949)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Getting the type of 'self' (line 950)
        self_1700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 15), 'self')
        # Obtaining the member 'instance' of a type (line 950)
        instance_1701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 950, 15), self_1700, 'instance')
        # Getting the type of 'None' (line 950)
        None_1702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 36), 'None')
        # Applying the binary operator 'isnot' (line 950)
        result_is_not_1703 = python_operator(stypy.reporting.localization.Localization(__file__, 950, 15), 'isnot', instance_1701, None_1702)
        
        # Testing if the type of an if condition is none (line 950)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 950, 12), result_is_not_1703):
            pass
        else:
            
            # Testing the type of an if condition (line 950)
            if_condition_1704 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 950, 12), result_is_not_1703)
            # Assigning a type to the variable 'if_condition_1704' (line 950)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 950, 12), 'if_condition_1704', if_condition_1704)
            # SSA begins for if statement (line 950)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to delattr(...): (line 952)
            # Processing the call arguments (line 952)
            # Getting the type of 'self' (line 952)
            self_1706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 24), 'self', False)
            # Obtaining the member 'instance' of a type (line 952)
            instance_1707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 952, 24), self_1706, 'instance')
            # Getting the type of 'member_name' (line 952)
            member_name_1708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 39), 'member_name', False)
            # Processing the call keyword arguments (line 952)
            kwargs_1709 = {}
            # Getting the type of 'delattr' (line 952)
            delattr_1705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 16), 'delattr', False)
            # Calling delattr(args, kwargs) (line 952)
            delattr_call_result_1710 = invoke(stypy.reporting.localization.Localization(__file__, 952, 16), delattr_1705, *[instance_1707, member_name_1708], **kwargs_1709)
            
            # Getting the type of 'None' (line 953)
            None_1711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 23), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 953)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 953, 16), 'stypy_return_type', None_1711)
            # SSA join for if statement (line 950)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to supports_structural_reflection(...): (line 955)
        # Processing the call arguments (line 955)
        # Getting the type of 'self' (line 955)
        self_1714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 83), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 955)
        python_entity_1715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 83), self_1714, 'python_entity')
        # Processing the call keyword arguments (line 955)
        kwargs_1716 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 955)
        type_inference_proxy_management_copy_1712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 15), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'supports_structural_reflection' of a type (line 955)
        supports_structural_reflection_1713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 15), type_inference_proxy_management_copy_1712, 'supports_structural_reflection')
        # Calling supports_structural_reflection(args, kwargs) (line 955)
        supports_structural_reflection_call_result_1717 = invoke(stypy.reporting.localization.Localization(__file__, 955, 15), supports_structural_reflection_1713, *[python_entity_1715], **kwargs_1716)
        
        # Testing if the type of an if condition is none (line 955)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 955, 12), supports_structural_reflection_call_result_1717):
            pass
        else:
            
            # Testing the type of an if condition (line 955)
            if_condition_1718 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 955, 12), supports_structural_reflection_call_result_1717)
            # Assigning a type to the variable 'if_condition_1718' (line 955)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 955, 12), 'if_condition_1718', if_condition_1718)
            # SSA begins for if statement (line 955)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to delattr(...): (line 957)
            # Processing the call arguments (line 957)
            # Getting the type of 'self' (line 957)
            self_1720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 24), 'self', False)
            # Obtaining the member 'python_entity' of a type (line 957)
            python_entity_1721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 957, 24), self_1720, 'python_entity')
            # Getting the type of 'member_name' (line 957)
            member_name_1722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 44), 'member_name', False)
            # Processing the call keyword arguments (line 957)
            kwargs_1723 = {}
            # Getting the type of 'delattr' (line 957)
            delattr_1719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 16), 'delattr', False)
            # Calling delattr(args, kwargs) (line 957)
            delattr_call_result_1724 = invoke(stypy.reporting.localization.Localization(__file__, 957, 16), delattr_1719, *[python_entity_1721, member_name_1722], **kwargs_1723)
            
            # Getting the type of 'None' (line 958)
            None_1725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 23), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 958)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 958, 16), 'stypy_return_type', None_1725)
            # SSA join for if statement (line 955)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA branch for the except part of a try statement (line 949)
        # SSA branch for the except 'Exception' branch of a try statement (line 949)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'Exception' (line 959)
        Exception_1726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 15), 'Exception')
        # Assigning a type to the variable 'exc' (line 959)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 959, 8), 'exc', Exception_1726)
        
        # Call to TypeError(...): (line 960)
        # Processing the call arguments (line 960)
        # Getting the type of 'localization' (line 960)
        localization_1728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 29), 'localization', False)
        
        # Call to format(...): (line 961)
        # Processing the call arguments (line 961)
        
        # Call to __repr__(...): (line 962)
        # Processing the call keyword arguments (line 962)
        kwargs_1733 = {}
        # Getting the type of 'self' (line 962)
        self_1731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 36), 'self', False)
        # Obtaining the member '__repr__' of a type (line 962)
        repr___1732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 962, 36), self_1731, '__repr__')
        # Calling __repr__(args, kwargs) (line 962)
        repr___call_result_1734 = invoke(stypy.reporting.localization.Localization(__file__, 962, 36), repr___1732, *[], **kwargs_1733)
        
        
        # Call to str(...): (line 962)
        # Processing the call arguments (line 962)
        # Getting the type of 'exc' (line 962)
        exc_1736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 57), 'exc', False)
        # Processing the call keyword arguments (line 962)
        kwargs_1737 = {}
        # Getting the type of 'str' (line 962)
        str_1735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 53), 'str', False)
        # Calling str(args, kwargs) (line 962)
        str_call_result_1738 = invoke(stypy.reporting.localization.Localization(__file__, 962, 53), str_1735, *[exc_1736], **kwargs_1737)
        
        # Getting the type of 'member_name' (line 962)
        member_name_1739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 63), 'member_name', False)
        # Processing the call keyword arguments (line 961)
        kwargs_1740 = {}
        str_1729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 961, 29), 'str', "'{2}' member deletion is impossible: Cannot modify the structure of '{0}': {1}")
        # Obtaining the member 'format' of a type (line 961)
        format_1730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 961, 29), str_1729, 'format')
        # Calling format(args, kwargs) (line 961)
        format_call_result_1741 = invoke(stypy.reporting.localization.Localization(__file__, 961, 29), format_1730, *[repr___call_result_1734, str_call_result_1738, member_name_1739], **kwargs_1740)
        
        # Processing the call keyword arguments (line 960)
        kwargs_1742 = {}
        # Getting the type of 'TypeError' (line 960)
        TypeError_1727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 19), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 960)
        TypeError_call_result_1743 = invoke(stypy.reporting.localization.Localization(__file__, 960, 19), TypeError_1727, *[localization_1728, format_call_result_1741], **kwargs_1742)
        
        # Assigning a type to the variable 'stypy_return_type' (line 960)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 960, 12), 'stypy_return_type', TypeError_call_result_1743)
        # SSA join for try-except statement (line 949)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to TypeError(...): (line 964)
        # Processing the call arguments (line 964)
        # Getting the type of 'localization' (line 964)
        localization_1745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 25), 'localization', False)
        
        # Call to format(...): (line 965)
        # Processing the call arguments (line 965)
        # Getting the type of 'member_name' (line 966)
        member_name_1748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 51), 'member_name', False)
        # Processing the call keyword arguments (line 965)
        kwargs_1749 = {}
        str_1746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 965, 25), 'str', "'{0}' member deletion is impossible: Cannot modify the structure of a python library type or instance")
        # Obtaining the member 'format' of a type (line 965)
        format_1747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 965, 25), str_1746, 'format')
        # Calling format(args, kwargs) (line 965)
        format_call_result_1750 = invoke(stypy.reporting.localization.Localization(__file__, 965, 25), format_1747, *[member_name_1748], **kwargs_1749)
        
        # Processing the call keyword arguments (line 964)
        kwargs_1751 = {}
        # Getting the type of 'TypeError' (line 964)
        TypeError_1744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 15), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 964)
        TypeError_call_result_1752 = invoke(stypy.reporting.localization.Localization(__file__, 964, 15), TypeError_1744, *[localization_1745, format_call_result_1750], **kwargs_1751)
        
        # Assigning a type to the variable 'stypy_return_type' (line 964)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 964, 8), 'stypy_return_type', TypeError_call_result_1752)
        
        # ################# End of 'delete_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'delete_member' in the type store
        # Getting the type of 'stypy_return_type' (line 937)
        stypy_return_type_1753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1753)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'delete_member'
        return stypy_return_type_1753


    @norecursion
    def change_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'change_type'
        module_type_store = module_type_store.open_function_context('change_type', 968, 4, False)
        # Assigning a type to the variable 'self' (line 969)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 969, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.change_type.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.change_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.change_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.change_type.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.change_type')
        TypeInferenceProxy.change_type.__dict__.__setitem__('stypy_param_names_list', ['localization', 'new_type'])
        TypeInferenceProxy.change_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.change_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.change_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.change_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.change_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.change_type.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.change_type', ['localization', 'new_type'], None, None, defaults, varargs, kwargs)

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

        str_1754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 981, (-1)), 'str', "\n        Changes the type of the stored entity, provided it is an instance (so it supports structural reflection).\n        Type change is only available in Python for instances of user-defined classes.\n\n        You can only assign to the __class__ attribute of an instance of a user-defined class\n        (i.e. defined using the class keyword), and the new value must also be a user-defined class.\n        Whether the classes are new-style or old-style does not matter. (You can't mix them, though.\n        You can't turn an old-style class instance into a new-style class instance.)\n\n        :param localization: Call localization data\n        :param new_type: New type of the instance.\n        :return: A TypeError or None\n        ")
        
        # Assigning a Call to a Name (line 982):
        
        # Assigning a Call to a Name (line 982):
        
        # Call to __change_instance_type_checks(...): (line 982)
        # Processing the call arguments (line 982)
        # Getting the type of 'localization' (line 982)
        localization_1757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 52), 'localization', False)
        # Getting the type of 'new_type' (line 982)
        new_type_1758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 66), 'new_type', False)
        # Processing the call keyword arguments (line 982)
        kwargs_1759 = {}
        # Getting the type of 'self' (line 982)
        self_1755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 17), 'self', False)
        # Obtaining the member '__change_instance_type_checks' of a type (line 982)
        change_instance_type_checks_1756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 982, 17), self_1755, '__change_instance_type_checks')
        # Calling __change_instance_type_checks(args, kwargs) (line 982)
        change_instance_type_checks_call_result_1760 = invoke(stypy.reporting.localization.Localization(__file__, 982, 17), change_instance_type_checks_1756, *[localization_1757, new_type_1758], **kwargs_1759)
        
        # Assigning a type to the variable 'result' (line 982)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 982, 8), 'result', change_instance_type_checks_call_result_1760)
        
        # Type idiom detected: calculating its left and rigth part (line 984)
        # Getting the type of 'TypeError' (line 984)
        TypeError_1761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 30), 'TypeError')
        # Getting the type of 'result' (line 984)
        result_1762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 22), 'result')
        
        (may_be_1763, more_types_in_union_1764) = may_be_subtype(TypeError_1761, result_1762)

        if may_be_1763:

            if more_types_in_union_1764:
                # Runtime conditional SSA (line 984)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'result' (line 984)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 984, 8), 'result', remove_not_subtype_from_union(result_1762, TypeError))
            # Getting the type of 'result' (line 985)
            result_1765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 19), 'result')
            # Assigning a type to the variable 'stypy_return_type' (line 985)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 985, 12), 'stypy_return_type', result_1765)

            if more_types_in_union_1764:
                # SSA join for if statement (line 984)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to is_user_defined_class(...): (line 988)
        # Processing the call arguments (line 988)
        # Getting the type of 'new_type' (line 988)
        new_type_1768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 70), 'new_type', False)
        # Obtaining the member 'python_entity' of a type (line 988)
        python_entity_1769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 988, 70), new_type_1768, 'python_entity')
        # Processing the call keyword arguments (line 988)
        kwargs_1770 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 988)
        type_inference_proxy_management_copy_1766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 11), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'is_user_defined_class' of a type (line 988)
        is_user_defined_class_1767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 988, 11), type_inference_proxy_management_copy_1766, 'is_user_defined_class')
        # Calling is_user_defined_class(args, kwargs) (line 988)
        is_user_defined_class_call_result_1771 = invoke(stypy.reporting.localization.Localization(__file__, 988, 11), is_user_defined_class_1767, *[python_entity_1769], **kwargs_1770)
        
        # Testing if the type of an if condition is none (line 988)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 988, 8), is_user_defined_class_call_result_1771):
            
            # Assigning a Attribute to a Attribute (line 991):
            
            # Assigning a Attribute to a Attribute (line 991):
            # Getting the type of 'new_type' (line 991)
            new_type_1776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 33), 'new_type')
            # Obtaining the member 'python_entity' of a type (line 991)
            python_entity_1777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 991, 33), new_type_1776, 'python_entity')
            # Getting the type of 'self' (line 991)
            self_1778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 12), 'self')
            # Setting the type of the member 'python_entity' of a type (line 991)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 991, 12), self_1778, 'python_entity', python_entity_1777)
        else:
            
            # Testing the type of an if condition (line 988)
            if_condition_1772 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 988, 8), is_user_defined_class_call_result_1771)
            # Assigning a type to the variable 'if_condition_1772' (line 988)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 988, 8), 'if_condition_1772', if_condition_1772)
            # SSA begins for if statement (line 988)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Attribute (line 989):
            
            # Assigning a Attribute to a Attribute (line 989):
            # Getting the type of 'types' (line 989)
            types_1773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 33), 'types')
            # Obtaining the member 'InstanceType' of a type (line 989)
            InstanceType_1774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 989, 33), types_1773, 'InstanceType')
            # Getting the type of 'self' (line 989)
            self_1775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 12), 'self')
            # Setting the type of the member 'python_entity' of a type (line 989)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 989, 12), self_1775, 'python_entity', InstanceType_1774)
            # SSA branch for the else part of an if statement (line 988)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Attribute to a Attribute (line 991):
            
            # Assigning a Attribute to a Attribute (line 991):
            # Getting the type of 'new_type' (line 991)
            new_type_1776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 33), 'new_type')
            # Obtaining the member 'python_entity' of a type (line 991)
            python_entity_1777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 991, 33), new_type_1776, 'python_entity')
            # Getting the type of 'self' (line 991)
            self_1778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 12), 'self')
            # Setting the type of the member 'python_entity' of a type (line 991)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 991, 12), self_1778, 'python_entity', python_entity_1777)
            # SSA join for if statement (line 988)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to setattr(...): (line 993)
        # Processing the call arguments (line 993)
        # Getting the type of 'self' (line 993)
        self_1780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 16), 'self', False)
        # Obtaining the member 'instance' of a type (line 993)
        instance_1781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 993, 16), self_1780, 'instance')
        str_1782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 993, 31), 'str', '__class__')
        # Getting the type of 'new_type' (line 993)
        new_type_1783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 44), 'new_type', False)
        # Obtaining the member 'python_entity' of a type (line 993)
        python_entity_1784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 993, 44), new_type_1783, 'python_entity')
        # Processing the call keyword arguments (line 993)
        kwargs_1785 = {}
        # Getting the type of 'setattr' (line 993)
        setattr_1779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 8), 'setattr', False)
        # Calling setattr(args, kwargs) (line 993)
        setattr_call_result_1786 = invoke(stypy.reporting.localization.Localization(__file__, 993, 8), setattr_1779, *[instance_1781, str_1782, python_entity_1784], **kwargs_1785)
        
        # Getting the type of 'None' (line 994)
        None_1787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 994, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 994)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 994, 8), 'stypy_return_type', None_1787)
        
        # ################# End of 'change_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'change_type' in the type store
        # Getting the type of 'stypy_return_type' (line 968)
        stypy_return_type_1788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1788)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'change_type'
        return stypy_return_type_1788


    @norecursion
    def change_base_types(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'change_base_types'
        module_type_store = module_type_store.open_function_context('change_base_types', 996, 4, False)
        # Assigning a type to the variable 'self' (line 997)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 997, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.change_base_types.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.change_base_types.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.change_base_types.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.change_base_types.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.change_base_types')
        TypeInferenceProxy.change_base_types.__dict__.__setitem__('stypy_param_names_list', ['localization', 'new_types'])
        TypeInferenceProxy.change_base_types.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.change_base_types.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.change_base_types.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.change_base_types.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.change_base_types.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.change_base_types.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.change_base_types', ['localization', 'new_types'], None, None, defaults, varargs, kwargs)

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

        str_1789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1008, (-1)), 'str', '\n        Changes, if possible, the base types of the hold Python class. For determining if the change is possible, a\n        series of checks (defined before) are made.\n\n        For new-style classes, changing of the mro is not possible, you need to define a metaclass that does the trick\n\n        Old-style classes admits changing its __bases__ attribute (its a tuple), so we can add or substitute\n\n        :param localization: Call localization data\n        :param new_types: New base types (in the form of a tuple)\n        :return: A TypeError or None\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 1009)
        # Getting the type of 'new_types' (line 1009)
        new_types_1790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 20), 'new_types')
        # Getting the type of 'tuple' (line 1009)
        tuple_1791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 34), 'tuple')
        
        (may_be_1792, more_types_in_union_1793) = may_not_be_type(new_types_1790, tuple_1791)

        if may_be_1792:

            if more_types_in_union_1793:
                # Runtime conditional SSA (line 1009)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'new_types' (line 1009)
            new_types_1794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 8), 'new_types')
            # Assigning a type to the variable 'new_types' (line 1009)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1009, 8), 'new_types', remove_type_from_union(new_types_1794, tuple_1791))
            
            # Call to TypeError(...): (line 1010)
            # Processing the call arguments (line 1010)
            # Getting the type of 'localization' (line 1010)
            localization_1796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 29), 'localization', False)
            str_1797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1010, 43), 'str', 'New subtypes have to be specified using a tuple')
            # Processing the call keyword arguments (line 1010)
            kwargs_1798 = {}
            # Getting the type of 'TypeError' (line 1010)
            TypeError_1795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 1010)
            TypeError_call_result_1799 = invoke(stypy.reporting.localization.Localization(__file__, 1010, 19), TypeError_1795, *[localization_1796, str_1797], **kwargs_1798)
            
            # Assigning a type to the variable 'stypy_return_type' (line 1010)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1010, 12), 'stypy_return_type', TypeError_call_result_1799)

            if more_types_in_union_1793:
                # SSA join for if statement (line 1009)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'new_types' (line 1012)
        new_types_1800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1012, 25), 'new_types')
        # Assigning a type to the variable 'new_types_1800' (line 1012)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1012, 8), 'new_types_1800', new_types_1800)
        # Testing if the for loop is going to be iterated (line 1012)
        # Testing the type of a for loop iterable (line 1012)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1012, 8), new_types_1800)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 1012, 8), new_types_1800):
            # Getting the type of the for loop variable (line 1012)
            for_loop_var_1801 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1012, 8), new_types_1800)
            # Assigning a type to the variable 'base_type' (line 1012)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1012, 8), 'base_type', for_loop_var_1801)
            # SSA begins for a for statement (line 1012)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 1013):
            
            # Assigning a Call to a Name (line 1013):
            
            # Call to __change_class_base_types_checks(...): (line 1013)
            # Processing the call arguments (line 1013)
            # Getting the type of 'localization' (line 1013)
            localization_1804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1013, 58), 'localization', False)
            # Getting the type of 'base_type' (line 1013)
            base_type_1805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1013, 72), 'base_type', False)
            # Processing the call keyword arguments (line 1013)
            kwargs_1806 = {}
            # Getting the type of 'self' (line 1013)
            self_1802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1013, 20), 'self', False)
            # Obtaining the member '__change_class_base_types_checks' of a type (line 1013)
            change_class_base_types_checks_1803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1013, 20), self_1802, '__change_class_base_types_checks')
            # Calling __change_class_base_types_checks(args, kwargs) (line 1013)
            change_class_base_types_checks_call_result_1807 = invoke(stypy.reporting.localization.Localization(__file__, 1013, 20), change_class_base_types_checks_1803, *[localization_1804, base_type_1805], **kwargs_1806)
            
            # Assigning a type to the variable 'check' (line 1013)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1013, 12), 'check', change_class_base_types_checks_call_result_1807)
            
            # Type idiom detected: calculating its left and rigth part (line 1014)
            # Getting the type of 'TypeError' (line 1014)
            TypeError_1808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1014, 33), 'TypeError')
            # Getting the type of 'check' (line 1014)
            check_1809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1014, 26), 'check')
            
            (may_be_1810, more_types_in_union_1811) = may_be_subtype(TypeError_1808, check_1809)

            if may_be_1810:

                if more_types_in_union_1811:
                    # Runtime conditional SSA (line 1014)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'check' (line 1014)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1014, 12), 'check', remove_not_subtype_from_union(check_1809, TypeError))
                # Getting the type of 'check' (line 1015)
                check_1812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1015, 23), 'check')
                # Assigning a type to the variable 'stypy_return_type' (line 1015)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1015, 16), 'stypy_return_type', check_1812)

                if more_types_in_union_1811:
                    # SSA join for if statement (line 1014)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 1017):
        
        # Assigning a Call to a Name (line 1017):
        
        # Call to map(...): (line 1017)
        # Processing the call arguments (line 1017)

        @norecursion
        def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_1'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 1017, 27, True)
            # Passed parameters checking function
            _stypy_temp_lambda_1.stypy_localization = localization
            _stypy_temp_lambda_1.stypy_type_of_self = None
            _stypy_temp_lambda_1.stypy_type_store = module_type_store
            _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
            _stypy_temp_lambda_1.stypy_param_names_list = ['tproxy']
            _stypy_temp_lambda_1.stypy_varargs_param_name = None
            _stypy_temp_lambda_1.stypy_kwargs_param_name = None
            _stypy_temp_lambda_1.stypy_call_defaults = defaults
            _stypy_temp_lambda_1.stypy_call_varargs = varargs
            _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['tproxy'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_1', ['tproxy'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'tproxy' (line 1017)
            tproxy_1814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 42), 'tproxy', False)
            # Obtaining the member 'python_entity' of a type (line 1017)
            python_entity_1815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1017, 42), tproxy_1814, 'python_entity')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 1017)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1017, 27), 'stypy_return_type', python_entity_1815)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_1' in the type store
            # Getting the type of 'stypy_return_type' (line 1017)
            stypy_return_type_1816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 27), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_1816)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_1'
            return stypy_return_type_1816

        # Assigning a type to the variable '_stypy_temp_lambda_1' (line 1017)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1017, 27), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
        # Getting the type of '_stypy_temp_lambda_1' (line 1017)
        _stypy_temp_lambda_1_1817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 27), '_stypy_temp_lambda_1')
        # Getting the type of 'new_types' (line 1017)
        new_types_1818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 64), 'new_types', False)
        # Processing the call keyword arguments (line 1017)
        kwargs_1819 = {}
        # Getting the type of 'map' (line 1017)
        map_1813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 23), 'map', False)
        # Calling map(args, kwargs) (line 1017)
        map_call_result_1820 = invoke(stypy.reporting.localization.Localization(__file__, 1017, 23), map_1813, *[_stypy_temp_lambda_1_1817, new_types_1818], **kwargs_1819)
        
        # Assigning a type to the variable 'base_classes' (line 1017)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1017, 8), 'base_classes', map_call_result_1820)
        
        # Assigning a Call to a Attribute (line 1019):
        
        # Assigning a Call to a Attribute (line 1019):
        
        # Call to tuple(...): (line 1019)
        # Processing the call arguments (line 1019)
        # Getting the type of 'base_classes' (line 1019)
        base_classes_1822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1019, 45), 'base_classes', False)
        # Processing the call keyword arguments (line 1019)
        kwargs_1823 = {}
        # Getting the type of 'tuple' (line 1019)
        tuple_1821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1019, 39), 'tuple', False)
        # Calling tuple(args, kwargs) (line 1019)
        tuple_call_result_1824 = invoke(stypy.reporting.localization.Localization(__file__, 1019, 39), tuple_1821, *[base_classes_1822], **kwargs_1823)
        
        # Getting the type of 'self' (line 1019)
        self_1825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1019, 8), 'self')
        # Obtaining the member 'python_entity' of a type (line 1019)
        python_entity_1826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1019, 8), self_1825, 'python_entity')
        # Setting the type of the member '__bases__' of a type (line 1019)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1019, 8), python_entity_1826, '__bases__', tuple_call_result_1824)
        # Getting the type of 'None' (line 1020)
        None_1827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1020, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 1020)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1020, 8), 'stypy_return_type', None_1827)
        
        # ################# End of 'change_base_types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'change_base_types' in the type store
        # Getting the type of 'stypy_return_type' (line 996)
        stypy_return_type_1828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 996, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1828)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'change_base_types'
        return stypy_return_type_1828


    @norecursion
    def add_base_types(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_base_types'
        module_type_store = module_type_store.open_function_context('add_base_types', 1022, 4, False)
        # Assigning a type to the variable 'self' (line 1023)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1023, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.add_base_types.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.add_base_types.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.add_base_types.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.add_base_types.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.add_base_types')
        TypeInferenceProxy.add_base_types.__dict__.__setitem__('stypy_param_names_list', ['localization', 'new_types'])
        TypeInferenceProxy.add_base_types.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.add_base_types.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.add_base_types.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.add_base_types.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.add_base_types.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.add_base_types.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.add_base_types', ['localization', 'new_types'], None, None, defaults, varargs, kwargs)

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

        str_1829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, (-1)), 'str', '\n        Adds, if possible, the base types of the hold Python class existing base types.\n        For determining if the change is possible, a series of checks (defined before) are made.\n\n        :param localization: Call localization data\n        :param new_types: New base types (in the form of a tuple)\n        :return: A TypeError or None\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 1031)
        # Getting the type of 'new_types' (line 1031)
        new_types_1830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 20), 'new_types')
        # Getting the type of 'tuple' (line 1031)
        tuple_1831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 34), 'tuple')
        
        (may_be_1832, more_types_in_union_1833) = may_not_be_type(new_types_1830, tuple_1831)

        if may_be_1832:

            if more_types_in_union_1833:
                # Runtime conditional SSA (line 1031)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'new_types' (line 1031)
            new_types_1834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 8), 'new_types')
            # Assigning a type to the variable 'new_types' (line 1031)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1031, 8), 'new_types', remove_type_from_union(new_types_1834, tuple_1831))
            
            # Call to TypeError(...): (line 1032)
            # Processing the call arguments (line 1032)
            # Getting the type of 'localization' (line 1032)
            localization_1836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 29), 'localization', False)
            str_1837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1032, 43), 'str', 'New subtypes have to be specified using a tuple')
            # Processing the call keyword arguments (line 1032)
            kwargs_1838 = {}
            # Getting the type of 'TypeError' (line 1032)
            TypeError_1835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 1032)
            TypeError_call_result_1839 = invoke(stypy.reporting.localization.Localization(__file__, 1032, 19), TypeError_1835, *[localization_1836, str_1837], **kwargs_1838)
            
            # Assigning a type to the variable 'stypy_return_type' (line 1032)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1032, 12), 'stypy_return_type', TypeError_call_result_1839)

            if more_types_in_union_1833:
                # SSA join for if statement (line 1031)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'new_types' (line 1034)
        new_types_1840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 25), 'new_types')
        # Assigning a type to the variable 'new_types_1840' (line 1034)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1034, 8), 'new_types_1840', new_types_1840)
        # Testing if the for loop is going to be iterated (line 1034)
        # Testing the type of a for loop iterable (line 1034)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1034, 8), new_types_1840)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 1034, 8), new_types_1840):
            # Getting the type of the for loop variable (line 1034)
            for_loop_var_1841 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1034, 8), new_types_1840)
            # Assigning a type to the variable 'base_type' (line 1034)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1034, 8), 'base_type', for_loop_var_1841)
            # SSA begins for a for statement (line 1034)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 1035):
            
            # Assigning a Call to a Name (line 1035):
            
            # Call to __change_class_base_types_checks(...): (line 1035)
            # Processing the call arguments (line 1035)
            # Getting the type of 'localization' (line 1035)
            localization_1844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 58), 'localization', False)
            # Getting the type of 'base_type' (line 1035)
            base_type_1845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 72), 'base_type', False)
            # Processing the call keyword arguments (line 1035)
            kwargs_1846 = {}
            # Getting the type of 'self' (line 1035)
            self_1842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 20), 'self', False)
            # Obtaining the member '__change_class_base_types_checks' of a type (line 1035)
            change_class_base_types_checks_1843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1035, 20), self_1842, '__change_class_base_types_checks')
            # Calling __change_class_base_types_checks(args, kwargs) (line 1035)
            change_class_base_types_checks_call_result_1847 = invoke(stypy.reporting.localization.Localization(__file__, 1035, 20), change_class_base_types_checks_1843, *[localization_1844, base_type_1845], **kwargs_1846)
            
            # Assigning a type to the variable 'check' (line 1035)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1035, 12), 'check', change_class_base_types_checks_call_result_1847)
            
            # Type idiom detected: calculating its left and rigth part (line 1036)
            # Getting the type of 'TypeError' (line 1036)
            TypeError_1848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 33), 'TypeError')
            # Getting the type of 'check' (line 1036)
            check_1849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 26), 'check')
            
            (may_be_1850, more_types_in_union_1851) = may_be_subtype(TypeError_1848, check_1849)

            if may_be_1850:

                if more_types_in_union_1851:
                    # Runtime conditional SSA (line 1036)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'check' (line 1036)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1036, 12), 'check', remove_not_subtype_from_union(check_1849, TypeError))
                # Getting the type of 'check' (line 1037)
                check_1852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 23), 'check')
                # Assigning a type to the variable 'stypy_return_type' (line 1037)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 16), 'stypy_return_type', check_1852)

                if more_types_in_union_1851:
                    # SSA join for if statement (line 1036)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 1039):
        
        # Assigning a Call to a Name (line 1039):
        
        # Call to map(...): (line 1039)
        # Processing the call arguments (line 1039)

        @norecursion
        def _stypy_temp_lambda_2(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_2'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_2', 1039, 27, True)
            # Passed parameters checking function
            _stypy_temp_lambda_2.stypy_localization = localization
            _stypy_temp_lambda_2.stypy_type_of_self = None
            _stypy_temp_lambda_2.stypy_type_store = module_type_store
            _stypy_temp_lambda_2.stypy_function_name = '_stypy_temp_lambda_2'
            _stypy_temp_lambda_2.stypy_param_names_list = ['tproxy']
            _stypy_temp_lambda_2.stypy_varargs_param_name = None
            _stypy_temp_lambda_2.stypy_kwargs_param_name = None
            _stypy_temp_lambda_2.stypy_call_defaults = defaults
            _stypy_temp_lambda_2.stypy_call_varargs = varargs
            _stypy_temp_lambda_2.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_2', ['tproxy'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_2', ['tproxy'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'tproxy' (line 1039)
            tproxy_1854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 42), 'tproxy', False)
            # Obtaining the member 'python_entity' of a type (line 1039)
            python_entity_1855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1039, 42), tproxy_1854, 'python_entity')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 1039)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 27), 'stypy_return_type', python_entity_1855)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_2' in the type store
            # Getting the type of 'stypy_return_type' (line 1039)
            stypy_return_type_1856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 27), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_1856)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_2'
            return stypy_return_type_1856

        # Assigning a type to the variable '_stypy_temp_lambda_2' (line 1039)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 27), '_stypy_temp_lambda_2', _stypy_temp_lambda_2)
        # Getting the type of '_stypy_temp_lambda_2' (line 1039)
        _stypy_temp_lambda_2_1857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 27), '_stypy_temp_lambda_2')
        # Getting the type of 'new_types' (line 1039)
        new_types_1858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 64), 'new_types', False)
        # Processing the call keyword arguments (line 1039)
        kwargs_1859 = {}
        # Getting the type of 'map' (line 1039)
        map_1853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 23), 'map', False)
        # Calling map(args, kwargs) (line 1039)
        map_call_result_1860 = invoke(stypy.reporting.localization.Localization(__file__, 1039, 23), map_1853, *[_stypy_temp_lambda_2_1857, new_types_1858], **kwargs_1859)
        
        # Assigning a type to the variable 'base_classes' (line 1039)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 8), 'base_classes', map_call_result_1860)
        
        # Getting the type of 'self' (line 1040)
        self_1861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 8), 'self')
        # Obtaining the member 'python_entity' of a type (line 1040)
        python_entity_1862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1040, 8), self_1861, 'python_entity')
        # Obtaining the member '__bases__' of a type (line 1040)
        bases___1863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1040, 8), python_entity_1862, '__bases__')
        
        # Call to tuple(...): (line 1040)
        # Processing the call arguments (line 1040)
        # Getting the type of 'base_classes' (line 1040)
        base_classes_1865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 46), 'base_classes', False)
        # Processing the call keyword arguments (line 1040)
        kwargs_1866 = {}
        # Getting the type of 'tuple' (line 1040)
        tuple_1864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 40), 'tuple', False)
        # Calling tuple(args, kwargs) (line 1040)
        tuple_call_result_1867 = invoke(stypy.reporting.localization.Localization(__file__, 1040, 40), tuple_1864, *[base_classes_1865], **kwargs_1866)
        
        # Applying the binary operator '+=' (line 1040)
        result_iadd_1868 = python_operator(stypy.reporting.localization.Localization(__file__, 1040, 8), '+=', bases___1863, tuple_call_result_1867)
        # Getting the type of 'self' (line 1040)
        self_1869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 8), 'self')
        # Obtaining the member 'python_entity' of a type (line 1040)
        python_entity_1870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1040, 8), self_1869, 'python_entity')
        # Setting the type of the member '__bases__' of a type (line 1040)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1040, 8), python_entity_1870, '__bases__', result_iadd_1868)
        
        # Getting the type of 'None' (line 1041)
        None_1871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1041, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 1041)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1041, 8), 'stypy_return_type', None_1871)
        
        # ################# End of 'add_base_types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_base_types' in the type store
        # Getting the type of 'stypy_return_type' (line 1022)
        stypy_return_type_1872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1022, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1872)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_base_types'
        return stypy_return_type_1872


    @norecursion
    def clone(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'clone'
        module_type_store = module_type_store.open_function_context('clone', 1045, 4, False)
        # Assigning a type to the variable 'self' (line 1046)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1046, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.clone.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.clone.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.clone.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.clone.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.clone')
        TypeInferenceProxy.clone.__dict__.__setitem__('stypy_param_names_list', [])
        TypeInferenceProxy.clone.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.clone.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.clone.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.clone.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.clone.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.clone.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.clone', [], None, None, defaults, varargs, kwargs)

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

        str_1873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1051, (-1)), 'str', '\n        Clones the type proxy, making an independent copy of the stored python entity. Physical cloning is not\n        performed if the hold python entity do not support intercession, as its structure is immutable.\n\n        :return: A clone of this proxy\n        ')
        
        # Evaluating a boolean operation
        
        
        # Call to supports_structural_reflection(...): (line 1052)
        # Processing the call keyword arguments (line 1052)
        kwargs_1876 = {}
        # Getting the type of 'self' (line 1052)
        self_1874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 15), 'self', False)
        # Obtaining the member 'supports_structural_reflection' of a type (line 1052)
        supports_structural_reflection_1875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1052, 15), self_1874, 'supports_structural_reflection')
        # Calling supports_structural_reflection(args, kwargs) (line 1052)
        supports_structural_reflection_call_result_1877 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 15), supports_structural_reflection_1875, *[], **kwargs_1876)
        
        # Applying the 'not' unary operator (line 1052)
        result_not__1878 = python_operator(stypy.reporting.localization.Localization(__file__, 1052, 11), 'not', supports_structural_reflection_call_result_1877)
        
        
        
        # Call to can_store_elements(...): (line 1052)
        # Processing the call keyword arguments (line 1052)
        kwargs_1881 = {}
        # Getting the type of 'self' (line 1052)
        self_1879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 61), 'self', False)
        # Obtaining the member 'can_store_elements' of a type (line 1052)
        can_store_elements_1880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1052, 61), self_1879, 'can_store_elements')
        # Calling can_store_elements(args, kwargs) (line 1052)
        can_store_elements_call_result_1882 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 61), can_store_elements_1880, *[], **kwargs_1881)
        
        # Applying the 'not' unary operator (line 1052)
        result_not__1883 = python_operator(stypy.reporting.localization.Localization(__file__, 1052, 57), 'not', can_store_elements_call_result_1882)
        
        # Applying the binary operator 'and' (line 1052)
        result_and_keyword_1884 = python_operator(stypy.reporting.localization.Localization(__file__, 1052, 11), 'and', result_not__1878, result_not__1883)
        
        
        # Call to can_store_keypairs(...): (line 1053)
        # Processing the call keyword arguments (line 1053)
        kwargs_1887 = {}
        # Getting the type of 'self' (line 1053)
        self_1885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 20), 'self', False)
        # Obtaining the member 'can_store_keypairs' of a type (line 1053)
        can_store_keypairs_1886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1053, 20), self_1885, 'can_store_keypairs')
        # Calling can_store_keypairs(args, kwargs) (line 1053)
        can_store_keypairs_call_result_1888 = invoke(stypy.reporting.localization.Localization(__file__, 1053, 20), can_store_keypairs_1886, *[], **kwargs_1887)
        
        # Applying the 'not' unary operator (line 1053)
        result_not__1889 = python_operator(stypy.reporting.localization.Localization(__file__, 1053, 16), 'not', can_store_keypairs_call_result_1888)
        
        # Applying the binary operator 'and' (line 1052)
        result_and_keyword_1890 = python_operator(stypy.reporting.localization.Localization(__file__, 1052, 11), 'and', result_and_keyword_1884, result_not__1889)
        
        # Testing if the type of an if condition is none (line 1052)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 1052, 8), result_and_keyword_1890):
            
            # Call to create_duplicate(...): (line 1056)
            # Processing the call arguments (line 1056)
            # Getting the type of 'self' (line 1056)
            self_1895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 73), 'self', False)
            # Processing the call keyword arguments (line 1056)
            kwargs_1896 = {}
            # Getting the type of 'type_inference_proxy_management_copy' (line 1056)
            type_inference_proxy_management_copy_1893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 19), 'type_inference_proxy_management_copy', False)
            # Obtaining the member 'create_duplicate' of a type (line 1056)
            create_duplicate_1894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1056, 19), type_inference_proxy_management_copy_1893, 'create_duplicate')
            # Calling create_duplicate(args, kwargs) (line 1056)
            create_duplicate_call_result_1897 = invoke(stypy.reporting.localization.Localization(__file__, 1056, 19), create_duplicate_1894, *[self_1895], **kwargs_1896)
            
            # Assigning a type to the variable 'stypy_return_type' (line 1056)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1056, 12), 'stypy_return_type', create_duplicate_call_result_1897)
        else:
            
            # Testing the type of an if condition (line 1052)
            if_condition_1891 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1052, 8), result_and_keyword_1890)
            # Assigning a type to the variable 'if_condition_1891' (line 1052)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1052, 8), 'if_condition_1891', if_condition_1891)
            # SSA begins for if statement (line 1052)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'self' (line 1054)
            self_1892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 19), 'self')
            # Assigning a type to the variable 'stypy_return_type' (line 1054)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1054, 12), 'stypy_return_type', self_1892)
            # SSA branch for the else part of an if statement (line 1052)
            module_type_store.open_ssa_branch('else')
            
            # Call to create_duplicate(...): (line 1056)
            # Processing the call arguments (line 1056)
            # Getting the type of 'self' (line 1056)
            self_1895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 73), 'self', False)
            # Processing the call keyword arguments (line 1056)
            kwargs_1896 = {}
            # Getting the type of 'type_inference_proxy_management_copy' (line 1056)
            type_inference_proxy_management_copy_1893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 19), 'type_inference_proxy_management_copy', False)
            # Obtaining the member 'create_duplicate' of a type (line 1056)
            create_duplicate_1894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1056, 19), type_inference_proxy_management_copy_1893, 'create_duplicate')
            # Calling create_duplicate(args, kwargs) (line 1056)
            create_duplicate_call_result_1897 = invoke(stypy.reporting.localization.Localization(__file__, 1056, 19), create_duplicate_1894, *[self_1895], **kwargs_1896)
            
            # Assigning a type to the variable 'stypy_return_type' (line 1056)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1056, 12), 'stypy_return_type', create_duplicate_call_result_1897)
            # SSA join for if statement (line 1052)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'clone(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clone' in the type store
        # Getting the type of 'stypy_return_type' (line 1045)
        stypy_return_type_1898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1898)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clone'
        return stypy_return_type_1898


    @norecursion
    def dir(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dir'
        module_type_store = module_type_store.open_function_context('dir', 1060, 4, False)
        # Assigning a type to the variable 'self' (line 1061)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1061, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.dir.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.dir.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.dir.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.dir.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.dir')
        TypeInferenceProxy.dir.__dict__.__setitem__('stypy_param_names_list', [])
        TypeInferenceProxy.dir.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.dir.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.dir.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.dir.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.dir.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.dir.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.dir', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dir', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dir(...)' code ##################

        str_1899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1064, (-1)), 'str', '\n        Calls the dir Python builtin over the stored Python object and returns the result\n        :return: list of strings\n        ')
        
        # Call to dir(...): (line 1065)
        # Processing the call arguments (line 1065)
        # Getting the type of 'self' (line 1065)
        self_1901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 19), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 1065)
        python_entity_1902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1065, 19), self_1901, 'python_entity')
        # Processing the call keyword arguments (line 1065)
        kwargs_1903 = {}
        # Getting the type of 'dir' (line 1065)
        dir_1900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 15), 'dir', False)
        # Calling dir(args, kwargs) (line 1065)
        dir_call_result_1904 = invoke(stypy.reporting.localization.Localization(__file__, 1065, 15), dir_1900, *[python_entity_1902], **kwargs_1903)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1065)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1065, 8), 'stypy_return_type', dir_call_result_1904)
        
        # ################# End of 'dir(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dir' in the type store
        # Getting the type of 'stypy_return_type' (line 1060)
        stypy_return_type_1905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1905)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dir'
        return stypy_return_type_1905


    @norecursion
    def is_user_defined_class(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_user_defined_class'
        module_type_store = module_type_store.open_function_context('is_user_defined_class', 1084, 4, False)
        # Assigning a type to the variable 'self' (line 1085)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1085, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.is_user_defined_class.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.is_user_defined_class.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.is_user_defined_class.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.is_user_defined_class.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.is_user_defined_class')
        TypeInferenceProxy.is_user_defined_class.__dict__.__setitem__('stypy_param_names_list', [])
        TypeInferenceProxy.is_user_defined_class.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.is_user_defined_class.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.is_user_defined_class.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.is_user_defined_class.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.is_user_defined_class.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.is_user_defined_class.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.is_user_defined_class', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_user_defined_class', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_user_defined_class(...)' code ##################

        str_1906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1088, (-1)), 'str', '\n        Determines whether this proxy holds an user-defined class or not\n        :return:\n        ')
        
        # Call to is_user_defined_class(...): (line 1089)
        # Processing the call arguments (line 1089)
        # Getting the type of 'self' (line 1089)
        self_1909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 74), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 1089)
        python_entity_1910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1089, 74), self_1909, 'python_entity')
        # Processing the call keyword arguments (line 1089)
        kwargs_1911 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 1089)
        type_inference_proxy_management_copy_1907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 15), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'is_user_defined_class' of a type (line 1089)
        is_user_defined_class_1908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1089, 15), type_inference_proxy_management_copy_1907, 'is_user_defined_class')
        # Calling is_user_defined_class(args, kwargs) (line 1089)
        is_user_defined_class_call_result_1912 = invoke(stypy.reporting.localization.Localization(__file__, 1089, 15), is_user_defined_class_1908, *[python_entity_1910], **kwargs_1911)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1089)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1089, 8), 'stypy_return_type', is_user_defined_class_call_result_1912)
        
        # ################# End of 'is_user_defined_class(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_user_defined_class' in the type store
        # Getting the type of 'stypy_return_type' (line 1084)
        stypy_return_type_1913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1084, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1913)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_user_defined_class'
        return stypy_return_type_1913


    @norecursion
    def __annotate_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__annotate_type'
        module_type_store = module_type_store.open_function_context('__annotate_type', 1093, 4, False)
        # Assigning a type to the variable 'self' (line 1094)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1094, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceProxy.__annotate_type.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.__annotate_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceProxy.__annotate_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.__annotate_type.__dict__.__setitem__('stypy_function_name', 'TypeInferenceProxy.__annotate_type')
        TypeInferenceProxy.__annotate_type.__dict__.__setitem__('stypy_param_names_list', ['line', 'column', 'name', 'type_'])
        TypeInferenceProxy.__annotate_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.__annotate_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.__annotate_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.__annotate_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.__annotate_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.__annotate_type.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceProxy.__annotate_type', ['line', 'column', 'name', 'type_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__annotate_type', localization, ['line', 'column', 'name', 'type_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__annotate_type(...)' code ##################

        str_1914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1101, (-1)), 'str', '\n        Annotate a type into the proxy type annotation record\n        :param line: Source code line when the type change is performed\n        :param column: Source code column when the type change is performed\n        :param name: Name of the variable whose type is changed\n        :param type_: New type\n        :return: None\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 1102)
        str_1915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1102, 25), 'str', 'annotation_record')
        # Getting the type of 'self' (line 1102)
        self_1916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 19), 'self')
        
        (may_be_1917, more_types_in_union_1918) = may_provide_member(str_1915, self_1916)

        if may_be_1917:

            if more_types_in_union_1918:
                # Runtime conditional SSA (line 1102)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'self' (line 1102)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1102, 8), 'self', remove_not_member_provider_from_union(self_1916, 'annotation_record'))
            
            # Call to annotate_type(...): (line 1103)
            # Processing the call arguments (line 1103)
            # Getting the type of 'line' (line 1103)
            line_1922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 49), 'line', False)
            # Getting the type of 'column' (line 1103)
            column_1923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 55), 'column', False)
            # Getting the type of 'name' (line 1103)
            name_1924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 63), 'name', False)
            # Getting the type of 'type_' (line 1103)
            type__1925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 69), 'type_', False)
            # Processing the call keyword arguments (line 1103)
            kwargs_1926 = {}
            # Getting the type of 'self' (line 1103)
            self_1919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 12), 'self', False)
            # Obtaining the member 'annotation_record' of a type (line 1103)
            annotation_record_1920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1103, 12), self_1919, 'annotation_record')
            # Obtaining the member 'annotate_type' of a type (line 1103)
            annotate_type_1921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1103, 12), annotation_record_1920, 'annotate_type')
            # Calling annotate_type(args, kwargs) (line 1103)
            annotate_type_call_result_1927 = invoke(stypy.reporting.localization.Localization(__file__, 1103, 12), annotate_type_1921, *[line_1922, column_1923, name_1924, type__1925], **kwargs_1926)
            

            if more_types_in_union_1918:
                # SSA join for if statement (line 1102)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__annotate_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__annotate_type' in the type store
        # Getting the type of 'stypy_return_type' (line 1093)
        stypy_return_type_1928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1928)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__annotate_type'
        return stypy_return_type_1928


# Assigning a type to the variable 'TypeInferenceProxy' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'TypeInferenceProxy', TypeInferenceProxy)

# Assigning a Call to a Name (line 40):

# Call to dict(...): (line 40)
# Processing the call keyword arguments (line 40)
kwargs_1930 = {}
# Getting the type of 'dict' (line 40)
dict_1929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 23), 'dict', False)
# Calling dict(args, kwargs) (line 40)
dict_call_result_1931 = invoke(stypy.reporting.localization.Localization(__file__, 40, 23), dict_1929, *[], **kwargs_1930)

# Getting the type of 'TypeInferenceProxy'
TypeInferenceProxy_1932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeInferenceProxy')
# Setting the type of the member 'type_proxy_cache' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeInferenceProxy_1932, 'type_proxy_cache', dict_call_result_1931)

# Assigning a Name to a Name (line 48):
# Getting the type of 'True' (line 48)
True_1933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'True')
# Getting the type of 'TypeInferenceProxy'
TypeInferenceProxy_1934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeInferenceProxy')
# Setting the type of the member 'annotate_types' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeInferenceProxy_1934, 'annotate_types', True_1933)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
