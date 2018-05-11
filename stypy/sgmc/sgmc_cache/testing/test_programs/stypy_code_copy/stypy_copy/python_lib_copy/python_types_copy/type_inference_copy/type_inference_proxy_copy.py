
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
1067:     def dict(self, localization):
1068:         '''
1069:         Equivalent to call __dict__ over the stored Python instance
1070:         :param localization:
1071:         :return:
1072:         '''
1073:         members = self.dir()
1074:         ret_dict = TypeInferenceProxy.instance(dict)
1075:         ret_dict.set_type_instance(True)
1076:         for member in members:
1077:             str_instance = TypeInferenceProxy.instance(str, value=member)
1078: 
1079:             value = self.get_type_of_member(localization, member)
1080:             ret_dict.add_key_and_value_type(localization, (str_instance, value), False)
1081: 
1082:         return ret_dict
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
import_10140 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'type_inference_proxy_management_copy')

if (type(import_10140) is not StypyTypeError):

    if (import_10140 != 'pyd_module'):
        __import__(import_10140)
        sys_modules_10141 = sys.modules[import_10140]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'type_inference_proxy_management_copy', sys_modules_10141.module_type_store, module_type_store)
    else:
        import type_inference_proxy_management_copy

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'type_inference_proxy_management_copy', type_inference_proxy_management_copy, module_type_store)

else:
    # Assigning a type to the variable 'type_inference_proxy_management_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'type_inference_proxy_management_copy', import_10140)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_10142 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy')

if (type(import_10142) is not StypyTypeError):

    if (import_10142 != 'pyd_module'):
        __import__(import_10142)
        sys_modules_10143 = sys.modules[import_10142]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', sys_modules_10143.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_10143, sys_modules_10143.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', import_10142)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy import TypeWarning' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_10144 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy')

if (type(import_10144) is not StypyTypeError):

    if (import_10144 != 'pyd_module'):
        __import__(import_10144)
        sys_modules_10145 = sys.modules[import_10144]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy', sys_modules_10145.module_type_store, module_type_store, ['TypeWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_10145, sys_modules_10145.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy import TypeWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy', None, module_type_store, ['TypeWarning'], [TypeWarning])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy', import_10144)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy import call_handlers_copy' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_10146 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy')

if (type(import_10146) is not StypyTypeError):

    if (import_10146 != 'pyd_module'):
        __import__(import_10146)
        sys_modules_10147 = sys.modules[import_10146]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy', sys_modules_10147.module_type_store, module_type_store, ['call_handlers_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_10147, sys_modules_10147.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy import call_handlers_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy', None, module_type_store, ['call_handlers_copy'], [call_handlers_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy', import_10146)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy import Type' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_10148 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy')

if (type(import_10148) is not StypyTypeError):

    if (import_10148 != 'pyd_module'):
        __import__(import_10148)
        sys_modules_10149 = sys.modules[import_10148]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy', sys_modules_10149.module_type_store, module_type_store, ['Type'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_10149, sys_modules_10149.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy import Type

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy', None, module_type_store, ['Type'], [Type])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy', import_10148)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy import type_equivalence_copy' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_10150 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy')

if (type(import_10150) is not StypyTypeError):

    if (import_10150 != 'pyd_module'):
        __import__(import_10150)
        sys_modules_10151 = sys.modules[import_10150]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy', sys_modules_10151.module_type_store, module_type_store, ['type_equivalence_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_10151, sys_modules_10151.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy import type_equivalence_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy', None, module_type_store, ['type_equivalence_copy'], [type_equivalence_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy', import_10150)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy' statement (line 10)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_10152 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_10152) is not StypyTypeError):

    if (import_10152 != 'pyd_module'):
        __import__(import_10152)
        sys_modules_10153 = sys.modules[import_10152]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', sys_modules_10153.module_type_store, module_type_store, ['union_type_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_10153, sys_modules_10153.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['union_type_copy'], [union_type_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', import_10152)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import undefined_type_copy' statement (line 11)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_10154 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_10154) is not StypyTypeError):

    if (import_10154 != 'pyd_module'):
        __import__(import_10154)
        sys_modules_10155 = sys.modules[import_10154]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', sys_modules_10155.module_type_store, module_type_store, ['undefined_type_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_10155, sys_modules_10155.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import undefined_type_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['undefined_type_copy'], [undefined_type_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', import_10154)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import simple_python_types' statement (line 12)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_10156 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy')

if (type(import_10156) is not StypyTypeError):

    if (import_10156 != 'pyd_module'):
        __import__(import_10156)
        sys_modules_10157 = sys.modules[import_10156]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', sys_modules_10157.module_type_store, module_type_store, ['simple_python_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_10157, sys_modules_10157.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import simple_python_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', None, module_type_store, ['simple_python_types'], [simple_python_types])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', import_10156)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy.type_annotation_record_copy import TypeAnnotationRecord' statement (line 13)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_10158 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy.type_annotation_record_copy')

if (type(import_10158) is not StypyTypeError):

    if (import_10158 != 'pyd_module'):
        __import__(import_10158)
        sys_modules_10159 = sys.modules[import_10158]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy.type_annotation_record_copy', sys_modules_10159.module_type_store, module_type_store, ['TypeAnnotationRecord'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_10159, sys_modules_10159.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy.type_annotation_record_copy import TypeAnnotationRecord

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy.type_annotation_record_copy', None, module_type_store, ['TypeAnnotationRecord'], [TypeAnnotationRecord])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy.type_annotation_record_copy' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy.type_annotation_record_copy', import_10158)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy import type_store_copy' statement (line 14)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_10160 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'testing.test_programs.stypy_code_copy.stypy_copy')

if (type(import_10160) is not StypyTypeError):

    if (import_10160 != 'pyd_module'):
        __import__(import_10160)
        sys_modules_10161 = sys.modules[import_10160]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', sys_modules_10161.module_type_store, module_type_store, ['type_store_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_10161, sys_modules_10161.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy import type_store_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', None, module_type_store, ['type_store_copy'], [type_store_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', import_10160)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy import stypy_parameters_copy' statement (line 15)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_10162 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'testing.test_programs.stypy_code_copy.stypy_copy')

if (type(import_10162) is not StypyTypeError):

    if (import_10162 != 'pyd_module'):
        __import__(import_10162)
        sys_modules_10163 = sys.modules[import_10162]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', sys_modules_10163.module_type_store, module_type_store, ['stypy_parameters_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_10163, sys_modules_10163.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy import stypy_parameters_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', None, module_type_store, ['stypy_parameters_copy'], [stypy_parameters_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', import_10162)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

# Declaration of the 'TypeInferenceProxy' class
# Getting the type of 'Type' (line 18)
Type_10164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 25), 'Type')

class TypeInferenceProxy(Type_10164, ):
    str_10165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, (-1)), 'str', '\n    The type inference proxy is the main class of stypy. Its main purpose is to represent any kind of Python type,\n     holding a reference to it. It is also responsible of a lot of possible operations that can be done with the\n     contained type, including:\n\n     - Returning/setting the type of any member of the Python entity it holds.\n     - Invoke any of its associated callable members, returning the invokation result\n     - Support structural reflection operations, it the enclosed object is able to support them\n     - Obtain relationships with other entities (modules that contain a represented function, class that contains\n     a represented method,...)\n     - Manipulate stored types (if the represented entity is able to store other types)\n     - Clone itself to support the SSA algorithm\n     - Respond to builtin operations such as dir and __dict__ calls\n     - Hold values for the represented object type\n\n     All Python entities (functions, variables, methods, classes, modules,...) might be enclosed in a type inference\n     proxy. For those method that are not applicable to the enclosed Python entity, the class will return a TypeError.\n    ')
    
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

        str_10166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, (-1)), 'str', '\n        Gets the python entity that can be considered the "parent" of the passed entity\n        :param parent: Any Python entity\n        :return: The parent of this entity, if any\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 59)
        str_10167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 27), 'str', '__module__')
        # Getting the type of 'parent' (line 59)
        parent_10168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 19), 'parent')
        
        (may_be_10169, more_types_in_union_10170) = may_provide_member(str_10167, parent_10168)

        if may_be_10169:

            if more_types_in_union_10170:
                # Runtime conditional SSA (line 59)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'parent' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'parent', remove_not_member_provider_from_union(parent_10168, '__module__'))
            
            # Call to instance(...): (line 60)
            # Processing the call arguments (line 60)
            
            # Call to getmodule(...): (line 60)
            # Processing the call arguments (line 60)
            # Getting the type of 'parent' (line 60)
            parent_10175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 65), 'parent', False)
            # Processing the call keyword arguments (line 60)
            kwargs_10176 = {}
            # Getting the type of 'inspect' (line 60)
            inspect_10173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 47), 'inspect', False)
            # Obtaining the member 'getmodule' of a type (line 60)
            getmodule_10174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 47), inspect_10173, 'getmodule')
            # Calling getmodule(args, kwargs) (line 60)
            getmodule_call_result_10177 = invoke(stypy.reporting.localization.Localization(__file__, 60, 47), getmodule_10174, *[parent_10175], **kwargs_10176)
            
            # Processing the call keyword arguments (line 60)
            kwargs_10178 = {}
            # Getting the type of 'TypeInferenceProxy' (line 60)
            TypeInferenceProxy_10171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 19), 'TypeInferenceProxy', False)
            # Obtaining the member 'instance' of a type (line 60)
            instance_10172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 19), TypeInferenceProxy_10171, 'instance')
            # Calling instance(args, kwargs) (line 60)
            instance_call_result_10179 = invoke(stypy.reporting.localization.Localization(__file__, 60, 19), instance_10172, *[getmodule_call_result_10177], **kwargs_10178)
            
            # Assigning a type to the variable 'stypy_return_type' (line 60)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'stypy_return_type', instance_call_result_10179)

            if more_types_in_union_10170:
                # SSA join for if statement (line 59)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 61)
        str_10180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 27), 'str', '__class__')
        # Getting the type of 'parent' (line 61)
        parent_10181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 19), 'parent')
        
        (may_be_10182, more_types_in_union_10183) = may_provide_member(str_10180, parent_10181)

        if may_be_10182:

            if more_types_in_union_10183:
                # Runtime conditional SSA (line 61)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'parent' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'parent', remove_not_member_provider_from_union(parent_10181, '__class__'))
            
            # Call to instance(...): (line 62)
            # Processing the call arguments (line 62)
            # Getting the type of 'parent' (line 62)
            parent_10186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 47), 'parent', False)
            # Obtaining the member '__class__' of a type (line 62)
            class___10187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 47), parent_10186, '__class__')
            # Processing the call keyword arguments (line 62)
            kwargs_10188 = {}
            # Getting the type of 'TypeInferenceProxy' (line 62)
            TypeInferenceProxy_10184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 19), 'TypeInferenceProxy', False)
            # Obtaining the member 'instance' of a type (line 62)
            instance_10185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 19), TypeInferenceProxy_10184, 'instance')
            # Calling instance(args, kwargs) (line 62)
            instance_call_result_10189 = invoke(stypy.reporting.localization.Localization(__file__, 62, 19), instance_10185, *[class___10187], **kwargs_10188)
            
            # Assigning a type to the variable 'stypy_return_type' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'stypy_return_type', instance_call_result_10189)

            if more_types_in_union_10183:
                # SSA join for if statement (line 61)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'None' (line 64)
        None_10190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'stypy_return_type', None_10190)
        
        # ################# End of '__get_parent_proxy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__get_parent_proxy' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_10191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10191)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__get_parent_proxy'
        return stypy_return_type_10191


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

        str_10192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, (-1)), 'str', '\n        Changes the parent object of the represented object to the one specified. This is used to trace the nesting\n        of proxies that hold types that are placed inside other proxies represented entities. This property is NOT\n        related with dynamic inheritance.\n        :param parent: The new parent object or None. If the passed parent is None, the class tries to autocalculate it.\n        If there is no possible parent, it is assigned to None\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 74)
        # Getting the type of 'parent' (line 74)
        parent_10193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'parent')
        # Getting the type of 'None' (line 74)
        None_10194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), 'None')
        
        (may_be_10195, more_types_in_union_10196) = may_not_be_none(parent_10193, None_10194)

        if may_be_10195:

            if more_types_in_union_10196:
                # Runtime conditional SSA (line 74)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 75):
            
            # Assigning a Name to a Attribute (line 75):
            # Getting the type of 'parent' (line 75)
            parent_10197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 32), 'parent')
            # Getting the type of 'self' (line 75)
            self_10198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'self')
            # Setting the type of the member 'parent_proxy' of a type (line 75)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), self_10198, 'parent_proxy', parent_10197)

            if more_types_in_union_10196:
                # Runtime conditional SSA for else branch (line 74)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_10195) or more_types_in_union_10196):
            
            
            # Call to ismodule(...): (line 77)
            # Processing the call arguments (line 77)
            # Getting the type of 'self' (line 77)
            self_10201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 36), 'self', False)
            # Obtaining the member 'python_entity' of a type (line 77)
            python_entity_10202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 36), self_10201, 'python_entity')
            # Processing the call keyword arguments (line 77)
            kwargs_10203 = {}
            # Getting the type of 'inspect' (line 77)
            inspect_10199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'inspect', False)
            # Obtaining the member 'ismodule' of a type (line 77)
            ismodule_10200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 19), inspect_10199, 'ismodule')
            # Calling ismodule(args, kwargs) (line 77)
            ismodule_call_result_10204 = invoke(stypy.reporting.localization.Localization(__file__, 77, 19), ismodule_10200, *[python_entity_10202], **kwargs_10203)
            
            # Applying the 'not' unary operator (line 77)
            result_not__10205 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 15), 'not', ismodule_call_result_10204)
            
            # Testing if the type of an if condition is none (line 77)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 77, 12), result_not__10205):
                
                # Assigning a Name to a Attribute (line 80):
                
                # Assigning a Name to a Attribute (line 80):
                # Getting the type of 'None' (line 80)
                None_10214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 36), 'None')
                # Getting the type of 'self' (line 80)
                self_10215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'self')
                # Setting the type of the member 'parent_proxy' of a type (line 80)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 16), self_10215, 'parent_proxy', None_10214)
            else:
                
                # Testing the type of an if condition (line 77)
                if_condition_10206 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 12), result_not__10205)
                # Assigning a type to the variable 'if_condition_10206' (line 77)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'if_condition_10206', if_condition_10206)
                # SSA begins for if statement (line 77)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Attribute (line 78):
                
                # Assigning a Call to a Attribute (line 78):
                
                # Call to __get_parent_proxy(...): (line 78)
                # Processing the call arguments (line 78)
                # Getting the type of 'self' (line 78)
                self_10209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 74), 'self', False)
                # Obtaining the member 'python_entity' of a type (line 78)
                python_entity_10210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 74), self_10209, 'python_entity')
                # Processing the call keyword arguments (line 78)
                kwargs_10211 = {}
                # Getting the type of 'TypeInferenceProxy' (line 78)
                TypeInferenceProxy_10207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 36), 'TypeInferenceProxy', False)
                # Obtaining the member '__get_parent_proxy' of a type (line 78)
                get_parent_proxy_10208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 36), TypeInferenceProxy_10207, '__get_parent_proxy')
                # Calling __get_parent_proxy(args, kwargs) (line 78)
                get_parent_proxy_call_result_10212 = invoke(stypy.reporting.localization.Localization(__file__, 78, 36), get_parent_proxy_10208, *[python_entity_10210], **kwargs_10211)
                
                # Getting the type of 'self' (line 78)
                self_10213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'self')
                # Setting the type of the member 'parent_proxy' of a type (line 78)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 16), self_10213, 'parent_proxy', get_parent_proxy_call_result_10212)
                # SSA branch for the else part of an if statement (line 77)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Attribute (line 80):
                
                # Assigning a Name to a Attribute (line 80):
                # Getting the type of 'None' (line 80)
                None_10214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 36), 'None')
                # Getting the type of 'self' (line 80)
                self_10215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'self')
                # Setting the type of the member 'parent_proxy' of a type (line 80)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 16), self_10215, 'parent_proxy', None_10214)
                # SSA join for if statement (line 77)
                module_type_store = module_type_store.join_ssa_context()
                


            if (may_be_10195 and more_types_in_union_10196):
                # SSA join for if statement (line 74)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__assign_parent_proxy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__assign_parent_proxy' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_10216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10216)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__assign_parent_proxy'
        return stypy_return_type_10216


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

        str_10217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, (-1)), 'str', '\n        Performs all the possible checks to see if a base type change is possible for the currently hold python entity.\n        This includes:\n        - Making sure that the currently hold object represent a class. No base type change is possible if the hold\n        entity is not a class. For checking the availability of an instance type change, see the\n        "__change_instance_type_checks" private method.\n        - Making sure that the hold class is not a new-style class. New style Python classes cannot change its base\n        type directly, as its __mro__ (Method Resolution Order) property is readonly. For this purpose a metaclass\n        has to be created, like in this example:\n\n        class change_mro_meta(type):\n            def __new__(cls, cls_name, cls_bases, cls_dict):\n                    out_cls = super(change_mro_meta, cls).__new__(cls, cls_name, cls_bases, cls_dict)\n                    out_cls.change_mro = False\n                    out_cls.hack_mro   = classmethod(cls.hack_mro)\n                    out_cls.fix_mro    = classmethod(cls.fix_mro)\n                    out_cls.recalc_mro = classmethod(cls.recalc_mro)\n                    return out_cls\n\n            @staticmethod\n            def hack_mro(cls):\n                cls.change_mro = True\n                cls.recalc_mro()\n\n            @staticmethod\n            def fix_mro(cls):\n                cls.change_mro = False\n                cls.recalc_mro()\n\n            @staticmethod\n            def recalc_mro(cls):\n                # Changing a class\' base causes __mro__ recalculation\n                cls.__bases__  = cls.__bases__ + tuple()\n\n            def mro(cls):\n                default_mro = super(change_mro_meta, cls).mro()\n                if hasattr(cls, "change_mro") and cls.change_mro:\n                    return default_mro[1:2] + default_mro\n                else:\n                    return default_mro\n\n        - Making sure that new base class do not belong to a different class style as the current one: base type of\n        old-style classes can only be changed to another old-style class.\n\n        :param localization: Call localization data\n        :param new_type: New base type to change to\n        :return: A Type error specifying the problem encountered with the base type change or None if no error is found\n        ')
        
        
        # Call to is_class(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'self' (line 131)
        self_10220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 61), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 131)
        python_entity_10221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 61), self_10220, 'python_entity')
        # Processing the call keyword arguments (line 131)
        kwargs_10222 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 131)
        type_inference_proxy_management_copy_10218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'is_class' of a type (line 131)
        is_class_10219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 15), type_inference_proxy_management_copy_10218, 'is_class')
        # Calling is_class(args, kwargs) (line 131)
        is_class_call_result_10223 = invoke(stypy.reporting.localization.Localization(__file__, 131, 15), is_class_10219, *[python_entity_10221], **kwargs_10222)
        
        # Applying the 'not' unary operator (line 131)
        result_not__10224 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 11), 'not', is_class_call_result_10223)
        
        # Testing if the type of an if condition is none (line 131)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 131, 8), result_not__10224):
            pass
        else:
            
            # Testing the type of an if condition (line 131)
            if_condition_10225 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 131, 8), result_not__10224)
            # Assigning a type to the variable 'if_condition_10225' (line 131)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'if_condition_10225', if_condition_10225)
            # SSA begins for if statement (line 131)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 132)
            # Processing the call arguments (line 132)
            # Getting the type of 'localization' (line 132)
            localization_10227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 29), 'localization', False)
            str_10228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 43), 'str', 'Cannot change the base type of a non-class Python entity')
            # Processing the call keyword arguments (line 132)
            kwargs_10229 = {}
            # Getting the type of 'TypeError' (line 132)
            TypeError_10226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 132)
            TypeError_call_result_10230 = invoke(stypy.reporting.localization.Localization(__file__, 132, 19), TypeError_10226, *[localization_10227, str_10228], **kwargs_10229)
            
            # Assigning a type to the variable 'stypy_return_type' (line 132)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'stypy_return_type', TypeError_call_result_10230)
            # SSA join for if statement (line 131)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to is_new_style_class(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'self' (line 134)
        self_10233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 67), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 134)
        python_entity_10234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 67), self_10233, 'python_entity')
        # Processing the call keyword arguments (line 134)
        kwargs_10235 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 134)
        type_inference_proxy_management_copy_10231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 11), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'is_new_style_class' of a type (line 134)
        is_new_style_class_10232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 11), type_inference_proxy_management_copy_10231, 'is_new_style_class')
        # Calling is_new_style_class(args, kwargs) (line 134)
        is_new_style_class_call_result_10236 = invoke(stypy.reporting.localization.Localization(__file__, 134, 11), is_new_style_class_10232, *[python_entity_10234], **kwargs_10235)
        
        # Testing if the type of an if condition is none (line 134)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 134, 8), is_new_style_class_call_result_10236):
            pass
        else:
            
            # Testing the type of an if condition (line 134)
            if_condition_10237 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 8), is_new_style_class_call_result_10236)
            # Assigning a type to the variable 'if_condition_10237' (line 134)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'if_condition_10237', if_condition_10237)
            # SSA begins for if statement (line 134)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 135)
            # Processing the call arguments (line 135)
            # Getting the type of 'localization' (line 135)
            localization_10239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 29), 'localization', False)
            str_10240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 29), 'str', 'Cannot change the class hierarchy of a new-style class: The __mro__ (Method Resolution Order) property is readonly')
            # Processing the call keyword arguments (line 135)
            kwargs_10241 = {}
            # Getting the type of 'TypeError' (line 135)
            TypeError_10238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 135)
            TypeError_call_result_10242 = invoke(stypy.reporting.localization.Localization(__file__, 135, 19), TypeError_10238, *[localization_10239, str_10240], **kwargs_10241)
            
            # Assigning a type to the variable 'stypy_return_type' (line 135)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'stypy_return_type', TypeError_call_result_10242)
            # SSA join for if statement (line 134)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 139)
        self_10243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 11), 'self')
        # Obtaining the member 'instance' of a type (line 139)
        instance_10244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 11), self_10243, 'instance')
        # Getting the type of 'None' (line 139)
        None_10245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 32), 'None')
        # Applying the binary operator 'isnot' (line 139)
        result_is_not_10246 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 11), 'isnot', instance_10244, None_10245)
        
        # Testing if the type of an if condition is none (line 139)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 139, 8), result_is_not_10246):
            pass
        else:
            
            # Testing the type of an if condition (line 139)
            if_condition_10247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 8), result_is_not_10246)
            # Assigning a type to the variable 'if_condition_10247' (line 139)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'if_condition_10247', if_condition_10247)
            # SSA begins for if statement (line 139)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 140)
            # Processing the call arguments (line 140)
            # Getting the type of 'localization' (line 140)
            localization_10249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 29), 'localization', False)
            str_10250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 43), 'str', 'Cannot change the class hierarchy of a class using an instance')
            # Processing the call keyword arguments (line 140)
            kwargs_10251 = {}
            # Getting the type of 'TypeError' (line 140)
            TypeError_10248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 140)
            TypeError_call_result_10252 = invoke(stypy.reporting.localization.Localization(__file__, 140, 19), TypeError_10248, *[localization_10249, str_10250], **kwargs_10251)
            
            # Assigning a type to the variable 'stypy_return_type' (line 140)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'stypy_return_type', TypeError_call_result_10252)
            # SSA join for if statement (line 139)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 142):
        
        # Assigning a Call to a Name (line 142):
        
        # Call to is_old_style_class(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'self' (line 142)
        self_10255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 85), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 142)
        python_entity_10256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 85), self_10255, 'python_entity')
        # Processing the call keyword arguments (line 142)
        kwargs_10257 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 142)
        type_inference_proxy_management_copy_10253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 29), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'is_old_style_class' of a type (line 142)
        is_old_style_class_10254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 29), type_inference_proxy_management_copy_10253, 'is_old_style_class')
        # Calling is_old_style_class(args, kwargs) (line 142)
        is_old_style_class_call_result_10258 = invoke(stypy.reporting.localization.Localization(__file__, 142, 29), is_old_style_class_10254, *[python_entity_10256], **kwargs_10257)
        
        # Assigning a type to the variable 'old_style_existing' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'old_style_existing', is_old_style_class_call_result_10258)
        
        # Type idiom detected: calculating its left and rigth part (line 143)
        # Getting the type of 'TypeError' (line 143)
        TypeError_10259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 36), 'TypeError')
        # Getting the type of 'new_type' (line 143)
        new_type_10260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 26), 'new_type')
        
        (may_be_10261, more_types_in_union_10262) = may_not_be_subtype(TypeError_10259, new_type_10260)

        if may_be_10261:

            if more_types_in_union_10262:
                # Runtime conditional SSA (line 143)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'new_type' (line 143)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'new_type', remove_subtype_from_union(new_type_10260, TypeError))
            
            # Assigning a Call to a Name (line 144):
            
            # Assigning a Call to a Name (line 144):
            
            # Call to is_old_style_class(...): (line 144)
            # Processing the call arguments (line 144)
            # Getting the type of 'new_type' (line 144)
            new_type_10265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 84), 'new_type', False)
            # Obtaining the member 'python_entity' of a type (line 144)
            python_entity_10266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 84), new_type_10265, 'python_entity')
            # Processing the call keyword arguments (line 144)
            kwargs_10267 = {}
            # Getting the type of 'type_inference_proxy_management_copy' (line 144)
            type_inference_proxy_management_copy_10263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 28), 'type_inference_proxy_management_copy', False)
            # Obtaining the member 'is_old_style_class' of a type (line 144)
            is_old_style_class_10264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 28), type_inference_proxy_management_copy_10263, 'is_old_style_class')
            # Calling is_old_style_class(args, kwargs) (line 144)
            is_old_style_class_call_result_10268 = invoke(stypy.reporting.localization.Localization(__file__, 144, 28), is_old_style_class_10264, *[python_entity_10266], **kwargs_10267)
            
            # Assigning a type to the variable 'old_style_new' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'old_style_new', is_old_style_class_call_result_10268)

            if more_types_in_union_10262:
                # Runtime conditional SSA for else branch (line 143)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_10261) or more_types_in_union_10262):
            # Assigning a type to the variable 'new_type' (line 143)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'new_type', remove_not_subtype_from_union(new_type_10260, TypeError))
            
            # Call to TypeError(...): (line 146)
            # Processing the call arguments (line 146)
            # Getting the type of 'localization' (line 146)
            localization_10270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 29), 'localization', False)
            str_10271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 43), 'str', 'Cannot change the class hierarchy to a type error')
            # Processing the call keyword arguments (line 146)
            kwargs_10272 = {}
            # Getting the type of 'TypeError' (line 146)
            TypeError_10269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 146)
            TypeError_call_result_10273 = invoke(stypy.reporting.localization.Localization(__file__, 146, 19), TypeError_10269, *[localization_10270, str_10271], **kwargs_10272)
            
            # Assigning a type to the variable 'stypy_return_type' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'stypy_return_type', TypeError_call_result_10273)

            if (may_be_10261 and more_types_in_union_10262):
                # SSA join for if statement (line 143)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'old_style_existing' (line 149)
        old_style_existing_10274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'old_style_existing')
        # Getting the type of 'old_style_new' (line 149)
        old_style_new_10275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 37), 'old_style_new')
        # Applying the binary operator '==' (line 149)
        result_eq_10276 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 15), '==', old_style_existing_10274, old_style_new_10275)
        
        # Applying the 'not' unary operator (line 149)
        result_not__10277 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 11), 'not', result_eq_10276)
        
        # Testing if the type of an if condition is none (line 149)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 149, 8), result_not__10277):
            pass
        else:
            
            # Testing the type of an if condition (line 149)
            if_condition_10278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 8), result_not__10277)
            # Assigning a type to the variable 'if_condition_10278' (line 149)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'if_condition_10278', if_condition_10278)
            # SSA begins for if statement (line 149)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 150)
            # Processing the call arguments (line 150)
            # Getting the type of 'localization' (line 150)
            localization_10280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 29), 'localization', False)
            str_10281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 43), 'str', 'Cannot change the class hierarchy from an old-style Python parent class to a new-style Python parent class')
            # Processing the call keyword arguments (line 150)
            kwargs_10282 = {}
            # Getting the type of 'TypeError' (line 150)
            TypeError_10279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 150)
            TypeError_call_result_10283 = invoke(stypy.reporting.localization.Localization(__file__, 150, 19), TypeError_10279, *[localization_10280, str_10281], **kwargs_10282)
            
            # Assigning a type to the variable 'stypy_return_type' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'stypy_return_type', TypeError_call_result_10283)
            # SSA join for if statement (line 149)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'None' (line 153)
        None_10284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'stypy_return_type', None_10284)
        
        # ################# End of '__change_class_base_types_checks(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__change_class_base_types_checks' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_10285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10285)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__change_class_base_types_checks'
        return stypy_return_type_10285


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

        str_10286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, (-1)), 'str', '\n        Performs all the checks that ensure that changing the type of an instance is possible. This includes:\n        - Making sure that we are changing the type of an user-defined class instance. Type change for Python\n        library classes instances is not possible.\n        - Making sure that the old instance type and the new instance type are of the same class style, as mixing\n        old-style and new-style types is not possible in Python.\n\n        :param localization: Call localization data\n        :param new_type: New instance type.\n        :return:\n        ')
        
        
        # Call to is_class(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'self' (line 168)
        self_10289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 61), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 168)
        python_entity_10290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 61), self_10289, 'python_entity')
        # Processing the call keyword arguments (line 168)
        kwargs_10291 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 168)
        type_inference_proxy_management_copy_10287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 15), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'is_class' of a type (line 168)
        is_class_10288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 15), type_inference_proxy_management_copy_10287, 'is_class')
        # Calling is_class(args, kwargs) (line 168)
        is_class_call_result_10292 = invoke(stypy.reporting.localization.Localization(__file__, 168, 15), is_class_10288, *[python_entity_10290], **kwargs_10291)
        
        # Applying the 'not' unary operator (line 168)
        result_not__10293 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 11), 'not', is_class_call_result_10292)
        
        # Testing if the type of an if condition is none (line 168)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 168, 8), result_not__10293):
            pass
        else:
            
            # Testing the type of an if condition (line 168)
            if_condition_10294 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 8), result_not__10293)
            # Assigning a type to the variable 'if_condition_10294' (line 168)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'if_condition_10294', if_condition_10294)
            # SSA begins for if statement (line 168)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 169)
            # Processing the call arguments (line 169)
            # Getting the type of 'localization' (line 169)
            localization_10296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 29), 'localization', False)
            str_10297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 43), 'str', 'Cannot change the type of a Python entity that it is not a class')
            # Processing the call keyword arguments (line 169)
            kwargs_10298 = {}
            # Getting the type of 'TypeError' (line 169)
            TypeError_10295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 169)
            TypeError_call_result_10299 = invoke(stypy.reporting.localization.Localization(__file__, 169, 19), TypeError_10295, *[localization_10296, str_10297], **kwargs_10298)
            
            # Assigning a type to the variable 'stypy_return_type' (line 169)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'stypy_return_type', TypeError_call_result_10299)
            # SSA join for if statement (line 168)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to is_user_defined_class(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'self' (line 172)
        self_10302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 74), 'self', False)
        # Obtaining the member 'instance' of a type (line 172)
        instance_10303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 74), self_10302, 'instance')
        # Obtaining the member '__class__' of a type (line 172)
        class___10304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 74), instance_10303, '__class__')
        # Processing the call keyword arguments (line 172)
        kwargs_10305 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 172)
        type_inference_proxy_management_copy_10300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'is_user_defined_class' of a type (line 172)
        is_user_defined_class_10301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 15), type_inference_proxy_management_copy_10300, 'is_user_defined_class')
        # Calling is_user_defined_class(args, kwargs) (line 172)
        is_user_defined_class_call_result_10306 = invoke(stypy.reporting.localization.Localization(__file__, 172, 15), is_user_defined_class_10301, *[class___10304], **kwargs_10305)
        
        # Applying the 'not' unary operator (line 172)
        result_not__10307 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 11), 'not', is_user_defined_class_call_result_10306)
        
        # Testing if the type of an if condition is none (line 172)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 172, 8), result_not__10307):
            pass
        else:
            
            # Testing the type of an if condition (line 172)
            if_condition_10308 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 8), result_not__10307)
            # Assigning a type to the variable 'if_condition_10308' (line 172)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'if_condition_10308', if_condition_10308)
            # SSA begins for if statement (line 172)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 173)
            # Processing the call arguments (line 173)
            # Getting the type of 'localization' (line 173)
            localization_10310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 29), 'localization', False)
            str_10311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 43), 'str', 'Cannot change the type of an instance of a non user-defined class')
            # Processing the call keyword arguments (line 173)
            kwargs_10312 = {}
            # Getting the type of 'TypeError' (line 173)
            TypeError_10309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 173)
            TypeError_call_result_10313 = invoke(stypy.reporting.localization.Localization(__file__, 173, 19), TypeError_10309, *[localization_10310, str_10311], **kwargs_10312)
            
            # Assigning a type to the variable 'stypy_return_type' (line 173)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'stypy_return_type', TypeError_call_result_10313)
            # SSA join for if statement (line 172)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Type idiom detected: calculating its left and rigth part (line 176)
        # Getting the type of 'self' (line 176)
        self_10314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 11), 'self')
        # Obtaining the member 'instance' of a type (line 176)
        instance_10315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 11), self_10314, 'instance')
        # Getting the type of 'None' (line 176)
        None_10316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 28), 'None')
        
        (may_be_10317, more_types_in_union_10318) = may_be_none(instance_10315, None_10316)

        if may_be_10317:

            if more_types_in_union_10318:
                # Runtime conditional SSA (line 176)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to TypeError(...): (line 177)
            # Processing the call arguments (line 177)
            # Getting the type of 'localization' (line 177)
            localization_10320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 29), 'localization', False)
            str_10321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 43), 'str', 'Cannot change the type of a class object; Type change is only possiblewith class instances')
            # Processing the call keyword arguments (line 177)
            kwargs_10322 = {}
            # Getting the type of 'TypeError' (line 177)
            TypeError_10319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 177)
            TypeError_call_result_10323 = invoke(stypy.reporting.localization.Localization(__file__, 177, 19), TypeError_10319, *[localization_10320, str_10321], **kwargs_10322)
            
            # Assigning a type to the variable 'stypy_return_type' (line 177)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'stypy_return_type', TypeError_call_result_10323)

            if more_types_in_union_10318:
                # SSA join for if statement (line 176)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 180):
        
        # Assigning a Call to a Name (line 180):
        
        # Call to is_old_style_class(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'self' (line 180)
        self_10326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 85), 'self', False)
        # Obtaining the member 'instance' of a type (line 180)
        instance_10327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 85), self_10326, 'instance')
        # Obtaining the member '__class__' of a type (line 180)
        class___10328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 85), instance_10327, '__class__')
        # Processing the call keyword arguments (line 180)
        kwargs_10329 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 180)
        type_inference_proxy_management_copy_10324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 29), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'is_old_style_class' of a type (line 180)
        is_old_style_class_10325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 29), type_inference_proxy_management_copy_10324, 'is_old_style_class')
        # Calling is_old_style_class(args, kwargs) (line 180)
        is_old_style_class_call_result_10330 = invoke(stypy.reporting.localization.Localization(__file__, 180, 29), is_old_style_class_10325, *[class___10328], **kwargs_10329)
        
        # Assigning a type to the variable 'old_style_existing' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'old_style_existing', is_old_style_class_call_result_10330)
        
        # Assigning a Call to a Name (line 181):
        
        # Assigning a Call to a Name (line 181):
        
        # Call to is_old_style_class(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'new_type' (line 181)
        new_type_10333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 80), 'new_type', False)
        # Obtaining the member 'python_entity' of a type (line 181)
        python_entity_10334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 80), new_type_10333, 'python_entity')
        # Processing the call keyword arguments (line 181)
        kwargs_10335 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 181)
        type_inference_proxy_management_copy_10331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 24), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'is_old_style_class' of a type (line 181)
        is_old_style_class_10332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 24), type_inference_proxy_management_copy_10331, 'is_old_style_class')
        # Calling is_old_style_class(args, kwargs) (line 181)
        is_old_style_class_call_result_10336 = invoke(stypy.reporting.localization.Localization(__file__, 181, 24), is_old_style_class_10332, *[python_entity_10334], **kwargs_10335)
        
        # Assigning a type to the variable 'old_style_new' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'old_style_new', is_old_style_class_call_result_10336)
        
        
        # Getting the type of 'old_style_existing' (line 184)
        old_style_existing_10337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 15), 'old_style_existing')
        # Getting the type of 'old_style_new' (line 184)
        old_style_new_10338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 37), 'old_style_new')
        # Applying the binary operator '==' (line 184)
        result_eq_10339 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 15), '==', old_style_existing_10337, old_style_new_10338)
        
        # Applying the 'not' unary operator (line 184)
        result_not__10340 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 11), 'not', result_eq_10339)
        
        # Testing if the type of an if condition is none (line 184)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 184, 8), result_not__10340):
            pass
        else:
            
            # Testing the type of an if condition (line 184)
            if_condition_10341 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 8), result_not__10340)
            # Assigning a type to the variable 'if_condition_10341' (line 184)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'if_condition_10341', if_condition_10341)
            # SSA begins for if statement (line 184)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 185)
            # Processing the call arguments (line 185)
            # Getting the type of 'localization' (line 185)
            localization_10343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 29), 'localization', False)
            str_10344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 43), 'str', 'Cannot change the type of an instances from an old-style Python class to a new-style Python class or viceversa')
            # Processing the call keyword arguments (line 185)
            kwargs_10345 = {}
            # Getting the type of 'TypeError' (line 185)
            TypeError_10342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 185)
            TypeError_call_result_10346 = invoke(stypy.reporting.localization.Localization(__file__, 185, 19), TypeError_10342, *[localization_10343, str_10344], **kwargs_10345)
            
            # Assigning a type to the variable 'stypy_return_type' (line 185)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'stypy_return_type', TypeError_call_result_10346)
            # SSA join for if statement (line 184)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'None' (line 188)
        None_10347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'stypy_return_type', None_10347)
        
        # ################# End of '__change_instance_type_checks(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__change_instance_type_checks' in the type store
        # Getting the type of 'stypy_return_type' (line 155)
        stypy_return_type_10348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10348)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__change_instance_type_checks'
        return stypy_return_type_10348


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 192)
        None_10349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 43), 'None')
        # Getting the type of 'None' (line 192)
        None_10350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 56), 'None')
        # Getting the type of 'None' (line 192)
        None_10351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 71), 'None')
        # Getting the type of 'undefined_type_copy' (line 192)
        undefined_type_copy_10352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 83), 'undefined_type_copy')
        # Obtaining the member 'UndefinedType' of a type (line 192)
        UndefinedType_10353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 83), undefined_type_copy_10352, 'UndefinedType')
        defaults = [None_10349, None_10350, None_10351, UndefinedType_10353]
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

        str_10354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, (-1)), 'str', '\n        Creates a new Type inference proxy for the passed python entity (function, module, class, ...). This constructor\n        should NOT be called directly. Use the instance(...) method instead to take advantage of the implemented\n        type memoization of this class.\n        :param python_entity: Represented python entity.\n        :param name: Name of the represented Python entity. If nothing is provided, the Python entity __name__ property\n        value is used instead. Instances have an special name indicating that this entity holds an instance of a class.\n        :param parent: Parent proxy object. If nothing is provided, the parent proxy is autocalculated, if possible.\n        :param instance: Instance of the represented class. If this proxy holds a class, it is possible that instead\n        of representing the class is representing a particular class instance. This is important to properly model\n        instance intercession, as altering the structure of single class instances is possible.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 205)
        # Getting the type of 'name' (line 205)
        name_10355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 11), 'name')
        # Getting the type of 'None' (line 205)
        None_10356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 19), 'None')
        
        (may_be_10357, more_types_in_union_10358) = may_be_none(name_10355, None_10356)

        if may_be_10357:

            if more_types_in_union_10358:
                # Runtime conditional SSA (line 205)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 206)
            str_10359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 38), 'str', '__name__')
            # Getting the type of 'python_entity' (line 206)
            python_entity_10360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 23), 'python_entity')
            
            (may_be_10361, more_types_in_union_10362) = may_provide_member(str_10359, python_entity_10360)

            if may_be_10361:

                if more_types_in_union_10362:
                    # Runtime conditional SSA (line 206)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'python_entity' (line 206)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'python_entity', remove_not_member_provider_from_union(python_entity_10360, '__name__'))
                
                # Assigning a Attribute to a Attribute (line 207):
                
                # Assigning a Attribute to a Attribute (line 207):
                # Getting the type of 'python_entity' (line 207)
                python_entity_10363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 28), 'python_entity')
                # Obtaining the member '__name__' of a type (line 207)
                name___10364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 28), python_entity_10363, '__name__')
                # Getting the type of 'self' (line 207)
                self_10365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'self')
                # Setting the type of the member 'name' of a type (line 207)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 16), self_10365, 'name', name___10364)

                if more_types_in_union_10362:
                    # Runtime conditional SSA for else branch (line 206)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_10361) or more_types_in_union_10362):
                # Assigning a type to the variable 'python_entity' (line 206)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'python_entity', remove_member_provider_from_union(python_entity_10360, '__name__'))
                
                # Type idiom detected: calculating its left and rigth part (line 209)
                str_10366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 42), 'str', '__class__')
                # Getting the type of 'python_entity' (line 209)
                python_entity_10367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 27), 'python_entity')
                
                (may_be_10368, more_types_in_union_10369) = may_provide_member(str_10366, python_entity_10367)

                if may_be_10368:

                    if more_types_in_union_10369:
                        # Runtime conditional SSA (line 209)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Assigning a type to the variable 'python_entity' (line 209)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 16), 'python_entity', remove_not_member_provider_from_union(python_entity_10367, '__class__'))
                    
                    # Assigning a Attribute to a Attribute (line 210):
                    
                    # Assigning a Attribute to a Attribute (line 210):
                    # Getting the type of 'python_entity' (line 210)
                    python_entity_10370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 32), 'python_entity')
                    # Obtaining the member '__class__' of a type (line 210)
                    class___10371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 32), python_entity_10370, '__class__')
                    # Obtaining the member '__name__' of a type (line 210)
                    name___10372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 32), class___10371, '__name__')
                    # Getting the type of 'self' (line 210)
                    self_10373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 20), 'self')
                    # Setting the type of the member 'name' of a type (line 210)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 20), self_10373, 'name', name___10372)

                    if more_types_in_union_10369:
                        # Runtime conditional SSA for else branch (line 209)
                        module_type_store.open_ssa_branch('idiom else')



                if ((not may_be_10368) or more_types_in_union_10369):
                    # Assigning a type to the variable 'python_entity' (line 209)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 16), 'python_entity', remove_member_provider_from_union(python_entity_10367, '__class__'))
                    
                    # Type idiom detected: calculating its left and rigth part (line 212)
                    str_10374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 46), 'str', '__module__')
                    # Getting the type of 'python_entity' (line 212)
                    python_entity_10375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 31), 'python_entity')
                    
                    (may_be_10376, more_types_in_union_10377) = may_provide_member(str_10374, python_entity_10375)

                    if may_be_10376:

                        if more_types_in_union_10377:
                            # Runtime conditional SSA (line 212)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                        else:
                            module_type_store = module_type_store

                        # Assigning a type to the variable 'python_entity' (line 212)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), 'python_entity', remove_not_member_provider_from_union(python_entity_10375, '__module__'))
                        
                        # Assigning a Attribute to a Attribute (line 213):
                        
                        # Assigning a Attribute to a Attribute (line 213):
                        # Getting the type of 'python_entity' (line 213)
                        python_entity_10378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 36), 'python_entity')
                        # Obtaining the member '__module__' of a type (line 213)
                        module___10379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 36), python_entity_10378, '__module__')
                        # Getting the type of 'self' (line 213)
                        self_10380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 24), 'self')
                        # Setting the type of the member 'name' of a type (line 213)
                        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 24), self_10380, 'name', module___10379)

                        if more_types_in_union_10377:
                            # SSA join for if statement (line 212)
                            module_type_store = module_type_store.join_ssa_context()


                    

                    if (may_be_10368 and more_types_in_union_10369):
                        # SSA join for if statement (line 209)
                        module_type_store = module_type_store.join_ssa_context()


                

                if (may_be_10361 and more_types_in_union_10362):
                    # SSA join for if statement (line 206)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Type idiom detected: calculating its left and rigth part (line 215)
            # Getting the type of 'instance' (line 215)
            instance_10381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'instance')
            # Getting the type of 'None' (line 215)
            None_10382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 31), 'None')
            
            (may_be_10383, more_types_in_union_10384) = may_not_be_none(instance_10381, None_10382)

            if may_be_10383:

                if more_types_in_union_10384:
                    # Runtime conditional SSA (line 215)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a BinOp to a Attribute (line 216):
                
                # Assigning a BinOp to a Attribute (line 216):
                str_10385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 28), 'str', '<')
                # Getting the type of 'self' (line 216)
                self_10386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 34), 'self')
                # Obtaining the member 'name' of a type (line 216)
                name_10387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 34), self_10386, 'name')
                # Applying the binary operator '+' (line 216)
                result_add_10388 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 28), '+', str_10385, name_10387)
                
                str_10389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 46), 'str', ' instance>')
                # Applying the binary operator '+' (line 216)
                result_add_10390 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 44), '+', result_add_10388, str_10389)
                
                # Getting the type of 'self' (line 216)
                self_10391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'self')
                # Setting the type of the member 'name' of a type (line 216)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 16), self_10391, 'name', result_add_10390)

                if more_types_in_union_10384:
                    # SSA join for if statement (line 215)
                    module_type_store = module_type_store.join_ssa_context()


            

            if more_types_in_union_10358:
                # Runtime conditional SSA for else branch (line 205)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_10357) or more_types_in_union_10358):
            
            # Assigning a Name to a Attribute (line 218):
            
            # Assigning a Name to a Attribute (line 218):
            # Getting the type of 'name' (line 218)
            name_10392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 24), 'name')
            # Getting the type of 'self' (line 218)
            self_10393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'self')
            # Setting the type of the member 'name' of a type (line 218)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 12), self_10393, 'name', name_10392)

            if (may_be_10357 and more_types_in_union_10358):
                # SSA join for if statement (line 205)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 220):
        
        # Assigning a Name to a Attribute (line 220):
        # Getting the type of 'python_entity' (line 220)
        python_entity_10394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 29), 'python_entity')
        # Getting the type of 'self' (line 220)
        self_10395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'self')
        # Setting the type of the member 'python_entity' of a type (line 220)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 8), self_10395, 'python_entity', python_entity_10394)
        
        # Call to __assign_parent_proxy(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'parent' (line 221)
        parent_10398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 35), 'parent', False)
        # Processing the call keyword arguments (line 221)
        kwargs_10399 = {}
        # Getting the type of 'self' (line 221)
        self_10396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'self', False)
        # Obtaining the member '__assign_parent_proxy' of a type (line 221)
        assign_parent_proxy_10397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), self_10396, '__assign_parent_proxy')
        # Calling __assign_parent_proxy(args, kwargs) (line 221)
        assign_parent_proxy_call_result_10400 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), assign_parent_proxy_10397, *[parent_10398], **kwargs_10399)
        
        
        # Assigning a Name to a Attribute (line 222):
        
        # Assigning a Name to a Attribute (line 222):
        # Getting the type of 'instance' (line 222)
        instance_10401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 24), 'instance')
        # Getting the type of 'self' (line 222)
        self_10402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'self')
        # Setting the type of the member 'instance' of a type (line 222)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), self_10402, 'instance', instance_10401)
        
        # Type idiom detected: calculating its left and rigth part (line 223)
        # Getting the type of 'instance' (line 223)
        instance_10403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'instance')
        # Getting the type of 'None' (line 223)
        None_10404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 27), 'None')
        
        (may_be_10405, more_types_in_union_10406) = may_not_be_none(instance_10403, None_10404)

        if may_be_10405:

            if more_types_in_union_10406:
                # Runtime conditional SSA (line 223)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to set_type_instance(...): (line 224)
            # Processing the call arguments (line 224)
            # Getting the type of 'True' (line 224)
            True_10409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 35), 'True', False)
            # Processing the call keyword arguments (line 224)
            kwargs_10410 = {}
            # Getting the type of 'self' (line 224)
            self_10407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'self', False)
            # Obtaining the member 'set_type_instance' of a type (line 224)
            set_type_instance_10408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 12), self_10407, 'set_type_instance')
            # Calling set_type_instance(args, kwargs) (line 224)
            set_type_instance_call_result_10411 = invoke(stypy.reporting.localization.Localization(__file__, 224, 12), set_type_instance_10408, *[True_10409], **kwargs_10410)
            

            if more_types_in_union_10406:
                # SSA join for if statement (line 223)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Attribute (line 227):
        
        # Assigning a Call to a Attribute (line 227):
        
        # Call to list(...): (line 227)
        # Processing the call keyword arguments (line 227)
        kwargs_10413 = {}
        # Getting the type of 'list' (line 227)
        list_10412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 34), 'list', False)
        # Calling list(args, kwargs) (line 227)
        list_call_result_10414 = invoke(stypy.reporting.localization.Localization(__file__, 227, 34), list_10412, *[], **kwargs_10413)
        
        # Getting the type of 'self' (line 227)
        self_10415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'self')
        # Setting the type of the member 'additional_members' of a type (line 227)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), self_10415, 'additional_members', list_call_result_10414)
        
        # Assigning a Name to a Attribute (line 230):
        
        # Assigning a Name to a Attribute (line 230):
        # Getting the type of 'None' (line 230)
        None_10416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 23), 'None')
        # Getting the type of 'self' (line 230)
        self_10417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'self')
        # Setting the type of the member 'type_of' of a type (line 230)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), self_10417, 'type_of', None_10416)
        
        # Assigning a Name to a Attribute (line 234):
        
        # Assigning a Name to a Attribute (line 234):
        # Getting the type of 'True' (line 234)
        True_10418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 31), 'True')
        # Getting the type of 'self' (line 234)
        self_10419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'self')
        # Setting the type of the member 'known_structure' of a type (line 234)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), self_10419, 'known_structure', True_10418)
        
        # Getting the type of 'value' (line 236)
        value_10420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 11), 'value')
        # Getting the type of 'undefined_type_copy' (line 236)
        undefined_type_copy_10421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 24), 'undefined_type_copy')
        # Obtaining the member 'UndefinedType' of a type (line 236)
        UndefinedType_10422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 24), undefined_type_copy_10421, 'UndefinedType')
        # Applying the binary operator 'isnot' (line 236)
        result_is_not_10423 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 11), 'isnot', value_10420, UndefinedType_10422)
        
        # Testing if the type of an if condition is none (line 236)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 236, 8), result_is_not_10423):
            pass
        else:
            
            # Testing the type of an if condition (line 236)
            if_condition_10424 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 8), result_is_not_10423)
            # Assigning a type to the variable 'if_condition_10424' (line 236)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'if_condition_10424', if_condition_10424)
            # SSA begins for if statement (line 236)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 237):
            
            # Assigning a Name to a Attribute (line 237):
            # Getting the type of 'value' (line 237)
            value_10425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 25), 'value')
            # Getting the type of 'self' (line 237)
            self_10426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'self')
            # Setting the type of the member 'value' of a type (line 237)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), self_10426, 'value', value_10425)
            
            # Call to set_type_instance(...): (line 238)
            # Processing the call arguments (line 238)
            # Getting the type of 'True' (line 238)
            True_10429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 35), 'True', False)
            # Processing the call keyword arguments (line 238)
            kwargs_10430 = {}
            # Getting the type of 'self' (line 238)
            self_10427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'self', False)
            # Obtaining the member 'set_type_instance' of a type (line 238)
            set_type_instance_10428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 12), self_10427, 'set_type_instance')
            # Calling set_type_instance(args, kwargs) (line 238)
            set_type_instance_call_result_10431 = invoke(stypy.reporting.localization.Localization(__file__, 238, 12), set_type_instance_10428, *[True_10429], **kwargs_10430)
            
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

        str_10432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, (-1)), 'str', '\n        String representation of this proxy and its contents. Python builtin types have a very concise representation.\n        The method have been stripped down of much of its information gathering code to favor a more concise and clear\n        representation of entities.\n        :return: str\n        ')
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'self' (line 281)
        self_10434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 22), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 281)
        python_entity_10435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 22), self_10434, 'python_entity')
        # Getting the type of 'types' (line 281)
        types_10436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 42), 'types', False)
        # Obtaining the member 'InstanceType' of a type (line 281)
        InstanceType_10437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 42), types_10436, 'InstanceType')
        # Processing the call keyword arguments (line 281)
        kwargs_10438 = {}
        # Getting the type of 'isinstance' (line 281)
        isinstance_10433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 281)
        isinstance_call_result_10439 = invoke(stypy.reporting.localization.Localization(__file__, 281, 11), isinstance_10433, *[python_entity_10435, InstanceType_10437], **kwargs_10438)
        
        
        # Call to isinstance(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'self' (line 281)
        self_10441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 76), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 281)
        python_entity_10442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 76), self_10441, 'python_entity')
        # Getting the type of 'types' (line 281)
        types_10443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 96), 'types', False)
        # Obtaining the member 'ClassType' of a type (line 281)
        ClassType_10444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 96), types_10443, 'ClassType')
        # Processing the call keyword arguments (line 281)
        kwargs_10445 = {}
        # Getting the type of 'isinstance' (line 281)
        isinstance_10440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 65), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 281)
        isinstance_call_result_10446 = invoke(stypy.reporting.localization.Localization(__file__, 281, 65), isinstance_10440, *[python_entity_10442, ClassType_10444], **kwargs_10445)
        
        # Applying the binary operator 'or' (line 281)
        result_or_keyword_10447 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 11), 'or', isinstance_call_result_10439, isinstance_call_result_10446)
        
        # Testing if the type of an if condition is none (line 281)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 281, 8), result_or_keyword_10447):
            pass
        else:
            
            # Testing the type of an if condition (line 281)
            if_condition_10448 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 281, 8), result_or_keyword_10447)
            # Assigning a type to the variable 'if_condition_10448' (line 281)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'if_condition_10448', if_condition_10448)
            # SSA begins for if statement (line 281)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'self' (line 282)
            self_10449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 19), 'self')
            # Obtaining the member 'name' of a type (line 282)
            name_10450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 19), self_10449, 'name')
            # Assigning a type to the variable 'stypy_return_type' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'stypy_return_type', name_10450)
            # SSA join for if statement (line 281)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 285)
        self_10451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 11), 'self')
        # Obtaining the member 'python_entity' of a type (line 285)
        python_entity_10452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 11), self_10451, 'python_entity')
        # Getting the type of 'simple_python_types' (line 285)
        simple_python_types_10453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 33), 'simple_python_types')
        # Applying the binary operator 'in' (line 285)
        result_contains_10454 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 11), 'in', python_entity_10452, simple_python_types_10453)
        
        # Testing if the type of an if condition is none (line 285)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 285, 8), result_contains_10454):
            pass
        else:
            
            # Testing the type of an if condition (line 285)
            if_condition_10455 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 285, 8), result_contains_10454)
            # Assigning a type to the variable 'if_condition_10455' (line 285)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'if_condition_10455', if_condition_10455)
            # SSA begins for if statement (line 285)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to get_python_type(...): (line 286)
            # Processing the call keyword arguments (line 286)
            kwargs_10458 = {}
            # Getting the type of 'self' (line 286)
            self_10456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 19), 'self', False)
            # Obtaining the member 'get_python_type' of a type (line 286)
            get_python_type_10457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 19), self_10456, 'get_python_type')
            # Calling get_python_type(args, kwargs) (line 286)
            get_python_type_call_result_10459 = invoke(stypy.reporting.localization.Localization(__file__, 286, 19), get_python_type_10457, *[], **kwargs_10458)
            
            # Obtaining the member '__name__' of a type (line 286)
            name___10460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 19), get_python_type_call_result_10459, '__name__')
            # Assigning a type to the variable 'stypy_return_type' (line 286)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'stypy_return_type', name___10460)
            # SSA join for if statement (line 285)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Str to a Name (line 288):
        
        # Assigning a Str to a Name (line 288):
        str_10461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 21), 'str', '')
        # Assigning a type to the variable 'parent_str' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'parent_str', str_10461)
        
        # Assigning a Str to a Name (line 301):
        
        # Assigning a Str to a Name (line 301):
        str_10462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 19), 'str', '')
        # Assigning a type to the variable 'str_mark' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'str_mark', str_10462)
        
        # Getting the type of 'self' (line 317)
        self_10463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 11), 'self')
        # Obtaining the member 'instance' of a type (line 317)
        instance_10464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 11), self_10463, 'instance')
        # Getting the type of 'None' (line 317)
        None_10465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 32), 'None')
        # Applying the binary operator 'isnot' (line 317)
        result_is_not_10466 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 11), 'isnot', instance_10464, None_10465)
        
        # Testing if the type of an if condition is none (line 317)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 317, 8), result_is_not_10466):
            
            # Assigning a Str to a Name (line 321):
            
            # Assigning a Str to a Name (line 321):
            str_10475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 28), 'str', '')
            # Assigning a type to the variable 'instance_type' (line 321)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'instance_type', str_10475)
        else:
            
            # Testing the type of an if condition (line 317)
            if_condition_10467 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 317, 8), result_is_not_10466)
            # Assigning a type to the variable 'if_condition_10467' (line 317)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'if_condition_10467', if_condition_10467)
            # SSA begins for if statement (line 317)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 318):
            
            # Assigning a BinOp to a Name (line 318):
            # Getting the type of 'self' (line 318)
            self_10468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 28), 'self')
            # Obtaining the member 'instance' of a type (line 318)
            instance_10469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 28), self_10468, 'instance')
            # Obtaining the member '__class__' of a type (line 318)
            class___10470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 28), instance_10469, '__class__')
            # Obtaining the member '__name__' of a type (line 318)
            name___10471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 28), class___10470, '__name__')
            str_10472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 63), 'str', ' instance')
            # Applying the binary operator '+' (line 318)
            result_add_10473 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 28), '+', name___10471, str_10472)
            
            # Assigning a type to the variable 'instance_type' (line 318)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'instance_type', result_add_10473)
            # Getting the type of 'instance_type' (line 319)
            instance_type_10474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 19), 'instance_type')
            # Assigning a type to the variable 'stypy_return_type' (line 319)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'stypy_return_type', instance_type_10474)
            # SSA branch for the else part of an if statement (line 317)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Str to a Name (line 321):
            
            # Assigning a Str to a Name (line 321):
            str_10475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 28), 'str', '')
            # Assigning a type to the variable 'instance_type' (line 321)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'instance_type', str_10475)
            # SSA join for if statement (line 317)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to hasattr(...): (line 324)
        # Processing the call arguments (line 324)
        # Getting the type of 'self' (line 324)
        self_10477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 19), 'self', False)
        # Getting the type of 'self' (line 324)
        self_10478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 25), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 324)
        contained_elements_property_name_10479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 25), self_10478, 'contained_elements_property_name')
        # Processing the call keyword arguments (line 324)
        kwargs_10480 = {}
        # Getting the type of 'hasattr' (line 324)
        hasattr_10476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 11), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 324)
        hasattr_call_result_10481 = invoke(stypy.reporting.localization.Localization(__file__, 324, 11), hasattr_10476, *[self_10477, contained_elements_property_name_10479], **kwargs_10480)
        
        # Testing if the type of an if condition is none (line 324)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 324, 8), hasattr_call_result_10481):
            
            # Call to can_store_elements(...): (line 329)
            # Processing the call keyword arguments (line 329)
            kwargs_10509 = {}
            # Getting the type of 'self' (line 329)
            self_10507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 15), 'self', False)
            # Obtaining the member 'can_store_elements' of a type (line 329)
            can_store_elements_10508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 15), self_10507, 'can_store_elements')
            # Calling can_store_elements(args, kwargs) (line 329)
            can_store_elements_call_result_10510 = invoke(stypy.reporting.localization.Localization(__file__, 329, 15), can_store_elements_10508, *[], **kwargs_10509)
            
            # Testing if the type of an if condition is none (line 329)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 329, 12), can_store_elements_call_result_10510):
                
                # Call to can_store_keypairs(...): (line 334)
                # Processing the call keyword arguments (line 334)
                kwargs_10526 = {}
                # Getting the type of 'self' (line 334)
                self_10524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 19), 'self', False)
                # Obtaining the member 'can_store_keypairs' of a type (line 334)
                can_store_keypairs_10525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 19), self_10524, 'can_store_keypairs')
                # Calling can_store_keypairs(args, kwargs) (line 334)
                can_store_keypairs_call_result_10527 = invoke(stypy.reporting.localization.Localization(__file__, 334, 19), can_store_keypairs_10525, *[], **kwargs_10526)
                
                # Testing if the type of an if condition is none (line 334)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 334, 16), can_store_keypairs_call_result_10527):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 334)
                    if_condition_10528 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 16), can_store_keypairs_call_result_10527)
                    # Assigning a type to the variable 'if_condition_10528' (line 334)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'if_condition_10528', if_condition_10528)
                    # SSA begins for if statement (line 334)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Str to a Name (line 335):
                    
                    # Assigning a Str to a Name (line 335):
                    str_10529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 36), 'str', '{}')
                    # Assigning a type to the variable 'contained_str' (line 335)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 20), 'contained_str', str_10529)
                    
                    # Call to format(...): (line 336)
                    # Processing the call arguments (line 336)
                    
                    # Call to get_python_type(...): (line 336)
                    # Processing the call keyword arguments (line 336)
                    kwargs_10534 = {}
                    # Getting the type of 'self' (line 336)
                    self_10532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 40), 'self', False)
                    # Obtaining the member 'get_python_type' of a type (line 336)
                    get_python_type_10533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 40), self_10532, 'get_python_type')
                    # Calling get_python_type(args, kwargs) (line 336)
                    get_python_type_call_result_10535 = invoke(stypy.reporting.localization.Localization(__file__, 336, 40), get_python_type_10533, *[], **kwargs_10534)
                    
                    # Obtaining the member '__name__' of a type (line 336)
                    name___10536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 40), get_python_type_call_result_10535, '__name__')
                    # Processing the call keyword arguments (line 336)
                    kwargs_10537 = {}
                    str_10530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 27), 'str', '{0}')
                    # Obtaining the member 'format' of a type (line 336)
                    format_10531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 27), str_10530, 'format')
                    # Calling format(args, kwargs) (line 336)
                    format_call_result_10538 = invoke(stypy.reporting.localization.Localization(__file__, 336, 27), format_10531, *[name___10536], **kwargs_10537)
                    
                    # Getting the type of 'contained_str' (line 337)
                    contained_str_10539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 29), 'contained_str')
                    # Applying the binary operator '+' (line 336)
                    result_add_10540 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 27), '+', format_call_result_10538, contained_str_10539)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 336)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 20), 'stypy_return_type', result_add_10540)
                    # SSA join for if statement (line 334)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 329)
                if_condition_10511 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 329, 12), can_store_elements_call_result_10510)
                # Assigning a type to the variable 'if_condition_10511' (line 329)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'if_condition_10511', if_condition_10511)
                # SSA begins for if statement (line 329)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Str to a Name (line 330):
                
                # Assigning a Str to a Name (line 330):
                str_10512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 32), 'str', '[]')
                # Assigning a type to the variable 'contained_str' (line 330)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'contained_str', str_10512)
                
                # Call to format(...): (line 331)
                # Processing the call arguments (line 331)
                
                # Call to get_python_type(...): (line 331)
                # Processing the call keyword arguments (line 331)
                kwargs_10517 = {}
                # Getting the type of 'self' (line 331)
                self_10515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 36), 'self', False)
                # Obtaining the member 'get_python_type' of a type (line 331)
                get_python_type_10516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 36), self_10515, 'get_python_type')
                # Calling get_python_type(args, kwargs) (line 331)
                get_python_type_call_result_10518 = invoke(stypy.reporting.localization.Localization(__file__, 331, 36), get_python_type_10516, *[], **kwargs_10517)
                
                # Obtaining the member '__name__' of a type (line 331)
                name___10519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 36), get_python_type_call_result_10518, '__name__')
                # Processing the call keyword arguments (line 331)
                kwargs_10520 = {}
                str_10513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 23), 'str', '{0}')
                # Obtaining the member 'format' of a type (line 331)
                format_10514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 23), str_10513, 'format')
                # Calling format(args, kwargs) (line 331)
                format_call_result_10521 = invoke(stypy.reporting.localization.Localization(__file__, 331, 23), format_10514, *[name___10519], **kwargs_10520)
                
                # Getting the type of 'contained_str' (line 332)
                contained_str_10522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 25), 'contained_str')
                # Applying the binary operator '+' (line 331)
                result_add_10523 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 23), '+', format_call_result_10521, contained_str_10522)
                
                # Assigning a type to the variable 'stypy_return_type' (line 331)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'stypy_return_type', result_add_10523)
                # SSA branch for the else part of an if statement (line 329)
                module_type_store.open_ssa_branch('else')
                
                # Call to can_store_keypairs(...): (line 334)
                # Processing the call keyword arguments (line 334)
                kwargs_10526 = {}
                # Getting the type of 'self' (line 334)
                self_10524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 19), 'self', False)
                # Obtaining the member 'can_store_keypairs' of a type (line 334)
                can_store_keypairs_10525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 19), self_10524, 'can_store_keypairs')
                # Calling can_store_keypairs(args, kwargs) (line 334)
                can_store_keypairs_call_result_10527 = invoke(stypy.reporting.localization.Localization(__file__, 334, 19), can_store_keypairs_10525, *[], **kwargs_10526)
                
                # Testing if the type of an if condition is none (line 334)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 334, 16), can_store_keypairs_call_result_10527):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 334)
                    if_condition_10528 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 16), can_store_keypairs_call_result_10527)
                    # Assigning a type to the variable 'if_condition_10528' (line 334)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'if_condition_10528', if_condition_10528)
                    # SSA begins for if statement (line 334)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Str to a Name (line 335):
                    
                    # Assigning a Str to a Name (line 335):
                    str_10529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 36), 'str', '{}')
                    # Assigning a type to the variable 'contained_str' (line 335)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 20), 'contained_str', str_10529)
                    
                    # Call to format(...): (line 336)
                    # Processing the call arguments (line 336)
                    
                    # Call to get_python_type(...): (line 336)
                    # Processing the call keyword arguments (line 336)
                    kwargs_10534 = {}
                    # Getting the type of 'self' (line 336)
                    self_10532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 40), 'self', False)
                    # Obtaining the member 'get_python_type' of a type (line 336)
                    get_python_type_10533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 40), self_10532, 'get_python_type')
                    # Calling get_python_type(args, kwargs) (line 336)
                    get_python_type_call_result_10535 = invoke(stypy.reporting.localization.Localization(__file__, 336, 40), get_python_type_10533, *[], **kwargs_10534)
                    
                    # Obtaining the member '__name__' of a type (line 336)
                    name___10536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 40), get_python_type_call_result_10535, '__name__')
                    # Processing the call keyword arguments (line 336)
                    kwargs_10537 = {}
                    str_10530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 27), 'str', '{0}')
                    # Obtaining the member 'format' of a type (line 336)
                    format_10531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 27), str_10530, 'format')
                    # Calling format(args, kwargs) (line 336)
                    format_call_result_10538 = invoke(stypy.reporting.localization.Localization(__file__, 336, 27), format_10531, *[name___10536], **kwargs_10537)
                    
                    # Getting the type of 'contained_str' (line 337)
                    contained_str_10539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 29), 'contained_str')
                    # Applying the binary operator '+' (line 336)
                    result_add_10540 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 27), '+', format_call_result_10538, contained_str_10539)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 336)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 20), 'stypy_return_type', result_add_10540)
                    # SSA join for if statement (line 334)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 329)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 324)
            if_condition_10482 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 324, 8), hasattr_call_result_10481)
            # Assigning a type to the variable 'if_condition_10482' (line 324)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'if_condition_10482', if_condition_10482)
            # SSA begins for if statement (line 324)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 325):
            
            # Assigning a BinOp to a Name (line 325):
            str_10483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 28), 'str', '[')
            
            # Call to str(...): (line 325)
            # Processing the call arguments (line 325)
            
            # Call to getattr(...): (line 325)
            # Processing the call arguments (line 325)
            # Getting the type of 'self' (line 325)
            self_10486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 46), 'self', False)
            # Getting the type of 'self' (line 325)
            self_10487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 52), 'self', False)
            # Obtaining the member 'contained_elements_property_name' of a type (line 325)
            contained_elements_property_name_10488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 52), self_10487, 'contained_elements_property_name')
            # Processing the call keyword arguments (line 325)
            kwargs_10489 = {}
            # Getting the type of 'getattr' (line 325)
            getattr_10485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 38), 'getattr', False)
            # Calling getattr(args, kwargs) (line 325)
            getattr_call_result_10490 = invoke(stypy.reporting.localization.Localization(__file__, 325, 38), getattr_10485, *[self_10486, contained_elements_property_name_10488], **kwargs_10489)
            
            # Processing the call keyword arguments (line 325)
            kwargs_10491 = {}
            # Getting the type of 'str' (line 325)
            str_10484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 34), 'str', False)
            # Calling str(args, kwargs) (line 325)
            str_call_result_10492 = invoke(stypy.reporting.localization.Localization(__file__, 325, 34), str_10484, *[getattr_call_result_10490], **kwargs_10491)
            
            # Applying the binary operator '+' (line 325)
            result_add_10493 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 28), '+', str_10483, str_call_result_10492)
            
            str_10494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 94), 'str', ']')
            # Applying the binary operator '+' (line 325)
            result_add_10495 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 92), '+', result_add_10493, str_10494)
            
            # Assigning a type to the variable 'contained_str' (line 325)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'contained_str', result_add_10495)
            
            # Call to format(...): (line 326)
            # Processing the call arguments (line 326)
            
            # Call to get_python_type(...): (line 326)
            # Processing the call keyword arguments (line 326)
            kwargs_10500 = {}
            # Getting the type of 'self' (line 326)
            self_10498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 32), 'self', False)
            # Obtaining the member 'get_python_type' of a type (line 326)
            get_python_type_10499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 32), self_10498, 'get_python_type')
            # Calling get_python_type(args, kwargs) (line 326)
            get_python_type_call_result_10501 = invoke(stypy.reporting.localization.Localization(__file__, 326, 32), get_python_type_10499, *[], **kwargs_10500)
            
            # Obtaining the member '__name__' of a type (line 326)
            name___10502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 32), get_python_type_call_result_10501, '__name__')
            # Processing the call keyword arguments (line 326)
            kwargs_10503 = {}
            str_10496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 19), 'str', '{0}')
            # Obtaining the member 'format' of a type (line 326)
            format_10497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 19), str_10496, 'format')
            # Calling format(args, kwargs) (line 326)
            format_call_result_10504 = invoke(stypy.reporting.localization.Localization(__file__, 326, 19), format_10497, *[name___10502], **kwargs_10503)
            
            # Getting the type of 'contained_str' (line 327)
            contained_str_10505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 21), 'contained_str')
            # Applying the binary operator '+' (line 326)
            result_add_10506 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 19), '+', format_call_result_10504, contained_str_10505)
            
            # Assigning a type to the variable 'stypy_return_type' (line 326)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'stypy_return_type', result_add_10506)
            # SSA branch for the else part of an if statement (line 324)
            module_type_store.open_ssa_branch('else')
            
            # Call to can_store_elements(...): (line 329)
            # Processing the call keyword arguments (line 329)
            kwargs_10509 = {}
            # Getting the type of 'self' (line 329)
            self_10507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 15), 'self', False)
            # Obtaining the member 'can_store_elements' of a type (line 329)
            can_store_elements_10508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 15), self_10507, 'can_store_elements')
            # Calling can_store_elements(args, kwargs) (line 329)
            can_store_elements_call_result_10510 = invoke(stypy.reporting.localization.Localization(__file__, 329, 15), can_store_elements_10508, *[], **kwargs_10509)
            
            # Testing if the type of an if condition is none (line 329)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 329, 12), can_store_elements_call_result_10510):
                
                # Call to can_store_keypairs(...): (line 334)
                # Processing the call keyword arguments (line 334)
                kwargs_10526 = {}
                # Getting the type of 'self' (line 334)
                self_10524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 19), 'self', False)
                # Obtaining the member 'can_store_keypairs' of a type (line 334)
                can_store_keypairs_10525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 19), self_10524, 'can_store_keypairs')
                # Calling can_store_keypairs(args, kwargs) (line 334)
                can_store_keypairs_call_result_10527 = invoke(stypy.reporting.localization.Localization(__file__, 334, 19), can_store_keypairs_10525, *[], **kwargs_10526)
                
                # Testing if the type of an if condition is none (line 334)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 334, 16), can_store_keypairs_call_result_10527):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 334)
                    if_condition_10528 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 16), can_store_keypairs_call_result_10527)
                    # Assigning a type to the variable 'if_condition_10528' (line 334)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'if_condition_10528', if_condition_10528)
                    # SSA begins for if statement (line 334)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Str to a Name (line 335):
                    
                    # Assigning a Str to a Name (line 335):
                    str_10529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 36), 'str', '{}')
                    # Assigning a type to the variable 'contained_str' (line 335)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 20), 'contained_str', str_10529)
                    
                    # Call to format(...): (line 336)
                    # Processing the call arguments (line 336)
                    
                    # Call to get_python_type(...): (line 336)
                    # Processing the call keyword arguments (line 336)
                    kwargs_10534 = {}
                    # Getting the type of 'self' (line 336)
                    self_10532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 40), 'self', False)
                    # Obtaining the member 'get_python_type' of a type (line 336)
                    get_python_type_10533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 40), self_10532, 'get_python_type')
                    # Calling get_python_type(args, kwargs) (line 336)
                    get_python_type_call_result_10535 = invoke(stypy.reporting.localization.Localization(__file__, 336, 40), get_python_type_10533, *[], **kwargs_10534)
                    
                    # Obtaining the member '__name__' of a type (line 336)
                    name___10536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 40), get_python_type_call_result_10535, '__name__')
                    # Processing the call keyword arguments (line 336)
                    kwargs_10537 = {}
                    str_10530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 27), 'str', '{0}')
                    # Obtaining the member 'format' of a type (line 336)
                    format_10531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 27), str_10530, 'format')
                    # Calling format(args, kwargs) (line 336)
                    format_call_result_10538 = invoke(stypy.reporting.localization.Localization(__file__, 336, 27), format_10531, *[name___10536], **kwargs_10537)
                    
                    # Getting the type of 'contained_str' (line 337)
                    contained_str_10539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 29), 'contained_str')
                    # Applying the binary operator '+' (line 336)
                    result_add_10540 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 27), '+', format_call_result_10538, contained_str_10539)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 336)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 20), 'stypy_return_type', result_add_10540)
                    # SSA join for if statement (line 334)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 329)
                if_condition_10511 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 329, 12), can_store_elements_call_result_10510)
                # Assigning a type to the variable 'if_condition_10511' (line 329)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'if_condition_10511', if_condition_10511)
                # SSA begins for if statement (line 329)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Str to a Name (line 330):
                
                # Assigning a Str to a Name (line 330):
                str_10512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 32), 'str', '[]')
                # Assigning a type to the variable 'contained_str' (line 330)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'contained_str', str_10512)
                
                # Call to format(...): (line 331)
                # Processing the call arguments (line 331)
                
                # Call to get_python_type(...): (line 331)
                # Processing the call keyword arguments (line 331)
                kwargs_10517 = {}
                # Getting the type of 'self' (line 331)
                self_10515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 36), 'self', False)
                # Obtaining the member 'get_python_type' of a type (line 331)
                get_python_type_10516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 36), self_10515, 'get_python_type')
                # Calling get_python_type(args, kwargs) (line 331)
                get_python_type_call_result_10518 = invoke(stypy.reporting.localization.Localization(__file__, 331, 36), get_python_type_10516, *[], **kwargs_10517)
                
                # Obtaining the member '__name__' of a type (line 331)
                name___10519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 36), get_python_type_call_result_10518, '__name__')
                # Processing the call keyword arguments (line 331)
                kwargs_10520 = {}
                str_10513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 23), 'str', '{0}')
                # Obtaining the member 'format' of a type (line 331)
                format_10514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 23), str_10513, 'format')
                # Calling format(args, kwargs) (line 331)
                format_call_result_10521 = invoke(stypy.reporting.localization.Localization(__file__, 331, 23), format_10514, *[name___10519], **kwargs_10520)
                
                # Getting the type of 'contained_str' (line 332)
                contained_str_10522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 25), 'contained_str')
                # Applying the binary operator '+' (line 331)
                result_add_10523 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 23), '+', format_call_result_10521, contained_str_10522)
                
                # Assigning a type to the variable 'stypy_return_type' (line 331)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'stypy_return_type', result_add_10523)
                # SSA branch for the else part of an if statement (line 329)
                module_type_store.open_ssa_branch('else')
                
                # Call to can_store_keypairs(...): (line 334)
                # Processing the call keyword arguments (line 334)
                kwargs_10526 = {}
                # Getting the type of 'self' (line 334)
                self_10524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 19), 'self', False)
                # Obtaining the member 'can_store_keypairs' of a type (line 334)
                can_store_keypairs_10525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 19), self_10524, 'can_store_keypairs')
                # Calling can_store_keypairs(args, kwargs) (line 334)
                can_store_keypairs_call_result_10527 = invoke(stypy.reporting.localization.Localization(__file__, 334, 19), can_store_keypairs_10525, *[], **kwargs_10526)
                
                # Testing if the type of an if condition is none (line 334)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 334, 16), can_store_keypairs_call_result_10527):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 334)
                    if_condition_10528 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 16), can_store_keypairs_call_result_10527)
                    # Assigning a type to the variable 'if_condition_10528' (line 334)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'if_condition_10528', if_condition_10528)
                    # SSA begins for if statement (line 334)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Str to a Name (line 335):
                    
                    # Assigning a Str to a Name (line 335):
                    str_10529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 36), 'str', '{}')
                    # Assigning a type to the variable 'contained_str' (line 335)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 20), 'contained_str', str_10529)
                    
                    # Call to format(...): (line 336)
                    # Processing the call arguments (line 336)
                    
                    # Call to get_python_type(...): (line 336)
                    # Processing the call keyword arguments (line 336)
                    kwargs_10534 = {}
                    # Getting the type of 'self' (line 336)
                    self_10532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 40), 'self', False)
                    # Obtaining the member 'get_python_type' of a type (line 336)
                    get_python_type_10533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 40), self_10532, 'get_python_type')
                    # Calling get_python_type(args, kwargs) (line 336)
                    get_python_type_call_result_10535 = invoke(stypy.reporting.localization.Localization(__file__, 336, 40), get_python_type_10533, *[], **kwargs_10534)
                    
                    # Obtaining the member '__name__' of a type (line 336)
                    name___10536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 40), get_python_type_call_result_10535, '__name__')
                    # Processing the call keyword arguments (line 336)
                    kwargs_10537 = {}
                    str_10530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 27), 'str', '{0}')
                    # Obtaining the member 'format' of a type (line 336)
                    format_10531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 27), str_10530, 'format')
                    # Calling format(args, kwargs) (line 336)
                    format_call_result_10538 = invoke(stypy.reporting.localization.Localization(__file__, 336, 27), format_10531, *[name___10536], **kwargs_10537)
                    
                    # Getting the type of 'contained_str' (line 337)
                    contained_str_10539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 29), 'contained_str')
                    # Applying the binary operator '+' (line 336)
                    result_add_10540 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 27), '+', format_call_result_10538, contained_str_10539)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 336)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 20), 'stypy_return_type', result_add_10540)
                    # SSA join for if statement (line 334)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 329)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 324)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Str to a Name (line 339):
        
        # Assigning a Str to a Name (line 339):
        str_10541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 19), 'str', '')
        # Assigning a type to the variable 'own_name' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'own_name', str_10541)
        
        # Call to format(...): (line 346)
        # Processing the call arguments (line 346)
        
        # Call to get_python_type(...): (line 346)
        # Processing the call keyword arguments (line 346)
        kwargs_10546 = {}
        # Getting the type of 'self' (line 346)
        self_10544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 37), 'self', False)
        # Obtaining the member 'get_python_type' of a type (line 346)
        get_python_type_10545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 37), self_10544, 'get_python_type')
        # Calling get_python_type(args, kwargs) (line 346)
        get_python_type_call_result_10547 = invoke(stypy.reporting.localization.Localization(__file__, 346, 37), get_python_type_10545, *[], **kwargs_10546)
        
        # Obtaining the member '__name__' of a type (line 346)
        name___10548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 37), get_python_type_call_result_10547, '__name__')
        # Getting the type of 'own_name' (line 346)
        own_name_10549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 70), 'own_name', False)
        # Getting the type of 'instance_type' (line 346)
        instance_type_10550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 80), 'instance_type', False)
        # Getting the type of 'str_mark' (line 346)
        str_mark_10551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 95), 'str_mark', False)
        # Processing the call keyword arguments (line 346)
        kwargs_10552 = {}
        str_10542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 15), 'str', '{0}{3}{1}{2}')
        # Obtaining the member 'format' of a type (line 346)
        format_10543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 15), str_10542, 'format')
        # Calling format(args, kwargs) (line 346)
        format_call_result_10553 = invoke(stypy.reporting.localization.Localization(__file__, 346, 15), format_10543, *[name___10548, own_name_10549, instance_type_10550, str_mark_10551], **kwargs_10552)
        
        # Getting the type of 'parent_str' (line 346)
        parent_str_10554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 107), 'parent_str')
        # Applying the binary operator '+' (line 346)
        result_add_10555 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 15), '+', format_call_result_10553, parent_str_10554)
        
        # Assigning a type to the variable 'stypy_return_type' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'stypy_return_type', result_add_10555)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 273)
        stypy_return_type_10556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10556)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_10556


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

        str_10557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, (-1)), 'str', '\n        Determines if a property of two objects have the same value.\n        :param property_name: Name of the property to test\n        :param obj1: First object\n        :param obj2: Second object\n        :return: bool (True if same value or both object do not have the property\n        ')
        
        # Evaluating a boolean operation
        
        # Call to hasattr(...): (line 362)
        # Processing the call arguments (line 362)
        # Getting the type of 'obj1' (line 362)
        obj1_10559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 19), 'obj1', False)
        # Getting the type of 'property_name' (line 362)
        property_name_10560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 25), 'property_name', False)
        # Processing the call keyword arguments (line 362)
        kwargs_10561 = {}
        # Getting the type of 'hasattr' (line 362)
        hasattr_10558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 11), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 362)
        hasattr_call_result_10562 = invoke(stypy.reporting.localization.Localization(__file__, 362, 11), hasattr_10558, *[obj1_10559, property_name_10560], **kwargs_10561)
        
        
        # Call to hasattr(...): (line 362)
        # Processing the call arguments (line 362)
        # Getting the type of 'obj2' (line 362)
        obj2_10564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 52), 'obj2', False)
        # Getting the type of 'property_name' (line 362)
        property_name_10565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 58), 'property_name', False)
        # Processing the call keyword arguments (line 362)
        kwargs_10566 = {}
        # Getting the type of 'hasattr' (line 362)
        hasattr_10563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 44), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 362)
        hasattr_call_result_10567 = invoke(stypy.reporting.localization.Localization(__file__, 362, 44), hasattr_10563, *[obj2_10564, property_name_10565], **kwargs_10566)
        
        # Applying the binary operator 'and' (line 362)
        result_and_keyword_10568 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 11), 'and', hasattr_call_result_10562, hasattr_call_result_10567)
        
        # Testing if the type of an if condition is none (line 362)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 362, 8), result_and_keyword_10568):
            pass
        else:
            
            # Testing the type of an if condition (line 362)
            if_condition_10569 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 362, 8), result_and_keyword_10568)
            # Assigning a type to the variable 'if_condition_10569' (line 362)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'if_condition_10569', if_condition_10569)
            # SSA begins for if statement (line 362)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            
            # Call to getattr(...): (line 363)
            # Processing the call arguments (line 363)
            # Getting the type of 'obj1' (line 363)
            obj1_10571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 27), 'obj1', False)
            # Getting the type of 'property_name' (line 363)
            property_name_10572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 33), 'property_name', False)
            # Processing the call keyword arguments (line 363)
            kwargs_10573 = {}
            # Getting the type of 'getattr' (line 363)
            getattr_10570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 19), 'getattr', False)
            # Calling getattr(args, kwargs) (line 363)
            getattr_call_result_10574 = invoke(stypy.reporting.localization.Localization(__file__, 363, 19), getattr_10570, *[obj1_10571, property_name_10572], **kwargs_10573)
            
            
            # Call to getattr(...): (line 363)
            # Processing the call arguments (line 363)
            # Getting the type of 'obj2' (line 363)
            obj2_10576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 59), 'obj2', False)
            # Getting the type of 'property_name' (line 363)
            property_name_10577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 65), 'property_name', False)
            # Processing the call keyword arguments (line 363)
            kwargs_10578 = {}
            # Getting the type of 'getattr' (line 363)
            getattr_10575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 51), 'getattr', False)
            # Calling getattr(args, kwargs) (line 363)
            getattr_call_result_10579 = invoke(stypy.reporting.localization.Localization(__file__, 363, 51), getattr_10575, *[obj2_10576, property_name_10577], **kwargs_10578)
            
            # Applying the binary operator '==' (line 363)
            result_eq_10580 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 19), '==', getattr_call_result_10574, getattr_call_result_10579)
            
            # Applying the 'not' unary operator (line 363)
            result_not__10581 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 15), 'not', result_eq_10580)
            
            # Testing if the type of an if condition is none (line 363)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 363, 12), result_not__10581):
                pass
            else:
                
                # Testing the type of an if condition (line 363)
                if_condition_10582 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 363, 12), result_not__10581)
                # Assigning a type to the variable 'if_condition_10582' (line 363)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'if_condition_10582', if_condition_10582)
                # SSA begins for if statement (line 363)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'False' (line 364)
                False_10583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 364)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 16), 'stypy_return_type', False_10583)
                # SSA join for if statement (line 363)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 362)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'True' (line 366)
        True_10584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'stypy_return_type', True_10584)
        
        # ################# End of '__equal_property_value(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__equal_property_value' in the type store
        # Getting the type of 'stypy_return_type' (line 353)
        stypy_return_type_10585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10585)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__equal_property_value'
        return stypy_return_type_10585


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

        str_10586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, (-1)), 'str', '\n        Determines if the passed argument is an UndefinedType or contains an UndefinedType\n        :param value: Any Type\n        :return: Tuple (bool, int) (contains an undefined type, the value holds n more types)\n        ')
        
        # Call to isinstance(...): (line 375)
        # Processing the call arguments (line 375)
        # Getting the type of 'value' (line 375)
        value_10588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 22), 'value', False)
        # Getting the type of 'union_type_copy' (line 375)
        union_type_copy_10589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 29), 'union_type_copy', False)
        # Obtaining the member 'UnionType' of a type (line 375)
        UnionType_10590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 29), union_type_copy_10589, 'UnionType')
        # Processing the call keyword arguments (line 375)
        kwargs_10591 = {}
        # Getting the type of 'isinstance' (line 375)
        isinstance_10587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 375)
        isinstance_call_result_10592 = invoke(stypy.reporting.localization.Localization(__file__, 375, 11), isinstance_10587, *[value_10588, UnionType_10590], **kwargs_10591)
        
        # Testing if the type of an if condition is none (line 375)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 375, 8), isinstance_call_result_10592):
            
            # Call to isinstance(...): (line 380)
            # Processing the call arguments (line 380)
            # Getting the type of 'value' (line 380)
            value_10614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 26), 'value', False)
            # Getting the type of 'undefined_type_copy' (line 380)
            undefined_type_copy_10615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 33), 'undefined_type_copy', False)
            # Obtaining the member 'UndefinedType' of a type (line 380)
            UndefinedType_10616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 33), undefined_type_copy_10615, 'UndefinedType')
            # Processing the call keyword arguments (line 380)
            kwargs_10617 = {}
            # Getting the type of 'isinstance' (line 380)
            isinstance_10613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 380)
            isinstance_call_result_10618 = invoke(stypy.reporting.localization.Localization(__file__, 380, 15), isinstance_10613, *[value_10614, UndefinedType_10616], **kwargs_10617)
            
            # Testing if the type of an if condition is none (line 380)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 380, 12), isinstance_call_result_10618):
                pass
            else:
                
                # Testing the type of an if condition (line 380)
                if_condition_10619 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 380, 12), isinstance_call_result_10618)
                # Assigning a type to the variable 'if_condition_10619' (line 380)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'if_condition_10619', if_condition_10619)
                # SSA begins for if statement (line 380)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Obtaining an instance of the builtin type 'tuple' (line 381)
                tuple_10620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 381)
                # Adding element type (line 381)
                # Getting the type of 'True' (line 381)
                True_10621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 23), 'True')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), tuple_10620, True_10621)
                # Adding element type (line 381)
                int_10622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 29), 'int')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), tuple_10620, int_10622)
                
                # Assigning a type to the variable 'stypy_return_type' (line 381)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 16), 'stypy_return_type', tuple_10620)
                # SSA join for if statement (line 380)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 375)
            if_condition_10593 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 375, 8), isinstance_call_result_10592)
            # Assigning a type to the variable 'if_condition_10593' (line 375)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'if_condition_10593', if_condition_10593)
            # SSA begins for if statement (line 375)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'value' (line 376)
            value_10594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 25), 'value')
            # Obtaining the member 'types' of a type (line 376)
            types_10595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 25), value_10594, 'types')
            # Assigning a type to the variable 'types_10595' (line 376)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'types_10595', types_10595)
            # Testing if the for loop is going to be iterated (line 376)
            # Testing the type of a for loop iterable (line 376)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 376, 12), types_10595)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 376, 12), types_10595):
                # Getting the type of the for loop variable (line 376)
                for_loop_var_10596 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 376, 12), types_10595)
                # Assigning a type to the variable 'type_' (line 376)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'type_', for_loop_var_10596)
                # SSA begins for a for statement (line 376)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to isinstance(...): (line 377)
                # Processing the call arguments (line 377)
                # Getting the type of 'type_' (line 377)
                type__10598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 30), 'type_', False)
                # Getting the type of 'undefined_type_copy' (line 377)
                undefined_type_copy_10599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 37), 'undefined_type_copy', False)
                # Obtaining the member 'UndefinedType' of a type (line 377)
                UndefinedType_10600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 37), undefined_type_copy_10599, 'UndefinedType')
                # Processing the call keyword arguments (line 377)
                kwargs_10601 = {}
                # Getting the type of 'isinstance' (line 377)
                isinstance_10597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 19), 'isinstance', False)
                # Calling isinstance(args, kwargs) (line 377)
                isinstance_call_result_10602 = invoke(stypy.reporting.localization.Localization(__file__, 377, 19), isinstance_10597, *[type__10598, UndefinedType_10600], **kwargs_10601)
                
                # Testing if the type of an if condition is none (line 377)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 377, 16), isinstance_call_result_10602):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 377)
                    if_condition_10603 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 377, 16), isinstance_call_result_10602)
                    # Assigning a type to the variable 'if_condition_10603' (line 377)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 16), 'if_condition_10603', if_condition_10603)
                    # SSA begins for if statement (line 377)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 378)
                    tuple_10604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 27), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 378)
                    # Adding element type (line 378)
                    # Getting the type of 'True' (line 378)
                    True_10605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 27), 'True')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 27), tuple_10604, True_10605)
                    # Adding element type (line 378)
                    
                    # Call to len(...): (line 378)
                    # Processing the call arguments (line 378)
                    # Getting the type of 'value' (line 378)
                    value_10607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 37), 'value', False)
                    # Obtaining the member 'types' of a type (line 378)
                    types_10608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 37), value_10607, 'types')
                    # Processing the call keyword arguments (line 378)
                    kwargs_10609 = {}
                    # Getting the type of 'len' (line 378)
                    len_10606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 33), 'len', False)
                    # Calling len(args, kwargs) (line 378)
                    len_call_result_10610 = invoke(stypy.reporting.localization.Localization(__file__, 378, 33), len_10606, *[types_10608], **kwargs_10609)
                    
                    int_10611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 52), 'int')
                    # Applying the binary operator '-' (line 378)
                    result_sub_10612 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 33), '-', len_call_result_10610, int_10611)
                    
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 27), tuple_10604, result_sub_10612)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 378)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 20), 'stypy_return_type', tuple_10604)
                    # SSA join for if statement (line 377)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA branch for the else part of an if statement (line 375)
            module_type_store.open_ssa_branch('else')
            
            # Call to isinstance(...): (line 380)
            # Processing the call arguments (line 380)
            # Getting the type of 'value' (line 380)
            value_10614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 26), 'value', False)
            # Getting the type of 'undefined_type_copy' (line 380)
            undefined_type_copy_10615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 33), 'undefined_type_copy', False)
            # Obtaining the member 'UndefinedType' of a type (line 380)
            UndefinedType_10616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 33), undefined_type_copy_10615, 'UndefinedType')
            # Processing the call keyword arguments (line 380)
            kwargs_10617 = {}
            # Getting the type of 'isinstance' (line 380)
            isinstance_10613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 380)
            isinstance_call_result_10618 = invoke(stypy.reporting.localization.Localization(__file__, 380, 15), isinstance_10613, *[value_10614, UndefinedType_10616], **kwargs_10617)
            
            # Testing if the type of an if condition is none (line 380)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 380, 12), isinstance_call_result_10618):
                pass
            else:
                
                # Testing the type of an if condition (line 380)
                if_condition_10619 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 380, 12), isinstance_call_result_10618)
                # Assigning a type to the variable 'if_condition_10619' (line 380)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'if_condition_10619', if_condition_10619)
                # SSA begins for if statement (line 380)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Obtaining an instance of the builtin type 'tuple' (line 381)
                tuple_10620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 381)
                # Adding element type (line 381)
                # Getting the type of 'True' (line 381)
                True_10621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 23), 'True')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), tuple_10620, True_10621)
                # Adding element type (line 381)
                int_10622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 29), 'int')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), tuple_10620, int_10622)
                
                # Assigning a type to the variable 'stypy_return_type' (line 381)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 16), 'stypy_return_type', tuple_10620)
                # SSA join for if statement (line 380)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 375)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Obtaining an instance of the builtin type 'tuple' (line 383)
        tuple_10623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 383)
        # Adding element type (line 383)
        # Getting the type of 'False' (line 383)
        False_10624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 15), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 15), tuple_10623, False_10624)
        # Adding element type (line 383)
        int_10625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 15), tuple_10623, int_10625)
        
        # Assigning a type to the variable 'stypy_return_type' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'stypy_return_type', tuple_10623)
        
        # ################# End of 'contains_an_undefined_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'contains_an_undefined_type' in the type store
        # Getting the type of 'stypy_return_type' (line 368)
        stypy_return_type_10626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10626)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'contains_an_undefined_type'
        return stypy_return_type_10626


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

        str_10627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, (-1)), 'str', '\n        Type proxy equality. The equality algorithm is represented as follows:\n        - Both objects have to be type inference proxies.\n        - Both objects have to hold the same type of python entity\n        - Both objects held entity name has to be the same (same class, same function, same module, ...), if the\n        proxy is not holding an instance\n        - If the hold entity do not support structural reflection, comparison will be done using the is operator\n        (reference comparison)\n        - If not, comparison by structure is performed (same amount of members, same types for these members)\n\n        :param other: The other object to compare with\n        :return: bool\n        ')
        
        # Getting the type of 'self' (line 399)
        self_10628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 11), 'self')
        # Getting the type of 'other' (line 399)
        other_10629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 19), 'other')
        # Applying the binary operator 'is' (line 399)
        result_is__10630 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 11), 'is', self_10628, other_10629)
        
        # Testing if the type of an if condition is none (line 399)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 399, 8), result_is__10630):
            pass
        else:
            
            # Testing the type of an if condition (line 399)
            if_condition_10631 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 399, 8), result_is__10630)
            # Assigning a type to the variable 'if_condition_10631' (line 399)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'if_condition_10631', if_condition_10631)
            # SSA begins for if statement (line 399)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'True' (line 400)
            True_10632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 400)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'stypy_return_type', True_10632)
            # SSA join for if statement (line 399)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        
        # Call to type(...): (line 402)
        # Processing the call arguments (line 402)
        # Getting the type of 'other' (line 402)
        other_10634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 20), 'other', False)
        # Processing the call keyword arguments (line 402)
        kwargs_10635 = {}
        # Getting the type of 'type' (line 402)
        type_10633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 15), 'type', False)
        # Calling type(args, kwargs) (line 402)
        type_call_result_10636 = invoke(stypy.reporting.localization.Localization(__file__, 402, 15), type_10633, *[other_10634], **kwargs_10635)
        
        # Getting the type of 'TypeInferenceProxy' (line 402)
        TypeInferenceProxy_10637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 30), 'TypeInferenceProxy')
        # Applying the binary operator 'is' (line 402)
        result_is__10638 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 15), 'is', type_call_result_10636, TypeInferenceProxy_10637)
        
        # Applying the 'not' unary operator (line 402)
        result_not__10639 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 11), 'not', result_is__10638)
        
        # Testing if the type of an if condition is none (line 402)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 402, 8), result_not__10639):
            pass
        else:
            
            # Testing the type of an if condition (line 402)
            if_condition_10640 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 402, 8), result_not__10639)
            # Assigning a type to the variable 'if_condition_10640' (line 402)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'if_condition_10640', if_condition_10640)
            # SSA begins for if statement (line 402)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'False' (line 403)
            False_10641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 403)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'stypy_return_type', False_10641)
            # SSA join for if statement (line 402)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Type idiom detected: calculating its left and rigth part (line 406)
        # Getting the type of 'self' (line 406)
        self_10642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 20), 'self')
        # Obtaining the member 'python_entity' of a type (line 406)
        python_entity_10643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 20), self_10642, 'python_entity')
        
        # Call to type(...): (line 406)
        # Processing the call arguments (line 406)
        # Getting the type of 'other' (line 406)
        other_10645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 48), 'other', False)
        # Obtaining the member 'python_entity' of a type (line 406)
        python_entity_10646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 48), other_10645, 'python_entity')
        # Processing the call keyword arguments (line 406)
        kwargs_10647 = {}
        # Getting the type of 'type' (line 406)
        type_10644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 43), 'type', False)
        # Calling type(args, kwargs) (line 406)
        type_call_result_10648 = invoke(stypy.reporting.localization.Localization(__file__, 406, 43), type_10644, *[python_entity_10646], **kwargs_10647)
        
        
        (may_be_10649, more_types_in_union_10650) = may_not_be_type(python_entity_10643, type_call_result_10648)

        if may_be_10649:

            if more_types_in_union_10650:
                # Runtime conditional SSA (line 406)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 406)
            self_10651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'self')
            # Obtaining the member 'python_entity' of a type (line 406)
            python_entity_10652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 8), self_10651, 'python_entity')
            # Setting the type of the member 'python_entity' of a type (line 406)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 8), self_10651, 'python_entity', remove_type_from_union(python_entity_10652, type_call_result_10648))
            # Getting the type of 'False' (line 407)
            False_10653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 407)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'stypy_return_type', False_10653)

            if more_types_in_union_10650:
                # SSA join for if statement (line 406)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'self' (line 410)
        self_10654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'self')
        # Obtaining the member 'instance' of a type (line 410)
        instance_10655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 12), self_10654, 'instance')
        # Getting the type of 'None' (line 410)
        None_10656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 29), 'None')
        # Applying the binary operator 'is' (line 410)
        result_is__10657 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 12), 'is', instance_10655, None_10656)
        
        
        # Getting the type of 'other' (line 410)
        other_10658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 38), 'other')
        # Obtaining the member 'instance' of a type (line 410)
        instance_10659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 38), other_10658, 'instance')
        # Getting the type of 'None' (line 410)
        None_10660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 56), 'None')
        # Applying the binary operator 'is' (line 410)
        result_is__10661 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 38), 'is', instance_10659, None_10660)
        
        # Applying the binary operator '^' (line 410)
        result_xor_10662 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 11), '^', result_is__10657, result_is__10661)
        
        # Testing if the type of an if condition is none (line 410)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 410, 8), result_xor_10662):
            pass
        else:
            
            # Testing the type of an if condition (line 410)
            if_condition_10663 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 410, 8), result_xor_10662)
            # Assigning a type to the variable 'if_condition_10663' (line 410)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'if_condition_10663', if_condition_10663)
            # SSA begins for if statement (line 410)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'False' (line 411)
            False_10664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 411)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 12), 'stypy_return_type', False_10664)
            # SSA join for if statement (line 410)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 414):
        
        # Assigning a Call to a Name (line 414):
        
        # Call to is_type_instance(...): (line 414)
        # Processing the call keyword arguments (line 414)
        kwargs_10667 = {}
        # Getting the type of 'self' (line 414)
        self_10665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 28), 'self', False)
        # Obtaining the member 'is_type_instance' of a type (line 414)
        is_type_instance_10666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 28), self_10665, 'is_type_instance')
        # Calling is_type_instance(args, kwargs) (line 414)
        is_type_instance_call_result_10668 = invoke(stypy.reporting.localization.Localization(__file__, 414, 28), is_type_instance_10666, *[], **kwargs_10667)
        
        # Assigning a type to the variable 'self_instantiated' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'self_instantiated', is_type_instance_call_result_10668)
        
        # Assigning a Call to a Name (line 415):
        
        # Assigning a Call to a Name (line 415):
        
        # Call to is_type_instance(...): (line 415)
        # Processing the call keyword arguments (line 415)
        kwargs_10671 = {}
        # Getting the type of 'other' (line 415)
        other_10669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 29), 'other', False)
        # Obtaining the member 'is_type_instance' of a type (line 415)
        is_type_instance_10670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 29), other_10669, 'is_type_instance')
        # Calling is_type_instance(args, kwargs) (line 415)
        is_type_instance_call_result_10672 = invoke(stypy.reporting.localization.Localization(__file__, 415, 29), is_type_instance_10670, *[], **kwargs_10671)
        
        # Assigning a type to the variable 'other_instantiated' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'other_instantiated', is_type_instance_call_result_10672)
        
        # Getting the type of 'self_instantiated' (line 416)
        self_instantiated_10673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 11), 'self_instantiated')
        # Getting the type of 'other_instantiated' (line 416)
        other_instantiated_10674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 32), 'other_instantiated')
        # Applying the binary operator '!=' (line 416)
        result_ne_10675 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 11), '!=', self_instantiated_10673, other_instantiated_10674)
        
        # Testing if the type of an if condition is none (line 416)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 416, 8), result_ne_10675):
            pass
        else:
            
            # Testing the type of an if condition (line 416)
            if_condition_10676 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 416, 8), result_ne_10675)
            # Assigning a type to the variable 'if_condition_10676' (line 416)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'if_condition_10676', if_condition_10676)
            # SSA begins for if statement (line 416)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'False' (line 417)
            False_10677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 417)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'stypy_return_type', False_10677)
            # SSA join for if statement (line 416)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Attribute to a Name (line 419):
        
        # Assigning a Attribute to a Name (line 419):
        # Getting the type of 'self' (line 419)
        self_10678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 22), 'self')
        # Obtaining the member 'python_entity' of a type (line 419)
        python_entity_10679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 22), self_10678, 'python_entity')
        # Assigning a type to the variable 'self_entity' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'self_entity', python_entity_10679)
        
        # Assigning a Attribute to a Name (line 420):
        
        # Assigning a Attribute to a Name (line 420):
        # Getting the type of 'other' (line 420)
        other_10680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 23), 'other')
        # Obtaining the member 'python_entity' of a type (line 420)
        python_entity_10681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 23), other_10680, 'python_entity')
        # Assigning a type to the variable 'other_entity' (line 420)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'other_entity', python_entity_10681)
        
        # Getting the type of 'Type' (line 423)
        Type_10682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 25), 'Type')
        # Obtaining the member 'special_properties_for_equality' of a type (line 423)
        special_properties_for_equality_10683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 25), Type_10682, 'special_properties_for_equality')
        # Assigning a type to the variable 'special_properties_for_equality_10683' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'special_properties_for_equality_10683', special_properties_for_equality_10683)
        # Testing if the for loop is going to be iterated (line 423)
        # Testing the type of a for loop iterable (line 423)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 423, 8), special_properties_for_equality_10683)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 423, 8), special_properties_for_equality_10683):
            # Getting the type of the for loop variable (line 423)
            for_loop_var_10684 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 423, 8), special_properties_for_equality_10683)
            # Assigning a type to the variable 'prop_name' (line 423)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'prop_name', for_loop_var_10684)
            # SSA begins for a for statement (line 423)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to __equal_property_value(...): (line 424)
            # Processing the call arguments (line 424)
            # Getting the type of 'prop_name' (line 424)
            prop_name_10687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 47), 'prop_name', False)
            # Getting the type of 'self_entity' (line 424)
            self_entity_10688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 58), 'self_entity', False)
            # Getting the type of 'other_entity' (line 424)
            other_entity_10689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 71), 'other_entity', False)
            # Processing the call keyword arguments (line 424)
            kwargs_10690 = {}
            # Getting the type of 'self' (line 424)
            self_10685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 19), 'self', False)
            # Obtaining the member '__equal_property_value' of a type (line 424)
            equal_property_value_10686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 19), self_10685, '__equal_property_value')
            # Calling __equal_property_value(args, kwargs) (line 424)
            equal_property_value_call_result_10691 = invoke(stypy.reporting.localization.Localization(__file__, 424, 19), equal_property_value_10686, *[prop_name_10687, self_entity_10688, other_entity_10689], **kwargs_10690)
            
            # Applying the 'not' unary operator (line 424)
            result_not__10692 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 15), 'not', equal_property_value_call_result_10691)
            
            # Testing if the type of an if condition is none (line 424)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 424, 12), result_not__10692):
                pass
            else:
                
                # Testing the type of an if condition (line 424)
                if_condition_10693 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 424, 12), result_not__10692)
                # Assigning a type to the variable 'if_condition_10693' (line 424)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'if_condition_10693', if_condition_10693)
                # SSA begins for if statement (line 424)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'False' (line 425)
                False_10694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 425)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 16), 'stypy_return_type', False_10694)
                # SSA join for if statement (line 424)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to __equal_property_value(...): (line 428)
        # Processing the call arguments (line 428)
        # Getting the type of 'TypeInferenceProxy' (line 428)
        TypeInferenceProxy_10697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 43), 'TypeInferenceProxy', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 428)
        contained_elements_property_name_10698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 43), TypeInferenceProxy_10697, 'contained_elements_property_name')
        # Getting the type of 'self_entity' (line 428)
        self_entity_10699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 96), 'self_entity', False)
        # Getting the type of 'other_entity' (line 429)
        other_entity_10700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 43), 'other_entity', False)
        # Processing the call keyword arguments (line 428)
        kwargs_10701 = {}
        # Getting the type of 'self' (line 428)
        self_10695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 15), 'self', False)
        # Obtaining the member '__equal_property_value' of a type (line 428)
        equal_property_value_10696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 15), self_10695, '__equal_property_value')
        # Calling __equal_property_value(args, kwargs) (line 428)
        equal_property_value_call_result_10702 = invoke(stypy.reporting.localization.Localization(__file__, 428, 15), equal_property_value_10696, *[contained_elements_property_name_10698, self_entity_10699, other_entity_10700], **kwargs_10701)
        
        # Applying the 'not' unary operator (line 428)
        result_not__10703 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 11), 'not', equal_property_value_call_result_10702)
        
        # Testing if the type of an if condition is none (line 428)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 428, 8), result_not__10703):
            pass
        else:
            
            # Testing the type of an if condition (line 428)
            if_condition_10704 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 428, 8), result_not__10703)
            # Assigning a type to the variable 'if_condition_10704' (line 428)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'if_condition_10704', if_condition_10704)
            # SSA begins for if statement (line 428)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'False' (line 430)
            False_10705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 430)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'stypy_return_type', False_10705)
            # SSA join for if statement (line 428)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Type idiom detected: calculating its left and rigth part (line 433)
        # Getting the type of 'self' (line 433)
        self_10706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 11), 'self')
        # Obtaining the member 'instance' of a type (line 433)
        instance_10707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 11), self_10706, 'instance')
        # Getting the type of 'None' (line 433)
        None_10708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 28), 'None')
        
        (may_be_10709, more_types_in_union_10710) = may_be_none(instance_10707, None_10708)

        if may_be_10709:

            if more_types_in_union_10710:
                # Runtime conditional SSA (line 433)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Evaluating a boolean operation
            
            # Call to supports_structural_reflection(...): (line 437)
            # Processing the call keyword arguments (line 437)
            kwargs_10713 = {}
            # Getting the type of 'self' (line 437)
            self_10711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 15), 'self', False)
            # Obtaining the member 'supports_structural_reflection' of a type (line 437)
            supports_structural_reflection_10712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 15), self_10711, 'supports_structural_reflection')
            # Calling supports_structural_reflection(args, kwargs) (line 437)
            supports_structural_reflection_call_result_10714 = invoke(stypy.reporting.localization.Localization(__file__, 437, 15), supports_structural_reflection_10712, *[], **kwargs_10713)
            
            
            # Call to supports_structural_reflection(...): (line 437)
            # Processing the call keyword arguments (line 437)
            kwargs_10717 = {}
            # Getting the type of 'other' (line 437)
            other_10715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 57), 'other', False)
            # Obtaining the member 'supports_structural_reflection' of a type (line 437)
            supports_structural_reflection_10716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 57), other_10715, 'supports_structural_reflection')
            # Calling supports_structural_reflection(args, kwargs) (line 437)
            supports_structural_reflection_call_result_10718 = invoke(stypy.reporting.localization.Localization(__file__, 437, 57), supports_structural_reflection_10716, *[], **kwargs_10717)
            
            # Applying the binary operator 'and' (line 437)
            result_and_keyword_10719 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 15), 'and', supports_structural_reflection_call_result_10714, supports_structural_reflection_call_result_10718)
            
            # Testing if the type of an if condition is none (line 437)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 437, 12), result_and_keyword_10719):
                
                
                # Call to type(...): (line 442)
                # Processing the call arguments (line 442)
                # Getting the type of 'self_entity' (line 442)
                self_entity_10729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 28), 'self_entity', False)
                # Processing the call keyword arguments (line 442)
                kwargs_10730 = {}
                # Getting the type of 'type' (line 442)
                type_10728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 23), 'type', False)
                # Calling type(args, kwargs) (line 442)
                type_call_result_10731 = invoke(stypy.reporting.localization.Localization(__file__, 442, 23), type_10728, *[self_entity_10729], **kwargs_10730)
                
                
                # Call to type(...): (line 442)
                # Processing the call arguments (line 442)
                # Getting the type of 'other_entity' (line 442)
                other_entity_10733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 49), 'other_entity', False)
                # Processing the call keyword arguments (line 442)
                kwargs_10734 = {}
                # Getting the type of 'type' (line 442)
                type_10732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 44), 'type', False)
                # Calling type(args, kwargs) (line 442)
                type_call_result_10735 = invoke(stypy.reporting.localization.Localization(__file__, 442, 44), type_10732, *[other_entity_10733], **kwargs_10734)
                
                # Applying the binary operator 'is' (line 442)
                result_is__10736 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 23), 'is', type_call_result_10731, type_call_result_10735)
                
                # Assigning a type to the variable 'stypy_return_type' (line 442)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 16), 'stypy_return_type', result_is__10736)
            else:
                
                # Testing the type of an if condition (line 437)
                if_condition_10720 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 437, 12), result_and_keyword_10719)
                # Assigning a type to the variable 'if_condition_10720' (line 437)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'if_condition_10720', if_condition_10720)
                # SSA begins for if statement (line 437)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to structural_equivalence(...): (line 439)
                # Processing the call arguments (line 439)
                # Getting the type of 'self_entity' (line 439)
                self_entity_10723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 68), 'self_entity', False)
                # Getting the type of 'other_entity' (line 439)
                other_entity_10724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 81), 'other_entity', False)
                # Getting the type of 'True' (line 439)
                True_10725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 95), 'True', False)
                # Processing the call keyword arguments (line 439)
                kwargs_10726 = {}
                # Getting the type of 'type_equivalence_copy' (line 439)
                type_equivalence_copy_10721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 23), 'type_equivalence_copy', False)
                # Obtaining the member 'structural_equivalence' of a type (line 439)
                structural_equivalence_10722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 23), type_equivalence_copy_10721, 'structural_equivalence')
                # Calling structural_equivalence(args, kwargs) (line 439)
                structural_equivalence_call_result_10727 = invoke(stypy.reporting.localization.Localization(__file__, 439, 23), structural_equivalence_10722, *[self_entity_10723, other_entity_10724, True_10725], **kwargs_10726)
                
                # Assigning a type to the variable 'stypy_return_type' (line 439)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 16), 'stypy_return_type', structural_equivalence_call_result_10727)
                # SSA branch for the else part of an if statement (line 437)
                module_type_store.open_ssa_branch('else')
                
                
                # Call to type(...): (line 442)
                # Processing the call arguments (line 442)
                # Getting the type of 'self_entity' (line 442)
                self_entity_10729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 28), 'self_entity', False)
                # Processing the call keyword arguments (line 442)
                kwargs_10730 = {}
                # Getting the type of 'type' (line 442)
                type_10728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 23), 'type', False)
                # Calling type(args, kwargs) (line 442)
                type_call_result_10731 = invoke(stypy.reporting.localization.Localization(__file__, 442, 23), type_10728, *[self_entity_10729], **kwargs_10730)
                
                
                # Call to type(...): (line 442)
                # Processing the call arguments (line 442)
                # Getting the type of 'other_entity' (line 442)
                other_entity_10733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 49), 'other_entity', False)
                # Processing the call keyword arguments (line 442)
                kwargs_10734 = {}
                # Getting the type of 'type' (line 442)
                type_10732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 44), 'type', False)
                # Calling type(args, kwargs) (line 442)
                type_call_result_10735 = invoke(stypy.reporting.localization.Localization(__file__, 442, 44), type_10732, *[other_entity_10733], **kwargs_10734)
                
                # Applying the binary operator 'is' (line 442)
                result_is__10736 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 23), 'is', type_call_result_10731, type_call_result_10735)
                
                # Assigning a type to the variable 'stypy_return_type' (line 442)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 16), 'stypy_return_type', result_is__10736)
                # SSA join for if statement (line 437)
                module_type_store = module_type_store.join_ssa_context()
                


            if more_types_in_union_10710:
                # Runtime conditional SSA for else branch (line 433)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_10709) or more_types_in_union_10710):
            
            # Evaluating a boolean operation
            
            # Call to supports_structural_reflection(...): (line 447)
            # Processing the call keyword arguments (line 447)
            kwargs_10739 = {}
            # Getting the type of 'self' (line 447)
            self_10737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 15), 'self', False)
            # Obtaining the member 'supports_structural_reflection' of a type (line 447)
            supports_structural_reflection_10738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 15), self_10737, 'supports_structural_reflection')
            # Calling supports_structural_reflection(args, kwargs) (line 447)
            supports_structural_reflection_call_result_10740 = invoke(stypy.reporting.localization.Localization(__file__, 447, 15), supports_structural_reflection_10738, *[], **kwargs_10739)
            
            
            # Call to supports_structural_reflection(...): (line 447)
            # Processing the call keyword arguments (line 447)
            kwargs_10743 = {}
            # Getting the type of 'other' (line 447)
            other_10741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 57), 'other', False)
            # Obtaining the member 'supports_structural_reflection' of a type (line 447)
            supports_structural_reflection_10742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 57), other_10741, 'supports_structural_reflection')
            # Calling supports_structural_reflection(args, kwargs) (line 447)
            supports_structural_reflection_call_result_10744 = invoke(stypy.reporting.localization.Localization(__file__, 447, 57), supports_structural_reflection_10742, *[], **kwargs_10743)
            
            # Applying the binary operator 'and' (line 447)
            result_and_keyword_10745 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 15), 'and', supports_structural_reflection_call_result_10740, supports_structural_reflection_call_result_10744)
            
            # Testing if the type of an if condition is none (line 447)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 447, 12), result_and_keyword_10745):
                
                # Assigning a Compare to a Name (line 460):
                
                # Assigning a Compare to a Name (line 460):
                
                
                # Call to type(...): (line 460)
                # Processing the call arguments (line 460)
                # Getting the type of 'self' (line 460)
                self_10770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 34), 'self', False)
                # Obtaining the member 'python_entity' of a type (line 460)
                python_entity_10771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 34), self_10770, 'python_entity')
                # Processing the call keyword arguments (line 460)
                kwargs_10772 = {}
                # Getting the type of 'type' (line 460)
                type_10769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 29), 'type', False)
                # Calling type(args, kwargs) (line 460)
                type_call_result_10773 = invoke(stypy.reporting.localization.Localization(__file__, 460, 29), type_10769, *[python_entity_10771], **kwargs_10772)
                
                
                # Call to type(...): (line 460)
                # Processing the call arguments (line 460)
                # Getting the type of 'other' (line 460)
                other_10775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 62), 'other', False)
                # Obtaining the member 'python_entity' of a type (line 460)
                python_entity_10776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 62), other_10775, 'python_entity')
                # Processing the call keyword arguments (line 460)
                kwargs_10777 = {}
                # Getting the type of 'type' (line 460)
                type_10774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 57), 'type', False)
                # Calling type(args, kwargs) (line 460)
                type_call_result_10778 = invoke(stypy.reporting.localization.Localization(__file__, 460, 57), type_10774, *[python_entity_10776], **kwargs_10777)
                
                # Applying the binary operator 'is' (line 460)
                result_is__10779 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 29), 'is', type_call_result_10773, type_call_result_10778)
                
                # Assigning a type to the variable 'equivalent' (line 460)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 16), 'equivalent', result_is__10779)
                
                # Getting the type of 'equivalent' (line 461)
                equivalent_10780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 23), 'equivalent')
                # Applying the 'not' unary operator (line 461)
                result_not__10781 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 19), 'not', equivalent_10780)
                
                # Testing if the type of an if condition is none (line 461)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 461, 16), result_not__10781):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 461)
                    if_condition_10782 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 461, 16), result_not__10781)
                    # Assigning a type to the variable 'if_condition_10782' (line 461)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 16), 'if_condition_10782', if_condition_10782)
                    # SSA begins for if statement (line 461)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'False' (line 462)
                    False_10783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 27), 'False')
                    # Assigning a type to the variable 'stypy_return_type' (line 462)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 20), 'stypy_return_type', False_10783)
                    # SSA join for if statement (line 461)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Attribute to a Name (line 465):
                
                # Assigning a Attribute to a Name (line 465):
                # Getting the type of 'self' (line 465)
                self_10784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 30), 'self')
                # Obtaining the member 'instance' of a type (line 465)
                instance_10785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 30), self_10784, 'instance')
                # Assigning a type to the variable 'self_entity' (line 465)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 16), 'self_entity', instance_10785)
                
                # Assigning a Attribute to a Name (line 466):
                
                # Assigning a Attribute to a Name (line 466):
                # Getting the type of 'other' (line 466)
                other_10786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 31), 'other')
                # Obtaining the member 'instance' of a type (line 466)
                instance_10787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 31), other_10786, 'instance')
                # Assigning a type to the variable 'other_entity' (line 466)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 16), 'other_entity', instance_10787)
                
                # Getting the type of 'self_entity' (line 467)
                self_entity_10788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 23), 'self_entity')
                # Getting the type of 'other_entity' (line 467)
                other_entity_10789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 38), 'other_entity')
                # Applying the binary operator 'is' (line 467)
                result_is__10790 = python_operator(stypy.reporting.localization.Localization(__file__, 467, 23), 'is', self_entity_10788, other_entity_10789)
                
                # Assigning a type to the variable 'stypy_return_type' (line 467)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 16), 'stypy_return_type', result_is__10790)
            else:
                
                # Testing the type of an if condition (line 447)
                if_condition_10746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 447, 12), result_and_keyword_10745)
                # Assigning a type to the variable 'if_condition_10746' (line 447)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'if_condition_10746', if_condition_10746)
                # SSA begins for if statement (line 447)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 449):
                
                # Assigning a Call to a Name (line 449):
                
                # Call to structural_equivalence(...): (line 449)
                # Processing the call arguments (line 449)
                # Getting the type of 'self_entity' (line 449)
                self_entity_10749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 74), 'self_entity', False)
                # Getting the type of 'other_entity' (line 449)
                other_entity_10750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 87), 'other_entity', False)
                # Getting the type of 'True' (line 449)
                True_10751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 101), 'True', False)
                # Processing the call keyword arguments (line 449)
                kwargs_10752 = {}
                # Getting the type of 'type_equivalence_copy' (line 449)
                type_equivalence_copy_10747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 29), 'type_equivalence_copy', False)
                # Obtaining the member 'structural_equivalence' of a type (line 449)
                structural_equivalence_10748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 29), type_equivalence_copy_10747, 'structural_equivalence')
                # Calling structural_equivalence(args, kwargs) (line 449)
                structural_equivalence_call_result_10753 = invoke(stypy.reporting.localization.Localization(__file__, 449, 29), structural_equivalence_10748, *[self_entity_10749, other_entity_10750, True_10751], **kwargs_10752)
                
                # Assigning a type to the variable 'equivalent' (line 449)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 16), 'equivalent', structural_equivalence_call_result_10753)
                
                # Getting the type of 'equivalent' (line 451)
                equivalent_10754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 23), 'equivalent')
                # Applying the 'not' unary operator (line 451)
                result_not__10755 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 19), 'not', equivalent_10754)
                
                # Testing if the type of an if condition is none (line 451)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 451, 16), result_not__10755):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 451)
                    if_condition_10756 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 451, 16), result_not__10755)
                    # Assigning a type to the variable 'if_condition_10756' (line 451)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 16), 'if_condition_10756', if_condition_10756)
                    # SSA begins for if statement (line 451)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'False' (line 452)
                    False_10757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 27), 'False')
                    # Assigning a type to the variable 'stypy_return_type' (line 452)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 20), 'stypy_return_type', False_10757)
                    # SSA join for if statement (line 451)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Attribute to a Name (line 455):
                
                # Assigning a Attribute to a Name (line 455):
                # Getting the type of 'self' (line 455)
                self_10758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 30), 'self')
                # Obtaining the member 'instance' of a type (line 455)
                instance_10759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 30), self_10758, 'instance')
                # Assigning a type to the variable 'self_entity' (line 455)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 16), 'self_entity', instance_10759)
                
                # Assigning a Attribute to a Name (line 456):
                
                # Assigning a Attribute to a Name (line 456):
                # Getting the type of 'other' (line 456)
                other_10760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 31), 'other')
                # Obtaining the member 'instance' of a type (line 456)
                instance_10761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 31), other_10760, 'instance')
                # Assigning a type to the variable 'other_entity' (line 456)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'other_entity', instance_10761)
                
                # Call to structural_equivalence(...): (line 457)
                # Processing the call arguments (line 457)
                # Getting the type of 'self_entity' (line 457)
                self_entity_10764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 68), 'self_entity', False)
                # Getting the type of 'other_entity' (line 457)
                other_entity_10765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 81), 'other_entity', False)
                # Getting the type of 'False' (line 457)
                False_10766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 95), 'False', False)
                # Processing the call keyword arguments (line 457)
                kwargs_10767 = {}
                # Getting the type of 'type_equivalence_copy' (line 457)
                type_equivalence_copy_10762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 23), 'type_equivalence_copy', False)
                # Obtaining the member 'structural_equivalence' of a type (line 457)
                structural_equivalence_10763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 23), type_equivalence_copy_10762, 'structural_equivalence')
                # Calling structural_equivalence(args, kwargs) (line 457)
                structural_equivalence_call_result_10768 = invoke(stypy.reporting.localization.Localization(__file__, 457, 23), structural_equivalence_10763, *[self_entity_10764, other_entity_10765, False_10766], **kwargs_10767)
                
                # Assigning a type to the variable 'stypy_return_type' (line 457)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 16), 'stypy_return_type', structural_equivalence_call_result_10768)
                # SSA branch for the else part of an if statement (line 447)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Compare to a Name (line 460):
                
                # Assigning a Compare to a Name (line 460):
                
                
                # Call to type(...): (line 460)
                # Processing the call arguments (line 460)
                # Getting the type of 'self' (line 460)
                self_10770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 34), 'self', False)
                # Obtaining the member 'python_entity' of a type (line 460)
                python_entity_10771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 34), self_10770, 'python_entity')
                # Processing the call keyword arguments (line 460)
                kwargs_10772 = {}
                # Getting the type of 'type' (line 460)
                type_10769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 29), 'type', False)
                # Calling type(args, kwargs) (line 460)
                type_call_result_10773 = invoke(stypy.reporting.localization.Localization(__file__, 460, 29), type_10769, *[python_entity_10771], **kwargs_10772)
                
                
                # Call to type(...): (line 460)
                # Processing the call arguments (line 460)
                # Getting the type of 'other' (line 460)
                other_10775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 62), 'other', False)
                # Obtaining the member 'python_entity' of a type (line 460)
                python_entity_10776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 62), other_10775, 'python_entity')
                # Processing the call keyword arguments (line 460)
                kwargs_10777 = {}
                # Getting the type of 'type' (line 460)
                type_10774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 57), 'type', False)
                # Calling type(args, kwargs) (line 460)
                type_call_result_10778 = invoke(stypy.reporting.localization.Localization(__file__, 460, 57), type_10774, *[python_entity_10776], **kwargs_10777)
                
                # Applying the binary operator 'is' (line 460)
                result_is__10779 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 29), 'is', type_call_result_10773, type_call_result_10778)
                
                # Assigning a type to the variable 'equivalent' (line 460)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 16), 'equivalent', result_is__10779)
                
                # Getting the type of 'equivalent' (line 461)
                equivalent_10780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 23), 'equivalent')
                # Applying the 'not' unary operator (line 461)
                result_not__10781 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 19), 'not', equivalent_10780)
                
                # Testing if the type of an if condition is none (line 461)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 461, 16), result_not__10781):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 461)
                    if_condition_10782 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 461, 16), result_not__10781)
                    # Assigning a type to the variable 'if_condition_10782' (line 461)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 16), 'if_condition_10782', if_condition_10782)
                    # SSA begins for if statement (line 461)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'False' (line 462)
                    False_10783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 27), 'False')
                    # Assigning a type to the variable 'stypy_return_type' (line 462)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 20), 'stypy_return_type', False_10783)
                    # SSA join for if statement (line 461)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Attribute to a Name (line 465):
                
                # Assigning a Attribute to a Name (line 465):
                # Getting the type of 'self' (line 465)
                self_10784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 30), 'self')
                # Obtaining the member 'instance' of a type (line 465)
                instance_10785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 30), self_10784, 'instance')
                # Assigning a type to the variable 'self_entity' (line 465)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 16), 'self_entity', instance_10785)
                
                # Assigning a Attribute to a Name (line 466):
                
                # Assigning a Attribute to a Name (line 466):
                # Getting the type of 'other' (line 466)
                other_10786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 31), 'other')
                # Obtaining the member 'instance' of a type (line 466)
                instance_10787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 31), other_10786, 'instance')
                # Assigning a type to the variable 'other_entity' (line 466)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 16), 'other_entity', instance_10787)
                
                # Getting the type of 'self_entity' (line 467)
                self_entity_10788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 23), 'self_entity')
                # Getting the type of 'other_entity' (line 467)
                other_entity_10789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 38), 'other_entity')
                # Applying the binary operator 'is' (line 467)
                result_is__10790 = python_operator(stypy.reporting.localization.Localization(__file__, 467, 23), 'is', self_entity_10788, other_entity_10789)
                
                # Assigning a type to the variable 'stypy_return_type' (line 467)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 16), 'stypy_return_type', result_is__10790)
                # SSA join for if statement (line 447)
                module_type_store = module_type_store.join_ssa_context()
                


            if (may_be_10709 and more_types_in_union_10710):
                # SSA join for if statement (line 433)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 385)
        stypy_return_type_10791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10791)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_10791


    @staticmethod
    @norecursion
    def instance(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 472)
        None_10792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 37), 'None')
        # Getting the type of 'None' (line 472)
        None_10793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 50), 'None')
        # Getting the type of 'None' (line 472)
        None_10794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 65), 'None')
        # Getting the type of 'undefined_type_copy' (line 472)
        undefined_type_copy_10795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 77), 'undefined_type_copy')
        # Obtaining the member 'UndefinedType' of a type (line 472)
        UndefinedType_10796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 77), undefined_type_copy_10795, 'UndefinedType')
        defaults = [None_10792, None_10793, None_10794, UndefinedType_10796]
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

        str_10797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, (-1)), 'str', '\n        Creates a new Type inference proxy for the passed python entity (function, module, class, ...). This is the\n        preferred way to create proxy instances, as this method implement a memoization optimization.\n\n        :param python_entity: Represented python entity.\n        :param name: Name of the represented Python entity. If nothing is provided, the Python entity __name__ property\n        value is used instead. Instances have an special name indicating that this entity holds an instance of a class.\n        :param parent: Parent proxy object. If nothing is provided, the parent proxy is autocalculated, if possible.\n        :param instance: Instance of the represented class. If this proxy holds a class, it is possible that instead\n        of representing the class is representing a particular class instance. This is important to properly model\n        instance intercession, as altering the structure of single instances is possible.\n        ')
        
        # Call to isinstance(...): (line 493)
        # Processing the call arguments (line 493)
        # Getting the type of 'python_entity' (line 493)
        python_entity_10799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 22), 'python_entity', False)
        # Getting the type of 'Type' (line 493)
        Type_10800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 37), 'Type', False)
        # Processing the call keyword arguments (line 493)
        kwargs_10801 = {}
        # Getting the type of 'isinstance' (line 493)
        isinstance_10798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 493)
        isinstance_call_result_10802 = invoke(stypy.reporting.localization.Localization(__file__, 493, 11), isinstance_10798, *[python_entity_10799, Type_10800], **kwargs_10801)
        
        # Testing if the type of an if condition is none (line 493)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 493, 8), isinstance_call_result_10802):
            pass
        else:
            
            # Testing the type of an if condition (line 493)
            if_condition_10803 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 493, 8), isinstance_call_result_10802)
            # Assigning a type to the variable 'if_condition_10803' (line 493)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'if_condition_10803', if_condition_10803)
            # SSA begins for if statement (line 493)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'python_entity' (line 494)
            python_entity_10804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 19), 'python_entity')
            # Assigning a type to the variable 'stypy_return_type' (line 494)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'stypy_return_type', python_entity_10804)
            # SSA join for if statement (line 493)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to TypeInferenceProxy(...): (line 496)
        # Processing the call arguments (line 496)
        # Getting the type of 'python_entity' (line 496)
        python_entity_10806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 34), 'python_entity', False)
        # Getting the type of 'name' (line 496)
        name_10807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 49), 'name', False)
        # Getting the type of 'parent' (line 496)
        parent_10808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 55), 'parent', False)
        # Getting the type of 'instance' (line 496)
        instance_10809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 63), 'instance', False)
        # Getting the type of 'value' (line 496)
        value_10810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 73), 'value', False)
        # Processing the call keyword arguments (line 496)
        kwargs_10811 = {}
        # Getting the type of 'TypeInferenceProxy' (line 496)
        TypeInferenceProxy_10805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 15), 'TypeInferenceProxy', False)
        # Calling TypeInferenceProxy(args, kwargs) (line 496)
        TypeInferenceProxy_call_result_10812 = invoke(stypy.reporting.localization.Localization(__file__, 496, 15), TypeInferenceProxy_10805, *[python_entity_10806, name_10807, parent_10808, instance_10809, value_10810], **kwargs_10811)
        
        # Assigning a type to the variable 'stypy_return_type' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'stypy_return_type', TypeInferenceProxy_call_result_10812)
        
        # ################# End of 'instance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'instance' in the type store
        # Getting the type of 'stypy_return_type' (line 471)
        stypy_return_type_10813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10813)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'instance'
        return stypy_return_type_10813


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

        str_10814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, (-1)), 'str', '\n        Returns the Python entity (function, method, class, object, module...) represented by this Type.\n        :return: A Python entity\n        ')
        # Getting the type of 'self' (line 505)
        self_10815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 15), 'self')
        # Obtaining the member 'python_entity' of a type (line 505)
        python_entity_10816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 15), self_10815, 'python_entity')
        # Assigning a type to the variable 'stypy_return_type' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'stypy_return_type', python_entity_10816)
        
        # ################# End of 'get_python_entity(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_python_entity' in the type store
        # Getting the type of 'stypy_return_type' (line 500)
        stypy_return_type_10817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10817)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_python_entity'
        return stypy_return_type_10817


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

        str_10818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, (-1)), 'str', '\n        Get the python type of the hold entity. This is equivalent to call the type(hold_python_entity). If a user-\n        defined class instance is hold, a types.InstanceType is returned (as Python does)\n        :return: A python type\n        ')
        
        
        # Call to isclass(...): (line 513)
        # Processing the call arguments (line 513)
        # Getting the type of 'self' (line 513)
        self_10821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 31), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 513)
        python_entity_10822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 31), self_10821, 'python_entity')
        # Processing the call keyword arguments (line 513)
        kwargs_10823 = {}
        # Getting the type of 'inspect' (line 513)
        inspect_10819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 15), 'inspect', False)
        # Obtaining the member 'isclass' of a type (line 513)
        isclass_10820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 15), inspect_10819, 'isclass')
        # Calling isclass(args, kwargs) (line 513)
        isclass_call_result_10824 = invoke(stypy.reporting.localization.Localization(__file__, 513, 15), isclass_10820, *[python_entity_10822], **kwargs_10823)
        
        # Applying the 'not' unary operator (line 513)
        result_not__10825 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 11), 'not', isclass_call_result_10824)
        
        # Testing if the type of an if condition is none (line 513)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 513, 8), result_not__10825):
            pass
        else:
            
            # Testing the type of an if condition (line 513)
            if_condition_10826 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 513, 8), result_not__10825)
            # Assigning a type to the variable 'if_condition_10826' (line 513)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'if_condition_10826', if_condition_10826)
            # SSA begins for if statement (line 513)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to type(...): (line 514)
            # Processing the call arguments (line 514)
            # Getting the type of 'self' (line 514)
            self_10828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 24), 'self', False)
            # Obtaining the member 'python_entity' of a type (line 514)
            python_entity_10829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 24), self_10828, 'python_entity')
            # Processing the call keyword arguments (line 514)
            kwargs_10830 = {}
            # Getting the type of 'type' (line 514)
            type_10827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 19), 'type', False)
            # Calling type(args, kwargs) (line 514)
            type_call_result_10831 = invoke(stypy.reporting.localization.Localization(__file__, 514, 19), type_10827, *[python_entity_10829], **kwargs_10830)
            
            # Assigning a type to the variable 'stypy_return_type' (line 514)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 12), 'stypy_return_type', type_call_result_10831)
            # SSA join for if statement (line 513)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Evaluating a boolean operation
        
        # Call to is_user_defined_class(...): (line 516)
        # Processing the call arguments (line 516)
        # Getting the type of 'self' (line 516)
        self_10834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 70), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 516)
        python_entity_10835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 70), self_10834, 'python_entity')
        # Processing the call keyword arguments (line 516)
        kwargs_10836 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 516)
        type_inference_proxy_management_copy_10832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 11), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'is_user_defined_class' of a type (line 516)
        is_user_defined_class_10833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 11), type_inference_proxy_management_copy_10832, 'is_user_defined_class')
        # Calling is_user_defined_class(args, kwargs) (line 516)
        is_user_defined_class_call_result_10837 = invoke(stypy.reporting.localization.Localization(__file__, 516, 11), is_user_defined_class_10833, *[python_entity_10835], **kwargs_10836)
        
        
        # Getting the type of 'self' (line 516)
        self_10838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 94), 'self')
        # Obtaining the member 'instance' of a type (line 516)
        instance_10839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 94), self_10838, 'instance')
        # Getting the type of 'None' (line 516)
        None_10840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 115), 'None')
        # Applying the binary operator 'isnot' (line 516)
        result_is_not_10841 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 94), 'isnot', instance_10839, None_10840)
        
        # Applying the binary operator 'and' (line 516)
        result_and_keyword_10842 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 11), 'and', is_user_defined_class_call_result_10837, result_is_not_10841)
        
        # Testing if the type of an if condition is none (line 516)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 516, 8), result_and_keyword_10842):
            pass
        else:
            
            # Testing the type of an if condition (line 516)
            if_condition_10843 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 516, 8), result_and_keyword_10842)
            # Assigning a type to the variable 'if_condition_10843' (line 516)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'if_condition_10843', if_condition_10843)
            # SSA begins for if statement (line 516)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'types' (line 517)
            types_10844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 19), 'types')
            # Obtaining the member 'InstanceType' of a type (line 517)
            InstanceType_10845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 19), types_10844, 'InstanceType')
            # Assigning a type to the variable 'stypy_return_type' (line 517)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 12), 'stypy_return_type', InstanceType_10845)
            # SSA join for if statement (line 516)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'self' (line 519)
        self_10846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 15), 'self')
        # Obtaining the member 'python_entity' of a type (line 519)
        python_entity_10847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 15), self_10846, 'python_entity')
        # Assigning a type to the variable 'stypy_return_type' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'stypy_return_type', python_entity_10847)
        
        # ################# End of 'get_python_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_python_type' in the type store
        # Getting the type of 'stypy_return_type' (line 507)
        stypy_return_type_10848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10848)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_python_type'
        return stypy_return_type_10848


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

        str_10849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, (-1)), 'str', '\n        Gets the stored class instance (if any). Class instances are only stored for instance intercession purposes, as\n        we need an entity to store these kind of changes.\n        :return:\n        ')
        # Getting the type of 'self' (line 527)
        self_10850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 15), 'self')
        # Obtaining the member 'instance' of a type (line 527)
        instance_10851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 15), self_10850, 'instance')
        # Assigning a type to the variable 'stypy_return_type' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'stypy_return_type', instance_10851)
        
        # ################# End of 'get_instance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_instance' in the type store
        # Getting the type of 'stypy_return_type' (line 521)
        stypy_return_type_10852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10852)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_instance'
        return stypy_return_type_10852


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

        str_10853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, (-1)), 'str', '\n        Determines if this proxy holds a value to the type it represents\n        :return:\n        ')
        
        # Call to hasattr(...): (line 534)
        # Processing the call arguments (line 534)
        # Getting the type of 'self' (line 534)
        self_10855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 23), 'self', False)
        str_10856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 29), 'str', 'value')
        # Processing the call keyword arguments (line 534)
        kwargs_10857 = {}
        # Getting the type of 'hasattr' (line 534)
        hasattr_10854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 15), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 534)
        hasattr_call_result_10858 = invoke(stypy.reporting.localization.Localization(__file__, 534, 15), hasattr_10854, *[self_10855, str_10856], **kwargs_10857)
        
        # Assigning a type to the variable 'stypy_return_type' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 8), 'stypy_return_type', hasattr_call_result_10858)
        
        # ################# End of 'has_value(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_value' in the type store
        # Getting the type of 'stypy_return_type' (line 529)
        stypy_return_type_10859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10859)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_value'
        return stypy_return_type_10859


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

        str_10860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, (-1)), 'str', '\n        Gets the value held by this proxy\n        :return: Value of the proxt\n        ')
        # Getting the type of 'self' (line 541)
        self_10861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 15), 'self')
        # Obtaining the member 'value' of a type (line 541)
        value_10862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 15), self_10861, 'value')
        # Assigning a type to the variable 'stypy_return_type' (line 541)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'stypy_return_type', value_10862)
        
        # ################# End of 'get_value(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_value' in the type store
        # Getting the type of 'stypy_return_type' (line 536)
        stypy_return_type_10863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10863)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_value'
        return stypy_return_type_10863


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

        str_10864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, (-1)), 'str', '\n        Sets the value held by this proxy. No type check is performed\n        :return: Value of the proxt\n        ')
        
        # Assigning a Name to a Attribute (line 548):
        
        # Assigning a Name to a Attribute (line 548):
        # Getting the type of 'value' (line 548)
        value_10865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 21), 'value')
        # Getting the type of 'self' (line 548)
        self_10866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'self')
        # Setting the type of the member 'value' of a type (line 548)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 8), self_10866, 'value', value_10865)
        
        # ################# End of 'set_value(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_value' in the type store
        # Getting the type of 'stypy_return_type' (line 543)
        stypy_return_type_10867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10867)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_value'
        return stypy_return_type_10867


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
        True_10868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 14), 'True')
        # Assigning a type to the variable 'True_10868' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'True_10868', True_10868)
        # Testing if the while is going to be iterated (line 553)
        # Testing the type of an if condition (line 553)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 553, 8), True_10868)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 553, 8), True_10868):
            
            # Assigning a Attribute to a Name (line 554):
            
            # Assigning a Attribute to a Name (line 554):
            # Getting the type of 'self' (line 554)
            self_10869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 22), 'self')
            # Obtaining the member 'parent_proxy' of a type (line 554)
            parent_proxy_10870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 22), self_10869, 'parent_proxy')
            # Obtaining the member 'python_entity' of a type (line 554)
            python_entity_10871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 22), parent_proxy_10870, 'python_entity')
            # Assigning a type to the variable 'current' (line 554)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 12), 'current', python_entity_10871)
            
            # Type idiom detected: calculating its left and rigth part (line 555)
            # Getting the type of 'current' (line 555)
            current_10872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 15), 'current')
            # Getting the type of 'None' (line 555)
            None_10873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 26), 'None')
            
            (may_be_10874, more_types_in_union_10875) = may_be_none(current_10872, None_10873)

            if may_be_10874:

                if more_types_in_union_10875:
                    # Runtime conditional SSA (line 555)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                str_10876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 23), 'str', '')
                # Assigning a type to the variable 'stypy_return_type' (line 556)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 16), 'stypy_return_type', str_10876)

                if more_types_in_union_10875:
                    # SSA join for if statement (line 555)
                    module_type_store = module_type_store.join_ssa_context()


            
            # Getting the type of 'current' (line 555)
            current_10877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 12), 'current')
            # Assigning a type to the variable 'current' (line 555)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 12), 'current', remove_type_from_union(current_10877, types.NoneType))
            
            # Call to isinstance(...): (line 557)
            # Processing the call arguments (line 557)
            # Getting the type of 'current' (line 557)
            current_10879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 26), 'current', False)
            # Getting the type of 'types' (line 557)
            types_10880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 35), 'types', False)
            # Obtaining the member 'ModuleType' of a type (line 557)
            ModuleType_10881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 35), types_10880, 'ModuleType')
            # Processing the call keyword arguments (line 557)
            kwargs_10882 = {}
            # Getting the type of 'isinstance' (line 557)
            isinstance_10878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 557)
            isinstance_call_result_10883 = invoke(stypy.reporting.localization.Localization(__file__, 557, 15), isinstance_10878, *[current_10879, ModuleType_10881], **kwargs_10882)
            
            # Testing if the type of an if condition is none (line 557)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 557, 12), isinstance_call_result_10883):
                pass
            else:
                
                # Testing the type of an if condition (line 557)
                if_condition_10884 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 557, 12), isinstance_call_result_10883)
                # Assigning a type to the variable 'if_condition_10884' (line 557)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 12), 'if_condition_10884', if_condition_10884)
                # SSA begins for if statement (line 557)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'current' (line 558)
                current_10885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 23), 'current')
                # Obtaining the member '__file__' of a type (line 558)
                file___10886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 23), current_10885, '__file__')
                # Assigning a type to the variable 'stypy_return_type' (line 558)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 16), 'stypy_return_type', file___10886)
                # SSA join for if statement (line 557)
                module_type_store = module_type_store.join_ssa_context()
                


        
        
        # ################# End of '__get_module_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__get_module_file' in the type store
        # Getting the type of 'stypy_return_type' (line 552)
        stypy_return_type_10887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10887)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__get_module_file'
        return stypy_return_type_10887


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

        str_10888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, (-1)), 'str', '\n        Returns the type of the passed member name or a TypeError if the stored entity has no member with the mentioned\n        name.\n        :param localization: Call localization data\n        :param member_name: Member name\n        :return: A type proxy with the member type or a TypeError\n        ')
        
        
        # SSA begins for try-except statement (line 568)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Type idiom detected: calculating its left and rigth part (line 569)
        # Getting the type of 'self' (line 569)
        self_10889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 15), 'self')
        # Obtaining the member 'instance' of a type (line 569)
        instance_10890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 15), self_10889, 'instance')
        # Getting the type of 'None' (line 569)
        None_10891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 32), 'None')
        
        (may_be_10892, more_types_in_union_10893) = may_be_none(instance_10890, None_10891)

        if may_be_10892:

            if more_types_in_union_10893:
                # Runtime conditional SSA (line 569)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to instance(...): (line 570)
            # Processing the call arguments (line 570)
            
            # Call to getattr(...): (line 570)
            # Processing the call arguments (line 570)
            # Getting the type of 'self' (line 570)
            self_10897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 59), 'self', False)
            # Obtaining the member 'python_entity' of a type (line 570)
            python_entity_10898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 59), self_10897, 'python_entity')
            # Getting the type of 'member_name' (line 570)
            member_name_10899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 79), 'member_name', False)
            # Processing the call keyword arguments (line 570)
            kwargs_10900 = {}
            # Getting the type of 'getattr' (line 570)
            getattr_10896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 51), 'getattr', False)
            # Calling getattr(args, kwargs) (line 570)
            getattr_call_result_10901 = invoke(stypy.reporting.localization.Localization(__file__, 570, 51), getattr_10896, *[python_entity_10898, member_name_10899], **kwargs_10900)
            
            # Getting the type of 'self' (line 571)
            self_10902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 51), 'self', False)
            # Obtaining the member 'name' of a type (line 571)
            name_10903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 51), self_10902, 'name')
            str_10904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 63), 'str', '.')
            # Applying the binary operator '+' (line 571)
            result_add_10905 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 51), '+', name_10903, str_10904)
            
            # Getting the type of 'member_name' (line 571)
            member_name_10906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 69), 'member_name', False)
            # Applying the binary operator '+' (line 571)
            result_add_10907 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 67), '+', result_add_10905, member_name_10906)
            
            # Processing the call keyword arguments (line 570)
            # Getting the type of 'self' (line 572)
            self_10908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 58), 'self', False)
            keyword_10909 = self_10908
            kwargs_10910 = {'parent': keyword_10909}
            # Getting the type of 'TypeInferenceProxy' (line 570)
            TypeInferenceProxy_10894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 23), 'TypeInferenceProxy', False)
            # Obtaining the member 'instance' of a type (line 570)
            instance_10895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 23), TypeInferenceProxy_10894, 'instance')
            # Calling instance(args, kwargs) (line 570)
            instance_call_result_10911 = invoke(stypy.reporting.localization.Localization(__file__, 570, 23), instance_10895, *[getattr_call_result_10901, result_add_10907], **kwargs_10910)
            
            # Assigning a type to the variable 'stypy_return_type' (line 570)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 16), 'stypy_return_type', instance_call_result_10911)

            if more_types_in_union_10893:
                # Runtime conditional SSA for else branch (line 569)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_10892) or more_types_in_union_10893):
            
            # Call to hasattr(...): (line 575)
            # Processing the call arguments (line 575)
            # Getting the type of 'self' (line 575)
            self_10913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 27), 'self', False)
            # Obtaining the member 'instance' of a type (line 575)
            instance_10914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 27), self_10913, 'instance')
            # Getting the type of 'member_name' (line 575)
            member_name_10915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 42), 'member_name', False)
            # Processing the call keyword arguments (line 575)
            kwargs_10916 = {}
            # Getting the type of 'hasattr' (line 575)
            hasattr_10912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 19), 'hasattr', False)
            # Calling hasattr(args, kwargs) (line 575)
            hasattr_call_result_10917 = invoke(stypy.reporting.localization.Localization(__file__, 575, 19), hasattr_10912, *[instance_10914, member_name_10915], **kwargs_10916)
            
            # Testing if the type of an if condition is none (line 575)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 575, 16), hasattr_call_result_10917):
                
                # Assigning a Call to a Name (line 584):
                
                # Assigning a Call to a Name (line 584):
                
                # Call to get_original_program_from_type_inference_file(...): (line 584)
                # Processing the call arguments (line 584)
                
                # Call to __get_module_file(...): (line 585)
                # Processing the call keyword arguments (line 585)
                kwargs_10941 = {}
                # Getting the type of 'self' (line 585)
                self_10939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 24), 'self', False)
                # Obtaining the member '__get_module_file' of a type (line 585)
                get_module_file_10940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 24), self_10939, '__get_module_file')
                # Calling __get_module_file(args, kwargs) (line 585)
                get_module_file_call_result_10942 = invoke(stypy.reporting.localization.Localization(__file__, 585, 24), get_module_file_10940, *[], **kwargs_10941)
                
                # Processing the call keyword arguments (line 584)
                kwargs_10943 = {}
                # Getting the type of 'stypy_parameters_copy' (line 584)
                stypy_parameters_copy_10937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 34), 'stypy_parameters_copy', False)
                # Obtaining the member 'get_original_program_from_type_inference_file' of a type (line 584)
                get_original_program_from_type_inference_file_10938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 34), stypy_parameters_copy_10937, 'get_original_program_from_type_inference_file')
                # Calling get_original_program_from_type_inference_file(args, kwargs) (line 584)
                get_original_program_from_type_inference_file_call_result_10944 = invoke(stypy.reporting.localization.Localization(__file__, 584, 34), get_original_program_from_type_inference_file_10938, *[get_module_file_call_result_10942], **kwargs_10943)
                
                # Assigning a type to the variable 'module_path' (line 584)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 20), 'module_path', get_original_program_from_type_inference_file_call_result_10944)
                
                # Assigning a Call to a Name (line 586):
                
                # Assigning a Call to a Name (line 586):
                
                # Call to get_type_store_of_module(...): (line 586)
                # Processing the call arguments (line 586)
                # Getting the type of 'module_path' (line 586)
                module_path_10949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 86), 'module_path', False)
                # Processing the call keyword arguments (line 586)
                kwargs_10950 = {}
                # Getting the type of 'type_store_copy' (line 586)
                type_store_copy_10945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 25), 'type_store_copy', False)
                # Obtaining the member 'typestore' of a type (line 586)
                typestore_10946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 25), type_store_copy_10945, 'typestore')
                # Obtaining the member 'TypeStore' of a type (line 586)
                TypeStore_10947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 25), typestore_10946, 'TypeStore')
                # Obtaining the member 'get_type_store_of_module' of a type (line 586)
                get_type_store_of_module_10948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 25), TypeStore_10947, 'get_type_store_of_module')
                # Calling get_type_store_of_module(args, kwargs) (line 586)
                get_type_store_of_module_call_result_10951 = invoke(stypy.reporting.localization.Localization(__file__, 586, 25), get_type_store_of_module_10948, *[module_path_10949], **kwargs_10950)
                
                # Assigning a type to the variable 'ts' (line 586)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 20), 'ts', get_type_store_of_module_call_result_10951)
                
                # Assigning a Call to a Name (line 587):
                
                # Assigning a Call to a Name (line 587):
                
                # Call to get_type_of(...): (line 587)
                # Processing the call arguments (line 587)
                # Getting the type of 'localization' (line 587)
                localization_10954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 41), 'localization', False)
                # Getting the type of 'self' (line 587)
                self_10955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 55), 'self', False)
                # Obtaining the member 'python_entity' of a type (line 587)
                python_entity_10956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 55), self_10955, 'python_entity')
                # Obtaining the member '__name__' of a type (line 587)
                name___10957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 55), python_entity_10956, '__name__')
                # Processing the call keyword arguments (line 587)
                kwargs_10958 = {}
                # Getting the type of 'ts' (line 587)
                ts_10952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 26), 'ts', False)
                # Obtaining the member 'get_type_of' of a type (line 587)
                get_type_of_10953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 26), ts_10952, 'get_type_of')
                # Calling get_type_of(args, kwargs) (line 587)
                get_type_of_call_result_10959 = invoke(stypy.reporting.localization.Localization(__file__, 587, 26), get_type_of_10953, *[localization_10954, name___10957], **kwargs_10958)
                
                # Assigning a type to the variable 'typ' (line 587)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 20), 'typ', get_type_of_call_result_10959)
                
                # Call to get_type_of_member(...): (line 588)
                # Processing the call arguments (line 588)
                # Getting the type of 'localization' (line 588)
                localization_10962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 50), 'localization', False)
                # Getting the type of 'member_name' (line 588)
                member_name_10963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 64), 'member_name', False)
                # Processing the call keyword arguments (line 588)
                kwargs_10964 = {}
                # Getting the type of 'typ' (line 588)
                typ_10960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 27), 'typ', False)
                # Obtaining the member 'get_type_of_member' of a type (line 588)
                get_type_of_member_10961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 27), typ_10960, 'get_type_of_member')
                # Calling get_type_of_member(args, kwargs) (line 588)
                get_type_of_member_call_result_10965 = invoke(stypy.reporting.localization.Localization(__file__, 588, 27), get_type_of_member_10961, *[localization_10962, member_name_10963], **kwargs_10964)
                
                # Assigning a type to the variable 'stypy_return_type' (line 588)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 20), 'stypy_return_type', get_type_of_member_call_result_10965)
            else:
                
                # Testing the type of an if condition (line 575)
                if_condition_10918 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 575, 16), hasattr_call_result_10917)
                # Assigning a type to the variable 'if_condition_10918' (line 575)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 16), 'if_condition_10918', if_condition_10918)
                # SSA begins for if statement (line 575)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to instance(...): (line 576)
                # Processing the call arguments (line 576)
                
                # Call to getattr(...): (line 576)
                # Processing the call arguments (line 576)
                # Getting the type of 'self' (line 576)
                self_10922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 63), 'self', False)
                # Obtaining the member 'instance' of a type (line 576)
                instance_10923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 63), self_10922, 'instance')
                # Getting the type of 'member_name' (line 576)
                member_name_10924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 78), 'member_name', False)
                # Processing the call keyword arguments (line 576)
                kwargs_10925 = {}
                # Getting the type of 'getattr' (line 576)
                getattr_10921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 55), 'getattr', False)
                # Calling getattr(args, kwargs) (line 576)
                getattr_call_result_10926 = invoke(stypy.reporting.localization.Localization(__file__, 576, 55), getattr_10921, *[instance_10923, member_name_10924], **kwargs_10925)
                
                # Getting the type of 'self' (line 577)
                self_10927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 55), 'self', False)
                # Obtaining the member 'name' of a type (line 577)
                name_10928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 55), self_10927, 'name')
                str_10929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 67), 'str', '.')
                # Applying the binary operator '+' (line 577)
                result_add_10930 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 55), '+', name_10928, str_10929)
                
                # Getting the type of 'member_name' (line 577)
                member_name_10931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 73), 'member_name', False)
                # Applying the binary operator '+' (line 577)
                result_add_10932 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 71), '+', result_add_10930, member_name_10931)
                
                # Processing the call keyword arguments (line 576)
                # Getting the type of 'self' (line 578)
                self_10933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 62), 'self', False)
                keyword_10934 = self_10933
                kwargs_10935 = {'parent': keyword_10934}
                # Getting the type of 'TypeInferenceProxy' (line 576)
                TypeInferenceProxy_10919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 27), 'TypeInferenceProxy', False)
                # Obtaining the member 'instance' of a type (line 576)
                instance_10920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 27), TypeInferenceProxy_10919, 'instance')
                # Calling instance(args, kwargs) (line 576)
                instance_call_result_10936 = invoke(stypy.reporting.localization.Localization(__file__, 576, 27), instance_10920, *[getattr_call_result_10926, result_add_10932], **kwargs_10935)
                
                # Assigning a type to the variable 'stypy_return_type' (line 576)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 20), 'stypy_return_type', instance_call_result_10936)
                # SSA branch for the else part of an if statement (line 575)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Name (line 584):
                
                # Assigning a Call to a Name (line 584):
                
                # Call to get_original_program_from_type_inference_file(...): (line 584)
                # Processing the call arguments (line 584)
                
                # Call to __get_module_file(...): (line 585)
                # Processing the call keyword arguments (line 585)
                kwargs_10941 = {}
                # Getting the type of 'self' (line 585)
                self_10939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 24), 'self', False)
                # Obtaining the member '__get_module_file' of a type (line 585)
                get_module_file_10940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 24), self_10939, '__get_module_file')
                # Calling __get_module_file(args, kwargs) (line 585)
                get_module_file_call_result_10942 = invoke(stypy.reporting.localization.Localization(__file__, 585, 24), get_module_file_10940, *[], **kwargs_10941)
                
                # Processing the call keyword arguments (line 584)
                kwargs_10943 = {}
                # Getting the type of 'stypy_parameters_copy' (line 584)
                stypy_parameters_copy_10937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 34), 'stypy_parameters_copy', False)
                # Obtaining the member 'get_original_program_from_type_inference_file' of a type (line 584)
                get_original_program_from_type_inference_file_10938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 34), stypy_parameters_copy_10937, 'get_original_program_from_type_inference_file')
                # Calling get_original_program_from_type_inference_file(args, kwargs) (line 584)
                get_original_program_from_type_inference_file_call_result_10944 = invoke(stypy.reporting.localization.Localization(__file__, 584, 34), get_original_program_from_type_inference_file_10938, *[get_module_file_call_result_10942], **kwargs_10943)
                
                # Assigning a type to the variable 'module_path' (line 584)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 20), 'module_path', get_original_program_from_type_inference_file_call_result_10944)
                
                # Assigning a Call to a Name (line 586):
                
                # Assigning a Call to a Name (line 586):
                
                # Call to get_type_store_of_module(...): (line 586)
                # Processing the call arguments (line 586)
                # Getting the type of 'module_path' (line 586)
                module_path_10949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 86), 'module_path', False)
                # Processing the call keyword arguments (line 586)
                kwargs_10950 = {}
                # Getting the type of 'type_store_copy' (line 586)
                type_store_copy_10945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 25), 'type_store_copy', False)
                # Obtaining the member 'typestore' of a type (line 586)
                typestore_10946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 25), type_store_copy_10945, 'typestore')
                # Obtaining the member 'TypeStore' of a type (line 586)
                TypeStore_10947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 25), typestore_10946, 'TypeStore')
                # Obtaining the member 'get_type_store_of_module' of a type (line 586)
                get_type_store_of_module_10948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 25), TypeStore_10947, 'get_type_store_of_module')
                # Calling get_type_store_of_module(args, kwargs) (line 586)
                get_type_store_of_module_call_result_10951 = invoke(stypy.reporting.localization.Localization(__file__, 586, 25), get_type_store_of_module_10948, *[module_path_10949], **kwargs_10950)
                
                # Assigning a type to the variable 'ts' (line 586)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 20), 'ts', get_type_store_of_module_call_result_10951)
                
                # Assigning a Call to a Name (line 587):
                
                # Assigning a Call to a Name (line 587):
                
                # Call to get_type_of(...): (line 587)
                # Processing the call arguments (line 587)
                # Getting the type of 'localization' (line 587)
                localization_10954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 41), 'localization', False)
                # Getting the type of 'self' (line 587)
                self_10955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 55), 'self', False)
                # Obtaining the member 'python_entity' of a type (line 587)
                python_entity_10956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 55), self_10955, 'python_entity')
                # Obtaining the member '__name__' of a type (line 587)
                name___10957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 55), python_entity_10956, '__name__')
                # Processing the call keyword arguments (line 587)
                kwargs_10958 = {}
                # Getting the type of 'ts' (line 587)
                ts_10952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 26), 'ts', False)
                # Obtaining the member 'get_type_of' of a type (line 587)
                get_type_of_10953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 26), ts_10952, 'get_type_of')
                # Calling get_type_of(args, kwargs) (line 587)
                get_type_of_call_result_10959 = invoke(stypy.reporting.localization.Localization(__file__, 587, 26), get_type_of_10953, *[localization_10954, name___10957], **kwargs_10958)
                
                # Assigning a type to the variable 'typ' (line 587)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 20), 'typ', get_type_of_call_result_10959)
                
                # Call to get_type_of_member(...): (line 588)
                # Processing the call arguments (line 588)
                # Getting the type of 'localization' (line 588)
                localization_10962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 50), 'localization', False)
                # Getting the type of 'member_name' (line 588)
                member_name_10963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 64), 'member_name', False)
                # Processing the call keyword arguments (line 588)
                kwargs_10964 = {}
                # Getting the type of 'typ' (line 588)
                typ_10960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 27), 'typ', False)
                # Obtaining the member 'get_type_of_member' of a type (line 588)
                get_type_of_member_10961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 27), typ_10960, 'get_type_of_member')
                # Calling get_type_of_member(args, kwargs) (line 588)
                get_type_of_member_call_result_10965 = invoke(stypy.reporting.localization.Localization(__file__, 588, 27), get_type_of_member_10961, *[localization_10962, member_name_10963], **kwargs_10964)
                
                # Assigning a type to the variable 'stypy_return_type' (line 588)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 20), 'stypy_return_type', get_type_of_member_call_result_10965)
                # SSA join for if statement (line 575)
                module_type_store = module_type_store.join_ssa_context()
                


            if (may_be_10892 and more_types_in_union_10893):
                # SSA join for if statement (line 569)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA branch for the except part of a try statement (line 568)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 568)
        module_type_store.open_ssa_branch('except')
        
        # Call to TypeError(...): (line 593)
        # Processing the call arguments (line 593)
        # Getting the type of 'localization' (line 593)
        localization_10967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 29), 'localization', False)
        
        # Call to format(...): (line 594)
        # Processing the call arguments (line 594)
        
        # Call to get_python_type(...): (line 594)
        # Processing the call keyword arguments (line 594)
        kwargs_10972 = {}
        # Getting the type of 'self' (line 594)
        self_10970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 62), 'self', False)
        # Obtaining the member 'get_python_type' of a type (line 594)
        get_python_type_10971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 62), self_10970, 'get_python_type')
        # Calling get_python_type(args, kwargs) (line 594)
        get_python_type_call_result_10973 = invoke(stypy.reporting.localization.Localization(__file__, 594, 62), get_python_type_10971, *[], **kwargs_10972)
        
        # Obtaining the member '__name__' of a type (line 594)
        name___10974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 62), get_python_type_call_result_10973, '__name__')
        # Getting the type of 'member_name' (line 594)
        member_name_10975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 95), 'member_name', False)
        # Processing the call keyword arguments (line 594)
        kwargs_10976 = {}
        str_10968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 29), 'str', "{0} has no member '{1}'")
        # Obtaining the member 'format' of a type (line 594)
        format_10969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 29), str_10968, 'format')
        # Calling format(args, kwargs) (line 594)
        format_call_result_10977 = invoke(stypy.reporting.localization.Localization(__file__, 594, 29), format_10969, *[name___10974, member_name_10975], **kwargs_10976)
        
        # Processing the call keyword arguments (line 593)
        kwargs_10978 = {}
        # Getting the type of 'TypeError' (line 593)
        TypeError_10966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 19), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 593)
        TypeError_call_result_10979 = invoke(stypy.reporting.localization.Localization(__file__, 593, 19), TypeError_10966, *[localization_10967, format_call_result_10977], **kwargs_10978)
        
        # Assigning a type to the variable 'stypy_return_type' (line 593)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 12), 'stypy_return_type', TypeError_call_result_10979)
        # SSA join for try-except statement (line 568)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_type_of_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_type_of_member' in the type store
        # Getting the type of 'stypy_return_type' (line 560)
        stypy_return_type_10980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10980)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_type_of_member'
        return stypy_return_type_10980


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

        str_10981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, (-1)), 'str', '\n        Set the type of a member of the represented object. If the member do not exist, it is created with the passed\n        name and types (except iif the represented object do not support reflection, in that case a TypeError is\n        returned)\n        :param localization: Caller information\n        :param member_name: Name of the member\n        :param member_type: Type of the member\n        :return: None or a TypeError\n        ')
        
        
        # SSA begins for try-except statement (line 606)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Tuple (line 607):
        
        # Assigning a Call to a Name:
        
        # Call to contains_an_undefined_type(...): (line 607)
        # Processing the call arguments (line 607)
        # Getting the type of 'member_type' (line 607)
        member_type_10984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 100), 'member_type', False)
        # Processing the call keyword arguments (line 607)
        kwargs_10985 = {}
        # Getting the type of 'TypeInferenceProxy' (line 607)
        TypeInferenceProxy_10982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 54), 'TypeInferenceProxy', False)
        # Obtaining the member 'contains_an_undefined_type' of a type (line 607)
        contains_an_undefined_type_10983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 54), TypeInferenceProxy_10982, 'contains_an_undefined_type')
        # Calling contains_an_undefined_type(args, kwargs) (line 607)
        contains_an_undefined_type_call_result_10986 = invoke(stypy.reporting.localization.Localization(__file__, 607, 54), contains_an_undefined_type_10983, *[member_type_10984], **kwargs_10985)
        
        # Assigning a type to the variable 'call_assignment_10131' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'call_assignment_10131', contains_an_undefined_type_call_result_10986)
        
        # Assigning a Call to a Name (line 607):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_10131' (line 607)
        call_assignment_10131_10987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'call_assignment_10131', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_10988 = stypy_get_value_from_tuple(call_assignment_10131_10987, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_10132' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'call_assignment_10132', stypy_get_value_from_tuple_call_result_10988)
        
        # Assigning a Name to a Name (line 607):
        # Getting the type of 'call_assignment_10132' (line 607)
        call_assignment_10132_10989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'call_assignment_10132')
        # Assigning a type to the variable 'contains_undefined' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'contains_undefined', call_assignment_10132_10989)
        
        # Assigning a Call to a Name (line 607):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_10131' (line 607)
        call_assignment_10131_10990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'call_assignment_10131', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_10991 = stypy_get_value_from_tuple(call_assignment_10131_10990, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_10133' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'call_assignment_10133', stypy_get_value_from_tuple_call_result_10991)
        
        # Assigning a Name to a Name (line 607):
        # Getting the type of 'call_assignment_10133' (line 607)
        call_assignment_10133_10992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'call_assignment_10133')
        # Assigning a type to the variable 'more_types_in_value' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 32), 'more_types_in_value', call_assignment_10133_10992)
        # Getting the type of 'contains_undefined' (line 608)
        contains_undefined_10993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 15), 'contains_undefined')
        # Testing if the type of an if condition is none (line 608)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 608, 12), contains_undefined_10993):
            pass
        else:
            
            # Testing the type of an if condition (line 608)
            if_condition_10994 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 608, 12), contains_undefined_10993)
            # Assigning a type to the variable 'if_condition_10994' (line 608)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 12), 'if_condition_10994', if_condition_10994)
            # SSA begins for if statement (line 608)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'more_types_in_value' (line 609)
            more_types_in_value_10995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 19), 'more_types_in_value')
            int_10996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 42), 'int')
            # Applying the binary operator '==' (line 609)
            result_eq_10997 = python_operator(stypy.reporting.localization.Localization(__file__, 609, 19), '==', more_types_in_value_10995, int_10996)
            
            # Testing if the type of an if condition is none (line 609)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 609, 16), result_eq_10997):
                
                # Call to instance(...): (line 613)
                # Processing the call arguments (line 613)
                # Getting the type of 'localization' (line 613)
                localization_11013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 41), 'localization', False)
                
                # Call to format(...): (line 614)
                # Processing the call arguments (line 614)
                # Getting the type of 'self' (line 615)
                self_11016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 48), 'self', False)
                # Obtaining the member 'parent_proxy' of a type (line 615)
                parent_proxy_11017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 48), self_11016, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 615)
                name_11018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 48), parent_proxy_11017, 'name')
                # Getting the type of 'member_name' (line 615)
                member_name_11019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 72), 'member_name', False)
                # Processing the call keyword arguments (line 614)
                kwargs_11020 = {}
                str_11014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 41), 'str', 'Potentialy assigning to {0}.{1} the value of a previously undefined variable')
                # Obtaining the member 'format' of a type (line 614)
                format_11015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 41), str_11014, 'format')
                # Calling format(args, kwargs) (line 614)
                format_call_result_11021 = invoke(stypy.reporting.localization.Localization(__file__, 614, 41), format_11015, *[name_11018, member_name_11019], **kwargs_11020)
                
                # Processing the call keyword arguments (line 613)
                kwargs_11022 = {}
                # Getting the type of 'TypeWarning' (line 613)
                TypeWarning_11011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 20), 'TypeWarning', False)
                # Obtaining the member 'instance' of a type (line 613)
                instance_11012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 20), TypeWarning_11011, 'instance')
                # Calling instance(args, kwargs) (line 613)
                instance_call_result_11023 = invoke(stypy.reporting.localization.Localization(__file__, 613, 20), instance_11012, *[localization_11013, format_call_result_11021], **kwargs_11022)
                
            else:
                
                # Testing the type of an if condition (line 609)
                if_condition_10998 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 609, 16), result_eq_10997)
                # Assigning a type to the variable 'if_condition_10998' (line 609)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 16), 'if_condition_10998', if_condition_10998)
                # SSA begins for if statement (line 609)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to TypeError(...): (line 610)
                # Processing the call arguments (line 610)
                # Getting the type of 'localization' (line 610)
                localization_11000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 30), 'localization', False)
                
                # Call to format(...): (line 610)
                # Processing the call arguments (line 610)
                # Getting the type of 'self' (line 611)
                self_11003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 37), 'self', False)
                # Obtaining the member 'parent_proxy' of a type (line 611)
                parent_proxy_11004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 37), self_11003, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 611)
                name_11005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 37), parent_proxy_11004, 'name')
                # Getting the type of 'member_name' (line 611)
                member_name_11006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 61), 'member_name', False)
                # Processing the call keyword arguments (line 610)
                kwargs_11007 = {}
                str_11001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 44), 'str', 'Assigning to {0}.{1} the value of a previously undefined variable')
                # Obtaining the member 'format' of a type (line 610)
                format_11002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 44), str_11001, 'format')
                # Calling format(args, kwargs) (line 610)
                format_call_result_11008 = invoke(stypy.reporting.localization.Localization(__file__, 610, 44), format_11002, *[name_11005, member_name_11006], **kwargs_11007)
                
                # Processing the call keyword arguments (line 610)
                kwargs_11009 = {}
                # Getting the type of 'TypeError' (line 610)
                TypeError_10999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 20), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 610)
                TypeError_call_result_11010 = invoke(stypy.reporting.localization.Localization(__file__, 610, 20), TypeError_10999, *[localization_11000, format_call_result_11008], **kwargs_11009)
                
                # SSA branch for the else part of an if statement (line 609)
                module_type_store.open_ssa_branch('else')
                
                # Call to instance(...): (line 613)
                # Processing the call arguments (line 613)
                # Getting the type of 'localization' (line 613)
                localization_11013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 41), 'localization', False)
                
                # Call to format(...): (line 614)
                # Processing the call arguments (line 614)
                # Getting the type of 'self' (line 615)
                self_11016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 48), 'self', False)
                # Obtaining the member 'parent_proxy' of a type (line 615)
                parent_proxy_11017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 48), self_11016, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 615)
                name_11018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 48), parent_proxy_11017, 'name')
                # Getting the type of 'member_name' (line 615)
                member_name_11019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 72), 'member_name', False)
                # Processing the call keyword arguments (line 614)
                kwargs_11020 = {}
                str_11014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 41), 'str', 'Potentialy assigning to {0}.{1} the value of a previously undefined variable')
                # Obtaining the member 'format' of a type (line 614)
                format_11015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 41), str_11014, 'format')
                # Calling format(args, kwargs) (line 614)
                format_call_result_11021 = invoke(stypy.reporting.localization.Localization(__file__, 614, 41), format_11015, *[name_11018, member_name_11019], **kwargs_11020)
                
                # Processing the call keyword arguments (line 613)
                kwargs_11022 = {}
                # Getting the type of 'TypeWarning' (line 613)
                TypeWarning_11011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 20), 'TypeWarning', False)
                # Obtaining the member 'instance' of a type (line 613)
                instance_11012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 20), TypeWarning_11011, 'instance')
                # Calling instance(args, kwargs) (line 613)
                instance_call_result_11023 = invoke(stypy.reporting.localization.Localization(__file__, 613, 20), instance_11012, *[localization_11013, format_call_result_11021], **kwargs_11022)
                
                # SSA join for if statement (line 609)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 608)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 617)
        self_11024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 15), 'self')
        # Obtaining the member 'instance' of a type (line 617)
        instance_11025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 15), self_11024, 'instance')
        # Getting the type of 'None' (line 617)
        None_11026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 36), 'None')
        # Applying the binary operator 'isnot' (line 617)
        result_is_not_11027 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 15), 'isnot', instance_11025, None_11026)
        
        # Testing if the type of an if condition is none (line 617)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 617, 12), result_is_not_11027):
            pass
        else:
            
            # Testing the type of an if condition (line 617)
            if_condition_11028 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 617, 12), result_is_not_11027)
            # Assigning a type to the variable 'if_condition_11028' (line 617)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 12), 'if_condition_11028', if_condition_11028)
            # SSA begins for if statement (line 617)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to setattr(...): (line 619)
            # Processing the call arguments (line 619)
            # Getting the type of 'self' (line 619)
            self_11030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 24), 'self', False)
            # Obtaining the member 'instance' of a type (line 619)
            instance_11031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 24), self_11030, 'instance')
            # Getting the type of 'member_name' (line 619)
            member_name_11032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 39), 'member_name', False)
            # Getting the type of 'member_type' (line 619)
            member_type_11033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 52), 'member_type', False)
            # Processing the call keyword arguments (line 619)
            kwargs_11034 = {}
            # Getting the type of 'setattr' (line 619)
            setattr_11029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 16), 'setattr', False)
            # Calling setattr(args, kwargs) (line 619)
            setattr_call_result_11035 = invoke(stypy.reporting.localization.Localization(__file__, 619, 16), setattr_11029, *[instance_11031, member_name_11032, member_type_11033], **kwargs_11034)
            
            # Getting the type of 'self' (line 620)
            self_11036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 19), 'self')
            # Obtaining the member 'annotate_types' of a type (line 620)
            annotate_types_11037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 19), self_11036, 'annotate_types')
            # Testing if the type of an if condition is none (line 620)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 620, 16), annotate_types_11037):
                pass
            else:
                
                # Testing the type of an if condition (line 620)
                if_condition_11038 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 620, 16), annotate_types_11037)
                # Assigning a type to the variable 'if_condition_11038' (line 620)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 16), 'if_condition_11038', if_condition_11038)
                # SSA begins for if statement (line 620)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to __annotate_type(...): (line 621)
                # Processing the call arguments (line 621)
                # Getting the type of 'localization' (line 621)
                localization_11041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 41), 'localization', False)
                # Obtaining the member 'line' of a type (line 621)
                line_11042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 41), localization_11041, 'line')
                # Getting the type of 'localization' (line 621)
                localization_11043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 60), 'localization', False)
                # Obtaining the member 'column' of a type (line 621)
                column_11044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 60), localization_11043, 'column')
                # Getting the type of 'member_name' (line 621)
                member_name_11045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 81), 'member_name', False)
                # Getting the type of 'member_type' (line 622)
                member_type_11046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 41), 'member_type', False)
                # Processing the call keyword arguments (line 621)
                kwargs_11047 = {}
                # Getting the type of 'self' (line 621)
                self_11039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 20), 'self', False)
                # Obtaining the member '__annotate_type' of a type (line 621)
                annotate_type_11040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 20), self_11039, '__annotate_type')
                # Calling __annotate_type(args, kwargs) (line 621)
                annotate_type_call_result_11048 = invoke(stypy.reporting.localization.Localization(__file__, 621, 20), annotate_type_11040, *[line_11042, column_11044, member_name_11045, member_type_11046], **kwargs_11047)
                
                # SSA join for if statement (line 620)
                module_type_store = module_type_store.join_ssa_context()
                

            # Getting the type of 'None' (line 623)
            None_11049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 23), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 623)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 16), 'stypy_return_type', None_11049)
            # SSA join for if statement (line 617)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Evaluating a boolean operation
        
        # Call to supports_structural_reflection(...): (line 625)
        # Processing the call arguments (line 625)
        # Getting the type of 'self' (line 625)
        self_11052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 83), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 625)
        python_entity_11053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 83), self_11052, 'python_entity')
        # Processing the call keyword arguments (line 625)
        kwargs_11054 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 625)
        type_inference_proxy_management_copy_11050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 15), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'supports_structural_reflection' of a type (line 625)
        supports_structural_reflection_11051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 15), type_inference_proxy_management_copy_11050, 'supports_structural_reflection')
        # Calling supports_structural_reflection(args, kwargs) (line 625)
        supports_structural_reflection_call_result_11055 = invoke(stypy.reporting.localization.Localization(__file__, 625, 15), supports_structural_reflection_11051, *[python_entity_11053], **kwargs_11054)
        
        
        # Call to hasattr(...): (line 625)
        # Processing the call arguments (line 625)
        # Getting the type of 'self' (line 626)
        self_11057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 20), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 626)
        python_entity_11058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 20), self_11057, 'python_entity')
        # Getting the type of 'member_name' (line 626)
        member_name_11059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 40), 'member_name', False)
        # Processing the call keyword arguments (line 625)
        kwargs_11060 = {}
        # Getting the type of 'hasattr' (line 625)
        hasattr_11056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 106), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 625)
        hasattr_call_result_11061 = invoke(stypy.reporting.localization.Localization(__file__, 625, 106), hasattr_11056, *[python_entity_11058, member_name_11059], **kwargs_11060)
        
        # Applying the binary operator 'or' (line 625)
        result_or_keyword_11062 = python_operator(stypy.reporting.localization.Localization(__file__, 625, 15), 'or', supports_structural_reflection_call_result_11055, hasattr_call_result_11061)
        
        # Testing if the type of an if condition is none (line 625)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 625, 12), result_or_keyword_11062):
            pass
        else:
            
            # Testing the type of an if condition (line 625)
            if_condition_11063 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 625, 12), result_or_keyword_11062)
            # Assigning a type to the variable 'if_condition_11063' (line 625)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 12), 'if_condition_11063', if_condition_11063)
            # SSA begins for if statement (line 625)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to setattr(...): (line 628)
            # Processing the call arguments (line 628)
            # Getting the type of 'self' (line 628)
            self_11065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 24), 'self', False)
            # Obtaining the member 'python_entity' of a type (line 628)
            python_entity_11066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 24), self_11065, 'python_entity')
            # Getting the type of 'member_name' (line 628)
            member_name_11067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 44), 'member_name', False)
            # Getting the type of 'member_type' (line 628)
            member_type_11068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 57), 'member_type', False)
            # Processing the call keyword arguments (line 628)
            kwargs_11069 = {}
            # Getting the type of 'setattr' (line 628)
            setattr_11064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 16), 'setattr', False)
            # Calling setattr(args, kwargs) (line 628)
            setattr_call_result_11070 = invoke(stypy.reporting.localization.Localization(__file__, 628, 16), setattr_11064, *[python_entity_11066, member_name_11067, member_type_11068], **kwargs_11069)
            
            # Getting the type of 'self' (line 629)
            self_11071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 19), 'self')
            # Obtaining the member 'annotate_types' of a type (line 629)
            annotate_types_11072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 19), self_11071, 'annotate_types')
            # Testing if the type of an if condition is none (line 629)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 629, 16), annotate_types_11072):
                pass
            else:
                
                # Testing the type of an if condition (line 629)
                if_condition_11073 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 629, 16), annotate_types_11072)
                # Assigning a type to the variable 'if_condition_11073' (line 629)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 16), 'if_condition_11073', if_condition_11073)
                # SSA begins for if statement (line 629)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to __annotate_type(...): (line 630)
                # Processing the call arguments (line 630)
                # Getting the type of 'localization' (line 630)
                localization_11076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 41), 'localization', False)
                # Obtaining the member 'line' of a type (line 630)
                line_11077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 41), localization_11076, 'line')
                # Getting the type of 'localization' (line 630)
                localization_11078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 60), 'localization', False)
                # Obtaining the member 'column' of a type (line 630)
                column_11079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 60), localization_11078, 'column')
                # Getting the type of 'member_name' (line 630)
                member_name_11080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 81), 'member_name', False)
                # Getting the type of 'member_type' (line 631)
                member_type_11081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 41), 'member_type', False)
                # Processing the call keyword arguments (line 630)
                kwargs_11082 = {}
                # Getting the type of 'self' (line 630)
                self_11074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 20), 'self', False)
                # Obtaining the member '__annotate_type' of a type (line 630)
                annotate_type_11075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 20), self_11074, '__annotate_type')
                # Calling __annotate_type(args, kwargs) (line 630)
                annotate_type_call_result_11083 = invoke(stypy.reporting.localization.Localization(__file__, 630, 20), annotate_type_11075, *[line_11077, column_11079, member_name_11080, member_type_11081], **kwargs_11082)
                
                # SSA join for if statement (line 629)
                module_type_store = module_type_store.join_ssa_context()
                

            # Getting the type of 'None' (line 632)
            None_11084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 23), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 632)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 16), 'stypy_return_type', None_11084)
            # SSA join for if statement (line 625)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA branch for the except part of a try statement (line 606)
        # SSA branch for the except 'Exception' branch of a try statement (line 606)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'Exception' (line 633)
        Exception_11085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 15), 'Exception')
        # Assigning a type to the variable 'exc' (line 633)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 8), 'exc', Exception_11085)
        
        # Call to TypeError(...): (line 634)
        # Processing the call arguments (line 634)
        # Getting the type of 'localization' (line 634)
        localization_11087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 29), 'localization', False)
        
        # Call to format(...): (line 635)
        # Processing the call arguments (line 635)
        
        # Call to __repr__(...): (line 635)
        # Processing the call keyword arguments (line 635)
        kwargs_11092 = {}
        # Getting the type of 'self' (line 635)
        self_11090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 80), 'self', False)
        # Obtaining the member '__repr__' of a type (line 635)
        repr___11091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 80), self_11090, '__repr__')
        # Calling __repr__(args, kwargs) (line 635)
        repr___call_result_11093 = invoke(stypy.reporting.localization.Localization(__file__, 635, 80), repr___11091, *[], **kwargs_11092)
        
        
        # Call to str(...): (line 635)
        # Processing the call arguments (line 635)
        # Getting the type of 'exc' (line 635)
        exc_11095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 101), 'exc', False)
        # Processing the call keyword arguments (line 635)
        kwargs_11096 = {}
        # Getting the type of 'str' (line 635)
        str_11094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 97), 'str', False)
        # Calling str(args, kwargs) (line 635)
        str_call_result_11097 = invoke(stypy.reporting.localization.Localization(__file__, 635, 97), str_11094, *[exc_11095], **kwargs_11096)
        
        # Processing the call keyword arguments (line 635)
        kwargs_11098 = {}
        str_11088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 29), 'str', "Cannot modify the structure of '{0}': {1}")
        # Obtaining the member 'format' of a type (line 635)
        format_11089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 29), str_11088, 'format')
        # Calling format(args, kwargs) (line 635)
        format_call_result_11099 = invoke(stypy.reporting.localization.Localization(__file__, 635, 29), format_11089, *[repr___call_result_11093, str_call_result_11097], **kwargs_11098)
        
        # Processing the call keyword arguments (line 634)
        kwargs_11100 = {}
        # Getting the type of 'TypeError' (line 634)
        TypeError_11086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 19), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 634)
        TypeError_call_result_11101 = invoke(stypy.reporting.localization.Localization(__file__, 634, 19), TypeError_11086, *[localization_11087, format_call_result_11099], **kwargs_11100)
        
        # Assigning a type to the variable 'stypy_return_type' (line 634)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 12), 'stypy_return_type', TypeError_call_result_11101)
        # SSA join for try-except statement (line 606)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to TypeError(...): (line 637)
        # Processing the call arguments (line 637)
        # Getting the type of 'localization' (line 637)
        localization_11103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 25), 'localization', False)
        str_11104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 25), 'str', 'Cannot modify the structure of a python library type or instance')
        # Processing the call keyword arguments (line 637)
        kwargs_11105 = {}
        # Getting the type of 'TypeError' (line 637)
        TypeError_11102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 15), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 637)
        TypeError_call_result_11106 = invoke(stypy.reporting.localization.Localization(__file__, 637, 15), TypeError_11102, *[localization_11103, str_11104], **kwargs_11105)
        
        # Assigning a type to the variable 'stypy_return_type' (line 637)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 8), 'stypy_return_type', TypeError_call_result_11106)
        
        # ################# End of 'set_type_of_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_type_of_member' in the type store
        # Getting the type of 'stypy_return_type' (line 596)
        stypy_return_type_11107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11107)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_type_of_member'
        return stypy_return_type_11107


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

        str_11108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, (-1)), 'str', '\n        Invoke a callable member of the hold python entity with the specified arguments and keyword arguments.\n        NOTE: Calling a class constructor returns a type proxy of an instance of this class. But an instance object\n        is only stored if the instances of this class support structural reflection.\n\n        :param localization: Call localization data\n        :param args: Arguments of the call\n        :param kwargs: Keyword arguments of the call\n        :return:\n        ')
        
        
        # Call to callable(...): (line 655)
        # Processing the call arguments (line 655)
        # Getting the type of 'self' (line 655)
        self_11110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 24), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 655)
        python_entity_11111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 24), self_11110, 'python_entity')
        # Processing the call keyword arguments (line 655)
        kwargs_11112 = {}
        # Getting the type of 'callable' (line 655)
        callable_11109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 15), 'callable', False)
        # Calling callable(args, kwargs) (line 655)
        callable_call_result_11113 = invoke(stypy.reporting.localization.Localization(__file__, 655, 15), callable_11109, *[python_entity_11111], **kwargs_11112)
        
        # Applying the 'not' unary operator (line 655)
        result_not__11114 = python_operator(stypy.reporting.localization.Localization(__file__, 655, 11), 'not', callable_call_result_11113)
        
        # Testing if the type of an if condition is none (line 655)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 655, 8), result_not__11114):
            
            # Assigning a Call to a Name (line 659):
            
            # Assigning a Call to a Name (line 659):
            
            # Call to perform_call(...): (line 659)
            # Processing the call arguments (line 659)
            # Getting the type of 'self' (line 659)
            self_11123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 54), 'self', False)
            # Getting the type of 'self' (line 659)
            self_11124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 60), 'self', False)
            # Obtaining the member 'python_entity' of a type (line 659)
            python_entity_11125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 60), self_11124, 'python_entity')
            # Getting the type of 'localization' (line 659)
            localization_11126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 80), 'localization', False)
            # Getting the type of 'args' (line 659)
            args_11127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 95), 'args', False)
            # Processing the call keyword arguments (line 659)
            # Getting the type of 'kwargs' (line 659)
            kwargs_11128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 103), 'kwargs', False)
            kwargs_11129 = {'kwargs_11128': kwargs_11128}
            # Getting the type of 'call_handlers_copy' (line 659)
            call_handlers_copy_11121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 22), 'call_handlers_copy', False)
            # Obtaining the member 'perform_call' of a type (line 659)
            perform_call_11122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 22), call_handlers_copy_11121, 'perform_call')
            # Calling perform_call(args, kwargs) (line 659)
            perform_call_call_result_11130 = invoke(stypy.reporting.localization.Localization(__file__, 659, 22), perform_call_11122, *[self_11123, python_entity_11125, localization_11126, args_11127], **kwargs_11129)
            
            # Assigning a type to the variable 'result_' (line 659)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 12), 'result_', perform_call_call_result_11130)
            
            # Evaluating a boolean operation
            
            # Call to is_type_changing_method(...): (line 661)
            # Processing the call arguments (line 661)
            # Getting the type of 'self' (line 661)
            self_11133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 60), 'self', False)
            # Obtaining the member 'name' of a type (line 661)
            name_11134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 60), self_11133, 'name')
            # Processing the call keyword arguments (line 661)
            kwargs_11135 = {}
            # Getting the type of 'TypeAnnotationRecord' (line 661)
            TypeAnnotationRecord_11131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 15), 'TypeAnnotationRecord', False)
            # Obtaining the member 'is_type_changing_method' of a type (line 661)
            is_type_changing_method_11132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 15), TypeAnnotationRecord_11131, 'is_type_changing_method')
            # Calling is_type_changing_method(args, kwargs) (line 661)
            is_type_changing_method_call_result_11136 = invoke(stypy.reporting.localization.Localization(__file__, 661, 15), is_type_changing_method_11132, *[name_11134], **kwargs_11135)
            
            # Getting the type of 'self' (line 661)
            self_11137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 75), 'self')
            # Obtaining the member 'annotate_types' of a type (line 661)
            annotate_types_11138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 75), self_11137, 'annotate_types')
            # Applying the binary operator 'and' (line 661)
            result_and_keyword_11139 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 15), 'and', is_type_changing_method_call_result_11136, annotate_types_11138)
            
            # Testing if the type of an if condition is none (line 661)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 661, 12), result_and_keyword_11139):
                pass
            else:
                
                # Testing the type of an if condition (line 661)
                if_condition_11140 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 661, 12), result_and_keyword_11139)
                # Assigning a type to the variable 'if_condition_11140' (line 661)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 12), 'if_condition_11140', if_condition_11140)
                # SSA begins for if statement (line 661)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to __annotate_type(...): (line 662)
                # Processing the call arguments (line 662)
                # Getting the type of 'localization' (line 662)
                localization_11143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 37), 'localization', False)
                # Obtaining the member 'line' of a type (line 662)
                line_11144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 37), localization_11143, 'line')
                # Getting the type of 'localization' (line 662)
                localization_11145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 56), 'localization', False)
                # Obtaining the member 'column' of a type (line 662)
                column_11146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 56), localization_11145, 'column')
                # Getting the type of 'self' (line 662)
                self_11147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 77), 'self', False)
                # Obtaining the member 'parent_proxy' of a type (line 662)
                parent_proxy_11148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 77), self_11147, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 662)
                name_11149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 77), parent_proxy_11148, 'name')
                
                # Call to get_python_type(...): (line 663)
                # Processing the call keyword arguments (line 663)
                kwargs_11153 = {}
                # Getting the type of 'self' (line 663)
                self_11150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 37), 'self', False)
                # Obtaining the member 'parent_proxy' of a type (line 663)
                parent_proxy_11151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 37), self_11150, 'parent_proxy')
                # Obtaining the member 'get_python_type' of a type (line 663)
                get_python_type_11152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 37), parent_proxy_11151, 'get_python_type')
                # Calling get_python_type(args, kwargs) (line 663)
                get_python_type_call_result_11154 = invoke(stypy.reporting.localization.Localization(__file__, 663, 37), get_python_type_11152, *[], **kwargs_11153)
                
                # Processing the call keyword arguments (line 662)
                kwargs_11155 = {}
                # Getting the type of 'self' (line 662)
                self_11141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 16), 'self', False)
                # Obtaining the member '__annotate_type' of a type (line 662)
                annotate_type_11142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 16), self_11141, '__annotate_type')
                # Calling __annotate_type(args, kwargs) (line 662)
                annotate_type_call_result_11156 = invoke(stypy.reporting.localization.Localization(__file__, 662, 16), annotate_type_11142, *[line_11144, column_11146, name_11149, get_python_type_call_result_11154], **kwargs_11155)
                
                # SSA join for if statement (line 661)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Type idiom detected: calculating its left and rigth part (line 666)
            # Getting the type of 'TypeError' (line 666)
            TypeError_11157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 35), 'TypeError')
            # Getting the type of 'result_' (line 666)
            result__11158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 26), 'result_')
            
            (may_be_11159, more_types_in_union_11160) = may_be_subtype(TypeError_11157, result__11158)

            if may_be_11159:

                if more_types_in_union_11160:
                    # Runtime conditional SSA (line 666)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'result_' (line 666)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 12), 'result_', remove_not_subtype_from_union(result__11158, TypeError))
                # Getting the type of 'result_' (line 667)
                result__11161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 23), 'result_')
                # Assigning a type to the variable 'stypy_return_type' (line 667)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 16), 'stypy_return_type', result__11161)

                if more_types_in_union_11160:
                    # SSA join for if statement (line 666)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Call to isinstance(...): (line 669)
            # Processing the call arguments (line 669)
            # Getting the type of 'result_' (line 669)
            result__11163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 26), 'result_', False)
            # Getting the type of 'Type' (line 669)
            Type_11164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 35), 'Type', False)
            # Processing the call keyword arguments (line 669)
            kwargs_11165 = {}
            # Getting the type of 'isinstance' (line 669)
            isinstance_11162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 669)
            isinstance_call_result_11166 = invoke(stypy.reporting.localization.Localization(__file__, 669, 15), isinstance_11162, *[result__11163, Type_11164], **kwargs_11165)
            
            # Testing if the type of an if condition is none (line 669)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 669, 12), isinstance_call_result_11166):
                pass
            else:
                
                # Testing the type of an if condition (line 669)
                if_condition_11167 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 669, 12), isinstance_call_result_11166)
                # Assigning a type to the variable 'if_condition_11167' (line 669)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 12), 'if_condition_11167', if_condition_11167)
                # SSA begins for if statement (line 669)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to set_type_instance(...): (line 670)
                # Processing the call arguments (line 670)
                # Getting the type of 'True' (line 670)
                True_11170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 42), 'True', False)
                # Processing the call keyword arguments (line 670)
                kwargs_11171 = {}
                # Getting the type of 'result_' (line 670)
                result__11168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 16), 'result_', False)
                # Obtaining the member 'set_type_instance' of a type (line 670)
                set_type_instance_11169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 16), result__11168, 'set_type_instance')
                # Calling set_type_instance(args, kwargs) (line 670)
                set_type_instance_call_result_11172 = invoke(stypy.reporting.localization.Localization(__file__, 670, 16), set_type_instance_11169, *[True_11170], **kwargs_11171)
                
                # Getting the type of 'result_' (line 671)
                result__11173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 23), 'result_')
                # Assigning a type to the variable 'stypy_return_type' (line 671)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 16), 'stypy_return_type', result__11173)
                # SSA join for if statement (line 669)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to isclass(...): (line 675)
            # Processing the call arguments (line 675)
            # Getting the type of 'self' (line 675)
            self_11176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 31), 'self', False)
            # Obtaining the member 'python_entity' of a type (line 675)
            python_entity_11177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 31), self_11176, 'python_entity')
            # Processing the call keyword arguments (line 675)
            kwargs_11178 = {}
            # Getting the type of 'inspect' (line 675)
            inspect_11174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 15), 'inspect', False)
            # Obtaining the member 'isclass' of a type (line 675)
            isclass_11175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 15), inspect_11174, 'isclass')
            # Calling isclass(args, kwargs) (line 675)
            isclass_call_result_11179 = invoke(stypy.reporting.localization.Localization(__file__, 675, 15), isclass_11175, *[python_entity_11177], **kwargs_11178)
            
            # Testing if the type of an if condition is none (line 675)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 675, 12), isclass_call_result_11179):
                
                # Assigning a Name to a Name (line 685):
                
                # Assigning a Name to a Name (line 685):
                # Getting the type of 'None' (line 685)
                None_11193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 27), 'None')
                # Assigning a type to the variable 'instance' (line 685)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 16), 'instance', None_11193)
            else:
                
                # Testing the type of an if condition (line 675)
                if_condition_11180 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 675, 12), isclass_call_result_11179)
                # Assigning a type to the variable 'if_condition_11180' (line 675)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 12), 'if_condition_11180', if_condition_11180)
                # SSA begins for if statement (line 675)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to supports_structural_reflection(...): (line 677)
                # Processing the call arguments (line 677)
                # Getting the type of 'result_' (line 677)
                result__11183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 87), 'result_', False)
                # Processing the call keyword arguments (line 677)
                kwargs_11184 = {}
                # Getting the type of 'type_inference_proxy_management_copy' (line 677)
                type_inference_proxy_management_copy_11181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 19), 'type_inference_proxy_management_copy', False)
                # Obtaining the member 'supports_structural_reflection' of a type (line 677)
                supports_structural_reflection_11182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 19), type_inference_proxy_management_copy_11181, 'supports_structural_reflection')
                # Calling supports_structural_reflection(args, kwargs) (line 677)
                supports_structural_reflection_call_result_11185 = invoke(stypy.reporting.localization.Localization(__file__, 677, 19), supports_structural_reflection_11182, *[result__11183], **kwargs_11184)
                
                # Testing if the type of an if condition is none (line 677)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 677, 16), supports_structural_reflection_call_result_11185):
                    
                    # Assigning a Name to a Name (line 683):
                    
                    # Assigning a Name to a Name (line 683):
                    # Getting the type of 'None' (line 683)
                    None_11192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 31), 'None')
                    # Assigning a type to the variable 'instance' (line 683)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 20), 'instance', None_11192)
                else:
                    
                    # Testing the type of an if condition (line 677)
                    if_condition_11186 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 677, 16), supports_structural_reflection_call_result_11185)
                    # Assigning a type to the variable 'if_condition_11186' (line 677)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 16), 'if_condition_11186', if_condition_11186)
                    # SSA begins for if statement (line 677)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Name (line 678):
                    
                    # Assigning a Name to a Name (line 678):
                    # Getting the type of 'result_' (line 678)
                    result__11187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 31), 'result_')
                    # Assigning a type to the variable 'instance' (line 678)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 20), 'instance', result__11187)
                    
                    # Assigning a Call to a Name (line 681):
                    
                    # Assigning a Call to a Name (line 681):
                    
                    # Call to type(...): (line 681)
                    # Processing the call arguments (line 681)
                    # Getting the type of 'result_' (line 681)
                    result__11189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 35), 'result_', False)
                    # Processing the call keyword arguments (line 681)
                    kwargs_11190 = {}
                    # Getting the type of 'type' (line 681)
                    type_11188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 30), 'type', False)
                    # Calling type(args, kwargs) (line 681)
                    type_call_result_11191 = invoke(stypy.reporting.localization.Localization(__file__, 681, 30), type_11188, *[result__11189], **kwargs_11190)
                    
                    # Assigning a type to the variable 'result_' (line 681)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 20), 'result_', type_call_result_11191)
                    # SSA branch for the else part of an if statement (line 677)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Name to a Name (line 683):
                    
                    # Assigning a Name to a Name (line 683):
                    # Getting the type of 'None' (line 683)
                    None_11192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 31), 'None')
                    # Assigning a type to the variable 'instance' (line 683)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 20), 'instance', None_11192)
                    # SSA join for if statement (line 677)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA branch for the else part of an if statement (line 675)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Name (line 685):
                
                # Assigning a Name to a Name (line 685):
                # Getting the type of 'None' (line 685)
                None_11193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 27), 'None')
                # Assigning a type to the variable 'instance' (line 685)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 16), 'instance', None_11193)
                # SSA join for if statement (line 675)
                module_type_store = module_type_store.join_ssa_context()
                

            
            
            # Call to isinstance(...): (line 688)
            # Processing the call arguments (line 688)
            # Getting the type of 'result_' (line 688)
            result__11195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 30), 'result_', False)
            # Getting the type of 'Type' (line 688)
            Type_11196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 39), 'Type', False)
            # Processing the call keyword arguments (line 688)
            kwargs_11197 = {}
            # Getting the type of 'isinstance' (line 688)
            isinstance_11194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 688)
            isinstance_call_result_11198 = invoke(stypy.reporting.localization.Localization(__file__, 688, 19), isinstance_11194, *[result__11195, Type_11196], **kwargs_11197)
            
            # Applying the 'not' unary operator (line 688)
            result_not__11199 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 15), 'not', isinstance_call_result_11198)
            
            # Testing if the type of an if condition is none (line 688)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 688, 12), result_not__11199):
                
                # Call to set_type_instance(...): (line 694)
                # Processing the call arguments (line 694)
                # Getting the type of 'True' (line 694)
                True_11216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 42), 'True', False)
                # Processing the call keyword arguments (line 694)
                kwargs_11217 = {}
                # Getting the type of 'result_' (line 694)
                result__11214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 16), 'result_', False)
                # Obtaining the member 'set_type_instance' of a type (line 694)
                set_type_instance_11215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 16), result__11214, 'set_type_instance')
                # Calling set_type_instance(args, kwargs) (line 694)
                set_type_instance_call_result_11218 = invoke(stypy.reporting.localization.Localization(__file__, 694, 16), set_type_instance_11215, *[True_11216], **kwargs_11217)
                
                # Getting the type of 'result_' (line 695)
                result__11219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 23), 'result_')
                # Assigning a type to the variable 'stypy_return_type' (line 695)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 16), 'stypy_return_type', result__11219)
            else:
                
                # Testing the type of an if condition (line 688)
                if_condition_11200 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 688, 12), result_not__11199)
                # Assigning a type to the variable 'if_condition_11200' (line 688)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 12), 'if_condition_11200', if_condition_11200)
                # SSA begins for if statement (line 688)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 689):
                
                # Assigning a Call to a Name (line 689):
                
                # Call to instance(...): (line 689)
                # Processing the call arguments (line 689)
                # Getting the type of 'result_' (line 689)
                result__11203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 50), 'result_', False)
                # Processing the call keyword arguments (line 689)
                # Getting the type of 'instance' (line 689)
                instance_11204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 68), 'instance', False)
                keyword_11205 = instance_11204
                kwargs_11206 = {'instance': keyword_11205}
                # Getting the type of 'TypeInferenceProxy' (line 689)
                TypeInferenceProxy_11201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 22), 'TypeInferenceProxy', False)
                # Obtaining the member 'instance' of a type (line 689)
                instance_11202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 22), TypeInferenceProxy_11201, 'instance')
                # Calling instance(args, kwargs) (line 689)
                instance_call_result_11207 = invoke(stypy.reporting.localization.Localization(__file__, 689, 22), instance_11202, *[result__11203], **kwargs_11206)
                
                # Assigning a type to the variable 'ret' (line 689)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 16), 'ret', instance_call_result_11207)
                
                # Call to set_type_instance(...): (line 690)
                # Processing the call arguments (line 690)
                # Getting the type of 'True' (line 690)
                True_11210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 38), 'True', False)
                # Processing the call keyword arguments (line 690)
                kwargs_11211 = {}
                # Getting the type of 'ret' (line 690)
                ret_11208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 16), 'ret', False)
                # Obtaining the member 'set_type_instance' of a type (line 690)
                set_type_instance_11209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 16), ret_11208, 'set_type_instance')
                # Calling set_type_instance(args, kwargs) (line 690)
                set_type_instance_call_result_11212 = invoke(stypy.reporting.localization.Localization(__file__, 690, 16), set_type_instance_11209, *[True_11210], **kwargs_11211)
                
                # Getting the type of 'ret' (line 692)
                ret_11213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 23), 'ret')
                # Assigning a type to the variable 'stypy_return_type' (line 692)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 16), 'stypy_return_type', ret_11213)
                # SSA branch for the else part of an if statement (line 688)
                module_type_store.open_ssa_branch('else')
                
                # Call to set_type_instance(...): (line 694)
                # Processing the call arguments (line 694)
                # Getting the type of 'True' (line 694)
                True_11216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 42), 'True', False)
                # Processing the call keyword arguments (line 694)
                kwargs_11217 = {}
                # Getting the type of 'result_' (line 694)
                result__11214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 16), 'result_', False)
                # Obtaining the member 'set_type_instance' of a type (line 694)
                set_type_instance_11215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 16), result__11214, 'set_type_instance')
                # Calling set_type_instance(args, kwargs) (line 694)
                set_type_instance_call_result_11218 = invoke(stypy.reporting.localization.Localization(__file__, 694, 16), set_type_instance_11215, *[True_11216], **kwargs_11217)
                
                # Getting the type of 'result_' (line 695)
                result__11219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 23), 'result_')
                # Assigning a type to the variable 'stypy_return_type' (line 695)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 16), 'stypy_return_type', result__11219)
                # SSA join for if statement (line 688)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 655)
            if_condition_11115 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 655, 8), result_not__11114)
            # Assigning a type to the variable 'if_condition_11115' (line 655)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 8), 'if_condition_11115', if_condition_11115)
            # SSA begins for if statement (line 655)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 656)
            # Processing the call arguments (line 656)
            # Getting the type of 'localization' (line 656)
            localization_11117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 29), 'localization', False)
            str_11118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 43), 'str', 'Cannot invoke on a non callable type')
            # Processing the call keyword arguments (line 656)
            kwargs_11119 = {}
            # Getting the type of 'TypeError' (line 656)
            TypeError_11116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 656)
            TypeError_call_result_11120 = invoke(stypy.reporting.localization.Localization(__file__, 656, 19), TypeError_11116, *[localization_11117, str_11118], **kwargs_11119)
            
            # Assigning a type to the variable 'stypy_return_type' (line 656)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 12), 'stypy_return_type', TypeError_call_result_11120)
            # SSA branch for the else part of an if statement (line 655)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 659):
            
            # Assigning a Call to a Name (line 659):
            
            # Call to perform_call(...): (line 659)
            # Processing the call arguments (line 659)
            # Getting the type of 'self' (line 659)
            self_11123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 54), 'self', False)
            # Getting the type of 'self' (line 659)
            self_11124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 60), 'self', False)
            # Obtaining the member 'python_entity' of a type (line 659)
            python_entity_11125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 60), self_11124, 'python_entity')
            # Getting the type of 'localization' (line 659)
            localization_11126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 80), 'localization', False)
            # Getting the type of 'args' (line 659)
            args_11127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 95), 'args', False)
            # Processing the call keyword arguments (line 659)
            # Getting the type of 'kwargs' (line 659)
            kwargs_11128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 103), 'kwargs', False)
            kwargs_11129 = {'kwargs_11128': kwargs_11128}
            # Getting the type of 'call_handlers_copy' (line 659)
            call_handlers_copy_11121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 22), 'call_handlers_copy', False)
            # Obtaining the member 'perform_call' of a type (line 659)
            perform_call_11122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 22), call_handlers_copy_11121, 'perform_call')
            # Calling perform_call(args, kwargs) (line 659)
            perform_call_call_result_11130 = invoke(stypy.reporting.localization.Localization(__file__, 659, 22), perform_call_11122, *[self_11123, python_entity_11125, localization_11126, args_11127], **kwargs_11129)
            
            # Assigning a type to the variable 'result_' (line 659)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 12), 'result_', perform_call_call_result_11130)
            
            # Evaluating a boolean operation
            
            # Call to is_type_changing_method(...): (line 661)
            # Processing the call arguments (line 661)
            # Getting the type of 'self' (line 661)
            self_11133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 60), 'self', False)
            # Obtaining the member 'name' of a type (line 661)
            name_11134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 60), self_11133, 'name')
            # Processing the call keyword arguments (line 661)
            kwargs_11135 = {}
            # Getting the type of 'TypeAnnotationRecord' (line 661)
            TypeAnnotationRecord_11131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 15), 'TypeAnnotationRecord', False)
            # Obtaining the member 'is_type_changing_method' of a type (line 661)
            is_type_changing_method_11132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 15), TypeAnnotationRecord_11131, 'is_type_changing_method')
            # Calling is_type_changing_method(args, kwargs) (line 661)
            is_type_changing_method_call_result_11136 = invoke(stypy.reporting.localization.Localization(__file__, 661, 15), is_type_changing_method_11132, *[name_11134], **kwargs_11135)
            
            # Getting the type of 'self' (line 661)
            self_11137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 75), 'self')
            # Obtaining the member 'annotate_types' of a type (line 661)
            annotate_types_11138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 75), self_11137, 'annotate_types')
            # Applying the binary operator 'and' (line 661)
            result_and_keyword_11139 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 15), 'and', is_type_changing_method_call_result_11136, annotate_types_11138)
            
            # Testing if the type of an if condition is none (line 661)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 661, 12), result_and_keyword_11139):
                pass
            else:
                
                # Testing the type of an if condition (line 661)
                if_condition_11140 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 661, 12), result_and_keyword_11139)
                # Assigning a type to the variable 'if_condition_11140' (line 661)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 12), 'if_condition_11140', if_condition_11140)
                # SSA begins for if statement (line 661)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to __annotate_type(...): (line 662)
                # Processing the call arguments (line 662)
                # Getting the type of 'localization' (line 662)
                localization_11143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 37), 'localization', False)
                # Obtaining the member 'line' of a type (line 662)
                line_11144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 37), localization_11143, 'line')
                # Getting the type of 'localization' (line 662)
                localization_11145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 56), 'localization', False)
                # Obtaining the member 'column' of a type (line 662)
                column_11146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 56), localization_11145, 'column')
                # Getting the type of 'self' (line 662)
                self_11147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 77), 'self', False)
                # Obtaining the member 'parent_proxy' of a type (line 662)
                parent_proxy_11148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 77), self_11147, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 662)
                name_11149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 77), parent_proxy_11148, 'name')
                
                # Call to get_python_type(...): (line 663)
                # Processing the call keyword arguments (line 663)
                kwargs_11153 = {}
                # Getting the type of 'self' (line 663)
                self_11150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 37), 'self', False)
                # Obtaining the member 'parent_proxy' of a type (line 663)
                parent_proxy_11151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 37), self_11150, 'parent_proxy')
                # Obtaining the member 'get_python_type' of a type (line 663)
                get_python_type_11152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 37), parent_proxy_11151, 'get_python_type')
                # Calling get_python_type(args, kwargs) (line 663)
                get_python_type_call_result_11154 = invoke(stypy.reporting.localization.Localization(__file__, 663, 37), get_python_type_11152, *[], **kwargs_11153)
                
                # Processing the call keyword arguments (line 662)
                kwargs_11155 = {}
                # Getting the type of 'self' (line 662)
                self_11141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 16), 'self', False)
                # Obtaining the member '__annotate_type' of a type (line 662)
                annotate_type_11142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 16), self_11141, '__annotate_type')
                # Calling __annotate_type(args, kwargs) (line 662)
                annotate_type_call_result_11156 = invoke(stypy.reporting.localization.Localization(__file__, 662, 16), annotate_type_11142, *[line_11144, column_11146, name_11149, get_python_type_call_result_11154], **kwargs_11155)
                
                # SSA join for if statement (line 661)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Type idiom detected: calculating its left and rigth part (line 666)
            # Getting the type of 'TypeError' (line 666)
            TypeError_11157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 35), 'TypeError')
            # Getting the type of 'result_' (line 666)
            result__11158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 26), 'result_')
            
            (may_be_11159, more_types_in_union_11160) = may_be_subtype(TypeError_11157, result__11158)

            if may_be_11159:

                if more_types_in_union_11160:
                    # Runtime conditional SSA (line 666)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'result_' (line 666)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 12), 'result_', remove_not_subtype_from_union(result__11158, TypeError))
                # Getting the type of 'result_' (line 667)
                result__11161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 23), 'result_')
                # Assigning a type to the variable 'stypy_return_type' (line 667)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 16), 'stypy_return_type', result__11161)

                if more_types_in_union_11160:
                    # SSA join for if statement (line 666)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Call to isinstance(...): (line 669)
            # Processing the call arguments (line 669)
            # Getting the type of 'result_' (line 669)
            result__11163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 26), 'result_', False)
            # Getting the type of 'Type' (line 669)
            Type_11164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 35), 'Type', False)
            # Processing the call keyword arguments (line 669)
            kwargs_11165 = {}
            # Getting the type of 'isinstance' (line 669)
            isinstance_11162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 669)
            isinstance_call_result_11166 = invoke(stypy.reporting.localization.Localization(__file__, 669, 15), isinstance_11162, *[result__11163, Type_11164], **kwargs_11165)
            
            # Testing if the type of an if condition is none (line 669)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 669, 12), isinstance_call_result_11166):
                pass
            else:
                
                # Testing the type of an if condition (line 669)
                if_condition_11167 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 669, 12), isinstance_call_result_11166)
                # Assigning a type to the variable 'if_condition_11167' (line 669)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 12), 'if_condition_11167', if_condition_11167)
                # SSA begins for if statement (line 669)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to set_type_instance(...): (line 670)
                # Processing the call arguments (line 670)
                # Getting the type of 'True' (line 670)
                True_11170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 42), 'True', False)
                # Processing the call keyword arguments (line 670)
                kwargs_11171 = {}
                # Getting the type of 'result_' (line 670)
                result__11168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 16), 'result_', False)
                # Obtaining the member 'set_type_instance' of a type (line 670)
                set_type_instance_11169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 16), result__11168, 'set_type_instance')
                # Calling set_type_instance(args, kwargs) (line 670)
                set_type_instance_call_result_11172 = invoke(stypy.reporting.localization.Localization(__file__, 670, 16), set_type_instance_11169, *[True_11170], **kwargs_11171)
                
                # Getting the type of 'result_' (line 671)
                result__11173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 23), 'result_')
                # Assigning a type to the variable 'stypy_return_type' (line 671)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 16), 'stypy_return_type', result__11173)
                # SSA join for if statement (line 669)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to isclass(...): (line 675)
            # Processing the call arguments (line 675)
            # Getting the type of 'self' (line 675)
            self_11176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 31), 'self', False)
            # Obtaining the member 'python_entity' of a type (line 675)
            python_entity_11177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 31), self_11176, 'python_entity')
            # Processing the call keyword arguments (line 675)
            kwargs_11178 = {}
            # Getting the type of 'inspect' (line 675)
            inspect_11174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 15), 'inspect', False)
            # Obtaining the member 'isclass' of a type (line 675)
            isclass_11175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 15), inspect_11174, 'isclass')
            # Calling isclass(args, kwargs) (line 675)
            isclass_call_result_11179 = invoke(stypy.reporting.localization.Localization(__file__, 675, 15), isclass_11175, *[python_entity_11177], **kwargs_11178)
            
            # Testing if the type of an if condition is none (line 675)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 675, 12), isclass_call_result_11179):
                
                # Assigning a Name to a Name (line 685):
                
                # Assigning a Name to a Name (line 685):
                # Getting the type of 'None' (line 685)
                None_11193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 27), 'None')
                # Assigning a type to the variable 'instance' (line 685)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 16), 'instance', None_11193)
            else:
                
                # Testing the type of an if condition (line 675)
                if_condition_11180 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 675, 12), isclass_call_result_11179)
                # Assigning a type to the variable 'if_condition_11180' (line 675)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 12), 'if_condition_11180', if_condition_11180)
                # SSA begins for if statement (line 675)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to supports_structural_reflection(...): (line 677)
                # Processing the call arguments (line 677)
                # Getting the type of 'result_' (line 677)
                result__11183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 87), 'result_', False)
                # Processing the call keyword arguments (line 677)
                kwargs_11184 = {}
                # Getting the type of 'type_inference_proxy_management_copy' (line 677)
                type_inference_proxy_management_copy_11181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 19), 'type_inference_proxy_management_copy', False)
                # Obtaining the member 'supports_structural_reflection' of a type (line 677)
                supports_structural_reflection_11182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 19), type_inference_proxy_management_copy_11181, 'supports_structural_reflection')
                # Calling supports_structural_reflection(args, kwargs) (line 677)
                supports_structural_reflection_call_result_11185 = invoke(stypy.reporting.localization.Localization(__file__, 677, 19), supports_structural_reflection_11182, *[result__11183], **kwargs_11184)
                
                # Testing if the type of an if condition is none (line 677)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 677, 16), supports_structural_reflection_call_result_11185):
                    
                    # Assigning a Name to a Name (line 683):
                    
                    # Assigning a Name to a Name (line 683):
                    # Getting the type of 'None' (line 683)
                    None_11192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 31), 'None')
                    # Assigning a type to the variable 'instance' (line 683)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 20), 'instance', None_11192)
                else:
                    
                    # Testing the type of an if condition (line 677)
                    if_condition_11186 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 677, 16), supports_structural_reflection_call_result_11185)
                    # Assigning a type to the variable 'if_condition_11186' (line 677)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 16), 'if_condition_11186', if_condition_11186)
                    # SSA begins for if statement (line 677)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Name (line 678):
                    
                    # Assigning a Name to a Name (line 678):
                    # Getting the type of 'result_' (line 678)
                    result__11187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 31), 'result_')
                    # Assigning a type to the variable 'instance' (line 678)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 20), 'instance', result__11187)
                    
                    # Assigning a Call to a Name (line 681):
                    
                    # Assigning a Call to a Name (line 681):
                    
                    # Call to type(...): (line 681)
                    # Processing the call arguments (line 681)
                    # Getting the type of 'result_' (line 681)
                    result__11189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 35), 'result_', False)
                    # Processing the call keyword arguments (line 681)
                    kwargs_11190 = {}
                    # Getting the type of 'type' (line 681)
                    type_11188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 30), 'type', False)
                    # Calling type(args, kwargs) (line 681)
                    type_call_result_11191 = invoke(stypy.reporting.localization.Localization(__file__, 681, 30), type_11188, *[result__11189], **kwargs_11190)
                    
                    # Assigning a type to the variable 'result_' (line 681)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 20), 'result_', type_call_result_11191)
                    # SSA branch for the else part of an if statement (line 677)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Name to a Name (line 683):
                    
                    # Assigning a Name to a Name (line 683):
                    # Getting the type of 'None' (line 683)
                    None_11192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 31), 'None')
                    # Assigning a type to the variable 'instance' (line 683)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 20), 'instance', None_11192)
                    # SSA join for if statement (line 677)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA branch for the else part of an if statement (line 675)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Name (line 685):
                
                # Assigning a Name to a Name (line 685):
                # Getting the type of 'None' (line 685)
                None_11193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 27), 'None')
                # Assigning a type to the variable 'instance' (line 685)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 16), 'instance', None_11193)
                # SSA join for if statement (line 675)
                module_type_store = module_type_store.join_ssa_context()
                

            
            
            # Call to isinstance(...): (line 688)
            # Processing the call arguments (line 688)
            # Getting the type of 'result_' (line 688)
            result__11195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 30), 'result_', False)
            # Getting the type of 'Type' (line 688)
            Type_11196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 39), 'Type', False)
            # Processing the call keyword arguments (line 688)
            kwargs_11197 = {}
            # Getting the type of 'isinstance' (line 688)
            isinstance_11194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 688)
            isinstance_call_result_11198 = invoke(stypy.reporting.localization.Localization(__file__, 688, 19), isinstance_11194, *[result__11195, Type_11196], **kwargs_11197)
            
            # Applying the 'not' unary operator (line 688)
            result_not__11199 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 15), 'not', isinstance_call_result_11198)
            
            # Testing if the type of an if condition is none (line 688)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 688, 12), result_not__11199):
                
                # Call to set_type_instance(...): (line 694)
                # Processing the call arguments (line 694)
                # Getting the type of 'True' (line 694)
                True_11216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 42), 'True', False)
                # Processing the call keyword arguments (line 694)
                kwargs_11217 = {}
                # Getting the type of 'result_' (line 694)
                result__11214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 16), 'result_', False)
                # Obtaining the member 'set_type_instance' of a type (line 694)
                set_type_instance_11215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 16), result__11214, 'set_type_instance')
                # Calling set_type_instance(args, kwargs) (line 694)
                set_type_instance_call_result_11218 = invoke(stypy.reporting.localization.Localization(__file__, 694, 16), set_type_instance_11215, *[True_11216], **kwargs_11217)
                
                # Getting the type of 'result_' (line 695)
                result__11219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 23), 'result_')
                # Assigning a type to the variable 'stypy_return_type' (line 695)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 16), 'stypy_return_type', result__11219)
            else:
                
                # Testing the type of an if condition (line 688)
                if_condition_11200 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 688, 12), result_not__11199)
                # Assigning a type to the variable 'if_condition_11200' (line 688)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 12), 'if_condition_11200', if_condition_11200)
                # SSA begins for if statement (line 688)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 689):
                
                # Assigning a Call to a Name (line 689):
                
                # Call to instance(...): (line 689)
                # Processing the call arguments (line 689)
                # Getting the type of 'result_' (line 689)
                result__11203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 50), 'result_', False)
                # Processing the call keyword arguments (line 689)
                # Getting the type of 'instance' (line 689)
                instance_11204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 68), 'instance', False)
                keyword_11205 = instance_11204
                kwargs_11206 = {'instance': keyword_11205}
                # Getting the type of 'TypeInferenceProxy' (line 689)
                TypeInferenceProxy_11201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 22), 'TypeInferenceProxy', False)
                # Obtaining the member 'instance' of a type (line 689)
                instance_11202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 22), TypeInferenceProxy_11201, 'instance')
                # Calling instance(args, kwargs) (line 689)
                instance_call_result_11207 = invoke(stypy.reporting.localization.Localization(__file__, 689, 22), instance_11202, *[result__11203], **kwargs_11206)
                
                # Assigning a type to the variable 'ret' (line 689)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 16), 'ret', instance_call_result_11207)
                
                # Call to set_type_instance(...): (line 690)
                # Processing the call arguments (line 690)
                # Getting the type of 'True' (line 690)
                True_11210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 38), 'True', False)
                # Processing the call keyword arguments (line 690)
                kwargs_11211 = {}
                # Getting the type of 'ret' (line 690)
                ret_11208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 16), 'ret', False)
                # Obtaining the member 'set_type_instance' of a type (line 690)
                set_type_instance_11209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 16), ret_11208, 'set_type_instance')
                # Calling set_type_instance(args, kwargs) (line 690)
                set_type_instance_call_result_11212 = invoke(stypy.reporting.localization.Localization(__file__, 690, 16), set_type_instance_11209, *[True_11210], **kwargs_11211)
                
                # Getting the type of 'ret' (line 692)
                ret_11213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 23), 'ret')
                # Assigning a type to the variable 'stypy_return_type' (line 692)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 16), 'stypy_return_type', ret_11213)
                # SSA branch for the else part of an if statement (line 688)
                module_type_store.open_ssa_branch('else')
                
                # Call to set_type_instance(...): (line 694)
                # Processing the call arguments (line 694)
                # Getting the type of 'True' (line 694)
                True_11216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 42), 'True', False)
                # Processing the call keyword arguments (line 694)
                kwargs_11217 = {}
                # Getting the type of 'result_' (line 694)
                result__11214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 16), 'result_', False)
                # Obtaining the member 'set_type_instance' of a type (line 694)
                set_type_instance_11215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 16), result__11214, 'set_type_instance')
                # Calling set_type_instance(args, kwargs) (line 694)
                set_type_instance_call_result_11218 = invoke(stypy.reporting.localization.Localization(__file__, 694, 16), set_type_instance_11215, *[True_11216], **kwargs_11217)
                
                # Getting the type of 'result_' (line 695)
                result__11219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 23), 'result_')
                # Assigning a type to the variable 'stypy_return_type' (line 695)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 16), 'stypy_return_type', result__11219)
                # SSA join for if statement (line 688)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 655)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'invoke(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'invoke' in the type store
        # Getting the type of 'stypy_return_type' (line 642)
        stypy_return_type_11220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11220)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'invoke'
        return stypy_return_type_11220


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

        str_11221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, (-1)), 'str', '\n        For represented containers, this method checks if we are trying to store Undefined variables inside them\n        :param localization: Caller information\n        :param value: Value we are trying to store\n        :return:\n        ')
        
        # Assigning a Call to a Tuple (line 706):
        
        # Assigning a Call to a Name:
        
        # Call to contains_an_undefined_type(...): (line 706)
        # Processing the call arguments (line 706)
        # Getting the type of 'value' (line 706)
        value_11224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 96), 'value', False)
        # Processing the call keyword arguments (line 706)
        kwargs_11225 = {}
        # Getting the type of 'TypeInferenceProxy' (line 706)
        TypeInferenceProxy_11222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 50), 'TypeInferenceProxy', False)
        # Obtaining the member 'contains_an_undefined_type' of a type (line 706)
        contains_an_undefined_type_11223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 50), TypeInferenceProxy_11222, 'contains_an_undefined_type')
        # Calling contains_an_undefined_type(args, kwargs) (line 706)
        contains_an_undefined_type_call_result_11226 = invoke(stypy.reporting.localization.Localization(__file__, 706, 50), contains_an_undefined_type_11223, *[value_11224], **kwargs_11225)
        
        # Assigning a type to the variable 'call_assignment_10134' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'call_assignment_10134', contains_an_undefined_type_call_result_11226)
        
        # Assigning a Call to a Name (line 706):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_10134' (line 706)
        call_assignment_10134_11227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'call_assignment_10134', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_11228 = stypy_get_value_from_tuple(call_assignment_10134_11227, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_10135' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'call_assignment_10135', stypy_get_value_from_tuple_call_result_11228)
        
        # Assigning a Name to a Name (line 706):
        # Getting the type of 'call_assignment_10135' (line 706)
        call_assignment_10135_11229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'call_assignment_10135')
        # Assigning a type to the variable 'contains_undefined' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'contains_undefined', call_assignment_10135_11229)
        
        # Assigning a Call to a Name (line 706):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_10134' (line 706)
        call_assignment_10134_11230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'call_assignment_10134', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_11231 = stypy_get_value_from_tuple(call_assignment_10134_11230, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_10136' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'call_assignment_10136', stypy_get_value_from_tuple_call_result_11231)
        
        # Assigning a Name to a Name (line 706):
        # Getting the type of 'call_assignment_10136' (line 706)
        call_assignment_10136_11232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'call_assignment_10136')
        # Assigning a type to the variable 'more_types_in_value' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 28), 'more_types_in_value', call_assignment_10136_11232)
        # Getting the type of 'contains_undefined' (line 707)
        contains_undefined_11233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 11), 'contains_undefined')
        # Testing if the type of an if condition is none (line 707)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 707, 8), contains_undefined_11233):
            pass
        else:
            
            # Testing the type of an if condition (line 707)
            if_condition_11234 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 707, 8), contains_undefined_11233)
            # Assigning a type to the variable 'if_condition_11234' (line 707)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 8), 'if_condition_11234', if_condition_11234)
            # SSA begins for if statement (line 707)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'more_types_in_value' (line 708)
            more_types_in_value_11235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 15), 'more_types_in_value')
            int_11236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 38), 'int')
            # Applying the binary operator '==' (line 708)
            result_eq_11237 = python_operator(stypy.reporting.localization.Localization(__file__, 708, 15), '==', more_types_in_value_11235, int_11236)
            
            # Testing if the type of an if condition is none (line 708)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 708, 12), result_eq_11237):
                
                # Call to instance(...): (line 712)
                # Processing the call arguments (line 712)
                # Getting the type of 'localization' (line 712)
                localization_11251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 37), 'localization', False)
                
                # Call to format(...): (line 713)
                # Processing the call arguments (line 713)
                # Getting the type of 'self' (line 714)
                self_11254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 44), 'self', False)
                # Obtaining the member 'name' of a type (line 714)
                name_11255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 44), self_11254, 'name')
                # Processing the call keyword arguments (line 713)
                kwargs_11256 = {}
                str_11252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 37), 'str', "Potentially storing in '{0}' the value of a previously undefined variable")
                # Obtaining the member 'format' of a type (line 713)
                format_11253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 37), str_11252, 'format')
                # Calling format(args, kwargs) (line 713)
                format_call_result_11257 = invoke(stypy.reporting.localization.Localization(__file__, 713, 37), format_11253, *[name_11255], **kwargs_11256)
                
                # Processing the call keyword arguments (line 712)
                kwargs_11258 = {}
                # Getting the type of 'TypeWarning' (line 712)
                TypeWarning_11249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 16), 'TypeWarning', False)
                # Obtaining the member 'instance' of a type (line 712)
                instance_11250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 16), TypeWarning_11249, 'instance')
                # Calling instance(args, kwargs) (line 712)
                instance_call_result_11259 = invoke(stypy.reporting.localization.Localization(__file__, 712, 16), instance_11250, *[localization_11251, format_call_result_11257], **kwargs_11258)
                
            else:
                
                # Testing the type of an if condition (line 708)
                if_condition_11238 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 708, 12), result_eq_11237)
                # Assigning a type to the variable 'if_condition_11238' (line 708)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 12), 'if_condition_11238', if_condition_11238)
                # SSA begins for if statement (line 708)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to TypeError(...): (line 709)
                # Processing the call arguments (line 709)
                # Getting the type of 'localization' (line 709)
                localization_11240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 26), 'localization', False)
                
                # Call to format(...): (line 709)
                # Processing the call arguments (line 709)
                # Getting the type of 'self' (line 710)
                self_11243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 33), 'self', False)
                # Obtaining the member 'name' of a type (line 710)
                name_11244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 33), self_11243, 'name')
                # Processing the call keyword arguments (line 709)
                kwargs_11245 = {}
                str_11241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 40), 'str', "Storing in '{0}' the value of a previously undefined variable")
                # Obtaining the member 'format' of a type (line 709)
                format_11242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 40), str_11241, 'format')
                # Calling format(args, kwargs) (line 709)
                format_call_result_11246 = invoke(stypy.reporting.localization.Localization(__file__, 709, 40), format_11242, *[name_11244], **kwargs_11245)
                
                # Processing the call keyword arguments (line 709)
                kwargs_11247 = {}
                # Getting the type of 'TypeError' (line 709)
                TypeError_11239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 16), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 709)
                TypeError_call_result_11248 = invoke(stypy.reporting.localization.Localization(__file__, 709, 16), TypeError_11239, *[localization_11240, format_call_result_11246], **kwargs_11247)
                
                # SSA branch for the else part of an if statement (line 708)
                module_type_store.open_ssa_branch('else')
                
                # Call to instance(...): (line 712)
                # Processing the call arguments (line 712)
                # Getting the type of 'localization' (line 712)
                localization_11251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 37), 'localization', False)
                
                # Call to format(...): (line 713)
                # Processing the call arguments (line 713)
                # Getting the type of 'self' (line 714)
                self_11254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 44), 'self', False)
                # Obtaining the member 'name' of a type (line 714)
                name_11255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 44), self_11254, 'name')
                # Processing the call keyword arguments (line 713)
                kwargs_11256 = {}
                str_11252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 37), 'str', "Potentially storing in '{0}' the value of a previously undefined variable")
                # Obtaining the member 'format' of a type (line 713)
                format_11253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 37), str_11252, 'format')
                # Calling format(args, kwargs) (line 713)
                format_call_result_11257 = invoke(stypy.reporting.localization.Localization(__file__, 713, 37), format_11253, *[name_11255], **kwargs_11256)
                
                # Processing the call keyword arguments (line 712)
                kwargs_11258 = {}
                # Getting the type of 'TypeWarning' (line 712)
                TypeWarning_11249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 16), 'TypeWarning', False)
                # Obtaining the member 'instance' of a type (line 712)
                instance_11250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 16), TypeWarning_11249, 'instance')
                # Calling instance(args, kwargs) (line 712)
                instance_call_result_11259 = invoke(stypy.reporting.localization.Localization(__file__, 712, 16), instance_11250, *[localization_11251, format_call_result_11257], **kwargs_11258)
                
                # SSA join for if statement (line 708)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 707)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Obtaining an instance of the builtin type 'tuple' (line 715)
        tuple_11260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 715)
        # Adding element type (line 715)
        # Getting the type of 'contains_undefined' (line 715)
        contains_undefined_11261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 15), 'contains_undefined')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 715, 15), tuple_11260, contains_undefined_11261)
        # Adding element type (line 715)
        # Getting the type of 'more_types_in_value' (line 715)
        more_types_in_value_11262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 35), 'more_types_in_value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 715, 15), tuple_11260, more_types_in_value_11262)
        
        # Assigning a type to the variable 'stypy_return_type' (line 715)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 8), 'stypy_return_type', tuple_11260)
        
        # ################# End of '__check_undefined_stored_value(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__check_undefined_stored_value' in the type store
        # Getting the type of 'stypy_return_type' (line 699)
        stypy_return_type_11263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11263)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__check_undefined_stored_value'
        return stypy_return_type_11263


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

        str_11264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, (-1)), 'str', '\n        Determines if this proxy represents a Python type able to store elements (lists, tuples, ...)\n        :return: bool\n        ')
        
        # Assigning a BoolOp to a Name (line 722):
        
        # Assigning a BoolOp to a Name (line 722):
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        
        str_11265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 23), 'str', 'dictionary-')
        # Getting the type of 'self' (line 722)
        self_11266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 40), 'self')
        # Obtaining the member 'name' of a type (line 722)
        name_11267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 40), self_11266, 'name')
        # Applying the binary operator 'in' (line 722)
        result_contains_11268 = python_operator(stypy.reporting.localization.Localization(__file__, 722, 23), 'in', str_11265, name_11267)
        
        
        str_11269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 54), 'str', 'iterator')
        # Getting the type of 'self' (line 722)
        self_11270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 68), 'self')
        # Obtaining the member 'name' of a type (line 722)
        name_11271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 68), self_11270, 'name')
        # Applying the binary operator 'in' (line 722)
        result_contains_11272 = python_operator(stypy.reporting.localization.Localization(__file__, 722, 54), 'in', str_11269, name_11271)
        
        # Applying the binary operator 'and' (line 722)
        result_and_keyword_11273 = python_operator(stypy.reporting.localization.Localization(__file__, 722, 23), 'and', result_contains_11268, result_contains_11272)
        
        
        # Evaluating a boolean operation
        
        str_11274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 83), 'str', 'iterator')
        # Getting the type of 'self' (line 722)
        self_11275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 97), 'self')
        # Obtaining the member 'name' of a type (line 722)
        name_11276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 97), self_11275, 'name')
        # Applying the binary operator 'in' (line 722)
        result_contains_11277 = python_operator(stypy.reporting.localization.Localization(__file__, 722, 83), 'in', str_11274, name_11276)
        
        
        str_11278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 83), 'str', 'dict')
        # Getting the type of 'self' (line 723)
        self_11279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 97), 'self')
        # Obtaining the member 'name' of a type (line 723)
        name_11280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 97), self_11279, 'name')
        # Applying the binary operator 'notin' (line 723)
        result_contains_11281 = python_operator(stypy.reporting.localization.Localization(__file__, 723, 83), 'notin', str_11278, name_11280)
        
        # Applying the binary operator 'and' (line 722)
        result_and_keyword_11282 = python_operator(stypy.reporting.localization.Localization(__file__, 722, 83), 'and', result_contains_11277, result_contains_11281)
        
        # Applying the binary operator 'or' (line 722)
        result_or_keyword_11283 = python_operator(stypy.reporting.localization.Localization(__file__, 722, 22), 'or', result_and_keyword_11273, result_and_keyword_11282)
        
        # Assigning a type to the variable 'is_iterator' (line 722)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 8), 'is_iterator', result_or_keyword_11283)
        
        # Assigning a List to a Name (line 725):
        
        # Assigning a List to a Name (line 725):
        
        # Obtaining an instance of the builtin type 'list' (line 725)
        list_11284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 725)
        # Adding element type (line 725)
        # Getting the type of 'list' (line 725)
        list_11285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 27), 'list')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 26), list_11284, list_11285)
        # Adding element type (line 725)
        # Getting the type of 'set' (line 725)
        set_11286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 33), 'set')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 26), list_11284, set_11286)
        # Adding element type (line 725)
        # Getting the type of 'tuple' (line 725)
        tuple_11287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 38), 'tuple')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 26), list_11284, tuple_11287)
        # Adding element type (line 725)
        # Getting the type of 'types' (line 725)
        types_11288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 45), 'types')
        # Obtaining the member 'GeneratorType' of a type (line 725)
        GeneratorType_11289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 45), types_11288, 'GeneratorType')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 26), list_11284, GeneratorType_11289)
        # Adding element type (line 725)
        # Getting the type of 'bytearray' (line 725)
        bytearray_11290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 66), 'bytearray')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 26), list_11284, bytearray_11290)
        # Adding element type (line 725)
        # Getting the type of 'slice' (line 725)
        slice_11291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 77), 'slice')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 26), list_11284, slice_11291)
        # Adding element type (line 725)
        # Getting the type of 'range' (line 725)
        range_11292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 84), 'range')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 26), list_11284, range_11292)
        # Adding element type (line 725)
        # Getting the type of 'xrange' (line 725)
        xrange_11293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 91), 'xrange')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 26), list_11284, xrange_11293)
        # Adding element type (line 725)
        # Getting the type of 'enumerate' (line 725)
        enumerate_11294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 99), 'enumerate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 26), list_11284, enumerate_11294)
        # Adding element type (line 725)
        # Getting the type of 'reversed' (line 725)
        reversed_11295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 110), 'reversed')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 26), list_11284, reversed_11295)
        # Adding element type (line 725)
        # Getting the type of 'frozenset' (line 726)
        frozenset_11296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 27), 'frozenset')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 26), list_11284, frozenset_11296)
        
        # Assigning a type to the variable 'data_structures' (line 725)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 725, 8), 'data_structures', list_11284)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 727)
        self_11297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 16), 'self')
        # Obtaining the member 'python_entity' of a type (line 727)
        python_entity_11298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 16), self_11297, 'python_entity')
        # Getting the type of 'data_structures' (line 727)
        data_structures_11299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 38), 'data_structures')
        # Applying the binary operator 'in' (line 727)
        result_contains_11300 = python_operator(stypy.reporting.localization.Localization(__file__, 727, 16), 'in', python_entity_11298, data_structures_11299)
        
        # Getting the type of 'is_iterator' (line 727)
        is_iterator_11301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 58), 'is_iterator')
        # Applying the binary operator 'or' (line 727)
        result_or_keyword_11302 = python_operator(stypy.reporting.localization.Localization(__file__, 727, 15), 'or', result_contains_11300, is_iterator_11301)
        
        # Assigning a type to the variable 'stypy_return_type' (line 727)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 8), 'stypy_return_type', result_or_keyword_11302)
        
        # ################# End of 'can_store_elements(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'can_store_elements' in the type store
        # Getting the type of 'stypy_return_type' (line 717)
        stypy_return_type_11303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11303)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'can_store_elements'
        return stypy_return_type_11303


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

        str_11304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, (-1)), 'str', '\n        Determines if this proxy represents a Python type able to store keypairs (dict, dict iterators)\n        :return: bool\n        ')
        
        # Assigning a BoolOp to a Name (line 734):
        
        # Assigning a BoolOp to a Name (line 734):
        
        # Evaluating a boolean operation
        
        str_11305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 22), 'str', 'iterator')
        # Getting the type of 'self' (line 734)
        self_11306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 36), 'self')
        # Obtaining the member 'name' of a type (line 734)
        name_11307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 36), self_11306, 'name')
        # Applying the binary operator 'in' (line 734)
        result_contains_11308 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 22), 'in', str_11305, name_11307)
        
        
        str_11309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 50), 'str', 'dict')
        # Getting the type of 'self' (line 734)
        self_11310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 60), 'self')
        # Obtaining the member 'name' of a type (line 734)
        name_11311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 60), self_11310, 'name')
        # Applying the binary operator 'in' (line 734)
        result_contains_11312 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 50), 'in', str_11309, name_11311)
        
        # Applying the binary operator 'and' (line 734)
        result_and_keyword_11313 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 22), 'and', result_contains_11308, result_contains_11312)
        
        # Assigning a type to the variable 'is_iterator' (line 734)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 8), 'is_iterator', result_and_keyword_11313)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 736)
        self_11314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 15), 'self')
        # Obtaining the member 'python_entity' of a type (line 736)
        python_entity_11315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 15), self_11314, 'python_entity')
        # Getting the type of 'dict' (line 736)
        dict_11316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 37), 'dict')
        # Applying the binary operator 'is' (line 736)
        result_is__11317 = python_operator(stypy.reporting.localization.Localization(__file__, 736, 15), 'is', python_entity_11315, dict_11316)
        
        # Getting the type of 'is_iterator' (line 736)
        is_iterator_11318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 45), 'is_iterator')
        # Applying the binary operator 'or' (line 736)
        result_or_keyword_11319 = python_operator(stypy.reporting.localization.Localization(__file__, 736, 15), 'or', result_is__11317, is_iterator_11318)
        
        # Assigning a type to the variable 'stypy_return_type' (line 736)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 8), 'stypy_return_type', result_or_keyword_11319)
        
        # ################# End of 'can_store_keypairs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'can_store_keypairs' in the type store
        # Getting the type of 'stypy_return_type' (line 729)
        stypy_return_type_11320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11320)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'can_store_keypairs'
        return stypy_return_type_11320


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

        str_11321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, (-1)), 'str', '\n        Determines if a proxy able to store elements can be considered empty (no elements were inserted through its\n        lifespan\n        :return: None or TypeError\n        ')
        
        # Evaluating a boolean operation
        
        
        # Call to can_store_elements(...): (line 744)
        # Processing the call keyword arguments (line 744)
        kwargs_11324 = {}
        # Getting the type of 'self' (line 744)
        self_11322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 15), 'self', False)
        # Obtaining the member 'can_store_elements' of a type (line 744)
        can_store_elements_11323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 15), self_11322, 'can_store_elements')
        # Calling can_store_elements(args, kwargs) (line 744)
        can_store_elements_call_result_11325 = invoke(stypy.reporting.localization.Localization(__file__, 744, 15), can_store_elements_11323, *[], **kwargs_11324)
        
        # Applying the 'not' unary operator (line 744)
        result_not__11326 = python_operator(stypy.reporting.localization.Localization(__file__, 744, 11), 'not', can_store_elements_call_result_11325)
        
        
        
        # Call to can_store_keypairs(...): (line 744)
        # Processing the call keyword arguments (line 744)
        kwargs_11329 = {}
        # Getting the type of 'self' (line 744)
        self_11327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 49), 'self', False)
        # Obtaining the member 'can_store_keypairs' of a type (line 744)
        can_store_keypairs_11328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 49), self_11327, 'can_store_keypairs')
        # Calling can_store_keypairs(args, kwargs) (line 744)
        can_store_keypairs_call_result_11330 = invoke(stypy.reporting.localization.Localization(__file__, 744, 49), can_store_keypairs_11328, *[], **kwargs_11329)
        
        # Applying the 'not' unary operator (line 744)
        result_not__11331 = python_operator(stypy.reporting.localization.Localization(__file__, 744, 45), 'not', can_store_keypairs_call_result_11330)
        
        # Applying the binary operator 'and' (line 744)
        result_and_keyword_11332 = python_operator(stypy.reporting.localization.Localization(__file__, 744, 11), 'and', result_not__11326, result_not__11331)
        
        # Testing if the type of an if condition is none (line 744)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 744, 8), result_and_keyword_11332):
            pass
        else:
            
            # Testing the type of an if condition (line 744)
            if_condition_11333 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 744, 8), result_and_keyword_11332)
            # Assigning a type to the variable 'if_condition_11333' (line 744)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 8), 'if_condition_11333', if_condition_11333)
            # SSA begins for if statement (line 744)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 745)
            # Processing the call arguments (line 745)
            # Getting the type of 'None' (line 745)
            None_11335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 29), 'None', False)
            str_11336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 29), 'str', 'STYPY CRITICAL ERROR: Attempt to determine if a container is empty over a python type ({0}) that is not able to do it')
            # Processing the call keyword arguments (line 745)
            kwargs_11337 = {}
            # Getting the type of 'TypeError' (line 745)
            TypeError_11334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 745)
            TypeError_call_result_11338 = invoke(stypy.reporting.localization.Localization(__file__, 745, 19), TypeError_11334, *[None_11335, str_11336], **kwargs_11337)
            
            # Assigning a type to the variable 'stypy_return_type' (line 745)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 12), 'stypy_return_type', TypeError_call_result_11338)
            # SSA join for if statement (line 744)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to hasattr(...): (line 748)
        # Processing the call arguments (line 748)
        # Getting the type of 'self' (line 748)
        self_11340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 23), 'self', False)
        # Getting the type of 'self' (line 748)
        self_11341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 29), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 748)
        contained_elements_property_name_11342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 29), self_11341, 'contained_elements_property_name')
        # Processing the call keyword arguments (line 748)
        kwargs_11343 = {}
        # Getting the type of 'hasattr' (line 748)
        hasattr_11339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 15), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 748)
        hasattr_call_result_11344 = invoke(stypy.reporting.localization.Localization(__file__, 748, 15), hasattr_11339, *[self_11340, contained_elements_property_name_11342], **kwargs_11343)
        
        # Assigning a type to the variable 'stypy_return_type' (line 748)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 748, 8), 'stypy_return_type', hasattr_call_result_11344)
        
        # ################# End of 'is_empty(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_empty' in the type store
        # Getting the type of 'stypy_return_type' (line 738)
        stypy_return_type_11345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11345)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_empty'
        return stypy_return_type_11345


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

        str_11346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, (-1)), 'str', '\n        Obtains the elements stored by this type, returning an error if this is called over a proxy that represent\n        a non element holding Python type\n        :return: None or TypeError\n        ')
        
        # Evaluating a boolean operation
        
        
        # Call to can_store_elements(...): (line 756)
        # Processing the call keyword arguments (line 756)
        kwargs_11349 = {}
        # Getting the type of 'self' (line 756)
        self_11347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 15), 'self', False)
        # Obtaining the member 'can_store_elements' of a type (line 756)
        can_store_elements_11348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 15), self_11347, 'can_store_elements')
        # Calling can_store_elements(args, kwargs) (line 756)
        can_store_elements_call_result_11350 = invoke(stypy.reporting.localization.Localization(__file__, 756, 15), can_store_elements_11348, *[], **kwargs_11349)
        
        # Applying the 'not' unary operator (line 756)
        result_not__11351 = python_operator(stypy.reporting.localization.Localization(__file__, 756, 11), 'not', can_store_elements_call_result_11350)
        
        
        
        # Call to can_store_keypairs(...): (line 756)
        # Processing the call keyword arguments (line 756)
        kwargs_11354 = {}
        # Getting the type of 'self' (line 756)
        self_11352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 49), 'self', False)
        # Obtaining the member 'can_store_keypairs' of a type (line 756)
        can_store_keypairs_11353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 49), self_11352, 'can_store_keypairs')
        # Calling can_store_keypairs(args, kwargs) (line 756)
        can_store_keypairs_call_result_11355 = invoke(stypy.reporting.localization.Localization(__file__, 756, 49), can_store_keypairs_11353, *[], **kwargs_11354)
        
        # Applying the 'not' unary operator (line 756)
        result_not__11356 = python_operator(stypy.reporting.localization.Localization(__file__, 756, 45), 'not', can_store_keypairs_call_result_11355)
        
        # Applying the binary operator 'and' (line 756)
        result_and_keyword_11357 = python_operator(stypy.reporting.localization.Localization(__file__, 756, 11), 'and', result_not__11351, result_not__11356)
        
        # Testing if the type of an if condition is none (line 756)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 756, 8), result_and_keyword_11357):
            pass
        else:
            
            # Testing the type of an if condition (line 756)
            if_condition_11358 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 756, 8), result_and_keyword_11357)
            # Assigning a type to the variable 'if_condition_11358' (line 756)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 8), 'if_condition_11358', if_condition_11358)
            # SSA begins for if statement (line 756)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 757)
            # Processing the call arguments (line 757)
            # Getting the type of 'None' (line 757)
            None_11360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 29), 'None', False)
            str_11361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 29), 'str', 'STYPY CRITICAL ERROR: Attempt to return stored elements over a python type ({0}) that is not able to do it')
            # Processing the call keyword arguments (line 757)
            kwargs_11362 = {}
            # Getting the type of 'TypeError' (line 757)
            TypeError_11359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 757)
            TypeError_call_result_11363 = invoke(stypy.reporting.localization.Localization(__file__, 757, 19), TypeError_11359, *[None_11360, str_11361], **kwargs_11362)
            
            # Assigning a type to the variable 'stypy_return_type' (line 757)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 757, 12), 'stypy_return_type', TypeError_call_result_11363)
            # SSA join for if statement (line 756)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to hasattr(...): (line 760)
        # Processing the call arguments (line 760)
        # Getting the type of 'self' (line 760)
        self_11365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 19), 'self', False)
        # Getting the type of 'self' (line 760)
        self_11366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 25), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 760)
        contained_elements_property_name_11367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 25), self_11366, 'contained_elements_property_name')
        # Processing the call keyword arguments (line 760)
        kwargs_11368 = {}
        # Getting the type of 'hasattr' (line 760)
        hasattr_11364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 11), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 760)
        hasattr_call_result_11369 = invoke(stypy.reporting.localization.Localization(__file__, 760, 11), hasattr_11364, *[self_11365, contained_elements_property_name_11367], **kwargs_11368)
        
        # Testing if the type of an if condition is none (line 760)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 760, 8), hasattr_call_result_11369):
            
            # Call to UndefinedType(...): (line 763)
            # Processing the call keyword arguments (line 763)
            kwargs_11379 = {}
            # Getting the type of 'undefined_type_copy' (line 763)
            undefined_type_copy_11377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 19), 'undefined_type_copy', False)
            # Obtaining the member 'UndefinedType' of a type (line 763)
            UndefinedType_11378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 19), undefined_type_copy_11377, 'UndefinedType')
            # Calling UndefinedType(args, kwargs) (line 763)
            UndefinedType_call_result_11380 = invoke(stypy.reporting.localization.Localization(__file__, 763, 19), UndefinedType_11378, *[], **kwargs_11379)
            
            # Assigning a type to the variable 'stypy_return_type' (line 763)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 12), 'stypy_return_type', UndefinedType_call_result_11380)
        else:
            
            # Testing the type of an if condition (line 760)
            if_condition_11370 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 760, 8), hasattr_call_result_11369)
            # Assigning a type to the variable 'if_condition_11370' (line 760)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'if_condition_11370', if_condition_11370)
            # SSA begins for if statement (line 760)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to getattr(...): (line 761)
            # Processing the call arguments (line 761)
            # Getting the type of 'self' (line 761)
            self_11372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 27), 'self', False)
            # Getting the type of 'self' (line 761)
            self_11373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 33), 'self', False)
            # Obtaining the member 'contained_elements_property_name' of a type (line 761)
            contained_elements_property_name_11374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 33), self_11373, 'contained_elements_property_name')
            # Processing the call keyword arguments (line 761)
            kwargs_11375 = {}
            # Getting the type of 'getattr' (line 761)
            getattr_11371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 19), 'getattr', False)
            # Calling getattr(args, kwargs) (line 761)
            getattr_call_result_11376 = invoke(stypy.reporting.localization.Localization(__file__, 761, 19), getattr_11371, *[self_11372, contained_elements_property_name_11374], **kwargs_11375)
            
            # Assigning a type to the variable 'stypy_return_type' (line 761)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'stypy_return_type', getattr_call_result_11376)
            # SSA branch for the else part of an if statement (line 760)
            module_type_store.open_ssa_branch('else')
            
            # Call to UndefinedType(...): (line 763)
            # Processing the call keyword arguments (line 763)
            kwargs_11379 = {}
            # Getting the type of 'undefined_type_copy' (line 763)
            undefined_type_copy_11377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 19), 'undefined_type_copy', False)
            # Obtaining the member 'UndefinedType' of a type (line 763)
            UndefinedType_11378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 19), undefined_type_copy_11377, 'UndefinedType')
            # Calling UndefinedType(args, kwargs) (line 763)
            UndefinedType_call_result_11380 = invoke(stypy.reporting.localization.Localization(__file__, 763, 19), UndefinedType_11378, *[], **kwargs_11379)
            
            # Assigning a type to the variable 'stypy_return_type' (line 763)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 12), 'stypy_return_type', UndefinedType_call_result_11380)
            # SSA join for if statement (line 760)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'get_elements_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_elements_type' in the type store
        # Getting the type of 'stypy_return_type' (line 750)
        stypy_return_type_11381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11381)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_elements_type'
        return stypy_return_type_11381


    @norecursion
    def set_elements_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 765)
        True_11382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 79), 'True')
        defaults = [True_11382]
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

        str_11383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, (-1)), 'str', '\n        Sets the elements stored by this type, returning an error if this is called over a proxy that represent\n        a non element holding Python type. It also checks if we are trying to store an undefined variable.\n        :param localization: Caller information\n        :param elements_type: New stored elements type\n        :param record_annotation: Whether to annotate the type change or not\n        :return: The stored elements type\n        ')
        
        # Evaluating a boolean operation
        
        
        # Call to can_store_elements(...): (line 774)
        # Processing the call keyword arguments (line 774)
        kwargs_11386 = {}
        # Getting the type of 'self' (line 774)
        self_11384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 15), 'self', False)
        # Obtaining the member 'can_store_elements' of a type (line 774)
        can_store_elements_11385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 15), self_11384, 'can_store_elements')
        # Calling can_store_elements(args, kwargs) (line 774)
        can_store_elements_call_result_11387 = invoke(stypy.reporting.localization.Localization(__file__, 774, 15), can_store_elements_11385, *[], **kwargs_11386)
        
        # Applying the 'not' unary operator (line 774)
        result_not__11388 = python_operator(stypy.reporting.localization.Localization(__file__, 774, 11), 'not', can_store_elements_call_result_11387)
        
        
        
        # Call to can_store_keypairs(...): (line 774)
        # Processing the call keyword arguments (line 774)
        kwargs_11391 = {}
        # Getting the type of 'self' (line 774)
        self_11389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 49), 'self', False)
        # Obtaining the member 'can_store_keypairs' of a type (line 774)
        can_store_keypairs_11390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 49), self_11389, 'can_store_keypairs')
        # Calling can_store_keypairs(args, kwargs) (line 774)
        can_store_keypairs_call_result_11392 = invoke(stypy.reporting.localization.Localization(__file__, 774, 49), can_store_keypairs_11390, *[], **kwargs_11391)
        
        # Applying the 'not' unary operator (line 774)
        result_not__11393 = python_operator(stypy.reporting.localization.Localization(__file__, 774, 45), 'not', can_store_keypairs_call_result_11392)
        
        # Applying the binary operator 'and' (line 774)
        result_and_keyword_11394 = python_operator(stypy.reporting.localization.Localization(__file__, 774, 11), 'and', result_not__11388, result_not__11393)
        
        # Testing if the type of an if condition is none (line 774)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 774, 8), result_and_keyword_11394):
            pass
        else:
            
            # Testing the type of an if condition (line 774)
            if_condition_11395 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 774, 8), result_and_keyword_11394)
            # Assigning a type to the variable 'if_condition_11395' (line 774)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 8), 'if_condition_11395', if_condition_11395)
            # SSA begins for if statement (line 774)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 775)
            # Processing the call arguments (line 775)
            # Getting the type of 'localization' (line 775)
            localization_11397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 29), 'localization', False)
            
            # Call to format(...): (line 776)
            # Processing the call arguments (line 776)
            
            # Call to get_python_type(...): (line 777)
            # Processing the call keyword arguments (line 777)
            kwargs_11402 = {}
            # Getting the type of 'self' (line 777)
            self_11400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 64), 'self', False)
            # Obtaining the member 'get_python_type' of a type (line 777)
            get_python_type_11401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 64), self_11400, 'get_python_type')
            # Calling get_python_type(args, kwargs) (line 777)
            get_python_type_call_result_11403 = invoke(stypy.reporting.localization.Localization(__file__, 777, 64), get_python_type_11401, *[], **kwargs_11402)
            
            # Processing the call keyword arguments (line 776)
            kwargs_11404 = {}
            str_11398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 29), 'str', 'STYPY CRITICAL ERROR: Attempt to set stored elements types over a python type ({0}) that is not able to do it')
            # Obtaining the member 'format' of a type (line 776)
            format_11399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 29), str_11398, 'format')
            # Calling format(args, kwargs) (line 776)
            format_call_result_11405 = invoke(stypy.reporting.localization.Localization(__file__, 776, 29), format_11399, *[get_python_type_call_result_11403], **kwargs_11404)
            
            # Processing the call keyword arguments (line 775)
            kwargs_11406 = {}
            # Getting the type of 'TypeError' (line 775)
            TypeError_11396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 775)
            TypeError_call_result_11407 = invoke(stypy.reporting.localization.Localization(__file__, 775, 19), TypeError_11396, *[localization_11397, format_call_result_11405], **kwargs_11406)
            
            # Assigning a type to the variable 'stypy_return_type' (line 775)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 12), 'stypy_return_type', TypeError_call_result_11407)
            # SSA join for if statement (line 774)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Tuple (line 779):
        
        # Assigning a Call to a Name:
        
        # Call to contains_an_undefined_type(...): (line 779)
        # Processing the call arguments (line 779)
        # Getting the type of 'elements_type' (line 779)
        elements_type_11410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 96), 'elements_type', False)
        # Processing the call keyword arguments (line 779)
        kwargs_11411 = {}
        # Getting the type of 'TypeInferenceProxy' (line 779)
        TypeInferenceProxy_11408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 50), 'TypeInferenceProxy', False)
        # Obtaining the member 'contains_an_undefined_type' of a type (line 779)
        contains_an_undefined_type_11409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 50), TypeInferenceProxy_11408, 'contains_an_undefined_type')
        # Calling contains_an_undefined_type(args, kwargs) (line 779)
        contains_an_undefined_type_call_result_11412 = invoke(stypy.reporting.localization.Localization(__file__, 779, 50), contains_an_undefined_type_11409, *[elements_type_11410], **kwargs_11411)
        
        # Assigning a type to the variable 'call_assignment_10137' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'call_assignment_10137', contains_an_undefined_type_call_result_11412)
        
        # Assigning a Call to a Name (line 779):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_10137' (line 779)
        call_assignment_10137_11413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'call_assignment_10137', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_11414 = stypy_get_value_from_tuple(call_assignment_10137_11413, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_10138' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'call_assignment_10138', stypy_get_value_from_tuple_call_result_11414)
        
        # Assigning a Name to a Name (line 779):
        # Getting the type of 'call_assignment_10138' (line 779)
        call_assignment_10138_11415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'call_assignment_10138')
        # Assigning a type to the variable 'contains_undefined' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'contains_undefined', call_assignment_10138_11415)
        
        # Assigning a Call to a Name (line 779):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_10137' (line 779)
        call_assignment_10137_11416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'call_assignment_10137', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_11417 = stypy_get_value_from_tuple(call_assignment_10137_11416, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_10139' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'call_assignment_10139', stypy_get_value_from_tuple_call_result_11417)
        
        # Assigning a Name to a Name (line 779):
        # Getting the type of 'call_assignment_10139' (line 779)
        call_assignment_10139_11418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'call_assignment_10139')
        # Assigning a type to the variable 'more_types_in_value' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 28), 'more_types_in_value', call_assignment_10139_11418)
        # Getting the type of 'contains_undefined' (line 780)
        contains_undefined_11419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 11), 'contains_undefined')
        # Testing if the type of an if condition is none (line 780)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 780, 8), contains_undefined_11419):
            pass
        else:
            
            # Testing the type of an if condition (line 780)
            if_condition_11420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 780, 8), contains_undefined_11419)
            # Assigning a type to the variable 'if_condition_11420' (line 780)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 8), 'if_condition_11420', if_condition_11420)
            # SSA begins for if statement (line 780)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'more_types_in_value' (line 781)
            more_types_in_value_11421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 15), 'more_types_in_value')
            int_11422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 38), 'int')
            # Applying the binary operator '==' (line 781)
            result_eq_11423 = python_operator(stypy.reporting.localization.Localization(__file__, 781, 15), '==', more_types_in_value_11421, int_11422)
            
            # Testing if the type of an if condition is none (line 781)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 781, 12), result_eq_11423):
                
                # Call to instance(...): (line 785)
                # Processing the call arguments (line 785)
                # Getting the type of 'localization' (line 785)
                localization_11437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 37), 'localization', False)
                
                # Call to format(...): (line 786)
                # Processing the call arguments (line 786)
                # Getting the type of 'self' (line 787)
                self_11440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 44), 'self', False)
                # Obtaining the member 'name' of a type (line 787)
                name_11441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 44), self_11440, 'name')
                # Processing the call keyword arguments (line 786)
                kwargs_11442 = {}
                str_11438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 37), 'str', "Potentially storing in '{0}' the value of a previously undefined variable")
                # Obtaining the member 'format' of a type (line 786)
                format_11439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 37), str_11438, 'format')
                # Calling format(args, kwargs) (line 786)
                format_call_result_11443 = invoke(stypy.reporting.localization.Localization(__file__, 786, 37), format_11439, *[name_11441], **kwargs_11442)
                
                # Processing the call keyword arguments (line 785)
                kwargs_11444 = {}
                # Getting the type of 'TypeWarning' (line 785)
                TypeWarning_11435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 16), 'TypeWarning', False)
                # Obtaining the member 'instance' of a type (line 785)
                instance_11436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 16), TypeWarning_11435, 'instance')
                # Calling instance(args, kwargs) (line 785)
                instance_call_result_11445 = invoke(stypy.reporting.localization.Localization(__file__, 785, 16), instance_11436, *[localization_11437, format_call_result_11443], **kwargs_11444)
                
            else:
                
                # Testing the type of an if condition (line 781)
                if_condition_11424 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 781, 12), result_eq_11423)
                # Assigning a type to the variable 'if_condition_11424' (line 781)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 781, 12), 'if_condition_11424', if_condition_11424)
                # SSA begins for if statement (line 781)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to TypeError(...): (line 782)
                # Processing the call arguments (line 782)
                # Getting the type of 'localization' (line 782)
                localization_11426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 26), 'localization', False)
                
                # Call to format(...): (line 782)
                # Processing the call arguments (line 782)
                # Getting the type of 'self' (line 783)
                self_11429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 33), 'self', False)
                # Obtaining the member 'name' of a type (line 783)
                name_11430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 33), self_11429, 'name')
                # Processing the call keyword arguments (line 782)
                kwargs_11431 = {}
                str_11427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, 40), 'str', "Storing in '{0}' the value of a previously undefined variable")
                # Obtaining the member 'format' of a type (line 782)
                format_11428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 40), str_11427, 'format')
                # Calling format(args, kwargs) (line 782)
                format_call_result_11432 = invoke(stypy.reporting.localization.Localization(__file__, 782, 40), format_11428, *[name_11430], **kwargs_11431)
                
                # Processing the call keyword arguments (line 782)
                kwargs_11433 = {}
                # Getting the type of 'TypeError' (line 782)
                TypeError_11425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 16), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 782)
                TypeError_call_result_11434 = invoke(stypy.reporting.localization.Localization(__file__, 782, 16), TypeError_11425, *[localization_11426, format_call_result_11432], **kwargs_11433)
                
                # SSA branch for the else part of an if statement (line 781)
                module_type_store.open_ssa_branch('else')
                
                # Call to instance(...): (line 785)
                # Processing the call arguments (line 785)
                # Getting the type of 'localization' (line 785)
                localization_11437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 37), 'localization', False)
                
                # Call to format(...): (line 786)
                # Processing the call arguments (line 786)
                # Getting the type of 'self' (line 787)
                self_11440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 44), 'self', False)
                # Obtaining the member 'name' of a type (line 787)
                name_11441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 44), self_11440, 'name')
                # Processing the call keyword arguments (line 786)
                kwargs_11442 = {}
                str_11438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 37), 'str', "Potentially storing in '{0}' the value of a previously undefined variable")
                # Obtaining the member 'format' of a type (line 786)
                format_11439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 37), str_11438, 'format')
                # Calling format(args, kwargs) (line 786)
                format_call_result_11443 = invoke(stypy.reporting.localization.Localization(__file__, 786, 37), format_11439, *[name_11441], **kwargs_11442)
                
                # Processing the call keyword arguments (line 785)
                kwargs_11444 = {}
                # Getting the type of 'TypeWarning' (line 785)
                TypeWarning_11435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 16), 'TypeWarning', False)
                # Obtaining the member 'instance' of a type (line 785)
                instance_11436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 16), TypeWarning_11435, 'instance')
                # Calling instance(args, kwargs) (line 785)
                instance_call_result_11445 = invoke(stypy.reporting.localization.Localization(__file__, 785, 16), instance_11436, *[localization_11437, format_call_result_11443], **kwargs_11444)
                
                # SSA join for if statement (line 781)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 780)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to setattr(...): (line 789)
        # Processing the call arguments (line 789)
        # Getting the type of 'self' (line 789)
        self_11447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 16), 'self', False)
        # Getting the type of 'self' (line 789)
        self_11448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 22), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 789)
        contained_elements_property_name_11449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 789, 22), self_11448, 'contained_elements_property_name')
        # Getting the type of 'elements_type' (line 789)
        elements_type_11450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 61), 'elements_type', False)
        # Processing the call keyword arguments (line 789)
        kwargs_11451 = {}
        # Getting the type of 'setattr' (line 789)
        setattr_11446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 8), 'setattr', False)
        # Calling setattr(args, kwargs) (line 789)
        setattr_call_result_11452 = invoke(stypy.reporting.localization.Localization(__file__, 789, 8), setattr_11446, *[self_11447, contained_elements_property_name_11449, elements_type_11450], **kwargs_11451)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'record_annotation' (line 790)
        record_annotation_11453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 11), 'record_annotation')
        # Getting the type of 'self' (line 790)
        self_11454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 33), 'self')
        # Obtaining the member 'annotate_types' of a type (line 790)
        annotate_types_11455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 790, 33), self_11454, 'annotate_types')
        # Applying the binary operator 'and' (line 790)
        result_and_keyword_11456 = python_operator(stypy.reporting.localization.Localization(__file__, 790, 11), 'and', record_annotation_11453, annotate_types_11455)
        
        # Testing if the type of an if condition is none (line 790)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 790, 8), result_and_keyword_11456):
            pass
        else:
            
            # Testing the type of an if condition (line 790)
            if_condition_11457 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 790, 8), result_and_keyword_11456)
            # Assigning a type to the variable 'if_condition_11457' (line 790)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 790, 8), 'if_condition_11457', if_condition_11457)
            # SSA begins for if statement (line 790)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to __annotate_type(...): (line 791)
            # Processing the call arguments (line 791)
            # Getting the type of 'localization' (line 791)
            localization_11460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 33), 'localization', False)
            # Obtaining the member 'line' of a type (line 791)
            line_11461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 33), localization_11460, 'line')
            # Getting the type of 'localization' (line 791)
            localization_11462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 52), 'localization', False)
            # Obtaining the member 'column' of a type (line 791)
            column_11463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 52), localization_11462, 'column')
            str_11464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, 73), 'str', '<container elements type>')
            
            # Call to getattr(...): (line 792)
            # Processing the call arguments (line 792)
            # Getting the type of 'self' (line 792)
            self_11466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 41), 'self', False)
            # Getting the type of 'self' (line 792)
            self_11467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 47), 'self', False)
            # Obtaining the member 'contained_elements_property_name' of a type (line 792)
            contained_elements_property_name_11468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 47), self_11467, 'contained_elements_property_name')
            # Processing the call keyword arguments (line 792)
            kwargs_11469 = {}
            # Getting the type of 'getattr' (line 792)
            getattr_11465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 33), 'getattr', False)
            # Calling getattr(args, kwargs) (line 792)
            getattr_call_result_11470 = invoke(stypy.reporting.localization.Localization(__file__, 792, 33), getattr_11465, *[self_11466, contained_elements_property_name_11468], **kwargs_11469)
            
            # Processing the call keyword arguments (line 791)
            kwargs_11471 = {}
            # Getting the type of 'self' (line 791)
            self_11458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 12), 'self', False)
            # Obtaining the member '__annotate_type' of a type (line 791)
            annotate_type_11459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 12), self_11458, '__annotate_type')
            # Calling __annotate_type(args, kwargs) (line 791)
            annotate_type_call_result_11472 = invoke(stypy.reporting.localization.Localization(__file__, 791, 12), annotate_type_11459, *[line_11461, column_11463, str_11464, getattr_call_result_11470], **kwargs_11471)
            
            # SSA join for if statement (line 790)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'set_elements_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_elements_type' in the type store
        # Getting the type of 'stypy_return_type' (line 765)
        stypy_return_type_11473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11473)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_elements_type'
        return stypy_return_type_11473


    @norecursion
    def add_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 794)
        True_11474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 62), 'True')
        defaults = [True_11474]
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

        str_11475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 802, (-1)), 'str', '\n        Adds type_ to the elements stored by this type, returning an error if this is called over a proxy that represent\n        a non element holding Python type. It also checks if we are trying to store an undefined variable.\n        :param localization: Caller information\n        :param type_: Type to store\n        :param record_annotation: Whether to annotate the type change or not\n        :return: None or TypeError\n        ')
        
        
        # Call to can_store_elements(...): (line 803)
        # Processing the call keyword arguments (line 803)
        kwargs_11478 = {}
        # Getting the type of 'self' (line 803)
        self_11476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 15), 'self', False)
        # Obtaining the member 'can_store_elements' of a type (line 803)
        can_store_elements_11477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 803, 15), self_11476, 'can_store_elements')
        # Calling can_store_elements(args, kwargs) (line 803)
        can_store_elements_call_result_11479 = invoke(stypy.reporting.localization.Localization(__file__, 803, 15), can_store_elements_11477, *[], **kwargs_11478)
        
        # Applying the 'not' unary operator (line 803)
        result_not__11480 = python_operator(stypy.reporting.localization.Localization(__file__, 803, 11), 'not', can_store_elements_call_result_11479)
        
        # Testing if the type of an if condition is none (line 803)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 803, 8), result_not__11480):
            pass
        else:
            
            # Testing the type of an if condition (line 803)
            if_condition_11481 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 803, 8), result_not__11480)
            # Assigning a type to the variable 'if_condition_11481' (line 803)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 803, 8), 'if_condition_11481', if_condition_11481)
            # SSA begins for if statement (line 803)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 804)
            # Processing the call arguments (line 804)
            # Getting the type of 'localization' (line 804)
            localization_11483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 29), 'localization', False)
            
            # Call to format(...): (line 805)
            # Processing the call arguments (line 805)
            
            # Call to get_python_type(...): (line 806)
            # Processing the call keyword arguments (line 806)
            kwargs_11488 = {}
            # Getting the type of 'self' (line 806)
            self_11486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 53), 'self', False)
            # Obtaining the member 'get_python_type' of a type (line 806)
            get_python_type_11487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 53), self_11486, 'get_python_type')
            # Calling get_python_type(args, kwargs) (line 806)
            get_python_type_call_result_11489 = invoke(stypy.reporting.localization.Localization(__file__, 806, 53), get_python_type_11487, *[], **kwargs_11488)
            
            # Processing the call keyword arguments (line 805)
            kwargs_11490 = {}
            str_11484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 29), 'str', 'STYPY CRITICAL ERROR: Attempt to store elements over a python type ({0}) that is not able to do it')
            # Obtaining the member 'format' of a type (line 805)
            format_11485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 29), str_11484, 'format')
            # Calling format(args, kwargs) (line 805)
            format_call_result_11491 = invoke(stypy.reporting.localization.Localization(__file__, 805, 29), format_11485, *[get_python_type_call_result_11489], **kwargs_11490)
            
            # Processing the call keyword arguments (line 804)
            kwargs_11492 = {}
            # Getting the type of 'TypeError' (line 804)
            TypeError_11482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 804)
            TypeError_call_result_11493 = invoke(stypy.reporting.localization.Localization(__file__, 804, 19), TypeError_11482, *[localization_11483, format_call_result_11491], **kwargs_11492)
            
            # Assigning a type to the variable 'stypy_return_type' (line 804)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 804, 12), 'stypy_return_type', TypeError_call_result_11493)
            # SSA join for if statement (line 803)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Name to a Name (line 808):
        
        # Assigning a Name to a Name (line 808):
        # Getting the type of 'None' (line 808)
        None_11494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 24), 'None')
        # Assigning a type to the variable 'existing_type' (line 808)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 8), 'existing_type', None_11494)
        
        # Call to hasattr(...): (line 809)
        # Processing the call arguments (line 809)
        # Getting the type of 'self' (line 809)
        self_11496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 19), 'self', False)
        # Getting the type of 'self' (line 809)
        self_11497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 25), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 809)
        contained_elements_property_name_11498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 809, 25), self_11497, 'contained_elements_property_name')
        # Processing the call keyword arguments (line 809)
        kwargs_11499 = {}
        # Getting the type of 'hasattr' (line 809)
        hasattr_11495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 11), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 809)
        hasattr_call_result_11500 = invoke(stypy.reporting.localization.Localization(__file__, 809, 11), hasattr_11495, *[self_11496, contained_elements_property_name_11498], **kwargs_11499)
        
        # Testing if the type of an if condition is none (line 809)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 809, 8), hasattr_call_result_11500):
            pass
        else:
            
            # Testing the type of an if condition (line 809)
            if_condition_11501 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 809, 8), hasattr_call_result_11500)
            # Assigning a type to the variable 'if_condition_11501' (line 809)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 8), 'if_condition_11501', if_condition_11501)
            # SSA begins for if statement (line 809)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 810):
            
            # Assigning a Call to a Name (line 810):
            
            # Call to getattr(...): (line 810)
            # Processing the call arguments (line 810)
            # Getting the type of 'self' (line 810)
            self_11503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 36), 'self', False)
            # Getting the type of 'self' (line 810)
            self_11504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 42), 'self', False)
            # Obtaining the member 'contained_elements_property_name' of a type (line 810)
            contained_elements_property_name_11505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 42), self_11504, 'contained_elements_property_name')
            # Processing the call keyword arguments (line 810)
            kwargs_11506 = {}
            # Getting the type of 'getattr' (line 810)
            getattr_11502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 28), 'getattr', False)
            # Calling getattr(args, kwargs) (line 810)
            getattr_call_result_11507 = invoke(stypy.reporting.localization.Localization(__file__, 810, 28), getattr_11502, *[self_11503, contained_elements_property_name_11505], **kwargs_11506)
            
            # Assigning a type to the variable 'existing_type' (line 810)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 12), 'existing_type', getattr_call_result_11507)
            # SSA join for if statement (line 809)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 812):
        
        # Assigning a Call to a Name (line 812):
        
        # Call to add(...): (line 812)
        # Processing the call arguments (line 812)
        # Getting the type of 'existing_type' (line 812)
        existing_type_11511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 55), 'existing_type', False)
        # Getting the type of 'type_' (line 812)
        type__11512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 70), 'type_', False)
        # Processing the call keyword arguments (line 812)
        kwargs_11513 = {}
        # Getting the type of 'union_type_copy' (line 812)
        union_type_copy_11508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 25), 'union_type_copy', False)
        # Obtaining the member 'UnionType' of a type (line 812)
        UnionType_11509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 25), union_type_copy_11508, 'UnionType')
        # Obtaining the member 'add' of a type (line 812)
        add_11510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 25), UnionType_11509, 'add')
        # Calling add(args, kwargs) (line 812)
        add_call_result_11514 = invoke(stypy.reporting.localization.Localization(__file__, 812, 25), add_11510, *[existing_type_11511, type__11512], **kwargs_11513)
        
        # Assigning a type to the variable 'value_to_store' (line 812)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 8), 'value_to_store', add_call_result_11514)
        
        # Call to __check_undefined_stored_value(...): (line 813)
        # Processing the call arguments (line 813)
        # Getting the type of 'localization' (line 813)
        localization_11517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 44), 'localization', False)
        # Getting the type of 'value_to_store' (line 813)
        value_to_store_11518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 58), 'value_to_store', False)
        # Processing the call keyword arguments (line 813)
        kwargs_11519 = {}
        # Getting the type of 'self' (line 813)
        self_11515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 8), 'self', False)
        # Obtaining the member '__check_undefined_stored_value' of a type (line 813)
        check_undefined_stored_value_11516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 813, 8), self_11515, '__check_undefined_stored_value')
        # Calling __check_undefined_stored_value(args, kwargs) (line 813)
        check_undefined_stored_value_call_result_11520 = invoke(stypy.reporting.localization.Localization(__file__, 813, 8), check_undefined_stored_value_11516, *[localization_11517, value_to_store_11518], **kwargs_11519)
        
        
        # Call to setattr(...): (line 815)
        # Processing the call arguments (line 815)
        # Getting the type of 'self' (line 815)
        self_11522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 16), 'self', False)
        # Getting the type of 'self' (line 815)
        self_11523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 22), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 815)
        contained_elements_property_name_11524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 815, 22), self_11523, 'contained_elements_property_name')
        # Getting the type of 'value_to_store' (line 815)
        value_to_store_11525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 61), 'value_to_store', False)
        # Processing the call keyword arguments (line 815)
        kwargs_11526 = {}
        # Getting the type of 'setattr' (line 815)
        setattr_11521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 8), 'setattr', False)
        # Calling setattr(args, kwargs) (line 815)
        setattr_call_result_11527 = invoke(stypy.reporting.localization.Localization(__file__, 815, 8), setattr_11521, *[self_11522, contained_elements_property_name_11524, value_to_store_11525], **kwargs_11526)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'record_annotation' (line 817)
        record_annotation_11528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 11), 'record_annotation')
        # Getting the type of 'self' (line 817)
        self_11529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 33), 'self')
        # Obtaining the member 'annotate_types' of a type (line 817)
        annotate_types_11530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 33), self_11529, 'annotate_types')
        # Applying the binary operator 'and' (line 817)
        result_and_keyword_11531 = python_operator(stypy.reporting.localization.Localization(__file__, 817, 11), 'and', record_annotation_11528, annotate_types_11530)
        
        # Testing if the type of an if condition is none (line 817)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 817, 8), result_and_keyword_11531):
            pass
        else:
            
            # Testing the type of an if condition (line 817)
            if_condition_11532 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 817, 8), result_and_keyword_11531)
            # Assigning a type to the variable 'if_condition_11532' (line 817)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 8), 'if_condition_11532', if_condition_11532)
            # SSA begins for if statement (line 817)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to __annotate_type(...): (line 818)
            # Processing the call arguments (line 818)
            # Getting the type of 'localization' (line 818)
            localization_11535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 33), 'localization', False)
            # Obtaining the member 'line' of a type (line 818)
            line_11536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 818, 33), localization_11535, 'line')
            # Getting the type of 'localization' (line 818)
            localization_11537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 52), 'localization', False)
            # Obtaining the member 'column' of a type (line 818)
            column_11538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 818, 52), localization_11537, 'column')
            str_11539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 73), 'str', '<container elements type>')
            
            # Call to getattr(...): (line 819)
            # Processing the call arguments (line 819)
            # Getting the type of 'self' (line 819)
            self_11541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 41), 'self', False)
            # Getting the type of 'self' (line 819)
            self_11542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 47), 'self', False)
            # Obtaining the member 'contained_elements_property_name' of a type (line 819)
            contained_elements_property_name_11543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 819, 47), self_11542, 'contained_elements_property_name')
            # Processing the call keyword arguments (line 819)
            kwargs_11544 = {}
            # Getting the type of 'getattr' (line 819)
            getattr_11540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 33), 'getattr', False)
            # Calling getattr(args, kwargs) (line 819)
            getattr_call_result_11545 = invoke(stypy.reporting.localization.Localization(__file__, 819, 33), getattr_11540, *[self_11541, contained_elements_property_name_11543], **kwargs_11544)
            
            # Processing the call keyword arguments (line 818)
            kwargs_11546 = {}
            # Getting the type of 'self' (line 818)
            self_11533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 12), 'self', False)
            # Obtaining the member '__annotate_type' of a type (line 818)
            annotate_type_11534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 818, 12), self_11533, '__annotate_type')
            # Calling __annotate_type(args, kwargs) (line 818)
            annotate_type_call_result_11547 = invoke(stypy.reporting.localization.Localization(__file__, 818, 12), annotate_type_11534, *[line_11536, column_11538, str_11539, getattr_call_result_11545], **kwargs_11546)
            
            # SSA join for if statement (line 817)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'add_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_type' in the type store
        # Getting the type of 'stypy_return_type' (line 794)
        stypy_return_type_11548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11548)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_type'
        return stypy_return_type_11548


    @norecursion
    def add_types_from_list(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 821)
        True_11549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 77), 'True')
        defaults = [True_11549]
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

        str_11550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, (-1)), 'str', '\n        Adds the types on type_list to the elements stored by this type, returning an error if this is called over a\n        proxy that represent a non element holding Python type. It also checks if we are trying to store an undefined\n        variable.\n        :param localization: Caller information\n        :param type_list: List of types to add\n        :param record_annotation: Whether to annotate the type change or not\n        :return: None or TypeError\n        ')
        
        
        # Call to can_store_elements(...): (line 831)
        # Processing the call keyword arguments (line 831)
        kwargs_11553 = {}
        # Getting the type of 'self' (line 831)
        self_11551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 15), 'self', False)
        # Obtaining the member 'can_store_elements' of a type (line 831)
        can_store_elements_11552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 831, 15), self_11551, 'can_store_elements')
        # Calling can_store_elements(args, kwargs) (line 831)
        can_store_elements_call_result_11554 = invoke(stypy.reporting.localization.Localization(__file__, 831, 15), can_store_elements_11552, *[], **kwargs_11553)
        
        # Applying the 'not' unary operator (line 831)
        result_not__11555 = python_operator(stypy.reporting.localization.Localization(__file__, 831, 11), 'not', can_store_elements_call_result_11554)
        
        # Testing if the type of an if condition is none (line 831)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 831, 8), result_not__11555):
            pass
        else:
            
            # Testing the type of an if condition (line 831)
            if_condition_11556 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 831, 8), result_not__11555)
            # Assigning a type to the variable 'if_condition_11556' (line 831)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'if_condition_11556', if_condition_11556)
            # SSA begins for if statement (line 831)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 832)
            # Processing the call arguments (line 832)
            # Getting the type of 'localization' (line 832)
            localization_11558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 29), 'localization', False)
            
            # Call to format(...): (line 833)
            # Processing the call arguments (line 833)
            
            # Call to get_python_type(...): (line 834)
            # Processing the call keyword arguments (line 834)
            kwargs_11563 = {}
            # Getting the type of 'self' (line 834)
            self_11561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 53), 'self', False)
            # Obtaining the member 'get_python_type' of a type (line 834)
            get_python_type_11562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 53), self_11561, 'get_python_type')
            # Calling get_python_type(args, kwargs) (line 834)
            get_python_type_call_result_11564 = invoke(stypy.reporting.localization.Localization(__file__, 834, 53), get_python_type_11562, *[], **kwargs_11563)
            
            # Processing the call keyword arguments (line 833)
            kwargs_11565 = {}
            str_11559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 833, 29), 'str', 'STYPY CRITICAL ERROR: Attempt to store elements over a python type ({0}) that is not able to do it')
            # Obtaining the member 'format' of a type (line 833)
            format_11560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 833, 29), str_11559, 'format')
            # Calling format(args, kwargs) (line 833)
            format_call_result_11566 = invoke(stypy.reporting.localization.Localization(__file__, 833, 29), format_11560, *[get_python_type_call_result_11564], **kwargs_11565)
            
            # Processing the call keyword arguments (line 832)
            kwargs_11567 = {}
            # Getting the type of 'TypeError' (line 832)
            TypeError_11557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 832)
            TypeError_call_result_11568 = invoke(stypy.reporting.localization.Localization(__file__, 832, 19), TypeError_11557, *[localization_11558, format_call_result_11566], **kwargs_11567)
            
            # Assigning a type to the variable 'stypy_return_type' (line 832)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 832, 12), 'stypy_return_type', TypeError_call_result_11568)
            # SSA join for if statement (line 831)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to hasattr(...): (line 836)
        # Processing the call arguments (line 836)
        # Getting the type of 'self' (line 836)
        self_11570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 19), 'self', False)
        # Getting the type of 'self' (line 836)
        self_11571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 25), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 836)
        contained_elements_property_name_11572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 836, 25), self_11571, 'contained_elements_property_name')
        # Processing the call keyword arguments (line 836)
        kwargs_11573 = {}
        # Getting the type of 'hasattr' (line 836)
        hasattr_11569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 11), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 836)
        hasattr_call_result_11574 = invoke(stypy.reporting.localization.Localization(__file__, 836, 11), hasattr_11569, *[self_11570, contained_elements_property_name_11572], **kwargs_11573)
        
        # Testing if the type of an if condition is none (line 836)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 836, 8), hasattr_call_result_11574):
            pass
        else:
            
            # Testing the type of an if condition (line 836)
            if_condition_11575 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 836, 8), hasattr_call_result_11574)
            # Assigning a type to the variable 'if_condition_11575' (line 836)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 836, 8), 'if_condition_11575', if_condition_11575)
            # SSA begins for if statement (line 836)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 837):
            
            # Assigning a Call to a Name (line 837):
            
            # Call to getattr(...): (line 837)
            # Processing the call arguments (line 837)
            # Getting the type of 'self' (line 837)
            self_11577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 36), 'self', False)
            # Getting the type of 'self' (line 837)
            self_11578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 42), 'self', False)
            # Obtaining the member 'contained_elements_property_name' of a type (line 837)
            contained_elements_property_name_11579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 42), self_11578, 'contained_elements_property_name')
            # Processing the call keyword arguments (line 837)
            kwargs_11580 = {}
            # Getting the type of 'getattr' (line 837)
            getattr_11576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 28), 'getattr', False)
            # Calling getattr(args, kwargs) (line 837)
            getattr_call_result_11581 = invoke(stypy.reporting.localization.Localization(__file__, 837, 28), getattr_11576, *[self_11577, contained_elements_property_name_11579], **kwargs_11580)
            
            # Assigning a type to the variable 'existing_type' (line 837)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 12), 'existing_type', getattr_call_result_11581)
            
            # Assigning a BinOp to a Name (line 838):
            
            # Assigning a BinOp to a Name (line 838):
            
            # Obtaining an instance of the builtin type 'list' (line 838)
            list_11582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 24), 'list')
            # Adding type elements to the builtin type 'list' instance (line 838)
            # Adding element type (line 838)
            # Getting the type of 'existing_type' (line 838)
            existing_type_11583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 25), 'existing_type')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 838, 24), list_11582, existing_type_11583)
            
            # Getting the type of 'type_list' (line 838)
            type_list_11584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 42), 'type_list')
            # Applying the binary operator '+' (line 838)
            result_add_11585 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 24), '+', list_11582, type_list_11584)
            
            # Assigning a type to the variable 'type_list' (line 838)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 12), 'type_list', result_add_11585)
            # SSA join for if statement (line 836)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to setattr(...): (line 840)
        # Processing the call arguments (line 840)
        # Getting the type of 'self' (line 840)
        self_11587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 16), 'self', False)
        # Getting the type of 'self' (line 840)
        self_11588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 22), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 840)
        contained_elements_property_name_11589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 22), self_11588, 'contained_elements_property_name')
        
        # Call to create_union_type_from_types(...): (line 841)
        # Getting the type of 'type_list' (line 841)
        type_list_11593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 72), 'type_list', False)
        # Processing the call keyword arguments (line 841)
        kwargs_11594 = {}
        # Getting the type of 'union_type_copy' (line 841)
        union_type_copy_11590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 16), 'union_type_copy', False)
        # Obtaining the member 'UnionType' of a type (line 841)
        UnionType_11591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 841, 16), union_type_copy_11590, 'UnionType')
        # Obtaining the member 'create_union_type_from_types' of a type (line 841)
        create_union_type_from_types_11592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 841, 16), UnionType_11591, 'create_union_type_from_types')
        # Calling create_union_type_from_types(args, kwargs) (line 841)
        create_union_type_from_types_call_result_11595 = invoke(stypy.reporting.localization.Localization(__file__, 841, 16), create_union_type_from_types_11592, *[type_list_11593], **kwargs_11594)
        
        # Processing the call keyword arguments (line 840)
        kwargs_11596 = {}
        # Getting the type of 'setattr' (line 840)
        setattr_11586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 8), 'setattr', False)
        # Calling setattr(args, kwargs) (line 840)
        setattr_call_result_11597 = invoke(stypy.reporting.localization.Localization(__file__, 840, 8), setattr_11586, *[self_11587, contained_elements_property_name_11589, create_union_type_from_types_call_result_11595], **kwargs_11596)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'record_annotation' (line 842)
        record_annotation_11598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 11), 'record_annotation')
        # Getting the type of 'self' (line 842)
        self_11599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 33), 'self')
        # Obtaining the member 'annotate_types' of a type (line 842)
        annotate_types_11600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 842, 33), self_11599, 'annotate_types')
        # Applying the binary operator 'and' (line 842)
        result_and_keyword_11601 = python_operator(stypy.reporting.localization.Localization(__file__, 842, 11), 'and', record_annotation_11598, annotate_types_11600)
        
        # Testing if the type of an if condition is none (line 842)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 842, 8), result_and_keyword_11601):
            pass
        else:
            
            # Testing the type of an if condition (line 842)
            if_condition_11602 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 842, 8), result_and_keyword_11601)
            # Assigning a type to the variable 'if_condition_11602' (line 842)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 842, 8), 'if_condition_11602', if_condition_11602)
            # SSA begins for if statement (line 842)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to __annotate_type(...): (line 843)
            # Processing the call arguments (line 843)
            # Getting the type of 'localization' (line 843)
            localization_11605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 33), 'localization', False)
            # Obtaining the member 'line' of a type (line 843)
            line_11606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 33), localization_11605, 'line')
            # Getting the type of 'localization' (line 843)
            localization_11607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 52), 'localization', False)
            # Obtaining the member 'column' of a type (line 843)
            column_11608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 52), localization_11607, 'column')
            str_11609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 843, 73), 'str', '<container elements type>')
            
            # Call to getattr(...): (line 844)
            # Processing the call arguments (line 844)
            # Getting the type of 'self' (line 844)
            self_11611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 41), 'self', False)
            # Getting the type of 'self' (line 844)
            self_11612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 47), 'self', False)
            # Obtaining the member 'contained_elements_property_name' of a type (line 844)
            contained_elements_property_name_11613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 47), self_11612, 'contained_elements_property_name')
            # Processing the call keyword arguments (line 844)
            kwargs_11614 = {}
            # Getting the type of 'getattr' (line 844)
            getattr_11610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 33), 'getattr', False)
            # Calling getattr(args, kwargs) (line 844)
            getattr_call_result_11615 = invoke(stypy.reporting.localization.Localization(__file__, 844, 33), getattr_11610, *[self_11611, contained_elements_property_name_11613], **kwargs_11614)
            
            # Processing the call keyword arguments (line 843)
            kwargs_11616 = {}
            # Getting the type of 'self' (line 843)
            self_11603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 12), 'self', False)
            # Obtaining the member '__annotate_type' of a type (line 843)
            annotate_type_11604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 12), self_11603, '__annotate_type')
            # Calling __annotate_type(args, kwargs) (line 843)
            annotate_type_call_result_11617 = invoke(stypy.reporting.localization.Localization(__file__, 843, 12), annotate_type_11604, *[line_11606, column_11608, str_11609, getattr_call_result_11615], **kwargs_11616)
            
            # SSA join for if statement (line 842)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'add_types_from_list(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_types_from_list' in the type store
        # Getting the type of 'stypy_return_type' (line 821)
        stypy_return_type_11618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11618)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_types_from_list'
        return stypy_return_type_11618


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

        str_11619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 851, (-1)), 'str', '\n        Helper method to see if the stored keypairs contains a key equal to the passed one.\n        :param key:\n        :return:\n        ')
        
        # Assigning a Call to a Name (line 852):
        
        # Assigning a Call to a Name (line 852):
        
        # Call to getattr(...): (line 852)
        # Processing the call arguments (line 852)
        # Getting the type of 'self' (line 852)
        self_11621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 36), 'self', False)
        # Getting the type of 'self' (line 852)
        self_11622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 42), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 852)
        contained_elements_property_name_11623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 852, 42), self_11622, 'contained_elements_property_name')
        # Processing the call keyword arguments (line 852)
        kwargs_11624 = {}
        # Getting the type of 'getattr' (line 852)
        getattr_11620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 28), 'getattr', False)
        # Calling getattr(args, kwargs) (line 852)
        getattr_call_result_11625 = invoke(stypy.reporting.localization.Localization(__file__, 852, 28), getattr_11620, *[self_11621, contained_elements_property_name_11623], **kwargs_11624)
        
        # Assigning a type to the variable 'existing_type_map' (line 852)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 852, 8), 'existing_type_map', getattr_call_result_11625)
        
        # Assigning a Call to a Name (line 853):
        
        # Assigning a Call to a Name (line 853):
        
        # Call to keys(...): (line 853)
        # Processing the call keyword arguments (line 853)
        kwargs_11628 = {}
        # Getting the type of 'existing_type_map' (line 853)
        existing_type_map_11626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 15), 'existing_type_map', False)
        # Obtaining the member 'keys' of a type (line 853)
        keys_11627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 853, 15), existing_type_map_11626, 'keys')
        # Calling keys(args, kwargs) (line 853)
        keys_call_result_11629 = invoke(stypy.reporting.localization.Localization(__file__, 853, 15), keys_11627, *[], **kwargs_11628)
        
        # Assigning a type to the variable 'keys' (line 853)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 853, 8), 'keys', keys_call_result_11629)
        
        # Getting the type of 'keys' (line 854)
        keys_11630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 23), 'keys')
        # Assigning a type to the variable 'keys_11630' (line 854)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 854, 8), 'keys_11630', keys_11630)
        # Testing if the for loop is going to be iterated (line 854)
        # Testing the type of a for loop iterable (line 854)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 854, 8), keys_11630)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 854, 8), keys_11630):
            # Getting the type of the for loop variable (line 854)
            for_loop_var_11631 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 854, 8), keys_11630)
            # Assigning a type to the variable 'element' (line 854)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 854, 8), 'element', for_loop_var_11631)
            # SSA begins for a for statement (line 854)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'key' (line 855)
            key_11632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 15), 'key')
            # Getting the type of 'element' (line 855)
            element_11633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 22), 'element')
            # Applying the binary operator '==' (line 855)
            result_eq_11634 = python_operator(stypy.reporting.localization.Localization(__file__, 855, 15), '==', key_11632, element_11633)
            
            # Testing if the type of an if condition is none (line 855)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 855, 12), result_eq_11634):
                pass
            else:
                
                # Testing the type of an if condition (line 855)
                if_condition_11635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 855, 12), result_eq_11634)
                # Assigning a type to the variable 'if_condition_11635' (line 855)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 855, 12), 'if_condition_11635', if_condition_11635)
                # SSA begins for if statement (line 855)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'True' (line 856)
                True_11636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 23), 'True')
                # Assigning a type to the variable 'stypy_return_type' (line 856)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 856, 16), 'stypy_return_type', True_11636)
                # SSA join for if statement (line 855)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'False' (line 857)
        False_11637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 857)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 857, 8), 'stypy_return_type', False_11637)
        
        # ################# End of '__exist_key(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__exist_key' in the type store
        # Getting the type of 'stypy_return_type' (line 846)
        stypy_return_type_11638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11638)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__exist_key'
        return stypy_return_type_11638


    @norecursion
    def add_key_and_value_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 859)
        True_11639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 81), 'True')
        defaults = [True_11639]
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

        str_11640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 867, (-1)), 'str', '\n        Adds type_tuple to the elements stored by this type, returning an error if this is called over a proxy that\n        represent a non keypair holding Python type. It also checks if we are trying to store an undefined variable.\n        :param localization: Caller information\n        :param type_tuple: Tuple of types to store (key type, value type)\n        :param record_annotation: Whether to annotate the type change or not\n        :return: None or TypeError\n        ')
        
        # Assigning a Subscript to a Name (line 868):
        
        # Assigning a Subscript to a Name (line 868):
        
        # Obtaining the type of the subscript
        int_11641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 25), 'int')
        # Getting the type of 'type_tuple' (line 868)
        type_tuple_11642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 14), 'type_tuple')
        # Obtaining the member '__getitem__' of a type (line 868)
        getitem___11643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 868, 14), type_tuple_11642, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 868)
        subscript_call_result_11644 = invoke(stypy.reporting.localization.Localization(__file__, 868, 14), getitem___11643, int_11641)
        
        # Assigning a type to the variable 'key' (line 868)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 8), 'key', subscript_call_result_11644)
        
        # Assigning a Subscript to a Name (line 869):
        
        # Assigning a Subscript to a Name (line 869):
        
        # Obtaining the type of the subscript
        int_11645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 869, 27), 'int')
        # Getting the type of 'type_tuple' (line 869)
        type_tuple_11646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 16), 'type_tuple')
        # Obtaining the member '__getitem__' of a type (line 869)
        getitem___11647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 869, 16), type_tuple_11646, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 869)
        subscript_call_result_11648 = invoke(stypy.reporting.localization.Localization(__file__, 869, 16), getitem___11647, int_11645)
        
        # Assigning a type to the variable 'value' (line 869)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 869, 8), 'value', subscript_call_result_11648)
        
        
        # Call to can_store_keypairs(...): (line 871)
        # Processing the call keyword arguments (line 871)
        kwargs_11651 = {}
        # Getting the type of 'self' (line 871)
        self_11649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 15), 'self', False)
        # Obtaining the member 'can_store_keypairs' of a type (line 871)
        can_store_keypairs_11650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 15), self_11649, 'can_store_keypairs')
        # Calling can_store_keypairs(args, kwargs) (line 871)
        can_store_keypairs_call_result_11652 = invoke(stypy.reporting.localization.Localization(__file__, 871, 15), can_store_keypairs_11650, *[], **kwargs_11651)
        
        # Applying the 'not' unary operator (line 871)
        result_not__11653 = python_operator(stypy.reporting.localization.Localization(__file__, 871, 11), 'not', can_store_keypairs_call_result_11652)
        
        # Testing if the type of an if condition is none (line 871)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 871, 8), result_not__11653):
            pass
        else:
            
            # Testing the type of an if condition (line 871)
            if_condition_11654 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 871, 8), result_not__11653)
            # Assigning a type to the variable 'if_condition_11654' (line 871)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 871, 8), 'if_condition_11654', if_condition_11654)
            # SSA begins for if statement (line 871)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to can_store_elements(...): (line 872)
            # Processing the call keyword arguments (line 872)
            kwargs_11657 = {}
            # Getting the type of 'self' (line 872)
            self_11655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 19), 'self', False)
            # Obtaining the member 'can_store_elements' of a type (line 872)
            can_store_elements_11656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 872, 19), self_11655, 'can_store_elements')
            # Calling can_store_elements(args, kwargs) (line 872)
            can_store_elements_call_result_11658 = invoke(stypy.reporting.localization.Localization(__file__, 872, 19), can_store_elements_11656, *[], **kwargs_11657)
            
            # Applying the 'not' unary operator (line 872)
            result_not__11659 = python_operator(stypy.reporting.localization.Localization(__file__, 872, 15), 'not', can_store_elements_call_result_11658)
            
            # Testing if the type of an if condition is none (line 872)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 872, 12), result_not__11659):
                
                
                # Call to get_python_type(...): (line 877)
                # Processing the call keyword arguments (line 877)
                kwargs_11675 = {}
                # Getting the type of 'key' (line 877)
                key_11673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 19), 'key', False)
                # Obtaining the member 'get_python_type' of a type (line 877)
                get_python_type_11674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 19), key_11673, 'get_python_type')
                # Calling get_python_type(args, kwargs) (line 877)
                get_python_type_call_result_11676 = invoke(stypy.reporting.localization.Localization(__file__, 877, 19), get_python_type_11674, *[], **kwargs_11675)
                
                # Getting the type of 'int' (line 877)
                int_11677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 48), 'int')
                # Applying the binary operator 'isnot' (line 877)
                result_is_not_11678 = python_operator(stypy.reporting.localization.Localization(__file__, 877, 19), 'isnot', get_python_type_call_result_11676, int_11677)
                
                # Testing if the type of an if condition is none (line 877)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 877, 16), result_is_not_11678):
                    
                    # Call to add_type(...): (line 881)
                    # Processing the call arguments (line 881)
                    # Getting the type of 'localization' (line 881)
                    localization_11687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 34), 'localization', False)
                    # Getting the type of 'value' (line 881)
                    value_11688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 48), 'value', False)
                    # Getting the type of 'record_annotation' (line 881)
                    record_annotation_11689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 55), 'record_annotation', False)
                    # Processing the call keyword arguments (line 881)
                    kwargs_11690 = {}
                    # Getting the type of 'self' (line 881)
                    self_11685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 20), 'self', False)
                    # Obtaining the member 'add_type' of a type (line 881)
                    add_type_11686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 881, 20), self_11685, 'add_type')
                    # Calling add_type(args, kwargs) (line 881)
                    add_type_call_result_11691 = invoke(stypy.reporting.localization.Localization(__file__, 881, 20), add_type_11686, *[localization_11687, value_11688, record_annotation_11689], **kwargs_11690)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 882)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 20), 'stypy_return_type', types.NoneType)
                else:
                    
                    # Testing the type of an if condition (line 877)
                    if_condition_11679 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 877, 16), result_is_not_11678)
                    # Assigning a type to the variable 'if_condition_11679' (line 877)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 877, 16), 'if_condition_11679', if_condition_11679)
                    # SSA begins for if statement (line 877)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to TypeError(...): (line 878)
                    # Processing the call arguments (line 878)
                    # Getting the type of 'localization' (line 878)
                    localization_11681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 37), 'localization', False)
                    str_11682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 37), 'str', 'STYPY CRITICAL ERROR: Attempt to store keypairs on a python collection')
                    # Processing the call keyword arguments (line 878)
                    kwargs_11683 = {}
                    # Getting the type of 'TypeError' (line 878)
                    TypeError_11680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 27), 'TypeError', False)
                    # Calling TypeError(args, kwargs) (line 878)
                    TypeError_call_result_11684 = invoke(stypy.reporting.localization.Localization(__file__, 878, 27), TypeError_11680, *[localization_11681, str_11682], **kwargs_11683)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 878)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 20), 'stypy_return_type', TypeError_call_result_11684)
                    # SSA branch for the else part of an if statement (line 877)
                    module_type_store.open_ssa_branch('else')
                    
                    # Call to add_type(...): (line 881)
                    # Processing the call arguments (line 881)
                    # Getting the type of 'localization' (line 881)
                    localization_11687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 34), 'localization', False)
                    # Getting the type of 'value' (line 881)
                    value_11688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 48), 'value', False)
                    # Getting the type of 'record_annotation' (line 881)
                    record_annotation_11689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 55), 'record_annotation', False)
                    # Processing the call keyword arguments (line 881)
                    kwargs_11690 = {}
                    # Getting the type of 'self' (line 881)
                    self_11685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 20), 'self', False)
                    # Obtaining the member 'add_type' of a type (line 881)
                    add_type_11686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 881, 20), self_11685, 'add_type')
                    # Calling add_type(args, kwargs) (line 881)
                    add_type_call_result_11691 = invoke(stypy.reporting.localization.Localization(__file__, 881, 20), add_type_11686, *[localization_11687, value_11688, record_annotation_11689], **kwargs_11690)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 882)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 20), 'stypy_return_type', types.NoneType)
                    # SSA join for if statement (line 877)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 872)
                if_condition_11660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 872, 12), result_not__11659)
                # Assigning a type to the variable 'if_condition_11660' (line 872)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 872, 12), 'if_condition_11660', if_condition_11660)
                # SSA begins for if statement (line 872)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to TypeError(...): (line 873)
                # Processing the call arguments (line 873)
                # Getting the type of 'localization' (line 873)
                localization_11662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 33), 'localization', False)
                
                # Call to format(...): (line 874)
                # Processing the call arguments (line 874)
                
                # Call to get_python_type(...): (line 875)
                # Processing the call keyword arguments (line 875)
                kwargs_11667 = {}
                # Getting the type of 'self' (line 875)
                self_11665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 49), 'self', False)
                # Obtaining the member 'get_python_type' of a type (line 875)
                get_python_type_11666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 875, 49), self_11665, 'get_python_type')
                # Calling get_python_type(args, kwargs) (line 875)
                get_python_type_call_result_11668 = invoke(stypy.reporting.localization.Localization(__file__, 875, 49), get_python_type_11666, *[], **kwargs_11667)
                
                # Processing the call keyword arguments (line 874)
                kwargs_11669 = {}
                str_11663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, 33), 'str', 'STYPY CRITICAL ERROR: Attempt to store keypairs over a python type ({0}) that is nota dict')
                # Obtaining the member 'format' of a type (line 874)
                format_11664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 874, 33), str_11663, 'format')
                # Calling format(args, kwargs) (line 874)
                format_call_result_11670 = invoke(stypy.reporting.localization.Localization(__file__, 874, 33), format_11664, *[get_python_type_call_result_11668], **kwargs_11669)
                
                # Processing the call keyword arguments (line 873)
                kwargs_11671 = {}
                # Getting the type of 'TypeError' (line 873)
                TypeError_11661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 23), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 873)
                TypeError_call_result_11672 = invoke(stypy.reporting.localization.Localization(__file__, 873, 23), TypeError_11661, *[localization_11662, format_call_result_11670], **kwargs_11671)
                
                # Assigning a type to the variable 'stypy_return_type' (line 873)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 873, 16), 'stypy_return_type', TypeError_call_result_11672)
                # SSA branch for the else part of an if statement (line 872)
                module_type_store.open_ssa_branch('else')
                
                
                # Call to get_python_type(...): (line 877)
                # Processing the call keyword arguments (line 877)
                kwargs_11675 = {}
                # Getting the type of 'key' (line 877)
                key_11673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 19), 'key', False)
                # Obtaining the member 'get_python_type' of a type (line 877)
                get_python_type_11674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 19), key_11673, 'get_python_type')
                # Calling get_python_type(args, kwargs) (line 877)
                get_python_type_call_result_11676 = invoke(stypy.reporting.localization.Localization(__file__, 877, 19), get_python_type_11674, *[], **kwargs_11675)
                
                # Getting the type of 'int' (line 877)
                int_11677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 48), 'int')
                # Applying the binary operator 'isnot' (line 877)
                result_is_not_11678 = python_operator(stypy.reporting.localization.Localization(__file__, 877, 19), 'isnot', get_python_type_call_result_11676, int_11677)
                
                # Testing if the type of an if condition is none (line 877)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 877, 16), result_is_not_11678):
                    
                    # Call to add_type(...): (line 881)
                    # Processing the call arguments (line 881)
                    # Getting the type of 'localization' (line 881)
                    localization_11687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 34), 'localization', False)
                    # Getting the type of 'value' (line 881)
                    value_11688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 48), 'value', False)
                    # Getting the type of 'record_annotation' (line 881)
                    record_annotation_11689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 55), 'record_annotation', False)
                    # Processing the call keyword arguments (line 881)
                    kwargs_11690 = {}
                    # Getting the type of 'self' (line 881)
                    self_11685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 20), 'self', False)
                    # Obtaining the member 'add_type' of a type (line 881)
                    add_type_11686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 881, 20), self_11685, 'add_type')
                    # Calling add_type(args, kwargs) (line 881)
                    add_type_call_result_11691 = invoke(stypy.reporting.localization.Localization(__file__, 881, 20), add_type_11686, *[localization_11687, value_11688, record_annotation_11689], **kwargs_11690)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 882)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 20), 'stypy_return_type', types.NoneType)
                else:
                    
                    # Testing the type of an if condition (line 877)
                    if_condition_11679 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 877, 16), result_is_not_11678)
                    # Assigning a type to the variable 'if_condition_11679' (line 877)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 877, 16), 'if_condition_11679', if_condition_11679)
                    # SSA begins for if statement (line 877)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to TypeError(...): (line 878)
                    # Processing the call arguments (line 878)
                    # Getting the type of 'localization' (line 878)
                    localization_11681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 37), 'localization', False)
                    str_11682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 37), 'str', 'STYPY CRITICAL ERROR: Attempt to store keypairs on a python collection')
                    # Processing the call keyword arguments (line 878)
                    kwargs_11683 = {}
                    # Getting the type of 'TypeError' (line 878)
                    TypeError_11680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 27), 'TypeError', False)
                    # Calling TypeError(args, kwargs) (line 878)
                    TypeError_call_result_11684 = invoke(stypy.reporting.localization.Localization(__file__, 878, 27), TypeError_11680, *[localization_11681, str_11682], **kwargs_11683)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 878)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 20), 'stypy_return_type', TypeError_call_result_11684)
                    # SSA branch for the else part of an if statement (line 877)
                    module_type_store.open_ssa_branch('else')
                    
                    # Call to add_type(...): (line 881)
                    # Processing the call arguments (line 881)
                    # Getting the type of 'localization' (line 881)
                    localization_11687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 34), 'localization', False)
                    # Getting the type of 'value' (line 881)
                    value_11688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 48), 'value', False)
                    # Getting the type of 'record_annotation' (line 881)
                    record_annotation_11689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 55), 'record_annotation', False)
                    # Processing the call keyword arguments (line 881)
                    kwargs_11690 = {}
                    # Getting the type of 'self' (line 881)
                    self_11685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 20), 'self', False)
                    # Obtaining the member 'add_type' of a type (line 881)
                    add_type_11686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 881, 20), self_11685, 'add_type')
                    # Calling add_type(args, kwargs) (line 881)
                    add_type_call_result_11691 = invoke(stypy.reporting.localization.Localization(__file__, 881, 20), add_type_11686, *[localization_11687, value_11688, record_annotation_11689], **kwargs_11690)
                    
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
        self_11693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 23), 'self', False)
        # Getting the type of 'self' (line 884)
        self_11694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 29), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 884)
        contained_elements_property_name_11695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 884, 29), self_11694, 'contained_elements_property_name')
        # Processing the call keyword arguments (line 884)
        kwargs_11696 = {}
        # Getting the type of 'hasattr' (line 884)
        hasattr_11692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 15), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 884)
        hasattr_call_result_11697 = invoke(stypy.reporting.localization.Localization(__file__, 884, 15), hasattr_11692, *[self_11693, contained_elements_property_name_11695], **kwargs_11696)
        
        # Applying the 'not' unary operator (line 884)
        result_not__11698 = python_operator(stypy.reporting.localization.Localization(__file__, 884, 11), 'not', hasattr_call_result_11697)
        
        # Testing if the type of an if condition is none (line 884)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 884, 8), result_not__11698):
            pass
        else:
            
            # Testing the type of an if condition (line 884)
            if_condition_11699 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 884, 8), result_not__11698)
            # Assigning a type to the variable 'if_condition_11699' (line 884)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 884, 8), 'if_condition_11699', if_condition_11699)
            # SSA begins for if statement (line 884)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to setattr(...): (line 885)
            # Processing the call arguments (line 885)
            # Getting the type of 'self' (line 885)
            self_11701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 20), 'self', False)
            # Getting the type of 'self' (line 885)
            self_11702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 26), 'self', False)
            # Obtaining the member 'contained_elements_property_name' of a type (line 885)
            contained_elements_property_name_11703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 885, 26), self_11702, 'contained_elements_property_name')
            
            # Call to dict(...): (line 885)
            # Processing the call keyword arguments (line 885)
            kwargs_11705 = {}
            # Getting the type of 'dict' (line 885)
            dict_11704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 65), 'dict', False)
            # Calling dict(args, kwargs) (line 885)
            dict_call_result_11706 = invoke(stypy.reporting.localization.Localization(__file__, 885, 65), dict_11704, *[], **kwargs_11705)
            
            # Processing the call keyword arguments (line 885)
            kwargs_11707 = {}
            # Getting the type of 'setattr' (line 885)
            setattr_11700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 12), 'setattr', False)
            # Calling setattr(args, kwargs) (line 885)
            setattr_call_result_11708 = invoke(stypy.reporting.localization.Localization(__file__, 885, 12), setattr_11700, *[self_11701, contained_elements_property_name_11703, dict_call_result_11706], **kwargs_11707)
            
            # SSA join for if statement (line 884)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 887):
        
        # Assigning a Call to a Name (line 887):
        
        # Call to getattr(...): (line 887)
        # Processing the call arguments (line 887)
        # Getting the type of 'self' (line 887)
        self_11710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 36), 'self', False)
        # Getting the type of 'self' (line 887)
        self_11711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 42), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 887)
        contained_elements_property_name_11712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 42), self_11711, 'contained_elements_property_name')
        # Processing the call keyword arguments (line 887)
        kwargs_11713 = {}
        # Getting the type of 'getattr' (line 887)
        getattr_11709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 28), 'getattr', False)
        # Calling getattr(args, kwargs) (line 887)
        getattr_call_result_11714 = invoke(stypy.reporting.localization.Localization(__file__, 887, 28), getattr_11709, *[self_11710, contained_elements_property_name_11712], **kwargs_11713)
        
        # Assigning a type to the variable 'existing_type_map' (line 887)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 887, 8), 'existing_type_map', getattr_call_result_11714)
        
        # Call to __check_undefined_stored_value(...): (line 889)
        # Processing the call arguments (line 889)
        # Getting the type of 'localization' (line 889)
        localization_11717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 44), 'localization', False)
        # Getting the type of 'value' (line 889)
        value_11718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 58), 'value', False)
        # Processing the call keyword arguments (line 889)
        kwargs_11719 = {}
        # Getting the type of 'self' (line 889)
        self_11715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 8), 'self', False)
        # Obtaining the member '__check_undefined_stored_value' of a type (line 889)
        check_undefined_stored_value_11716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 889, 8), self_11715, '__check_undefined_stored_value')
        # Calling __check_undefined_stored_value(args, kwargs) (line 889)
        check_undefined_stored_value_call_result_11720 = invoke(stypy.reporting.localization.Localization(__file__, 889, 8), check_undefined_stored_value_11716, *[localization_11717, value_11718], **kwargs_11719)
        
        
        # Call to __exist_key(...): (line 892)
        # Processing the call arguments (line 892)
        # Getting the type of 'key' (line 892)
        key_11723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 28), 'key', False)
        # Processing the call keyword arguments (line 892)
        kwargs_11724 = {}
        # Getting the type of 'self' (line 892)
        self_11721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 11), 'self', False)
        # Obtaining the member '__exist_key' of a type (line 892)
        exist_key_11722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 892, 11), self_11721, '__exist_key')
        # Calling __exist_key(args, kwargs) (line 892)
        exist_key_call_result_11725 = invoke(stypy.reporting.localization.Localization(__file__, 892, 11), exist_key_11722, *[key_11723], **kwargs_11724)
        
        # Testing if the type of an if condition is none (line 892)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 892, 8), exist_key_call_result_11725):
            
            # Assigning a Name to a Subscript (line 899):
            
            # Assigning a Name to a Subscript (line 899):
            # Getting the type of 'value' (line 899)
            value_11755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 37), 'value')
            # Getting the type of 'existing_type_map' (line 899)
            existing_type_map_11756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 12), 'existing_type_map')
            # Getting the type of 'key' (line 899)
            key_11757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 30), 'key')
            # Storing an element on a container (line 899)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 899, 12), existing_type_map_11756, (key_11757, value_11755))
        else:
            
            # Testing the type of an if condition (line 892)
            if_condition_11726 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 892, 8), exist_key_call_result_11725)
            # Assigning a type to the variable 'if_condition_11726' (line 892)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 8), 'if_condition_11726', if_condition_11726)
            # SSA begins for if statement (line 892)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 894):
            
            # Assigning a Call to a Name (line 894):
            
            # Call to index(...): (line 894)
            # Processing the call arguments (line 894)
            # Getting the type of 'key' (line 894)
            key_11732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 62), 'key', False)
            # Processing the call keyword arguments (line 894)
            kwargs_11733 = {}
            
            # Call to keys(...): (line 894)
            # Processing the call keyword arguments (line 894)
            kwargs_11729 = {}
            # Getting the type of 'existing_type_map' (line 894)
            existing_type_map_11727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 31), 'existing_type_map', False)
            # Obtaining the member 'keys' of a type (line 894)
            keys_11728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 894, 31), existing_type_map_11727, 'keys')
            # Calling keys(args, kwargs) (line 894)
            keys_call_result_11730 = invoke(stypy.reporting.localization.Localization(__file__, 894, 31), keys_11728, *[], **kwargs_11729)
            
            # Obtaining the member 'index' of a type (line 894)
            index_11731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 894, 31), keys_call_result_11730, 'index')
            # Calling index(args, kwargs) (line 894)
            index_call_result_11734 = invoke(stypy.reporting.localization.Localization(__file__, 894, 31), index_11731, *[key_11732], **kwargs_11733)
            
            # Assigning a type to the variable 'stored_key_index' (line 894)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 894, 12), 'stored_key_index', index_call_result_11734)
            
            # Assigning a Subscript to a Name (line 895):
            
            # Assigning a Subscript to a Name (line 895):
            
            # Obtaining the type of the subscript
            # Getting the type of 'stored_key_index' (line 895)
            stored_key_index_11735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 50), 'stored_key_index')
            
            # Call to keys(...): (line 895)
            # Processing the call keyword arguments (line 895)
            kwargs_11738 = {}
            # Getting the type of 'existing_type_map' (line 895)
            existing_type_map_11736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 25), 'existing_type_map', False)
            # Obtaining the member 'keys' of a type (line 895)
            keys_11737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 895, 25), existing_type_map_11736, 'keys')
            # Calling keys(args, kwargs) (line 895)
            keys_call_result_11739 = invoke(stypy.reporting.localization.Localization(__file__, 895, 25), keys_11737, *[], **kwargs_11738)
            
            # Obtaining the member '__getitem__' of a type (line 895)
            getitem___11740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 895, 25), keys_call_result_11739, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 895)
            subscript_call_result_11741 = invoke(stypy.reporting.localization.Localization(__file__, 895, 25), getitem___11740, stored_key_index_11735)
            
            # Assigning a type to the variable 'stored_key' (line 895)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 895, 12), 'stored_key', subscript_call_result_11741)
            
            # Assigning a Subscript to a Name (line 896):
            
            # Assigning a Subscript to a Name (line 896):
            
            # Obtaining the type of the subscript
            # Getting the type of 'stored_key' (line 896)
            stored_key_11742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 46), 'stored_key')
            # Getting the type of 'existing_type_map' (line 896)
            existing_type_map_11743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 28), 'existing_type_map')
            # Obtaining the member '__getitem__' of a type (line 896)
            getitem___11744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 896, 28), existing_type_map_11743, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 896)
            subscript_call_result_11745 = invoke(stypy.reporting.localization.Localization(__file__, 896, 28), getitem___11744, stored_key_11742)
            
            # Assigning a type to the variable 'existing_type' (line 896)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 896, 12), 'existing_type', subscript_call_result_11745)
            
            # Assigning a Call to a Subscript (line 897):
            
            # Assigning a Call to a Subscript (line 897):
            
            # Call to add(...): (line 897)
            # Processing the call arguments (line 897)
            # Getting the type of 'existing_type' (line 897)
            existing_type_11749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 74), 'existing_type', False)
            # Getting the type of 'value' (line 897)
            value_11750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 89), 'value', False)
            # Processing the call keyword arguments (line 897)
            kwargs_11751 = {}
            # Getting the type of 'union_type_copy' (line 897)
            union_type_copy_11746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 44), 'union_type_copy', False)
            # Obtaining the member 'UnionType' of a type (line 897)
            UnionType_11747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 44), union_type_copy_11746, 'UnionType')
            # Obtaining the member 'add' of a type (line 897)
            add_11748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 44), UnionType_11747, 'add')
            # Calling add(args, kwargs) (line 897)
            add_call_result_11752 = invoke(stypy.reporting.localization.Localization(__file__, 897, 44), add_11748, *[existing_type_11749, value_11750], **kwargs_11751)
            
            # Getting the type of 'existing_type_map' (line 897)
            existing_type_map_11753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 12), 'existing_type_map')
            # Getting the type of 'stored_key' (line 897)
            stored_key_11754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 30), 'stored_key')
            # Storing an element on a container (line 897)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 12), existing_type_map_11753, (stored_key_11754, add_call_result_11752))
            # SSA branch for the else part of an if statement (line 892)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Subscript (line 899):
            
            # Assigning a Name to a Subscript (line 899):
            # Getting the type of 'value' (line 899)
            value_11755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 37), 'value')
            # Getting the type of 'existing_type_map' (line 899)
            existing_type_map_11756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 12), 'existing_type_map')
            # Getting the type of 'key' (line 899)
            key_11757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 30), 'key')
            # Storing an element on a container (line 899)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 899, 12), existing_type_map_11756, (key_11757, value_11755))
            # SSA join for if statement (line 892)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Evaluating a boolean operation
        # Getting the type of 'record_annotation' (line 901)
        record_annotation_11758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 11), 'record_annotation')
        # Getting the type of 'self' (line 901)
        self_11759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 33), 'self')
        # Obtaining the member 'annotate_types' of a type (line 901)
        annotate_types_11760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 901, 33), self_11759, 'annotate_types')
        # Applying the binary operator 'and' (line 901)
        result_and_keyword_11761 = python_operator(stypy.reporting.localization.Localization(__file__, 901, 11), 'and', record_annotation_11758, annotate_types_11760)
        
        # Testing if the type of an if condition is none (line 901)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 901, 8), result_and_keyword_11761):
            pass
        else:
            
            # Testing the type of an if condition (line 901)
            if_condition_11762 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 901, 8), result_and_keyword_11761)
            # Assigning a type to the variable 'if_condition_11762' (line 901)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 901, 8), 'if_condition_11762', if_condition_11762)
            # SSA begins for if statement (line 901)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to __annotate_type(...): (line 902)
            # Processing the call arguments (line 902)
            # Getting the type of 'localization' (line 902)
            localization_11765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 33), 'localization', False)
            # Obtaining the member 'line' of a type (line 902)
            line_11766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 902, 33), localization_11765, 'line')
            # Getting the type of 'localization' (line 902)
            localization_11767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 52), 'localization', False)
            # Obtaining the member 'column' of a type (line 902)
            column_11768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 902, 52), localization_11767, 'column')
            str_11769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 902, 73), 'str', '<dictionary elements type>')
            
            # Call to getattr(...): (line 903)
            # Processing the call arguments (line 903)
            # Getting the type of 'self' (line 903)
            self_11771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 41), 'self', False)
            # Getting the type of 'self' (line 903)
            self_11772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 47), 'self', False)
            # Obtaining the member 'contained_elements_property_name' of a type (line 903)
            contained_elements_property_name_11773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 903, 47), self_11772, 'contained_elements_property_name')
            # Processing the call keyword arguments (line 903)
            kwargs_11774 = {}
            # Getting the type of 'getattr' (line 903)
            getattr_11770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 33), 'getattr', False)
            # Calling getattr(args, kwargs) (line 903)
            getattr_call_result_11775 = invoke(stypy.reporting.localization.Localization(__file__, 903, 33), getattr_11770, *[self_11771, contained_elements_property_name_11773], **kwargs_11774)
            
            # Processing the call keyword arguments (line 902)
            kwargs_11776 = {}
            # Getting the type of 'self' (line 902)
            self_11763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 12), 'self', False)
            # Obtaining the member '__annotate_type' of a type (line 902)
            annotate_type_11764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 902, 12), self_11763, '__annotate_type')
            # Calling __annotate_type(args, kwargs) (line 902)
            annotate_type_call_result_11777 = invoke(stypy.reporting.localization.Localization(__file__, 902, 12), annotate_type_11764, *[line_11766, column_11768, str_11769, getattr_call_result_11775], **kwargs_11776)
            
            # SSA join for if statement (line 901)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'add_key_and_value_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_key_and_value_type' in the type store
        # Getting the type of 'stypy_return_type' (line 859)
        stypy_return_type_11778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11778)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_key_and_value_type'
        return stypy_return_type_11778


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

        str_11779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 912, (-1)), 'str', '\n        Get the poosible values associated to a key type on a keypair storing proxy\n\n        :param localization: Caller information\n        :param key: Key type\n        :return: Value type list\n        ')
        
        # Assigning a Call to a Name (line 913):
        
        # Assigning a Call to a Name (line 913):
        
        # Call to getattr(...): (line 913)
        # Processing the call arguments (line 913)
        # Getting the type of 'self' (line 913)
        self_11781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 36), 'self', False)
        # Getting the type of 'self' (line 913)
        self_11782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 42), 'self', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 913)
        contained_elements_property_name_11783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 913, 42), self_11782, 'contained_elements_property_name')
        # Processing the call keyword arguments (line 913)
        kwargs_11784 = {}
        # Getting the type of 'getattr' (line 913)
        getattr_11780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 28), 'getattr', False)
        # Calling getattr(args, kwargs) (line 913)
        getattr_call_result_11785 = invoke(stypy.reporting.localization.Localization(__file__, 913, 28), getattr_11780, *[self_11781, contained_elements_property_name_11783], **kwargs_11784)
        
        # Assigning a type to the variable 'existing_type_map' (line 913)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 913, 8), 'existing_type_map', getattr_call_result_11785)
        
        
        # SSA begins for try-except statement (line 915)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 917):
        
        # Assigning a Call to a Name (line 917):
        
        # Call to index(...): (line 917)
        # Processing the call arguments (line 917)
        # Getting the type of 'key' (line 917)
        key_11791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 62), 'key', False)
        # Processing the call keyword arguments (line 917)
        kwargs_11792 = {}
        
        # Call to keys(...): (line 917)
        # Processing the call keyword arguments (line 917)
        kwargs_11788 = {}
        # Getting the type of 'existing_type_map' (line 917)
        existing_type_map_11786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 31), 'existing_type_map', False)
        # Obtaining the member 'keys' of a type (line 917)
        keys_11787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 917, 31), existing_type_map_11786, 'keys')
        # Calling keys(args, kwargs) (line 917)
        keys_call_result_11789 = invoke(stypy.reporting.localization.Localization(__file__, 917, 31), keys_11787, *[], **kwargs_11788)
        
        # Obtaining the member 'index' of a type (line 917)
        index_11790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 917, 31), keys_call_result_11789, 'index')
        # Calling index(args, kwargs) (line 917)
        index_call_result_11793 = invoke(stypy.reporting.localization.Localization(__file__, 917, 31), index_11790, *[key_11791], **kwargs_11792)
        
        # Assigning a type to the variable 'stored_key_index' (line 917)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 917, 12), 'stored_key_index', index_call_result_11793)
        
        # Assigning a Subscript to a Name (line 918):
        
        # Assigning a Subscript to a Name (line 918):
        
        # Obtaining the type of the subscript
        # Getting the type of 'stored_key_index' (line 918)
        stored_key_index_11794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 50), 'stored_key_index')
        
        # Call to keys(...): (line 918)
        # Processing the call keyword arguments (line 918)
        kwargs_11797 = {}
        # Getting the type of 'existing_type_map' (line 918)
        existing_type_map_11795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 25), 'existing_type_map', False)
        # Obtaining the member 'keys' of a type (line 918)
        keys_11796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 918, 25), existing_type_map_11795, 'keys')
        # Calling keys(args, kwargs) (line 918)
        keys_call_result_11798 = invoke(stypy.reporting.localization.Localization(__file__, 918, 25), keys_11796, *[], **kwargs_11797)
        
        # Obtaining the member '__getitem__' of a type (line 918)
        getitem___11799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 918, 25), keys_call_result_11798, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 918)
        subscript_call_result_11800 = invoke(stypy.reporting.localization.Localization(__file__, 918, 25), getitem___11799, stored_key_index_11794)
        
        # Assigning a type to the variable 'stored_key' (line 918)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 918, 12), 'stored_key', subscript_call_result_11800)
        
        # Assigning a Subscript to a Name (line 919):
        
        # Assigning a Subscript to a Name (line 919):
        
        # Obtaining the type of the subscript
        # Getting the type of 'stored_key' (line 919)
        stored_key_11801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 38), 'stored_key')
        # Getting the type of 'existing_type_map' (line 919)
        existing_type_map_11802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 20), 'existing_type_map')
        # Obtaining the member '__getitem__' of a type (line 919)
        getitem___11803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 919, 20), existing_type_map_11802, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 919)
        subscript_call_result_11804 = invoke(stypy.reporting.localization.Localization(__file__, 919, 20), getitem___11803, stored_key_11801)
        
        # Assigning a type to the variable 'value' (line 919)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 919, 12), 'value', subscript_call_result_11804)
        # Getting the type of 'value' (line 920)
        value_11805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 19), 'value')
        # Assigning a type to the variable 'stypy_return_type' (line 920)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 920, 12), 'stypy_return_type', value_11805)
        # SSA branch for the except part of a try statement (line 915)
        # SSA branch for the except '<any exception>' branch of a try statement (line 915)
        module_type_store.open_ssa_branch('except')
        
        # Call to TypeError(...): (line 922)
        # Processing the call arguments (line 922)
        # Getting the type of 'localization' (line 922)
        localization_11807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 29), 'localization', False)
        
        # Call to format(...): (line 922)
        # Processing the call arguments (line 922)
        # Getting the type of 'key' (line 922)
        key_11810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 93), 'key', False)
        # Processing the call keyword arguments (line 922)
        kwargs_11811 = {}
        str_11808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 43), 'str', "No value is associated to key type '{0}'")
        # Obtaining the member 'format' of a type (line 922)
        format_11809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 43), str_11808, 'format')
        # Calling format(args, kwargs) (line 922)
        format_call_result_11812 = invoke(stypy.reporting.localization.Localization(__file__, 922, 43), format_11809, *[key_11810], **kwargs_11811)
        
        # Processing the call keyword arguments (line 922)
        kwargs_11813 = {}
        # Getting the type of 'TypeError' (line 922)
        TypeError_11806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 19), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 922)
        TypeError_call_result_11814 = invoke(stypy.reporting.localization.Localization(__file__, 922, 19), TypeError_11806, *[localization_11807, format_call_result_11812], **kwargs_11813)
        
        # Assigning a type to the variable 'stypy_return_type' (line 922)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 12), 'stypy_return_type', TypeError_call_result_11814)
        # SSA join for try-except statement (line 915)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_values_from_key(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_values_from_key' in the type store
        # Getting the type of 'stypy_return_type' (line 905)
        stypy_return_type_11815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11815)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_values_from_key'
        return stypy_return_type_11815


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

        str_11816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 933, (-1)), 'str', '\n        Determines whether the stored python entity supports intercession. This means that this proxy stores an\n        instance (which are created precisely for this purpose) or the stored entity has a dict as the type of\n        its __dict__ property (and not a dictproxy instance, that is read-only).\n\n        :return: bool\n        ')
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 934)
        self_11817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 15), 'self')
        # Obtaining the member 'instance' of a type (line 934)
        instance_11818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 934, 15), self_11817, 'instance')
        # Getting the type of 'None' (line 934)
        None_11819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 36), 'None')
        # Applying the binary operator 'isnot' (line 934)
        result_is_not_11820 = python_operator(stypy.reporting.localization.Localization(__file__, 934, 15), 'isnot', instance_11818, None_11819)
        
        
        # Call to supports_structural_reflection(...): (line 934)
        # Processing the call arguments (line 934)
        # Getting the type of 'self' (line 935)
        self_11823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 12), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 935)
        python_entity_11824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 935, 12), self_11823, 'python_entity')
        # Processing the call keyword arguments (line 934)
        kwargs_11825 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 934)
        type_inference_proxy_management_copy_11821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 44), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'supports_structural_reflection' of a type (line 934)
        supports_structural_reflection_11822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 934, 44), type_inference_proxy_management_copy_11821, 'supports_structural_reflection')
        # Calling supports_structural_reflection(args, kwargs) (line 934)
        supports_structural_reflection_call_result_11826 = invoke(stypy.reporting.localization.Localization(__file__, 934, 44), supports_structural_reflection_11822, *[python_entity_11824], **kwargs_11825)
        
        # Applying the binary operator 'or' (line 934)
        result_or_keyword_11827 = python_operator(stypy.reporting.localization.Localization(__file__, 934, 15), 'or', result_is_not_11820, supports_structural_reflection_call_result_11826)
        
        # Assigning a type to the variable 'stypy_return_type' (line 934)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 934, 8), 'stypy_return_type', result_or_keyword_11827)
        
        # ################# End of 'supports_structural_reflection(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'supports_structural_reflection' in the type store
        # Getting the type of 'stypy_return_type' (line 926)
        stypy_return_type_11828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11828)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'supports_structural_reflection'
        return stypy_return_type_11828


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

        str_11829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 948, (-1)), 'str', '\n        Set the type of the member whose name is passed to the specified value. There are cases in which deepcopies of\n        the stored python entities are not supported when cloning the type proxy (cloning is needed for SSA), but\n        structural reflection is supported. Therefore, the additional_members attribute have to be created to still\n        support structural reflection while maintaining the ability to create fully independent clones of the stored\n        python entity.\n\n        :param localization: Call localization data\n        :param member_name: Member name\n        :return:\n        ')
        
        
        # SSA begins for try-except statement (line 949)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Getting the type of 'self' (line 950)
        self_11830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 15), 'self')
        # Obtaining the member 'instance' of a type (line 950)
        instance_11831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 950, 15), self_11830, 'instance')
        # Getting the type of 'None' (line 950)
        None_11832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 36), 'None')
        # Applying the binary operator 'isnot' (line 950)
        result_is_not_11833 = python_operator(stypy.reporting.localization.Localization(__file__, 950, 15), 'isnot', instance_11831, None_11832)
        
        # Testing if the type of an if condition is none (line 950)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 950, 12), result_is_not_11833):
            pass
        else:
            
            # Testing the type of an if condition (line 950)
            if_condition_11834 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 950, 12), result_is_not_11833)
            # Assigning a type to the variable 'if_condition_11834' (line 950)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 950, 12), 'if_condition_11834', if_condition_11834)
            # SSA begins for if statement (line 950)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to delattr(...): (line 952)
            # Processing the call arguments (line 952)
            # Getting the type of 'self' (line 952)
            self_11836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 24), 'self', False)
            # Obtaining the member 'instance' of a type (line 952)
            instance_11837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 952, 24), self_11836, 'instance')
            # Getting the type of 'member_name' (line 952)
            member_name_11838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 39), 'member_name', False)
            # Processing the call keyword arguments (line 952)
            kwargs_11839 = {}
            # Getting the type of 'delattr' (line 952)
            delattr_11835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 16), 'delattr', False)
            # Calling delattr(args, kwargs) (line 952)
            delattr_call_result_11840 = invoke(stypy.reporting.localization.Localization(__file__, 952, 16), delattr_11835, *[instance_11837, member_name_11838], **kwargs_11839)
            
            # Getting the type of 'None' (line 953)
            None_11841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 23), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 953)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 953, 16), 'stypy_return_type', None_11841)
            # SSA join for if statement (line 950)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to supports_structural_reflection(...): (line 955)
        # Processing the call arguments (line 955)
        # Getting the type of 'self' (line 955)
        self_11844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 83), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 955)
        python_entity_11845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 83), self_11844, 'python_entity')
        # Processing the call keyword arguments (line 955)
        kwargs_11846 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 955)
        type_inference_proxy_management_copy_11842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 15), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'supports_structural_reflection' of a type (line 955)
        supports_structural_reflection_11843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 15), type_inference_proxy_management_copy_11842, 'supports_structural_reflection')
        # Calling supports_structural_reflection(args, kwargs) (line 955)
        supports_structural_reflection_call_result_11847 = invoke(stypy.reporting.localization.Localization(__file__, 955, 15), supports_structural_reflection_11843, *[python_entity_11845], **kwargs_11846)
        
        # Testing if the type of an if condition is none (line 955)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 955, 12), supports_structural_reflection_call_result_11847):
            pass
        else:
            
            # Testing the type of an if condition (line 955)
            if_condition_11848 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 955, 12), supports_structural_reflection_call_result_11847)
            # Assigning a type to the variable 'if_condition_11848' (line 955)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 955, 12), 'if_condition_11848', if_condition_11848)
            # SSA begins for if statement (line 955)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to delattr(...): (line 957)
            # Processing the call arguments (line 957)
            # Getting the type of 'self' (line 957)
            self_11850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 24), 'self', False)
            # Obtaining the member 'python_entity' of a type (line 957)
            python_entity_11851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 957, 24), self_11850, 'python_entity')
            # Getting the type of 'member_name' (line 957)
            member_name_11852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 44), 'member_name', False)
            # Processing the call keyword arguments (line 957)
            kwargs_11853 = {}
            # Getting the type of 'delattr' (line 957)
            delattr_11849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 16), 'delattr', False)
            # Calling delattr(args, kwargs) (line 957)
            delattr_call_result_11854 = invoke(stypy.reporting.localization.Localization(__file__, 957, 16), delattr_11849, *[python_entity_11851, member_name_11852], **kwargs_11853)
            
            # Getting the type of 'None' (line 958)
            None_11855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 23), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 958)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 958, 16), 'stypy_return_type', None_11855)
            # SSA join for if statement (line 955)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA branch for the except part of a try statement (line 949)
        # SSA branch for the except 'Exception' branch of a try statement (line 949)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'Exception' (line 959)
        Exception_11856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 15), 'Exception')
        # Assigning a type to the variable 'exc' (line 959)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 959, 8), 'exc', Exception_11856)
        
        # Call to TypeError(...): (line 960)
        # Processing the call arguments (line 960)
        # Getting the type of 'localization' (line 960)
        localization_11858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 29), 'localization', False)
        
        # Call to format(...): (line 961)
        # Processing the call arguments (line 961)
        
        # Call to __repr__(...): (line 962)
        # Processing the call keyword arguments (line 962)
        kwargs_11863 = {}
        # Getting the type of 'self' (line 962)
        self_11861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 36), 'self', False)
        # Obtaining the member '__repr__' of a type (line 962)
        repr___11862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 962, 36), self_11861, '__repr__')
        # Calling __repr__(args, kwargs) (line 962)
        repr___call_result_11864 = invoke(stypy.reporting.localization.Localization(__file__, 962, 36), repr___11862, *[], **kwargs_11863)
        
        
        # Call to str(...): (line 962)
        # Processing the call arguments (line 962)
        # Getting the type of 'exc' (line 962)
        exc_11866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 57), 'exc', False)
        # Processing the call keyword arguments (line 962)
        kwargs_11867 = {}
        # Getting the type of 'str' (line 962)
        str_11865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 53), 'str', False)
        # Calling str(args, kwargs) (line 962)
        str_call_result_11868 = invoke(stypy.reporting.localization.Localization(__file__, 962, 53), str_11865, *[exc_11866], **kwargs_11867)
        
        # Getting the type of 'member_name' (line 962)
        member_name_11869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 63), 'member_name', False)
        # Processing the call keyword arguments (line 961)
        kwargs_11870 = {}
        str_11859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 961, 29), 'str', "'{2}' member deletion is impossible: Cannot modify the structure of '{0}': {1}")
        # Obtaining the member 'format' of a type (line 961)
        format_11860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 961, 29), str_11859, 'format')
        # Calling format(args, kwargs) (line 961)
        format_call_result_11871 = invoke(stypy.reporting.localization.Localization(__file__, 961, 29), format_11860, *[repr___call_result_11864, str_call_result_11868, member_name_11869], **kwargs_11870)
        
        # Processing the call keyword arguments (line 960)
        kwargs_11872 = {}
        # Getting the type of 'TypeError' (line 960)
        TypeError_11857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 19), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 960)
        TypeError_call_result_11873 = invoke(stypy.reporting.localization.Localization(__file__, 960, 19), TypeError_11857, *[localization_11858, format_call_result_11871], **kwargs_11872)
        
        # Assigning a type to the variable 'stypy_return_type' (line 960)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 960, 12), 'stypy_return_type', TypeError_call_result_11873)
        # SSA join for try-except statement (line 949)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to TypeError(...): (line 964)
        # Processing the call arguments (line 964)
        # Getting the type of 'localization' (line 964)
        localization_11875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 25), 'localization', False)
        
        # Call to format(...): (line 965)
        # Processing the call arguments (line 965)
        # Getting the type of 'member_name' (line 966)
        member_name_11878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 51), 'member_name', False)
        # Processing the call keyword arguments (line 965)
        kwargs_11879 = {}
        str_11876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 965, 25), 'str', "'{0}' member deletion is impossible: Cannot modify the structure of a python library type or instance")
        # Obtaining the member 'format' of a type (line 965)
        format_11877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 965, 25), str_11876, 'format')
        # Calling format(args, kwargs) (line 965)
        format_call_result_11880 = invoke(stypy.reporting.localization.Localization(__file__, 965, 25), format_11877, *[member_name_11878], **kwargs_11879)
        
        # Processing the call keyword arguments (line 964)
        kwargs_11881 = {}
        # Getting the type of 'TypeError' (line 964)
        TypeError_11874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 15), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 964)
        TypeError_call_result_11882 = invoke(stypy.reporting.localization.Localization(__file__, 964, 15), TypeError_11874, *[localization_11875, format_call_result_11880], **kwargs_11881)
        
        # Assigning a type to the variable 'stypy_return_type' (line 964)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 964, 8), 'stypy_return_type', TypeError_call_result_11882)
        
        # ################# End of 'delete_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'delete_member' in the type store
        # Getting the type of 'stypy_return_type' (line 937)
        stypy_return_type_11883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11883)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'delete_member'
        return stypy_return_type_11883


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

        str_11884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 981, (-1)), 'str', "\n        Changes the type of the stored entity, provided it is an instance (so it supports structural reflection).\n        Type change is only available in Python for instances of user-defined classes.\n\n        You can only assign to the __class__ attribute of an instance of a user-defined class\n        (i.e. defined using the class keyword), and the new value must also be a user-defined class.\n        Whether the classes are new-style or old-style does not matter. (You can't mix them, though.\n        You can't turn an old-style class instance into a new-style class instance.)\n\n        :param localization: Call localization data\n        :param new_type: New type of the instance.\n        :return: A TypeError or None\n        ")
        
        # Assigning a Call to a Name (line 982):
        
        # Assigning a Call to a Name (line 982):
        
        # Call to __change_instance_type_checks(...): (line 982)
        # Processing the call arguments (line 982)
        # Getting the type of 'localization' (line 982)
        localization_11887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 52), 'localization', False)
        # Getting the type of 'new_type' (line 982)
        new_type_11888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 66), 'new_type', False)
        # Processing the call keyword arguments (line 982)
        kwargs_11889 = {}
        # Getting the type of 'self' (line 982)
        self_11885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 17), 'self', False)
        # Obtaining the member '__change_instance_type_checks' of a type (line 982)
        change_instance_type_checks_11886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 982, 17), self_11885, '__change_instance_type_checks')
        # Calling __change_instance_type_checks(args, kwargs) (line 982)
        change_instance_type_checks_call_result_11890 = invoke(stypy.reporting.localization.Localization(__file__, 982, 17), change_instance_type_checks_11886, *[localization_11887, new_type_11888], **kwargs_11889)
        
        # Assigning a type to the variable 'result' (line 982)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 982, 8), 'result', change_instance_type_checks_call_result_11890)
        
        # Type idiom detected: calculating its left and rigth part (line 984)
        # Getting the type of 'TypeError' (line 984)
        TypeError_11891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 30), 'TypeError')
        # Getting the type of 'result' (line 984)
        result_11892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 22), 'result')
        
        (may_be_11893, more_types_in_union_11894) = may_be_subtype(TypeError_11891, result_11892)

        if may_be_11893:

            if more_types_in_union_11894:
                # Runtime conditional SSA (line 984)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'result' (line 984)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 984, 8), 'result', remove_not_subtype_from_union(result_11892, TypeError))
            # Getting the type of 'result' (line 985)
            result_11895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 19), 'result')
            # Assigning a type to the variable 'stypy_return_type' (line 985)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 985, 12), 'stypy_return_type', result_11895)

            if more_types_in_union_11894:
                # SSA join for if statement (line 984)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to is_user_defined_class(...): (line 988)
        # Processing the call arguments (line 988)
        # Getting the type of 'new_type' (line 988)
        new_type_11898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 70), 'new_type', False)
        # Obtaining the member 'python_entity' of a type (line 988)
        python_entity_11899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 988, 70), new_type_11898, 'python_entity')
        # Processing the call keyword arguments (line 988)
        kwargs_11900 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 988)
        type_inference_proxy_management_copy_11896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 11), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'is_user_defined_class' of a type (line 988)
        is_user_defined_class_11897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 988, 11), type_inference_proxy_management_copy_11896, 'is_user_defined_class')
        # Calling is_user_defined_class(args, kwargs) (line 988)
        is_user_defined_class_call_result_11901 = invoke(stypy.reporting.localization.Localization(__file__, 988, 11), is_user_defined_class_11897, *[python_entity_11899], **kwargs_11900)
        
        # Testing if the type of an if condition is none (line 988)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 988, 8), is_user_defined_class_call_result_11901):
            
            # Assigning a Attribute to a Attribute (line 991):
            
            # Assigning a Attribute to a Attribute (line 991):
            # Getting the type of 'new_type' (line 991)
            new_type_11906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 33), 'new_type')
            # Obtaining the member 'python_entity' of a type (line 991)
            python_entity_11907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 991, 33), new_type_11906, 'python_entity')
            # Getting the type of 'self' (line 991)
            self_11908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 12), 'self')
            # Setting the type of the member 'python_entity' of a type (line 991)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 991, 12), self_11908, 'python_entity', python_entity_11907)
        else:
            
            # Testing the type of an if condition (line 988)
            if_condition_11902 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 988, 8), is_user_defined_class_call_result_11901)
            # Assigning a type to the variable 'if_condition_11902' (line 988)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 988, 8), 'if_condition_11902', if_condition_11902)
            # SSA begins for if statement (line 988)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Attribute (line 989):
            
            # Assigning a Attribute to a Attribute (line 989):
            # Getting the type of 'types' (line 989)
            types_11903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 33), 'types')
            # Obtaining the member 'InstanceType' of a type (line 989)
            InstanceType_11904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 989, 33), types_11903, 'InstanceType')
            # Getting the type of 'self' (line 989)
            self_11905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 12), 'self')
            # Setting the type of the member 'python_entity' of a type (line 989)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 989, 12), self_11905, 'python_entity', InstanceType_11904)
            # SSA branch for the else part of an if statement (line 988)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Attribute to a Attribute (line 991):
            
            # Assigning a Attribute to a Attribute (line 991):
            # Getting the type of 'new_type' (line 991)
            new_type_11906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 33), 'new_type')
            # Obtaining the member 'python_entity' of a type (line 991)
            python_entity_11907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 991, 33), new_type_11906, 'python_entity')
            # Getting the type of 'self' (line 991)
            self_11908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 12), 'self')
            # Setting the type of the member 'python_entity' of a type (line 991)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 991, 12), self_11908, 'python_entity', python_entity_11907)
            # SSA join for if statement (line 988)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to setattr(...): (line 993)
        # Processing the call arguments (line 993)
        # Getting the type of 'self' (line 993)
        self_11910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 16), 'self', False)
        # Obtaining the member 'instance' of a type (line 993)
        instance_11911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 993, 16), self_11910, 'instance')
        str_11912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 993, 31), 'str', '__class__')
        # Getting the type of 'new_type' (line 993)
        new_type_11913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 44), 'new_type', False)
        # Obtaining the member 'python_entity' of a type (line 993)
        python_entity_11914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 993, 44), new_type_11913, 'python_entity')
        # Processing the call keyword arguments (line 993)
        kwargs_11915 = {}
        # Getting the type of 'setattr' (line 993)
        setattr_11909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 8), 'setattr', False)
        # Calling setattr(args, kwargs) (line 993)
        setattr_call_result_11916 = invoke(stypy.reporting.localization.Localization(__file__, 993, 8), setattr_11909, *[instance_11911, str_11912, python_entity_11914], **kwargs_11915)
        
        # Getting the type of 'None' (line 994)
        None_11917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 994, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 994)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 994, 8), 'stypy_return_type', None_11917)
        
        # ################# End of 'change_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'change_type' in the type store
        # Getting the type of 'stypy_return_type' (line 968)
        stypy_return_type_11918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11918)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'change_type'
        return stypy_return_type_11918


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

        str_11919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1008, (-1)), 'str', '\n        Changes, if possible, the base types of the hold Python class. For determining if the change is possible, a\n        series of checks (defined before) are made.\n\n        For new-style classes, changing of the mro is not possible, you need to define a metaclass that does the trick\n\n        Old-style classes admits changing its __bases__ attribute (its a tuple), so we can add or substitute\n\n        :param localization: Call localization data\n        :param new_types: New base types (in the form of a tuple)\n        :return: A TypeError or None\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 1009)
        # Getting the type of 'new_types' (line 1009)
        new_types_11920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 20), 'new_types')
        # Getting the type of 'tuple' (line 1009)
        tuple_11921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 34), 'tuple')
        
        (may_be_11922, more_types_in_union_11923) = may_not_be_type(new_types_11920, tuple_11921)

        if may_be_11922:

            if more_types_in_union_11923:
                # Runtime conditional SSA (line 1009)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'new_types' (line 1009)
            new_types_11924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 8), 'new_types')
            # Assigning a type to the variable 'new_types' (line 1009)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1009, 8), 'new_types', remove_type_from_union(new_types_11924, tuple_11921))
            
            # Call to TypeError(...): (line 1010)
            # Processing the call arguments (line 1010)
            # Getting the type of 'localization' (line 1010)
            localization_11926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 29), 'localization', False)
            str_11927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1010, 43), 'str', 'New subtypes have to be specified using a tuple')
            # Processing the call keyword arguments (line 1010)
            kwargs_11928 = {}
            # Getting the type of 'TypeError' (line 1010)
            TypeError_11925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 1010)
            TypeError_call_result_11929 = invoke(stypy.reporting.localization.Localization(__file__, 1010, 19), TypeError_11925, *[localization_11926, str_11927], **kwargs_11928)
            
            # Assigning a type to the variable 'stypy_return_type' (line 1010)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1010, 12), 'stypy_return_type', TypeError_call_result_11929)

            if more_types_in_union_11923:
                # SSA join for if statement (line 1009)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'new_types' (line 1012)
        new_types_11930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1012, 25), 'new_types')
        # Assigning a type to the variable 'new_types_11930' (line 1012)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1012, 8), 'new_types_11930', new_types_11930)
        # Testing if the for loop is going to be iterated (line 1012)
        # Testing the type of a for loop iterable (line 1012)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1012, 8), new_types_11930)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 1012, 8), new_types_11930):
            # Getting the type of the for loop variable (line 1012)
            for_loop_var_11931 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1012, 8), new_types_11930)
            # Assigning a type to the variable 'base_type' (line 1012)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1012, 8), 'base_type', for_loop_var_11931)
            # SSA begins for a for statement (line 1012)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 1013):
            
            # Assigning a Call to a Name (line 1013):
            
            # Call to __change_class_base_types_checks(...): (line 1013)
            # Processing the call arguments (line 1013)
            # Getting the type of 'localization' (line 1013)
            localization_11934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1013, 58), 'localization', False)
            # Getting the type of 'base_type' (line 1013)
            base_type_11935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1013, 72), 'base_type', False)
            # Processing the call keyword arguments (line 1013)
            kwargs_11936 = {}
            # Getting the type of 'self' (line 1013)
            self_11932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1013, 20), 'self', False)
            # Obtaining the member '__change_class_base_types_checks' of a type (line 1013)
            change_class_base_types_checks_11933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1013, 20), self_11932, '__change_class_base_types_checks')
            # Calling __change_class_base_types_checks(args, kwargs) (line 1013)
            change_class_base_types_checks_call_result_11937 = invoke(stypy.reporting.localization.Localization(__file__, 1013, 20), change_class_base_types_checks_11933, *[localization_11934, base_type_11935], **kwargs_11936)
            
            # Assigning a type to the variable 'check' (line 1013)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1013, 12), 'check', change_class_base_types_checks_call_result_11937)
            
            # Type idiom detected: calculating its left and rigth part (line 1014)
            # Getting the type of 'TypeError' (line 1014)
            TypeError_11938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1014, 33), 'TypeError')
            # Getting the type of 'check' (line 1014)
            check_11939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1014, 26), 'check')
            
            (may_be_11940, more_types_in_union_11941) = may_be_subtype(TypeError_11938, check_11939)

            if may_be_11940:

                if more_types_in_union_11941:
                    # Runtime conditional SSA (line 1014)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'check' (line 1014)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1014, 12), 'check', remove_not_subtype_from_union(check_11939, TypeError))
                # Getting the type of 'check' (line 1015)
                check_11942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1015, 23), 'check')
                # Assigning a type to the variable 'stypy_return_type' (line 1015)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1015, 16), 'stypy_return_type', check_11942)

                if more_types_in_union_11941:
                    # SSA join for if statement (line 1014)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 1017):
        
        # Assigning a Call to a Name (line 1017):
        
        # Call to map(...): (line 1017)
        # Processing the call arguments (line 1017)

        @norecursion
        def _stypy_temp_lambda_19(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_19'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_19', 1017, 27, True)
            # Passed parameters checking function
            _stypy_temp_lambda_19.stypy_localization = localization
            _stypy_temp_lambda_19.stypy_type_of_self = None
            _stypy_temp_lambda_19.stypy_type_store = module_type_store
            _stypy_temp_lambda_19.stypy_function_name = '_stypy_temp_lambda_19'
            _stypy_temp_lambda_19.stypy_param_names_list = ['tproxy']
            _stypy_temp_lambda_19.stypy_varargs_param_name = None
            _stypy_temp_lambda_19.stypy_kwargs_param_name = None
            _stypy_temp_lambda_19.stypy_call_defaults = defaults
            _stypy_temp_lambda_19.stypy_call_varargs = varargs
            _stypy_temp_lambda_19.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_19', ['tproxy'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_19', ['tproxy'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'tproxy' (line 1017)
            tproxy_11944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 42), 'tproxy', False)
            # Obtaining the member 'python_entity' of a type (line 1017)
            python_entity_11945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1017, 42), tproxy_11944, 'python_entity')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 1017)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1017, 27), 'stypy_return_type', python_entity_11945)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_19' in the type store
            # Getting the type of 'stypy_return_type' (line 1017)
            stypy_return_type_11946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 27), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_11946)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_19'
            return stypy_return_type_11946

        # Assigning a type to the variable '_stypy_temp_lambda_19' (line 1017)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1017, 27), '_stypy_temp_lambda_19', _stypy_temp_lambda_19)
        # Getting the type of '_stypy_temp_lambda_19' (line 1017)
        _stypy_temp_lambda_19_11947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 27), '_stypy_temp_lambda_19')
        # Getting the type of 'new_types' (line 1017)
        new_types_11948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 64), 'new_types', False)
        # Processing the call keyword arguments (line 1017)
        kwargs_11949 = {}
        # Getting the type of 'map' (line 1017)
        map_11943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 23), 'map', False)
        # Calling map(args, kwargs) (line 1017)
        map_call_result_11950 = invoke(stypy.reporting.localization.Localization(__file__, 1017, 23), map_11943, *[_stypy_temp_lambda_19_11947, new_types_11948], **kwargs_11949)
        
        # Assigning a type to the variable 'base_classes' (line 1017)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1017, 8), 'base_classes', map_call_result_11950)
        
        # Assigning a Call to a Attribute (line 1019):
        
        # Assigning a Call to a Attribute (line 1019):
        
        # Call to tuple(...): (line 1019)
        # Processing the call arguments (line 1019)
        # Getting the type of 'base_classes' (line 1019)
        base_classes_11952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1019, 45), 'base_classes', False)
        # Processing the call keyword arguments (line 1019)
        kwargs_11953 = {}
        # Getting the type of 'tuple' (line 1019)
        tuple_11951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1019, 39), 'tuple', False)
        # Calling tuple(args, kwargs) (line 1019)
        tuple_call_result_11954 = invoke(stypy.reporting.localization.Localization(__file__, 1019, 39), tuple_11951, *[base_classes_11952], **kwargs_11953)
        
        # Getting the type of 'self' (line 1019)
        self_11955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1019, 8), 'self')
        # Obtaining the member 'python_entity' of a type (line 1019)
        python_entity_11956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1019, 8), self_11955, 'python_entity')
        # Setting the type of the member '__bases__' of a type (line 1019)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1019, 8), python_entity_11956, '__bases__', tuple_call_result_11954)
        # Getting the type of 'None' (line 1020)
        None_11957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1020, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 1020)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1020, 8), 'stypy_return_type', None_11957)
        
        # ################# End of 'change_base_types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'change_base_types' in the type store
        # Getting the type of 'stypy_return_type' (line 996)
        stypy_return_type_11958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 996, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11958)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'change_base_types'
        return stypy_return_type_11958


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

        str_11959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, (-1)), 'str', '\n        Adds, if possible, the base types of the hold Python class existing base types.\n        For determining if the change is possible, a series of checks (defined before) are made.\n\n        :param localization: Call localization data\n        :param new_types: New base types (in the form of a tuple)\n        :return: A TypeError or None\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 1031)
        # Getting the type of 'new_types' (line 1031)
        new_types_11960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 20), 'new_types')
        # Getting the type of 'tuple' (line 1031)
        tuple_11961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 34), 'tuple')
        
        (may_be_11962, more_types_in_union_11963) = may_not_be_type(new_types_11960, tuple_11961)

        if may_be_11962:

            if more_types_in_union_11963:
                # Runtime conditional SSA (line 1031)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'new_types' (line 1031)
            new_types_11964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 8), 'new_types')
            # Assigning a type to the variable 'new_types' (line 1031)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1031, 8), 'new_types', remove_type_from_union(new_types_11964, tuple_11961))
            
            # Call to TypeError(...): (line 1032)
            # Processing the call arguments (line 1032)
            # Getting the type of 'localization' (line 1032)
            localization_11966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 29), 'localization', False)
            str_11967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1032, 43), 'str', 'New subtypes have to be specified using a tuple')
            # Processing the call keyword arguments (line 1032)
            kwargs_11968 = {}
            # Getting the type of 'TypeError' (line 1032)
            TypeError_11965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 19), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 1032)
            TypeError_call_result_11969 = invoke(stypy.reporting.localization.Localization(__file__, 1032, 19), TypeError_11965, *[localization_11966, str_11967], **kwargs_11968)
            
            # Assigning a type to the variable 'stypy_return_type' (line 1032)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1032, 12), 'stypy_return_type', TypeError_call_result_11969)

            if more_types_in_union_11963:
                # SSA join for if statement (line 1031)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'new_types' (line 1034)
        new_types_11970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 25), 'new_types')
        # Assigning a type to the variable 'new_types_11970' (line 1034)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1034, 8), 'new_types_11970', new_types_11970)
        # Testing if the for loop is going to be iterated (line 1034)
        # Testing the type of a for loop iterable (line 1034)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1034, 8), new_types_11970)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 1034, 8), new_types_11970):
            # Getting the type of the for loop variable (line 1034)
            for_loop_var_11971 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1034, 8), new_types_11970)
            # Assigning a type to the variable 'base_type' (line 1034)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1034, 8), 'base_type', for_loop_var_11971)
            # SSA begins for a for statement (line 1034)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 1035):
            
            # Assigning a Call to a Name (line 1035):
            
            # Call to __change_class_base_types_checks(...): (line 1035)
            # Processing the call arguments (line 1035)
            # Getting the type of 'localization' (line 1035)
            localization_11974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 58), 'localization', False)
            # Getting the type of 'base_type' (line 1035)
            base_type_11975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 72), 'base_type', False)
            # Processing the call keyword arguments (line 1035)
            kwargs_11976 = {}
            # Getting the type of 'self' (line 1035)
            self_11972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 20), 'self', False)
            # Obtaining the member '__change_class_base_types_checks' of a type (line 1035)
            change_class_base_types_checks_11973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1035, 20), self_11972, '__change_class_base_types_checks')
            # Calling __change_class_base_types_checks(args, kwargs) (line 1035)
            change_class_base_types_checks_call_result_11977 = invoke(stypy.reporting.localization.Localization(__file__, 1035, 20), change_class_base_types_checks_11973, *[localization_11974, base_type_11975], **kwargs_11976)
            
            # Assigning a type to the variable 'check' (line 1035)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1035, 12), 'check', change_class_base_types_checks_call_result_11977)
            
            # Type idiom detected: calculating its left and rigth part (line 1036)
            # Getting the type of 'TypeError' (line 1036)
            TypeError_11978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 33), 'TypeError')
            # Getting the type of 'check' (line 1036)
            check_11979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 26), 'check')
            
            (may_be_11980, more_types_in_union_11981) = may_be_subtype(TypeError_11978, check_11979)

            if may_be_11980:

                if more_types_in_union_11981:
                    # Runtime conditional SSA (line 1036)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'check' (line 1036)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1036, 12), 'check', remove_not_subtype_from_union(check_11979, TypeError))
                # Getting the type of 'check' (line 1037)
                check_11982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 23), 'check')
                # Assigning a type to the variable 'stypy_return_type' (line 1037)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 16), 'stypy_return_type', check_11982)

                if more_types_in_union_11981:
                    # SSA join for if statement (line 1036)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 1039):
        
        # Assigning a Call to a Name (line 1039):
        
        # Call to map(...): (line 1039)
        # Processing the call arguments (line 1039)

        @norecursion
        def _stypy_temp_lambda_20(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_20'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_20', 1039, 27, True)
            # Passed parameters checking function
            _stypy_temp_lambda_20.stypy_localization = localization
            _stypy_temp_lambda_20.stypy_type_of_self = None
            _stypy_temp_lambda_20.stypy_type_store = module_type_store
            _stypy_temp_lambda_20.stypy_function_name = '_stypy_temp_lambda_20'
            _stypy_temp_lambda_20.stypy_param_names_list = ['tproxy']
            _stypy_temp_lambda_20.stypy_varargs_param_name = None
            _stypy_temp_lambda_20.stypy_kwargs_param_name = None
            _stypy_temp_lambda_20.stypy_call_defaults = defaults
            _stypy_temp_lambda_20.stypy_call_varargs = varargs
            _stypy_temp_lambda_20.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_20', ['tproxy'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_20', ['tproxy'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'tproxy' (line 1039)
            tproxy_11984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 42), 'tproxy', False)
            # Obtaining the member 'python_entity' of a type (line 1039)
            python_entity_11985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1039, 42), tproxy_11984, 'python_entity')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 1039)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 27), 'stypy_return_type', python_entity_11985)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_20' in the type store
            # Getting the type of 'stypy_return_type' (line 1039)
            stypy_return_type_11986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 27), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_11986)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_20'
            return stypy_return_type_11986

        # Assigning a type to the variable '_stypy_temp_lambda_20' (line 1039)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 27), '_stypy_temp_lambda_20', _stypy_temp_lambda_20)
        # Getting the type of '_stypy_temp_lambda_20' (line 1039)
        _stypy_temp_lambda_20_11987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 27), '_stypy_temp_lambda_20')
        # Getting the type of 'new_types' (line 1039)
        new_types_11988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 64), 'new_types', False)
        # Processing the call keyword arguments (line 1039)
        kwargs_11989 = {}
        # Getting the type of 'map' (line 1039)
        map_11983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 23), 'map', False)
        # Calling map(args, kwargs) (line 1039)
        map_call_result_11990 = invoke(stypy.reporting.localization.Localization(__file__, 1039, 23), map_11983, *[_stypy_temp_lambda_20_11987, new_types_11988], **kwargs_11989)
        
        # Assigning a type to the variable 'base_classes' (line 1039)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 8), 'base_classes', map_call_result_11990)
        
        # Getting the type of 'self' (line 1040)
        self_11991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 8), 'self')
        # Obtaining the member 'python_entity' of a type (line 1040)
        python_entity_11992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1040, 8), self_11991, 'python_entity')
        # Obtaining the member '__bases__' of a type (line 1040)
        bases___11993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1040, 8), python_entity_11992, '__bases__')
        
        # Call to tuple(...): (line 1040)
        # Processing the call arguments (line 1040)
        # Getting the type of 'base_classes' (line 1040)
        base_classes_11995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 46), 'base_classes', False)
        # Processing the call keyword arguments (line 1040)
        kwargs_11996 = {}
        # Getting the type of 'tuple' (line 1040)
        tuple_11994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 40), 'tuple', False)
        # Calling tuple(args, kwargs) (line 1040)
        tuple_call_result_11997 = invoke(stypy.reporting.localization.Localization(__file__, 1040, 40), tuple_11994, *[base_classes_11995], **kwargs_11996)
        
        # Applying the binary operator '+=' (line 1040)
        result_iadd_11998 = python_operator(stypy.reporting.localization.Localization(__file__, 1040, 8), '+=', bases___11993, tuple_call_result_11997)
        # Getting the type of 'self' (line 1040)
        self_11999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 8), 'self')
        # Obtaining the member 'python_entity' of a type (line 1040)
        python_entity_12000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1040, 8), self_11999, 'python_entity')
        # Setting the type of the member '__bases__' of a type (line 1040)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1040, 8), python_entity_12000, '__bases__', result_iadd_11998)
        
        # Getting the type of 'None' (line 1041)
        None_12001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1041, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 1041)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1041, 8), 'stypy_return_type', None_12001)
        
        # ################# End of 'add_base_types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_base_types' in the type store
        # Getting the type of 'stypy_return_type' (line 1022)
        stypy_return_type_12002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1022, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12002)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_base_types'
        return stypy_return_type_12002


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

        str_12003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1051, (-1)), 'str', '\n        Clones the type proxy, making an independent copy of the stored python entity. Physical cloning is not\n        performed if the hold python entity do not support intercession, as its structure is immutable.\n\n        :return: A clone of this proxy\n        ')
        
        # Evaluating a boolean operation
        
        
        # Call to supports_structural_reflection(...): (line 1052)
        # Processing the call keyword arguments (line 1052)
        kwargs_12006 = {}
        # Getting the type of 'self' (line 1052)
        self_12004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 15), 'self', False)
        # Obtaining the member 'supports_structural_reflection' of a type (line 1052)
        supports_structural_reflection_12005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1052, 15), self_12004, 'supports_structural_reflection')
        # Calling supports_structural_reflection(args, kwargs) (line 1052)
        supports_structural_reflection_call_result_12007 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 15), supports_structural_reflection_12005, *[], **kwargs_12006)
        
        # Applying the 'not' unary operator (line 1052)
        result_not__12008 = python_operator(stypy.reporting.localization.Localization(__file__, 1052, 11), 'not', supports_structural_reflection_call_result_12007)
        
        
        
        # Call to can_store_elements(...): (line 1052)
        # Processing the call keyword arguments (line 1052)
        kwargs_12011 = {}
        # Getting the type of 'self' (line 1052)
        self_12009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 61), 'self', False)
        # Obtaining the member 'can_store_elements' of a type (line 1052)
        can_store_elements_12010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1052, 61), self_12009, 'can_store_elements')
        # Calling can_store_elements(args, kwargs) (line 1052)
        can_store_elements_call_result_12012 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 61), can_store_elements_12010, *[], **kwargs_12011)
        
        # Applying the 'not' unary operator (line 1052)
        result_not__12013 = python_operator(stypy.reporting.localization.Localization(__file__, 1052, 57), 'not', can_store_elements_call_result_12012)
        
        # Applying the binary operator 'and' (line 1052)
        result_and_keyword_12014 = python_operator(stypy.reporting.localization.Localization(__file__, 1052, 11), 'and', result_not__12008, result_not__12013)
        
        
        # Call to can_store_keypairs(...): (line 1053)
        # Processing the call keyword arguments (line 1053)
        kwargs_12017 = {}
        # Getting the type of 'self' (line 1053)
        self_12015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 20), 'self', False)
        # Obtaining the member 'can_store_keypairs' of a type (line 1053)
        can_store_keypairs_12016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1053, 20), self_12015, 'can_store_keypairs')
        # Calling can_store_keypairs(args, kwargs) (line 1053)
        can_store_keypairs_call_result_12018 = invoke(stypy.reporting.localization.Localization(__file__, 1053, 20), can_store_keypairs_12016, *[], **kwargs_12017)
        
        # Applying the 'not' unary operator (line 1053)
        result_not__12019 = python_operator(stypy.reporting.localization.Localization(__file__, 1053, 16), 'not', can_store_keypairs_call_result_12018)
        
        # Applying the binary operator 'and' (line 1052)
        result_and_keyword_12020 = python_operator(stypy.reporting.localization.Localization(__file__, 1052, 11), 'and', result_and_keyword_12014, result_not__12019)
        
        # Testing if the type of an if condition is none (line 1052)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 1052, 8), result_and_keyword_12020):
            
            # Call to create_duplicate(...): (line 1056)
            # Processing the call arguments (line 1056)
            # Getting the type of 'self' (line 1056)
            self_12025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 73), 'self', False)
            # Processing the call keyword arguments (line 1056)
            kwargs_12026 = {}
            # Getting the type of 'type_inference_proxy_management_copy' (line 1056)
            type_inference_proxy_management_copy_12023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 19), 'type_inference_proxy_management_copy', False)
            # Obtaining the member 'create_duplicate' of a type (line 1056)
            create_duplicate_12024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1056, 19), type_inference_proxy_management_copy_12023, 'create_duplicate')
            # Calling create_duplicate(args, kwargs) (line 1056)
            create_duplicate_call_result_12027 = invoke(stypy.reporting.localization.Localization(__file__, 1056, 19), create_duplicate_12024, *[self_12025], **kwargs_12026)
            
            # Assigning a type to the variable 'stypy_return_type' (line 1056)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1056, 12), 'stypy_return_type', create_duplicate_call_result_12027)
        else:
            
            # Testing the type of an if condition (line 1052)
            if_condition_12021 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1052, 8), result_and_keyword_12020)
            # Assigning a type to the variable 'if_condition_12021' (line 1052)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1052, 8), 'if_condition_12021', if_condition_12021)
            # SSA begins for if statement (line 1052)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'self' (line 1054)
            self_12022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 19), 'self')
            # Assigning a type to the variable 'stypy_return_type' (line 1054)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1054, 12), 'stypy_return_type', self_12022)
            # SSA branch for the else part of an if statement (line 1052)
            module_type_store.open_ssa_branch('else')
            
            # Call to create_duplicate(...): (line 1056)
            # Processing the call arguments (line 1056)
            # Getting the type of 'self' (line 1056)
            self_12025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 73), 'self', False)
            # Processing the call keyword arguments (line 1056)
            kwargs_12026 = {}
            # Getting the type of 'type_inference_proxy_management_copy' (line 1056)
            type_inference_proxy_management_copy_12023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 19), 'type_inference_proxy_management_copy', False)
            # Obtaining the member 'create_duplicate' of a type (line 1056)
            create_duplicate_12024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1056, 19), type_inference_proxy_management_copy_12023, 'create_duplicate')
            # Calling create_duplicate(args, kwargs) (line 1056)
            create_duplicate_call_result_12027 = invoke(stypy.reporting.localization.Localization(__file__, 1056, 19), create_duplicate_12024, *[self_12025], **kwargs_12026)
            
            # Assigning a type to the variable 'stypy_return_type' (line 1056)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1056, 12), 'stypy_return_type', create_duplicate_call_result_12027)
            # SSA join for if statement (line 1052)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'clone(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clone' in the type store
        # Getting the type of 'stypy_return_type' (line 1045)
        stypy_return_type_12028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12028)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clone'
        return stypy_return_type_12028


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

        str_12029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1064, (-1)), 'str', '\n        Calls the dir Python builtin over the stored Python object and returns the result\n        :return: list of strings\n        ')
        
        # Call to dir(...): (line 1065)
        # Processing the call arguments (line 1065)
        # Getting the type of 'self' (line 1065)
        self_12031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 19), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 1065)
        python_entity_12032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1065, 19), self_12031, 'python_entity')
        # Processing the call keyword arguments (line 1065)
        kwargs_12033 = {}
        # Getting the type of 'dir' (line 1065)
        dir_12030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 15), 'dir', False)
        # Calling dir(args, kwargs) (line 1065)
        dir_call_result_12034 = invoke(stypy.reporting.localization.Localization(__file__, 1065, 15), dir_12030, *[python_entity_12032], **kwargs_12033)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1065)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1065, 8), 'stypy_return_type', dir_call_result_12034)
        
        # ################# End of 'dir(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dir' in the type store
        # Getting the type of 'stypy_return_type' (line 1060)
        stypy_return_type_12035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12035)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dir'
        return stypy_return_type_12035


    @staticmethod
    @norecursion
    def dict(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dict'
        module_type_store = module_type_store.open_function_context('dict', 1067, 4, False)
        
        # Passed parameters checking function
        TypeInferenceProxy.dict.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceProxy.dict.__dict__.__setitem__('stypy_type_of_self', None)
        TypeInferenceProxy.dict.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceProxy.dict.__dict__.__setitem__('stypy_function_name', 'dict')
        TypeInferenceProxy.dict.__dict__.__setitem__('stypy_param_names_list', ['self', 'localization'])
        TypeInferenceProxy.dict.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceProxy.dict.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceProxy.dict.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceProxy.dict.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceProxy.dict.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceProxy.dict.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, 'dict', ['self', 'localization'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dict', localization, ['localization'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dict(...)' code ##################

        str_12036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1072, (-1)), 'str', '\n        Equivalent to call __dict__ over the stored Python instance\n        :param localization:\n        :return:\n        ')
        
        # Assigning a Call to a Name (line 1073):
        
        # Assigning a Call to a Name (line 1073):
        
        # Call to dir(...): (line 1073)
        # Processing the call keyword arguments (line 1073)
        kwargs_12039 = {}
        # Getting the type of 'self' (line 1073)
        self_12037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 18), 'self', False)
        # Obtaining the member 'dir' of a type (line 1073)
        dir_12038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1073, 18), self_12037, 'dir')
        # Calling dir(args, kwargs) (line 1073)
        dir_call_result_12040 = invoke(stypy.reporting.localization.Localization(__file__, 1073, 18), dir_12038, *[], **kwargs_12039)
        
        # Assigning a type to the variable 'members' (line 1073)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1073, 8), 'members', dir_call_result_12040)
        
        # Assigning a Call to a Name (line 1074):
        
        # Assigning a Call to a Name (line 1074):
        
        # Call to instance(...): (line 1074)
        # Processing the call arguments (line 1074)
        # Getting the type of 'dict' (line 1074)
        dict_12043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 47), 'dict', False)
        # Processing the call keyword arguments (line 1074)
        kwargs_12044 = {}
        # Getting the type of 'TypeInferenceProxy' (line 1074)
        TypeInferenceProxy_12041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 19), 'TypeInferenceProxy', False)
        # Obtaining the member 'instance' of a type (line 1074)
        instance_12042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1074, 19), TypeInferenceProxy_12041, 'instance')
        # Calling instance(args, kwargs) (line 1074)
        instance_call_result_12045 = invoke(stypy.reporting.localization.Localization(__file__, 1074, 19), instance_12042, *[dict_12043], **kwargs_12044)
        
        # Assigning a type to the variable 'ret_dict' (line 1074)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1074, 8), 'ret_dict', instance_call_result_12045)
        
        # Call to set_type_instance(...): (line 1075)
        # Processing the call arguments (line 1075)
        # Getting the type of 'True' (line 1075)
        True_12048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 35), 'True', False)
        # Processing the call keyword arguments (line 1075)
        kwargs_12049 = {}
        # Getting the type of 'ret_dict' (line 1075)
        ret_dict_12046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 8), 'ret_dict', False)
        # Obtaining the member 'set_type_instance' of a type (line 1075)
        set_type_instance_12047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1075, 8), ret_dict_12046, 'set_type_instance')
        # Calling set_type_instance(args, kwargs) (line 1075)
        set_type_instance_call_result_12050 = invoke(stypy.reporting.localization.Localization(__file__, 1075, 8), set_type_instance_12047, *[True_12048], **kwargs_12049)
        
        
        # Getting the type of 'members' (line 1076)
        members_12051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 22), 'members')
        # Assigning a type to the variable 'members_12051' (line 1076)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1076, 8), 'members_12051', members_12051)
        # Testing if the for loop is going to be iterated (line 1076)
        # Testing the type of a for loop iterable (line 1076)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1076, 8), members_12051)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 1076, 8), members_12051):
            # Getting the type of the for loop variable (line 1076)
            for_loop_var_12052 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1076, 8), members_12051)
            # Assigning a type to the variable 'member' (line 1076)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1076, 8), 'member', for_loop_var_12052)
            # SSA begins for a for statement (line 1076)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 1077):
            
            # Assigning a Call to a Name (line 1077):
            
            # Call to instance(...): (line 1077)
            # Processing the call arguments (line 1077)
            # Getting the type of 'str' (line 1077)
            str_12055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 55), 'str', False)
            # Processing the call keyword arguments (line 1077)
            # Getting the type of 'member' (line 1077)
            member_12056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 66), 'member', False)
            keyword_12057 = member_12056
            kwargs_12058 = {'value': keyword_12057}
            # Getting the type of 'TypeInferenceProxy' (line 1077)
            TypeInferenceProxy_12053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 27), 'TypeInferenceProxy', False)
            # Obtaining the member 'instance' of a type (line 1077)
            instance_12054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1077, 27), TypeInferenceProxy_12053, 'instance')
            # Calling instance(args, kwargs) (line 1077)
            instance_call_result_12059 = invoke(stypy.reporting.localization.Localization(__file__, 1077, 27), instance_12054, *[str_12055], **kwargs_12058)
            
            # Assigning a type to the variable 'str_instance' (line 1077)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1077, 12), 'str_instance', instance_call_result_12059)
            
            # Assigning a Call to a Name (line 1079):
            
            # Assigning a Call to a Name (line 1079):
            
            # Call to get_type_of_member(...): (line 1079)
            # Processing the call arguments (line 1079)
            # Getting the type of 'localization' (line 1079)
            localization_12062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 44), 'localization', False)
            # Getting the type of 'member' (line 1079)
            member_12063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 58), 'member', False)
            # Processing the call keyword arguments (line 1079)
            kwargs_12064 = {}
            # Getting the type of 'self' (line 1079)
            self_12060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 20), 'self', False)
            # Obtaining the member 'get_type_of_member' of a type (line 1079)
            get_type_of_member_12061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1079, 20), self_12060, 'get_type_of_member')
            # Calling get_type_of_member(args, kwargs) (line 1079)
            get_type_of_member_call_result_12065 = invoke(stypy.reporting.localization.Localization(__file__, 1079, 20), get_type_of_member_12061, *[localization_12062, member_12063], **kwargs_12064)
            
            # Assigning a type to the variable 'value' (line 1079)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1079, 12), 'value', get_type_of_member_call_result_12065)
            
            # Call to add_key_and_value_type(...): (line 1080)
            # Processing the call arguments (line 1080)
            # Getting the type of 'localization' (line 1080)
            localization_12068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 44), 'localization', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 1080)
            tuple_12069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1080, 59), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 1080)
            # Adding element type (line 1080)
            # Getting the type of 'str_instance' (line 1080)
            str_instance_12070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 59), 'str_instance', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1080, 59), tuple_12069, str_instance_12070)
            # Adding element type (line 1080)
            # Getting the type of 'value' (line 1080)
            value_12071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 73), 'value', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1080, 59), tuple_12069, value_12071)
            
            # Getting the type of 'False' (line 1080)
            False_12072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 81), 'False', False)
            # Processing the call keyword arguments (line 1080)
            kwargs_12073 = {}
            # Getting the type of 'ret_dict' (line 1080)
            ret_dict_12066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 12), 'ret_dict', False)
            # Obtaining the member 'add_key_and_value_type' of a type (line 1080)
            add_key_and_value_type_12067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1080, 12), ret_dict_12066, 'add_key_and_value_type')
            # Calling add_key_and_value_type(args, kwargs) (line 1080)
            add_key_and_value_type_call_result_12074 = invoke(stypy.reporting.localization.Localization(__file__, 1080, 12), add_key_and_value_type_12067, *[localization_12068, tuple_12069, False_12072], **kwargs_12073)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'ret_dict' (line 1082)
        ret_dict_12075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1082, 15), 'ret_dict')
        # Assigning a type to the variable 'stypy_return_type' (line 1082)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1082, 8), 'stypy_return_type', ret_dict_12075)
        
        # ################# End of 'dict(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dict' in the type store
        # Getting the type of 'stypy_return_type' (line 1067)
        stypy_return_type_12076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12076)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dict'
        return stypy_return_type_12076


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

        str_12077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1088, (-1)), 'str', '\n        Determines whether this proxy holds an user-defined class or not\n        :return:\n        ')
        
        # Call to is_user_defined_class(...): (line 1089)
        # Processing the call arguments (line 1089)
        # Getting the type of 'self' (line 1089)
        self_12080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 74), 'self', False)
        # Obtaining the member 'python_entity' of a type (line 1089)
        python_entity_12081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1089, 74), self_12080, 'python_entity')
        # Processing the call keyword arguments (line 1089)
        kwargs_12082 = {}
        # Getting the type of 'type_inference_proxy_management_copy' (line 1089)
        type_inference_proxy_management_copy_12078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 15), 'type_inference_proxy_management_copy', False)
        # Obtaining the member 'is_user_defined_class' of a type (line 1089)
        is_user_defined_class_12079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1089, 15), type_inference_proxy_management_copy_12078, 'is_user_defined_class')
        # Calling is_user_defined_class(args, kwargs) (line 1089)
        is_user_defined_class_call_result_12083 = invoke(stypy.reporting.localization.Localization(__file__, 1089, 15), is_user_defined_class_12079, *[python_entity_12081], **kwargs_12082)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1089)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1089, 8), 'stypy_return_type', is_user_defined_class_call_result_12083)
        
        # ################# End of 'is_user_defined_class(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_user_defined_class' in the type store
        # Getting the type of 'stypy_return_type' (line 1084)
        stypy_return_type_12084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1084, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12084)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_user_defined_class'
        return stypy_return_type_12084


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

        str_12085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1101, (-1)), 'str', '\n        Annotate a type into the proxy type annotation record\n        :param line: Source code line when the type change is performed\n        :param column: Source code column when the type change is performed\n        :param name: Name of the variable whose type is changed\n        :param type_: New type\n        :return: None\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 1102)
        str_12086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1102, 25), 'str', 'annotation_record')
        # Getting the type of 'self' (line 1102)
        self_12087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 19), 'self')
        
        (may_be_12088, more_types_in_union_12089) = may_provide_member(str_12086, self_12087)

        if may_be_12088:

            if more_types_in_union_12089:
                # Runtime conditional SSA (line 1102)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'self' (line 1102)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1102, 8), 'self', remove_not_member_provider_from_union(self_12087, 'annotation_record'))
            
            # Call to annotate_type(...): (line 1103)
            # Processing the call arguments (line 1103)
            # Getting the type of 'line' (line 1103)
            line_12093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 49), 'line', False)
            # Getting the type of 'column' (line 1103)
            column_12094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 55), 'column', False)
            # Getting the type of 'name' (line 1103)
            name_12095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 63), 'name', False)
            # Getting the type of 'type_' (line 1103)
            type__12096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 69), 'type_', False)
            # Processing the call keyword arguments (line 1103)
            kwargs_12097 = {}
            # Getting the type of 'self' (line 1103)
            self_12090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 12), 'self', False)
            # Obtaining the member 'annotation_record' of a type (line 1103)
            annotation_record_12091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1103, 12), self_12090, 'annotation_record')
            # Obtaining the member 'annotate_type' of a type (line 1103)
            annotate_type_12092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1103, 12), annotation_record_12091, 'annotate_type')
            # Calling annotate_type(args, kwargs) (line 1103)
            annotate_type_call_result_12098 = invoke(stypy.reporting.localization.Localization(__file__, 1103, 12), annotate_type_12092, *[line_12093, column_12094, name_12095, type__12096], **kwargs_12097)
            

            if more_types_in_union_12089:
                # SSA join for if statement (line 1102)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__annotate_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__annotate_type' in the type store
        # Getting the type of 'stypy_return_type' (line 1093)
        stypy_return_type_12099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12099)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__annotate_type'
        return stypy_return_type_12099


# Assigning a type to the variable 'TypeInferenceProxy' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'TypeInferenceProxy', TypeInferenceProxy)

# Assigning a Call to a Name (line 40):

# Call to dict(...): (line 40)
# Processing the call keyword arguments (line 40)
kwargs_12102 = {}
# Getting the type of 'TypeInferenceProxy'
TypeInferenceProxy_12100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeInferenceProxy', False)
# Obtaining the member 'dict' of a type
dict_12101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeInferenceProxy_12100, 'dict')
# Calling dict(args, kwargs) (line 40)
dict_call_result_12103 = invoke(stypy.reporting.localization.Localization(__file__, 40, 23), dict_12101, *[], **kwargs_12102)

# Getting the type of 'TypeInferenceProxy'
TypeInferenceProxy_12104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeInferenceProxy')
# Setting the type of the member 'type_proxy_cache' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeInferenceProxy_12104, 'type_proxy_cache', dict_call_result_12103)

# Assigning a Name to a Name (line 48):
# Getting the type of 'True' (line 48)
True_12105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'True')
# Getting the type of 'TypeInferenceProxy'
TypeInferenceProxy_12106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeInferenceProxy')
# Setting the type of the member 'annotate_types' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeInferenceProxy_12106, 'annotate_types', True_12105)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
