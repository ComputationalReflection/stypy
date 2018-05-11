
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import os
2: 
3: from ...stypy_copy import python_interface_copy
4: from ...stypy_copy.errors_copy.type_error_copy import TypeError
5: from ...stypy_copy.errors_copy.undefined_type_error_copy import UndefinedTypeError
6: from ...stypy_copy.errors_copy.type_warning_copy import TypeWarning, UnreferencedLocalVariableTypeWarning
7: from function_context_copy import FunctionContext
8: from ...stypy_copy.python_lib_copy.python_types_copy import non_python_type_copy
9: from ...stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import type_inference_proxy_copy, localization_copy
10: from type_annotation_record_copy import TypeAnnotationRecord
11: from ...stypy_copy import stypy_parameters_copy
12: 
13: 
14: class TypeStore(non_python_type_copy.NonPythonType):
15:     '''
16:     A TypeStore contains all the registered variable, function names and types within a particular file (module).
17:     It functions like a central storage of type information for the file, and allows any program to perform type
18:     queries for any variable within the module.
19: 
20:     The TypeStore allows flow-sensitive type storage, as it allows us to create nested contexts in which
21:     [<variable_name>, <variable_type>] pairs are stored for any particular function or method. Following Python
22:     semantics a variable in a nested context shadows a same-name variable in an outer context. If a variable is not
23:     found in the topmost context, it is searched in the more global ones.
24: 
25:     Please note that the TypeStore abstracts away context search semantics, as it only allows the user to create and
26:     destroy them.
27:     '''
28: 
29:     type_stores_of_modules = dict()
30: 
31:     @staticmethod
32:     def get_type_store_of_module(module_name):
33:         '''
34:         Obtains the type store associated with a module name
35:         :param module_name: Module name
36:         :return: TypeStore object of that module
37:         '''
38:         try:
39:             return TypeStore.type_stores_of_modules[module_name]
40:         except:
41:             return None
42: 
43:     def __load_predefined_variables(self):
44:         self.set_type_of(localization_copy.Localization(self.program_name, 1, 1), '__file__', str)
45:         self.set_type_of(localization_copy.Localization(self.program_name, 1, 1), '__doc__', str)
46:         self.set_type_of(localization_copy.Localization(self.program_name, 1, 1), '__name__', str)
47:         self.set_type_of(localization_copy.Localization(self.program_name, 1, 1), '__package__', str)
48: 
49:     def __init__(self, file_name):
50:         '''
51:         Creates a type store for the passed file name (module)
52:         :param file_name: file name to create the TypeStore for
53:         :return:
54:         '''
55:         file_name = file_name.replace("\\", "/")
56:         self.program_name = file_name.replace(stypy_parameters_copy.type_inference_file_postfix, "")
57:         self.program_name = self.program_name.replace(stypy_parameters_copy.type_inference_file_directory_name + "/", "")
58: 
59:         # At least every module must have a main function context
60:         main_context = FunctionContext(file_name, True)
61: 
62:         # Create an annotation record for the program, reusing the existing one if it was previously created
63:         main_context.annotation_record = TypeAnnotationRecord.get_instance_for_file(self.program_name)
64: 
65:         # Initializes the context stack
66:         self.context_stack = [main_context]
67: 
68:         # Teared down function contexts are stored for reporting variables created during the execution for debugging
69:         # purposes
70:         self.last_function_contexts = []
71: 
72:         # Configure if some warnings are given
73:         self.test_unreferenced_var = stypy_parameters_copy.ENABLE_CODING_ADVICES
74: 
75:         # External modules used by this module have its own type store. These secondary type stores are stored here
76:         # to access them when needed.
77:         self.external_modules = []
78:         self.__load_predefined_variables()
79: 
80:         file_cache = os.path.abspath(self.program_name).replace('\\', '/')
81:         # Register ourselves in the collection of created type stores
82:         TypeStore.type_stores_of_modules[file_cache] = self
83: 
84:     def add_external_module(self, stypy_object):
85:         '''
86:         Adds a external module to the list of modules used by this one
87:         :param stypy_object:
88:         :return:
89:         '''
90:         self.external_modules.append(stypy_object)
91:         module_type_store = stypy_object.get_analyzed_program_type_store()
92:         module_type_store.last_function_contexts = self.last_function_contexts
93: 
94:     def get_all_processed_function_contexts(self):
95:         '''
96:         Obtain a list of all the function context that were ever used during the program execution (active + past ones)
97:         :return: List of function contexts
98:         '''
99:         return self.context_stack + self.last_function_contexts
100: 
101:     def set_check_unreferenced_vars(self, state):
102:         '''
103:         On some occasions, such as when invoking methods or reading default values from parameters, the unreferenced
104:         var checks must be disabled to ensure proper behavior.
105:         :param state: bool value. However, if coding advices are disabled, this method has no functionality, they are
106:         always set to False
107:         :return:
108:         '''
109:         if not stypy_parameters_copy.ENABLE_CODING_ADVICES:
110:             self.test_unreferenced_var = False
111:             return
112:         self.test_unreferenced_var = state
113: 
114:     def set_context(self, function_name="", lineno=-1, col_offset=-1):
115:         '''
116:         Creates a new function context in the top position of the context stack
117:         '''
118:         context = FunctionContext(function_name)
119:         context.declaration_line = lineno
120:         context.declaration_column = col_offset
121:         context.annotation_record = TypeAnnotationRecord.get_instance_for_file(self.program_name)
122: 
123:         self.context_stack.insert(0, context)
124: 
125:     def unset_context(self):
126:         '''
127:         Pops and returns the topmost context in the context stack
128:         :return:
129:         '''
130:         # Invariant
131:         assert len(self.context_stack) > 0
132: 
133:         context = self.context_stack.pop(0)
134:         self.last_function_contexts.append(context)
135: 
136:         return context
137: 
138:     def get_context(self):
139:         '''
140:         Gets the current (topmost) context.
141:         :return: The current context
142:         '''
143:         return self.context_stack[0]
144: 
145:     def mark_as_global(self, localization, name):
146:         '''
147:         Mark a variable as global in the current function context
148:         :param localization: Caller information
149:         :param name: variable name
150:         :return:
151:         '''
152:         ret = None
153:         self.set_check_unreferenced_vars(False)
154:         var_type = self.get_context().get_type_of(name)
155:         self.set_check_unreferenced_vars(True)
156:         if var_type is not None:
157:             ret = TypeWarning(localization,
158:                               "SyntaxWarning: name '{0}' is used before global declaration".format(name))
159:             if not self.get_context() == self.get_global_context():
160:                 # Declaring a variable as global once it has a value promotes it to global
161:                 self.get_global_context().set_type_of(name, var_type, localization)
162: 
163:         unreferenced_var_warnings = filter(lambda warn: isinstance(warn, UnreferencedLocalVariableTypeWarning) and
164:                                                         warn.name == name and warn.context == self.get_context(),
165:                                            TypeWarning.get_warning_msgs())
166: 
167:         if len(unreferenced_var_warnings) > 0:
168:             ret = TypeWarning(localization,
169:                               "SyntaxWarning: name '{0}' is used before global declaration".format(name))
170: 
171:         global_vars = self.get_context().global_vars
172: 
173:         if name not in global_vars:
174:             global_vars.append(name)
175:         return ret
176: 
177:     def get_global_context(self):
178:         '''
179:         Gets the main function context, the last element in the context stack
180:         :return:
181:         '''
182:         return self.context_stack[-1]
183: 
184:     def get_type_of(self, localization, name):
185:         '''
186:         Gets the type of the variable name, implemented the mentioned context-search semantics
187:         :param localization: Caller information
188:         :param name: Variable name
189:         :return:
190:         '''
191:         ret = self.__get_type_of_from_function_context(localization, name, self.get_context())
192: 
193:         # If it is not found, look among builtins as python does.
194:         if isinstance(ret, UndefinedTypeError):
195:             member = python_interface_copy.import_from(localization, name)
196: 
197:             if isinstance(member, TypeError):
198:                 member.msg = "Could not find a definition for the name '{0}' in the current context. Are you missing " \
199:                              "an import?".format(name)
200: 
201:             if member is not None:
202:                 # If found here, it is not an error any more
203:                 TypeError.remove_error_msg(ret)
204:                 # ret_member = type_inference_proxy.TypeInferenceProxy.instance(type(member.python_entity))
205:                 # ret_member.type_of = member
206:                 return member
207:                 # return ret_member
208: 
209:         return ret
210: 
211:     def get_context_of(self, name):
212:         '''
213:         Returns the function context in which a variable is first defined
214:         :param name: Variable name
215:         :return:
216:         '''
217:         for context in self.context_stack:
218:             if name in context:
219:                 return context
220: 
221:         return None
222: 
223:     def set_type_of(self, localization, name, type_):
224:         '''
225:         Set the type of a variable using the context semantics previously mentioned.
226: 
227:         Only simple a=b assignments are supported, as multiple assignments are solved by AST desugaring, so all of them
228:         are converted to equivalent simple ones.
229:         '''
230:         if not isinstance(type_, type_inference_proxy_copy.Type):
231:             type_ = type_inference_proxy_copy.TypeInferenceProxy.instance(type_)
232: 
233:         type_.annotation_record = TypeAnnotationRecord.get_instance_for_file(self.program_name)
234:         # Destination is a single name of a variable
235:         return self.__set_type_of(localization, name, type_)
236: 
237:     def set_type_store(self, type_store, clone=False):
238:         '''
239:         Assign to this type store the attributes of the passed type store, cloning the passed
240:         type store if indicated. This operation is needed to implement the SSA algorithm
241:         :param type_store: Type store to assign to this one
242:         :param clone: Clone the passed type store before assigning its values
243:         :return:
244:         '''
245:         if clone:
246:             type_store = TypeStore.__clone_type_store(type_store)
247: 
248:         self.program_name = type_store.program_name
249:         self.context_stack = type_store.context_stack
250:         self.last_function_contexts = type_store.last_function_contexts
251:         self.external_modules = type_store.external_modules
252:         self.test_unreferenced_var = type_store.test_unreferenced_var
253: 
254:     def clone_type_store(self):
255:         '''
256:         Clone this type store
257:         :return: A clone of this type store
258:         '''
259:         return TypeStore.__clone_type_store(self)
260: 
261:     def get_public_names_and_types(self):
262:         '''
263:         Gets all the public variables within this type store function contexts and its types
264:         in a {name: type} dictionary
265:         :return: {name: type} dictionary
266:         '''
267:         name_type_dict = {}
268:         cont = len(self.context_stack) - 1
269:         # Run through the contexts in inverse order (more global to more local) and store its name - type pairs. This
270:         # way local variables that shadows global ones take precedence.
271:         for i in range(len(self.context_stack)):
272:             ctx = self.context_stack[cont]
273: 
274:             for name in ctx.types_of:
275:                 if name.startswith("__"):
276:                     continue
277:                 name_type_dict[name] = ctx.types_of[name]
278: 
279:             cont -= 1
280: 
281:         return name_type_dict
282: 
283:     def get_last_function_context_for(self, context_name):
284:         '''
285:         Gets the last used function context whose name is the one passed to this function
286:         :param context_name: Context name to search
287:         :return: Function context
288:         '''
289:         context = None
290: 
291:         for last_context in self.last_function_contexts:
292:             if last_context.function_name == context_name:
293:                 context = last_context
294: 
295:         if context is None:
296:             for context in self.context_stack:
297:                 if context_name == context.function_name:
298:                     return context
299: 
300:         return context
301: 
302:     def add_alias(self, alias, member_name):
303:         '''
304:         Adds an alias to the current function context
305:         :param alias: Alias name
306:         :param member_name: Aliased variable name
307:         :return:
308:         '''
309:         self.get_context().add_alias(alias, member_name)
310: 
311:     def del_type_of(self, localization, name):
312:         '''
313:         Delete a variable for the first function context that defines it (using the context
314:         search semantics we mentioned)
315:         :param localization:
316:         :param name:
317:         :return:
318:         '''
319:         ret = self.__del_type_of_from_function_context(localization, name, self.get_context())
320: 
321:         return ret
322: 
323:     def store_return_type_of_current_context(self, return_type):
324:         '''
325:         Changes the return type of the current function context
326:         :param return_type: Type
327:         :return:
328:         '''
329:         self.get_context().return_type = return_type
330: 
331:     # ########################################## NON - PUBLIC INTERFACE ##########################################
332: 
333:     def __get_type_of_from_function_context(self, localization, name, f_context):
334:         '''
335:         Search the stored function contexts for the type associated to a name.
336:         As we follows the program flow, a correct program ensures that if this query is performed the name actually HAS
337:         a type (it has been assigned a value previously in the previous executed statements). If the name is not found,
338:          we have detected a programmer error within the source file (usage of a previously undeclared name). The
339:          method is orthogonal to variables and functions.
340:         :param name: Name of the element whose type we want to know
341:         :return:
342:         '''
343: 
344:         # Obtain the current context
345:         current_context = f_context
346:         # Get global context (module-level)
347:         global_context = self.get_global_context()
348: 
349:         # Is this global? (marked previously with the global keyword)
350:         if name in current_context.global_vars:
351:             # Search the name within the global context
352:             type_ = global_context.get_type_of(name)
353:             # If it does not exist, we cannot read it (no value was provided for it)
354:             if type_ is None:
355:                 return TypeError(localization, "Attempted to read the uninitialized global '%s'" % name)
356:             else:
357:                 # If it exist, return its type
358:                 return type_
359: 
360:         top_context_reached = False
361: 
362:         # If the name is not a global, we run from the more local to the more global context looking for the name
363:         for context in self.context_stack:
364:             if context == f_context:
365:                 top_context_reached = True
366: 
367:             if not top_context_reached:
368:                 continue
369: 
370:             type_ = context.get_type_of(name)
371: 
372:             if type_ is None:
373:                 continue
374: 
375:             '''
376:             The type of name is found. In this case, we test if the name is also present into the global context.
377:             If it is, and was not marked as a global till now, we generate a warning indicating that if a write access
378:             is performed to name and it is still not marked as global, then Python will throw a runtime error
379:             complaining that name has been referenced without being assigned first. global have to be used to avoid
380:             this error.
381:             '''
382:             # Not marked as global & defined in a non local context & we are not within the global context & is a var
383:             if self.test_unreferenced_var:
384:                 if name not in current_context.global_vars and \
385:                         not context == self.get_context() \
386:                         and not current_context == global_context:
387:                     UnreferencedLocalVariableTypeWarning(localization, name, current_context)
388: 
389:             return type_
390: 
391:         return UndefinedTypeError(localization, "The variable '%s' does not exist" % str(name))
392: 
393:     def __set_type_of(self, localization, name, type_):
394:         '''
395:         Cases:
396: 
397:         - Exist in the local context:
398:             Is marked as global: It means that the global keyword was used after one assignment ->
399:          assign the variable in the global context and remove from the local
400:             Is not marked as global: Update
401:         - Don't exist in the local context:
402:             Is global: Go to the global context and assign
403:             Is not global: Create (Update). Shadows more global same-name element
404:         '''
405:         global_context = self.get_global_context()
406:         is_marked_as_global = name in self.get_context().global_vars
407:         exist_in_local_context = name in self.get_context()
408: 
409:         if exist_in_local_context:
410:             if is_marked_as_global:
411:                 global_context.set_type_of(name, type_, localization)
412:                 del self.get_context().types_of[name]
413:                 return TypeWarning(localization, "You used the global keyword on '{0}' after assigning a value to it. "
414:                                                  "It is valid, but will throw a warning on execution. "
415:                                                  "Please consider moving the global statement before "
416:                                                  "any assignment is done to '{0}'".format(name))
417:             else:
418:                 self.get_context().set_type_of(name, type_, localization)
419:         else:
420:             if is_marked_as_global:
421:                 global_context.set_type_of(name, type_, localization)
422:             else:
423:                 '''Special case:
424:                     If:
425:                         - A variable do not exist in the local context
426:                         - This variable is not marked as global
427:                         - There exist unreferenced type warnings in this scope typed to this variable.
428:                     Then:
429:                         - For each unreferenced type warning found:
430:                             - Generate a unreferenced variable error with the warning data
431:                             - Delete warning
432:                             - Mark the type of the variable as ErrorType
433:                 '''
434:                 unreferenced_type_warnings = filter(lambda warning:
435:                                                     warning.__class__ == UnreferencedLocalVariableTypeWarning,
436:                                                     TypeWarning.get_warning_msgs())
437: 
438:                 if len(unreferenced_type_warnings) > 0:
439:                     our_unreferenced_type_warnings_in_this_context = filter(lambda warning:
440:                                                                             warning.context == self.get_context() and
441:                                                                             warning.name == name,
442:                                                                             unreferenced_type_warnings)
443: 
444:                     for utw in our_unreferenced_type_warnings_in_this_context:
445:                         TypeError(localization, "UnboundLocalError: local variable '{0}' "
446:                                                 "referenced before assignment".format(name))
447:                         TypeWarning.warnings.remove(utw)
448: 
449:                     # Unreferenced local errors tied to 'name'
450:                     if len(our_unreferenced_type_warnings_in_this_context) > 0:
451:                         self.get_context().set_type_of(name, TypeError(localization, "Attempted to use '{0}' previously"
452:                                                                                      " to its definition".format(name)),
453:                                                        localization)
454:                         return self.get_context().get_type_of(name)
455: 
456:                 contains_undefined, more_types_in_value = type_inference_proxy.TypeInferenceProxy. \
457:                     contains_an_undefined_type(type_)
458:                 if contains_undefined:
459:                     if more_types_in_value == 0:
460:                         TypeError(localization, "Assigning to '{0}' the value of an undefined variable".
461:                                   format(name))
462:                     else:
463:                         TypeWarning.instance(localization,
464:                                              "Potentialy assigning to '{0}' the value of an undefined variable".
465:                                              format(name))
466: 
467:                 self.get_context().set_type_of(name, type_, localization)
468:         return None
469: 
470:     @staticmethod
471:     def __clone_type_store(type_store):
472:         '''
473:         Clones the type store; eventually it must also clone the values (classes)
474:         because they can be modified with intercession
475:         '''
476: 
477:         cloned_obj = TypeStore(type_store.program_name)
478:         cloned_obj.context_stack = []
479:         for context in type_store.context_stack:
480:             cloned_obj.context_stack.append(context.clone())
481: 
482:         cloned_obj.last_function_contexts = type_store.last_function_contexts
483:         cloned_obj.external_modules = type_store.external_modules
484:         cloned_obj.test_unreferenced_var = type_store.test_unreferenced_var
485: 
486:         return cloned_obj
487: 
488:     # TODO: Remove?
489:     # def __get_last_function_context_for(self, context_name):
490:     #     context = None
491:     #     try:
492:     #         context = self.last_function_contexts[context_name]
493:     #     except KeyError:
494:     #         for context in self.context_stack:
495:     #             if context_name == context.function_name:
496:     #                 return context
497:     #
498:     #     return context
499: 
500:     def __del_type_of_from_function_context(self, localization, name, f_context):
501:         '''
502:         Search the stored function contexts for the type associated to a name.
503:         As we follows the program flow, a correct program ensures that if this query is performed the name actually HAS
504:         a type (it has been assigned a value previously in the previous executed statements). If the name is not found,
505:          we have detected a programmer error within the source file (usage of a previously undeclared name). The
506:          method is orthogonal to variables and functions.
507:         :param name: Name of the element whose type we want to know
508:         :return:
509:         '''
510: 
511:         # Obtain the current context
512:         current_context = f_context
513:         # Get global context (module-level)
514:         global_context = self.get_global_context()
515: 
516:         # Is this global? (marked previously with the global keyword)
517:         if name in current_context.global_vars:
518:             # Search the name within the global context
519:             type_ = global_context.get_type_of(name)
520:             # If it does not exist, we cannot read it (no value was provided for it)
521:             if type_ is None:
522:                 return TypeError(localization, "Attempted to delete the uninitialized global '%s'" % name)
523:             else:
524:                 # If it exist, delete it
525:                 return global_context.del_type_of(name)
526: 
527:         top_context_reached = False
528: 
529:         # If the name is not a global, we run from the more local to the more global context looking for the name
530:         for context in self.context_stack:
531:             if context == f_context:
532:                 top_context_reached = True
533: 
534:             if not top_context_reached:
535:                 continue
536: 
537:             type_ = context.get_type_of(name)
538: 
539:             if type_ is None:
540:                 continue
541: 
542:             return context.del_type_of(name)
543: 
544:         return UndefinedTypeError(localization, "The variable '%s' does not exist" % str(name))
545: 
546:     # ############################################# SPECIAL METHODS #############################################
547: 
548:     def __len__(self):
549:         '''
550:         len operator, returning the number of function context stored in this type store
551:         :return:
552:         '''
553:         return len(self.context_stack)
554: 
555:     def __iter__(self):
556:         '''
557:         Iterator interface, to traverse function contexts
558:         :return:
559:         '''
560:         for f_context in self.context_stack:
561:             yield f_context
562: 
563:     def __getitem__(self, item):
564:         '''
565:         Returns the nth function context in the context stack
566:         :param item: Index of the function context
567:         :return: A Function context or an exception if the position is not valid
568:         '''
569:         return self.context_stack[item]
570: 
571:     def __contains__(self, item):
572:         '''
573:         in operator, to see if a variable is defined in a function context of the current context stack
574:         :param item: variable
575:         :return: bool
576:         '''
577:         type_ = self.get_type_of(None, item)
578: 
579:         return not (type_.__class__ == TypeError)
580: 
581:     def __repr__(self):
582:         '''
583:         Textual representation of the type store
584:         :return: str
585:         '''
586:         txt = "Type store of file '" + str(self.program_name.split("/")[-1]) + "'\n"
587:         txt += "Active contexts:\n"
588: 
589:         for context in self.context_stack:
590:             txt += str(context)
591: 
592:         if len(self.last_function_contexts) > 0:
593:             txt += "Other contexts created during execution:\n"
594:             for context in self.last_function_contexts:
595:                 txt += str(context)
596: 
597:         return txt
598: 
599:     def __str__(self):
600:         return self.__repr__()
601: 
602:     # ############################## MEMBER TYPE GET / SET ###############################
603: 
604:     def get_type_of_member(self, localization, member_name):
605:         '''
606:         Proxy for get_type_of, to comply with NonPythonType interface
607:         :param localization: Caller information
608:         :param member_name: Member name
609:         :return:
610:         '''
611:         return self.get_type_of(localization, member_name)
612: 
613:     def set_type_of_member(self, localization, member_name, member_value):
614:         '''
615:         Proxy for set_type_of, to comply with NonPythonType interface
616:         :param localization: Caller information
617:         :param member_name: Member name
618:         :return:
619:         '''
620:         return self.set_type_of(localization, member_name, member_value)
621: 
622:     # ############################## STRUCTURAL REFLECTION ###############################
623: 
624:     def delete_member(self, localization, member):
625:         '''
626:         Proxy for del_type_of, to comply with NonPythonType interface
627:         :param localization: Caller information
628:         :param member: Member name
629:         :return:
630:         '''
631:         return self.del_type_of(localization, member)
632: 
633:     def supports_structural_reflection(self):
634:         '''
635:         TypeStores (modules) always support structural reflection
636:         :return: True
637:         '''
638:         return True
639: 
640:     # ############################## TYPE CLONING ###############################
641: 
642:     def clone(self):
643:         '''
644:         Proxy for clone_type_store, to comply with NonPythonType interface
645:         :return:
646:         '''
647:         return self.clone_type_store()
648: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import os' statement (line 1)
import os

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy import python_interface_copy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')
import_17654 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy')

if (type(import_17654) is not StypyTypeError):

    if (import_17654 != 'pyd_module'):
        __import__(import_17654)
        sys_modules_17655 = sys.modules[import_17654]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', sys_modules_17655.module_type_store, module_type_store, ['python_interface_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_17655, sys_modules_17655.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy import python_interface_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', None, module_type_store, ['python_interface_copy'], [python_interface_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', import_17654)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')
import_17656 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy')

if (type(import_17656) is not StypyTypeError):

    if (import_17656 != 'pyd_module'):
        __import__(import_17656)
        sys_modules_17657 = sys.modules[import_17656]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', sys_modules_17657.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_17657, sys_modules_17657.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', import_17656)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.undefined_type_error_copy import UndefinedTypeError' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')
import_17658 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.undefined_type_error_copy')

if (type(import_17658) is not StypyTypeError):

    if (import_17658 != 'pyd_module'):
        __import__(import_17658)
        sys_modules_17659 = sys.modules[import_17658]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.undefined_type_error_copy', sys_modules_17659.module_type_store, module_type_store, ['UndefinedTypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_17659, sys_modules_17659.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.undefined_type_error_copy import UndefinedTypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.undefined_type_error_copy', None, module_type_store, ['UndefinedTypeError'], [UndefinedTypeError])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.undefined_type_error_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.undefined_type_error_copy', import_17658)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy import TypeWarning, UnreferencedLocalVariableTypeWarning' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')
import_17660 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy')

if (type(import_17660) is not StypyTypeError):

    if (import_17660 != 'pyd_module'):
        __import__(import_17660)
        sys_modules_17661 = sys.modules[import_17660]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy', sys_modules_17661.module_type_store, module_type_store, ['TypeWarning', 'UnreferencedLocalVariableTypeWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_17661, sys_modules_17661.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy import TypeWarning, UnreferencedLocalVariableTypeWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy', None, module_type_store, ['TypeWarning', 'UnreferencedLocalVariableTypeWarning'], [TypeWarning, UnreferencedLocalVariableTypeWarning])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy', import_17660)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from function_context_copy import FunctionContext' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')
import_17662 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'function_context_copy')

if (type(import_17662) is not StypyTypeError):

    if (import_17662 != 'pyd_module'):
        __import__(import_17662)
        sys_modules_17663 = sys.modules[import_17662]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'function_context_copy', sys_modules_17663.module_type_store, module_type_store, ['FunctionContext'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_17663, sys_modules_17663.module_type_store, module_type_store)
    else:
        from function_context_copy import FunctionContext

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'function_context_copy', None, module_type_store, ['FunctionContext'], [FunctionContext])

else:
    # Assigning a type to the variable 'function_context_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'function_context_copy', import_17662)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy import non_python_type_copy' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')
import_17664 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy')

if (type(import_17664) is not StypyTypeError):

    if (import_17664 != 'pyd_module'):
        __import__(import_17664)
        sys_modules_17665 = sys.modules[import_17664]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy', sys_modules_17665.module_type_store, module_type_store, ['non_python_type_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_17665, sys_modules_17665.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy import non_python_type_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy', None, module_type_store, ['non_python_type_copy'], [non_python_type_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy', import_17664)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import type_inference_proxy_copy, localization_copy' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')
import_17666 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_17666) is not StypyTypeError):

    if (import_17666 != 'pyd_module'):
        __import__(import_17666)
        sys_modules_17667 = sys.modules[import_17666]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', sys_modules_17667.module_type_store, module_type_store, ['type_inference_proxy_copy', 'localization_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_17667, sys_modules_17667.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import type_inference_proxy_copy, localization_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['type_inference_proxy_copy', 'localization_copy'], [type_inference_proxy_copy, localization_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', import_17666)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from type_annotation_record_copy import TypeAnnotationRecord' statement (line 10)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')
import_17668 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'type_annotation_record_copy')

if (type(import_17668) is not StypyTypeError):

    if (import_17668 != 'pyd_module'):
        __import__(import_17668)
        sys_modules_17669 = sys.modules[import_17668]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'type_annotation_record_copy', sys_modules_17669.module_type_store, module_type_store, ['TypeAnnotationRecord'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_17669, sys_modules_17669.module_type_store, module_type_store)
    else:
        from type_annotation_record_copy import TypeAnnotationRecord

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'type_annotation_record_copy', None, module_type_store, ['TypeAnnotationRecord'], [TypeAnnotationRecord])

else:
    # Assigning a type to the variable 'type_annotation_record_copy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'type_annotation_record_copy', import_17668)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy import stypy_parameters_copy' statement (line 11)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')
import_17670 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'testing.test_programs.stypy_code_copy.stypy_copy')

if (type(import_17670) is not StypyTypeError):

    if (import_17670 != 'pyd_module'):
        __import__(import_17670)
        sys_modules_17671 = sys.modules[import_17670]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', sys_modules_17671.module_type_store, module_type_store, ['stypy_parameters_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_17671, sys_modules_17671.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy import stypy_parameters_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', None, module_type_store, ['stypy_parameters_copy'], [stypy_parameters_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', import_17670)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')

# Declaration of the 'TypeStore' class
# Getting the type of 'non_python_type_copy' (line 14)
non_python_type_copy_17672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 16), 'non_python_type_copy')
# Obtaining the member 'NonPythonType' of a type (line 14)
NonPythonType_17673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 16), non_python_type_copy_17672, 'NonPythonType')

class TypeStore(NonPythonType_17673, ):
    str_17674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, (-1)), 'str', '\n    A TypeStore contains all the registered variable, function names and types within a particular file (module).\n    It functions like a central storage of type information for the file, and allows any program to perform type\n    queries for any variable within the module.\n\n    The TypeStore allows flow-sensitive type storage, as it allows us to create nested contexts in which\n    [<variable_name>, <variable_type>] pairs are stored for any particular function or method. Following Python\n    semantics a variable in a nested context shadows a same-name variable in an outer context. If a variable is not\n    found in the topmost context, it is searched in the more global ones.\n\n    Please note that the TypeStore abstracts away context search semantics, as it only allows the user to create and\n    destroy them.\n    ')
    
    # Assigning a Call to a Name (line 29):

    @staticmethod
    @norecursion
    def get_type_store_of_module(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_type_store_of_module'
        module_type_store = module_type_store.open_function_context('get_type_store_of_module', 31, 4, False)
        
        # Passed parameters checking function
        TypeStore.get_type_store_of_module.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.get_type_store_of_module.__dict__.__setitem__('stypy_type_of_self', None)
        TypeStore.get_type_store_of_module.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.get_type_store_of_module.__dict__.__setitem__('stypy_function_name', 'get_type_store_of_module')
        TypeStore.get_type_store_of_module.__dict__.__setitem__('stypy_param_names_list', ['module_name'])
        TypeStore.get_type_store_of_module.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.get_type_store_of_module.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.get_type_store_of_module.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.get_type_store_of_module.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.get_type_store_of_module.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.get_type_store_of_module.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, 'get_type_store_of_module', ['module_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_type_store_of_module', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_type_store_of_module(...)' code ##################

        str_17675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, (-1)), 'str', '\n        Obtains the type store associated with a module name\n        :param module_name: Module name\n        :return: TypeStore object of that module\n        ')
        
        
        # SSA begins for try-except statement (line 38)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Obtaining the type of the subscript
        # Getting the type of 'module_name' (line 39)
        module_name_17676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 52), 'module_name')
        # Getting the type of 'TypeStore' (line 39)
        TypeStore_17677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 19), 'TypeStore')
        # Obtaining the member 'type_stores_of_modules' of a type (line 39)
        type_stores_of_modules_17678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 19), TypeStore_17677, 'type_stores_of_modules')
        # Obtaining the member '__getitem__' of a type (line 39)
        getitem___17679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 19), type_stores_of_modules_17678, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
        subscript_call_result_17680 = invoke(stypy.reporting.localization.Localization(__file__, 39, 19), getitem___17679, module_name_17676)
        
        # Assigning a type to the variable 'stypy_return_type' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'stypy_return_type', subscript_call_result_17680)
        # SSA branch for the except part of a try statement (line 38)
        # SSA branch for the except '<any exception>' branch of a try statement (line 38)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'None' (line 41)
        None_17681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'stypy_return_type', None_17681)
        # SSA join for try-except statement (line 38)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_type_store_of_module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_type_store_of_module' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_17682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17682)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_type_store_of_module'
        return stypy_return_type_17682


    @norecursion
    def __load_predefined_variables(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__load_predefined_variables'
        module_type_store = module_type_store.open_function_context('__load_predefined_variables', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.__load_predefined_variables.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.__load_predefined_variables.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.__load_predefined_variables.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.__load_predefined_variables.__dict__.__setitem__('stypy_function_name', 'TypeStore.__load_predefined_variables')
        TypeStore.__load_predefined_variables.__dict__.__setitem__('stypy_param_names_list', [])
        TypeStore.__load_predefined_variables.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.__load_predefined_variables.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.__load_predefined_variables.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.__load_predefined_variables.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.__load_predefined_variables.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.__load_predefined_variables.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.__load_predefined_variables', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__load_predefined_variables', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__load_predefined_variables(...)' code ##################

        
        # Call to set_type_of(...): (line 44)
        # Processing the call arguments (line 44)
        
        # Call to Localization(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'self' (line 44)
        self_17687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 56), 'self', False)
        # Obtaining the member 'program_name' of a type (line 44)
        program_name_17688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 56), self_17687, 'program_name')
        int_17689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 75), 'int')
        int_17690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 78), 'int')
        # Processing the call keyword arguments (line 44)
        kwargs_17691 = {}
        # Getting the type of 'localization_copy' (line 44)
        localization_copy_17685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 25), 'localization_copy', False)
        # Obtaining the member 'Localization' of a type (line 44)
        Localization_17686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 25), localization_copy_17685, 'Localization')
        # Calling Localization(args, kwargs) (line 44)
        Localization_call_result_17692 = invoke(stypy.reporting.localization.Localization(__file__, 44, 25), Localization_17686, *[program_name_17688, int_17689, int_17690], **kwargs_17691)
        
        str_17693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 82), 'str', '__file__')
        # Getting the type of 'str' (line 44)
        str_17694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 94), 'str', False)
        # Processing the call keyword arguments (line 44)
        kwargs_17695 = {}
        # Getting the type of 'self' (line 44)
        self_17683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'self', False)
        # Obtaining the member 'set_type_of' of a type (line 44)
        set_type_of_17684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), self_17683, 'set_type_of')
        # Calling set_type_of(args, kwargs) (line 44)
        set_type_of_call_result_17696 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), set_type_of_17684, *[Localization_call_result_17692, str_17693, str_17694], **kwargs_17695)
        
        
        # Call to set_type_of(...): (line 45)
        # Processing the call arguments (line 45)
        
        # Call to Localization(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'self' (line 45)
        self_17701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 56), 'self', False)
        # Obtaining the member 'program_name' of a type (line 45)
        program_name_17702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 56), self_17701, 'program_name')
        int_17703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 75), 'int')
        int_17704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 78), 'int')
        # Processing the call keyword arguments (line 45)
        kwargs_17705 = {}
        # Getting the type of 'localization_copy' (line 45)
        localization_copy_17699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), 'localization_copy', False)
        # Obtaining the member 'Localization' of a type (line 45)
        Localization_17700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 25), localization_copy_17699, 'Localization')
        # Calling Localization(args, kwargs) (line 45)
        Localization_call_result_17706 = invoke(stypy.reporting.localization.Localization(__file__, 45, 25), Localization_17700, *[program_name_17702, int_17703, int_17704], **kwargs_17705)
        
        str_17707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 82), 'str', '__doc__')
        # Getting the type of 'str' (line 45)
        str_17708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 93), 'str', False)
        # Processing the call keyword arguments (line 45)
        kwargs_17709 = {}
        # Getting the type of 'self' (line 45)
        self_17697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self', False)
        # Obtaining the member 'set_type_of' of a type (line 45)
        set_type_of_17698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_17697, 'set_type_of')
        # Calling set_type_of(args, kwargs) (line 45)
        set_type_of_call_result_17710 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), set_type_of_17698, *[Localization_call_result_17706, str_17707, str_17708], **kwargs_17709)
        
        
        # Call to set_type_of(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Call to Localization(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'self' (line 46)
        self_17715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 56), 'self', False)
        # Obtaining the member 'program_name' of a type (line 46)
        program_name_17716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 56), self_17715, 'program_name')
        int_17717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 75), 'int')
        int_17718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 78), 'int')
        # Processing the call keyword arguments (line 46)
        kwargs_17719 = {}
        # Getting the type of 'localization_copy' (line 46)
        localization_copy_17713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 25), 'localization_copy', False)
        # Obtaining the member 'Localization' of a type (line 46)
        Localization_17714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 25), localization_copy_17713, 'Localization')
        # Calling Localization(args, kwargs) (line 46)
        Localization_call_result_17720 = invoke(stypy.reporting.localization.Localization(__file__, 46, 25), Localization_17714, *[program_name_17716, int_17717, int_17718], **kwargs_17719)
        
        str_17721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 82), 'str', '__name__')
        # Getting the type of 'str' (line 46)
        str_17722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 94), 'str', False)
        # Processing the call keyword arguments (line 46)
        kwargs_17723 = {}
        # Getting the type of 'self' (line 46)
        self_17711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self', False)
        # Obtaining the member 'set_type_of' of a type (line 46)
        set_type_of_17712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_17711, 'set_type_of')
        # Calling set_type_of(args, kwargs) (line 46)
        set_type_of_call_result_17724 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), set_type_of_17712, *[Localization_call_result_17720, str_17721, str_17722], **kwargs_17723)
        
        
        # Call to set_type_of(...): (line 47)
        # Processing the call arguments (line 47)
        
        # Call to Localization(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'self' (line 47)
        self_17729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 56), 'self', False)
        # Obtaining the member 'program_name' of a type (line 47)
        program_name_17730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 56), self_17729, 'program_name')
        int_17731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 75), 'int')
        int_17732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 78), 'int')
        # Processing the call keyword arguments (line 47)
        kwargs_17733 = {}
        # Getting the type of 'localization_copy' (line 47)
        localization_copy_17727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 25), 'localization_copy', False)
        # Obtaining the member 'Localization' of a type (line 47)
        Localization_17728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 25), localization_copy_17727, 'Localization')
        # Calling Localization(args, kwargs) (line 47)
        Localization_call_result_17734 = invoke(stypy.reporting.localization.Localization(__file__, 47, 25), Localization_17728, *[program_name_17730, int_17731, int_17732], **kwargs_17733)
        
        str_17735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 82), 'str', '__package__')
        # Getting the type of 'str' (line 47)
        str_17736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 97), 'str', False)
        # Processing the call keyword arguments (line 47)
        kwargs_17737 = {}
        # Getting the type of 'self' (line 47)
        self_17725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'self', False)
        # Obtaining the member 'set_type_of' of a type (line 47)
        set_type_of_17726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), self_17725, 'set_type_of')
        # Calling set_type_of(args, kwargs) (line 47)
        set_type_of_call_result_17738 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), set_type_of_17726, *[Localization_call_result_17734, str_17735, str_17736], **kwargs_17737)
        
        
        # ################# End of '__load_predefined_variables(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__load_predefined_variables' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_17739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17739)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__load_predefined_variables'
        return stypy_return_type_17739


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 49, 4, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.__init__', ['file_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['file_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_17740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, (-1)), 'str', '\n        Creates a type store for the passed file name (module)\n        :param file_name: file name to create the TypeStore for\n        :return:\n        ')
        
        # Assigning a Call to a Name (line 55):
        
        # Assigning a Call to a Name (line 55):
        
        # Call to replace(...): (line 55)
        # Processing the call arguments (line 55)
        str_17743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 38), 'str', '\\')
        str_17744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 44), 'str', '/')
        # Processing the call keyword arguments (line 55)
        kwargs_17745 = {}
        # Getting the type of 'file_name' (line 55)
        file_name_17741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 20), 'file_name', False)
        # Obtaining the member 'replace' of a type (line 55)
        replace_17742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 20), file_name_17741, 'replace')
        # Calling replace(args, kwargs) (line 55)
        replace_call_result_17746 = invoke(stypy.reporting.localization.Localization(__file__, 55, 20), replace_17742, *[str_17743, str_17744], **kwargs_17745)
        
        # Assigning a type to the variable 'file_name' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'file_name', replace_call_result_17746)
        
        # Assigning a Call to a Attribute (line 56):
        
        # Assigning a Call to a Attribute (line 56):
        
        # Call to replace(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'stypy_parameters_copy' (line 56)
        stypy_parameters_copy_17749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 46), 'stypy_parameters_copy', False)
        # Obtaining the member 'type_inference_file_postfix' of a type (line 56)
        type_inference_file_postfix_17750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 46), stypy_parameters_copy_17749, 'type_inference_file_postfix')
        str_17751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 97), 'str', '')
        # Processing the call keyword arguments (line 56)
        kwargs_17752 = {}
        # Getting the type of 'file_name' (line 56)
        file_name_17747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 28), 'file_name', False)
        # Obtaining the member 'replace' of a type (line 56)
        replace_17748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 28), file_name_17747, 'replace')
        # Calling replace(args, kwargs) (line 56)
        replace_call_result_17753 = invoke(stypy.reporting.localization.Localization(__file__, 56, 28), replace_17748, *[type_inference_file_postfix_17750, str_17751], **kwargs_17752)
        
        # Getting the type of 'self' (line 56)
        self_17754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self')
        # Setting the type of the member 'program_name' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_17754, 'program_name', replace_call_result_17753)
        
        # Assigning a Call to a Attribute (line 57):
        
        # Assigning a Call to a Attribute (line 57):
        
        # Call to replace(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'stypy_parameters_copy' (line 57)
        stypy_parameters_copy_17758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 54), 'stypy_parameters_copy', False)
        # Obtaining the member 'type_inference_file_directory_name' of a type (line 57)
        type_inference_file_directory_name_17759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 54), stypy_parameters_copy_17758, 'type_inference_file_directory_name')
        str_17760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 113), 'str', '/')
        # Applying the binary operator '+' (line 57)
        result_add_17761 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 54), '+', type_inference_file_directory_name_17759, str_17760)
        
        str_17762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 118), 'str', '')
        # Processing the call keyword arguments (line 57)
        kwargs_17763 = {}
        # Getting the type of 'self' (line 57)
        self_17755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 28), 'self', False)
        # Obtaining the member 'program_name' of a type (line 57)
        program_name_17756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 28), self_17755, 'program_name')
        # Obtaining the member 'replace' of a type (line 57)
        replace_17757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 28), program_name_17756, 'replace')
        # Calling replace(args, kwargs) (line 57)
        replace_call_result_17764 = invoke(stypy.reporting.localization.Localization(__file__, 57, 28), replace_17757, *[result_add_17761, str_17762], **kwargs_17763)
        
        # Getting the type of 'self' (line 57)
        self_17765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self')
        # Setting the type of the member 'program_name' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_17765, 'program_name', replace_call_result_17764)
        
        # Assigning a Call to a Name (line 60):
        
        # Assigning a Call to a Name (line 60):
        
        # Call to FunctionContext(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'file_name' (line 60)
        file_name_17767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 39), 'file_name', False)
        # Getting the type of 'True' (line 60)
        True_17768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 50), 'True', False)
        # Processing the call keyword arguments (line 60)
        kwargs_17769 = {}
        # Getting the type of 'FunctionContext' (line 60)
        FunctionContext_17766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'FunctionContext', False)
        # Calling FunctionContext(args, kwargs) (line 60)
        FunctionContext_call_result_17770 = invoke(stypy.reporting.localization.Localization(__file__, 60, 23), FunctionContext_17766, *[file_name_17767, True_17768], **kwargs_17769)
        
        # Assigning a type to the variable 'main_context' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'main_context', FunctionContext_call_result_17770)
        
        # Assigning a Call to a Attribute (line 63):
        
        # Assigning a Call to a Attribute (line 63):
        
        # Call to get_instance_for_file(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'self' (line 63)
        self_17773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 84), 'self', False)
        # Obtaining the member 'program_name' of a type (line 63)
        program_name_17774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 84), self_17773, 'program_name')
        # Processing the call keyword arguments (line 63)
        kwargs_17775 = {}
        # Getting the type of 'TypeAnnotationRecord' (line 63)
        TypeAnnotationRecord_17771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 41), 'TypeAnnotationRecord', False)
        # Obtaining the member 'get_instance_for_file' of a type (line 63)
        get_instance_for_file_17772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 41), TypeAnnotationRecord_17771, 'get_instance_for_file')
        # Calling get_instance_for_file(args, kwargs) (line 63)
        get_instance_for_file_call_result_17776 = invoke(stypy.reporting.localization.Localization(__file__, 63, 41), get_instance_for_file_17772, *[program_name_17774], **kwargs_17775)
        
        # Getting the type of 'main_context' (line 63)
        main_context_17777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'main_context')
        # Setting the type of the member 'annotation_record' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), main_context_17777, 'annotation_record', get_instance_for_file_call_result_17776)
        
        # Assigning a List to a Attribute (line 66):
        
        # Assigning a List to a Attribute (line 66):
        
        # Obtaining an instance of the builtin type 'list' (line 66)
        list_17778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 66)
        # Adding element type (line 66)
        # Getting the type of 'main_context' (line 66)
        main_context_17779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 30), 'main_context')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 29), list_17778, main_context_17779)
        
        # Getting the type of 'self' (line 66)
        self_17780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self')
        # Setting the type of the member 'context_stack' of a type (line 66)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_17780, 'context_stack', list_17778)
        
        # Assigning a List to a Attribute (line 70):
        
        # Assigning a List to a Attribute (line 70):
        
        # Obtaining an instance of the builtin type 'list' (line 70)
        list_17781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 70)
        
        # Getting the type of 'self' (line 70)
        self_17782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'self')
        # Setting the type of the member 'last_function_contexts' of a type (line 70)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), self_17782, 'last_function_contexts', list_17781)
        
        # Assigning a Attribute to a Attribute (line 73):
        
        # Assigning a Attribute to a Attribute (line 73):
        # Getting the type of 'stypy_parameters_copy' (line 73)
        stypy_parameters_copy_17783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 37), 'stypy_parameters_copy')
        # Obtaining the member 'ENABLE_CODING_ADVICES' of a type (line 73)
        ENABLE_CODING_ADVICES_17784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 37), stypy_parameters_copy_17783, 'ENABLE_CODING_ADVICES')
        # Getting the type of 'self' (line 73)
        self_17785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self')
        # Setting the type of the member 'test_unreferenced_var' of a type (line 73)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_17785, 'test_unreferenced_var', ENABLE_CODING_ADVICES_17784)
        
        # Assigning a List to a Attribute (line 77):
        
        # Assigning a List to a Attribute (line 77):
        
        # Obtaining an instance of the builtin type 'list' (line 77)
        list_17786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 77)
        
        # Getting the type of 'self' (line 77)
        self_17787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'self')
        # Setting the type of the member 'external_modules' of a type (line 77)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), self_17787, 'external_modules', list_17786)
        
        # Call to __load_predefined_variables(...): (line 78)
        # Processing the call keyword arguments (line 78)
        kwargs_17790 = {}
        # Getting the type of 'self' (line 78)
        self_17788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self', False)
        # Obtaining the member '__load_predefined_variables' of a type (line 78)
        load_predefined_variables_17789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_17788, '__load_predefined_variables')
        # Calling __load_predefined_variables(args, kwargs) (line 78)
        load_predefined_variables_call_result_17791 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), load_predefined_variables_17789, *[], **kwargs_17790)
        
        
        # Assigning a Call to a Name (line 80):
        
        # Assigning a Call to a Name (line 80):
        
        # Call to replace(...): (line 80)
        # Processing the call arguments (line 80)
        str_17800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 64), 'str', '\\')
        str_17801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 70), 'str', '/')
        # Processing the call keyword arguments (line 80)
        kwargs_17802 = {}
        
        # Call to abspath(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'self' (line 80)
        self_17795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 37), 'self', False)
        # Obtaining the member 'program_name' of a type (line 80)
        program_name_17796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 37), self_17795, 'program_name')
        # Processing the call keyword arguments (line 80)
        kwargs_17797 = {}
        # Getting the type of 'os' (line 80)
        os_17792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 21), 'os', False)
        # Obtaining the member 'path' of a type (line 80)
        path_17793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 21), os_17792, 'path')
        # Obtaining the member 'abspath' of a type (line 80)
        abspath_17794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 21), path_17793, 'abspath')
        # Calling abspath(args, kwargs) (line 80)
        abspath_call_result_17798 = invoke(stypy.reporting.localization.Localization(__file__, 80, 21), abspath_17794, *[program_name_17796], **kwargs_17797)
        
        # Obtaining the member 'replace' of a type (line 80)
        replace_17799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 21), abspath_call_result_17798, 'replace')
        # Calling replace(args, kwargs) (line 80)
        replace_call_result_17803 = invoke(stypy.reporting.localization.Localization(__file__, 80, 21), replace_17799, *[str_17800, str_17801], **kwargs_17802)
        
        # Assigning a type to the variable 'file_cache' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'file_cache', replace_call_result_17803)
        
        # Assigning a Name to a Subscript (line 82):
        
        # Assigning a Name to a Subscript (line 82):
        # Getting the type of 'self' (line 82)
        self_17804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 55), 'self')
        # Getting the type of 'TypeStore' (line 82)
        TypeStore_17805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'TypeStore')
        # Obtaining the member 'type_stores_of_modules' of a type (line 82)
        type_stores_of_modules_17806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), TypeStore_17805, 'type_stores_of_modules')
        # Getting the type of 'file_cache' (line 82)
        file_cache_17807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 41), 'file_cache')
        # Storing an element on a container (line 82)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 8), type_stores_of_modules_17806, (file_cache_17807, self_17804))
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def add_external_module(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_external_module'
        module_type_store = module_type_store.open_function_context('add_external_module', 84, 4, False)
        # Assigning a type to the variable 'self' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.add_external_module.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.add_external_module.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.add_external_module.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.add_external_module.__dict__.__setitem__('stypy_function_name', 'TypeStore.add_external_module')
        TypeStore.add_external_module.__dict__.__setitem__('stypy_param_names_list', ['stypy_object'])
        TypeStore.add_external_module.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.add_external_module.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.add_external_module.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.add_external_module.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.add_external_module.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.add_external_module.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.add_external_module', ['stypy_object'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_external_module', localization, ['stypy_object'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_external_module(...)' code ##################

        str_17808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, (-1)), 'str', '\n        Adds a external module to the list of modules used by this one\n        :param stypy_object:\n        :return:\n        ')
        
        # Call to append(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'stypy_object' (line 90)
        stypy_object_17812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 37), 'stypy_object', False)
        # Processing the call keyword arguments (line 90)
        kwargs_17813 = {}
        # Getting the type of 'self' (line 90)
        self_17809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self', False)
        # Obtaining the member 'external_modules' of a type (line 90)
        external_modules_17810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_17809, 'external_modules')
        # Obtaining the member 'append' of a type (line 90)
        append_17811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), external_modules_17810, 'append')
        # Calling append(args, kwargs) (line 90)
        append_call_result_17814 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), append_17811, *[stypy_object_17812], **kwargs_17813)
        
        
        # Assigning a Call to a Name (line 91):
        
        # Assigning a Call to a Name (line 91):
        
        # Call to get_analyzed_program_type_store(...): (line 91)
        # Processing the call keyword arguments (line 91)
        kwargs_17817 = {}
        # Getting the type of 'stypy_object' (line 91)
        stypy_object_17815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 28), 'stypy_object', False)
        # Obtaining the member 'get_analyzed_program_type_store' of a type (line 91)
        get_analyzed_program_type_store_17816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 28), stypy_object_17815, 'get_analyzed_program_type_store')
        # Calling get_analyzed_program_type_store(args, kwargs) (line 91)
        get_analyzed_program_type_store_call_result_17818 = invoke(stypy.reporting.localization.Localization(__file__, 91, 28), get_analyzed_program_type_store_17816, *[], **kwargs_17817)
        
        # Assigning a type to the variable 'module_type_store' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'module_type_store', get_analyzed_program_type_store_call_result_17818)
        
        # Assigning a Attribute to a Attribute (line 92):
        
        # Assigning a Attribute to a Attribute (line 92):
        # Getting the type of 'self' (line 92)
        self_17819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 51), 'self')
        # Obtaining the member 'last_function_contexts' of a type (line 92)
        last_function_contexts_17820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 51), self_17819, 'last_function_contexts')
        # Getting the type of 'module_type_store' (line 92)
        module_type_store_17821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'module_type_store')
        # Setting the type of the member 'last_function_contexts' of a type (line 92)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), module_type_store_17821, 'last_function_contexts', last_function_contexts_17820)
        
        # ################# End of 'add_external_module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_external_module' in the type store
        # Getting the type of 'stypy_return_type' (line 84)
        stypy_return_type_17822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17822)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_external_module'
        return stypy_return_type_17822


    @norecursion
    def get_all_processed_function_contexts(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_all_processed_function_contexts'
        module_type_store = module_type_store.open_function_context('get_all_processed_function_contexts', 94, 4, False)
        # Assigning a type to the variable 'self' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.get_all_processed_function_contexts.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.get_all_processed_function_contexts.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.get_all_processed_function_contexts.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.get_all_processed_function_contexts.__dict__.__setitem__('stypy_function_name', 'TypeStore.get_all_processed_function_contexts')
        TypeStore.get_all_processed_function_contexts.__dict__.__setitem__('stypy_param_names_list', [])
        TypeStore.get_all_processed_function_contexts.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.get_all_processed_function_contexts.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.get_all_processed_function_contexts.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.get_all_processed_function_contexts.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.get_all_processed_function_contexts.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.get_all_processed_function_contexts.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.get_all_processed_function_contexts', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_all_processed_function_contexts', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_all_processed_function_contexts(...)' code ##################

        str_17823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, (-1)), 'str', '\n        Obtain a list of all the function context that were ever used during the program execution (active + past ones)\n        :return: List of function contexts\n        ')
        # Getting the type of 'self' (line 99)
        self_17824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 15), 'self')
        # Obtaining the member 'context_stack' of a type (line 99)
        context_stack_17825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 15), self_17824, 'context_stack')
        # Getting the type of 'self' (line 99)
        self_17826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 36), 'self')
        # Obtaining the member 'last_function_contexts' of a type (line 99)
        last_function_contexts_17827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 36), self_17826, 'last_function_contexts')
        # Applying the binary operator '+' (line 99)
        result_add_17828 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 15), '+', context_stack_17825, last_function_contexts_17827)
        
        # Assigning a type to the variable 'stypy_return_type' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'stypy_return_type', result_add_17828)
        
        # ################# End of 'get_all_processed_function_contexts(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_all_processed_function_contexts' in the type store
        # Getting the type of 'stypy_return_type' (line 94)
        stypy_return_type_17829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17829)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_all_processed_function_contexts'
        return stypy_return_type_17829


    @norecursion
    def set_check_unreferenced_vars(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_check_unreferenced_vars'
        module_type_store = module_type_store.open_function_context('set_check_unreferenced_vars', 101, 4, False)
        # Assigning a type to the variable 'self' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.set_check_unreferenced_vars.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.set_check_unreferenced_vars.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.set_check_unreferenced_vars.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.set_check_unreferenced_vars.__dict__.__setitem__('stypy_function_name', 'TypeStore.set_check_unreferenced_vars')
        TypeStore.set_check_unreferenced_vars.__dict__.__setitem__('stypy_param_names_list', ['state'])
        TypeStore.set_check_unreferenced_vars.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.set_check_unreferenced_vars.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.set_check_unreferenced_vars.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.set_check_unreferenced_vars.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.set_check_unreferenced_vars.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.set_check_unreferenced_vars.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.set_check_unreferenced_vars', ['state'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_check_unreferenced_vars', localization, ['state'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_check_unreferenced_vars(...)' code ##################

        str_17830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, (-1)), 'str', '\n        On some occasions, such as when invoking methods or reading default values from parameters, the unreferenced\n        var checks must be disabled to ensure proper behavior.\n        :param state: bool value. However, if coding advices are disabled, this method has no functionality, they are\n        always set to False\n        :return:\n        ')
        
        # Getting the type of 'stypy_parameters_copy' (line 109)
        stypy_parameters_copy_17831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 15), 'stypy_parameters_copy')
        # Obtaining the member 'ENABLE_CODING_ADVICES' of a type (line 109)
        ENABLE_CODING_ADVICES_17832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 15), stypy_parameters_copy_17831, 'ENABLE_CODING_ADVICES')
        # Applying the 'not' unary operator (line 109)
        result_not__17833 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 11), 'not', ENABLE_CODING_ADVICES_17832)
        
        # Testing if the type of an if condition is none (line 109)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 109, 8), result_not__17833):
            pass
        else:
            
            # Testing the type of an if condition (line 109)
            if_condition_17834 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 8), result_not__17833)
            # Assigning a type to the variable 'if_condition_17834' (line 109)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'if_condition_17834', if_condition_17834)
            # SSA begins for if statement (line 109)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 110):
            
            # Assigning a Name to a Attribute (line 110):
            # Getting the type of 'False' (line 110)
            False_17835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 41), 'False')
            # Getting the type of 'self' (line 110)
            self_17836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'self')
            # Setting the type of the member 'test_unreferenced_var' of a type (line 110)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), self_17836, 'test_unreferenced_var', False_17835)
            # Assigning a type to the variable 'stypy_return_type' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 109)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Name to a Attribute (line 112):
        
        # Assigning a Name to a Attribute (line 112):
        # Getting the type of 'state' (line 112)
        state_17837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 37), 'state')
        # Getting the type of 'self' (line 112)
        self_17838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'self')
        # Setting the type of the member 'test_unreferenced_var' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), self_17838, 'test_unreferenced_var', state_17837)
        
        # ################# End of 'set_check_unreferenced_vars(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_check_unreferenced_vars' in the type store
        # Getting the type of 'stypy_return_type' (line 101)
        stypy_return_type_17839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17839)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_check_unreferenced_vars'
        return stypy_return_type_17839


    @norecursion
    def set_context(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_17840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 40), 'str', '')
        int_17841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 51), 'int')
        int_17842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 66), 'int')
        defaults = [str_17840, int_17841, int_17842]
        # Create a new context for function 'set_context'
        module_type_store = module_type_store.open_function_context('set_context', 114, 4, False)
        # Assigning a type to the variable 'self' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.set_context.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.set_context.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.set_context.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.set_context.__dict__.__setitem__('stypy_function_name', 'TypeStore.set_context')
        TypeStore.set_context.__dict__.__setitem__('stypy_param_names_list', ['function_name', 'lineno', 'col_offset'])
        TypeStore.set_context.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.set_context.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.set_context.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.set_context.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.set_context.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.set_context.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.set_context', ['function_name', 'lineno', 'col_offset'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_context', localization, ['function_name', 'lineno', 'col_offset'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_context(...)' code ##################

        str_17843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, (-1)), 'str', '\n        Creates a new function context in the top position of the context stack\n        ')
        
        # Assigning a Call to a Name (line 118):
        
        # Assigning a Call to a Name (line 118):
        
        # Call to FunctionContext(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'function_name' (line 118)
        function_name_17845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 34), 'function_name', False)
        # Processing the call keyword arguments (line 118)
        kwargs_17846 = {}
        # Getting the type of 'FunctionContext' (line 118)
        FunctionContext_17844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 18), 'FunctionContext', False)
        # Calling FunctionContext(args, kwargs) (line 118)
        FunctionContext_call_result_17847 = invoke(stypy.reporting.localization.Localization(__file__, 118, 18), FunctionContext_17844, *[function_name_17845], **kwargs_17846)
        
        # Assigning a type to the variable 'context' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'context', FunctionContext_call_result_17847)
        
        # Assigning a Name to a Attribute (line 119):
        
        # Assigning a Name to a Attribute (line 119):
        # Getting the type of 'lineno' (line 119)
        lineno_17848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 35), 'lineno')
        # Getting the type of 'context' (line 119)
        context_17849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'context')
        # Setting the type of the member 'declaration_line' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), context_17849, 'declaration_line', lineno_17848)
        
        # Assigning a Name to a Attribute (line 120):
        
        # Assigning a Name to a Attribute (line 120):
        # Getting the type of 'col_offset' (line 120)
        col_offset_17850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 37), 'col_offset')
        # Getting the type of 'context' (line 120)
        context_17851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'context')
        # Setting the type of the member 'declaration_column' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), context_17851, 'declaration_column', col_offset_17850)
        
        # Assigning a Call to a Attribute (line 121):
        
        # Assigning a Call to a Attribute (line 121):
        
        # Call to get_instance_for_file(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'self' (line 121)
        self_17854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 79), 'self', False)
        # Obtaining the member 'program_name' of a type (line 121)
        program_name_17855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 79), self_17854, 'program_name')
        # Processing the call keyword arguments (line 121)
        kwargs_17856 = {}
        # Getting the type of 'TypeAnnotationRecord' (line 121)
        TypeAnnotationRecord_17852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 36), 'TypeAnnotationRecord', False)
        # Obtaining the member 'get_instance_for_file' of a type (line 121)
        get_instance_for_file_17853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 36), TypeAnnotationRecord_17852, 'get_instance_for_file')
        # Calling get_instance_for_file(args, kwargs) (line 121)
        get_instance_for_file_call_result_17857 = invoke(stypy.reporting.localization.Localization(__file__, 121, 36), get_instance_for_file_17853, *[program_name_17855], **kwargs_17856)
        
        # Getting the type of 'context' (line 121)
        context_17858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'context')
        # Setting the type of the member 'annotation_record' of a type (line 121)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), context_17858, 'annotation_record', get_instance_for_file_call_result_17857)
        
        # Call to insert(...): (line 123)
        # Processing the call arguments (line 123)
        int_17862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 34), 'int')
        # Getting the type of 'context' (line 123)
        context_17863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 37), 'context', False)
        # Processing the call keyword arguments (line 123)
        kwargs_17864 = {}
        # Getting the type of 'self' (line 123)
        self_17859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'self', False)
        # Obtaining the member 'context_stack' of a type (line 123)
        context_stack_17860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), self_17859, 'context_stack')
        # Obtaining the member 'insert' of a type (line 123)
        insert_17861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), context_stack_17860, 'insert')
        # Calling insert(args, kwargs) (line 123)
        insert_call_result_17865 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), insert_17861, *[int_17862, context_17863], **kwargs_17864)
        
        
        # ################# End of 'set_context(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_context' in the type store
        # Getting the type of 'stypy_return_type' (line 114)
        stypy_return_type_17866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17866)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_context'
        return stypy_return_type_17866


    @norecursion
    def unset_context(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'unset_context'
        module_type_store = module_type_store.open_function_context('unset_context', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.unset_context.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.unset_context.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.unset_context.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.unset_context.__dict__.__setitem__('stypy_function_name', 'TypeStore.unset_context')
        TypeStore.unset_context.__dict__.__setitem__('stypy_param_names_list', [])
        TypeStore.unset_context.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.unset_context.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.unset_context.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.unset_context.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.unset_context.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.unset_context.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.unset_context', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'unset_context', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'unset_context(...)' code ##################

        str_17867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, (-1)), 'str', '\n        Pops and returns the topmost context in the context stack\n        :return:\n        ')
        # Evaluating assert statement condition
        
        
        # Call to len(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'self' (line 131)
        self_17869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 19), 'self', False)
        # Obtaining the member 'context_stack' of a type (line 131)
        context_stack_17870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 19), self_17869, 'context_stack')
        # Processing the call keyword arguments (line 131)
        kwargs_17871 = {}
        # Getting the type of 'len' (line 131)
        len_17868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'len', False)
        # Calling len(args, kwargs) (line 131)
        len_call_result_17872 = invoke(stypy.reporting.localization.Localization(__file__, 131, 15), len_17868, *[context_stack_17870], **kwargs_17871)
        
        int_17873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 41), 'int')
        # Applying the binary operator '>' (line 131)
        result_gt_17874 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 15), '>', len_call_result_17872, int_17873)
        
        assert_17875 = result_gt_17874
        # Assigning a type to the variable 'assert_17875' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'assert_17875', result_gt_17874)
        
        # Assigning a Call to a Name (line 133):
        
        # Assigning a Call to a Name (line 133):
        
        # Call to pop(...): (line 133)
        # Processing the call arguments (line 133)
        int_17879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 41), 'int')
        # Processing the call keyword arguments (line 133)
        kwargs_17880 = {}
        # Getting the type of 'self' (line 133)
        self_17876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 18), 'self', False)
        # Obtaining the member 'context_stack' of a type (line 133)
        context_stack_17877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 18), self_17876, 'context_stack')
        # Obtaining the member 'pop' of a type (line 133)
        pop_17878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 18), context_stack_17877, 'pop')
        # Calling pop(args, kwargs) (line 133)
        pop_call_result_17881 = invoke(stypy.reporting.localization.Localization(__file__, 133, 18), pop_17878, *[int_17879], **kwargs_17880)
        
        # Assigning a type to the variable 'context' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'context', pop_call_result_17881)
        
        # Call to append(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'context' (line 134)
        context_17885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 43), 'context', False)
        # Processing the call keyword arguments (line 134)
        kwargs_17886 = {}
        # Getting the type of 'self' (line 134)
        self_17882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'self', False)
        # Obtaining the member 'last_function_contexts' of a type (line 134)
        last_function_contexts_17883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), self_17882, 'last_function_contexts')
        # Obtaining the member 'append' of a type (line 134)
        append_17884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), last_function_contexts_17883, 'append')
        # Calling append(args, kwargs) (line 134)
        append_call_result_17887 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), append_17884, *[context_17885], **kwargs_17886)
        
        # Getting the type of 'context' (line 136)
        context_17888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 15), 'context')
        # Assigning a type to the variable 'stypy_return_type' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'stypy_return_type', context_17888)
        
        # ################# End of 'unset_context(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'unset_context' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_17889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17889)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'unset_context'
        return stypy_return_type_17889


    @norecursion
    def get_context(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_context'
        module_type_store = module_type_store.open_function_context('get_context', 138, 4, False)
        # Assigning a type to the variable 'self' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.get_context.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.get_context.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.get_context.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.get_context.__dict__.__setitem__('stypy_function_name', 'TypeStore.get_context')
        TypeStore.get_context.__dict__.__setitem__('stypy_param_names_list', [])
        TypeStore.get_context.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.get_context.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.get_context.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.get_context.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.get_context.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.get_context.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.get_context', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_context', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_context(...)' code ##################

        str_17890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, (-1)), 'str', '\n        Gets the current (topmost) context.\n        :return: The current context\n        ')
        
        # Obtaining the type of the subscript
        int_17891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 34), 'int')
        # Getting the type of 'self' (line 143)
        self_17892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 15), 'self')
        # Obtaining the member 'context_stack' of a type (line 143)
        context_stack_17893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 15), self_17892, 'context_stack')
        # Obtaining the member '__getitem__' of a type (line 143)
        getitem___17894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 15), context_stack_17893, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 143)
        subscript_call_result_17895 = invoke(stypy.reporting.localization.Localization(__file__, 143, 15), getitem___17894, int_17891)
        
        # Assigning a type to the variable 'stypy_return_type' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'stypy_return_type', subscript_call_result_17895)
        
        # ################# End of 'get_context(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_context' in the type store
        # Getting the type of 'stypy_return_type' (line 138)
        stypy_return_type_17896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17896)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_context'
        return stypy_return_type_17896


    @norecursion
    def mark_as_global(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mark_as_global'
        module_type_store = module_type_store.open_function_context('mark_as_global', 145, 4, False)
        # Assigning a type to the variable 'self' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.mark_as_global.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.mark_as_global.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.mark_as_global.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.mark_as_global.__dict__.__setitem__('stypy_function_name', 'TypeStore.mark_as_global')
        TypeStore.mark_as_global.__dict__.__setitem__('stypy_param_names_list', ['localization', 'name'])
        TypeStore.mark_as_global.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.mark_as_global.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.mark_as_global.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.mark_as_global.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.mark_as_global.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.mark_as_global.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.mark_as_global', ['localization', 'name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mark_as_global', localization, ['localization', 'name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mark_as_global(...)' code ##################

        str_17897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, (-1)), 'str', '\n        Mark a variable as global in the current function context\n        :param localization: Caller information\n        :param name: variable name\n        :return:\n        ')
        
        # Assigning a Name to a Name (line 152):
        
        # Assigning a Name to a Name (line 152):
        # Getting the type of 'None' (line 152)
        None_17898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 14), 'None')
        # Assigning a type to the variable 'ret' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'ret', None_17898)
        
        # Call to set_check_unreferenced_vars(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'False' (line 153)
        False_17901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 41), 'False', False)
        # Processing the call keyword arguments (line 153)
        kwargs_17902 = {}
        # Getting the type of 'self' (line 153)
        self_17899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'self', False)
        # Obtaining the member 'set_check_unreferenced_vars' of a type (line 153)
        set_check_unreferenced_vars_17900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), self_17899, 'set_check_unreferenced_vars')
        # Calling set_check_unreferenced_vars(args, kwargs) (line 153)
        set_check_unreferenced_vars_call_result_17903 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), set_check_unreferenced_vars_17900, *[False_17901], **kwargs_17902)
        
        
        # Assigning a Call to a Name (line 154):
        
        # Assigning a Call to a Name (line 154):
        
        # Call to get_type_of(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'name' (line 154)
        name_17909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 50), 'name', False)
        # Processing the call keyword arguments (line 154)
        kwargs_17910 = {}
        
        # Call to get_context(...): (line 154)
        # Processing the call keyword arguments (line 154)
        kwargs_17906 = {}
        # Getting the type of 'self' (line 154)
        self_17904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 19), 'self', False)
        # Obtaining the member 'get_context' of a type (line 154)
        get_context_17905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 19), self_17904, 'get_context')
        # Calling get_context(args, kwargs) (line 154)
        get_context_call_result_17907 = invoke(stypy.reporting.localization.Localization(__file__, 154, 19), get_context_17905, *[], **kwargs_17906)
        
        # Obtaining the member 'get_type_of' of a type (line 154)
        get_type_of_17908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 19), get_context_call_result_17907, 'get_type_of')
        # Calling get_type_of(args, kwargs) (line 154)
        get_type_of_call_result_17911 = invoke(stypy.reporting.localization.Localization(__file__, 154, 19), get_type_of_17908, *[name_17909], **kwargs_17910)
        
        # Assigning a type to the variable 'var_type' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'var_type', get_type_of_call_result_17911)
        
        # Call to set_check_unreferenced_vars(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'True' (line 155)
        True_17914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 41), 'True', False)
        # Processing the call keyword arguments (line 155)
        kwargs_17915 = {}
        # Getting the type of 'self' (line 155)
        self_17912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'self', False)
        # Obtaining the member 'set_check_unreferenced_vars' of a type (line 155)
        set_check_unreferenced_vars_17913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), self_17912, 'set_check_unreferenced_vars')
        # Calling set_check_unreferenced_vars(args, kwargs) (line 155)
        set_check_unreferenced_vars_call_result_17916 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), set_check_unreferenced_vars_17913, *[True_17914], **kwargs_17915)
        
        
        # Type idiom detected: calculating its left and rigth part (line 156)
        # Getting the type of 'var_type' (line 156)
        var_type_17917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'var_type')
        # Getting the type of 'None' (line 156)
        None_17918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 27), 'None')
        
        (may_be_17919, more_types_in_union_17920) = may_not_be_none(var_type_17917, None_17918)

        if may_be_17919:

            if more_types_in_union_17920:
                # Runtime conditional SSA (line 156)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 157):
            
            # Assigning a Call to a Name (line 157):
            
            # Call to TypeWarning(...): (line 157)
            # Processing the call arguments (line 157)
            # Getting the type of 'localization' (line 157)
            localization_17922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 30), 'localization', False)
            
            # Call to format(...): (line 158)
            # Processing the call arguments (line 158)
            # Getting the type of 'name' (line 158)
            name_17925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 99), 'name', False)
            # Processing the call keyword arguments (line 158)
            kwargs_17926 = {}
            str_17923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 30), 'str', "SyntaxWarning: name '{0}' is used before global declaration")
            # Obtaining the member 'format' of a type (line 158)
            format_17924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 30), str_17923, 'format')
            # Calling format(args, kwargs) (line 158)
            format_call_result_17927 = invoke(stypy.reporting.localization.Localization(__file__, 158, 30), format_17924, *[name_17925], **kwargs_17926)
            
            # Processing the call keyword arguments (line 157)
            kwargs_17928 = {}
            # Getting the type of 'TypeWarning' (line 157)
            TypeWarning_17921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 18), 'TypeWarning', False)
            # Calling TypeWarning(args, kwargs) (line 157)
            TypeWarning_call_result_17929 = invoke(stypy.reporting.localization.Localization(__file__, 157, 18), TypeWarning_17921, *[localization_17922, format_call_result_17927], **kwargs_17928)
            
            # Assigning a type to the variable 'ret' (line 157)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'ret', TypeWarning_call_result_17929)
            
            
            
            # Call to get_context(...): (line 159)
            # Processing the call keyword arguments (line 159)
            kwargs_17932 = {}
            # Getting the type of 'self' (line 159)
            self_17930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 19), 'self', False)
            # Obtaining the member 'get_context' of a type (line 159)
            get_context_17931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 19), self_17930, 'get_context')
            # Calling get_context(args, kwargs) (line 159)
            get_context_call_result_17933 = invoke(stypy.reporting.localization.Localization(__file__, 159, 19), get_context_17931, *[], **kwargs_17932)
            
            
            # Call to get_global_context(...): (line 159)
            # Processing the call keyword arguments (line 159)
            kwargs_17936 = {}
            # Getting the type of 'self' (line 159)
            self_17934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 41), 'self', False)
            # Obtaining the member 'get_global_context' of a type (line 159)
            get_global_context_17935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 41), self_17934, 'get_global_context')
            # Calling get_global_context(args, kwargs) (line 159)
            get_global_context_call_result_17937 = invoke(stypy.reporting.localization.Localization(__file__, 159, 41), get_global_context_17935, *[], **kwargs_17936)
            
            # Applying the binary operator '==' (line 159)
            result_eq_17938 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 19), '==', get_context_call_result_17933, get_global_context_call_result_17937)
            
            # Applying the 'not' unary operator (line 159)
            result_not__17939 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 15), 'not', result_eq_17938)
            
            # Testing if the type of an if condition is none (line 159)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 159, 12), result_not__17939):
                pass
            else:
                
                # Testing the type of an if condition (line 159)
                if_condition_17940 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 12), result_not__17939)
                # Assigning a type to the variable 'if_condition_17940' (line 159)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'if_condition_17940', if_condition_17940)
                # SSA begins for if statement (line 159)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to set_type_of(...): (line 161)
                # Processing the call arguments (line 161)
                # Getting the type of 'name' (line 161)
                name_17946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 54), 'name', False)
                # Getting the type of 'var_type' (line 161)
                var_type_17947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 60), 'var_type', False)
                # Getting the type of 'localization' (line 161)
                localization_17948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 70), 'localization', False)
                # Processing the call keyword arguments (line 161)
                kwargs_17949 = {}
                
                # Call to get_global_context(...): (line 161)
                # Processing the call keyword arguments (line 161)
                kwargs_17943 = {}
                # Getting the type of 'self' (line 161)
                self_17941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'self', False)
                # Obtaining the member 'get_global_context' of a type (line 161)
                get_global_context_17942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 16), self_17941, 'get_global_context')
                # Calling get_global_context(args, kwargs) (line 161)
                get_global_context_call_result_17944 = invoke(stypy.reporting.localization.Localization(__file__, 161, 16), get_global_context_17942, *[], **kwargs_17943)
                
                # Obtaining the member 'set_type_of' of a type (line 161)
                set_type_of_17945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 16), get_global_context_call_result_17944, 'set_type_of')
                # Calling set_type_of(args, kwargs) (line 161)
                set_type_of_call_result_17950 = invoke(stypy.reporting.localization.Localization(__file__, 161, 16), set_type_of_17945, *[name_17946, var_type_17947, localization_17948], **kwargs_17949)
                
                # SSA join for if statement (line 159)
                module_type_store = module_type_store.join_ssa_context()
                


            if more_types_in_union_17920:
                # SSA join for if statement (line 156)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 163):
        
        # Assigning a Call to a Name (line 163):
        
        # Call to filter(...): (line 163)
        # Processing the call arguments (line 163)

        @norecursion
        def _stypy_temp_lambda_26(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_26'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_26', 163, 43, True)
            # Passed parameters checking function
            _stypy_temp_lambda_26.stypy_localization = localization
            _stypy_temp_lambda_26.stypy_type_of_self = None
            _stypy_temp_lambda_26.stypy_type_store = module_type_store
            _stypy_temp_lambda_26.stypy_function_name = '_stypy_temp_lambda_26'
            _stypy_temp_lambda_26.stypy_param_names_list = ['warn']
            _stypy_temp_lambda_26.stypy_varargs_param_name = None
            _stypy_temp_lambda_26.stypy_kwargs_param_name = None
            _stypy_temp_lambda_26.stypy_call_defaults = defaults
            _stypy_temp_lambda_26.stypy_call_varargs = varargs
            _stypy_temp_lambda_26.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_26', ['warn'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_26', ['warn'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Evaluating a boolean operation
            
            # Call to isinstance(...): (line 163)
            # Processing the call arguments (line 163)
            # Getting the type of 'warn' (line 163)
            warn_17953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 67), 'warn', False)
            # Getting the type of 'UnreferencedLocalVariableTypeWarning' (line 163)
            UnreferencedLocalVariableTypeWarning_17954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 73), 'UnreferencedLocalVariableTypeWarning', False)
            # Processing the call keyword arguments (line 163)
            kwargs_17955 = {}
            # Getting the type of 'isinstance' (line 163)
            isinstance_17952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 56), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 163)
            isinstance_call_result_17956 = invoke(stypy.reporting.localization.Localization(__file__, 163, 56), isinstance_17952, *[warn_17953, UnreferencedLocalVariableTypeWarning_17954], **kwargs_17955)
            
            
            # Getting the type of 'warn' (line 164)
            warn_17957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 56), 'warn', False)
            # Obtaining the member 'name' of a type (line 164)
            name_17958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 56), warn_17957, 'name')
            # Getting the type of 'name' (line 164)
            name_17959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 69), 'name', False)
            # Applying the binary operator '==' (line 164)
            result_eq_17960 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 56), '==', name_17958, name_17959)
            
            # Applying the binary operator 'and' (line 163)
            result_and_keyword_17961 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 56), 'and', isinstance_call_result_17956, result_eq_17960)
            
            # Getting the type of 'warn' (line 164)
            warn_17962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 78), 'warn', False)
            # Obtaining the member 'context' of a type (line 164)
            context_17963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 78), warn_17962, 'context')
            
            # Call to get_context(...): (line 164)
            # Processing the call keyword arguments (line 164)
            kwargs_17966 = {}
            # Getting the type of 'self' (line 164)
            self_17964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 94), 'self', False)
            # Obtaining the member 'get_context' of a type (line 164)
            get_context_17965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 94), self_17964, 'get_context')
            # Calling get_context(args, kwargs) (line 164)
            get_context_call_result_17967 = invoke(stypy.reporting.localization.Localization(__file__, 164, 94), get_context_17965, *[], **kwargs_17966)
            
            # Applying the binary operator '==' (line 164)
            result_eq_17968 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 78), '==', context_17963, get_context_call_result_17967)
            
            # Applying the binary operator 'and' (line 163)
            result_and_keyword_17969 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 56), 'and', result_and_keyword_17961, result_eq_17968)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 163)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 43), 'stypy_return_type', result_and_keyword_17969)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_26' in the type store
            # Getting the type of 'stypy_return_type' (line 163)
            stypy_return_type_17970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 43), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_17970)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_26'
            return stypy_return_type_17970

        # Assigning a type to the variable '_stypy_temp_lambda_26' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 43), '_stypy_temp_lambda_26', _stypy_temp_lambda_26)
        # Getting the type of '_stypy_temp_lambda_26' (line 163)
        _stypy_temp_lambda_26_17971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 43), '_stypy_temp_lambda_26')
        
        # Call to get_warning_msgs(...): (line 165)
        # Processing the call keyword arguments (line 165)
        kwargs_17974 = {}
        # Getting the type of 'TypeWarning' (line 165)
        TypeWarning_17972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 43), 'TypeWarning', False)
        # Obtaining the member 'get_warning_msgs' of a type (line 165)
        get_warning_msgs_17973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 43), TypeWarning_17972, 'get_warning_msgs')
        # Calling get_warning_msgs(args, kwargs) (line 165)
        get_warning_msgs_call_result_17975 = invoke(stypy.reporting.localization.Localization(__file__, 165, 43), get_warning_msgs_17973, *[], **kwargs_17974)
        
        # Processing the call keyword arguments (line 163)
        kwargs_17976 = {}
        # Getting the type of 'filter' (line 163)
        filter_17951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 36), 'filter', False)
        # Calling filter(args, kwargs) (line 163)
        filter_call_result_17977 = invoke(stypy.reporting.localization.Localization(__file__, 163, 36), filter_17951, *[_stypy_temp_lambda_26_17971, get_warning_msgs_call_result_17975], **kwargs_17976)
        
        # Assigning a type to the variable 'unreferenced_var_warnings' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'unreferenced_var_warnings', filter_call_result_17977)
        
        
        # Call to len(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'unreferenced_var_warnings' (line 167)
        unreferenced_var_warnings_17979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 15), 'unreferenced_var_warnings', False)
        # Processing the call keyword arguments (line 167)
        kwargs_17980 = {}
        # Getting the type of 'len' (line 167)
        len_17978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'len', False)
        # Calling len(args, kwargs) (line 167)
        len_call_result_17981 = invoke(stypy.reporting.localization.Localization(__file__, 167, 11), len_17978, *[unreferenced_var_warnings_17979], **kwargs_17980)
        
        int_17982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 44), 'int')
        # Applying the binary operator '>' (line 167)
        result_gt_17983 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 11), '>', len_call_result_17981, int_17982)
        
        # Testing if the type of an if condition is none (line 167)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 167, 8), result_gt_17983):
            pass
        else:
            
            # Testing the type of an if condition (line 167)
            if_condition_17984 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 8), result_gt_17983)
            # Assigning a type to the variable 'if_condition_17984' (line 167)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'if_condition_17984', if_condition_17984)
            # SSA begins for if statement (line 167)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 168):
            
            # Assigning a Call to a Name (line 168):
            
            # Call to TypeWarning(...): (line 168)
            # Processing the call arguments (line 168)
            # Getting the type of 'localization' (line 168)
            localization_17986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 30), 'localization', False)
            
            # Call to format(...): (line 169)
            # Processing the call arguments (line 169)
            # Getting the type of 'name' (line 169)
            name_17989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 99), 'name', False)
            # Processing the call keyword arguments (line 169)
            kwargs_17990 = {}
            str_17987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 30), 'str', "SyntaxWarning: name '{0}' is used before global declaration")
            # Obtaining the member 'format' of a type (line 169)
            format_17988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 30), str_17987, 'format')
            # Calling format(args, kwargs) (line 169)
            format_call_result_17991 = invoke(stypy.reporting.localization.Localization(__file__, 169, 30), format_17988, *[name_17989], **kwargs_17990)
            
            # Processing the call keyword arguments (line 168)
            kwargs_17992 = {}
            # Getting the type of 'TypeWarning' (line 168)
            TypeWarning_17985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 18), 'TypeWarning', False)
            # Calling TypeWarning(args, kwargs) (line 168)
            TypeWarning_call_result_17993 = invoke(stypy.reporting.localization.Localization(__file__, 168, 18), TypeWarning_17985, *[localization_17986, format_call_result_17991], **kwargs_17992)
            
            # Assigning a type to the variable 'ret' (line 168)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'ret', TypeWarning_call_result_17993)
            # SSA join for if statement (line 167)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Attribute to a Name (line 171):
        
        # Assigning a Attribute to a Name (line 171):
        
        # Call to get_context(...): (line 171)
        # Processing the call keyword arguments (line 171)
        kwargs_17996 = {}
        # Getting the type of 'self' (line 171)
        self_17994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 22), 'self', False)
        # Obtaining the member 'get_context' of a type (line 171)
        get_context_17995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 22), self_17994, 'get_context')
        # Calling get_context(args, kwargs) (line 171)
        get_context_call_result_17997 = invoke(stypy.reporting.localization.Localization(__file__, 171, 22), get_context_17995, *[], **kwargs_17996)
        
        # Obtaining the member 'global_vars' of a type (line 171)
        global_vars_17998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 22), get_context_call_result_17997, 'global_vars')
        # Assigning a type to the variable 'global_vars' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'global_vars', global_vars_17998)
        
        # Getting the type of 'name' (line 173)
        name_17999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 11), 'name')
        # Getting the type of 'global_vars' (line 173)
        global_vars_18000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 23), 'global_vars')
        # Applying the binary operator 'notin' (line 173)
        result_contains_18001 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 11), 'notin', name_17999, global_vars_18000)
        
        # Testing if the type of an if condition is none (line 173)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 173, 8), result_contains_18001):
            pass
        else:
            
            # Testing the type of an if condition (line 173)
            if_condition_18002 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 8), result_contains_18001)
            # Assigning a type to the variable 'if_condition_18002' (line 173)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'if_condition_18002', if_condition_18002)
            # SSA begins for if statement (line 173)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 174)
            # Processing the call arguments (line 174)
            # Getting the type of 'name' (line 174)
            name_18005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 31), 'name', False)
            # Processing the call keyword arguments (line 174)
            kwargs_18006 = {}
            # Getting the type of 'global_vars' (line 174)
            global_vars_18003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'global_vars', False)
            # Obtaining the member 'append' of a type (line 174)
            append_18004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 12), global_vars_18003, 'append')
            # Calling append(args, kwargs) (line 174)
            append_call_result_18007 = invoke(stypy.reporting.localization.Localization(__file__, 174, 12), append_18004, *[name_18005], **kwargs_18006)
            
            # SSA join for if statement (line 173)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'ret' (line 175)
        ret_18008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 15), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'stypy_return_type', ret_18008)
        
        # ################# End of 'mark_as_global(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mark_as_global' in the type store
        # Getting the type of 'stypy_return_type' (line 145)
        stypy_return_type_18009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18009)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mark_as_global'
        return stypy_return_type_18009


    @norecursion
    def get_global_context(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_global_context'
        module_type_store = module_type_store.open_function_context('get_global_context', 177, 4, False)
        # Assigning a type to the variable 'self' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.get_global_context.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.get_global_context.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.get_global_context.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.get_global_context.__dict__.__setitem__('stypy_function_name', 'TypeStore.get_global_context')
        TypeStore.get_global_context.__dict__.__setitem__('stypy_param_names_list', [])
        TypeStore.get_global_context.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.get_global_context.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.get_global_context.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.get_global_context.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.get_global_context.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.get_global_context.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.get_global_context', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_global_context', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_global_context(...)' code ##################

        str_18010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, (-1)), 'str', '\n        Gets the main function context, the last element in the context stack\n        :return:\n        ')
        
        # Obtaining the type of the subscript
        int_18011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 34), 'int')
        # Getting the type of 'self' (line 182)
        self_18012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'self')
        # Obtaining the member 'context_stack' of a type (line 182)
        context_stack_18013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 15), self_18012, 'context_stack')
        # Obtaining the member '__getitem__' of a type (line 182)
        getitem___18014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 15), context_stack_18013, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 182)
        subscript_call_result_18015 = invoke(stypy.reporting.localization.Localization(__file__, 182, 15), getitem___18014, int_18011)
        
        # Assigning a type to the variable 'stypy_return_type' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'stypy_return_type', subscript_call_result_18015)
        
        # ################# End of 'get_global_context(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_global_context' in the type store
        # Getting the type of 'stypy_return_type' (line 177)
        stypy_return_type_18016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18016)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_global_context'
        return stypy_return_type_18016


    @norecursion
    def get_type_of(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_type_of'
        module_type_store = module_type_store.open_function_context('get_type_of', 184, 4, False)
        # Assigning a type to the variable 'self' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.get_type_of.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.get_type_of.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.get_type_of.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.get_type_of.__dict__.__setitem__('stypy_function_name', 'TypeStore.get_type_of')
        TypeStore.get_type_of.__dict__.__setitem__('stypy_param_names_list', ['localization', 'name'])
        TypeStore.get_type_of.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.get_type_of.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.get_type_of.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.get_type_of.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.get_type_of.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.get_type_of.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.get_type_of', ['localization', 'name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_type_of', localization, ['localization', 'name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_type_of(...)' code ##################

        str_18017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, (-1)), 'str', '\n        Gets the type of the variable name, implemented the mentioned context-search semantics\n        :param localization: Caller information\n        :param name: Variable name\n        :return:\n        ')
        
        # Assigning a Call to a Name (line 191):
        
        # Assigning a Call to a Name (line 191):
        
        # Call to __get_type_of_from_function_context(...): (line 191)
        # Processing the call arguments (line 191)
        # Getting the type of 'localization' (line 191)
        localization_18020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 55), 'localization', False)
        # Getting the type of 'name' (line 191)
        name_18021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 69), 'name', False)
        
        # Call to get_context(...): (line 191)
        # Processing the call keyword arguments (line 191)
        kwargs_18024 = {}
        # Getting the type of 'self' (line 191)
        self_18022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 75), 'self', False)
        # Obtaining the member 'get_context' of a type (line 191)
        get_context_18023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 75), self_18022, 'get_context')
        # Calling get_context(args, kwargs) (line 191)
        get_context_call_result_18025 = invoke(stypy.reporting.localization.Localization(__file__, 191, 75), get_context_18023, *[], **kwargs_18024)
        
        # Processing the call keyword arguments (line 191)
        kwargs_18026 = {}
        # Getting the type of 'self' (line 191)
        self_18018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 14), 'self', False)
        # Obtaining the member '__get_type_of_from_function_context' of a type (line 191)
        get_type_of_from_function_context_18019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 14), self_18018, '__get_type_of_from_function_context')
        # Calling __get_type_of_from_function_context(args, kwargs) (line 191)
        get_type_of_from_function_context_call_result_18027 = invoke(stypy.reporting.localization.Localization(__file__, 191, 14), get_type_of_from_function_context_18019, *[localization_18020, name_18021, get_context_call_result_18025], **kwargs_18026)
        
        # Assigning a type to the variable 'ret' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'ret', get_type_of_from_function_context_call_result_18027)
        
        # Call to isinstance(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'ret' (line 194)
        ret_18029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 22), 'ret', False)
        # Getting the type of 'UndefinedTypeError' (line 194)
        UndefinedTypeError_18030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 27), 'UndefinedTypeError', False)
        # Processing the call keyword arguments (line 194)
        kwargs_18031 = {}
        # Getting the type of 'isinstance' (line 194)
        isinstance_18028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 194)
        isinstance_call_result_18032 = invoke(stypy.reporting.localization.Localization(__file__, 194, 11), isinstance_18028, *[ret_18029, UndefinedTypeError_18030], **kwargs_18031)
        
        # Testing if the type of an if condition is none (line 194)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 194, 8), isinstance_call_result_18032):
            pass
        else:
            
            # Testing the type of an if condition (line 194)
            if_condition_18033 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 8), isinstance_call_result_18032)
            # Assigning a type to the variable 'if_condition_18033' (line 194)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'if_condition_18033', if_condition_18033)
            # SSA begins for if statement (line 194)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 195):
            
            # Assigning a Call to a Name (line 195):
            
            # Call to import_from(...): (line 195)
            # Processing the call arguments (line 195)
            # Getting the type of 'localization' (line 195)
            localization_18036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 55), 'localization', False)
            # Getting the type of 'name' (line 195)
            name_18037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 69), 'name', False)
            # Processing the call keyword arguments (line 195)
            kwargs_18038 = {}
            # Getting the type of 'python_interface_copy' (line 195)
            python_interface_copy_18034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 21), 'python_interface_copy', False)
            # Obtaining the member 'import_from' of a type (line 195)
            import_from_18035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 21), python_interface_copy_18034, 'import_from')
            # Calling import_from(args, kwargs) (line 195)
            import_from_call_result_18039 = invoke(stypy.reporting.localization.Localization(__file__, 195, 21), import_from_18035, *[localization_18036, name_18037], **kwargs_18038)
            
            # Assigning a type to the variable 'member' (line 195)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'member', import_from_call_result_18039)
            
            # Type idiom detected: calculating its left and rigth part (line 197)
            # Getting the type of 'TypeError' (line 197)
            TypeError_18040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 34), 'TypeError')
            # Getting the type of 'member' (line 197)
            member_18041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 26), 'member')
            
            (may_be_18042, more_types_in_union_18043) = may_be_subtype(TypeError_18040, member_18041)

            if may_be_18042:

                if more_types_in_union_18043:
                    # Runtime conditional SSA (line 197)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'member' (line 197)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'member', remove_not_subtype_from_union(member_18041, TypeError))
                
                # Assigning a Call to a Attribute (line 198):
                
                # Assigning a Call to a Attribute (line 198):
                
                # Call to format(...): (line 198)
                # Processing the call arguments (line 198)
                # Getting the type of 'name' (line 199)
                name_18046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 49), 'name', False)
                # Processing the call keyword arguments (line 198)
                kwargs_18047 = {}
                str_18044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 29), 'str', "Could not find a definition for the name '{0}' in the current context. Are you missing an import?")
                # Obtaining the member 'format' of a type (line 198)
                format_18045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 29), str_18044, 'format')
                # Calling format(args, kwargs) (line 198)
                format_call_result_18048 = invoke(stypy.reporting.localization.Localization(__file__, 198, 29), format_18045, *[name_18046], **kwargs_18047)
                
                # Getting the type of 'member' (line 198)
                member_18049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'member')
                # Setting the type of the member 'msg' of a type (line 198)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 16), member_18049, 'msg', format_call_result_18048)

                if more_types_in_union_18043:
                    # SSA join for if statement (line 197)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Type idiom detected: calculating its left and rigth part (line 201)
            # Getting the type of 'member' (line 201)
            member_18050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'member')
            # Getting the type of 'None' (line 201)
            None_18051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 29), 'None')
            
            (may_be_18052, more_types_in_union_18053) = may_not_be_none(member_18050, None_18051)

            if may_be_18052:

                if more_types_in_union_18053:
                    # Runtime conditional SSA (line 201)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to remove_error_msg(...): (line 203)
                # Processing the call arguments (line 203)
                # Getting the type of 'ret' (line 203)
                ret_18056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 43), 'ret', False)
                # Processing the call keyword arguments (line 203)
                kwargs_18057 = {}
                # Getting the type of 'TypeError' (line 203)
                TypeError_18054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'TypeError', False)
                # Obtaining the member 'remove_error_msg' of a type (line 203)
                remove_error_msg_18055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 16), TypeError_18054, 'remove_error_msg')
                # Calling remove_error_msg(args, kwargs) (line 203)
                remove_error_msg_call_result_18058 = invoke(stypy.reporting.localization.Localization(__file__, 203, 16), remove_error_msg_18055, *[ret_18056], **kwargs_18057)
                
                # Getting the type of 'member' (line 206)
                member_18059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 23), 'member')
                # Assigning a type to the variable 'stypy_return_type' (line 206)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 16), 'stypy_return_type', member_18059)

                if more_types_in_union_18053:
                    # SSA join for if statement (line 201)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 194)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'ret' (line 209)
        ret_18060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 15), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'stypy_return_type', ret_18060)
        
        # ################# End of 'get_type_of(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_type_of' in the type store
        # Getting the type of 'stypy_return_type' (line 184)
        stypy_return_type_18061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18061)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_type_of'
        return stypy_return_type_18061


    @norecursion
    def get_context_of(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_context_of'
        module_type_store = module_type_store.open_function_context('get_context_of', 211, 4, False)
        # Assigning a type to the variable 'self' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.get_context_of.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.get_context_of.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.get_context_of.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.get_context_of.__dict__.__setitem__('stypy_function_name', 'TypeStore.get_context_of')
        TypeStore.get_context_of.__dict__.__setitem__('stypy_param_names_list', ['name'])
        TypeStore.get_context_of.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.get_context_of.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.get_context_of.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.get_context_of.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.get_context_of.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.get_context_of.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.get_context_of', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_context_of', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_context_of(...)' code ##################

        str_18062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, (-1)), 'str', '\n        Returns the function context in which a variable is first defined\n        :param name: Variable name\n        :return:\n        ')
        
        # Getting the type of 'self' (line 217)
        self_18063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 23), 'self')
        # Obtaining the member 'context_stack' of a type (line 217)
        context_stack_18064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 23), self_18063, 'context_stack')
        # Assigning a type to the variable 'context_stack_18064' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'context_stack_18064', context_stack_18064)
        # Testing if the for loop is going to be iterated (line 217)
        # Testing the type of a for loop iterable (line 217)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 217, 8), context_stack_18064)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 217, 8), context_stack_18064):
            # Getting the type of the for loop variable (line 217)
            for_loop_var_18065 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 217, 8), context_stack_18064)
            # Assigning a type to the variable 'context' (line 217)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'context', for_loop_var_18065)
            # SSA begins for a for statement (line 217)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'name' (line 218)
            name_18066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), 'name')
            # Getting the type of 'context' (line 218)
            context_18067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 23), 'context')
            # Applying the binary operator 'in' (line 218)
            result_contains_18068 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 15), 'in', name_18066, context_18067)
            
            # Testing if the type of an if condition is none (line 218)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 218, 12), result_contains_18068):
                pass
            else:
                
                # Testing the type of an if condition (line 218)
                if_condition_18069 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 218, 12), result_contains_18068)
                # Assigning a type to the variable 'if_condition_18069' (line 218)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'if_condition_18069', if_condition_18069)
                # SSA begins for if statement (line 218)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'context' (line 219)
                context_18070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 23), 'context')
                # Assigning a type to the variable 'stypy_return_type' (line 219)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'stypy_return_type', context_18070)
                # SSA join for if statement (line 218)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'None' (line 221)
        None_18071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'stypy_return_type', None_18071)
        
        # ################# End of 'get_context_of(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_context_of' in the type store
        # Getting the type of 'stypy_return_type' (line 211)
        stypy_return_type_18072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18072)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_context_of'
        return stypy_return_type_18072


    @norecursion
    def set_type_of(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_type_of'
        module_type_store = module_type_store.open_function_context('set_type_of', 223, 4, False)
        # Assigning a type to the variable 'self' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.set_type_of.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.set_type_of.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.set_type_of.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.set_type_of.__dict__.__setitem__('stypy_function_name', 'TypeStore.set_type_of')
        TypeStore.set_type_of.__dict__.__setitem__('stypy_param_names_list', ['localization', 'name', 'type_'])
        TypeStore.set_type_of.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.set_type_of.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.set_type_of.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.set_type_of.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.set_type_of.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.set_type_of.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.set_type_of', ['localization', 'name', 'type_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_type_of', localization, ['localization', 'name', 'type_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_type_of(...)' code ##################

        str_18073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, (-1)), 'str', '\n        Set the type of a variable using the context semantics previously mentioned.\n\n        Only simple a=b assignments are supported, as multiple assignments are solved by AST desugaring, so all of them\n        are converted to equivalent simple ones.\n        ')
        
        
        # Call to isinstance(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'type_' (line 230)
        type__18075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 26), 'type_', False)
        # Getting the type of 'type_inference_proxy_copy' (line 230)
        type_inference_proxy_copy_18076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 33), 'type_inference_proxy_copy', False)
        # Obtaining the member 'Type' of a type (line 230)
        Type_18077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 33), type_inference_proxy_copy_18076, 'Type')
        # Processing the call keyword arguments (line 230)
        kwargs_18078 = {}
        # Getting the type of 'isinstance' (line 230)
        isinstance_18074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 230)
        isinstance_call_result_18079 = invoke(stypy.reporting.localization.Localization(__file__, 230, 15), isinstance_18074, *[type__18075, Type_18077], **kwargs_18078)
        
        # Applying the 'not' unary operator (line 230)
        result_not__18080 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 11), 'not', isinstance_call_result_18079)
        
        # Testing if the type of an if condition is none (line 230)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 230, 8), result_not__18080):
            pass
        else:
            
            # Testing the type of an if condition (line 230)
            if_condition_18081 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 8), result_not__18080)
            # Assigning a type to the variable 'if_condition_18081' (line 230)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'if_condition_18081', if_condition_18081)
            # SSA begins for if statement (line 230)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 231):
            
            # Assigning a Call to a Name (line 231):
            
            # Call to instance(...): (line 231)
            # Processing the call arguments (line 231)
            # Getting the type of 'type_' (line 231)
            type__18085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 74), 'type_', False)
            # Processing the call keyword arguments (line 231)
            kwargs_18086 = {}
            # Getting the type of 'type_inference_proxy_copy' (line 231)
            type_inference_proxy_copy_18082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'type_inference_proxy_copy', False)
            # Obtaining the member 'TypeInferenceProxy' of a type (line 231)
            TypeInferenceProxy_18083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 20), type_inference_proxy_copy_18082, 'TypeInferenceProxy')
            # Obtaining the member 'instance' of a type (line 231)
            instance_18084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 20), TypeInferenceProxy_18083, 'instance')
            # Calling instance(args, kwargs) (line 231)
            instance_call_result_18087 = invoke(stypy.reporting.localization.Localization(__file__, 231, 20), instance_18084, *[type__18085], **kwargs_18086)
            
            # Assigning a type to the variable 'type_' (line 231)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'type_', instance_call_result_18087)
            # SSA join for if statement (line 230)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Attribute (line 233):
        
        # Assigning a Call to a Attribute (line 233):
        
        # Call to get_instance_for_file(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'self' (line 233)
        self_18090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 77), 'self', False)
        # Obtaining the member 'program_name' of a type (line 233)
        program_name_18091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 77), self_18090, 'program_name')
        # Processing the call keyword arguments (line 233)
        kwargs_18092 = {}
        # Getting the type of 'TypeAnnotationRecord' (line 233)
        TypeAnnotationRecord_18088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 34), 'TypeAnnotationRecord', False)
        # Obtaining the member 'get_instance_for_file' of a type (line 233)
        get_instance_for_file_18089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 34), TypeAnnotationRecord_18088, 'get_instance_for_file')
        # Calling get_instance_for_file(args, kwargs) (line 233)
        get_instance_for_file_call_result_18093 = invoke(stypy.reporting.localization.Localization(__file__, 233, 34), get_instance_for_file_18089, *[program_name_18091], **kwargs_18092)
        
        # Getting the type of 'type_' (line 233)
        type__18094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'type_')
        # Setting the type of the member 'annotation_record' of a type (line 233)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 8), type__18094, 'annotation_record', get_instance_for_file_call_result_18093)
        
        # Call to __set_type_of(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'localization' (line 235)
        localization_18097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 34), 'localization', False)
        # Getting the type of 'name' (line 235)
        name_18098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 48), 'name', False)
        # Getting the type of 'type_' (line 235)
        type__18099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 54), 'type_', False)
        # Processing the call keyword arguments (line 235)
        kwargs_18100 = {}
        # Getting the type of 'self' (line 235)
        self_18095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 15), 'self', False)
        # Obtaining the member '__set_type_of' of a type (line 235)
        set_type_of_18096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 15), self_18095, '__set_type_of')
        # Calling __set_type_of(args, kwargs) (line 235)
        set_type_of_call_result_18101 = invoke(stypy.reporting.localization.Localization(__file__, 235, 15), set_type_of_18096, *[localization_18097, name_18098, type__18099], **kwargs_18100)
        
        # Assigning a type to the variable 'stypy_return_type' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'stypy_return_type', set_type_of_call_result_18101)
        
        # ################# End of 'set_type_of(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_type_of' in the type store
        # Getting the type of 'stypy_return_type' (line 223)
        stypy_return_type_18102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18102)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_type_of'
        return stypy_return_type_18102


    @norecursion
    def set_type_store(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 237)
        False_18103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 47), 'False')
        defaults = [False_18103]
        # Create a new context for function 'set_type_store'
        module_type_store = module_type_store.open_function_context('set_type_store', 237, 4, False)
        # Assigning a type to the variable 'self' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.set_type_store.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.set_type_store.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.set_type_store.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.set_type_store.__dict__.__setitem__('stypy_function_name', 'TypeStore.set_type_store')
        TypeStore.set_type_store.__dict__.__setitem__('stypy_param_names_list', ['type_store', 'clone'])
        TypeStore.set_type_store.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.set_type_store.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.set_type_store.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.set_type_store.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.set_type_store.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.set_type_store.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.set_type_store', ['type_store', 'clone'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_type_store', localization, ['type_store', 'clone'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_type_store(...)' code ##################

        str_18104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, (-1)), 'str', '\n        Assign to this type store the attributes of the passed type store, cloning the passed\n        type store if indicated. This operation is needed to implement the SSA algorithm\n        :param type_store: Type store to assign to this one\n        :param clone: Clone the passed type store before assigning its values\n        :return:\n        ')
        # Getting the type of 'clone' (line 245)
        clone_18105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 11), 'clone')
        # Testing if the type of an if condition is none (line 245)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 245, 8), clone_18105):
            pass
        else:
            
            # Testing the type of an if condition (line 245)
            if_condition_18106 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 245, 8), clone_18105)
            # Assigning a type to the variable 'if_condition_18106' (line 245)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'if_condition_18106', if_condition_18106)
            # SSA begins for if statement (line 245)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 246):
            
            # Assigning a Call to a Name (line 246):
            
            # Call to __clone_type_store(...): (line 246)
            # Processing the call arguments (line 246)
            # Getting the type of 'type_store' (line 246)
            type_store_18109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 54), 'type_store', False)
            # Processing the call keyword arguments (line 246)
            kwargs_18110 = {}
            # Getting the type of 'TypeStore' (line 246)
            TypeStore_18107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 25), 'TypeStore', False)
            # Obtaining the member '__clone_type_store' of a type (line 246)
            clone_type_store_18108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 25), TypeStore_18107, '__clone_type_store')
            # Calling __clone_type_store(args, kwargs) (line 246)
            clone_type_store_call_result_18111 = invoke(stypy.reporting.localization.Localization(__file__, 246, 25), clone_type_store_18108, *[type_store_18109], **kwargs_18110)
            
            # Assigning a type to the variable 'type_store' (line 246)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'type_store', clone_type_store_call_result_18111)
            # SSA join for if statement (line 245)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Attribute to a Attribute (line 248):
        
        # Assigning a Attribute to a Attribute (line 248):
        # Getting the type of 'type_store' (line 248)
        type_store_18112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 28), 'type_store')
        # Obtaining the member 'program_name' of a type (line 248)
        program_name_18113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 28), type_store_18112, 'program_name')
        # Getting the type of 'self' (line 248)
        self_18114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'self')
        # Setting the type of the member 'program_name' of a type (line 248)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), self_18114, 'program_name', program_name_18113)
        
        # Assigning a Attribute to a Attribute (line 249):
        
        # Assigning a Attribute to a Attribute (line 249):
        # Getting the type of 'type_store' (line 249)
        type_store_18115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 29), 'type_store')
        # Obtaining the member 'context_stack' of a type (line 249)
        context_stack_18116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 29), type_store_18115, 'context_stack')
        # Getting the type of 'self' (line 249)
        self_18117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'self')
        # Setting the type of the member 'context_stack' of a type (line 249)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), self_18117, 'context_stack', context_stack_18116)
        
        # Assigning a Attribute to a Attribute (line 250):
        
        # Assigning a Attribute to a Attribute (line 250):
        # Getting the type of 'type_store' (line 250)
        type_store_18118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 38), 'type_store')
        # Obtaining the member 'last_function_contexts' of a type (line 250)
        last_function_contexts_18119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 38), type_store_18118, 'last_function_contexts')
        # Getting the type of 'self' (line 250)
        self_18120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'self')
        # Setting the type of the member 'last_function_contexts' of a type (line 250)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), self_18120, 'last_function_contexts', last_function_contexts_18119)
        
        # Assigning a Attribute to a Attribute (line 251):
        
        # Assigning a Attribute to a Attribute (line 251):
        # Getting the type of 'type_store' (line 251)
        type_store_18121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 32), 'type_store')
        # Obtaining the member 'external_modules' of a type (line 251)
        external_modules_18122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 32), type_store_18121, 'external_modules')
        # Getting the type of 'self' (line 251)
        self_18123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'self')
        # Setting the type of the member 'external_modules' of a type (line 251)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), self_18123, 'external_modules', external_modules_18122)
        
        # Assigning a Attribute to a Attribute (line 252):
        
        # Assigning a Attribute to a Attribute (line 252):
        # Getting the type of 'type_store' (line 252)
        type_store_18124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 37), 'type_store')
        # Obtaining the member 'test_unreferenced_var' of a type (line 252)
        test_unreferenced_var_18125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 37), type_store_18124, 'test_unreferenced_var')
        # Getting the type of 'self' (line 252)
        self_18126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'self')
        # Setting the type of the member 'test_unreferenced_var' of a type (line 252)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 8), self_18126, 'test_unreferenced_var', test_unreferenced_var_18125)
        
        # ################# End of 'set_type_store(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_type_store' in the type store
        # Getting the type of 'stypy_return_type' (line 237)
        stypy_return_type_18127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18127)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_type_store'
        return stypy_return_type_18127


    @norecursion
    def clone_type_store(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'clone_type_store'
        module_type_store = module_type_store.open_function_context('clone_type_store', 254, 4, False)
        # Assigning a type to the variable 'self' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.clone_type_store.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.clone_type_store.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.clone_type_store.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.clone_type_store.__dict__.__setitem__('stypy_function_name', 'TypeStore.clone_type_store')
        TypeStore.clone_type_store.__dict__.__setitem__('stypy_param_names_list', [])
        TypeStore.clone_type_store.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.clone_type_store.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.clone_type_store.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.clone_type_store.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.clone_type_store.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.clone_type_store.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.clone_type_store', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'clone_type_store', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'clone_type_store(...)' code ##################

        str_18128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, (-1)), 'str', '\n        Clone this type store\n        :return: A clone of this type store\n        ')
        
        # Call to __clone_type_store(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'self' (line 259)
        self_18131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 44), 'self', False)
        # Processing the call keyword arguments (line 259)
        kwargs_18132 = {}
        # Getting the type of 'TypeStore' (line 259)
        TypeStore_18129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 15), 'TypeStore', False)
        # Obtaining the member '__clone_type_store' of a type (line 259)
        clone_type_store_18130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 15), TypeStore_18129, '__clone_type_store')
        # Calling __clone_type_store(args, kwargs) (line 259)
        clone_type_store_call_result_18133 = invoke(stypy.reporting.localization.Localization(__file__, 259, 15), clone_type_store_18130, *[self_18131], **kwargs_18132)
        
        # Assigning a type to the variable 'stypy_return_type' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'stypy_return_type', clone_type_store_call_result_18133)
        
        # ################# End of 'clone_type_store(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clone_type_store' in the type store
        # Getting the type of 'stypy_return_type' (line 254)
        stypy_return_type_18134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18134)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clone_type_store'
        return stypy_return_type_18134


    @norecursion
    def get_public_names_and_types(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_public_names_and_types'
        module_type_store = module_type_store.open_function_context('get_public_names_and_types', 261, 4, False)
        # Assigning a type to the variable 'self' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.get_public_names_and_types.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.get_public_names_and_types.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.get_public_names_and_types.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.get_public_names_and_types.__dict__.__setitem__('stypy_function_name', 'TypeStore.get_public_names_and_types')
        TypeStore.get_public_names_and_types.__dict__.__setitem__('stypy_param_names_list', [])
        TypeStore.get_public_names_and_types.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.get_public_names_and_types.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.get_public_names_and_types.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.get_public_names_and_types.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.get_public_names_and_types.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.get_public_names_and_types.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.get_public_names_and_types', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_public_names_and_types', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_public_names_and_types(...)' code ##################

        str_18135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, (-1)), 'str', '\n        Gets all the public variables within this type store function contexts and its types\n        in a {name: type} dictionary\n        :return: {name: type} dictionary\n        ')
        
        # Assigning a Dict to a Name (line 267):
        
        # Assigning a Dict to a Name (line 267):
        
        # Obtaining an instance of the builtin type 'dict' (line 267)
        dict_18136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 25), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 267)
        
        # Assigning a type to the variable 'name_type_dict' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'name_type_dict', dict_18136)
        
        # Assigning a BinOp to a Name (line 268):
        
        # Assigning a BinOp to a Name (line 268):
        
        # Call to len(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'self' (line 268)
        self_18138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 19), 'self', False)
        # Obtaining the member 'context_stack' of a type (line 268)
        context_stack_18139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 19), self_18138, 'context_stack')
        # Processing the call keyword arguments (line 268)
        kwargs_18140 = {}
        # Getting the type of 'len' (line 268)
        len_18137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 15), 'len', False)
        # Calling len(args, kwargs) (line 268)
        len_call_result_18141 = invoke(stypy.reporting.localization.Localization(__file__, 268, 15), len_18137, *[context_stack_18139], **kwargs_18140)
        
        int_18142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 41), 'int')
        # Applying the binary operator '-' (line 268)
        result_sub_18143 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 15), '-', len_call_result_18141, int_18142)
        
        # Assigning a type to the variable 'cont' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'cont', result_sub_18143)
        
        
        # Call to range(...): (line 271)
        # Processing the call arguments (line 271)
        
        # Call to len(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'self' (line 271)
        self_18146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 27), 'self', False)
        # Obtaining the member 'context_stack' of a type (line 271)
        context_stack_18147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 27), self_18146, 'context_stack')
        # Processing the call keyword arguments (line 271)
        kwargs_18148 = {}
        # Getting the type of 'len' (line 271)
        len_18145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 23), 'len', False)
        # Calling len(args, kwargs) (line 271)
        len_call_result_18149 = invoke(stypy.reporting.localization.Localization(__file__, 271, 23), len_18145, *[context_stack_18147], **kwargs_18148)
        
        # Processing the call keyword arguments (line 271)
        kwargs_18150 = {}
        # Getting the type of 'range' (line 271)
        range_18144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 17), 'range', False)
        # Calling range(args, kwargs) (line 271)
        range_call_result_18151 = invoke(stypy.reporting.localization.Localization(__file__, 271, 17), range_18144, *[len_call_result_18149], **kwargs_18150)
        
        # Assigning a type to the variable 'range_call_result_18151' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'range_call_result_18151', range_call_result_18151)
        # Testing if the for loop is going to be iterated (line 271)
        # Testing the type of a for loop iterable (line 271)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 271, 8), range_call_result_18151)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 271, 8), range_call_result_18151):
            # Getting the type of the for loop variable (line 271)
            for_loop_var_18152 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 271, 8), range_call_result_18151)
            # Assigning a type to the variable 'i' (line 271)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'i', for_loop_var_18152)
            # SSA begins for a for statement (line 271)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Name (line 272):
            
            # Assigning a Subscript to a Name (line 272):
            
            # Obtaining the type of the subscript
            # Getting the type of 'cont' (line 272)
            cont_18153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 37), 'cont')
            # Getting the type of 'self' (line 272)
            self_18154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 18), 'self')
            # Obtaining the member 'context_stack' of a type (line 272)
            context_stack_18155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 18), self_18154, 'context_stack')
            # Obtaining the member '__getitem__' of a type (line 272)
            getitem___18156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 18), context_stack_18155, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 272)
            subscript_call_result_18157 = invoke(stypy.reporting.localization.Localization(__file__, 272, 18), getitem___18156, cont_18153)
            
            # Assigning a type to the variable 'ctx' (line 272)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'ctx', subscript_call_result_18157)
            
            # Getting the type of 'ctx' (line 274)
            ctx_18158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 24), 'ctx')
            # Obtaining the member 'types_of' of a type (line 274)
            types_of_18159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 24), ctx_18158, 'types_of')
            # Assigning a type to the variable 'types_of_18159' (line 274)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'types_of_18159', types_of_18159)
            # Testing if the for loop is going to be iterated (line 274)
            # Testing the type of a for loop iterable (line 274)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 274, 12), types_of_18159)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 274, 12), types_of_18159):
                # Getting the type of the for loop variable (line 274)
                for_loop_var_18160 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 274, 12), types_of_18159)
                # Assigning a type to the variable 'name' (line 274)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'name', for_loop_var_18160)
                # SSA begins for a for statement (line 274)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to startswith(...): (line 275)
                # Processing the call arguments (line 275)
                str_18163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 35), 'str', '__')
                # Processing the call keyword arguments (line 275)
                kwargs_18164 = {}
                # Getting the type of 'name' (line 275)
                name_18161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 19), 'name', False)
                # Obtaining the member 'startswith' of a type (line 275)
                startswith_18162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 19), name_18161, 'startswith')
                # Calling startswith(args, kwargs) (line 275)
                startswith_call_result_18165 = invoke(stypy.reporting.localization.Localization(__file__, 275, 19), startswith_18162, *[str_18163], **kwargs_18164)
                
                # Testing if the type of an if condition is none (line 275)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 275, 16), startswith_call_result_18165):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 275)
                    if_condition_18166 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 16), startswith_call_result_18165)
                    # Assigning a type to the variable 'if_condition_18166' (line 275)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'if_condition_18166', if_condition_18166)
                    # SSA begins for if statement (line 275)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # SSA join for if statement (line 275)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Subscript to a Subscript (line 277):
                
                # Assigning a Subscript to a Subscript (line 277):
                
                # Obtaining the type of the subscript
                # Getting the type of 'name' (line 277)
                name_18167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 52), 'name')
                # Getting the type of 'ctx' (line 277)
                ctx_18168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 39), 'ctx')
                # Obtaining the member 'types_of' of a type (line 277)
                types_of_18169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 39), ctx_18168, 'types_of')
                # Obtaining the member '__getitem__' of a type (line 277)
                getitem___18170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 39), types_of_18169, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 277)
                subscript_call_result_18171 = invoke(stypy.reporting.localization.Localization(__file__, 277, 39), getitem___18170, name_18167)
                
                # Getting the type of 'name_type_dict' (line 277)
                name_type_dict_18172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 16), 'name_type_dict')
                # Getting the type of 'name' (line 277)
                name_18173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 31), 'name')
                # Storing an element on a container (line 277)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 16), name_type_dict_18172, (name_18173, subscript_call_result_18171))
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Getting the type of 'cont' (line 279)
            cont_18174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'cont')
            int_18175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 20), 'int')
            # Applying the binary operator '-=' (line 279)
            result_isub_18176 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 12), '-=', cont_18174, int_18175)
            # Assigning a type to the variable 'cont' (line 279)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'cont', result_isub_18176)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'name_type_dict' (line 281)
        name_type_dict_18177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 15), 'name_type_dict')
        # Assigning a type to the variable 'stypy_return_type' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'stypy_return_type', name_type_dict_18177)
        
        # ################# End of 'get_public_names_and_types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_public_names_and_types' in the type store
        # Getting the type of 'stypy_return_type' (line 261)
        stypy_return_type_18178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18178)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_public_names_and_types'
        return stypy_return_type_18178


    @norecursion
    def get_last_function_context_for(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_last_function_context_for'
        module_type_store = module_type_store.open_function_context('get_last_function_context_for', 283, 4, False)
        # Assigning a type to the variable 'self' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.get_last_function_context_for.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.get_last_function_context_for.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.get_last_function_context_for.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.get_last_function_context_for.__dict__.__setitem__('stypy_function_name', 'TypeStore.get_last_function_context_for')
        TypeStore.get_last_function_context_for.__dict__.__setitem__('stypy_param_names_list', ['context_name'])
        TypeStore.get_last_function_context_for.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.get_last_function_context_for.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.get_last_function_context_for.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.get_last_function_context_for.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.get_last_function_context_for.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.get_last_function_context_for.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.get_last_function_context_for', ['context_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_last_function_context_for', localization, ['context_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_last_function_context_for(...)' code ##################

        str_18179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, (-1)), 'str', '\n        Gets the last used function context whose name is the one passed to this function\n        :param context_name: Context name to search\n        :return: Function context\n        ')
        
        # Assigning a Name to a Name (line 289):
        
        # Assigning a Name to a Name (line 289):
        # Getting the type of 'None' (line 289)
        None_18180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 18), 'None')
        # Assigning a type to the variable 'context' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'context', None_18180)
        
        # Getting the type of 'self' (line 291)
        self_18181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 28), 'self')
        # Obtaining the member 'last_function_contexts' of a type (line 291)
        last_function_contexts_18182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 28), self_18181, 'last_function_contexts')
        # Assigning a type to the variable 'last_function_contexts_18182' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'last_function_contexts_18182', last_function_contexts_18182)
        # Testing if the for loop is going to be iterated (line 291)
        # Testing the type of a for loop iterable (line 291)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 291, 8), last_function_contexts_18182)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 291, 8), last_function_contexts_18182):
            # Getting the type of the for loop variable (line 291)
            for_loop_var_18183 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 291, 8), last_function_contexts_18182)
            # Assigning a type to the variable 'last_context' (line 291)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'last_context', for_loop_var_18183)
            # SSA begins for a for statement (line 291)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'last_context' (line 292)
            last_context_18184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 15), 'last_context')
            # Obtaining the member 'function_name' of a type (line 292)
            function_name_18185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 15), last_context_18184, 'function_name')
            # Getting the type of 'context_name' (line 292)
            context_name_18186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 45), 'context_name')
            # Applying the binary operator '==' (line 292)
            result_eq_18187 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 15), '==', function_name_18185, context_name_18186)
            
            # Testing if the type of an if condition is none (line 292)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 292, 12), result_eq_18187):
                pass
            else:
                
                # Testing the type of an if condition (line 292)
                if_condition_18188 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 12), result_eq_18187)
                # Assigning a type to the variable 'if_condition_18188' (line 292)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'if_condition_18188', if_condition_18188)
                # SSA begins for if statement (line 292)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 293):
                
                # Assigning a Name to a Name (line 293):
                # Getting the type of 'last_context' (line 293)
                last_context_18189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 26), 'last_context')
                # Assigning a type to the variable 'context' (line 293)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 16), 'context', last_context_18189)
                # SSA join for if statement (line 292)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Type idiom detected: calculating its left and rigth part (line 295)
        # Getting the type of 'context' (line 295)
        context_18190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 11), 'context')
        # Getting the type of 'None' (line 295)
        None_18191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 22), 'None')
        
        (may_be_18192, more_types_in_union_18193) = may_be_none(context_18190, None_18191)

        if may_be_18192:

            if more_types_in_union_18193:
                # Runtime conditional SSA (line 295)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 'self' (line 296)
            self_18194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 27), 'self')
            # Obtaining the member 'context_stack' of a type (line 296)
            context_stack_18195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 27), self_18194, 'context_stack')
            # Assigning a type to the variable 'context_stack_18195' (line 296)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'context_stack_18195', context_stack_18195)
            # Testing if the for loop is going to be iterated (line 296)
            # Testing the type of a for loop iterable (line 296)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 296, 12), context_stack_18195)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 296, 12), context_stack_18195):
                # Getting the type of the for loop variable (line 296)
                for_loop_var_18196 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 296, 12), context_stack_18195)
                # Assigning a type to the variable 'context' (line 296)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'context', for_loop_var_18196)
                # SSA begins for a for statement (line 296)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'context_name' (line 297)
                context_name_18197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 19), 'context_name')
                # Getting the type of 'context' (line 297)
                context_18198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 35), 'context')
                # Obtaining the member 'function_name' of a type (line 297)
                function_name_18199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 35), context_18198, 'function_name')
                # Applying the binary operator '==' (line 297)
                result_eq_18200 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 19), '==', context_name_18197, function_name_18199)
                
                # Testing if the type of an if condition is none (line 297)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 297, 16), result_eq_18200):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 297)
                    if_condition_18201 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 297, 16), result_eq_18200)
                    # Assigning a type to the variable 'if_condition_18201' (line 297)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 16), 'if_condition_18201', if_condition_18201)
                    # SSA begins for if statement (line 297)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'context' (line 298)
                    context_18202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 27), 'context')
                    # Assigning a type to the variable 'stypy_return_type' (line 298)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 20), 'stypy_return_type', context_18202)
                    # SSA join for if statement (line 297)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            

            if more_types_in_union_18193:
                # SSA join for if statement (line 295)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'context' (line 300)
        context_18203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 15), 'context')
        # Assigning a type to the variable 'stypy_return_type' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'stypy_return_type', context_18203)
        
        # ################# End of 'get_last_function_context_for(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_last_function_context_for' in the type store
        # Getting the type of 'stypy_return_type' (line 283)
        stypy_return_type_18204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18204)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_last_function_context_for'
        return stypy_return_type_18204


    @norecursion
    def add_alias(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_alias'
        module_type_store = module_type_store.open_function_context('add_alias', 302, 4, False)
        # Assigning a type to the variable 'self' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.add_alias.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.add_alias.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.add_alias.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.add_alias.__dict__.__setitem__('stypy_function_name', 'TypeStore.add_alias')
        TypeStore.add_alias.__dict__.__setitem__('stypy_param_names_list', ['alias', 'member_name'])
        TypeStore.add_alias.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.add_alias.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.add_alias.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.add_alias.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.add_alias.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.add_alias.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.add_alias', ['alias', 'member_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_alias', localization, ['alias', 'member_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_alias(...)' code ##################

        str_18205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, (-1)), 'str', '\n        Adds an alias to the current function context\n        :param alias: Alias name\n        :param member_name: Aliased variable name\n        :return:\n        ')
        
        # Call to add_alias(...): (line 309)
        # Processing the call arguments (line 309)
        # Getting the type of 'alias' (line 309)
        alias_18211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 37), 'alias', False)
        # Getting the type of 'member_name' (line 309)
        member_name_18212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 44), 'member_name', False)
        # Processing the call keyword arguments (line 309)
        kwargs_18213 = {}
        
        # Call to get_context(...): (line 309)
        # Processing the call keyword arguments (line 309)
        kwargs_18208 = {}
        # Getting the type of 'self' (line 309)
        self_18206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'self', False)
        # Obtaining the member 'get_context' of a type (line 309)
        get_context_18207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 8), self_18206, 'get_context')
        # Calling get_context(args, kwargs) (line 309)
        get_context_call_result_18209 = invoke(stypy.reporting.localization.Localization(__file__, 309, 8), get_context_18207, *[], **kwargs_18208)
        
        # Obtaining the member 'add_alias' of a type (line 309)
        add_alias_18210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 8), get_context_call_result_18209, 'add_alias')
        # Calling add_alias(args, kwargs) (line 309)
        add_alias_call_result_18214 = invoke(stypy.reporting.localization.Localization(__file__, 309, 8), add_alias_18210, *[alias_18211, member_name_18212], **kwargs_18213)
        
        
        # ################# End of 'add_alias(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_alias' in the type store
        # Getting the type of 'stypy_return_type' (line 302)
        stypy_return_type_18215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18215)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_alias'
        return stypy_return_type_18215


    @norecursion
    def del_type_of(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'del_type_of'
        module_type_store = module_type_store.open_function_context('del_type_of', 311, 4, False)
        # Assigning a type to the variable 'self' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.del_type_of.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.del_type_of.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.del_type_of.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.del_type_of.__dict__.__setitem__('stypy_function_name', 'TypeStore.del_type_of')
        TypeStore.del_type_of.__dict__.__setitem__('stypy_param_names_list', ['localization', 'name'])
        TypeStore.del_type_of.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.del_type_of.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.del_type_of.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.del_type_of.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.del_type_of.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.del_type_of.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.del_type_of', ['localization', 'name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'del_type_of', localization, ['localization', 'name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'del_type_of(...)' code ##################

        str_18216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, (-1)), 'str', '\n        Delete a variable for the first function context that defines it (using the context\n        search semantics we mentioned)\n        :param localization:\n        :param name:\n        :return:\n        ')
        
        # Assigning a Call to a Name (line 319):
        
        # Assigning a Call to a Name (line 319):
        
        # Call to __del_type_of_from_function_context(...): (line 319)
        # Processing the call arguments (line 319)
        # Getting the type of 'localization' (line 319)
        localization_18219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 55), 'localization', False)
        # Getting the type of 'name' (line 319)
        name_18220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 69), 'name', False)
        
        # Call to get_context(...): (line 319)
        # Processing the call keyword arguments (line 319)
        kwargs_18223 = {}
        # Getting the type of 'self' (line 319)
        self_18221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 75), 'self', False)
        # Obtaining the member 'get_context' of a type (line 319)
        get_context_18222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 75), self_18221, 'get_context')
        # Calling get_context(args, kwargs) (line 319)
        get_context_call_result_18224 = invoke(stypy.reporting.localization.Localization(__file__, 319, 75), get_context_18222, *[], **kwargs_18223)
        
        # Processing the call keyword arguments (line 319)
        kwargs_18225 = {}
        # Getting the type of 'self' (line 319)
        self_18217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 14), 'self', False)
        # Obtaining the member '__del_type_of_from_function_context' of a type (line 319)
        del_type_of_from_function_context_18218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 14), self_18217, '__del_type_of_from_function_context')
        # Calling __del_type_of_from_function_context(args, kwargs) (line 319)
        del_type_of_from_function_context_call_result_18226 = invoke(stypy.reporting.localization.Localization(__file__, 319, 14), del_type_of_from_function_context_18218, *[localization_18219, name_18220, get_context_call_result_18224], **kwargs_18225)
        
        # Assigning a type to the variable 'ret' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'ret', del_type_of_from_function_context_call_result_18226)
        # Getting the type of 'ret' (line 321)
        ret_18227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 15), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'stypy_return_type', ret_18227)
        
        # ################# End of 'del_type_of(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'del_type_of' in the type store
        # Getting the type of 'stypy_return_type' (line 311)
        stypy_return_type_18228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18228)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'del_type_of'
        return stypy_return_type_18228


    @norecursion
    def store_return_type_of_current_context(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'store_return_type_of_current_context'
        module_type_store = module_type_store.open_function_context('store_return_type_of_current_context', 323, 4, False)
        # Assigning a type to the variable 'self' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.store_return_type_of_current_context.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.store_return_type_of_current_context.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.store_return_type_of_current_context.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.store_return_type_of_current_context.__dict__.__setitem__('stypy_function_name', 'TypeStore.store_return_type_of_current_context')
        TypeStore.store_return_type_of_current_context.__dict__.__setitem__('stypy_param_names_list', ['return_type'])
        TypeStore.store_return_type_of_current_context.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.store_return_type_of_current_context.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.store_return_type_of_current_context.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.store_return_type_of_current_context.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.store_return_type_of_current_context.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.store_return_type_of_current_context.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.store_return_type_of_current_context', ['return_type'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'store_return_type_of_current_context', localization, ['return_type'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'store_return_type_of_current_context(...)' code ##################

        str_18229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, (-1)), 'str', '\n        Changes the return type of the current function context\n        :param return_type: Type\n        :return:\n        ')
        
        # Assigning a Name to a Attribute (line 329):
        
        # Assigning a Name to a Attribute (line 329):
        # Getting the type of 'return_type' (line 329)
        return_type_18230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 41), 'return_type')
        
        # Call to get_context(...): (line 329)
        # Processing the call keyword arguments (line 329)
        kwargs_18233 = {}
        # Getting the type of 'self' (line 329)
        self_18231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'self', False)
        # Obtaining the member 'get_context' of a type (line 329)
        get_context_18232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), self_18231, 'get_context')
        # Calling get_context(args, kwargs) (line 329)
        get_context_call_result_18234 = invoke(stypy.reporting.localization.Localization(__file__, 329, 8), get_context_18232, *[], **kwargs_18233)
        
        # Setting the type of the member 'return_type' of a type (line 329)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), get_context_call_result_18234, 'return_type', return_type_18230)
        
        # ################# End of 'store_return_type_of_current_context(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'store_return_type_of_current_context' in the type store
        # Getting the type of 'stypy_return_type' (line 323)
        stypy_return_type_18235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18235)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'store_return_type_of_current_context'
        return stypy_return_type_18235


    @norecursion
    def __get_type_of_from_function_context(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__get_type_of_from_function_context'
        module_type_store = module_type_store.open_function_context('__get_type_of_from_function_context', 333, 4, False)
        # Assigning a type to the variable 'self' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.__get_type_of_from_function_context.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.__get_type_of_from_function_context.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.__get_type_of_from_function_context.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.__get_type_of_from_function_context.__dict__.__setitem__('stypy_function_name', 'TypeStore.__get_type_of_from_function_context')
        TypeStore.__get_type_of_from_function_context.__dict__.__setitem__('stypy_param_names_list', ['localization', 'name', 'f_context'])
        TypeStore.__get_type_of_from_function_context.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.__get_type_of_from_function_context.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.__get_type_of_from_function_context.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.__get_type_of_from_function_context.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.__get_type_of_from_function_context.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.__get_type_of_from_function_context.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.__get_type_of_from_function_context', ['localization', 'name', 'f_context'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__get_type_of_from_function_context', localization, ['localization', 'name', 'f_context'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__get_type_of_from_function_context(...)' code ##################

        str_18236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, (-1)), 'str', '\n        Search the stored function contexts for the type associated to a name.\n        As we follows the program flow, a correct program ensures that if this query is performed the name actually HAS\n        a type (it has been assigned a value previously in the previous executed statements). If the name is not found,\n         we have detected a programmer error within the source file (usage of a previously undeclared name). The\n         method is orthogonal to variables and functions.\n        :param name: Name of the element whose type we want to know\n        :return:\n        ')
        
        # Assigning a Name to a Name (line 345):
        
        # Assigning a Name to a Name (line 345):
        # Getting the type of 'f_context' (line 345)
        f_context_18237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 26), 'f_context')
        # Assigning a type to the variable 'current_context' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'current_context', f_context_18237)
        
        # Assigning a Call to a Name (line 347):
        
        # Assigning a Call to a Name (line 347):
        
        # Call to get_global_context(...): (line 347)
        # Processing the call keyword arguments (line 347)
        kwargs_18240 = {}
        # Getting the type of 'self' (line 347)
        self_18238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 25), 'self', False)
        # Obtaining the member 'get_global_context' of a type (line 347)
        get_global_context_18239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 25), self_18238, 'get_global_context')
        # Calling get_global_context(args, kwargs) (line 347)
        get_global_context_call_result_18241 = invoke(stypy.reporting.localization.Localization(__file__, 347, 25), get_global_context_18239, *[], **kwargs_18240)
        
        # Assigning a type to the variable 'global_context' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'global_context', get_global_context_call_result_18241)
        
        # Getting the type of 'name' (line 350)
        name_18242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 11), 'name')
        # Getting the type of 'current_context' (line 350)
        current_context_18243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 19), 'current_context')
        # Obtaining the member 'global_vars' of a type (line 350)
        global_vars_18244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 19), current_context_18243, 'global_vars')
        # Applying the binary operator 'in' (line 350)
        result_contains_18245 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 11), 'in', name_18242, global_vars_18244)
        
        # Testing if the type of an if condition is none (line 350)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 350, 8), result_contains_18245):
            pass
        else:
            
            # Testing the type of an if condition (line 350)
            if_condition_18246 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 350, 8), result_contains_18245)
            # Assigning a type to the variable 'if_condition_18246' (line 350)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'if_condition_18246', if_condition_18246)
            # SSA begins for if statement (line 350)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 352):
            
            # Assigning a Call to a Name (line 352):
            
            # Call to get_type_of(...): (line 352)
            # Processing the call arguments (line 352)
            # Getting the type of 'name' (line 352)
            name_18249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 47), 'name', False)
            # Processing the call keyword arguments (line 352)
            kwargs_18250 = {}
            # Getting the type of 'global_context' (line 352)
            global_context_18247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 20), 'global_context', False)
            # Obtaining the member 'get_type_of' of a type (line 352)
            get_type_of_18248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 20), global_context_18247, 'get_type_of')
            # Calling get_type_of(args, kwargs) (line 352)
            get_type_of_call_result_18251 = invoke(stypy.reporting.localization.Localization(__file__, 352, 20), get_type_of_18248, *[name_18249], **kwargs_18250)
            
            # Assigning a type to the variable 'type_' (line 352)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'type_', get_type_of_call_result_18251)
            
            # Type idiom detected: calculating its left and rigth part (line 354)
            # Getting the type of 'type_' (line 354)
            type__18252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 15), 'type_')
            # Getting the type of 'None' (line 354)
            None_18253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 24), 'None')
            
            (may_be_18254, more_types_in_union_18255) = may_be_none(type__18252, None_18253)

            if may_be_18254:

                if more_types_in_union_18255:
                    # Runtime conditional SSA (line 354)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to TypeError(...): (line 355)
                # Processing the call arguments (line 355)
                # Getting the type of 'localization' (line 355)
                localization_18257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 33), 'localization', False)
                str_18258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 47), 'str', "Attempted to read the uninitialized global '%s'")
                # Getting the type of 'name' (line 355)
                name_18259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 99), 'name', False)
                # Applying the binary operator '%' (line 355)
                result_mod_18260 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 47), '%', str_18258, name_18259)
                
                # Processing the call keyword arguments (line 355)
                kwargs_18261 = {}
                # Getting the type of 'TypeError' (line 355)
                TypeError_18256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 23), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 355)
                TypeError_call_result_18262 = invoke(stypy.reporting.localization.Localization(__file__, 355, 23), TypeError_18256, *[localization_18257, result_mod_18260], **kwargs_18261)
                
                # Assigning a type to the variable 'stypy_return_type' (line 355)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 16), 'stypy_return_type', TypeError_call_result_18262)

                if more_types_in_union_18255:
                    # Runtime conditional SSA for else branch (line 354)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_18254) or more_types_in_union_18255):
                # Getting the type of 'type_' (line 358)
                type__18263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 23), 'type_')
                # Assigning a type to the variable 'stypy_return_type' (line 358)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 16), 'stypy_return_type', type__18263)

                if (may_be_18254 and more_types_in_union_18255):
                    # SSA join for if statement (line 354)
                    module_type_store = module_type_store.join_ssa_context()


            
            # Getting the type of 'type_' (line 354)
            type__18264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'type_')
            # Assigning a type to the variable 'type_' (line 354)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'type_', remove_type_from_union(type__18264, types.NoneType))
            # SSA join for if statement (line 350)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Name to a Name (line 360):
        
        # Assigning a Name to a Name (line 360):
        # Getting the type of 'False' (line 360)
        False_18265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 30), 'False')
        # Assigning a type to the variable 'top_context_reached' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'top_context_reached', False_18265)
        
        # Getting the type of 'self' (line 363)
        self_18266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 23), 'self')
        # Obtaining the member 'context_stack' of a type (line 363)
        context_stack_18267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 23), self_18266, 'context_stack')
        # Assigning a type to the variable 'context_stack_18267' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'context_stack_18267', context_stack_18267)
        # Testing if the for loop is going to be iterated (line 363)
        # Testing the type of a for loop iterable (line 363)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 363, 8), context_stack_18267)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 363, 8), context_stack_18267):
            # Getting the type of the for loop variable (line 363)
            for_loop_var_18268 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 363, 8), context_stack_18267)
            # Assigning a type to the variable 'context' (line 363)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'context', for_loop_var_18268)
            # SSA begins for a for statement (line 363)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'context' (line 364)
            context_18269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 15), 'context')
            # Getting the type of 'f_context' (line 364)
            f_context_18270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 26), 'f_context')
            # Applying the binary operator '==' (line 364)
            result_eq_18271 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 15), '==', context_18269, f_context_18270)
            
            # Testing if the type of an if condition is none (line 364)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 364, 12), result_eq_18271):
                pass
            else:
                
                # Testing the type of an if condition (line 364)
                if_condition_18272 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 364, 12), result_eq_18271)
                # Assigning a type to the variable 'if_condition_18272' (line 364)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'if_condition_18272', if_condition_18272)
                # SSA begins for if statement (line 364)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 365):
                
                # Assigning a Name to a Name (line 365):
                # Getting the type of 'True' (line 365)
                True_18273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 38), 'True')
                # Assigning a type to the variable 'top_context_reached' (line 365)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 16), 'top_context_reached', True_18273)
                # SSA join for if statement (line 364)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'top_context_reached' (line 367)
            top_context_reached_18274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 19), 'top_context_reached')
            # Applying the 'not' unary operator (line 367)
            result_not__18275 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 15), 'not', top_context_reached_18274)
            
            # Testing if the type of an if condition is none (line 367)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 367, 12), result_not__18275):
                pass
            else:
                
                # Testing the type of an if condition (line 367)
                if_condition_18276 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 367, 12), result_not__18275)
                # Assigning a type to the variable 'if_condition_18276' (line 367)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'if_condition_18276', if_condition_18276)
                # SSA begins for if statement (line 367)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 367)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Call to a Name (line 370):
            
            # Assigning a Call to a Name (line 370):
            
            # Call to get_type_of(...): (line 370)
            # Processing the call arguments (line 370)
            # Getting the type of 'name' (line 370)
            name_18279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 40), 'name', False)
            # Processing the call keyword arguments (line 370)
            kwargs_18280 = {}
            # Getting the type of 'context' (line 370)
            context_18277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 20), 'context', False)
            # Obtaining the member 'get_type_of' of a type (line 370)
            get_type_of_18278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 20), context_18277, 'get_type_of')
            # Calling get_type_of(args, kwargs) (line 370)
            get_type_of_call_result_18281 = invoke(stypy.reporting.localization.Localization(__file__, 370, 20), get_type_of_18278, *[name_18279], **kwargs_18280)
            
            # Assigning a type to the variable 'type_' (line 370)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'type_', get_type_of_call_result_18281)
            
            # Type idiom detected: calculating its left and rigth part (line 372)
            # Getting the type of 'type_' (line 372)
            type__18282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 15), 'type_')
            # Getting the type of 'None' (line 372)
            None_18283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 24), 'None')
            
            (may_be_18284, more_types_in_union_18285) = may_be_none(type__18282, None_18283)

            if may_be_18284:

                if more_types_in_union_18285:
                    # Runtime conditional SSA (line 372)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store


                if more_types_in_union_18285:
                    # SSA join for if statement (line 372)
                    module_type_store = module_type_store.join_ssa_context()


            
            str_18286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, (-1)), 'str', '\n            The type of name is found. In this case, we test if the name is also present into the global context.\n            If it is, and was not marked as a global till now, we generate a warning indicating that if a write access\n            is performed to name and it is still not marked as global, then Python will throw a runtime error\n            complaining that name has been referenced without being assigned first. global have to be used to avoid\n            this error.\n            ')
            # Getting the type of 'self' (line 383)
            self_18287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 15), 'self')
            # Obtaining the member 'test_unreferenced_var' of a type (line 383)
            test_unreferenced_var_18288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 15), self_18287, 'test_unreferenced_var')
            # Testing if the type of an if condition is none (line 383)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 383, 12), test_unreferenced_var_18288):
                pass
            else:
                
                # Testing the type of an if condition (line 383)
                if_condition_18289 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 383, 12), test_unreferenced_var_18288)
                # Assigning a type to the variable 'if_condition_18289' (line 383)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'if_condition_18289', if_condition_18289)
                # SSA begins for if statement (line 383)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Evaluating a boolean operation
                
                # Getting the type of 'name' (line 384)
                name_18290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 19), 'name')
                # Getting the type of 'current_context' (line 384)
                current_context_18291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 31), 'current_context')
                # Obtaining the member 'global_vars' of a type (line 384)
                global_vars_18292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 31), current_context_18291, 'global_vars')
                # Applying the binary operator 'notin' (line 384)
                result_contains_18293 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 19), 'notin', name_18290, global_vars_18292)
                
                
                
                # Getting the type of 'context' (line 385)
                context_18294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 28), 'context')
                
                # Call to get_context(...): (line 385)
                # Processing the call keyword arguments (line 385)
                kwargs_18297 = {}
                # Getting the type of 'self' (line 385)
                self_18295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 39), 'self', False)
                # Obtaining the member 'get_context' of a type (line 385)
                get_context_18296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 39), self_18295, 'get_context')
                # Calling get_context(args, kwargs) (line 385)
                get_context_call_result_18298 = invoke(stypy.reporting.localization.Localization(__file__, 385, 39), get_context_18296, *[], **kwargs_18297)
                
                # Applying the binary operator '==' (line 385)
                result_eq_18299 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 28), '==', context_18294, get_context_call_result_18298)
                
                # Applying the 'not' unary operator (line 385)
                result_not__18300 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 24), 'not', result_eq_18299)
                
                # Applying the binary operator 'and' (line 384)
                result_and_keyword_18301 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 19), 'and', result_contains_18293, result_not__18300)
                
                
                # Getting the type of 'current_context' (line 386)
                current_context_18302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 32), 'current_context')
                # Getting the type of 'global_context' (line 386)
                global_context_18303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 51), 'global_context')
                # Applying the binary operator '==' (line 386)
                result_eq_18304 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 32), '==', current_context_18302, global_context_18303)
                
                # Applying the 'not' unary operator (line 386)
                result_not__18305 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 28), 'not', result_eq_18304)
                
                # Applying the binary operator 'and' (line 384)
                result_and_keyword_18306 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 19), 'and', result_and_keyword_18301, result_not__18305)
                
                # Testing if the type of an if condition is none (line 384)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 384, 16), result_and_keyword_18306):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 384)
                    if_condition_18307 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 384, 16), result_and_keyword_18306)
                    # Assigning a type to the variable 'if_condition_18307' (line 384)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 16), 'if_condition_18307', if_condition_18307)
                    # SSA begins for if statement (line 384)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to UnreferencedLocalVariableTypeWarning(...): (line 387)
                    # Processing the call arguments (line 387)
                    # Getting the type of 'localization' (line 387)
                    localization_18309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 57), 'localization', False)
                    # Getting the type of 'name' (line 387)
                    name_18310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 71), 'name', False)
                    # Getting the type of 'current_context' (line 387)
                    current_context_18311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 77), 'current_context', False)
                    # Processing the call keyword arguments (line 387)
                    kwargs_18312 = {}
                    # Getting the type of 'UnreferencedLocalVariableTypeWarning' (line 387)
                    UnreferencedLocalVariableTypeWarning_18308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 20), 'UnreferencedLocalVariableTypeWarning', False)
                    # Calling UnreferencedLocalVariableTypeWarning(args, kwargs) (line 387)
                    UnreferencedLocalVariableTypeWarning_call_result_18313 = invoke(stypy.reporting.localization.Localization(__file__, 387, 20), UnreferencedLocalVariableTypeWarning_18308, *[localization_18309, name_18310, current_context_18311], **kwargs_18312)
                    
                    # SSA join for if statement (line 384)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 383)
                module_type_store = module_type_store.join_ssa_context()
                

            # Getting the type of 'type_' (line 389)
            type__18314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 19), 'type_')
            # Assigning a type to the variable 'stypy_return_type' (line 389)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'stypy_return_type', type__18314)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to UndefinedTypeError(...): (line 391)
        # Processing the call arguments (line 391)
        # Getting the type of 'localization' (line 391)
        localization_18316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 34), 'localization', False)
        str_18317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 48), 'str', "The variable '%s' does not exist")
        
        # Call to str(...): (line 391)
        # Processing the call arguments (line 391)
        # Getting the type of 'name' (line 391)
        name_18319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 89), 'name', False)
        # Processing the call keyword arguments (line 391)
        kwargs_18320 = {}
        # Getting the type of 'str' (line 391)
        str_18318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 85), 'str', False)
        # Calling str(args, kwargs) (line 391)
        str_call_result_18321 = invoke(stypy.reporting.localization.Localization(__file__, 391, 85), str_18318, *[name_18319], **kwargs_18320)
        
        # Applying the binary operator '%' (line 391)
        result_mod_18322 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 48), '%', str_18317, str_call_result_18321)
        
        # Processing the call keyword arguments (line 391)
        kwargs_18323 = {}
        # Getting the type of 'UndefinedTypeError' (line 391)
        UndefinedTypeError_18315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 15), 'UndefinedTypeError', False)
        # Calling UndefinedTypeError(args, kwargs) (line 391)
        UndefinedTypeError_call_result_18324 = invoke(stypy.reporting.localization.Localization(__file__, 391, 15), UndefinedTypeError_18315, *[localization_18316, result_mod_18322], **kwargs_18323)
        
        # Assigning a type to the variable 'stypy_return_type' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'stypy_return_type', UndefinedTypeError_call_result_18324)
        
        # ################# End of '__get_type_of_from_function_context(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__get_type_of_from_function_context' in the type store
        # Getting the type of 'stypy_return_type' (line 333)
        stypy_return_type_18325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18325)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__get_type_of_from_function_context'
        return stypy_return_type_18325


    @norecursion
    def __set_type_of(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__set_type_of'
        module_type_store = module_type_store.open_function_context('__set_type_of', 393, 4, False)
        # Assigning a type to the variable 'self' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.__set_type_of.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.__set_type_of.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.__set_type_of.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.__set_type_of.__dict__.__setitem__('stypy_function_name', 'TypeStore.__set_type_of')
        TypeStore.__set_type_of.__dict__.__setitem__('stypy_param_names_list', ['localization', 'name', 'type_'])
        TypeStore.__set_type_of.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.__set_type_of.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.__set_type_of.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.__set_type_of.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.__set_type_of.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.__set_type_of.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.__set_type_of', ['localization', 'name', 'type_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__set_type_of', localization, ['localization', 'name', 'type_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__set_type_of(...)' code ##################

        str_18326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, (-1)), 'str', "\n        Cases:\n\n        - Exist in the local context:\n            Is marked as global: It means that the global keyword was used after one assignment ->\n         assign the variable in the global context and remove from the local\n            Is not marked as global: Update\n        - Don't exist in the local context:\n            Is global: Go to the global context and assign\n            Is not global: Create (Update). Shadows more global same-name element\n        ")
        
        # Assigning a Call to a Name (line 405):
        
        # Assigning a Call to a Name (line 405):
        
        # Call to get_global_context(...): (line 405)
        # Processing the call keyword arguments (line 405)
        kwargs_18329 = {}
        # Getting the type of 'self' (line 405)
        self_18327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 25), 'self', False)
        # Obtaining the member 'get_global_context' of a type (line 405)
        get_global_context_18328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 25), self_18327, 'get_global_context')
        # Calling get_global_context(args, kwargs) (line 405)
        get_global_context_call_result_18330 = invoke(stypy.reporting.localization.Localization(__file__, 405, 25), get_global_context_18328, *[], **kwargs_18329)
        
        # Assigning a type to the variable 'global_context' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'global_context', get_global_context_call_result_18330)
        
        # Assigning a Compare to a Name (line 406):
        
        # Assigning a Compare to a Name (line 406):
        
        # Getting the type of 'name' (line 406)
        name_18331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 30), 'name')
        
        # Call to get_context(...): (line 406)
        # Processing the call keyword arguments (line 406)
        kwargs_18334 = {}
        # Getting the type of 'self' (line 406)
        self_18332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 38), 'self', False)
        # Obtaining the member 'get_context' of a type (line 406)
        get_context_18333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 38), self_18332, 'get_context')
        # Calling get_context(args, kwargs) (line 406)
        get_context_call_result_18335 = invoke(stypy.reporting.localization.Localization(__file__, 406, 38), get_context_18333, *[], **kwargs_18334)
        
        # Obtaining the member 'global_vars' of a type (line 406)
        global_vars_18336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 38), get_context_call_result_18335, 'global_vars')
        # Applying the binary operator 'in' (line 406)
        result_contains_18337 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 30), 'in', name_18331, global_vars_18336)
        
        # Assigning a type to the variable 'is_marked_as_global' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'is_marked_as_global', result_contains_18337)
        
        # Assigning a Compare to a Name (line 407):
        
        # Assigning a Compare to a Name (line 407):
        
        # Getting the type of 'name' (line 407)
        name_18338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 33), 'name')
        
        # Call to get_context(...): (line 407)
        # Processing the call keyword arguments (line 407)
        kwargs_18341 = {}
        # Getting the type of 'self' (line 407)
        self_18339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 41), 'self', False)
        # Obtaining the member 'get_context' of a type (line 407)
        get_context_18340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 41), self_18339, 'get_context')
        # Calling get_context(args, kwargs) (line 407)
        get_context_call_result_18342 = invoke(stypy.reporting.localization.Localization(__file__, 407, 41), get_context_18340, *[], **kwargs_18341)
        
        # Applying the binary operator 'in' (line 407)
        result_contains_18343 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 33), 'in', name_18338, get_context_call_result_18342)
        
        # Assigning a type to the variable 'exist_in_local_context' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'exist_in_local_context', result_contains_18343)
        # Getting the type of 'exist_in_local_context' (line 409)
        exist_in_local_context_18344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 11), 'exist_in_local_context')
        # Testing if the type of an if condition is none (line 409)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 409, 8), exist_in_local_context_18344):
            # Getting the type of 'is_marked_as_global' (line 420)
            is_marked_as_global_18387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 15), 'is_marked_as_global')
            # Testing if the type of an if condition is none (line 420)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 420, 12), is_marked_as_global_18387):
                str_18396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, (-1)), 'str', 'Special case:\n                    If:\n                        - A variable do not exist in the local context\n                        - This variable is not marked as global\n                        - There exist unreferenced type warnings in this scope typed to this variable.\n                    Then:\n                        - For each unreferenced type warning found:\n                            - Generate a unreferenced variable error with the warning data\n                            - Delete warning\n                            - Mark the type of the variable as ErrorType\n                ')
                
                # Assigning a Call to a Name (line 434):
                
                # Assigning a Call to a Name (line 434):
                
                # Call to filter(...): (line 434)
                # Processing the call arguments (line 434)

                @norecursion
                def _stypy_temp_lambda_27(localization, *varargs, **kwargs):
                    global module_type_store
                    # Assign values to the parameters with defaults
                    defaults = []
                    # Create a new context for function '_stypy_temp_lambda_27'
                    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_27', 434, 52, True)
                    # Passed parameters checking function
                    _stypy_temp_lambda_27.stypy_localization = localization
                    _stypy_temp_lambda_27.stypy_type_of_self = None
                    _stypy_temp_lambda_27.stypy_type_store = module_type_store
                    _stypy_temp_lambda_27.stypy_function_name = '_stypy_temp_lambda_27'
                    _stypy_temp_lambda_27.stypy_param_names_list = ['warning']
                    _stypy_temp_lambda_27.stypy_varargs_param_name = None
                    _stypy_temp_lambda_27.stypy_kwargs_param_name = None
                    _stypy_temp_lambda_27.stypy_call_defaults = defaults
                    _stypy_temp_lambda_27.stypy_call_varargs = varargs
                    _stypy_temp_lambda_27.stypy_call_kwargs = kwargs
                    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_27', ['warning'], None, None, defaults, varargs, kwargs)

                    if is_error_type(arguments):
                        # Destroy the current context
                        module_type_store = module_type_store.close_function_context()
                        return arguments

                    # Stacktrace push for error reporting
                    localization.set_stack_trace('_stypy_temp_lambda_27', ['warning'], arguments)
                    # Default return type storage variable (SSA)
                    # Assigning a type to the variable 'stypy_return_type'
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                    
                    
                    # ################# Begin of the lambda function code ##################

                    
                    # Getting the type of 'warning' (line 435)
                    warning_18398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 52), 'warning', False)
                    # Obtaining the member '__class__' of a type (line 435)
                    class___18399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 52), warning_18398, '__class__')
                    # Getting the type of 'UnreferencedLocalVariableTypeWarning' (line 435)
                    UnreferencedLocalVariableTypeWarning_18400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 73), 'UnreferencedLocalVariableTypeWarning', False)
                    # Applying the binary operator '==' (line 435)
                    result_eq_18401 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 52), '==', class___18399, UnreferencedLocalVariableTypeWarning_18400)
                    
                    # Assigning the return type of the lambda function
                    # Assigning a type to the variable 'stypy_return_type' (line 434)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 52), 'stypy_return_type', result_eq_18401)
                    
                    # ################# End of the lambda function code ##################

                    # Stacktrace pop (error reporting)
                    localization.unset_stack_trace()
                    
                    # Storing the return type of function '_stypy_temp_lambda_27' in the type store
                    # Getting the type of 'stypy_return_type' (line 434)
                    stypy_return_type_18402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 52), 'stypy_return_type')
                    module_type_store.store_return_type_of_current_context(stypy_return_type_18402)
                    
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    
                    # Return type of the function '_stypy_temp_lambda_27'
                    return stypy_return_type_18402

                # Assigning a type to the variable '_stypy_temp_lambda_27' (line 434)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 52), '_stypy_temp_lambda_27', _stypy_temp_lambda_27)
                # Getting the type of '_stypy_temp_lambda_27' (line 434)
                _stypy_temp_lambda_27_18403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 52), '_stypy_temp_lambda_27')
                
                # Call to get_warning_msgs(...): (line 436)
                # Processing the call keyword arguments (line 436)
                kwargs_18406 = {}
                # Getting the type of 'TypeWarning' (line 436)
                TypeWarning_18404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 52), 'TypeWarning', False)
                # Obtaining the member 'get_warning_msgs' of a type (line 436)
                get_warning_msgs_18405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 52), TypeWarning_18404, 'get_warning_msgs')
                # Calling get_warning_msgs(args, kwargs) (line 436)
                get_warning_msgs_call_result_18407 = invoke(stypy.reporting.localization.Localization(__file__, 436, 52), get_warning_msgs_18405, *[], **kwargs_18406)
                
                # Processing the call keyword arguments (line 434)
                kwargs_18408 = {}
                # Getting the type of 'filter' (line 434)
                filter_18397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 45), 'filter', False)
                # Calling filter(args, kwargs) (line 434)
                filter_call_result_18409 = invoke(stypy.reporting.localization.Localization(__file__, 434, 45), filter_18397, *[_stypy_temp_lambda_27_18403, get_warning_msgs_call_result_18407], **kwargs_18408)
                
                # Assigning a type to the variable 'unreferenced_type_warnings' (line 434)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 16), 'unreferenced_type_warnings', filter_call_result_18409)
                
                
                # Call to len(...): (line 438)
                # Processing the call arguments (line 438)
                # Getting the type of 'unreferenced_type_warnings' (line 438)
                unreferenced_type_warnings_18411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 23), 'unreferenced_type_warnings', False)
                # Processing the call keyword arguments (line 438)
                kwargs_18412 = {}
                # Getting the type of 'len' (line 438)
                len_18410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 19), 'len', False)
                # Calling len(args, kwargs) (line 438)
                len_call_result_18413 = invoke(stypy.reporting.localization.Localization(__file__, 438, 19), len_18410, *[unreferenced_type_warnings_18411], **kwargs_18412)
                
                int_18414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 53), 'int')
                # Applying the binary operator '>' (line 438)
                result_gt_18415 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 19), '>', len_call_result_18413, int_18414)
                
                # Testing if the type of an if condition is none (line 438)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 438, 16), result_gt_18415):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 438)
                    if_condition_18416 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 438, 16), result_gt_18415)
                    # Assigning a type to the variable 'if_condition_18416' (line 438)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 'if_condition_18416', if_condition_18416)
                    # SSA begins for if statement (line 438)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 439):
                    
                    # Assigning a Call to a Name (line 439):
                    
                    # Call to filter(...): (line 439)
                    # Processing the call arguments (line 439)

                    @norecursion
                    def _stypy_temp_lambda_28(localization, *varargs, **kwargs):
                        global module_type_store
                        # Assign values to the parameters with defaults
                        defaults = []
                        # Create a new context for function '_stypy_temp_lambda_28'
                        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_28', 439, 76, True)
                        # Passed parameters checking function
                        _stypy_temp_lambda_28.stypy_localization = localization
                        _stypy_temp_lambda_28.stypy_type_of_self = None
                        _stypy_temp_lambda_28.stypy_type_store = module_type_store
                        _stypy_temp_lambda_28.stypy_function_name = '_stypy_temp_lambda_28'
                        _stypy_temp_lambda_28.stypy_param_names_list = ['warning']
                        _stypy_temp_lambda_28.stypy_varargs_param_name = None
                        _stypy_temp_lambda_28.stypy_kwargs_param_name = None
                        _stypy_temp_lambda_28.stypy_call_defaults = defaults
                        _stypy_temp_lambda_28.stypy_call_varargs = varargs
                        _stypy_temp_lambda_28.stypy_call_kwargs = kwargs
                        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_28', ['warning'], None, None, defaults, varargs, kwargs)

                        if is_error_type(arguments):
                            # Destroy the current context
                            module_type_store = module_type_store.close_function_context()
                            return arguments

                        # Stacktrace push for error reporting
                        localization.set_stack_trace('_stypy_temp_lambda_28', ['warning'], arguments)
                        # Default return type storage variable (SSA)
                        # Assigning a type to the variable 'stypy_return_type'
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                        
                        
                        # ################# Begin of the lambda function code ##################

                        
                        # Evaluating a boolean operation
                        
                        # Getting the type of 'warning' (line 440)
                        warning_18418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 76), 'warning', False)
                        # Obtaining the member 'context' of a type (line 440)
                        context_18419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 76), warning_18418, 'context')
                        
                        # Call to get_context(...): (line 440)
                        # Processing the call keyword arguments (line 440)
                        kwargs_18422 = {}
                        # Getting the type of 'self' (line 440)
                        self_18420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 95), 'self', False)
                        # Obtaining the member 'get_context' of a type (line 440)
                        get_context_18421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 95), self_18420, 'get_context')
                        # Calling get_context(args, kwargs) (line 440)
                        get_context_call_result_18423 = invoke(stypy.reporting.localization.Localization(__file__, 440, 95), get_context_18421, *[], **kwargs_18422)
                        
                        # Applying the binary operator '==' (line 440)
                        result_eq_18424 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 76), '==', context_18419, get_context_call_result_18423)
                        
                        
                        # Getting the type of 'warning' (line 441)
                        warning_18425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 76), 'warning', False)
                        # Obtaining the member 'name' of a type (line 441)
                        name_18426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 76), warning_18425, 'name')
                        # Getting the type of 'name' (line 441)
                        name_18427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 92), 'name', False)
                        # Applying the binary operator '==' (line 441)
                        result_eq_18428 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 76), '==', name_18426, name_18427)
                        
                        # Applying the binary operator 'and' (line 440)
                        result_and_keyword_18429 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 76), 'and', result_eq_18424, result_eq_18428)
                        
                        # Assigning the return type of the lambda function
                        # Assigning a type to the variable 'stypy_return_type' (line 439)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 76), 'stypy_return_type', result_and_keyword_18429)
                        
                        # ################# End of the lambda function code ##################

                        # Stacktrace pop (error reporting)
                        localization.unset_stack_trace()
                        
                        # Storing the return type of function '_stypy_temp_lambda_28' in the type store
                        # Getting the type of 'stypy_return_type' (line 439)
                        stypy_return_type_18430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 76), 'stypy_return_type')
                        module_type_store.store_return_type_of_current_context(stypy_return_type_18430)
                        
                        # Destroy the current context
                        module_type_store = module_type_store.close_function_context()
                        
                        # Return type of the function '_stypy_temp_lambda_28'
                        return stypy_return_type_18430

                    # Assigning a type to the variable '_stypy_temp_lambda_28' (line 439)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 76), '_stypy_temp_lambda_28', _stypy_temp_lambda_28)
                    # Getting the type of '_stypy_temp_lambda_28' (line 439)
                    _stypy_temp_lambda_28_18431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 76), '_stypy_temp_lambda_28')
                    # Getting the type of 'unreferenced_type_warnings' (line 442)
                    unreferenced_type_warnings_18432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 76), 'unreferenced_type_warnings', False)
                    # Processing the call keyword arguments (line 439)
                    kwargs_18433 = {}
                    # Getting the type of 'filter' (line 439)
                    filter_18417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 69), 'filter', False)
                    # Calling filter(args, kwargs) (line 439)
                    filter_call_result_18434 = invoke(stypy.reporting.localization.Localization(__file__, 439, 69), filter_18417, *[_stypy_temp_lambda_28_18431, unreferenced_type_warnings_18432], **kwargs_18433)
                    
                    # Assigning a type to the variable 'our_unreferenced_type_warnings_in_this_context' (line 439)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 20), 'our_unreferenced_type_warnings_in_this_context', filter_call_result_18434)
                    
                    # Getting the type of 'our_unreferenced_type_warnings_in_this_context' (line 444)
                    our_unreferenced_type_warnings_in_this_context_18435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 31), 'our_unreferenced_type_warnings_in_this_context')
                    # Assigning a type to the variable 'our_unreferenced_type_warnings_in_this_context_18435' (line 444)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 20), 'our_unreferenced_type_warnings_in_this_context_18435', our_unreferenced_type_warnings_in_this_context_18435)
                    # Testing if the for loop is going to be iterated (line 444)
                    # Testing the type of a for loop iterable (line 444)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 444, 20), our_unreferenced_type_warnings_in_this_context_18435)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 444, 20), our_unreferenced_type_warnings_in_this_context_18435):
                        # Getting the type of the for loop variable (line 444)
                        for_loop_var_18436 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 444, 20), our_unreferenced_type_warnings_in_this_context_18435)
                        # Assigning a type to the variable 'utw' (line 444)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 20), 'utw', for_loop_var_18436)
                        # SSA begins for a for statement (line 444)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Call to TypeError(...): (line 445)
                        # Processing the call arguments (line 445)
                        # Getting the type of 'localization' (line 445)
                        localization_18438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 34), 'localization', False)
                        
                        # Call to format(...): (line 445)
                        # Processing the call arguments (line 445)
                        # Getting the type of 'name' (line 446)
                        name_18441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 86), 'name', False)
                        # Processing the call keyword arguments (line 445)
                        kwargs_18442 = {}
                        str_18439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 48), 'str', "UnboundLocalError: local variable '{0}' referenced before assignment")
                        # Obtaining the member 'format' of a type (line 445)
                        format_18440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 48), str_18439, 'format')
                        # Calling format(args, kwargs) (line 445)
                        format_call_result_18443 = invoke(stypy.reporting.localization.Localization(__file__, 445, 48), format_18440, *[name_18441], **kwargs_18442)
                        
                        # Processing the call keyword arguments (line 445)
                        kwargs_18444 = {}
                        # Getting the type of 'TypeError' (line 445)
                        TypeError_18437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 24), 'TypeError', False)
                        # Calling TypeError(args, kwargs) (line 445)
                        TypeError_call_result_18445 = invoke(stypy.reporting.localization.Localization(__file__, 445, 24), TypeError_18437, *[localization_18438, format_call_result_18443], **kwargs_18444)
                        
                        
                        # Call to remove(...): (line 447)
                        # Processing the call arguments (line 447)
                        # Getting the type of 'utw' (line 447)
                        utw_18449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 52), 'utw', False)
                        # Processing the call keyword arguments (line 447)
                        kwargs_18450 = {}
                        # Getting the type of 'TypeWarning' (line 447)
                        TypeWarning_18446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 24), 'TypeWarning', False)
                        # Obtaining the member 'warnings' of a type (line 447)
                        warnings_18447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 24), TypeWarning_18446, 'warnings')
                        # Obtaining the member 'remove' of a type (line 447)
                        remove_18448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 24), warnings_18447, 'remove')
                        # Calling remove(args, kwargs) (line 447)
                        remove_call_result_18451 = invoke(stypy.reporting.localization.Localization(__file__, 447, 24), remove_18448, *[utw_18449], **kwargs_18450)
                        
                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    
                    
                    # Call to len(...): (line 450)
                    # Processing the call arguments (line 450)
                    # Getting the type of 'our_unreferenced_type_warnings_in_this_context' (line 450)
                    our_unreferenced_type_warnings_in_this_context_18453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 27), 'our_unreferenced_type_warnings_in_this_context', False)
                    # Processing the call keyword arguments (line 450)
                    kwargs_18454 = {}
                    # Getting the type of 'len' (line 450)
                    len_18452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 23), 'len', False)
                    # Calling len(args, kwargs) (line 450)
                    len_call_result_18455 = invoke(stypy.reporting.localization.Localization(__file__, 450, 23), len_18452, *[our_unreferenced_type_warnings_in_this_context_18453], **kwargs_18454)
                    
                    int_18456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 77), 'int')
                    # Applying the binary operator '>' (line 450)
                    result_gt_18457 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 23), '>', len_call_result_18455, int_18456)
                    
                    # Testing if the type of an if condition is none (line 450)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 450, 20), result_gt_18457):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 450)
                        if_condition_18458 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 450, 20), result_gt_18457)
                        # Assigning a type to the variable 'if_condition_18458' (line 450)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 20), 'if_condition_18458', if_condition_18458)
                        # SSA begins for if statement (line 450)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to set_type_of(...): (line 451)
                        # Processing the call arguments (line 451)
                        # Getting the type of 'name' (line 451)
                        name_18464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 55), 'name', False)
                        
                        # Call to TypeError(...): (line 451)
                        # Processing the call arguments (line 451)
                        # Getting the type of 'localization' (line 451)
                        localization_18466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 71), 'localization', False)
                        
                        # Call to format(...): (line 451)
                        # Processing the call arguments (line 451)
                        # Getting the type of 'name' (line 452)
                        name_18469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 113), 'name', False)
                        # Processing the call keyword arguments (line 451)
                        kwargs_18470 = {}
                        str_18467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 85), 'str', "Attempted to use '{0}' previously to its definition")
                        # Obtaining the member 'format' of a type (line 451)
                        format_18468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 85), str_18467, 'format')
                        # Calling format(args, kwargs) (line 451)
                        format_call_result_18471 = invoke(stypy.reporting.localization.Localization(__file__, 451, 85), format_18468, *[name_18469], **kwargs_18470)
                        
                        # Processing the call keyword arguments (line 451)
                        kwargs_18472 = {}
                        # Getting the type of 'TypeError' (line 451)
                        TypeError_18465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 61), 'TypeError', False)
                        # Calling TypeError(args, kwargs) (line 451)
                        TypeError_call_result_18473 = invoke(stypy.reporting.localization.Localization(__file__, 451, 61), TypeError_18465, *[localization_18466, format_call_result_18471], **kwargs_18472)
                        
                        # Getting the type of 'localization' (line 453)
                        localization_18474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 55), 'localization', False)
                        # Processing the call keyword arguments (line 451)
                        kwargs_18475 = {}
                        
                        # Call to get_context(...): (line 451)
                        # Processing the call keyword arguments (line 451)
                        kwargs_18461 = {}
                        # Getting the type of 'self' (line 451)
                        self_18459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 24), 'self', False)
                        # Obtaining the member 'get_context' of a type (line 451)
                        get_context_18460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 24), self_18459, 'get_context')
                        # Calling get_context(args, kwargs) (line 451)
                        get_context_call_result_18462 = invoke(stypy.reporting.localization.Localization(__file__, 451, 24), get_context_18460, *[], **kwargs_18461)
                        
                        # Obtaining the member 'set_type_of' of a type (line 451)
                        set_type_of_18463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 24), get_context_call_result_18462, 'set_type_of')
                        # Calling set_type_of(args, kwargs) (line 451)
                        set_type_of_call_result_18476 = invoke(stypy.reporting.localization.Localization(__file__, 451, 24), set_type_of_18463, *[name_18464, TypeError_call_result_18473, localization_18474], **kwargs_18475)
                        
                        
                        # Call to get_type_of(...): (line 454)
                        # Processing the call arguments (line 454)
                        # Getting the type of 'name' (line 454)
                        name_18482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 62), 'name', False)
                        # Processing the call keyword arguments (line 454)
                        kwargs_18483 = {}
                        
                        # Call to get_context(...): (line 454)
                        # Processing the call keyword arguments (line 454)
                        kwargs_18479 = {}
                        # Getting the type of 'self' (line 454)
                        self_18477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 31), 'self', False)
                        # Obtaining the member 'get_context' of a type (line 454)
                        get_context_18478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 31), self_18477, 'get_context')
                        # Calling get_context(args, kwargs) (line 454)
                        get_context_call_result_18480 = invoke(stypy.reporting.localization.Localization(__file__, 454, 31), get_context_18478, *[], **kwargs_18479)
                        
                        # Obtaining the member 'get_type_of' of a type (line 454)
                        get_type_of_18481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 31), get_context_call_result_18480, 'get_type_of')
                        # Calling get_type_of(args, kwargs) (line 454)
                        get_type_of_call_result_18484 = invoke(stypy.reporting.localization.Localization(__file__, 454, 31), get_type_of_18481, *[name_18482], **kwargs_18483)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 454)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 24), 'stypy_return_type', get_type_of_call_result_18484)
                        # SSA join for if statement (line 450)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 438)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Call to a Tuple (line 456):
                
                # Assigning a Call to a Name:
                
                # Call to contains_an_undefined_type(...): (line 456)
                # Processing the call arguments (line 456)
                # Getting the type of 'type_' (line 457)
                type__18488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 47), 'type_', False)
                # Processing the call keyword arguments (line 456)
                kwargs_18489 = {}
                # Getting the type of 'type_inference_proxy' (line 456)
                type_inference_proxy_18485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 58), 'type_inference_proxy', False)
                # Obtaining the member 'TypeInferenceProxy' of a type (line 456)
                TypeInferenceProxy_18486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 58), type_inference_proxy_18485, 'TypeInferenceProxy')
                # Obtaining the member 'contains_an_undefined_type' of a type (line 456)
                contains_an_undefined_type_18487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 58), TypeInferenceProxy_18486, 'contains_an_undefined_type')
                # Calling contains_an_undefined_type(args, kwargs) (line 456)
                contains_an_undefined_type_call_result_18490 = invoke(stypy.reporting.localization.Localization(__file__, 456, 58), contains_an_undefined_type_18487, *[type__18488], **kwargs_18489)
                
                # Assigning a type to the variable 'call_assignment_17651' (line 456)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17651', contains_an_undefined_type_call_result_18490)
                
                # Assigning a Call to a Name (line 456):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_17651' (line 456)
                call_assignment_17651_18491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17651', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_18492 = stypy_get_value_from_tuple(call_assignment_17651_18491, 2, 0)
                
                # Assigning a type to the variable 'call_assignment_17652' (line 456)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17652', stypy_get_value_from_tuple_call_result_18492)
                
                # Assigning a Name to a Name (line 456):
                # Getting the type of 'call_assignment_17652' (line 456)
                call_assignment_17652_18493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17652')
                # Assigning a type to the variable 'contains_undefined' (line 456)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'contains_undefined', call_assignment_17652_18493)
                
                # Assigning a Call to a Name (line 456):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_17651' (line 456)
                call_assignment_17651_18494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17651', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_18495 = stypy_get_value_from_tuple(call_assignment_17651_18494, 2, 1)
                
                # Assigning a type to the variable 'call_assignment_17653' (line 456)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17653', stypy_get_value_from_tuple_call_result_18495)
                
                # Assigning a Name to a Name (line 456):
                # Getting the type of 'call_assignment_17653' (line 456)
                call_assignment_17653_18496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17653')
                # Assigning a type to the variable 'more_types_in_value' (line 456)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 36), 'more_types_in_value', call_assignment_17653_18496)
                # Getting the type of 'contains_undefined' (line 458)
                contains_undefined_18497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 19), 'contains_undefined')
                # Testing if the type of an if condition is none (line 458)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 458, 16), contains_undefined_18497):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 458)
                    if_condition_18498 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 458, 16), contains_undefined_18497)
                    # Assigning a type to the variable 'if_condition_18498' (line 458)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 16), 'if_condition_18498', if_condition_18498)
                    # SSA begins for if statement (line 458)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'more_types_in_value' (line 459)
                    more_types_in_value_18499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 23), 'more_types_in_value')
                    int_18500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 46), 'int')
                    # Applying the binary operator '==' (line 459)
                    result_eq_18501 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 23), '==', more_types_in_value_18499, int_18500)
                    
                    # Testing if the type of an if condition is none (line 459)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 459, 20), result_eq_18501):
                        
                        # Call to instance(...): (line 463)
                        # Processing the call arguments (line 463)
                        # Getting the type of 'localization' (line 463)
                        localization_18514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 45), 'localization', False)
                        
                        # Call to format(...): (line 464)
                        # Processing the call arguments (line 464)
                        # Getting the type of 'name' (line 465)
                        name_18517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 52), 'name', False)
                        # Processing the call keyword arguments (line 464)
                        kwargs_18518 = {}
                        str_18515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 45), 'str', "Potentialy assigning to '{0}' the value of an undefined variable")
                        # Obtaining the member 'format' of a type (line 464)
                        format_18516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 45), str_18515, 'format')
                        # Calling format(args, kwargs) (line 464)
                        format_call_result_18519 = invoke(stypy.reporting.localization.Localization(__file__, 464, 45), format_18516, *[name_18517], **kwargs_18518)
                        
                        # Processing the call keyword arguments (line 463)
                        kwargs_18520 = {}
                        # Getting the type of 'TypeWarning' (line 463)
                        TypeWarning_18512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 24), 'TypeWarning', False)
                        # Obtaining the member 'instance' of a type (line 463)
                        instance_18513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 24), TypeWarning_18512, 'instance')
                        # Calling instance(args, kwargs) (line 463)
                        instance_call_result_18521 = invoke(stypy.reporting.localization.Localization(__file__, 463, 24), instance_18513, *[localization_18514, format_call_result_18519], **kwargs_18520)
                        
                    else:
                        
                        # Testing the type of an if condition (line 459)
                        if_condition_18502 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 459, 20), result_eq_18501)
                        # Assigning a type to the variable 'if_condition_18502' (line 459)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 20), 'if_condition_18502', if_condition_18502)
                        # SSA begins for if statement (line 459)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to TypeError(...): (line 460)
                        # Processing the call arguments (line 460)
                        # Getting the type of 'localization' (line 460)
                        localization_18504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 34), 'localization', False)
                        
                        # Call to format(...): (line 460)
                        # Processing the call arguments (line 460)
                        # Getting the type of 'name' (line 461)
                        name_18507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 41), 'name', False)
                        # Processing the call keyword arguments (line 460)
                        kwargs_18508 = {}
                        str_18505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 48), 'str', "Assigning to '{0}' the value of an undefined variable")
                        # Obtaining the member 'format' of a type (line 460)
                        format_18506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 48), str_18505, 'format')
                        # Calling format(args, kwargs) (line 460)
                        format_call_result_18509 = invoke(stypy.reporting.localization.Localization(__file__, 460, 48), format_18506, *[name_18507], **kwargs_18508)
                        
                        # Processing the call keyword arguments (line 460)
                        kwargs_18510 = {}
                        # Getting the type of 'TypeError' (line 460)
                        TypeError_18503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 24), 'TypeError', False)
                        # Calling TypeError(args, kwargs) (line 460)
                        TypeError_call_result_18511 = invoke(stypy.reporting.localization.Localization(__file__, 460, 24), TypeError_18503, *[localization_18504, format_call_result_18509], **kwargs_18510)
                        
                        # SSA branch for the else part of an if statement (line 459)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to instance(...): (line 463)
                        # Processing the call arguments (line 463)
                        # Getting the type of 'localization' (line 463)
                        localization_18514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 45), 'localization', False)
                        
                        # Call to format(...): (line 464)
                        # Processing the call arguments (line 464)
                        # Getting the type of 'name' (line 465)
                        name_18517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 52), 'name', False)
                        # Processing the call keyword arguments (line 464)
                        kwargs_18518 = {}
                        str_18515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 45), 'str', "Potentialy assigning to '{0}' the value of an undefined variable")
                        # Obtaining the member 'format' of a type (line 464)
                        format_18516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 45), str_18515, 'format')
                        # Calling format(args, kwargs) (line 464)
                        format_call_result_18519 = invoke(stypy.reporting.localization.Localization(__file__, 464, 45), format_18516, *[name_18517], **kwargs_18518)
                        
                        # Processing the call keyword arguments (line 463)
                        kwargs_18520 = {}
                        # Getting the type of 'TypeWarning' (line 463)
                        TypeWarning_18512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 24), 'TypeWarning', False)
                        # Obtaining the member 'instance' of a type (line 463)
                        instance_18513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 24), TypeWarning_18512, 'instance')
                        # Calling instance(args, kwargs) (line 463)
                        instance_call_result_18521 = invoke(stypy.reporting.localization.Localization(__file__, 463, 24), instance_18513, *[localization_18514, format_call_result_18519], **kwargs_18520)
                        
                        # SSA join for if statement (line 459)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 458)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Call to set_type_of(...): (line 467)
                # Processing the call arguments (line 467)
                # Getting the type of 'name' (line 467)
                name_18527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 47), 'name', False)
                # Getting the type of 'type_' (line 467)
                type__18528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 53), 'type_', False)
                # Getting the type of 'localization' (line 467)
                localization_18529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 60), 'localization', False)
                # Processing the call keyword arguments (line 467)
                kwargs_18530 = {}
                
                # Call to get_context(...): (line 467)
                # Processing the call keyword arguments (line 467)
                kwargs_18524 = {}
                # Getting the type of 'self' (line 467)
                self_18522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 16), 'self', False)
                # Obtaining the member 'get_context' of a type (line 467)
                get_context_18523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 16), self_18522, 'get_context')
                # Calling get_context(args, kwargs) (line 467)
                get_context_call_result_18525 = invoke(stypy.reporting.localization.Localization(__file__, 467, 16), get_context_18523, *[], **kwargs_18524)
                
                # Obtaining the member 'set_type_of' of a type (line 467)
                set_type_of_18526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 16), get_context_call_result_18525, 'set_type_of')
                # Calling set_type_of(args, kwargs) (line 467)
                set_type_of_call_result_18531 = invoke(stypy.reporting.localization.Localization(__file__, 467, 16), set_type_of_18526, *[name_18527, type__18528, localization_18529], **kwargs_18530)
                
            else:
                
                # Testing the type of an if condition (line 420)
                if_condition_18388 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 420, 12), is_marked_as_global_18387)
                # Assigning a type to the variable 'if_condition_18388' (line 420)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 12), 'if_condition_18388', if_condition_18388)
                # SSA begins for if statement (line 420)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to set_type_of(...): (line 421)
                # Processing the call arguments (line 421)
                # Getting the type of 'name' (line 421)
                name_18391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 43), 'name', False)
                # Getting the type of 'type_' (line 421)
                type__18392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 49), 'type_', False)
                # Getting the type of 'localization' (line 421)
                localization_18393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 56), 'localization', False)
                # Processing the call keyword arguments (line 421)
                kwargs_18394 = {}
                # Getting the type of 'global_context' (line 421)
                global_context_18389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 16), 'global_context', False)
                # Obtaining the member 'set_type_of' of a type (line 421)
                set_type_of_18390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 16), global_context_18389, 'set_type_of')
                # Calling set_type_of(args, kwargs) (line 421)
                set_type_of_call_result_18395 = invoke(stypy.reporting.localization.Localization(__file__, 421, 16), set_type_of_18390, *[name_18391, type__18392, localization_18393], **kwargs_18394)
                
                # SSA branch for the else part of an if statement (line 420)
                module_type_store.open_ssa_branch('else')
                str_18396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, (-1)), 'str', 'Special case:\n                    If:\n                        - A variable do not exist in the local context\n                        - This variable is not marked as global\n                        - There exist unreferenced type warnings in this scope typed to this variable.\n                    Then:\n                        - For each unreferenced type warning found:\n                            - Generate a unreferenced variable error with the warning data\n                            - Delete warning\n                            - Mark the type of the variable as ErrorType\n                ')
                
                # Assigning a Call to a Name (line 434):
                
                # Assigning a Call to a Name (line 434):
                
                # Call to filter(...): (line 434)
                # Processing the call arguments (line 434)

                @norecursion
                def _stypy_temp_lambda_27(localization, *varargs, **kwargs):
                    global module_type_store
                    # Assign values to the parameters with defaults
                    defaults = []
                    # Create a new context for function '_stypy_temp_lambda_27'
                    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_27', 434, 52, True)
                    # Passed parameters checking function
                    _stypy_temp_lambda_27.stypy_localization = localization
                    _stypy_temp_lambda_27.stypy_type_of_self = None
                    _stypy_temp_lambda_27.stypy_type_store = module_type_store
                    _stypy_temp_lambda_27.stypy_function_name = '_stypy_temp_lambda_27'
                    _stypy_temp_lambda_27.stypy_param_names_list = ['warning']
                    _stypy_temp_lambda_27.stypy_varargs_param_name = None
                    _stypy_temp_lambda_27.stypy_kwargs_param_name = None
                    _stypy_temp_lambda_27.stypy_call_defaults = defaults
                    _stypy_temp_lambda_27.stypy_call_varargs = varargs
                    _stypy_temp_lambda_27.stypy_call_kwargs = kwargs
                    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_27', ['warning'], None, None, defaults, varargs, kwargs)

                    if is_error_type(arguments):
                        # Destroy the current context
                        module_type_store = module_type_store.close_function_context()
                        return arguments

                    # Stacktrace push for error reporting
                    localization.set_stack_trace('_stypy_temp_lambda_27', ['warning'], arguments)
                    # Default return type storage variable (SSA)
                    # Assigning a type to the variable 'stypy_return_type'
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                    
                    
                    # ################# Begin of the lambda function code ##################

                    
                    # Getting the type of 'warning' (line 435)
                    warning_18398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 52), 'warning', False)
                    # Obtaining the member '__class__' of a type (line 435)
                    class___18399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 52), warning_18398, '__class__')
                    # Getting the type of 'UnreferencedLocalVariableTypeWarning' (line 435)
                    UnreferencedLocalVariableTypeWarning_18400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 73), 'UnreferencedLocalVariableTypeWarning', False)
                    # Applying the binary operator '==' (line 435)
                    result_eq_18401 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 52), '==', class___18399, UnreferencedLocalVariableTypeWarning_18400)
                    
                    # Assigning the return type of the lambda function
                    # Assigning a type to the variable 'stypy_return_type' (line 434)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 52), 'stypy_return_type', result_eq_18401)
                    
                    # ################# End of the lambda function code ##################

                    # Stacktrace pop (error reporting)
                    localization.unset_stack_trace()
                    
                    # Storing the return type of function '_stypy_temp_lambda_27' in the type store
                    # Getting the type of 'stypy_return_type' (line 434)
                    stypy_return_type_18402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 52), 'stypy_return_type')
                    module_type_store.store_return_type_of_current_context(stypy_return_type_18402)
                    
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    
                    # Return type of the function '_stypy_temp_lambda_27'
                    return stypy_return_type_18402

                # Assigning a type to the variable '_stypy_temp_lambda_27' (line 434)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 52), '_stypy_temp_lambda_27', _stypy_temp_lambda_27)
                # Getting the type of '_stypy_temp_lambda_27' (line 434)
                _stypy_temp_lambda_27_18403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 52), '_stypy_temp_lambda_27')
                
                # Call to get_warning_msgs(...): (line 436)
                # Processing the call keyword arguments (line 436)
                kwargs_18406 = {}
                # Getting the type of 'TypeWarning' (line 436)
                TypeWarning_18404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 52), 'TypeWarning', False)
                # Obtaining the member 'get_warning_msgs' of a type (line 436)
                get_warning_msgs_18405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 52), TypeWarning_18404, 'get_warning_msgs')
                # Calling get_warning_msgs(args, kwargs) (line 436)
                get_warning_msgs_call_result_18407 = invoke(stypy.reporting.localization.Localization(__file__, 436, 52), get_warning_msgs_18405, *[], **kwargs_18406)
                
                # Processing the call keyword arguments (line 434)
                kwargs_18408 = {}
                # Getting the type of 'filter' (line 434)
                filter_18397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 45), 'filter', False)
                # Calling filter(args, kwargs) (line 434)
                filter_call_result_18409 = invoke(stypy.reporting.localization.Localization(__file__, 434, 45), filter_18397, *[_stypy_temp_lambda_27_18403, get_warning_msgs_call_result_18407], **kwargs_18408)
                
                # Assigning a type to the variable 'unreferenced_type_warnings' (line 434)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 16), 'unreferenced_type_warnings', filter_call_result_18409)
                
                
                # Call to len(...): (line 438)
                # Processing the call arguments (line 438)
                # Getting the type of 'unreferenced_type_warnings' (line 438)
                unreferenced_type_warnings_18411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 23), 'unreferenced_type_warnings', False)
                # Processing the call keyword arguments (line 438)
                kwargs_18412 = {}
                # Getting the type of 'len' (line 438)
                len_18410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 19), 'len', False)
                # Calling len(args, kwargs) (line 438)
                len_call_result_18413 = invoke(stypy.reporting.localization.Localization(__file__, 438, 19), len_18410, *[unreferenced_type_warnings_18411], **kwargs_18412)
                
                int_18414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 53), 'int')
                # Applying the binary operator '>' (line 438)
                result_gt_18415 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 19), '>', len_call_result_18413, int_18414)
                
                # Testing if the type of an if condition is none (line 438)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 438, 16), result_gt_18415):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 438)
                    if_condition_18416 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 438, 16), result_gt_18415)
                    # Assigning a type to the variable 'if_condition_18416' (line 438)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 'if_condition_18416', if_condition_18416)
                    # SSA begins for if statement (line 438)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 439):
                    
                    # Assigning a Call to a Name (line 439):
                    
                    # Call to filter(...): (line 439)
                    # Processing the call arguments (line 439)

                    @norecursion
                    def _stypy_temp_lambda_28(localization, *varargs, **kwargs):
                        global module_type_store
                        # Assign values to the parameters with defaults
                        defaults = []
                        # Create a new context for function '_stypy_temp_lambda_28'
                        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_28', 439, 76, True)
                        # Passed parameters checking function
                        _stypy_temp_lambda_28.stypy_localization = localization
                        _stypy_temp_lambda_28.stypy_type_of_self = None
                        _stypy_temp_lambda_28.stypy_type_store = module_type_store
                        _stypy_temp_lambda_28.stypy_function_name = '_stypy_temp_lambda_28'
                        _stypy_temp_lambda_28.stypy_param_names_list = ['warning']
                        _stypy_temp_lambda_28.stypy_varargs_param_name = None
                        _stypy_temp_lambda_28.stypy_kwargs_param_name = None
                        _stypy_temp_lambda_28.stypy_call_defaults = defaults
                        _stypy_temp_lambda_28.stypy_call_varargs = varargs
                        _stypy_temp_lambda_28.stypy_call_kwargs = kwargs
                        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_28', ['warning'], None, None, defaults, varargs, kwargs)

                        if is_error_type(arguments):
                            # Destroy the current context
                            module_type_store = module_type_store.close_function_context()
                            return arguments

                        # Stacktrace push for error reporting
                        localization.set_stack_trace('_stypy_temp_lambda_28', ['warning'], arguments)
                        # Default return type storage variable (SSA)
                        # Assigning a type to the variable 'stypy_return_type'
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                        
                        
                        # ################# Begin of the lambda function code ##################

                        
                        # Evaluating a boolean operation
                        
                        # Getting the type of 'warning' (line 440)
                        warning_18418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 76), 'warning', False)
                        # Obtaining the member 'context' of a type (line 440)
                        context_18419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 76), warning_18418, 'context')
                        
                        # Call to get_context(...): (line 440)
                        # Processing the call keyword arguments (line 440)
                        kwargs_18422 = {}
                        # Getting the type of 'self' (line 440)
                        self_18420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 95), 'self', False)
                        # Obtaining the member 'get_context' of a type (line 440)
                        get_context_18421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 95), self_18420, 'get_context')
                        # Calling get_context(args, kwargs) (line 440)
                        get_context_call_result_18423 = invoke(stypy.reporting.localization.Localization(__file__, 440, 95), get_context_18421, *[], **kwargs_18422)
                        
                        # Applying the binary operator '==' (line 440)
                        result_eq_18424 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 76), '==', context_18419, get_context_call_result_18423)
                        
                        
                        # Getting the type of 'warning' (line 441)
                        warning_18425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 76), 'warning', False)
                        # Obtaining the member 'name' of a type (line 441)
                        name_18426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 76), warning_18425, 'name')
                        # Getting the type of 'name' (line 441)
                        name_18427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 92), 'name', False)
                        # Applying the binary operator '==' (line 441)
                        result_eq_18428 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 76), '==', name_18426, name_18427)
                        
                        # Applying the binary operator 'and' (line 440)
                        result_and_keyword_18429 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 76), 'and', result_eq_18424, result_eq_18428)
                        
                        # Assigning the return type of the lambda function
                        # Assigning a type to the variable 'stypy_return_type' (line 439)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 76), 'stypy_return_type', result_and_keyword_18429)
                        
                        # ################# End of the lambda function code ##################

                        # Stacktrace pop (error reporting)
                        localization.unset_stack_trace()
                        
                        # Storing the return type of function '_stypy_temp_lambda_28' in the type store
                        # Getting the type of 'stypy_return_type' (line 439)
                        stypy_return_type_18430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 76), 'stypy_return_type')
                        module_type_store.store_return_type_of_current_context(stypy_return_type_18430)
                        
                        # Destroy the current context
                        module_type_store = module_type_store.close_function_context()
                        
                        # Return type of the function '_stypy_temp_lambda_28'
                        return stypy_return_type_18430

                    # Assigning a type to the variable '_stypy_temp_lambda_28' (line 439)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 76), '_stypy_temp_lambda_28', _stypy_temp_lambda_28)
                    # Getting the type of '_stypy_temp_lambda_28' (line 439)
                    _stypy_temp_lambda_28_18431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 76), '_stypy_temp_lambda_28')
                    # Getting the type of 'unreferenced_type_warnings' (line 442)
                    unreferenced_type_warnings_18432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 76), 'unreferenced_type_warnings', False)
                    # Processing the call keyword arguments (line 439)
                    kwargs_18433 = {}
                    # Getting the type of 'filter' (line 439)
                    filter_18417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 69), 'filter', False)
                    # Calling filter(args, kwargs) (line 439)
                    filter_call_result_18434 = invoke(stypy.reporting.localization.Localization(__file__, 439, 69), filter_18417, *[_stypy_temp_lambda_28_18431, unreferenced_type_warnings_18432], **kwargs_18433)
                    
                    # Assigning a type to the variable 'our_unreferenced_type_warnings_in_this_context' (line 439)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 20), 'our_unreferenced_type_warnings_in_this_context', filter_call_result_18434)
                    
                    # Getting the type of 'our_unreferenced_type_warnings_in_this_context' (line 444)
                    our_unreferenced_type_warnings_in_this_context_18435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 31), 'our_unreferenced_type_warnings_in_this_context')
                    # Assigning a type to the variable 'our_unreferenced_type_warnings_in_this_context_18435' (line 444)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 20), 'our_unreferenced_type_warnings_in_this_context_18435', our_unreferenced_type_warnings_in_this_context_18435)
                    # Testing if the for loop is going to be iterated (line 444)
                    # Testing the type of a for loop iterable (line 444)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 444, 20), our_unreferenced_type_warnings_in_this_context_18435)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 444, 20), our_unreferenced_type_warnings_in_this_context_18435):
                        # Getting the type of the for loop variable (line 444)
                        for_loop_var_18436 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 444, 20), our_unreferenced_type_warnings_in_this_context_18435)
                        # Assigning a type to the variable 'utw' (line 444)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 20), 'utw', for_loop_var_18436)
                        # SSA begins for a for statement (line 444)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Call to TypeError(...): (line 445)
                        # Processing the call arguments (line 445)
                        # Getting the type of 'localization' (line 445)
                        localization_18438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 34), 'localization', False)
                        
                        # Call to format(...): (line 445)
                        # Processing the call arguments (line 445)
                        # Getting the type of 'name' (line 446)
                        name_18441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 86), 'name', False)
                        # Processing the call keyword arguments (line 445)
                        kwargs_18442 = {}
                        str_18439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 48), 'str', "UnboundLocalError: local variable '{0}' referenced before assignment")
                        # Obtaining the member 'format' of a type (line 445)
                        format_18440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 48), str_18439, 'format')
                        # Calling format(args, kwargs) (line 445)
                        format_call_result_18443 = invoke(stypy.reporting.localization.Localization(__file__, 445, 48), format_18440, *[name_18441], **kwargs_18442)
                        
                        # Processing the call keyword arguments (line 445)
                        kwargs_18444 = {}
                        # Getting the type of 'TypeError' (line 445)
                        TypeError_18437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 24), 'TypeError', False)
                        # Calling TypeError(args, kwargs) (line 445)
                        TypeError_call_result_18445 = invoke(stypy.reporting.localization.Localization(__file__, 445, 24), TypeError_18437, *[localization_18438, format_call_result_18443], **kwargs_18444)
                        
                        
                        # Call to remove(...): (line 447)
                        # Processing the call arguments (line 447)
                        # Getting the type of 'utw' (line 447)
                        utw_18449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 52), 'utw', False)
                        # Processing the call keyword arguments (line 447)
                        kwargs_18450 = {}
                        # Getting the type of 'TypeWarning' (line 447)
                        TypeWarning_18446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 24), 'TypeWarning', False)
                        # Obtaining the member 'warnings' of a type (line 447)
                        warnings_18447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 24), TypeWarning_18446, 'warnings')
                        # Obtaining the member 'remove' of a type (line 447)
                        remove_18448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 24), warnings_18447, 'remove')
                        # Calling remove(args, kwargs) (line 447)
                        remove_call_result_18451 = invoke(stypy.reporting.localization.Localization(__file__, 447, 24), remove_18448, *[utw_18449], **kwargs_18450)
                        
                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    
                    
                    # Call to len(...): (line 450)
                    # Processing the call arguments (line 450)
                    # Getting the type of 'our_unreferenced_type_warnings_in_this_context' (line 450)
                    our_unreferenced_type_warnings_in_this_context_18453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 27), 'our_unreferenced_type_warnings_in_this_context', False)
                    # Processing the call keyword arguments (line 450)
                    kwargs_18454 = {}
                    # Getting the type of 'len' (line 450)
                    len_18452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 23), 'len', False)
                    # Calling len(args, kwargs) (line 450)
                    len_call_result_18455 = invoke(stypy.reporting.localization.Localization(__file__, 450, 23), len_18452, *[our_unreferenced_type_warnings_in_this_context_18453], **kwargs_18454)
                    
                    int_18456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 77), 'int')
                    # Applying the binary operator '>' (line 450)
                    result_gt_18457 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 23), '>', len_call_result_18455, int_18456)
                    
                    # Testing if the type of an if condition is none (line 450)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 450, 20), result_gt_18457):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 450)
                        if_condition_18458 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 450, 20), result_gt_18457)
                        # Assigning a type to the variable 'if_condition_18458' (line 450)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 20), 'if_condition_18458', if_condition_18458)
                        # SSA begins for if statement (line 450)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to set_type_of(...): (line 451)
                        # Processing the call arguments (line 451)
                        # Getting the type of 'name' (line 451)
                        name_18464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 55), 'name', False)
                        
                        # Call to TypeError(...): (line 451)
                        # Processing the call arguments (line 451)
                        # Getting the type of 'localization' (line 451)
                        localization_18466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 71), 'localization', False)
                        
                        # Call to format(...): (line 451)
                        # Processing the call arguments (line 451)
                        # Getting the type of 'name' (line 452)
                        name_18469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 113), 'name', False)
                        # Processing the call keyword arguments (line 451)
                        kwargs_18470 = {}
                        str_18467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 85), 'str', "Attempted to use '{0}' previously to its definition")
                        # Obtaining the member 'format' of a type (line 451)
                        format_18468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 85), str_18467, 'format')
                        # Calling format(args, kwargs) (line 451)
                        format_call_result_18471 = invoke(stypy.reporting.localization.Localization(__file__, 451, 85), format_18468, *[name_18469], **kwargs_18470)
                        
                        # Processing the call keyword arguments (line 451)
                        kwargs_18472 = {}
                        # Getting the type of 'TypeError' (line 451)
                        TypeError_18465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 61), 'TypeError', False)
                        # Calling TypeError(args, kwargs) (line 451)
                        TypeError_call_result_18473 = invoke(stypy.reporting.localization.Localization(__file__, 451, 61), TypeError_18465, *[localization_18466, format_call_result_18471], **kwargs_18472)
                        
                        # Getting the type of 'localization' (line 453)
                        localization_18474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 55), 'localization', False)
                        # Processing the call keyword arguments (line 451)
                        kwargs_18475 = {}
                        
                        # Call to get_context(...): (line 451)
                        # Processing the call keyword arguments (line 451)
                        kwargs_18461 = {}
                        # Getting the type of 'self' (line 451)
                        self_18459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 24), 'self', False)
                        # Obtaining the member 'get_context' of a type (line 451)
                        get_context_18460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 24), self_18459, 'get_context')
                        # Calling get_context(args, kwargs) (line 451)
                        get_context_call_result_18462 = invoke(stypy.reporting.localization.Localization(__file__, 451, 24), get_context_18460, *[], **kwargs_18461)
                        
                        # Obtaining the member 'set_type_of' of a type (line 451)
                        set_type_of_18463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 24), get_context_call_result_18462, 'set_type_of')
                        # Calling set_type_of(args, kwargs) (line 451)
                        set_type_of_call_result_18476 = invoke(stypy.reporting.localization.Localization(__file__, 451, 24), set_type_of_18463, *[name_18464, TypeError_call_result_18473, localization_18474], **kwargs_18475)
                        
                        
                        # Call to get_type_of(...): (line 454)
                        # Processing the call arguments (line 454)
                        # Getting the type of 'name' (line 454)
                        name_18482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 62), 'name', False)
                        # Processing the call keyword arguments (line 454)
                        kwargs_18483 = {}
                        
                        # Call to get_context(...): (line 454)
                        # Processing the call keyword arguments (line 454)
                        kwargs_18479 = {}
                        # Getting the type of 'self' (line 454)
                        self_18477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 31), 'self', False)
                        # Obtaining the member 'get_context' of a type (line 454)
                        get_context_18478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 31), self_18477, 'get_context')
                        # Calling get_context(args, kwargs) (line 454)
                        get_context_call_result_18480 = invoke(stypy.reporting.localization.Localization(__file__, 454, 31), get_context_18478, *[], **kwargs_18479)
                        
                        # Obtaining the member 'get_type_of' of a type (line 454)
                        get_type_of_18481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 31), get_context_call_result_18480, 'get_type_of')
                        # Calling get_type_of(args, kwargs) (line 454)
                        get_type_of_call_result_18484 = invoke(stypy.reporting.localization.Localization(__file__, 454, 31), get_type_of_18481, *[name_18482], **kwargs_18483)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 454)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 24), 'stypy_return_type', get_type_of_call_result_18484)
                        # SSA join for if statement (line 450)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 438)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Call to a Tuple (line 456):
                
                # Assigning a Call to a Name:
                
                # Call to contains_an_undefined_type(...): (line 456)
                # Processing the call arguments (line 456)
                # Getting the type of 'type_' (line 457)
                type__18488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 47), 'type_', False)
                # Processing the call keyword arguments (line 456)
                kwargs_18489 = {}
                # Getting the type of 'type_inference_proxy' (line 456)
                type_inference_proxy_18485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 58), 'type_inference_proxy', False)
                # Obtaining the member 'TypeInferenceProxy' of a type (line 456)
                TypeInferenceProxy_18486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 58), type_inference_proxy_18485, 'TypeInferenceProxy')
                # Obtaining the member 'contains_an_undefined_type' of a type (line 456)
                contains_an_undefined_type_18487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 58), TypeInferenceProxy_18486, 'contains_an_undefined_type')
                # Calling contains_an_undefined_type(args, kwargs) (line 456)
                contains_an_undefined_type_call_result_18490 = invoke(stypy.reporting.localization.Localization(__file__, 456, 58), contains_an_undefined_type_18487, *[type__18488], **kwargs_18489)
                
                # Assigning a type to the variable 'call_assignment_17651' (line 456)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17651', contains_an_undefined_type_call_result_18490)
                
                # Assigning a Call to a Name (line 456):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_17651' (line 456)
                call_assignment_17651_18491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17651', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_18492 = stypy_get_value_from_tuple(call_assignment_17651_18491, 2, 0)
                
                # Assigning a type to the variable 'call_assignment_17652' (line 456)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17652', stypy_get_value_from_tuple_call_result_18492)
                
                # Assigning a Name to a Name (line 456):
                # Getting the type of 'call_assignment_17652' (line 456)
                call_assignment_17652_18493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17652')
                # Assigning a type to the variable 'contains_undefined' (line 456)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'contains_undefined', call_assignment_17652_18493)
                
                # Assigning a Call to a Name (line 456):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_17651' (line 456)
                call_assignment_17651_18494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17651', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_18495 = stypy_get_value_from_tuple(call_assignment_17651_18494, 2, 1)
                
                # Assigning a type to the variable 'call_assignment_17653' (line 456)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17653', stypy_get_value_from_tuple_call_result_18495)
                
                # Assigning a Name to a Name (line 456):
                # Getting the type of 'call_assignment_17653' (line 456)
                call_assignment_17653_18496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17653')
                # Assigning a type to the variable 'more_types_in_value' (line 456)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 36), 'more_types_in_value', call_assignment_17653_18496)
                # Getting the type of 'contains_undefined' (line 458)
                contains_undefined_18497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 19), 'contains_undefined')
                # Testing if the type of an if condition is none (line 458)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 458, 16), contains_undefined_18497):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 458)
                    if_condition_18498 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 458, 16), contains_undefined_18497)
                    # Assigning a type to the variable 'if_condition_18498' (line 458)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 16), 'if_condition_18498', if_condition_18498)
                    # SSA begins for if statement (line 458)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'more_types_in_value' (line 459)
                    more_types_in_value_18499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 23), 'more_types_in_value')
                    int_18500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 46), 'int')
                    # Applying the binary operator '==' (line 459)
                    result_eq_18501 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 23), '==', more_types_in_value_18499, int_18500)
                    
                    # Testing if the type of an if condition is none (line 459)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 459, 20), result_eq_18501):
                        
                        # Call to instance(...): (line 463)
                        # Processing the call arguments (line 463)
                        # Getting the type of 'localization' (line 463)
                        localization_18514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 45), 'localization', False)
                        
                        # Call to format(...): (line 464)
                        # Processing the call arguments (line 464)
                        # Getting the type of 'name' (line 465)
                        name_18517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 52), 'name', False)
                        # Processing the call keyword arguments (line 464)
                        kwargs_18518 = {}
                        str_18515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 45), 'str', "Potentialy assigning to '{0}' the value of an undefined variable")
                        # Obtaining the member 'format' of a type (line 464)
                        format_18516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 45), str_18515, 'format')
                        # Calling format(args, kwargs) (line 464)
                        format_call_result_18519 = invoke(stypy.reporting.localization.Localization(__file__, 464, 45), format_18516, *[name_18517], **kwargs_18518)
                        
                        # Processing the call keyword arguments (line 463)
                        kwargs_18520 = {}
                        # Getting the type of 'TypeWarning' (line 463)
                        TypeWarning_18512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 24), 'TypeWarning', False)
                        # Obtaining the member 'instance' of a type (line 463)
                        instance_18513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 24), TypeWarning_18512, 'instance')
                        # Calling instance(args, kwargs) (line 463)
                        instance_call_result_18521 = invoke(stypy.reporting.localization.Localization(__file__, 463, 24), instance_18513, *[localization_18514, format_call_result_18519], **kwargs_18520)
                        
                    else:
                        
                        # Testing the type of an if condition (line 459)
                        if_condition_18502 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 459, 20), result_eq_18501)
                        # Assigning a type to the variable 'if_condition_18502' (line 459)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 20), 'if_condition_18502', if_condition_18502)
                        # SSA begins for if statement (line 459)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to TypeError(...): (line 460)
                        # Processing the call arguments (line 460)
                        # Getting the type of 'localization' (line 460)
                        localization_18504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 34), 'localization', False)
                        
                        # Call to format(...): (line 460)
                        # Processing the call arguments (line 460)
                        # Getting the type of 'name' (line 461)
                        name_18507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 41), 'name', False)
                        # Processing the call keyword arguments (line 460)
                        kwargs_18508 = {}
                        str_18505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 48), 'str', "Assigning to '{0}' the value of an undefined variable")
                        # Obtaining the member 'format' of a type (line 460)
                        format_18506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 48), str_18505, 'format')
                        # Calling format(args, kwargs) (line 460)
                        format_call_result_18509 = invoke(stypy.reporting.localization.Localization(__file__, 460, 48), format_18506, *[name_18507], **kwargs_18508)
                        
                        # Processing the call keyword arguments (line 460)
                        kwargs_18510 = {}
                        # Getting the type of 'TypeError' (line 460)
                        TypeError_18503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 24), 'TypeError', False)
                        # Calling TypeError(args, kwargs) (line 460)
                        TypeError_call_result_18511 = invoke(stypy.reporting.localization.Localization(__file__, 460, 24), TypeError_18503, *[localization_18504, format_call_result_18509], **kwargs_18510)
                        
                        # SSA branch for the else part of an if statement (line 459)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to instance(...): (line 463)
                        # Processing the call arguments (line 463)
                        # Getting the type of 'localization' (line 463)
                        localization_18514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 45), 'localization', False)
                        
                        # Call to format(...): (line 464)
                        # Processing the call arguments (line 464)
                        # Getting the type of 'name' (line 465)
                        name_18517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 52), 'name', False)
                        # Processing the call keyword arguments (line 464)
                        kwargs_18518 = {}
                        str_18515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 45), 'str', "Potentialy assigning to '{0}' the value of an undefined variable")
                        # Obtaining the member 'format' of a type (line 464)
                        format_18516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 45), str_18515, 'format')
                        # Calling format(args, kwargs) (line 464)
                        format_call_result_18519 = invoke(stypy.reporting.localization.Localization(__file__, 464, 45), format_18516, *[name_18517], **kwargs_18518)
                        
                        # Processing the call keyword arguments (line 463)
                        kwargs_18520 = {}
                        # Getting the type of 'TypeWarning' (line 463)
                        TypeWarning_18512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 24), 'TypeWarning', False)
                        # Obtaining the member 'instance' of a type (line 463)
                        instance_18513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 24), TypeWarning_18512, 'instance')
                        # Calling instance(args, kwargs) (line 463)
                        instance_call_result_18521 = invoke(stypy.reporting.localization.Localization(__file__, 463, 24), instance_18513, *[localization_18514, format_call_result_18519], **kwargs_18520)
                        
                        # SSA join for if statement (line 459)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 458)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Call to set_type_of(...): (line 467)
                # Processing the call arguments (line 467)
                # Getting the type of 'name' (line 467)
                name_18527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 47), 'name', False)
                # Getting the type of 'type_' (line 467)
                type__18528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 53), 'type_', False)
                # Getting the type of 'localization' (line 467)
                localization_18529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 60), 'localization', False)
                # Processing the call keyword arguments (line 467)
                kwargs_18530 = {}
                
                # Call to get_context(...): (line 467)
                # Processing the call keyword arguments (line 467)
                kwargs_18524 = {}
                # Getting the type of 'self' (line 467)
                self_18522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 16), 'self', False)
                # Obtaining the member 'get_context' of a type (line 467)
                get_context_18523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 16), self_18522, 'get_context')
                # Calling get_context(args, kwargs) (line 467)
                get_context_call_result_18525 = invoke(stypy.reporting.localization.Localization(__file__, 467, 16), get_context_18523, *[], **kwargs_18524)
                
                # Obtaining the member 'set_type_of' of a type (line 467)
                set_type_of_18526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 16), get_context_call_result_18525, 'set_type_of')
                # Calling set_type_of(args, kwargs) (line 467)
                set_type_of_call_result_18531 = invoke(stypy.reporting.localization.Localization(__file__, 467, 16), set_type_of_18526, *[name_18527, type__18528, localization_18529], **kwargs_18530)
                
                # SSA join for if statement (line 420)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 409)
            if_condition_18345 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 409, 8), exist_in_local_context_18344)
            # Assigning a type to the variable 'if_condition_18345' (line 409)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'if_condition_18345', if_condition_18345)
            # SSA begins for if statement (line 409)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'is_marked_as_global' (line 410)
            is_marked_as_global_18346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 15), 'is_marked_as_global')
            # Testing if the type of an if condition is none (line 410)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 410, 12), is_marked_as_global_18346):
                
                # Call to set_type_of(...): (line 418)
                # Processing the call arguments (line 418)
                # Getting the type of 'name' (line 418)
                name_18382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 47), 'name', False)
                # Getting the type of 'type_' (line 418)
                type__18383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 53), 'type_', False)
                # Getting the type of 'localization' (line 418)
                localization_18384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 60), 'localization', False)
                # Processing the call keyword arguments (line 418)
                kwargs_18385 = {}
                
                # Call to get_context(...): (line 418)
                # Processing the call keyword arguments (line 418)
                kwargs_18379 = {}
                # Getting the type of 'self' (line 418)
                self_18377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 16), 'self', False)
                # Obtaining the member 'get_context' of a type (line 418)
                get_context_18378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 16), self_18377, 'get_context')
                # Calling get_context(args, kwargs) (line 418)
                get_context_call_result_18380 = invoke(stypy.reporting.localization.Localization(__file__, 418, 16), get_context_18378, *[], **kwargs_18379)
                
                # Obtaining the member 'set_type_of' of a type (line 418)
                set_type_of_18381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 16), get_context_call_result_18380, 'set_type_of')
                # Calling set_type_of(args, kwargs) (line 418)
                set_type_of_call_result_18386 = invoke(stypy.reporting.localization.Localization(__file__, 418, 16), set_type_of_18381, *[name_18382, type__18383, localization_18384], **kwargs_18385)
                
            else:
                
                # Testing the type of an if condition (line 410)
                if_condition_18347 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 410, 12), is_marked_as_global_18346)
                # Assigning a type to the variable 'if_condition_18347' (line 410)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'if_condition_18347', if_condition_18347)
                # SSA begins for if statement (line 410)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to set_type_of(...): (line 411)
                # Processing the call arguments (line 411)
                # Getting the type of 'name' (line 411)
                name_18350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 43), 'name', False)
                # Getting the type of 'type_' (line 411)
                type__18351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 49), 'type_', False)
                # Getting the type of 'localization' (line 411)
                localization_18352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 56), 'localization', False)
                # Processing the call keyword arguments (line 411)
                kwargs_18353 = {}
                # Getting the type of 'global_context' (line 411)
                global_context_18348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 16), 'global_context', False)
                # Obtaining the member 'set_type_of' of a type (line 411)
                set_type_of_18349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 16), global_context_18348, 'set_type_of')
                # Calling set_type_of(args, kwargs) (line 411)
                set_type_of_call_result_18354 = invoke(stypy.reporting.localization.Localization(__file__, 411, 16), set_type_of_18349, *[name_18350, type__18351, localization_18352], **kwargs_18353)
                
                # Deleting a member
                
                # Call to get_context(...): (line 412)
                # Processing the call keyword arguments (line 412)
                kwargs_18357 = {}
                # Getting the type of 'self' (line 412)
                self_18355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 20), 'self', False)
                # Obtaining the member 'get_context' of a type (line 412)
                get_context_18356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 20), self_18355, 'get_context')
                # Calling get_context(args, kwargs) (line 412)
                get_context_call_result_18358 = invoke(stypy.reporting.localization.Localization(__file__, 412, 20), get_context_18356, *[], **kwargs_18357)
                
                # Obtaining the member 'types_of' of a type (line 412)
                types_of_18359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 20), get_context_call_result_18358, 'types_of')
                
                # Obtaining the type of the subscript
                # Getting the type of 'name' (line 412)
                name_18360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 48), 'name')
                
                # Call to get_context(...): (line 412)
                # Processing the call keyword arguments (line 412)
                kwargs_18363 = {}
                # Getting the type of 'self' (line 412)
                self_18361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 20), 'self', False)
                # Obtaining the member 'get_context' of a type (line 412)
                get_context_18362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 20), self_18361, 'get_context')
                # Calling get_context(args, kwargs) (line 412)
                get_context_call_result_18364 = invoke(stypy.reporting.localization.Localization(__file__, 412, 20), get_context_18362, *[], **kwargs_18363)
                
                # Obtaining the member 'types_of' of a type (line 412)
                types_of_18365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 20), get_context_call_result_18364, 'types_of')
                # Obtaining the member '__getitem__' of a type (line 412)
                getitem___18366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 20), types_of_18365, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 412)
                subscript_call_result_18367 = invoke(stypy.reporting.localization.Localization(__file__, 412, 20), getitem___18366, name_18360)
                
                del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 16), types_of_18359, subscript_call_result_18367)
                
                # Call to TypeWarning(...): (line 413)
                # Processing the call arguments (line 413)
                # Getting the type of 'localization' (line 413)
                localization_18369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 35), 'localization', False)
                
                # Call to format(...): (line 413)
                # Processing the call arguments (line 413)
                # Getting the type of 'name' (line 416)
                name_18372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 90), 'name', False)
                # Processing the call keyword arguments (line 413)
                kwargs_18373 = {}
                str_18370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 49), 'str', "You used the global keyword on '{0}' after assigning a value to it. It is valid, but will throw a warning on execution. Please consider moving the global statement before any assignment is done to '{0}'")
                # Obtaining the member 'format' of a type (line 413)
                format_18371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 49), str_18370, 'format')
                # Calling format(args, kwargs) (line 413)
                format_call_result_18374 = invoke(stypy.reporting.localization.Localization(__file__, 413, 49), format_18371, *[name_18372], **kwargs_18373)
                
                # Processing the call keyword arguments (line 413)
                kwargs_18375 = {}
                # Getting the type of 'TypeWarning' (line 413)
                TypeWarning_18368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 23), 'TypeWarning', False)
                # Calling TypeWarning(args, kwargs) (line 413)
                TypeWarning_call_result_18376 = invoke(stypy.reporting.localization.Localization(__file__, 413, 23), TypeWarning_18368, *[localization_18369, format_call_result_18374], **kwargs_18375)
                
                # Assigning a type to the variable 'stypy_return_type' (line 413)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 16), 'stypy_return_type', TypeWarning_call_result_18376)
                # SSA branch for the else part of an if statement (line 410)
                module_type_store.open_ssa_branch('else')
                
                # Call to set_type_of(...): (line 418)
                # Processing the call arguments (line 418)
                # Getting the type of 'name' (line 418)
                name_18382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 47), 'name', False)
                # Getting the type of 'type_' (line 418)
                type__18383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 53), 'type_', False)
                # Getting the type of 'localization' (line 418)
                localization_18384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 60), 'localization', False)
                # Processing the call keyword arguments (line 418)
                kwargs_18385 = {}
                
                # Call to get_context(...): (line 418)
                # Processing the call keyword arguments (line 418)
                kwargs_18379 = {}
                # Getting the type of 'self' (line 418)
                self_18377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 16), 'self', False)
                # Obtaining the member 'get_context' of a type (line 418)
                get_context_18378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 16), self_18377, 'get_context')
                # Calling get_context(args, kwargs) (line 418)
                get_context_call_result_18380 = invoke(stypy.reporting.localization.Localization(__file__, 418, 16), get_context_18378, *[], **kwargs_18379)
                
                # Obtaining the member 'set_type_of' of a type (line 418)
                set_type_of_18381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 16), get_context_call_result_18380, 'set_type_of')
                # Calling set_type_of(args, kwargs) (line 418)
                set_type_of_call_result_18386 = invoke(stypy.reporting.localization.Localization(__file__, 418, 16), set_type_of_18381, *[name_18382, type__18383, localization_18384], **kwargs_18385)
                
                # SSA join for if statement (line 410)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA branch for the else part of an if statement (line 409)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'is_marked_as_global' (line 420)
            is_marked_as_global_18387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 15), 'is_marked_as_global')
            # Testing if the type of an if condition is none (line 420)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 420, 12), is_marked_as_global_18387):
                str_18396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, (-1)), 'str', 'Special case:\n                    If:\n                        - A variable do not exist in the local context\n                        - This variable is not marked as global\n                        - There exist unreferenced type warnings in this scope typed to this variable.\n                    Then:\n                        - For each unreferenced type warning found:\n                            - Generate a unreferenced variable error with the warning data\n                            - Delete warning\n                            - Mark the type of the variable as ErrorType\n                ')
                
                # Assigning a Call to a Name (line 434):
                
                # Assigning a Call to a Name (line 434):
                
                # Call to filter(...): (line 434)
                # Processing the call arguments (line 434)

                @norecursion
                def _stypy_temp_lambda_27(localization, *varargs, **kwargs):
                    global module_type_store
                    # Assign values to the parameters with defaults
                    defaults = []
                    # Create a new context for function '_stypy_temp_lambda_27'
                    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_27', 434, 52, True)
                    # Passed parameters checking function
                    _stypy_temp_lambda_27.stypy_localization = localization
                    _stypy_temp_lambda_27.stypy_type_of_self = None
                    _stypy_temp_lambda_27.stypy_type_store = module_type_store
                    _stypy_temp_lambda_27.stypy_function_name = '_stypy_temp_lambda_27'
                    _stypy_temp_lambda_27.stypy_param_names_list = ['warning']
                    _stypy_temp_lambda_27.stypy_varargs_param_name = None
                    _stypy_temp_lambda_27.stypy_kwargs_param_name = None
                    _stypy_temp_lambda_27.stypy_call_defaults = defaults
                    _stypy_temp_lambda_27.stypy_call_varargs = varargs
                    _stypy_temp_lambda_27.stypy_call_kwargs = kwargs
                    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_27', ['warning'], None, None, defaults, varargs, kwargs)

                    if is_error_type(arguments):
                        # Destroy the current context
                        module_type_store = module_type_store.close_function_context()
                        return arguments

                    # Stacktrace push for error reporting
                    localization.set_stack_trace('_stypy_temp_lambda_27', ['warning'], arguments)
                    # Default return type storage variable (SSA)
                    # Assigning a type to the variable 'stypy_return_type'
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                    
                    
                    # ################# Begin of the lambda function code ##################

                    
                    # Getting the type of 'warning' (line 435)
                    warning_18398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 52), 'warning', False)
                    # Obtaining the member '__class__' of a type (line 435)
                    class___18399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 52), warning_18398, '__class__')
                    # Getting the type of 'UnreferencedLocalVariableTypeWarning' (line 435)
                    UnreferencedLocalVariableTypeWarning_18400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 73), 'UnreferencedLocalVariableTypeWarning', False)
                    # Applying the binary operator '==' (line 435)
                    result_eq_18401 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 52), '==', class___18399, UnreferencedLocalVariableTypeWarning_18400)
                    
                    # Assigning the return type of the lambda function
                    # Assigning a type to the variable 'stypy_return_type' (line 434)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 52), 'stypy_return_type', result_eq_18401)
                    
                    # ################# End of the lambda function code ##################

                    # Stacktrace pop (error reporting)
                    localization.unset_stack_trace()
                    
                    # Storing the return type of function '_stypy_temp_lambda_27' in the type store
                    # Getting the type of 'stypy_return_type' (line 434)
                    stypy_return_type_18402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 52), 'stypy_return_type')
                    module_type_store.store_return_type_of_current_context(stypy_return_type_18402)
                    
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    
                    # Return type of the function '_stypy_temp_lambda_27'
                    return stypy_return_type_18402

                # Assigning a type to the variable '_stypy_temp_lambda_27' (line 434)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 52), '_stypy_temp_lambda_27', _stypy_temp_lambda_27)
                # Getting the type of '_stypy_temp_lambda_27' (line 434)
                _stypy_temp_lambda_27_18403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 52), '_stypy_temp_lambda_27')
                
                # Call to get_warning_msgs(...): (line 436)
                # Processing the call keyword arguments (line 436)
                kwargs_18406 = {}
                # Getting the type of 'TypeWarning' (line 436)
                TypeWarning_18404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 52), 'TypeWarning', False)
                # Obtaining the member 'get_warning_msgs' of a type (line 436)
                get_warning_msgs_18405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 52), TypeWarning_18404, 'get_warning_msgs')
                # Calling get_warning_msgs(args, kwargs) (line 436)
                get_warning_msgs_call_result_18407 = invoke(stypy.reporting.localization.Localization(__file__, 436, 52), get_warning_msgs_18405, *[], **kwargs_18406)
                
                # Processing the call keyword arguments (line 434)
                kwargs_18408 = {}
                # Getting the type of 'filter' (line 434)
                filter_18397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 45), 'filter', False)
                # Calling filter(args, kwargs) (line 434)
                filter_call_result_18409 = invoke(stypy.reporting.localization.Localization(__file__, 434, 45), filter_18397, *[_stypy_temp_lambda_27_18403, get_warning_msgs_call_result_18407], **kwargs_18408)
                
                # Assigning a type to the variable 'unreferenced_type_warnings' (line 434)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 16), 'unreferenced_type_warnings', filter_call_result_18409)
                
                
                # Call to len(...): (line 438)
                # Processing the call arguments (line 438)
                # Getting the type of 'unreferenced_type_warnings' (line 438)
                unreferenced_type_warnings_18411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 23), 'unreferenced_type_warnings', False)
                # Processing the call keyword arguments (line 438)
                kwargs_18412 = {}
                # Getting the type of 'len' (line 438)
                len_18410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 19), 'len', False)
                # Calling len(args, kwargs) (line 438)
                len_call_result_18413 = invoke(stypy.reporting.localization.Localization(__file__, 438, 19), len_18410, *[unreferenced_type_warnings_18411], **kwargs_18412)
                
                int_18414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 53), 'int')
                # Applying the binary operator '>' (line 438)
                result_gt_18415 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 19), '>', len_call_result_18413, int_18414)
                
                # Testing if the type of an if condition is none (line 438)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 438, 16), result_gt_18415):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 438)
                    if_condition_18416 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 438, 16), result_gt_18415)
                    # Assigning a type to the variable 'if_condition_18416' (line 438)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 'if_condition_18416', if_condition_18416)
                    # SSA begins for if statement (line 438)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 439):
                    
                    # Assigning a Call to a Name (line 439):
                    
                    # Call to filter(...): (line 439)
                    # Processing the call arguments (line 439)

                    @norecursion
                    def _stypy_temp_lambda_28(localization, *varargs, **kwargs):
                        global module_type_store
                        # Assign values to the parameters with defaults
                        defaults = []
                        # Create a new context for function '_stypy_temp_lambda_28'
                        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_28', 439, 76, True)
                        # Passed parameters checking function
                        _stypy_temp_lambda_28.stypy_localization = localization
                        _stypy_temp_lambda_28.stypy_type_of_self = None
                        _stypy_temp_lambda_28.stypy_type_store = module_type_store
                        _stypy_temp_lambda_28.stypy_function_name = '_stypy_temp_lambda_28'
                        _stypy_temp_lambda_28.stypy_param_names_list = ['warning']
                        _stypy_temp_lambda_28.stypy_varargs_param_name = None
                        _stypy_temp_lambda_28.stypy_kwargs_param_name = None
                        _stypy_temp_lambda_28.stypy_call_defaults = defaults
                        _stypy_temp_lambda_28.stypy_call_varargs = varargs
                        _stypy_temp_lambda_28.stypy_call_kwargs = kwargs
                        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_28', ['warning'], None, None, defaults, varargs, kwargs)

                        if is_error_type(arguments):
                            # Destroy the current context
                            module_type_store = module_type_store.close_function_context()
                            return arguments

                        # Stacktrace push for error reporting
                        localization.set_stack_trace('_stypy_temp_lambda_28', ['warning'], arguments)
                        # Default return type storage variable (SSA)
                        # Assigning a type to the variable 'stypy_return_type'
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                        
                        
                        # ################# Begin of the lambda function code ##################

                        
                        # Evaluating a boolean operation
                        
                        # Getting the type of 'warning' (line 440)
                        warning_18418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 76), 'warning', False)
                        # Obtaining the member 'context' of a type (line 440)
                        context_18419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 76), warning_18418, 'context')
                        
                        # Call to get_context(...): (line 440)
                        # Processing the call keyword arguments (line 440)
                        kwargs_18422 = {}
                        # Getting the type of 'self' (line 440)
                        self_18420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 95), 'self', False)
                        # Obtaining the member 'get_context' of a type (line 440)
                        get_context_18421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 95), self_18420, 'get_context')
                        # Calling get_context(args, kwargs) (line 440)
                        get_context_call_result_18423 = invoke(stypy.reporting.localization.Localization(__file__, 440, 95), get_context_18421, *[], **kwargs_18422)
                        
                        # Applying the binary operator '==' (line 440)
                        result_eq_18424 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 76), '==', context_18419, get_context_call_result_18423)
                        
                        
                        # Getting the type of 'warning' (line 441)
                        warning_18425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 76), 'warning', False)
                        # Obtaining the member 'name' of a type (line 441)
                        name_18426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 76), warning_18425, 'name')
                        # Getting the type of 'name' (line 441)
                        name_18427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 92), 'name', False)
                        # Applying the binary operator '==' (line 441)
                        result_eq_18428 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 76), '==', name_18426, name_18427)
                        
                        # Applying the binary operator 'and' (line 440)
                        result_and_keyword_18429 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 76), 'and', result_eq_18424, result_eq_18428)
                        
                        # Assigning the return type of the lambda function
                        # Assigning a type to the variable 'stypy_return_type' (line 439)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 76), 'stypy_return_type', result_and_keyword_18429)
                        
                        # ################# End of the lambda function code ##################

                        # Stacktrace pop (error reporting)
                        localization.unset_stack_trace()
                        
                        # Storing the return type of function '_stypy_temp_lambda_28' in the type store
                        # Getting the type of 'stypy_return_type' (line 439)
                        stypy_return_type_18430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 76), 'stypy_return_type')
                        module_type_store.store_return_type_of_current_context(stypy_return_type_18430)
                        
                        # Destroy the current context
                        module_type_store = module_type_store.close_function_context()
                        
                        # Return type of the function '_stypy_temp_lambda_28'
                        return stypy_return_type_18430

                    # Assigning a type to the variable '_stypy_temp_lambda_28' (line 439)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 76), '_stypy_temp_lambda_28', _stypy_temp_lambda_28)
                    # Getting the type of '_stypy_temp_lambda_28' (line 439)
                    _stypy_temp_lambda_28_18431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 76), '_stypy_temp_lambda_28')
                    # Getting the type of 'unreferenced_type_warnings' (line 442)
                    unreferenced_type_warnings_18432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 76), 'unreferenced_type_warnings', False)
                    # Processing the call keyword arguments (line 439)
                    kwargs_18433 = {}
                    # Getting the type of 'filter' (line 439)
                    filter_18417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 69), 'filter', False)
                    # Calling filter(args, kwargs) (line 439)
                    filter_call_result_18434 = invoke(stypy.reporting.localization.Localization(__file__, 439, 69), filter_18417, *[_stypy_temp_lambda_28_18431, unreferenced_type_warnings_18432], **kwargs_18433)
                    
                    # Assigning a type to the variable 'our_unreferenced_type_warnings_in_this_context' (line 439)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 20), 'our_unreferenced_type_warnings_in_this_context', filter_call_result_18434)
                    
                    # Getting the type of 'our_unreferenced_type_warnings_in_this_context' (line 444)
                    our_unreferenced_type_warnings_in_this_context_18435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 31), 'our_unreferenced_type_warnings_in_this_context')
                    # Assigning a type to the variable 'our_unreferenced_type_warnings_in_this_context_18435' (line 444)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 20), 'our_unreferenced_type_warnings_in_this_context_18435', our_unreferenced_type_warnings_in_this_context_18435)
                    # Testing if the for loop is going to be iterated (line 444)
                    # Testing the type of a for loop iterable (line 444)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 444, 20), our_unreferenced_type_warnings_in_this_context_18435)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 444, 20), our_unreferenced_type_warnings_in_this_context_18435):
                        # Getting the type of the for loop variable (line 444)
                        for_loop_var_18436 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 444, 20), our_unreferenced_type_warnings_in_this_context_18435)
                        # Assigning a type to the variable 'utw' (line 444)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 20), 'utw', for_loop_var_18436)
                        # SSA begins for a for statement (line 444)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Call to TypeError(...): (line 445)
                        # Processing the call arguments (line 445)
                        # Getting the type of 'localization' (line 445)
                        localization_18438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 34), 'localization', False)
                        
                        # Call to format(...): (line 445)
                        # Processing the call arguments (line 445)
                        # Getting the type of 'name' (line 446)
                        name_18441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 86), 'name', False)
                        # Processing the call keyword arguments (line 445)
                        kwargs_18442 = {}
                        str_18439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 48), 'str', "UnboundLocalError: local variable '{0}' referenced before assignment")
                        # Obtaining the member 'format' of a type (line 445)
                        format_18440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 48), str_18439, 'format')
                        # Calling format(args, kwargs) (line 445)
                        format_call_result_18443 = invoke(stypy.reporting.localization.Localization(__file__, 445, 48), format_18440, *[name_18441], **kwargs_18442)
                        
                        # Processing the call keyword arguments (line 445)
                        kwargs_18444 = {}
                        # Getting the type of 'TypeError' (line 445)
                        TypeError_18437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 24), 'TypeError', False)
                        # Calling TypeError(args, kwargs) (line 445)
                        TypeError_call_result_18445 = invoke(stypy.reporting.localization.Localization(__file__, 445, 24), TypeError_18437, *[localization_18438, format_call_result_18443], **kwargs_18444)
                        
                        
                        # Call to remove(...): (line 447)
                        # Processing the call arguments (line 447)
                        # Getting the type of 'utw' (line 447)
                        utw_18449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 52), 'utw', False)
                        # Processing the call keyword arguments (line 447)
                        kwargs_18450 = {}
                        # Getting the type of 'TypeWarning' (line 447)
                        TypeWarning_18446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 24), 'TypeWarning', False)
                        # Obtaining the member 'warnings' of a type (line 447)
                        warnings_18447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 24), TypeWarning_18446, 'warnings')
                        # Obtaining the member 'remove' of a type (line 447)
                        remove_18448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 24), warnings_18447, 'remove')
                        # Calling remove(args, kwargs) (line 447)
                        remove_call_result_18451 = invoke(stypy.reporting.localization.Localization(__file__, 447, 24), remove_18448, *[utw_18449], **kwargs_18450)
                        
                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    
                    
                    # Call to len(...): (line 450)
                    # Processing the call arguments (line 450)
                    # Getting the type of 'our_unreferenced_type_warnings_in_this_context' (line 450)
                    our_unreferenced_type_warnings_in_this_context_18453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 27), 'our_unreferenced_type_warnings_in_this_context', False)
                    # Processing the call keyword arguments (line 450)
                    kwargs_18454 = {}
                    # Getting the type of 'len' (line 450)
                    len_18452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 23), 'len', False)
                    # Calling len(args, kwargs) (line 450)
                    len_call_result_18455 = invoke(stypy.reporting.localization.Localization(__file__, 450, 23), len_18452, *[our_unreferenced_type_warnings_in_this_context_18453], **kwargs_18454)
                    
                    int_18456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 77), 'int')
                    # Applying the binary operator '>' (line 450)
                    result_gt_18457 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 23), '>', len_call_result_18455, int_18456)
                    
                    # Testing if the type of an if condition is none (line 450)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 450, 20), result_gt_18457):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 450)
                        if_condition_18458 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 450, 20), result_gt_18457)
                        # Assigning a type to the variable 'if_condition_18458' (line 450)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 20), 'if_condition_18458', if_condition_18458)
                        # SSA begins for if statement (line 450)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to set_type_of(...): (line 451)
                        # Processing the call arguments (line 451)
                        # Getting the type of 'name' (line 451)
                        name_18464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 55), 'name', False)
                        
                        # Call to TypeError(...): (line 451)
                        # Processing the call arguments (line 451)
                        # Getting the type of 'localization' (line 451)
                        localization_18466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 71), 'localization', False)
                        
                        # Call to format(...): (line 451)
                        # Processing the call arguments (line 451)
                        # Getting the type of 'name' (line 452)
                        name_18469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 113), 'name', False)
                        # Processing the call keyword arguments (line 451)
                        kwargs_18470 = {}
                        str_18467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 85), 'str', "Attempted to use '{0}' previously to its definition")
                        # Obtaining the member 'format' of a type (line 451)
                        format_18468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 85), str_18467, 'format')
                        # Calling format(args, kwargs) (line 451)
                        format_call_result_18471 = invoke(stypy.reporting.localization.Localization(__file__, 451, 85), format_18468, *[name_18469], **kwargs_18470)
                        
                        # Processing the call keyword arguments (line 451)
                        kwargs_18472 = {}
                        # Getting the type of 'TypeError' (line 451)
                        TypeError_18465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 61), 'TypeError', False)
                        # Calling TypeError(args, kwargs) (line 451)
                        TypeError_call_result_18473 = invoke(stypy.reporting.localization.Localization(__file__, 451, 61), TypeError_18465, *[localization_18466, format_call_result_18471], **kwargs_18472)
                        
                        # Getting the type of 'localization' (line 453)
                        localization_18474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 55), 'localization', False)
                        # Processing the call keyword arguments (line 451)
                        kwargs_18475 = {}
                        
                        # Call to get_context(...): (line 451)
                        # Processing the call keyword arguments (line 451)
                        kwargs_18461 = {}
                        # Getting the type of 'self' (line 451)
                        self_18459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 24), 'self', False)
                        # Obtaining the member 'get_context' of a type (line 451)
                        get_context_18460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 24), self_18459, 'get_context')
                        # Calling get_context(args, kwargs) (line 451)
                        get_context_call_result_18462 = invoke(stypy.reporting.localization.Localization(__file__, 451, 24), get_context_18460, *[], **kwargs_18461)
                        
                        # Obtaining the member 'set_type_of' of a type (line 451)
                        set_type_of_18463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 24), get_context_call_result_18462, 'set_type_of')
                        # Calling set_type_of(args, kwargs) (line 451)
                        set_type_of_call_result_18476 = invoke(stypy.reporting.localization.Localization(__file__, 451, 24), set_type_of_18463, *[name_18464, TypeError_call_result_18473, localization_18474], **kwargs_18475)
                        
                        
                        # Call to get_type_of(...): (line 454)
                        # Processing the call arguments (line 454)
                        # Getting the type of 'name' (line 454)
                        name_18482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 62), 'name', False)
                        # Processing the call keyword arguments (line 454)
                        kwargs_18483 = {}
                        
                        # Call to get_context(...): (line 454)
                        # Processing the call keyword arguments (line 454)
                        kwargs_18479 = {}
                        # Getting the type of 'self' (line 454)
                        self_18477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 31), 'self', False)
                        # Obtaining the member 'get_context' of a type (line 454)
                        get_context_18478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 31), self_18477, 'get_context')
                        # Calling get_context(args, kwargs) (line 454)
                        get_context_call_result_18480 = invoke(stypy.reporting.localization.Localization(__file__, 454, 31), get_context_18478, *[], **kwargs_18479)
                        
                        # Obtaining the member 'get_type_of' of a type (line 454)
                        get_type_of_18481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 31), get_context_call_result_18480, 'get_type_of')
                        # Calling get_type_of(args, kwargs) (line 454)
                        get_type_of_call_result_18484 = invoke(stypy.reporting.localization.Localization(__file__, 454, 31), get_type_of_18481, *[name_18482], **kwargs_18483)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 454)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 24), 'stypy_return_type', get_type_of_call_result_18484)
                        # SSA join for if statement (line 450)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 438)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Call to a Tuple (line 456):
                
                # Assigning a Call to a Name:
                
                # Call to contains_an_undefined_type(...): (line 456)
                # Processing the call arguments (line 456)
                # Getting the type of 'type_' (line 457)
                type__18488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 47), 'type_', False)
                # Processing the call keyword arguments (line 456)
                kwargs_18489 = {}
                # Getting the type of 'type_inference_proxy' (line 456)
                type_inference_proxy_18485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 58), 'type_inference_proxy', False)
                # Obtaining the member 'TypeInferenceProxy' of a type (line 456)
                TypeInferenceProxy_18486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 58), type_inference_proxy_18485, 'TypeInferenceProxy')
                # Obtaining the member 'contains_an_undefined_type' of a type (line 456)
                contains_an_undefined_type_18487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 58), TypeInferenceProxy_18486, 'contains_an_undefined_type')
                # Calling contains_an_undefined_type(args, kwargs) (line 456)
                contains_an_undefined_type_call_result_18490 = invoke(stypy.reporting.localization.Localization(__file__, 456, 58), contains_an_undefined_type_18487, *[type__18488], **kwargs_18489)
                
                # Assigning a type to the variable 'call_assignment_17651' (line 456)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17651', contains_an_undefined_type_call_result_18490)
                
                # Assigning a Call to a Name (line 456):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_17651' (line 456)
                call_assignment_17651_18491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17651', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_18492 = stypy_get_value_from_tuple(call_assignment_17651_18491, 2, 0)
                
                # Assigning a type to the variable 'call_assignment_17652' (line 456)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17652', stypy_get_value_from_tuple_call_result_18492)
                
                # Assigning a Name to a Name (line 456):
                # Getting the type of 'call_assignment_17652' (line 456)
                call_assignment_17652_18493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17652')
                # Assigning a type to the variable 'contains_undefined' (line 456)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'contains_undefined', call_assignment_17652_18493)
                
                # Assigning a Call to a Name (line 456):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_17651' (line 456)
                call_assignment_17651_18494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17651', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_18495 = stypy_get_value_from_tuple(call_assignment_17651_18494, 2, 1)
                
                # Assigning a type to the variable 'call_assignment_17653' (line 456)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17653', stypy_get_value_from_tuple_call_result_18495)
                
                # Assigning a Name to a Name (line 456):
                # Getting the type of 'call_assignment_17653' (line 456)
                call_assignment_17653_18496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17653')
                # Assigning a type to the variable 'more_types_in_value' (line 456)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 36), 'more_types_in_value', call_assignment_17653_18496)
                # Getting the type of 'contains_undefined' (line 458)
                contains_undefined_18497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 19), 'contains_undefined')
                # Testing if the type of an if condition is none (line 458)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 458, 16), contains_undefined_18497):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 458)
                    if_condition_18498 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 458, 16), contains_undefined_18497)
                    # Assigning a type to the variable 'if_condition_18498' (line 458)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 16), 'if_condition_18498', if_condition_18498)
                    # SSA begins for if statement (line 458)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'more_types_in_value' (line 459)
                    more_types_in_value_18499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 23), 'more_types_in_value')
                    int_18500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 46), 'int')
                    # Applying the binary operator '==' (line 459)
                    result_eq_18501 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 23), '==', more_types_in_value_18499, int_18500)
                    
                    # Testing if the type of an if condition is none (line 459)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 459, 20), result_eq_18501):
                        
                        # Call to instance(...): (line 463)
                        # Processing the call arguments (line 463)
                        # Getting the type of 'localization' (line 463)
                        localization_18514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 45), 'localization', False)
                        
                        # Call to format(...): (line 464)
                        # Processing the call arguments (line 464)
                        # Getting the type of 'name' (line 465)
                        name_18517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 52), 'name', False)
                        # Processing the call keyword arguments (line 464)
                        kwargs_18518 = {}
                        str_18515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 45), 'str', "Potentialy assigning to '{0}' the value of an undefined variable")
                        # Obtaining the member 'format' of a type (line 464)
                        format_18516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 45), str_18515, 'format')
                        # Calling format(args, kwargs) (line 464)
                        format_call_result_18519 = invoke(stypy.reporting.localization.Localization(__file__, 464, 45), format_18516, *[name_18517], **kwargs_18518)
                        
                        # Processing the call keyword arguments (line 463)
                        kwargs_18520 = {}
                        # Getting the type of 'TypeWarning' (line 463)
                        TypeWarning_18512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 24), 'TypeWarning', False)
                        # Obtaining the member 'instance' of a type (line 463)
                        instance_18513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 24), TypeWarning_18512, 'instance')
                        # Calling instance(args, kwargs) (line 463)
                        instance_call_result_18521 = invoke(stypy.reporting.localization.Localization(__file__, 463, 24), instance_18513, *[localization_18514, format_call_result_18519], **kwargs_18520)
                        
                    else:
                        
                        # Testing the type of an if condition (line 459)
                        if_condition_18502 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 459, 20), result_eq_18501)
                        # Assigning a type to the variable 'if_condition_18502' (line 459)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 20), 'if_condition_18502', if_condition_18502)
                        # SSA begins for if statement (line 459)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to TypeError(...): (line 460)
                        # Processing the call arguments (line 460)
                        # Getting the type of 'localization' (line 460)
                        localization_18504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 34), 'localization', False)
                        
                        # Call to format(...): (line 460)
                        # Processing the call arguments (line 460)
                        # Getting the type of 'name' (line 461)
                        name_18507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 41), 'name', False)
                        # Processing the call keyword arguments (line 460)
                        kwargs_18508 = {}
                        str_18505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 48), 'str', "Assigning to '{0}' the value of an undefined variable")
                        # Obtaining the member 'format' of a type (line 460)
                        format_18506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 48), str_18505, 'format')
                        # Calling format(args, kwargs) (line 460)
                        format_call_result_18509 = invoke(stypy.reporting.localization.Localization(__file__, 460, 48), format_18506, *[name_18507], **kwargs_18508)
                        
                        # Processing the call keyword arguments (line 460)
                        kwargs_18510 = {}
                        # Getting the type of 'TypeError' (line 460)
                        TypeError_18503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 24), 'TypeError', False)
                        # Calling TypeError(args, kwargs) (line 460)
                        TypeError_call_result_18511 = invoke(stypy.reporting.localization.Localization(__file__, 460, 24), TypeError_18503, *[localization_18504, format_call_result_18509], **kwargs_18510)
                        
                        # SSA branch for the else part of an if statement (line 459)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to instance(...): (line 463)
                        # Processing the call arguments (line 463)
                        # Getting the type of 'localization' (line 463)
                        localization_18514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 45), 'localization', False)
                        
                        # Call to format(...): (line 464)
                        # Processing the call arguments (line 464)
                        # Getting the type of 'name' (line 465)
                        name_18517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 52), 'name', False)
                        # Processing the call keyword arguments (line 464)
                        kwargs_18518 = {}
                        str_18515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 45), 'str', "Potentialy assigning to '{0}' the value of an undefined variable")
                        # Obtaining the member 'format' of a type (line 464)
                        format_18516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 45), str_18515, 'format')
                        # Calling format(args, kwargs) (line 464)
                        format_call_result_18519 = invoke(stypy.reporting.localization.Localization(__file__, 464, 45), format_18516, *[name_18517], **kwargs_18518)
                        
                        # Processing the call keyword arguments (line 463)
                        kwargs_18520 = {}
                        # Getting the type of 'TypeWarning' (line 463)
                        TypeWarning_18512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 24), 'TypeWarning', False)
                        # Obtaining the member 'instance' of a type (line 463)
                        instance_18513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 24), TypeWarning_18512, 'instance')
                        # Calling instance(args, kwargs) (line 463)
                        instance_call_result_18521 = invoke(stypy.reporting.localization.Localization(__file__, 463, 24), instance_18513, *[localization_18514, format_call_result_18519], **kwargs_18520)
                        
                        # SSA join for if statement (line 459)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 458)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Call to set_type_of(...): (line 467)
                # Processing the call arguments (line 467)
                # Getting the type of 'name' (line 467)
                name_18527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 47), 'name', False)
                # Getting the type of 'type_' (line 467)
                type__18528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 53), 'type_', False)
                # Getting the type of 'localization' (line 467)
                localization_18529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 60), 'localization', False)
                # Processing the call keyword arguments (line 467)
                kwargs_18530 = {}
                
                # Call to get_context(...): (line 467)
                # Processing the call keyword arguments (line 467)
                kwargs_18524 = {}
                # Getting the type of 'self' (line 467)
                self_18522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 16), 'self', False)
                # Obtaining the member 'get_context' of a type (line 467)
                get_context_18523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 16), self_18522, 'get_context')
                # Calling get_context(args, kwargs) (line 467)
                get_context_call_result_18525 = invoke(stypy.reporting.localization.Localization(__file__, 467, 16), get_context_18523, *[], **kwargs_18524)
                
                # Obtaining the member 'set_type_of' of a type (line 467)
                set_type_of_18526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 16), get_context_call_result_18525, 'set_type_of')
                # Calling set_type_of(args, kwargs) (line 467)
                set_type_of_call_result_18531 = invoke(stypy.reporting.localization.Localization(__file__, 467, 16), set_type_of_18526, *[name_18527, type__18528, localization_18529], **kwargs_18530)
                
            else:
                
                # Testing the type of an if condition (line 420)
                if_condition_18388 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 420, 12), is_marked_as_global_18387)
                # Assigning a type to the variable 'if_condition_18388' (line 420)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 12), 'if_condition_18388', if_condition_18388)
                # SSA begins for if statement (line 420)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to set_type_of(...): (line 421)
                # Processing the call arguments (line 421)
                # Getting the type of 'name' (line 421)
                name_18391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 43), 'name', False)
                # Getting the type of 'type_' (line 421)
                type__18392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 49), 'type_', False)
                # Getting the type of 'localization' (line 421)
                localization_18393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 56), 'localization', False)
                # Processing the call keyword arguments (line 421)
                kwargs_18394 = {}
                # Getting the type of 'global_context' (line 421)
                global_context_18389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 16), 'global_context', False)
                # Obtaining the member 'set_type_of' of a type (line 421)
                set_type_of_18390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 16), global_context_18389, 'set_type_of')
                # Calling set_type_of(args, kwargs) (line 421)
                set_type_of_call_result_18395 = invoke(stypy.reporting.localization.Localization(__file__, 421, 16), set_type_of_18390, *[name_18391, type__18392, localization_18393], **kwargs_18394)
                
                # SSA branch for the else part of an if statement (line 420)
                module_type_store.open_ssa_branch('else')
                str_18396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, (-1)), 'str', 'Special case:\n                    If:\n                        - A variable do not exist in the local context\n                        - This variable is not marked as global\n                        - There exist unreferenced type warnings in this scope typed to this variable.\n                    Then:\n                        - For each unreferenced type warning found:\n                            - Generate a unreferenced variable error with the warning data\n                            - Delete warning\n                            - Mark the type of the variable as ErrorType\n                ')
                
                # Assigning a Call to a Name (line 434):
                
                # Assigning a Call to a Name (line 434):
                
                # Call to filter(...): (line 434)
                # Processing the call arguments (line 434)

                @norecursion
                def _stypy_temp_lambda_27(localization, *varargs, **kwargs):
                    global module_type_store
                    # Assign values to the parameters with defaults
                    defaults = []
                    # Create a new context for function '_stypy_temp_lambda_27'
                    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_27', 434, 52, True)
                    # Passed parameters checking function
                    _stypy_temp_lambda_27.stypy_localization = localization
                    _stypy_temp_lambda_27.stypy_type_of_self = None
                    _stypy_temp_lambda_27.stypy_type_store = module_type_store
                    _stypy_temp_lambda_27.stypy_function_name = '_stypy_temp_lambda_27'
                    _stypy_temp_lambda_27.stypy_param_names_list = ['warning']
                    _stypy_temp_lambda_27.stypy_varargs_param_name = None
                    _stypy_temp_lambda_27.stypy_kwargs_param_name = None
                    _stypy_temp_lambda_27.stypy_call_defaults = defaults
                    _stypy_temp_lambda_27.stypy_call_varargs = varargs
                    _stypy_temp_lambda_27.stypy_call_kwargs = kwargs
                    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_27', ['warning'], None, None, defaults, varargs, kwargs)

                    if is_error_type(arguments):
                        # Destroy the current context
                        module_type_store = module_type_store.close_function_context()
                        return arguments

                    # Stacktrace push for error reporting
                    localization.set_stack_trace('_stypy_temp_lambda_27', ['warning'], arguments)
                    # Default return type storage variable (SSA)
                    # Assigning a type to the variable 'stypy_return_type'
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                    
                    
                    # ################# Begin of the lambda function code ##################

                    
                    # Getting the type of 'warning' (line 435)
                    warning_18398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 52), 'warning', False)
                    # Obtaining the member '__class__' of a type (line 435)
                    class___18399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 52), warning_18398, '__class__')
                    # Getting the type of 'UnreferencedLocalVariableTypeWarning' (line 435)
                    UnreferencedLocalVariableTypeWarning_18400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 73), 'UnreferencedLocalVariableTypeWarning', False)
                    # Applying the binary operator '==' (line 435)
                    result_eq_18401 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 52), '==', class___18399, UnreferencedLocalVariableTypeWarning_18400)
                    
                    # Assigning the return type of the lambda function
                    # Assigning a type to the variable 'stypy_return_type' (line 434)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 52), 'stypy_return_type', result_eq_18401)
                    
                    # ################# End of the lambda function code ##################

                    # Stacktrace pop (error reporting)
                    localization.unset_stack_trace()
                    
                    # Storing the return type of function '_stypy_temp_lambda_27' in the type store
                    # Getting the type of 'stypy_return_type' (line 434)
                    stypy_return_type_18402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 52), 'stypy_return_type')
                    module_type_store.store_return_type_of_current_context(stypy_return_type_18402)
                    
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    
                    # Return type of the function '_stypy_temp_lambda_27'
                    return stypy_return_type_18402

                # Assigning a type to the variable '_stypy_temp_lambda_27' (line 434)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 52), '_stypy_temp_lambda_27', _stypy_temp_lambda_27)
                # Getting the type of '_stypy_temp_lambda_27' (line 434)
                _stypy_temp_lambda_27_18403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 52), '_stypy_temp_lambda_27')
                
                # Call to get_warning_msgs(...): (line 436)
                # Processing the call keyword arguments (line 436)
                kwargs_18406 = {}
                # Getting the type of 'TypeWarning' (line 436)
                TypeWarning_18404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 52), 'TypeWarning', False)
                # Obtaining the member 'get_warning_msgs' of a type (line 436)
                get_warning_msgs_18405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 52), TypeWarning_18404, 'get_warning_msgs')
                # Calling get_warning_msgs(args, kwargs) (line 436)
                get_warning_msgs_call_result_18407 = invoke(stypy.reporting.localization.Localization(__file__, 436, 52), get_warning_msgs_18405, *[], **kwargs_18406)
                
                # Processing the call keyword arguments (line 434)
                kwargs_18408 = {}
                # Getting the type of 'filter' (line 434)
                filter_18397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 45), 'filter', False)
                # Calling filter(args, kwargs) (line 434)
                filter_call_result_18409 = invoke(stypy.reporting.localization.Localization(__file__, 434, 45), filter_18397, *[_stypy_temp_lambda_27_18403, get_warning_msgs_call_result_18407], **kwargs_18408)
                
                # Assigning a type to the variable 'unreferenced_type_warnings' (line 434)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 16), 'unreferenced_type_warnings', filter_call_result_18409)
                
                
                # Call to len(...): (line 438)
                # Processing the call arguments (line 438)
                # Getting the type of 'unreferenced_type_warnings' (line 438)
                unreferenced_type_warnings_18411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 23), 'unreferenced_type_warnings', False)
                # Processing the call keyword arguments (line 438)
                kwargs_18412 = {}
                # Getting the type of 'len' (line 438)
                len_18410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 19), 'len', False)
                # Calling len(args, kwargs) (line 438)
                len_call_result_18413 = invoke(stypy.reporting.localization.Localization(__file__, 438, 19), len_18410, *[unreferenced_type_warnings_18411], **kwargs_18412)
                
                int_18414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 53), 'int')
                # Applying the binary operator '>' (line 438)
                result_gt_18415 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 19), '>', len_call_result_18413, int_18414)
                
                # Testing if the type of an if condition is none (line 438)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 438, 16), result_gt_18415):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 438)
                    if_condition_18416 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 438, 16), result_gt_18415)
                    # Assigning a type to the variable 'if_condition_18416' (line 438)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 'if_condition_18416', if_condition_18416)
                    # SSA begins for if statement (line 438)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 439):
                    
                    # Assigning a Call to a Name (line 439):
                    
                    # Call to filter(...): (line 439)
                    # Processing the call arguments (line 439)

                    @norecursion
                    def _stypy_temp_lambda_28(localization, *varargs, **kwargs):
                        global module_type_store
                        # Assign values to the parameters with defaults
                        defaults = []
                        # Create a new context for function '_stypy_temp_lambda_28'
                        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_28', 439, 76, True)
                        # Passed parameters checking function
                        _stypy_temp_lambda_28.stypy_localization = localization
                        _stypy_temp_lambda_28.stypy_type_of_self = None
                        _stypy_temp_lambda_28.stypy_type_store = module_type_store
                        _stypy_temp_lambda_28.stypy_function_name = '_stypy_temp_lambda_28'
                        _stypy_temp_lambda_28.stypy_param_names_list = ['warning']
                        _stypy_temp_lambda_28.stypy_varargs_param_name = None
                        _stypy_temp_lambda_28.stypy_kwargs_param_name = None
                        _stypy_temp_lambda_28.stypy_call_defaults = defaults
                        _stypy_temp_lambda_28.stypy_call_varargs = varargs
                        _stypy_temp_lambda_28.stypy_call_kwargs = kwargs
                        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_28', ['warning'], None, None, defaults, varargs, kwargs)

                        if is_error_type(arguments):
                            # Destroy the current context
                            module_type_store = module_type_store.close_function_context()
                            return arguments

                        # Stacktrace push for error reporting
                        localization.set_stack_trace('_stypy_temp_lambda_28', ['warning'], arguments)
                        # Default return type storage variable (SSA)
                        # Assigning a type to the variable 'stypy_return_type'
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                        
                        
                        # ################# Begin of the lambda function code ##################

                        
                        # Evaluating a boolean operation
                        
                        # Getting the type of 'warning' (line 440)
                        warning_18418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 76), 'warning', False)
                        # Obtaining the member 'context' of a type (line 440)
                        context_18419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 76), warning_18418, 'context')
                        
                        # Call to get_context(...): (line 440)
                        # Processing the call keyword arguments (line 440)
                        kwargs_18422 = {}
                        # Getting the type of 'self' (line 440)
                        self_18420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 95), 'self', False)
                        # Obtaining the member 'get_context' of a type (line 440)
                        get_context_18421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 95), self_18420, 'get_context')
                        # Calling get_context(args, kwargs) (line 440)
                        get_context_call_result_18423 = invoke(stypy.reporting.localization.Localization(__file__, 440, 95), get_context_18421, *[], **kwargs_18422)
                        
                        # Applying the binary operator '==' (line 440)
                        result_eq_18424 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 76), '==', context_18419, get_context_call_result_18423)
                        
                        
                        # Getting the type of 'warning' (line 441)
                        warning_18425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 76), 'warning', False)
                        # Obtaining the member 'name' of a type (line 441)
                        name_18426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 76), warning_18425, 'name')
                        # Getting the type of 'name' (line 441)
                        name_18427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 92), 'name', False)
                        # Applying the binary operator '==' (line 441)
                        result_eq_18428 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 76), '==', name_18426, name_18427)
                        
                        # Applying the binary operator 'and' (line 440)
                        result_and_keyword_18429 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 76), 'and', result_eq_18424, result_eq_18428)
                        
                        # Assigning the return type of the lambda function
                        # Assigning a type to the variable 'stypy_return_type' (line 439)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 76), 'stypy_return_type', result_and_keyword_18429)
                        
                        # ################# End of the lambda function code ##################

                        # Stacktrace pop (error reporting)
                        localization.unset_stack_trace()
                        
                        # Storing the return type of function '_stypy_temp_lambda_28' in the type store
                        # Getting the type of 'stypy_return_type' (line 439)
                        stypy_return_type_18430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 76), 'stypy_return_type')
                        module_type_store.store_return_type_of_current_context(stypy_return_type_18430)
                        
                        # Destroy the current context
                        module_type_store = module_type_store.close_function_context()
                        
                        # Return type of the function '_stypy_temp_lambda_28'
                        return stypy_return_type_18430

                    # Assigning a type to the variable '_stypy_temp_lambda_28' (line 439)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 76), '_stypy_temp_lambda_28', _stypy_temp_lambda_28)
                    # Getting the type of '_stypy_temp_lambda_28' (line 439)
                    _stypy_temp_lambda_28_18431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 76), '_stypy_temp_lambda_28')
                    # Getting the type of 'unreferenced_type_warnings' (line 442)
                    unreferenced_type_warnings_18432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 76), 'unreferenced_type_warnings', False)
                    # Processing the call keyword arguments (line 439)
                    kwargs_18433 = {}
                    # Getting the type of 'filter' (line 439)
                    filter_18417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 69), 'filter', False)
                    # Calling filter(args, kwargs) (line 439)
                    filter_call_result_18434 = invoke(stypy.reporting.localization.Localization(__file__, 439, 69), filter_18417, *[_stypy_temp_lambda_28_18431, unreferenced_type_warnings_18432], **kwargs_18433)
                    
                    # Assigning a type to the variable 'our_unreferenced_type_warnings_in_this_context' (line 439)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 20), 'our_unreferenced_type_warnings_in_this_context', filter_call_result_18434)
                    
                    # Getting the type of 'our_unreferenced_type_warnings_in_this_context' (line 444)
                    our_unreferenced_type_warnings_in_this_context_18435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 31), 'our_unreferenced_type_warnings_in_this_context')
                    # Assigning a type to the variable 'our_unreferenced_type_warnings_in_this_context_18435' (line 444)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 20), 'our_unreferenced_type_warnings_in_this_context_18435', our_unreferenced_type_warnings_in_this_context_18435)
                    # Testing if the for loop is going to be iterated (line 444)
                    # Testing the type of a for loop iterable (line 444)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 444, 20), our_unreferenced_type_warnings_in_this_context_18435)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 444, 20), our_unreferenced_type_warnings_in_this_context_18435):
                        # Getting the type of the for loop variable (line 444)
                        for_loop_var_18436 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 444, 20), our_unreferenced_type_warnings_in_this_context_18435)
                        # Assigning a type to the variable 'utw' (line 444)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 20), 'utw', for_loop_var_18436)
                        # SSA begins for a for statement (line 444)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Call to TypeError(...): (line 445)
                        # Processing the call arguments (line 445)
                        # Getting the type of 'localization' (line 445)
                        localization_18438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 34), 'localization', False)
                        
                        # Call to format(...): (line 445)
                        # Processing the call arguments (line 445)
                        # Getting the type of 'name' (line 446)
                        name_18441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 86), 'name', False)
                        # Processing the call keyword arguments (line 445)
                        kwargs_18442 = {}
                        str_18439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 48), 'str', "UnboundLocalError: local variable '{0}' referenced before assignment")
                        # Obtaining the member 'format' of a type (line 445)
                        format_18440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 48), str_18439, 'format')
                        # Calling format(args, kwargs) (line 445)
                        format_call_result_18443 = invoke(stypy.reporting.localization.Localization(__file__, 445, 48), format_18440, *[name_18441], **kwargs_18442)
                        
                        # Processing the call keyword arguments (line 445)
                        kwargs_18444 = {}
                        # Getting the type of 'TypeError' (line 445)
                        TypeError_18437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 24), 'TypeError', False)
                        # Calling TypeError(args, kwargs) (line 445)
                        TypeError_call_result_18445 = invoke(stypy.reporting.localization.Localization(__file__, 445, 24), TypeError_18437, *[localization_18438, format_call_result_18443], **kwargs_18444)
                        
                        
                        # Call to remove(...): (line 447)
                        # Processing the call arguments (line 447)
                        # Getting the type of 'utw' (line 447)
                        utw_18449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 52), 'utw', False)
                        # Processing the call keyword arguments (line 447)
                        kwargs_18450 = {}
                        # Getting the type of 'TypeWarning' (line 447)
                        TypeWarning_18446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 24), 'TypeWarning', False)
                        # Obtaining the member 'warnings' of a type (line 447)
                        warnings_18447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 24), TypeWarning_18446, 'warnings')
                        # Obtaining the member 'remove' of a type (line 447)
                        remove_18448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 24), warnings_18447, 'remove')
                        # Calling remove(args, kwargs) (line 447)
                        remove_call_result_18451 = invoke(stypy.reporting.localization.Localization(__file__, 447, 24), remove_18448, *[utw_18449], **kwargs_18450)
                        
                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    
                    
                    # Call to len(...): (line 450)
                    # Processing the call arguments (line 450)
                    # Getting the type of 'our_unreferenced_type_warnings_in_this_context' (line 450)
                    our_unreferenced_type_warnings_in_this_context_18453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 27), 'our_unreferenced_type_warnings_in_this_context', False)
                    # Processing the call keyword arguments (line 450)
                    kwargs_18454 = {}
                    # Getting the type of 'len' (line 450)
                    len_18452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 23), 'len', False)
                    # Calling len(args, kwargs) (line 450)
                    len_call_result_18455 = invoke(stypy.reporting.localization.Localization(__file__, 450, 23), len_18452, *[our_unreferenced_type_warnings_in_this_context_18453], **kwargs_18454)
                    
                    int_18456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 77), 'int')
                    # Applying the binary operator '>' (line 450)
                    result_gt_18457 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 23), '>', len_call_result_18455, int_18456)
                    
                    # Testing if the type of an if condition is none (line 450)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 450, 20), result_gt_18457):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 450)
                        if_condition_18458 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 450, 20), result_gt_18457)
                        # Assigning a type to the variable 'if_condition_18458' (line 450)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 20), 'if_condition_18458', if_condition_18458)
                        # SSA begins for if statement (line 450)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to set_type_of(...): (line 451)
                        # Processing the call arguments (line 451)
                        # Getting the type of 'name' (line 451)
                        name_18464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 55), 'name', False)
                        
                        # Call to TypeError(...): (line 451)
                        # Processing the call arguments (line 451)
                        # Getting the type of 'localization' (line 451)
                        localization_18466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 71), 'localization', False)
                        
                        # Call to format(...): (line 451)
                        # Processing the call arguments (line 451)
                        # Getting the type of 'name' (line 452)
                        name_18469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 113), 'name', False)
                        # Processing the call keyword arguments (line 451)
                        kwargs_18470 = {}
                        str_18467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 85), 'str', "Attempted to use '{0}' previously to its definition")
                        # Obtaining the member 'format' of a type (line 451)
                        format_18468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 85), str_18467, 'format')
                        # Calling format(args, kwargs) (line 451)
                        format_call_result_18471 = invoke(stypy.reporting.localization.Localization(__file__, 451, 85), format_18468, *[name_18469], **kwargs_18470)
                        
                        # Processing the call keyword arguments (line 451)
                        kwargs_18472 = {}
                        # Getting the type of 'TypeError' (line 451)
                        TypeError_18465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 61), 'TypeError', False)
                        # Calling TypeError(args, kwargs) (line 451)
                        TypeError_call_result_18473 = invoke(stypy.reporting.localization.Localization(__file__, 451, 61), TypeError_18465, *[localization_18466, format_call_result_18471], **kwargs_18472)
                        
                        # Getting the type of 'localization' (line 453)
                        localization_18474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 55), 'localization', False)
                        # Processing the call keyword arguments (line 451)
                        kwargs_18475 = {}
                        
                        # Call to get_context(...): (line 451)
                        # Processing the call keyword arguments (line 451)
                        kwargs_18461 = {}
                        # Getting the type of 'self' (line 451)
                        self_18459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 24), 'self', False)
                        # Obtaining the member 'get_context' of a type (line 451)
                        get_context_18460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 24), self_18459, 'get_context')
                        # Calling get_context(args, kwargs) (line 451)
                        get_context_call_result_18462 = invoke(stypy.reporting.localization.Localization(__file__, 451, 24), get_context_18460, *[], **kwargs_18461)
                        
                        # Obtaining the member 'set_type_of' of a type (line 451)
                        set_type_of_18463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 24), get_context_call_result_18462, 'set_type_of')
                        # Calling set_type_of(args, kwargs) (line 451)
                        set_type_of_call_result_18476 = invoke(stypy.reporting.localization.Localization(__file__, 451, 24), set_type_of_18463, *[name_18464, TypeError_call_result_18473, localization_18474], **kwargs_18475)
                        
                        
                        # Call to get_type_of(...): (line 454)
                        # Processing the call arguments (line 454)
                        # Getting the type of 'name' (line 454)
                        name_18482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 62), 'name', False)
                        # Processing the call keyword arguments (line 454)
                        kwargs_18483 = {}
                        
                        # Call to get_context(...): (line 454)
                        # Processing the call keyword arguments (line 454)
                        kwargs_18479 = {}
                        # Getting the type of 'self' (line 454)
                        self_18477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 31), 'self', False)
                        # Obtaining the member 'get_context' of a type (line 454)
                        get_context_18478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 31), self_18477, 'get_context')
                        # Calling get_context(args, kwargs) (line 454)
                        get_context_call_result_18480 = invoke(stypy.reporting.localization.Localization(__file__, 454, 31), get_context_18478, *[], **kwargs_18479)
                        
                        # Obtaining the member 'get_type_of' of a type (line 454)
                        get_type_of_18481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 31), get_context_call_result_18480, 'get_type_of')
                        # Calling get_type_of(args, kwargs) (line 454)
                        get_type_of_call_result_18484 = invoke(stypy.reporting.localization.Localization(__file__, 454, 31), get_type_of_18481, *[name_18482], **kwargs_18483)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 454)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 24), 'stypy_return_type', get_type_of_call_result_18484)
                        # SSA join for if statement (line 450)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 438)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Call to a Tuple (line 456):
                
                # Assigning a Call to a Name:
                
                # Call to contains_an_undefined_type(...): (line 456)
                # Processing the call arguments (line 456)
                # Getting the type of 'type_' (line 457)
                type__18488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 47), 'type_', False)
                # Processing the call keyword arguments (line 456)
                kwargs_18489 = {}
                # Getting the type of 'type_inference_proxy' (line 456)
                type_inference_proxy_18485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 58), 'type_inference_proxy', False)
                # Obtaining the member 'TypeInferenceProxy' of a type (line 456)
                TypeInferenceProxy_18486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 58), type_inference_proxy_18485, 'TypeInferenceProxy')
                # Obtaining the member 'contains_an_undefined_type' of a type (line 456)
                contains_an_undefined_type_18487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 58), TypeInferenceProxy_18486, 'contains_an_undefined_type')
                # Calling contains_an_undefined_type(args, kwargs) (line 456)
                contains_an_undefined_type_call_result_18490 = invoke(stypy.reporting.localization.Localization(__file__, 456, 58), contains_an_undefined_type_18487, *[type__18488], **kwargs_18489)
                
                # Assigning a type to the variable 'call_assignment_17651' (line 456)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17651', contains_an_undefined_type_call_result_18490)
                
                # Assigning a Call to a Name (line 456):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_17651' (line 456)
                call_assignment_17651_18491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17651', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_18492 = stypy_get_value_from_tuple(call_assignment_17651_18491, 2, 0)
                
                # Assigning a type to the variable 'call_assignment_17652' (line 456)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17652', stypy_get_value_from_tuple_call_result_18492)
                
                # Assigning a Name to a Name (line 456):
                # Getting the type of 'call_assignment_17652' (line 456)
                call_assignment_17652_18493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17652')
                # Assigning a type to the variable 'contains_undefined' (line 456)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'contains_undefined', call_assignment_17652_18493)
                
                # Assigning a Call to a Name (line 456):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_17651' (line 456)
                call_assignment_17651_18494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17651', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_18495 = stypy_get_value_from_tuple(call_assignment_17651_18494, 2, 1)
                
                # Assigning a type to the variable 'call_assignment_17653' (line 456)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17653', stypy_get_value_from_tuple_call_result_18495)
                
                # Assigning a Name to a Name (line 456):
                # Getting the type of 'call_assignment_17653' (line 456)
                call_assignment_17653_18496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'call_assignment_17653')
                # Assigning a type to the variable 'more_types_in_value' (line 456)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 36), 'more_types_in_value', call_assignment_17653_18496)
                # Getting the type of 'contains_undefined' (line 458)
                contains_undefined_18497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 19), 'contains_undefined')
                # Testing if the type of an if condition is none (line 458)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 458, 16), contains_undefined_18497):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 458)
                    if_condition_18498 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 458, 16), contains_undefined_18497)
                    # Assigning a type to the variable 'if_condition_18498' (line 458)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 16), 'if_condition_18498', if_condition_18498)
                    # SSA begins for if statement (line 458)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'more_types_in_value' (line 459)
                    more_types_in_value_18499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 23), 'more_types_in_value')
                    int_18500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 46), 'int')
                    # Applying the binary operator '==' (line 459)
                    result_eq_18501 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 23), '==', more_types_in_value_18499, int_18500)
                    
                    # Testing if the type of an if condition is none (line 459)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 459, 20), result_eq_18501):
                        
                        # Call to instance(...): (line 463)
                        # Processing the call arguments (line 463)
                        # Getting the type of 'localization' (line 463)
                        localization_18514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 45), 'localization', False)
                        
                        # Call to format(...): (line 464)
                        # Processing the call arguments (line 464)
                        # Getting the type of 'name' (line 465)
                        name_18517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 52), 'name', False)
                        # Processing the call keyword arguments (line 464)
                        kwargs_18518 = {}
                        str_18515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 45), 'str', "Potentialy assigning to '{0}' the value of an undefined variable")
                        # Obtaining the member 'format' of a type (line 464)
                        format_18516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 45), str_18515, 'format')
                        # Calling format(args, kwargs) (line 464)
                        format_call_result_18519 = invoke(stypy.reporting.localization.Localization(__file__, 464, 45), format_18516, *[name_18517], **kwargs_18518)
                        
                        # Processing the call keyword arguments (line 463)
                        kwargs_18520 = {}
                        # Getting the type of 'TypeWarning' (line 463)
                        TypeWarning_18512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 24), 'TypeWarning', False)
                        # Obtaining the member 'instance' of a type (line 463)
                        instance_18513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 24), TypeWarning_18512, 'instance')
                        # Calling instance(args, kwargs) (line 463)
                        instance_call_result_18521 = invoke(stypy.reporting.localization.Localization(__file__, 463, 24), instance_18513, *[localization_18514, format_call_result_18519], **kwargs_18520)
                        
                    else:
                        
                        # Testing the type of an if condition (line 459)
                        if_condition_18502 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 459, 20), result_eq_18501)
                        # Assigning a type to the variable 'if_condition_18502' (line 459)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 20), 'if_condition_18502', if_condition_18502)
                        # SSA begins for if statement (line 459)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to TypeError(...): (line 460)
                        # Processing the call arguments (line 460)
                        # Getting the type of 'localization' (line 460)
                        localization_18504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 34), 'localization', False)
                        
                        # Call to format(...): (line 460)
                        # Processing the call arguments (line 460)
                        # Getting the type of 'name' (line 461)
                        name_18507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 41), 'name', False)
                        # Processing the call keyword arguments (line 460)
                        kwargs_18508 = {}
                        str_18505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 48), 'str', "Assigning to '{0}' the value of an undefined variable")
                        # Obtaining the member 'format' of a type (line 460)
                        format_18506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 48), str_18505, 'format')
                        # Calling format(args, kwargs) (line 460)
                        format_call_result_18509 = invoke(stypy.reporting.localization.Localization(__file__, 460, 48), format_18506, *[name_18507], **kwargs_18508)
                        
                        # Processing the call keyword arguments (line 460)
                        kwargs_18510 = {}
                        # Getting the type of 'TypeError' (line 460)
                        TypeError_18503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 24), 'TypeError', False)
                        # Calling TypeError(args, kwargs) (line 460)
                        TypeError_call_result_18511 = invoke(stypy.reporting.localization.Localization(__file__, 460, 24), TypeError_18503, *[localization_18504, format_call_result_18509], **kwargs_18510)
                        
                        # SSA branch for the else part of an if statement (line 459)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to instance(...): (line 463)
                        # Processing the call arguments (line 463)
                        # Getting the type of 'localization' (line 463)
                        localization_18514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 45), 'localization', False)
                        
                        # Call to format(...): (line 464)
                        # Processing the call arguments (line 464)
                        # Getting the type of 'name' (line 465)
                        name_18517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 52), 'name', False)
                        # Processing the call keyword arguments (line 464)
                        kwargs_18518 = {}
                        str_18515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 45), 'str', "Potentialy assigning to '{0}' the value of an undefined variable")
                        # Obtaining the member 'format' of a type (line 464)
                        format_18516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 45), str_18515, 'format')
                        # Calling format(args, kwargs) (line 464)
                        format_call_result_18519 = invoke(stypy.reporting.localization.Localization(__file__, 464, 45), format_18516, *[name_18517], **kwargs_18518)
                        
                        # Processing the call keyword arguments (line 463)
                        kwargs_18520 = {}
                        # Getting the type of 'TypeWarning' (line 463)
                        TypeWarning_18512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 24), 'TypeWarning', False)
                        # Obtaining the member 'instance' of a type (line 463)
                        instance_18513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 24), TypeWarning_18512, 'instance')
                        # Calling instance(args, kwargs) (line 463)
                        instance_call_result_18521 = invoke(stypy.reporting.localization.Localization(__file__, 463, 24), instance_18513, *[localization_18514, format_call_result_18519], **kwargs_18520)
                        
                        # SSA join for if statement (line 459)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 458)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Call to set_type_of(...): (line 467)
                # Processing the call arguments (line 467)
                # Getting the type of 'name' (line 467)
                name_18527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 47), 'name', False)
                # Getting the type of 'type_' (line 467)
                type__18528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 53), 'type_', False)
                # Getting the type of 'localization' (line 467)
                localization_18529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 60), 'localization', False)
                # Processing the call keyword arguments (line 467)
                kwargs_18530 = {}
                
                # Call to get_context(...): (line 467)
                # Processing the call keyword arguments (line 467)
                kwargs_18524 = {}
                # Getting the type of 'self' (line 467)
                self_18522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 16), 'self', False)
                # Obtaining the member 'get_context' of a type (line 467)
                get_context_18523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 16), self_18522, 'get_context')
                # Calling get_context(args, kwargs) (line 467)
                get_context_call_result_18525 = invoke(stypy.reporting.localization.Localization(__file__, 467, 16), get_context_18523, *[], **kwargs_18524)
                
                # Obtaining the member 'set_type_of' of a type (line 467)
                set_type_of_18526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 16), get_context_call_result_18525, 'set_type_of')
                # Calling set_type_of(args, kwargs) (line 467)
                set_type_of_call_result_18531 = invoke(stypy.reporting.localization.Localization(__file__, 467, 16), set_type_of_18526, *[name_18527, type__18528, localization_18529], **kwargs_18530)
                
                # SSA join for if statement (line 420)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 409)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'None' (line 468)
        None_18532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'stypy_return_type', None_18532)
        
        # ################# End of '__set_type_of(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__set_type_of' in the type store
        # Getting the type of 'stypy_return_type' (line 393)
        stypy_return_type_18533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18533)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__set_type_of'
        return stypy_return_type_18533


    @staticmethod
    @norecursion
    def __clone_type_store(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__clone_type_store'
        module_type_store = module_type_store.open_function_context('__clone_type_store', 470, 4, False)
        
        # Passed parameters checking function
        TypeStore.__clone_type_store.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.__clone_type_store.__dict__.__setitem__('stypy_type_of_self', None)
        TypeStore.__clone_type_store.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.__clone_type_store.__dict__.__setitem__('stypy_function_name', '__clone_type_store')
        TypeStore.__clone_type_store.__dict__.__setitem__('stypy_param_names_list', ['type_store'])
        TypeStore.__clone_type_store.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.__clone_type_store.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.__clone_type_store.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.__clone_type_store.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.__clone_type_store.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.__clone_type_store.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, '__clone_type_store', ['type_store'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__clone_type_store', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__clone_type_store(...)' code ##################

        str_18534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, (-1)), 'str', '\n        Clones the type store; eventually it must also clone the values (classes)\n        because they can be modified with intercession\n        ')
        
        # Assigning a Call to a Name (line 477):
        
        # Assigning a Call to a Name (line 477):
        
        # Call to TypeStore(...): (line 477)
        # Processing the call arguments (line 477)
        # Getting the type of 'type_store' (line 477)
        type_store_18536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 31), 'type_store', False)
        # Obtaining the member 'program_name' of a type (line 477)
        program_name_18537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 31), type_store_18536, 'program_name')
        # Processing the call keyword arguments (line 477)
        kwargs_18538 = {}
        # Getting the type of 'TypeStore' (line 477)
        TypeStore_18535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 21), 'TypeStore', False)
        # Calling TypeStore(args, kwargs) (line 477)
        TypeStore_call_result_18539 = invoke(stypy.reporting.localization.Localization(__file__, 477, 21), TypeStore_18535, *[program_name_18537], **kwargs_18538)
        
        # Assigning a type to the variable 'cloned_obj' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'cloned_obj', TypeStore_call_result_18539)
        
        # Assigning a List to a Attribute (line 478):
        
        # Assigning a List to a Attribute (line 478):
        
        # Obtaining an instance of the builtin type 'list' (line 478)
        list_18540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 478)
        
        # Getting the type of 'cloned_obj' (line 478)
        cloned_obj_18541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'cloned_obj')
        # Setting the type of the member 'context_stack' of a type (line 478)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 8), cloned_obj_18541, 'context_stack', list_18540)
        
        # Getting the type of 'type_store' (line 479)
        type_store_18542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 23), 'type_store')
        # Obtaining the member 'context_stack' of a type (line 479)
        context_stack_18543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 23), type_store_18542, 'context_stack')
        # Assigning a type to the variable 'context_stack_18543' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'context_stack_18543', context_stack_18543)
        # Testing if the for loop is going to be iterated (line 479)
        # Testing the type of a for loop iterable (line 479)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 479, 8), context_stack_18543)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 479, 8), context_stack_18543):
            # Getting the type of the for loop variable (line 479)
            for_loop_var_18544 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 479, 8), context_stack_18543)
            # Assigning a type to the variable 'context' (line 479)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'context', for_loop_var_18544)
            # SSA begins for a for statement (line 479)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to append(...): (line 480)
            # Processing the call arguments (line 480)
            
            # Call to clone(...): (line 480)
            # Processing the call keyword arguments (line 480)
            kwargs_18550 = {}
            # Getting the type of 'context' (line 480)
            context_18548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 44), 'context', False)
            # Obtaining the member 'clone' of a type (line 480)
            clone_18549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 44), context_18548, 'clone')
            # Calling clone(args, kwargs) (line 480)
            clone_call_result_18551 = invoke(stypy.reporting.localization.Localization(__file__, 480, 44), clone_18549, *[], **kwargs_18550)
            
            # Processing the call keyword arguments (line 480)
            kwargs_18552 = {}
            # Getting the type of 'cloned_obj' (line 480)
            cloned_obj_18545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 12), 'cloned_obj', False)
            # Obtaining the member 'context_stack' of a type (line 480)
            context_stack_18546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 12), cloned_obj_18545, 'context_stack')
            # Obtaining the member 'append' of a type (line 480)
            append_18547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 12), context_stack_18546, 'append')
            # Calling append(args, kwargs) (line 480)
            append_call_result_18553 = invoke(stypy.reporting.localization.Localization(__file__, 480, 12), append_18547, *[clone_call_result_18551], **kwargs_18552)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Attribute to a Attribute (line 482):
        
        # Assigning a Attribute to a Attribute (line 482):
        # Getting the type of 'type_store' (line 482)
        type_store_18554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 44), 'type_store')
        # Obtaining the member 'last_function_contexts' of a type (line 482)
        last_function_contexts_18555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 44), type_store_18554, 'last_function_contexts')
        # Getting the type of 'cloned_obj' (line 482)
        cloned_obj_18556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'cloned_obj')
        # Setting the type of the member 'last_function_contexts' of a type (line 482)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 8), cloned_obj_18556, 'last_function_contexts', last_function_contexts_18555)
        
        # Assigning a Attribute to a Attribute (line 483):
        
        # Assigning a Attribute to a Attribute (line 483):
        # Getting the type of 'type_store' (line 483)
        type_store_18557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 38), 'type_store')
        # Obtaining the member 'external_modules' of a type (line 483)
        external_modules_18558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 38), type_store_18557, 'external_modules')
        # Getting the type of 'cloned_obj' (line 483)
        cloned_obj_18559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'cloned_obj')
        # Setting the type of the member 'external_modules' of a type (line 483)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 8), cloned_obj_18559, 'external_modules', external_modules_18558)
        
        # Assigning a Attribute to a Attribute (line 484):
        
        # Assigning a Attribute to a Attribute (line 484):
        # Getting the type of 'type_store' (line 484)
        type_store_18560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 43), 'type_store')
        # Obtaining the member 'test_unreferenced_var' of a type (line 484)
        test_unreferenced_var_18561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 43), type_store_18560, 'test_unreferenced_var')
        # Getting the type of 'cloned_obj' (line 484)
        cloned_obj_18562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'cloned_obj')
        # Setting the type of the member 'test_unreferenced_var' of a type (line 484)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 8), cloned_obj_18562, 'test_unreferenced_var', test_unreferenced_var_18561)
        # Getting the type of 'cloned_obj' (line 486)
        cloned_obj_18563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 15), 'cloned_obj')
        # Assigning a type to the variable 'stypy_return_type' (line 486)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'stypy_return_type', cloned_obj_18563)
        
        # ################# End of '__clone_type_store(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__clone_type_store' in the type store
        # Getting the type of 'stypy_return_type' (line 470)
        stypy_return_type_18564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18564)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__clone_type_store'
        return stypy_return_type_18564


    @norecursion
    def __del_type_of_from_function_context(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__del_type_of_from_function_context'
        module_type_store = module_type_store.open_function_context('__del_type_of_from_function_context', 500, 4, False)
        # Assigning a type to the variable 'self' (line 501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.__del_type_of_from_function_context.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.__del_type_of_from_function_context.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.__del_type_of_from_function_context.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.__del_type_of_from_function_context.__dict__.__setitem__('stypy_function_name', 'TypeStore.__del_type_of_from_function_context')
        TypeStore.__del_type_of_from_function_context.__dict__.__setitem__('stypy_param_names_list', ['localization', 'name', 'f_context'])
        TypeStore.__del_type_of_from_function_context.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.__del_type_of_from_function_context.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.__del_type_of_from_function_context.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.__del_type_of_from_function_context.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.__del_type_of_from_function_context.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.__del_type_of_from_function_context.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.__del_type_of_from_function_context', ['localization', 'name', 'f_context'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__del_type_of_from_function_context', localization, ['localization', 'name', 'f_context'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__del_type_of_from_function_context(...)' code ##################

        str_18565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, (-1)), 'str', '\n        Search the stored function contexts for the type associated to a name.\n        As we follows the program flow, a correct program ensures that if this query is performed the name actually HAS\n        a type (it has been assigned a value previously in the previous executed statements). If the name is not found,\n         we have detected a programmer error within the source file (usage of a previously undeclared name). The\n         method is orthogonal to variables and functions.\n        :param name: Name of the element whose type we want to know\n        :return:\n        ')
        
        # Assigning a Name to a Name (line 512):
        
        # Assigning a Name to a Name (line 512):
        # Getting the type of 'f_context' (line 512)
        f_context_18566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 26), 'f_context')
        # Assigning a type to the variable 'current_context' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'current_context', f_context_18566)
        
        # Assigning a Call to a Name (line 514):
        
        # Assigning a Call to a Name (line 514):
        
        # Call to get_global_context(...): (line 514)
        # Processing the call keyword arguments (line 514)
        kwargs_18569 = {}
        # Getting the type of 'self' (line 514)
        self_18567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 25), 'self', False)
        # Obtaining the member 'get_global_context' of a type (line 514)
        get_global_context_18568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 25), self_18567, 'get_global_context')
        # Calling get_global_context(args, kwargs) (line 514)
        get_global_context_call_result_18570 = invoke(stypy.reporting.localization.Localization(__file__, 514, 25), get_global_context_18568, *[], **kwargs_18569)
        
        # Assigning a type to the variable 'global_context' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'global_context', get_global_context_call_result_18570)
        
        # Getting the type of 'name' (line 517)
        name_18571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 11), 'name')
        # Getting the type of 'current_context' (line 517)
        current_context_18572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 19), 'current_context')
        # Obtaining the member 'global_vars' of a type (line 517)
        global_vars_18573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 19), current_context_18572, 'global_vars')
        # Applying the binary operator 'in' (line 517)
        result_contains_18574 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 11), 'in', name_18571, global_vars_18573)
        
        # Testing if the type of an if condition is none (line 517)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 517, 8), result_contains_18574):
            pass
        else:
            
            # Testing the type of an if condition (line 517)
            if_condition_18575 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 517, 8), result_contains_18574)
            # Assigning a type to the variable 'if_condition_18575' (line 517)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'if_condition_18575', if_condition_18575)
            # SSA begins for if statement (line 517)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 519):
            
            # Assigning a Call to a Name (line 519):
            
            # Call to get_type_of(...): (line 519)
            # Processing the call arguments (line 519)
            # Getting the type of 'name' (line 519)
            name_18578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 47), 'name', False)
            # Processing the call keyword arguments (line 519)
            kwargs_18579 = {}
            # Getting the type of 'global_context' (line 519)
            global_context_18576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 20), 'global_context', False)
            # Obtaining the member 'get_type_of' of a type (line 519)
            get_type_of_18577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 20), global_context_18576, 'get_type_of')
            # Calling get_type_of(args, kwargs) (line 519)
            get_type_of_call_result_18580 = invoke(stypy.reporting.localization.Localization(__file__, 519, 20), get_type_of_18577, *[name_18578], **kwargs_18579)
            
            # Assigning a type to the variable 'type_' (line 519)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 12), 'type_', get_type_of_call_result_18580)
            
            # Type idiom detected: calculating its left and rigth part (line 521)
            # Getting the type of 'type_' (line 521)
            type__18581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 15), 'type_')
            # Getting the type of 'None' (line 521)
            None_18582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 24), 'None')
            
            (may_be_18583, more_types_in_union_18584) = may_be_none(type__18581, None_18582)

            if may_be_18583:

                if more_types_in_union_18584:
                    # Runtime conditional SSA (line 521)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to TypeError(...): (line 522)
                # Processing the call arguments (line 522)
                # Getting the type of 'localization' (line 522)
                localization_18586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 33), 'localization', False)
                str_18587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 47), 'str', "Attempted to delete the uninitialized global '%s'")
                # Getting the type of 'name' (line 522)
                name_18588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 101), 'name', False)
                # Applying the binary operator '%' (line 522)
                result_mod_18589 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 47), '%', str_18587, name_18588)
                
                # Processing the call keyword arguments (line 522)
                kwargs_18590 = {}
                # Getting the type of 'TypeError' (line 522)
                TypeError_18585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 23), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 522)
                TypeError_call_result_18591 = invoke(stypy.reporting.localization.Localization(__file__, 522, 23), TypeError_18585, *[localization_18586, result_mod_18589], **kwargs_18590)
                
                # Assigning a type to the variable 'stypy_return_type' (line 522)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 16), 'stypy_return_type', TypeError_call_result_18591)

                if more_types_in_union_18584:
                    # Runtime conditional SSA for else branch (line 521)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_18583) or more_types_in_union_18584):
                
                # Call to del_type_of(...): (line 525)
                # Processing the call arguments (line 525)
                # Getting the type of 'name' (line 525)
                name_18594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 50), 'name', False)
                # Processing the call keyword arguments (line 525)
                kwargs_18595 = {}
                # Getting the type of 'global_context' (line 525)
                global_context_18592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 23), 'global_context', False)
                # Obtaining the member 'del_type_of' of a type (line 525)
                del_type_of_18593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 23), global_context_18592, 'del_type_of')
                # Calling del_type_of(args, kwargs) (line 525)
                del_type_of_call_result_18596 = invoke(stypy.reporting.localization.Localization(__file__, 525, 23), del_type_of_18593, *[name_18594], **kwargs_18595)
                
                # Assigning a type to the variable 'stypy_return_type' (line 525)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 16), 'stypy_return_type', del_type_of_call_result_18596)

                if (may_be_18583 and more_types_in_union_18584):
                    # SSA join for if statement (line 521)
                    module_type_store = module_type_store.join_ssa_context()


            
            # Getting the type of 'type_' (line 521)
            type__18597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'type_')
            # Assigning a type to the variable 'type_' (line 521)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'type_', remove_type_from_union(type__18597, types.NoneType))
            # SSA join for if statement (line 517)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Name to a Name (line 527):
        
        # Assigning a Name to a Name (line 527):
        # Getting the type of 'False' (line 527)
        False_18598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 30), 'False')
        # Assigning a type to the variable 'top_context_reached' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'top_context_reached', False_18598)
        
        # Getting the type of 'self' (line 530)
        self_18599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 23), 'self')
        # Obtaining the member 'context_stack' of a type (line 530)
        context_stack_18600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 23), self_18599, 'context_stack')
        # Assigning a type to the variable 'context_stack_18600' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'context_stack_18600', context_stack_18600)
        # Testing if the for loop is going to be iterated (line 530)
        # Testing the type of a for loop iterable (line 530)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 530, 8), context_stack_18600)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 530, 8), context_stack_18600):
            # Getting the type of the for loop variable (line 530)
            for_loop_var_18601 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 530, 8), context_stack_18600)
            # Assigning a type to the variable 'context' (line 530)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'context', for_loop_var_18601)
            # SSA begins for a for statement (line 530)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'context' (line 531)
            context_18602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 15), 'context')
            # Getting the type of 'f_context' (line 531)
            f_context_18603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 26), 'f_context')
            # Applying the binary operator '==' (line 531)
            result_eq_18604 = python_operator(stypy.reporting.localization.Localization(__file__, 531, 15), '==', context_18602, f_context_18603)
            
            # Testing if the type of an if condition is none (line 531)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 531, 12), result_eq_18604):
                pass
            else:
                
                # Testing the type of an if condition (line 531)
                if_condition_18605 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 531, 12), result_eq_18604)
                # Assigning a type to the variable 'if_condition_18605' (line 531)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 12), 'if_condition_18605', if_condition_18605)
                # SSA begins for if statement (line 531)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 532):
                
                # Assigning a Name to a Name (line 532):
                # Getting the type of 'True' (line 532)
                True_18606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 38), 'True')
                # Assigning a type to the variable 'top_context_reached' (line 532)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 16), 'top_context_reached', True_18606)
                # SSA join for if statement (line 531)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'top_context_reached' (line 534)
            top_context_reached_18607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 19), 'top_context_reached')
            # Applying the 'not' unary operator (line 534)
            result_not__18608 = python_operator(stypy.reporting.localization.Localization(__file__, 534, 15), 'not', top_context_reached_18607)
            
            # Testing if the type of an if condition is none (line 534)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 534, 12), result_not__18608):
                pass
            else:
                
                # Testing the type of an if condition (line 534)
                if_condition_18609 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 534, 12), result_not__18608)
                # Assigning a type to the variable 'if_condition_18609' (line 534)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'if_condition_18609', if_condition_18609)
                # SSA begins for if statement (line 534)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 534)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Call to a Name (line 537):
            
            # Assigning a Call to a Name (line 537):
            
            # Call to get_type_of(...): (line 537)
            # Processing the call arguments (line 537)
            # Getting the type of 'name' (line 537)
            name_18612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 40), 'name', False)
            # Processing the call keyword arguments (line 537)
            kwargs_18613 = {}
            # Getting the type of 'context' (line 537)
            context_18610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 20), 'context', False)
            # Obtaining the member 'get_type_of' of a type (line 537)
            get_type_of_18611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 20), context_18610, 'get_type_of')
            # Calling get_type_of(args, kwargs) (line 537)
            get_type_of_call_result_18614 = invoke(stypy.reporting.localization.Localization(__file__, 537, 20), get_type_of_18611, *[name_18612], **kwargs_18613)
            
            # Assigning a type to the variable 'type_' (line 537)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 12), 'type_', get_type_of_call_result_18614)
            
            # Type idiom detected: calculating its left and rigth part (line 539)
            # Getting the type of 'type_' (line 539)
            type__18615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 15), 'type_')
            # Getting the type of 'None' (line 539)
            None_18616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 24), 'None')
            
            (may_be_18617, more_types_in_union_18618) = may_be_none(type__18615, None_18616)

            if may_be_18617:

                if more_types_in_union_18618:
                    # Runtime conditional SSA (line 539)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store


                if more_types_in_union_18618:
                    # SSA join for if statement (line 539)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Call to del_type_of(...): (line 542)
            # Processing the call arguments (line 542)
            # Getting the type of 'name' (line 542)
            name_18621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 39), 'name', False)
            # Processing the call keyword arguments (line 542)
            kwargs_18622 = {}
            # Getting the type of 'context' (line 542)
            context_18619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 19), 'context', False)
            # Obtaining the member 'del_type_of' of a type (line 542)
            del_type_of_18620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 19), context_18619, 'del_type_of')
            # Calling del_type_of(args, kwargs) (line 542)
            del_type_of_call_result_18623 = invoke(stypy.reporting.localization.Localization(__file__, 542, 19), del_type_of_18620, *[name_18621], **kwargs_18622)
            
            # Assigning a type to the variable 'stypy_return_type' (line 542)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 12), 'stypy_return_type', del_type_of_call_result_18623)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to UndefinedTypeError(...): (line 544)
        # Processing the call arguments (line 544)
        # Getting the type of 'localization' (line 544)
        localization_18625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 34), 'localization', False)
        str_18626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 48), 'str', "The variable '%s' does not exist")
        
        # Call to str(...): (line 544)
        # Processing the call arguments (line 544)
        # Getting the type of 'name' (line 544)
        name_18628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 89), 'name', False)
        # Processing the call keyword arguments (line 544)
        kwargs_18629 = {}
        # Getting the type of 'str' (line 544)
        str_18627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 85), 'str', False)
        # Calling str(args, kwargs) (line 544)
        str_call_result_18630 = invoke(stypy.reporting.localization.Localization(__file__, 544, 85), str_18627, *[name_18628], **kwargs_18629)
        
        # Applying the binary operator '%' (line 544)
        result_mod_18631 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 48), '%', str_18626, str_call_result_18630)
        
        # Processing the call keyword arguments (line 544)
        kwargs_18632 = {}
        # Getting the type of 'UndefinedTypeError' (line 544)
        UndefinedTypeError_18624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 15), 'UndefinedTypeError', False)
        # Calling UndefinedTypeError(args, kwargs) (line 544)
        UndefinedTypeError_call_result_18633 = invoke(stypy.reporting.localization.Localization(__file__, 544, 15), UndefinedTypeError_18624, *[localization_18625, result_mod_18631], **kwargs_18632)
        
        # Assigning a type to the variable 'stypy_return_type' (line 544)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 8), 'stypy_return_type', UndefinedTypeError_call_result_18633)
        
        # ################# End of '__del_type_of_from_function_context(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__del_type_of_from_function_context' in the type store
        # Getting the type of 'stypy_return_type' (line 500)
        stypy_return_type_18634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18634)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__del_type_of_from_function_context'
        return stypy_return_type_18634


    @norecursion
    def __len__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__len__'
        module_type_store = module_type_store.open_function_context('__len__', 548, 4, False)
        # Assigning a type to the variable 'self' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.__len__.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.__len__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.__len__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.__len__.__dict__.__setitem__('stypy_function_name', 'TypeStore.__len__')
        TypeStore.__len__.__dict__.__setitem__('stypy_param_names_list', [])
        TypeStore.__len__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.__len__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.__len__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.__len__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.__len__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.__len__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.__len__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__len__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__len__(...)' code ##################

        str_18635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, (-1)), 'str', '\n        len operator, returning the number of function context stored in this type store\n        :return:\n        ')
        
        # Call to len(...): (line 553)
        # Processing the call arguments (line 553)
        # Getting the type of 'self' (line 553)
        self_18637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 19), 'self', False)
        # Obtaining the member 'context_stack' of a type (line 553)
        context_stack_18638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 19), self_18637, 'context_stack')
        # Processing the call keyword arguments (line 553)
        kwargs_18639 = {}
        # Getting the type of 'len' (line 553)
        len_18636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 15), 'len', False)
        # Calling len(args, kwargs) (line 553)
        len_call_result_18640 = invoke(stypy.reporting.localization.Localization(__file__, 553, 15), len_18636, *[context_stack_18638], **kwargs_18639)
        
        # Assigning a type to the variable 'stypy_return_type' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'stypy_return_type', len_call_result_18640)
        
        # ################# End of '__len__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__len__' in the type store
        # Getting the type of 'stypy_return_type' (line 548)
        stypy_return_type_18641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18641)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__len__'
        return stypy_return_type_18641


    @norecursion
    def __iter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iter__'
        module_type_store = module_type_store.open_function_context('__iter__', 555, 4, False)
        # Assigning a type to the variable 'self' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.__iter__.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.__iter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.__iter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.__iter__.__dict__.__setitem__('stypy_function_name', 'TypeStore.__iter__')
        TypeStore.__iter__.__dict__.__setitem__('stypy_param_names_list', [])
        TypeStore.__iter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.__iter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.__iter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.__iter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.__iter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.__iter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.__iter__', [], None, None, defaults, varargs, kwargs)

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

        str_18642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, (-1)), 'str', '\n        Iterator interface, to traverse function contexts\n        :return:\n        ')
        
        # Getting the type of 'self' (line 560)
        self_18643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 25), 'self')
        # Obtaining the member 'context_stack' of a type (line 560)
        context_stack_18644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 25), self_18643, 'context_stack')
        # Assigning a type to the variable 'context_stack_18644' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'context_stack_18644', context_stack_18644)
        # Testing if the for loop is going to be iterated (line 560)
        # Testing the type of a for loop iterable (line 560)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 560, 8), context_stack_18644)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 560, 8), context_stack_18644):
            # Getting the type of the for loop variable (line 560)
            for_loop_var_18645 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 560, 8), context_stack_18644)
            # Assigning a type to the variable 'f_context' (line 560)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'f_context', for_loop_var_18645)
            # SSA begins for a for statement (line 560)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Creating a generator
            # Getting the type of 'f_context' (line 561)
            f_context_18646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 18), 'f_context')
            GeneratorType_18647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 12), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 12), GeneratorType_18647, f_context_18646)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 12), 'stypy_return_type', GeneratorType_18647)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of '__iter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iter__' in the type store
        # Getting the type of 'stypy_return_type' (line 555)
        stypy_return_type_18648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18648)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iter__'
        return stypy_return_type_18648


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 563, 4, False)
        # Assigning a type to the variable 'self' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.__getitem__.__dict__.__setitem__('stypy_function_name', 'TypeStore.__getitem__')
        TypeStore.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['item'])
        TypeStore.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.__getitem__', ['item'], None, None, defaults, varargs, kwargs)

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

        str_18649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, (-1)), 'str', '\n        Returns the nth function context in the context stack\n        :param item: Index of the function context\n        :return: A Function context or an exception if the position is not valid\n        ')
        
        # Obtaining the type of the subscript
        # Getting the type of 'item' (line 569)
        item_18650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 34), 'item')
        # Getting the type of 'self' (line 569)
        self_18651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 15), 'self')
        # Obtaining the member 'context_stack' of a type (line 569)
        context_stack_18652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 15), self_18651, 'context_stack')
        # Obtaining the member '__getitem__' of a type (line 569)
        getitem___18653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 15), context_stack_18652, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 569)
        subscript_call_result_18654 = invoke(stypy.reporting.localization.Localization(__file__, 569, 15), getitem___18653, item_18650)
        
        # Assigning a type to the variable 'stypy_return_type' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'stypy_return_type', subscript_call_result_18654)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 563)
        stypy_return_type_18655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18655)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_18655


    @norecursion
    def __contains__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__contains__'
        module_type_store = module_type_store.open_function_context('__contains__', 571, 4, False)
        # Assigning a type to the variable 'self' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.__contains__.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.__contains__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.__contains__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.__contains__.__dict__.__setitem__('stypy_function_name', 'TypeStore.__contains__')
        TypeStore.__contains__.__dict__.__setitem__('stypy_param_names_list', ['item'])
        TypeStore.__contains__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.__contains__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.__contains__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.__contains__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.__contains__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.__contains__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.__contains__', ['item'], None, None, defaults, varargs, kwargs)

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

        str_18656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, (-1)), 'str', '\n        in operator, to see if a variable is defined in a function context of the current context stack\n        :param item: variable\n        :return: bool\n        ')
        
        # Assigning a Call to a Name (line 577):
        
        # Assigning a Call to a Name (line 577):
        
        # Call to get_type_of(...): (line 577)
        # Processing the call arguments (line 577)
        # Getting the type of 'None' (line 577)
        None_18659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 33), 'None', False)
        # Getting the type of 'item' (line 577)
        item_18660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 39), 'item', False)
        # Processing the call keyword arguments (line 577)
        kwargs_18661 = {}
        # Getting the type of 'self' (line 577)
        self_18657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 16), 'self', False)
        # Obtaining the member 'get_type_of' of a type (line 577)
        get_type_of_18658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 16), self_18657, 'get_type_of')
        # Calling get_type_of(args, kwargs) (line 577)
        get_type_of_call_result_18662 = invoke(stypy.reporting.localization.Localization(__file__, 577, 16), get_type_of_18658, *[None_18659, item_18660], **kwargs_18661)
        
        # Assigning a type to the variable 'type_' (line 577)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), 'type_', get_type_of_call_result_18662)
        
        
        # Getting the type of 'type_' (line 579)
        type__18663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 20), 'type_')
        # Obtaining the member '__class__' of a type (line 579)
        class___18664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 20), type__18663, '__class__')
        # Getting the type of 'TypeError' (line 579)
        TypeError_18665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 39), 'TypeError')
        # Applying the binary operator '==' (line 579)
        result_eq_18666 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 20), '==', class___18664, TypeError_18665)
        
        # Applying the 'not' unary operator (line 579)
        result_not__18667 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 15), 'not', result_eq_18666)
        
        # Assigning a type to the variable 'stypy_return_type' (line 579)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 8), 'stypy_return_type', result_not__18667)
        
        # ################# End of '__contains__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__contains__' in the type store
        # Getting the type of 'stypy_return_type' (line 571)
        stypy_return_type_18668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18668)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__contains__'
        return stypy_return_type_18668


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 581, 4, False)
        # Assigning a type to the variable 'self' (line 582)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'TypeStore.stypy__repr__')
        TypeStore.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        TypeStore.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        str_18669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, (-1)), 'str', '\n        Textual representation of the type store\n        :return: str\n        ')
        
        # Assigning a BinOp to a Name (line 586):
        
        # Assigning a BinOp to a Name (line 586):
        str_18670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 14), 'str', "Type store of file '")
        
        # Call to str(...): (line 586)
        # Processing the call arguments (line 586)
        
        # Obtaining the type of the subscript
        int_18672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 72), 'int')
        
        # Call to split(...): (line 586)
        # Processing the call arguments (line 586)
        str_18676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 67), 'str', '/')
        # Processing the call keyword arguments (line 586)
        kwargs_18677 = {}
        # Getting the type of 'self' (line 586)
        self_18673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 43), 'self', False)
        # Obtaining the member 'program_name' of a type (line 586)
        program_name_18674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 43), self_18673, 'program_name')
        # Obtaining the member 'split' of a type (line 586)
        split_18675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 43), program_name_18674, 'split')
        # Calling split(args, kwargs) (line 586)
        split_call_result_18678 = invoke(stypy.reporting.localization.Localization(__file__, 586, 43), split_18675, *[str_18676], **kwargs_18677)
        
        # Obtaining the member '__getitem__' of a type (line 586)
        getitem___18679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 43), split_call_result_18678, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 586)
        subscript_call_result_18680 = invoke(stypy.reporting.localization.Localization(__file__, 586, 43), getitem___18679, int_18672)
        
        # Processing the call keyword arguments (line 586)
        kwargs_18681 = {}
        # Getting the type of 'str' (line 586)
        str_18671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 39), 'str', False)
        # Calling str(args, kwargs) (line 586)
        str_call_result_18682 = invoke(stypy.reporting.localization.Localization(__file__, 586, 39), str_18671, *[subscript_call_result_18680], **kwargs_18681)
        
        # Applying the binary operator '+' (line 586)
        result_add_18683 = python_operator(stypy.reporting.localization.Localization(__file__, 586, 14), '+', str_18670, str_call_result_18682)
        
        str_18684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 79), 'str', "'\n")
        # Applying the binary operator '+' (line 586)
        result_add_18685 = python_operator(stypy.reporting.localization.Localization(__file__, 586, 77), '+', result_add_18683, str_18684)
        
        # Assigning a type to the variable 'txt' (line 586)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 8), 'txt', result_add_18685)
        
        # Getting the type of 'txt' (line 587)
        txt_18686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'txt')
        str_18687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 15), 'str', 'Active contexts:\n')
        # Applying the binary operator '+=' (line 587)
        result_iadd_18688 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 8), '+=', txt_18686, str_18687)
        # Assigning a type to the variable 'txt' (line 587)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'txt', result_iadd_18688)
        
        
        # Getting the type of 'self' (line 589)
        self_18689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 23), 'self')
        # Obtaining the member 'context_stack' of a type (line 589)
        context_stack_18690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 23), self_18689, 'context_stack')
        # Assigning a type to the variable 'context_stack_18690' (line 589)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 8), 'context_stack_18690', context_stack_18690)
        # Testing if the for loop is going to be iterated (line 589)
        # Testing the type of a for loop iterable (line 589)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 589, 8), context_stack_18690)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 589, 8), context_stack_18690):
            # Getting the type of the for loop variable (line 589)
            for_loop_var_18691 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 589, 8), context_stack_18690)
            # Assigning a type to the variable 'context' (line 589)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 8), 'context', for_loop_var_18691)
            # SSA begins for a for statement (line 589)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'txt' (line 590)
            txt_18692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 12), 'txt')
            
            # Call to str(...): (line 590)
            # Processing the call arguments (line 590)
            # Getting the type of 'context' (line 590)
            context_18694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 23), 'context', False)
            # Processing the call keyword arguments (line 590)
            kwargs_18695 = {}
            # Getting the type of 'str' (line 590)
            str_18693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 19), 'str', False)
            # Calling str(args, kwargs) (line 590)
            str_call_result_18696 = invoke(stypy.reporting.localization.Localization(__file__, 590, 19), str_18693, *[context_18694], **kwargs_18695)
            
            # Applying the binary operator '+=' (line 590)
            result_iadd_18697 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 12), '+=', txt_18692, str_call_result_18696)
            # Assigning a type to the variable 'txt' (line 590)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 12), 'txt', result_iadd_18697)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 592)
        # Processing the call arguments (line 592)
        # Getting the type of 'self' (line 592)
        self_18699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 15), 'self', False)
        # Obtaining the member 'last_function_contexts' of a type (line 592)
        last_function_contexts_18700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 15), self_18699, 'last_function_contexts')
        # Processing the call keyword arguments (line 592)
        kwargs_18701 = {}
        # Getting the type of 'len' (line 592)
        len_18698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 11), 'len', False)
        # Calling len(args, kwargs) (line 592)
        len_call_result_18702 = invoke(stypy.reporting.localization.Localization(__file__, 592, 11), len_18698, *[last_function_contexts_18700], **kwargs_18701)
        
        int_18703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 46), 'int')
        # Applying the binary operator '>' (line 592)
        result_gt_18704 = python_operator(stypy.reporting.localization.Localization(__file__, 592, 11), '>', len_call_result_18702, int_18703)
        
        # Testing if the type of an if condition is none (line 592)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 592, 8), result_gt_18704):
            pass
        else:
            
            # Testing the type of an if condition (line 592)
            if_condition_18705 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 592, 8), result_gt_18704)
            # Assigning a type to the variable 'if_condition_18705' (line 592)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'if_condition_18705', if_condition_18705)
            # SSA begins for if statement (line 592)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'txt' (line 593)
            txt_18706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 12), 'txt')
            str_18707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 19), 'str', 'Other contexts created during execution:\n')
            # Applying the binary operator '+=' (line 593)
            result_iadd_18708 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 12), '+=', txt_18706, str_18707)
            # Assigning a type to the variable 'txt' (line 593)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 12), 'txt', result_iadd_18708)
            
            
            # Getting the type of 'self' (line 594)
            self_18709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 27), 'self')
            # Obtaining the member 'last_function_contexts' of a type (line 594)
            last_function_contexts_18710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 27), self_18709, 'last_function_contexts')
            # Assigning a type to the variable 'last_function_contexts_18710' (line 594)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 12), 'last_function_contexts_18710', last_function_contexts_18710)
            # Testing if the for loop is going to be iterated (line 594)
            # Testing the type of a for loop iterable (line 594)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 594, 12), last_function_contexts_18710)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 594, 12), last_function_contexts_18710):
                # Getting the type of the for loop variable (line 594)
                for_loop_var_18711 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 594, 12), last_function_contexts_18710)
                # Assigning a type to the variable 'context' (line 594)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 12), 'context', for_loop_var_18711)
                # SSA begins for a for statement (line 594)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'txt' (line 595)
                txt_18712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 16), 'txt')
                
                # Call to str(...): (line 595)
                # Processing the call arguments (line 595)
                # Getting the type of 'context' (line 595)
                context_18714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 27), 'context', False)
                # Processing the call keyword arguments (line 595)
                kwargs_18715 = {}
                # Getting the type of 'str' (line 595)
                str_18713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 23), 'str', False)
                # Calling str(args, kwargs) (line 595)
                str_call_result_18716 = invoke(stypy.reporting.localization.Localization(__file__, 595, 23), str_18713, *[context_18714], **kwargs_18715)
                
                # Applying the binary operator '+=' (line 595)
                result_iadd_18717 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 16), '+=', txt_18712, str_call_result_18716)
                # Assigning a type to the variable 'txt' (line 595)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 16), 'txt', result_iadd_18717)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 592)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'txt' (line 597)
        txt_18718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 15), 'txt')
        # Assigning a type to the variable 'stypy_return_type' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'stypy_return_type', txt_18718)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 581)
        stypy_return_type_18719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18719)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_18719


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 599, 4, False)
        # Assigning a type to the variable 'self' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.stypy__str__.__dict__.__setitem__('stypy_function_name', 'TypeStore.stypy__str__')
        TypeStore.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        TypeStore.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.stypy__str__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to __repr__(...): (line 600)
        # Processing the call keyword arguments (line 600)
        kwargs_18722 = {}
        # Getting the type of 'self' (line 600)
        self_18720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 15), 'self', False)
        # Obtaining the member '__repr__' of a type (line 600)
        repr___18721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 15), self_18720, '__repr__')
        # Calling __repr__(args, kwargs) (line 600)
        repr___call_result_18723 = invoke(stypy.reporting.localization.Localization(__file__, 600, 15), repr___18721, *[], **kwargs_18722)
        
        # Assigning a type to the variable 'stypy_return_type' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'stypy_return_type', repr___call_result_18723)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 599)
        stypy_return_type_18724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18724)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_18724


    @norecursion
    def get_type_of_member(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_type_of_member'
        module_type_store = module_type_store.open_function_context('get_type_of_member', 604, 4, False)
        # Assigning a type to the variable 'self' (line 605)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.get_type_of_member.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.get_type_of_member.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.get_type_of_member.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.get_type_of_member.__dict__.__setitem__('stypy_function_name', 'TypeStore.get_type_of_member')
        TypeStore.get_type_of_member.__dict__.__setitem__('stypy_param_names_list', ['localization', 'member_name'])
        TypeStore.get_type_of_member.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.get_type_of_member.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.get_type_of_member.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.get_type_of_member.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.get_type_of_member.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.get_type_of_member.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.get_type_of_member', ['localization', 'member_name'], None, None, defaults, varargs, kwargs)

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

        str_18725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, (-1)), 'str', '\n        Proxy for get_type_of, to comply with NonPythonType interface\n        :param localization: Caller information\n        :param member_name: Member name\n        :return:\n        ')
        
        # Call to get_type_of(...): (line 611)
        # Processing the call arguments (line 611)
        # Getting the type of 'localization' (line 611)
        localization_18728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 32), 'localization', False)
        # Getting the type of 'member_name' (line 611)
        member_name_18729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 46), 'member_name', False)
        # Processing the call keyword arguments (line 611)
        kwargs_18730 = {}
        # Getting the type of 'self' (line 611)
        self_18726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 15), 'self', False)
        # Obtaining the member 'get_type_of' of a type (line 611)
        get_type_of_18727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 15), self_18726, 'get_type_of')
        # Calling get_type_of(args, kwargs) (line 611)
        get_type_of_call_result_18731 = invoke(stypy.reporting.localization.Localization(__file__, 611, 15), get_type_of_18727, *[localization_18728, member_name_18729], **kwargs_18730)
        
        # Assigning a type to the variable 'stypy_return_type' (line 611)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 8), 'stypy_return_type', get_type_of_call_result_18731)
        
        # ################# End of 'get_type_of_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_type_of_member' in the type store
        # Getting the type of 'stypy_return_type' (line 604)
        stypy_return_type_18732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18732)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_type_of_member'
        return stypy_return_type_18732


    @norecursion
    def set_type_of_member(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_type_of_member'
        module_type_store = module_type_store.open_function_context('set_type_of_member', 613, 4, False)
        # Assigning a type to the variable 'self' (line 614)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.set_type_of_member.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.set_type_of_member.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.set_type_of_member.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.set_type_of_member.__dict__.__setitem__('stypy_function_name', 'TypeStore.set_type_of_member')
        TypeStore.set_type_of_member.__dict__.__setitem__('stypy_param_names_list', ['localization', 'member_name', 'member_value'])
        TypeStore.set_type_of_member.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.set_type_of_member.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.set_type_of_member.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.set_type_of_member.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.set_type_of_member.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.set_type_of_member.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.set_type_of_member', ['localization', 'member_name', 'member_value'], None, None, defaults, varargs, kwargs)

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

        str_18733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, (-1)), 'str', '\n        Proxy for set_type_of, to comply with NonPythonType interface\n        :param localization: Caller information\n        :param member_name: Member name\n        :return:\n        ')
        
        # Call to set_type_of(...): (line 620)
        # Processing the call arguments (line 620)
        # Getting the type of 'localization' (line 620)
        localization_18736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 32), 'localization', False)
        # Getting the type of 'member_name' (line 620)
        member_name_18737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 46), 'member_name', False)
        # Getting the type of 'member_value' (line 620)
        member_value_18738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 59), 'member_value', False)
        # Processing the call keyword arguments (line 620)
        kwargs_18739 = {}
        # Getting the type of 'self' (line 620)
        self_18734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 15), 'self', False)
        # Obtaining the member 'set_type_of' of a type (line 620)
        set_type_of_18735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 15), self_18734, 'set_type_of')
        # Calling set_type_of(args, kwargs) (line 620)
        set_type_of_call_result_18740 = invoke(stypy.reporting.localization.Localization(__file__, 620, 15), set_type_of_18735, *[localization_18736, member_name_18737, member_value_18738], **kwargs_18739)
        
        # Assigning a type to the variable 'stypy_return_type' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 8), 'stypy_return_type', set_type_of_call_result_18740)
        
        # ################# End of 'set_type_of_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_type_of_member' in the type store
        # Getting the type of 'stypy_return_type' (line 613)
        stypy_return_type_18741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18741)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_type_of_member'
        return stypy_return_type_18741


    @norecursion
    def delete_member(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'delete_member'
        module_type_store = module_type_store.open_function_context('delete_member', 624, 4, False)
        # Assigning a type to the variable 'self' (line 625)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.delete_member.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.delete_member.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.delete_member.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.delete_member.__dict__.__setitem__('stypy_function_name', 'TypeStore.delete_member')
        TypeStore.delete_member.__dict__.__setitem__('stypy_param_names_list', ['localization', 'member'])
        TypeStore.delete_member.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.delete_member.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.delete_member.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.delete_member.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.delete_member.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.delete_member.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.delete_member', ['localization', 'member'], None, None, defaults, varargs, kwargs)

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

        str_18742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, (-1)), 'str', '\n        Proxy for del_type_of, to comply with NonPythonType interface\n        :param localization: Caller information\n        :param member: Member name\n        :return:\n        ')
        
        # Call to del_type_of(...): (line 631)
        # Processing the call arguments (line 631)
        # Getting the type of 'localization' (line 631)
        localization_18745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 32), 'localization', False)
        # Getting the type of 'member' (line 631)
        member_18746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 46), 'member', False)
        # Processing the call keyword arguments (line 631)
        kwargs_18747 = {}
        # Getting the type of 'self' (line 631)
        self_18743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 15), 'self', False)
        # Obtaining the member 'del_type_of' of a type (line 631)
        del_type_of_18744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 15), self_18743, 'del_type_of')
        # Calling del_type_of(args, kwargs) (line 631)
        del_type_of_call_result_18748 = invoke(stypy.reporting.localization.Localization(__file__, 631, 15), del_type_of_18744, *[localization_18745, member_18746], **kwargs_18747)
        
        # Assigning a type to the variable 'stypy_return_type' (line 631)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 'stypy_return_type', del_type_of_call_result_18748)
        
        # ################# End of 'delete_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'delete_member' in the type store
        # Getting the type of 'stypy_return_type' (line 624)
        stypy_return_type_18749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18749)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'delete_member'
        return stypy_return_type_18749


    @norecursion
    def supports_structural_reflection(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'supports_structural_reflection'
        module_type_store = module_type_store.open_function_context('supports_structural_reflection', 633, 4, False)
        # Assigning a type to the variable 'self' (line 634)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.supports_structural_reflection.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.supports_structural_reflection.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.supports_structural_reflection.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.supports_structural_reflection.__dict__.__setitem__('stypy_function_name', 'TypeStore.supports_structural_reflection')
        TypeStore.supports_structural_reflection.__dict__.__setitem__('stypy_param_names_list', [])
        TypeStore.supports_structural_reflection.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.supports_structural_reflection.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.supports_structural_reflection.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.supports_structural_reflection.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.supports_structural_reflection.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.supports_structural_reflection.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.supports_structural_reflection', [], None, None, defaults, varargs, kwargs)

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

        str_18750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, (-1)), 'str', '\n        TypeStores (modules) always support structural reflection\n        :return: True\n        ')
        # Getting the type of 'True' (line 638)
        True_18751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 638)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 8), 'stypy_return_type', True_18751)
        
        # ################# End of 'supports_structural_reflection(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'supports_structural_reflection' in the type store
        # Getting the type of 'stypy_return_type' (line 633)
        stypy_return_type_18752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18752)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'supports_structural_reflection'
        return stypy_return_type_18752


    @norecursion
    def clone(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'clone'
        module_type_store = module_type_store.open_function_context('clone', 642, 4, False)
        # Assigning a type to the variable 'self' (line 643)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeStore.clone.__dict__.__setitem__('stypy_localization', localization)
        TypeStore.clone.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeStore.clone.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeStore.clone.__dict__.__setitem__('stypy_function_name', 'TypeStore.clone')
        TypeStore.clone.__dict__.__setitem__('stypy_param_names_list', [])
        TypeStore.clone.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeStore.clone.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeStore.clone.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeStore.clone.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeStore.clone.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeStore.clone.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeStore.clone', [], None, None, defaults, varargs, kwargs)

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

        str_18753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, (-1)), 'str', '\n        Proxy for clone_type_store, to comply with NonPythonType interface\n        :return:\n        ')
        
        # Call to clone_type_store(...): (line 647)
        # Processing the call keyword arguments (line 647)
        kwargs_18756 = {}
        # Getting the type of 'self' (line 647)
        self_18754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 15), 'self', False)
        # Obtaining the member 'clone_type_store' of a type (line 647)
        clone_type_store_18755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 15), self_18754, 'clone_type_store')
        # Calling clone_type_store(args, kwargs) (line 647)
        clone_type_store_call_result_18757 = invoke(stypy.reporting.localization.Localization(__file__, 647, 15), clone_type_store_18755, *[], **kwargs_18756)
        
        # Assigning a type to the variable 'stypy_return_type' (line 647)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'stypy_return_type', clone_type_store_call_result_18757)
        
        # ################# End of 'clone(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clone' in the type store
        # Getting the type of 'stypy_return_type' (line 642)
        stypy_return_type_18758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18758)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clone'
        return stypy_return_type_18758


# Assigning a type to the variable 'TypeStore' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'TypeStore', TypeStore)

# Assigning a Call to a Name (line 29):

# Call to dict(...): (line 29)
# Processing the call keyword arguments (line 29)
kwargs_18760 = {}
# Getting the type of 'dict' (line 29)
dict_18759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 29), 'dict', False)
# Calling dict(args, kwargs) (line 29)
dict_call_result_18761 = invoke(stypy.reporting.localization.Localization(__file__, 29, 29), dict_18759, *[], **kwargs_18760)

# Getting the type of 'TypeStore'
TypeStore_18762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeStore')
# Setting the type of the member 'type_stores_of_modules' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeStore_18762, 'type_stores_of_modules', dict_call_result_18761)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
